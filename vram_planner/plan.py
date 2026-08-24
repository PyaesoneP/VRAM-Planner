"""Turning a model file plus a budget into a layer split."""
import os
from .const import _mib
from .paths import user_path
from .gguf import load_gguf, parse_meta_only
from .model import classify_tensors, extract_config, ot_regex
from .kv import (KV_TYPE_BYTES, kv_bytes_per_token, kv_bytes_per_token_growing,
                 kv_bytes_total, kv_bytes_total_at, max_ctx_for_kv_budget,
                 recurrent_bytes, resolve_kv_lengths, swa_cache_len)
from .compute import MTP_SPEC_CONST_MIB, MTP_SPEC_PER_SEQ_MIB, compute_buffer_split, compute_buffer_terms, graph_is_split, output_head_on_gpu, vision_config, vision_grid, vision_peak_mib
from .gpu import gpu_list, platform_support, cached_bandwidth
from .speed import estimate_speed
from .calib import calibration_status
from .cards import load_card, remember_card


def default_vram_budget(gpu, basis="total", reserve_mib=512):
    """The VRAM a plan should be built against. One rule, two callers.

    This function exists because there used to be two rules. The browser
    prefilled its budget from free VRAM at page load and sent a 0 reserve; the
    speed sweep seeded its stage-A ladder from TOTAL VRAM with a 512 MiB
    reserve. Both are defensible and running both is not: on a 16 GiB card with
    11.5 GiB free they differ by several -ngl rungs, so the planner and the
    sweep recommended different configs for no reason either of them stated.

    `basis="total"` is the default and is what the sweep measures under.
    preflight() refuses to run a campaign unless the card is essentially empty,
    so free VRAM at planning time is transient state that the rows will never
    be measured under - seeding off it produces a garbage ladder exactly when
    something happens to be loaded, which is when someone is most likely to be
    planning.

    `basis="free"` answers the different question - "will this load right now,
    next to what I already have open" - and is the user's to choose. It is a
    real question; it is just not the one a campaign answers.

    Returns MiB, never negative.
    """
    gpu = gpu or {}
    total = float(gpu.get("total_mib") or 0)
    free = float(gpu.get("free_mib") or 0)
    have = free if basis == "free" else total
    return max(0.0, have - max(0.0, float(reserve_mib or 0)))


# How close to the budget counts as "tight" rather than simply fitting. The
# compute buffer is the one estimated term, and its held-out error is 22.5%
# mean on the buffer alone - so a plan landing inside the last few percent of
# the budget is inside the estimate's own error bar and should not be reported
# with the same confidence as one with room to spare.
VERDICT_TIGHT_FRAC = 0.97

# Cells the DFlash draft KV cache holds per block of draft context, as a
# multiple of the trained block depth. Observed in the reference DFlash log
# (1792 cells over a 16-block drafter), and deliberately round - it sizes a
# cache whose true layout is llama.cpp-internal. The number it produces is a
# small fraction of the drafter's weights either way; see find_drafter_for().
DRAFT_KV_BLOCK_SLACK = 8.0


def _verdict(plan, inputs=None, ram_free_mib=None):
    """The plan's answer as a state and a word, rather than as prose.

    `headline` stays what it is - the sentence, and what the CLI prints. This
    is the part the UI leads with, and it is derived here so the browser cannot
    invent its own reading of the same fields. It used to: the step tab read
    r.verdict.fits and r.totals.vram_mib, neither of which this module has ever
    returned, so the tab said "fits" for every plan including the ones that did
    not.

    Two bases are in play and they differ by the reserve and the safety margin:

      * the EFFECTIVE one the split was searched against - usage excluding the
        driver reserve, budget already reduced by it - which is the pair that
        must be compared to decide whether the plan fits;
      * the RAW one the memory bar draws and the user checks against Task
        Manager, which counts the reserve as occupied and the budget as typed.

    The decision is made on the first and reported on the second, because a tab
    reading 10.25 GiB beside a bar reading 11,013 MiB is two answers to one
    question.
    """
    inputs = inputs or {}
    used = float(plan.get("vram_used_mib") or 0.0)
    budget = float(plan.get("vram_budget_mib") or 0.0)
    reserve = float(inputs.get("gpu_reserve_mib") or 0.0)
    raw_budget = float(inputs.get("vram_budget_mib") or 0.0) or budget
    ram = float(plan.get("ram_used_mib") or 0.0)
    if plan.get("attention_overflow") or plan.get("kv_overflow"):
        state, word = "no_fit", "DOES NOT FIT"
    elif budget and used > budget:
        # It loads - WDDM spills into shared system memory rather than failing -
        # and that is the worst outcome here, not the safest: the numbers still
        # look like numbers while the speed is off a cliff.
        state, word = "spills", "SPILLS TO SHARED MEMORY"
    elif plan.get("ram_ok") is False:
        state, word = "no_fit", "DOES NOT FIT IN RAM"
    elif budget and used > budget * VERDICT_TIGHT_FRAC:
        state, word = "tight", "FITS, BARELY"
    else:
        state, word = "fits", "FITS"
    return {
        "state": state, "word": word,
        # Reported on the raw basis, so this agrees with the memory bar.
        "vram_mib": used + reserve, "vram_budget_mib": raw_budget,
        # ...and the pair the decision was actually made on, for anyone who
        # needs to reproduce it.
        "vram_used_eff_mib": used, "vram_budget_eff_mib": budget,
        "gpu_reserve_mib": reserve,
        "ram_mib": ram,
        "ram_budget_mib": plan.get("ram_budget_mib"),
        "ram_free_mib": ram_free_mib,
        # The three knobs a launcher needs, normalised across the planners
        # so a caller does not have to know which one produced this plan.
        "ngl": plan.get("n_gpu_layers"),
        "ncmoe": plan.get("n_cpu_moe") or 0,
        "nffn": plan.get("n_cpu_ffn") or 0,
    }


def find_mmproj(model_path):
    """A multimodal model ships a separate vision/audio projector next to the
    weights (mmproj-*.gguf). LM Studio loads it with the model and puts it on the
    GPU, and it counts it in the "model size" it shows you - so it must be part of
    the VRAM budget. Returns {path, bytes, tensor_bytes, vision} or None.

    `vision` carries the tower's geometry (patch size, merge, block count, head
    count) so the caller can size the ENCODER TRANSIENTS as well as the weights.
    Charging only the weights is what let a plan read "fits" and then OOM on the
    first image - see vision_peak_mib()."""
    try:
        d = os.path.dirname(os.path.abspath(model_path))
        base = os.path.basename(model_path).lower()
        if base.startswith("mmproj"):
            return None                              # the projector itself was picked
        for fn in sorted(os.listdir(d)):
            if fn.lower().startswith("mmproj") and fn.lower().endswith(".gguf"):
                p = os.path.join(d, fn)
                nb = os.path.getsize(p)
                try:
                    tb = sum(t["n_bytes"] for t in load_gguf(p)["tensors"])
                except Exception:
                    tb = nb
                try:
                    vis = vision_config(parse_meta_only(p))
                except Exception:
                    vis = None
                return {"path": p, "name": fn, "bytes": nb, "tensor_bytes": tb,
                        "vision": vis}
    except Exception:
        pass
    return None


def find_drafter_for(model_path):
    """The DFlash drafter next to `model_path`, full detail.

    The cheaper sibling in sweep.py (which reads only the GGUF header) exists
    because the flag-building path must not pay for a full load; this one reads
    the whole file because a plan needs the exact tensor bytes and the draft-
    cache geometry. Returns {path, name, bytes, tensor_bytes, block_size, cfg}
    or None. A drafter cannot draft itself - a path to the drafter file returns
    None, same rule as find_mmproj()."""
    d = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path).lower()
    if base.startswith("dflash") or not os.path.isdir(d):
        return None
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".gguf"):
            continue
        p = os.path.join(d, fn)
        try:
            meta = parse_meta_only(p)
        except (OSError, ValueError):
            continue
        if (meta.get("general.architecture") or "").lower() != "dflash":
            continue
        nb = os.path.getsize(p)
        try:
            g = load_gguf(p)
            tb = sum(t["n_bytes"] for t in g["tensors"])
            cfg = extract_config(g)
        except Exception:
            tb, cfg = nb, {}
        bs = meta.get("dflash.block_size")
        return {"path": p, "name": fn, "bytes": nb, "tensor_bytes": tb,
                "block_size": int(bs) if isinstance(bs, int) and bs > 0 else 16,
                "cfg": cfg}
    return None


def load_drafter(path):
    """Full detail on an explicitly picked drafter file, whatever it is.

    Unlike find_drafter_for() - which only ever returns a DFlash drafter found
    next to a model - this takes the file the user chose: a `dflash-*.gguf`, or
    a full model carrying MTP blocks, which llama.cpp will happily draft with
    (-md + --spec-type draft-mtp). Returns {path, name, bytes, tensor_bytes,
    cfg, meta} or raises ValueError when the file is not readable. The caller
    decides what the file IS; this only says what it costs."""
    path = user_path(path)          # "Copy as path" quotes are not part of it
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise ValueError("drafter file not found: %s" % path)
    meta = parse_meta_only(p)
    try:
        g = load_gguf(p)
        tb = sum(t["n_bytes"] for t in g["tensors"])
        try:
            cfg = extract_config(g)
        except Exception:
            cfg = {}
    except (OSError, ValueError) as e:
        raise ValueError("could not read drafter %s: %s" % (path, e))
    return {"path": p, "name": os.path.basename(p), "bytes": os.path.getsize(p),
            "tensor_bytes": tb, "cfg": cfg, "meta": meta}


def analyze(path, ctx, kv_type, n_ubatch, flash_attn,
            vram_budget_mib, ram_budget_mib, gpu_reserve_mib,
            compute_override_mib, safety_pct,
            gpu_layers_override=None, ram_free_mib=None, n_seq=1,
            include_mmproj=True, n_cpu_moe_override=None, n_cpu_ffn_override=None,
            bw_vram_gbs=None, bw_ram_gbs=None, ram_eff=None, ctx_fill=None,
            bw_note="", mtp_spec=False, dflash=False, drafter=None,
            image_px=None, vision_flash_attn=True, spec_kv="f16",
            plan_mode=None, mmproj_place=None):
    # `mmproj_place` supersedes the older boolean: "vram" is the projector loaded
    # to the GPU (include_mmproj=True), "none" is not loading it at all
    # (include_mmproj=False), and "ram" is --no-mmproj-offload - the file loads,
    # but its bytes are charged to the host. The boolean is still accepted so the
    # calibration and fit callers, which have no opinion here, need no edits.
    if mmproj_place is None:
        mmproj_place = "vram" if include_mmproj else "none"
    if mmproj_place not in ("vram", "ram", "none"):
        raise ValueError("mmproj_place must be vram, ram or none, not %r" % (mmproj_place,))
    include_mmproj = mmproj_place != "none"
    mmproj_on_gpu = mmproj_place == "vram"
    # The file is authoritative when it is here; the stored card stands in when it
    # is not. Reading a real file also refreshes the card, so the library builds
    # up as a side effect of ordinary use rather than needing to be curated.
    from_card = False
    if os.path.exists(path):
        model = load_gguf(path)
        cfg = extract_config(model)
        cl = classify_tensors(model, cfg)
        mmproj = find_mmproj(path)
        file_bytes, n_shards = model["file_bytes"], len(model["shards"])
        remember_card(path, cfg, cl, mmproj, file_bytes, n_shards)
    else:
        card = load_card(path)
        if not card:
            raise FileNotFoundError(
                "%s is not on disk and no stored card exists for it. Analyse it "
                "once while the file is present to record one." % os.path.basename(path))
        cfg, cl, mmproj, cmeta = card
        file_bytes, n_shards = cmeta["file_bytes"], cmeta["n_shards"]
        from_card = True

    n_layers = cfg["n_layers"] or 0
    warnings = []
    if from_card:
        # Said plainly rather than hidden: everything below is real arithmetic on
        # real recorded numbers, but nothing here was re-read from the weights.
        warnings.append(
            "Planned from a stored model card - %s is not on disk. Layer sizes, "
            "KV geometry and the projector are the ones recorded when the file was "
            "last read, so this is exact for that file and wrong for any other "
            "build or quant sharing its name." % os.path.basename(path))
    if cl["unknown_types"]:
        warnings.append("Unrecognized quant type(s) %s. Their sizes were recovered from the "
                        "gaps between tensor offsets rather than from the type table, which "
                        "is exact but includes any padding, so the total may run a few bytes "
                        "high. Everything else is unaffected." % ", ".join(cl["unknown_types"]))
    if n_layers == 0:
        warnings.append("Could not read block_count from metadata; layer math unavailable.")
    _plat = platform_support()
    if not _plat["supported"]:
        warnings.append(_plat["reason"])
    _gpus = gpu_list()
    if len(_gpus) > 1:
        warnings.append("%d GPUs detected. This plan targets GPU 0 (%s) only - multi-GPU "
                        "splits (llama.cpp --tensor-split) are not modelled, so the layer "
                        "counts below are for a single card."
                        % (len(_gpus), _gpus[0].get("name", "?")))

    # exact byte quantities -> MiB
    weights_mib = _mib(cl["weights_total"])
    embed_mib   = _mib(cl["embed_bytes"])
    output_mib  = _mib(cl["output_bytes"])
    expert_total_mib = _mib(cl["expert_bytes_total"])
    per_layer_max_mib  = _mib(cl["per_layer_max"])
    per_layer_mean_mib = _mib(cl["per_layer_mean"])
    expert_layer_mean_mib = _mib(cl["expert_layer_mean"])
    ffn_dense_total_mib = _mib(cl["ffn_dense_total"])
    ffn_layer_mean_mib = _mib(cl["ffn_layer_mean"])
    file_on_disk_mib = _mib(file_bytes)

    # Resolve each block's cache length first: with sliding-window attention the
    # windowed blocks cap out at their window and every later number depends on it.
    resolve_kv_lengths(cfg, ctx, n_ubatch, n_seq, flash_attn)
    swa_set = set(cfg["swa_layers"])
    n_swa_layers = len(swa_set)

    kv_per_tok = kv_bytes_per_token(cfg, kv_type)      # bytes/token (attn layers only)
    kv_total_mib = _mib(kv_bytes_total(cfg, kv_type))
    n_attn = len(cfg["attn_layers"]) or 1
    # per KV-bearing layer, not per block - on hybrid models those differ 4x+
    kv_per_layer_mib = kv_total_mib / n_attn
    kv_swa_mib = _mib(kv_bytes_total(cfg, kv_type, sorted(swa_set)))
    kv_global_mib = kv_total_mib - kv_swa_mib
    kv_grow_per_tok_mib = _mib(kv_bytes_per_token_growing(cfg, kv_type))

    # fixed recurrent state (hybrid/SSM models only); does not grow with context
    rec_total_mib = _mib(recurrent_bytes(cfg, range(n_layers), n_seq))
    rec_per_layer_mib = (rec_total_mib / len(cfg["ssm_layers"])) if cfg["ssm_layers"] else 0.0

    compute_terms = compute_buffer_terms(cfg, ctx, n_ubatch, flash_attn, n_seq, kv_type)
    # planning value: assume the whole graph runs on the GPU at full offload. The
    # plan builders re-split it once they know where the layers landed - including
    # the floor, which now depends on how many blocks are resident.
    compute_mib = compute_override_mib if compute_override_mib else \
        round(compute_terms["graph"] + compute_terms["floor_base"]
              + compute_terms["floor_per_layer"] * n_layers, 1)

    def compute_fn(ngl, output_on_gpu=None, any_on_cpu=None):
        """GPU/CPU compute-buffer split for a candidate layer count.

        output_on_gpu is accepted and ignored: the logits buffer is host memory at
        every offload level, measured. The parameter stays so callers that pass it
        keep working, and because where the HEAD runs is still meaningful."""
        ngl = max(0, min(int(ngl), n_layers))
        if any_on_cpu is None:
            any_on_cpu = graph_is_split(n_layers, ngl)
        return compute_buffer_split(compute_terms, ngl > 0, any_on_cpu,
                                    ngl, compute_override_mib)

    # MTP speculative decoding: the draft blocks run after all, so they get the KV
    # cache back - at the draft cache's OWN quant (spec_kv, f16 unless -ctkd/-ctvd
    # say otherwise), whatever the target's KV quant - plus a fixed pool and one
    # slot per sequence. Comes off the top of the budget like the projector.
    #
    # The KV terms below are all the measured-f16 geometry (mtp_kv_per_token is
    # the f16 bytes per token, and the DFlash cache is sized by kv_bytes_per_token
    # at f16), so the ones that scale do so by the quant's bytes per element; the
    # pool and per-sequence slot are graph buffers and do not.
    spec_kv_bpe = KV_TYPE_BYTES.get(spec_kv, 2.0) / 2.0
    spec_mib = 0.0
    if mtp_spec and cfg.get("mtp_kv_per_token"):
        spec_mib = (_mib(cfg["mtp_kv_per_token"] * ctx * spec_kv_bpe)
                    + MTP_SPEC_CONST_MIB + MTP_SPEC_PER_SEQ_MIB * max(1, n_seq))

    # DFlash speculative decoding: the drafter is a SECOND model file sharing the
    # process, so its weights are resident in VRAM (the draft runs on the GPU by
    # construction), and it keeps a small KV cache of its own plus draft-graph
    # buffers. All three are priced here, and all three are DERIVED rather than
    # measured - the drafter's exact tensor bytes, then its geometry through the
    # calibrated graph coefficients - so the result says so, and a campaign's
    # stage-D rows exist to replace the derivation with reality.
    dflash_info = None
    drafter_mtp = None     # an explicit drafter pick that is a model with MTP blocks
    mtp_fallback = False   # dflash asked, no drafter found, priced the model's own MTP
    if drafter:
        # An explicit pick overrides discovery entirely: the user named the
        # file, so the plan prices exactly that file and must not silently
        # swap in something found next to the model instead.
        det = load_drafter(drafter)
        if (det["meta"].get("general.architecture") or "").lower() == "dflash":
            bs = det["meta"].get("dflash.block_size")
            dflash_info = dict(det,
                               block_size=int(bs) if isinstance(bs, int) and bs > 0 else 16)
        elif det["cfg"].get("n_mtp_layers"):
            drafter_mtp = det
        else:
            raise ValueError(
                "%s is neither a DFlash drafter (architecture 'dflash') nor a "
                "model with MTP blocks of its own, so it cannot draft this "
                "model. Pick a dflash-*.gguf or a model whose name carries MTP."
                % det["name"])
    elif dflash:
        dflash_info = find_drafter_for(path)
        if not dflash_info:
            # No drafter to price. If the model can draft itself, the plan can
            # still be complete - llama.cpp runs one scheme per server, so the
            # model's own MTP cache is what it would actually use - but only
            # with the substitution said out loud. A model with neither is a
            # request that cannot be honored at all, and refusing beats
            # guessing (this is the deliberate ValueError the UI and selftest
            # rely on).
            if cfg.get("mtp_kv_per_token"):
                mtp_fallback = True
                if not mtp_spec:
                    spec_mib = (_mib(cfg["mtp_kv_per_token"] * ctx * spec_kv_bpe)
                                + MTP_SPEC_CONST_MIB
                                + MTP_SPEC_PER_SEQ_MIB * max(1, n_seq))
                warnings.append(
                    "no DFlash drafter next to %s - the model drafts itself, so "
                    "this plan prices its MTP draft cache instead. llama.cpp "
                    "runs one speculative scheme per server."
                    % os.path.basename(path))
            else:
                raise ValueError(
                    "no DFlash drafter next to %s - the scheme needs a separate "
                    "dflash-*.gguf whose architecture is 'dflash' in the same "
                    "directory, and this model has no MTP blocks of its own to "
                    "price instead. Run the plan without DFlash to split the "
                    "model on its own." % os.path.basename(path))
    if dflash_info:
        if mtp_spec:
            warnings.append("DFlash and MTP are both on - llama.cpp runs one "
                            "speculative scheme per server, so this plan prices "
                            "the DFlash drafter only.")
        dc = dflash_info.get("cfg") or {}
        block = dflash_info["block_size"]
        weights_d = _mib(dflash_info["tensor_bytes"])
        # The draft KV cache is the drafter's own KV, priced at the draft cache's
        # quant (spec_kv) the way the MTP draft cache is, whatever the target's KV
        # quant - held at the trained block depth times a fixed slack for the
        # cells llama.cpp keeps around each block (its draft cache in the
        # reference log carried ~7 blocks per cell row).
        try:
            draft_kv_mib = _mib(kv_bytes_per_token(dc, spec_kv)
                                * block * DRAFT_KV_BLOCK_SLACK)
        except Exception:
            draft_kv_mib = 0.0
        # Draft-graph buffers at the block depth, through the same calibrated
        # coefficients the target's graph is priced with - the drafter's own
        # geometry, so its attention and logits buffers are not the target's.
        try:
            draft_graph_mib = compute_buffer_terms(dc, block, n_ubatch,
                                                   flash_attn, n_seq,
                                                   kv_type)["graph"]
        except Exception:
            draft_graph_mib = 0.0
        spec_mib = weights_d + draft_kv_mib + draft_graph_mib
        warnings.append(
            "The DFlash drafter's working set is derived: drafter weights "
            "(exact bytes), its KV cache and draft graph (from the drafter's "
            "geometry and this machine's calibration). The campaign's stage-D "
            "rows measure the real cost.")
    elif drafter_mtp:
        if mtp_spec:
            warnings.append("MTP drafting with an external draft model - the "
                            "model's own MTP blocks stay idle, so this plan "
                            "prices the drafter's MTP cache only.")
        drafter_mib = _mib(drafter_mtp["tensor_bytes"])
        drafter_cache_mib = (_mib(drafter_mtp["cfg"]["mtp_kv_per_token"] * ctx
                                  * spec_kv_bpe)
                             + MTP_SPEC_CONST_MIB
                             + MTP_SPEC_PER_SEQ_MIB * max(1, n_seq))
        spec_mib = drafter_mib + drafter_cache_mib
        warnings.append(
            "The %s draft model's cost is derived: its weights (exact bytes) "
            "and its MTP draft cache (from the drafter's geometry and this "
            "machine's calibration)." % drafter_mtp["name"])

    # vision/audio projector: loaded to the GPU alongside the model, so it comes
    # off the top of the budget before any layer split is planned
    mmproj_mib = _mib(mmproj["tensor_bytes"]) if (mmproj and include_mmproj) else 0.0

    # Vision encoder transients. The projector's weights (above) are resident; this
    # is the pool the tower needs WHILE it encodes an image, and it is quadratic in
    # patch count. It is only reserved when the caller names an image size, because
    # a text-only plan should not carry headroom it will never use - but a plan made
    # without it is a text-only plan, and says so in the warnings below.
    # Whether the tower fuses attention is worth ~40x here and is a property of the
    # BACKEND, not the model, so report both ends the way the speed roofline does
    # rather than pretending to know. vis_peak is the one actually reserved.
    vis_cfg = (mmproj or {}).get("vision") if include_mmproj else None
    vis_grid = vis_peak = vis_ceiling = None
    vis_mib = 0.0
    if vis_cfg and image_px:
        try:
            vis_grid = vision_grid(vis_cfg, image_px[0], image_px[1])
            vis_peak = vision_peak_mib(vis_cfg, vis_grid, vision_flash_attn)
            vis_ceiling = vision_peak_mib(vis_cfg, vis_grid, False)
            vis_mib = vis_peak["total_mib"]
        except Exception:
            vis_grid = vis_peak = vis_ceiling = None
            vis_mib = 0.0

    # usable VRAM after reserving driver/OS headroom and a safety margin
    eff_vram = max(0.0, (vram_budget_mib - gpu_reserve_mib) * (1.0 - safety_pct / 100.0))
    # A projector pinned to system RAM (--no-mmproj-offload) costs no VRAM, so it
    # must not be taken off the top here - doing so would shrink every split by a
    # gigabyte the card never spends. Its bytes are charged to RAM below instead.
    gpu_held_out = (mmproj_mib + vis_mib) if mmproj_on_gpu else 0.0
    eff_vram = max(0.0, eff_vram - gpu_held_out - spec_mib)

    result = {
        "ok": True, "warnings": warnings, "config": cfg,
        "model_name": cfg["name"], "arch": cfg["arch"],
        "shards": n_shards,
        "params_total": cl["params_total"], "active_params": cl["active_params"],
        "is_moe": cl["is_moe"], "n_expert": cfg["n_expert"], "n_expert_used": cfg["n_expert_used"],
        "n_expert_layers": cl["n_expert_layers"],
        "bpw": (cl["weights_total"] * 8.0 / cl["params_total"]) if cl["params_total"] else 0,
        "quant_hist": {k: _mib(v) for k, v in cl["quant_hist"].items()},
        "sizes_mib": {
            "weights": weights_mib, "file_on_disk": file_on_disk_mib,
            "embed": embed_mib, "output": output_mib,
            "expert_total": expert_total_mib,
            "per_layer_max": per_layer_max_mib, "per_layer_mean": per_layer_mean_mib,
            "expert_layer_mean": expert_layer_mean_mib,
            "kv_total": kv_total_mib, "kv_per_layer": kv_per_layer_mib,
            "kv_per_token_kib": kv_per_tok / 1024.0,
            "kv_swa": kv_swa_mib, "kv_global": kv_global_mib,
            "kv_grow_per_token_kib": kv_grow_per_tok_mib * 1024.0,
            "recurrent_total": rec_total_mib, "recurrent_per_layer": rec_per_layer_mib,
            "compute": compute_mib, "mmproj": mmproj_mib,
            "compute_graph": compute_terms["graph"], "compute_output": compute_terms["output"],
            "compute_floor": round(compute_terms["floor_base"]
                                   + compute_terms["floor_per_layer"] * n_layers, 1),
            "bundle_on_disk": file_on_disk_mib + (_mib(mmproj["bytes"]) if mmproj else 0.0),
        },
        "dflash": ({"name": dflash_info["name"], "mib": spec_mib,
                    "weights_mib": weights_d, "kv_mib": draft_kv_mib,
                    "graph_mib": draft_graph_mib, "block_size": block,
                    "file_mib": _mib(dflash_info["bytes"]), "derived": True}
                   if dflash_info else None),
        "drafter": (
            {"kind": "mtp", "name": drafter_mtp["name"], "path": drafter_mtp["path"],
             "mib": spec_mib, "weights_mib": drafter_mib,
             "cache_mib": drafter_cache_mib,
             "file_mib": _mib(drafter_mtp["bytes"]), "derived": True,
             "depth": int(drafter_mtp["cfg"]["n_mtp_layers"] or 0)}
            if drafter_mtp else
            ({"kind": "dflash", "name": dflash_info["name"], "path": dflash_info["path"],
              "mib": spec_mib, "weights_mib": weights_d,
              "cache_mib": draft_kv_mib + draft_graph_mib,
              "file_mib": _mib(dflash_info["bytes"]), "derived": True,
              "depth": block}
             if dflash_info else None)),
        "mmproj": ({"name": mmproj["name"], "mib": _mib(mmproj["tensor_bytes"]),
                    "file_mib": _mib(mmproj["bytes"]), "included": bool(include_mmproj),
                    "place": mmproj_place}
                   if mmproj else None),
        "vision": ({"config": vis_cfg, "grid": vis_grid, "peak": vis_peak,
                    "ceiling": vis_ceiling, "reserved_mib": vis_mib,
                    "flash_attn_assumed": bool(vision_flash_attn),
                    "derived": True} if vis_cfg else None),
        "hybrid": {
            "is_hybrid": cfg["is_hybrid"],
            "n_attn_layers": len(cfg["attn_layers"]),
            "n_ssm_layers": len(cfg["ssm_layers"]),
            "attn_layers": cfg["attn_layers"],
            "interval": cfg["full_attention_interval"],
        },
        "calibration": calibration_status(),
        "from_card": from_card,
        "swa": {
            "enabled": bool(cfg["swa_layers"]),
            "n_swa": cfg["n_swa"],
            "n_swa_layers": n_swa_layers,
            "n_global_layers": len(cfg["attn_layers"]) - n_swa_layers,
            "window_cache_tokens": swa_cache_len(cfg, ctx, n_ubatch, n_seq, flash_attn),
            "head_dim": cfg["head_dim_k_swa"],
            "head_dim_global": cfg["head_dim_k"],
            "source": cfg["swa_source"],
        },
        "inputs": {
            "context": ctx, "kv_type": kv_type, "n_ubatch": n_ubatch, "n_seq": n_seq,
            "mtp_spec": bool(mtp_spec), "dflash": bool(dflash),
            "spec_kv": spec_kv,
            "mtp_depth": (int(cfg.get("n_mtp_layers") or 0)
                          if (mtp_spec or mtp_fallback) else None),
            "drafter": (os.path.abspath(drafter) if drafter
                        else (dflash_info["path"] if dflash_info else None)),
            "drafter_kind": ("mtp" if drafter_mtp else "dflash")
                            if (drafter_mtp or dflash_info) else None,
            "drafter_depth": (int(drafter_mtp["cfg"]["n_mtp_layers"] or 0)
                              if drafter_mtp else
                              (block if dflash_info else None)),
            "flash_attn": flash_attn, "vram_budget_mib": vram_budget_mib,
            "ram_budget_mib": ram_budget_mib, "gpu_reserve_mib": gpu_reserve_mib,
            "eff_vram_mib": eff_vram, "safety_pct": safety_pct,
            "n_ctx_train": cfg["n_ctx_train"],
            "mmproj_place": mmproj_place,
            # Which question was asked. Filled in below once the dispatch knows
            # whether the two-plan regime even applies - a model with one answer
            # has no mode, and saying "ceiling" there would be an invention.
            "plan_mode": None,
        },
    }

    # minimum footprint check
    min_total_mib = weights_mib + kv_total_mib
    if min_total_mib > (vram_budget_mib + ram_budget_mib):
        warnings.append("Model weights + KV (%.0f MiB) exceed VRAM+RAM budget (%.0f MiB). "
                        "Use a smaller quant or shorter context."
                        % (min_total_mib, vram_budget_mib + ram_budget_mib))

    # ---- KV cache table across context sizes ----
    kv_table = []
    for c in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if cfg["n_ctx_train"] and c > cfg["n_ctx_train"]:
            continue
        kv_table.append({"ctx": c,
                         "kv_mib": _mib(kv_bytes_total_at(cfg, kv_type, c, n_ubatch,
                                                          n_seq, flash_attn))})
    result["kv_table"] = kv_table

    if n_layers == 0:
        result["plan"] = {"kind": "unknown"}
        return result

    # Weights that reach VRAM beyond the transformer blocks. THE single largest
    # error in the end-to-end total before this was measured: on the three dense
    # models swept, CUDA0.model carries 626-843 MiB more than the blocks account
    # for, appearing as soon as -ngl is 1 and not growing after. On the two MoE
    # models it carries 15-143 MiB, i.e. nothing.
    #
    # Scored against the process counter over 144 loads, charging the token
    # embeddings on dense models only takes the total from 22.7% mean / 54.4% worst
    # to 7.1% / 39.6% - and does it uniformly, every model between 4% and 10%,
    # rather than by trading one architecture off against another. Charging them on
    # every model gets 14.6%; charging the output head instead gets 20.4%; charging
    # both gets 35.6%.
    #
    # The MECHANISM is not established - five models over three architectures is
    # enough to measure the split but not to explain it - so this is the first thing
    # to re-check when more architectures are swept, especially a dense MoE-free
    # model with untied embeddings. It errs in the safe direction for the case it is
    # least sure of: over-charging VRAM makes a plan conservative, under-charging it
    # makes the plan overcommit and the load spill.
    embed_on_gpu_mib = 0.0 if cl["is_moe"] else embed_mib

    def gpu_extra_weights(ngl):
        """Non-block weights resident in VRAM at this offload level."""
        return embed_on_gpu_mib if ngl > 0 else 0.0

    block_weights_mib = weights_mib - embed_mib - output_mib
    compute_full_mib = compute_fn(n_layers)["gpu"]
    full_need = (block_weights_mib + gpu_extra_weights(n_layers)
                 + kv_total_mib + rec_total_mib + compute_full_mib)
    fully_fits = full_need <= eff_vram

    # max context that fits fully on GPU (all weights on GPU). KV is piecewise
    # linear in ctx once sliding-window layers are in play, and the compute buffer
    # itself grows with ctx, so re-solve rather than divide by a per-token cost.
    def _gpu_room(c):
        # full offload, so every block is resident and the floor is at its largest
        t = compute_buffer_terms(cfg, c, n_ubatch, flash_attn, n_seq, kv_type)
        cb = compute_override_mib if compute_override_mib else \
            round(t["graph"] + t["floor_base"] + t["floor_per_layer"] * n_layers, 1)
        return eff_vram - block_weights_mib - rec_total_mib - cb
    max_ctx_gpu = max_ctx_for_kv_budget(cfg, kv_type, _gpu_room(ctx), n_ubatch,
                                        n_seq, flash_attn)
    for _ in range(4):                       # settle the compute-buffer feedback
        nxt = max_ctx_for_kv_budget(cfg, kv_type, _gpu_room(max(1, max_ctx_gpu)),
                                    n_ubatch, n_seq, flash_attn)
        if nxt == max_ctx_gpu:
            break
        max_ctx_gpu = nxt

    # Bandwidths are resolved BEFORE the dispatch, not with the roofline: the
    # two-plan default below compares the plans at the requested context, and it
    # must use the same numbers the rooflines print, or the pick would be made
    # on a speed the card never shows.
    # A number with no stated source is a silently wrong one, so the old made-up
    # 500/50 default is gone: a side whose bandwidth is unknown degrades the
    # roofline to n/a, and only the byte split survives - which is still true.
    bw_v, bw_r = bw_vram_gbs, bw_ram_gbs
    bw_src = {"vram": "given" if bw_v else None,
              "ram": "given" if bw_r else None}
    if None in bw_src.values():
        auto = cached_bandwidth()
        if bw_src["vram"] is None and auto.get("vram_gbs"):
            bw_v, bw_src["vram"] = auto["vram_gbs"], "auto"
        if bw_src["ram"] is None and auto.get("ram_gbs"):
            bw_r, bw_src["ram"] = auto["ram_gbs"], "auto"

    # Calls made before the ceiling/fit retag (old clients, stored sweep
    # configs) still say "speed" / "context"; the plans they name are the
    # ceiling and the fit, so read them as such instead of as "auto".
    plan_mode = {"speed": "ceiling", "context": "fit"}.get(plan_mode, plan_mode)
    plan_pick = None
    # An explicit knob value means "verify the config I actually ran": the pair
    # is costed exactly, whatever the plans above would have suggested. The
    # regime planners below only ever answer their own questions, so an override
    # routes away from them rather than being dropped - analyze() used to do
    # exactly that, returning the identical plan for -ngl 29 and -ngl 50.
    if cl["is_moe"]:
        if n_cpu_ffn_override is not None:
            warnings.append(
                "-ot pins DENSE FFN tensors (blk.N.ffn_gate/up/down), and this model "
                "routes its experts instead - there are no such tensors to pin, so the "
                "CPU FFN blocks value was dropped. --n-cpu-moe is this model's knob.")
            n_cpu_ffn_override = None
        plan = _plan_moe(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib, kv_per_layer_mib,
                         compute_mib, weights_mib, expert_total_mib, expert_layer_mean_mib,
                         per_layer_max_mib, per_layer_mean_mib, fully_fits, full_need,
                         max_ctx_gpu, ctx, kv_type, flash_attn, rec_total=rec_total_mib,
                         n_seq=n_seq, embed=embed_mib, output=output_mib,
                         ngl_override=gpu_layers_override, n_cpu_moe_override=n_cpu_moe_override,
                         compute_fn=compute_fn)
    elif gpu_layers_override is not None or n_cpu_ffn_override is not None:
        plan = _plan_dense(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib, kv_per_layer_mib,
                           compute_mib, weights_mib, embed_mib, output_mib,
                           per_layer_max_mib, per_layer_mean_mib, fully_fits, full_need,
                           max_ctx_gpu, ctx, kv_type, flash_attn,
                           ngl_override=gpu_layers_override, n_seq=n_seq,
                           n_cpu_ffn_override=n_cpu_ffn_override,
                           compute_fn=compute_fn, gpu_extra=gpu_extra_weights)
    elif not fully_fits and cl["ffn_dense_total"] > 0:
        # Two-plan regime: the model does not fit on the GPU at this context, so
        # there is no single answer - there is one per question being asked.
        #   * the CEILING plan keeps every layer's attention and KV on the GPU
        #     (ngl = all) and exiles every dense FFN to the CPU (-ot); its
        #     answer is the LARGEST context that still fits that way.
        #   * the FIT plan keeps the requested context pinned and walks -ot
        #     first, then -ngl, until it fits; its answer is the LEAST exile
        #     that does, and that is also the fastest way to hold this context.
        # Both plans load the same bytes, so the only thing that differs between
        # them at the requested context is HOW MANY weights sit in VRAM - and
        # that is exactly what the pick compares below.
        ceiling_plan = _plan_dense_ceiling(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib,
                                           weights_mib, ffn_dense_total_mib, rec_total_mib,
                                           max_ctx_gpu, ctx, kv_type, flash_attn,
                                           n_ubatch=n_ubatch, n_seq=n_seq,
                                           compute_fn=compute_fn, gpu_extra=gpu_extra_weights,
                                           compute_override_mib=compute_override_mib,
                                           layer_max=per_layer_max_mib, layer_mean=per_layer_mean_mib,
                                           ffn_layer_mean=ffn_layer_mean_mib)
        fit_plan = _plan_dense_fit(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib,
                                   weights_mib, embed_mib, output_mib,
                                   rec_total_mib, max_ctx_gpu, ctx, kv_type,
                                   flash_attn,
                                   n_seq=n_seq, compute_fn=compute_fn,
                                   gpu_extra=gpu_extra_weights,
                                   layer_max=per_layer_max_mib,
                                   layer_mean=per_layer_mean_mib,
                                   ffn_layer_mean=ffn_layer_mean_mib)
        result["plans"] = {"ceiling": ceiling_plan, "fit": fit_plan}
        # Which plan is REPORTED is the caller's question to ask; both are always
        # returned so the browser can toggle between them without a round trip.
        #
        # The default is the plan predicted FASTER AT THE REQUESTED CONTEXT.
        # The old rule keyed on coverage (the ceiling plan's max_ctx >= ctx),
        # but coverage only ever says which plan CAN answer the context, not
        # which one runs it faster - and when both can, the fit plan usually
        # wins, because it leaves the least FFN in RAM and streams the least
        # per token. Pricing both plans at the same context, with the same
        # fill, leaves the weight split as the only variable, so the
        # comparison is the question itself.
        c_fit = bool(ceiling_plan.get("vram_ok")) and bool(ceiling_plan.get("ram_ok", True))
        f_fit = bool(fit_plan.get("vram_ok")) and bool(fit_plan.get("ram_ok", True))
        # Both plans get priced at the requested context, pinned and auto alike:
        # the card shows both numbers either way, and an explicit pin is still
        # the answer to "which is faster here" even when it loses.
        def _at_ctx(p_):
            try:
                sp = _tps_at_ctx(p_, cl, cfg, ctx, kv_type, bw_v, bw_r, ram_eff, ctx_fill)
            except Exception as e:
                return {"error": "%s: %s" % (type(e).__name__, e), "ok": False}
            if "missing" in sp:
                sp["ok"] = False
                sides = " and ".join({"bw_vram_gbs": "VRAM",
                                      "bw_ram_gbs": "system RAM"}[m]
                                     for m in sp["missing"])
                sp["reason"] = "no bandwidth for %s, so this plan cannot be priced." % sides
            else:
                sp["ok"] = True
            return sp
        c_sp = _at_ctx(ceiling_plan)
        f_sp = _at_ctx(fit_plan)
        c_hi = c_sp.get("tok_s_hi") if "missing" not in c_sp else None
        f_hi = f_sp.get("tok_s_hi") if "missing" not in f_sp else None
        if plan_mode == "ceiling":
            mode = "ceiling"
        elif plan_mode == "fit":
            mode = "fit"
        elif not c_fit:
            # The ceiling plan cannot hold the context, so it cannot cover it -
            # the old auto rule's test, with the fit plan as the only remaining
            # answer (the infeasible-fit-alone case is unreachable: if the fit
            # plan cannot hold the context, neither can the ceiling one).
            mode = "fit"
        elif not f_fit:
            mode = "ceiling"
        elif c_hi is None and f_hi is None:
            mode = "fit"
        elif c_hi is None:
            mode = "fit"
        elif f_hi is None:
            mode = "ceiling"
        elif f_hi > c_hi:
            mode = "fit"
        elif c_hi > f_hi:
            mode = "ceiling"
        else:
            # Exact tie: the splits are identical, and the ceiling plan's
            # context is the more informative number to report.
            mode = "ceiling"
        result["plan_mode"] = mode
        plan_pick = {
            "ctx": ctx,
            "pick": mode,
            "ceiling": {"tok_s_lo": c_sp.get("tok_s_lo"), "tok_s_hi": c_hi,
                        "ok": c_sp.get("ok", False), "reason": c_sp.get("reason")},
            "fit": {"tok_s_lo": f_sp.get("tok_s_lo"), "tok_s_hi": f_hi,
                    "ok": f_sp.get("ok", False), "reason": f_sp.get("reason")},
        }
        if plan_mode in ("ceiling", "fit"):
            plan_pick["rule"] = "pinned"
            plan_pick["reason"] = "Pinned: %s." % plan_mode
        elif not c_fit:
            plan_pick["rule"] = "only_feasible"
            plan_pick["reason"] = "The ceiling plan does not load at %s - the fit plan is the only one that does." % f"{ctx:,}"
        elif not f_fit:
            plan_pick["rule"] = "only_feasible"
            plan_pick["reason"] = "The fit plan does not load at %s - the ceiling plan is the only one that does." % f"{ctx:,}"
        elif c_hi is None and f_hi is None:
            plan_pick["rule"] = "no_bandwidth"
            plan_pick["reason"] = "No bandwidth is recorded, so the split decides: the fit plan keeps more weights in VRAM and streams less per token."
        elif f_hi > c_hi:
            plan_pick["rule"] = "faster_at_ctx"
            plan_pick["reason"] = "Predicted faster at %s: %.1f vs %.1f tok/s." % (f"{ctx:,}", f_hi, c_hi)
        elif c_hi > f_hi:
            plan_pick["rule"] = "faster_at_ctx"
            plan_pick["reason"] = "Predicted faster at %s: %.1f vs %.1f tok/s." % (f"{ctx:,}", c_hi, f_hi)
        else:
            plan_pick["rule"] = "tie"
            plan_pick["reason"] = "Predicted equal at %s - the ceiling plan's context is the more informative number." % f"{ctx:,}"
        plan = result["plans"][mode]
    else:
        plan = _plan_dense(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib, kv_per_layer_mib,
                           compute_mib, weights_mib, embed_mib, output_mib,
                           per_layer_max_mib, per_layer_mean_mib, fully_fits, full_need,
                           max_ctx_gpu, ctx, kv_type, flash_attn,
                           ngl_override=gpu_layers_override, n_seq=n_seq,
                           compute_fn=compute_fn, gpu_extra=gpu_extra_weights)
    # The projector and the MTP draft cache were held out of eff_vram while planning,
    # so the split search never had to think about them. Fold them back into the
    # reported totals - but into the BUDGET as well as the usage, or the two numbers
    # end up on different bases and a plan that fits reads as though it does not.
    # On gemma-4-31B with its 1145 MiB projector that showed as "11045 / 10070" next
    # to a green verdict, which is exactly the sort of thing that makes a user stop
    # trusting the tool.
    # In the two-plan regime every plan is folded the same way, so the toggle
    # shows each one on exactly the basis the selected one is reported on.
    targets = list(result["plans"].values()) if "plans" in result else [plan]
    held_out = 0.0        # folded back into the GPU totals
    host_extra = 0.0      # charged to RAM instead, when the projector sits there
    if mmproj_mib > 0:
        for p_ in targets:
            p_["mmproj_mib"] = mmproj_mib
            p_["mmproj_place"] = mmproj_place
        if mmproj_on_gpu:
            held_out += mmproj_mib
        else:
            host_extra += mmproj_mib
    if spec_mib > 0:
        for p_ in targets:
            p_["spec_mib"] = spec_mib
        held_out += spec_mib
    if vis_mib > 0:
        for p_ in targets:
            p_["vision_mib"] = vis_mib
        # The encoder's transients land wherever the tower runs, which is the same
        # side its weights were placed on.
        if mmproj_on_gpu:
            held_out += vis_mib
        else:
            host_extra += vis_mib
    for p_ in targets:
        if held_out > 0:
            p_["vram_used_mib"] = p_.get("vram_used_mib", 0.0) + held_out
            if p_.get("vram_budget_mib") is not None:
                p_["vram_budget_mib"] = p_["vram_budget_mib"] + held_out
        if host_extra > 0:
            p_["ram_used_mib"] = p_.get("ram_used_mib", 0.0) + host_extra
        # Recorded on the plan so the verdict and the UI read one budget rather
        # than each reaching for a different field.
        p_["ram_budget_mib"] = ram_budget_mib
        p_["verdict"] = _verdict(p_, result["inputs"], ram_free_mib)
        # The projector's placement is a launch flag, not a split, so it is
        # stamped onto the command here rather than threaded through every
        # planner - none of which has an opinion about where it goes.
        if include_mmproj and not mmproj_on_gpu and p_.get("llama_cmd"):
            p_["llama_cmd"] = p_["llama_cmd"] + " --no-mmproj-offload"
    result["inputs"]["plan_mode"] = result.get("plan_mode")
    result["plan"] = plan
    result["plan_pick"] = plan_pick

    # ---- speed roofline ----------------------------------------------------
    def _roofline(p_):
        # The plan carries its own placement: n_gpu_layers says where the blocks
        # are, n_cpu_ffn says which of their dense FFN tensors -ot exiled to
        # RAM. per_token_bytes() charges each byte to the side that streams it
        # every token, and a plan is scored on its OWN split - it needs only the
        # bandwidth of the side it actually reads from, and nothing about the
        # other plan.
        try:
            ngl = p_.get("n_gpu_layers")
            if ngl is None:
                ngl = n_layers if p_.get("fits_fully") else 0
            n_cpu_ffn = p_.get("n_cpu_ffn") or 0
            gpu_blocks = list(range(max(0, n_layers - int(ngl)), n_layers))
            # A ceiling plan's context IS its answer, so its roofline is read at
            # the window it actually proposes rather than at the one that was typed.
            plan_ctx = int(p_.get("max_ctx") or ctx)
            fill = ctx_fill if ctx_fill is not None else min(plan_ctx, 8192)
            sp = estimate_speed(cfg, cl, gpu_blocks, fill, kv_type,
                                bw_v, bw_r,
                                cpu_head=True,   # the head stays in RAM at every -ngl
                                ram_eff=ram_eff, n_cpu_moe=p_.get("n_cpu_moe", 0) or 0,
                                n_cpu_ffn=n_cpu_ffn)
            sp["n_gpu_layers"] = int(ngl)
            sp["n_cpu_moe"] = p_.get("n_cpu_moe", 0) or 0
            sp["n_cpu_ffn"] = n_cpu_ffn
            sp["bw_vram_source"] = bw_src["vram"] or "none"
            sp["bw_ram_source"] = bw_src["ram"] or "none"
            sp["bw_auto"] = bw_note or ""
            if "missing" in sp:
                sp["ok"] = False
                sides = " and ".join({"bw_vram_gbs": "VRAM",
                                      "bw_ram_gbs": "system RAM"}[m]
                                     for m in sp["missing"])
                sp["reason"] = ("no bandwidth for %s, so there is no honest number. "
                                "The per-token byte split above is still valid - "
                                "enter a bandwidth to get a speed." % sides)
            else:
                sp["ok"] = True
            sp["notes"] = _speed_notes(sp, flash_attn, bool(mtp_spec or dflash))
            return sp
        except Exception as e:
            return {"error": "%s: %s" % (type(e).__name__, e), "ok": False}

    # Every plan carries its own, because the two plans run at genuinely different
    # speeds and the browser toggles between them WITHOUT asking again - a single
    # top-level roofline would keep showing the selected plan's number under the
    # other plan's split.
    for p_ in targets:
        p_["speed"] = _roofline(p_)
    result["speed"] = plan["speed"]

    # ---- vision warnings ---------------------------------------------------
    if vis_cfg and vis_peak:
        warnings.append("Vision encoder transients are DERIVED, NOT MEASURED - no sweep backs "
                        "them, unlike every other term here. Treat %.0f MiB as an order of "
                        "magnitude." % vis_peak["total_mib"])
        if vis_peak["scores_mib"] > 0.5 * vis_peak["total_mib"]:
            warnings.append("At %dx%d that image is %s patches, and the %s attention scores "
                            "over them are %.0f MiB of the %.0f MiB peak. This term is "
                            "QUADRATIC in pixels: halving each dimension cuts it ~4x."
                            % (vis_grid["width"], vis_grid["height"],
                               f"{vis_grid['n_patches']:,}", vis_cfg["projector"],
                               vis_peak["scores_mib"], vis_peak["total_mib"]))
        # The fused/unfused gap is the dominant uncertainty, so name it explicitly
        # rather than letting a single number imply more confidence than there is.
        if vis_ceiling and vis_ceiling["total_mib"] > 1.5 * vis_peak["total_mib"]:
            warnings.append("Reserved %.0f MiB assuming the vision tower FUSES attention. If it "
                            "does not, the peak is %.0f MiB instead - a %.0fx swing, and the "
                            "whole uncertainty in this estimate. Head dim is %d; CUDA supports "
                            "it (fattn.cu) but not on the tensor-core path. Check your load log "
                            "for 'flash attention is enabled' to settle it for your machine."
                            % (vis_peak["total_mib"], vis_ceiling["total_mib"],
                               vis_ceiling["total_mib"] / max(1.0, vis_peak["total_mib"]),
                               vis_cfg.get("head_dim") or 0))
        if vis_grid["image_tokens"] > max(1, ctx) * 0.25:
            warnings.append("Each %dx%d image also adds %s tokens to the context - %.0f%% of "
                            "the %s you configured. The KV and compute cost of those tokens "
                            "is on top of the encoder peak."
                            % (vis_grid["width"], vis_grid["height"],
                               f"{vis_grid['image_tokens']:,}",
                               100.0 * vis_grid["image_tokens"] / max(1, ctx), f"{ctx:,}"))
    elif vis_cfg and not image_px:
        warnings.append("%s carries a vision encoder, but no image size was given, so this is "
                        "a TEXT-ONLY plan: the projector's %.0f MiB of weights are charged and "
                        "its encoder transients are not. Pass an image size to reserve room for "
                        "them - at screenshot resolutions they are the larger of the two."
                        % (mmproj["name"], mmproj_mib))

    if mmproj and not include_mmproj:
        warnings.append("A vision projector (%s, %.0f MiB) sits next to this model. LM Studio "
                        "loads it onto the GPU with the model and counts it in the size it shows. "
                        "It is NOT included in this plan - tick 'Load vision projector' to count it."
                        % (mmproj["name"], _mib(mmproj["tensor_bytes"])))

    # hardware-tied warnings comparing the plan to the *dedicated* VRAM and *live free* RAM
    vram_used = plan.get("vram_used_mib", 0)
    ram_used = plan.get("ram_used_mib", 0)
    if vram_used > vram_budget_mib:
        warnings.append("Plan needs %.0f MiB VRAM but only %.0f MiB is available - the overflow "
                        "spills into shared GPU memory (system RAM used as VRAM), which is very slow. "
                        "Lower layers or context." % (vram_used, vram_budget_mib))
    if ram_free_mib and ram_used > ram_free_mib and ram_used <= ram_budget_mib:
        warnings.append("Plan needs %.0f MiB RAM but only %.0f MiB is free right now - about %.0f MiB "
                        "will come from standby/compression/paging. It loads (LM Studio will push RAM "
                        "toward 100%%) but gets slower. Close apps for headroom."
                        % (ram_used, ram_free_mib, ram_used - ram_free_mib))
    return result


def _speed_notes(sp, flash_attn, has_spec):
    """Where the speed model is thin, said out loud.

    The byte counts are exact, and the bandwidth uncertainty is already visible
    as the bracket width, so these name the parts of the machine the estimate
    does not cover at all - the UI prints them next to the number rather than
    letting a clean bracket imply more coverage than there is."""
    if "error" in sp or sp.get("missing"):
        return []
    notes = []
    if not flash_attn:
        notes.append("Flash attention is off, but the estimate assumes the tiled "
                     "KV read path - expect the low end of the bracket.")
    if has_spec:
        notes.append("Speculative decoding is not modelled: the number is the "
                     "per-accepted-token floor, before rejection overhead.")
    notes.append("Decode only, bandwidth bound: the compute term (FLOPs) is not "
                 "modelled, so a small active model or a very short context can "
                 "run slower than this.")
    return notes


def _tps_at_ctx(plan_, cl, cfg, ctx, kv_type, bw_v, bw_r, ram_eff, ctx_fill=None):
    """A plan's predicted tok/s at the REQUESTED ctx, priced on its own split.

    Each plan's card roofline is priced at the context that plan proposes, so
    the two numbers on screen answer different questions and cannot be
    compared. This prices both plans at the context the user asked for, with
    the same fill, leaving the byte split - weights on GPU vs weights in RAM -
    as the only thing that differs. It must stay a mirror of analyze()'s
    roofline estimate_speed() call (same cpu_head, same ram_eff), or the
    number it uses to choose would not be the number the card prints."""
    n_layers = cfg["n_layers"] or 0
    ngl = plan_.get("n_gpu_layers")
    if ngl is None:
        ngl = n_layers if plan_.get("fits_fully") else 0
    fill = ctx_fill if ctx_fill is not None else min(ctx, 8192)
    return estimate_speed(cfg, cl, list(range(max(0, n_layers - int(ngl)), n_layers)),
                          fill, kv_type, bw_v, bw_r,
                          cpu_head=True,
                          ram_eff=ram_eff, n_cpu_moe=plan_.get("n_cpu_moe", 0) or 0,
                          n_cpu_ffn=plan_.get("n_cpu_ffn") or 0)


def _llama_flags(ctx, kv_type, flash_attn, ngl=None, n_cpu_moe=None,
                 ot_all_experts=False, n_cpu_ffn=None):
    parts = ["llama-server", "-m <model.gguf>"]
    parts.append("-ngl %s" % (ngl if ngl is not None else 999))
    if n_cpu_ffn:
        # Same spelling sweep.build_argv() and the launchers emit - one source.
        parts.append('-ot "%s"' % ot_regex(n_cpu_ffn))
    if n_cpu_moe is not None:
        parts.append("--n-cpu-moe %d" % n_cpu_moe)
    if ot_all_experts:
        parts.append(r'-ot "\.ffn_.*_exps\.=CPU"')
    parts.append("-c %d" % ctx)
    if flash_attn or kv_type != "f16":
        parts.append("-fa")
    if kv_type != "f16":
        parts.append("-ctk %s -ctv %s" % (kv_type, kv_type))
    return " ".join(parts)


def _dense_cost(cfg, cl, ngl, n_cpu_ffn, layer_max, layer_mean, ffn_layer_mean):
    """Exact BLOCK weight bytes on each side for a (-ngl, n_cpu_ffn) pair.

    The single source of dense block-weight accounting: the classic split, the
    verification path, and the ceiling/fit planners all read through this,
    because blocks are not interchangeable (hybrid attn vs SSM, quant varies
    per block) and the -ot pin moves tens out of blocks that STAY on the GPU.
    llama.cpp offloads the LAST n_gpu_layers blocks, so which blocks land on
    the GPU matters. Embeddings, the output head, KV and recurrent state are
    not here - the caller combines them with what it already knows."""
    n_layers = cfg["n_layers"] or 0
    ngl = max(0, min(int(ngl), n_layers))
    n_cpu_ffn = max(0, min(int(n_cpu_ffn or 0), n_layers))
    per_layer = cl.get("per_layer_bytes") or {}
    per_ffn   = cl.get("per_layer_ffn_bytes") or {}
    on_gpu = list(range(max(0, n_layers - ngl), n_layers))
    s = set(on_gpu)
    gpu_b = cpu_b = 0.0
    for i in range(n_layers):
        b = per_layer.get(i, 0.0)
        # A pinned block that is resident hands its FFN to the CPU only; a
        # pinned block already on the CPU carries the FFN there with the rest.
        f = min(b, per_ffn.get(i, 0.0)) if (i < n_cpu_ffn and i in s) else 0.0
        if i in s:
            gpu_b += b - f
        else:
            cpu_b += b
        cpu_b += f
    if per_layer:
        gpu_w, cpu_w = _mib(gpu_b), _mib(cpu_b)
    else:
        # metadata-only parse: no per-block table, so the pin costs the MEAN
        # FFN per pinned block that is resident - the declared approximation
        # the classic path uses for everything else.
        first = max(0, n_layers - ngl)
        n_pin = sum(1 for i in range(min(n_cpu_ffn, n_layers)) if i >= first)
        gpu_w = min(ngl, n_layers) * layer_max - n_pin * ffn_layer_mean
        cpu_w = (n_layers - ngl) * layer_mean + n_pin * ffn_layer_mean
    return {"n_gpu_layers": ngl, "n_cpu_ffn": n_cpu_ffn, "on_gpu": on_gpu,
            "cpu_blocks": [i for i in range(n_layers) if i not in s],
            "gpu_block_mib": gpu_w, "cpu_block_mib": cpu_w}


def _plan_dense(cfg, cl, eff_vram, ram, kv_total, kv_layer, compute, weights,
                embed, output, layer_max, layer_mean, fully_fits, full_need,
                max_ctx_gpu, ctx, kv_type, flash_attn, ngl_override=None, n_seq=1,
                n_cpu_ffn_override=None,
                compute_fn=None, gpu_extra=None):
    n_layers = cfg["n_layers"]
    gpu_extra = gpu_extra or (lambda ngl: 0.0)

    def build(ngl, forced, n_cpu_ffn=0):
        ngl = max(0, min(int(ngl), n_layers))
        n_cpu_ffn = max(0, min(int(n_cpu_ffn or 0), n_layers))
        full = (ngl >= n_layers)
        cost = _dense_cost(cfg, cl, ngl, n_cpu_ffn, layer_max, layer_mean, 0.0)
        # A -ot pin splits the graph even at full -ngl, so the surcharge has to
        # know about it; passing only ngl would read every FFN pin as unsplit.
        cb = (compute_fn(ngl, any_on_cpu=graph_is_split(n_layers, ngl, 0, n_cpu_ffn))
              if compute_fn else {"gpu": compute, "cpu": 0.0})
        # The output head always stays in system RAM; the token embeddings follow
        # gpu_extra() - see the note where embed_on_gpu_mib is derived.
        extra = gpu_extra(ngl)
        on_gpu, cpu_blocks = cost["on_gpu"], cost["cpu_blocks"]
        gpu_w  = cost["gpu_block_mib"] + extra
        cpu_w  = cost["cpu_block_mib"] + embed + output - extra
        if full:
            gpu_kv, cpu_kv = kv_total, 0.0
        else:
            gpu_kv = _mib(kv_bytes_total(cfg, kv_type, on_gpu))
            cpu_kv = kv_total - gpu_kv
        gpu_rec = _mib(recurrent_bytes(cfg, on_gpu, n_seq))
        cpu_rec = _mib(recurrent_bytes(cfg, cpu_blocks, n_seq))
        cpu_layers = n_layers - ngl
        vram_used = gpu_w + gpu_kv + gpu_rec + cb["gpu"]
        ram_used = cpu_w + cpu_kv + cpu_rec + cb["cpu"]
        ram_ok = ram_used <= ram
        vram_ok = vram_used <= eff_vram
        if full:
            head = "All %d layers on GPU." % n_layers
            if not forced:
                head += " Room for up to ~%s tokens of context." % f"{max_ctx_gpu:,}"
        else:
            head = "Split: %d of %d layers on GPU, %d on CPU." % (ngl, n_layers, cpu_layers)
        if forced:
            head = "Verifying your config - not a recommendation. " + head
            if n_cpu_ffn:
                head += " The first %d blocks' dense FFN sits on the CPU (-ot)." % n_cpu_ffn
        if forced and not vram_ok:
            head += "  WARNING: needs %.0f MiB VRAM (> %.0f budget) - spills to shared memory (slow)." % (
                vram_used, eff_vram)
        if not ram_ok:
            head += "  WARNING: needs %.0f MiB RAM (> %.0f budget)." % (ram_used, ram)
        ls = ["GPU Offload / GPU Layers: %s" % ("max (all %d)" % n_layers if full else ngl),
              "Context Length: %d" % ctx,
              "Offload KV Cache to GPU Memory: ON (keeps GPU-layer KV in VRAM - the default)",
              "Limit to Dedicated GPU Memory: ON (avoid slow shared-memory spill)",
              ("Flash Attention: ON  (KV @ %s)" % kv_type) if kv_type != "f16"
              else "Flash Attention: optional"]
        if n_cpu_ffn:
            ls.append("Dense FFN of the first %d blocks pinned to the CPU via -ot "
                      "(LM Studio has no dense-FFN-only toggle - run the llama.cpp "
                      "command below)." % n_cpu_ffn)
        return {
            "kind": "dense", "fits_fully": full, "n_gpu_layers": ngl, "cpu_layers": cpu_layers,
            "n_cpu_ffn": n_cpu_ffn,
            "forced_ngl": forced, "vram_ok": vram_ok,
            "vram_used_mib": vram_used, "vram_budget_mib": eff_vram,
            "ram_used_mib": ram_used, "ram_ok": ram_ok, "max_ctx_gpu": max_ctx_gpu,
            "gpu_weights_mib": gpu_w, "gpu_kv_mib": gpu_kv, "compute_mib": cb["gpu"],
            "cpu_compute_mib": cb["cpu"], "gpu_recurrent_mib": gpu_rec,
            "cpu_weights_mib": cpu_w, "cpu_kv_mib": cpu_kv, "cpu_recurrent_mib": cpu_rec,
            "gpu_attn_layers": 0 if full else sum(
                1 for i in on_gpu if i in set(cfg["attn_layers"])),
            "n_attn_layers": len(cfg["attn_layers"]),
            "lmstudio": ls,
            "llama_cmd": _llama_flags(ctx, kv_type, flash_attn, ngl=(99 if full else ngl),
                                      n_cpu_ffn=n_cpu_ffn or None),
            "headline": head,
        }

    # An explicit knob value verifies the exact config it names: with only one
    # typed, the other takes the full-GPU extreme, the same rule _plan_moe uses.
    if ngl_override is not None or n_cpu_ffn_override is not None:
        ngl = n_layers if ngl_override is None else ngl_override
        ncpu = 0 if n_cpu_ffn_override is None else n_cpu_ffn_override
        return build(ngl, True, ncpu)
    if fully_fits:
        return build(n_layers, False)
    # Blocks are not interchangeable (hybrid attn vs SSM, and quant varies per
    # block), so search downward for the largest ngl that actually fits rather
    # than dividing by an average. No -ot pin: plain layer counts are what this
    # planner answers; the regime planners own the pinned-FFN question.
    for ngl in range(n_layers, -1, -1):
        p = build(ngl, False)
        if p["vram_used_mib"] <= eff_vram:
            return p
    return build(0, False)


def _plan_dense_ceiling(cfg, cl, eff_vram, ram, kv_total, weights, ffn_total,
                        rec_total, max_ctx_gpu, ctx, kv_type, flash_attn, n_ubatch, n_seq,
                        compute_fn, gpu_extra, compute_override_mib,
                        layer_max, layer_mean, ffn_layer_mean):
    """Plan the CEILING: every layer's attention and KV on the GPU (ngl = all),
    every dense FFN exiled to the CPU (-ot). The answer is the LARGEST context
    that still fits that way.

    This is the regime in which decode pays nothing for KV over PCIe - the
    cache re-read every token stays in VRAM - at the price of streaming the
    dense FFN from system RAM every token (per_token_bytes() charges exactly
    that). It is the dense answer to the MoE expert split: the only weights big
    enough to be worth moving out of VRAM. The FFN count is pinned to ALL on
    purpose - buying any of it back would spend bytes the context is here to
    get, and the speed sweep (which sweeps context at this exact pin) finds
    where the regime actually stops being fast."""
    n_layers = cfg["n_layers"] or 0
    n_cpu_ffn = n_layers
    gpu_extra = gpu_extra or (lambda ngl: 0.0)
    cost = _dense_cost(cfg, cl, n_layers, n_cpu_ffn, layer_max, layer_mean, ffn_layer_mean)
    extra = gpu_extra(n_layers)
    head_ram = _mib(cl["embed_bytes"] + cl["output_bytes"]) - extra
    # What VRAM holds besides KV and compute: the non-FFN part of every block
    # plus the embeddings the measured rule keeps on the GPU at any -ngl > 0.
    fixed = cost["gpu_block_mib"] + extra + rec_total

    def cb_split_at(c):
        # The graph spans both backends - the FFN tensors sit in RAM while their
        # blocks' attention runs on the GPU - so the split surcharge applies even
        # though the layer count is full.
        t = compute_buffer_terms(cfg, c, n_ubatch, flash_attn, n_seq, kv_type)
        return compute_buffer_split(t, True, True, n_layers, compute_override_mib)

    def cb_at(c):
        return cb_split_at(c)["gpu"]

    def kv_at(c):
        return _mib(kv_bytes_total_at(cfg, kv_type, c, n_ubatch, n_seq, flash_attn))

    def vram_at(c):
        return fixed + cb_at(c) + kv_at(c)

    # The largest context whose OWN total lands under the budget.
    #
    # This used to be a fixed point over max_ctx_for_kv_budget(): solve KV
    # against the room left by the compute buffer, recompute the buffer at the
    # answer, repeat. But the buffer grows with ctx, so the iteration can settle
    # on a context that costs a little more than the budget it was solved
    # against - and this number is not a hint, it is the plan, so "a little
    # more" is a plan that does not fit reported as one that does.
    #
    # vram_at() is monotone non-decreasing in ctx (KV is piecewise linear -
    # sliding-window layers cap at their window - and every other term is flat
    # or rising), so a bisection on the total itself is exact and needs no
    # settling. The KV solver still supplies the opening bracket, which is what
    # keeps this to ~20 evaluations of pure arithmetic.
    seed = max_ctx_for_kv_budget(
        cfg, kv_type, max(0.0, eff_vram - fixed - cb_at(ctx)),
        n_ubatch, n_seq, flash_attn)
    hi = max(int(seed) * 2, int(ctx), 1)
    # A context past what the model was trained on is not a config, whatever the
    # arithmetic says - the sweep clamps it the same way.
    if cfg["n_ctx_train"]:
        hi = min(hi, int(cfg["n_ctx_train"]))
    if hi < 1 or vram_at(1) > eff_vram:
        max_ctx = 0
    else:
        lo = 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if vram_at(mid) <= eff_vram:
                lo = mid
            else:
                hi = mid - 1
        max_ctx = lo

    # The plan is priced at the context it PROPOSES, not at the one that was
    # typed. Those are different numbers by construction - max_ctx is the answer
    # to "how much context can I have this way" - and pricing the answer at the
    # question produced a plan that said "87,194 tokens fit" above a bar reading
    # 161% OVER, because the bar was the footprint of the 131,072 that did not.
    # Below zero there is no proposal to price, so the requested context stands
    # and the overflow it shows is the honest reason the regime is infeasible.
    priced_ctx = max_ctx if max_ctx > 0 else ctx
    cb = cb_split_at(priced_ctx)
    vram_used = vram_at(priced_ctx)
    vram_ok = max_ctx > 0 and vram_used <= eff_vram
    ffn_cpu_w = cost["cpu_block_mib"] + head_ram
    ram_used = ffn_cpu_w + cb["cpu"]
    ram_ok = ram_used <= ram

    if max_ctx <= 0:
        head = ("No context fits with every layer on the GPU: attention weights alone are "
                "%.0f MiB, and with the compute buffer and recurrent state that is %.0f MiB "
                "before a single token of KV. This plan is infeasible on this card - the "
                "fit plan trades layers for context instead."
                % (cost["gpu_block_mib"] + extra, fixed + cb_at(1)))
    else:
        head = ("All %d layers' attention and KV on the GPU; every dense FFN on the CPU (-ot). "
                "Largest context that still fits that way: ~%s tokens."
                % (n_layers, f"{max_ctx:,}"))
    if not ram_ok:
        head += "  WARNING: FFN + head need %.0f MiB RAM (> %.0f budget)." % (ram_used, ram)
    ls = ["GPU Offload / GPU Layers: max (all %d)" % n_layers,
          "Every dense FFN pinned to the CPU via -ot (LM Studio has no dense-FFN-only "
          "toggle - run the llama.cpp command below).",
          "Context Length: up to ~%s tokens" % f"{max_ctx:,}"]
    return {
        "kind": "dense_ceiling", "mode": "ceiling", "fits_fully": False,
        "n_gpu_layers": n_layers, "cpu_layers": 0, "n_cpu_ffn": n_cpu_ffn,
        "vram_ok": vram_ok,
        "vram_used_mib": vram_used, "vram_budget_mib": eff_vram,
        "ram_used_mib": ram_used, "ram_ok": ram_ok,
        "max_ctx": max_ctx, "max_ctx_gpu": max_ctx_gpu,
        "gpu_weights_mib": cost["gpu_block_mib"] + extra,
        "gpu_kv_mib": kv_at(priced_ctx),
        "compute_mib": cb["gpu"], "cpu_compute_mib": cb["cpu"],
        "gpu_recurrent_mib": rec_total,
        "cpu_weights_mib": ffn_cpu_w, "cpu_kv_mib": 0.0, "cpu_recurrent_mib": 0.0,
        "forced_ngl": False,
        "lmstudio": ls,
        "llama_cmd": _llama_flags(priced_ctx, kv_type, flash_attn, ngl=99,
                                  n_cpu_ffn=n_layers),
        "headline": head,
    }


def _plan_dense_fit(cfg, cl, eff_vram, ram, kv_total, weights, embed, output,
                    rec_total, max_ctx_gpu, ctx, kv_type, flash_attn, n_seq,
                    compute_fn, gpu_extra, layer_max, layer_mean, ffn_layer_mean):
    """Plan the FIT: the requested context is pinned and it is the thing being
    protected. Two knobs pay for it, in this order:

      1. The dense FFN is exiled to the CPU (-ot), and the search asks how
         LITTLE of it has to go. Exiling FFN frees VRAM without costing any KV,
         so with every block still on the GPU the answer is the SMALLEST
         n_cpu_ffn that fits - exactly the shape of the MoE planner's
         --n-cpu-moe search, and for the same reason: those are the only weights
         big enough to be worth moving, and moving them does not drag KV along.
      2. Only when even a full FFN exile is not enough does -ngl come down. That
         is the expensive step - a whole block leaving the GPU takes its KV with
         it - so it is the fallback, not the first move.

    The old planner pinned n_cpu_ffn at every block unconditionally and went
    straight to walking -ngl. On a card with room to spare that left VRAM unused
    and streamed FFN weights over PCIe every token for no reason.

    The sibling (ceiling plan) pins the other way: -ngl and the FFN exile are
    both at maximum and the CONTEXT is what is free."""
    n_layers = cfg["n_layers"] or 0
    gpu_extra = gpu_extra or (lambda ngl: 0.0)

    def build(ngl, n_cpu_ffn):
        cost = _dense_cost(cfg, cl, ngl, n_cpu_ffn, layer_max, layer_mean, ffn_layer_mean)
        cb = (compute_fn(ngl, any_on_cpu=graph_is_split(n_layers, ngl, 0, n_cpu_ffn))
              if compute_fn else {"gpu": 0.0, "cpu": 0.0})
        extra = gpu_extra(ngl)
        gpu_w = cost["gpu_block_mib"] + extra
        cpu_w = cost["cpu_block_mib"] + embed + output - extra
        if ngl >= n_layers:
            gpu_kv, gpu_rec = kv_total, rec_total
        else:
            gpu_kv  = _mib(kv_bytes_total(cfg, kv_type, cost["on_gpu"]))
            gpu_rec = _mib(recurrent_bytes(cfg, cost["on_gpu"], n_seq))
        vram_used = gpu_w + gpu_kv + gpu_rec + cb["gpu"]
        ram_used = cpu_w + (kv_total - gpu_kv) + (rec_total - gpu_rec) + cb["cpu"]
        return cost, cb, extra, gpu_w, cpu_w, gpu_kv, gpu_rec, vram_used, ram_used

    # Phase 1: every block on the GPU, and as much FFN kept there as fits.
    # Ascending, so the first hit is the LEAST FFN exiled - the most weight left
    # in VRAM. n_cpu_ffn=0 is "nothing exiled at all", which is the right answer
    # whenever the whole model fits at this context.
    best = None
    n_cpu_ffn = n_layers
    for f in range(0, n_layers + 1):
        c = build(n_layers, f)
        if c[7] <= eff_vram:
            best, n_cpu_ffn = (n_layers, c), f
            break
    # Phase 2: a full FFN exile still does not fit, so blocks have to go too.
    # Blocks are not interchangeable, so search downward for the largest ngl
    # that does - the same rule the classic planner uses.
    if best is None:
        n_cpu_ffn = n_layers
        for ngl in range(n_layers - 1, -1, -1):
            c = build(ngl, n_cpu_ffn)
            if c[7] <= eff_vram:
                best = (ngl, c)
                break
    if best is None:
        ngl, c = 0, build(0, n_cpu_ffn)
        cost, cb, extra, gpu_w, cpu_w, gpu_kv, gpu_rec, vram_used, ram_used = c
        head = ("The context cannot fit even with nothing on the GPU: %.0f MiB of KV alone "
                "exceeds the %.0f MiB RAM budget plus whatever the card keeps. Lower the "
                "context or the KV quant." % (kv_total, ram))
        overflow = True
    else:
        ngl, c = best
        cost, cb, extra, gpu_w, cpu_w, gpu_kv, gpu_rec, vram_used, ram_used = c
        if n_cpu_ffn <= 0:
            head = ("Context %s pinned: all %d layers on the GPU, whole - nothing "
                    "had to be exiled to fit it." % (f"{ctx:,}", n_layers))
        elif ngl >= n_layers:
            head = ("Context %s pinned: all %d layers on the GPU, with the first %d "
                    "blocks' dense FFN on the CPU (-ot) to pay for it. The other %d "
                    "keep their FFN in VRAM."
                    % (f"{ctx:,}", n_layers, n_cpu_ffn, n_layers - n_cpu_ffn))
        else:
            head = ("Context %s pinned: %d of %d layers on GPU, every dense FFN on "
                    "the CPU (-ot). The other %d layers run on the CPU, KV and all "
                    "- that is the price of the context."
                    % (f"{ctx:,}", ngl, n_layers, n_layers - ngl))
        overflow = False
    if not (ram_used <= ram):
        head += "  WARNING: needs %.0f MiB RAM (> %.0f budget)." % (ram_used, ram)
    ls = ["GPU Offload / GPU Layers: %s"
          % ("max (all %d)" % n_layers if ngl >= n_layers else ngl)]
    if n_cpu_ffn:
        ls.append("The first %d blocks' dense FFN pinned to the CPU via -ot (LM Studio "
                  "has no dense-FFN-only toggle - run the llama.cpp command below)."
                  % n_cpu_ffn)
    ls.append("Context Length: %s tokens" % f"{ctx:,}")
    return {
        "kind": "dense_fit", "mode": "fit", "fits_fully": False,
        "n_gpu_layers": ngl, "cpu_layers": n_layers - ngl, "n_cpu_ffn": n_cpu_ffn,
        "kv_overflow": overflow or None,
        "vram_ok": vram_used <= eff_vram,
        "vram_used_mib": vram_used, "vram_budget_mib": eff_vram,
        "ram_used_mib": ram_used, "ram_ok": ram_used <= ram,
        "max_ctx": ctx, "max_ctx_gpu": max_ctx_gpu,
        "gpu_weights_mib": gpu_w, "gpu_kv_mib": gpu_kv, "compute_mib": cb["gpu"],
        "cpu_compute_mib": cb["cpu"], "gpu_recurrent_mib": gpu_rec,
        "cpu_weights_mib": cpu_w, "cpu_kv_mib": kv_total - gpu_kv,
        "cpu_recurrent_mib": rec_total - gpu_rec,
        "forced_ngl": False,
        "lmstudio": ls,
        "llama_cmd": _llama_flags(ctx, kv_type, flash_attn,
                                  ngl=(99 if ngl >= n_layers else ngl),
                                  n_cpu_ffn=n_cpu_ffn or None),
        "headline": head,
    }


def _plan_moe(cfg, cl, eff_vram, ram, kv_total, kv_layer, compute, weights,
              expert_total, expert_layer_mean, layer_max, layer_mean,
              fully_fits, full_need, max_ctx_gpu, ctx, kv_type, flash_attn,
              rec_total=0.0, n_seq=1, ngl_override=None, n_cpu_moe_override=None,
              embed=0.0, output=0.0, compute_fn=None):
    """Plan an MoE split.

    Two knobs, and they are not the same one:
      * -ngl N          -> the last N blocks live on the GPU (attention + KV too)
      * --n-cpu-moe M   -> the routed experts of the FIRST M blocks go to the CPU

    The efficient MoE config is ngl = all blocks (so every layer's attention and
    KV stay on the GPU) plus the smallest --n-cpu-moe that fits, because experts
    are the only weights big enough to be worth moving and only 8-of-256 of them
    run per token. Whole-layer offload is the fallback for when even that fails.

    Everything is summed from the real per-block tensor bytes - expert blocks are
    not interchangeable, and on a hybrid MoE only some blocks carry KV at all.
    """
    n_layers = cfg["n_layers"]
    n_exp_layers = cl["n_expert_layers"]
    per_layer = cl.get("per_layer_bytes") or {}
    per_expert = cl.get("per_layer_expert_bytes") or {}

    def cost(ngl, n_cpu_moe):
        """Exact GPU/CPU split for a given (-ngl, --n-cpu-moe) pair."""
        ngl = max(0, min(int(ngl), n_layers))
        n_cpu_moe = max(0, min(int(n_cpu_moe), n_layers))
        on_gpu = list(range(n_layers - ngl, n_layers))
        gpu_w = cpu_w = 0.0
        for i in range(n_layers):
            b = per_layer.get(i, 0)
            e = per_expert.get(i, 0)
            if i in set(on_gpu):
                # experts of the first n_cpu_moe blocks are pinned to the CPU
                if i < n_cpu_moe:
                    gpu_w += b - e
                    cpu_w += e
                else:
                    gpu_w += b
            else:
                cpu_w += b
        gpu_w, cpu_w = _mib(gpu_w), _mib(cpu_w)
        # embeddings and the output head stay in system RAM at every layer count -
        # see the note in _plan_dense.build()
        cpu_w += embed + output
        gpu_kv = _mib(kv_bytes_total(cfg, kv_type, on_gpu))
        gpu_rec = _mib(recurrent_bytes(cfg, on_gpu, n_seq))
        rest = [i for i in range(n_layers) if i not in set(on_gpu)]
        # experts pinned to the CPU also make the CPU run graph ops, even at full -ngl
        cb = (compute_fn(ngl, any_on_cpu=graph_is_split(n_layers, ngl, n_cpu_moe))
              if compute_fn else {"gpu": compute, "cpu": 0.0})
        return {
            "n_gpu_layers": ngl, "n_cpu_moe": n_cpu_moe,
            "experts_on_gpu": max(0, ngl - max(0, n_cpu_moe - (n_layers - ngl))),
            "gpu_weights_mib": gpu_w, "gpu_kv_mib": gpu_kv, "gpu_recurrent_mib": gpu_rec,
            "compute_mib": cb["gpu"], "cpu_compute_mib": cb["cpu"],
            "cpu_weights_mib": cpu_w, "cpu_kv_mib": kv_total - gpu_kv,
            "cpu_recurrent_mib": _mib(recurrent_bytes(cfg, rest, n_seq)),
            "vram_used_mib": gpu_w + gpu_kv + gpu_rec + cb["gpu"],
        }

    def finish(c, head, ls, cmd, forced=False):
        c["ram_used_mib"] = (c["cpu_weights_mib"] + c["cpu_kv_mib"]
                             + c["cpu_recurrent_mib"] + c.get("cpu_compute_mib", 0.0))
        c["ram_ok"] = c["ram_used_mib"] <= ram
        c["vram_ok"] = c["vram_used_mib"] <= eff_vram
        c["fits_fully"] = (c["n_gpu_layers"] >= n_layers and c["n_cpu_moe"] == 0)
        c["kind"] = "moe"; c["n_expert_layers"] = n_exp_layers
        c["vram_budget_mib"] = eff_vram; c["max_ctx_gpu"] = max_ctx_gpu
        c["forced_ngl"] = forced
        if forced and not c["vram_ok"]:
            head += ("  WARNING: needs %.0f MiB VRAM (> %.0f budget) - spills to shared memory (slow)."
                     % (c["vram_used_mib"], eff_vram))
        if not c["ram_ok"]:
            head += "  WARNING: needs %.0f MiB RAM (> %.0f MiB budget)." % (c["ram_used_mib"], ram)
        c["headline"] = head; c["lmstudio"] = ls; c["llama_cmd"] = cmd
        return c

    # ---- verifying a config you already ran -------------------------------
    if ngl_override is not None or n_cpu_moe_override is not None:
        ngl = n_layers if ngl_override is None else ngl_override
        ncm = 0 if n_cpu_moe_override is None else n_cpu_moe_override
        c = cost(ngl, ncm)
        head = ("Your config: %d of %d blocks on GPU, experts of the first %d on CPU."
                % (c["n_gpu_layers"], n_layers, c["n_cpu_moe"]))
        return finish(c, head,
                      ["GPU Offload / GPU Layers: %d" % c["n_gpu_layers"],
                       "Num CPU Expert Layers: %d" % c["n_cpu_moe"],
                       "Context Length: %d" % ctx,
                       "This is a verification of the numbers you entered, not a recommendation."],
                      _llama_flags(ctx, kv_type, flash_attn, ngl=999,
                                   n_cpu_moe=c["n_cpu_moe"]) if c["n_cpu_moe"] else
                      _llama_flags(ctx, kv_type, flash_attn, ngl=c["n_gpu_layers"]),
                      forced=True)

    # ---- everything on GPU --------------------------------------------------
    if fully_fits:
        c = cost(n_layers, 0)
        return finish(c, "Whole MoE fits on GPU. Keep experts on GPU. Room for ~%s tokens."
                      % f"{max_ctx_gpu:,}",
                      ["GPU Offload / GPU Layers: max",
                       "Force Model Expert Weights onto CPU: OFF (whole model fits)",
                       "Context Length: %d" % ctx],
                      _llama_flags(ctx, kv_type, flash_attn, ngl=99))

    # ---- keep every block on the GPU, push out only as many experts as needed
    for ncm in range(0, n_layers + 1):
        c = cost(n_layers, ncm)
        if c["vram_used_mib"] <= eff_vram:
            experts_gpu = n_layers - ncm
            if ncm == 0:
                head = "All experts fit on GPU alongside attention + KV."
                ls = ["GPU Offload / GPU Layers: max",
                      "Force Model Expert Weights onto CPU: OFF", "Context Length: %d" % ctx]
                cmd = _llama_flags(ctx, kv_type, flash_attn, ngl=99)
            elif experts_gpu == 0:
                head = "Attention + KV on GPU; ALL experts on CPU (%d blocks)." % ncm
                ls = ["GPU Offload / GPU Layers: max",
                      "Force Model Expert Weights onto CPU: ON", "Context Length: %d" % ctx]
                cmd = _llama_flags(ctx, kv_type, flash_attn, ngl=999, ot_all_experts=True)
            else:
                head = ("Attention + KV on GPU; experts for %d blocks on CPU, %d on GPU."
                        % (ncm, experts_gpu))
                ls = ["GPU Offload / GPU Layers: max",
                      "0.4.x: set 'Num CPU Expert Layers' (Number of layers to keep experts "
                      "on CPU) to %d - NOT the GPU Offload slider." % ncm,
                      "0.3.x: 'Force Model Expert Weights onto CPU' offloads ALL experts; "
                      "use the llama.cpp command below for a partial split.",
                      "Context Length: %d" % ctx]
                cmd = _llama_flags(ctx, kv_type, flash_attn, ngl=999, n_cpu_moe=ncm)
            return finish(c, head, ls, cmd)

    # ---- even attention + KV alone don't fit: fall back to whole-block offload
    best = None
    for ngl in range(n_layers, -1, -1):
        c = cost(ngl, n_layers)                  # all experts on CPU
        if c["vram_used_mib"] <= eff_vram:
            best = c
            break
    if best is None:
        best = cost(0, n_layers)
    c = best
    c["attention_overflow"] = True
    head = ("Attention + KV for all %d blocks (%.0f MiB KV at %s ctx) exceed the %.0f MiB budget "
            "even with every expert on CPU - falling back to %d whole blocks on GPU. "
            "Lower the context to keep attention on the GPU."
            % (n_layers, kv_total, f"{ctx:,}", eff_vram, c["n_gpu_layers"]))
    return finish(c, head,
                  ["Even attention+KV exceed VRAM at this context.",
                   "Lower Context Length (KV cache is the cost) before reducing GPU Layers.",
                   "GPU Offload / GPU Layers: %d" % c["n_gpu_layers"],
                   "Num CPU Expert Layers: %d" % n_layers],
                  _llama_flags(ctx, kv_type, flash_attn, ngl=c["n_gpu_layers"],
                               n_cpu_moe=n_layers))
