"""Turning a model file plus a budget into a layer split."""
import os
from .const import _mib
from .gguf import load_gguf
from .model import classify_tensors, extract_config
from .kv import kv_bytes_per_token, kv_bytes_per_token_growing, kv_bytes_total, kv_bytes_total_at, max_ctx_for_kv_budget, recurrent_bytes, resolve_kv_lengths, swa_cache_len
from .compute import MTP_SPEC_CONST_MIB, MTP_SPEC_PER_SEQ_MIB, compute_buffer_split, compute_buffer_terms, graph_is_split, output_head_on_gpu
from .gpu import gpu_list, platform_support
from .speed import estimate_speed
from .calib import calibration_status


def find_mmproj(model_path):
    """A multimodal model ships a separate vision/audio projector next to the
    weights (mmproj-*.gguf). LM Studio loads it with the model and puts it on the
    GPU, and it counts it in the "model size" it shows you - so it must be part of
    the VRAM budget. Returns {path, bytes, tensor_bytes} or None."""
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
                return {"path": p, "name": fn, "bytes": nb, "tensor_bytes": tb}
    except Exception:
        pass
    return None


def analyze(path, ctx, kv_type, n_ubatch, flash_attn,
            vram_budget_mib, ram_budget_mib, gpu_reserve_mib,
            compute_override_mib, safety_pct, kv_on_gpu=False,
            gpu_layers_override=None, ram_free_mib=None, n_seq=1,
            include_mmproj=True, n_cpu_moe_override=None,
            bw_vram_gbs=None, bw_ram_gbs=None, ram_eff=None, ctx_fill=None,
            bw_note="", mtp_spec=False):
    model = load_gguf(path)
    cfg = extract_config(model)
    cl = classify_tensors(model, cfg)
    mmproj = find_mmproj(path)

    n_layers = cfg["n_layers"] or 0
    warnings = []
    if cl["unknown_types"]:
        warnings.append("Unrecognized quant type(s) %s - sizes for those tensors may be off. "
                        "Check the total against file size below."
                        % ", ".join(cl["unknown_types"]))
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
    file_on_disk_mib = _mib(model["file_bytes"])

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
    # planning value: assume the whole graph runs on the GPU. The plan builders
    # re-split it once they know where the layers and the output head landed.
    compute_mib = compute_override_mib if compute_override_mib else \
        round(compute_terms["graph"] + compute_terms["logits"], 1)

    def compute_fn(ngl, output_on_gpu=None, any_on_cpu=None):
        """GPU/CPU compute-buffer split for a candidate layer count."""
        ngl = max(0, min(int(ngl), n_layers))
        if output_on_gpu is None:
            output_on_gpu = output_head_on_gpu(n_layers, ngl)
        if any_on_cpu is None:
            any_on_cpu = graph_is_split(n_layers, ngl)
        return compute_buffer_split(compute_terms, ngl > 0, any_on_cpu,
                                    output_on_gpu, compute_override_mib)

    # MTP speculative decoding: the draft blocks run after all, so they get the KV
    # cache back - at f16, whatever the configured KV quant - plus a fixed pool and
    # one slot per sequence. Comes off the top of the budget like the projector.
    spec_mib = 0.0
    if mtp_spec and cfg.get("mtp_kv_per_token"):
        spec_mib = (_mib(cfg["mtp_kv_per_token"] * ctx)
                    + MTP_SPEC_CONST_MIB + MTP_SPEC_PER_SEQ_MIB * max(1, n_seq))

    # vision/audio projector: loaded to the GPU alongside the model, so it comes
    # off the top of the budget before any layer split is planned
    mmproj_mib = _mib(mmproj["tensor_bytes"]) if (mmproj and include_mmproj) else 0.0

    # usable VRAM after reserving driver/OS headroom and a safety margin
    eff_vram = max(0.0, (vram_budget_mib - gpu_reserve_mib) * (1.0 - safety_pct / 100.0))
    eff_vram = max(0.0, eff_vram - mmproj_mib - spec_mib)

    result = {
        "ok": True, "warnings": warnings, "config": cfg,
        "model_name": cfg["name"], "arch": cfg["arch"],
        "shards": len(model["shards"]),
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
            "compute_graph": compute_terms["graph"], "compute_logits": compute_terms["logits"],
            "bundle_on_disk": file_on_disk_mib + (_mib(mmproj["bytes"]) if mmproj else 0.0),
        },
        "mmproj": ({"name": mmproj["name"], "mib": _mib(mmproj["tensor_bytes"]),
                    "file_mib": _mib(mmproj["bytes"]), "included": bool(include_mmproj)}
                   if mmproj else None),
        "hybrid": {
            "is_hybrid": cfg["is_hybrid"],
            "n_attn_layers": len(cfg["attn_layers"]),
            "n_ssm_layers": len(cfg["ssm_layers"]),
            "attn_layers": cfg["attn_layers"],
            "interval": cfg["full_attention_interval"],
        },
        "calibration": calibration_status(),
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
            "mtp_spec": bool(mtp_spec),
            "flash_attn": flash_attn, "vram_budget_mib": vram_budget_mib,
            "ram_budget_mib": ram_budget_mib, "gpu_reserve_mib": gpu_reserve_mib,
            "eff_vram_mib": eff_vram, "safety_pct": safety_pct,
            "n_ctx_train": cfg["n_ctx_train"],
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

    # Weights that actually reach VRAM: the transformer blocks only. The token
    # embeddings and output head stay in system RAM even at full offload.
    block_weights_mib = weights_mib - embed_mib - output_mib
    compute_full_mib = compute_fn(n_layers)["gpu"]
    full_need = block_weights_mib + kv_total_mib + rec_total_mib + compute_full_mib
    fully_fits = full_need <= eff_vram

    # max context that fits fully on GPU (all weights on GPU). KV is piecewise
    # linear in ctx once sliding-window layers are in play, and the compute buffer
    # itself grows with ctx, so re-solve rather than divide by a per-token cost.
    def _gpu_room(c):
        t = compute_buffer_terms(cfg, c, n_ubatch, flash_attn, n_seq, kv_type)
        cb = compute_override_mib if compute_override_mib else round(t["graph"] + t["logits"], 1)
        return eff_vram - block_weights_mib - rec_total_mib - cb
    max_ctx_gpu = max_ctx_for_kv_budget(cfg, kv_type, _gpu_room(ctx), n_ubatch,
                                        n_seq, flash_attn)
    for _ in range(4):                       # settle the compute-buffer feedback
        nxt = max_ctx_for_kv_budget(cfg, kv_type, _gpu_room(max(1, max_ctx_gpu)),
                                    n_ubatch, n_seq, flash_attn)
        if nxt == max_ctx_gpu:
            break
        max_ctx_gpu = nxt

    if cl["is_moe"]:
        plan = _plan_moe(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib, kv_per_layer_mib,
                         compute_mib, weights_mib, expert_total_mib, expert_layer_mean_mib,
                         per_layer_max_mib, per_layer_mean_mib, fully_fits, full_need,
                         max_ctx_gpu, ctx, kv_type, flash_attn, rec_total=rec_total_mib,
                         n_seq=n_seq, embed=embed_mib, output=output_mib,
                         ngl_override=gpu_layers_override, n_cpu_moe_override=n_cpu_moe_override,
                         compute_fn=compute_fn)
    elif kv_on_gpu and not fully_fits and cl["ffn_dense_total"] > 0:
        plan = _plan_dense_kv_on_gpu(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib,
                                     compute_mib, weights_mib, ffn_dense_total_mib,
                                     ffn_layer_mean_mib, max_ctx_gpu, ctx, kv_type, flash_attn,
                                     rec_total=rec_total_mib, n_ubatch=n_ubatch, n_seq=n_seq)
    else:
        plan = _plan_dense(cfg, cl, eff_vram, ram_budget_mib, kv_total_mib, kv_per_layer_mib,
                           compute_mib, weights_mib, embed_mib, output_mib,
                           per_layer_max_mib, per_layer_mean_mib, fully_fits, full_need,
                           max_ctx_gpu, ctx, kv_type, flash_attn,
                           ngl_override=gpu_layers_override, n_seq=n_seq,
                           compute_fn=compute_fn)
    # the projector was held out of eff_vram while planning; fold it back into the
    # reported totals so the bars and warnings show the real GPU footprint
    if mmproj_mib > 0:
        plan["mmproj_mib"] = mmproj_mib
        plan["vram_used_mib"] = plan.get("vram_used_mib", 0.0) + mmproj_mib
    if spec_mib > 0:
        plan["spec_mib"] = spec_mib
        plan["vram_used_mib"] = plan.get("vram_used_mib", 0.0) + spec_mib
    result["plan"] = plan

    # ---- speed roofline ----------------------------------------------------
    try:
        # KV-on-GPU mode keeps every block on the GPU but exiles the dense FFN,
        # so its placement cannot be described by a layer count alone.
        n_cpu_ffn = 0
        if plan.get("kind") == "dense_kv_gpu":
            ngl = n_layers
            n_cpu_ffn = int(plan.get("ffn_on_cpu") or n_layers)
            plan.setdefault("n_gpu_layers", n_layers)
        else:
            ngl = plan.get("n_gpu_layers")
            if ngl is None:
                ngl = n_layers if plan.get("fits_fully") else 0
        gpu_blocks = list(range(max(0, n_layers - int(ngl)), n_layers))
        fill = ctx_fill if ctx_fill is not None else min(ctx, 8192)
        sp = estimate_speed(cfg, cl, gpu_blocks, fill, kv_type,
                            bw_vram_gbs or 500.0, bw_ram_gbs or 50.0,
                            cpu_head=True,   # the head stays in RAM at every -ngl
                            ram_eff=ram_eff, n_cpu_moe=plan.get("n_cpu_moe", 0) or 0,
                            n_cpu_ffn=n_cpu_ffn)
        sp["n_gpu_layers"] = int(ngl)
        sp["n_cpu_moe"] = plan.get("n_cpu_moe", 0) or 0
        sp["n_cpu_ffn"] = n_cpu_ffn
        sp["bw_auto"] = bw_note or ""
        result["speed"] = sp
    except Exception as e:
        result["speed"] = {"error": "%s: %s" % (type(e).__name__, e)}

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


def _llama_flags(ctx, kv_type, flash_attn, ngl=None, n_cpu_moe=None, ot_all_experts=False):
    parts = ["llama-server", "-m <model.gguf>"]
    parts.append("-ngl %s" % (ngl if ngl is not None else 999))
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


def _plan_dense(cfg, cl, eff_vram, ram, kv_total, kv_layer, compute, weights,
                embed, output, layer_max, layer_mean, fully_fits, full_need,
                max_ctx_gpu, ctx, kv_type, flash_attn, ngl_override=None, n_seq=1,
                compute_fn=None):
    n_layers = cfg["n_layers"]
    per_layer = cl.get("per_layer_bytes") or {}
    # llama.cpp offloads the LAST n_gpu_layers blocks (i_gpu_start = n_layer - ngl),
    # so which blocks land on the GPU matters: on a hybrid model the attention
    # blocks are the expensive ones and they are not evenly spread.
    def gpu_blocks(ngl):
        return list(range(max(0, n_layers - ngl), n_layers))

    def build(ngl, forced):
        ngl = max(0, min(int(ngl), n_layers))
        full = (ngl >= n_layers)
        cb = compute_fn(ngl) if compute_fn else {"gpu": compute, "cpu": 0.0}
        if full:
            # Even at full offload the token embeddings and the output head stay in
            # system RAM - measured across an -ngl sweep, no step matching their
            # size appears at any layer count, and charging them to VRAM would
            # leave less room for the compute buffer than the CUDA context alone
            # occupies. Only the transformer blocks go to the GPU.
            gpu_w, gpu_kv = weights - embed - output, kv_total
            gpu_rec = _mib(recurrent_bytes(cfg, range(n_layers), n_seq))
            vram_used = gpu_w + kv_total + gpu_rec + cb["gpu"]
            cpu_w, cpu_kv, cpu_rec = embed + output, 0.0, 0.0
            cpu_layers = 0
        else:
            on_gpu = gpu_blocks(ngl)
            off = set(on_gpu)
            gpu_w  = _mib(sum(per_layer.get(i, 0) for i in on_gpu)) or (ngl * layer_max)
            gpu_kv = _mib(kv_bytes_total(cfg, kv_type, on_gpu))
            gpu_rec = _mib(recurrent_bytes(cfg, on_gpu, n_seq))
            vram_used = gpu_w + gpu_kv + gpu_rec + cb["gpu"]
            cpu_layers = n_layers - ngl
            rest = [i for i in range(n_layers) if i not in off]
            cpu_w = (_mib(sum(per_layer.get(i, 0) for i in rest)) or (cpu_layers * layer_mean)) \
                    + embed + output
            cpu_kv = kv_total - gpu_kv
            cpu_rec = _mib(recurrent_bytes(cfg, rest, n_seq))
        ram_used = cpu_w + cpu_kv + cpu_rec + cb["cpu"]
        ram_ok = ram_used <= ram
        vram_ok = vram_used <= eff_vram
        if full:
            head = "All %d layers on GPU." % n_layers
            if not forced:
                head += " Room for up to ~%s tokens of context." % f"{max_ctx_gpu:,}"
        else:
            head = "Split: %d of %d layers on GPU, %d on CPU." % (ngl, n_layers, cpu_layers)
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
        return {
            "kind": "dense", "fits_fully": full, "n_gpu_layers": ngl, "cpu_layers": cpu_layers,
            "forced_ngl": forced, "vram_ok": vram_ok,
            "vram_used_mib": vram_used, "vram_budget_mib": eff_vram,
            "ram_used_mib": ram_used, "ram_ok": ram_ok, "max_ctx_gpu": max_ctx_gpu,
            "gpu_weights_mib": gpu_w, "gpu_kv_mib": gpu_kv, "compute_mib": cb["gpu"],
            "cpu_compute_mib": cb["cpu"], "gpu_recurrent_mib": gpu_rec,
            "cpu_weights_mib": cpu_w, "cpu_kv_mib": cpu_kv, "cpu_recurrent_mib": cpu_rec,
            "gpu_attn_layers": 0 if full else sum(
                1 for i in gpu_blocks(ngl) if i in set(cfg["attn_layers"])),
            "n_attn_layers": len(cfg["attn_layers"]),
            "lmstudio": ls,
            "llama_cmd": _llama_flags(ctx, kv_type, flash_attn, ngl=(99 if full else ngl)),
            "headline": head,
        }

    if ngl_override is not None:
        return build(ngl_override, True)
    if fully_fits:
        return build(n_layers, False)
    # Blocks are not interchangeable (hybrid attn vs SSM, and quant varies per
    # block), so search downward for the largest ngl that actually fits rather
    # than dividing by an average.
    for ngl in range(n_layers, -1, -1):
        p = build(ngl, False)
        if p["vram_used_mib"] <= eff_vram:
            return p
    return build(0, False)


def _ffn_cpu_flag():
    return r'-ot "\.ffn_(gate|up|down)\.weight=CPU"'


def _plan_dense_kv_on_gpu(cfg, cl, eff_vram, ram, kv_total, compute, weights,
                          ffn_total, ffn_layer_mean, max_ctx_gpu, ctx, kv_type, flash_attn,
                          rec_total=0.0, n_ubatch=512, n_seq=1):
    """Keep ALL KV cache (and attention) on GPU by offloading dense FFN weights to CPU.
    Dense analog of MoE expert offload. Note: FFN runs every token, so this is slower
    for dense models than a normal layer split."""
    n_layers = cfg["n_layers"]
    n_ffn = cl["n_ffn_layers"]
    # Weights that actually reach VRAM: blocks minus the dense FFN we are exiling.
    # Embeddings and the output head stay in system RAM - see _plan_dense.build().
    head_ram = _mib(cl["embed_bytes"] + cl["output_bytes"])
    onchip_attn = weights - ffn_total - head_ram
    # base = attention + norms + full KV, all on GPU
    base_gpu = onchip_attn + kv_total + rec_total + compute
    if base_gpu <= eff_vram:
        spare = eff_vram - base_gpu
        ffn_on_gpu = int(spare / ffn_layer_mean) if ffn_layer_mean > 0 else 0
        ffn_on_gpu = max(0, min(ffn_on_gpu, n_ffn))
        ffn_on_cpu = n_ffn - ffn_on_gpu
        vram_used = base_gpu + ffn_on_gpu * ffn_layer_mean
        cpu_weights = ffn_on_cpu * ffn_layer_mean + head_ram
        ram_ok = cpu_weights <= ram
        if ffn_on_cpu == 0:
            head = "All KV on GPU and the whole model fits - no FFN offload needed."
            cmd = _llama_flags(ctx, kv_type, flash_attn, ngl=99)
            ls = ["GPU Offload / GPU Layers: max", "Context Length: %d" % ctx]
        elif ffn_on_gpu == 0:
            head = "All %s tokens of KV on GPU; every FFN layer on CPU (slow generation for dense)." % f"{ctx:,}"
            cmd = "llama-server -m <model.gguf> -ngl 999 %s -c %d%s" % (
                _ffn_cpu_flag(), ctx,
                (" -fa -ctk %s -ctv %s" % (kv_type, kv_type)) if kv_type != "f16"
                else (" -fa" if flash_attn else ""))
            ls = ["GPU Offload / GPU Layers: max",
                  "This mode needs llama.cpp's -ot (LM Studio has no dense-FFN-only toggle).",
                  "Context Length: %d" % ctx]
        else:
            head = "All KV on GPU; FFN for %d layers on CPU, %d on GPU." % (ffn_on_cpu, ffn_on_gpu)
            cmd = "llama-server -m <model.gguf> -ngl 999 %s -c %d%s" % (
                _ffn_cpu_flag(), ctx,
                (" -fa -ctk %s -ctv %s" % (kv_type, kv_type)) if kv_type != "f16"
                else (" -fa" if flash_attn else ""))
            ls = ["GPU Offload / GPU Layers: max",
                  "Partial dense-FFN offload needs llama.cpp -ot (below).",
                  "Context Length: %d" % ctx]
        if not ram_ok:
            head += "  WARNING: FFN needs %.0f MiB RAM (> %.0f MiB budget)." % (cpu_weights, ram)
        return {
            "kind": "dense_kv_gpu", "fits_fully": False, "kv_on_gpu": True,
            "ffn_on_cpu": ffn_on_cpu, "ffn_on_gpu": ffn_on_gpu, "n_ffn_layers": n_ffn,
            "vram_used_mib": vram_used, "vram_budget_mib": eff_vram,
            "ram_used_mib": cpu_weights, "ram_ok": ram_ok, "max_ctx_gpu": max_ctx_gpu,
            "gpu_weights_mib": onchip_attn + ffn_on_gpu * ffn_layer_mean,
            "gpu_kv_mib": kv_total, "compute_mib": compute, "gpu_recurrent_mib": rec_total,
            "cpu_weights_mib": cpu_weights, "cpu_kv_mib": 0.0,
            "lmstudio": ls, "llama_cmd": cmd, "headline": head,
        }
    # even attention + full KV don't fit -> the attention weights pinned to the GPU are the wall
    # attention weights only - the KV cache cannot live apart from them
    onchip = onchip_attn
    attn_base = onchip + rec_total + compute
    room = eff_vram - attn_base
    max_ctx_kv_gpu = max_ctx_for_kv_budget(cfg, kv_type, room, n_ubatch, n_seq, flash_attn)
    if room <= 0:
        head = ("KV fits, but it can't be separated from its layers: all KV on GPU forces all "
                "attention weights onto the GPU too (%.0f MiB) - that alone + the compute buffer "
                "leaves ~0 for KV. Use a smaller quant, or a normal layer split." % onchip)
    else:
        head = ("All KV on GPU also pins %.0f MiB of attention weights to the GPU; with the %.0f MiB "
                "KV that's over the %.0f MiB budget. Max context this way: ~%s tokens."
                % (onchip, kv_total, eff_vram, f"{max_ctx_kv_gpu:,}"))
    return {
        "kind": "dense_kv_gpu", "fits_fully": False, "kv_on_gpu": True, "kv_overflow": True,
        "ffn_on_cpu": n_ffn, "ffn_on_gpu": 0, "n_ffn_layers": n_ffn,
        "n_gpu_layers": n_layers,
        "vram_used_mib": base_gpu, "vram_budget_mib": eff_vram, "ram_used_mib": ffn_total,
        "gpu_weights_mib": onchip_attn, "gpu_kv_mib": kv_total, "compute_mib": compute,
        "gpu_recurrent_mib": rec_total,
        "cpu_weights_mib": ffn_total + head_ram, "cpu_kv_mib": 0.0, "max_ctx_gpu": max_ctx_gpu,
        "max_ctx_kv_gpu": max_ctx_kv_gpu,
        "lmstudio": ["Attention weights that must stay on GPU: %.0f MiB (KV can't live apart from them)." % onchip,
                     "Those + %.0f MiB KV + %.0f MiB compute = %.0f MiB (budget %.0f MiB)."
                     % (kv_total, compute, base_gpu, eff_vram),
                     "Max context with all KV on GPU (FFN on CPU): ~%s tokens." % f"{max_ctx_kv_gpu:,}",
                     "For KV-on-GPU at usable context, drop to a smaller quant (e.g. Q4_K_M)."],
        "llama_cmd": "llama-server -m <model.gguf> -ngl 999 %s -c %d -fa -ctk %s -ctv %s" % (
            _ffn_cpu_flag(), max(1024, max_ctx_kv_gpu), kv_type, kv_type),
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
