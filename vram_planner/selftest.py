"""Synthetic GGUF writer and the self-test suite."""
import os, struct
from .const import _mib
from .gguf import GGML_TYPES, _parse_one, load_gguf
from .model import RE_EXPS, classify_tensors, extract_config
from .compute import CB_CUDA_CTX_MIB, CB_DEFAULTS, CB_SPLIT_GRAPH_MIB, CB_SPLIT_PER_TOKEN, compute_buffer_split, compute_buffer_terms, graph_is_split, output_head_on_gpu, vision_grid, vision_peak_mib
from .lmstudio import REF_GPU, current_backend, read_lmstudio_runtime, resolve_runtime_ngl
from .calib import CALIB_SCHEMA, CALIB_TERMS, _CALIB_CACHE, _active_gpu, _design, _struct_offset, calib_coeffs, fit_calibration, mark_unreliable
from .plan import analyze, find_mmproj
from .speed import estimate_speed


# ---------------------------------------------------------------------------
# Self-test: build synthetic GGUFs, validate parser byte-sizes + planning math
# ---------------------------------------------------------------------------
def _ws(f, s):  b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)

def _wu32(f, v): f.write(struct.pack("<I", v))

def _wu64(f, v): f.write(struct.pack("<Q", v))

def _kv_u32(f, k, v): _ws(f, k); _wu32(f, 4); _wu32(f, v)

def _kv_str(f, k, v): _ws(f, k); _wu32(f, 8); _ws(f, v)

def _kv_arr(f, k, v):                       # array of u32 (per-layer metadata)
    _ws(f, k); _wu32(f, 9); _wu32(f, 4); _wu64(f, len(v))
    for x in v: _wu32(f, int(x))


def _tensor_bytes(dims, type_id):
    ne = 1
    for d in dims: ne *= d
    _, block, tsize = GGML_TYPES[type_id]
    nb = (ne // block) * tsize
    if ne % block: nb += tsize
    return nb


def _write_gguf(path, meta_u32, meta_str, tensors):
    align = 32
    ms = dict(meta_str); ms.setdefault("general.architecture", "test")
    kvs = [("general.architecture", "str", ms["general.architecture"])]
    for k, v in ms.items():
        if k != "general.architecture":
            kvs.append((k, "str", v))
    for k, v in meta_u32.items():
        kvs.append((k, "arr" if isinstance(v, (list, tuple)) else "u32", v))
    with open(path, "wb") as f:
        f.write(b"GGUF"); _wu32(f, 3)
        _wu64(f, len(tensors)); _wu64(f, len(kvs))
        for (k, t, v) in kvs:
            {"str": _kv_str, "arr": _kv_arr, "u32": _kv_u32}[t](f, k, v)
        offset = 0
        sizes = []
        for (name, dims, tid) in tensors:
            _ws(f, name); _wu32(f, len(dims))
            for d in dims: _wu64(f, d)
            _wu32(f, tid); _wu64(f, offset)
            sz = _tensor_bytes(dims, tid); sizes.append(sz)
            aligned = ((sz + align - 1) // align) * align
            offset += aligned
        here = f.tell()
        pad = ((here + align - 1) // align) * align - here
        f.write(b"\x00" * pad)
        if offset > 0:                       # sparse data region (no real bytes allocated)
            f.seek(offset - 1, 1); f.write(b"\x00")


def self_test(require_refs=False):
    """Run the suite. Returns 0 on success, 1 on failure.

    `require_refs` turns every skipped real-measurement section into a failure. The
    synthetic sections check that the code does what it says; only the sections that
    replay real llama-server loads check that what it says is TRUE, and those need
    the reference hardware and models. Without this flag the suite can print PASSED
    having never compared itself to reality."""
    skipped_real = []
    # The suite validates the CODE, so it must run against the shipped defaults.
    # Letting the user's own calibration load here would mean the tests measure
    # their fit instead - and fail on a perfectly good build.
    # (set on the module, not rebound locally - the flag lives in calib and every
    # other module reads it from there)
    from . import calib
    _CALIB_CACHE.clear()
    calib._CALIB_LOADED = True
    import tempfile
    tmp = tempfile.mkdtemp(prefix="vramtest_")
    ok = True

    # The suite analyses synthetic fixtures (dense/moe/hybrid/swa.gguf), and
    # analyze() records a model card for everything it reads. Left alone that
    # writes four fake models into the user's real card store, which then offers
    # them in the model list. Point the store at the scratch directory for the
    # whole run - the test must not have side effects on user data.
    from . import cards as _cardsmod
    _real_cards_store = _cardsmod._cards_store
    _cardsmod._cards_store = lambda: os.path.join(tmp, "test_cards.json")
    try:
        return _run_suite(require_refs, tmp, skipped_real)
    finally:
        _cardsmod._cards_store = _real_cards_store


def _run_suite(require_refs, tmp, skipped_real):
    from . import calib
    ok = True

    # 1) byte-precision test
    p1 = os.path.join(tmp, "prec.gguf")
    tens = [("a.f32", [10], 0), ("b.q4k", [256], 12), ("c.q6k", [256, 1], 14),
            ("d.q8_0", [64], 8)]
    _write_gguf(p1, {"test.block_count": 0}, {"general.architecture": "test"}, tens)
    _, _, parsed = _parse_one(p1)
    expect = {"a.f32": 40, "b.q4k": 144, "c.q6k": 210, "d.q8_0": 68}
    for t in parsed:
        e = expect[t["name"]]
        got = t["n_bytes"]
        print("  tensor %-8s type=%-6s bytes=%-6d expect=%-6d %s"
              % (t["name"], t["type_name"], got, e, "OK" if got == e else "FAIL"))
        ok = ok and (got == e)

    # 2) realistic DENSE model -> planning
    nL, hid, nh, nkv = 8, 512, 8, 2
    hd = hid // nh
    dense_t = [("token_embd.weight", [hid, 4000], 12)]
    for i in range(nL):
        dense_t += [
            ("blk.%d.attn_q.weight" % i, [hid, hid], 12),
            ("blk.%d.attn_k.weight" % i, [hid, nkv*hd], 12),
            ("blk.%d.attn_v.weight" % i, [hid, nkv*hd], 12),
            ("blk.%d.attn_output.weight" % i, [hid, hid], 12),
            ("blk.%d.ffn_gate.weight" % i, [hid, 1536], 12),
            ("blk.%d.ffn_up.weight" % i, [hid, 1536], 12),
            ("blk.%d.ffn_down.weight" % i, [1536, hid], 12),
            ("blk.%d.attn_norm.weight" % i, [hid], 0),
            ("blk.%d.ffn_norm.weight" % i, [hid], 0),
        ]
    dense_t += [("output_norm.weight", [hid], 0), ("output.weight", [hid, 4000], 14)]
    p2 = os.path.join(tmp, "dense.gguf")
    _write_gguf(p2, {"llama.block_count": nL, "llama.attention.head_count": nh,
                     "llama.attention.head_count_kv": nkv, "llama.embedding_length": hid,
                     "llama.context_length": 8192, "llama.feed_forward_length": 1536},
                {"general.architecture": "llama", "general.name": "DenseTest"}, dense_t)
    r = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=200, ram_budget_mib=8000,
                gpu_reserve_mib=64, compute_override_mib=40, safety_pct=0)
    p = r["plan"]
    print("  DENSE  is_moe=%s ngl=%s/%s  vram_used=%.0f  ram_used=%.0f"
          % (r["is_moe"], p.get("n_gpu_layers"), nL, p.get("vram_used_mib", 0), p.get("ram_used_mib", 0)))
    ok = ok and (r["is_moe"] is False) and (0 <= p["n_gpu_layers"] <= nL)

    # 3) MoE model -> expert offload plan
    nExp, used = 16, 2
    moe_t = [("token_embd.weight", [hid, 4000], 12)]
    for i in range(nL):
        moe_t += [
            ("blk.%d.attn_q.weight" % i, [hid, hid], 12),
            ("blk.%d.attn_k.weight" % i, [hid, nkv*hd], 12),
            ("blk.%d.attn_v.weight" % i, [hid, nkv*hd], 12),
            ("blk.%d.attn_output.weight" % i, [hid, hid], 12),
            ("blk.%d.ffn_gate_inp.weight" % i, [hid, nExp], 0),
            ("blk.%d.ffn_gate_exps.weight" % i, [hid, 768, nExp], 12),
            ("blk.%d.ffn_up_exps.weight" % i, [hid, 768, nExp], 12),
            ("blk.%d.ffn_down_exps.weight" % i, [768, hid, nExp], 12),
            ("blk.%d.attn_norm.weight" % i, [hid], 0),
            ("blk.%d.ffn_norm.weight" % i, [hid], 0),
        ]
    moe_t += [("output_norm.weight", [hid], 0), ("output.weight", [hid, 4000], 14)]
    p3 = os.path.join(tmp, "moe.gguf")
    _write_gguf(p3, {"qwen3moe.block_count": nL, "qwen3moe.attention.head_count": nh,
                     "qwen3moe.attention.head_count_kv": nkv, "qwen3moe.embedding_length": hid,
                     "qwen3moe.context_length": 8192, "qwen3moe.expert_count": nExp,
                     "qwen3moe.expert_used_count": used},
                {"general.architecture": "qwen3moe", "general.name": "MoETest"}, moe_t)
    r = analyze(p3, 4096, "q8_0", 512, True, vram_budget_mib=120, ram_budget_mib=8000,
                gpu_reserve_mib=32, compute_override_mib=30, safety_pct=0)
    p = r["plan"]
    print("  MoE    is_moe=%s kind=%s n_cpu_moe=%s experts_on_gpu=%s  active=%.2fB total=%.2fB"
          % (r["is_moe"], p.get("kind"), p.get("n_cpu_moe"), p.get("experts_on_gpu"),
             r["active_params"]/1e9, r["params_total"]/1e9))
    ok = ok and (r["is_moe"] is True) and (p["kind"] == "moe") and (r["active_params"] < r["params_total"])

    # 3b) DFlash drafter: a SEPARATE file next to the model whose architecture is
    #     "dflash". It is a drafter, not a model: it must be found for its target,
    #     priced by the plan, swept by stage D at its own block depth, and handed
    #     to llama.cpp with -md - and it must never surface as a model itself.
    dl = 3
    draft_t = []
    for i in range(dl):
        draft_t += [
            ("blk.%d.attn_q.weight" % i, [hid, hid], 12),
            ("blk.%d.attn_k.weight" % i, [hid, nkv * hd], 12),
            ("blk.%d.attn_v.weight" % i, [hid, nkv * hd], 12),
            ("blk.%d.attn_output.weight" % i, [hid, hid], 12),
            ("blk.%d.ffn_gate.weight" % i, [hid, 1536], 12),
            ("blk.%d.ffn_up.weight" % i, [hid, 1536], 12),
            ("blk.%d.ffn_down.weight" % i, [1536, hid], 12),
            ("blk.%d.attn_norm.weight" % i, [hid], 0),
            ("blk.%d.ffn_norm.weight" % i, [hid], 0),
        ]
    pd = os.path.join(tmp, "dflash-moe.gguf")
    _write_gguf(pd, {"dflash.block_count": dl, "dflash.block_size": 8,
                     "dflash.attention.head_count": nh,
                     "dflash.attention.head_count_kv": nkv,
                     "dflash.embedding_length": hid,
                     "dflash.context_length": 8192,
                     "dflash.feed_forward_length": 1536,
                     "dflash.target_layers": [0, 2, 4]},
                {"general.architecture": "dflash", "general.name": "DraftTest"},
                draft_t)
    from .sweep import (build_argv, discover_models, find_drafter_for as sdf)
    from .plan import find_drafter_for as pdf
    df_ok = (sdf(p3) == pd and sdf(pd) is None and pdf(p3)["block_size"] == 8
             and "dflash-moe.gguf" not in
             [os.path.basename(x) for x in discover_models(tmp)])
    print("  DFLASH finder pairs model<->drafter, never the drafter itself, and "
          "discovery skips it  %s" % ("OK" if df_ok else "FAIL"))
    ok = ok and df_ok
    rd = analyze(p3, 4096, "q8_0", 512, True, vram_budget_mib=120,
                 ram_budget_mib=8000, gpu_reserve_mib=32, compute_override_mib=30,
                 safety_pct=0, dflash=True)
    d = rd["dflash"] or {}
    exp_w = _mib(sum(_tensor_bytes(t[1], t[2]) for t in draft_t))
    plan_ok = (d.get("derived") is True and d.get("block_size") == 8
               and abs(d.get("weights_mib", 0) - exp_w) < 0.1
               and rd["inputs"]["dflash"] is True
               and abs(rd["plan"].get("spec_mib", 0) - d.get("mib", 0)) < 0.1
               and d.get("mib", 0) > exp_w)
    print("  DFLASH plan charges drafter exactly (%d MiB weights) + KV + graph  %s"
          % (exp_w, "OK" if plan_ok else "FAIL"))
    ok = ok and plan_ok
    # without the file there is no scheme to price, and that must be said out loud
    nod = os.path.join(tmp, "nodraft")
    os.makedirs(nod, exist_ok=True)
    pn = os.path.join(nod, "m.gguf")
    with open(p1, "rb") as fi, open(pn, "wb") as fo:
        fo.write(fi.read())
    nofile_ok = False
    try:
        analyze(pn, 4096, "f16", 512, True, vram_budget_mib=2000,
                ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                safety_pct=0, dflash=True)
    except ValueError:
        nofile_ok = True
    print("  DFLASH analyze without a drafter raises, naming the gap  %s"
          % ("OK" if nofile_ok else "FAIL"))
    ok = ok and nofile_ok
    # ...but a model with MTP blocks of its own is not stuck: llama.cpp runs one
    # scheme per server, so the plan prices the model's own MTP cache instead
    # and says so out loud, rather than failing a plan that can still be
    # complete. (The stale-checkbox path the UI now prevents entirely.)
    mtpp = os.path.join(nod, "mtp.gguf")
    _write_gguf(mtpp, {"llama.block_count": nL, "llama.attention.head_count": nh,
                       "llama.attention.head_count_kv": nkv,
                       "llama.embedding_length": hid, "llama.context_length": 8192,
                       "llama.feed_forward_length": 1536,
                       "llama.nextn_predict_layers": 1},
                {"general.architecture": "llama", "general.name": "MTPTest"},
                dense_t)
    fb = analyze(mtpp, 4096, "f16", 512, True, vram_budget_mib=2000,
                 ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                 safety_pct=0, dflash=True)
    fb_ok = (fb["dflash"] is None and fb["inputs"]["dflash"] is True
             and fb["plan"].get("spec_mib", 0) > 0
             and any("MTP" in w for w in (fb.get("warnings") or [])))
    print("  DFLASH analyze without a drafter falls back to MTP, warning said  %s"
          % ("OK" if fb_ok else "FAIL"))
    ok = ok and fb_ok
    # The same synthetic MTP model prices the draft-cache quant test in section
    # 15 - aliased here because the reference-model section reuses the `mtpp`
    # name for the real Qwen3.5-9B-MTP file, and section 15 runs after it.
    mtps = mtpp
    # 3c) EXPLICIT drafter pick: the user names the file, so discovery plays no
    #     part - the plan prices exactly that file. A dflash-*.gguf drafts as
    #     DFlash; a model with MTP blocks drafts as an MTP draft model (-md +
    #     --spec-type draft-mtp); anything else is refused rather than guessed.
    #     p2 sits NEXT TO a dflash drafter, so an explicit MTP pick that still
    #     prices the MTP file proves the pick overrides discovery.
    em = analyze(p2, 4096, "f16", 512, True, vram_budget_mib=2000,
                 ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                 safety_pct=0, drafter=mtpp)
    em_ok = (em["drafter"] and em["drafter"]["kind"] == "mtp"
             and em["drafter"]["name"] == "mtp.gguf"
             and em["drafter"]["mib"] > 0 and em["drafter"]["depth"] == 1
             and em["dflash"] is None and em["plan"].get("spec_mib", 0) > 0
             and em["inputs"]["drafter"] == os.path.abspath(mtpp)
             and em["inputs"]["drafter_kind"] == "mtp"
             and any("draft model" in w for w in (em.get("warnings") or [])))
    print("  DRAFTER explicit MTP model drafts as MTP, overrides discovery  %s"
          % ("OK" if em_ok else "FAIL"))
    ok = ok and em_ok
    ed = analyze(p3, 4096, "f16", 512, True, vram_budget_mib=2000,
                 ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                 safety_pct=0, drafter=pd)
    ed_ok = (ed["drafter"] and ed["drafter"]["kind"] == "dflash"
             and ed["drafter"]["depth"] == 8 and ed["dflash"] is not None
             and ed["inputs"]["drafter_kind"] == "dflash")
    print("  DRAFTER explicit dflash-*.gguf drafts as DFlash  %s"
          % ("OK" if ed_ok else "FAIL"))
    ok = ok and ed_ok
    refused = missing = False
    try:
        analyze(p2, 4096, "f16", 512, True, vram_budget_mib=2000,
                ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                safety_pct=0, drafter=p1)
    except ValueError:
        refused = True
    try:
        analyze(p2, 4096, "f16", 512, True, vram_budget_mib=2000,
                ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=0,
                safety_pct=0, drafter=os.path.join(nod, "nope.gguf"))
    except ValueError:
        missing = True
    print("  DRAFTER a plain model file is refused, a missing one too  %s"
          % ("OK" if refused and missing else "FAIL"))
    ok = ok and refused and missing
    # stage B sweeps the drafter's OWN depth ladder - every depth the block
    # allows, ascending - because the draft cache grows with depth and the
    # monotone wall prunes the deeper half of it once the first depth OOMs
    # (see the wall tests below).
    from .bench import stage_configs
    b0 = {"ctx": 4096, "kv": "f16", "fa": True, "seq": 1, "ub": 512, "ngl": 0}
    dd = stage_configs("b", dict(b0), 8, False, "ngl", [1], facts={"n_mtp_layers": 0},
                       drafter={"path": pd, "block_size": 8})
    dmax = sorted({c["spec_n_max"] for c in dd if c["spec"] == "draft-dflash"})
    nd = [c for c in stage_configs("b", dict(b0), 8, False, "ngl", [1],
                                   facts={"n_mtp_layers": 0})
          if c["spec"] == "draft-dflash"]
    av = build_argv("EXE", p3, dict(b0, spec="draft-dflash", spec_n_max=99),
                    8231, probe=False)
    clamped = ("-md" in av and "--spec-draft-n-max" in av
               and av[av.index("--spec-draft-n-max") + 1] == "8")
    nofile2 = False
    try:
        build_argv("EXE", pn, dict(b0, spec="draft-dflash", spec_n_max=8),
                   8231, probe=False)
    except ValueError:
        nofile2 = True
    grid_ok = (dmax == list(range(1, 9)) and not nd and clamped and nofile2)
    print("  DFLASH stage D depths %s; argv clamps 99 -> %s; no drafter -> refused  %s"
          % (dmax, av[av.index("--spec-draft-n-max") + 1] if clamped else "?",
             "OK" if grid_ok else "FAIL"))
    ok = ok and grid_ok
    # ...and the CHAINED rebuild - what the web campaign actually runs - must
    # carry the drafter too, or stage D silently loses its dflash rows mid-
    # campaign while the ngram rows still appear, which reads exactly like a
    # model that has no drafter.
    from .bench import grid_context
    gb, gnl, gmoe, gaxis, grungs = grid_context(
        {"n_layers": 8}, base=dict(b0), model_path=p3, drafter={"path": pd,
                                                                "block_size": 8})
    chained = [c for c in stage_configs("b", gb, gnl, gmoe, gaxis, grungs,
                                        facts={"n_mtp_layers": 0},
                                        drafter={"path": pd, "block_size": 8})
               if c["spec"] == "draft-dflash"]
    chain_ok = bool(chained) and "md" in chained[0] and "md" in gb
    print("  DFLASH chained stage rebuild keeps the drafter (md in baseline + rows)  %s"
          % ("OK" if chain_ok else "FAIL"))
    ok = ok and chain_ok
    # ...and an external MTP drafter is the OTHER scheme the picker exists for:
    # a separate GGUF with nextn blocks, which no discovery can find. The gate
    # passes on the drafter's blocks instead of the model's facts, and the
    # drafter rides every row the way DFlash's does. The model's own blocks (no
    # md) stay distinguishable from an external drafter (md present) even when
    # both exist.
    #
    # The depths are NOT capped at what the drafter trained. --spec-draft-n-max
    # is a draft RUN LENGTH, not a count of prediction heads: measured rows gave
    # fifteen different results past a one-nextn-layer drafter, peaking at depth
    # 4. The cap used to reduce the ladder to a single rung, which ended stage
    # B's wall walk the moment it found a config that fitted.
    from .bench import (_draft_depths, _spec_axis, _spec_ctx_rung,
                        STAGE_B_SPEC as _SDS)
    # Read from the campaign's own table: a widened ladder is a deliberate
    # change, and restating it here would only ever go stale.
    MTP_DEPTHS = sorted(n for s, n in _SDS if s == "draft-mtp")
    dm = stage_configs("b", dict(b0), 8, False, "ngl", [1], facts={"n_mtp_layers": 0},
                       drafter={"path": mtpp, "kind": "mtp", "depth": 2})
    mtp_rows = [c for c in dm if c["spec"] == "draft-mtp"]
    mtp_ok = (sorted({c["spec_n_max"] for c in mtp_rows}) == MTP_DEPTHS
              and all(c["md"] == os.path.abspath(mtpp) for c in mtp_rows)
              and not [c for c in dm if c["spec"] == "draft-dflash"]
              and any(c["spec"] == "ngram-mod" for c in dm))
    own_rows = [c for c in stage_configs("b", dict(b0), 8, False, "ngl", [1],
                                         facts={"n_mtp_layers": 2})
                if c["spec"] == "draft-mtp"]
    own_ok = (sorted({c["spec_n_max"] for c in own_rows}) == MTP_DEPTHS
              and all("md" not in c for c in own_rows))
    capped = (_draft_depths("draft-mtp", {"kind": "mtp", "depth": 2})
              == _draft_depths("draft-mtp") == MTP_DEPTHS)
    print("  MTP external drafter: stage B carries -md, depths uncapped  %s"
          % ("OK" if mtp_ok and own_ok and capped else "FAIL"))
    ok = ok and mtp_ok and own_ok and capped
    # The fit sweep gets the same scheme from the same spelling: classify the
    # named file (dflash / MTP / refuse), then build_grid appends the spec rows
    # with the drafter's md - the draft cache is real VRAM the plan only derives,
    # so the allocation sweep measures it too.
    from .sweep import build_grid, classify_drafter
    def load_drafter_q(pth):
        from .plan import load_drafter
        return bool(load_drafter(pth).get("name"))
    drf_mtp = classify_drafter(mtpp)
    drf_dfl = classify_drafter(pd)
    drf_bad = drf_missing = False
    try:
        classify_drafter(p1)
    except ValueError:
        drf_bad = True
    try:
        classify_drafter(os.path.join(nod, "nope.gguf"))
    except ValueError:
        drf_missing = True
    # A path a PERSON pasted. Windows Explorer's "Copy as path" wraps it in
    # double quotes, and a text field has no shell to strip them - so abspath()
    # read `"C:\...` as RELATIVE, prepended the working directory, and reported
    # a file not found that was sitting exactly where the message pointed. Both
    # drafter readers take the quotes off, and so does every path field on the
    # web form; a quote INSIDE a name is legal on POSIX and stays.
    from .paths import user_path
    from .web import clean_paths
    drf_quoted = (classify_drafter('"%s"' % mtpp) == drf_mtp
                  and classify_drafter("  %s  " % mtpp) == drf_mtp
                  and load_drafter_q('"%s"' % pd)
                  and user_path('"/a/b.gguf"') == "/a/b.gguf"
                  and user_path("'/a/b.gguf'") == "/a/b.gguf"
                  # not a matched pair, and not this function's business
                  and user_path('/a/b"c.gguf') == '/a/b"c.gguf'
                  and user_path('"/a/b.gguf') == '"/a/b.gguf'
                  and user_path(None) == "" and user_path('""') == ""
                  # the body cleaner touches paths and nothing else
                  and clean_paths({"path": '"/m.gguf"', "mmproj": True,
                                   "model_name": 'x"y'})
                  == {"path": "/m.gguf", "mmproj": True, "model_name": 'x"y'})

    fg = [c for c in build_grid({"n_layers": 8, "n_ctx_train": 8192,
                                 "is_moe": False}, drafter=drf_mtp)
          if c.get("spec")]
    fg_ok = (drf_mtp["kind"] == "mtp" and drf_mtp["depth"] == 1
             and drf_dfl["kind"] == "dflash" and drf_dfl["block_size"] == 8
             and drf_bad and drf_missing and drf_quoted
             and {c["spec"] for c in fg} == {"draft-mtp"}
             and {c["spec_n_max"] for c in fg} == {1}
             and all(c["md"] == os.path.abspath(mtpp) for c in fg))
    print("  MTP external drafter: fit grid adds draft-mtp rows with the md  %s"
          % ("OK" if fg_ok else "FAIL"))
    ok = ok and fg_ok

    # 3b) MONOTONE WALLS: one hard OOM proves every worse rung in the same
    #     family fails the same way, so the queued loads that would re-prove it
    #     are pruned. Only `oom` prunes; genfail/exit/timeout say nothing about
    #     the next rung; a `spilled` row LOADED, which is a measurement, not a
    #     wall. Family = every knob equal except the tagged axis.
    from .bench import _same_family, _worse, _prune_queue, _tag_axes, \
        _depth_extras, _spec_retry, _draft_depths, _spec_give_back, \
        _freer_rung, _spec_wall_note, _spec_ctx_rung
    from .sweep import _key
    wa = dict(ctx=4096, kv="f16", fa=True, seq=1, ub=512, ngl=20, fill=2048)
    wall_ok = True
    wall_ok = wall_ok and _same_family(dict(wa, ngl=22), dict(wa, ngl=30), "ngl") \
        and not _same_family(dict(wa, ngl=22), dict(wa, ngl=30, ub=1024), "ngl") \
        and not _same_family(dict(wa, ngl=22, spec="draft-mtp"),
                             dict(wa, ngl=30), "ngl")
    wall_ok = wall_ok and _worse(dict(wa, ngl=22), dict(wa, ngl=30), "ngl", "up") \
        and not _worse(dict(wa, ngl=22), dict(wa, ngl=30), "ngl", "down")
    # ncmoe normalises like _key does: absent means 0, so an absent-vs-0 pair is
    # the same config, and on the MoE axis worse means SMALLER.
    wall_ok = wall_ok and _worse(dict(wa, ncmoe=6), dict(wa, ncmoe=2),
                                 "ncmoe", "down") \
        and not _worse(dict(wa, ncmoe=6), dict(wa, ncmoe=2), "ncmoe", "up")
    print("  WALL same-family / worse-direction  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # The queue prune: an OOM at ngl 24 drops the pending ngl 28/32 in the same
    # family, keeps an ub=1024 sibling (different family), keeps everything
    # BEFORE the OOM (already run or running), and drops nothing on a failure
    # that proves nothing (the oom-only gate lives in run_group: a genfail,
    # exit or timeout row never calls _prune_queue).
    wq = [dict(wa, ngl=20, _wall=[("ngl", "up")]),
          dict(wa, ngl=24, _wall=[("ngl", "up")]),
          dict(wa, ngl=28, _wall=[("ngl", "up")]),
          dict(wa, ngl=32, _wall=[("ngl", "up")]),
          dict(wa, ngl=28, ub=1024, _wall=[("ngl", "up")])]
    wwalls = {_key("m", c): c.get("_wall") for c in wq}
    keep, drop = _prune_queue([dict(c) for c in wq], wwalls, 1, "m", dict(wa, ngl=24))
    drop_keys = [_key("m", c) for c in drop]
    wall_ok = ([c["ngl"] for c in keep] == [20, 24, 28]
               and keep[2]["ub"] == 1024               # different family: kept
               and sorted(c["ngl"] for c in drop) == [28, 32]
               and all(c["ub"] == 512 for c in drop)
               and _key("m", keep[2]) not in drop_keys
               and len(keep) == 3)
    print("  WALL queue prune on oom only, family-isolated  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # --speed-axes ladders get tagged by the same table: every monotone axis
    # named in the ladder tags every row of it (family = everything else equal,
    # so a cross-product prunes along each axis within each value of the
    # others); ncmoe only on an MoE; un-orderable axes stay untagged, so those
    # rows are all measured exactly as typed.
    tagged = [dict(wa, ngl=20), dict(wa, ngl=28), dict(wa, ub=1024)]
    _tag_axes(tagged, {"ngl", "ub", "temp"}, is_moe=False)
    wall_ok = (all(c.get("_wall") == [("ngl", "up"), ("ub", "up")] for c in tagged))
    tag1 = [dict(wa, ngl=20), dict(wa, ngl=28)]
    _tag_axes(tag1, {"ngl"}, is_moe=False)
    wall_ok = wall_ok and all(c.get("_wall") == [("ngl", "up")] for c in tag1)
    tag2 = [dict(wa, ncmoe=6), dict(wa, ngl=20, ncmoe=6)]
    _tag_axes(tag2, {"ncmoe"}, is_moe=True)
    wall_ok = wall_ok and tag2[0].get("_wall") == [("ncmoe", "down")]
    tag3 = [dict(wa, temp=1.0), dict(wa, spec="ngram-mod")]
    _tag_axes(tag3, {"temp", "spec"}, is_moe=False)
    wall_ok = wall_ok and "_wall" not in tag3[0] and "_wall" not in tag3[1]
    print("  WALL --speed-axes tagging (monotone only, ncmoe MoE-only)  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # Configs carry their tags out of the grid. Stage A's come from the search
    # rather than from a list - one place builds them (config_for), so the tag,
    # the stage letter and the axis value cannot be assembled two ways - and
    # stage B's from stage_configs, spec_n_max up within the draft family.
    from .bench import (_MONOTONE_AXES, _MONOTONE_MOE, _MONOTONE_FFN,
                        _WallSearch, wall_values)
    wsa = _WallSearch("ngl", [20, 24, 28], dict(wa, fill=2048))
    sa = [wsa.config_for(v) for v in wsa.values]
    wsf = _WallSearch("n_cpu_ffn", wall_values("n_cpu_ffn", 32), dict(wa))
    sb = [c for c in stage_configs("b", dict(wa), 32, False, "ngl", [1])
          if (c.get("spec") or "none") != "none"]
    wall_ok = (all(c.get("_wall") == [("ngl", "up")] for c in sa)
               and all(c.get("stage") == "A" for c in sa)
               and [c["ngl"] for c in sa] == [20, 24, 28]
               # the FFN search runs the other way, and says so in its own tag
               and wsf.wall == [("n_cpu_ffn", "down")]
               and wsf.values[0] == 32 and wsf.values[-1] == 0
               and all(c.get("_wall") == [("spec_n_max", "up")] for c in sb)
               and _MONOTONE_AXES == {"ngl": "up", "ub": "up",
                                      "ctx": "up", "spec_n_max": "up"}
               and _MONOTONE_MOE == {"ncmoe": "down"}
               and _MONOTONE_FFN == {"n_cpu_ffn": "down"})
    print("  WALL stage tags (search ngl-up / ffn-down, stage B nmax-up)  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # Depth extras: the top three fitting rungs from a stage-A ladder, at each
    # extra fill, ranked by the wall axis - and only from rows that actually
    # fit (status ok), so an OOM-only ladder yields nothing.
    rows = [{"status": "ok", "config": dict(wa, ngl=24, fill=2048, stage="A")},
            {"status": "ok", "config": dict(wa, ngl=28, fill=2048, stage="A")},
            {"status": "ok", "config": dict(wa, ngl=20, fill=2048, stage="A")},
            {"status": "oom", "config": dict(wa, ngl=32, fill=2048, stage="A")},
            {"status": "ok", "config": dict(wa, ngl=24, fill=2048, stage="C")}]
    ex = _depth_extras(rows, dict(wa, fill=2048), {"is_moe": False},
                       [8192, 16384], "ngl", "up")
    wall_ok = (len(ex) == 6 and _key("m", ex[0]) != _key("m", ex[3])
               and ex[0]["ngl"] == 28 and ex[0]["fill"] == 8192
               and ex[3]["ngl"] == 28 and ex[3]["fill"] == 16384
               and {c["ngl"] for c in ex} == {24, 28, 20}
               and all(c["stage"] == "A" and c.get("_wall") == [("ngl", "up")]
                       for c in ex))
    exo = _depth_extras(rows, dict(wa, fill=2048), {"is_moe": False},
                        [8192], "ngl", "up")
    wall_ok = wall_ok and _key("m", exo[0]) in [_key("m", c) for c in ex]
    exn = _depth_extras([{"status": "oom",
                          "config": dict(wa, ngl=32, fill=2048, stage="A")}],
                        dict(wa, fill=2048), {"is_moe": False}, [8192], "ngl", "up")
    wall_ok = wall_ok and not exn
    print("  WALL depth extras: top-3 fitting rungs x extra fills  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # The draft wall walk: one walk per FAMILY, rung by rung, depth ladder
    # nested inside. First OOM starts the walk one rung freer at the failed
    # depth; a fit continues the ladder at that rung; an OOM advances the rung
    # and resumes at the depth that just failed; n-gram schemes never walk.
    tr = {}
    dw = dict(wa, ngl=24, spec="draft-dflash", spec_n_max=3)
    facts8 = {"is_moe": False, "n_layers": 8}
    drf = {"path": pd, "block_size": 4}
    wall_ok = _draft_depths("draft-dflash", drf) == [1, 2, 3, 4]
    n1 = _spec_retry(dict(dw), {"status": "oom"}, facts8, tr, drafter=drf)
    wall_ok = wall_ok and n1 is not None and n1["ngl"] == 23 \
        and n1["spec_n_max"] == 3 and tr.get("draft-dflash")
    n2 = _spec_retry(n1, {"status": "ok"}, facts8, tr, drafter=drf)
    wall_ok = wall_ok and n2 is not None and n2["ngl"] == 23 \
        and n2["spec_n_max"] == 4
    n3 = _spec_retry(n2, {"status": "ok"}, facts8, tr, drafter=drf)
    wall_ok = wall_ok and n3 is None            # ladder exhausted at this rung
    # ...and a probe that OOMs at depth 4 walks one rung freer, resuming at the
    # depth that failed, not at the bottom of the ladder.
    tr2 = {}
    q1 = _spec_retry(dict(dw), {"status": "oom"}, facts8, tr2, drafter=drf)
    q2 = _spec_retry(q1, {"status": "oom"}, facts8, tr2, drafter=drf)
    wall_ok = wall_ok and q2 is not None and q2["ngl"] == 22 \
        and q2["spec_n_max"] == 3
    # On an MoE the walk goes UP, and a genfail row never walks at all.
    tr3 = {}
    dwm = dict(wa, ncmoe=6, ngl=0, spec="draft-mtp", spec_n_max=2)
    m1 = _spec_retry(dict(dwm), {"status": "oom"}, {"is_moe": True, "n_layers": 8},
                     tr3)
    wall_ok = wall_ok and m1 is not None and m1["ncmoe"] == 7 \
        and m1["spec_n_max"] == 2
    m2 = _spec_retry(m1, {"status": "ok"}, {"is_moe": True, "n_layers": 8}, tr3)
    wall_ok = wall_ok and m2 is not None and m2["ncmoe"] == 7 \
        and m2["spec_n_max"] == 3
    wall_ok = wall_ok and _spec_retry(dict(wa, spec="ngram-mod"),
                                      {"status": "oom"}, facts8, {}) is None \
        and _spec_retry(dict(wa, spec="draft-gram-l2"),
                        {"status": "oom"}, facts8, {}) is None \
        and _spec_retry(dict(wa, spec="draft-dflash", spec_n_max=2),
                        {"status": "genfail"}, facts8, {}, drafter=drf) is None
    print("  WALL draft walk: per-family rungs, nested depth ladder  %s"
          % ("OK" if wall_ok else "FAIL"))
    ok = ok and wall_ok
    # In the dense CEILING plan -ngl is pinned at every block - that pin IS the
    # plan - so the walk frees VRAM by giving back CONTEXT instead. Rungs are a
    # tenth of the window, snapped to CTX_LADDER_STEP, and -ngl must not move:
    # a walk that traded layers away here would report the fit plan's
    # numbers as the ceiling plan's.
    tr4 = {}
    dws = dict(wa, ngl=8, ctx=40960, spec="draft-dflash", spec_n_max=3)
    s1 = _spec_retry(dict(dws), {"status": "oom"}, facts8, tr4, drafter=drf,
                     axis="ctx")
    spd_ok = (s1 is not None and s1["ctx"] == 36864 and s1["ngl"] == 8
              and s1["spec_n_max"] == 3)
    # a fit continues the depth ladder at the SAME context, like any other axis
    s2 = _spec_retry(s1, {"status": "ok"}, facts8, tr4, drafter=drf,
                     axis="ctx")
    spd_ok = spd_ok and s2 is not None and s2["ctx"] == 36864 and s2["spec_n_max"] == 4
    # another OOM steps down again, a tenth of the NEW window
    tr5 = {}
    t1 = _spec_retry(dict(dws), {"status": "oom"}, facts8, tr5, drafter=drf,
                     axis="ctx")
    t2 = _spec_retry(t1, {"status": "oom"}, facts8, tr5, drafter=drf,
                     axis="ctx")
    spd_ok = spd_ok and t2 is not None and t2["ctx"] == 32768 and t2["ngl"] == 8
    # the step never stalls: a tenth of a small window rounds to zero, so the
    # floor is one CTX_LADDER_STEP, and below it there is no window to give back
    spd_ok = (spd_ok and _spec_ctx_rung(2048) == 1024
              and _spec_ctx_rung(1024) is None and _spec_ctx_rung(0) is None)
    # the FIT plan is untouched - it still walks layers, because there the
    # context is the thing being protected
    tr6 = {}
    c1 = _spec_retry(dict(dws), {"status": "oom"}, facts8, tr6, drafter=drf,
                     axis="ngl")
    spd_ok = spd_ok and c1 is not None and c1["ngl"] == 7 and c1["ctx"] == 40960
    # ...and an MoE ignores the mode entirely: its axis was never -ngl
    spd_ok = spd_ok and _spec_axis({"is_moe": True}, "ceiling") == "ncmoe"
    print("  WALL ceiling plan walks context, not layers  %s"
          % ("OK" if spd_ok else "FAIL"))
    ok = ok and spd_ok
    # ...unless stage A phase 2 left dense FFN blocks ON the card, in which case
    # THOSE are what the draft cache reclaims. They were bought with the VRAM
    # phase 1 had left over and buying them was worth a few per cent; the
    # context in this mode is what the campaign went looking for.
    #
    # This is not a preference between two levers. Muse-Glimmer-30B at ngl 52 on
    # a 5070 Ti Laptop: one FFN block is ~214 MiB, one context rung at 131072
    # against a q8_0 cache is ~66 MiB. The real campaign walked context down all
    # seven rungs - 131072 to 62464, less than half the window - freed 463 MiB,
    # and never fitted the 1556 MiB dflash drafter. Three FFN blocks free more
    # than the whole context axis can, because that axis caps out at the size of
    # the KV cache and a drafter is usually bigger.
    tr8 = {}
    dwf = dict(wa, ngl=8, ctx=40960, n_cpu_ffn=5, spec="draft-dflash",
               spec_n_max=3)
    f1 = _spec_retry(dict(dwf), {"status": "oom"}, facts8, tr8, drafter=drf,
                     axis="ctx")
    ffn_ok = (f1 is not None and f1["n_cpu_ffn"] == 6 and f1["ctx"] == 40960
              and f1["ngl"] == 8 and f1["spec_n_max"] == 3)
    f2 = _spec_retry(f1, {"status": "oom"}, facts8, tr8, drafter=drf, axis="ctx")
    ffn_ok = ffn_ok and f2 is not None and f2["n_cpu_ffn"] == 7 \
        and f2["ctx"] == 40960
    # a fit continues the depth ladder at that rung, like on any other axis
    f3 = _spec_retry(f2, {"status": "ok"}, facts8, tr8, drafter=drf, axis="ctx")
    ffn_ok = ffn_ok and f3 is not None and f3["n_cpu_ffn"] == 7 \
        and f3["spec_n_max"] == 4
    # Exhausted at n_layers, the walk falls through to stage A's own axis and
    # carries on there - the rungs it would have walked had phase 2 never run,
    # so nothing is lost by trying the FFN first.
    #
    # And it goes until a rung FITS or the axis runs out. There is no budget:
    # five was the old bound, then seven, and seven was three rungs short on the
    # first model that tested it - Muse-Glimmer-30B's dflash drafter fits at
    # n_cpu_ffn 34 and the walk stopped at 31, having spent every load it was
    # allowed to conclude the opposite of the truth.
    tr9, c9, seen9, ran_out = {}, dict(dwf), [], False
    for _ in range(400):
        nx = _spec_retry(c9, {"status": "oom"}, facts8, tr9, drafter=drf,
                         axis="ctx")
        if nx is None:
            ran_out = True
            break
        seen9.append((nx["n_cpu_ffn"], nx["ctx"]))
        c9 = nx
    ctxs = [x for v, x in seen9[3:]]
    ffn_ok = (ffn_ok and ran_out                       # ended on its own terms
              and [v for v, _ in seen9[:3]] == [6, 7, 8]
              and all(x == 40960 for _, x in seen9[:3])   # context untouched
              and seen9[3][0] == 8 and seen9[3][1] < 40960  # then the context
              and all(v == 8 for v, _ in seen9[3:])
              and ctxs == sorted(ctxs, reverse=True) and len(set(ctxs)) == len(ctxs)
              # ...all the way to the end of it, not to a rung count
              and _spec_ctx_rung(ctxs[-1]) is None)
    # Phase 2 never ran, so there is no FFN on the card to give back and the
    # walk is exactly what it was: straight down the context.
    ffn_ok = ffn_ok and _spec_retry(dict(dws), {"status": "oom"}, facts8, {},
                                    drafter=drf, axis="ctx")["ctx"] == 36864
    # An MoE has no dense FFN to give back - --n-cpu-moe IS its FFN knob and is
    # already the axis - so the rule can only ever name what the caller passed.
    ffn_ok = (ffn_ok
              and _spec_give_back(dict(dwf), {"is_moe": True, "n_layers": 8},
                                  "ncmoe") == "ncmoe"
              and _spec_give_back(dict(dwf), facts8, "ctx") == "n_cpu_ffn"
              and _spec_give_back(dict(dwf, n_cpu_ffn=8), facts8, "ctx") == "ctx"
              # "freer" is read from the monotone tables, not from a second copy
              and _freer_rung("n_cpu_ffn", 7, 8) == 8
              and _freer_rung("n_cpu_ffn", 8, 8) is None
              and _freer_rung("ngl", 8, 8) == 7 and _freer_rung("ngl", 1, 8) is None)
    # The all-OOM note has to suggest the same ladder the walk would have
    # spent, or the advice contradicts the eight OOM lines above it. In the
    # ceiling plan -ngl is pinned at every block, so an ngl ladder there proposes
    # leaving the plan rather than making room inside it.
    nt, nt2 = [], []
    nout = [{"status": "oom", "config": {"spec": "draft-dflash"}}]
    ntodo = [{"spec": "draft-dflash", "spec_n_max": 1}, {"spec": "ngram-mod"}]
    _spec_wall_note(nout, ntodo, dict(wa, ngl=52, ncmoe=0, n_cpu_ffn=24),
                    nt.append, facts={"n_layers": 52, "is_moe": False})
    _spec_wall_note(nout, ntodo, dict(wa, ngl=52, ncmoe=0, n_cpu_ffn=52),
                    nt2.append, facts={"n_layers": 52, "is_moe": False})
    ffn_ok = (ffn_ok
              and any("n_cpu_ffn=25,26,27,28" in ln for ln in nt)
              and not any("ngl=" in ln for ln in nt)
              and any("ngl=48,49,50,51" in ln for ln in nt2))
    print("  WALL phase 2 FFN blocks given back before the context  %s"
          % ("OK" if ffn_ok else "FAIL"))
    ok = ok and ffn_ok
    # The MTP depth ladder is NOT capped at an external drafter's
    # nextn_predict_layers. --spec-draft-n-max is a draft RUN LENGTH: measured
    # rows on gemma-4-12B gave fifteen different results with acceptance
    # decaying smoothly (0.822 at depth 1, 0.540 at 4, 0.215 at 15) and the best
    # throughput at depth 4, so a clamp is not what llama.cpp does. The cap made
    # the ladder [1] for a one-nextn-layer drafter, and stage D's wall walk -
    # having just spent two loads finding a context where the draft cache fits -
    # stopped the instant depth 1 fitted, with the depth question unmeasured.
    d1 = {"kind": "mtp", "depth": 1, "path": pd}
    cap_ok = (_draft_depths("draft-mtp", d1) == _draft_depths("draft-mtp")
              and len(_draft_depths("draft-mtp", d1)) > 1
              # dflash IS a block count, and stays one
              and _draft_depths("draft-dflash", {"block_size": 4}) == [1, 2, 3, 4])
    # ...so the walk keeps going after the rung that fits: two OOMs down the
    # context axis, then the depth ladder at the context that worked.
    tr7, c7 = {}, dict(wa, ngl=8, ctx=40960, spec="draft-mtp", spec_n_max=1)
    seen7 = []
    for st in ("oom", "oom", "ok", "ok"):
        nx = _spec_retry(c7, {"status": st}, facts8, tr7, drafter=d1, axis="ctx")
        if nx is None:
            break
        seen7.append((nx["ctx"], nx["spec_n_max"]))
        c7 = nx
    cap_ok = cap_ok and len(seen7) == 4 and seen7[0][1] == 1 and seen7[1][1] == 1 \
        and seen7[2][0] == seen7[1][0] and seen7[2][1] == 2 and seen7[3][1] == 3
    print("  WALL after a fitting rung the depth ladder continues  %s"
          % ("OK" if cap_ok else "FAIL"))
    ok = ok and cap_ok
    # Promotion between stages: which row becomes the next stage's baseline.
    # Two gates decide it and they must widen TOGETHER on a swept axis - a row
    # that cannot be compared is never a candidate, and a winner that is not
    # carried is found and thrown away. ctx is the only axis that is both gated
    # by comparable() and absent from CARRY_KEYS, because until the ceiling plan
    # it was only ever the campaign's definition and never its result.
    # CARRY_KEYS is imported again further down, which makes it a local for
    # this whole function - so it has to be bound here too, not just there.
    from .bench import best_config, carry_keys, CHAIN_MARGIN, CARRY_KEYS
    pbase = dict(wa, ctx=98304, ngl=65, n_cpu_ffn=65, ub=512, fill=2048,
                 spec="none", spec_kv="f16", kv="q8_0", fa=True, seq=1)
    def prow(c, tok):
        return {"status": "ok", "tok_s": tok, "model": "M", "n_predict": 128,
                "repeat": 3, "distinct_ratio": 0.9, "copyback_ratio": 0.1,
                "config": dict(pbase, ctx=c, stage="A")}
    # stage A swept context and found a rung both FASTER and larger than the
    # planner's opening guess
    prows = [prow(49152, 8.0), prow(73728, 7.6), prow(98304, 7.2), prow(122880, 9.9)]
    pa, _ = best_config(prows, "M", pbase, n_predict=128, repeat=3, swept="ctx")
    pn, _ = best_config(prows, "M", pbase, n_predict=128, repeat=3, swept="ngl")
    promo_ok = (pa is not None and pa["ctx"] == 122880          # swept: promoted
                and "ctx" in carry_keys("ctx")
                # not swept: ctx is the campaign's definition, so the other
                # rungs are other experiments and the baseline does not move
                and pn is not None and pn["ctx"] == 98304
                and "ctx" not in carry_keys("ngl")
                and "ctx" not in CARRY_KEYS)
    # the margin still gates it: a win inside the noise floor does not move the
    # baseline, or the campaign's PATH depends on jitter
    inside, _ = best_config(prows, "M", pbase, n_predict=128, repeat=3,
                            swept="ctx", incumbent_tok_s=9.9 / (1.0 + CHAIN_MARGIN / 2))
    outside, _ = best_config(prows, "M", pbase, n_predict=128, repeat=3,
                             swept="ctx", incumbent_tok_s=9.0)
    promo_ok = promo_ok and inside is None and outside is not None
    # ...and an untrustworthy row is never promoted however fast it looks: a
    # spilled row's speed is off a cliff for a reason unrelated to the setting
    sp = prow(122880, 99.0); sp["spilled"] = True
    bad, badw = best_config([prow(98304, 7.2), sp], "M", pbase, n_predict=128,
                            repeat=3, swept="ctx")
    promo_ok = promo_ok and badw["tok_s"] == 7.2 and bad["ctx"] == 98304
    print("  PROMO stage B: fastest trustworthy comparable row, by margin  %s"
          % ("OK" if promo_ok else "FAIL"))
    ok = ok and promo_ok
    # ...but stage B is the ONLY stage that asks a speed question. Stage A takes
    # the value at the wall instead - the largest that loads, or the smallest on
    # an axis where less means more resident. Ranking those by tok/s ranks
    # jitter: measured rows move 4.2% of tok/s across a DOUBLING of context,
    # and downward, so fastest-wins promoted the smallest window on the ladder
    # in the one mode whose whole purpose is the largest.
    from .bench import stage_objective
    def _pick(rws, letter, gaxis, **kw):
        ax, ob = stage_objective(letter, gaxis)
        return best_config(rws, "M", pbase, n_predict=128, repeat=3,
                           swept=(ax or ""), axis=ax, objective=ob, **kw)
    def _pr(tok, **kw):
        return {"status": "ok", "tok_s": tok, "model": "M", "n_predict": 128,
                "repeat": 3, "distinct_ratio": 0.9, "copyback_ratio": 0.1,
                "config": dict(pbase, stage=kw.pop("stage", "A"), **kw)}
    obj_ok = (stage_objective("a", "ctx") == ("ctx", "extreme")
              and stage_objective("a", "ncmoe") == ("ncmoe", "extreme")
              and stage_objective("a", "n_cpu_ffn") == ("n_cpu_ffn", "extreme")
              and stage_objective("b", "ctx") == (None, "fastest"))
    # the real shape: tok/s drifts DOWN as the window grows
    lad = [_pr(7.15, ctx=74752), _pr(6.89, ctx=111616), _pr(6.85, ctx=148480)]
    ca, _ = _pick(lad, "a", "ctx")
    old, _ = best_config(lad, "M", pbase, n_predict=128, repeat=3, swept="ctx")
    obj_ok = obj_ok and ca["ctx"] == 148480 and old["ctx"] == 74752
    # -ngl runs the other way round in the fit plan, ncmoe the other way again
    cn, _ = _pick([_pr(6.2, ngl=61), _pr(6.9, ngl=63), _pr(6.85, ngl=65)], "a", "ngl")
    cm, _ = _pick([_pr(6.0, ncmoe=33), _pr(6.6, ncmoe=31), _pr(6.5, ncmoe=29)],
                  "a", "ncmoe")
    obj_ok = obj_ok and cn["ngl"] == 65 and cm["ncmoe"] == 29
    # phase 2 promotes the SMALLEST FFN exile that loaded - fewer blocks moved
    # off the card is more resident, the same direction ncmoe runs
    cf, _ = _pick([_pr(6.2, n_cpu_ffn=65), _pr(6.9, n_cpu_ffn=40),
                   _pr(6.85, n_cpu_ffn=31)], "a", "n_cpu_ffn")
    obj_ok = obj_ok and cf["n_cpu_ffn"] == 31
    # stage B stays a speed question - no ordering of draft depths implies an
    # answer, only the acceptance rate does
    cd, wd = _pick([_pr(6.9, spec="none", stage="B"),
                    _pr(9.4, spec="draft-mtp", spec_n_max=2, stage="B"),
                    _pr(8.1, spec="draft-mtp", spec_n_max=5, stage="B")], "b", "ctx")
    obj_ok = obj_ok and cd["spec_n_max"] == 2 and wd["tok_s"] == 9.4
    # NO SLACK. The largest value that loaded wins even when it is much slower:
    # a big window measured slow is still the big window that fits, and that is
    # what the ceiling plan exists to find. The 5% slack this replaces could hand
    # back half the context to buy 6% of a number that moves 4.2% across a
    # doubling of the window anyway.
    cg, _ = _pick([_pr(7.0, ctx=74752), _pr(3.0, ctx=148480)], "a", "ctx")
    obj_ok = obj_ok and cg["ctx"] == 148480
    # a spilled row is not trustworthy() and still wins stage A, because stage A
    # asks whether it LOADED and the answer is yes. Stage B, which reads its
    # tok/s, still refuses it - see the block above.
    spl = _pr(6.5, ctx=148480); spl["spilled"] = True
    cs, _ = _pick([_pr(7.0, ctx=74752), spl], "a", "ctx")
    obj_ok = obj_ok and cs["ctx"] == 148480
    # ...but a row that did not load never wins, whatever its status says
    for bad_status in ("oom", "skipped", "genfail", "timeout", "exit"):
        br = _pr(9.9, ctx=148480); br["status"] = bad_status
        cb, _ = _pick([_pr(7.0, ctx=74752), br], "a", "ctx")
        obj_ok = obj_ok and cb["ctx"] == 74752
    # tok/s only settles a tie between rows at the SAME value
    t1, t2 = _pr(6.0, ctx=74752), _pr(8.0, ctx=74752)
    ct, wt = _pick([t1, t2], "a", "ctx")
    obj_ok = obj_ok and wt["tok_s"] == 8.0
    print("  PROMO stage A takes the wall with no slack, B takes the fastest  %s"
          % ("OK" if obj_ok else "FAIL"))
    ok = ok and obj_ok
    # Which baseline a stage's rows are judged against. comparable() gates ctx,
    # and in the dense SPEED mode nobody pins one: the campaign opens at
    # SPEED_BASE's context and stage A goes looking for the real wall. Judging
    # stage C against the OPENING base then rejects every row it just measured -
    # they all sit at what A found - so no ubatch is ever promoted and stage D
    # inherits A's. It went unseen because stage A itself is immune: swept="ctx"
    # switches that same gate off, so only the stages after it are affected.
    from .bench import grid_context
    cfacts = {"n_layers": 65, "n_ctx_train": 262144, "arch": "qwen3"}
    carried = dict(pbase, ctx=148480, stage="A")
    cgb, _n, _m, _ax, _r = grid_context(cfacts, base=dict(pbase), carried=carried)
    # phase 2's rows sit at the context phase 1 found, not at the opening one
    crows = [_pr(7.11, ctx=148480, n_cpu_ffn=65, stage="A"),
             _pr(6.90, ctx=148480, n_cpu_ffn=40, stage="A")]
    opened, _ = best_config(crows, "M", pbase, n_predict=128, repeat=3,
                            swept="n_cpu_ffn", axis="n_cpu_ffn",
                            objective="extreme")
    stage, _ = best_config(crows, "M", cgb, n_predict=128, repeat=3,
                           swept="n_cpu_ffn", axis="n_cpu_ffn",
                           objective="extreme")
    base_ok = (cgb["ctx"] == 148480 and pbase["ctx"] != 148480
               and opened is None and stage is not None
               and stage["n_cpu_ffn"] == 40 and stage["ctx"] == 148480)
    print("  PROMO a stage is judged against the base it RAN at, not the "
          "campaign's opening one  %s" % ("OK" if base_ok else "FAIL"))
    ok = ok and base_ok
    # the launcher carries the drafter as a checked path variable, in both shells
    from .launch import command_lines, launch_script
    csh = dict(b0, spec="draft-dflash", spec_n_max=8, md=pd)
    shb = "\n".join(command_lines(launch_script(p3, csh, shell="bash")))
    shp = "\n".join(command_lines(launch_script(p3, csh, shell="powershell")))
    launch_ok = ('-md "$DRAFT"' in shb and 'for f in "$MODEL" "$DRAFT"' in shb
                 and "-md $draft" in shp and "@($model, $draft)" in shp)
    print("  DFLASH launcher passes -md with a checked path, both shells  %s"
          % ("OK" if launch_ok else "FAIL"))
    ok = ok and launch_ok
    # an external MTP drafter rides the same checked-path machinery, and the
    # model's own MTP (draft-mtp without an md) must NOT collect a drafter
    # from the folder - the re-resolve only ever looks for dflash files, and
    # draft-mtp only enters it when a file was named.
    csm = dict(b0, spec="draft-mtp", spec_n_max=2, md=mtpp)
    shb2 = "\n".join(command_lines(launch_script(p3, csm, shell="bash")))
    shp2 = "\n".join(command_lines(launch_script(p3, csm, shell="powershell")))
    mav = build_argv("EXE", p2, dict(b0, spec="draft-mtp", spec_n_max=2, md=mtpp),
                     8231, probe=False)
    oav = build_argv("EXE", p2, dict(b0, spec="draft-mtp", spec_n_max=2),
                     8231, probe=False)
    mtpmd_ok = ('-md "$DRAFT"' in shb2 and "-md $draft" in shp2
                and "-md" in mav and "-md" not in oav)
    print("  DFLASH external MTP drafter rides -md in argv and both launchers  %s"
          % ("OK" if mtpmd_ok else "FAIL"))
    ok = ok and mtpmd_ok

    # 4) hybrid attention/SSM: only every Nth block may carry a KV cache
    hyb_t = [("token_embd.weight", [hid, 4000], 12)]
    for i in range(nL):
        if i % 4 == 3:
            hyb_t += [("blk.%d.attn_k.weight" % i, [hid, nkv*hd], 12),
                      ("blk.%d.attn_v.weight" % i, [hid, nkv*hd], 12),
                      ("blk.%d.attn_q.weight" % i, [hid, hid], 12)]
        else:
            hyb_t += [("blk.%d.ssm_conv1d.weight" % i, [4, 1024], 0),
                      ("blk.%d.ssm_out.weight" % i, [512, hid], 8)]
        hyb_t += [("blk.%d.ffn_gate.weight" % i, [hid, 1536], 12),
                  ("blk.%d.ffn_up.weight" % i, [hid, 1536], 12),
                  ("blk.%d.ffn_down.weight" % i, [1536, hid], 12)]
    hyb_t += [("output.weight", [hid, 4000], 14)]
    p4 = os.path.join(tmp, "hybrid.gguf")
    _write_gguf(p4, {"h.block_count": nL, "h.attention.head_count": nh,
                     "h.attention.head_count_kv": nkv, "h.embedding_length": hid,
                     "h.context_length": 8192, "h.full_attention_interval": 4,
                     "h.ssm.state_size": 64, "h.ssm.inner_size": 512,
                     "h.ssm.conv_kernel": 4},
                {"general.architecture": "h", "general.name": "HybridTest"}, hyb_t)
    rh = analyze(p4, 4096, "f16", 512, False, vram_budget_mib=200, ram_budget_mib=8000,
                 gpu_reserve_mib=64, compute_override_mib=40, safety_pct=0,
                 bw_vram_gbs=600, bw_ram_gbs=80, ctx_fill=1024)
    n_attn = len(rh["config"]["attn_layers"])
    hyb_ok = (n_attn == nL // 4 and rh["hybrid"]["is_hybrid"]
              and rh["sizes_mib"]["recurrent_total"] > 0)
    print("  HYBRID KV-bearing=%d of %d (expect %d) recurrent=%.1f MiB  %s"
          % (n_attn, nL, nL // 4, rh["sizes_mib"]["recurrent_total"],
             "OK" if hyb_ok else "FAIL"))
    ok = ok and hyb_ok

    # 5) The two-plan regime. A dense model that does not fit has one answer per
    #    question, so analyze() computes BOTH and reports one: the CEILING plan
    #    pins every block on the GPU with the dense FFN exiled (-ngl all, -ot all)
    #    and solves for the largest context; the FIT plan holds the context and
    #    walks -ot (then -ngl) down to the least exile that fits.
    def _two(**kw):
        return analyze(p2, 4096, "f16", 512, False, vram_budget_mib=30,
                       ram_budget_mib=8000, gpu_reserve_mib=0,
                       compute_override_mib=5, safety_pct=0,
                       bw_vram_gbs=600, bw_ram_gbs=80, ctx_fill=1024, **kw)
    rk = _two()
    pl = rk.get("plans") or {}
    ceil_plan, fit_plan = pl.get("ceiling") or {}, pl.get("fit") or {}
    nL2 = rk["config"]["n_layers"]
    # Each plan carries its OWN roofline, because the browser toggles between them
    # without asking again - a single top-level one would show the selected plan's
    # speed under the other plan's split.
    sk = ceil_plan.get("speed") or {}
    two_ok = (ceil_plan.get("kind") == "dense_ceiling"
              and ceil_plan.get("n_gpu_layers") == nL2
              and ceil_plan.get("n_cpu_ffn") == nL2
              and fit_plan.get("kind") == "dense_fit"
              and fit_plan.get("max_ctx") == 4096
              # the fit plan pays in the CHEAPEST currency first: it exiles
              # only as much dense FFN as it has to, and gives the rest back to
              # the GPU. Whole blocks move only when a full exile is not enough.
              and 0 <= (fit_plan.get("n_cpu_ffn") or 0) <= nL2
              and (fit_plan.get("n_gpu_layers") == nL2
                   or fit_plan.get("n_cpu_ffn") == nL2)
              # the exiled FFN must be charged to the CPU side of the roofline
              and sk.get("cpu_mib", 0) > 0
              and rk["plan"] is pl[rk["plan_mode"]])
    print("  2PLAN  ceiling ngl=%s ffn=%s max_ctx=%s | fit ngl=%s ffn=%s | reported=%s  %s"
          % (ceil_plan.get("n_gpu_layers"), ceil_plan.get("n_cpu_ffn"),
             ceil_plan.get("max_ctx"), fit_plan.get("n_gpu_layers"),
             fit_plan.get("n_cpu_ffn"),
             rk.get("plan_mode"), "OK" if two_ok else "FAIL"))
    ok = ok and two_ok
    # 5aa) The fit plan's FFN give-back, which is the dense answer to the
    #      MoE expert split: exiling FFN frees VRAM without costing any KV, so
    #      with every block on the GPU the search wants the SMALLEST exile that
    #      fits. A bigger card must therefore keep MORE FFN in VRAM, never less
    #      - the old planner pinned every block unconditionally and left the
    #      spare VRAM unused while streaming those weights over PCIe per token.
    ffn_at = []
    for budget in (30, 60, 120, 400):
        rr = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=budget,
                     ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=5,
                     safety_pct=0, plan_mode="fit")
        cp = (rr.get("plans") or {}).get("fit") or rr["plan"]
        ffn_at.append((budget, cp.get("n_gpu_layers"), cp.get("n_cpu_ffn") or 0))
    # more VRAM must never mean MORE exiled, and somewhere it must mean less -
    # otherwise the give-back is not happening at all
    give_ok = all(a[2] >= b[2] for a, b in zip(ffn_at, ffn_at[1:]))
    give_ok = give_ok and ffn_at[0][2] > ffn_at[-1][2]
    # ...and the emitted -ot count is the plan's, not a hardcoded "all blocks"
    rr = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=60, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=5, safety_pct=0,
                 plan_mode="fit")
    cp = (rr.get("plans") or {}).get("fit") or rr["plan"]
    cmd = cp.get("llama_cmd") or ""
    give_ok = give_ok and (("-ot" in cmd) == bool(cp.get("n_cpu_ffn")))
    print("  2PLAN  fit plan gives FFN back as VRAM allows: %s  %s"
          % (" ".join("%s->ot%s" % (b, f) for b, _, f in ffn_at),
             "OK" if give_ok else "FAIL"))
    ok = ok and give_ok

    # 5a) The automatic rule, and the override of it. Absent a mode the plan
    #     PREDICTED FASTER AT THE REQUESTED CONTEXT wins. The old rule keyed on
    #     coverage (the ceiling plan's max_ctx >= ctx), but coverage only says
    #     which plan CAN answer the context - when both can, the fit plan usually
    #     streams less per token and is the faster one, which is the inversion
    #     this assertion exists to catch. An explicit mode is obeyed either way,
    #     which is what the UI toggle rides on.
    auto = rk.get("plan_mode")
    pp = rk.get("plan_pick") or {}
    pp_c, pp_f = pp.get("ceiling") or {}, pp.get("fit") or {}
    if pp_c.get("tok_s_hi") and pp_f.get("tok_s_hi"):
        expect = "fit" if pp_f["tok_s_hi"] > pp_c["tok_s_hi"] else "ceiling"
    else:
        expect = pp.get("pick")
    forced = _two(plan_mode="fit")
    forced_c = _two(plan_mode="ceiling")
    mode_ok = (auto == expect
               and pp.get("pick") == auto
               and bool(pp.get("rule")) and bool(pp.get("reason"))
               and forced.get("plan_mode") == "fit"
               and forced["plan"]["kind"] == "dense_fit"
               and forced_c.get("plan_mode") == "ceiling"
               and forced_c["plan"]["kind"] == "dense_ceiling"
               # both plans are always returned, whichever one is reported
               and set((forced.get("plans") or {})) == {"ceiling", "fit"})
    print("  PMODE  auto=%s (expect %s, rule=%s), forced fit/ceiling honoured  %s"
          % (auto, expect, pp.get("rule"), "OK" if mode_ok else "FAIL"))
    ok = ok and mode_ok

    # 5b) An explicit knob means "verify the config I actually ran", so it must
    #     reach a planner that HAS that knob rather than being dropped. The mode
    #     planners answer only their own question and would return byte-identical
    #     plans for -ngl 1 and -ngl 3, which is what this catches.
    ro_a = _two(gpu_layers_override=1)
    ro_b = _two(gpu_layers_override=3)
    rf_a = _two(n_cpu_ffn_override=2)
    rf_b = _two(n_cpu_ffn_override=6)
    ovr_ok = (ro_a["plan"]["kind"] == "dense"                     # routed away from the modes
              and ro_a["plan"]["n_gpu_layers"] == 1               # ...and honoured
              and ro_b["plan"]["n_gpu_layers"] == 3
              and ro_a["plan"]["vram_used_mib"] != ro_b["plan"]["vram_used_mib"]
              and rf_a["plan"]["kind"] == "dense"
              and rf_a["plan"]["n_cpu_ffn"] == 2
              and rf_b["plan"]["n_cpu_ffn"] == 6
              and rf_a["plan"]["vram_used_mib"] != rf_b["plan"]["vram_used_mib"]
               # an override is not a mode: neither plans dict should be offered
               and "plans" not in ro_a and "plans" not in rf_a
               # the recommendation path (no override) must STILL reach the two
               # plans
               and rk["plan"]["kind"] in ("dense_ceiling", "dense_fit"))
    print("  NGL-OVR ngl 1->%.0f MiB, 3->%.0f MiB | ffn 2->%.0f MiB, 6->%.0f MiB  %s"
          % (ro_a["plan"]["vram_used_mib"], ro_b["plan"]["vram_used_mib"],
             rf_a["plan"]["vram_used_mib"], rf_b["plan"]["vram_used_mib"],
             "OK" if ovr_ok else "FAIL"))
    ok = ok and ovr_ok

    # 5c) Vision transients. Derived, not measured - so what is asserted here is the
    #     SHAPE, not the magnitude: scores are quadratic in patch count while the
    #     activation terms are linear, image tokens follow the spatial merge, and
    #     flash attention removes the quadratic term entirely. Those are the claims
    #     the code makes; the coefficients are not claims at all.
    vcfg = {"projector": "test", "patch": 16, "merge": 2, "hidden": 1152,
            "ffn_len": 4304, "blocks": 27, "heads": 16, "image_size": 768,
            "projection_dim": 5120}
    g1 = vision_grid(vcfg, 1024, 1024)
    g2 = vision_grid(vcfg, 2048, 2048)          # 2x each side -> 4x patches
    # flash_attn=False explicitly: the quadratic term only EXISTS unfused, and that
    # is the scaling being asserted. The default is True (see vision_peak_mib), so
    # relying on it here would silently test 0 == 0.
    p1 = vision_peak_mib(vcfg, g1, flash_attn=False)
    pk2 = vision_peak_mib(vcfg, g2, flash_attn=False)
    pfa = vision_peak_mib(vcfg, g2, flash_attn=True)
    vis_ok = (g1["n_patches"] == 64 * 64 and g2["n_patches"] == 4 * g1["n_patches"]
              and g1["image_tokens"] == g1["n_patches"] // 4          # merge 2x2
              # scores go as patches^2 -> 16x for 4x the patches
              and abs(pk2["scores_mib"] / p1["scores_mib"] - 16.0) < 0.01
              # activations go as patches -> 4x
              and abs(pk2["act_mib"] / p1["act_mib"] - 4.0) < 0.01
              and pfa["scores_mib"] == 0.0 and pfa["total_mib"] < pk2["total_mib"]
              # snapped down to a whole merge block (16*2 = 32)
              and vision_grid(vcfg, 1000, 1000)["width"] == 992)
    print("  VISION 1024px=%s patches -> %s tok; 2048px scores x%.1f, act x%.1f; "
          "fa removes scores=%s  %s"
          % (f"{g1['n_patches']:,}", f"{g1['image_tokens']:,}",
             pk2["scores_mib"] / p1["scores_mib"], pk2["act_mib"] / p1["act_mib"],
             pfa["scores_mib"] == 0.0, "OK" if vis_ok else "FAIL"))
    ok = ok and vis_ok

    # 5d) AC1 - no bandwidth anywhere, not even the probe: no plan may fall back
    #     on a fabricated figure. Every plan degrades to n/a, names the side it
    #     cannot score, and keeps only the byte split that is still true. Patch
    #     the probe as plan.py sees it, not the machine.
    from . import plan as _planmod
    _real_probe = _planmod.cached_bandwidth
    _planmod.cached_bandwidth = lambda fresh=False: {"vram_gbs": None, "ram_gbs": None}
    try:
        rna = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=30,
                      ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=5,
                      safety_pct=0, bw_vram_gbs=None, bw_ram_gbs=None, ctx_fill=1024)
    finally:
        _planmod.cached_bandwidth = _real_probe
    na_sp = [p_.get("speed") for p_ in (rna.get("plans") or {}).values()
             if isinstance(p_.get("speed"), dict)]
    na_ok = len(na_sp) >= 2
    for sp_ in na_sp:
        na_ok = (na_ok and sp_.get("ok") is False
                 and sp_.get("tok_s_hi") is None and sp_.get("tok_s_lo") is None
                 and sp_.get("missing")
                 and set(sp_.get("missing")) <= {"bw_vram_gbs", "bw_ram_gbs"}
                 and sp_.get("bw_vram_gbs") is None and sp_.get("bw_ram_gbs") is None
                 and sp_.get("bw_vram_source") == "none"
                 and sp_.get("bw_ram_source") == "none"
                 and bool(sp_.get("reason"))
                 and (sp_.get("gpu_mib", 0) > 0 or sp_.get("cpu_mib", 0) > 0))
    # per-side need: the probe above only covers what a whole plan does; this
    # checks estimate_speed directly - a plan that reads one device needs only
    # that device's number (head and KV follow the blocks off the GPU)
    _cl2 = classify_tensors(load_gguf(p2), rk["config"])
    _allb = list(range(nL2))
    # a stale calibrated value must never ride a degenerate payload - the UI
    # keys its hero on `calibrated`, and None.toFixed() is a whole results
    # panel replaced by an error string
    _deg = estimate_speed(rk["config"], _cl2, _allb, 1024,
                          "f16", 50.0, None, cpu_head=True, ram_eff=0.6)
    na_ok = (na_ok
             and "missing" not in estimate_speed(rk["config"], _cl2, _allb, 1024,
                                                 "f16", 50.0, None, cpu_head=False)
             and "missing" in estimate_speed(rk["config"], _cl2, _allb, 1024,
                                             "f16", None, 80.0, cpu_head=False)
             and "missing" not in estimate_speed(rk["config"], _cl2, _allb, 1024,
                                                 "f16", 50.0, 80.0, cpu_head=True)
             and _deg.get("tok_s") is None and _deg.get("calibrated") is False
             and "ram_eff" not in _deg)
    print("  SPEED-NA probe unavailable: %d/%d plans n/a (missing=%s), byte split kept  %s"
          % (sum(1 for sp_ in na_sp if sp_.get("ok") is False), len(na_sp),
             sorted({m for sp_ in na_sp for m in (sp_.get("missing") or [])}),
             "OK" if na_ok else "FAIL"))
    ok = ok and na_ok

    # 5e) AC2 - a plan's prediction is standalone: recomputed from that plan's OWN
    #     placement only (its -ngl / -ot, nothing about the sibling plan) it must
    #     reproduce the number analyze attached, to the bit.
    def _alone(p_):
        ngl = p_.get("n_gpu_layers")
        if ngl is None:
            ngl = nL2 if p_.get("fits_fully") else 0
        return estimate_speed(rk["config"], _cl2,
                              list(range(max(0, nL2 - int(ngl)), nL2)),
                              1024, "f16", 600.0, 80.0, cpu_head=True,
                              n_cpu_moe=p_.get("n_cpu_moe") or 0,
                              n_cpu_ffn=p_.get("n_cpu_ffn") or 0)
    solo_ok = True
    for _nm in ("ceiling", "fit"):
        _a = (pl.get(_nm) or {}).get("speed") or {}
        _b = _alone(pl.get(_nm) or {})
        for _k in ("tok_s_hi", "tok_s_lo", "gpu_mib", "cpu_mib", "expert_frac", "ctx_fill"):
            solo_ok = solo_ok and _a.get(_k) is not None and _a.get(_k) == _b.get(_k)
    _cxsp = (pl.get("fit") or {}).get("speed") or {}
    print("  SPEED-SOLO recomputed from own split: ceiling %.2f/%.2f, fit %.2f/%.2f  %s"
          % (sk.get("tok_s_hi") or 0.0, sk.get("tok_s_lo") or 0.0,
             _cxsp.get("tok_s_hi") or 0.0, _cxsp.get("tok_s_lo") or 0.0,
             "OK" if solo_ok else "FAIL"))
    ok = ok and solo_ok

    # 5f) AC3 - synthetic bandwidths, real consequence. The two plans bill the
    #     same bytes to different sides: the ceiling plan exiles the whole dense
    #     FFN to RAM (streaming it every token) but keeps every block's KV on the
    #     GPU; the fit plan keeps as much FFN on the GPU as the card allows and,
    #     once the card is tight enough, starts evicting whole blocks -
    #     weights AND KV - to the slow side. A generous card favours the fit
    #     plan (FFN stays on the fast side); a tight one favours the ceiling
    #     plan (all KV stays on the fast side), so the PREDICTED ordering must
    #     flip somewhere in the family. A property of placement, not of the model:
    #     no single card can show it.
    flip_rows = []
    for b in (30, 25, 20, 15, 10, 8):
        fr = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=b,
                     ram_budget_mib=8000, gpu_reserve_mib=0, compute_override_mib=5,
                     safety_pct=0, bw_vram_gbs=50, bw_ram_gbs=25)
        fps = fr.get("plans") or {}
        hi = {k: (v.get("speed") or {}).get("tok_s_hi") for k, v in fps.items()}
        if hi.get("ceiling") is None or hi.get("fit") is None:
            continue
        flip_rows.append((b, hi["ceiling"], hi["fit"]))
    flip_ok = (len(flip_rows) >= 3
               and any(s - c > 0 for _, s, c in flip_rows)
               and any(s - c < 0 for _, s, c in flip_rows))
    print("  SPEED-FLIP bw 50/25, predicted tok_s_hi ceiling/fit as the card shrinks: "
          "%s  %s"
          % (" ".join("b%d:%.2f/%.2f" % (b, s, c) for b, s, c in flip_rows),
             "OK" if flip_ok else "FAIL"))
    ok = ok and flip_ok

    # 6) sliding-window attention: windowed layers must cap at their window (and
    #    use their own head dims), or KV is overstated by 10-20x at long context.
    swa_t = [("token_embd.weight", [hid, 4000], 12)]
    for i in range(nL):
        swa_t += [("blk.%d.attn_q.weight" % i, [hid, hid], 12),
                  ("blk.%d.attn_k.weight" % i, [hid, nkv*hd], 12),
                  ("blk.%d.attn_v.weight" % i, [hid, nkv*hd], 12),
                  ("blk.%d.ffn_down.weight" % i, [1536, hid], 12)]
    swa_t += [("output.weight", [hid, 4000], 14)]
    p5 = os.path.join(tmp, "swa.gguf")
    _write_gguf(p5, {"g.block_count": nL, "g.attention.head_count": nh,
                     "g.attention.head_count_kv": nkv, "g.embedding_length": hid,
                     "g.context_length": 32768, "g.feed_forward_length": 1536,
                     "g.attention.key_length": 128, "g.attention.value_length": 128,
                     "g.attention.key_length_swa": 64, "g.attention.value_length_swa": 64,
                     "g.attention.sliding_window": 1024,
                     # 1 = windowed, 0 = full attention; last of every 4 is global
                     "g.attention.sliding_window_pattern": [1, 1, 1, 0] * (nL // 4)},
                {"general.architecture": "g", "general.name": "SWATest"}, swa_t)
    rs = analyze(p5, 32768, "f16", 512, True, vram_budget_mib=4000, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=10, safety_pct=0, n_seq=1)
    sw, ss = rs["swa"], rs["sizes_mib"]
    n_glob, n_win = nL // 4, nL - nL // 4
    win_tok = 1024 + 512                                    # window*seq + ubatch, pads to 1536
    exp_kv = _mib(nkv * (128 + 128) * 2.0 * 32768 * n_glob +
                  nkv * (64 + 64) * 2.0 * win_tok * n_win)
    # and the flat part must stay flat: doubling ctx only grows the global layers
    kvt = {t["ctx"]: t["kv_mib"] for t in rs["kv_table"]}
    flat_ok = abs((kvt[32768] - kvt[16384]) -
                  _mib(nkv * (128 + 128) * 2.0 * 16384 * n_glob)) < 0.5
    swa_ok = (sw["enabled"] and sw["n_swa_layers"] == n_win and sw["n_global_layers"] == n_glob
              and sw["window_cache_tokens"] == win_tok
              and abs(ss["kv_total"] - exp_kv) < 0.5 and flat_ok)
    print("  SWA    %d windowed / %d global, window=%d tok  kv=%.1f MiB (expect %.1f) "
          "naive=%.1f  %s"
          % (sw["n_swa_layers"], sw["n_global_layers"], sw["window_cache_tokens"],
             ss["kv_total"], exp_kv, _mib(nkv * 256 * 2.0 * 32768 * nL),
             "OK" if swa_ok else "FAIL"))
    ok = ok and swa_ok

    # 7) the compute buffer splits across backends, the floor tracks how many blocks
    #    are resident, and the output tensor stays on the host at BOTH placements.
    #    That last one is the correction: it used to follow the output head onto the
    #    GPU, and across 144 measured loads there is no CUDA0.output buffer at all.
    rc = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=60, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=None, safety_pct=0,
                 gpu_layers_override=2)
    rf = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=99999, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=None, safety_pct=0)
    out, gr = rc["sizes_mib"]["compute_output"], rc["sizes_mib"]["compute_graph"]
    n_all = rc["config"]["n_layers"]
    floor2 = CB_CUDA_CTX_MIB + CB_DEFAULTS["floor"] * 2
    floor_all = CB_CUDA_CTX_MIB + CB_DEFAULTS["floor"] * n_all
    split_x = CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * 4096)
    # The output tensor is 4 bytes x n_vocab, for ONE output token - not the ubatch.
    # Asserted against the formula rather than "> 0" because the synthetic model has
    # a tiny vocabulary and rounds to 0.0 MiB, which is correct and would otherwise
    # look like a failure.
    exp_out = round(_mib(4.0 * rc["config"]["n_vocab"] * 1), 1)   # n_seq = 1 here
    cb_ok = (abs(rc["plan"]["compute_mib"] - (gr + floor2 + split_x)) < 0.05
             and abs(rc["plan"]["cpu_compute_mib"] - (gr + out)) < 0.05
             # full offload: no surcharge, the bigger floor, and the output tensor
             # still on the host - so the CPU side is exactly the output tensor
             and abs(rf["plan"]["cpu_compute_mib"] - out) < 0.05
             and abs(rf["plan"]["compute_mib"] - (gr + floor_all)) < 0.05
             and abs(out - exp_out) < 0.05)
    print("  CMPBUF split ngl=2: gpu=%.1f cpu=%.1f | full offload: gpu=%.1f cpu=%.1f "
          "(graph=%.1f output=%.2f floor %.1f->%.1f split=%.1f)  %s"
          % (rc["plan"]["compute_mib"], rc["plan"]["cpu_compute_mib"],
             rf["plan"]["compute_mib"], rf["plan"]["cpu_compute_mib"], gr, out,
             floor2, floor_all, split_x, "OK" if cb_ok else "FAIL"))
    ok = ok and cb_ok

    # 8) regression-lock the measured VRAM model. These are real llama-server runs
    #    (RTX 5070 Ti, CUDA 12, llama.cpp 2.27.1, gemma-4-31B-it-QAT-Q4_0), read off
    #    the GPU process-memory counter. They only run when that model is present.
    mroot = os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
    REF = {
        "g31": (os.path.join(mroot, "lmstudio-community", "gemma-4-31B-it-QAT-GGUF",
                             "gemma-4-31B-it-QAT-Q4_0.gguf"), "gemma4 dense+SWA"),
        "g26": (os.path.join(mroot, "lmstudio-community", "gemma-4-26B-A4B-it-QAT-GGUF",
                             "gemma-4-26B-A4B-it-QAT-Q4_0.gguf"), "gemma4 MoE+SWA"),
        "q27": (os.path.join(mroot, "unsloth", "Qwen3.6-27B-GGUF",
                             "Qwen3.6-27B-UD-Q4_K_XL.gguf"), "qwen35 hybrid-SSM"),
        "q35": (os.path.join(mroot, "unsloth", "Qwen3.6-35B-A3B-GGUF",
                             "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), "qwen35moe hybrid"),
        "q9":  (os.path.join(mroot, "unsloth", "Qwen3.5-9B-MTP-GGUF",
                             "Qwen3.5-9B-Q6_K.gguf"), "qwen35 hybrid + MTP"),
    }
    # model, ctx, ngl, seq, ubatch, flash, kv, measured MiB, projector
    obs = [("g31", 32768,  4, 1, 512, True,  "q8_0",  2797.7, False),
           ("g31", 65536,  4, 1, 512, True,  "q8_0",  3357.7, False),
           ("g31", 32768,  8, 1, 512, True,  "q8_0",  4029.7, False),
           ("g31", 131072, 1, 1, 512, True,  "q8_0",  3091.8, False),
           ("g31", 32768,  1, 1, 512, True,  "q8_0",  1819.6, False),
           ("g31", 32768,  1, 1, 512, False, "f16",   3699.8, False),
           ("g31", 65536,  1, 1, 512, False, "f16",   6067.9, False),
           ("g31", 131072, 4, 4, 512, True,  "q8_0",  5826.0, True),   # LM Studio run
           ("g31", 262144, 14,2, 512, True,  "q8_0", 11541.0, True),   # LM Studio run
           ("g26", 32768,  2, 1, 512, True,  "f16",   1827.6, False),
           ("g26", 32768,  6, 1, 512, True,  "f16",   3627.6, False),
           ("g26", 65536,  2, 1, 512, True,  "f16",   1865.7, False),
           ("q27", 32768,  2, 1, 512, True,  "f16",   1801.7, False),
           ("q27", 32768,  6, 1, 512, True,  "f16",   2947.7, False),
           ("q27", 65536,  2, 1, 512, True,  "f16",   2121.7, False),
           ("q35", 32768,  2, 1, 512, True,  "f16",   1637.6, False),
           ("q35", 32768,  6, 1, 512, True,  "f16",   3751.7, False),
           ("q35", 65536,  2, 1, 512, True,  "f16",   1797.7, False),
           # ubatch sweeps - these are what showed the activation scratch scales
           # with hidden size, not with FFN width
           ("q27", 32768,  2, 1, 128, True,  "f16",   1741.6, False),
           ("q27", 32768,  2, 1, 2048,True,  "f16",   2364.0, False),
           ("q35", 32768,  2, 1, 128, True,  "f16",   1519.6, False),
           ("q35", 32768,  2, 1, 2048,True,  "f16",   1849.9, False),
           # minimal probes - the scaling terms nearly vanish, exposing the pool
           ("g31", 2048,   1, 1, 64,  True,  "f16",   1353.5, False),
           ("g26", 2048,   1, 1, 64,  True,  "f16",   1019.5, False),
           ("q27", 2048,   1, 1, 64,  True,  "f16",   1301.5, False),
           ("q35", 2048,   1, 1, 64,  True,  "f16",    895.5, False),
           # multi-sequence: exercises the recurrent state and the SWA window
           ("q27", 32768,  6, 8, 512, True,  "f16",   3035.7, False),
           ("q27", 32768,  6, 2, 512, True,  "f16",   2967.6, False),
           # FULL offload, where embeddings and the output head decide the answer.
           # This model is 22% embed+head, so charging them to VRAM was a ~1.6 GB
           # error that stayed hidden on the bigger models above.
           ("q9",  8192,  33, 1, 512, True,  "q8_0",  6827.6, False),
           ("q9",  8192,  99, 1, 512, True,  "q8_0",  6983.6, False),
           ("q9",  65536, 99, 1, 512, True,  "q8_0",  8215.7, False),
           ("q9",  131072,99, 1, 512, True,  "q8_0",  9623.9, False),
           ("q9",  8192,   1, 1, 512, True,  "q8_0",  1087.6, False),
           ("q9",  8192,   2, 1, 512, True,  "q8_0",  1285.6, False),
           ("q9",  8192,  31, 1, 512, True,  "q8_0",  6477.6, False),
           # PARTIAL offload at long context, and -ngl 0. These are the rows that
           # exposed the compute buffer being charged nothing to the GPU unless a
           # layer landed there - which is what let plans overcommit and spill.
           # -ngl 0 is not a case anyone plans for, but it isolates the pool
           # perfectly: no weights, no KV on the GPU, so what is left is the term
           # that used to be invisible.
           ("q9",  32768,  0, 1, 512, True,  "q8_0",   471.6, False),
           ("q9",  65536,  0, 1, 512, True,  "q8_0",   699.7, False),
           ("q9",  131072, 0, 1, 512, True,  "q8_0",  1155.8, False),
           ("q9",  262144, 0, 1, 512, True,  "q8_0",  2068.1, False),
           ("q9",  32768,  8, 1, 512, True,  "q8_0",  2661.7, False),
           ("q9",  32768, 16, 1, 512, True,  "q8_0",  4191.7, False),
           ("q9",  32768, 23, 1, 512, True,  "q8_0",  5543.7, False),
           ("q9",  131072,23, 1, 512, True,  "q8_0",  7451.9, False),
           ("q9",  262144,23, 1, 512, True,  "q8_0",  9996.1, False),
           ("q9",  32768, 33, 1, 512, True,  "q8_0",  7355.7, False),
           ("q9",  32768, 33, 1, 512, True,  "q8_0",  9301.7, True)]
    # Thresholds are what the model actually achieves, stated out loud rather than
    # aspired to. Mean is the number that describes typical use; worst is dominated
    # by tiny-context, near-zero-offload corners where the total is ~1 GiB and a
    # 200 MiB miss reads as 20%. Both are asserted so a regression in either fails.
    # The signed bias is printed because DIRECTION matters more than magnitude here:
    # over-predicting makes a plan conservative, under-predicting makes it spill.
    VRAMFIT_MEAN_MAX, VRAMFIT_WORST_MAX = 12.0, 45.0
    this_gpu = _active_gpu()
    hw_match = (this_gpu == REF_GPU)
    seen, worst, worst_lbl, errs, bias = set(), 0.0, "", [], []
    for key, ctx, ngl, seq, ub, fa, kt, meas, proj in obs:
        if not hw_match:
            break
        path, _ = REF[key]
        if not os.path.isfile(path):
            continue
        rr = analyze(path, ctx, kt, ub, fa, vram_budget_mib=11805,
                     ram_budget_mib=30165, gpu_reserve_mib=0,
                     compute_override_mib=None, safety_pct=0, n_seq=seq,
                     gpu_layers_override=ngl, include_mmproj=proj)
        got = rr["plan"]["vram_used_mib"]
        err = abs(got - meas) / meas * 100.0
        errs.append(err); bias.append((got - meas) / meas * 100.0); seen.add(key)
        if err > worst:
            worst, worst_lbl = err, "%s ctx %d ngl %d fa %d" % (key, ctx, ngl, fa)
    n = len(errs)
    if not hw_match:
        # A skip used to leave `ok` untouched, so the suite printed PASSED having
        # checked nothing against reality. Say so loudly instead.
        print("  VRAMFIT *** SKIPPED - NOT VALIDATED *** (reference data measured on "
              "%s; this is %s)" % (REF_GPU, this_gpu or "no NVIDIA GPU"))
        skipped_real.append("VRAMFIT (wrong GPU)")
    elif n:
        mean = sum(errs) / n
        vram_ok = mean <= VRAMFIT_MEAN_MAX and worst <= VRAMFIT_WORST_MAX
        print("  VRAMFIT mean %.1f%% (<=%.0f) worst %.1f%% (<=%.0f, %s) bias %+.1f%% "
              "over %d runs / %d models  %s"
              % (mean, VRAMFIT_MEAN_MAX, worst, VRAMFIT_WORST_MAX, worst_lbl,
                 sum(bias) / n, n, len(seen), "OK" if vram_ok else "FAIL"))
        ok = ok and vram_ok
    else:
        print("  VRAMFIT *** SKIPPED - NOT VALIDATED *** (no reference models present)")
        skipped_real.append("VRAMFIT (no models)")

    # 8b) the recurrent (SSM) state must scale with n_seq exactly as llama.cpp
    #     allocates it. Measured by holding ctx/ngl fixed and varying -np, which
    #     leaves weights and KV untouched: q27 -np 1 -> 8 moved 2947.7 -> 3035.7.
    q27p = REF["q27"][0]
    if not hw_match:
        print("  SSMSEQ  skipped (reference data measured on %s)" % REF_GPU)
        skipped_real.append("SSMSEQ")
    elif os.path.isfile(q27p):
        def _vram(seq):
            rr = analyze(q27p, 32768, "f16", 512, True, vram_budget_mib=1 << 20,
                         ram_budget_mib=1 << 20, gpu_reserve_mib=0,
                         compute_override_mib=0.001, safety_pct=0, n_seq=seq,
                         gpu_layers_override=6, include_mmproj=False)
            p = rr["plan"]
            return p["gpu_weights_mib"] + p["gpu_kv_mib"] + p.get("gpu_recurrent_mib", 0)
        pred_d, meas_d = _vram(8) - _vram(1), 3035.7 - 2947.7
        rec_ok = abs(pred_d - meas_d) <= 8.0
        print("  SSMSEQ -np 1->8 recurrent growth: predicted %.1f MiB, measured %.1f  %s"
              % (pred_d, meas_d, "OK" if rec_ok else "FAIL"))
        ok = ok and rec_ok

    # 8c) routed experts must be recognised whatever the tensors are called. Gemma 4
    #     MoE fuses gate+up into ffn_gate_up_exps; matching only (gate|up|down)_exps
    #     silently counted 2/3 of the expert weight as non-offloadable dense weight,
    #     which broke both the active-param count and the whole expert-offload plan.
    for nm, want in [("blk.0.ffn_gate_exps.weight", True), ("blk.0.ffn_up_exps.weight", True),
                     ("blk.0.ffn_down_exps.weight", True), ("blk.0.ffn_gate_up_exps.weight", True),
                     ("blk.0.ffn_down_exps.scale", True), ("blk.0.ffn_gate_inp.weight", False),
                     ("blk.0.ffn_down_shexp.weight", False), ("blk.0.ffn_down.weight", False)]:
        got = bool(RE_EXPS.match(nm))
        if got != want:
            print("  EXPERTS %-34s expected=%s got=%s  FAIL" % (nm, want, got))
            ok = False
    # and the whole-model check: "A4B" in the name means ~4B active parameters
    g26p = REF["g26"][0]
    if os.path.isfile(g26p):
        rg = analyze(g26p, 8192, "q8_0", 512, True, vram_budget_mib=11509,
                     ram_budget_mib=30165, gpu_reserve_mib=0, compute_override_mib=None,
                     safety_pct=5, n_seq=1, include_mmproj=False)
        act_b = rg["active_params"] / 1e9
        exp_ok = 3.0 <= act_b <= 5.0 and rg["plan"]["kind"] == "moe"
        print("  EXPERTS gemma-4-26B-A4B active params %.2fB (name says ~4B)  %s"
              % (act_b, "OK" if exp_ok else "FAIL"))
        ok = ok and exp_ok

    # 8d) LM Studio writes "max" for a full offload. Parsing only digits skipped
    #     those blocks entirely and reported the PREVIOUS load's numbers, which is
    #     worse than reporting nothing - it looks like a valid reading.
    logtxt = (
        "[LM Studio] Model load size estimate with raw num offload layers '17' "
        "and context length '8192':\n  Total: 9.67 GB\n"
        "[LM Studio] Resolved GPU config options:\n  Num Offload Layers: 17\n"
        "  Num CPU Expert Layers: 0\n  Main GPU: 0\n"
        "[LM Studio] Model load size estimate with raw num offload layers 'max' "
        "and context length '262144':\n  Total: 35.58 GB\n"
        "[LM Studio] Resolved GPU config options:\n  Num Offload Layers: max\n"
        "  Num CPU Expert Layers: 29\n  Main GPU: 0\n")
    tmplog = os.path.join(tmp, "main.log")
    open(tmplog, "w", encoding="utf-8").write(logtxt)
    _oldapp = os.environ.get("APPDATA")
    os.makedirs(os.path.join(tmp, "fakeapp", "LM Studio", "logs"), exist_ok=True)
    open(os.path.join(tmp, "fakeapp", "LM Studio", "logs", "main.log"),
         "w", encoding="utf-8").write(logtxt)
    os.environ["APPDATA"] = os.path.join(tmp, "fakeapp")
    try:
        rt = read_lmstudio_runtime()
    finally:
        if _oldapp is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = _oldapp
    rt_ok = (rt is not None and rt.get("all_layers") is True
             and resolve_runtime_ngl(rt, 30) == 30
             and rt.get("context") == 262144 and rt.get("n_cpu_moe") == 29)
    print("  LMSLOG 'max' offload parsed: ngl=%s ctx=%s cpu_experts=%s  %s"
          % (resolve_runtime_ngl(rt, 30) if rt else None,
             (rt or {}).get("context"), (rt or {}).get("n_cpu_moe"),
             "OK" if rt_ok else "FAIL"))
    ok = ok and rt_ok

    # 8e) multi-token-prediction blocks. They sit at the end of the block list and
    #     look like ordinary transformer blocks, so they get counted twice over:
    #     once in the weights (correct - measurement shows they ARE resident) and
    #     once in the KV cache (wrong - they never grow one). Measured on
    #     Qwen3.5-9B-MTP: the ctx-slope over 8k->131k is 2640.3 MiB, which matches
    #     8 attention layers (-1.5%), not the 9 the block list implies (+10.9%).
    mtpp = os.path.join(mroot, "unsloth", "Qwen3.5-9B-MTP-GGUF", "Qwen3.5-9B-Q6_K.gguf")
    if os.path.isfile(mtpp):
        mc = extract_config(load_gguf(mtpp))
        struct_ok = (mc["n_mtp_layers"] == 1 and mc["mtp_layers"] == [32]
                     and 32 not in mc["attn_layers"]
                     and mc["kv_heads_per_layer"][32] == 0
                     and len(mc["attn_layers"]) == 8)
        def _v(ctx):
            rr = analyze(mtpp, ctx, "q8_0", 512, True, vram_budget_mib=1 << 20,
                         ram_budget_mib=1 << 20, gpu_reserve_mib=0,
                         compute_override_mib=None, safety_pct=0, n_seq=1,
                         gpu_layers_override=mc["n_layers"], include_mmproj=False)
            return rr["plan"]["vram_used_mib"]
        # The structural facts are the actual assertion - they are exact. The slope
        # is the evidence that justified them, kept here only as a sanity net; a
        # tight bound on it would really be testing the compute-buffer coefficients,
        # which are fitted and which calibration legitimately moves.
        slope = _v(131072) - _v(8192)
        slope_err = abs(slope - 2640.3) / 2640.3 * 100.0
        mtp_ok = struct_ok and slope_err <= 10.0
        print("  MTP    %d nextn block(s), %d attention layers (not %d), ctx-slope "
              "%.0f MiB vs measured 2640 (%.1f%% on shipped defaults)  %s"
              % (mc["n_mtp_layers"], len(mc["attn_layers"]), len(mc["attn_layers"]) + 1,
                 slope, slope_err, "OK" if mtp_ok else "FAIL"))
        ok = ok and mtp_ok
    else:
        print("  MTP    skipped (reference MTP model not present)")
        skipped_real.append("MTP")

    # 9) the calibration fitter must recover known coefficients from synthetic rows,
    #    and must REFUSE to free terms the data cannot identify (fitting four
    #    coefficients to two points would be worse than shipping the defaults).
    # "ctx" is bytes per context token now, not a fraction of a KV token, so the
    # synthetic truth has to live on that scale or the plausibility guard will
    # (correctly) refuse to free it.
    # "floor" is MiB of kernel modules per resident block, so the synthetic truth
    # lives on that scale - a few MiB, not the hundreds the old intercept carried.
    truth = {"floor": 9.0, "ctx": 1100.0, "act": 40.0, "nofa": 5.0}
    def synth(ctx, ub, fa, hid=4096, nh=32, kvtok=8192, ngl=16, n_layers=32,
              n_cpu_moe=0, n_vocab=32000, moe_width=0):
        r = {"ctx": ctx, "ub": ub, "fa": fa, "hidden": hid, "n_head": nh,
             "kv_type": "f16", "kv_tok_ctx": kvtok * ctx, "exact_mib": 1000.0,
             "measured_mib": 1000.0, "ngl": ngl, "n_layers": n_layers,
             "n_cpu_moe": n_cpu_moe, "n_vocab": n_vocab, "moe_width": moe_width}
        d = _design(r)
        r["overhead_mib"] = _struct_offset(r) + sum(truth[k] * d[k] for k in truth)
        return r
    rich = [synth(8192, 512, True), synth(32768, 512, True), synth(131072, 512, True),
            synth(32768, 128, True), synth(32768, 2048, True), synth(32768, 512, False),
            synth(65536, 1024, True)]
    fr = fit_calibration(rich)
    got = fr["coeffs"]
    recovered = all(abs(got[k] - truth[k]) <= abs(truth[k]) * 0.02 for k in truth)
    print("  CALFIT rich: freed %-24s recovered=%s residual %.2f%%  %s"
          % (",".join(fr["free"]), recovered, fr["residual_pct"],
             "OK" if (recovered and len(fr["free"]) == 4) else "FAIL"))
    ok = ok and recovered and len(fr["free"]) == 4

    # one measurement: only the constant may move, everything else stays at prior
    one = fit_calibration([synth(32768, 512, True)])
    lean_ok = (one["free"] == ["floor"]
               and all(abs(one["coeffs"][k] - CB_DEFAULTS[k]) < 1e-9
                       for k in ("ctx", "act", "nofa")))
    # and that single constant must still reproduce the observation exactly
    lean_ok = lean_ok and one["residual_pct"] < 0.01
    print("  CALFIT lean: freed %-24s others held at prior=%s  %s"
          % (",".join(one["free"]), lean_ok, "OK" if lean_ok else "FAIL"))
    ok = ok and lean_ok

    # rows that vary only in context must not free the ubatch/no-FA terms
    ctxonly = fit_calibration([synth(8192, 512, True), synth(32768, 512, True),
                               synth(131072, 512, True)])
    sel_ok = ("ctx" in ctxonly["free"] and "act" not in ctxonly["free"]
              and "nofa" not in ctxonly["free"])
    print("  CALFIT ctx-only: freed %-20s (act/nofa correctly withheld=%s)  %s"
          % (",".join(ctxonly["free"]), sel_ok, "OK" if sel_ok else "FAIL"))
    ok = ok and sel_ok

    # Several models at ONE ubatch make the hidden*ubatch column look varied when
    # only hidden changed - and hidden is confounded with the per-hidden structural
    # constant. Fitting "act" from that yielded a coefficient 9x the prior on real
    # user data. The knob, not the regressor, has to have moved.
    multimodel = [synth(32768, 512, True, hid=h, kvtok=k)
                  for h, k in ((2048, 4096), (2816, 8192), (5120, 8192), (5376, 16384))]
    mm = fit_calibration(multimodel)
    mm_ok = "act" not in mm["free"] and mm["coeffs"]["act"] == CB_DEFAULTS["act"]
    print("  CALFIT 4 models @ one ubatch: freed %-14s (act held at prior=%s)  %s"
          % (",".join(mm["free"]), mm_ok, "OK" if mm_ok else "FAIL"))
    ok = ok and mm_ok

    # a physically impossible or wildly extrapolated slope must be rejected in
    # favour of the shipped prior rather than published as a calibration
    absurd = [dict(r) for r in rich]
    for r in absurd:
        r["overhead_mib"] = r["overhead_mib"] * 50.0 + 5000.0
    ab = fit_calibration(absurd)
    # Rejecting outright is a valid answer; publishing an implausible slope is not.
    ab_ok = ab is None or (
        all(ab["coeffs"][t] == CB_DEFAULTS[t] or 0.1 <= ab["coeffs"][t] / CB_DEFAULTS[t] <= 10.0
            for t in ("ctx", "act", "nofa"))
        and all(ab["coeffs"][t] >= 0 for t in ("ctx", "act", "nofa")))
    print("  CALFIT absurd data: %-28s  %s"
          % ("rejected entirely" if ab is None else "freed " + ",".join(ab["free"]),
             "OK" if ab_ok else "FAIL"))
    ok = ok and ab_ok

    # A single measurement must still produce a usable fit - rejecting it stranded
    # real measurements as "calibrated from 0 measurements".
    # A real sweep row: Qwen3.6-35B-A3B at ctx 32768, -ngl 20 of 40. The allocator
    # reported 10155.2 MiB of weights + 320.0 of KV on the card and 317.0 of compute
    # buffer; the process counter read 10997.7, so the exact terms are 10475.2 and
    # the overhead this fit has to explain is 522.5.
    one_real = [{"ctx": 32768, "ub": 512, "fa": True, "hidden": 2048, "n_head": 16,
                 "kv_tok_ctx": 32768 * 512.0, "exact_mib": 10475.2, "overhead_mib": 522.5,
                 "measured_mib": 10997.7, "ngl": 20, "n_layers": 40, "n_cpu_moe": 0,
                 "n_vocab": 248320, "moe_width": 4096}]
    single = fit_calibration(one_real)
    single_ok = (single is not None and single["free"] == ["floor"]
                 and single["coeffs"]["floor"] >= 0.0)
    print("  CALFIT single real measurement: fitted=%s floor=%.2f MiB/layer  %s"
          % (single is not None, (single or {}).get("coeffs", {}).get("floor", 0.0),
             "OK" if single_ok else "FAIL"))
    ok = ok and single_ok

    # 9b) THE symmetry that broke. Every term the planner charges to VRAM must be
    #     removed from the observation by _struct_offset or _design, or the fit
    #     absorbs it into `const` and prediction then bills it a second time.
    #     --n-cpu-moe is the case that regressed: ngl == n_layers there, so a
    #     `ngl < n_layers` test called the graph unsplit while the planner - right,
    #     the experts really do run on the CPU - called it split. Worth ~850 MiB of
    #     phantom VRAM on a 26B MoE at 262k, two expert layers' worth.
    sym_ok = True
    for label, sngl, sncm in (("dense partial", 16, 0), ("dense full", 32, 0),
                              ("experts on cpu", 32, 20), ("moe partial", 24, 20)):
        srow = synth(65536, 512, True, ngl=sngl, n_cpu_moe=sncm)
        scfg = {"hidden": srow["hidden"], "n_head": srow["n_head"],
                "n_vocab": srow["n_vocab"], "attn_layers": [0], "n_layers": 32}
        st = compute_buffer_terms(scfg, srow["ctx"], srow["ub"], srow["fa"], 1, "f16")
        charged = compute_buffer_split(st, True, graph_is_split(32, sngl, sncm),
                                       sngl)["gpu"]
        sd = _design(srow)
        removed = _struct_offset(srow) + sum(calib_coeffs()[k] * sd[k] for k in CALIB_TERMS)
        hit = abs(charged - removed) <= 0.5
        sym_ok = sym_ok and hit
        print("  CALSYM %-15s ngl=%2d n_cpu_moe=%2d  planner charges %8.1f, fit removes "
              "%8.1f  %s" % (label, sngl, sncm, charged, removed, "OK" if hit else "FAIL"))
    ok = ok and sym_ok

    # ...and a measurement whose reading did not respond to the config must be
    # flagged, not fitted: overhead = measured - exact turns a clamped reading into
    # a compute buffer that shrinks as layers are added.
    flat = [dict(synth(262144, 512, True, ngl=n), model="flat.gguf", gpu="g",
                 exact_mib=8000.0 + 500.0 * i, measured_mib=11500.0)
            for i, n in enumerate((18, 22, 26))]
    mark_unreliable(flat)
    flat_ok = all(r.get("unreliable") for r in flat)
    moving = [dict(synth(262144, 512, True, ngl=n), model="ok.gguf", gpu="g",
                   exact_mib=8000.0 + 500.0 * i, measured_mib=9000.0 + 500.0 * i)
              for i, n in enumerate((18, 22, 26))]
    mark_unreliable(moving)
    flat_ok = flat_ok and not any(r.get("unreliable") for r in moving)
    print("  CALFLAT clamped rows flagged=%s, responsive rows kept=%s  %s"
          % (all(r.get("unreliable") for r in flat),
             not any(r.get("unreliable") for r in moving), "OK" if flat_ok else "FAIL"))
    ok = ok and flat_ok

    # 9c) THE FREEZE. calib_coeffs() must be a pure read: the same call, twice,
    #     with the store changing underneath, has to return the same numbers.
    #     Before the fit was stored, every call refitted from whatever rows
    #     happened to be present, so pressing Measure in the UI - or a schema bump
    #     re-deriving overhead_mib on every row - silently moved the coefficients
    #     of a plan already on screen. The symptom is a planner that reports two
    #     different answers for one config minutes apart, which is indistinguish-
    #     able from a bug in the model itself.
    import json as _json, tempfile as _tf, vram_planner.calib as _cal
    _saved_cache, _saved_loaded = dict(_CALIB_CACHE), _cal._CALIB_LOADED
    _saved_store = _cal._calib_store
    _tmp = os.path.join(_tf.gettempdir(), "vram_planner_calfrz_store.json")
    def _row(ngl, overhead):
        return dict(synth(32768, 512, True, ngl=ngl), model="frz.gguf", gpu="g",
                    backend=current_backend(), exact_mib=10000.0,
                    measured_mib=10000.0 + overhead, overhead_mib=overhead)
    try:
        _cal._calib_store = lambda: _tmp
        _cal._CALIB_LOADED = True                # we drive the load path by hand

        # one measurement, fitted and stored
        _json.dump({"rows": [_row(20, 520.0)], "schema": CALIB_SCHEMA},
                   open(_tmp, "w", encoding="utf-8"))
        _cal.refresh_calibration("g", force=True)
        first = calib_coeffs("g")["floor"]
        stored = _json.load(open(_tmp, encoding="utf-8")).get("fits", {}).get("g", {})

        # now the rows change underneath - a Measure from another window, or a
        # migration re-deriving overhead_mib. Reads must not notice.
        d = _json.load(open(_tmp, encoding="utf-8"))
        d["rows"] = [_row(20, 520.0), _row(30, 2400.0), _row(10, 90.0)]
        _json.dump(d, open(_tmp, "w", encoding="utf-8"))
        _cal.refresh_calibration("g")            # a read, not a refit
        second = calib_coeffs("g")["floor"]

        # ...until the refit is actually asked for.
        _cal.refresh_calibration("g", force=True)
        third = calib_coeffs("g")["floor"]

        frozen_ok = (stored.get("coeffs", {}).get("floor") == first
                     and second == first and third != first)
        print("  CALFRZ fit stored=%.2f, survives new rows=%.2f, refits on demand=%.2f"
              "  %s" % (first, second, third, "OK" if frozen_ok else "FAIL"))
        ok = ok and frozen_ok

        _CALIB_CACHE["g"] = {"n": 3, "free": ["floor"], "residual_pct": 1.0,
                             "coeffs": dict(CB_DEFAULTS, floor=7.25),
                             "backend": "b1", "fit_schema": _cal.CALIB_FIT_SCHEMA,
                             "row_schema": CALIB_SCHEMA}

        # ...and a fit made under a different llama.cpp build must be REPORTED as
        # outdated, not quietly replaced. Silently refitting on a build change is
        # exactly the drift this design removes.
        _CALIB_CACHE["g"]["backend"] = "some-old-build"
        stale_msg = _cal._outdated(_CALIB_CACHE["g"])
        cur_b = current_backend()
        # With no detectable backend there is nothing to compare, so no claim.
        stale_ok = bool(stale_msg) if cur_b else stale_msg == ""
        print("  CALFRZ build change reported not applied: %s  %s"
              % (("%r" % stale_msg[:38]) if stale_msg else "no backend to compare",
                 "OK" if stale_ok else "FAIL"))
        ok = ok and stale_ok
    finally:
        _cal._calib_store = _saved_store
        _CALIB_CACHE.clear()
        _CALIB_CACHE.update(_saved_cache)
        _cal._CALIB_LOADED = _saved_loaded
        try: os.remove(_tmp)
        except OSError: pass

    # 9d) A card must reproduce the file EXACTLY. It is not an approximation or a
    #     summary - it is the same three structures analyze() would have computed,
    #     stored. If a plan from a card differs from a plan from the file by even a
    #     MiB, the card is lying and every downstream number inherits it.
    #     The failure mode this guards is silent: JSON has no integer keys, so
    #     per_layer_bytes round-trips as {"0": n} and every per_layer[i] lookup
    #     misses, which reads as a model with no layers rather than as an error.
    import vram_planner.cards as _cards
    card_ok = True
    try:
        _mfile = load_gguf(p3)
        _mcfg = extract_config(_mfile)
        _card = _cards.make_card(p3, _mcfg, classify_tensors(_mfile, _mcfg),
                                 find_mmproj(p3))
        _cfg2, _cl2, _mm2, _meta2 = _cards._rehydrate(_json.loads(_json.dumps(_card)))
        # the int-keyed maps are the whole hazard
        _cl1 = classify_tensors(_mfile, _mcfg)
        key_ok = all(_cl2[k] == _cl1[k] for k in
                     ("per_layer_bytes", "per_layer_expert_bytes", "per_layer_ffn_bytes"))
        # ...and the plans themselves must agree to the MiB
        _saved_store2 = _cards._cards_store
        _tmp2 = os.path.join(_tf.gettempdir(), "vram_planner_cards_test.json")
        try:
            _cards._cards_store = lambda: _tmp2
            _cards.save_cards({"cards": {os.path.basename(p3): _card}})
            a = analyze(p3, 32768, "q8_0", 512, True, vram_budget_mib=9000,
                        ram_budget_mib=32000, gpu_reserve_mib=512,
                        compute_override_mib=0, safety_pct=5)
            b = analyze(os.path.basename(p3), 32768, "q8_0", 512, True,
                        vram_budget_mib=9000, ram_budget_mib=32000,
                        gpu_reserve_mib=512, compute_override_mib=0, safety_pct=5)
        finally:
            _cards._cards_store = _saved_store2
            try: os.remove(_tmp2)
            except OSError: pass
        # Every key, not a chosen few: a hand-picked list is exactly how a field
        # that only the file path populates slips through unnoticed.
        pdiff = [k for k in a["plan"] if a["plan"][k] != b["plan"][k]]
        cdiff = [k for k in a["config"] if a["config"][k] != b["config"][k]]
        sdiff = [k for k in (a.get("speed") or {})
                 if (a.get("speed") or {})[k] != (b.get("speed") or {}).get(k)]
        plan_ok = (not pdiff and not cdiff and not sdiff
                   and b["from_card"] and not a["from_card"])
        card_ok = key_ok and plan_ok
        print("  CARD  int keys survive JSON=%s | card==file across %d plan / %d config "
              "/ %d speed keys%s  %s"
              % (key_ok, len(a["plan"]), len(a["config"]), len(a.get("speed") or {}),
                 "" if plan_ok else "  DIFFER: %s" % (pdiff + cdiff + sdiff)[:6],
                 "OK" if card_ok else "FAIL"))
    except Exception as e:
        card_ok = False
        print("  CARD  raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and card_ok

    # 10) --n-cpu-moe must move expert bytes to the RAM side of the speed model
    rm2 = analyze(p3, 4096, "f16", 512, False, vram_budget_mib=300, ram_budget_mib=8000,
                  gpu_reserve_mib=0, compute_override_mib=5, safety_pct=0,
                  gpu_layers_override=nL, n_cpu_moe_override=nL // 2,
                  bw_vram_gbs=600, bw_ram_gbs=80, ctx_fill=1024)
    sm = rm2.get("speed") or {}
    moe_ok = sm.get("cpu_mib", 0) > 0 and rm2["plan"]["n_cpu_moe"] == nL // 2
    print("  MoE-SPD n_cpu_moe=%s cpu_bytes/token=%.1f MiB  %s"
          % (rm2["plan"]["n_cpu_moe"], sm.get("cpu_mib", 0), "OK" if moe_ok else "FAIL"))
    ok = ok and moe_ok

    # 11) build_argv's probe flag, and the launch scripts built on top of it
    try:
        from .launch import command_lines, launch_script
        from .sweep import build_argv, build_grid, model_facts

        # 11a) probe=True must be what it always was. bench and sweep both depend
        # on this, and a silent change to the probe command line would invalidate
        # every recorded row without invalidating any recorded row's LOOK.
        cg = build_grid(model_facts(p3))[0]
        pro = build_argv("EXE", "M.gguf", cg, 8231)
        usr = build_argv("EXE", "M.gguf", cg, 8231, probe=False)
        dropped = [x for x in pro if x not in usr]
        probe_ok = (pro[:pro.index("--port") + 2] == usr[:usr.index("--port") + 2]
                    and set(dropped) == {"--cache-ram", "0", "-v", "--no-warmup"}
                    and not [x for x in usr if x not in pro])
        print("  ARGV  probe=False drops only %s  %s"
              % (sorted(set(dropped)), "OK" if probe_ok else "FAIL"))

        # 11b) A generated script must never EXECUTE a flag that this build
        # ignores. Comments naming them are the useful part, so the check reads
        # what runs, not what the file contains - see command_lines().
        DEAD = ("--draft-max", "--no-mmap", "--mlock")
        base = {"ctx": 8192, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512, "ngl": 20}
        cases = [
            ("plain",  dict(base), {}),
            ("spec",   dict(base, spec="draft-mtp", spec_n_max=2), {}),
            ("projram", dict(base, mmproj_offload=False), {"mmproj": "mm.gguf"}),
            ("moe",    dict(base, ngl=99, ncmoe=12), {}),
        ]
        gen_ok, why = True, []
        for shell in ("powershell", "bash"):
            for name, c, extra in cases:
                run = "\n".join(command_lines(
                    launch_script("m.gguf", c, shell=shell, **extra)))
                for d in DEAD:
                    if d in run:
                        gen_ok = False; why.append("%s/%s runs %s" % (shell, name, d))
                # a flag appears exactly when the config implies it
                for flag, want in (("--spec-draft-n-max", bool(c.get("spec"))),
                                   ("--no-mmproj-offload", c.get("mmproj_offload") is False),
                                   ("--n-cpu-moe", bool(c.get("ncmoe"))),
                                   ("--load-mode", True)):
                    if (flag in run) != want:
                        gen_ok = False
                        why.append("%s/%s %s=%s" % (shell, name, flag, flag in run))
                # nothing invents a sampler the caller did not choose
                for s in ("--temp", "--top-k", "--top-p", "--min-p"):
                    if s in run:
                        gen_ok = False; why.append("%s/%s invented %s" % (shell, name, s))
        run = "\n".join(command_lines(launch_script(
            "m.gguf", base, shell="bash", sampling={"temp": 1.0, "min_p": ""})))
        if "--temp" not in run or "--min-p" in run:
            gen_ok = False; why.append("blank sampler handling")
        # The browser assets, checked as BYTES. A stray control character in a
        # string literal is invisible in an editor, parses fine, and passes
        # node --check - and then breaks at runtime in a way that points
        # nowhere near itself. A NUL landed in campaignId()'s separator, the id
        # went out as data-id, came back through el.dataset.id with the NUL
        # dropped, stopped matching the id campaignRow() computed, and every
        # campaign silently refused to open. Nothing in the file looked wrong.
        ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
        ctl_bad = []
        for fn in sorted(os.listdir(ui_dir)) if os.path.isdir(ui_dir) else []:
            if not fn.endswith((".js", ".css", ".html")):
                continue
            raw_b = open(os.path.join(ui_dir, fn), "rb").read()
            for i, b in enumerate(raw_b):
                if b < 9 or 13 < b < 32:
                    ctl_bad.append("%s byte %d = 0x%02x (line %d)"
                                   % (fn, i, b, raw_b[:i].count(b"\n") + 1))
                    break
        if ctl_bad:
            gen_ok = False; why.append("control chars: " + "; ".join(ctl_bad))

        print("  SCRIPT no dead flags, flags match config, blank samplers omitted%s  %s"
              % ("" if gen_ok else "  " + "; ".join(why[:4]), "OK" if gen_ok else "FAIL"))

        # 11c) The chat template. --jinja HAS to precede --chat-template-file:
        # without it a build accepts only its built-in template NAMES and rejects
        # a path, so getting the order wrong produces a script that looks right
        # and silently serves the wrong template.
        from .launch import ps_quote, sh_quote, template_args
        tmpl_ok, twhy = True, []
        # A space and an apostrophe: the two characters that break a shell literal.
        tf = "C:\\models\\it's a dir\\my chat.jinja"
        want = {"powershell": ps_quote(tf), "bash": sh_quote(tf.replace("\\", "/"))}
        for shell in ("powershell", "bash"):
            run = command_lines(launch_script(
                "m.gguf", base, shell=shell, chat_template_file=tf,
                chat_template_kwargs='{"enable_thinking": false}'))
            js = [i for i, l in enumerate(run) if "--jinja" in l]
            fs = [i for i, l in enumerate(run) if "--chat-template-file" in l]
            ks = [i for i, l in enumerate(run) if "--chat-template-kwargs" in l]
            if not (js and fs and ks and min(js) < min(fs)):
                tmpl_ok = False
                twhy.append("%s ordering jinja=%s file=%s" % (shell, js, fs))
            # The path and the JSON survive quoting - as a correctly ESCAPED
            # literal, which is not the same string that went in: the apostrophe
            # is doubled for PowerShell and spliced out and back for sh.
            text = "\n".join(run)
            if want[shell] not in text:
                tmpl_ok = False; twhy.append("%s mis-quoted the path" % shell)
            if '{"enable_thinking":false}' not in text:
                tmpl_ok = False; twhy.append("%s lost the kwargs JSON" % shell)
            # and blank means BLANK - no flag, not an empty one
            blank = "\n".join(command_lines(launch_script("m.gguf", base, shell=shell)))
            if "chat-template" in blank or "--jinja" in blank:
                tmpl_ok = False; twhy.append("%s emits template flags when unset" % shell)
        # kwargs that are not a JSON OBJECT are rejected at generation time, which
        # is the difference between a message and a server that will not start
        for bad in ("not json", "[1,2]", '"str"', "3"):
            try:
                template_args(None, bad)
                tmpl_ok = False; twhy.append("accepted %r" % bad)
            except ValueError:
                pass
        if template_args(None, {"b": 1, "a": 2})[1] != '{"a":2,"b":1}':
            tmpl_ok = False; twhy.append("dict kwargs not serialised stably")

        # Windows Explorer's "Copy as path" - the normal way anyone produces a
        # path to paste - always wraps it in double quotes, and they are not
        # part of the filename. Left on, Test-Path fails for a file that plainly
        # exists and the launcher dies on its own guard.
        want_p = "C:\\d\\chat.jinja"
        for given in ('"C:\\d\\chat.jinja"', "'C:\\d\\chat.jinja'",
                      '  "C:\\d\\chat.jinja"  '):
            if template_args(given, None)[0] != want_p:
                tmpl_ok = False; twhy.append("quotes kept in %r" % given)
        # ...but a quote that is part of the name survives, and an unbalanced
        # one is not a wrapper
        for keep in ("C:\\it's\\chat.jinja", '"C:\\d\\chat.jinja'):
            if template_args(keep, None)[0] != keep:
                tmpl_ok = False; twhy.append("mangled %r" % keep)

        # The header's divergence line. A row that names no sampler was measured
        # GREEDY - that is a fact about it, not a gap - and greedy is
        # speculation's best case, so a script running temp 1.0 must not quote
        # the acceptance rate as though it applied.
        from .launch import _config_divergence
        meas = {"config": {"ngl": 28, "spec": "draft-mtp", "spec_n_max": 2}}
        dv = _config_divergence(meas["config"], meas,
                                {"temp": 1.0, "top_k": 20, "top_p": 0.95,
                                 "repeat_penalty": 1.05})
        if "temp 0.0->1.0" not in dv or "top_k 0->20" not in dv:
            tmpl_ok = False; twhy.append("greedy baseline not reported: %r" % dv)
        # rep_pen/pres_pen are spelled differently on the two sides; without the
        # alias they could never diverge however far apart they were set
        if "rep_pen 1.0->1.05" not in dv:
            tmpl_ok = False; twhy.append("rep_pen alias not applied: %r" % dv)
        # and matching samplers still report nothing
        if _config_divergence(meas["config"], meas,
                              {"temp": 0.0, "top_k": 0, "top_p": 1.0}):
            tmpl_ok = False; twhy.append("greedy-vs-greedy reported a difference")

        # --reasoning-preserve. The trap: a Qwen3 template HAS a preserve_thinking
        # variable, so setting it in --chat-template-kwargs looks like it works -
        # but llama-server strips <think> out of the history BEFORE rendering, so
        # the variable has nothing left to act on. It has to be a real flag.
        from .launch import reasoning_args
        for shell, want in (("powershell", "$ReasoningPreserve = 'on'"),
                            ("bash", 'REASONINGPRESERVE="${REASONINGPRESERVE:-on}"')):
            t = "\n".join(command_lines(launch_script(
                "m.gguf", base, shell=shell, reasoning="on",
                reasoning_preserve="on")))
            # BOTH branches are in the script on purpose - it is a parameter, so
            # it has to be flippable at launch without regenerating the file.
            # What the caller chose is the DEFAULT, not the only branch present.
            if "--reasoning-preserve" not in t or "--no-reasoning-preserve" not in t:
                tmpl_ok = False; twhy.append("%s has only one preserve branch" % shell)
            if want not in t:
                tmpl_ok = False; twhy.append("%s default is not 'on'" % shell)
            if "--reasoning" not in t:
                tmpl_ok = False; twhy.append("%s lost --reasoning" % shell)
            # unset stays unset: 'default' is a real third value here, because
            # the flag pair's own default is "whatever the template says"
            blank = "\n".join(command_lines(launch_script("m.gguf", base, shell=shell)))
            if "reasoning" in blank.lower():
                tmpl_ok = False; twhy.append("%s emits reasoning when unset" % shell)
        if reasoning_args(None, None) != (None, None) \
                or reasoning_args("ON", True) != ("on", "on") \
                or reasoning_args(None, False) != (None, "off") \
                or reasoning_args(None, "default") != (None, None):
            tmpl_ok = False; twhy.append("reasoning_args normalisation")
        for bad in (("yes", None), (None, "maybe")):
            try:
                reasoning_args(*bad)
                tmpl_ok = False; twhy.append("accepted %r" % (bad,))
            except ValueError:
                pass
        print("  TMPL  --jinja precedes the template flags, bad kwargs refused%s  %s"
              % ("" if tmpl_ok else "  " + "; ".join(twhy[:4]), "OK" if tmpl_ok else "FAIL"))

        # Windows PowerShell 5.1 re-quotes native arguments and DROPS double
        # quotes already inside a value, so a valid '{"a":1}' reaches the exe as
        # {a:1} and llama-server rejects it with a JSON parse error pointing at
        # column 2 - which reads like the JSON was wrong. The kwargs are the only
        # value here that contains quotes, so they are the only one that needs the
        # escape, and it must be conditional: PowerShell 7.3+ passes the argument
        # through intact and the backslashes would arrive literally.
        psk = command_lines(launch_script(
            "m.gguf", base, shell="powershell",
            chat_template_kwargs='{"enable_thinking": false}'))
        add = [l for l in psk if "--chat-template-kwargs" in l and "+=" in l]
        quote_ok = (any("PSNativeCommandArgumentPassing" in l for l in psk)
                    and any("-replace" in l and "\\\"" in l for l in psk)
                    and bool(add) and "$ChatTemplateKwargs" not in add[0])
        # bash needs none of this: execve takes argv directly, nothing re-parses it
        bsh = command_lines(launch_script(
            "m.gguf", base, shell="bash",
            chat_template_kwargs='{"enable_thinking": false}'))
        quote_ok = quote_ok and not any("-replace" in l for l in bsh)
        print("  TMPL  PowerShell escapes the kwargs quotes, bash does not  %s"
              % ("OK" if quote_ok else "FAIL"))
        tmpl_ok = tmpl_ok and quote_ok

        # 11d) A script built from an old speed row names a model this machine may
        # no longer have. It must still emit every flag - those are the valuable
        # part - and must SAY the path did not resolve rather than looking fine.
        miss = launch_script("Deleted.gguf", base, shell="bash", path_resolved=False)
        lost_ok = ("NOT found on disk" in miss
                   and "--load-mode" in "\n".join(command_lines(miss)))
        print("  TMPL  unresolved model still generates, and says so  %s"
              % ("OK" if lost_ok else "FAIL"))

        # 11e) A launcher built from a measured row must SAY when the launched
        # config differs from the row's config. The row's tok/s and fit evidence
        # belong to the row as measured - a script that adds MTP on top of a row
        # measured without it, carrying the row's ngl into a config that OOMs on
        # it, would otherwise look exactly like the row it is not.
        from .launch import _config_divergence
        rowcfg = dict(base, spec="none")
        meas = {"tok_s": 9.45, "config": rowcfg}
        same = launch_script("m.gguf", dict(base, spec="none"), shell="bash",
                             measured=meas)
        div = launch_script("m.gguf", dict(base, spec="draft-mtp", spec_n_max=2),
                            shell="bash", measured=meas)
        div_ok = ("DIFFERS from the measured row" in div
                  and "spec none->draft-mtp" in div
                  and "DIFFERS" not in same
                  # a config that omits a knob groups with one that never set it
                  and _config_divergence(dict(base), meas) == "")
        print("  SCRIPT launcher says when the launched config differs from the row  %s"
              % ("OK" if div_ok else "FAIL"))
        script_ok = probe_ok and gen_ok and tmpl_ok and lost_ok and div_ok
    except Exception as e:
        script_ok = False
        print("  SCRIPT raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and script_ok

    # 12) the job runner: one at a time, and a cancel that is actually observed
    try:
        import threading, time as _time
        import time as _t
        from .job import Job
        j = Job()
        gate, seen = threading.Event(), {}

        def work(job):
            job.set_total(2)
            job._append("started")
            gate.wait(5)
            seen["cancelled"] = job.cancelled()
            job._add_row({"status": "ok", "tok_s": 1.0})
            return {"rows": 1}

        started, _ = j.start("test", work)
        second, _ = j.start("test", work)      # must be refused, not queued
        j.cancel()
        gate.set()
        for _ in range(50):
            if not j.running: break
            _time.sleep(0.05)
        snap = j.snapshot()
        job_ok = (started and not second and seen.get("cancelled") is True
                  and snap["status"] == "cancelled" and snap["done"] == 1
                  and snap["total"] == 2)
        print("  JOB   second start refused=%s | cancel seen=%s | status=%s  %s"
              % (not second, seen.get("cancelled"), snap["status"],
                 "OK" if job_ok else "FAIL"))
    except Exception as e:
        job_ok = False
        print("  JOB   raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and job_ok

    # 13) chaining, and the findings drawn out of recorded rows. Both are places
    # where being wrong is worse than being absent: a bad chained baseline bends
    # every stage after it in one direction, and an insight that compares across
    # experiments manufactures an effect out of the difference between the runs.
    try:
        from .bench import (axis_effects, best_config, comparable, pareto,
                            depth_curve)
        from .sweep import _key as _skey

        B = {"ctx": 32768, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512,
             "ngl": 28, "fill": 2048}

        def row(tok, cfg=None, **kw):
            c = dict(B); c.update(cfg or {})
            r = {"model": "M.gguf", "gpu": "G", "_file": "f.jsonl", "status": "ok",
                 "tok_s": tok, "n_predict": 128, "repeat": 3, "config": c,
                 "proc_vram_mib": kw.pop("vram", 8000)}
            r.update(kw); return r

        INC = 8.0
        refuse = [
            ("spilled",      [row(20.0, {"ngl": 31}, spilled=True)]),
            ("looping",      [row(20.0, {"ngl": 31}, distinct_ratio=0.3)]),
            ("other fill",   [row(20.0, {"ngl": 31, "fill": 64000})]),
            ("other kv",     [row(20.0, {"ngl": 31, "kv": "f16"})]),
            ("other npred",  [dict(row(20.0, {"ngl": 31}), n_predict=32)]),
            # ubatch became a frozen setting when stage C was retired, so it
            # gates like kv and fill do. The store is full of ub 256/1024/2048
            # rows that stage C measured back when it WAS an axis; ungated they
            # would set the baseline for a campaign frozen at 512.
            ("other ub",     [row(20.0, {"ngl": 31, "ub": 2048})]),
            ("inside 2%",    [row(INC * 1.01, {"ngl": 31})]),
        ]
        chain_ok, cwhy = True, []
        for label, rows_ in refuse:
            c, _w = best_config(rows_, "M.gguf", B, 128, 3, incumbent_tok_s=INC)
            if c is not None:
                chain_ok = False; cwhy.append("took a %s row" % label)
        c, w = best_config([row(INC * 1.05, {"ngl": 31})], "M.gguf", B, 128, 3,
                           incumbent_tok_s=INC)
        if not c or c["ngl"] != 31:
            chain_ok = False; cwhy.append("refused a legitimate +5% challenger")
        # only the placement and speculation knobs carry; the campaign's own
        # definition never does - and `ub` moved onto that side of the line
        # with stage C, so a promoted baseline keeps the frozen value rather
        # than the winning row's.
        c, _ = best_config([row(9.9, {"ngl": 31, "spec": "draft-mtp",
                                      "spec_n_max": 2, "stage": "B"})],
                           "M.gguf", B, 128, 3, incumbent_tok_s=INC)
        if not c or c["ngl"] != 31 or c["spec"] != "draft-mtp" or "stage" in c \
                or c["ub"] != B["ub"] or c["ctx"] != B["ctx"] \
                or c["fill"] != B["fill"]:
            chain_ok = False; cwhy.append("carried the wrong keys")
        from vram_planner.bench import CARRY_KEYS as _CK
        if "ub" in _CK:
            chain_ok = False; cwhy.append("ub is still a carried key")
        # Samplers are a condition of the measurement, not a config knob: greedy
        # is speculation's best case, so a greedy row and a sampled one are two
        # experiments. They must not meet in a baseline OR in an effect size.
        samp_ok = (not comparable(row(9.0), "M.gguf", dict(B, temp=0.7), 128, 3)
                   and comparable(row(9.0, {"temp": 0.7}), "M.gguf",
                                  dict(B, temp=0.7), 128, 3)
                   # and a config that omits a sampler groups with one that sets
                   # it to its neutral value - they are the same measurement
                   and comparable(row(9.0, {"rep_pen": 1.0, "pres_pen": 0.0}),
                                  "M.gguf", B, 128, 3))
        mixed_s = [row(5.0, {"ub": 512}), row(6.0, {"ub": 1024}),
                   row(99.0, {"ub": 2048, "temp": 0.7})]
        ub_s = [e for e in axis_effects(mixed_s)["effects"] if e["axis"] == "ub"][0]
        samp_ok = samp_ok and sorted(v["value"] for v in ub_s["values"]) == [512, 1024]
        print("  CHAIN samplers are a condition, not a knob: greedy never meets "
              "sampled  %s" % ("OK" if samp_ok else "FAIL"))

        # rep_pen/pres_pen are real --speed-axes names, so a ladder over either
        # has to produce distinct resume keys - and a config that omits them must
        # still key identically to every row already on disk.
        kb = {"ctx": 32768, "ngl": 28, "ub": 512, "seq": 1, "fa": True, "kv": "q8_0"}
        key_ok = (_skey("M", kb) == _skey("M", dict(kb, rep_pen=1.0, pres_pen=0.0))
                  and len({_skey("M", dict(kb, rep_pen=v))
                           for v in (1.0, 1.05, 1.1)}) == 3
                  and len({_skey("M", dict(kb, pres_pen=v))
                           for v in (0.0, 0.5, 1.0)}) == 3)
        print("  CHAIN sampler ladders resume distinctly, old rows key unchanged  %s"
              % ("OK" if key_ok else "FAIL"))

        # --speed-axes and --speed-chain are mutually exclusive: chaining rebuilds
        # each STAGE from the previous winner and an explicit ladder has no
        # stages. The ladder is the more specific instruction and must win, out
        # loud - silently running the staged grid instead would burn the same
        # hours measuring something nobody asked for.
        from .bench import resolve_search
        ax = {"kv": ["f16", "q8_0"]}
        excl_ok = (resolve_search(ax, True)[0] is False       # ladder wins
                   and resolve_search(ax, True)[1]            # and says so
                   and resolve_search(None, True) == (True, None)   # chain alone
                   and resolve_search(ax, False) == (False, None))  # axes alone
        print("  CHAIN --speed-axes overrides --speed-chain, and says so  %s"
              % ("OK" if excl_ok else "FAIL"))
        chain_ok = chain_ok and excl_ok
        chain_ok = chain_ok and samp_ok and key_ok
        print("  CHAIN untrustworthy/incomparable rows refused, 2%% margin held%s  %s"
              % ("" if chain_ok else "  " + "; ".join(cwhy[:3]), "OK" if chain_ok else "FAIL"))

        # An effect may only ever be computed inside one experiment.
        mixed = [row(5.0, {"ub": 512}), row(6.0, {"ub": 1024}),
                 dict(row(99.0, {"ub": 2048}), gpu="OTHER GPU"),
                 dict(row(98.0, {"ub": 2048}), _file="other.jsonl"),
                 dict(row(97.0, {"ub": 2048}), n_predict=32),
                 row(96.0, {"ub": 2048, "fill": 64000})]
        eff = {e["axis"]: e for e in axis_effects(mixed)["effects"]}
        ub = eff.get("ub") or {}
        vals = sorted(v["value"] for v in ub.get("values", []))
        split_ok = (vals == [512, 1024] and not ub.get("single")
                    and abs((ub.get("gain_pct") or 0) - 20.0) < 0.1)
        # a row that stands alone is REPORTED as that, not quietly dropped
        alone = {e["axis"]: e for e in axis_effects([row(5.0)])["effects"]}
        alone_ok = alone.get("ub", {}).get("single") is True
        # and untrustworthy rows never reach a conclusion, though rank_rows lists them
        dirty = [row(5.0, {"ub": 512}), row(6.0, {"ub": 1024}),
                 row(50.0, {"ub": 2048}, spilled=True)]
        dax = {e["axis"]: e for e in axis_effects(dirty)["effects"]}
        clean_ok = (2048 not in [v["value"] for v in dax["ub"]["values"]]
                    and axis_effects(dirty)["n_excluded"] == 1)
        print("  FIND  comparisons never cross experiments (%s), lone value kept, "
              "spilled excluded  %s"
              % (vals, "OK" if (split_ok and alone_ok and clean_ok) else "FAIL"))

        # Pareto: exactly the non-dominated set.
        pts = [row(10.0, {"ngl": 1}, vram=5000),   # fastest, biggest    -> in
               row(9.0, {"ngl": 2}, vram=4000),    # slower, smaller     -> in
               row(8.0, {"ngl": 3}, vram=4500),    # dominated by ngl 2  -> out
               row(7.0, {"ngl": 4}, vram=3000)]    # slowest, smallest   -> in
        got = sorted((r["config"]["ngl"]) for r in pareto(pts))
        par_ok = got == [1, 2, 4]
        # depth: only configs actually measured at more than one fill
        dep = depth_curve([row(10.0, {"fill": 2048}), row(5.0, {"fill": 32768}),
                           row(9.0, {"fill": 2048, "ngl": 99})])
        dep_ok = (len(dep) == 1 and len(dep[0]["points"]) == 2
                  and abs(dep[0]["drop_pct"] - 50.0) < 0.1)
        print("  FIND  pareto frontier %s, depth curve needs two depths  %s"
              % (got, "OK" if (par_ok and dep_ok) else "FAIL"))

        # A looping row must be VISIBLE while the campaign runs, not only dropped
        # from conclusions an hour later. The two must also use one threshold: a
        # row warned about and then kept - or dropped having never been flagged -
        # is worse than either rule alone.
        from .bench import _fmt_row, trustworthy, LOOP_RATIO
        loop = row(2.85, {"spec": "draft-mtp", "spec_n_max": 2},
                   distinct_ratio=0.184, accept_rate=1.0)
        quiet = row(2.85, {"spec": "draft-mtp", "spec_n_max": 2},
                    distinct_ratio=1.0, accept_rate=1.0)
        edge = row(2.85, distinct_ratio=LOOP_RATIO)          # exactly at the line
        lt = _fmt_row(loop["config"], loop)
        # the acceptance rate is the trap: 100% on looping text reads as the
        # drafter excelling, so the line has to say which one it is
        warn_ok = ("LOOPING" in lt and "82%" in lt and "not the drafter" in lt
                   and "LOOPING" not in _fmt_row(quiet["config"], quiet)
                   and "LOOPING" not in _fmt_row(edge["config"], edge))
        # non-speculative looping still warns, but without the acceptance clause
        plain = row(2.10, distinct_ratio=0.184)
        pt = _fmt_row(plain["config"], plain)
        warn_ok = warn_ok and "LOOPING" in pt and "drafter" not in pt
        # one threshold, both directions
        gate_ok = (not trustworthy(loop) and trustworthy(quiet)
                   and trustworthy(edge))
        print("  FIND  looping row warned at run time and gated by one threshold  %s"
              % ("OK" if (warn_ok and gate_ok) else "FAIL"))

        # Copying the prompt back is the second degenerate mode, and it is the
        # one distinct_ratio cannot see: a verbatim copy's 8-word windows are
        # all distinct, so it reads a clean 1.00 - healthy - while inflating
        # speculative acceptance exactly as much as looping does. The copy gate
        # must mark the row at run time and refuse it, at one shared threshold.
        from .bench import (COPY_RATIO, verify_config,
                            _copyback_ratio, _distinct_ratio)
        copied = row(2.9, {"spec": "draft-mtp", "spec_n_max": 2},
                     distinct_ratio=1.0, copyback_ratio=0.96, accept_rate=1.0)
        ct = _fmt_row(copied["config"], copied)
        copy_ok = ("COPYING" in ct and "96%" in ct and "not the drafter" in ct
                   and not trustworthy(copied) and "LOOPING" not in ct)
        # a copy is one mode and a loop is the other; the note names what happened
        plainc = row(2.9, distinct_ratio=1.0, copyback_ratio=0.9)
        pt2 = _fmt_row(plainc["config"], plainc)
        copy_ok = copy_ok and ("COPYING" in pt2 and "drafter" not in pt2
                               and not trustworthy(plainc))
        # a row that loops on the prompt's own text is BOTH modes - still marked,
        # still refused, named as the copy it is
        both = row(2.9, distinct_ratio=0.1, copyback_ratio=0.9)
        bt = _fmt_row(both["config"], both)
        copy_ok = copy_ok and ("COPYING" in bt and "excluded" in bt
                               and not trustworthy(both))
        # exactly at the line passes, like the looping edge above
        edgec = row(2.9, distinct_ratio=1.0, copyback_ratio=COPY_RATIO)
        copy_ok = copy_ok and trustworthy(edgec)
        # the metric itself: verbatim windows hit, original prose misses, and a
        # copy is exactly the case distinct_ratio calls healthy
        prompt = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
                  "lambda mu nu xi omicron pi rho sigma tau upsilon chi psi omega")
        verbatim = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
                    "lambda mu nu")
        fresh = "completely different words that have no relation to the corpus"
        metric_ok = (_copyback_ratio(verbatim, prompt) == 1.0
                     and _copyback_ratio(fresh, prompt) == 0.0
                     and _distinct_ratio(verbatim) == 1.0
                     and _copyback_ratio("a b c", prompt) is None)
        print("  FIND  copying detected, marked, and gated (distinct_ratio misses it)  %s"
              % ("OK" if (copy_ok and metric_ok) else "FAIL"))

        # /completion is raw continuation. An instruction-tuned model handed a
        # wall of source with no role markers has nothing telling it a REQUEST
        # was made, so it continues the document - which is what produced 23
        # unusable rows. The prompt now goes through the model's own chat
        # template, and a build too old to offer one must say so per row rather
        # than quietly reproducing the old behaviour.
        import hashlib
        from .bench import (apply_template, PROMPT_SCHEME, prompt_identity,
                            corpus_text)
        calls = []

        def fake_post(url, path, payload, timeout=600):
            calls.append(path)
            if path == "/apply-template":
                return {"prompt": "<|im_start|>user\n"
                                  + payload["messages"][0]["content"]
                                  + "<|im_end|>\n<|im_start|>assistant\n"}
            raise RuntimeError("no such endpoint")

        import vram_planner.bench as _b
        real_post = _b._post
        try:
            _b._post = fake_post
            got, applied = apply_template("u", "hello")
            tmpl_ok = (applied is True and "<|im_start|>user" in got
                       and "hello" in got)
            # a build without the endpoint falls back rather than failing the
            # campaign - but reports it, so the row can be marked RAW
            _b._post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404"))
            got2, applied2 = apply_template("u", "hello")
            tmpl_ok = tmpl_ok and got2 == "hello" and applied2 is False
        finally:
            _b._post = real_post
        # an untemplated row is named even when its output looks healthy: the
        # defect is in how the row was produced, not in what came out
        rawrow = row(3.9, distinct_ratio=1.0, copyback_ratio=0.0, templated=False)
        rt = _fmt_row(rawrow["config"], rawrow)
        okrow = row(3.9, distinct_ratio=1.0, copyback_ratio=0.0, templated=True)
        tmpl_ok = (tmpl_ok and "RAW" in rt
                   and "RAW" not in _fmt_row(okrow["config"], okrow))
        # changing HOW a prompt is assembled must re-key it, or rows measured on
        # raw continuation would merge with templated ones and average two
        # different experiments together
        h = hashlib.sha256()
        h.update(corpus_text().encode("utf-8"))
        h.update(_b.INSTRUCTION.encode("utf-8"))
        scheme_ok = (h.hexdigest() != prompt_identity()
                     and PROMPT_SCHEME in ("chat",))
        print("  FIND  prompt goes through the chat template, raw builds say so  %s"
              % ("OK" if (tmpl_ok and scheme_ok) else "FAIL"))

        # A campaign starts a server every few minutes on the same port. The
        # previous socket is still in TIME_WAIT, llama-server does not set
        # SO_REUSEADDR, and the row dies as EXIT - which reads as a crash, so
        # the config looks like evidence about the wall when it is a harness
        # fault. Two rows of the 32k campaign were lost exactly this way.
        import socket as _sock
        from .bench import free_port, BENCH_PORT
        held = _sock.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        try:
            alt = free_port(taken)
            port_ok = (alt != taken and 1024 < alt < 65536
                       # and an unused port is handed back unchanged, so the
                       # ordinary case still lands on the documented default
                       and free_port(BENCH_PORT) in (BENCH_PORT,))
            # probing must not itself leave the port unusable for the server
            port_ok = port_ok and free_port(alt) == alt
        finally:
            held.close()
        print("  BENCH busy port falls back instead of dying as EXIT  %s"
              % ("OK" if port_ok else "FAIL"))

        # Stage D could not succeed. A and B pick the fastest split that FITS,
        # so it is the one with the least headroom; D then asks for a draft KV
        # cache llama.cpp keeps at f16 whatever -ctk says. It OOMed on both
        # models tried and speculation was the WINNER on both once given room:
        # ngl 31 -> 28 (3.95 vs 3.25), ncmoe 29 -> 34 (54.87 vs 47.17).
        from .bench import _spec_retry
        moe, dense = {"is_moe": True, "n_layers": 41}, {"is_moe": False, "n_layers": 65}
        oom, okrow = {"status": "oom"}, {"status": "ok"}

        def walk(c0, facts, key):
            t, c, seen = {}, dict(c0), [c0[key]]
            while True:
                n = _spec_retry(c, oom, facts, t)
                if n is None:
                    return seen
                c = n; seen.append(c[key])
        # The directions are OPPOSITE and getting it backwards walks into the
        # wall: more n_cpu_moe frees VRAM, more ngl consumes it.
        retry_ok = (
            34 in walk({"spec": "draft-mtp", "spec_n_max": 2, "ncmoe": 29}, moe, "ncmoe")
            and 28 in walk({"spec": "draft-mtp", "spec_n_max": 2, "ngl": 31}, dense, "ngl")
            # a row that LOADED is not retried
            and _spec_retry({"spec": "draft-mtp", "ncmoe": 29}, okrow, moe, {}) is None
            # n-gram speculators allocate no second cache, so their OOM is about
            # the model and walking would only prove it more slowly
            and _spec_retry({"spec": "ngram-mod", "ncmoe": 29}, oom, moe, {}) is None
            and _spec_retry({"spec": "none", "ncmoe": 29}, oom, moe, {}) is None
            # no rung budget: it walks to the END of the axis, because a walk
            # that stops short reports "speculation does not fit" while meaning
            # "I stopped looking". The axis itself is the bound.
            and walk({"spec": "draft-mtp", "ncmoe": 0}, moe, "ncmoe")
                == list(range(0, 42))
            # and it cannot walk off either end
            and _spec_retry({"spec": "draft-mtp", "ncmoe": 41}, oom, moe, {}) is None
            and _spec_retry({"spec": "draft-mtp", "ngl": 1}, oom, dense, {}) is None)
        print("  SPEC  a draft config that OOMs is retried with room, not written off  %s"
              % ("OK" if retry_ok else "FAIL"))
        port_ok = port_ok and retry_ok

        # A knob that silently does not apply is worse than one that is refused.
        # rounds re-runs the stages from the WINNER, so without chaining there is
        # no winner to re-run them from - it was accepted in that state and did
        # precisely nothing, with nothing said.
        from .bench import resolve_rounds
        rnd_ok = (resolve_rounds(True, 3) == (3, None)
                  and resolve_rounds(True, 1) == (1, None)
                  and resolve_rounds(False, 1) == (1, None)
                  and resolve_rounds(False, 3)[0] == 1
                  and "ignored" in (resolve_rounds(False, 3)[1] or "")
                  # nonsense normalises rather than raising mid-campaign
                  and resolve_rounds(True, 0) == (1, None)
                  and resolve_rounds(True, None) == (1, None))
        # ...and the two controls that cannot act are hidden by CSS, not by JS:
        # the grid pane is rewritten every 1.5s while a campaign runs, so a
        # script toggle is one missed redraw away from being wrong.
        css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ui", "app.css"), encoding="utf-8").read()
        rnd_ok = (rnd_ok and "#swchain:not(:checked)) > #swroundsfield" in css
                  and "#swverify:not(:checked)) > #swverifyfield" in css)
        print("  ROUND rounds refused without chaining, dead controls hidden  %s"
              % ("OK" if rnd_ok else "FAIL"))
        port_ok = port_ok and rnd_ok

        # prefill is taken from the FIRST request after the server comes up, so
        # it also paid for faulting CPU-resident weights in. bench_one already
        # throws that pass's DECODE away for the same reason; it kept its
        # prefill. A tiny throwaway generation now goes first, and rows say
        # which side of that they were measured on - old ones must not be
        # averaged with new ones as though the number meant the same thing.
        import inspect as _i
        src = _i.getsource(_b.bench_one)
        warm_ok = ('row["prefill_warm"] = True' in src
                   # the throwaway has to come BEFORE the timed cold pass, or it
                   # warms nothing that matters
                   and src.index("seed=999") < src.index("seed=1000")
                   and "prefill_warm" in _i.getsource(_b._slim))
        # the badge fires on rows lacking the flag, and only when there is a
        # prefill figure to distrust
        js = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ui", "app.js"), encoding="utf-8").read()
        warm_ok = warm_ok and "r.prefill_warm !== true && r.prefill_tok_s" in js
        print("  WARM  prefill measured after the fault-in, and old rows say so  %s"
              % ("OK" if warm_ok else "FAIL"))
        port_ok = port_ok and warm_ok

        # Stop, twice. The soft stop lands between configs and keeps the store
        # free of half-measurements, but nearly all of a config's wall time is
        # one blocking request to the server, so at a deep fill it is minutes
        # away. A second press kills the server, which ends that request at
        # once, and the abandoned row is DISCARDED - a genfail written here
        # would be indistinguishable on disk from a real one and the next
        # campaign would carry it forward as a wall that does not exist.
        import time as _t
        from .job import Job
        class _P(object):
            def __init__(self): self.killed = False
            def poll(self): return None
            def kill(self): self.killed = True
        jb = Job()
        stop_ok = jb.cancel() == (False, "nothing running")   # idle refuses
        jb.status, jb.started = "running", _t.time()
        proc = _P()
        jb.set_live_proc(proc)
        ok1, m1 = jb.cancel()
        stop_ok = (stop_ok and ok1 and m1 == "stopping"
                   and jb.cancelled() and not jb.aborting()
                   # the first press must NOT kill: the row in flight is still
                   # going to be finished and recorded
                   and not proc.killed
                   and jb.snapshot()["cancelling"] and not jb.snapshot()["aborting"])
        ok2, m2 = jb.cancel()
        stop_ok = (stop_ok and ok2 and m2 == "aborting"
                   and jb.aborting() and proc.killed
                   and jb.snapshot()["aborting"])
        # nothing to kill between configs is not an error
        jb2 = Job(); jb2.status, jb2.started = "running", _t.time()
        jb2.cancel()
        stop_ok = stop_ok and jb2.cancel()[0] and jb2.aborting()
        print("  STOP  first press finishes the config, second abandons it  %s"
              % ("OK" if stop_ok else "FAIL"))
        port_ok = port_ok and stop_ok

        # WDDM does not fail an allocation past the dedicated budget - it moves
        # part of the process to system RAM and keeps going, so the row says ok
        # while every token that touches the moved bytes crosses PCIe. The old
        # detector could not see it: its floor test only catches a NONSENSICAL
        # floor, and a partial demotion leaves a plausible one.
        from .bench import (demoted, spill_note, _infer_demotion,
                            SHARED_SPILL_MIB, FLOOR_DROP_MIB)
        from .sweep import suspect_reason
        # The verdict comes from the EXCESS over the ladder, never from the raw
        # counter. Shared Usage also counts host memory the process holds on
        # PURPOSE - pinned staging buffers, --no-mmproj-offload - so a threshold
        # on the absolute value fires on every healthy row. It did: 474.0 MiB
        # flat across ngl 26/27/28 with the projector in RAM, unmoved by ngl,
        # and the row carrying it was the fastest ever measured on that model.
        meas_ok = (demoted({"shared_excess": SHARED_SPILL_MIB + 1})
                   and not demoted({"shared_excess": SHARED_SPILL_MIB})
                   # the raw reading, however large, is not the question
                   and not demoted({"shared_mib": 9999.0})
                   # unmeasured is not the same as clean
                   and not demoted({"shared_excess": None})
                   and not demoted({}))
        note = spill_note({"shared_excess": 246.0})
        meas_ok = meas_ok and "246" in note and "PCIe" in note
        # mid-campaign there is no ladder, so the counter is reported as a fact
        # and given no verdict rather than being called a spill
        raw = spill_note({"shared_mib": 474.0})
        meas_ok = meas_ok and "474" in raw and "SPILLED" not in raw

        # The real 65k ladder: three rungs at a flat 474.0 with the projector in
        # RAM, all of it deliberate. Nothing here is a spill, and the fastest row
        # is one of them - the shape that broke the first version.
        def srow(ngl, shared, tok, **kw):
            r = {"model": "M.gguf", "gpu": "G", "_file": "f.jsonl", "status": "ok",
                 "tok_s": tok, "floor_mib": 1050.0 + ngl, "shared_mib": shared,
                 "spilled": True,      # what the first detector wrote to disk
                 "config": {"ctx": 131072, "kv": "q8_0", "ub": 512, "ngl": ngl,
                            "fill": 65536, "mmproj_offload": False,
                            "spec": "draft-mtp", "spec_n_max": 2}}
            r.update(kw)
            return r
        flat = [srow(26, 474.0, 3.61), srow(27, 474.0, 3.86), srow(28, 474.0, 3.95)]
        _infer_demotion(flat)
        meas_ok = (meas_ok
                   and all(r["shared_excess"] == 0.0 for r in flat)
                   # ...and the stored verdict is CORRECTED, not carried forward,
                   # or every row of that campaign stays outside trustworthy()
                   # and best_config() has nothing left to pick
                   and not any(r["spilled"] for r in flat)
                   and all(trustworthy(r) for r in flat))
        # one rung carrying 240 MiB the others do not IS the driver moving
        # something, and that is the only thing the counter can say
        step = flat + [srow(29, 714.0, 2.10)]
        _infer_demotion(step)
        meas_ok = (meas_ok and step[-1]["shared_excess"] == 240.0
                   and step[-1]["spilled"] and not trustworthy(step[-1]))
        # A LONE measured row has no ladder to be excess over, so nothing in the
        # grouping can reach it - and it would otherwise keep the stored verdict
        # of a detector that no longer exists, permanently untrustworthy. This is
        # the real ngl 31 reference row, the only measured row in its group.
        lone = [srow(31, 238.0, 3.25, spec="none")]
        lone[0]["config"] = dict(lone[0]["config"], spec="none", spec_n_max=0)
        _infer_demotion(lone)
        meas_ok = (meas_ok and lone[0].get("shared_excess") is None
                   and not lone[0]["spilled"] and trustworthy(lone[0]))
        # ...unless the row's own NUMBERS are unsound, which never needed the
        # counter. Judged by unsound_reason() on the reading itself, not by the
        # stored `suspect` string: that string is the ALLOCATION FIT's verdict
        # and rejects every -ot row on principle, which is every row of every
        # dense campaign - a gate that fires on all of them promotes none.
        sus = [srow(31, 238.0, 3.25, floor_mib=-500.0)]
        _infer_demotion(sus)
        meas_ok = meas_ok and sus[0]["spilled"]
        # ...and an -ot pin alone is NOT that: it excludes a row from the fit
        # and says nothing about whether its tok/s can be believed.
        otr = [srow(31, 238.0, 3.25)]
        otr[0]["config"] = dict(otr[0]["config"], n_cpu_ffn=65)
        otr[0]["suspect"] = suspect_reason(otr[0])
        _infer_demotion(otr)
        meas_ok = (meas_ok and otr[0]["suspect"] and not otr[0]["spilled"]
                   and trustworthy(otr[0]))

        # The inference, for rows recorded before the counter existed. These are
        # the real numbers from the ngl ladder: five rungs agreeing within 12
        # MiB, then one that fell 246 MiB when the process could not grow.
        def frow(ngl, floor, **kw):
            r = {"model": "M.gguf", "gpu": "G", "_file": "f.jsonl", "status": "ok",
                 "tok_s": 3.0, "floor_mib": floor,
                 "config": {"ctx": 131072, "kv": "q8_0", "ub": 512, "ngl": ngl,
                            "fill": 65536}}
            r.update(kw)
            return r

        ladder = [frow(24, 1348.0), frow(25, 1351.9), frow(26, 1355.8),
                  frow(27, 1356.3), frow(28, 1360.2), frow(29, 1114.1)]
        _infer_demotion(ladder)
        inf_ok = (ladder[-1].get("spill_inferred") > FLOOR_DROP_MIB
                  and not any(r.get("spill_inferred") for r in ladder[:-1])
                  # marked but NOT gated: an inference is weaker than a reading,
                  # and a ladder slowing at its top rung is the wall being found
                  # rather than a row to hide
                  and not ladder[-1].get("spilled")
                  and trustworthy(ladder[-1]))
        # a measured row is judged on its reading, never on the inference - a
        # measurement beats a deduction about the same fact
        measured = ladder[:-1] + [frow(29, 1114.1, shared_mib=300.0)]
        _infer_demotion(measured)
        inf_ok = inf_ok and not measured[-1].get("spill_inferred")
        # Speculation moves the floor by ~800 MiB legitimately: llama.cpp does
        # not report the draft KV cache in alloc_gpu, so it lands in floor. The
        # first version of this grouped across it, the median landed between the
        # two populations, and every ORDINARY row read as a 380 MiB collapse.
        # These are those real floors.
        spec_mix = [frow(n, f) for n, f in
                    ((28, 224.2), (30, 230.0), (31, 232.5), (32, 236.4))]
        for n, f, nmax in ((28, 1000.0, 1), (26, 1049.6, 2), (28, 1058.0, 2),
                           (28, 1118.0, 3)):
            r = frow(n, f)
            r["config"] = dict(r["config"], spec="draft-mtp", spec_n_max=nmax)
            spec_mix.append(r)
        _infer_demotion(spec_mix)
        inf_ok = inf_ok and not any(r.get("spill_inferred") for r in spec_mix)
        # the projector moves the floor the same way, at ~1100 MiB
        mixed = ladder[:-1] + [frow(28, 230.0)]
        mixed[-1]["config"] = dict(mixed[-1]["config"], mmproj_offload=False)
        _infer_demotion(mixed)
        inf_ok = inf_ok and not mixed[-1].get("spill_inferred")
        # and the inference names itself as one rather than claiming a reading
        inote = spill_note({"spill_inferred": 246.1})
        inf_ok = inf_ok and "floor fell" in inote and "246" in inote
        print("  SPILL demotion measured from the counter, inferred for old rows  %s"
              % ("OK" if (meas_ok and inf_ok) else "FAIL"))

        # The frozen corpus is an experiment condition. Two rows measured
        # against different corpora - or one recorded before the corpus was
        # frozen at all - are different experiments and must never meet in a
        # baseline or an effect size.
        pid_a, pid_b = "a" * 40, "b" * 40
        id_ok = (not comparable(row(9.0, prompt_id=pid_a), "M.gguf", B, 128, 3,
                                prompt_id=pid_b)
                 and comparable(row(9.0, prompt_id=pid_a), "M.gguf", B, 128, 3,
                                prompt_id=pid_a)
                 # a row recorded before the freeze has no id: it is its own
                 # experiment, not the campaign's
                 and not comparable(row(9.0), "M.gguf", B, 128, 3, prompt_id=pid_a)
                 # ...and ids also split effect sizes, never merging the corpora
                 and sorted(v["value"] for v in [e for e in axis_effects(
                     [row(5.0, {"ub": 512}, prompt_id=pid_a),
                      row(6.0, {"ub": 1024}, prompt_id=pid_a),
                      row(99.0, {"ub": 2048})])["effects"]
                     if e["axis"] == "ub"][0]["values"]) == [512, 1024])
        # --speed-verify composes the winner's knobs with the production
        # config, so the row actually loaded is the config actually launched.
        vc = verify_config({"ngl": 28, "ctx": 131072, "stage": "A", "fill": 100000},
                           {"spec": ["draft-mtp"], "spec_n_max": [2], "temp": [1.0]})
        vc_ok = (vc == {"ngl": 28, "ctx": 131072, "fill": 100000,
                        "spec": "draft-mtp", "spec_n_max": 2, "temp": 1.0}
                 and "stage" not in vc)
        print("  FIND  prompt_id splits experiments, verify composes winner+production  %s"
              % ("OK" if (id_ok and vc_ok) else "FAIL"))

        # The chat template is the other half of what a tok/s number means: a
        # thinking template spends tokens reasoning before it answers. So it
        # splits experiments exactly like prompt_id, in BOTH directions - and
        # its identity is the file's CONTENT, not its path, or editing a
        # template in place would leave the old rows looking current.
        import tempfile as _tf
        from vram_planner.bench import template_identity, _ANY
        from vram_planner.sweep import build_argv
        td = _tf.mkdtemp()
        tp = os.path.join(td, "chat template.jinja")
        with open(tp, "w", encoding="utf-8") as fh:
            fh.write("{{ messages[0].content }}")
        kw = '{"enable_thinking":true}'
        t1 = template_identity(tp, kw)
        with open(tp, "a", encoding="utf-8") as fh:
            fh.write("\n{# edited #}")
        t2 = template_identity(tp, kw)
        ti_ok = (t1 and t2 and t1 != t2                     # content, not path
                 and template_identity(tp, None) != t2      # kwargs count too
                 and template_identity(None, None) is None) # nothing pinned
        # comparable() filters both ways, and _ANY means "do not filter" -
        # which is what keeps every row recorded before this from vanishing.
        cmp_ok = (comparable(row(9.0, template_id=t1), "M.gguf", B, 128, 3,
                             template_id=t1)
                  and not comparable(row(9.0, template_id=t1), "M.gguf", B, 128, 3,
                                     template_id=t2)
                  # a campaign that pinned NO template must not inherit a
                  # baseline from one that did
                  and not comparable(row(9.0, template_id=t1), "M.gguf", B, 128, 3,
                                     template_id=None)
                  and not comparable(row(9.0), "M.gguf", B, 128, 3, template_id=t1)
                  and comparable(row(9.0, template_id=t1), "M.gguf", B, 128, 3)
                  and comparable(row(9.0), "M.gguf", B, 128, 3, template_id=_ANY)
                  # ...and it splits effect sizes, never merging two templates
                  and sorted(v["value"] for v in [e for e in axis_effects(
                      [row(5.0, {"ub": 512}, template_id=t1),
                       row(6.0, {"ub": 1024}, template_id=t1),
                       row(99.0, {"ub": 2048}, template_id=t2)])["effects"]
                      if e["axis"] == "ub"][0]["values"]) == [512, 1024])
        # --jinja MUST precede --chat-template-file or the build rejects a path,
        # and a path that does not exist is a SILENT fallback to the GGUF's own
        # template - a whole campaign measured against something nobody chose.
        bc = {"ctx": 4096, "ub": 512, "seq": 1, "ngl": 28, "fa": True, "kv": "q8_0"}
        av = build_argv("s.exe", "m.gguf", dict(bc, chat_template_file=tp,
                                                chat_template_kwargs=kw), 8232)
        arg_ok = (av.index("--jinja") < av.index("--chat-template-file")
                  and av[av.index("--chat-template-file") + 1] == tp
                  and av[av.index("--chat-template-kwargs") + 1] == kw
                  # no flags at all when nothing is pinned
                  and "--jinja" not in build_argv("s.exe", "m.gguf", dict(bc), 8232)
                  # probe=False is launch.py's, which emits these itself at
                  # script runtime; a second copy here would double them
                  and "--jinja" not in build_argv(
                      "s.exe", "m.gguf", dict(bc, chat_template_file=tp), 8232,
                      probe=False))
        try:
            build_argv("s.exe", "m.gguf",
                       dict(bc, chat_template_file=tp + ".nope"), 8232)
            arg_ok = False
        except ValueError:
            pass
        # Samplers are frozen for a campaign, like ctx and kv, and they have to
        # reach the CONFIG - sampling_of() reads them there and _key() hashes
        # them, so a campaign re-run at real settings must not resume its greedy
        # rows as though they were the same measurement. They are not: greedy
        # makes the target's token deterministic, so it is speculation's best
        # case, and an acceptance rate measured there is an upper bound.
        from vram_planner.bench import (stage_configs, sampling_of,
                                        _SWEEP_SAMPLER_KEYS)
        from vram_planner.sweep import _key
        b_greedy = {"ctx": 4096, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512,
                    "ngl": 28, "fill": 2048}
        b_real = dict(b_greedy, temp=1.0, top_k=20, top_p=0.95, rep_pen=1.05)
        # Stage A's configs come from the search, so that is where the campaign
        # settings have to survive to - a sampler that reached stage_configs but
        # not config_for would leave every stage-A row measured greedy.
        from vram_planner.bench import _WallSearch
        c_g = _WallSearch("ngl", [27, 28], b_greedy).config_for(28)
        c_r = _WallSearch("ngl", [27, 28], b_real).config_for(28)
        samp_ok = (sampling_of(c_g)["temperature"] == 0.0
                   and sampling_of(c_r)["temperature"] == 1.0
                   and sampling_of(c_r)["top_k"] == 20
                   and sampling_of(c_r)["repeat_penalty"] == 1.05
                   # ...and the two are different rows, so resume re-measures
                   and _key("m.gguf", c_g) != _key("m.gguf", c_r)
                   # every name the sweep uses survives the trip
                   and all(k in b_real or k in ("min_p", "pres_pen")
                           for k in _SWEEP_SAMPLER_KEYS))
        # the web layer's field names map onto those, or the two cards would
        # measure and launch under settings that only LOOK like each other
        from vram_planner.web import Handler
        mapped = {Handler._SWEEP_SAMPLER.get(k, k) for k in
                  ("temp", "top_k", "top_p", "min_p", "repeat_penalty",
                   "presence_penalty")}
        samp_ok = samp_ok and mapped == set(_SWEEP_SAMPLER_KEYS)
        print("  TMPL  template splits experiments by content, --jinja leads it  %s"
              % ("OK" if (ti_ok and cmp_ok and arg_ok and samp_ok) else "FAIL"))
        find_ok = (chain_ok and split_ok and alone_ok and clean_ok and par_ok
                   and dep_ok and warn_ok and gate_ok
                   and copy_ok and metric_ok and id_ok and vc_ok
                   and tmpl_ok and scheme_ok and port_ok
                   and meas_ok and inf_ok
                   and ti_ok and cmp_ok and arg_ok and samp_ok)
    except Exception as e:
        find_ok = False
        print("  CHAIN/FIND raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and find_ok

    # 14) the -ot dense-FFN tensor knob: n_cpu_ffn, end to end - the regex,
    # the argv, the resume key, the fit gate, stage E's ladder and walls, and
    # the launch scripts it lands in.
    print("\n  -ot dense-FFN tensor offload (n_cpu_ffn)")
    try:
        import re as _re
        from vram_planner.sweep import (ot_regex, _key, build_argv,
                                        parse_overrides, suspect_reason)
        from vram_planner.bench import (stage_configs, build_speed_grid,
                                        _tag_axes, _prune_queue, wall_values,
                                        _carry_summary, SPEED_BASE, STAGES,
                                        CARRY_KEYS, _CONFIG_KEYS, _MONOTONE_FFN)

        # 14a) the regex: every block number spelled out, so the `0` in blk.0
        # can never match the `0` inside blk.10 - the `\.` after the number
        # anchors it either way, but spelling the alternation makes the intent
        # literal. Zero pins nothing, and no flag means no regex.
        ot_ok = (ot_regex(3) == r"blk\.(0|1|2)\.ffn_(gate|up|down)\.weight=CPU"
                 and ot_regex(0) == ""
                 and _re.search(r"blk\.0\.", "blk.10.ffn_up.weight") is None
                 and _re.search(r"blk\.(0|1|2)\.", "blk.2.ffn_up.weight") is not None
                 and "10" in ot_regex(11))
        print("  OT    regex is per-block and anchored  %s"
              % ("OK" if ot_ok else "FAIL"))

        # 14b) build_argv emits the flag exactly when the pin is set - probe
        # and launch alike - and before --n-cpu-moe, because llama.cpp resolves
        # the user's own --override-tensor entries before the ones --n-cpu-moe
        # generates, so an explicit pin must be allowed to win.
        b0 = {"ctx": 8192, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512, "ngl": 20}
        av0 = build_argv("EXE", "M.gguf", dict(b0), 8231)
        av4 = build_argv("EXE", "M.gguf", dict(b0, n_cpu_ffn=4), 8231)
        av8 = build_argv("EXE", "M.gguf", dict(b0, n_cpu_ffn=8), 8231, probe=False)
        avx = build_argv("EXE", "M.gguf", dict(b0, n_cpu_ffn=4, ncmoe=4), 8231,
                         probe=False)
        ot_ok = ot_ok and (
            "--override-tensor" not in av0
            and av4[av4.index("--override-tensor"):
                    av4.index("--override-tensor") + 2] ==
                ["--override-tensor", ot_regex(4)]
            and "--override-tensor" in av8
            and avx.index("--override-tensor") < avx.index("--n-cpu-moe"))
        print("  OT    build_argv emits -ot exactly when pinned  %s"
              % ("OK" if ot_ok else "FAIL"))

        # 14c) the resume key: a different pin is a different measurement, and
        # absent means 0 - rows recorded before the key existed keep keying
        # the same way, so a resumed campaign does not re-run its history.
        ot_ok = ot_ok and (_key("m.gguf", dict(b0)) == _key("m.gguf",
                                                            dict(b0, n_cpu_ffn=0))
                           and _key("m.gguf", dict(b0)) != _key("m.gguf",
                                                                dict(b0, n_cpu_ffn=8)))
        # ...and it is both carried by chaining (a later stage must rebuild at
        # the pin that won) and part of an experiment's identity (two rows at
        # different pins never meet in an effect size).
        ot_ok = ot_ok and ("n_cpu_ffn" in CARRY_KEYS and "n_cpu_ffn" in _CONFIG_KEYS)

        # 14d) the axes grammar reads it like any other int, so --speed-axes
        # and --speed-verify-overrides reach it without a special case.
        ot_ok = ot_ok and parse_overrides(["n_cpu_ffn=0,4,8"]) == \
            {"n_cpu_ffn": [0, 4, 8]}

        # 14e) the fit gate: an ALLOCATION row carrying the pin cannot be
        # fitted - the allocation equation prices whole blocks and has no term
        # for a tensor split - so it stays a visible probe and never corrupts
        # a coefficient. The speed row is a different store and no gate exists
        # there, which is the point.
        good = {"status": "ok", "floor_mib": 200.0, "gpu_free_after_mib": 1000.0,
                "config": dict(b0)}
        pinned = dict(good, config=dict(b0, n_cpu_ffn=4))
        ot_ok = ot_ok and suspect_reason(pinned) and not suspect_reason(good)
        print("  OT    key / argv / fit gate  %s" % ("OK" if ot_ok else "FAIL"))

        # 14f) the pin is no longer SWEPT as a stage. Both dense plan modes
        # exile every block's dense FFN, so a ladder over n_cpu_ffn from a
        # planner seed has no question left to answer - what survives of it is
        # stage A's phase 2, which walks the exile back DOWN once the wall is
        # known, and the hand-set override (--speed-ot).
        #
        # The campaign is A and B. The retired letters - c (ubatch, now a
        # frozen setting), e (the old -ot ladder), and the old projector stage -
        # produce nothing at all rather than something surprising.
        fd = {"arch": "test", "n_layers": 32, "n_ctx_train": 0, "is_moe": False}
        fm = {"arch": "test", "n_layers": 32, "n_ctx_train": 0, "is_moe": True}
        gd = build_speed_grid(fd, base=dict(SPEED_BASE, ctx=8192))
        gm = build_speed_grid(fm, base=dict(SPEED_BASE, ctx=8192))
        blank = dict(SPEED_BASE, ctx=8192, ngl=20)
        ot_ok = ot_ok and (
            STAGES == "ab"
            # no row is stamped with a letter the campaign no longer runs
            and not any(c.get("stage") in ("C", "E") for c in gd + gm)
            # stage B is the only one that builds a list, and it builds
            # speculation rows
            and all(c.get("stage") == "B" for c in gd + gm)
            and any((c.get("spec") or "none") != "none" for c in gd)
            and stage_configs("e", blank, 32, False, "ngl", [1]) == []
            and stage_configs("c", blank, 32, False, "ngl", [1]) == []
            # ...and stage A builds none either: it is a search, so returning
            # the whole domain would misstate its cost by an order of magnitude
            and stage_configs("a", blank, 32, False, "ngl",
                              wall_values("ngl", 32)) == [])
        print("  OT    campaign is %s; the retired letters build nothing  %s"
              % (STAGES, "OK" if ot_ok else "FAIL"))

        # 14g) the monotone wall: n_cpu_ffn is dense-only and runs DOWNWARD -
        # more blocks on the CPU means less VRAM, so an OOM at 16 proves 8
        # fails too. An explicit --speed-axes ladder over it is tagged the same
        # way, and untagged on an MoE where the axis means nothing.
        ot_ok = ot_ok and _MONOTONE_FFN == {"n_cpu_ffn": "down"}
        t1 = [dict(b0, n_cpu_ffn=8), dict(b0, n_cpu_ffn=16)]
        _tag_axes(t1, {"n_cpu_ffn"}, is_moe=False)
        t2 = [dict(b0, n_cpu_ffn=8)]
        _tag_axes(t2, {"n_cpu_ffn"}, is_moe=True)
        ot_ok = ot_ok and (t1[0].get("_wall") == [("n_cpu_ffn", "down")]
                           and "_wall" not in t2[0])
        # ...and the wall pruning: on a DOWNWARD axis, worse means FEWER blocks
        # pinned, so an OOM at 8 in a descending ladder proves the pending 4
        # and 0 fail too and they are dropped - while a sibling at another
        # ubatch is a different family and survives. An ascending ladder
        # OOMing on its LAST rung has nothing pending: the rungs below it
        # already ran, and a run row is a measurement, never pruned.
        eq = [dict(b0, n_cpu_ffn=16, _wall=[("n_cpu_ffn", "down")]),
              dict(b0, n_cpu_ffn=8, _wall=[("n_cpu_ffn", "down")]),
              dict(b0, n_cpu_ffn=4, _wall=[("n_cpu_ffn", "down")]),
              dict(b0, n_cpu_ffn=0, _wall=[("n_cpu_ffn", "down")]),
              dict(b0, n_cpu_ffn=4, ub=1024, _wall=[("n_cpu_ffn", "down")])]
        eqw = {_key("m", c): c.get("_wall") for c in eq}
        kp, dp = _prune_queue([dict(c) for c in eq], eqw, 1, "m",
                              dict(b0, n_cpu_ffn=8))
        ot_ok = ot_ok and ([c["n_cpu_ffn"] for c in kp] == [16, 8, 4]
                           and kp[-1]["ub"] == 1024
                           and sorted(c["n_cpu_ffn"] for c in dp) == [0, 4]
                           and all(c["ub"] == 512 for c in dp))
        print("  OT    monotone wall is dense-only and downward  %s"
              % ("OK" if ot_ok else "FAIL"))

        # 14h) the launch scripts: a config with the pin declares the regex as
        # a per-launch parameter (which tensors move is a legitimate edit), a
        # config without it never mentions the flag, and the carried-baseline
        # summary names the pin so a chained campaign at ffn-cpu 8 does not
        # read as the plain split.
        from vram_planner.launch import launch_script
        lop = launch_script("m.gguf", dict(b0, n_cpu_ffn=4), shell="powershell")
        lob = launch_script("m.gguf", dict(b0, n_cpu_ffn=4), shell="bash")
        lno = launch_script("m.gguf", dict(b0), shell="powershell")
        ot_ok = ot_ok and (
            "--override-tensor" in lop and "$OverrideTensor" in lop
            and r"blk\.(0|1|2|3)\.ffn_(gate|up|down)\.weight=CPU" in lop
            and "--override-tensor" in lob and '"$OVERRIDETENSOR"' in lob
            and "--override-tensor" not in lno
            and "ffn-cpu 8" in _carry_summary(dict(b0, n_cpu_ffn=8))
            and "ffn-cpu" not in _carry_summary(dict(b0)))
        print("  OT    scripts declare the pin, summary names it  %s"
              % ("OK" if ot_ok else "FAIL"))
    except Exception as e:
        ot_ok = False
        print("  OT raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and ot_ok

    # 15) the DRAFT cache's OWN quant (spec_kv): llama.cpp keeps the draft
    #     model's KV cache at f16 whatever -ctk/-ctv say - -ctkd/-ctvd are the
    #     flags that move it, and the sweep's knob is frozen like kv's. The
    #     planner prices the draft cache at the same quant, the launch scripts
    #     carry it, and rows measured under different draft quants never meet.
    print("\n  Draft KV cache quant (spec_kv)")
    try:
        from vram_planner.bench import (SPEED_BASE, _CONFIG_KEYS, EFFECT_AXES,
                                        CARRY_KEYS, _carry_summary, comparable,
                                        _CONFIG_DEFAULTS)
        from vram_planner.launch import launch_script, _config_divergence
        from vram_planner.sweep import _key, build_argv
        from vram_planner.plan import load_drafter
        from vram_planner.kv import kv_bytes_per_token
        from vram_planner.compute import MTP_SPEC_CONST_MIB, MTP_SPEC_PER_SEQ_MIB

        b0 = dict(SPEED_BASE, ctx=8192, ngl=20)
        d0 = dict(b0, spec="draft-mtp", spec_n_max=2)

        # 15a) argv: the flags ride the draft schemes - both K and V, the same
        #     way the target's two are moved - and only there: a config without
        #     a draft scheme has no draft cache to quantise. The campaign's own
        #     default names f16 explicitly rather than leaving llama.cpp to
        #     guess; only a config that predates the knob is silent, because
        #     silence IS llama.cpp's f16.
        dA = dict(d0)
        dA.pop("spec_kv", None)
        av0 = build_argv("EXE", "M.gguf", d0, 8231)
        avq = build_argv("EXE", "M.gguf", dict(d0, spec_kv="q8_0"), 8231)
        avn = build_argv("EXE", "M.gguf", dict(b0, spec_kv="q8_0"), 8231)
        avA = build_argv("EXE", "M.gguf", dA, 8231)
        sk_ok = (av0[av0.index("-ctkd"):av0.index("-ctkd") + 4] ==
                     ["-ctkd", "f16", "-ctvd", "f16"]
                 and avq[avq.index("-ctkd"):avq.index("-ctkd") + 4] ==
                     ["-ctkd", "q8_0", "-ctvd", "q8_0"]
                 and "-ctkd" not in avn and "-ctkd" not in avA)

        # 15b) identity: the draft quant is part of what a speculative row IS.
        #     Absent normalises to llama.cpp's f16 default, so rows recorded
        #     before the knob existed keep keying the same way, and a q8_0 row
        #     is a different measurement that must never merge with an f16 one.
        sk_ok = sk_ok and (_key("m", d0) == _key("m", dict(d0, spec_kv="f16"))
                           and _key("m", d0) != _key("m", dict(d0, spec_kv="q8_0"))
                           and "spec_kv" in _CONFIG_KEYS
                           and "spec_kv" not in CARRY_KEYS
                           and _CONFIG_DEFAULTS["spec_kv"] == "f16"
                           and ("spec_kv", "draft KV quant", ("spec_kv",), "f16")
                               in EFFECT_AXES)
        print("  SPECKV argv / key / identity  %s" % ("OK" if sk_ok else "FAIL"))

        # 15c) the planner prices the draft cache at the quant, not at f16:
        #     the MTP cache term visibly shrinks from f16 to q8_0 (the pool and
        #     per-sequence slot are graph buffers and do not move), and the
        #     DFlash drafter's KV cache scales by the same bytes-per-element
        #     ratio - checked through the exact pricing path it is handed to.
        a16 = analyze(mtps, 4096, "f16", 512, True, vram_budget_mib=2000,
                      ram_budget_mib=8000, gpu_reserve_mib=0,
                      compute_override_mib=0, safety_pct=0, mtp_spec=True,
                      spec_kv="f16")
        aq = analyze(mtps, 4096, "f16", 512, True, vram_budget_mib=2000,
                     ram_budget_mib=8000, gpu_reserve_mib=0,
                     compute_override_mib=0, safety_pct=0, mtp_spec=True,
                     spec_kv="q8_0")
        pool = MTP_SPEC_CONST_MIB + MTP_SPEC_PER_SEQ_MIB * 1
        kv16, kvq = a16["plan"]["spec_mib"] - pool, aq["plan"]["spec_mib"] - pool
        dc = load_drafter(pd)["cfg"]
        sk_ok = sk_ok and (a16["inputs"]["spec_kv"] == "f16"
                           and aq["inputs"]["spec_kv"] == "q8_0"
                           and 0.4 < kv16 - kvq < 1.4
                           and abs(kvq / kv16 - 34.0 / 64.0) < 0.08
                           and abs(kv_bytes_per_token(dc, "q8_0")
                                   / kv_bytes_per_token(dc, "f16")
                                   - 34.0 / 64.0) < 0.001)
        print("  SPECKV planner prices the draft cache at the quant  %s"
              % ("OK" if sk_ok else "FAIL"))

        # 15d) comparability: a q8_0-draft row is a different experiment from
        #     an f16 one - a baseline and an effect size may only be built
        #     inside one quant - and absent compares as f16 on both sides.
        rw = {"config": dict(d0), "tok_s": 1.0, "status": "ok",
              "model": "m.gguf"}
        rq = {"config": dict(d0, spec_kv="q8_0"), "tok_s": 1.0, "status": "ok",
              "model": "m.gguf"}
        sk_ok = sk_ok and (comparable(rw, "m.gguf", d0)
                           and comparable(rw, "m.gguf", dict(d0, spec_kv="f16"))
                           and not comparable(rw, "m.gguf",
                                              dict(d0, spec_kv="q8_0"))
                           and not comparable(rq, "m.gguf", d0))
        print("  SPECKV rows at different draft quants never meet  %s"
              % ("OK" if sk_ok else "FAIL"))

        # 15e) the launch scripts carry the flags with the quant, a script
        #     launched from an f16-measured row at q8_0 says so in the header,
        #     and a carried baseline names the quant instead of reading as the
        #     plain f16 speculation.
        lq = launch_script("m.gguf", dict(d0, spec_kv="q8_0"), shell="powershell")
        lb = launch_script("m.gguf", dict(d0, spec_kv="q8_0"), shell="bash")
        ln = launch_script("m.gguf", d0, shell="powershell")
        div = _config_divergence(dict(d0, spec_kv="q8_0"),
                                 {"config": dict(d0)}, sampling=None)
        sk_ok = sk_ok and ("-ctkd q8_0 -ctvd q8_0" in lq
                           and "-ctkd" in lb
                           and "-ctkd f16 -ctvd f16" in ln
                           and "spec_kv" in div
                           and "spec-kv q8_0" in
                               _carry_summary(dict(d0, spec_kv="q8_0"))
                           and "spec-kv" not in _carry_summary(d0))
        print("  SPECKV scripts, divergence, carried baseline  %s"
              % ("OK" if sk_ok else "FAIL"))
    except Exception as e:
        sk_ok = False
        print("  SPECKV raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and sk_ok

    # 16) SKIP: abandoning ONE config mid-campaign without ending it. The
    #     controlling layer (terminal S key, web Skip button) kills the server,
    #     so the run fails promptly instead of sitting out its generation
    #     timeout, and the row is stamped `skipped` - recorded and keyed so a
    #     resumed campaign never re-measures it, but never a measurement: no
    #     wall, no baseline, no tok/s.
    print("\n  Skip a run")
    try:
        import time as _t
        from .bench import _skip_stamp, trustworthy, _fmt_row
        from .job import Job
        from .launch import launch_script

        b0 = dict(SPEED_BASE, ctx=8192, ngl=20)

        # 16a) the stamping decision, exactly as bench_one sees it. The counter
        #     only grows, so a press binds to the run it was made during, and a
        #     press between runs is consumed by the comparison: the next run's
        #     base already includes it, so nothing later is ever misstamped.
        n = [0]
        sk = lambda: n[0]
        skip_ok = (not _skip_stamp({"status": "genfail"}, sk, 0)
                   and not _skip_stamp({"status": "ok"}, None, 0))
        n[0] = 1
        skip_ok = (skip_ok
                   and _skip_stamp({"status": "genfail"}, sk, 0)
                   and _skip_stamp({"status": "exit"}, sk, 0)
                   # complete measurements stay what they are: an OOM row is
                   # the wall the ladder is read from, and overwriting it with
                   # a skip would throw away the one result the run produced
                   and not _skip_stamp({"status": "ok"}, sk, 0)
                   and not _skip_stamp({"status": "oom"}, sk, 0)
                   and not _skip_stamp({"status": "genfail"}, sk, 1))
        print("  SKIP  the press binds to the run it was made during  %s"
              % ("OK" if skip_ok else "FAIL"))

        # 16b) a skipped row is evidence of the judgement, never a measurement.
        r = {"status": "skipped", "gen_error": "skipped by user",
             "tok_s": 12.5, "config": dict(b0)}
        line = _fmt_row(dict(b0), r)
        skip_ok = skip_ok and (not trustworthy(r)
                               and "SKIPPED" in line
                               and "skipped by user" in line)
        print("  SKIP  recorded but never a measurement  %s"
              % ("OK" if skip_ok else "FAIL"))

        # 16c) the web button's backend: idle refuses, a live server is killed,
        #     the counter bumps once per press and leaves stop/abort alone.
        class _P(object):
            def __init__(self): self.killed = False
            def poll(self): return None
            def kill(self): self.killed = True
        jb = Job()
        js_ok = jb.skip() == (False, "nothing running")   # idle refuses
        jb.status, jb.started = "running", _t.time()
        proc = _P()
        jb.set_live_proc(proc)
        ok3, m3 = jb.skip()
        js_ok = (js_ok and ok3 and m3 == "skipping" and proc.killed
                 and jb.skipped() == 1 and jb.status == "running"
                 and not jb.cancelled() and not jb.aborting()
                 # a press between configs (nothing live) is not an error...
                 and jb.skip()[0]
                 # ...and it is consumed: the count is what stamps, not the
                 # event, so it never bleeds into the next config
                 and jb.skipped() == 2)
        print("  SKIP  web button kills the server, counts the press  %s"
              % ("OK" if js_ok else "FAIL"))
        skip_ok = skip_ok and js_ok

        # 16d) a launcher built over a skipped row says NOT MEASURED - the row
        #     exists so the campaign does not re-run the config, not as
        #     evidence, and a script that claimed "MEASURED" over it would be
        #     the lie the header exists to prevent. The header is comment lines,
        #     so the raw text is checked, not command_lines().
        s = launch_script("m.gguf", dict(b0), shell="bash",
                          measured={"status": "skipped", "config": dict(b0)})
        skip_ok = skip_ok and ("NOT MEASURED" in s and "SKIPPED" in s
                               and "PREDICTED" not in s
                               and "tok/s" not in s)
        print("  SKIP  launcher over a skipped row says so  %s"
              % ("OK" if skip_ok else "FAIL"))
    except Exception as e:
        skip_ok = False
        print("  SKIP raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and skip_ok

    # 16.5) STOP: a second press has to end the config in flight, and the
    #     config in flight spends most of its wall time LOADING. serve() used to
    #     hand the process to the campaign only after the ready line, so for the
    #     whole load - 20-120s normally, up to the timeout when something is
    #     wrong - there was nothing for a hard stop to kill, and the flag it set
    #     was not looked at again until the config finished on its own.
    print("\n  Stop the config in flight")
    try:
        import tempfile as _tf, threading as _th, time as _t
        from .bench import SPEED_BASE
        from .job import Job
        from .sweep import serve, _Server, KILL_WAIT_S
        import vram_planner.sweep as _sw

        # 16.5a) the process is published BEFORE the ready line is waited for.
        #     Faked at the Popen boundary, because the point being pinned is the
        #     ORDER of two sp_calls, not llama.cpp.
        sp_seen = []

        class _Proc(object):
            def __init__(self): self.killed, self.n = False, 0
            def poll(self):
                # never exits on its own; the abort has to be what ends it
                self.n += 1
                return None
            def kill(self): self.killed = True
            def terminate(self): self.killed = True
            def wait(self, timeout=None): return 0

        sp_proc = _Proc()
        # A SHIM module object, not a mutation of the real `subprocess`.
        # subprocess.check_output goes through Popen, so patching the stdlib
        # module in place leaks a fake process into every later section that
        # shells out to nvidia-smi - which is most of them.
        class _FakeSub(object):
            CREATE_NEW_PROCESS_GROUP = 0x200
            DEVNULL = -3
            @staticmethod
            def Popen(*a, **kw):
                return sp_proc
        real_popen, real_free = _sw.subprocess, _sw._free_mib
        _sw.subprocess = _FakeSub
        _sw._free_mib = lambda: 1000.0
        try:
            sp_flag = _th.Event()
            with serve({"exe": "x", "dir": ".", "vendor": []}, "m.gguf",
                       dict(SPEED_BASE, ctx=1024, ngl=1), port=8299, timeout=30.0,
                       log_dir=_tf.mkdtemp(), on_proc=sp_seen.append,
                       should_abort=lambda: (sp_seen and sp_flag.set()) or sp_flag.is_set()) as srv:
                pass
            hstop_ok = (sp_seen == [sp_proc]              # published, exactly once
                       and srv.status == "aborted" # the WAIT ended, not the timeout
                       and sp_proc.killed)            # and the process went with it
        finally:
            _sw.subprocess, _sw._free_mib = real_popen, real_free
        print("  STOP  the server is published during the load, not after  %s"
              % ("OK" if hstop_ok else "FAIL"))

        # 16.5b) Job.cancel: first press is sp_soft, every press after it kills
        #     whatever is live. It used to kill only on the exact second press,
        #     so a second press that landed between configs killed nothing and a
        #     third returned "aborting" and also did nothing.
        sp_job = Job()
        hstop_ok = hstop_ok and sp_job.cancel() == (False, "nothing running")
        sp_job.status, sp_job.started = "running", _t.time()
        sp_a = _Proc()
        sp_job.set_live_proc(sp_a)
        sp_ok1, sp_m1 = sp_job.cancel()
        sp_soft = (sp_ok1 and sp_m1 == "stopping" and not sp_a.killed
                and sp_job.cancelled() and not sp_job.aborting())
        sp_ok2, sp_m2 = sp_job.cancel()
        sp_hard = sp_ok2 and sp_m2 == "aborting" and sp_a.killed and sp_job.aborting()
        # a press landing between configs kills nothing and is not an error;
        # the NEXT one still kills the server that came up meanwhile
        sp_job.set_live_proc(None)
        sp_ok3, _m3 = sp_job.cancel()
        sp_b = _Proc()
        sp_job.set_live_proc(sp_b)
        sp_ok4, _m4 = sp_job.cancel()
        hstop_ok = hstop_ok and sp_soft and sp_hard and sp_ok3 and sp_ok4 and sp_b.killed
        print("  STOP  first press soft, every press after it kills  %s"
              % ("OK" if hstop_ok else "FAIL"))

        # 16.5c) an abandoned row is discarded, so nothing slow is measured for
        #     it: finish_row skips the two nvidia-smi reads and marks the row
        #     instead of leaving the gap looking like a failed reading.
        sp_srv = _Server(8299, os.path.join(_tf.mkdtemp(), "r.log"))
        sp_srv.status, sp_srv.free_before = "aborted", 900.0
        sp_calls = [0]

        def _counted():
            sp_calls[0] += 1
            return 1234.0

        real_free, real_total = _sw._free_mib, _sw._total_mib
        _sw._free_mib = _sw._total_mib = _counted
        try:
            sp_ra = _sw.finish_row({"status": "error", "config": {}}, sp_srv, aborted=True)
            sp_rb = _sw.finish_row({"status": "error", "config": {}}, sp_srv)
        finally:
            _sw._free_mib, _sw._total_mib = real_free, real_total
        hstop_ok = hstop_ok and (sp_ra.get("aborted") is True
                               and "gpu_free_after_mib" not in sp_ra
                               and sp_calls[0] == 2            # only the second row read
                               and sp_rb.get("gpu_free_after_mib") == 1234.0
                               and "aborted" not in sp_rb
                               # the reading taken BEFORE the run is kept either
                               # way: it is the one measured while it mattered
                               and sp_ra["gpu_free_before_mib"] == 900.0)
        # ...and the kill waits are short enough that a stop feels like one
        hstop_ok = hstop_ok and 0 < KILL_WAIT_S <= 10
        print("  STOP  an abandoned row costs no telemetry  %s"
              % ("OK" if hstop_ok else "FAIL"))
    except Exception as e:
        hstop_ok = False
        print("  STOP raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and hstop_ok

    # 16.6) STAGE C IS RETIRED: ubatch is a setting, not an axis. It drives
    #     PREFILL, and the campaign measures decode - this tool's own insights
    #     put the whole ladder at +1.3% against +37% for speculation - so four
    #     loads an hour apiece were being spent to re-derive that. It is frozen
    #     from the form now, like ctx and kv, and the old letters still resolve
    #     rather than silently meaning nothing.
    print("\n  Retired stage C")
    try:
        from .bench import (resolve_stages, STAGES, CARRY_KEYS, comparable,
                            speed_sweep)
        from .web import Handler as _H

        # 16.6a) the letters. `d` IS stage B - the same speculation stage under
        #     its new name - so it translates and says so; `c` has no stage to
        #     map to, so it is dropped and says so, naming the flag that still
        #     does it. Neither is allowed to mean nothing quietly, which is the
        #     rule resolve_search() and resolve_rounds() already follow.
        c_ok = (STAGES == "ab"
                and resolve_stages("ab") == ("ab", [])
                and resolve_stages("acd")[0] == "ab"
                and resolve_stages("d")[0] == "b"
                and resolve_stages("a") == ("a", [])
                # order is the CAMPAIGN's, not the typing order
                and resolve_stages("ba")[0] == "ab"
                # both translations are said out loud
                and any("stage B now" in n for n in resolve_stages("d")[1])
                and any("--speed-axes" in n for n in resolve_stages("c")[1])
                # asking for only the retired stage still runs something, and
                # says why - answering with zero loads and silence is worse
                and resolve_stages("c")[0] == "ab"
                # declining to choose is not an error and needs no note
                and resolve_stages("") == ("ab", []))
        print("  NOUB  the old stage letters still resolve, out loud  %s"
              % ("OK" if c_ok else "FAIL"))

        # 16.6b) ub crossed the line from result to definition, and it had to
        #     cross in BOTH lists at once - out of CARRY_KEYS and into
        #     comparable(). Half of it would have let the ub-1024 rows every old
        #     stage C left on disk set the baseline for a campaign frozen at 512.
        ub_b = {"ctx": 8192, "kv": "q8_0", "ub": 512, "seq": 1, "fa": True,
                "fill": 2048, "spec_kv": "f16"}
        def _ubrow(u):
            return {"model": "M", "status": "ok", "tok_s": 9.0, "n_predict": 128,
                    "repeat": 3, "config": dict(ub_b, ub=u)}
        c_ok = c_ok and ("ub" not in CARRY_KEYS
                         and comparable(_ubrow(512), "M", ub_b, 128, 3)
                         and not comparable(_ubrow(1024), "M", ub_b, 128, 3)
                         and not comparable(_ubrow(2048), "M", ub_b, 128, 3))

        # 16.6c) the value reaches the campaign. The plan form has always had
        #     the field and plan.analyze() has always taken it; what was missing
        #     was the wire between them, so every row was measured at 512
        #     whatever the plan above it said.
        args = _H._speed_args(_H, {"path": "", "n_ubatch": 1024})
        c_ok = c_ok and args["ub"] == 1024 and args["stages"] == "ab"
        # ...and its default matches the form's, so a body that predates the
        # field does not silently change what gets measured
        c_ok = c_ok and _H._speed_args(_H, {"path": ""})["ub"] == 512
        # the CLI spells it --speed-ub and speed_sweep takes it by that name
        import inspect as _insp
        c_ok = c_ok and "ub" in _insp.signature(speed_sweep).parameters
        print("  NOUB  ubatch is frozen: gated, not carried, and wired  %s"
              % ("OK" if c_ok else "FAIL"))
    except Exception as e:
        c_ok = False
        print("  NOUB raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and c_ok

    # 16.6d) ZERO IS A VALUE. "0 CPU FFN blocks" is an answer - keep every
    #     dense FFN on the card - and it was being read as a blank field and
    #     dropped, twice over: the browser's `0` failed `v not in (None, "",
    #     False)` because Python has 0 == False, and even when a 0 survived that,
    #     grid_context() applied the plan mode's every-block pin OVER it. The
    #     user set the knob to zero and watched the FFN get offloaded anyway.
    print("\n  Zero as an override")
    try:
        from .web import given, read_ui
        from .bench import ngl_ladder, grid_context

        # the wire: 0 is given, blank is not, and False is still a checkbox
        z_ok = (given(0) and given("0") and given(0.0)
                and not given(None) and not given("") and not given(False))
        # every override the plan form sends, not just the one that was noticed
        args_z = _H._speed_args(_H, {"path": "", "ot": 0})
        z_ok = z_ok and args_z["ot"] == 0
        z_ok = z_ok and _H._speed_args(_H, {"path": ""})["ot"] is None

        # ...and the knob is ONE knob. The plan form's override and the sweep
        # card's -ot field are two inputs for the same number, and the campaign
        # only ever read the second: planning at 0 and then sweeping measured
        # every block exiled, at every rung, which is the opposite of what was
        # asked for. The specific field wins; blank inherits; both blank is the
        # plan mode's own pin.
        def _ot(**kw):
            return _H._speed_args(_H, dict(kw, path=""))["ot"]
        z_ok = z_ok and (_ot(n_cpu_ffn_override=0) == 0
                         and _ot(n_cpu_ffn_override=8) == 8
                         and _ot(n_cpu_ffn_override=8, ot=0) == 0
                         and _ot(n_cpu_ffn_override=0, ot=8) == 8
                         and _ot() is None)
        # the browser does the same thing, so the two halves cannot drift
        _js = read_ui("app.js")
        z_ok = z_ok and ("plannedFfn" in _js
                         and 'v("swot") !== null ? v("swot") : plannedFfn()' in _js)

        # the pin: planner_split() cannot answer for a path that is not a model,
        # so the ladder falls back to its own seed - which is the interesting
        # case, because the FFN pin does not come from the seed at all.
        zc = {"ctx": 8192, "kv": "q8_0", "ub": 512, "fa": True, "seq": 1}
        _ax, _v, pins_d = ngl_ladder("no-such-model.gguf", zc, 48)
        _ax2, _v2, pins_0 = ngl_ladder("no-such-model.gguf", zc, 48, ffn_pin=0)
        _ax3, _v3, pins_9 = ngl_ladder("no-such-model.gguf", zc, 48, ffn_pin=9)
        # absent means the mode's every-block pin, which is what makes it the
        # mode; 0 and 9 are answers, and 0 is not "absent" spelled differently
        z_ok = z_ok and (pins_d["n_cpu_ffn"] == 48 and pins_0["n_cpu_ffn"] == 0
                         and pins_9["n_cpu_ffn"] == 9)
        # ...and it is clamped, not trusted: more blocks than the model has is
        # a typo, not a config
        z_ok = z_ok and ngl_ladder("no-such-model.gguf", zc, 48,
                                   ffn_pin=999)[2]["n_cpu_ffn"] == 48

        # the clobber: the pin has to reach the LADDER, because grid_context
        # applies pins over the baseline - a hand-set value riding in on `base`
        # was overwritten by the mode's own, silently.
        zf = {"n_layers": 48, "n_ctx_train": 32768, "is_moe": False}
        zb = dict(zc, n_cpu_ffn=0)
        z_ok = z_ok and grid_context(zf, base=zb,
                                     model_path="no-such-model.gguf")[0]["n_cpu_ffn"] == 48
        z_ok = z_ok and grid_context(zf, base=zb, model_path="no-such-model.gguf",
                                     ffn_pin=0)[0]["n_cpu_ffn"] == 0
        print("  ZERO  0 blocks on CPU is a config, not a blank field  %s"
              % ("OK" if z_ok else "FAIL"))
    except Exception as e:
        z_ok = False
        print("  ZERO raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and z_ok

    # 16.7) SPILL GATES: a row wrongly called spilled is a row stage B cannot
    #     rank and the recommendation card will not show. Three of the four
    #     clauses were firing on healthy rows.
    print("\n  Spill gates")
    try:
        from .sweep import unsound_reason, suspect_reason, FLOOR_MAX_MIB, \
            FLOOR_MIN_MIB

        gb0 = {"status": "ok", "floor_mib": 220.0, "gpu_free_after_mib": 4000.0,
               "config": {"ctx": 8192, "ub": 512, "ngl": 20}}

        # 16.7a) an unreadable floor is a fact about the TELEMETRY, not about
        #     the run. get_gpu_processes() failing, or the process being gone by
        #     the time it was sampled, used to read as a spill and cost stage B
        #     a candidate every time nvidia-smi hiccupped. demoted() already
        #     states the rule: unmeasured is not evidence of absence.
        sp_ok = (unsound_reason(dict(gb0, floor_mib=None)) == ""
                 # a genuinely negative floor is still the WDDM signature and
                 # still fires: the allocator asked for more than the process
                 # holds, so the reading is the cap and not the need
                 and unsound_reason(dict(gb0, floor_mib=-2000.0))
                 and unsound_reason(dict(gb0, floor_mib=FLOOR_MIN_MIB + 1.0)) == "")

        # 16.7b) FLOOR_MAX_MIB says "a floor this large is not a CUDA context",
        #     and on a row carrying a draft cache or an in-VRAM projector that
        #     premise is false - llama.cpp reports neither in its buffer lines,
        #     so both land in floor. A draft-mtp row measured 1050 MiB where its
        #     plain twin sat at 230, and a deep depth at a large context clears
        #     4096 legitimately: exactly the rows stage B exists to measure.
        big = dict(gb0, floor_mib=FLOOR_MAX_MIB + 500.0)
        sp_ok = sp_ok and (
            unsound_reason(big)                       # plain row: still flagged
            and unsound_reason(dict(big, config=dict(big["config"],
                                                     spec="draft-mtp",
                                                     spec_n_max=4))) == ""
            and unsound_reason(dict(big, config=dict(big["config"],
                                                     mmproj="m.gguf"))) == ""
            # ...but a projector deliberately left in RAM allocates nothing on
            # the card, so it earns no exemption
            and unsound_reason(dict(big, config=dict(big["config"],
                                                     mmproj="m.gguf",
                                                     mmproj_offload=False)))
            and unsound_reason(dict(big, config=dict(big["config"],
                                                     spec="none"))))

        # 16.7c) "the card had almost nothing left" is not a defect in a SPEED
        #     row - it is the definition of the row stage A is looking for, so
        #     every winner at the wall read that way and was gated out. It is
        #     also measured after teardown, when a low reading is the driver not
        #     having released yet. It belongs to the allocation FIT, which does
        #     want headroom around the point it is pricing.
        tight = dict(gb0, gpu_free_after_mib=100.0)
        sp_ok = sp_ok and (unsound_reason(tight) == ""
                           and suspect_reason(tight)
                           and suspect_reason(gb0) == "")
        # the fit gate still rejects everything unsound_reason does, plus the
        # tensor split it has no term for
        sp_ok = sp_ok and (suspect_reason(dict(gb0, floor_mib=-2000.0))
                           and suspect_reason(dict(gb0, config=dict(
                               gb0["config"], n_cpu_ffn=8))))
        # and nothing judges a row that never loaded
        sp_ok = sp_ok and (unsound_reason(dict(gb0, status="oom")) == ""
                           and suspect_reason(dict(gb0, status="oom")) == "")
        print("  SPILL unmeasured is not a verdict; the wall is not a defect  %s"
              % ("OK" if sp_ok else "FAIL"))
    except Exception as e:
        sp_ok = False
        print("  SPILL raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and sp_ok

    # 17) MODE-AWARE STAGE A: which knob the wall is made of. A dense model that
    #     does not fit has one answer per question, and stage A has to ladder the
    #     axis belonging to the question being asked - laddering -ngl for a speed
    #     plan measures a config that plan never proposes, and laddering ctx for a
    #     context plan sweeps the one number the user pinned. On an MoE neither
    #     applies: the axis is --n-cpu-moe and always was.
    print("\n  Mode-aware stage A")
    try:
        from .bench import (_WallSearch, wall_values, wall_probes,
                            ctx_values, ctx_search_step, axis_direction,
                            CTX_STEP, CTX_SEARCH_DIVISIONS,
                            _MONOTONE_AXES, _prune_queue)
        from .sweep import _key

        b17 = dict(b0, spec="draft-mtp", spec_n_max=3, ub=256, n_cpu_ffn=32)

        # 17a) the axis reaches the rows: each stage-A config varies exactly the
        #     named knob and carries that axis's own wall direction, so the
        #     first OOM prunes the right half of an explicit ladder and the
        #     search reads the same fact when it bisects.
        w_ngl = _WallSearch("ngl", [18, 20, 22], dict(b17, ngl=20))
        w_ctx = _WallSearch("ctx", [16384, 32768, 65536], dict(b17, ngl=32))
        w_moe = _WallSearch("ncmoe", [6, 5, 4], dict(b17, ngl=32))
        sa_ngl = [w_ngl.config_for(v) for v in w_ngl.values]
        sa_ctx = [w_ctx.config_for(v) for v in w_ctx.values]
        sa_moe = [w_moe.config_for(v) for v in w_moe.values]
        m_ok = ([c["ngl"] for c in sa_ngl] == [18, 20, 22]
                and all(c["_wall"] == [("ngl", "up")] for c in sa_ngl)
                and [c["ctx"] for c in sa_ctx] == [16384, 32768, 65536]
                # the layers stay put while the window moves - that IS the mode
                and all(c["ngl"] == 32 and c["n_cpu_ffn"] == 32 for c in sa_ctx)
                and all(c["_wall"] == [("ctx", "up")] for c in sa_ctx)
                and [c["ncmoe"] for c in sa_moe] == [6, 5, 4]
                and all(c["_wall"] == [("ncmoe", "down")] for c in sa_moe)
                and all(c["stage"] == "A"
                        for c in sa_ngl + sa_ctx + sa_moe))
        print("  MODEA each axis varies its own knob, tagged its own way  %s"
              % ("OK" if m_ok else "FAIL"))

        # 17b) there is now ONE direction table, read through axis_direction().
        #     There used to be two (_AXIS_DIRECTION beside _MONOTONE_*) that
        #     agreed "by construction", which is a guarantee that lasts until
        #     somebody edits one of them - a disagreement would have pruned the
        #     wrong half of a ladder in silence.
        m_ok = (m_ok
                and axis_direction("ngl") == _MONOTONE_AXES["ngl"] == "up"
                and axis_direction("ctx") == _MONOTONE_AXES["ctx"] == "up"
                and axis_direction("ncmoe") == "down"
                and axis_direction("n_cpu_ffn") == "down"
                # ...and wall_values() reads it, so the domain of every search
                # is ordered ascending in VRAM COST whichever way its axis runs
                and wall_values("ngl", 4) == [0, 1, 2, 3, 4]
                and wall_values("ncmoe", 4) == [4, 3, 2, 1, 0]
                and wall_values("n_cpu_ffn", 4) == [4, 3, 2, 1, 0])
        # ...and it prunes: an OOM at 32768 in an ascending ctx ladder proves
        # 65536 fails the same way, while a sibling at another ubatch survives
        # because it is a different family.
        q = [dict(c) for c in sa_ctx] + [dict(sa_ctx[-1], ub=1024)]
        walls = {_key("m", c): c.get("_wall") for c in q}
        # the OOM row IS the queued config with its wall tag stripped, which is
        # what run_group hands over - _same_family compares every other key.
        oom17 = dict(q[1]); oom17.pop("_wall", None)
        keep, drop = _prune_queue(q, walls, 1, "m", oom17)
        m_ok = m_ok and [c["ctx"] for c in drop] == [65536] and len(keep) == 3
        print("  MODEA one ctx OOM prunes the larger windows only  %s"
              % ("OK" if m_ok else "FAIL"))

        # 17c) the ctx domain: the WHOLE trained range, in steps of
        #     n_ctx_train/64 and never finer than CTX_STEP, so the rungs read as
        #     sizes a person would type and the search costs the same handful of
        #     probes on any model. The trained maximum is always in it, because
        #     "the whole window fits" is the answer that starts phase 2.
        dom = ctx_values(262144)
        m_ok = (m_ok
                and dom == sorted(set(dom))
                and dom[-1] == 262144 and dom[0] == ctx_search_step(262144)
                and ctx_search_step(262144) == 4096
                and all(v % CTX_STEP == 0 for v in dom)
                and len(dom) == CTX_SEARCH_DIVISIONS
                # the floor takes over on a small window: 1024-token rungs
                and ctx_search_step(32768) == CTX_STEP
                and ctx_values(32768)[-1] == 32768
                # a context the step does not divide still ends AT the maximum
                and ctx_values(100000)[-1] == 100000
                and ctx_values(0) == [] and ctx_values(None) == [])
        # ...and the cost is log2, not linear. That is the whole reason the
        # domain can afford to be the whole range instead of a bracket around
        # a planner seed that might be centred anywhere.
        m_ok = m_ok and (wall_probes(dom) <= 7 and wall_probes(range(66)) <= 7
                         and wall_probes([]) == 0 and wall_probes([1]) == 1)
        print("  MODEA ctx domain is the whole range, at log2 cost  %s"
              % ("OK" if m_ok else "FAIL"))

        # 17d) the search converges on the exact wall, from anywhere in the
        #     domain, within its own advertised budget - and it is DETERMINISTIC:
        #     the same statuses give the same probes, so a resumed campaign
        #     retraces its own path instead of exploring a different one.
        def _converge(axis, values, fits, known=None):
            w = _WallSearch(axis, values, dict(b17), known_fits=known)
            cap, taken = w.budget, []
            while True:
                c = w.next()
                if c is None:
                    break
                taken.append(c[axis])
                w.feed({"status": "ok" if fits(c[axis]) else "oom"})
            return w, cap, taken

        conv_ok = True
        for nl in (1, 7, 32, 65):
            vals = wall_values("ngl", nl)
            for wall in range(0, nl + 1):
                w, cap, taken = _converge("ngl", vals, lambda v, x=wall: v <= x)
                w2, _c2, taken2 = _converge("ngl", vals, lambda v, x=wall: v <= x)
                conv_ok = conv_ok and (w.winner == wall and len(taken) <= cap
                                       and taken == taken2
                                       and w.topped_out == (wall == nl))
        # the FFN walk runs the other way and is seeded with the value phase 1
        # already proved, so it never spends a probe re-measuring it
        fv = wall_values("n_cpu_ffn", 32)
        wf, capf, takenf = _converge("n_cpu_ffn", fv,
                                     lambda v: v >= 12, known=32)
        conv_ok = conv_ok and (wf.winner == 12 and 32 not in takenf
                               and len(takenf) <= capf)
        # nothing loading is not a winner, and --limit can cut a search short
        wn, _cn, _tn = _converge("ngl", wall_values("ngl", 8), lambda v: False)
        wl = _WallSearch("ngl", wall_values("ngl", 65), dict(b17), max_probes=2)
        nl_taken = 0
        while wl.next() is not None:
            nl_taken += 1
            wl.feed({"status": "ok"})
        conv_ok = conv_ok and (wn.winner is None and not wn.topped_out
                               and nl_taken == 2)
        # a status that is not `ok` never counts as fitting - not oom, and not
        # the four ways a run can be interrupted either. The predicate has to be
        # total for a bisection, and this is the direction that cannot promote
        # something the user skipped.
        for st in ("oom", "skipped", "genfail", "timeout", "exit", "aborted"):
            wq = _WallSearch("ngl", [1, 2], dict(b17))
            while True:
                c = wq.next()
                if c is None:
                    break
                wq.feed({"status": "ok" if c["ngl"] == 1 else st})
            conv_ok = conv_ok and wq.winner == 1
        m_ok = m_ok and conv_ok
        print("  MODEA the search converges on the wall, deterministically  %s"
              % ("OK" if conv_ok else "FAIL"))
    except Exception as e:
        m_ok = False
        print("  MODEA raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and m_ok

    # 18) SLOW RUN: a config that does not OOM can still be unusable - WDDM
    #     spills it into shared memory, or the machine is busy - and then every
    #     pass crawls and the row, when it finally lands, is a plausible-looking
    #     number that is not worth the minutes it took. Three sides of the same
    #     fix: the mid-run abort (a pass at a fraction of what the model
    #     normally does ends the measurement, status `spilled`), the recorded
    #     rows verdict (a stored row at a fraction of its campaign's own median
    #     is flagged the same way a spill is, so it stops being evidence), and
    #     the resume filter (a judged row - skipped by the user, spilled by the
    #     machine - is never re-measured).
    print("\n  Slow-run abort and collapse")
    try:
        from .bench import _abort_floor, _resume_recorded, _infer_demotion, \
            SLOW_FRAC, trustworthy
        from .sweep import _key

        def hrow(tok, model="M.gguf", ngl=27, **kw):
            r = {"model": model, "gpu": "G", "_file": "f.jsonl", "status": "ok",
                 "tok_s": tok, "floor_mib": 1050.0,
                 "config": {"ctx": 131072, "kv": "q8_0", "ub": 512, "seq": 1,
                            "fa": True, "ngl": ngl, "fill": 2048}}
            r.update(kw)
            return r

        # 18a) the abort floor is 15% of the median of the healthy rows measured
        #     under the SAME CONDITIONS; None until there is something to judge
        #     against. A spilled, looping or copying row is not a healthy row -
        #     it must not drag the floor down onto the configs that follow it.
        c18 = {"ctx": 131072, "kv": "q8_0", "ub": 512, "seq": 1, "fa": True,
               "ngl": 28, "fill": 2048}
        af = _abort_floor([hrow(5.6), hrow(5.8), hrow(5.58)], c18, "M.gguf")
        sr_ok = (af is not None and abs(af - 0.15 * 5.6) < 0.001
                 and _abort_floor([hrow(5.6)], c18, "M.gguf") is None
                 and _abort_floor([], c18, "M.gguf") is None
                 and _abort_floor([hrow(5.6)], c18, "Other.gguf") is None
                 and _abort_floor([hrow(5.6, spilled=True), hrow(5.8)],
                                  c18, "M.gguf") is None
                 and _abort_floor([hrow(5.6, distinct_ratio=0.2), hrow(5.8)],
                                  c18, "M.gguf") is None)
        # ...and it groups the way _infer_demotion() groups, which is the whole
        # point: the two make the same judgement, so they must make it from the
        # same rows. Decode falls ~6x from 2k to 120k of fill on one config, so
        # a floor drawn from shallow rows aborts a legitimate deep one at 15% of
        # a number it could never reach - and records it `spilled`.
        deep = dict(c18, fill=120000)
        shallow = [hrow(9.45), hrow(9.40), hrow(9.50)]
        sr_ok = sr_ok and _abort_floor(shallow, deep, "M.gguf") is None
        # the deep rows judge each other, and the shallow ones stay out of it
        deep_rows = [dict(hrow(t), config=dict(deep)) for t in (1.49, 1.51, 1.47)]
        afd = _abort_floor(deep_rows + shallow, deep, "M.gguf")
        sr_ok = sr_ok and afd is not None and abs(afd - 0.15 * 1.49) < 0.001
        # speculation is the same story pointing the other way: a draft row runs
        # far faster, so it must not raise the floor under the plain baseline
        spec_c = dict(c18, spec="draft-mtp", spec_n_max=2)
        spec_rows = [dict(hrow(t), config=dict(spec_c)) for t in (13.0, 13.2)]
        sr_ok = sr_ok and _abort_floor(spec_rows, c18, "M.gguf") is None
        print("  SLOWRUN abort floor: 15%% of the healthy median, grouped  %s"
              % ("OK" if sr_ok else "FAIL"))

        # 18b) resume: a judged row is never re-measured. `ok`/`oom` are
        #     measurements, `skipped`/`spilled` are judgements (by the user and
        #     by the machine), and everything else failed for a reason that may
        #     not hold tomorrow. A row measured under other settings is a
        #     different experiment, not this campaign's work already done.
        rows18 = [hrow(5.6, ngl=27, n_predict=128, repeat=3, prompt_id="p",
                       template_id=None),
                  dict(hrow(5.0, ngl=28, n_predict=128, repeat=3, prompt_id="p",
                            template_id=None), status="oom"),
                  dict(hrow(0.5, ngl=29, n_predict=128, repeat=3, prompt_id="p",
                            template_id=None), status="spilled"),
                  dict(hrow(0.0, ngl=30, n_predict=128, repeat=3, prompt_id="p",
                            template_id=None), status="skipped"),
                  dict(hrow(4.0, ngl=31, n_predict=128, repeat=3, prompt_id="p",
                            template_id=None), status="genfail"),
                  hrow(5.0, ngl=27, n_predict=32, repeat=1, prompt_id="p",
                       template_id=None),
                  hrow(5.0, ngl=27, n_predict=128, repeat=3, prompt_id="q",
                       template_id=None),
                  hrow(5.0, ngl=27, n_predict=128, repeat=3, prompt_id="p",
                       template_id="t")]
        rec = _resume_recorded(rows18, 128, 3, "p", None)
        sr_ok = (sr_ok and len(rec) == 4
                 and all(rec[k]["status"] in ("ok", "oom", "skipped", "spilled")
                         for k in rec))
        print("  SLOWRUN resume keeps judgements, drops the failed  %s"
              % ("OK" if sr_ok else "FAIL"))

        # 18c) the recorded-rows verdict: a row at a fraction of its campaign's
        #     own median tok/s is flagged like a spill - excluded from every
        #     conclusion - with the ratio named. The wall being found moves a
        #     ladder by tens of percent, never by 6x, so the top rung is safe.
        def crows(tok, **kw):
            r = hrow(tok, **kw)
            r["config"] = dict(r["config"], fill=65536)
            return r
        grp = [crows(5.6), crows(5.8), crows(0.576)]
        _infer_demotion(grp)
        sr_ok = (sr_ok and grp[-1]["spilled"]
                 and grp[-1].get("collapse_inferred") is not None
                 and grp[-1]["collapse_inferred"] < SLOW_FRAC
                 and not grp[0].get("spilled") and not grp[1].get("spilled")
                 and trustworthy(grp[0]) and not trustworthy(grp[-1]))
        wall = [crows(5.9), crows(5.6), crows(4.5)]
        _infer_demotion(wall)
        sr_ok = (sr_ok and not any(r.get("spilled") for r in wall)
                 and all(trustworthy(r) for r in wall))
        # a lone row has no campaign to be a fraction of - the grouping needs 3
        lone18 = [crows(0.4)]
        _infer_demotion(lone18)
        sr_ok = (sr_ok and not lone18[0].get("spilled"))
        # a verdict from a memory signal survives: the collapse pass runs after
        # the counter/floor verdicts, so a demoted row keeps its demotion
        sus18 = [crows(5.6), crows(5.8), crows(3.9, floor_mib=-500.0,
                                                  shared_mib=700.0)]
        _infer_demotion(sus18)
        sr_ok = (sr_ok and sus18[-1]["spilled"])
        print("  SLOWRUN stored rows at a fraction of the median are flagged  %s"
              % ("OK" if sr_ok else "FAIL"))

        # 18d) the Skip button's route: the endpoint exists in the handler, so
        #     the allow-list must carry it too. This is the actual bug this
        #     section exists for - the button 404'd and the fetch swallowed it.
        import vram_planner.web as _web
        sr_ok = (sr_ok and "/api/speed/skip" in _web._POSTS
                 and "/api/speed/stop" in _web._POSTS)
        print("  SLOWRUN skip route is on the POST allow-list  %s"
              % ("OK" if sr_ok else "FAIL"))

        # 18e) a launcher over a spilled row says NOT MEASURED, like a skipped
        #     one - the row exists so the campaign does not re-measure it, not
        #     as evidence.
        s18 = launch_script("m.gguf", dict(b0), shell="bash",
                            measured={"status": "spilled", "config": dict(b0)})
        sr_ok = (sr_ok and "NOT MEASURED" in s18 and "SPILLED" in s18
                 and "PREDICTED" not in s18 and "tok/s" not in s18)
        print("  SLOWRUN launcher over a spilled row says so  %s"
              % ("OK" if sr_ok else "FAIL"))
    except Exception as e:
        sr_ok = False
        print("  SLOWRUN raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and sr_ok

    # 19) WALL SIGNATURE: current llama.cpp Windows builds die at the wall
    #     with a GGML_ASSERT(buffer) abort - the NULL-buffer check in
    #     ggml_backend_alloc_ctx_tensors - instead of a clean "failed to
    #     allocate" line. Before the assert was added to RE_OOM, every such
    #     row landed as `exit`: it never pruned the queue, never recorded a
    #     wall, and a resumed sweep re-measured all the rungs it had already
    #     crashed on. An assert that appears AFTER the server is up is
    #     something else's problem and stays non-OOM.
    print("\n  Allocation-wall signature")
    try:
        from .sweep import parse_log
        crash = ("0.08.170.077 I load_hparams: model size:         884.62 MiB\n"
                 "llm-engine/llama.cpp/ggml/src/ggml-backend.cpp:179: "
                 "GGML_ASSERT(buffer) failed\n")
        clean = ("0.07.025.102 E graph_reserve: failed to allocate compute "
                 "buffers\n")
        as_ = parse_log(crash)
        no = parse_log("0.01.000.000 I llama_server: listening on http://...\n"
                       + crash)
        cl = parse_log(clean)
        ok_ = (as_["oom"] and not no["oom"] and cl["oom"])

        # 19b) ...and the other way round: `error loading model` was IN RE_OOM,
        #     so every way a FILE can be refused was recorded as the card
        #     running out. It is not a harmless mislabel. _draft_probe() reads
        #     an OOM on a speculative row as "free some VRAM and retry" and
        #     walks the FFN back to the CPU one rung at a time, so an MTP head
        #     shipped without the blocks its header promises - it loads as a
        #     draft model and llama.cpp indexes past the end - was re-refused at
        #     seven progressively freer splits and left seven rows behind that
        #     read as "speculation does not fit here".
        from .sweep import RE_OOM
        from .bench import _spec_retry
        drf = parse_log(
            "0.04.733.251 E common_speculative_init_result: failed to load "
            "draft model, 'mtp-Q.gguf': error loading model: invalid vector "
            "subscript\n")
        # a load error is NOT an OOM, and the reason survives for the row to say
        ok_ = ok_ and (not drf["oom"]
                       and drf["load_error"] == "invalid vector subscript"
                       and not RE_OOM.search("error loading model: bad file"))
        # ...but the allocator failing INSIDE the loader still is one
        oom_in_load = parse_log("error loading model: unable to allocate "
                                "CUDA0 buffer of size 3808.00 MiB\n")
        ok_ = ok_ and oom_in_load["oom"] and oom_in_load["load_error"] is None
        # and the walk leaves a loadfail row alone - this is the whole point.
        # The identical row as an OOM still walks, so the gate is the STATUS and
        # not some accident of the config.
        _spec_c = {"spec": "draft-mtp", "spec_n_max": 1, "ngl": 64,
                   "n_cpu_ffn": 0, "ctx": 4096}
        _facts = {"n_layers": 64, "is_moe": False}
        _drf = {"kind": "mtp", "depth": 4}
        ok_ = ok_ and (
            _spec_retry(_spec_c, {"config": _spec_c, "status": "loadfail"},
                        _facts, {}, _drf, "ctx") is None
            and _spec_retry(_spec_c, {"config": _spec_c, "status": "oom"},
                            _facts, {}, _drf, "ctx") is not None)
        print("  WALLSIG assert-only log is oom, post-ready assert is not  %s"
              % ("OK" if ok_ else "FAIL"))
    except Exception as e:
        ok_ = False
        print("  WALLSIG raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and ok_

    # ---- the recommendation, and the two divergences it exists to report ----
    #
    # The planner and the speed sweep answered the same question two different
    # ways and neither said so. Three things are pinned here because each was a
    # silent wrong answer before it was a test.
    print("\n  Recommendation and budget parity")
    try:
        from vram_planner.plan import default_vram_budget
        from vram_planner.bench import PLAN_BASIS, PLAN_RESERVE_MIB
        from vram_planner.recommend import plan_config, recommend

        # 1) ONE budget rule. planner_split() used to compute total-minus-512
        #    inline while the browser prefilled free-minus-nothing. The helper is
        #    now the only definition; this pins it to what the sweep expects.
        g = {"total_mib": 16376.0, "free_mib": 11508.0}
        b_total = default_vram_budget(g, PLAN_BASIS, 0)
        b_free = default_vram_budget(g, "free", 0)
        b_res = default_vram_budget(g, PLAN_BASIS, PLAN_RESERVE_MIB)
        budget_ok = (b_total == 16376.0 and b_free == 11508.0
                     and b_res == 16376.0 - PLAN_RESERVE_MIB
                     and PLAN_BASIS == "total"
                     # never negative, however small the card or large the reserve
                     and default_vram_budget({"total_mib": 100}, "total", 4096) == 0.0
                     and default_vram_budget(None) == 0.0)
        print("  BUDGET one rule, total basis, reserve applied, never negative  %s"
              % ("OK" if budget_ok else "FAIL"))

        # 2) The verdict is COMPUTED, not inferred by the browser. The step tab
        #    read r.verdict.fits and r.totals.vram_mib, neither of which this API
        #    has ever returned - so it announced "fits" for every plan analyzed,
        #    including the ones that did not.
        r_no = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=8,
                       ram_budget_mib=8, gpu_reserve_mib=0,
                       compute_override_mib=40, safety_pct=0)
        r_yes = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=100000,
                        ram_budget_mib=100000, gpu_reserve_mib=0,
                        compute_override_mib=40, safety_pct=0)
        v_no, v_yes = r_no["plan"]["verdict"], r_yes["plan"]["verdict"]
        verdict_ok = (v_yes["state"] == "fits" and v_yes["word"] == "FITS"
                      and v_no["state"] in ("no_fit", "spills")
                      and v_no["word"] != v_yes["word"]
                      and v_yes["vram_mib"] > 0
                      # the two knobs a launcher needs, normalised across planners
                      and "ngl" in v_yes and "ncmoe" in v_yes)

        # The verdict must be reported on the SAME basis the memory bar draws,
        # or the step tab and the card under it give two answers to one
        # question - they differed by exactly the driver reserve, so the tab
        # read 10.25 GiB beside a bar reading 11,013 MiB.
        r_res = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=4000,
                        ram_budget_mib=40000, gpu_reserve_mib=512,
                        compute_override_mib=40, safety_pct=5)
        pl, vr = r_res["plan"], r_res["plan"]["verdict"]
        bar = sum(pl.get(k) or 0 for k in
                  ("gpu_weights_mib", "gpu_kv_mib", "gpu_recurrent_mib",
                   "mmproj_mib", "spec_mib", "compute_mib")) + 512
        basis_ok = (abs(vr["vram_mib"] - bar) < 0.5
                    and vr["vram_budget_mib"] == 4000
                    # ...while the fit decision stays on the effective pair
                    and abs(vr["vram_used_eff_mib"] - pl["vram_used_mib"]) < 0.5
                    and abs(vr["vram_budget_eff_mib"] - pl["vram_budget_mib"]) < 0.5
                    and vr["vram_budget_eff_mib"] < vr["vram_budget_mib"])
        verdict_ok = verdict_ok and basis_ok
        print("  VERDICT state server-side, starved plan is not 'fits', bar agrees  %s"
              % ("OK" if verdict_ok else "FAIL"))

        # 3) recommend(): a measured winner supersedes the estimate, an
        #    untrustworthy one never does however fast it reads, and every reason
        #    the two differ is named rather than left for the user to spot.
        pr = {"plan": {"n_gpu_layers": 41, "n_cpu_moe": 18, "fits_fully": False},
              "inputs": {"context": 32768, "kv_type": "q8_0", "n_ubatch": 512,
                         "n_seq": 1, "flash_attn": True, "vram_budget_mib": 11508}}
        base_cfg = {"ctx": 32768, "kv": "q8_0", "fa": True, "seq": 1, "fill": 2048}
        win = {"model": "m.gguf", "status": "ok", "tok_s": 8.41,
               "proc_vram_mib": 14880,
               "config": dict(base_cfg, ngl=47, ncmoe=12, ub=1024,
                              spec="draft-mtp", spec_n_max=2)}
        # Faster, and unusable: WDDM spilled it into shared system memory, so it
        # reports ok while running off a cliff. It must never win.
        spill = {"model": "m.gguf", "status": "ok", "tok_s": 9.9, "spilled": True,
                 "config": dict(base_cfg, ngl=50, ncmoe=0, ub=512)}
        got = recommend(pr, [win, spill], sweep_budget_mib=15864)
        kinds = {d["kind"] for d in got["deltas"]}
        rec_ok = (got["source"] == "measured"
                  and got["config"]["ngl"] == 47 and got["tok_s"] == 8.41
                  and got["n_trusted"] == 1 and got["n_rows"] == 2
                  # the objective difference is the headline: largest-that-fits
                  # is not fastest-measured, and nothing said so before
                  and "objective" in kinds
                  # priced against different amounts of VRAM
                  and "budget" in kinds
                  # knobs no plan has a field for
                  and "axis" in kinds)
        only_spill = recommend(pr, [spill], sweep_budget_mib=15864)
        rec_ok = rec_ok and (only_spill["source"] == "predicted"
                             and only_spill["config"]["ngl"] == 41
                             and {d["kind"] for d in only_spill["deltas"]} == {"untrusted"})
        none_at_all = recommend(pr, [])
        rec_ok = rec_ok and (none_at_all["source"] == "predicted"
                             and none_at_all["deltas"] == [])
        # A row measured at another context is a different experiment, not a
        # faster config - strict mode must not let it take over the answer.
        elsewhere = {"model": "m.gguf", "status": "ok", "tok_s": 40.0,
                     "config": dict(base_cfg, ctx=4096, ngl=60)}
        far = recommend(pr, [elsewhere, win], sweep_budget_mib=15864)
        rec_ok = rec_ok and far["config"]["ngl"] == 47
        # ...and with nothing comparable at all it still answers, saying so.
        loose = recommend(pr, [elsewhere], sweep_budget_mib=15864)
        rec_ok = rec_ok and (loose["source"] == "measured"
                             and "stale" in {d["kind"] for d in loose["deltas"]})
        # agreement is silent: same split, same budget, no unplannable knobs
        agree = {"model": "m.gguf", "status": "ok", "tok_s": 5.0,
                 "config": dict(base_cfg, ngl=41, ncmoe=18, ub=512)}
        quiet = recommend(pr, [agree], sweep_budget_mib=11508)
        rec_ok = rec_ok and quiet["deltas"] == []
        # ...and silence has to hold for the DEFAULT plan, which is the one
        # nearly every user sees. The raw budget field is not the number a split
        # is chosen against - the reserve and the safety margin come off first -
        # so comparing it against the sweep's effective basis reported a budget
        # delta on every plan ever made, including this one, where the entire
        # gap is the reserve the sweep had already taken off.
        pr_def = {"plan": {"n_gpu_layers": 41, "n_cpu_moe": 18, "fits_fully": False},
                  "inputs": {"context": 32768, "kv_type": "q8_0", "n_ubatch": 512,
                             "n_seq": 1, "flash_attn": True,
                             "vram_budget_mib": 16376, "gpu_reserve_mib": 512,
                             "safety_pct": 5}}
        sweep_eff = (16376 - 512) * 0.95          # what web.py hands recommend()
        parity = recommend(pr_def, [agree], sweep_budget_mib=sweep_eff)
        rec_ok = rec_ok and parity["deltas"] == []
        # ...while a basis that really does differ still says so: free-at-page-
        # load against card-total is the divergence this module was written for.
        pr_free = {"plan": pr_def["plan"],
                   "inputs": dict(pr_def["inputs"], vram_budget_mib=11508,
                                  gpu_reserve_mib=0)}
        split = recommend(pr_free, [agree], sweep_budget_mib=sweep_eff)
        rec_ok = rec_ok and {d["kind"] for d in split["deltas"]} == {"budget"}
        # comparable() also gates on the SAMPLERS, and a plan has none of those
        # either - so a campaign that swept real sampler settings matched
        # nothing, strict narrowing emptied, and the fallback recommended a
        # sampled row against a greedy assumption without a word. It falls back
        # still, because some answer beats none; it names it now.
        sampled = {"model": "m.gguf", "status": "ok", "tok_s": 7.0,
                   "config": dict(base_cfg, ngl=41, ncmoe=18, ub=512,
                                  temp=0.7, top_p=0.95)}
        samp = recommend(pr, [sampled], sweep_budget_mib=11508)
        samp_text = "".join(d["text"] for d in samp["deltas"] if d["kind"] == "samplers")
        rec_ok = rec_ok and (samp["source"] == "measured"
                             and {d["kind"] for d in samp["deltas"]} == {"samplers"}
                             and "temperature 0.7" in samp_text)
        # ...and stays quiet when the rows were greedy, which is what `agree` is.
        rec_ok = rec_ok and "samplers" not in {d["kind"] for d in quiet["deltas"]}
        # the plan's own config survives the trip into a row's vocabulary
        pc = plan_config(pr)
        rec_ok = rec_ok and pc["ngl"] == 41 and pc["ncmoe"] == 18 and pc["kv"] == "q8_0"
        # ...and so does the speculation it priced: an external MTP drafter
        # becomes spec draft-mtp + md, the model's own MTP blocks become a
        # scheme without a file, and a plan with neither invents neither.
        pc_dr = plan_config(em)
        rec_ok = rec_ok and pc_dr["spec"] == "draft-mtp" \
            and pc_dr["spec_n_max"] == 1 and pc_dr["md"] == em["inputs"]["drafter"]
        own_inp = {k: v for k, v in em["inputs"].items()
                   if k not in ("drafter", "drafter_kind", "drafter_depth")}
        pc_own = plan_config({"plan": em["plan"],
                              "inputs": dict(own_inp, mtp_depth=2)})
        rec_ok = rec_ok and pc_own["spec"] == "draft-mtp" \
            and pc_own["spec_n_max"] == 2 and "md" not in pc_own
        pc_dd = plan_config(ed)
        rec_ok = rec_ok and pc_dd["spec"] == "draft-dflash" \
            and pc_dd["spec_n_max"] == 8 and pc_dd["md"] == ed["inputs"]["drafter"]
        print("  RECOMM measured wins, spilled never does, every delta named    %s"
              % ("OK" if rec_ok else "FAIL"))
        rec_all = budget_ok and verdict_ok and rec_ok

        # 4) "Best" is per CATEGORY, because the two categories do not share an
        #    objective. Each dense mode pins one placement and leaves exactly one
        #    knob free, and along a knob that is monotone in VRAM the value worth
        #    having is the one at the wall - not the one that read fastest. Rows
        #    move ~4% of tok/s across a DOUBLING of context, and downward, so
        #    ranking a SPEED campaign by tok/s recommends the smallest window in
        #    the one mode whose whole purpose is the largest.
        from vram_planner.recommend import mode_axis
        from vram_planner.bench import axis_direction, pick_extreme

        nL = 48
        sp_pr = {"plan_mode": "ceiling",
                 "plan": {"kind": "dense_ceiling", "mode": "ceiling", "n_gpu_layers": nL,
                           "n_cpu_ffn": nL, "max_ctx": 65536, "fits_fully": False},
                 "config": {"n_layers": nL},
                 "inputs": {"context": 32768, "kv_type": "f16", "n_ubatch": 512,
                             "n_seq": 1, "flash_attn": True, "vram_budget_mib": 16376,
                             "gpu_reserve_mib": 512, "safety_pct": 5}}
        s_base = {"kv": "f16", "fa": True, "seq": 1, "fill": 2048}
        # The trap, exactly as measured: the SMALLEST window is the fastest row,
        # by a margin far inside the noise a context ladder produces anyway.
        s_rows = [{"model": "m.gguf", "status": "ok", "tok_s": t, "proc_vram_mib": 15000,
                   "config": dict(s_base, ctx=c, ngl=nL, n_cpu_ffn=nL, ub=512)}
                  for c, t in ((32768, 9.10), (65536, 8.95), (131072, 8.80))]
        s_got = recommend(sp_pr, s_rows, sweep_budget_mib=sweep_eff)
        mode_ok = (mode_axis(sp_pr) == ("ctx", "up")
                   and s_got["axis"] == "ctx" and s_got["objective"] == "extreme"
                   and s_got["source"] == "measured"
                   # the largest window that loaded, not the fastest row
                   and s_got["config"]["ctx"] == 131072
                   and "context" in s_got["goal"])
        # ...and it stays the largest that loaded even when it is much slower.
        # There used to be a 5% slack here that dropped a rung running below its
        # neighbours before the extreme was taken; it is gone, because it was a
        # speed judgement inside a question that is not about speed, and it
        # could hand back half the window to buy a difference the same ladder
        # produces as jitter.
        s_thrash = s_rows[:2] + [dict(s_rows[2], tok_s=2.0,
                                      config=dict(s_rows[2]["config"]))]
        mode_ok = mode_ok and recommend(sp_pr, s_thrash,
                                        sweep_budget_mib=sweep_eff)["config"]["ctx"] == 131072
        # What DOES remove such a row is a spill detector answering from the
        # memory counters, not an inference from the ranking - and the card
        # gates on those through trustworthy(), so the slow row drops out and
        # the next-largest window is recommended instead.
        s_spill = s_rows[:2] + [dict(s_rows[2], tok_s=2.0, spilled=True,
                                     config=dict(s_rows[2]["config"]))]
        mode_ok = mode_ok and recommend(sp_pr, s_spill,
                                        sweep_budget_mib=sweep_eff)["config"]["ctx"] == 65536

        # CONTEXT holds the window and pays in the cheapest currency first: while
        # every block still fits the free knob is the -ot exile and LESS exiled is
        # better, so the wall runs downward. That direction was wrong until
        # axis_direction() learned the dense table - it read n_cpu_ffn as "up"
        # and promoted the MOST exiled rung, the opposite of the answer.
        cx_pr = {"plan_mode": "fit",
                 "plan": {"kind": "dense_fit", "mode": "fit", "n_gpu_layers": nL,
                           "n_cpu_ffn": 30, "max_ctx": 32768, "fits_fully": False},
                 "config": {"n_layers": nL},
                 "inputs": dict(sp_pr["inputs"])}
        c_rows = [{"model": "m.gguf", "status": "ok", "tok_s": t, "proc_vram_mib": 15000,
                   "config": dict(s_base, ctx=32768, ngl=nL, n_cpu_ffn=f, ub=512)}
                  for f, t in ((26, 8.20), (30, 8.30), (40, 8.35))]
        c_got = recommend(cx_pr, c_rows, sweep_budget_mib=sweep_eff)
        mode_ok = mode_ok and (mode_axis(cx_pr) == ("n_cpu_ffn", "down")
                               and axis_direction("n_cpu_ffn") == "down"
                               and c_got["config"]["n_cpu_ffn"] == 26
                               and "FFN" in c_got["goal"])
        # ...and once a full exile is not enough, whole blocks leave and the free
        # knob becomes -ngl, which runs the other way. Same rule as ngl_ladder().
        cx_deep = {"plan_mode": "fit",
                   "plan": dict(cx_pr["plan"], n_gpu_layers=31, n_cpu_ffn=nL),
                   "config": {"n_layers": nL}, "inputs": dict(sp_pr["inputs"])}
        mode_ok = mode_ok and mode_axis(cx_deep) == ("ngl", "up")

        # A category exempts its own axis from the staleness gate, and nothing
        # else: a ceiling plan did not ASK for a context, it proposed one, so a row
        # that found a larger window is the answer improving. A row at another KV
        # quant is still a different experiment.
        wrong_kv = [dict(s_rows[2], config=dict(s_rows[2]["config"], kv="q8_0"))]
        kv_got = recommend(sp_pr, wrong_kv, sweep_budget_mib=sweep_eff)
        mode_ok = mode_ok and "stale" in {d["kind"] for d in kv_got["deltas"]}
        mode_ok = mode_ok and "stale" not in {d["kind"] for d in s_got["deltas"]}

        # Off the two-plan regime nothing is left free, so fastest-wins stands -
        # which is what every MoE and every fits-whole plan gets, and what the
        # cases above this one are still asserting.
        mode_ok = mode_ok and mode_axis(pr) == (None, None)             and recommend(pr, [win], sweep_budget_mib=15864)["objective"] == "fastest"
        # ...and a plan_mode with no matching plan is not a category either: the
        # browser can hold "ceiling" while looking at an MoE.
        mode_ok = mode_ok and mode_axis({"plan_mode": "ceiling",
                                         "plan": {"kind": "moe"}}) == (None, None)
        # an old result in a live session carries the old mode names; the
        # read-side mapping must keep answering with the new categories' axes
        mode_ok = mode_ok and mode_axis({"plan_mode": "speed",
                                         "plan": {"mode": "speed",
                                                  "kind": "dense_speed"}}
                                        ) == ("ctx", axis_direction("ctx"))
        mode_ok = mode_ok and mode_axis({"plan_mode": "context",
                                         "plan": {"mode": "context",
                                                  "kind": "dense_context"}}
                                        ) == ("ngl", axis_direction("ngl"))
        # The free axis only means something among rows that pinned everything
        # else the way this category does. Measured on the real store: the
        # ceiling card answered "the largest context that loaded" with a row at
        # -ngl 28, a fit campaign's row where most of the model is in RAM and
        # 128k naturally fits - the other plan's answer wearing this plan's badge.
        other_regime = {"model": "m.gguf", "status": "ok", "tok_s": 4.0,
                        "proc_vram_mib": 15000,
                        "config": dict(s_base, ctx=262144, ngl=20, n_cpu_ffn=0, ub=512)}
        pin_got = recommend(sp_pr, s_rows + [other_regime], sweep_budget_mib=sweep_eff)
        mode_ok = mode_ok and (pin_got["config"]["ctx"] == 131072
                               and pin_got["config"]["ngl"] == nL
                               and "regime" not in {d["kind"] for d in pin_got["deltas"]})
        # ...and with nothing measured in this layout at all it still answers,
        # saying which question the rows it had were answering instead.
        only_other = recommend(sp_pr, [other_regime], sweep_budget_mib=sweep_eff)
        mode_ok = mode_ok and (only_other["source"] == "measured"
                               and "regime" in {d["kind"] for d in only_other["deltas"]})
        # pick_extreme() is the same function the campaign promotes with, so the
        # card and the sweep cannot land on different rows - including when the
        # largest is the slowest, which is the case that used to split them.
        mode_ok = mode_ok and (pick_extreme(s_rows, "ctx")["config"]["ctx"] == 131072
                               and pick_extreme(s_thrash, "ctx")["config"]["ctx"] == 131072
                               and pick_extreme([], "ctx") is None)
        print("  RECMODE ceiling takes the widest window, fit the least exiled  %s"
              % ("OK" if mode_ok else "FAIL"))
        rec_all = rec_all and mode_ok
    except Exception as e:
        rec_all = False
        print("  RECOMMEND raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and rec_all

    # ---- forgetting a campaign -------------------------------------------
    #
    # Destructive, and what it destroys is hours of GPU time. Every guarantee it
    # makes is pinned here: it takes only its own campaign, it never unlinks the
    # file that campaign shares with others, and it leaves the removed rows
    # somewhere they can be moved back from.
    print("\n  Forgetting a campaign")
    try:
        import json as _json
        from vram_planner import bench as _b
        from vram_planner.bench import campaign_match, delete_campaign, sweep_index

        store = os.path.join(tmp, "speed")
        os.makedirs(store, exist_ok=True)
        _real_dir = _b.bench_dir
        _b.bench_dir = lambda: store
        try:
            fn = "GPU_A__build-1.jsonl"
            def _row(model, pid, tid, tok):
                return {"model": model, "gpu": "GPU A", "prompt_id": pid,
                        "template_id": tid, "status": "ok", "tok_s": tok,
                        "when": 1000 + tok, "n_predict": 128, "repeat": 3,
                        "config": {"ctx": 4096, "kv": "f16", "fa": True, "seq": 1,
                                   "ngl": int(tok), "ub": 512, "fill": 0}}
            # Three campaigns in ONE file: same model under two prompts, plus a
            # second model. Deleting the first must leave the other two intact -
            # a file is one GPU and one build, never one campaign.
            rows = ([_row("a.gguf", "p1", "t1", 1.0), _row("a.gguf", "p1", "t1", 2.0)]
                    + [_row("a.gguf", "p2", "t1", 3.0)]
                    + [_row("b.gguf", "p1", "t1", 4.0)])
            with open(os.path.join(store, fn), "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(_json.dumps(r) + "\n")
            # A line this tool did not write is not this tool's to discard.
            with open(os.path.join(store, fn), "a", encoding="utf-8", newline="\n") as f:
                f.write("{not json at all}\n")

            before = sweep_index(_b.load_speed_rows())
            res = delete_campaign(model="a.gguf", gpu="GPU A", file=fn,
                                  prompt_id="p1", template_id="t1")
            after = sweep_index(_b.load_speed_rows())
            left = {(g["model"], g["prompt_id"]) for g in after}
            del_ok = (len(before) == 3 and res.get("ok") and res["removed"] == 2
                      # the file survives, carrying the campaigns it shared with
                      and os.path.isfile(os.path.join(store, fn))
                      and len(after) == 2
                      and ("a.gguf", "p2") in left and ("b.gguf", "p1") in left
                      and ("a.gguf", "p1") not in left
                      # ...and the rows are recoverable, not gone
                      and res["backup"] and os.path.isfile(res["backup"])
                      and len(open(res["backup"], encoding="utf-8")
                              .read().strip().splitlines()) == 2
                      # the unparseable line was left exactly where it was
                      and "{not json at all}" in open(os.path.join(store, fn),
                                                      encoding="utf-8").read())
            # An empty prompt_id is a VALUE - the campaigns that pinned none -
            # and must never widen to every campaign of that model.
            wide = delete_campaign(model="b.gguf", gpu="GPU A", file=fn,
                                   prompt_id="", template_id="")
            del_ok = del_ok and not wide.get("ok") and wide.get("removed") == 0
            del_ok = del_ok and campaign_match(
                {"model": "b.gguf", "prompt_id": "p1"}, model="b.gguf", prompt_id="p1")
            # A name that matches nothing changes nothing.
            miss = delete_campaign(model="nope.gguf", gpu="GPU A", file=fn)
            del_ok = del_ok and not miss.get("ok") and len(sweep_index(
                _b.load_speed_rows())) == 2
            # ...and neither does one naming a file that is not there.
            gone = delete_campaign(model="a.gguf", gpu="GPU A", file="no-such.jsonl")
            del_ok = del_ok and not gone.get("ok")

            # A campaign appending WHILE the delete reads: the row that landed
            # is not in the rewrite, so replacing would forget a measurement
            # nobody asked to forget. The web server refuses this by checking
            # its own JOB, but a --forget-sweep in a terminal is a different
            # process and cannot see that job at all - so the guard that counts
            # is here, and it is the file itself that reports the collision.
            n_backups = len(os.listdir(_b.deleted_dir()))
            _real_read = _b._read_lines
            def _read_then_append(path):
                out = _real_read(path)
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(_json.dumps(_row("a.gguf", "p2", "t1", 9.0)) + "\n")
                return out
            _b._read_lines = _read_then_append
            try:
                race = delete_campaign(model="a.gguf", gpu="GPU A", file=fn,
                                       prompt_id="p2", template_id="t1")
            finally:
                _b._read_lines = _real_read
            still = {(g["model"], g["prompt_id"]) for g in sweep_index(_b.load_speed_rows())}
            del_ok = del_ok and (not race.get("ok")
                                 and "changed" in (race.get("error") or "")
                                 # the campaign is untouched, appended row and all
                                 and ("a.gguf", "p2") in still
                                 and "\"ngl\": 9" in open(os.path.join(store, fn),
                                                          encoding="utf-8").read()
                                 # and no backup left behind claiming otherwise
                                 and len(os.listdir(_b.deleted_dir())) == n_backups)

            # Two campaigns out of one file inside the same second. The backup
            # stamp is second-granular, so the second copy landed on the first
            # and the rows it was protecting went with it - a recovery file that
            # silently replaces another is worse than none, because it is the
            # thing the confirm dialog is not enough protection without.
            second = delete_campaign(model="b.gguf", gpu="GPU A", file=fn,
                                     prompt_id="p1", template_id="t1")
            backups = os.listdir(_b.deleted_dir())
            del_ok = del_ok and (second.get("ok") and second["removed"] == 1
                                 and len(backups) == n_backups + 1
                                 and len(set(backups)) == len(backups)
                                 and all(os.path.getsize(os.path.join(
                                     _b.deleted_dir(), b)) > 0 for b in backups))
            print("  FORGET takes one campaign, keeps the file, keeps a copy back  %s"
                  % ("OK" if del_ok else "FAIL"))
        finally:
            _b.bench_dir = _real_dir
    except Exception as e:
        del_ok = False
        print("  FORGET raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and del_ok

    # ---------------------------------------------------------------- 22 ----
    # The controls a model actually has, and getting a card back out.
    #
    # Two failures this pins, both of which were invisible rather than loud:
    #
    #  * every optional control - the projector placement, the MTP tick box,
    #    --n-cpu-moe against -ot - was revealed by render(), which runs only on
    #    an analyze RESPONSE. So a setting that changes the answer first appeared
    #    underneath the answer. /api/probe answers the same question from the
    #    header alone, and ONE function in the page acts on it.
    #  * #mtprow was nested inside #mmprojfield, so its own test could never fire
    #    for a model without a projector: a text-only model shipping MTP blocks
    #    could not show the box, while run() sent its checked value anyway.
    print("\n  Model probe and the card library")
    try:
        import inspect as _insp
        import vram_planner.web as _w2
        from vram_planner import cards as _cards_mod
        ui = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
        js = open(os.path.join(ui, "app.js"), encoding="utf-8").read()
        html = open(os.path.join(ui, "index.html"), encoding="utf-8").read()

        # both routes reachable: a handler without its allow-list entry 404s
        probe_ok = (hasattr(_w2.Handler, "_probe")
                    and hasattr(_w2.Handler, "_cards_forget")
                    and '"/api/probe"' in _insp.getsource(_w2.Handler.do_GET)
                    and "/api/cards/forget" in _w2._POSTS
                    and '"/api/cards/forget"' in _insp.getsource(_w2.Handler.do_POST))

        # the probe reads a card when the file is gone - that is what makes a
        # deleted model still selectable, and it must not raise on junk
        probe_ok = (probe_ok
                    and _w2.Handler._probe(None, "")["ok"] is False
                    and _w2.Handler._probe(None, "no-such-model.gguf")["ok"] is False)

        # ONE writer for the optional fields. Seven inline assignments in
        # render() is how they came to be analyze-only in the first place.
        # Only ASSIGNMENTS count. The field has to be on the LEFT of the `=`,
        # or reading one field's .hidden to derive another's (which is what the
        # speculation sublabel does) reads as a second writer.
        fields = ("mmprojfield", "visionrow", "mtprow", "ncpumoefield",
                  "ncpuffnfield", "dflashfield")
        writes = {}
        for f in fields:
            n = 0
            for ln in js.splitlines():
                if ".hidden = " not in ln:
                    continue
                lhs = ln.split(".hidden = ")[0]
                if lhs.lstrip().startswith("//"):
                    continue
                if lhs.rstrip().endswith('$("%s")' % f):
                    n += 1
            writes[f] = n
        # dflashfield is written by loadDrafters() too - the manual path input
        # lives inside it, so the picker has to stay reachable on its own
        probe_ok = (probe_ok and "function applyFieldVisibility(" in js
                    and all(writes[f] == 1 for f in fields if f != "dflashfield"))

        # ...and the MTP row is not a child of the projector field, or its test
        # is dead again the moment a model has no mmproj.
        after_mm = html.split('id="mmprojfield"', 1)[1]
        mm_block = after_mm.split("</div>", 1)[0]
        probe_ok = (probe_ok and 'id="mtprow"' not in mm_block
                    and 'id="visionrow"' not in mm_block
                    and 'id="mtprow"' in html and 'id="visionrow"' in html)
        print("  PROBE controls revealed on selection, MTP row is its own field  %s"
              % ("OK" if probe_ok else "FAIL"))

        # forgetting a card: real, reversible by re-reading the file, and it
        # refuses a name it does not have rather than reporting success
        from vram_planner.cards import (forget_card, list_cards, load_cards,
                                        make_card, save_cards)
        _real_store = _cards_mod._cards_store
        _cards_mod._cards_store = lambda: os.path.join(tmp, "cards_test.json")
        try:
            d = {"cards": {}}
            d["cards"]["A.gguf"] = make_card(
                "A.gguf", {"arch": "llama", "n_layers": 4}, {"params_total": 1}, None, 10, 1)
            d["cards"]["B.gguf"] = make_card(
                "B.gguf", {"arch": "llama", "n_layers": 4}, {"params_total": 1}, None, 10, 1)
            save_cards(d)
            card_ok = len(list_cards()) == 2
            r = _w2.Handler._cards_forget(None, {"name": "A.gguf"})
            card_ok = (card_ok and r["ok"] and len(r["cards"]) == 1
                       and r["cards"][0]["name"] == "B.gguf")
            # the other one is untouched - deleting is per name, never a wipe
            card_ok = card_ok and [c["name"] for c in list_cards()] == ["B.gguf"]
            # a name with no card is refused, and says so
            r2 = _w2.Handler._cards_forget(None, {"name": "A.gguf"})
            card_ok = card_ok and r2["ok"] is False and "no card" in r2["error"]
            r3 = _w2.Handler._cards_forget(None, {})
            card_ok = card_ok and r3["ok"] is False
            # the UI has a way in, and asks twice before doing it
            card_ok = (card_ok and '"card-forget"' in js
                       and '"card-forget-yes"' in js
                       and "/api/cards/forget" in js
                       and 'id="cardlist"' in html)
        finally:
            _cards_mod._cards_store = _real_store
        print("  CARDS  one card forgotten by name, the rest kept, UI can ask   %s"
              % ("OK" if card_ok else "FAIL"))
        probe_ok = probe_ok and card_ok
    except Exception as e:
        probe_ok = False
        print("  PROBE raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and probe_ok

    if skipped_real:
        print("\n  %d real-measurement section(s) did not run: %s"
              % (len(skipped_real), ", ".join(skipped_real)))
        print("  The synthetic sections check that the code does what it says. Only "
              "these check that\n  what it says is true. Pass --require-refs to make "
              "a skip here a failure.")
        if require_refs:
            print("  --require-refs was given, so this is a FAILURE.")
            ok = False
    print("\n  SELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
