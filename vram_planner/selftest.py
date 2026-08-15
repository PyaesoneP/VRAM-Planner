"""Synthetic GGUF writer and the self-test suite."""
import os, struct
from .const import _mib
from .gguf import GGML_TYPES, _parse_one, load_gguf
from .model import RE_EXPS, classify_tensors, extract_config
from .compute import CB_CUDA_CTX_MIB, CB_DEFAULTS, CB_SPLIT_GRAPH_MIB, CB_SPLIT_PER_TOKEN, compute_buffer_split, compute_buffer_terms, graph_is_split, output_head_on_gpu, vision_grid, vision_peak_mib
from .lmstudio import REF_GPU, current_backend, read_lmstudio_runtime, resolve_runtime_ngl
from .calib import CALIB_SCHEMA, CALIB_TERMS, _CALIB_CACHE, _active_gpu, _design, _struct_offset, calib_coeffs, fit_calibration, mark_unreliable
from .plan import analyze, find_mmproj


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

    # 5) KV-on-GPU mode exiles the dense FFN to RAM - the speed model must charge
    #    those bytes to the CPU side, not treat the whole model as GPU-resident.
    rk = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=30, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=5, safety_pct=0, kv_on_gpu=True,
                 bw_vram_gbs=600, bw_ram_gbs=80, ctx_fill=1024)
    sk = rk.get("speed") or {}
    # must actually take that branch, or the assertion below proves nothing
    kv_ok = (rk["plan"]["kind"] == "dense_kv_gpu" and sk.get("cpu_mib", 0) > 0
             and rk["plan"].get("ffn_on_cpu"))
    # 5b) An explicit layer count is a layer-level split, so it must reach a planner
    #     that HAS a layer count. KV-on-GPU offloads every block by definition and
    #     splits FFN tensors instead; it used to win the branch and drop the override
    #     silently, so analyze() returned byte-identical plans for different -ngl.
    ro_a = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=30, ram_budget_mib=8000,
                   gpu_reserve_mib=0, compute_override_mib=5, safety_pct=0, kv_on_gpu=True,
                   gpu_layers_override=1)
    ro_b = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=30, ram_budget_mib=8000,
                   gpu_reserve_mib=0, compute_override_mib=5, safety_pct=0, kv_on_gpu=True,
                   gpu_layers_override=3)
    ovr_ok = (ro_a["plan"]["kind"] == "dense"                     # routed away from kv_gpu
              and ro_a["plan"]["n_gpu_layers"] == 1               # ...and honoured
              and ro_b["plan"]["n_gpu_layers"] == 3
              and ro_a["plan"]["vram_used_mib"] != ro_b["plan"]["vram_used_mib"]
              # the recommendation path (no override) must STILL reach kv-on-gpu
              and rk["plan"]["kind"] == "dense_kv_gpu")
    print("  NGL-OVR kv_on_gpu + override: kind=%s ngl 1->%.0f MiB, 3->%.0f MiB (differ=%s), "
          "no-override still %s  %s"
          % (ro_a["plan"]["kind"], ro_a["plan"]["vram_used_mib"], ro_b["plan"]["vram_used_mib"],
             ro_a["plan"]["vram_used_mib"] != ro_b["plan"]["vram_used_mib"],
             rk["plan"]["kind"], "OK" if ovr_ok else "FAIL"))
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

    print("  KV-ON-GPU kind=%s ffn_on_cpu=%s cpu_bytes/token=%.1f MiB  %s"
          % (rk["plan"]["kind"], rk["plan"].get("ffn_on_cpu"), sk.get("cpu_mib", 0),
             "OK" if kv_ok else "FAIL"))
    ok = ok and kv_ok

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
        # only the six knobs carry; the campaign's own definition never does
        c, _ = best_config([row(9.9, {"ngl": 31, "ub": 2048, "spec": "draft-mtp",
                                      "spec_n_max": 2, "stage": "D"})],
                           "M.gguf", B, 128, 3, incumbent_tok_s=INC)
        if not c or c["ub"] != 2048 or c["spec"] != "draft-mtp" or "stage" in c \
                or c["ctx"] != B["ctx"] or c["fill"] != B["fill"]:
            chain_ok = False; cwhy.append("carried the wrong keys")
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

        # WDDM does not fail an allocation past the dedicated budget - it moves
        # part of the process to system RAM and keeps going, so the row says ok
        # while every token that touches the moved bytes crosses PCIe. The old
        # detector could not see it: its floor test only catches a NONSENSICAL
        # floor, and a partial demotion leaves a plausible one.
        from .bench import (demoted, spill_note, _infer_demotion,
                            SHARED_SPILL_MIB, FLOOR_DROP_MIB)
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
        # ...unless suspect_reason() flagged it, which never used the counter
        sus = [srow(31, 238.0, 3.25, suspect="floor_mib is negative")]
        _infer_demotion(sus)
        meas_ok = meas_ok and sus[0]["spilled"]

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
        print("  TMPL  template splits experiments by content, --jinja leads it  %s"
              % ("OK" if (ti_ok and cmp_ok and arg_ok) else "FAIL"))
        find_ok = (chain_ok and split_ok and alone_ok and clean_ok and par_ok
                   and dep_ok and warn_ok and gate_ok
                   and copy_ok and metric_ok and id_ok and vc_ok
                   and tmpl_ok and scheme_ok and port_ok
                   and meas_ok and inf_ok
                   and ti_ok and cmp_ok and arg_ok)
    except Exception as e:
        find_ok = False
        print("  CHAIN/FIND raised %s: %s  FAIL" % (type(e).__name__, e))
    ok = ok and find_ok

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
