"""Synthetic GGUF writer and the self-test suite."""
import os, struct
from .const import _mib
from .gguf import GGML_TYPES, _parse_one, load_gguf
from .model import RE_EXPS, extract_config
from .compute import CB_DEFAULTS, CB_SPLIT_GRAPH_MIB, CB_SPLIT_PER_TOKEN, compute_buffer_split, compute_buffer_terms, graph_is_split, output_head_on_gpu
from .lmstudio import REF_GPU, read_lmstudio_runtime, resolve_runtime_ngl
from .calib import CALIB_TERMS, _CALIB_CACHE, _active_gpu, _design, _struct_offset, calib_coeffs, fit_calibration, mark_unreliable
from .plan import analyze


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


def self_test():
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

    # 7) the logits term of the compute buffer follows the output head, not the GPU
    rc = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=60, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=None, safety_pct=0,
                 gpu_layers_override=2)
    rf = analyze(p2, 4096, "f16", 512, False, vram_budget_mib=99999, ram_budget_mib=8000,
                 gpu_reserve_mib=0, compute_override_mib=None, safety_pct=0)
    lg, gr = rc["sizes_mib"]["compute_logits"], rc["sizes_mib"]["compute_graph"]
    # partial offload: graph scratch on both sides plus the split surcharge on the
    # GPU, logits with the CPU-side head. Full offload: everything on the GPU, no
    # surcharge, nothing left in RAM.
    split_x = CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * 4096)
    cb_ok = (abs(rc["plan"]["compute_mib"] - (gr + split_x)) < 0.05
             and abs(rc["plan"]["cpu_compute_mib"] - (gr + lg)) < 0.05
             and rf["plan"]["cpu_compute_mib"] == 0
             and abs(rf["plan"]["compute_mib"] - (gr + lg)) < 0.05 and lg > 0)
    print("  CMPBUF split ngl=2: gpu=%.1f cpu=%.1f | full offload: gpu=%.1f cpu=%.1f "
          "(graph=%.1f logits=%.1f split=%.1f)  %s"
          % (rc["plan"]["compute_mib"], rc["plan"]["cpu_compute_mib"],
             rf["plan"]["compute_mib"], rf["plan"]["cpu_compute_mib"], gr, lg,
             split_x, "OK" if cb_ok else "FAIL"))
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
    this_gpu = _active_gpu()
    hw_match = (this_gpu == REF_GPU)
    seen, worst, worst_lbl, n = set(), 0.0, "", 0
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
        err = abs(rr["plan"]["vram_used_mib"] - meas) / meas * 100.0
        n += 1; seen.add(key)
        if err > worst:
            worst, worst_lbl = err, "%s ctx %d ngl %d fa %d" % (key, ctx, ngl, fa)
    if not hw_match:
        print("  VRAMFIT skipped (reference data measured on %s; this is %s)"
              % (REF_GPU, this_gpu or "no NVIDIA GPU"))
    elif n:
        vram_ok = worst <= 10.0
        print("  VRAMFIT worst %.1f%% (%s) over %d runs / %d architectures  %s"
              % (worst, worst_lbl, n, len(seen), "OK" if vram_ok else "FAIL"))
        ok = ok and vram_ok
    else:
        print("  VRAMFIT skipped (no reference models present)")

    # 8b) the recurrent (SSM) state must scale with n_seq exactly as llama.cpp
    #     allocates it. Measured by holding ctx/ngl fixed and varying -np, which
    #     leaves weights and KV untouched: q27 -np 1 -> 8 moved 2947.7 -> 3035.7.
    q27p = REF["q27"][0]
    if not hw_match:
        print("  SSMSEQ  skipped (reference data measured on %s)" % REF_GPU)
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

    # 9) the calibration fitter must recover known coefficients from synthetic rows,
    #    and must REFUSE to free terms the data cannot identify (fitting four
    #    coefficients to two points would be worse than shipping the defaults).
    # "ctx" is bytes per context token now, not a fraction of a KV token, so the
    # synthetic truth has to live on that scale or the plausibility guard will
    # (correctly) refuse to free it.
    truth = {"const": 900.0, "ctx": 1100.0, "act": 40.0, "nofa": 5.0}
    def synth(ctx, ub, fa, hid=4096, nh=32, kvtok=8192, ngl=16, n_layers=32,
              n_cpu_moe=0, n_vocab=32000):
        r = {"ctx": ctx, "ub": ub, "fa": fa, "hidden": hid, "n_head": nh,
             "kv_type": "f16", "kv_tok_ctx": kvtok * ctx, "exact_mib": 1000.0,
             "measured_mib": 1000.0, "ngl": ngl, "n_layers": n_layers,
             "n_cpu_moe": n_cpu_moe, "n_vocab": n_vocab}
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
    lean_ok = (one["free"] == ["const"]
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

    # A single measurement must still produce a usable fit. A small negative
    # additive constant is legitimate - the physical pool is const + the
    # structural per-hidden term - and rejecting it stranded real measurements
    # as "calibrated from 0 measurements".
    one_real = [{"ctx": 262144, "ub": 512, "fa": True, "hidden": 2048, "n_head": 16,
                 "kv_tok_ctx": 5368709120.0, "exact_mib": 8965.0, "overhead_mib": 943.6,
                 "measured_mib": 9908.6, "ngl": 40, "n_layers": 40, "n_cpu_moe": 37,
                 "n_vocab": 248320}]
    single = fit_calibration(one_real)
    single_ok = (single is not None and single["free"] == ["const"]
                 and single["coeffs"]["const"] + _struct_offset(one_real[0]) > 0)
    print("  CALFIT single real measurement: fitted=%s const=%.1f  %s"
          % (single is not None, (single or {}).get("coeffs", {}).get("const", 0.0),
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
                                       output_head_on_gpu(32, sngl))["gpu"]
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

    print("\n  SELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
