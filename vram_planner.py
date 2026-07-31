#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRAM Planner - plan how any GGUF model fits on your GPU + RAM.

Zero dependencies (Python 3.8+ standard library only). It:
  * parses the GGUF binary directly to read EXACT per-tensor byte sizes
    (same idea as `npx @huggingface/gguf --show-tensor`, but self-contained),
  * reads live free VRAM (via nvidia-smi) and free RAM,
  * computes weights + KV cache + compute buffer for any context / quant,
  * for DENSE models: how many layers fit on the GPU (-ngl),
  * for MoE models: how many layers' experts to keep on CPU (--n-cpu-moe / -ot),
  * serves a small web UI so you never type these commands by hand.

Run:   python vram_planner.py            (opens a browser)
       python vram_planner.py --port 8100 --no-browser
       python vram_planner.py --self-test   (validates the parser + math)
"""
import os, re, sys, json, math, glob, time, struct, argparse, subprocess, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

__version__ = "1.0.2"

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# GGML tensor type table: id -> (name, block_size_elements, bytes_per_block)
# tensor bytes = ceil(n_elements / block) * bytes_per_block
# ---------------------------------------------------------------------------
GGML_TYPES = {
    0:  ("F32",      1,   4),
    1:  ("F16",      1,   2),
    2:  ("Q4_0",    32,  18),
    3:  ("Q4_1",    32,  20),
    6:  ("Q5_0",    32,  22),
    7:  ("Q5_1",    32,  24),
    8:  ("Q8_0",    32,  34),
    9:  ("Q8_1",    32,  40),
    10: ("Q2_K",   256,  84),
    11: ("Q3_K",   256, 110),
    12: ("Q4_K",   256, 144),
    13: ("Q5_K",   256, 176),
    14: ("Q6_K",   256, 210),
    15: ("Q8_K",   256, 292),
    16: ("IQ2_XXS",256,  66),
    17: ("IQ2_XS", 256,  74),
    18: ("IQ3_XXS",256,  98),
    19: ("IQ1_S",  256,  50),
    20: ("IQ4_NL",  32,  18),
    21: ("IQ3_S",  256, 110),
    22: ("IQ2_S",  256,  82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8",       1,   1),
    25: ("I16",      1,   2),
    26: ("I32",      1,   4),
    27: ("I64",      1,   8),
    28: ("F64",      1,   8),
    29: ("IQ1_M",  256,  56),
    30: ("BF16",     1,   2),
    31: ("TQ1_0",  256,  54),
    32: ("TQ2_0",  256,  66),
    39: ("MXFP4",   32,  17),   # gpt-oss; best-effort, validated against file size
}

# KV cache bytes per element for each cache quantization
KV_TYPE_BYTES = {
    "f16":  2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}

# Sliding-window interleave stride, for architectures where llama.cpp hardcodes
# set_swa_pattern(p) instead of writing a pattern into the GGUF. Only consulted
# when the file gives a window size but no pattern; p = 1 means "every layer is
# windowed", which is also the fallback for any arch not listed here.
SWA_STRIDE_BY_ARCH = {
    "gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4,
    "gpt-oss": 2, "llama4": 4, "exaone4": 4, "hunyuan-moe": 2,
}

# llama.cpp pads the KV cache; 256 with flash attention, 32 without
KV_PAD_FA, KV_PAD_NOFA = 256, 32

# GGUF metadata value types
_SCALAR_FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
_T_STRING, _T_ARRAY = 8, 9

def _r(f, fmt):
    size = struct.calcsize('<' + fmt)
    data = f.read(size)
    if len(data) < size:
        raise EOFError("unexpected end of file while reading header")
    return struct.unpack('<' + fmt, data)[0]

def _read_str(f):
    n = _r(f, 'Q')
    return f.read(n).decode('utf-8', errors='replace')

def _read_value(f, vtype):
    """Read one metadata value. Big arrays are skipped (only their bytes consumed)."""
    if vtype in _SCALAR_FMT:
        return _r(f, _SCALAR_FMT[vtype])
    if vtype == _T_STRING:
        return _read_str(f)
    if vtype == _T_ARRAY:
        elem_type = _r(f, 'I')
        count = _r(f, 'Q')
        if elem_type == _T_STRING:
            for _ in range(count):
                ln = _r(f, 'Q'); f.seek(ln, 1)
            return None
        if elem_type == _T_ARRAY:
            for _ in range(count):
                _read_value(f, _T_ARRAY)
            return None
        fmt = _SCALAR_FMT.get(elem_type)
        if fmt is None:
            return None
        esize = struct.calcsize('<' + fmt)
        if count <= 8192:                      # keep small numeric arrays (e.g. per-layer kv heads)
            raw = f.read(esize * count)
            return list(struct.unpack('<' + fmt * count, raw))
        f.seek(esize * count, 1)               # skip huge arrays (tokenizer vocab, etc.)
        return None
    raise ValueError("unknown GGUF value type %d" % vtype)

SHARD_RE = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)

def _find_shards(path):
    """If path is one shard of a split GGUF, return all sibling shards in order."""
    d = os.path.dirname(path)
    base = os.path.basename(path)
    m = SHARD_RE.match(base)
    if not m:
        return [path]
    prefix, _, total = m.group(1), m.group(2), int(m.group(3))
    shards = []
    for i in range(1, total + 1):
        cand = os.path.join(d, "%s-%05d-of-%05d.gguf" % (prefix, i, total))
        if os.path.exists(cand):
            shards.append(cand)
    return shards or [path]

def _parse_one(path):
    """Parse a single GGUF file: returns (version, metadata_dict, tensor_list)."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError("not a GGUF file (bad magic): %s" % path)
        version = _r(f, 'I')
        tensor_count = _r(f, 'Q')
        kv_count = _r(f, 'Q')
        meta = {}
        for _ in range(kv_count):
            key = _read_str(f)
            vtype = _r(f, 'I')
            meta[key] = _read_value(f, vtype)
        tensors = []
        for _ in range(tensor_count):
            name = _read_str(f)
            n_dims = _r(f, 'I')
            dims = [_r(f, 'Q') for _ in range(n_dims)]
            type_id = _r(f, 'I')
            _offset = _r(f, 'Q')
            n_elem = 1
            for d in dims:
                n_elem *= d
            tname, block, tsize = GGML_TYPES.get(type_id, ("TYPE_%d" % type_id, None, None))
            if block:
                nbytes = (n_elem // block) * tsize
                if n_elem % block:
                    nbytes += tsize
                unknown = False
            else:
                nbytes = 0
                unknown = True
            tensors.append({
                "name": name, "dims": dims, "type_id": type_id, "type_name": tname,
                "n_elements": n_elem, "n_bytes": nbytes, "unknown_type": unknown,
            })
    return version, meta, tensors

def load_gguf(path):
    """Load a model (following shards). Metadata comes from the first shard."""
    shards = _find_shards(path)
    version, meta, tensors = _parse_one(shards[0])
    file_bytes = os.path.getsize(shards[0])
    for s in shards[1:]:
        _, _, extra = _parse_one(s)
        tensors.extend(extra)
        file_bytes += os.path.getsize(s)
    return {"path": path, "shards": shards, "version": version,
            "meta": meta, "tensors": tensors, "file_bytes": file_bytes}

def parse_meta_only(path):
    """Fast: read header + metadata only (skip the tensor table). Used during scan."""
    with open(path, "rb") as f:
        if f.read(4) != b'GGUF':
            raise ValueError("bad magic")
        _r(f, 'I')                       # version
        _r(f, 'Q')                       # tensor_count (ignored)
        kv_count = _r(f, 'Q')
        meta = {}
        for _ in range(kv_count):
            key = _read_str(f)
            meta[key] = _read_value(f, _r(f, 'I'))
    return meta

# ---------------------------------------------------------------------------
# Model config extraction (architecture-agnostic: match by key suffix)
# ---------------------------------------------------------------------------
RE_BLK  = re.compile(r'^blk\.(\d+)\.')
# Routed experts. Deliberately NOT an explicit (gate|up|down) list: architectures
# fuse them under names like ffn_gate_up_exps (Gemma 4 MoE), and missing one means
# its bytes are silently counted as non-offloadable dense weight - which wrecks
# both the active-parameter count and the whole expert-offload plan. Shared
# experts (ffn_*_shexp) run on every token and are correctly excluded: "shexp"
# does not contain "exps".
RE_EXPS = re.compile(r'^blk\.(\d+)\.ffn_[a-z0-9_]*exps\.')
RE_FFN  = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)\.weight')  # dense FFN weights

def _as_int(v, default=None):
    if v is None:
        return default
    if isinstance(v, list):
        return int(v[0]) if v else default
    try:
        return int(v)
    except Exception:
        return default

def extract_config(model):
    meta = model["meta"]
    arch = meta.get("general.architecture", "?")

    def g(suffix, default=None):
        k = "%s%s" % (arch, suffix)
        if k in meta:
            return meta[k]
        for kk, vv in meta.items():          # fall back to any arch prefix
            if kk.endswith(suffix):
                return vv
        return default

    n_layers   = _as_int(g(".block_count"))
    n_head     = _as_int(g(".attention.head_count"))
    n_head_kv  = g(".attention.head_count_kv")           # may be list (per-layer)
    hidden     = _as_int(g(".embedding_length"))
    n_ctx_train= _as_int(g(".context_length"))
    ffn_len    = _as_int(g(".feed_forward_length"))
    key_len    = _as_int(g(".attention.key_length"))
    val_len    = _as_int(g(".attention.value_length"))
    n_expert   = _as_int(g(".expert_count"), 0) or 0
    expert_ffn = _as_int(g(".expert_feed_forward_length"), 0) or 0
    n_exp_used = _as_int(g(".expert_used_count"), 0) or 0
    file_type  = _as_int(meta.get("general.file_type"))
    name       = (meta.get("general.name") or
                  os.path.splitext(os.path.basename(model["path"]))[0])

    head_dim_k = key_len or ((hidden // n_head) if (hidden and n_head) else None)
    head_dim_v = val_len or ((hidden // n_head) if (hidden and n_head) else None)

    # ---- which blocks actually hold a KV cache -------------------------------
    # Hybrid models (Qwen3-Next / Qwen3.5-3.6 "qwen35", Falcon-H, Jamba, Granite-4,
    # Mamba hybrids...) interleave a few real attention layers with many linear /
    # SSM layers. Only the attention layers grow a KV cache with context; the SSM
    # layers keep a fixed-size recurrent state. Read it off the tensor table --
    # that is ground truth and needs no per-arch table.
    attn_layers, ssm_layers, conv_dim, d_conv = set(), set(), 0, 0
    for t in model["tensors"]:
        m = RE_BLK.match(t["name"])
        if not m:
            continue
        li = int(m.group(1)); rest = t["name"][m.end():]
        if rest.startswith(("attn_k.", "attn_v.", "attn_k_b.", "attn_kv_")):
            attn_layers.add(li)
        elif rest.startswith("ssm_"):
            ssm_layers.add(li)
            if rest.startswith("ssm_conv1d.") and len(t["dims"]) >= 2:
                d_conv, conv_dim = t["dims"][0], t["dims"][1]
    # fused-QKV attention blocks have no separate attn_k/attn_v tensor
    for t in model["tensors"]:
        m = RE_BLK.match(t["name"])
        if not m:
            continue
        li = int(m.group(1))
        if t["name"][m.end():].startswith("attn_qkv.") and li not in ssm_layers:
            attn_layers.add(li)
    if not attn_layers and not ssm_layers:          # metadata-only parse / odd naming
        attn_layers = set(range(n_layers or 0))
    attn_layers = sorted(attn_layers)
    ssm_layers  = sorted(ssm_layers)

    # recurrent state per SSM layer per sequence (llama.cpp keeps both in f32):
    #   conv state = (d_conv - 1) * conv_dim      s state = d_state * d_inner
    ssm_d_state = _as_int(g(".ssm.state_size"), 0) or 0
    ssm_d_inner = _as_int(g(".ssm.inner_size"), 0) or 0
    recurrent_bytes = 4.0 * (max(0, d_conv - 1) * conv_dim + ssm_d_state * ssm_d_inner)

    # per-layer kv-head list, zero for layers with no KV cache
    if isinstance(n_head_kv, list) and len(n_head_kv) == (n_layers or len(n_head_kv)):
        kv_heads_per_layer = [int(x) for x in n_head_kv]
    else:
        kvh = _as_int(n_head_kv, n_head)
        kv_heads_per_layer = [kvh] * (n_layers or 0) if kvh else []
    aset = set(attn_layers)
    kv_heads_per_layer = [(h if i in aset else 0)
                          for i, h in enumerate(kv_heads_per_layer)]  # 0 for SSM and MTP

    n_vocab = 0
    for t in model["tensors"]:
        if "token_embd" in t["name"] and len(t["dims"]) >= 2:
            n_vocab = max(t["dims"])

    # ---- multi-token-prediction (MTP / "nextn") blocks -----------------------
    # DeepSeek-V3, GLM-4.5/4.6 and Qwen3.5 ship extra blocks that predict further
    # tokens ahead. They sit at the END of the block list, are counted in
    # block_count, and look exactly like ordinary transformer blocks in the tensor
    # table - same attn_q/k/v, same FFN - plus a few nextn.* tensors.
    #
    # Measured on Qwen3.5-9B-MTP (5 configurations, ctx 8k-131k): their weights
    # ARE loaded and resident, but they do NOT grow a KV cache, because they do
    # not run during ordinary decoding. Counting their KV inflated the cache by
    # 1/9 here. So: keep the weight, drop the cache.
    n_mtp = _as_int(g(".nextn_predict_layers"), 0) or 0
    if n_mtp <= 0 and any(".nextn." in t["name"] for t in model["tensors"]):
        seen = set()
        for t in model["tensors"]:
            if ".nextn." in t["name"]:
                mm = RE_BLK.match(t["name"])
                if mm:
                    seen.add(int(mm.group(1)))
        n_mtp = len(seen)
    n_mtp = max(0, min(n_mtp, (n_layers or 0)))
    mtp_layers = list(range((n_layers or 0) - n_mtp, n_layers or 0)) if n_mtp else []
    # Kept from before the heads are zeroed: with MTP speculative decoding enabled
    # (llama.cpp --spec-type draft-mtp, LM Studio's "Speculative Decoding: MTP")
    # these blocks DO run and DO grow a cache. Measured f16 regardless of
    # --cache-type-k/v, so the draft cache ignores the KV quant setting.
    mtp_kv_per_token = 0.0
    for i in mtp_layers:
        if i < len(kv_heads_per_layer):
            mtp_kv_per_token += kv_heads_per_layer[i] * ((head_dim_k or 0) + (head_dim_v or 0)) * 2.0
    if mtp_layers:
        mset = set(mtp_layers)
        attn_layers = [i for i in attn_layers if i not in mset]
        ssm_layers  = [i for i in ssm_layers if i not in mset]
        # kv_heads_per_layer was built above from the pre-MTP attention set, so it
        # has to be zeroed here too or the cache is still sized for these blocks
        for i in mtp_layers:
            if i < len(kv_heads_per_layer):
                kv_heads_per_layer[i] = 0

    # ---- sliding-window attention (SWA) --------------------------------------
    # Gemma 2/3/4, Mistral, Phi-3, Cohere2, gpt-oss, Llama-4 and friends cap most
    # layers to a fixed attention window. llama.cpp then builds TWO caches
    # (llama_kv_cache_unified_iswa): a full n_ctx one for the global layers and a
    # small ring buffer for the windowed ones. Those layers also often use smaller
    # head dims. Treating every layer as full-context overstates KV by 10-20x at
    # long context, so read every signal the file might carry.
    n_swa = _as_int(g(".attention.sliding_window"), 0) or 0
    swa_pattern = g(".attention.sliding_window_pattern")
    layer_types = g(".attention.layer_types")
    full_interval = _as_int(g(".full_attention_interval"), 0) or 0
    head_dim_k_swa = _as_int(g(".attention.key_length_swa"), 0) or head_dim_k
    head_dim_v_swa = _as_int(g(".attention.value_length_swa"), 0) or head_dim_v

    swa_layers, swa_source = set(), ""
    if n_swa > 0 and n_layers:
        rng = range(n_layers)
        if isinstance(swa_pattern, list) and swa_pattern:
            # explicit per-layer flag: 1 = windowed, 0 = full attention (Gemma 4)
            swa_layers = {i for i, x in enumerate(swa_pattern[:n_layers]) if _as_int(x, 0)}
            swa_source = "sliding_window_pattern (per-layer)"
        elif isinstance(layer_types, list) and layer_types:
            swa_layers = {i for i, x in enumerate(layer_types[:n_layers])
                          if "slid" in str(x).lower() or "local" in str(x).lower()}
            swa_source = "layer_types"
        else:
            # a stride: llama.cpp's set_swa_pattern(p) makes the LAST block of every
            # group of p the global one, the other p-1 windowed.
            p = _as_int(swa_pattern, 0) or 0
            src = "sliding_window_pattern (stride)"
            if p <= 0 and full_interval > 1 and not ssm_layers:
                # on hybrid SSM models this key means something else entirely
                p, src = full_interval, "full_attention_interval"
            if p <= 0:
                p, src = SWA_STRIDE_BY_ARCH.get(arch, 1), "arch default (%s)" % arch
            if p > 1:
                swa_layers, swa_source = {i for i in rng if (i + 1) % p != 0}, src
            else:
                # a window with no pattern at all = every layer is windowed
                swa_layers, swa_source = set(rng), "sliding_window (uniform)"
    swa_layers &= set(attn_layers)

    # per-layer (head_dim_k, head_dim_v, is_swa) - precomputed so the KV math and
    # the layer-split search stay O(1) per layer instead of re-deriving this.
    kv_layer_dims = [([head_dim_k_swa, head_dim_v_swa, 1] if i in swa_layers
                      else [head_dim_k, head_dim_v, 0]) for i in range(n_layers or 0)]

    return {
        "arch": arch, "name": name, "n_layers": n_layers, "n_head": n_head,
        "n_head_kv": _as_int(n_head_kv, n_head), "kv_heads_per_layer": kv_heads_per_layer,
        "hidden": hidden, "n_ctx_train": n_ctx_train, "ffn_len": ffn_len,
        "head_dim_k": head_dim_k, "head_dim_v": head_dim_v,
        "head_dim_k_swa": head_dim_k_swa, "head_dim_v_swa": head_dim_v_swa,
        "n_expert": n_expert, "n_expert_used": n_exp_used, "file_type": file_type,
        "expert_ffn_len": expert_ffn,
        "n_vocab": n_vocab,
        "attn_layers": attn_layers, "ssm_layers": ssm_layers,
        "is_hybrid": bool(ssm_layers) and bool(attn_layers),
        "recurrent_bytes_per_layer": recurrent_bytes,
        "full_attention_interval": _as_int(g(".full_attention_interval"), 0) or 0,
        "n_swa": n_swa, "swa_layers": sorted(swa_layers), "swa_source": swa_source,
        "n_mtp_layers": n_mtp, "mtp_layers": mtp_layers,
        "mtp_kv_per_token": mtp_kv_per_token,
        "kv_layer_dims": kv_layer_dims,
        "kv_ctx_per_layer": [],          # filled in by resolve_kv_lengths()
    }

def classify_tensors(model, cfg):
    """Sum tensor bytes per layer, split expert vs non-expert, find embed/output."""
    n_layers = cfg["n_layers"] or 0
    per_layer_total   = {}
    per_layer_expert  = {}
    per_layer_ffn     = {}           # dense FFN weights per layer (for KV-on-GPU offload)
    embed_bytes = 0
    output_bytes = 0                 # output.weight + output_norm + any other global tensor
    expert_bytes_total = 0
    expert_elems_total = 0
    ffn_dense_total = 0
    params_total = 0
    weights_total = 0
    unknown_types = set()
    quant_hist = {}

    for t in model["tensors"]:
        nb = t["n_bytes"]; ne = t["n_elements"]; nm = t["name"]
        weights_total += nb
        params_total  += ne
        quant_hist[t["type_name"]] = quant_hist.get(t["type_name"], 0) + nb
        if t["unknown_type"]:
            unknown_types.add(t["type_name"])
        m = RE_BLK.match(nm)
        if m:
            li = int(m.group(1))
            per_layer_total[li] = per_layer_total.get(li, 0) + nb
            if RE_EXPS.match(nm):
                per_layer_expert[li] = per_layer_expert.get(li, 0) + nb
                expert_bytes_total += nb
                expert_elems_total += ne
            elif RE_FFN.match(nm):
                per_layer_ffn[li] = per_layer_ffn.get(li, 0) + nb
                ffn_dense_total += nb
        else:
            if "token_embd" in nm:
                embed_bytes += nb
            else:
                output_bytes += nb

    is_moe = cfg["n_expert"] > 0 or expert_bytes_total > 0
    layer_totals = list(per_layer_total.values())
    per_layer_max  = max(layer_totals) if layer_totals else 0
    per_layer_mean = (sum(layer_totals) / len(layer_totals)) if layer_totals else 0

    expert_layers = sorted(per_layer_expert.keys())
    exp_vals = [per_layer_expert[i] for i in expert_layers]
    expert_layer_mean = (sum(exp_vals) / len(exp_vals)) if exp_vals else 0
    expert_layer_max  = max(exp_vals) if exp_vals else 0

    ffn_layers = sorted(per_layer_ffn.keys())
    ffn_vals = [per_layer_ffn[i] for i in ffn_layers]
    ffn_layer_mean = (sum(ffn_vals) / len(ffn_vals)) if ffn_vals else 0

    # active params for MoE (only n_expert_used of n_expert run per token)
    active_params = params_total
    if is_moe and cfg["n_expert"]:
        active_params = params_total - expert_elems_total + expert_elems_total * (
            cfg["n_expert_used"] / cfg["n_expert"])

    return {
        "per_layer_bytes": per_layer_total, "per_layer_expert_bytes": per_layer_expert,
        "per_layer_ffn_bytes": per_layer_ffn,
        "weights_total": weights_total, "params_total": params_total,
        "active_params": active_params, "embed_bytes": embed_bytes,
        "output_bytes": output_bytes, "expert_bytes_total": expert_bytes_total,
        "is_moe": is_moe, "n_expert_layers": len(expert_layers),
        "per_layer_max": per_layer_max, "per_layer_mean": per_layer_mean,
        "expert_layer_mean": expert_layer_mean, "expert_layer_max": expert_layer_max,
        "ffn_dense_total": ffn_dense_total, "n_ffn_layers": len(ffn_layers),
        "ffn_layer_mean": ffn_layer_mean,
        "unknown_types": sorted(unknown_types), "quant_hist": quant_hist,
    }

# ---------------------------------------------------------------------------
# Memory math
# ---------------------------------------------------------------------------
def kv_bytes_per_token_layer(cfg, kv_type, li):
    """KV-cache bytes for ONE token in ONE block (K + V). Zero for SSM/linear
    blocks, which carry a fixed recurrent state instead. Sliding-window blocks
    frequently use smaller head dims than the global ones (Gemma 4: 256 vs 512),
    so the dims are per-layer, not global."""
    b = KV_TYPE_BYTES.get(kv_type, 2.0)
    heads = cfg["kv_heads_per_layer"]
    dims = cfg.get("kv_layer_dims") or []
    if not heads or li >= len(heads):
        return 0.0
    if li < len(dims):
        hdk, hdv = dims[li][0] or 0, dims[li][1] or 0
    else:
        hdk, hdv = cfg["head_dim_k"] or 0, cfg["head_dim_v"] or 0
    if not (hdk and hdv):
        return 0.0
    return heads[li] * (hdk + hdv) * b

def is_swa_layer(cfg, li):
    dims = cfg.get("kv_layer_dims") or []
    return bool(li < len(dims) and dims[li][2])

def swa_cache_len(cfg, ctx, n_ubatch=512, n_seq=1, flash_attn=True):
    """Tokens held by the sliding-window cache, mirroring
    llama_kv_cache_unified_iswa: pad(n_swa * n_seq_max + n_ubatch), capped at the
    full context. Note this is CONSTANT in context length - past the window,
    raising ctx costs nothing on these layers."""
    n_swa = cfg.get("n_swa") or 0
    if n_swa <= 0:
        return ctx
    pad = KV_PAD_FA if flash_attn else KV_PAD_NOFA
    size = n_swa * max(1, n_seq) + max(1, n_ubatch)
    size = -(-size // pad) * pad
    return min(ctx, size)

def kv_cache_len_layer(cfg, li, ctx, n_ubatch=512, n_seq=1, flash_attn=True):
    """Tokens cached by ONE block: the whole context for a global block, only the
    window for a sliding-window block."""
    if is_swa_layer(cfg, li):
        return swa_cache_len(cfg, ctx, n_ubatch, n_seq, flash_attn)
    return ctx

def resolve_kv_lengths(cfg, ctx, n_ubatch=512, n_seq=1, flash_attn=True):
    """Bake the per-layer cache lengths into cfg so every downstream consumer
    (layer-split search, breakdown, speed model) agrees on them."""
    cfg["kv_ctx_per_layer"] = [
        kv_cache_len_layer(cfg, i, ctx, n_ubatch, n_seq, flash_attn)
        for i in range(cfg["n_layers"] or 0)]
    return cfg["kv_ctx_per_layer"]

def kv_bytes_layer(cfg, kv_type, li, ctx=None):
    """Total KV bytes held by ONE block, using the lengths resolved above."""
    lens = cfg.get("kv_ctx_per_layer") or []
    n = lens[li] if li < len(lens) else (ctx or 0)
    return kv_bytes_per_token_layer(cfg, kv_type, li) * n

def kv_bytes_total(cfg, kv_type, layers=None):
    """Total KV bytes across the given blocks (all of them by default)."""
    if layers is None:
        layers = range(cfg["n_layers"] or 0)
    return sum(kv_bytes_layer(cfg, kv_type, i) for i in layers)

def kv_bytes_total_at(cfg, kv_type, ctx, n_ubatch=512, n_seq=1, flash_attn=True, layers=None):
    """Total KV bytes at an arbitrary context, without disturbing cfg."""
    if layers is None:
        layers = range(cfg["n_layers"] or 0)
    return sum(kv_bytes_per_token_layer(cfg, kv_type, i) *
               kv_cache_len_layer(cfg, i, ctx, n_ubatch, n_seq, flash_attn)
               for i in layers)

def max_ctx_for_kv_budget(cfg, kv_type, budget_mib, n_ubatch=512, n_seq=1,
                          flash_attn=True, layers=None):
    """Largest context whose KV cache fits in budget_mib. With sliding-window
    layers KV is piecewise-linear in ctx (flat below the window, then only the
    global layers grow), so this bisects instead of dividing by a per-token cost."""
    if budget_mib <= 0:
        return 0
    hi = cfg.get("n_ctx_train") or (1 << 22)
    if _mib(kv_bytes_total_at(cfg, kv_type, hi, n_ubatch, n_seq, flash_attn, layers)) <= budget_mib:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _mib(kv_bytes_total_at(cfg, kv_type, mid, n_ubatch, n_seq, flash_attn, layers)) <= budget_mib:
            lo = mid
        else:
            hi = mid - 1
    return lo

def kv_bytes_per_token(cfg, kv_type):
    """Total KV-cache bytes added by ONE token across all layers (K + V). On a
    sliding-window model this is the *marginal* cost only while the context is
    shorter than the window; past it, the windowed layers stop growing."""
    heads = cfg["kv_heads_per_layer"]
    return sum(kv_bytes_per_token_layer(cfg, kv_type, i) for i in range(len(heads)))

def kv_bytes_per_token_growing(cfg, kv_type):
    """Bytes per token for the layers that keep growing past the window."""
    return sum(kv_bytes_per_token_layer(cfg, kv_type, i)
               for i in range(len(cfg["kv_heads_per_layer"])) if not is_swa_layer(cfg, i))

def recurrent_bytes(cfg, layers, n_seq):
    """Fixed-size conv + SSM state for the given recurrent blocks. Independent of
    context length; llama.cpp allocates one copy per sequence slot."""
    ssm = set(cfg.get("ssm_layers") or [])
    n = sum(1 for li in layers if li in ssm)
    return n * cfg.get("recurrent_bytes_per_layer", 0.0) * max(1, n_seq)

# Empirical coefficients for the GPU-side runtime overhead (compute buffer +
# backend workspace). These are MEASURED, not derived: see the note in
# compute_buffer_terms for why the textbook formula does not describe reality.
# Fitted jointly over 24 measured llama.cpp loads spanning four architectures
# (gemma4 dense + gemma4 MoE + qwen35 hybrid-SSM + qwen35moe), CUDA / llama.cpp
# 2.27.1. Mean error 4.8%, worst 12.1%. Fitting on a single model instead gave
# 14% mean / 32% worst on the other three - the shape held, the constants did not.
CB_ACT_PER_HIDDEN  = 56.12    # bytes per ubatch token per unit of hidden size
CB_CTX_FRAC        = 0.1164   # per ctx token, as a share of one token of f16 KV
CB_NOFA_HEAD_BYTES = 3.36     # f32 score matrix, per head per ctx token per ubatch token
# The fixed pool is not actually fixed: it scales with the residual-stream width.
# Measured directly with minimal probes (ctx 2048, ubatch 64, -ngl 1), where the
# scaling terms nearly vanish, it ran 342 MiB (hidden 2048) to 991 MiB (hidden
# 5120) - a 3x range that no single constant can cover. The per-machine part is
# the additive base, which is what calibration fits.
CB_CONST_PER_KHID  = 149.18   # MiB per 1000 units of hidden size

# MTP speculative decoding (llama.cpp --spec-type draft-mtp). Measured on
# Qwen3.5-9B-MTP at ctx 8k/32k/131k and --parallel 1/2/4: the draft cache costs
# one MTP block of f16 KV over the full context, plus a fixed pool and a
# per-sequence slot. Linear in both to within ~9%.
MTP_SPEC_CONST_MIB   = 104.0
MTP_SPEC_PER_SEQ_MIB = 100.0
# The constant covers CUDA context + driver + the fixed graph pool together: they
# cannot be separated from outside the process, and fitting them jointly is what
# the measurements support. So the GPU reserve field defaults to 0 - reserving on
# top of this would double-count. (One caveat: at -ngl 0, where no graph runs on
# the GPU at all, a bare CUDA context still measured ~714 MiB. That degenerate
# case is not modelled; it is also not a case anyone plans for.)
CB_GRAPH_CONST_MIB = 84.5     # additive base: CUDA context + driver, per machine

# These four are the ONLY numbers in the tool that are fitted rather than read
# from the model file, and they are the only ones that depend on the machine
# rather than the model. They ship as a prior and are refined per GPU from the
# user's own Measure runs - see fit_calibration().
CB_DEFAULTS = {"const": CB_GRAPH_CONST_MIB, "act": CB_ACT_PER_HIDDEN,
               "ctx": CB_CTX_FRAC, "nofa": CB_NOFA_HEAD_BYTES}
_CALIB_CACHE = {}                     # (gpu, backend) key -> fitted coefficient dict
_CALIB_LOADED = False                 # the store is read once, lazily

def _ensure_calibration():
    """Load the store on first use. Without this, calibration only ever applied
    inside serve(), so importing the module and calling analyze() silently fell
    back to the shipped defaults."""
    global _CALIB_LOADED
    if not _CALIB_LOADED:
        _CALIB_LOADED = True          # set first: a failed load must not retry forever
        try:
            refresh_calibration()
        except Exception:
            pass

def calib_coeffs(gpu=None):
    """Coefficients to use right now: the fit for this machine if one exists, else
    the shipped prior. Never a partial dict - unfitted terms keep their default."""
    _ensure_calibration()
    c = dict(CB_DEFAULTS)
    c.update((_CALIB_CACHE.get(gpu if gpu is not None else _active_gpu())
              or {}).get("coeffs") or {})
    return c

def _active_gpu():
    g = gpu_list()
    return (g[0].get("name") or "") if g else ""

def compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq=1):
    """The GPU-side runtime overhead beyond weights and KV, in MiB.

    This is the ONLY fuzzy term; weights and KV are exact. It is worth being
    explicit about how this is derived, because the obvious formula is wrong.

    Measuring llama.cpp (CUDA, 17 configurations, one model) shows the overhead:
      * does NOT depend on n_batch at all (identical to the byte at 512 vs 2048);
      * does NOT depend on how many layers are offloaded (one scratch pool, and
        it is the same size at -ngl 4 and -ngl 8);
      * grows LINEARLY in n_ctx, independent of n_ubatch - so it is NOT the
        [n_kv x n_ubatch] attention mask that the graph structure suggests. A
        regression with a ub*ctx term drives that term negative;
      * grows linearly in n_ubatch, independent of n_ctx;
      * has a large fixed floor (the CUDA context, ~700 MiB on a 12 GB card),
        which is what the GPU reserve field is for.

    So the model is additive and empirical:  const + f(ctx) + g(n_ubatch).
    The coefficients were fitted on one model on one GPU, so treat this as a good
    starting point and not as arithmetic - press Measure to pin it exactly for
    your machine. Weights and KV need no such caveat.

    Returned separately:
      * graph  - the pool above, which lives wherever the layers run.
      * logits - the output tensor [n_outputs x n_vocab], produced by the output
        matmul, so it belongs to whichever backend holds the head - NOT
        automatically to the GPU.
    """
    hidden  = cfg["hidden"] or 4096
    n_head  = cfg["n_head"] or 32
    n_vocab = cfg.get("n_vocab") or 32000
    ub      = max(1, n_ubatch)
    ctx     = max(1, context)

    attn = cfg.get("attn_layers") or []
    if not attn:
        return {"graph": round(calib_coeffs()["const"]
                                + CB_CONST_PER_KHID * (cfg["hidden"] or 4096) / 1000.0, 1),
                "logits": 0.0}

    # ubatch-sized activation scratch, in units of the FFN width
    K = calib_coeffs()
    act = K["act"] * ub * hidden

    # the ctx-linear term, expressed against one token of full-attention KV so it
    # carries across architectures (head count, GQA ratio and cache quant all move
    # it). Sliding-window layers contribute only their window, as they do for KV.
    kv_tok_global = sum(kv_bytes_per_token_layer(cfg, "f16", i)
                        for i in attn if not is_swa_layer(cfg, i))
    kv_tok_swa    = sum(kv_bytes_per_token_layer(cfg, "f16", i)
                        for i in attn if is_swa_layer(cfg, i))
    swa_len = swa_cache_len(cfg, ctx, ub, n_seq, flash_attn)
    ctx_term = K["ctx"] * (kv_tok_global * ctx + kv_tok_swa * swa_len)

    # without flash attention the full f32 score matrix is materialised, and it is
    # the one term that really is [n_kv x n_ubatch x n_head]
    scores = 0.0
    if not flash_attn:
        scores = K["nofa"] * n_head * ub * ctx
        # ^ this term is why flash attention is not optional at long context:
        #   at 64k ctx it alone is ~3.5 GiB on a 32-head model.

    graph = (K["const"] + CB_CONST_PER_KHID * hidden / 1000.0
             + _mib(act + ctx_term + scores))
    return {"graph": round(graph, 1), "logits": round(_mib(4.0 * ub * n_vocab), 1)}

def compute_buffer_split(terms, any_on_gpu, any_on_cpu, output_on_gpu, override_mib=None):
    """Split the compute buffer across backends. llama.cpp allocates a scratch
    pool per backend that runs part of the graph, so a partial offload pays the
    graph term on BOTH sides, while the logits tensor lands on exactly one - the
    one holding the output head. At -ngl 4 of 60 the head stays on the CPU, so
    charging its (often huge) logits buffer to VRAM overstates the GPU footprint.

    An override is a measured VRAM number, so it replaces the GPU side outright."""
    gpu = (terms["graph"] if any_on_gpu else 0.0) + (terms["logits"] if output_on_gpu else 0.0)
    cpu = (terms["graph"] if any_on_cpu else 0.0) + (0.0 if output_on_gpu else terms["logits"])
    if override_mib:
        gpu = float(override_mib)
    return {"gpu": round(gpu, 1), "cpu": round(cpu, 1)}

def compute_buffer_mib(cfg, context, n_ubatch, flash_attn, n_seq=1):
    """Total compute buffer across all backends (back-compat / whole-model view)."""
    t = compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq)
    return round(t["graph"] + t["logits"], 1)

def _mib(x):
    return x / MiB

# ---------------------------------------------------------------------------
# Speed: a memory-bandwidth roofline
# ---------------------------------------------------------------------------
# Token generation is bandwidth bound, not compute bound: every token streams
# each *active* weight exactly once. So
#     seconds/token = gpu_bytes/BW_vram + cpu_bytes/BW_ram
# The byte counts come from the tensor table and are exact. The bandwidths are
# not - real efficiency depends on access pattern, and scattered MoE expert
# gathers over CPU RAM are far below peak. That is why this reports a bracket
# until you calibrate it against one measured tok/s.

GPU_EFF   = 0.85       # GPU streams weights contiguously; close to peak
RAM_EFF_HI = 0.65      # contiguous-ish CPU reads
RAM_EFF_LO = 0.25      # scattered expert gather + CPU matmul limits

def per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head=True,
                    n_cpu_moe=0, n_cpu_ffn=0):
    """Bytes read per generated token, split by where they live.

    n_cpu_moe mirrors llama.cpp: the routed experts of the first N blocks sit in
    system RAM even though the rest of those blocks is on the GPU, so a block can
    contribute to both sides. n_cpu_ffn does the same for dense FFN weights, which
    is what the KV-on-GPU mode (-ot ffn=CPU) does. Unlike experts, a dense FFN runs
    on EVERY token, so those bytes are not discounted."""
    per = cl.get("per_layer_bytes") or {}
    exps = cl.get("per_layer_expert_bytes") or {}
    n_exp, n_used = cfg["n_expert"], cfg["n_expert_used"]
    frac = (n_used / n_exp) if (n_exp and n_used) else 1.0
    ffns = cl.get("per_layer_ffn_bytes") or {}
    on = set(gpu_blocks)
    # MTP blocks are resident but do not run during ordinary decoding, so their
    # weights are never streamed per token. Counting them would understate tok/s.
    mtp = set(cfg.get("mtp_layers") or [])
    gpu = cpu = 0.0
    for i in range(cfg["n_layers"] or 0):
        if i in mtp:
            continue
        dense = per.get(i, 0) - exps.get(i, 0)
        act_exp = exps.get(i, 0) * frac          # only n_used experts fire
        if i in on:
            if i < n_cpu_ffn and ffns.get(i):    # dense FFN pinned to CPU by -ot
                gpu += dense - ffns[i]
                cpu += ffns[i]
            else:
                gpu += dense
            if i < n_cpu_moe:
                cpu += act_exp                   # experts pinned to CPU by --n-cpu-moe
            else:
                gpu += act_exp
        else:
            cpu += dense + act_exp
    # token_embd is a row gather, not a full read; the output head is a real matmul
    head = cl.get("output_bytes", 0)
    if cpu_head:
        cpu += head
    else:
        gpu += head
    # KV grows as the context fills; read once per token on its own device. A
    # sliding-window layer only ever reads its window, so its per-token cost stops
    # growing there - that is the difference between a few hundred MiB and tens of
    # GiB per token on a long-context Gemma-class model.
    lens = cfg.get("kv_ctx_per_layer") or []
    fill = max(0, ctx_fill)
    for i in range(cfg["n_layers"] or 0):
        cap = lens[i] if i < len(lens) else fill
        b = kv_bytes_per_token_layer(cfg, kv_type, i) * min(fill, cap)
        if i in on:
            gpu += b
        else:
            cpu += b
    return {"gpu_bytes": gpu, "cpu_bytes": cpu,
            "gpu_mib": _mib(gpu), "cpu_mib": _mib(cpu), "expert_frac": frac}

def estimate_speed(cfg, cl, gpu_blocks, ctx_fill, kv_type,
                   bw_vram_gbs, bw_ram_gbs, cpu_head=True, ram_eff=None,
                   n_cpu_moe=0, n_cpu_ffn=0):
    """tok/s for generation. With ram_eff set (from a calibration) this returns a
    single number; without it, a hi/lo bracket."""
    b = per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head,
                        n_cpu_moe, n_cpu_ffn)
    def tps(re_):
        t = b["gpu_bytes"] / (bw_vram_gbs * GPU_EFF * 1e9)
        if b["cpu_bytes"] > 0:
            t += b["cpu_bytes"] / (bw_ram_gbs * re_ * 1e9)
        return (1.0 / t) if t > 0 else 0.0
    out = dict(b)
    out["bw_vram_gbs"] = bw_vram_gbs
    out["bw_ram_gbs"] = bw_ram_gbs
    out["ctx_fill"] = ctx_fill
    if ram_eff:
        out["tok_s"] = tps(ram_eff)
        out["ram_eff"] = ram_eff
        out["calibrated"] = True
    else:
        out["tok_s_hi"] = tps(RAM_EFF_HI)
        out["tok_s_lo"] = tps(RAM_EFF_LO)
        out["calibrated"] = False
    return out

def solve_ram_eff(cfg, cl, gpu_blocks, ctx_fill, kv_type,
                  bw_vram_gbs, bw_ram_gbs, measured_tok_s, cpu_head=True,
                  n_cpu_moe=0):
    """Back out the effective RAM bandwidth fraction from one measured tok/s.
    The GPU side is assumed to run at GPU_EFF (contiguous streaming), so the
    whole residual lands on the CPU term - which is the term that actually
    varies. Returns None if the measurement leaves no room for a CPU term."""
    if not measured_tok_s or measured_tok_s <= 0:
        return None
    b = per_token_bytes(cfg, cl, gpu_blocks, ctx_fill, kv_type, cpu_head, n_cpu_moe)
    t_total = 1.0 / measured_tok_s
    t_gpu = b["gpu_bytes"] / (bw_vram_gbs * GPU_EFF * 1e9)
    t_cpu = t_total - t_gpu
    if b["cpu_bytes"] <= 0 or t_cpu <= 0:
        return None
    return b["cpu_bytes"] / (t_cpu * bw_ram_gbs * 1e9)

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

    compute_terms = compute_buffer_terms(cfg, ctx, n_ubatch, flash_attn, n_seq)
    # planning value: assume the whole graph runs on the GPU. The plan builders
    # re-split it once they know where the layers and the output head landed.
    compute_mib = compute_override_mib if compute_override_mib else \
        round(compute_terms["graph"] + compute_terms["logits"], 1)

    def compute_fn(ngl, output_on_gpu=None, any_on_cpu=None):
        """GPU/CPU compute-buffer split for a candidate layer count."""
        ngl = max(0, min(int(ngl), n_layers))
        if output_on_gpu is None:
            output_on_gpu = (ngl >= n_layers)
        if any_on_cpu is None:
            any_on_cpu = (ngl < n_layers)
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
        t = compute_buffer_terms(cfg, c, n_ubatch, flash_attn, n_seq)
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
        cb = (compute_fn(ngl, any_on_cpu=(ngl < n_layers or n_cpu_moe > 0))
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

# ---------------------------------------------------------------------------
# Live system info + model discovery
# ---------------------------------------------------------------------------
_GPU_CACHE = {"when": 0.0, "val": None}
GPU_CACHE_TTL = 5.0     # seconds

def get_gpus(fresh=False):
    """Live GPU list. Memoised briefly because this shells out to nvidia-smi and
    sits on a hot path - calib_coeffs() -> _active_gpu() -> here runs several
    times per analyze(), and an 8s timeout each would be painful if the driver
    stalls. Pass fresh=True for the UI's refresh button."""
    if not fresh and _GPU_CACHE["val"] is not None and \
            (time.time() - _GPU_CACHE["when"]) < GPU_CACHE_TTL:
        return _GPU_CACHE["val"]
    val = _query_gpus()
    _GPU_CACHE["when"], _GPU_CACHE["val"] = time.time(), val
    return val

def gpu_list(fresh=False):
    """get_gpus() but always a list - it returns an {"error": ...} dict on failure."""
    g = get_gpus(fresh)
    return g if isinstance(g, list) else []

def platform_support():
    """Where this tool's numbers have actually been validated.

    Weights, KV cache, recurrent state and the projector are read from the GGUF
    and are exact on any platform. The compute-buffer coefficients are not: they
    were fitted against CUDA on Windows, and Metal / ROCm allocate their graphs
    differently. Rather than quietly report confident wrong totals there, say so."""
    gpus = gpu_list()
    vendor = "nvidia" if gpus else ""
    if sys.platform == "darwin":
        return {"supported": False, "gpu_vendor": "apple",
                "reason": "macOS / Metal is not validated. Weights and KV cache are still "
                          "exact, but the compute-buffer estimate was fitted on CUDA and "
                          "will be wrong here, and Measure cannot read GPU memory per "
                          "process. Treat the total as indicative until you calibrate."}
    if not gpus:
        return {"supported": False, "gpu_vendor": "",
                "reason": "No NVIDIA GPU found via nvidia-smi. Weights and KV cache are "
                          "still exact and you can type a VRAM budget manually, but the "
                          "compute-buffer estimate is CUDA-fitted and unvalidated on "
                          "other backends, and Measure is unavailable."}
    if os.name != "nt" and not sys.platform.startswith("linux"):
        return {"supported": False, "gpu_vendor": vendor,
                "reason": "Untested platform. The math is platform-independent but the "
                          "compute-buffer coefficients were fitted on Windows + CUDA."}
    return {"supported": True, "gpu_vendor": vendor, "reason": ""}

def _query_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8).decode("utf-8", "replace")
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 4:
                gpus.append({"name": p[0], "total_mib": float(p[1]),
                             "used_mib": float(p[2]), "free_mib": float(p[3])})
        return gpus
    except Exception as e:
        return {"error": str(e)}

def _lmstudio_home():
    for p in (os.environ.get("LMSTUDIO_HOME"),
              os.path.join(os.path.expanduser("~"), ".lmstudio"),
              os.path.join(os.path.expanduser("~"), ".cache", "lm-studio")):
        if p and os.path.isdir(p):
            return p
    return None

def scan_speed_history(limit=400):
    """Mine LM Studio's saved chats for real throughput measurements.

    Each generation records genInfo.stats (tokensPerSecond, numGpuLayers,
    timeToFirstTokenSec, prompt/predicted token counts) alongside the
    loadModelConfig it ran under - enough to match a measurement to a plan and
    calibrate against it."""
    home = _lmstudio_home()
    if not home:
        return {"error": "LM Studio directory not found"}
    conv = os.path.join(home, "conversations")
    if not os.path.isdir(conv):
        return {"error": "no conversations directory at %s" % conv}

    def cfgval(cfgobj, key, default=None):
        for f in (cfgobj or {}).get("fields", []):
            if f.get("key") == key:
                return f.get("value")
        return default

    recs = []
    try:
        files = sorted(glob.glob(os.path.join(conv, "*.conversation.json")),
                       key=os.path.getmtime, reverse=True)
    except Exception as e:
        return {"error": str(e)}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        when = os.path.getmtime(path)
        for msg in d.get("messages", []) or []:
            for ver in msg.get("versions", []) or []:
                for step in ver.get("steps", []) or []:
                    gi = step.get("genInfo") or {}
                    st = gi.get("stats") or {}
                    tps = st.get("tokensPerSecond")
                    if not tps:
                        continue
                    lc = gi.get("loadModelConfig") or {}
                    recs.append({
                        "model": gi.get("indexedModelIdentifier") or "?",
                        "tok_s": tps,
                        "n_gpu_layers": st.get("numGpuLayers"),
                        "ttft_s": st.get("timeToFirstTokenSec"),
                        "total_time_s": st.get("totalTimeSec"),
                        "prompt_tokens": st.get("promptTokensCount") or 0,
                        "predicted_tokens": st.get("predictedTokensCount") or 0,
                        "ctx": cfgval(lc, "llm.load.contextLength"),
                        "n_experts": cfgval(lc, "llm.load.numExperts"),
                        "cpu_expert_ratio": cfgval(lc, "llm.load.numCpuExpertLayersRatio"),
                        "offload_ratio": cfgval(lc, "llm.load.llama.acceleration.offloadRatio"),
                        "flash_attn": cfgval(lc, "llm.load.llama.flashAttention"),
                        "when": when,
                        "conversation": os.path.basename(path),
                    })
                    if len(recs) >= limit:
                        return recs
    return recs

RE_SRVLOG = re.compile(
    r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)\]\[INFO\]\[([^\]]+)\]\s+(.*)$")

def scan_server_logs(limit=300):
    """Server mode (OpenAI-compatible endpoint) never touches the chat history, so
    conversations/*.json stays empty for anything driven by an API client.

    The succinct server log still brackets each response:
        'Streaming response...'              -> t0
        'Prompt processing progress: 100.0%' -> prefill done
        'Finished streaming response'        -> t1
    That gives real prefill and decode wall times. It does NOT give token counts
    (LM Studio only logs those with verbose file logging on), so it cannot yield
    tok/s by itself - use it to see where the time goes, and Benchmark to measure."""
    home = _lmstudio_home()
    if not home:
        return {"error": "LM Studio directory not found"}
    root = os.path.join(home, "server-logs")
    if not os.path.isdir(root):
        return {"error": "no server-logs directory"}
    files = sorted(glob.glob(os.path.join(root, "*", "*.log")),
                   key=os.path.getmtime, reverse=True)[:14]
    import datetime
    out = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except Exception:
            continue
        cur = None
        for ln in lines:
            m = RE_SRVLOG.match(ln.strip())
            if not m:
                continue
            day, tm, model, msg = m.groups()
            try:
                ts = datetime.datetime.strptime(day + " " + tm, "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                continue
            if msg.startswith("Streaming response"):
                cur = {"model": model, "t0": ts, "t_prefill": None}
            elif cur and msg.startswith("Prompt processing progress: 100.0%"):
                cur["t_prefill"] = ts
            elif cur and msg.startswith("Finished streaming response"):
                pre = (cur["t_prefill"] - cur["t0"]) if cur["t_prefill"] else 0.0
                out.append({"model": cur["model"], "when": cur["t0"],
                            "prefill_s": round(pre, 1),
                            "decode_s": round(ts - (cur["t_prefill"] or cur["t0"]), 1),
                            "total_s": round(ts - cur["t0"], 1)})
                cur = None
                if len(out) >= limit:
                    break
    out.sort(key=lambda x: -x["when"])
    return out

def benchmark_server(base_url="http://localhost:1234", model=None, max_tokens=128,
                     timeout=180):
    """Measure the loaded model by asking it to generate. LM Studio's native
    /api/v0 returns a stats block with tokens_per_second and time_to_first_token;
    fall back to /v1 plus our own clock if that endpoint is not there.

    This runs a real generation on the loaded model, so it briefly uses the GPU."""
    import urllib.request, time
    def post(path, payload):
        req = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    if not model:
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/api/v0/models")
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            for m in d.get("data", []):
                if m.get("state") and m["state"] != "not-loaded":
                    model = m.get("id")
                    break
        except Exception as e:
            return {"error": "cannot reach %s: %s" % (base_url, e)}
    if not model:
        return {"error": "no model is loaded on %s" % base_url}

    body = {"model": model, "temperature": 0, "max_tokens": int(max_tokens),
            "stream": False,
            "messages": [{"role": "user",
                          "content": "Count from 1 to 100, one number per line."}]}
    t0 = time.time()
    try:
        d = post("/api/v0/chat/completions", body)
    except Exception:
        try:
            d = post("/v1/chat/completions", body)
        except Exception as e:
            return {"error": "benchmark request failed: %s" % e}
    wall = time.time() - t0
    st = d.get("stats") or {}
    usage = d.get("usage") or {}
    pred = st.get("predicted_tokens_count") or usage.get("completion_tokens") or 0
    prompt = st.get("prompt_tokens_count") or usage.get("prompt_tokens") or 0
    tps = st.get("tokens_per_second")
    ttft = st.get("time_to_first_token")
    if not tps and pred and wall > 0:
        # no stats block: approximate, subtracting prefill if we know it
        tps = pred / max(1e-6, wall - (ttft or 0))
    return {"model": model, "tok_s": tps, "ttft_s": ttft,
            "predicted_tokens": pred, "prompt_tokens": prompt,
            "wall_s": round(wall, 2), "source": "stats" if st else "wall-clock",
            "when": time.time()}

def _bench_store():
    return _user_file("speed_history.json")

def save_benchmark(rec):
    """Persist benchmarks next to the script so they survive restarts - server-mode
    users have no chat history to mine, so this becomes their history."""
    path = _bench_store()
    try:
        cur = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    except Exception:
        cur = []
    cur.insert(0, rec)
    try:
        json.dump(cur[:200], open(path, "w", encoding="utf-8"), indent=1)
    except Exception:
        pass
    return cur[:200]

def load_benchmarks():
    path = _bench_store()
    try:
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    except Exception:
        return []

def _norm_model(s):
    s = re.sub(r"\.gguf$", "", (s or ""), flags=re.I)
    s = re.sub(r"[-_.\s]+", "", s.lower())
    return s

def match_speed_history(recs, model_name, model_path=None):
    """Score history records against the model being planned. Exact-ish name
    match first; anything sharing a long substring counts as a weak match."""
    if not isinstance(recs, list):
        return []
    keys = [_norm_model(model_name)]
    if model_path:
        keys.append(_norm_model(os.path.basename(model_path)))
    out = []
    for r in recs:
        rid = _norm_model(r.get("model"))
        tail = rid.split("/")[-1] if "/" in rid else rid
        score = 0
        for k in keys:
            if not k:
                continue
            if k == rid or k == tail:
                score = max(score, 3)
            elif k in rid or rid in k or tail in k or k in tail:
                score = max(score, 2)
            else:
                # longest shared prefix of >=10 chars is still a decent signal
                n = 0
                while n < min(len(k), len(tail)) and k[n] == tail[n]:
                    n += 1
                if n >= 10:
                    score = max(score, 1)
        if score:
            r = dict(r); r["match"] = score
            out.append(r)
    out.sort(key=lambda x: (-x["match"], -(x.get("when") or 0)))
    return out

def get_bandwidth():
    """Best-effort peak memory bandwidth for this machine. Both numbers are
    editable in the UI - they are starting points, not gospel."""
    res = {"vram_gbs": None, "ram_gbs": None, "notes": []}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,clocks.max.memory,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8).decode("utf-8", "replace")
        p = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        name, mclk, mtot = p[0], float(p[1]), float(p[2])
        # nvidia-smi reports the memory clock; GDDR6/6X/7 transfer at 2x that.
        gbps = mclk * 2 / 1000.0
        gb = mtot / 1024.0
        bus = 384 if gb > 20 else 256 if gb > 13 else 192 if gb > 10 else 128
        res["vram_gbs"] = round(gbps * bus / 8.0, 0)
        res["notes"].append("%s: %.1f Gbps x %d-bit (bus width inferred from %.0f GB)"
                            % (name, gbps, bus, gb))
    except Exception:
        res["notes"].append("GPU bandwidth unavailable - enter it manually")
    if os.name == "nt":
        ps = (r"$m=Get-CimInstance Win32_PhysicalMemory;"
              r"'{0}|{1}|{2}' -f ($m|Measure-Object -Property Speed -Maximum).Maximum,"
              r"(($m|Measure-Object -Property DataWidth -Sum).Sum),$m.Count")
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                stderr=subprocess.DEVNULL, timeout=20).decode("utf-8", "replace")
            sp, width, cnt = out.strip().split("|")
            res["ram_gbs"] = round(float(sp) * float(width) / 8.0 / 1000.0, 1)
            res["notes"].append("RAM: %s MT/s x %s-bit total (%s modules)" % (sp, width, cnt))
        except Exception:
            res["notes"].append("RAM bandwidth unavailable - enter it manually")
    return res

def get_gpu_processes():
    """Per-process dedicated VRAM. This is what turns the compute buffer from an
    estimate into a measurement: load a model, read the real number, subtract the
    exact terms (weights + KV + recurrent + projector) and the remainder IS the
    compute buffer plus CUDA context.

    nvidia-smi cannot do this on Windows (WDDM hides per-process usage), so fall
    back to the GPU performance counters there."""
    procs = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8).decode("utf-8", "replace")
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3 and p[2].replace(".", "").isdigit():
                procs.append({"pid": p[0], "name": os.path.basename(p[1]),
                              "mib": float(p[2])})
    except Exception:
        pass
    if not procs and os.name == "nt":
        ps = (r"$c=Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -EA Stop;"
              r"$c.CounterSamples|?{$_.CookedValue -gt 8MB}|%{"
              r"$i=($_.InstanceName -split '_')[1];"
              r"$n=(Get-Process -Id $i -EA SilentlyContinue).ProcessName;"
              r"'{0}|{1}|{2}' -f $i,$n,[math]::Round($_.CookedValue/1MB,1)}")
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                stderr=subprocess.DEVNULL, timeout=25).decode("utf-8", "replace")
            for line in out.strip().splitlines():
                p = line.strip().split("|")
                if len(p) == 3 and p[2]:
                    procs.append({"pid": p[0], "name": p[1], "mib": float(p[2])})
        except Exception as e:
            return {"error": str(e)}
    procs.sort(key=lambda x: -x["mib"])
    # the inference process, if we can spot it
    for p in procs:
        n = (p["name"] or "").lower()
        p["is_engine"] = any(k in n for k in
                             ("llama-server", "llama_server", "llama-cpp", "ollama",
                              "koboldcpp", "llamacpp", "lms"))
    return procs

# "max" is a valid value here and means every block. Matching only digits made a
# max-offload load invisible, and the scan then silently reported the PREVIOUS
# load's layer count - the exact stale-config error the mismatch guard exists to
# prevent, so it has to be parsed, not skipped.
RE_LMS_NGL   = re.compile(r"Num Offload Layers:\s*(\d+|max)")
RE_LMS_NCM   = re.compile(r"Num CPU Expert Layers:\s*(\d+)")
RE_LMS_EST   = re.compile(r"raw num offload layers '(\d+|max)' and context length '(\d+)'")

def read_lmstudio_runtime():
    """What LM Studio ACTUALLY loaded, from %APPDATA%/LM Studio/logs/main.log.

    This matters because LM Studio silently overrides the GPU Offload slider to
    respect its own VRAM cap, so the number in the UI is not always the number in
    the engine. Measuring a running process against a plan built for a different
    layer count silently mis-attributes the difference to the compute buffer -
    which is exactly the way to get a wrong override. Returns
    {n_gpu_layers, n_cpu_moe, context} for the most recent load, or None."""
    p = os.path.join(os.environ.get("APPDATA", ""), "LM Studio", "logs", "main.log")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-4000:]
    except Exception:
        return None
    out, ctx = None, None
    for i, ln in enumerate(lines):
        m = RE_LMS_EST.search(ln)
        if m:
            ctx = int(m.group(2))
        if "Resolved GPU config options" in ln:
            blk = "".join(lines[i:i + 8])
            ngl = RE_LMS_NGL.search(blk)
            ncm = RE_LMS_NCM.search(blk)
            if ngl:
                mx = (ngl.group(1) == "max")
                out = {"n_gpu_layers": None if mx else int(ngl.group(1)),
                       "all_layers": mx,
                       "n_cpu_moe": int(ncm.group(1)) if ncm else 0,
                       "context": ctx}
    return out

def resolve_runtime_ngl(rt, n_layers):
    """LM Studio's layer count as a number, resolving 'max' against this model."""
    if not rt:
        return None
    return n_layers if rt.get("all_layers") else rt.get("n_gpu_layers")

# ---------------------------------------------------------------------------
# Per-machine calibration of the compute-buffer coefficients
# ---------------------------------------------------------------------------
# Weights and KV are read from the GGUF and need no calibration. The four
# overhead coefficients do: they vary with GPU, driver and llama.cpp build. Each
# Measure yields one training row (measured process VRAM minus the exact terms),
# so the tool refits from the user's own runs instead of asking anyone to
# understand the constants.
#
# The fit is PROGRESSIVE: it only frees as many coefficients as the data can
# actually identify, and leaves the rest at the shipped prior. One measurement
# moves the constant (the dominant per-machine term); several spread over context
# lengths also move the ctx slope; varied ubatch or model width moves the
# activation slope; a flash-attention-off run moves the score term. Fitting four
# coefficients to two points would be worse than not fitting at all.
CALIB_TERMS = ["const", "ctx", "act", "nofa"]   # freed in this order
CALIB_MIN_SPREAD = 0.25                          # min relative range to free a term

# The VRAMFIT / SSMSEQ self-tests replay real llama-server measurements. They are
# assertions about THIS hardware, so they only run on the machine they were taken
# on - otherwise a perfectly good build fails its own test suite on a stranger's GPU.
REF_GPU     = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
REF_BACKEND = "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1"

RE_BACKEND = re.compile(r"(llama\.cpp-[A-Za-z0-9_.+-]+)")

def detect_backend():
    """Which llama.cpp build is serving, e.g.
    'llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1'.

    llama.cpp changes how it allocates the compute graph between releases, so a
    backend upgrade quietly invalidates a calibration fitted on the old one. The
    build is not in any log, so read it from the engine process's own image path.
    Returns "" when it cannot be determined."""
    try:
        procs = get_gpu_processes()
        if isinstance(procs, dict):
            return ""
        eng = next((p for p in procs if p.get("is_engine")), None)
        if not eng:
            return ""
        pid = str(eng.get("pid") or "")
        path = ""
        if os.name == "nt":
            ps = ("(Get-CimInstance Win32_Process -Filter \"ProcessId=%s\")"
                  ".ExecutablePath" % pid)
            path = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                stderr=subprocess.DEVNULL, timeout=20).decode("utf-8", "replace")
        else:
            path = os.path.realpath("/proc/%s/exe" % pid)
        m = RE_BACKEND.search(path.replace("\\", "/").replace("/", " "))
        return m.group(1) if m else ""
    except Exception:
        return ""

_BACKEND_CACHE = {"val": None}

def current_backend(fresh=False):
    """Memoised detect_backend(). Detection costs a subprocess and needs the
    engine running, so it is resolved at most once per process unless a new
    measurement forces a re-check."""
    if fresh or _BACKEND_CACHE["val"] is None:
        _BACKEND_CACHE["val"] = detect_backend()
    return _BACKEND_CACHE["val"]

def _data_dir():
    """Where per-user state lives. Not next to the script: that breaks on a
    read-only or shared install, and it is how machine-identifying data ends up
    one `git add .` away from being published."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "vram-planner")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return os.path.dirname(os.path.abspath(__file__))
    return d

def _user_file(name):
    """Path in the data dir, migrating a pre-1.0 copy from beside the script."""
    new = os.path.join(_data_dir(), name)
    old = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.exists(new) and os.path.exists(old):
        try:
            os.replace(old, new)
        except Exception:
            return old
    return new

def _calib_store():
    return _user_file("vram_calibration.json")

def load_calibration():
    try:
        return json.load(open(_calib_store(), encoding="utf-8"))
    except Exception:
        return {"rows": []}

def save_calibration(data):
    try:
        json.dump(data, open(_calib_store(), "w", encoding="utf-8"), indent=1)
    except Exception:
        pass

def _design(row):
    """Regressors for one observation, in MiB, matching compute_buffer_terms."""
    return {"const": 1.0,
            "ctx":   _mib(row["kv_tok_ctx"]),
            "act":   _mib(row["hidden"] * row["ub"]),
            "nofa":  0.0 if row["fa"] else _mib(row["n_head"] * row["ub"] * row["ctx"])}

def _struct_offset(row):
    """Part of the graph pool that is structural (scales with hidden size) rather
    than per-machine. It is not fitted, so it comes off the observation before the
    additive base is solved for."""
    return CB_CONST_PER_KHID * row["hidden"] / 1000.0

def _solve(A, y):
    """Least squares by normal equations. Returns None if ill-conditioned."""
    n = len(A[0])
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)]
         + [sum(A[k][i] * y[k] for k in range(len(A)))] for i in range(n)]
    scale = [max(abs(M[i][i]), 1e-12) for i in range(n)]
    for i in range(n):
        pv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pv][i]) < 1e-9 * scale[i]:
            return None
        M[i], M[pv] = M[pv], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]

def fit_calibration(rows, prior=None):
    """Fit what the data supports. Returns {coeffs, free, n, residual_pct} or None."""
    prior = dict(prior or CB_DEFAULTS)
    rows = [r for r in rows if r.get("overhead_mib") is not None and r.get("hidden")]
    if len(rows) < 1:
        return None
    des = [_design(r) for r in rows]

    # a term is identifiable only if its regressor actually varies across the rows
    # A term is identifiable only if the knob it belongs to actually varied - and
    # the knob, not the regressor. The activation column is hidden*ubatch, so a
    # handful of different models measured at one ubatch makes it look varied when
    # all that changed was hidden size, which is confounded with the per-hidden
    # structural constant. Fitting it from that produced a coefficient 9x the
    # prior. So require spread in the knob a user can actually turn.
    knob = {"ctx": lambda r: float(r["ctx"]),
            "act": lambda r: float(r["ub"]),
            "nofa": lambda r: 0.0 if r["fa"] else 1.0}
    free = ["const"]
    for t in CALIB_TERMS[1:]:
        vals = [knob[t](r) for r in rows]
        lo, hi = min(vals), max(vals)
        if hi > 0 and (hi - lo) / hi >= CALIB_MIN_SPREAD:
            free.append(t)
    # never free more terms than we have rows to support (plus a degree of freedom)
    while len(free) > 1 and len(rows) < len(free) + 1:
        free.pop()

    coeffs = dict(prior)
    while free:
        fixed = [t for t in CALIB_TERMS if t not in free]
        A = [[d[t] for t in free] for d in des]
        y = [r["overhead_mib"] - _struct_offset(r) - sum(prior[t] * d[t] for t in fixed)
             for r, d in zip(rows, des)]
        x = _solve(A, y)
        # A fit is only accepted if it is finite and physically possible: these
        # terms are byte counts, so a negative one means the solver absorbed error
        # rather than measured anything. Slopes are also held within an order of
        # magnitude of the prior - beyond that the data is being extrapolated, and
        # the shipped default is the better answer.
        good = x is not None and all(v == v and abs(v) < 1e9 for v in x)
        if good:
            cand = dict(zip(free, x))
            for t, val in cand.items():
                if t == "const":
                    # An additive offset, not a byte count on its own: the physical
                    # pool is const + the structural per-hidden part, so a small
                    # negative const is a legitimate fit that trims a slightly
                    # over-predicting prior. Only reject nonsense magnitudes, and
                    # check the total stays positive on the rows we actually have.
                    if abs(val) > 20000 or any(val + _struct_offset(r) < 0 for r in rows):
                        good = False
                elif val < 0 or (prior[t] > 0 and not (0.1 <= val / prior[t] <= 10.0)):
                    good = False
        if good:
            coeffs = dict(prior)
            coeffs.update(cand)
            break
        free.pop()                       # drop the least-identifiable term and retry
    if not free:
        return None

    # in-sample residual: honest label, it is a fit quality not a prediction error
    errs = []
    for r, d in zip(rows, des):
        pred = _struct_offset(r) + sum(coeffs[t] * d[t] for t in CALIB_TERMS)
        base = max(1.0, r["overhead_mib"] + r.get("exact_mib", 0.0))
        errs.append(abs(pred - r["overhead_mib"]) / base * 100.0)
    return {"coeffs": coeffs, "free": free, "n": len(rows),
            "residual_pct": round(sum(errs) / len(errs), 1)}

def refresh_calibration(gpu=None):
    """Refit every GPU present in the store and publish the results.

    Rows are also filtered by llama.cpp build: a backend upgrade changes graph
    allocation, so mixing builds fits a curve through two different machines. If
    the current build is known and any row matches it, only those rows are used;
    otherwise everything for that GPU is used and the mismatch is reported."""
    data = load_calibration()
    cur = current_backend()
    by_gpu = {}
    for r in data.get("rows", []):
        by_gpu.setdefault(r.get("gpu") or "", []).append(r)
    _CALIB_CACHE.clear()
    for g, rows in by_gpu.items():
        # Only drop a row when its build is KNOWN and different. Rows recorded
        # before builds were tracked carry no backend; discarding those would
        # throw away good measurements from this very machine.
        used  = [r for r in rows if not cur or not r.get("backend") or r["backend"] == cur]
        stale = len(rows) - len(used)
        f = fit_calibration(used)
        if f:
            f["stale_rows"] = stale
            f["backend"] = cur
            _CALIB_CACHE[g] = f
    return _CALIB_CACHE.get(gpu if gpu is not None else _active_gpu())

def calibration_status(gpu=None):
    _ensure_calibration()
    g = gpu if gpu is not None else _active_gpu()
    f = _CALIB_CACHE.get(g)
    if not f:
        return {"calibrated": False, "gpu": g, "n": 0, "coeffs": dict(CB_DEFAULTS),
                "free": [], "residual_pct": None, "stale_rows": 0,
                "backend": current_backend()}
    return {"calibrated": True, "gpu": g, "n": f["n"], "coeffs": f["coeffs"],
            "free": f["free"], "residual_pct": f["residual_pct"],
            "stale_rows": f.get("stale_rows", 0), "backend": f.get("backend", "")}

def record_calibration(path, ctx, kv_type, n_ubatch, n_seq, flash_attn, ngl,
                       include_mmproj, measured_mib, gpu="", n_cpu_moe=0):
    """Turn one measurement into a training row and refit. The exact terms are
    recomputed here rather than trusted from the client, so a stale plan on screen
    cannot poison the store."""
    r = analyze(path, ctx, kv_type, n_ubatch, flash_attn,
                vram_budget_mib=1 << 20, ram_budget_mib=1 << 20, gpu_reserve_mib=0,
                compute_override_mib=0.001, safety_pct=0, n_seq=n_seq,
                gpu_layers_override=ngl, include_mmproj=include_mmproj,
                n_cpu_moe_override=(n_cpu_moe or None))
    cfg, p = r["config"], r["plan"]
    exact = (p.get("gpu_weights_mib", 0.0) + p.get("gpu_kv_mib", 0.0)
             + p.get("gpu_recurrent_mib", 0.0) + p.get("mmproj_mib", 0.0))
    overhead = measured_mib - exact
    if overhead <= 0:
        return {"error": "Measured %.0f MiB is below the exact terms (%.0f MiB) - the "
                         "loaded model is not the one analysed." % (measured_mib, exact)}
    attn = cfg["attn_layers"]
    kvg = sum(kv_bytes_per_token_layer(cfg, "f16", i) for i in attn if not is_swa_layer(cfg, i))
    kvs = sum(kv_bytes_per_token_layer(cfg, "f16", i) for i in attn if is_swa_layer(cfg, i))
    row = {"when": int(time.time()), "gpu": gpu or _active_gpu(),
           "backend": current_backend(fresh=True),
           "model": os.path.basename(path), "arch": cfg["arch"],
           "ctx": ctx, "ub": n_ubatch, "n_seq": n_seq, "fa": bool(flash_attn),
           "ngl": ngl, "n_cpu_moe": n_cpu_moe, "kv_type": kv_type,
           "hidden": cfg["hidden"] or 4096, "n_head": cfg["n_head"] or 32,
           "kv_tok_ctx": kvg * ctx + kvs * swa_cache_len(cfg, ctx, n_ubatch, n_seq, flash_attn),
           "exact_mib": round(exact, 1), "measured_mib": round(measured_mib, 1),
           "overhead_mib": round(overhead, 1)}
    data = load_calibration()
    rows = data.setdefault("rows", [])
    # one row per distinct config; a re-measure replaces the older reading
    key = (row["gpu"], row["model"], ctx, n_ubatch, n_seq, row["fa"], ngl, kv_type)
    rows[:] = [x for x in rows if (x.get("gpu"), x.get("model"), x.get("ctx"), x.get("ub"),
                                   x.get("n_seq"), x.get("fa"), x.get("ngl"),
                                   x.get("kv_type")) != key]
    rows.insert(0, row)
    del rows[400:]
    data["schema"] = 1
    save_calibration(data)
    refresh_calibration()
    return {"ok": True, "row": row, "status": calibration_status(row["gpu"])}

def get_ram():
    try:
        if os.name == "nt":
            import ctypes

            class MSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MSEX(); m.dwLength = ctypes.sizeof(MSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {"total_mib": m.ullTotalPhys / MiB, "free_mib": m.ullAvailPhys / MiB}
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0])   # kB
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return {"total_mib": info.get("MemTotal", 0) / 1024.0, "free_mib": avail / 1024.0}
    except Exception as e:
        return {"error": str(e), "total_mib": 0, "free_mib": 0}

def default_models_dir():
    home = os.path.expanduser("~")
    for c in [os.path.join(home, ".lmstudio", "models"),
              os.path.join(home, ".cache", "lm-studio", "models")]:
        if os.path.isdir(c):
            return c
    return os.path.join(home, ".lmstudio", "models")

def _quick_ctx(path):
    """Native context length from metadata only (fast). Returns int or None."""
    try:
        meta = parse_meta_only(path)
        for k, v in meta.items():
            if k.endswith(".context_length"):
                return _as_int(v)
    except Exception:
        pass
    return None

def scan_models(root):
    out = []
    if not root or not os.path.isdir(root):
        return out
    seen = set()
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".gguf"):
                continue
            full = os.path.join(dp, fn)
            m = SHARD_RE.match(fn)
            if m:
                if m.group(2) != "00001":
                    continue
                key = (dp, m.group(1))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size = sum(os.path.getsize(s) for s in _find_shards(full))
                except OSError:
                    size = os.path.getsize(full)
                out.append({"name": "%s  [split x%s]" % (m.group(1), m.group(3)),
                            "path": full, "size_mib": size / MiB, "n_ctx_train": _quick_ctx(full)})
            else:
                try:
                    out.append({"name": fn, "path": full,
                                "size_mib": os.path.getsize(full) / MiB,
                                "n_ctx_train": _quick_ctx(full)})
                except OSError:
                    pass
    out.sort(key=lambda x: x["name"].lower())
    return out

# ---------------------------------------------------------------------------
# Embedded web UI
# ---------------------------------------------------------------------------
HTML_PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRAM Planner</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121a23; --panel2:#0e141b; --line:#25313d;
  --ink:#e7eef5; --muted:#8ba0b3; --dim:#5b6b7b;
  --ok:#37d5b0; --warn:#f2b64c; --bad:#ff6a6a;
  --wt:#5aa9ff; --kv:#37d5b0; --cmp:#b98bff; --rsv:#3a4756;
  --mono:ui-monospace,"Cascadia Code","JetBrains Mono",Consolas,"DejaVu Sans Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:
   radial-gradient(1200px 600px at 80% -10%, #14202c 0%, transparent 60%),
   var(--bg); color:var(--ink); font-family:var(--sans); font-size:14px; line-height:1.45}
a{color:var(--wt)}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:14px 22px;justify-content:space-between;
   padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:20px}
.brand h1{font-family:var(--mono);font-weight:600;font-size:20px;letter-spacing:.14em;margin:0}
.brand .tag{color:var(--muted);font-size:12px;letter-spacing:.05em;margin-top:3px}
.brand .accent{color:var(--ok)}
.sys{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
.sys .g{min-width:190px}
.sys .lbl{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.08em;
   display:flex;justify-content:space-between;gap:10px}
.sys .lbl b{color:var(--ink);font-weight:600}
.meter{height:7px;border-radius:4px;background:#0c1219;border:1px solid var(--line);
   margin-top:5px;overflow:hidden}
.meter > i{display:block;height:100%;background:linear-gradient(90deg,var(--wt),var(--kv))}
button{font-family:var(--sans);cursor:pointer;border:1px solid var(--line);background:var(--panel2);
   color:var(--ink);border-radius:7px;padding:7px 12px;font-size:13px}
button:hover{border-color:#39485a}
button.ghost{background:transparent;color:var(--muted);padding:5px 9px;font-size:12px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:18px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px}
.card h2{font-family:var(--mono);font-size:12px;letter-spacing:.1em;color:var(--muted);
   text-transform:uppercase;margin:0 0 12px;font-weight:600}
.field{margin-bottom:13px}
.field label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
.field .hint{color:var(--dim);font-size:11px;margin-top:4px}
input[type=text],input[type=number],select{width:100%;background:var(--panel2);border:1px solid var(--line);
   color:var(--ink);border-radius:7px;padding:8px 10px;font-family:var(--mono);font-size:13px}
input:focus,select:focus{outline:none;border-color:var(--wt)}
.row{display:flex;gap:8px}
.row > *{flex:1}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
.chip{font-family:var(--mono);font-size:12px;padding:4px 9px;border:1px solid var(--line);
   border-radius:20px;background:var(--panel2);color:var(--muted);cursor:pointer}
.chip:hover{color:var(--ink)}
.chip.on{border-color:var(--kv);color:var(--kv)}
.check{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink);cursor:pointer}
.check input{width:16px;height:16px;accent-color:var(--kv)}
details.adv{margin-top:6px;border-top:1px dashed var(--line);padding-top:12px}
details.adv summary{cursor:pointer;color:var(--muted);font-size:12px;font-family:var(--mono);
   letter-spacing:.06em;list-style:none;margin-bottom:10px}
details.adv summary::-webkit-details-marker{display:none}
.go{width:100%;background:linear-gradient(90deg,#1c3a4f,#1a4a44);border-color:#2b5a52;
   color:#eafff9;font-weight:600;padding:11px;font-size:14px;letter-spacing:.03em;margin-top:6px}
.go:hover{filter:brightness(1.12)}
.results{display:flex;flex-direction:column;gap:16px;min-width:0}
.placeholder{color:var(--dim);font-size:13px;text-align:center;padding:40px 10px;
   font-family:var(--mono);letter-spacing:.05em}
.badge{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.06em;
   padding:2px 8px;border-radius:5px;border:1px solid var(--line)}
.badge.moe{color:var(--cmp);border-color:#4a3a63}
.badge.dense{color:var(--wt);border-color:#28405a}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px 16px}
.kv{font-family:var(--mono)}
.kv .k{color:var(--muted);font-size:11px;letter-spacing:.04em}
.kv .v{color:var(--ink);font-size:15px;margin-top:2px}
.verdict{border-left:3px solid var(--rsv);padding-left:12px}
.verdict.ok{border-color:var(--ok)} .verdict.warn{border-color:var(--warn)} .verdict.bad{border-color:var(--bad)}
.verdict .h{font-size:15px;font-weight:600}
.verdict.ok .h{color:var(--ok)} .verdict.warn .h{color:var(--warn)} .verdict.bad .h{color:var(--bad)}
.barwrap{margin-top:14px}
.bar-top{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;
   color:var(--muted);margin-bottom:5px}
.bar-top b{color:var(--ink)}
.bar{position:relative;height:26px;border-radius:6px;background:#0a1017;border:1px solid var(--line);
   overflow:hidden;display:flex}
.bar > span{height:100%;transition:width .5s cubic-bezier(.2,.7,.2,1)}
.bar .s-wt{background:var(--wt)} .bar .s-kv{background:var(--kv)}
.bar .s-cmp{background:var(--cmp)} .bar .s-rsv{background:repeating-linear-gradient(45deg,#2c3742,#2c3742 5px,#232d38 5px,#232d38 10px)}
.cap{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.85}
.cap:after{content:attr(data-l);position:absolute;top:-15px;left:50%;transform:translateX(-50%);
   font-family:var(--mono);font-size:9px;color:var(--muted);white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:9px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.legend .n{color:var(--ink)}
.setrow{font-family:var(--mono);font-size:13px;padding:5px 0;border-bottom:1px dashed var(--line);color:var(--ink)}
.setrow:last-child{border-bottom:none}
pre.cmd{background:#080c11;border:1px solid var(--line);border-radius:8px;padding:12px;
   font-family:var(--mono);font-size:12.5px;color:#d6e6f2;overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:0}
.cmdhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.04em}
td:first-child,th:first-child{text-align:left}
.warns{color:var(--warn);font-size:12.5px}
.warns div{padding:4px 0}
.muted{color:var(--muted)} .small{font-size:12px}
.note{color:var(--dim);font-size:11px;margin-top:8px;line-height:1.5}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--dim);border-top-color:var(--ok);
   border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <h1>VRAM<span class="accent">//</span>PLANNER</h1>
      <div class="tag">exact GGUF weights + KV cache &middot; live VRAM/RAM &middot; layer &amp; expert offload
        <span class="muted" id="ver"></span></div>
    </div>
    <div class="sys" id="sys"><span class="muted small">reading hardware...</span></div>
  </header>
  <div id="platwarn"></div>

  <div class="grid">
    <div class="card">
      <h2>Model &amp; run settings</h2>
      <div class="field">
        <label>Models folder</label>
        <div class="row">
          <input type="text" id="dir" placeholder="path to .gguf folder">
          <button class="ghost" style="flex:0 0 auto" onclick="scanModels()">Scan</button>
        </div>
        <div class="hint" id="scanhint"></div>
      </div>
      <div class="field">
        <label>Model</label>
        <select id="model" onchange="onPick()"><option value="">-- scan or paste a path --</option></select>
      </div>
      <div class="field">
        <label>...or full path to a .gguf</label>
        <input type="text" id="path" placeholder="C:\\...\\model.gguf">
      </div>
      <div class="field">
        <label>Context length (tokens)</label>
        <input type="number" id="ctx" value="8192" min="256" step="256">
        <div class="chips" id="ctxchips"></div>
      </div>
      <div class="row">
        <div class="field">
          <label>KV cache quant</label>
          <select id="kv">
            <option value="f16">f16 (2.00 B)</option>
            <option value="q8_0">q8_0 (1.06 B)</option>
            <option value="q5_1">q5_1 (0.75 B)</option>
            <option value="q4_0">q4_0 (0.56 B)</option>
          </select>
        </div>
        <div class="field">
          <label>ubatch = LM Studio "Physical Batch Size"</label>
          <input type="number" id="ubatch" value="512" min="16" step="16">
        </div>
        <div class="field">
          <label>seqs = "Max Concurrent Predictions"</label>
          <input type="number" id="nseq" value="1" min="1" step="1">
        </div>
      </div>
      <div class="hint">Match seqs to LM Studio's "Max Concurrent Predictions" (default 4). It
        sizes the recurrent state on hybrid models and the sliding-window cache on SWA models
        (Gemma, Mistral, gpt-oss) &mdash; llama.cpp allocates one window per sequence slot.</div>
      <div class="field">
        <label class="check"><input type="checkbox" id="fa"> Flash attention (needed for quant KV)</label>
      </div>
      <div class="field" id="mmprojfield" style="display:none">
        <label class="check"><input type="checkbox" id="mmproj" checked> Load vision projector (mmproj)</label>
        <div id="mtprow" style="display:none">
        <label class="check"><input type="checkbox" id="mtpspec" checked> MTP speculative decoding</label>
        <div class="hint">This model ships multi-token-prediction blocks. LM Studio turns this on by
          default for them (<span class="mono">Speculative Decoding: MTP</span>, llama.cpp
          <span class="mono">--spec-type draft-mtp</span>). It gives the draft blocks a KV cache back
          &mdash; at f16 regardless of your KV quant &mdash; plus a per-sequence slot. Untick only if
          you set Speculative Decoding to Off.</div></div>
        <div class="hint" id="mmprojhint"></div>
      </div>
      <div class="field">
        <label class="check"><input type="checkbox" id="kvgpu"> Force ALL layers' KV onto GPU (offload FFN)</label>
        <div class="hint">Advanced, llama.cpp <b>-ot</b> only. Different from LM Studio's "Offload KV
          Cache to GPU" (which only keeps <i>GPU-layer</i> KV in VRAM). This pushes FFN weights to CPU so
          every layer's KV sits in VRAM; dense generation gets slower. Leave OFF to model a normal
          LM Studio GPU-offload split.</div>
      </div>
      <details class="adv">
        <summary>Advanced budgets &amp; overrides</summary>
        <div class="field">
          <label>GPU layers (blank = auto; set to match/verify an LM Studio run)</label>
          <input type="number" id="ngl" step="1" min="0" placeholder="auto">
          <div class="hint">Enter LM Studio's "GPU Offload" number to see the exact VRAM/RAM that
            config uses, then compare against Task Manager. <b>Check the real value in the log</b>
            (<span class="mono">%APPDATA%\LM Studio\logs\main.log</span>, "Resolved GPU config
            options") &mdash; LM Studio silently adjusts the slider to respect its VRAM cap.</div>
        </div>
        <div class="field" id="ncpumoefield" style="display:none">
          <label>CPU expert layers (MoE; blank = auto)</label>
          <input type="number" id="ncpumoe" step="1" min="0" placeholder="auto">
          <div class="hint">LM Studio's "Number of layers to keep experts on CPU" /
            llama.cpp <b>--n-cpu-moe</b>. Different knob from GPU Offload: it moves only the
            routed experts of the first N blocks, leaving their attention and KV on the GPU.</div>
        </div>
        <div class="row">
          <div class="field"><label>VRAM budget (MiB)</label><input type="number" id="vram" step="64"></div>
          <div class="field"><label>RAM budget (MiB)</label><input type="number" id="ram" step="256"></div>
        </div>
        <div class="hint" style="margin:-6px 0 12px">RAM budget defaults to total installed, since a
          model can load into standby/paged memory (LM Studio pushes RAM to ~100%). The header shows
          what's free right now; the plan warns if it exceeds that.</div>
        <div class="row">
          <div class="field"><label>GPU reserve (MiB)</label><input type="number" id="reserve" value="0" step="64">
            <div class="hint">Extra headroom on top of the plan. Defaults to 0 because the CUDA
              context and driver overhead are already inside the compute-buffer estimate (they were
              fitted together against measured loads). Add to this only if you want the plan to
              leave room for something else on the GPU, e.g. your desktop.</div></div>
          <div class="field"><label>Safety %</label><input type="number" id="safety" value="5" step="1" min="0" max="40"></div>
        </div>
        <div class="row">
          <div class="field"><label>VRAM bandwidth (GB/s)</label>
            <input type="number" id="bwv" step="1" placeholder="auto"></div>
          <div class="field"><label>RAM bandwidth (GB/s)</label>
            <input type="number" id="bwr" step="0.1" placeholder="auto"></div>
        </div>
        <div class="hint" id="bwhint" style="margin:-6px 0 12px"></div>
        <div class="row">
          <div class="field"><label>Context filled (tokens)</label>
            <input type="number" id="ctxfill" step="256" placeholder="8192">
            <div class="hint">KV is re-read every token, so speed depends on how full the
              context actually is &mdash; not on the size you allocated.</div></div>
          <div class="field"><label>RAM efficiency (0 = bracket)</label>
            <input type="number" id="rameff" step="0.01" min="0" max="1" placeholder="0">
            <div class="hint">Set by Calibrate. Fraction of peak RAM bandwidth actually reached.</div></div>
        </div>
        <div class="field">
          <label>Compute buffer override (MiB, 0 = estimate)</label>
          <input type="number" id="compute" value="0" step="16">
          <div class="hint">Weights &amp; KV are exact. The compute buffer is estimated. You do not
            need the load log &mdash; if the model is loaded right now, press Measure and the tool reads
            the engine process's real VRAM and back-solves the true overhead.</div>
          <button class="ghost" style="margin-top:8px" onclick="calibrate()">&#9673; Measure running model</button>
          <div class="hint" id="calhint"></div>
        </div>
      </details>
      <button class="go" id="goBtn" onclick="run()">Analyze fit</button>
    </div>

    <div class="results" id="out">
      <div class="card"><div class="placeholder">Pick a model and press Analyze fit.</div></div>
    </div>
  </div>
</div>
<script src="/app.js"></script>
</body>
</html>'''

APP_JS = '''
const $ = id => document.getElementById(id);
let SYS = null, MODELS = [], LAST = null;

function fmt(m){ if(m==null||isNaN(m)) return "-"; return Math.round(m).toLocaleString()+" MiB"; }
function fmtG(m){ if(m==null||isNaN(m)) return "-"; return (m/1024).toFixed(2)+" GiB"; }
function fmtGB(m){ if(m==null||isNaN(m)) return "-"; return (m*1048576/1e9).toFixed(2)+" GB"; }
function B(n){ return (n/1e9).toFixed(2)+" B"; }
function clampPct(x){ return Math.max(0, Math.min(100, x)); }

function renderPlatform(p){
  const el = $("platwarn");
  if(!el) return;
  if(!p || p.supported){ el.innerHTML = ""; return; }
  el.innerHTML = '<div class="card warns" style="margin-bottom:14px">'+
    '<div style="color:var(--warn);font-weight:600;margin-bottom:4px">'+
    '&#9888; Unvalidated platform</div><div>'+escapeHtml(p.reason)+'</div></div>';
}

async function boot(){
  try{
    const s = await (await fetch("/api/system")).json();
    SYS = s;
    if(!$("dir").value) $("dir").value = s.default_dir || "";
    if(s.version) $("ver").textContent = " · v"+s.version;
    renderPlatform(s.platform);
    renderSys(s);
    prefillBudgets(s);
    if($("dir").value) scanModels();
  }catch(e){ $("sys").innerHTML = '<span class="muted small">system read failed: '+e+'</span>'; }
  buildCtxChips();
  loadBandwidth();
}

function prefillBudgets(s){
  let vfree = 0;
  if(Array.isArray(s.gpus) && s.gpus.length) vfree = s.gpus[0].free_mib;
  if(vfree && !$("vram").value) $("vram").value = Math.round(vfree);
  // RAM budget defaults to TOTAL installed (minus a small OS reserve): a model can load
  // into standby/paged memory, so "free right now" understates what will actually load.
  if(s.ram && s.ram.total_mib && !$("ram").value)
    $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
}

function renderSys(s){
  let h = "";
  if(Array.isArray(s.gpus) && s.gpus.length){
    s.gpus.forEach(g=>{
      const usedPct = clampPct(g.used_mib/g.total_mib*100);
      h += '<div class="g"><div class="lbl"><span>'+g.name+'</span>'+
           '<span><b>'+Math.round(g.free_mib).toLocaleString()+'</b> / '+Math.round(g.total_mib).toLocaleString()+' MiB free</span></div>'+
           '<div class="meter"><i style="width:'+usedPct+'%"></i></div></div>';
    });
  } else {
    h += '<div class="g"><div class="lbl"><span>GPU</span><span class="muted">nvidia-smi not found</span></div>'+
         '<div class="meter"></div></div>';
  }
  if(s.ram){
    const usedPct = clampPct((s.ram.total_mib-s.ram.free_mib)/s.ram.total_mib*100);
    h += '<div class="g"><div class="lbl"><span>System RAM</span>'+
         '<span><b>'+Math.round(s.ram.free_mib).toLocaleString()+'</b> / '+Math.round(s.ram.total_mib).toLocaleString()+' MiB free</span></div>'+
         '<div class="meter"><i style="width:'+usedPct+'%"></i></div></div>';
  }
  h += '<button class="ghost" onclick="refreshSys()">&#8635; refresh</button>';
  $("sys").innerHTML = h;
}

async function loadBandwidth(){
  try{
    const b = await (await fetch("/api/bandwidth")).json();
    if(b.vram_gbs && !$("bwv").value) $("bwv").value = b.vram_gbs;
    if(b.ram_gbs  && !$("bwr").value) $("bwr").value = b.ram_gbs;
    $("bwhint").innerHTML = (b.notes||[]).join("<br>");
  }catch(e){ $("bwhint").textContent = "bandwidth auto-detect failed: "+e; }
}

async function refreshSys(){
  const s = await (await fetch("/api/system")).json(); SYS = s;
  renderSys(s);
  if(Array.isArray(s.gpus)&&s.gpus.length) $("vram").value = Math.round(s.gpus[0].free_mib);
  if(s.ram) $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
}

function buildCtxChips(){
  const vals=[2048,4096,8192,16384,32768,65536,131072];
  $("ctxchips").innerHTML = vals.map(v=>'<span class="chip" data-v="'+v+'" onclick="setCtx('+v+')">'+
     (v>=1024?(v/1024)+"k":v)+'</span>').join("");
  markCtx();
}
function setCtx(v){ $("ctx").value=v; markCtx(); }
function markCtx(){
  const cur=parseInt($("ctx").value)||0;
  document.querySelectorAll("#ctxchips .chip").forEach(c=>{
    c.classList.toggle("on", parseInt(c.dataset.v)===cur);
  });
}
$("ctx") && $("ctx").addEventListener("input", markCtx);

async function scanModels(){
  const dir=$("dir").value.trim();
  $("scanhint").textContent="scanning...";
  try{
    const r = await (await fetch("/api/models?dir="+encodeURIComponent(dir))).json();
    MODELS = r.models||[];
    const sel=$("model");
    sel.innerHTML = '<option value="">-- '+MODELS.length+' models found --</option>'+
      MODELS.map((m,i)=>'<option value="'+i+'">'+
        m.name+'  ('+fmtG(m.size_mix||m.size_mib)+')'+
        (m.n_ctx_train?('  &middot; '+(m.n_ctx_train>=1024?(m.n_ctx_train/1024)+'k':m.n_ctx_train)+' ctx'):'')+
        '</option>').join("");
    $("scanhint").textContent = MODELS.length? (MODELS.length+" GGUF found in folder")
                                             : "no .gguf found here";
  }catch(e){ $("scanhint").textContent="scan failed: "+e; }
}
function onPick(){
  const i=$("model").value;
  if(i==="") return;
  const m=MODELS[parseInt(i)];
  if(!m) return;
  $("path").value="";
  if(m.n_ctx_train){ setCtxMax(m.n_ctx_train); }
}
let CTX_MAX = null;
function setCtxMax(nctx){
  CTX_MAX = nctx;
  $("ctx").max = nctx;
  // add/refresh a "max" chip
  let chip=document.getElementById("ctxmax");
  const label = (nctx>=1024?(nctx/1024)+"k":nctx);
  if(!chip){
    chip=document.createElement("span");
    chip.id="ctxmax"; chip.className="chip";
    $("ctxchips").appendChild(chip);
  }
  chip.textContent="max "+label;
  chip.setAttribute("data-v", nctx);
  chip.onclick=()=>setCtx(nctx);
  markCtx();
}

async function run(){
  let path = $("path").value.trim();
  if(!path){
    const i=$("model").value;
    if(i!=="" && MODELS[parseInt(i)]) path = MODELS[parseInt(i)].path;
  }
  if(!path){ alert("Pick a model from the list or paste a .gguf path."); return; }
  const body = {
    path: path,
    context: parseInt($("ctx").value)||8192,
    kv_type: $("kv").value,
    n_ubatch: parseInt($("ubatch").value)||512,
    n_seq: parseInt($("nseq").value)||1,
    include_mmproj: $("mmproj").checked,
    mtp_spec: $("mtpspec").checked,
    n_cpu_moe_override: $("ncpumoe").value===""? null : parseInt($("ncpumoe").value),
    bw_vram_gbs: parseFloat($("bwv").value)||0,
    bw_ram_gbs: parseFloat($("bwr").value)||0,
    ram_eff: parseFloat($("rameff").value)||0,
    ctx_fill: $("ctxfill").value===""? null : parseInt($("ctxfill").value),
    flash_attn: $("fa").checked,
    vram_budget_mib: parseFloat($("vram").value)||0,
    ram_budget_mib: parseFloat($("ram").value)||0,
    gpu_reserve_mib: parseFloat($("reserve").value)||0,
    compute_override_mib: parseFloat($("compute").value)||0,
    safety_pct: parseFloat($("safety").value)||0,
    kv_on_gpu: $("kvgpu").checked,
    gpu_layers_override: ($("ngl").value.trim()===""? null : parseInt($("ngl").value)),
    ram_free_mib: (SYS && SYS.ram ? SYS.ram.free_mib : null),
  };
  const btn=$("goBtn"); btn.disabled=true; const old=btn.textContent;
  btn.innerHTML='<span class="spin"></span>analyzing';
  $("out").innerHTML='<div class="card"><div class="placeholder"><span class="spin"></span>parsing GGUF...</div></div>';
  try{
    const res = await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const r = await res.json();
    if(!r.ok){ $("out").innerHTML='<div class="card warns"><b>Error:</b> '+(r.error||"unknown")+'</div>'; }
    else render(r);
  }catch(e){ $("out").innerHTML='<div class="card warns"><b>Request failed:</b> '+e+'</div>'; }
  btn.disabled=false; btn.textContent=old;
}

function kvBadge(c){ return c.is_moe
  ? '<span class="badge moe">MoE &middot; '+c.n_expert_used+'/'+c.n_expert+' experts</span>'
  : '<span class="badge dense">DENSE</span>'; }

function bar(title, capMib, used, segs){
  const sum = segs.reduce((a,s)=>a+(s.mib||0),0);
  const scale = Math.max(capMib, sum, 1);
  let inner = segs.map(s=>'<span class="'+s.cls+'" style="width:'+clampPct(s.mib/scale*100)+'%" title="'+s.label+': '+fmt(s.mib)+'"></span>').join("");
  const capPos = clampPct(capMib/scale*100);
  const over = sum>capMib;
  const utilTxt = capMib? (used/capMib*100).toFixed(0)+'%' : '-';
  let legend = segs.filter(s=>s.mib>0.5).map(s=>'<span><i style="background:'+s.color+'"></i>'+
      s.label+' <span class="n">'+fmt(s.mib)+'</span></span>').join("");
  return '<div class="barwrap"><div class="bar-top"><span>'+title+' &middot; using <b>'+fmt(used)+'</b> of '+fmt(capMib)+
     '</span><span style="color:'+(over?'var(--bad)':'var(--muted)')+'">'+utilTxt+(over?' OVER':'')+'</span></div>'+
     '<div class="bar">'+inner+'<span class="cap" data-l="'+fmt(capMib)+'" style="left:'+capPos+'%"></span></div>'+
     '<div class="legend">'+legend+'</div></div>';
}

function render(r){
  LAST = r;
  const c=r.config, p=r.plan, s=r.sizes_mib, inp=r.inputs;
  $("mmprojfield").style.display = r.mmproj? "" : "none";
  $("ncpumoefield").style.display = r.is_moe? "" : "none";
  { const row=$("mtprow"); if(row) row.style.display = c.n_mtp_layers ? "" : "none"; }
  if(r.mmproj) $("mmprojhint").innerHTML = r.mmproj.name+" &middot; "+fmt(r.mmproj.mib)+
    " of VRAM. LM Studio loads it with the model and includes it in the size it shows.";
  // verdict class
  let vcls="warn";
  if(p.fits_fully) vcls="ok";
  else if(p.attention_overflow || p.kv_overflow || p.ram_ok===false || p.vram_ok===false) vcls="bad";

  // ---- summary ----
  let sum = '<div class="card"><h2>Model</h2>'+
    '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px">'+
    '<div style="font-family:var(--mono);font-size:15px">'+ (r.model_name||"?") +'</div>'+ kvBadge(r) +'</div>'+
    '<div class="summary-grid">'+
      kvItem("architecture", c.arch)+
      kvItem("params (total)", B(r.params_total))+
      (r.is_moe? kvItem("params (active)", B(r.active_params)) : "")+
      kvItem("quant", (r.bpw).toFixed(2)+" bpw")+
      kvItem("layers", c.n_layers)+
      kvItem("hidden size", c.hidden)+
      kvItem("attn heads", c.n_head + " / kv " + c.n_head_kv)+
      (r.hybrid && r.hybrid.is_hybrid
        ? kvItem("KV-bearing layers", r.hybrid.n_attn_layers + " of " + c.n_layers +
            " (hybrid; " + r.hybrid.n_ssm_layers + " linear/SSM)")
        : "")+
      (r.swa && r.swa.enabled
        ? kvItem("sliding window", r.swa.n_swa.toLocaleString() + " tok &middot; " +
            r.swa.n_swa_layers + " windowed / " + r.swa.n_global_layers + " global")
        : "")+
      (c.n_mtp_layers
        ? kvItem("multi-token pred.", c.n_mtp_layers + " block" + (c.n_mtp_layers==1?"":"s") +
            ' <span class="muted">(' + (inp.mtp_spec ? "drafting: KV counted" : "idle: weights only") + ')</span>')
        : "")+
      kvItem("head dim", (c.head_dim_k||"-") +
        ((r.swa && r.swa.enabled && r.swa.head_dim!==r.swa.head_dim_global)
          ? '  <span class="muted">/ '+r.swa.head_dim+' swa</span>' : ""))+
      kvItem("native ctx", (c.n_ctx_train? c.n_ctx_train.toLocaleString():"-"))+
      kvItem("weights", fmtG(s.weights))+
    '</div></div>';

  // ---- verdict + bars ----
  const vramSegs=[
    {cls:"s-wt",  color:"var(--wt)",  label:"weights (GPU)", mib:p.gpu_weights_mib||0},
    {cls:"s-kv",  color:"var(--kv)",  label:"KV cache (GPU)", mib:p.gpu_kv_mib||0},
    {cls:"s-kv",  color:"#2aa88c",     label:"recurrent state", mib:p.gpu_recurrent_mib||0},
    {cls:"s-wt",  color:"#7d6cff",     label:"vision projector", mib:p.mmproj_mib||0},
    {cls:"s-kv",  color:"#e0a03a",     label:"MTP draft cache", mib:p.spec_mib||0},
    {cls:"s-cmp", color:"var(--cmp)", label:"compute buffer", mib:p.compute_mib||0},
    {cls:"s-rsv", color:"#2c3742",    label:"driver reserve", mib:inp.gpu_reserve_mib||0},
  ];
  const vramUsed=vramSegs.reduce((a,x)=>a+x.mib,0);
  const ramSegs=[
    {cls:"s-wt", color:"var(--wt)", label:"weights (CPU/RAM)", mib:p.cpu_weights_mib||0},
    {cls:"s-kv", color:"var(--kv)", label:"KV cache (RAM)", mib:p.cpu_kv_mib||0},
    {cls:"s-kv", color:"#2aa88c",   label:"recurrent state (RAM)", mib:p.cpu_recurrent_mib||0},
    {cls:"s-cmp",color:"var(--cmp)",label:"compute buffer (CPU)", mib:p.cpu_compute_mib||0},
  ];
  const ramUsed=ramSegs.reduce((a,x)=>a+x.mib,0);

  let verdict='<div class="card"><h2>Verdict</h2><div class="verdict '+vcls+'"><div class="h">'+
    p.headline+'</div></div>'+
    bar("VRAM", inp.vram_budget_mib, vramUsed, vramSegs);
  if(ramUsed>0.5) verdict += bar("System RAM", inp.ram_budget_mib, ramUsed, ramSegs);
  verdict += '<div class="note">Weights and KV cache are computed exactly from the GGUF tensor '+
    'table. '+
    ((r.calibration && r.calibration.calibrated)
      ? 'The compute buffer is <b style="color:var(--kv)">calibrated for your GPU</b> from '+
        r.calibration.n+' measurement'+(r.calibration.n==1?'':'s')+' ('+
        r.calibration.free.join(", ")+' fitted, in-sample '+r.calibration.residual_pct+'%). '
      : 'The compute buffer uses shipped defaults, fitted to measured llama.cpp runs across 4 '+
        'architectures (4.8% mean, 12% worst on one CUDA card). It is the only term that depends '+
        'on your hardware rather than the model &mdash; press <b>Measure running model</b> with a '+
        'model loaded to calibrate it for yours. ')+
    ((p.cpu_compute_mib>0.5)
      ? 'Split across backends here: '+fmt(p.compute_mib)+' on the GPU, '+fmt(p.cpu_compute_mib)+
        ' in RAM. llama.cpp gives every backend running part of the graph its own scratch pool, '+
        'and the '+fmt(s.compute_logits)+' logits tensor ('+inp.n_ubatch+' &times; '+
        (c.n_vocab||0).toLocaleString()+' vocab) belongs to whichever side holds the output head.'
      : 'All of it ('+fmt(p.compute_mib)+') is on the GPU at this split.')+
    '</div></div>';

  // ---- settings ----
  let setrows = (p.lmstudio||[]).map(x=>'<div class="setrow">'+x+'</div>').join("");
  let settings='<div class="card"><h2>Settings to use</h2>'+
    '<div style="margin-bottom:12px"><div class="k muted small" style="margin-bottom:6px;font-family:var(--mono);letter-spacing:.06em">LM STUDIO &middot; advanced load settings</div>'+
    setrows+'</div>'+
    '<div class="cmdhead"><div class="k muted small" style="font-family:var(--mono);letter-spacing:.06em">LLAMA.CPP</div>'+
    '<button class="ghost" onclick="copyCmd(this)">copy</button></div>'+
    '<pre class="cmd" id="cmd">'+escapeHtml(p.llama_cmd||"")+'</pre>';
  if(p.max_ctx_kv_gpu!=null)
    settings += '<div class="note">Max context with <b style="color:var(--ink)">all KV on GPU</b> '+
      '(FFN on CPU): ~<b style="color:var(--ink)">'+p.max_ctx_kv_gpu.toLocaleString()+'</b> tokens.</div>';
  else if(p.max_ctx_gpu!=null)
    settings += '<div class="note">Max context fully on GPU at this quant: ~<b style="color:var(--ink)">'+
      p.max_ctx_gpu.toLocaleString()+'</b> tokens.</div>';
  settings += '</div>';

  // ---- KV table ----
  let rows=(r.kv_table||[]).map(t=>{
    const cur = t.ctx===inp.context;
    return '<tr'+(cur?' style="color:var(--kv)"':'')+'><td>'+t.ctx.toLocaleString()+
      (cur?'  &larr;':'')+'</td><td>'+fmt(t.kv_mib)+'</td><td>'+fmtG(t.kv_mib)+'</td></tr>';
  }).join("");
  let kvcard='<div class="card"><h2>KV cache vs context ('+inp.kv_type+')</h2>'+
    '<table><thead><tr><th>context</th><th>KV size</th><th></th></tr></thead><tbody>'+rows+
    '</tbody></table><div class="note">Per token: '+s.kv_per_token_kib.toFixed(1)+' KiB across '+
    ((r.hybrid&&r.hybrid.is_hybrid)? r.hybrid.n_attn_layers+' KV-bearing of '+c.n_layers+' layers &mdash; '+
      'this is a hybrid model, the other '+r.hybrid.n_ssm_layers+' layers use a fixed recurrent state'
      : 'all '+c.n_layers+' layers')+'.</div>'+
    ((r.swa && r.swa.enabled)
      ? '<div class="note">This model uses <b style="color:var(--ink)">sliding-window attention</b>, so '+
        'the table is not a straight line. '+r.swa.n_swa_layers+' layers only ever cache '+
        r.swa.window_cache_tokens.toLocaleString()+' tokens (a '+r.swa.n_swa.toLocaleString()+
        '-token window &times; '+inp.n_seq+' seq + one ubatch, padded) &mdash; '+fmt(s.kv_swa)+
        ', flat no matter how long the context gets. Only the '+r.swa.n_global_layers+
        ' full-attention layers grow with context ('+fmt(s.kv_global)+' here, '+
        s.kv_grow_per_token_kib.toFixed(1)+' KiB/token). Detected from <code>'+r.swa.source+'</code>.</div>'
      : "")+'</div>';

  // ---- breakdown ----
  let brk='<div class="card"><h2>Memory breakdown</h2><table><tbody>'+
    brow("Model weights (exact)", fmt(s.weights)+"  ("+fmtG(s.weights)+")")+
    brow("&nbsp;&nbsp;token embeddings", fmt(s.embed))+
    brow("&nbsp;&nbsp;output / head", fmt(s.output))+
    (r.is_moe? brow("&nbsp;&nbsp;routed experts (offloadable)", fmt(s.expert_total)) : "")+
    brow("Per-layer weight (avg)", fmt(s.per_layer_mean))+
    brow("KV cache @ "+inp.context.toLocaleString()+" ctx", fmt(s.kv_total)+
      (r.hybrid && r.hybrid.is_hybrid
        ? '  <span class="muted">('+r.hybrid.n_attn_layers+' attn layers)</span>' : ""))+
    ((r.swa && r.swa.enabled)
      ? brow("&nbsp;&nbsp;full-attention layers ("+r.swa.n_global_layers+")", fmt(s.kv_global))+
        brow("&nbsp;&nbsp;sliding-window layers ("+r.swa.n_swa_layers+", fixed)", fmt(s.kv_swa))
      : "")+
    (s.recurrent_total>0.5? brow("Recurrent state (fixed, x"+inp.n_seq+" seq)", fmt(s.recurrent_total)) : "")+
    brow("Compute buffer (est.)", fmt(s.compute))+
    brow("&nbsp;&nbsp;graph scratch + attn mask", fmt(s.compute_graph))+
    brow("&nbsp;&nbsp;logits ("+inp.n_ubatch+" x "+(c.n_vocab||0).toLocaleString()+" vocab)",
      fmt(s.compute_logits)+'  <span class="muted">'+
      ((p.cpu_compute_mib>0.5)?'on CPU &mdash; the output head is not offloaded':'on GPU')+'</span>')+
    (r.mmproj? brow("Vision projector ("+r.mmproj.name+")",
        fmt(r.mmproj.mib)+(r.mmproj.included?"":'  <span class="muted">not loaded</span>')) : "")+
    brow("File on disk", fmtG(s.file_on_disk)+"  ("+fmtGB(s.file_on_disk)+")")+
    (r.mmproj? brow("&nbsp;&nbsp;+ projector = LM Studio's &quot;model size&quot;",
        fmtG(s.bundle_on_disk)+"  ("+fmtGB(s.bundle_on_disk)+")") : "")+
    '</tbody></table></div>';

  let warns="";
  if(r.warnings && r.warnings.length)
    warns='<div class="card warns"><h2 style="color:var(--warn)">Notes</h2>'+
      r.warnings.map(w=>'<div>&#9888; '+w+'</div>').join("")+'</div>';

  // ---- speed ----
  let speed = "";
  const sp = r.speed;
  if(sp && !sp.error){
    const num = sp.calibrated
      ? '<span style="color:var(--kv);font-size:26px;font-weight:600">'+sp.tok_s.toFixed(1)+'</span> tok/s'
      : '<span style="color:var(--ink);font-size:26px;font-weight:600">'+sp.tok_s_lo.toFixed(0)+
        '&ndash;'+sp.tok_s_hi.toFixed(0)+'</span> tok/s';
    speed = '<div class="card"><h2>Generation speed</h2>'+
      '<div style="margin:2px 0 14px">'+num+
      '<span class="muted small" style="margin-left:10px">'+
      (sp.calibrated? 'calibrated &middot; RAM at '+(sp.ram_eff*100).toFixed(0)+'% of peak'
                    : 'uncalibrated bracket &middot; measure once to collapse it')+
      '</span></div>'+
      '<table><tbody>'+
      brow("Read from VRAM per token", fmt(sp.gpu_mib))+
      brow("Read from system RAM per token", fmt(sp.cpu_mib)+
        (sp.cpu_mib>sp.gpu_mib/4? '  <span style="color:var(--warn)">&larr; the bottleneck</span>':""))+
      (sp.expert_frac<1? brow("Experts active per token",
          (sp.expert_frac*100).toFixed(2)+"% of expert weights") : "")+
      brow("Context assumed filled", sp.ctx_fill.toLocaleString()+" tokens")+
      brow("Bandwidth used", Math.round(sp.bw_vram_gbs)+" GB/s VRAM &middot; "+
           sp.bw_ram_gbs.toFixed(1)+" GB/s RAM")+
      '</tbody></table>'+
      '<div class="note">Generation is memory-bandwidth bound: every token reads each active '+
      'weight once. Byte counts are exact; the bandwidths are not, which is the whole width of '+
      'the bracket. Prompt processing is compute bound and is <b>not</b> modelled here.</div>'+
      '<div style="margin-top:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'+
      '<button class="ghost" onclick="benchNow()">&#9654; Benchmark the loaded model</button>'+
      '<span class="muted small">runs one short generation on the LM Studio server '+
      '(localhost:1234) &mdash; do it while your agent is idle</span></div>'+
      '<div id="benchout"></div>'+
      '<div id="spdhist"><div class="muted small">looking for past measurements...</div></div></div>';
  }

  $("out").innerHTML = sum + verdict + settings + speed + kvcard + brk + warns;
  if(sp && !sp.error) loadSpeedHistory(r);
}

async function benchNow(){
  const out = $("benchout");
  out.innerHTML = '<div class="note">generating... this occupies the GPU for a few seconds</div>';
  const r = LAST;
  let q = "?max_tokens=128";
  if(r && r.inputs) q += "&ctx="+r.inputs.context;
  if(r && r.plan && r.plan.n_gpu_layers!=null) q += "&ngl="+r.plan.n_gpu_layers;
  let d;
  try{ d = await (await fetch("/api/benchmark"+q)).json(); }
  catch(e){ out.innerHTML='<div class="note" style="color:var(--warn)">benchmark failed: '+e+'</div>'; return; }
  if(d.error || !d.tok_s){
    out.innerHTML='<div class="note" style="color:var(--warn)">'+(d.error||"no throughput returned")+
      '<br>Is a model loaded and the server running on localhost:1234?</div>'; return; }
  const fill = (d.prompt_tokens||0) + Math.round((d.predicted_tokens||0)/2);
  out.innerHTML = '<div class="note"><b style="color:var(--kv)">'+d.tok_s.toFixed(2)+' tok/s</b> measured on <b>'+
    d.model+'</b> &middot; '+d.predicted_tokens+' tokens'+
    (d.ttft_s!=null? ' &middot; TTFT '+Number(d.ttft_s).toFixed(2)+'s':'')+
    ' &middot; source: '+d.source+
    ' <button class="ghost" style="margin-left:8px" onclick="useMeasured('+d.tok_s+','+fill+')">calibrate from this</button>'+
    '<br>Saved to speed_history.json, so it shows up in the list below from now on.</div>';
  loadSpeedHistory(LAST);
}

async function loadSpeedHistory(r){
  const box = $("spdhist"); if(!box) return;
  let h;
  try{
    h = await (await fetch("/api/speedhistory?model="+encodeURIComponent(r.model_name||"")+
                           "&path="+encodeURIComponent(($("path").value||$("model").value||"")))).json();
  }catch(e){ box.innerHTML='<div class="muted small">history read failed: '+e+'</div>'; return; }
  if(h.error){ box.innerHTML='<div class="muted small">no measurement history: '+h.error+'</div>'; return; }
  const ms = h.matches||[];
  if(!ms.length){
    const n=(h.all||[]).length;
    box.innerHTML='<div class="note">No past measurements for <b>'+(r.model_name||"this model")+
      '</b> in LM Studio’s saved chats'+(n? ' ('+n+' record'+(n==1?'':'s')+' for other models)':'')+
      '. Generate once in LM Studio, then re-run &mdash; the tok/s it reports gets saved with the '+
      'chat and shows up here to calibrate against.</div>';
    return;
  }
  let rows = ms.slice(0,8).map((m,i)=>{
    const fill = (m.prompt_tokens||0) + Math.round((m.predicted_tokens||0)/2);
    return '<tr><td>'+m.tok_s.toFixed(2)+' tok/s</td>'+
      '<td class="muted">ngl '+(m.n_gpu_layers==null?"?":m.n_gpu_layers)+
        ' &middot; ctx '+(m.ctx? m.ctx.toLocaleString():"?")+
        ' &middot; ~'+fill.toLocaleString()+' filled</td>'+
      '<td style="text-align:right"><button class="ghost" onclick="useMeasured('+
        m.tok_s+','+fill+')">calibrate</button></td></tr>';
  }).join("");
  box.innerHTML = '<div style="margin-top:16px"><div class="k muted small" style="font-family:var(--mono);'+
    'letter-spacing:.06em;margin-bottom:6px">MEASURED IN LM STUDIO'+
    (ms[0].match<3? ' (approximate model match — check the config)':'')+'</div>'+
    '<table><tbody>'+rows+'</tbody></table>'+
    '<div class="note">Picking one sets <b>Context filled</b> to that run and solves for the real '+
    'RAM efficiency, then re-runs. Only calibrate against a run whose layer split matches the plan '+
    'above &mdash; otherwise you are fitting the wrong configuration.</div></div>' + srvLogPanel(h);
}

function srvLogPanel(h){
  const sl = h.server_log;
  if(!Array.isArray(sl) || !sl.length) return "";
  const n = Math.min(sl.length, 20), recent = sl.slice(0, n);
  const pre = recent.reduce((a,x)=>a+x.prefill_s,0), dec = recent.reduce((a,x)=>a+x.decode_s,0);
  if(pre+dec <= 0) return "";
  const pct = Math.round(pre/(pre+dec)*100);
  return '<div style="margin-top:16px"><div class="k muted small" style="font-family:var(--mono);'+
    'letter-spacing:.06em;margin-bottom:6px">SERVER MODE &middot; LAST '+n+' RESPONSES</div>'+
    '<table><tbody>'+
    brow("Prompt processing (prefill)", pre.toFixed(0)+" s total &middot; <b>"+pct+"%</b> of the time")+
    brow("Token generation (decode)", dec.toFixed(0)+" s total &middot; "+(100-pct)+"%")+
    '</tbody></table><div class="note">Wall times parsed from '+
    '<span class="mono">~/.lmstudio/server-logs</span>. The succinct log has no token counts, so this '+
    'cannot give tok/s &mdash; but it does show where your time actually goes. '+
    (pct>=50? '<b>Prefill dominates here</b>, and prefill is compute bound, so layer-split tuning '+
              'will not fix it &mdash; a shorter prompt or a reused prefix will.'
            : 'Decode dominates, which is what the bandwidth model above predicts.')+
    '</div></div>';
}

async function useMeasured(tokS, fill){
  $("ctxfill").value = Math.max(0, Math.round(fill));
  $("rameff").value = "";              // clear so the solve is not anchored to an old value
  const r = LAST; if(!r || !r.speed) return;
  // solve: t_total = gpu_bytes/(BWv*GPU_EFF) + cpu_bytes/(BWr*eff)
  const sp = r.speed, GPU_EFF = 0.85;
  const gpuB = sp.gpu_mib*1048576, cpuB = sp.cpu_mib*1048576;
  const tTot = 1/tokS, tGpu = gpuB/(sp.bw_vram_gbs*GPU_EFF*1e9), tCpu = tTot-tGpu;
  if(cpuB<=0 || tCpu<=0){
    $("calhint").innerHTML='<b style="color:var(--warn)">That measurement is faster than the GPU '+
      'term alone allows &mdash; it was a different config (or a different context fill). Not calibrating.</b>';
    return;
  }
  $("rameff").value = (cpuB/(tCpu*sp.bw_ram_gbs*1e9)).toFixed(3);
  run();
}

async function calibrate(){
  const h=$("calhint");
  if(!LAST){ h.innerHTML='<b style="color:var(--warn)">Press Analyze fit first, so the exact terms are known.</b>'; return; }
  h.textContent="reading GPU processes...";
  let r;
  try{ r = await (await fetch("/api/gpuprocs")).json(); }
  catch(e){ h.textContent="could not read GPU processes: "+e; return; }
  const procs = r.procs;
  if(!Array.isArray(procs) || !procs.length){
    h.innerHTML='<b style="color:var(--warn)">No GPU processes readable'+
      (procs&&procs.error? " ("+procs.error+")":"")+'.</b>'; return; }
  const eng = procs.find(p=>p.is_engine) || procs[0];
  const s=LAST.sizes_mib, p=LAST.plan, inp=LAST.inputs;

  // A measurement is only meaningful against a plan for the SAME config. LM Studio
  // silently overrides its own GPU Offload slider to respect its VRAM cap, so read
  // what it actually loaded and refuse to back-solve against a different split -
  // otherwise every layer of difference lands in the compute buffer as error.
  const rt = r.runtime;
  if(rt){
    let bad=[];
    if(rt.n_gpu_layers!=null && rt.n_gpu_layers!==p.n_gpu_layers)
      bad.push("GPU layers: engine is running <b>"+rt.n_gpu_layers+"</b>, this plan is for <b>"+
        p.n_gpu_layers+"</b>"+(inp&&LAST.plan.forced_ngl?"":" (auto-picked)"));
    if(rt.context!=null && rt.context!==inp.context)
      bad.push("Context: engine is running <b>"+rt.context.toLocaleString()+
        "</b>, this plan is for <b>"+inp.context.toLocaleString()+"</b>");
    if(bad.length){
      h.innerHTML='<b style="color:var(--warn)">Config mismatch - not measuring.</b><br>'+
        bad.join("<br>")+'<br>Set <b>GPU layers</b> to '+rt.n_gpu_layers+
        (rt.context!=null?' and <b>context</b> to '+rt.context.toLocaleString():'')+
        ', press Analyze fit, then Measure again. '+
        '<button class="ghost" onclick="matchRuntime('+rt.n_gpu_layers+','+(rt.context||0)+')">'+
        'do it for me</button>';
      return;
    }
  }

  // Record it as a calibration sample rather than a one-off override: the server
  // recomputes the exact terms, stores the row, and refits the coefficients for
  // this GPU. That makes every future plan better, not just this one.
  h.textContent="measuring and refitting...";
  let path = $("path").value.trim();
  if(!path){ const i=$("model").value; if(i!=="" && MODELS[parseInt(i)]) path = MODELS[parseInt(i)].path; }
  let res;
  try{
    res = await (await fetch("/api/calibrate",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({path:path, context:parseInt($("ctx").value)||8192,
        kv_type:$("kv").value, n_ubatch:parseInt($("ubatch").value)||512,
        n_seq:parseInt($("nseq").value)||1, flash_attn:$("fa").checked,
        include_mmproj:$("mmproj").checked})})).json();
  }catch(e){ h.textContent="calibration failed: "+e; return; }
  if(!res.ok){ h.innerHTML='<b style="color:var(--warn)">'+escapeHtml(res.error||"failed")+'</b>'; return; }

  const row=res.row, st=res.status;
  const pending = CALIB_TERMS.filter(t=>st.free.indexOf(t)<0);
  h.innerHTML = '<b style="color:var(--kv)">'+eng.name+'</b> is using <b>'+fmt(row.measured_mib)+
    '</b> at ngl '+row.ngl+' / ctx '+row.ctx.toLocaleString()+
    ' <span class="muted">(read from LM Studio\\'s log)</span>.<br>'+
    'minus exact terms '+fmt(row.exact_mib)+' &rarr; overhead <b>'+fmt(row.overhead_mib)+'</b>.<br>'+
    (st.calibrated
      ? '<b style="color:var(--kv)">Calibrated</b> from '+st.n+' measurement'+(st.n==1?'':'s')+
        ' on this GPU &mdash; fitted: '+st.free.join(", ")+
        ' (in-sample '+st.residual_pct+'%).'
      : '<b style="color:var(--warn)">Recorded, but not fitted yet.</b> The measurement is '+
        'saved; it did not produce a usable fit on its own, so the shipped defaults still '+
        'apply. Measure once more at a different context length.')+
    (pending.length? '<br><span class="muted">Still on defaults: '+pending.join(", ")+
      '. '+calibHint(pending)+'</span>' : '')+
    '<br>Press Analyze fit again.';
  run();
}

const CALIB_TERMS=["const","ctx","act","nofa"];
function calibHint(pending){
  const tips={ctx:"measure again at a very different context length",
              act:"measure a model with a different FFN width, or change ubatch",
              nofa:"measure once with flash attention off"};
  return pending.map(t=>tips[t]).filter(Boolean).join("; ")+".";
}

async function matchRuntime(ngl, ctx){
  $("ngl").value = ngl;
  if(ctx){ $("ctx").value = ctx; markCtx(); }
  await run();
  calibrate();
}

function kvItem(k,v){ return '<div class="kv"><div class="k">'+k+'</div><div class="v">'+(v==null?"-":v)+'</div></div>'; }
function brow(k,v){ return '<tr><td>'+k+'</td><td>'+v+'</td></tr>'; }
function escapeHtml(x){ return x.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function copyCmd(btn){
  const t=$("cmd").textContent;
  navigator.clipboard.writeText(t).then(()=>{ const o=btn.textContent; btn.textContent="copied"; setTimeout(()=>btn.textContent=o,1200); });
}
boot();
'''

# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _calibrate(self, data):
        """Record one Measure as a calibration row. Everything that matters is
        determined server-side: the engine's real VRAM, and the config LM Studio
        actually loaded (which is not always the one on screen)."""
        path = data.get("path") or ""
        if not os.path.exists(path):
            return {"ok": False, "error": "file not found: %s" % path}
        procs = get_gpu_processes()
        if isinstance(procs, dict):
            return {"ok": False, "error": "could not read GPU processes: %s"
                                          % procs.get("error", "?")}
        eng = next((p for p in procs if p.get("is_engine")), None)
        if not eng:
            return {"ok": False, "error": "No inference process on the GPU. Load the "
                                          "model in LM Studio first."}
        rt = read_lmstudio_runtime()
        try:
            n_layers = extract_config(load_gguf(path))["n_layers"] or 0
        except Exception as e:
            return {"ok": False, "error": "could not read model: %s" % e}
        ngl = resolve_runtime_ngl(rt, n_layers)
        ctx = (rt or {}).get("context") or int(data.get("context") or 0)
        if ngl is None:
            return {"ok": False, "error": "Could not read LM Studio's resolved GPU config "
                                          "from its log, so the layer count is unknown. "
                                          "Measurement skipped rather than guessed."}
        try:
            out = record_calibration(
                path, ctx, data.get("kv_type") or "f16",
                int(data.get("n_ubatch") or 512), int(data.get("n_seq") or 1),
                bool(data.get("flash_attn", True)), int(ngl),
                bool(data.get("include_mmproj", True)), float(eng["mib"]),
                gpu=_active_gpu(), n_cpu_moe=(rt or {}).get("n_cpu_moe") or 0)
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if out.get("error"):
            return {"ok": False, "error": out["error"]}
        out.update({"ok": True, "engine": eng,
                    "runtime": {"n_gpu_layers": ngl, "context": ctx,
                                "n_cpu_moe": (rt or {}).get("n_cpu_moe") or 0,
                                "all_layers": bool((rt or {}).get("all_layers"))}})
        return out

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        if u.path == "/app.js":
            return self._send(200, APP_JS, "application/javascript; charset=utf-8")
        if u.path == "/api/benchmark":
            q = parse_qs(u.query)
            res = benchmark_server(
                base_url=(q.get("url") or ["http://localhost:1234"])[0],
                max_tokens=int((q.get("max_tokens") or ["128"])[0]))
            if res.get("tok_s"):
                res["ctx"] = (q.get("ctx") or [None])[0]
                res["n_gpu_layers"] = (q.get("ngl") or [None])[0]
                save_benchmark(res)
            return self._send(200, res)
        if u.path == "/api/bandwidth":
            return self._send(200, get_bandwidth())
        if u.path == "/api/speedhistory":
            q = parse_qs(u.query)
            recs = scan_speed_history()
            if isinstance(recs, dict):
                recs = []
            for bm in load_benchmarks():          # server-mode measurements
                if bm.get("tok_s"):
                    recs.append({"model": bm.get("model") or "?", "tok_s": bm["tok_s"],
                                 "n_gpu_layers": bm.get("n_gpu_layers"),
                                 "ttft_s": bm.get("ttft_s"),
                                 "prompt_tokens": bm.get("prompt_tokens") or 0,
                                 "predicted_tokens": bm.get("predicted_tokens") or 0,
                                 "ctx": bm.get("ctx"), "when": bm.get("when"),
                                 "conversation": "benchmark", "benchmark": True})
            name = (q.get("model") or [""])[0]
            path = (q.get("path") or [""])[0]
            return self._send(200, {"matches": match_speed_history(recs, name, path),
                                    "all": recs, "server_log": scan_server_logs(60)})
        if u.path == "/api/gpuprocs":
            return self._send(200, {"procs": get_gpu_processes(),
                                    "runtime": read_lmstudio_runtime(),
                                    "calibration": calibration_status()})
        if u.path == "/api/system":
            fresh = parse_qs(u.query).get("fresh", ["0"])[0] == "1"
            return self._send(200, {"gpus": get_gpus(fresh=fresh), "ram": get_ram(),
                                    "default_dir": default_models_dir(),
                                    "version": __version__,
                                    "platform": platform_support()})
        if u.path == "/api/models":
            q = parse_qs(u.query)
            d = (q.get("dir", [""])[0]) or default_models_dir()
            try:
                return self._send(200, {"models": scan_models(d), "dir": d})
            except Exception as e:
                return self._send(200, {"models": [], "error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/api/analyze", "/api/calibrate"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._send(200, {"ok": False, "error": "bad request: %s" % e})
        if u.path == "/api/calibrate":
            return self._send(200, self._calibrate(data))
        try:
            path = data["path"]
            if not os.path.exists(path):
                return self._send(200, {"ok": False, "error": "file not found: %s" % path})
            res = analyze(
                path=path,
                ctx=int(data.get("context", 8192)),
                kv_type=data.get("kv_type", "f16"),
                n_ubatch=int(data.get("n_ubatch", 512)),
                n_seq=int(data.get("n_seq", 1) or 1),
                include_mmproj=bool(data.get("include_mmproj", True)),
                mtp_spec=bool(data.get("mtp_spec", True)),
                flash_attn=bool(data.get("flash_attn", False)),
                vram_budget_mib=float(data.get("vram_budget_mib", 0)),
                ram_budget_mib=float(data.get("ram_budget_mib", 0)),
                gpu_reserve_mib=float(data.get("gpu_reserve_mib", 512)),
                compute_override_mib=float(data.get("compute_override_mib", 0)),
                safety_pct=float(data.get("safety_pct", 5)),
                kv_on_gpu=bool(data.get("kv_on_gpu", False)),
                bw_vram_gbs=float(data.get("bw_vram_gbs") or 0) or None,
                bw_ram_gbs=float(data.get("bw_ram_gbs") or 0) or None,
                ram_eff=float(data.get("ram_eff") or 0) or None,
                ctx_fill=(int(data["ctx_fill"]) if data.get("ctx_fill") not in (None,"",False) else None),
                n_cpu_moe_override=(int(data["n_cpu_moe_override"])
                                    if data.get("n_cpu_moe_override") not in (None, "", False)
                                    else None),
                gpu_layers_override=(int(data["gpu_layers_override"])
                                     if data.get("gpu_layers_override") not in (None, "", False)
                                     else None),
                ram_free_mib=(float(data["ram_free_mib"])
                              if data.get("ram_free_mib") else None),
            )
            return self._send(200, res)
        except Exception as e:
            import traceback
            return self._send(200, {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                                    "trace": traceback.format_exc()})

def serve(host, port, open_browser):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0") else host, port)
    refresh_calibration()
    st = calibration_status()
    plat = platform_support()
    print("\n  VRAM Planner %s  running at  %s" % (__version__, url))
    print("  models folder default :  %s" % default_models_dir())
    print("  user data             :  %s" % _data_dir())
    print("  compute-buffer model  :  %s"
          % ("calibrated from %d measurement(s) on %s (fitted: %s, in-sample %.1f%%)"
             % (st["n"], st["gpu"] or "this GPU", ", ".join(st["free"]), st["residual_pct"])
             if st["calibrated"] else
             "shipped defaults - press Measure on a loaded model to calibrate"))
    if not plat["supported"]:
        print("\n  !! UNVALIDATED PLATFORM\n     %s" % plat["reason"])
    print("  press Ctrl+C to stop\n")
    if open_browser:
        try: webbrowser.open(url)
        except Exception: pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.server_close()

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
    global _CALIB_LOADED
    _CALIB_CACHE.clear()
    _CALIB_LOADED = True
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
    # partial offload: graph scratch on both sides, logits with the CPU-side head.
    # full offload: everything on the GPU, nothing left in RAM.
    cb_ok = (abs(rc["plan"]["compute_mib"] - gr) < 0.05
             and abs(rc["plan"]["cpu_compute_mib"] - (gr + lg)) < 0.05
             and rf["plan"]["cpu_compute_mib"] == 0
             and abs(rf["plan"]["compute_mib"] - (gr + lg)) < 0.05 and lg > 0)
    print("  CMPBUF split ngl=2: gpu=%.1f cpu=%.1f | full offload: gpu=%.1f cpu=%.1f "
          "(graph=%.1f logits=%.1f)  %s"
          % (rc["plan"]["compute_mib"], rc["plan"]["cpu_compute_mib"],
             rf["plan"]["compute_mib"], rf["plan"]["cpu_compute_mib"], gr, lg,
             "OK" if cb_ok else "FAIL"))
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
           ("q9",  8192,  31, 1, 512, True,  "q8_0",  6477.6, False)]
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
        vram_ok = worst <= 13.0
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
    truth = {"const": 900.0, "ctx": 0.20, "act": 40.0, "nofa": 5.0}
    def synth(ctx, ub, fa, hid=4096, nh=32, kvtok=8192):
        r = {"ctx": ctx, "ub": ub, "fa": fa, "hidden": hid, "n_head": nh,
             "kv_tok_ctx": kvtok * ctx, "exact_mib": 1000.0}
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
                 "kv_tok_ctx": 5368709120.0, "exact_mib": 8965.0, "overhead_mib": 943.6}]
    single = fit_calibration(one_real)
    single_ok = (single is not None and single["free"] == ["const"]
                 and single["coeffs"]["const"] + _struct_offset(one_real[0]) > 0)
    print("  CALFIT single real measurement: fitted=%s const=%.1f  %s"
          % (single is not None, (single or {}).get("coeffs", {}).get("const", 0.0),
             "OK" if single_ok else "FAIL"))
    ok = ok and single_ok

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

def main():
    ap = argparse.ArgumentParser(description="Plan GGUF model fit on GPU/RAM.")
    ap.add_argument("--version", action="version", version="vram-planner %s" % __version__)
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    serve(args.host, args.port, not args.no_browser)

if __name__ == "__main__":
    main()
