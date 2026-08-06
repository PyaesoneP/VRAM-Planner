"""Architecture-agnostic config extraction and tensor classification."""
import os, re
from .kv import recurrent_bytes


# Sliding-window interleave stride, for architectures where llama.cpp hardcodes
# set_swa_pattern(p) instead of writing a pattern into the GGUF. Only consulted
# when the file gives a window size but no pattern; p = 1 means "every layer is
# windowed", which is also the fallback for any arch not listed here.
SWA_STRIDE_BY_ARCH = {
    "gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4,
    "gpt-oss": 2, "llama4": 4, "exaone4": 4, "hunyuan-moe": 2,
}


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
    mla_layers, mla_k_tensor, mla_lora_tensor = set(), 0, 0
    for t in model["tensors"]:
        m = RE_BLK.match(t["name"])
        if not m:
            continue
        li = int(m.group(1)); rest = t["name"][m.end():]
        if rest.startswith(("attn_k.", "attn_v.", "attn_k_b.", "attn_kv_")):
            attn_layers.add(li)
            # ---- multi-head latent attention (DeepSeek-V2/V3, Kimi) ------------
            # These blocks do NOT cache K and V per head. They cache one compressed
            # latent per token: the output of attn_kv_a_mqa, which is
            # kv_lora_rank + qk_rope_head_dim wide (576 on V3) and shared across
            # every head. Sizing them as n_kv_heads * (head_dim_k + head_dim_v)
            # overstates the cache several-fold.
            #
            # Both widths are in the tensor table, which is ground truth:
            #   attn_kv_a_mqa  ne = [n_embd, kv_lora_rank + qk_rope]
            #   attn_kv_b      ne = [kv_lora_rank, n_head * (qk_nope + v_head_dim)]
            if rest.startswith("attn_kv_a_mqa.") and len(t["dims"]) >= 2:
                mla_layers.add(li)
                mla_k_tensor = max(mla_k_tensor, t["dims"][1])
            elif rest.startswith("attn_kv_b.") and len(t["dims"]) >= 1:
                mla_layers.add(li)
                mla_lora_tensor = max(mla_lora_tensor, t["dims"][0])
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

    # ---- MLA cache geometry --------------------------------------------------
    # Mirrors llama.cpp: it reads key_length_mla / value_length_mla when the file
    # carries them and otherwise derives them from kv_lora_rank and the rope
    # dimension. The tensor table is the last resort, and the most reliable of the
    # three - it is what the weights actually are rather than what a converter
    # claimed. The cached latent is shared across heads, so the head count for
    # these blocks is 1, not n_head_kv.
    kv_lora_rank = _as_int(g(".attention.kv_lora_rank"), 0) or 0
    qk_rope_dim  = _as_int(g(".rope.dimension_count"), 0) or 0
    head_dim_k_mla = (_as_int(g(".attention.key_length_mla"), 0) or 0
                      or (kv_lora_rank + qk_rope_dim if kv_lora_rank else 0)
                      or mla_k_tensor)
    head_dim_v_mla = (_as_int(g(".attention.value_length_mla"), 0) or 0
                      or kv_lora_rank or mla_lora_tensor)
    mla_layers &= set(attn_layers)
    is_mla = bool(mla_layers and head_dim_k_mla and head_dim_v_mla)
    if not is_mla:
        mla_layers = set()

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
    # one shared latent per token, not one K/V pair per head
    for i in mla_layers:
        if i < len(kv_heads_per_layer):
            kv_heads_per_layer[i] = 1

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
            if p <= 0 and arch in SWA_STRIDE_BY_ARCH:
                p, src = SWA_STRIDE_BY_ARCH[arch], "arch default (%s)" % arch
            if p > 1:
                swa_layers, swa_source = {i for i in rng if (i + 1) % p != 0}, src
            elif p == 1:
                # an explicit stride of 1 really does mean every layer is windowed
                swa_layers, swa_source = set(rng), "sliding_window (uniform)"
            else:
                # Unknown architecture, a window declared but no pattern anywhere.
                # Assuming "every layer windowed" is the cheap answer and the wrong
                # direction to be wrong in: it can understate KV by 10-20x at long
                # context, and this tool's job is to say whether something fits.
                # Charge the full cache and say the window was ignored.
                swa_layers, swa_source = set(), \
                    "ignored - %s declares a window but no pattern this tool knows" % arch
    swa_layers &= set(attn_layers)

    # per-layer (head_dim_k, head_dim_v, is_swa) - precomputed so the KV math and
    # the layer-split search stay O(1) per layer instead of re-deriving this.
    # MLA takes precedence over the SWA dims: no MLA architecture ships a sliding
    # window today, but if one does, the latent width is what gets cached either way.
    kv_layer_dims = [([head_dim_k_mla, head_dim_v_mla, 1 if i in swa_layers else 0]
                      if i in mla_layers else
                      [head_dim_k_swa, head_dim_v_swa, 1] if i in swa_layers
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
        "is_mla": is_mla, "mla_layers": sorted(mla_layers),
        "kv_lora_rank": kv_lora_rank,
        "head_dim_k_mla": head_dim_k_mla, "head_dim_v_mla": head_dim_v_mla,
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
