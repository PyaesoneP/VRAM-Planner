"""KV cache geometry: per-layer bytes, sliding windows, recurrent state."""
from .const import _mib


# KV cache bytes per element for each cache quantization
KV_TYPE_BYTES = {
    "f16":  2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}


# llama.cpp pads the KV cache; 256 with flash attention, 32 without
KV_PAD_FA, KV_PAD_NOFA = 256, 32


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
