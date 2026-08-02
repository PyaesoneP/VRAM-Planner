"""The compute buffer - the one fuzzy term - and where the graph runs."""
from .const import _mib


# Empirical coefficients for the GPU-side runtime overhead (compute buffer +
# backend workspace). These are MEASURED, not derived: see the note in
# compute_buffer_terms for why the textbook formula does not describe reality.
# Fitted over 36 loads spanning four architectures (gemma4 MoE + qwen35
# hybrid-SSM + qwen35moe + qwen35 27B), CUDA / llama.cpp 2.27.1, against the
# `CUDA0 compute buffer size` the allocator itself reports under -v. Earlier
# versions fitted against total process VRAM, where this term is a small share of
# a total dominated by exact weights and KV - that flattered the error badly. On
# the buffer alone the old form scores 54% mean / 311% worst; this one scores
# 20.7% mean / 76.6% worst. It is still the one estimated term in the tool.
CB_ACT_PER_HIDDEN  = 44.25    # bytes per ubatch token per unit of hidden size

# The ctx term is NOT a share of the KV cache, which is what this used to assume.
# Disproved directly: Qwen3.6-27B has exactly 2x Qwen3.5-9B's KV bytes per token
# and an identical 7296 B/token of compute buffer, and Gemma 4 26B's KV does not
# grow with context at all (sliding window, flat at 159.38 MiB) yet its buffer
# still grows at 7379 B/token. So it is charged per context token, flat, and
# sliding-window layers get no discount here even though they do for the cache.
CB_CTX_PER_TOKEN   = 1059.9   # bytes per ctx token

# ... plus the attention mask [n_kv x n_ubatch]. Fitted freely at 1.990 B, which
# is f16 to within a rounding error - a good sign the decomposition is real and
# not just a curve with enough knobs.
CB_MASK_PER_UB_TOK = 2.0      # bytes per (ubatch token x ctx token)

# A quantised KV cache costs MORE compute buffer than f16, not less: measured
# 5120 B/token at q8_0 against 1024 B/token at f16, same model and config. The
# cache is smaller but the kernels need scratch the f16 path does not.
CB_CTX_QUANT_BYTES = 3070.0   # extra bytes per ctx token when the cache is quantised

# When the graph spans both CPU and GPU, the CUDA pool is BIGGER than at full
# offload, not smaller. Measured on Qwen3.5-9B: 999.38 MiB at -ngl 0, 8 and 23
# alike against 719.66 at -ngl 33 (ctx 131072), and 315.38 against 239.66 (ctx
# 32768). So the surcharge is independent of how many layers landed on the GPU -
# it is binary in whether the graph is split at all - but it does grow with
# context, at 2176 B/token, which is one attention layer's worth of cache: the
# staging buffer for moving activations across the backend boundary.
CB_SPLIT_GRAPH_MIB = 7.7      # flat part of the split surcharge, MiB

CB_SPLIT_PER_TOKEN = 2176.0   # ... plus this per context token

CB_NOFA_HEAD_BYTES = 3.36     # f32 score matrix, per head per ctx token per ubatch token


# MTP speculative decoding (llama.cpp --spec-type draft-mtp). Measured on
# Qwen3.5-9B-MTP at ctx 8k/32k/131k and --parallel 1/2/4: the draft cache costs
# one MTP block of f16 KV over the full context, plus a fixed pool and a
# per-sequence slot. Linear in both to within ~9%.
MTP_SPEC_CONST_MIB   = 104.0

MTP_SPEC_PER_SEQ_MIB = 100.0

# Everything the process holds on top of the pool the allocator reports: the CUDA
# context, the driver, and the lazily-loaded kernel modules. The allocator log
# cannot see any of it, so these two come from total-process-VRAM measurements
# instead - 25 loads across four architectures - with the scaling terms above
# held fixed at their allocator-derived values. That division of labour is the
# point: the log fixes the slopes exactly, the perf counters fix the floor.
#
# It is genuinely large and it scales with the residual-stream width: ~147 MiB at
# hidden 2048 against ~759 MiB at hidden 5120. A bare CUDA context measured 156
# MiB on its own, so most of this is kernel modules, and wider models pull in
# more of them. The intercept is negative because it is an intercept and not a
# byte count; the pair is clamped at zero below.
CB_GRAPH_CONST_MIB = -102.1   # additive base, per machine (this is what Measure fits)

CB_CONST_PER_KHID  = 200.17   # MiB per 1000 units of hidden size

# ... but only once at least one layer actually runs on the GPU. At -ngl 0 the
# process holds a bare CUDA context and nothing else: dedicated VRAM minus the
# reported compute buffer came to 156.2 / 156.4 / 155.97 MiB across three
# configs, against the ~555 MiB the formula above gives for that model. CUDA
# loads kernel modules lazily, so with no layer offloaded almost none arrive.
# (This is also the "-ngl > n_layer costs a fixed ~156 MiB" anomaly that shipped
# as a known issue - the same allocation, seen from the other side.)
CB_CUDA_CTX_MIB    = 156.0    # bare CUDA context, when no layer is on the GPU


# These four are the ONLY numbers in the tool that are fitted rather than read
# from the model file, and they are the only ones that depend on the machine
# rather than the model. They ship as a prior and are refined per GPU from the
# user's own Measure runs - see fit_calibration().
CB_DEFAULTS = {"const": CB_GRAPH_CONST_MIB, "act": CB_ACT_PER_HIDDEN,
               "ctx": CB_CTX_PER_TOKEN, "nofa": CB_NOFA_HEAD_BYTES}


def compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq=1, kv_type="f16"):
    """The GPU-side runtime overhead beyond weights and KV, in MiB.

    This is the ONLY fuzzy term; weights and KV are exact. It is worth being
    explicit about how it is derived, because the obvious formula is wrong and so
    was the previous version of this one.

    The numbers behind it come from llama.cpp's own allocator: running with -v
    prints `CUDA0 compute buffer size = N MiB`, which is exact, where the earlier
    fit used total process VRAM and had to infer this term by subtraction. 36
    loads across four architectures show it:

      * IS allocated on the GPU even at -ngl 0, where no layer is offloaded at
        all. Measured 999.38 MiB at ctx 131072. The previous version charged the
        GPU nothing here, which is what made partial-offload plans overcommit.
      * does NOT depend on how many layers are offloaded - but DOES depend on
        whether the graph is split. Identical at -ngl 0, 8 and 23 (999.38 MiB),
        and smaller at full offload (719.66). See CB_SPLIT_GRAPH_MIB.
      * grows linearly in n_ctx, and separately in n_ctx x n_ubatch - the second
        being the f16 attention mask, which fitted freely to 1.990 B.
      * grows with n_ctx *flat*, not as a share of the KV cache: two models with
        2x different KV per token show the identical 7296 B/token, and a
        sliding-window model whose cache does not grow with context at all still
        grows here. Sliding-window layers therefore get no discount in this term.
      * costs MORE with a quantised KV cache than with f16 (5120 vs 1024
        B/token), the opposite of what the cache sizes do.
      * does NOT depend on n_batch, or on --parallel (719.66 at -np 1 and -np 4).

    Residuals are systematically positive at short context and negative at long,
    so the truth is almost certainly a max() over several buffers rather than a
    sum, and no linear form will close it. This one lands at 20.7% mean / 76.6%
    worst on the buffer in isolation. Press Measure to pin it for your machine;
    weights and KV need no such caveat.

    Returned separately:
      * graph  - the pool above, which lives wherever the layers run.
      * split_extra - the surcharge for a graph that spans CPU and GPU.
      * logits - the output tensor [n_outputs x n_vocab], produced by the output
        matmul, so it belongs to whichever backend holds the head - NOT
        automatically to the GPU.
    """
    hidden  = cfg["hidden"] or 4096
    n_head  = cfg["n_head"] or 32
    n_vocab = cfg.get("n_vocab") or 32000
    ub      = max(1, n_ubatch)
    ctx     = max(1, context)
    from .calib import calib_coeffs   # deferred: calib is fitted against THIS module
    K       = calib_coeffs()

    # the floor: CUDA context + driver + kernel modules. Clamped at zero because
    # the intercept is negative and a narrow enough model would drive it under.
    floor = max(0.0, K["const"] + CB_CONST_PER_KHID * hidden / 1000.0)

    attn = cfg.get("attn_layers") or []
    if not attn:
        return {"graph": round(floor, 1), "floor": round(floor, 1), "logits": 0.0,
                "split_extra": round(CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * ctx), 1)}

    # ubatch-sized activation scratch, in units of the residual-stream width
    act = K["act"] * ub * hidden

    # context terms: a flat per-token pool plus the [n_kv x n_ubatch] f16 mask,
    # with a surcharge when the cache is quantised. No SWA discount here - see
    # the note above; that is measured, not an oversight.
    ctx_term = K["ctx"] * ctx + CB_MASK_PER_UB_TOK * ub * ctx
    if kv_type != "f16":
        ctx_term += CB_CTX_QUANT_BYTES * ctx

    # without flash attention the full f32 score matrix is materialised, and it is
    # the one term that really is [n_kv x n_ubatch x n_head]
    scores = 0.0
    if not flash_attn:
        scores = K["nofa"] * n_head * ub * ctx
        # ^ this term is why flash attention is not optional at long context:
        #   at 64k ctx it alone is ~3.5 GiB on a 32-head model.

    graph = floor + _mib(act + ctx_term + scores)
    return {"graph": round(graph, 1), "floor": round(floor, 1),
            "logits": round(_mib(4.0 * ub * n_vocab), 1),
            "split_extra": round(CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * ctx), 1)}


def compute_buffer_split(terms, any_on_gpu, any_on_cpu, output_on_gpu, override_mib=None):
    """Split the compute buffer across backends. llama.cpp allocates a scratch
    pool per backend that runs part of the graph, so a partial offload pays the
    graph term on BOTH sides, while the logits tensor lands on exactly one - the
    one holding the output head. At -ngl 4 of 60 the head stays on the CPU, so
    charging its (often huge) logits buffer to VRAM overstates the GPU footprint.

    The scaling part of the graph pool is charged to the GPU whether or not a
    layer landed there - it is allocated at -ngl 0 too, measured at 999.38 MiB.
    Charging nothing there is what let partial-offload plans overcommit. Only the
    fixed floor is conditional, because that floor is lazily-loaded CUDA kernel
    modules and they do not arrive until a layer needs them.

    An override is a measured VRAM number, so it replaces the GPU side outright."""
    split_extra = terms.get("split_extra", 0.0) if any_on_cpu else 0.0
    graph = terms["graph"]
    if not any_on_gpu:
        graph -= max(0.0, terms.get("floor", 0.0) - CB_CUDA_CTX_MIB)
    gpu = graph + split_extra + (terms["logits"] if output_on_gpu else 0.0)
    cpu = (terms["graph"] if any_on_cpu else 0.0) + (0.0 if output_on_gpu else terms["logits"])
    if override_mib:
        gpu = float(override_mib)
    return {"gpu": round(gpu, 1), "cpu": round(cpu, 1)}


def graph_is_split(n_layers, ngl, n_cpu_moe=0):
    """Does the graph span both backends? THE definition of `any_on_cpu`.

    Both knobs move work to the CPU and they are independent: --n-cpu-moe pins
    the routed experts of the first M blocks there even at ngl == n_layers, so
    testing ngl alone reports every expert offload as unsplit. The planner and
    the calibration fit must read this from one place - when they did not, the
    fit saw those rows as unsplit and quietly absorbed the split surcharge into
    `const`, after which prediction charged it a second time. On a 26B MoE at
    262k that was ~850 MiB of phantom VRAM, about two expert layers."""
    return bool((n_layers and ngl is not None and ngl < n_layers)
                or (n_cpu_moe or 0) > 0)


def output_head_on_gpu(n_layers, ngl):
    """Does the output head - and so the logits tensor - land on the GPU? The
    head follows the last block, so it does exactly when every block is offloaded.
    --n-cpu-moe does not move it: routed experts are not the head."""
    return bool(n_layers and ngl is not None and ngl >= n_layers)


def compute_buffer_mib(cfg, context, n_ubatch, flash_attn, n_seq=1, kv_type="f16"):
    """Total compute buffer across all backends (back-compat / whole-model view)."""
    t = compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq, kv_type)
    return round(t["graph"] + t["logits"], 1)
