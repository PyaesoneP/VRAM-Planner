"""The compute buffer - the one fuzzy term - and where the graph runs."""
from .const import _mib


# Empirical coefficients for the GPU-side runtime overhead (compute buffer +
# backend workspace). These are MEASURED, not derived: see the note in
# compute_buffer_terms for why the textbook formula does not describe reality.
#
# The evidence is 146 usable loads over 5 models and 3 architectures (gemma4 dense
# and MoE, qwen35 hybrid-SSM at 9B and 27B, qwen35moe), CUDA / llama.cpp 2.27.1,
# collected by sweep.py against the `CUDA0 compute buffer size` the allocator itself
# reports under -v. Run `python -m vram_planner --fit` to score them on your own.
#
# Held out by architecture - fitted on two, scored on the third, never on itself -
# this form lands at 22.5% mean / 85.6% worst on the buffer in isolation, and the
# whole plan at 7.6% / 39.6% against the process counter. The figures this file used
# to quote (20.7% / 76.6%) were in-sample and are not comparable.
#
# Wholesale refitting was tried and REJECTED: it scores better in sample and worse
# out of it, every time. Freeing the context slope as well as the MoE term below
# takes the held-out error from 21.8% to 31.1%; freeing the activation slope too
# takes it to 33.0%. Three architectures do not identify six coefficients. Most of
# these numbers are therefore unchanged, on purpose.
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

# A routed-expert block materialises n_expert_used copies of the FFN activation for
# every token in the ubatch, and that working set does not depend on context at all.
# The model had no term for it, which is why both MoE models were under-predicted by
# 60-77% at short context - where nothing else is large enough to hide it - and
# converged back to the dense error at 262k once the context terms took over.
#
# This is the one coefficient added from the sweep, and it was added alone: fitting
# it by itself takes the held-out error from 28.5% to 21.8%, while ALSO refitting
# the context slope takes it back up to 31.1%, and refitting the activation slope
# as well to 33.0%. Three architectures cannot identify six coefficients, and
# leave-one-architecture-out says so every time it is asked.
#
# Note the width is expert_feed_forward_length, NOT feed_forward_length: llama.cpp
# reports n_ff = 0 for Qwen3.6-35B-A3B and keeps the real width in the expert key.
CB_MOE_ACT_PER_WIDTH = 54.4   # bytes per ubatch token per unit of routed-expert width

# The output tensor is f32 and sized for one output token PER SERVER SLOT, not for
# the ubatch. Measured at exactly 4 * n_vocab * n_parallel bytes: 1.00 MiB at vocab
# 262144 and 0.95 MiB at 248320 with -np 1, unchanged across ubatch 64 through 2048,
# and stepping to exactly 2/4/8x at -np 2/4/8. It is also HOST memory - across 146
# loads there is no CUDA0.output buffer at any offload level, only CUDA_Host.output.
#
# The previous form charged 4 * n_ubatch * n_vocab to VRAM whenever the head landed
# on the GPU, which at ubatch 2048 is 2 GiB of phantom VRAM on a large-vocabulary
# model. Both mistakes at once: the wrong multiplier and the wrong device.
CB_OUTPUT_BYTES    = 4.0      # f32 logits, per output token, on the host


# MTP speculative decoding (llama.cpp --spec-type draft-mtp). Measured on
# Qwen3.5-9B-MTP at ctx 8k/32k/131k and --parallel 1/2/4: the draft cache costs
# one MTP block of f16 KV over the full context, plus a fixed pool and a
# per-sequence slot. Linear in both to within ~9%.
MTP_SPEC_CONST_MIB   = 104.0

MTP_SPEC_PER_SEQ_MIB = 100.0

# Everything the process holds on top of the pool the allocator reports: the CUDA
# context, the driver, and the lazily-loaded kernel modules. The allocator log
# cannot see any of it, so it comes from total-process-VRAM measurements instead,
# with the scaling terms above held fixed at their allocator-derived values. That
# division of labour is the point: the log fixes the slopes exactly, the perf
# counters fix the floor.
#
# The bare CUDA context is the most reproducible number in this file. At -ngl 0 the
# process holds it and nothing else, and across all five models it measured 155.9 to
# 156.6 MiB - a 0.7 MiB spread over architectures spanning 9B to 35B.
CB_CUDA_CTX_MIB    = 156.0    # bare CUDA context, before any kernel module loads

# On top of that, the floor grows with how many blocks are actually resident,
# because CUDA loads kernel modules lazily and more layers pull in more of them.
# Measured 156 MiB at -ngl 0 rising to 463.6 MiB at -ngl 30 on gemma-4-26B.
#
# This replaces a pair of coefficients that made the floor a function of hidden size
# alone, gated on whether ANY layer was offloaded. That form had no way to express
# the ramp: it charged the full-offload floor at -ngl 1, and for a wide model it
# charged far more than the floor has ever been observed to be - 974 MiB for
# gemma-4-31B against a maximum of 463.6 MiB seen anywhere in 144 loads.
#
# It is the weakest term in the file and it is honest to say so: no candidate form
# fits it better than ~17 MiB mean absolute error, and the per-layer rate genuinely
# differs by architecture (~10 MiB/layer on gemma-4, ~2.5 on Qwen3.6-35B-A3B). It is
# also the term Measure is best placed to pin, since it is a property of the machine.
CB_FLOOR_PER_LAYER = 3.79     # MiB per block resident on the GPU


# These four are the ONLY numbers in the tool that are fitted rather than read
# from the model file, and they are the only ones that depend on the machine
# rather than the model. They ship as a prior and are refined per GPU from the
# user's own Measure runs - see fit_calibration().
CB_DEFAULTS = {"floor": CB_FLOOR_PER_LAYER, "act": CB_ACT_PER_HIDDEN,
               "ctx": CB_CTX_PER_TOKEN, "nofa": CB_NOFA_HEAD_BYTES}


def compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq=1, kv_type="f16",
                         coeffs=None):
    """The GPU-side runtime overhead beyond weights and KV, in MiB.

    This is the ONLY fuzzy term; weights and KV are exact. It is worth being
    explicit about how it is derived, because the obvious formula is wrong and so
    were both previous versions of this one.

    The numbers come from llama.cpp's own allocator: running with -v prints
    `CUDA0 compute buffer size = N MiB`, which is exact. 144 usable loads over five
    models and three architectures, collected by sweep.py, show it:

      * IS allocated on the GPU even at -ngl 0, where no layer is offloaded at all.
        Charging nothing there is what let partial-offload plans overcommit.
      * depends on whether the graph is SPLIT far more than on how many layers
        landed on the GPU: identical at -ngl 0, 1, 2, 4, 7, 8 and 15 on gemma-4-26B
        (501.13 MiB every time), and smaller once nothing is left on the CPU.
      * grows linearly in n_ctx, and separately in n_ctx x n_ubatch - the second
        being the f16 attention mask, which fitted freely to 1.990 B.
      * carries a large context-INDEPENDENT term on MoE models, the routed-expert
        activation scratch. See CB_MOE_ACT_PER_WIDTH; adding it is the single
        change that most improved held-out accuracy.
      * costs MORE with a quantised KV cache than with f16, the opposite of what
        the cache sizes do.
      * tracks the FULL context, not the per-slot context. This one is worth
        stating because the host-side buffer does the opposite: CUDA_Host.compute
        halves exactly as --parallel doubles, so llama.cpp sizes that one from
        n_ctx / n_parallel. Modelling the device side the same way made held-out
        error worse (24.3% against 21.8%), so the two really do differ.

    A max() over candidate peaks was tried, since ggml-alloc reports a peak over the
    graph rather than a sum, and the residual signs suggested it. It fits better in
    sample and generalises WORSE - 11.7% in sample against 28.2% held out, where the
    additive form gets 21.7% and 21.8%. It is not in here because the data does not
    support it yet, not because it was not tried; revisit with more architectures.

    Held out by architecture, this form scores 22.5% mean / 85.6% worst on the
    buffer in isolation, and takes the END-TO-END total from 22.7% to 7.6% - most of
    that from the output tensor and the floor below, not from the buffer itself. Run
    `python -m vram_planner --fit` to reproduce it on your own sweep.
    Press Measure to pin it for your machine; weights and KV need no such caveat.

    Returned separately:
      * graph  - the scaling pool, which lives wherever the layers run. This no
        longer includes the floor; compute_buffer_split() adds that, because it
        depends on how many blocks are resident and this function is not told.
      * floor_base / floor_per_layer - the CUDA context and the kernel modules.
      * split_extra - the surcharge for a graph that spans CPU and GPU.
      * output - the f32 logits tensor. HOST memory, and sized for one output
        token, not for the ubatch. See CB_OUTPUT_BYTES.
    """
    hidden  = cfg["hidden"] or 4096
    n_head  = cfg["n_head"] or 32
    n_vocab = cfg.get("n_vocab") or 32000
    ub      = max(1, n_ubatch)
    ctx     = max(1, context)
    # Routed-expert width. expert_feed_forward_length, not feed_forward_length -
    # they are different keys and a MoE may carry only the former.
    moe_w   = (cfg.get("n_expert_used") or 0) * (cfg.get("expert_ffn_len") or 0)
    # `coeffs` lets a caller score this model under a candidate fit without
    # touching the user's calibration store - which is what cross-validation has
    # to do, since the whole point is to predict with coefficients the row being
    # predicted took no part in producing.
    if coeffs is None:
        from .calib import calib_coeffs   # deferred: calib is fitted against THIS module
        coeffs = calib_coeffs()
    K = dict(CB_DEFAULTS)
    K.update(coeffs or {})

    # The output tensor: host memory, one output token per server slot. Kept in the
    # returned dict so the caller can show it, but it never reaches the GPU side.
    output = round(_mib(CB_OUTPUT_BYTES * n_vocab * max(1, n_seq)), 1)
    base = {"floor_base": CB_CUDA_CTX_MIB, "floor_per_layer": K["floor"],
            "output": output,
            "split_extra": round(CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * ctx), 1)}

    attn = cfg.get("attn_layers") or []
    if not attn:
        return dict(base, graph=0.0)

    # ubatch-sized activation scratch, in units of the residual-stream width...
    act = K["act"] * ub * hidden
    # ...plus, on a MoE, one copy per routed expert of the expert FFN activation.
    # Context-independent, which is why it shows up most clearly at short context.
    act += CB_MOE_ACT_PER_WIDTH * ub * moe_w

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

    return dict(base, graph=round(_mib(act + ctx_term + scores), 1))


def compute_buffer_split(terms, any_on_gpu, any_on_cpu, n_gpu_layers=0, override_mib=None):
    """Split the compute buffer across backends.

    llama.cpp allocates a scratch pool per backend that runs part of the graph, so a
    partial offload pays the graph term on BOTH sides. The scaling part is charged
    to the GPU whether or not a layer landed there - it is allocated at -ngl 0 too.
    Charging nothing there is what let partial-offload plans overcommit.

    The floor is charged per resident block on top of the bare CUDA context, rather
    than as a fixed amount gated on "is anything offloaded". Measured, the floor
    ramps from 156 MiB at -ngl 0 to 463.6 MiB at -ngl 30 on gemma-4-26B; the old
    step function charged the whole thing at -ngl 1.

    The output tensor is NOT placed here any more. It is host memory in every one of
    144 measured loads, at every offload level, so it goes to the CPU side
    unconditionally - see CB_OUTPUT_BYTES.

    An override is a measured VRAM number, so it replaces the GPU side outright."""
    split_extra = terms.get("split_extra", 0.0) if any_on_cpu else 0.0
    floor = terms.get("floor_base", CB_CUDA_CTX_MIB)
    if any_on_gpu:
        floor += terms.get("floor_per_layer", 0.0) * max(0, int(n_gpu_layers))
    gpu = terms["graph"] + floor + split_extra
    cpu = (terms["graph"] if any_on_cpu else 0.0) + terms.get("output", 0.0)
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
    """Does the output head land on the GPU? The head follows the last block, so it
    does exactly when every block is offloaded. --n-cpu-moe does not move it:
    routed experts are not the head.

    This no longer decides where the logits buffer is charged - it is host memory
    either way - but it still describes where the head's WEIGHTS run, which the
    planner and the calibration fit both need."""
    return bool(n_layers and ngl is not None and ngl >= n_layers)


def compute_buffer_mib(cfg, context, n_ubatch, flash_attn, n_seq=1, kv_type="f16",
                       n_gpu_layers=None):
    """Total compute buffer across all backends (back-compat / whole-model view).

    Assumes full offload unless told otherwise, since that is what a whole-model
    view means - and the floor now depends on how many blocks are resident."""
    t = compute_buffer_terms(cfg, context, n_ubatch, flash_attn, n_seq, kv_type)
    n = cfg.get("n_layers") or 0 if n_gpu_layers is None else n_gpu_layers
    return round(t["graph"] + t["floor_base"] + t["floor_per_layer"] * max(0, n)
                 + t["output"], 1)
