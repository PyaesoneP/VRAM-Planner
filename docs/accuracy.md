# What is exact, what is estimated, and how to fit it to your machine

Weights and KV are computed from the GGUF tensor table and are exact. One term is
not. This is how far off it is, how it was fitted, and how to re-fit it against your
own hardware.

[← back to the README](../README.md)

---

## What is exact vs estimated

**Exact**, straight from the GGUF tensor table and your context / KV quant — no
estimating: **weights**, **KV cache**, the **recurrent state** and the **projector**.
Per-layer offload deltas match llama.cpp to ~0.1 MiB.

**Estimated**: the compute buffer, and only that. The numbers come from llama.cpp's own
allocator — running with `-v` prints `CUDA0 compute buffer size = N MiB`, which is
exact. `python -m vram_planner --sweep` drives `llama-server` across a config grid and
records that line, the per-process VRAM counter, and llama.cpp's own layer placement,
one JSON row per load. The current model is fitted to **146 usable loads over 5 models
and 3 architectures**. Measured that way, the GPU-side overhead:

- **is allocated even at `-ngl 0`**, where not one layer is offloaded. The tool used to
  charge the GPU nothing here, which is exactly what let partial-offload plans
  overcommit and spill into shared memory;
- depends on whether the graph is *split* far more than on how many layers landed on
  the GPU — identical at `-ngl` 0, 1, 2, 4, 7, 8 and 15 on Gemma 4 26B (501.13 MiB every
  time), and smaller once nothing is left on the CPU. A split graph costs more, not less;
- grows linearly in `n_ctx`, **flat — not as a share of the KV cache**;
- grows separately in `n_ctx x n_ubatch` — the f16 attention mask, which fitted freely
  to **1.990 B** and is the one term whose textbook value the data actually confirms;
- carries a large **context-independent** term on MoE models: the routed-expert
  activation scratch, `n_expert_used` copies of the expert FFN activation per ubatch
  token. Adding it is the single change that most improved held-out accuracy;
- costs **more** with a quantised KV cache than with f16, the opposite of what the
  cache sizes do;
- tracks the **full** context, not the per-slot context — worth stating because the
  host-side buffer does the opposite. `CUDA_Host.compute` halves exactly as `--parallel`
  doubles, so llama.cpp sizes *that* one from `n_ctx / n_parallel`. Modelling the device
  side the same way made held-out error worse, so the two really do differ;
- and sits on a floor of **~156 MiB plus a few MiB per resident block** — the bare CUDA
  context, then lazily-loaded kernel modules as layers arrive. The 156 is the most
  reproducible number here: 155.9 to 156.6 MiB at `-ngl 0` across all five models.

That floor is invisible to the allocator log, so it comes from the process counter
instead. The log fixes the slopes exactly; the perf counters fix the floor.

Plus the f32 score matrix when flash attention is **off** — that term alone is ~3.5 GiB
at 64k context on a 32-head model, which is why FA is not optional at long context.

### Accuracy

Every number below is **held out by architecture** — fitted on two architectures and
scored on the third, never on itself. Reproduce with `python -m vram_planner --fit`.

| what | mean | worst |
| --- | --- | --- |
| **total GPU VRAM** (what the planner reports) | **7.6%** | 39.6% |
| the compute buffer alone | 22.5% | 85.6% |
| the floor, in MiB | 17.2 MiB | 226.7 MiB |

The total is uniformly good across all five models — 4.1%, 5.4%, 6.9%, 9.2%, 12.3% —
rather than a mean dragged around by one of them. The worst cases are all tiny-context,
near-zero-offload corners where the whole total is about 1 GiB, so a 200 MiB miss reads
as 20%. The buffer alone scores worse than the total because it *is* a slice of it; the
rest is weights and KV, which are exact.

Earlier versions of this file quoted 2.7% mean / 8.1% worst. That was an **in-sample**
number: the coefficients were fitted on the same loads the self-test then scored them
against, and it also concealed a pair of compensating errors — a floor formula that
charged up to 974 MiB where the floor has never been observed above 463.6, cancelling
against embeddings that were charged to RAM when llama.cpp had put them in VRAM.

A `max()` over candidate peaks was tried, since ggml-alloc reports a peak over the graph
rather than a sum. It fits better in sample and generalises **worse** — 11.7% in sample
against 28.2% held out, where the additive form gets 21.7% and 21.8%. It is not in the
tool because the data does not support it yet, not because it was not tried.

Press **Measure running model** — under **Expert** — to pin the machine-dependent
coefficients for your card (see below).

- **Parallel seqs** should match LM Studio's "Parallel" / llama.cpp `-np`. It sizes the
  recurrent state on hybrid models and the sliding-window cache on SWA models.
  Note `-np N` divides `-c` per slot, so total KV is unchanged.


## Calibration

Four compute-buffer coefficients are the only numbers here that depend on your
hardware rather than the model — `floor` (MiB of CUDA kernel modules per resident
block), `ctx`, `act` and `nofa`. Rather than ask you to understand them, the tool fits
them from your own runs.

Load a model in LM Studio, press **Measure running model**. The tool reads the engine
process's real VRAM, reads the config LM Studio *actually resolved* (not what the UI
shows — it silently overrides its own sliders), recomputes the exact terms server-side,
and refits.

It only frees as many coefficients as your data can identify — fitting four to one
measurement would be worse than shipping the defaults:

| measurements | what gets fitted |
|---|---|
| 0 | shipped defaults |
| 1 | the floor — MiB of CUDA kernel modules per resident block |
| 3+ spread over **context** | + the ctx slope |
| + varied **ubatch** | + the activation slope |
| + one flash-attention-**off** run | + the score term |

A term unlocks only when the knob it belongs to actually moved, and a fitted slope more
than 10x from the prior, or negative, is rejected in favour of the default. Rows are
keyed by GPU **and** llama.cpp build, so upgrading the backend does not silently reuse a
fit that no longer applies.

### The fit is frozen once made

Measuring writes the fitted coefficients to the store, and **nothing recomputes them on
its own** — not starting the server, not importing the package, not adding rows. The
same plan gives the same numbers today and next week.

This matters more than it sounds. When the fit was derived on demand, it was a function
of whatever the store happened to contain at that instant, so pressing Measure in one
window moved the coefficients under a plan already on screen in another, and a schema
bump re-derived `overhead_mib` on every stored row at import. Two runs of one config
disagreed with nothing in the config having changed, which is indistinguishable from a
bug in the model.

Only two things refit: pressing **Measure running model**, and `--recalibrate`.

```
python -m vram_planner --show-calibration    # the stored fit and where it came from
python -m vram_planner --recalibrate         # refit from stored rows, save, exit
```

A fit made under a different llama.cpp build, or by an older version of the fitter, is
**reported as outdated** in the terminal and the UI rather than silently replaced —
whether the numbers change is your call, not the tool's.


## Measuring it yourself

The accuracy numbers above are reproducible on your own hardware, and the model can be
re-fitted to it.

```
python -m vram_planner --sweep --dry-run     # what it will run, and for how long
python -m vram_planner --sweep               # hours; resumable, safe to interrupt
python -m vram_planner --fit                 # the held-out scorecard
```

`--sweep` finds a `llama-server` under LM Studio's `extensions/backends`, launches it
once per config with `-v`, and records the allocator's own buffer sizes, the
per-process VRAM counter, llama.cpp's layer placement and its SWA pattern — one JSON
row per load under `sweeps/`. A config that fails to allocate is recorded as `oom`
rather than dropped; a load taken with the card at the wall, where Windows spills to
shared memory and the counter reports the cap instead of the need, is detected and
excluded from fitting. `--probe MODEL ctx=A,B,C` runs an explicit ladder when something
in the grid does not interpolate.


## Known issues

- **Whether the token embeddings land in VRAM is measured, not understood.** On the
  three dense models swept they do, as soon as `-ngl` is 1, worth 626–843 MiB; on the
  two MoE models they do not. Charging them on dense models only takes the end-to-end
  error from 22.7% to 7.1%, and does it uniformly across all five, so it is clearly the
  right rule for these models — but the *mechanism* is unknown, and five models over
  three architectures is enough to measure the split and not to explain it. This is the
  first thing to re-check against a new architecture. It errs on the safe side for the
  case it is least sure of: over-charging VRAM makes a plan conservative, under-charging
  it makes the load spill.
- **The floor is the weakest term.** ~17 MiB mean absolute error, and the per-block rate
  genuinely differs by architecture (~10 MiB/layer on Gemma 4, ~2.5 on Qwen3.6-35B-A3B).
  It is also the term **Measure** is best placed to pin, being a property of the machine.
- **Non-CUDA backends are unvalidated.** See Supported platforms above.
- **Multi-GPU is not modelled.** `--tensor-split` is ignored; the plan targets GPU 0
  and the tool warns when it sees more than one card.
- **MLA is implemented but unmeasured.** DeepSeek-V2/V3 and Kimi cache a compressed
  latent, not per-head K and V; the width is read from the `attn_kv_a_mqa` tensor. No
  MLA model has been swept, so this is derived rather than validated.
- **Two measured anomalies I could not explain from outside the process.** On
  Qwen3.6-35B-A3B the recurrent state does not scale with `-np`, while Qwen3.6-27B's
  scales exactly as modelled (+88.0 MiB measured vs +87.3 predicted). And on
  Gemma 4 26B under *partial* offload, some blocks show no context-dependent cache
  growth at all. Neither affects full-offload planning, which is what these models are
  normally run with, and both are bounded.

- **Vision encoder transients are derived, not measured.** The projector's *weights*
  are exact (mmproj tensor bytes, charged off the top of the budget). Its
  *activations* — the pool the ViT needs while encoding an image — are computed from
  the tower's geometry and nothing else: no sweep backs them, so they are an order of
  magnitude rather than a prediction. They are also charged deliberately high (f32
  scores, attention unfused), because under-charging is the failure mode that makes a
  plan overcommit. Only reserved when you give an image size; without one the plan is
  text-only and says so.
