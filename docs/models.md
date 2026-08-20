# Model families, and what changes per family

Sliding-window, hybrid, multimodal and MoE models each break a different assumption
in the naive arithmetic. Here is what the planner does about each — and the stored
cards that let it plan a model you no longer have on disk.

[← back to the README](../README.md)

---

## Sliding-window attention (Gemma, Mistral, gpt-oss, Cohere2 ...)

Most recent long-context models do **not** grow a full KV cache on every layer. They
interleave a few *global* full-attention layers with many *windowed* ones capped at a
fixed span, and llama.cpp allocates two caches accordingly: a full `n_ctx` one for the
global layers, and a small ring buffer sized by the window for the rest. The windowed
layers also often use smaller head dims than the global ones.

The planner detects this from whatever signal the file carries, most explicit first:

1. `*.attention.sliding_window_pattern` as a per-layer 0/1 list (Gemma 4),
2. `*.attention.layer_types` string list,
3. a stride integer, or `*.full_attention_interval`,
4. `SWA_STRIDE_BY_ARCH` for architectures where llama.cpp hardcodes the stride
   (gemma2/3/3n, cohere2, gpt-oss, llama4, exaone4, hunyuan-moe),
5. an explicit stride of 1 still means "every layer is windowed", and
6. a window declared on an **unknown** architecture with no pattern anywhere is
   **not** assumed to be fully windowed any more — that guess can understate KV by
   10-20x at long context, and this tool's job is to say whether something fits.
   It charges every layer full and reports the window as ignored, so a plan errs
   conservative when the pattern is genuinely unknown.

A file with no window size falls through unchanged. Head dims come from
`key_length_swa` / `value_length_swa` where present.

**This is worth a lot.** Gemma 4 31B at 131k context, KV at q8_0:

| | naive (all layers full) | actual |
|---|---|---|
| 50 windowed layers | 106,000 MiB | ~638 MiB |
| 10 global layers | 8,240 MiB | 5,440 MiB |
| **total** | **114,240 MiB** | **6,078 MiB** |

Because the windowed layers are constant in context length, the KV-vs-context table is
**not a straight line** — past the window, only the global layers grow. Verified
against real llama.cpp: an `-ngl 4 -> 8` step predicted a 1231.8 MiB delta and measured
1232.0 MiB.


## Hybrid (attention + SSM) models
Qwen3.5/3.6 (`qwen35`), Qwen3-Next, Falcon-H, Jamba, Granite-4 and friends only run
**full attention on every Nth block** — the rest are linear/SSM blocks. Only the
attention blocks grow a KV cache with context; the SSM blocks hold a small
**fixed** recurrent state (conv + SSM state, f32, one copy per parallel sequence).

Qwen3.6-27B, for example, is 64 blocks but only **16** of them (every 4th) have KV.
Treating all 64 as KV-bearing overstates the cache by 4x — 17.0 GiB instead of
4.25 GiB at 128k/q8_0. The planner reads which blocks have `attn_k`/`attn_v` vs
`ssm_*` tensors straight from the tensor table, so it gets this right without an
architecture lookup table.

Because blocks are not interchangeable on these models (and llama.cpp offloads the
**last** `-ngl` blocks), the split is computed block by block rather than from a
per-layer average.


## Multimodal models and "why doesn't this match LM Studio's model size?"
Two separate things make the numbers look different:

1. **GiB vs GB.** The planner reports GiB (1024³). LM Studio reports the same bytes
   as GiB in the model panel and as GB (10⁹) in the loaded-models list, both
   labelled "GB". `17,612,564,704 bytes` = **16.40 GiB** = **17.61 GB**.
2. **The projector is part of what LM Studio calls the model.** A multimodal model
   ships `mmproj-*.gguf` next to the weights. LM Studio loads it onto the GPU and
   folds it into the size it displays.

For Qwen3.6-27B-UD-Q4_K_XL:

| | GiB | GB |
|---|---|---|
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` | 16.40 | 17.61 |
| `mmproj-F32.gguf` | 1.72 | 1.84 |
| **bundle — LM Studio's "model size"** | **18.12** | **19.46** |

The planner now finds the sibling projector, charges its **1,758 MiB** to VRAM
before planning the layer split, and shows both the model and the bundle size in
GiB and GB. Untick **Load vision projector (mmproj)**, under **Tune**, if you run
text-only.


## MoE models: two different knobs, and LM Studio may ignore both

- `-ngl N` puts the **last N blocks** on the GPU — attention, KV and experts together.
- `--n-cpu-moe M` moves only the **routed experts of the first M blocks** to the CPU,
  leaving their attention and KV on the GPU. LM Studio 0.4.x calls this
  *"Number of layers to keep experts on CPU"*.

The efficient config is `-ngl 999` plus the smallest `--n-cpu-moe` that fits: experts
are the only weights big enough to be worth moving, and only a few of them run per
token. Whole-block offload is the fallback for when even that won't fit. The planner
searches in that order and sums real per-block expert bytes, not an average.

**LM Studio silently overrides the sliders.** Check
`%APPDATA%\LM Studio\logs\main.log` for `Resolved GPU config options` — the
`Num Offload Layers` / `Num CPU Expert Layers` there is what actually ran. It also
sizes that decision using **8192 tokens**, not your context length:

```
Not using full context length for VRAM overflow calculations due to single GPU setup.
Instead, using '8192' as context length. Original context length: '262144'.
GPU offload layers was adjusted from 'max' to '16' to respect the strict GPU VRAM cap.
```

Put those two resolved numbers into **GPU layers** and **CPU expert layers** under
**Expert** to reproduce a run exactly, instead of what the UI displays.


## Model cards — planning without the weights

`analyze()` touches the `.gguf` for exactly three things: the config, the per-layer
tensor byte sums, and the projector next to it. All three are pure functions of the
header and tensor table, and together they are a few kilobytes. Everything after them is
arithmetic.

So they get cached as a **model card**, automatically, every time a model is read — about
8 KB each. When the file is gone, the card stands in:

```
python -m vram_planner --cards                 # what is stored
python -m vram_planner --add-card path/to.gguf # record one explicitly
python -m vram_planner --forget-card NAME.gguf # drop one
```

This means you can plan for a model you have **deleted**, or one you have not downloaded
yet — copy its card in — and on a machine that never held the weights at all. Cards for
models not on disk appear in the UI dropdown marked `○ … stored card, not on disk`, and
any plan built from one carries a warning saying so.

A card is not an approximation. It is the same three structures the file would have
produced, so a plan from a card is **identical to the plan from the file** — the
self-test asserts equality across every plan, config and speed key, not a sampled few.
The `CARD` check exists because the failure mode is silent: JSON has no integer keys, so
`per_layer_bytes` round-trips as `{"0": n}` and every lookup misses, which reads as a
model with no layers rather than as an error.

Cards are keyed by **file name**. Two genuinely different models sharing one name is the
single thing this cannot survive; a name whose size no longer matches is treated as stale
and rebuilt from the file.

They also rescue calibration rows. A stored measurement whose model has since been
deleted used to be stranded permanently on the next schema bump, because `overhead_mib`
could not be re-derived without the file. With a card it can.
