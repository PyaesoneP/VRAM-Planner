# Qwen3.8-27B-UD-Q4_K_XL on RTX 5070 Ti Laptop (12 GB) — speed findings

Measured 2026-08-18/19 with the tool's speed sweep (`llama.cpp 2.28.2`, CUDA 12, avx2).
Card `gpu_total_mib` 12227; ~500 MiB held by other processes, so the effective
budget is ~11.7 GB. All rows: draft-mtp, mmproj in RAM, KV/spec-KV q8_0,
ubatch 512, fill 2048, reasoning on (xhigh), the campaign's frozen corpus
(prompt `f3bef4861fc1`).

Rows live in
`C:\Users\pyaes\AppData\Local\vram-planner\speed\NVIDIA_GeForce_RTX_5070_Ti_Laptop_GPU__llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.28.2.jsonl`.

## What `n_cpu_ffn` does

`n_cpu_ffn=65` pins **every block's entire FFN** (gate/up/down, ~5.3 GB) to the
CPU via `-ot blk.(0..64).ffn_(gate|up|down).weight=CPU`. Attention, norms and KV
stay on the GPU. All rows with `-ot` carry the tool's `suspect`/`spilled` flags -
that is a known false positive ("the fit model cannot price dense FFN tensors on
the CPU"); every flagged row shown below verified clean (free VRAM counters
stable before/after, `free 11572 -> 11572`).

## Speed king (new, 2026-08-19)

All 65 layers + KV on the GPU (ngl 99 == ngl 65, clamped), FFN in RAM, context
walked down until it fits:

| ctx | status | tok/s | accept | VRAM used |
|-----|--------|-------|--------|-----------|
| 262144 | OOM | | | |
| 196608 | OOM | | | |
| 131072 | OOM | | | |
| **98304** | **fits** | **9.29** | 54% | 11148 MiB |
| 65536 | fits | 9.58 | 54% | 9672 MiB |
| 32768 | fits | 9.54 | 54% | 8196 MiB |

- The wall is **ctx 98304** (131072 OOMs; ~500 MiB of headroom left).
- Speed plateaus once the KV is small: 65536 and 32768 are the same.
- This is the fastest config measured on this card: 9.29 > 9.25 (old winner),
  +2.1 tok/s over the max-context config.

## The three candidate configs

| config | ctx | ngl | ffn | tok/s | VRAM | use for |
|--------|-----|-----|-----|-------|------|---------|
| speed king | 98304 | 99 | 65 | **9.29** | 11148 | fast chat |
| old winner | 131072 | 28 | 40 | 9.25 | 11370 | balanced |
| context king | 262144 | 30 | 65 | 7.19 | 11223 | max context (model max 262144) |

## Sweep history

### Baseline
- `ngram-mod`: **6.09 tok/s**.

### FFN sweep at ngl 28 (draft-mtp, n_max 2)
- ffn 0-32: OOM. ffn 40 is the least offload that fits.

| n_cpu_ffn | tok/s | notes |
|-----------|-------|-------|
| 0-32 | OOM | |
| **40** | **9.25** | VRAM 11370, accept 66% — old winner |
| 48 | 7.92 | |
| 56 | 7.28 | |
| 64 | 6.84 | |
| 65 | 6.19 | |

- OOM walks (each ffn OOM'd at ngl 28, row recorded one rung freer):
  ffn 0→ngl 27 (2.81, suspicious), 8→26 (7.81), 16→25 (7.81), 24→24 (7.93),
  32→23 (7.15).
- Depth ladder at ngl 23/ffn 40: n_max 3 = 7.08, n_max 5 = 6.12 (n_max 2 wins).

### ffn 65, ngl ladder at ctx 131072
- ngl 29-52 all fit: ~6.3-7.4 tok/s, flat; VRAM 7408 → 10912.
- Best ngl 52 = 7.41; ngl 56 = 2.84 (suspicious, re-measure). Wall ~58
  (ngl 60 OOM; walk 59/58 OOM; 57 genfail ConnectionReset).

### Max context test (ffn 65, ngl 28)
- ctx 196608 = 6.63 (VRAM 9186); **ctx 262144 = 6.36 (VRAM 11051) — model max
  fits**. No speed penalty for bigger context at this split.
- ngl ladder at ctx 262144: ngl 29 = 6.81 (11133), ngl 30 = 7.19 (11223),
  ngl 31 OOM — the context king.

## KV cache
- KV is pinned to the GPU by design ("KV on GPU, FFN in RAM"); llama.cpp places
  each layer's KV on the layer's device. No KV-offload axis in the tool.
- KV is the wall driver in every measurement (e.g. ngl 65's KV ~8.4 GB at
  ctx 262144 + 3 GB model + 1 GB compute + 0.7 GB mmproj/spec > 12.2 GB).
- `--no-kv-offload` would be ~10x slower at 131072 ctx.

## `--vmm` / CUDA-graphs warning (checked 2026-08-19)
- The warning ("pass `--vmm no` or turn off CUDA graphs ... split-device FFN")
  comes from LM Studio's engine docs — **not llama.cpp**. Neither flag exists in
  this build (verified: `error: invalid argument` for `--vmm`,
  `--cuda-graphs off`, `--no-cuda-graphs`) nor in current master's arg table
  (zero occurrences of "vmm" in `common/arg.cpp`).
- "VMM" is internal ggml-cuda virtual-memory allocation (`GGML_USE_VMM`,
  the "VMM: yes" device log line), not a CLI knob.
- CUDA graphs are compile-time (`USE_CUDA_GRAPH`) and **already auto-disabled**
  per graph when incompatible: multi-GPU, or the `mul_mat_id` fallback path —
  the path split-FFN (`-ot`) runs take ("the mul_mat_id fallback path
  synchronizes the stream, so we cannot use CUDA graphs").
- The old `LLAMACPP_DISABLE_CUDA_GRAPHS=1` env var (2024) is gone from current
  source; only `GGML_CUDA_GRAPH_OPT` remains (off by default).
- Two suspicious slow rows (2.8x tok/s, "host memory 538 MiB": ngl 27/ffn 0
  walk, ngl 56/ffn 65) are likely busy-moment readings, not a graph conflict;
  re-measure before trusting either.
