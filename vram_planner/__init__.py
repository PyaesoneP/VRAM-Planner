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

Run:   python -m vram_planner              (opens a browser)
       python -m vram_planner --port 8100 --no-browser
       python -m vram_planner --self-test  (validates the parser + math)

Layout. Each module may import only from those above it, so the import graph is
a DAG and the numbers have one home each:

    const     version, byte units
    gguf      the binary reader
    model     config extraction, tensor classification
    kv        cache geometry (exact)
    compute   the compute buffer (the one fitted term)
    paths     user data locations
    gpu       live hardware probes
    lmstudio  models, runtime config, server logs, backends
    speed     bandwidth roofline
    calib     fitting compute/ to this machine's measurements
    plan      analyze() and the layer-split planners
    web       UI and JSON endpoints
    selftest  synthetic GGUFs and the test suite
    cli       entry point

Two edges run backwards and are imported inside the function that needs them:
compute.compute_buffer_terms reads calib.calib_coeffs, and calib.record_calibration
calls plan.analyze. Both are noted at the call site.

`import vram_planner as v` re-exports the whole public surface, so v.analyze(),
v.load_gguf(), v.get_gpus() and friends keep working exactly as before.
"""
from .const import __version__, MiB, GiB, _mib
from .gguf import (GGML_TYPES, load_gguf, parse_meta_only, _find_shards, _parse_one,
                   _r, _read_str, _read_value)
from .model import extract_config, classify_tensors, SWA_STRIDE_BY_ARCH, _as_int
from .kv import (KV_TYPE_BYTES, KV_PAD_FA, KV_PAD_NOFA, kv_bytes_per_token_layer,
                 is_swa_layer, swa_cache_len, kv_cache_len_layer, resolve_kv_lengths,
                 kv_bytes_layer, kv_bytes_total, kv_bytes_total_at,
                 max_ctx_for_kv_budget, kv_bytes_per_token,
                 kv_bytes_per_token_growing, recurrent_bytes)
from .compute import (CB_DEFAULTS, CB_GRAPH_CONST_MIB, CB_CONST_PER_KHID,
                      CB_CUDA_CTX_MIB, CB_ACT_PER_HIDDEN, CB_CTX_PER_TOKEN,
                      CB_MASK_PER_UB_TOK, CB_CTX_QUANT_BYTES, CB_SPLIT_GRAPH_MIB,
                      CB_SPLIT_PER_TOKEN, CB_NOFA_HEAD_BYTES, MTP_SPEC_CONST_MIB,
                      MTP_SPEC_PER_SEQ_MIB, compute_buffer_terms, compute_buffer_split,
                      compute_buffer_mib, graph_is_split, output_head_on_gpu)
from .paths import _data_dir, _user_file
from .gpu import (get_gpus, gpu_list, platform_support, get_bandwidth,
                  get_gpu_processes, get_ram)
from .lmstudio import (scan_speed_history, scan_server_logs, benchmark_server,
                       save_benchmark, load_benchmarks, match_speed_history,
                       read_lmstudio_runtime, resolve_runtime_ngl, detect_backend,
                       current_backend, default_models_dir, scan_models,
                       _lmstudio_home, REF_GPU, REF_BACKEND)
from .speed import GPU_EFF, RAM_EFF_HI, RAM_EFF_LO, per_token_bytes, estimate_speed
from .calib import (CALIB_TERMS, CALIB_SCHEMA, CALIB_MIN_SPREAD, CALIB_MIN_FREE_MIB,
                    CALIB_FLAT_RATIO, CALIB_FLAT_MIN_MIB, calib_coeffs,
                    load_calibration, save_calibration, migrate_calibration,
                    mark_unreliable, fit_calibration, refresh_calibration,
                    calibration_status, record_calibration, _calib_store, _row_flags,
                    _design, _struct_offset, _active_gpu)
from .plan import analyze, find_mmproj
from .web import serve, Handler
from .selftest import self_test
from .cli import main
