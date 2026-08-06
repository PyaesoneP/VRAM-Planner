"""Drive llama-server across a config grid and record what the allocator reports.

The compute buffer is the one term this tool estimates, and until now it was fitted
against 44 rows typed in by hand from one laptop. That is not enough data to hold
any of it out, so the reported error has always been a fit residual rather than a
prediction error. This module removes the typing: it launches llama-server itself,
reads the numbers the allocator prints under -v, and writes one JSON row per config.

Two numbers come out of every run and they do different jobs - the same division of
labour compute.py already relies on:

  * `CUDA0 compute buffer size` is the allocator's own report. It is exact, and it
    fixes the SLOPES.
  * total process VRAM minus everything the allocator reports is the floor - CUDA
    context, driver, lazily-loaded kernel modules - which no log can see. It fixes
    the INTERCEPT.

A run that fails to allocate is recorded with status "oom" rather than dropped. An
OOM is a hard upper bound on what fits, and it is the one observation the old
Measure-button store could never contain: you cannot press Measure on a load that
never started.
"""
import json, os, re, subprocess, sys, time
from .const import MiB
from .gguf import load_gguf
from .model import extract_config
from .gpu import get_gpu_processes, gpu_list
from .lmstudio import default_models_dir
from .paths import _data_dir


# ---------------------------------------------------------------------------
# Finding a llama-server to drive
# ---------------------------------------------------------------------------
# LM Studio ships complete llama.cpp builds under extensions/backends, one
# directory per build, each with the build string in its name - the same string
# lmstudio.detect_backend() recovers from a running engine, so a sweep is tagged
# with exactly the build identity the calibration store keys on.
#
# The executable does NOT run from its own directory alone: the CUDA runtime DLLs
# live in a shared vendor package named by the build's backend-manifest.json, and
# without it on PATH the process dies with STATUS_DLL_NOT_FOUND (exit
# -1073741515) and no message at all. That is worth stating plainly because it
# looks exactly like a crash on startup.
BACKENDS_SUBDIR = os.path.join(".lmstudio", "extensions", "backends")

EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


def backends_dir():
    return os.path.join(os.path.expanduser("~"), BACKENDS_SUBDIR)


def find_backends(root=None):
    """Every llama.cpp build on this machine that we can actually launch.

    Returns a list of {build, exe, dir, vendor, gpu} newest-looking first. `build`
    is the directory name, which is the same string detect_backend() reports."""
    root = root or backends_dir()
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        exe = os.path.join(d, EXE_NAME)
        if not os.path.isfile(exe):
            continue
        vendor, gpu = [], ""
        try:
            man = json.load(open(os.path.join(d, "backend-manifest.json"),
                                 encoding="utf-8"))
            gpu = ((man.get("gpu") or {}).get("framework") or "").lower()
            for pkg in man.get("vendor_lib_package_names") or []:
                v = os.path.join(root, "vendor", pkg)
                if os.path.isdir(v):
                    vendor.append(v)
        except Exception:
            pass
        out.append({"build": name, "exe": exe, "dir": d, "vendor": vendor, "gpu": gpu})
    # prefer CUDA builds, then the highest version string
    out.sort(key=lambda b: (b["gpu"] != "cuda", _ver_key(b["build"])), reverse=False)
    out.sort(key=lambda b: (b["gpu"] != "cuda", [-x for x in _ver_key(b["build"])]))
    return out


def _ver_key(name):
    m = re.search(r"(\d+)\.(\d+)\.(\d+)\s*$", name)
    return [int(x) for x in m.groups()] if m else [0, 0, 0]


def pick_backend(want=None):
    """The build to sweep with: the one named, else the newest CUDA build."""
    bs = find_backends()
    if not bs:
        return None
    if want:
        return next((b for b in bs if b["build"] == want or want in b["build"]), None)
    return bs[0]


def _env_for(backend):
    env = dict(os.environ)
    parts = [backend["dir"]] + list(backend["vendor"]) + [env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(p for p in parts if p)
    # llama.cpp colours its log when it thinks it has a terminal; we are parsing it
    env["NO_COLOR"] = "1"
    env["LLAMA_LOG_COLORS"] = "0"
    return env


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------
# Every line carries a timestamp, a level and the emitting function before the
# content - `0.16.860.769 I sched_reserve:      CUDA0 compute buffer size = 336.25
# MiB` - so nothing here anchors at the start of a line.

# Matches every "<backend> <kind> buffer size = N MiB" line generically rather than
# naming CUDA0/CPU/CUDA_Host, because which backends appear depends on the build
# and on how the graph was split.
RE_BUF     = re.compile(r"(\S+)\s+(model|compute|KV|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB")

RE_KVCACHE = re.compile(r"llama_kv_cache:\s*size\s*=\s*([\d.]+) MiB \(\s*(\d+) cells,"
                        r"\s*(\d+) layers,\s*(\d+)/\s*(\d+) seqs\)")

# The ground truth the SWA heuristics in model.py have to guess at: llama.cpp says
# outright which device each layer landed on and whether it is a sliding-window
# layer. A sweep row therefore carries the real pattern, not SWA_STRIDE_BY_ARCH's
# opinion of it.
RE_LAYER   = re.compile(r"load_tensors:\s*layer\s+(\d+) assigned to device (\S+), is_swa = (\d)")

RE_OFFL    = re.compile(r"load_tensors:\s*offloaded (\d+)/(\d+) layers to GPU")

RE_CTXVAL  = re.compile(r"llama_context:\s*(n_ctx|n_batch|n_ubatch)\s*=\s*(\d+)")

RE_FA      = re.compile(r"llama_context:\s*flash_attn\s*=\s*(\w+)")

RE_NODES   = re.compile(r"graph nodes\s*=\s*(\d+)")

RE_SPLITS  = re.compile(r"graph splits\s*=\s*(\d+)")

# re.M matters: without it `$` only matches the end of the whole log and this finds
# exactly nothing. Keys are taken up to the `=` rather than as one token because
# several of them contain a space ("model type", "model params").
RE_INFO    = re.compile(r"print_info:\s*(.+?)\s*=\s*(.+?)\s*$", re.M)

RE_READY   = re.compile(r"llama_server: (model loaded|listening on http)")

# llama.cpp does not use one phrasing for a failed allocation, and some of these
# arrive from the CUDA runtime rather than from llama.cpp at all.
RE_OOM     = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|"
                        r"unable to allocate|ggml_backend_.*alloc.*failed|"
                        r"failed to create context|error loading model", re.I)

INFO_KEYS = ("arch", "n_layer", "n_embd", "n_head", "n_head_kv", "n_ff", "n_expert",
             "n_expert_used", "n_swa", "n_embd_head_k", "n_embd_head_v",
             "n_vocab", "model type", "model params")


def parse_log(text):
    """Everything worth keeping from one llama-server startup."""
    out = {"buffers": {}, "kv_caches": [], "layers": {}, "swa_layers": [],
           "info": {}, "ready": False, "oom": False}
    for m in RE_BUF.finditer(text):
        # a backend can report the same kind twice; keep the largest
        k = "%s.%s" % (m.group(1), m.group(2))
        v = float(m.group(3))
        out["buffers"][k] = max(out["buffers"].get(k, 0.0), v)
    # llama.cpp prints the cache summary once per reserve pass, so the same cache
    # shows up several times; the set of distinct caches is what we want.
    for m in RE_KVCACHE.finditer(text):
        kc = {"mib": float(m.group(1)), "cells": int(m.group(2)),
              "layers": int(m.group(3)), "seqs": int(m.group(5))}
        if kc not in out["kv_caches"]:
            out["kv_caches"].append(kc)
    for m in RE_LAYER.finditer(text):
        li, dev, swa = int(m.group(1)), m.group(2), m.group(3) == "1"
        out["layers"][li] = dev
        if swa and li not in out["swa_layers"]:
            out["swa_layers"].append(li)
    m = RE_OFFL.search(text)
    if m:
        out["offloaded"], out["n_layer_all"] = int(m.group(1)), int(m.group(2))
    for m in RE_CTXVAL.finditer(text):
        out[m.group(1)] = int(m.group(2))
    m = RE_FA.search(text)
    if m:
        out["flash_attn"] = (m.group(1) == "enabled")
    for rx, key in ((RE_NODES, "graph_nodes"), (RE_SPLITS, "graph_splits")):
        m = rx.search(text)
        if m:
            out[key] = int(m.group(1))
    for m in RE_INFO.finditer(text):
        if m.group(1) in INFO_KEYS:
            out["info"][m.group(1)] = m.group(2)
    out["swa_layers"].sort()
    out["ready"] = bool(RE_READY.search(text))
    # An OOM that happened *before* the server came up is a failed run; the same
    # words appearing after it is up are something else's problem.
    head = text.split("llama_server: model loaded")[0]
    out["oom"] = bool(RE_OOM.search(head)) and not out["ready"]
    return out


# A device buffer is one whose backend name ends in a device index - CUDA0, ROCm1.
# CUDA_Host does NOT match, and that is the point: it is pinned host memory, not
# VRAM, despite the name. Counting it as GPU is an easy and expensive mistake to
# make when reading these logs.
RE_DEVBUF = re.compile(r"^(CUDA|ROCm|Vulkan|Metal|SYCL)\d+\.")


def gpu_side(buffers):
    """Split the allocator's report into the part that lives in VRAM and the rest."""
    gpu = sum(v for k, v in buffers.items() if RE_DEVBUF.match(k))
    host = sum(v for k, v in buffers.items() if not RE_DEVBUF.match(k))
    return round(gpu, 2), round(host, 2)


# ---------------------------------------------------------------------------
# Running one config
# ---------------------------------------------------------------------------
def build_argv(exe, model_path, c, port):
    """The full command line for one config.

    -fa takes on|off|auto in current builds, not a bare boolean, and passing it as
    a flag silently swallows the next argument."""
    av = [exe, "-m", model_path,
          "-c", str(c["ctx"]), "-ub", str(c["ub"]), "-np", str(c["seq"]),
          "-ngl", str(c["ngl"]),
          "-fa", "on" if c["fa"] else "off",
          "-ctk", c["kv"], "-ctv", c["kv"],
          "--host", "127.0.0.1", "--port", str(port),
          "--no-warmup",          # we want the allocation, not a generated token
          "--cache-ram", "0",     # host-side prompt cache; noise for our purposes
          "-v"]
    if c.get("ncmoe"):
        av += ["--n-cpu-moe", str(c["ncmoe"])]
    return av


def run_one(backend, model_path, c, port=8231, timeout=420.0, settle=2.0, log_dir=None):
    """Launch one config, read what the allocator reports, kill it.

    stderr goes to a file rather than a pipe: llama.cpp emits thousands of lines
    under -v and a pipe nobody drains fills its buffer and deadlocks the child
    somewhere in the middle of loading."""
    log_dir = log_dir or os.path.join(_data_dir(), "sweep-logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    row = {"status": "error", "config": dict(c),
           "model": os.path.basename(model_path), "backend": backend["build"]}

    free_before = _free_mib()
    t0 = time.time()
    with open(log_path, "wb") as fh:
        kw = {}
        if os.name == "nt":
            # so terminate() cannot take our own console down with it
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            p = subprocess.Popen(build_argv(backend["exe"], model_path, c, port),
                                 stdout=subprocess.DEVNULL,
                                 stderr=fh, cwd=backend["dir"], env=_env_for(backend), **kw)
        except Exception as e:
            row["error"] = "launch failed: %s" % e
            return row

        proc_vram, parsed = None, {}
        try:
            while True:
                if time.time() - t0 > timeout:
                    row["status"], row["error"] = "timeout", "no ready line in %.0fs" % timeout
                    break
                rc = p.poll()
                text = _read(log_path)
                parsed = parse_log(text)
                if parsed["ready"]:
                    # let the allocator settle before reading the process counter
                    time.sleep(settle)
                    proc_vram = _proc_vram(p.pid)
                    row["status"] = "ok"
                    break
                if rc is not None:
                    row["status"] = "oom" if parsed["oom"] else "exit"
                    row["exit_code"] = rc
                    break
                time.sleep(0.5)
        finally:
            _kill(p)

    row["load_s"] = round(time.time() - t0, 1)
    row["log"] = parse_log(_read(log_path)) if not parsed else parsed
    if row["status"] == "ok":
        gpu_alloc, host_alloc = gpu_side(row["log"]["buffers"])
        row["alloc_gpu_mib"] = gpu_alloc
        row["alloc_host_mib"] = host_alloc
        row["proc_vram_mib"] = proc_vram
        # The floor: what the process holds that the allocator never reports.
        # Negative would mean the counter read low, which happens if we sampled
        # before allocation settled - keep it, do not clamp, so it shows up as bad
        # data rather than as a plausible small number.
        row["floor_mib"] = (round(proc_vram - gpu_alloc, 1)
                            if proc_vram is not None else None)
    row["gpu_free_before_mib"] = free_before
    row["gpu_free_after_mib"] = _free_mib()
    row["gpu_total_mib"] = _total_mib()
    row["suspect"] = suspect_reason(row)
    if log_dir and row["status"] != "ok":
        # keep the evidence for anything that did not work
        try:
            os.replace(log_path, os.path.join(log_dir, "fail-%d.log" % int(time.time())))
        except OSError:
            pass
    return row


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# A load that did not fit still starts on Windows: WDDM spills the remainder to
# shared system memory instead of failing, and the dedicated-memory counter then
# reports the cap rather than the need. The tell is unmistakable once you know it -
# the allocator asked for more than the process is holding, so the floor computes
# NEGATIVE, sometimes by gigabytes. Such a row is not a weak measurement, it is a
# measurement of the card's size, and fitting it drags every coefficient with it.
FLOOR_MIN_MIB = -32.0             # a little slack for counter jitter

FLOOR_MAX_MIB = 4096.0            # a floor larger than this is not a CUDA context


def suspect_reason(row):
    """Why this row should not be fitted, or "" if it looks sound."""
    if row.get("status") != "ok":
        return ""
    fl = row.get("floor_mib")
    if fl is None:
        return "per-process VRAM could not be read"
    if fl < FLOOR_MIN_MIB:
        return ("allocator asked for %.0f MiB more than the process holds - the card "
                "was at the wall and spilled to shared memory, so this reads the cap, "
                "not the need" % -fl)
    if fl > FLOOR_MAX_MIB:
        return "floor of %.0f MiB is too large to be a CUDA context" % fl
    free = row.get("gpu_free_after_mib")
    if free is not None and free < 192.0:
        return "only %.0f MiB of VRAM was free at measure time" % free
    return ""


def _free_mib():
    g = gpu_list(fresh=True)
    return g[0].get("free_mib") if g else None


def _total_mib():
    g = gpu_list(fresh=True)
    return g[0].get("total_mib") if g else None


def _proc_vram(pid):
    """Dedicated VRAM for our child, by pid. Matching on pid rather than on process
    name matters: LM Studio's own engine is also called llama-server, and reading
    its footprint instead of ours would be silent and completely wrong."""
    procs = get_gpu_processes()
    if isinstance(procs, dict):
        return None
    for p in procs:
        if str(p.get("pid")) == str(pid):
            return p["mib"]
    return None


def _kill(p):
    if p.poll() is not None:
        return
    try:
        p.terminate()
        p.wait(timeout=20)
    except Exception:
        try:
            p.kill()
            p.wait(timeout=20)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
# One knob at a time from a baseline, not a cross product. calib.py's spread check
# (CALIB_MIN_SPREAD) exists because a confounded sweep - several models at one
# ubatch, where hidden size moves but the knob does not - fitted the activation
# coefficient at 9x its prior. A one-at-a-time design keeps each coefficient
# identifiable from the rows that actually vary it, and costs O(sum of axes)
# loads instead of O(product).
CTX_AXIS   = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

UB_AXIS    = [64, 128, 256, 512, 1024, 2048]

SEQ_AXIS   = [1, 2, 4, 8]

KV_AXIS    = ["f16", "q8_0"]

BASE = {"ctx": 32768, "ngl": 2, "ub": 512, "seq": 1, "fa": True, "kv": "f16", "ncmoe": 0}


def model_facts(path):
    cfg = extract_config(load_gguf(path))
    return {"arch": cfg.get("arch") or "", "n_layers": cfg.get("n_layers") or 0,
            "n_ctx_train": cfg.get("n_ctx_train") or 0, "is_moe": bool(cfg.get("n_expert")),
            "hidden": cfg.get("hidden") or 0}


def build_grid(facts, base=None, max_ctx=None):
    """Configs for one model: a baseline, each axis swept, then a few corners."""
    b = dict(base or BASE)
    nl = facts["n_layers"] or 32
    ctx_cap = min(x for x in [max_ctx, facts["n_ctx_train"] or 10 ** 9] if x)
    b["ctx"] = min(b["ctx"], ctx_cap)
    b["ngl"] = min(b["ngl"], nl)

    ngl_axis = sorted({0, 1, 2, 4, 8, max(1, nl // 4), max(1, nl // 2),
                       max(1, nl - 1), nl})
    ngl_axis = [n for n in ngl_axis if n <= nl]

    cfgs, seen = [], set()

    def add(**kw):
        c = dict(b); c.update(kw)
        c["ctx"] = min(c["ctx"], ctx_cap)
        c["ngl"] = min(c["ngl"], nl)
        if not facts["is_moe"]:
            c["ncmoe"] = 0
        # --n-cpu-moe only means anything with the blocks actually on the GPU
        if c.get("ncmoe"):
            c["ngl"] = nl
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k)
            cfgs.append(c)

    add()
    for v in CTX_AXIS:
        if v <= ctx_cap:
            add(ctx=v)
    for v in ngl_axis:
        add(ngl=v)
    for v in UB_AXIS:
        add(ub=v)
    for v in SEQ_AXIS:
        add(seq=v)
    for v in KV_AXIS:
        add(kv=v)
    add(fa=False)
    # Flash attention off is the one axis whose term is quadratic in context, so it
    # needs its own short context ladder or the coefficient rides on one point.
    for v in (8192, 32768):
        if v <= ctx_cap:
            add(fa=False, ctx=v)
    if facts["is_moe"]:
        for v in sorted({1, 2, max(1, nl // 4), max(1, nl // 2), nl}):
            add(ncmoe=v, ngl=nl)
    # corners: the interactions a one-at-a-time design cannot see by construction
    for c in ({"ctx": min(131072, ctx_cap), "ub": 2048},
              {"ctx": min(131072, ctx_cap), "ngl": nl},
              {"ctx": min(8192, ctx_cap), "ngl": nl, "ub": 64},
              {"ctx": min(65536, ctx_cap), "kv": "q8_0", "seq": 4},
              {"ctx": min(32768, ctx_cap), "ngl": 0, "ub": 2048}):
        add(**c)
    return cfgs


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def sweep_path(gpu_name, build):
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s or "unknown")
    d = os.path.join(_data_dir(), "sweeps")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s__%s.jsonl" % (safe(gpu_name), safe(build)))


def _key(model, c):
    return (model, c["ctx"], c["ngl"], c["ub"], c["seq"], bool(c["fa"]),
            c["kv"], c.get("ncmoe") or 0)


def load_rows(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def discover_models(root=None, names=None):
    """GGUF files to sweep. Projectors are skipped - mmproj-*.gguf is not a model
    you can load on its own, and sweeping it just produces failed runs."""
    root = root or default_models_dir()
    out = []
    for dp, _dn, fns in os.walk(root or ""):
        for fn in sorted(fns):
            if not fn.lower().endswith(".gguf") or fn.lower().startswith("mmproj"):
                continue
            m = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", fn)
            if m and m.group(2) != "00001":
                continue
            if names and not any(n.lower() in fn.lower() for n in names):
                continue
            out.append(os.path.join(dp, fn))
    return out


def parse_overrides(specs):
    """"ctx=2048,4096 ngl=2" -> {"ctx": [2048, 4096], "ngl": [2]}."""
    out = {}
    for spec in specs or []:
        for part in spec.split():
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            vals = []
            for x in v.split(","):
                x = x.strip()
                if not x:
                    continue
                if k == "kv":
                    vals.append(x)
                elif k == "fa":
                    vals.append(x not in ("0", "off", "false"))
                else:
                    vals.append(int(x))
            if vals:
                out[k.strip()] = vals
    return out


def probe(model, axes, backend=None, timeout=420.0, port=8231, log=print):
    """Run an explicit cross product of configs and print the result immediately.

    The grid is a screening design and deliberately coarse. When it turns up
    something that does not interpolate - a plateau, a step, a value that sits
    above both its neighbours - the way to find out what it is, is a fine ladder
    through that region. Results still land in the sweep file, so a probe is also
    just more training data."""
    b = pick_backend(backend)
    if not b:
        log("No llama-server build found under %s" % backends_dir())
        return None
    paths = discover_models(names=[model] if isinstance(model, str) else model)
    if not paths:
        log("No model matching %r" % (model,))
        return None
    mp = paths[0]
    facts = model_facts(mp)
    base = dict(BASE)
    base["ngl"] = min(base["ngl"], facts["n_layers"] or base["ngl"])

    keys = [k for k in axes if k in base]
    combos = [dict(base)]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in axes[k]]

    gpu = (gpu_list(fresh=True) or [{}])[0].get("name") or ""
    path = sweep_path(gpu, b["build"])
    log("probe %s on %s: %d configs" % (os.path.basename(mp), b["build"], len(combos)))
    log("%-8s %-5s %-6s %-4s %-5s %-3s | %10s %10s %9s %7s"
        % ("ctx", "ngl", "ub", "np", "kv", "fa", "GPUcomp", "HOSTcomp", "floor", "splits"))
    out = []
    with open(path, "a", encoding="utf-8") as fh:
        for c in combos:
            row = run_one(b, mp, c, port=port, timeout=timeout)
            row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                        "when": int(time.time()), "gpu": gpu, "probe": True})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            out.append(row)
            if row["status"] != "ok":
                log("%-8d %-5d %-6d %-4d %-5s %-3d | %s %s"
                    % (c["ctx"], c["ngl"], c["ub"], c["seq"], c["kv"], c["fa"],
                       row["status"].upper(), row.get("error", "")))
                continue
            bufs = row["log"]["buffers"]
            log("%-8d %-5d %-6d %-4d %-5s %-3d | %10.2f %10.2f %9.1f %7s"
                % (c["ctx"], c["ngl"], c["ub"], c["seq"], c["kv"], c["fa"],
                   bufs.get("CUDA0.compute", 0.0), bufs.get("CUDA_Host.compute", 0.0),
                   row["floor_mib"] if row["floor_mib"] is not None else float("nan"),
                   row["log"].get("graph_splits")))
    return out


def sweep(models=None, backend=None, dry_run=False, out_path=None, timeout=420.0,
          port=8231, limit=None, log=print):
    """Run the grid for every model and append rows to the sweep file.

    Resumable by construction: the file is JSONL, appended one row at a time and
    flushed, and any config already present is skipped. A sweep is hours of loads;
    it has to survive being interrupted."""
    b = pick_backend(backend)
    if not b:
        log("No llama-server build found under %s" % backends_dir())
        return None
    gpu = (gpu_list(fresh=True) or [{}])[0].get("name") or ""
    path = out_path or sweep_path(gpu, b["build"])
    # Only a run that produced an answer counts as done. "ok" is an answer and so
    # is "oom" - that config genuinely does not fit. A timeout or a launch failure
    # is not, and must be retried rather than baked in as a permanent hole in the
    # grid by the very resume logic meant to protect the data.
    done = {_key(r["model"], r["config"]) for r in load_rows(path)
            if r.get("config") and r.get("status") in ("ok", "oom")}

    paths = models if models and os.path.isfile(str(models[0])) else \
        discover_models(names=models)
    if not paths:
        log("No models found under %s" % default_models_dir())
        return None

    plan, skipped = [], 0
    for mp in paths:
        try:
            facts = model_facts(mp)
        except Exception as e:
            log("  skip %s (%s)" % (os.path.basename(mp), e))
            continue
        for c in build_grid(facts):
            if _key(os.path.basename(mp), c) in done:
                skipped += 1
                continue
            plan.append((mp, facts, c))
    if limit:
        plan = plan[:limit]

    log("backend : %s" % b["build"])
    log("gpu     : %s" % (gpu or "unknown"))
    log("output  : %s" % path)
    log("models  : %d" % len(paths))
    log("configs : %d to run, %d already recorded" % (len(plan), skipped))
    # Loads observed at 17-120s depending on model size and how much lands on the
    # GPU; 60s per run is a usable middle for telling someone what they are in for.
    log("estimate : ~%.1f h at 60 s/load" % (len(plan) * 60.0 / 3600.0))
    if dry_run:
        for mp, _f, c in plan:
            log("  %-44s ctx %-7d ngl %-4d ub %-5d np %d fa %d %-5s ncmoe %d"
                % (os.path.basename(mp)[:44], c["ctx"], c["ngl"], c["ub"], c["seq"],
                   c["fa"], c["kv"], c.get("ncmoe") or 0))
        return {"planned": len(plan), "path": path, "dry_run": True}

    n_ok = n_oom = n_bad = 0
    with open(path, "a", encoding="utf-8") as fh:
        for i, (mp, facts, c) in enumerate(plan, 1):
            row = run_one(b, mp, c, port=port, timeout=timeout)
            row["arch"] = facts["arch"]
            row["n_layers"] = facts["n_layers"]
            row["when"] = int(time.time())
            row["gpu"] = gpu
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if row["status"] == "ok":
                n_ok += 1
                log("[%d/%d] %-30s ctx %-7d ngl %-4d ub %-5d %-5s -> gpu %8.2f  "
                    "floor %s  (%.0fs)"
                    % (i, len(plan), os.path.basename(mp)[:30], c["ctx"], c["ngl"],
                       c["ub"], c["kv"], row["alloc_gpu_mib"],
                       "%8.1f" % row["floor_mib"] if row["floor_mib"] is not None
                       else "       -", row["load_s"]))
            else:
                n_oom += row["status"] == "oom"
                n_bad += row["status"] not in ("ok", "oom")
                log("[%d/%d] %-30s ctx %-7d ngl %-4d -> %s %s"
                    % (i, len(plan), os.path.basename(mp)[:30], c["ctx"], c["ngl"],
                       row["status"].upper(), row.get("error", "")))
    log("done: %d ok, %d oom, %d failed -> %s" % (n_ok, n_oom, n_bad, path))
    return {"ok": n_ok, "oom": n_oom, "failed": n_bad, "path": path}
