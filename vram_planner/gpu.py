"""Live hardware probes: VRAM, RAM, bandwidth, per-process usage."""
import os, subprocess, sys, time
from .const import MiB


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


_BW_CACHE = {"when": 0.0, "val": None}

BW_CACHE_TTL = 120.0    # seconds


def cached_bandwidth(fresh=False):
    """get_bandwidth(), memoised for the planner path.

    analyze() runs in tight loops (calibration, sweeps) and the RAM probe shells
    out for up to 20s per call; a machine's peak bandwidth does not move within
    two minutes. Failures are cached too, so a driver that is down cannot put
    the whole planner on the slow probe path. The UI's refresh button and the
    /api/bandwidth endpoint call get_bandwidth() directly."""
    if not fresh and _BW_CACHE["val"] is not None and \
            (time.time() - _BW_CACHE["when"]) < BW_CACHE_TTL:
        return _BW_CACHE["val"]
    val = get_bandwidth()
    _BW_CACHE["when"], _BW_CACHE["val"] = time.time(), val
    return val


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


def gpu_shared_mib(pid):
    """Host memory this process's GPU work is reaching over PCIe, in MiB.

    The number that makes a WDDM demotion visible. When a process's dedicated
    allocations reach the driver's budget, Windows does not fail the request -
    it places some of it in system RAM and carries on. Dedicated usage then
    stops rising while the model keeps growing, and every token pays a PCIe
    round trip for whatever moved. That is why an ngl ladder can get SLOWER as
    it climbs while the row still says "ok".

    nvidia-smi cannot see this on WDDM. The GPU performance counters can, and
    the same counter set already backs get_gpu_processes' Windows fallback.

    Instances are per adapter and per segment, so one process can own several
    and they are summed. Returns None when the counters cannot be read - which
    is NOT the same as zero and must never be recorded as zero, because zero is
    the answer that means "measured, and nothing was demoted"."""
    if os.name != "nt":
        return None
    ps = (r"try { $c = Get-Counter '\GPU Process Memory(*)\Shared Usage' -EA Stop }"
          r" catch { 'NA'; exit };"
          r"$s = ($c.CounterSamples | Where-Object {"
          r" ($_.InstanceName -split '_')[1] -eq '%s' } |"
          r" Measure-Object CookedValue -Sum).Sum;"
          # No matching instance means the process holds nothing shared, which is
          # a real zero. Only a failure to READ the counters is unknown.
          r"if ($null -eq $s) { '0' } else { [math]::Round($s/1MB, 1) }" % str(pid))
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            stderr=subprocess.DEVNULL, timeout=25).decode("utf-8", "replace").strip()
    except Exception:
        return None
    try:
        return float(out.splitlines()[-1].strip())
    except (ValueError, IndexError):
        return None


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
