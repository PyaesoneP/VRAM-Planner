"""Reading LM Studio: models, runtime config, server logs, backends."""
import glob, json, os, re, subprocess, time
from .const import MiB
from .gguf import SHARD_RE, _find_shards, parse_meta_only
from .model import _as_int
from .paths import _user_file
from .gpu import get_gpu_processes


def _lmstudio_home():
    for p in (os.environ.get("LMSTUDIO_HOME"),
              os.path.join(os.path.expanduser("~"), ".lmstudio"),
              os.path.join(os.path.expanduser("~"), ".cache", "lm-studio")):
        if p and os.path.isdir(p):
            return p
    return None


def scan_speed_history(limit=400):
    """Mine LM Studio's saved chats for real throughput measurements.

    Each generation records genInfo.stats (tokensPerSecond, numGpuLayers,
    timeToFirstTokenSec, prompt/predicted token counts) alongside the
    loadModelConfig it ran under - enough to match a measurement to a plan and
    calibrate against it."""
    home = _lmstudio_home()
    if not home:
        return {"error": "LM Studio directory not found"}
    conv = os.path.join(home, "conversations")
    if not os.path.isdir(conv):
        return {"error": "no conversations directory at %s" % conv}

    def cfgval(cfgobj, key, default=None):
        for f in (cfgobj or {}).get("fields", []):
            if f.get("key") == key:
                return f.get("value")
        return default

    recs = []
    try:
        files = sorted(glob.glob(os.path.join(conv, "*.conversation.json")),
                       key=os.path.getmtime, reverse=True)
    except Exception as e:
        return {"error": str(e)}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        when = os.path.getmtime(path)
        for msg in d.get("messages", []) or []:
            for ver in msg.get("versions", []) or []:
                for step in ver.get("steps", []) or []:
                    gi = step.get("genInfo") or {}
                    st = gi.get("stats") or {}
                    tps = st.get("tokensPerSecond")
                    if not tps:
                        continue
                    lc = gi.get("loadModelConfig") or {}
                    recs.append({
                        "model": gi.get("indexedModelIdentifier") or "?",
                        "tok_s": tps,
                        "n_gpu_layers": st.get("numGpuLayers"),
                        "ttft_s": st.get("timeToFirstTokenSec"),
                        "total_time_s": st.get("totalTimeSec"),
                        "prompt_tokens": st.get("promptTokensCount") or 0,
                        "predicted_tokens": st.get("predictedTokensCount") or 0,
                        "ctx": cfgval(lc, "llm.load.contextLength"),
                        "n_experts": cfgval(lc, "llm.load.numExperts"),
                        "cpu_expert_ratio": cfgval(lc, "llm.load.numCpuExpertLayersRatio"),
                        "offload_ratio": cfgval(lc, "llm.load.llama.acceleration.offloadRatio"),
                        "flash_attn": cfgval(lc, "llm.load.llama.flashAttention"),
                        "when": when,
                        "conversation": os.path.basename(path),
                    })
                    if len(recs) >= limit:
                        return recs
    return recs


RE_SRVLOG = re.compile(
    r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)\]\[INFO\]\[([^\]]+)\]\s+(.*)$")


def scan_server_logs(limit=300):
    """Server mode (OpenAI-compatible endpoint) never touches the chat history, so
    conversations/*.json stays empty for anything driven by an API client.

    The succinct server log still brackets each response:
        'Streaming response...'              -> t0
        'Prompt processing progress: 100.0%' -> prefill done
        'Finished streaming response'        -> t1
    That gives real prefill and decode wall times. It does NOT give token counts
    (LM Studio only logs those with verbose file logging on), so it cannot yield
    tok/s by itself - use it to see where the time goes, and Benchmark to measure."""
    home = _lmstudio_home()
    if not home:
        return {"error": "LM Studio directory not found"}
    root = os.path.join(home, "server-logs")
    if not os.path.isdir(root):
        return {"error": "no server-logs directory"}
    files = sorted(glob.glob(os.path.join(root, "*", "*.log")),
                   key=os.path.getmtime, reverse=True)[:14]
    import datetime
    out = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except Exception:
            continue
        cur = None
        for ln in lines:
            m = RE_SRVLOG.match(ln.strip())
            if not m:
                continue
            day, tm, model, msg = m.groups()
            try:
                ts = datetime.datetime.strptime(day + " " + tm, "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                continue
            if msg.startswith("Streaming response"):
                cur = {"model": model, "t0": ts, "t_prefill": None}
            elif cur and msg.startswith("Prompt processing progress: 100.0%"):
                cur["t_prefill"] = ts
            elif cur and msg.startswith("Finished streaming response"):
                pre = (cur["t_prefill"] - cur["t0"]) if cur["t_prefill"] else 0.0
                out.append({"model": cur["model"], "when": cur["t0"],
                            "prefill_s": round(pre, 1),
                            "decode_s": round(ts - (cur["t_prefill"] or cur["t0"]), 1),
                            "total_s": round(ts - cur["t0"], 1)})
                cur = None
                if len(out) >= limit:
                    break
    out.sort(key=lambda x: -x["when"])
    return out


def benchmark_server(base_url="http://localhost:1234", model=None, max_tokens=128,
                     timeout=180):
    """Measure the loaded model by asking it to generate. LM Studio's native
    /api/v0 returns a stats block with tokens_per_second and time_to_first_token;
    fall back to /v1 plus our own clock if that endpoint is not there.

    This runs a real generation on the loaded model, so it briefly uses the GPU."""
    import urllib.request, time
    def post(path, payload):
        req = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    if not model:
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/api/v0/models")
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            for m in d.get("data", []):
                if m.get("state") and m["state"] != "not-loaded":
                    model = m.get("id")
                    break
        except Exception as e:
            return {"error": "cannot reach %s: %s" % (base_url, e)}
    if not model:
        return {"error": "no model is loaded on %s" % base_url}

    body = {"model": model, "temperature": 0, "max_tokens": int(max_tokens),
            "stream": False,
            "messages": [{"role": "user",
                          "content": "Count from 1 to 100, one number per line."}]}
    t0 = time.time()
    try:
        d = post("/api/v0/chat/completions", body)
    except Exception:
        try:
            d = post("/v1/chat/completions", body)
        except Exception as e:
            return {"error": "benchmark request failed: %s" % e}
    wall = time.time() - t0
    st = d.get("stats") or {}
    usage = d.get("usage") or {}
    pred = st.get("predicted_tokens_count") or usage.get("completion_tokens") or 0
    prompt = st.get("prompt_tokens_count") or usage.get("prompt_tokens") or 0
    tps = st.get("tokens_per_second")
    ttft = st.get("time_to_first_token")
    if not tps and pred and wall > 0:
        # no stats block: approximate, subtracting prefill if we know it
        tps = pred / max(1e-6, wall - (ttft or 0))
    return {"model": model, "tok_s": tps, "ttft_s": ttft,
            "predicted_tokens": pred, "prompt_tokens": prompt,
            "wall_s": round(wall, 2), "source": "stats" if st else "wall-clock",
            "when": time.time()}


def _bench_store():
    return _user_file("speed_history.json")


def save_benchmark(rec):
    """Persist benchmarks next to the script so they survive restarts - server-mode
    users have no chat history to mine, so this becomes their history."""
    path = _bench_store()
    try:
        cur = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    except Exception:
        cur = []
    cur.insert(0, rec)
    try:
        json.dump(cur[:200], open(path, "w", encoding="utf-8"), indent=1)
    except Exception:
        pass
    return cur[:200]


def load_benchmarks():
    path = _bench_store()
    try:
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    except Exception:
        return []


def _norm_model(s):
    s = re.sub(r"\.gguf$", "", (s or ""), flags=re.I)
    s = re.sub(r"[-_.\s]+", "", s.lower())
    return s


def match_speed_history(recs, model_name, model_path=None):
    """Score history records against the model being planned. Exact-ish name
    match first; anything sharing a long substring counts as a weak match."""
    if not isinstance(recs, list):
        return []
    keys = [_norm_model(model_name)]
    if model_path:
        keys.append(_norm_model(os.path.basename(model_path)))
    out = []
    for r in recs:
        rid = _norm_model(r.get("model"))
        tail = rid.split("/")[-1] if "/" in rid else rid
        score = 0
        for k in keys:
            if not k:
                continue
            if k == rid or k == tail:
                score = max(score, 3)
            elif k in rid or rid in k or tail in k or k in tail:
                score = max(score, 2)
            else:
                # longest shared prefix of >=10 chars is still a decent signal
                n = 0
                while n < min(len(k), len(tail)) and k[n] == tail[n]:
                    n += 1
                if n >= 10:
                    score = max(score, 1)
        if score:
            r = dict(r); r["match"] = score
            out.append(r)
    out.sort(key=lambda x: (-x["match"], -(x.get("when") or 0)))
    return out


# "max" is a valid value here and means every block. Matching only digits made a
# max-offload load invisible, and the scan then silently reported the PREVIOUS
# load's layer count - the exact stale-config error the mismatch guard exists to
# prevent, so it has to be parsed, not skipped.
RE_LMS_NGL   = re.compile(r"Num Offload Layers:\s*(\d+|max)")

RE_LMS_NCM   = re.compile(r"Num CPU Expert Layers:\s*(\d+)")

RE_LMS_EST   = re.compile(r"raw num offload layers '(\d+|max)' and context length '(\d+)'")


def read_lmstudio_runtime():
    """What LM Studio ACTUALLY loaded, from %APPDATA%/LM Studio/logs/main.log.

    This matters because LM Studio silently overrides the GPU Offload slider to
    respect its own VRAM cap, so the number in the UI is not always the number in
    the engine. Measuring a running process against a plan built for a different
    layer count silently mis-attributes the difference to the compute buffer -
    which is exactly the way to get a wrong override. Returns
    {n_gpu_layers, n_cpu_moe, context} for the most recent load, or None."""
    p = os.path.join(os.environ.get("APPDATA", ""), "LM Studio", "logs", "main.log")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-4000:]
    except Exception:
        return None
    out, ctx = None, None
    for i, ln in enumerate(lines):
        m = RE_LMS_EST.search(ln)
        if m:
            ctx = int(m.group(2))
        if "Resolved GPU config options" in ln:
            blk = "".join(lines[i:i + 8])
            ngl = RE_LMS_NGL.search(blk)
            ncm = RE_LMS_NCM.search(blk)
            if ngl:
                mx = (ngl.group(1) == "max")
                out = {"n_gpu_layers": None if mx else int(ngl.group(1)),
                       "all_layers": mx,
                       "n_cpu_moe": int(ncm.group(1)) if ncm else 0,
                       "context": ctx}
    return out


def resolve_runtime_ngl(rt, n_layers):
    """LM Studio's layer count as a number, resolving 'max' against this model."""
    if not rt:
        return None
    return n_layers if rt.get("all_layers") else rt.get("n_gpu_layers")


# The VRAMFIT / SSMSEQ self-tests replay real llama-server measurements. They are
# assertions about THIS hardware, so they only run on the machine they were taken
# on - otherwise a perfectly good build fails its own test suite on a stranger's GPU.
REF_GPU     = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"

REF_BACKEND = "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1"


RE_BACKEND = re.compile(r"(llama\.cpp-[A-Za-z0-9_.+-]+)")


def detect_backend():
    """Which llama.cpp build is serving, e.g.
    'llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1'.

    llama.cpp changes how it allocates the compute graph between releases, so a
    backend upgrade quietly invalidates a calibration fitted on the old one. The
    build is not in any log, so read it from the engine process's own image path.
    Returns "" when it cannot be determined."""
    try:
        procs = get_gpu_processes()
        if isinstance(procs, dict):
            return ""
        eng = next((p for p in procs if p.get("is_engine")), None)
        if not eng:
            return ""
        pid = str(eng.get("pid") or "")
        path = ""
        if os.name == "nt":
            ps = ("(Get-CimInstance Win32_Process -Filter \"ProcessId=%s\")"
                  ".ExecutablePath" % pid)
            path = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                stderr=subprocess.DEVNULL, timeout=20).decode("utf-8", "replace")
        else:
            path = os.path.realpath("/proc/%s/exe" % pid)
        m = RE_BACKEND.search(path.replace("\\", "/").replace("/", " "))
        return m.group(1) if m else ""
    except Exception:
        return ""


_BACKEND_CACHE = {"val": None}


def current_backend(fresh=False):
    """Memoised detect_backend(). Detection costs a subprocess and needs the
    engine running, so it is resolved at most once per process unless a new
    measurement forces a re-check."""
    if fresh or _BACKEND_CACHE["val"] is None:
        _BACKEND_CACHE["val"] = detect_backend()
    return _BACKEND_CACHE["val"]


def default_models_dir():
    home = os.path.expanduser("~")
    for c in [os.path.join(home, ".lmstudio", "models"),
              os.path.join(home, ".cache", "lm-studio", "models")]:
        if os.path.isdir(c):
            return c
    return os.path.join(home, ".lmstudio", "models")


def _quick_ctx(path):
    """Native context length from metadata only (fast). Returns int or None."""
    try:
        meta = parse_meta_only(path)
        for k, v in meta.items():
            if k.endswith(".context_length"):
                return _as_int(v)
    except Exception:
        pass
    return None


def scan_models(root):
    out = []
    if not root or not os.path.isdir(root):
        return out
    seen = set()
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".gguf"):
                continue
            full = os.path.join(dp, fn)
            m = SHARD_RE.match(fn)
            if m:
                if m.group(2) != "00001":
                    continue
                key = (dp, m.group(1))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size = sum(os.path.getsize(s) for s in _find_shards(full))
                except OSError:
                    size = os.path.getsize(full)
                out.append({"name": "%s  [split x%s]" % (m.group(1), m.group(3)),
                            "path": full, "size_mib": size / MiB, "n_ctx_train": _quick_ctx(full)})
            else:
                try:
                    out.append({"name": fn, "path": full,
                                "size_mib": os.path.getsize(full) / MiB,
                                "n_ctx_train": _quick_ctx(full)})
                except OSError:
                    pass
    out.sort(key=lambda x: x["name"].lower())
    return out
