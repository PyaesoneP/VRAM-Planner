"""The local web UI and its JSON endpoints."""
import json, os, webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs
from urllib.parse import urlparse
from .const import __version__
from .gguf import load_gguf
from .model import extract_config
from .paths import _data_dir
from .gpu import get_bandwidth, get_gpu_processes, get_gpus, get_ram, platform_support
from .lmstudio import benchmark_server, default_models_dir, load_benchmarks, match_speed_history, read_lmstudio_runtime, resolve_runtime_ngl, save_benchmark, scan_models, scan_server_logs, scan_speed_history
from .calib import _active_gpu, calibration_status, record_calibration, refresh_calibration
from .plan import analyze


# ---------------------------------------------------------------------------
# Static UI assets
# ---------------------------------------------------------------------------
# The UI lives in vram_planner/ui/ as real .html/.css/.js rather than inside
# Python string literals, so an editor can lint and format it. Only this fixed
# whitelist is served - the directory is never walked from a request path, so a
# crafted URL cannot reach outside it.
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
UI_FILES = {
    "/":         ("index.html", "text/html; charset=utf-8"),
    "/app.css":  ("app.css",    "text/css; charset=utf-8"),
    "/app.js":   ("app.js",     "application/javascript; charset=utf-8"),
}


def read_ui(name):
    """Read one UI file. Read per request, not cached at import: editing the CSS
    and hitting reload is the whole point of having them as files."""
    with open(os.path.join(UI_DIR, name), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _calibrate(self, data):
        """Record one Measure as a calibration row. Everything that matters is
        determined server-side: the engine's real VRAM, and the config LM Studio
        actually loaded (which is not always the one on screen)."""
        path = data.get("path") or ""
        if not os.path.exists(path):
            return {"ok": False, "error": "file not found: %s" % path}
        procs = get_gpu_processes()
        if isinstance(procs, dict):
            return {"ok": False, "error": "could not read GPU processes: %s"
                                          % procs.get("error", "?")}
        eng = next((p for p in procs if p.get("is_engine")), None)
        if not eng:
            return {"ok": False, "error": "No inference process on the GPU. Load the "
                                          "model in LM Studio first."}
        rt = read_lmstudio_runtime()
        try:
            n_layers = extract_config(load_gguf(path))["n_layers"] or 0
        except Exception as e:
            return {"ok": False, "error": "could not read model: %s" % e}
        ngl = resolve_runtime_ngl(rt, n_layers)
        ctx = (rt or {}).get("context") or int(data.get("context") or 0)
        if ngl is None:
            return {"ok": False, "error": "Could not read LM Studio's resolved GPU config "
                                          "from its log, so the layer count is unknown. "
                                          "Measurement skipped rather than guessed."}
        try:
            out = record_calibration(
                path, ctx, data.get("kv_type") or "f16",
                int(data.get("n_ubatch") or 512), int(data.get("n_seq") or 1),
                bool(data.get("flash_attn", True)), int(ngl),
                bool(data.get("include_mmproj", True)), float(eng["mib"]),
                gpu=_active_gpu(), n_cpu_moe=(rt or {}).get("n_cpu_moe") or 0)
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if out.get("error"):
            return {"ok": False, "error": out["error"]}
        out.update({"ok": True, "engine": eng,
                    "runtime": {"n_gpu_layers": ngl, "context": ctx,
                                "n_cpu_moe": (rt or {}).get("n_cpu_moe") or 0,
                                "all_layers": bool((rt or {}).get("all_layers"))}})
        return out

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in UI_FILES:
            name, ctype = UI_FILES[u.path]
            try:
                return self._send(200, read_ui(name), ctype)
            except OSError as e:
                return self._send(500, {"error": "UI asset %s unreadable: %s" % (name, e)})
        if u.path == "/api/benchmark":
            q = parse_qs(u.query)
            res = benchmark_server(
                base_url=(q.get("url") or ["http://localhost:1234"])[0],
                max_tokens=int((q.get("max_tokens") or ["128"])[0]))
            if res.get("tok_s"):
                res["ctx"] = (q.get("ctx") or [None])[0]
                res["n_gpu_layers"] = (q.get("ngl") or [None])[0]
                save_benchmark(res)
            return self._send(200, res)
        if u.path == "/api/bandwidth":
            return self._send(200, get_bandwidth())
        if u.path == "/api/speedhistory":
            q = parse_qs(u.query)
            recs = scan_speed_history()
            if isinstance(recs, dict):
                recs = []
            for bm in load_benchmarks():          # server-mode measurements
                if bm.get("tok_s"):
                    recs.append({"model": bm.get("model") or "?", "tok_s": bm["tok_s"],
                                 "n_gpu_layers": bm.get("n_gpu_layers"),
                                 "ttft_s": bm.get("ttft_s"),
                                 "prompt_tokens": bm.get("prompt_tokens") or 0,
                                 "predicted_tokens": bm.get("predicted_tokens") or 0,
                                 "ctx": bm.get("ctx"), "when": bm.get("when"),
                                 "conversation": "benchmark", "benchmark": True})
            name = (q.get("model") or [""])[0]
            path = (q.get("path") or [""])[0]
            return self._send(200, {"matches": match_speed_history(recs, name, path),
                                    "all": recs, "server_log": scan_server_logs(60)})
        if u.path == "/api/gpuprocs":
            return self._send(200, {"procs": get_gpu_processes(),
                                    "runtime": read_lmstudio_runtime(),
                                    "calibration": calibration_status()})
        if u.path == "/api/system":
            fresh = parse_qs(u.query).get("fresh", ["0"])[0] == "1"
            return self._send(200, {"gpus": get_gpus(fresh=fresh), "ram": get_ram(),
                                    "default_dir": default_models_dir(),
                                    "version": __version__,
                                    "platform": platform_support()})
        if u.path == "/api/models":
            q = parse_qs(u.query)
            d = (q.get("dir", [""])[0]) or default_models_dir()
            try:
                return self._send(200, {"models": scan_models(d), "dir": d})
            except Exception as e:
                return self._send(200, {"models": [], "error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/api/analyze", "/api/calibrate"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._send(200, {"ok": False, "error": "bad request: %s" % e})
        if u.path == "/api/calibrate":
            return self._send(200, self._calibrate(data))
        try:
            path = data["path"]
            if not os.path.exists(path):
                return self._send(200, {"ok": False, "error": "file not found: %s" % path})
            res = analyze(
                path=path,
                ctx=int(data.get("context", 8192)),
                kv_type=data.get("kv_type", "f16"),
                n_ubatch=int(data.get("n_ubatch", 512)),
                n_seq=int(data.get("n_seq", 1) or 1),
                include_mmproj=bool(data.get("include_mmproj", True)),
                mtp_spec=bool(data.get("mtp_spec", True)),
                flash_attn=bool(data.get("flash_attn", False)),
                vram_budget_mib=float(data.get("vram_budget_mib", 0)),
                ram_budget_mib=float(data.get("ram_budget_mib", 0)),
                gpu_reserve_mib=float(data.get("gpu_reserve_mib", 512)),
                compute_override_mib=float(data.get("compute_override_mib", 0)),
                safety_pct=float(data.get("safety_pct", 5)),
                kv_on_gpu=bool(data.get("kv_on_gpu", False)),
                bw_vram_gbs=float(data.get("bw_vram_gbs") or 0) or None,
                bw_ram_gbs=float(data.get("bw_ram_gbs") or 0) or None,
                ram_eff=float(data.get("ram_eff") or 0) or None,
                ctx_fill=(int(data["ctx_fill"]) if data.get("ctx_fill") not in (None,"",False) else None),
                n_cpu_moe_override=(int(data["n_cpu_moe_override"])
                                    if data.get("n_cpu_moe_override") not in (None, "", False)
                                    else None),
                gpu_layers_override=(int(data["gpu_layers_override"])
                                     if data.get("gpu_layers_override") not in (None, "", False)
                                     else None),
                ram_free_mib=(float(data["ram_free_mib"])
                              if data.get("ram_free_mib") else None),
            )
            return self._send(200, res)
        except Exception as e:
            import traceback
            return self._send(200, {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                                    "trace": traceback.format_exc()})


def serve(host, port, open_browser):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0") else host, port)
    refresh_calibration()
    st = calibration_status()
    plat = platform_support()
    print("\n  VRAM Planner %s  running at  %s" % (__version__, url))
    print("  models folder default :  %s" % default_models_dir())
    print("  user data             :  %s" % _data_dir())
    print("  compute-buffer model  :  %s"
          % ("calibrated from %d measurement(s) on %s (fitted: %s, in-sample %.1f%%)"
             % (st["n"], st["gpu"] or "this GPU", ", ".join(st["free"]), st["residual_pct"])
             if st["calibrated"] else
             "shipped defaults - press Measure on a loaded model to calibrate"))
    if st.get("skipped_rows"):
        print("  measurements skipped  :  %d (reading did not respond to the config, or "
              "no layer count recorded)" % st["skipped_rows"])
    if not plat["supported"]:
        print("\n  !! UNVALIDATED PLATFORM\n     %s" % plat["reason"])
    print("  press Ctrl+C to stop\n")
    if open_browser:
        try: webbrowser.open(url)
        except Exception: pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.server_close()
