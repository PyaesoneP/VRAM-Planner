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
from .cards import forget_card, have_card, list_cards
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

    # -- speed sweep ---------------------------------------------------------
    # bench/sweep are imported inside these handlers, not at module scope: they
    # pull in the subprocess machinery, and starting the web UI should not.

    def _speed_args(self, data):
        """The subset of --speed-sweep's arguments the UI exposes.

        ctx and kv come straight off the main form, because the settings you are
        planning for are the ones worth measuring - they are --speed-ctx and
        --speed-kv, which freeze those axes instead of sweeping them."""
        from .sweep import parse_overrides
        def as_int(k, default=None):
            v = data.get(k)
            return int(v) if v not in (None, "", False) else default
        path = data.get("path") or ""
        return {
            "models": [path] if path and os.path.exists(path) else None,
            "stages": (data.get("stages") or "abcd").lower(),
            "ctx": as_int("context"), "kv": data.get("kv_type") or None,
            "fill": as_int("fill"), "limit": as_int("limit"),
            "n_predict": as_int("n_predict", 128), "repeat": as_int("repeat", 3),
            "timeout": float(data.get("timeout") or 420.0),
            # Chained is the UI's default: it costs the same hours and measures
            # each knob at the split that won rather than at the planner's guess.
            "chain": bool(data.get("chain", True)),
            "rounds": max(1, as_int("rounds", 1) or 1),
            # verify is opt-in from the browser, like --speed-verify on the CLI
            "verify": bool(data.get("verify")),
            "verify_overrides": (parse_overrides(data.get("verify_overrides").split())
                                 if data.get("verify_overrides") else None),
            # Same two fields the launch-script card already carries, so a
            # campaign can be MEASURED under the template it will be RUN under.
            # Absent means the GGUF's own metadata template, which is what every
            # row recorded before this was measured against.
            "chat_template_file": data.get("chat_template_file") or None,
            "chat_template_kwargs": data.get("chat_template_kwargs") or None,
            "reasoning": data.get("reasoning") or None,
            "reasoning_preserve": data.get("reasoning_preserve") or None,
        }

    def _speed_start(self, data):
        from .bench import preflight_state, speed_sweep
        from .job import JOB
        if JOB.running:
            return {"ok": False, "running": True,
                    "error": "A sweep is already running. One at a time: a campaign "
                             "owns the GPU, so two at once would not be slower, they "
                             "would both be wrong."}
        st = preflight_state()
        if not st["ok"]:
            return {"ok": False, "preflight": st, "error": st["reason"]}
        kw = self._speed_args(data)
        if not kw["models"]:
            return {"ok": False, "error": "No such model file: %s" % (data.get("path") or "")}

        def add_row(row):
            # Keep only what the table renders. A full row carries every measured
            # pass, a ~600-char text sample and the parsed load log, and the whole
            # list of them is re-shipped on every 1.5s poll - so a long campaign
            # pays for its own history over and over. Slimming at the producer
            # keeps job.py free of any dependency on what a row means.
            from .bench import _slim
            job_row = _slim(row)
            job_row["config"] = row.get("config")
            JOB._add_row(job_row)

        def work(job):
            # skip_preflight: already checked above, and re-reading it here would
            # race against the driver still releasing memory.
            return speed_sweep(log=job._append, on_row=add_row,
                               should_stop=job.cancelled, on_total=job.set_total,
                               skip_preflight=True, **kw)

        ok, msg = JOB.start("speed sweep", work)
        return {"ok": ok, "error": None if ok else msg, "preflight": st}

    def _speed_plan(self, data):
        from .bench import speed_sweep
        kw = self._speed_args(data)
        if not kw["models"]:
            return {"ok": False, "error": "No such model file: %s" % (data.get("path") or "")}
        lines = []
        try:
            res = speed_sweep(dry_run=True, log=lines.append, **kw)
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if not res:
            return {"ok": False, "error": "\n".join(lines) or "could not build a grid"}
        res.update({"ok": True, "log": lines})
        return res

    def _resolve_model(self, data):
        """(path, resolved) for a script request.

        Speed rows record a model's BASENAME, not a path, so a script built from
        an old campaign - or one measured on another machine - has nothing to open
        directly. Look the name up under the models folder; failing that, hand back
        the name itself and say it did not resolve. Refusing outright would be the
        wrong call: the flags are the valuable part of a launcher and the path is
        one edit.

        `model_name` WINS over `path` when the two name different files. The page
        always sends the analyzed model's path, but the row being turned into a
        script may have come from another model's campaign in Past sweeps - and
        preferring the path there would build a launcher for the analyzed model
        carrying the other one's config and the other one's measured tok/s in its
        header. That is worse than any error: both halves look right."""
        path = data.get("path") or ""
        name = data.get("model_name") or ""
        if name and path and os.path.basename(path) != name:
            path = ""
        if path and os.path.exists(path):
            return path, True
        name = name or (os.path.basename(path) if path else "")
        if name:
            root = data.get("dir") or default_models_dir()
            try:
                for m in scan_models(root) or []:
                    if os.path.basename(m.get("path") or "") == name:
                        return m["path"], True
            except Exception:
                pass
        return (path or name or ""), False

    def _script(self, data):
        from .launch import LOAD_MODES, SHELLS, default_shell, launch_script, script_name
        path, resolved = self._resolve_model(data)
        c = data.get("config") or {}
        if not c:
            return {"ok": False, "error": "no config"}
        shell = data.get("shell") or default_shell()
        if shell not in SHELLS:
            return {"ok": False, "error": "unknown shell: %s" % shell}
        load_mode = data.get("load_mode") or "none"
        if load_mode not in LOAD_MODES:
            return {"ok": False, "error": "unknown load mode: %s" % load_mode}
        # The UI knows whether the plan includes the projector, but not where it
        # is - analyze() reports its name and size, not its path. `true` means
        # "find it", which is the same rule the sweep uses.
        mmproj = data.get("mmproj")
        if mmproj is True:
            from .bench import find_mmproj_for
            mmproj = find_mmproj_for(path) if os.path.exists(path) else None
        try:
            text = launch_script(
                path, c, mmproj=mmproj or None, shell=shell,
                sampling=data.get("sampling") or None,
                port=int(data.get("port") or 8080),
                bind_host=data.get("bind_host") or "127.0.0.1",
                load_mode=load_mode, measured=data.get("measured") or None,
                chat_template_file=data.get("chat_template_file") or None,
                chat_template_kwargs=data.get("chat_template_kwargs") or None,
                reasoning=data.get("reasoning") or None,
                reasoning_preserve=data.get("reasoning_preserve") or None,
                path_resolved=resolved)
        except ValueError as e:
            # Raised by template_args() for kwargs that are not a JSON object.
            # It is the user's typo, not a crash, so it reads as a message.
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        return {"ok": True, "text": text, "shell": shell, "path": path,
                "path_resolved": resolved, "filename": script_name(path, shell)}

    def _script_save(self, data):
        """Write the script and report where it went.

        Next to the model by default: on Windows a browser download lands in
        Downloads/ with a .ps1 that then trips execution policy, and the model
        folder is somewhere the user already knows. Falls back to the tool's own
        data directory when that folder is not writable."""
        res = self._script(data)
        if not res.get("ok"):
            return res
        # The RESOLVED path: a script built from an old row may name a model this
        # request never sent a path for, and saving beside it is still right when
        # the lookup found it.
        path = res.get("path") if res.get("path_resolved") else ""
        cands = []
        if path and os.path.isdir(os.path.dirname(os.path.abspath(path))):
            cands.append(os.path.dirname(os.path.abspath(path)))
        cands.append(_data_dir())
        for d in cands:
            dest = os.path.join(d, res["filename"])
            try:
                with open(dest, "w", encoding="utf-8", newline="\n") as f:
                    f.write(res["text"])
                if res["shell"] == "bash":
                    try:
                        os.chmod(dest, 0o755)
                    except OSError:
                        pass       # no chmod on Windows; the text is what matters
                res["saved"] = dest
                return res
            except OSError as e:
                res["error"] = "could not write %s: %s" % (dest, e)
        return {"ok": False, "error": res.get("error") or "could not write the script"}

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
        if u.path == "/api/speed/preflight":
            from .bench import preflight_state
            return self._send(200, preflight_state())
        if u.path == "/api/speed/status":
            from .job import JOB
            q = parse_qs(u.query)
            # A query string is user input even when only our own code writes it.
            # An unparseable `since` used to raise straight out of do_GET, which
            # drops the connection with no response at all - so a polling UI would
            # see a network error and have nothing to report. Bad offset, start
            # from the top.
            try:
                since = int((q.get("since") or ["0"])[0])
            except (TypeError, ValueError):
                since = 0
            return self._send(200, JOB.snapshot(since=since))
        if u.path == "/api/speed/rows":
            # Ranked history, with no job ever having run in this process - a
            # campaign from last week is exactly as usable as one from this hour.
            from .bench import bench_dir, load_speed_rows, rank_rows
            q = parse_qs(u.query)
            name = (q.get("model") or [""])[0]
            rows = load_speed_rows()
            if name:
                rows = [r for r in rows if r.get("model") == name]
            return self._send(200, {"rows": rank_rows(rows), "dir": bench_dir(),
                                    "n_all": len(rows)})
        if u.path == "/api/speed/history":
            # Every campaign on disk, with no model argument and nothing analyzed:
            # hours already spent should be readable the moment the page opens.
            from .bench import bench_dir, load_speed_rows, sweep_index
            rows = load_speed_rows()
            return self._send(200, {"ok": True, "campaigns": sweep_index(rows),
                                    "dir": bench_dir(), "n_rows": len(rows)})
        if u.path == "/api/speed/insights":
            from .bench import insights, load_speed_rows
            q = parse_qs(u.query)
            get = lambda k: (q.get(k) or [""])[0]
            rows = load_speed_rows()
            # Filter BEFORE analysing, not after: the whole correctness of
            # axis_effects() is that rows from different experiments never meet,
            # and narrowing to one campaign is the strongest form of that.
            for key, field in (("model", "model"), ("gpu", "gpu"), ("file", "_file")):
                v = get(key)
                if v:
                    rows = [r for r in rows if r.get(field) == v]
            if not rows:
                return self._send(200, {"ok": False, "error": "no rows match"})
            res = insights(rows)
            res["ok"] = True
            return self._send(200, res)
        if u.path == "/api/templates":
            # A browser <input type=file> cannot hand back a real path, so the
            # field is a text box - and typing an absolute Windows path by hand is
            # the entire friction this list removes.
            q = parse_qs(u.query)
            path = (q.get("path") or [""])[0]
            d = os.path.dirname(os.path.abspath(path)) if path else ""
            found = []
            if d and os.path.isdir(d):
                try:
                    for fn in sorted(os.listdir(d)):
                        if fn.lower().endswith((".jinja", ".jinja2", ".j2")):
                            found.append(os.path.join(d, fn))
                except OSError:
                    pass
            return self._send(200, {"ok": True, "dir": d, "templates": found})
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
            from .launch import LOAD_MODES, SHELLS, default_shell
            return self._send(200, {"gpus": get_gpus(fresh=fresh), "ram": get_ram(),
                                    "default_dir": default_models_dir(),
                                    "version": __version__,
                                    "platform": platform_support(),
                                    # which launcher this machine actually runs -
                                    # the browser cannot tell reliably
                                    "shell": default_shell(), "shells": list(SHELLS),
                                    "load_modes": list(LOAD_MODES)})
        if u.path == "/api/models":
            q = parse_qs(u.query)
            d = (q.get("dir", [""])[0]) or default_models_dir()
            try:
                found = scan_models(d)
            except Exception as e:
                return self._send(200, {"models": [], "error": str(e)})
            # Cards for models that are not on disk are offered alongside the real
            # files, flagged, so a deleted model stays plannable. A card whose file
            # IS present adds nothing and is not listed twice.
            here = {os.path.basename(m["path"]) for m in found}
            offline = [{"name": c["name"], "path": c["name"], "from_card": True,
                        "size_mib": (c["weights_bytes"] or c["file_bytes"]) / (1 << 20),
                        "n_ctx_train": 0}
                       for c in list_cards() if c["name"] not in here]
            return self._send(200, {"models": found + offline, "dir": d,
                                    "n_cards": len(offline)})
        if u.path == "/api/cards":
            return self._send(200, {"cards": list_cards()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        POSTS = ("/api/analyze", "/api/calibrate", "/api/script", "/api/script/save",
                 "/api/speed/plan", "/api/speed/start", "/api/speed/stop")
        if u.path not in POSTS:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            return self._send(200, {"ok": False, "error": "bad request: %s" % e})
        if u.path == "/api/calibrate":
            return self._send(200, self._calibrate(data))
        if u.path == "/api/script":
            return self._send(200, self._script(data))
        if u.path == "/api/script/save":
            return self._send(200, self._script_save(data))
        if u.path == "/api/speed/plan":
            return self._send(200, self._speed_plan(data))
        if u.path == "/api/speed/start":
            return self._send(200, self._speed_start(data))
        if u.path == "/api/speed/stop":
            from .job import JOB
            ok, msg = JOB.cancel()
            return self._send(200, {"ok": ok, "message": msg})
        try:
            path = data["path"]
            # A missing file is not an error when we have a card for it - that is
            # the whole point of the card. analyze() raises if there is neither.
            if not os.path.exists(path) and not have_card(path):
                return self._send(200, {"ok": False, "error":
                    "file not found and no stored card: %s" % path})
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
                # Vision: two numbers, or nothing. Absent means a text-only plan,
                # which analyze() warns about rather than guessing an image size.
                image_px=((int(data["image_w"]), int(data["image_h"]))
                          if data.get("image_w") and data.get("image_h") else None),
                vision_flash_attn=bool(data.get("vision_flash_attn", True)),
            )
            return self._send(200, res)
        except Exception as e:
            import traceback
            return self._send(200, {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                                    "trace": traceback.format_exc()})


def serve(host, port, open_browser):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0") else host, port)
    # A read, not a refit: the stored fit is what plans are made against until
    # someone presses Measure or runs --recalibrate. Starting the server must
    # not change anyone's numbers.
    refresh_calibration()
    st = calibration_status()
    plat = platform_support()
    print("\n  VRAM Planner %s  running at  %s" % (__version__, url))
    print("  models folder default :  %s" % default_models_dir())
    print("  user data             :  %s" % _data_dir())
    if st["calibrated"]:
        import datetime
        when = (datetime.datetime.fromtimestamp(st["when"]).strftime("%Y-%m-%d %H:%M")
                if st.get("when") else "an earlier version")
        print("  compute-buffer model  :  calibrated from %d measurement(s) on %s "
              "(fitted: %s, in-sample %.1f%%)"
              % (st["n"], st["gpu"] or "this GPU", ", ".join(st["free"]),
                 st["residual_pct"]))
        print("  fit frozen since      :  %s (press Measure or run --recalibrate "
              "to refit)" % when)
    else:
        print("  compute-buffer model  :  shipped defaults - press Measure on a "
              "loaded model to calibrate")
    if st.get("skipped_rows"):
        print("  measurements skipped  :  %d (reading did not respond to the config, or "
              "no layer count recorded)" % st["skipped_rows"])
    if st.get("outdated"):
        print("\n  !! STORED FIT MAY NOT MATCH THIS MACHINE\n     %s" % st["outdated"])
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
