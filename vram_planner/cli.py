"""Argument parsing and the process entry point."""
import argparse, datetime, os, sys
from .const import __version__
from .paths import _user_file
from .web import serve
from .selftest import self_test


def main():
    ap = argparse.ArgumentParser(description="Plan GGUF model fit on GPU/RAM.")
    ap.add_argument("--version", action="version", version="vram-planner %s" % __version__)
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--require-refs", action="store_true",
                    help="with --self-test: fail if any real-measurement section is "
                         "skipped, instead of passing having checked nothing")
    ap.add_argument("--sweep", action="store_true",
                    help="drive llama-server across a config grid and record what "
                         "the allocator reports (hours; see --dry-run first)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --sweep: print the configs and an estimate, run nothing")
    ap.add_argument("--models", nargs="*", default=None, metavar="NAME",
                    help="with --sweep: only models whose file name contains one of these")
    ap.add_argument("--backend", default=None, metavar="BUILD",
                    help="with --sweep: llama.cpp build to use (default: newest CUDA)")
    ap.add_argument("--limit", type=int, default=None,
                    help="with --sweep: stop after this many configs")
    ap.add_argument("--sweep-timeout", type=float, default=420.0, metavar="SECONDS")
    ap.add_argument("--probe", nargs="+", metavar="MODEL AXIS=V,V",
                    help="run an explicit config ladder on one model, e.g. "
                         "--probe gemma ctx=20480,24576,32768")
    ap.add_argument("--fit", action="store_true",
                    help="score the compute-buffer model against recorded sweep data, "
                         "held out - see --sweep")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refit the compute-buffer coefficients from the stored "
                         "measurements and save them. The fit is otherwise frozen: "
                         "nothing else refits it except pressing Measure")
    ap.add_argument("--show-calibration", action="store_true",
                    help="print the stored fit and where it came from, change nothing")
    ap.add_argument("--cards", action="store_true",
                    help="list stored model cards - models that can be planned "
                         "without the .gguf being on disk")
    ap.add_argument("--add-card", nargs="+", metavar="GGUF",
                    help="record a model card for these files so they stay plannable "
                         "after the weights are deleted")
    ap.add_argument("--forget-card", nargs="+", metavar="NAME",
                    help="delete stored model cards by file name")
    args = ap.parse_args()
    if args.add_card or args.forget_card or args.cards:
        from .cards import forget_card, list_cards, remember_card
        for p in (args.add_card or []):
            if not os.path.exists(p):
                print("skipped, not found: %s" % p)
                continue
            from .gguf import load_gguf
            from .model import classify_tensors, extract_config
            from .plan import find_mmproj
            m = load_gguf(p)
            cfg = extract_config(m)
            remember_card(p, cfg, classify_tensors(m, cfg), find_mmproj(p))
            print("recorded: %s" % os.path.basename(p))
        for n in (args.forget_card or []):
            print(("forgot: %s" if forget_card(n) else "no card for: %s") % n)
        cards = list_cards()
        if not cards:
            print("no model cards stored. One is recorded automatically every time "
                  "you analyse a model, or use --add-card.")
            sys.exit(0)
        print("\n%-46s %8s %7s %9s  %s" % ("model", "params", "layers", "weights", "recorded"))
        for c in cards:
            print("%-46s %7.1fB %7d %8.1fG  %s%s"
                  % (c["name"][:46], (c["params"] or 0) / 1e9, c["n_layers"],
                     (c["weights_bytes"] or 0) / 1e9,
                     datetime.datetime.fromtimestamp(c["when"]).strftime("%Y-%m-%d")
                     if c.get("when") else "?",
                     "  +mmproj" if c["has_mmproj"] else ""))
        print("\n%d card(s) in %s" % (len(cards), _user_file("model_cards.json")))
        sys.exit(0)
    if args.show_calibration or args.recalibrate:
        from .calib import calibration_status, refresh_calibration
        if args.recalibrate:
            refresh_calibration(force=True)
        st = calibration_status()
        if not st["calibrated"]:
            print("no stored fit for %s - the shipped defaults are in use.\n"
                  "Load a model in LM Studio and press Measure to make one."
                  % (st["gpu"] or "this GPU"))
            sys.exit(1)
        print("GPU        : %s" % st["gpu"])
        print("fitted     : %s from %d measurement(s)"
              % (datetime.datetime.fromtimestamp(st["when"]).strftime("%Y-%m-%d %H:%M")
                 if st.get("when") else "unknown", st["n"]))
        print("build      : %s" % (st["backend"] or "unknown"))
        print("free terms : %s (in-sample %.1f%%)"
              % (", ".join(st["free"]), st["residual_pct"]))
        for k, v in sorted(st["coeffs"].items()):
            print("  %-6s %s" % (k, v))
        if st["outdated"]:
            print("\n!! %s" % st["outdated"])
        sys.exit(0)
    if args.fit:
        from .fit import report
        sys.exit(0 if report() else 1)
    if args.probe:
        from .sweep import parse_overrides, probe
        r = probe(args.probe[0], parse_overrides(args.probe[1:]),
                  backend=args.backend, timeout=args.sweep_timeout)
        sys.exit(0 if r else 1)
    if args.self_test:
        sys.exit(self_test(require_refs=args.require_refs))
    if args.sweep:
        from .sweep import sweep      # deferred: only this path needs subprocess work
        r = sweep(models=args.models, backend=args.backend, dry_run=args.dry_run,
                  timeout=args.sweep_timeout, limit=args.limit)
        sys.exit(0 if r else 1)
    serve(args.host, args.port, not args.no_browser)
