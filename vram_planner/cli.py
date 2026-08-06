"""Argument parsing and the process entry point."""
import argparse, sys
from .const import __version__
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
    args = ap.parse_args()
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
