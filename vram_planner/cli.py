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
    ap.add_argument("--speed-sweep", action="store_true",
                    help="drive llama-server across a config grid and record how "
                         "fast it GENERATES - speculative decoding and prefill are "
                         "not modelled anywhere in this tool, so they can only be "
                         "measured. Rows go to speed/, never to sweeps/")
    ap.add_argument("--speed-axes", nargs="+", default=None, metavar="AXIS=V,V",
                    help="with --speed-sweep: run an explicit ladder instead of the "
                         "staged grid, e.g. --speed-axes ngl=26,28,30 spec=none")
    ap.add_argument("--speed-stages", default="abcd", metavar="LETTERS",
                    help="with --speed-sweep: which stages to run (a=ngl wall, "
                         "b=projector placement, c=ubatch, d=speculation)")
    ap.add_argument("--speed-fill", type=int, default=None, metavar="TOKENS",
                    help="with --speed-sweep: prompt length to measure at")
    ap.add_argument("--speed-ctx", type=int, default=None, metavar="N",
                    help="with --speed-sweep: freeze context length (clamped to the "
                         "model's trained context)")
    ap.add_argument("--speed-kv", default=None, metavar="TYPE",
                    help="with --speed-sweep: freeze KV cache quant, e.g. q8_0")
    ap.add_argument("--speed-chain", action="store_true",
                    help="with --speed-sweep: build each stage against the fastest "
                         "row measured so far instead of against the planner's "
                         "guess. Same number of loads, so the same hours - but "
                         "ubatch is then measured at the layer split that actually "
                         "won rather than at one nothing has confirmed")
    ap.add_argument("--speed-rounds", type=int, default=1, metavar="N",
                    help="with --speed-chain: re-run the stages from the winner N "
                         "times. Cheap - a config a later round revisits unchanged "
                         "is already recorded and is skipped")
    ap.add_argument("--speed-verify", action="store_true",
                    help="with --speed-sweep: after the campaign, load the winner "
                         "once more at the PRODUCTION config and only certify it if "
                         "it actually loads - a split that fitted the grid does not "
                         "automatically fit what you run")
    ap.add_argument("--speed-verify-overrides", nargs="+", default=None, metavar="AXIS=V,V",
                    help="with --speed-verify: the production config, e.g. "
                         "spec=draft-mtp spec_n_max=2 temp=1 top_k=20 top_p=0.95. "
                         "Overlaid on the winner's knobs; absent, the winner is "
                         "verified as itself")
    ap.add_argument("--chat-template-file", default=None, metavar="PATH",
                    help="with --speed-sweep: measure under this jinja chat "
                         "template instead of the GGUF's own. --jinja is passed "
                         "before it, as that build requires. Rows record the "
                         "template's CONTENT hash, so editing it re-measures")
    ap.add_argument("--chat-template-kwargs", default=None, metavar="JSON",
                    help='template variables as a JSON object, e.g. '
                         '\'{"enable_thinking":true,"reasoning_effort":"xhigh"}\'. '
                         "Validated before the first server starts")
    ap.add_argument("--refresh-corpus", action="store_true",
                    help="rebuild the frozen speed-sweep filler corpus from this "
                         "repository's README and sources. Rows record the corpus's "
                         "hash, so an old corpus and a new one are different "
                         "experiments and old rows stop resuming")
    ap.add_argument("--speed-report", action="store_true",
                    help="print every recorded speed row, fastest first")
    ap.add_argument("--insights", action="store_true",
                    help="with --speed-report: print what the campaigns FOUND - what "
                         "each knob was worth, how speed fell with context depth, and "
                         "which configs nothing beats on both speed and VRAM - instead "
                         "of the flat ranking")
    ap.add_argument("--n-predict", type=int, default=128,
                    help="with --speed-sweep: tokens generated per measured pass")
    ap.add_argument("--repeat", type=int, default=3,
                    help="with --speed-sweep: measured passes per config (median)")
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
    if args.speed_report:
        from .bench import report as speed_report, report_insights
        sys.exit(0 if (report_insights() if args.insights else speed_report()) else 1)
    if args.refresh_corpus:
        from .bench import prompt_identity, refresh_corpus
        before = prompt_identity()
        path = refresh_corpus()
        after = prompt_identity()
        print("wrote %s" % path)
        print("corpus identity: %s -> %s%s"
              % (before[:12], after[:12],
                 " (unchanged - commit it so rows keep resuming)" if before == after
                 else " - rows recorded against the old corpus will not be resumed"))
        sys.exit(0)
    if args.speed_sweep:
        from .bench import speed_sweep         # deferred: needs subprocess work
        from .sweep import parse_overrides
        axes = parse_overrides(args.speed_axes) if args.speed_axes else None
        overrides = (parse_overrides(args.speed_verify_overrides)
                     if args.speed_verify_overrides else None)
        r = speed_sweep(models=args.models, backend=args.backend,
                        dry_run=args.dry_run, timeout=args.sweep_timeout,
                        limit=args.limit, axes=axes, stages=args.speed_stages,
                        fill=args.speed_fill, ctx=args.speed_ctx, kv=args.speed_kv,
                        n_predict=args.n_predict, repeat=args.repeat,
                        chain=args.speed_chain, rounds=args.speed_rounds,
                        verify=args.speed_verify, verify_overrides=overrides,
                        chat_template_file=args.chat_template_file,
                        chat_template_kwargs=args.chat_template_kwargs)
        sys.exit(0 if r else 1)
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
