"""Argument parsing and the process entry point."""
import argparse, datetime, os, sys, threading, time
from .const import __version__
from .paths import _user_file
from .web import serve
from .selftest import self_test


def _tty_skip_harness(log):
    """The terminal's Skip: press S while a config is in flight.

    Returns (on_server, skip_count, stop) for speed_sweep, or (None, None,
    None) where there is no console to read - the feature does not exist there
    rather than stealing keys from something else. `on_server` publishes the
    live llama-server, `skip_count` answers bench_one's stamping question, and
    `stop` ends the listener thread after the campaign returns.

    Reading the console is a blocking business, so the listener is a daemon
    thread polling with msvcrt - Windows-only, and only when stdin is a real
    console, which is also when a keypress can reach anyone at all."""
    if os.name != "nt" or not sys.stdin.isatty():
        return None, None, None
    try:
        import msvcrt
    except ImportError:
        return None, None, None

    box = {"n": 0, "proc": None}
    lock = threading.Lock()
    stop = threading.Event()

    def on_server(p):
        with lock:
            box["proc"] = p

    def skip_count():
        with lock:
            return box["n"]

    def press():
        with lock:
            box["n"] += 1
            proc = box["proc"]
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception as e:
                log("skip    : could not kill the server: %s" % e)
        log("skip    : requested - abandoning the config in flight. Its row is "
            "recorded as skipped and will not be re-measured; the campaign "
            "continues with the next config.")

    def listen():
        try:
            while not stop.is_set():
                if msvcrt.kbhit() and msvcrt.getwch().lower() == "s":
                    press()
                time.sleep(0.15)
        except Exception:
            pass

    threading.Thread(target=listen, daemon=True,
                     name="vramplanner-skip").start()
    return on_server, skip_count, stop


def _forget_sweeps(names, assume_yes=False):
    """Forget recorded speed campaigns for the named models.

    Prints what matches BEFORE removing anything, because a campaign is the two
    hours of GPU time that produced it and a model name is an easy thing to
    mistype. A name matches a campaign whose model contains it, case-insensitive
    - the full file name is long and the point is not to make you type it - so
    the listing is what confirms which campaigns were actually meant.

    This runs in its own process and cannot see a campaign running in a browser,
    so it does not try to: delete_campaign() fingerprints the file and abandons
    a delete that would land on top of an append, which is the guard that holds
    across processes rather than only within one.
    """
    from .bench import delete_campaign, load_speed_rows, sweep_index
    groups = sweep_index(load_speed_rows())
    wanted = [g for g in groups
              if any(n.lower() in (g["model"] or "").lower() for n in names)]
    if not wanted:
        print("no recorded campaign matches: %s" % ", ".join(names))
        return 1
    print("\n%-42s %-22s %6s %10s  %s"
          % ("model", "gpu", "rows", "best tok/s", "recorded"))
    for g in wanted:
        print("%-42s %-22s %6d %10s  %s"
              % (g["model"][:42], (g["gpu"] or "?")[:22], g["n_rows"],
                 ("%.2f" % g["best_tok_s"]) if g["best_tok_s"] else "-",
                 datetime.datetime.fromtimestamp(g["last"]).strftime("%Y-%m-%d")
                 if g["last"] else "?"))
    total = sum(g["n_rows"] for g in wanted)
    print("\n%d campaign(s), %d row(s). They move to speed/deleted/, not to nothing."
          % (len(wanted), total))
    if not assume_yes:
        try:
            if input("Forget them? [y/N] ").strip().lower() not in ("y", "yes"):
                print("nothing deleted")
                return 1
        except EOFError:
            # No terminal to ask at. Refusing is the only safe reading of that.
            print("nothing deleted - no terminal to confirm at; pass --yes to skip "
                  "the prompt")
            return 1
    rc = 0
    for g in wanted:
        r = delete_campaign(model=g["model"], gpu=g["gpu"], file=g["file"],
                            prompt_id=g["prompt_id"] or "",
                            template_id=g["template_id"] or "")
        if r.get("ok"):
            print("forgot %d row(s) from %s -> %s"
                  % (r["removed"], r["file"], r.get("backup") or "(no backup)"))
        else:
            print("failed on %s: %s" % (g["model"], r.get("error")))
            rc = 1
    return rc


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
    ap.add_argument("--drafter", default=None, metavar="PATH",
                    help="with --sweep or --speed-sweep: draft every model with "
                         "this file. A dflash-*.gguf drafts as DFlash; a model with "
                         "MTP blocks drafts as draft-mtp; anything else is refused. "
                         "Absent, the speed sweep discovers dflash-*.gguf next to "
                         "each model as before, and the fit sweep plans no "
                         "speculation at all")
    ap.add_argument("--speed-axes", nargs="+", default=None, metavar="AXIS=V,V",
                    help="with --speed-sweep: run an explicit ladder instead of the "
                         "staged grid, e.g. --speed-axes ngl=26,28,30 spec=none. "
                         "n_cpu_ffn is the -ot axis: n_cpu_ffn=0,4,8 pins the first "
                         "blocks' dense FFN to the CPU. spec_kv is the draft "
                         "cache quant: spec_kv=f16,q8_0")
    # Kept a literal rather than bench.STAGES: cli.py imports bench lazily, and
    # an argparse default would drag the whole subprocess-heavy module into
    # every --help. selftest asserts the two agree.
    ap.add_argument("--speed-stages", default="ab", metavar="LETTERS",
                    help="with --speed-sweep: which stages to run (a=the wall, "
                         "b=speculation). Stage A's axis follows --speed-mode: "
                         "context for a speed plan, -ngl for a context plan, "
                         "--n-cpu-moe on an MoE; it then walks the dense FFN back "
                         "onto the card if the wall left room. The old letters "
                         "still work - d is stage B, and c was the ubatch sweep, "
                         "now a frozen setting (--speed-ub)")
    ap.add_argument("--speed-mode", default="auto", metavar="MODE",
                    choices=("auto", "speed", "context"),
                    help="with --speed-sweep, dense models: which question stage A "
                         "answers. 'speed' keeps every block's attention and KV on "
                         "the GPU with the dense FFN exiled (-ngl all, -ot) and "
                         "sweeps CONTEXT for the largest window that loads; "
                         "'context' holds the context and sweeps -ngl. 'auto' asks "
                         "the planner which one this model and card land in")
    ap.add_argument("--speed-fill", type=int, default=None, metavar="TOKENS",
                    help="with --speed-sweep: prompt length to measure at")
    ap.add_argument("--speed-fills", nargs="+", type=int, default=None,
                    metavar="TOKENS",
                    help="with --speed-sweep: the first value is the campaign fill, "
                         "the rest are DEEPER fills that re-measure the top stage-A "
                         "rungs once the wall is known - the depth slope the docs "
                         "tell you to measure, part of the campaign instead of a "
                         "second run. Refuses to combine with --speed-fill")
    ap.add_argument("--speed-ctx", type=int, default=None, metavar="N",
                    help="with --speed-sweep: freeze context length (clamped to the "
                         "model's trained context)")
    ap.add_argument("--speed-kv", default=None, metavar="TYPE",
                    help="with --speed-sweep: freeze KV cache quant, e.g. q8_0")
    ap.add_argument("--speed-ub", type=int, default=None, metavar="N",
                    help="with --speed-sweep: freeze the physical batch (-ub) for "
                         "the campaign. It used to be stage C's axis and is a "
                         "setting now: it drives prefill, and decode - which is "
                         "what the campaign measures - barely moves with it. "
                         'Sweep it anyway with --speed-axes "ub=256,512,1024"')
    ap.add_argument("--speed-spec-kv", default=None, metavar="TYPE",
                    help="with --speed-sweep: freeze the DRAFT model's KV cache "
                         "quant (-ctkd/-ctvd), e.g. q8_0 - f16 by default, the "
                         "way llama.cpp keeps it. Halves what speculation costs "
                         "in VRAM, at whatever the acceptance rate turns out to "
                         "be - which is why stage B measures it")
    ap.add_argument("--speed-chain", action="store_true",
                    help="with --speed-sweep: build each stage against the fastest "
                         "row measured so far instead of against the planner's "
                         "guess. Same number of loads, so the same hours - but "
                         "speculation is then measured at the split that actually "
                         "won rather than at one nothing has confirmed")
    ap.add_argument("--speed-rounds", type=int, default=1, metavar="N",
                    help="with --speed-chain: re-run the stages from the winner N "
                         "times. Cheap - a config a later round revisits unchanged "
                         "is already recorded and is skipped")
    ap.add_argument("--speed-ot", type=int, default=None, metavar="N",
                    help="with --speed-sweep: pin the first N blocks' dense FFN "
                         "tensors to the CPU (-ot) for the whole campaign - the "
                         "mode the planner prices as 'KV on GPU, FFN in RAM'. "
                         "Both plan modes pin every block, so this is only for "
                         "measuring a different split. Dense models only")
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
    ap.add_argument("--reasoning", default=None, choices=("auto", "on", "off"),
                    help="whether the model thinks. Replaces enable_thinking in "
                         "--chat-template-kwargs, which current builds accept and "
                         "then deprecate. Default auto: detect from the template")
    ap.add_argument("--reasoning-preserve", default=None,
                    choices=("default", "on", "off"),
                    help="keep the thinking trace across the WHOLE history, not "
                         "just the last assistant message. Cannot be set from the "
                         "template kwargs: llama-server strips <think> out of the "
                         "history before the template is rendered")
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
    ap.add_argument("--forget-sweep", nargs="+", metavar="MODEL",
                    help="forget recorded speed campaigns for these models. Lists "
                         "what matches and asks before removing anything; the rows "
                         "are moved to speed/deleted/ rather than dropped")
    ap.add_argument("--yes", action="store_true",
                    help="with --forget-sweep: skip the confirmation prompt")
    args = ap.parse_args()
    if args.forget_sweep:
        sys.exit(_forget_sweeps(args.forget_sweep, args.yes))
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
        if args.speed_fills and args.speed_fill is not None:
            print("--speed-fills and --speed-fill are the same question asked two "
                  "ways; the first --speed-fills value IS the campaign fill. Use "
                  "one of them.")
            sys.exit(2)
        overrides = (parse_overrides(args.speed_verify_overrides)
                     if args.speed_verify_overrides else None)
        # S skips the config in flight without ending the campaign: the run is
        # killed, its row recorded as skipped, and the next config starts. The
        # listener needs a real console, so it simply does not exist in pipes;
        # a dry run has no runs, so it is not armed there either.
        on_server, skip_count, skip_stop = None, None, None
        if not args.dry_run:
            on_server, skip_count, skip_stop = _tty_skip_harness(print)
        if on_server:
            print("  keys    : press S to skip the config in flight - it is recorded "
                  "as skipped and will not be re-measured; the campaign continues")
        try:
            r = speed_sweep(models=args.models, backend=args.backend,
                            dry_run=args.dry_run, timeout=args.sweep_timeout,
                            limit=args.limit, axes=axes, stages=args.speed_stages,
                            fill=args.speed_fill, fills=args.speed_fills,
                            ctx=args.speed_ctx, kv=args.speed_kv, ub=args.speed_ub,
                            spec_kv=args.speed_spec_kv,
                            n_predict=args.n_predict, repeat=args.repeat,
                            chain=args.speed_chain, rounds=args.speed_rounds,
                            verify=args.speed_verify, verify_overrides=overrides,
                            chat_template_file=args.chat_template_file,
                            chat_template_kwargs=args.chat_template_kwargs,
                            reasoning=args.reasoning,
                            reasoning_preserve=args.reasoning_preserve,
                            drafter=args.drafter, ot=args.speed_ot,
                            plan_mode=(None if args.speed_mode == "auto"
                                       else args.speed_mode),
                            on_server=on_server, skip_count=skip_count)
        finally:
            if skip_stop:
                skip_stop.set()
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
                  timeout=args.sweep_timeout, limit=args.limit,
                  drafter=args.drafter)
        sys.exit(0 if r else 1)
    serve(args.host, args.port, not args.no_browser)
