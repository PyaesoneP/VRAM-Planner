"""Drive llama-server across a config grid and record how fast it generates.

`sweep` answers "does this config fit, and what does the allocator reserve for
it". This module answers the question that is left over: "of the configs that
fit, which one is fastest". They are not the same question and the second one
cannot be derived from the first, because the two settings most worth tuning are
invisible to a memory model:

  * **Speculative decoding.** speed.py is a bandwidth roofline with no
    acceptance-rate term, and per_token_bytes() deliberately skips MTP blocks -
    they do not run during ordinary decode. The planner can price what
    speculation COSTS (an f16 draft cache, regardless of --cache-type-k) and
    nothing at all of what it saves. Only a generation can settle it.
  * **Prompt processing.** Not modelled anywhere in this tool, on purpose: it is
    compute bound rather than bandwidth bound. llama-server reports it for free
    in the same timings block as decode, so measuring one gets the other.

Rows land in `speed/`, NOT in `sweeps/`. That separation is load-bearing:
fit.load_sweep() globs every .jsonl under sweeps/ and fits any row that
suspect_reason() accepts, and a speculative row would sail straight through that
filter. compute.py has no term for a draft cache, so those megabytes would be
absorbed into `floor` and `ctx` and quietly corrupt every future plan. Same
format, same resume discipline, different tree.
"""
import json, os, re, statistics, time, urllib.error, urllib.request
from .gpu import gpu_list
from .lmstudio import default_models_dir
from .paths import _data_dir
from .sweep import (build_argv, discover_models, finish_row, model_facts,
                    pick_backend, backends_dir, serve, sweep_path, _key, load_rows)


BENCH_PORT = 8232          # 8231 is the allocation sweep's; 1234 is LM Studio's

N_PREDICT = 128            # tokens generated per measured pass

N_REPEAT = 3               # measured passes per config, median reported


def bench_dir():
    return os.path.join(_data_dir(), "speed")


def bench_path(gpu_name, build):
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s or "unknown")
    d = bench_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s__%s.jsonl" % (safe(gpu_name), safe(build)))


# ---------------------------------------------------------------------------
# Talking to the server
# ---------------------------------------------------------------------------
def _post(url, path, payload, timeout=600):
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def n_tokens(url, text, timeout=120):
    """Token count according to the model actually loaded.

    Using the server's own tokenizer is what lets "32k of context" mean 32k
    rather than a guess from a characters-per-token ratio, which is off by tens
    of percent between prose and code - exactly the two things this corpus mixes."""
    return len(_post(url, "/tokenize", {"content": text}, timeout).get("tokens") or [])


# The filler corpus is this repository: real prose and real Python, deterministic,
# always on disk next to the code that reads it, and representative of the mixed
# doc-and-source workload a coding daily-driver actually sees.
#
# This matters more than it looks. N-gram speculation predicts from repetition in
# the text, so a prompt built by repeating one paragraph would hand ngram-* a
# result it could never reproduce on real work, while uniformly random tokens
# would deny it one it genuinely deserves. Neither is a measurement. Real files
# are the only honest filler.
_CORPUS = None


def corpus_text(refresh=False):
    """Read once per process, then cached.

    The caching is not an optimisation, it is a correctness requirement. A
    campaign spans hours and re-reads this between configs; if the working tree
    is edited in the meantime - and it will be, since the harness and the files
    it reads live in the same repository - then later configs get a different
    prompt from earlier ones and the comparison quietly stops being one. Snapshot
    at the start, use the same bytes for every row."""
    global _CORPUS
    if _CORPUS is not None and not refresh:
        return _CORPUS
    here = os.path.dirname(os.path.abspath(__file__))
    parts = []
    readme = os.path.join(os.path.dirname(here), "README.md")
    if os.path.isfile(readme):
        parts.append(open(readme, encoding="utf-8", errors="replace").read())
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            parts.append(open(os.path.join(here, fn), encoding="utf-8",
                              errors="replace").read())
    _CORPUS = "\n\n".join(parts)
    return _CORPUS


INSTRUCTION = ("\n\nSummarise, in detail, what the code and documentation above "
               "do and how the pieces fit together.\n")


def build_prompt(url, fill_tokens, timeout=120):
    """A prompt of approximately `fill_tokens` tokens, ending in an instruction.

    Grows the corpus by repetition until it is long enough, then bisects on
    characters and re-tokenises to land close. Exactness is not the point -
    knowing the number is, so it goes in the row."""
    if not fill_tokens:
        return INSTRUCTION.strip(), None
    body = corpus_text()
    if not body.strip():
        body = "The quick brown fox jumps over the lazy dog. "
    per_char = max(1e-6, n_tokens(url, body[:20000], timeout) / 20000.0)
    want_chars = int(fill_tokens / per_char)
    # Past the length of the corpus there is nothing left to say, so the filler
    # starts repeating itself - and a verbatim repeat is trivially predictable,
    # which inflates any speculative acceptance rate measured on it. That does not
    # touch a non-speculative decode number (bandwidth does not care what the
    # tokens say) but it makes a speculative one optimistic, so the condition is
    # RECORDED rather than left for the reader to infer from the fill column.
    repeated = len(body) < want_chars
    while len(body) < want_chars:
        body = body + "\n\n" + body
    lo, hi = 0, min(len(body), int(want_chars * 1.6))
    best = body[:hi]
    for _ in range(6):
        mid = (lo + hi) // 2
        got = n_tokens(url, body[:mid], timeout)
        if got < fill_tokens:
            lo = mid
        else:
            hi = mid
            best = body[:mid]
        if abs(got - fill_tokens) < max(32, fill_tokens * 0.01):
            best = body[:mid]
            break
    return best + INSTRUCTION, repeated


def generate(url, prompt, n_predict=N_PREDICT, timeout=1800, cache_prompt=False):
    """One generation.

    temperature 0 makes the token stream reproducible across configs, and
    ignore_eos pins the work at exactly n_predict tokens - without it the model
    stops when it wants to and tok/s ends up measured over a handful of tokens,
    which is noise.

    `cache_prompt` is the difference between measuring prefill and measuring
    decode, and each pass wants one or the other. See bench_one().

    The whole `timings` object is kept verbatim rather than picked apart. It
    carries prompt_per_second as well as predicted_per_second, and on speculative
    runs it also carries draft_n / draft_n_accepted - the acceptance rate, which
    is the only thing that EXPLAINS a speculative result rather than just
    reporting it. Storing the dict whole means we get those whether or not a
    given build populates them."""
    d = _post(url, "/completion", {
        "prompt": prompt, "n_predict": int(n_predict), "temperature": 0,
        "ignore_eos": True, "cache_prompt": bool(cache_prompt), "stream": False},
        timeout)
    # A sample of what was actually generated. Not decoration: with temperature 0
    # and ignore_eos the model can fall into a repetition loop, and a repetition
    # loop is precisely what n-gram speculation predicts perfectly - so a
    # spectacular ngram acceptance rate can be an artefact of the harness rather
    # than a property of the setting. Keeping the text is what makes that
    # checkable after the fact instead of a suspicion.
    txt = d.get("content") or ""
    return {"timings": d.get("timings") or {},
            "tokens_predicted": d.get("tokens_predicted"),
            "tokens_evaluated": d.get("tokens_evaluated"),
            "sample": txt[:600],
            "distinct_ratio": _distinct_ratio(txt)}


def _distinct_ratio(text, n=8):
    """Fraction of n-word windows that are unique. Near 1.0 is healthy prose;
    down near 0 means the model is looping and any n-gram result is meaningless."""
    w = text.split()
    if len(w) < n + 1:
        return None
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return round(len(set(grams)) / len(grams), 3)


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------
def bench_one(backend, model_path, c, port=BENCH_PORT, timeout=420.0,
              n_predict=N_PREDICT, repeat=N_REPEAT, gen_timeout=1800, log=print):
    """Load one config, measure it, tear it down.

    Two kinds of pass, because prefill and decode want opposite things from the
    prompt cache:

      * ONE cold pass with cache_prompt off. It pays the full prefill, which is
        the only way to measure prefill, and its decode figure is thrown away -
        llama.cpp loads CUDA kernel modules lazily, so the opening pass pays for
        machinery every later pass gets free, and charging that to the config
        would penalise whichever one happened to run first.
      * `repeat` warm passes with cache_prompt on. They reuse the prefix, so each
        one is pure decode starting at the real context depth.

    Splitting them is what makes deep-fill measurement affordable at all. At the
    measured ~315 tok/s of prefill, a 120k-token prompt is six minutes; paying
    that four times per config would put a single row past twenty minutes."""
    log_dir = os.path.join(_data_dir(), "sweep-logs")
    row = {"status": "error", "config": dict(c),
           "model": os.path.basename(model_path), "backend": backend["build"],
           "speed": True, "n_predict": n_predict, "repeat": repeat}
    cfg = dict(c)
    cfg["warmup"] = True          # a speed run should not pay for lazy init
    with serve(backend, model_path, cfg, port=port, timeout=timeout,
               log_dir=log_dir, log_name="bench.log") as srv:
        if srv.ok:
            try:
                prompt, repeated = build_prompt(srv.url, c.get("fill") or 0)
                row["prompt_tokens"] = n_tokens(srv.url, prompt)
                row["corpus_repeated"] = bool(repeated)
                cold = generate(srv.url, prompt, min(16, n_predict), gen_timeout,
                                cache_prompt=False)
                row["cold"] = cold
                runs = [generate(srv.url, prompt, n_predict, gen_timeout,
                                 cache_prompt=True)
                        for _ in range(max(1, repeat))]
                row["runs"] = runs
                dec = [r["timings"].get("predicted_per_second") for r in runs]
                dec = [x for x in dec if x]
                pre = cold["timings"].get("prompt_per_second")
                row["tok_s"] = round(statistics.median(dec), 3) if dec else None
                row["tok_s_all"] = [round(x, 3) for x in dec]
                row["prefill_tok_s"] = round(pre, 1) if pre else None
                row["sample"] = runs[0].get("sample")
                dr = [r.get("distinct_ratio") for r in runs
                      if r.get("distinct_ratio") is not None]
                row["distinct_ratio"] = round(min(dr), 3) if dr else None
                # Acceptance is the number that explains a speculative result.
                # Absent on non-speculative runs, and absent on builds that do
                # not report it - both are fine, and both are visible as None
                # rather than as a fabricated zero.
                dn = [r["timings"].get("draft_n") for r in runs]
                da = [r["timings"].get("draft_n_accepted") for r in runs]
                if any(dn) and any(da):
                    tot_n = sum(x for x in dn if x)
                    tot_a = sum(x for x in da if x)
                    row["draft_n"] = tot_n
                    row["draft_accepted"] = tot_a
                    row["accept_rate"] = round(tot_a / tot_n, 4) if tot_n else None
            except Exception as e:
                row["status"] = "genfail"
                row["gen_error"] = "%s: %s" % (type(e).__name__, e)
    finish_row(row, srv, log_dir=log_dir)
    # finish_row takes its verdict from the server, which came up fine; a
    # generation that then failed is still a failed measurement.
    if row.get("gen_error"):
        row["status"] = "genfail"
    elif row["status"] == "ok" and not row.get("tok_s"):
        row["status"] = "genfail"
        row["gen_error"] = "server reported no predicted_per_second"
    # The Windows failure mode is not an OOM. WDDM spills past dedicated VRAM
    # into system RAM and the load SUCCEEDS, so the row says "ok" and only the
    # numbers give it away: a negative floor, and decode falling off a cliff.
    # Naming it on the row means the ngl ladder can be read without knowing that.
    row["spilled"] = bool(row.get("suspect")) and row.get("status") == "ok"
    return row


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
# Fixed for this campaign and not swept: context and KV quant are the two the
# user pinned, flash attention is mandatory with a quantised cache, and -np 1
# because every extra sequence buys recurrent state nobody asked for.
SPEED_BASE = {"ctx": 131072, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512,
              "ngl": 26, "ncmoe": 0, "fill": 2048, "spec": "none",
              "mmproj_offload": True}

# Stage A finds the wall, and the stages after it start from what A found. That
# ordering is the whole design: an ngl that spills makes every later comparison a
# comparison of spill behaviour.
STAGE_A_NGL = [26, 28, 30, 31, 32, 33, 34]

STAGE_C_UB = [256, 512, 1024, 2048]

STAGE_D_SPEC = [("none", 0), ("draft-mtp", 1), ("draft-mtp", 2), ("draft-mtp", 3),
                ("draft-mtp", 5), ("ngram-mod", 0), ("ngram-cache", 0),
                ("ngram-simple", 0)]


def build_speed_grid(facts, mmproj=None, base=None, stages="abcd"):
    """The staged grid, one knob at a time from a baseline.

    Deliberately not a cross product - the same reasoning as sweep.build_grid,
    where a confounded design fitted a coefficient at 9x its prior. Here the cost
    is worse than a bad fit: a cross product of these axes is hundreds of loads
    at roughly two minutes each."""
    b = dict(base or SPEED_BASE)
    if mmproj:
        b["mmproj"] = mmproj
    nl = facts.get("n_layers") or 65
    cfgs, seen = [], set()

    def add(**kw):
        c = dict(b)
        c.update(kw)
        c["ngl"] = min(c["ngl"], nl)
        k = tuple(sorted((str(x), str(y)) for x, y in c.items()))
        if k not in seen:
            seen.add(k)
            cfgs.append(c)

    if "a" in stages:
        for v in STAGE_A_NGL:
            add(ngl=v, stage="A")
    if "b" in stages and mmproj:
        # The projector can sit in system RAM instead. Text decode is unaffected,
        # so this is 885 MiB of VRAM back - several blocks - without giving up
        # images. Climb ngl again from where A ended, since the ceiling moved.
        for v in STAGE_A_NGL[1:] + [35, 36]:
            add(ngl=v, mmproj_offload=False, stage="B")
    if "c" in stages:
        for v in STAGE_C_UB:
            add(ub=v, stage="C")
    if "d" in stages:
        for sp, nmax in STAGE_D_SPEC:
            add(spec=sp, spec_n_max=nmax, stage="D")
    return cfgs


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def find_mmproj_for(model_path):
    d = os.path.dirname(os.path.abspath(model_path))
    for fn in sorted(os.listdir(d)):
        if fn.lower().startswith("mmproj") and fn.lower().endswith(".gguf"):
            return os.path.join(d, fn)
    return None


def preflight(log=print, need_free_pct=0.85):
    """Refuse to measure against a card someone else is already using.

    LM Studio holding a model resident does not make a run fail - it makes every
    row in the campaign wrong in the same direction, which is worse, because the
    numbers still look like numbers."""
    g = gpu_list(fresh=True)
    if not g:
        return True
    free, total = g[0].get("free_mib") or 0, g[0].get("total_mib") or 0
    if total and free < total * need_free_pct:
        log("Only %.0f of %.0f MiB VRAM is free (%.0f%%). Something is holding the "
            "card - LM Studio with a model loaded, most likely. Every row would be "
            "measured against a budget that is not yours.\nClose it and re-run."
            % (free, total, 100.0 * free / total))
        return False
    return True


def speed_sweep(models=None, backend=None, dry_run=False, timeout=420.0,
                port=BENCH_PORT, limit=None, axes=None, stages="abcd", fill=None,
                n_predict=N_PREDICT, repeat=N_REPEAT, log=print, skip_preflight=False):
    """Run the speed grid and append one row per config. Resumable like --sweep."""
    b = pick_backend(backend)
    if not b:
        log("No llama-server build found under %s" % backends_dir())
        return None
    paths = models if models and os.path.isfile(str(models[0])) else \
        discover_models(names=models)
    if not paths:
        log("No models found under %s" % default_models_dir())
        return None
    mp = paths[0]
    if len(paths) > 1:
        log("%d models matched; benchmarking %s" % (len(paths), os.path.basename(mp)))
    facts = model_facts(mp)
    mmproj = find_mmproj_for(mp)

    base = dict(SPEED_BASE)
    if fill is not None:
        base["fill"] = fill
    cfgs = build_speed_grid(facts, mmproj=mmproj, base=base, stages=stages)
    if axes:
        # An explicit ladder replaces the staged grid: this is how a stage gets
        # re-run at the ngl the previous stage actually settled on.
        combos = [dict(base, **({"mmproj": mmproj} if mmproj else {}))]
        for k, vals in axes.items():
            combos = [dict(c, **{k: v}) for c in combos for v in vals]
        cfgs = combos

    gpu = (gpu_list(fresh=True) or [{}])[0].get("name") or ""
    path = bench_path(gpu, b["build"])
    # Resume must also match how the row was MEASURED, not just what was
    # configured. A row taken at n_predict 32 / repeat 1 - a smoke test - is not
    # the same measurement as one taken at 128 / 3, and silently accepting it as
    # already-done would put a noisier number into the comparison than every
    # other row and give no sign it had happened.
    done = {_key(r["model"], r["config"]) for r in load_rows(path)
            if r.get("config") and r.get("status") in ("ok", "oom")
            and r.get("n_predict") == n_predict and r.get("repeat") == repeat}
    plan = [c for c in cfgs if _key(os.path.basename(mp), c) not in done]
    skipped = len(cfgs) - len(plan)
    if limit:
        plan = plan[:limit]

    log("backend : %s" % b["build"])
    log("gpu     : %s" % (gpu or "unknown"))
    log("model   : %s  (%d blocks, %s)" % (os.path.basename(mp), facts["n_layers"],
                                           facts["arch"]))
    log("mmproj  : %s" % (os.path.basename(mmproj) if mmproj else "none"))
    log("output  : %s" % path)
    log("configs : %d to run, %d already recorded" % (len(plan), skipped))
    log("estimate: ~%.1f h  (load + %d warm + %d x %d tokens per config)"
        % (len(plan) * (60.0 + repeat * n_predict / 4.0) / 3600.0, 16, repeat, n_predict))
    if dry_run:
        for c in plan:
            log("  %-2s ngl %-3d ub %-5d fill %-7d spec %-13s nmax %-2d mmproj %s"
                % (c.get("stage", "-"), c["ngl"], c["ub"], c.get("fill") or 0,
                   c.get("spec") or "none", c.get("spec_n_max") or 0,
                   "vram" if c.get("mmproj_offload") is not False else "ram"))
        return {"planned": len(plan), "path": path, "dry_run": True}
    if not skip_preflight and not preflight(log=log):
        return None

    log("")
    log("%-2s %-4s %-5s %-8s %-13s %-4s %-6s | %8s %9s %7s %6s"
        % ("st", "ngl", "ub", "fill", "spec", "nmax", "mmproj",
           "tok/s", "prefill", "VRAM", "accept"))
    out = []
    with open(path, "a", encoding="utf-8") as fh:
        for i, c in enumerate(plan, 1):
            row = bench_one(b, mp, c, port=port, timeout=timeout,
                            n_predict=n_predict, repeat=repeat, log=log)
            row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                        "when": int(time.time()), "gpu": gpu})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            out.append(row)
            log(_fmt_row(c, row, i, len(plan)))
    log("")
    log("%d rows -> %s" % (len(out), path))
    return {"rows": out, "path": path}


def _fmt_row(c, row, i=None, n=None):
    head = ("%-2s %-4d %-5d %-8d %-13s %-4d %-6s | "
            % (c.get("stage", "-"), c["ngl"], c["ub"], c.get("fill") or 0,
               c.get("spec") or "none", c.get("spec_n_max") or 0,
               "vram" if c.get("mmproj_offload") is not False else "ram"))
    if row.get("status") != "ok":
        return head + "%s %s" % (row["status"].upper(),
                                 row.get("gen_error") or row.get("error") or "")
    return head + "%8.2f %9.1f %7.0f %6s%s" % (
        row.get("tok_s") or 0.0, row.get("prefill_tok_s") or 0.0,
        row.get("proc_vram_mib") or 0.0,
        ("%.0f%%" % (100 * row["accept_rate"])) if row.get("accept_rate") else "-",
        ("  SPILLED" if row.get("spilled") else "")
        + ("  [filler repeats - speculative rate inflated]"
           if row.get("corpus_repeated") and row["config"].get("spec") not in (None, "none")
           else ""))


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------
def load_speed_rows(path=None):
    paths = [path] if path else (
        [os.path.join(bench_dir(), f) for f in sorted(os.listdir(bench_dir()))
         if f.endswith(".jsonl")] if os.path.isdir(bench_dir()) else [])
    rows = []
    for p in paths:
        rows.extend(load_rows(p))
    return rows


def report(path=None, log=print):
    """Every measured config, fastest first."""
    rows = [r for r in load_speed_rows(path) if r.get("status") == "ok" and r.get("tok_s")]
    if not rows:
        log("No speed rows recorded yet. Run --speed-sweep.")
        return False
    rows.sort(key=lambda r: -(r.get("tok_s") or 0))
    log("%-30s %-4s %-5s %-8s %-13s %-4s %-6s %8s %9s %7s %7s"
        % ("model", "ngl", "ub", "fill", "spec", "nmax", "mmproj", "tok/s",
           "prefill", "VRAM", "accept"))
    for r in rows:
        c = r["config"]
        log("%-30s %-4d %-5d %-8d %-13s %-4d %-6s %8.2f %9.1f %7.0f %7s%s"
            % (r["model"][:30], c["ngl"], c["ub"], c.get("fill") or 0,
               c.get("spec") or "none", c.get("spec_n_max") or 0,
               "vram" if c.get("mmproj_offload") is not False else "ram",
               r["tok_s"], r.get("prefill_tok_s") or 0.0,
               r.get("proc_vram_mib") or 0.0,
               ("%.0f%%" % (100 * r["accept_rate"])) if r.get("accept_rate") else "-",
               "  SPILLED" if r.get("spilled") else ""))
    return True
