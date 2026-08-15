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
import datetime, hashlib, json, os, re, socket, statistics, time, urllib.error, urllib.request
from .gpu import get_gpu_processes, gpu_list
from .lmstudio import default_models_dir
from .paths import _data_dir
from .sweep import (build_argv, discover_models, finish_row, model_facts,
                    pick_backend, backends_dir, serve, sweep_path, _key, load_rows)


BENCH_PORT = 8232          # 8231 is the allocation sweep's; 1234 is LM Studio's


def free_port(preferred=BENCH_PORT):
    """A port llama-server can actually bind, right now.

    One fixed port is not safe to reuse back to back. A campaign tears a server
    down and starts the next within seconds, but the old socket sits in
    TIME_WAIT for a minute or more afterwards, and llama-server does not set
    SO_REUSEADDR - it prints "couldn't bind HTTP server socket" and exits.

    That is not a hypothetical: it cost two rows of the 32k campaign, recorded
    as EXIT. EXIT reads like a crash or a driver fault, so those configs looked
    like evidence about the wall when they were nothing but a port collision -
    a harness failure wearing a result's clothes.

    Probing with bind() mirrors exactly what llama-server is about to attempt,
    and a socket that never listened leaves no TIME_WAIT of its own. The
    fallback asks the OS for any free port rather than guessing at offsets."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", int(preferred)))
        return int(preferred)
    except OSError:
        pass
    finally:
        s.close()
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()

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


# The filler corpus is a FROZEN snapshot of this repository: real prose and real
# Python, deterministic, always on disk next to the code that reads it, and
# representative of the mixed doc-and-source workload a coding daily-driver
# actually sees.
#
# It is a snapshot, not a live read, and that is the point. The corpus used to
# be re-read from the working tree, so a campaign spanning an edit - and the
# harness and the files it reads live in the same repository, so edits happen -
# quietly measured a different prompt before and after, with nothing on the row
# to say so. Benchmarks need frozen inputs: the corpus only changes when
# --refresh-corpus deliberately rebuilds it, which is a committed, visible
# event. Rows carry prompt_identity(), the hash of exactly the bytes a prompt
# is built from, so even a deliberate refresh can never be mistaken for the
# campaign it replaced.
#
# This matters more than it looks. N-gram speculation predicts from repetition in
# the text, so a prompt built by repeating one paragraph would hand ngram-* a
# result it could never reproduce on real work, while uniformly random tokens
# would deny it one it genuinely deserves. Neither is a measurement. Real files
# are the only honest filler.
_CORPUS = None
_CORPUS_ID = None

_CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_corpus.txt")


def _live_corpus_text():
    """README and this package's sources, joined exactly as the corpus always was."""
    here = os.path.dirname(os.path.abspath(__file__))
    parts = []
    readme = os.path.join(os.path.dirname(here), "README.md")
    if os.path.isfile(readme):
        parts.append(open(readme, encoding="utf-8", errors="replace").read())
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            parts.append(open(os.path.join(here, fn), encoding="utf-8",
                              errors="replace").read())
    return "\n\n".join(parts)


def refresh_corpus():
    """Rewrite the frozen corpus snapshot from the live sources.

    The one way the corpus changes. Deliberate, committed, and visible in the
    row store: the new bytes hash differently, so every row recorded against
    the old snapshot stops being comparable to new ones and a resumed campaign
    re-measures instead of reusing them."""
    global _CORPUS, _CORPUS_ID
    text = _live_corpus_text()
    with open(_CORPUS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    _CORPUS = text
    _CORPUS_ID = None
    return _CORPUS_PATH


def corpus_text(refresh=False):
    """The frozen snapshot, read once per process, then cached.

    The caching is not an optimisation, it is a correctness requirement. A
    campaign spans hours and must use the same bytes for every row. If the
    snapshot is missing - a fresh clone, or a package that never ran
    --refresh-corpus - it is rebuilt from the live sources on the spot, so
    nothing silently runs on an empty prompt."""
    global _CORPUS
    if _CORPUS is not None and not refresh:
        return _CORPUS
    if os.path.isfile(_CORPUS_PATH):
        with open(_CORPUS_PATH, encoding="utf-8", errors="replace") as fh:
            _CORPUS = fh.read()
    else:
        _CORPUS = _live_corpus_text()
    return _CORPUS


INSTRUCTION = ("\n\nSummarise, in detail, what the code and documentation above "
               "do and how the pieces fit together.\n")

# Bumped whenever the way a prompt is ASSEMBLED changes, not just its bytes.
# "chat" is the move from raw /completion continuation to the model's own chat
# template; rows measured either side of it are different experiments even
# though the corpus is identical, and the id is what stops them merging.
PROMPT_SCHEME = "chat"


def apply_template(url, text, timeout=120):
    """Wrap `text` as a user turn using the SERVER's own chat template.

    /completion is a raw-continuation endpoint: it hands the model exactly these
    bytes, with no role markers and nothing marking the text as a REQUEST. An
    instruction-tuned model given a wall of source that happens to end in a
    sentence has no signal that it was asked anything, so it does the only thing
    the endpoint asks for - it continues the document. In practice that means
    echoing the instruction back or copying the source, which is what every row
    of the 32k campaign did. That was never the model failing at depth; it was
    the benchmark measuring a base model that does not exist.

    Templating is also the only way these numbers describe what actually runs:
    production serves /v1/chat/completions with --jinja and a template.

    Returns (prompt, applied). A build without /apply-template falls back to the
    raw text rather than failing the campaign - visibly, because `applied` lands
    in every row and the scheme lands in the prompt id."""
    try:
        d = _post(url, "/apply-template",
                  {"messages": [{"role": "user", "content": text}]}, timeout)
    except Exception:
        return text, False
    out = d.get("prompt") if isinstance(d, dict) else None
    if isinstance(out, str) and out:
        return out, True
    return text, False


def prompt_identity():
    """The hash of exactly the bytes a prompt is built from: corpus + INSTRUCTION.

    Two rows with the same prompt_id and the same fill were measured against
    byte-identical prompts, whatever the working tree said when each was
    measured - which is what makes deep-fill numbers comparable across sessions
    at all. Rows without the field predate the freeze and are their own
    experiment; they group separately and are never resumed as already done."""
    global _CORPUS_ID
    if _CORPUS_ID is None:
        h = hashlib.sha256()
        h.update(corpus_text().encode("utf-8"))
        h.update(INSTRUCTION.encode("utf-8"))
        h.update(PROMPT_SCHEME.encode("utf-8"))
        _CORPUS_ID = h.hexdigest()
    return _CORPUS_ID


def build_prompt(url, fill_tokens, timeout=120):
    """A prompt of approximately `fill_tokens` tokens, ending in an instruction.

    Grows the corpus by repetition until it is long enough, then bisects on
    characters and re-tokenises to land close. Exactness is not the point -
    knowing the number is, so it goes in the row."""
    if not fill_tokens:
        p, applied = apply_template(url, INSTRUCTION.strip(), timeout)
        return p, None, applied
    body = corpus_text()
    if not body.strip():
        body = "The quick brown fox jumps over the lazy dog. "
    # The template's own markers cost tokens. Measured once against empty text
    # and subtracted from the target, so `fill 32768` still means a 32768-token
    # prompt rather than 32768 plus however many this model's template adds -
    # otherwise the depth a row claims and the depth it measured would drift
    # apart by a per-model constant.
    shell, applied = apply_template(url, "", timeout)
    overhead = n_tokens(url, shell, timeout) if applied else 0
    fill_tokens = max(1, fill_tokens - overhead)
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
    prompt, applied = apply_template(url, best + INSTRUCTION, timeout)
    return prompt, repeated, applied


def sampling_of(c):
    """The sampler settings for one config, defaulting to greedy.

    Greedy is the right DEFAULT because it makes the token stream reproducible
    across configs, so two rows differ by the setting under test and nothing
    else. It is the wrong thing to draw conclusions from when the question is
    speculative decoding: llama.cpp accepts a draft token when the target's own
    sampled token matches it, and under greedy that comparison is deterministic.
    Greedy is therefore speculation's best case, and an acceptance rate measured
    there is an upper bound on the one you will actually see. Sweep the real
    sampler settings before believing a speculative number."""
    return {"temperature": float(c.get("temp", 0) or 0),
            "top_k": int(c.get("top_k", 0) or 0),
            "top_p": float(c.get("top_p", 1.0) if c.get("top_p") is not None else 1.0),
            "min_p": float(c.get("min_p", 0) or 0),
            "repeat_penalty": float(c.get("rep_pen", 1.0) if c.get("rep_pen") is not None else 1.0),
            "presence_penalty": float(c.get("pres_pen", 0) or 0)}


def generate(url, prompt, n_predict=N_PREDICT, timeout=1800, cache_prompt=False,
             sampling=None, seed=None):
    """One generation.

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
    body = {"prompt": prompt, "n_predict": int(n_predict),
            "ignore_eos": True, "cache_prompt": bool(cache_prompt), "stream": False}
    body.update(sampling or {"temperature": 0})
    if seed is not None:
        # A fixed seed per repeat makes a sampled run reproducible, so the three
        # passes vary by nothing at all and the median means something.
        body["seed"] = int(seed)
    d = _post(url, "/completion", body, timeout)
    # A sample of what was actually generated. Not decoration: with temperature 0
    # and ignore_eos the model can fall into a repetition loop, and a repetition
    # loop is precisely what n-gram speculation predicts perfectly - so a
    # spectacular ngram acceptance rate can be an artefact of the harness rather
    # than a property of the setting. Keeping the text is what makes that
    # checkable after the fact instead of a suspicion.
    txt = d.get("content") or ""
    return {"timings": d.get("timings") or {},
            "seed": seed,
            "tokens_predicted": d.get("tokens_predicted"),
            "tokens_evaluated": d.get("tokens_evaluated"),
            "sample": txt[:600],
            "distinct_ratio": _distinct_ratio(txt),
            "copyback_ratio": _copyback_ratio(txt, prompt)}


def _distinct_ratio(text, n=8):
    """Fraction of n-word windows that are unique. Near 1.0 is healthy prose;
    down near 0 means the model is looping and any n-gram result is meaningless."""
    w = text.split()
    if len(w) < n + 1:
        return None
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return round(len(set(grams)) / len(grams), 3)


def _copyback_ratio(text, prompt, n=8):
    """Fraction of n-word windows in the output that appear verbatim in the prompt.

    Near 0.0 is a model doing its own work. Near 1.0 the model stopped
    generating and is copying its context back at you - which distinct_ratio
    cannot see, because a copy's windows are all distinct and it reads a clean
    1.00. Copying inflates speculative acceptance exactly as much as looping
    does: a drafter that predicts the next prompt word is trivially accepted by
    a target that then samples the same prompt word. Two symptoms, one meaning."""
    w = text.split()
    pw = prompt.split()
    if len(w) < n + 1 or len(pw) < n + 1:
        return None
    pgrams = set(zip(*(pw[i:] for i in range(n))))
    hits = sum(1 for g in zip(*(w[i:] for i in range(n))) if g in pgrams)
    return round(hits / (len(w) - n + 1), 3)


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
    # Resolved per config, not once per campaign: the collision only appears
    # BETWEEN rows, when the previous server's socket is still winding down.
    port = free_port(port)
    with serve(backend, model_path, cfg, port=port, timeout=timeout,
               log_dir=log_dir, log_name="bench.log") as srv:
        if srv.ok:
            try:
                prompt, repeated, templated = build_prompt(srv.url,
                                                           c.get("fill") or 0)
                row["prompt_tokens"] = n_tokens(srv.url, prompt)
                row["corpus_repeated"] = bool(repeated)
                # False means this build has no /apply-template and the row was
                # measured on raw continuation - the mode that produced 23
                # unusable rows. Recorded per row because it is a property of
                # the BACKEND, so it can differ between two rows in one file.
                row["templated"] = bool(templated)
                # Which frozen corpus this row measured against. Everything a
                # deep-fill number means lives or dies on this: two rows from
                # different corpora are different experiments, and resume and
                # every comparison below treat them that way.
                row["prompt_id"] = prompt_identity()
                samp = sampling_of(c)
                row["sampling"] = samp
                cold = generate(srv.url, prompt, min(16, n_predict), gen_timeout,
                                cache_prompt=False, sampling=samp, seed=1000)
                row["cold"] = cold
                runs = [generate(srv.url, prompt, n_predict, gen_timeout,
                                 cache_prompt=True, sampling=samp, seed=1000 + i)
                        for i in range(max(1, repeat))]
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
                cb = [r.get("copyback_ratio") for r in runs
                      if r.get("copyback_ratio") is not None]
                row["copyback_ratio"] = round(min(cb), 3) if cb else None
                # Acceptance is the number that explains a speculative result.
                # Absent on non-speculative runs, and absent on builds that do
                # not report it - both are fine, and both are visible as None
                # rather than as a fabricated zero.
                dn = [r["timings"].get("draft_n") for r in runs]
                da = [r["timings"].get("draft_n_accepted") for r in runs]
                # `any(da)` would treat a genuine 0% acceptance as "not reported"
                # and drop the whole group, so the row - and the launcher header
                # built from it - would say nothing where it should say that
                # speculation drafted N tokens and got none of them back. That is
                # the most useful thing a speculative row can tell you. Test for
                # PRESENCE instead, which still excludes builds that do not report
                # the field at all.
                if any(dn) and any(x is not None for x in da):
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
#
# The rungs are derived from the planner rather than fixed, because a fixed ladder
# is only ever right for the model it was written against. planner_ngl() asks
# plan.analyze() where the layers land for THIS model on THIS card, and the ladder
# brackets that: a few rungs below it, and enough above to find the real wall,
# since the planner is deliberately conservative and the measured ceiling is
# normally higher than it predicts.
NGL_BELOW, NGL_ABOVE = 2, 6


def planner_split(model_path, c):
    """Where the planner thinks the split falls, for seeding the ladder.

    Returns (n_gpu_layers, n_cpu_moe). The second only means anything on an MoE,
    and on an MoE it is the one that matters - see ngl_ladder().

    Imported here rather than at module scope only to keep the cost off the path
    of callers that never build a grid; plan sits above bench in the DAG, so the
    direction is fine."""
    from .plan import analyze                      # plan is above bench in the DAG
    from .gpu import get_gpus, get_ram
    try:
        g = (get_gpus() or [{}])[0]
        ram = get_ram() or {}
        # TOTAL, not free. The grid is built now and run later, and preflight()
        # refuses to run it unless the card is essentially empty - so free VRAM at
        # build time is transient state that has nothing to do with the conditions
        # the rows will be measured under. Seeding off it produces a garbage ladder
        # whenever anything happens to be loaded, which is exactly when someone is
        # most likely to be planning their next sweep.
        r = analyze(model_path, int(c["ctx"]), c["kv"], int(c["ub"]), bool(c["fa"]),
                    vram_budget_mib=float(g.get("total_mib") or 0),
                    ram_budget_mib=float(ram.get("total_mib") or 0),
                    gpu_reserve_mib=512, compute_override_mib=None, safety_pct=5,
                    n_seq=int(c.get("seq") or 1),
                    include_mmproj=(c.get("mmproj_offload") is not False),
                    mtp_spec=bool(c.get("spec") == "draft-mtp"))
        pl = r.get("plan") or {}
        n = pl.get("n_gpu_layers")
        return (int(n) if n is not None else None,
                int(pl.get("n_cpu_moe") or 0))
    except Exception:
        return None, None


def ngl_ladder(model_path, c, n_layers, is_moe=False):
    """The stage-A ladder, and which axis it walks.

    On a DENSE model the axis is -ngl: whole blocks move together, and the wall is
    the largest count that fits.

    On an MoE it is --n-cpu-moe, and -ngl is pinned at every block. The two knobs
    are not interchangeable and the difference is most of the model:
      * -ngl N       puts the last N blocks on the GPU - attention, KV and experts
                     together.
      * --n-cpu-moe M moves only the ROUTED EXPERTS of the first M blocks to the
                     CPU, leaving their attention and KV in VRAM.
    Experts are the only weights big enough to be worth moving (92% of the file on
    Qwen3.6-35B-A3B) and only n_expert_used of n_expert fire per token, so exiling
    them costs far less per token than exiling whole blocks - while whole-block
    offload drags KV to the CPU with it, which is the expensive thing to lose.
    Laddering -ngl on an MoE therefore measures the fallback strategy and never
    finds the good configuration at all.

    Lower n_cpu_moe means more experts resident, so here the wall is the SMALLEST
    value that fits and the ladder mostly walks downward from the planner's seed."""
    seed_ngl, seed_ncm = planner_split(model_path, c)
    if is_moe:
        if seed_ncm is None:
            seed_ncm = max(1, int(n_layers * 0.5))
        lo = max(0, seed_ncm - NGL_ABOVE)          # fewer on CPU = faster, if it fits
        hi = min(n_layers, seed_ncm + NGL_BELOW)
        return list(range(lo, hi + 1)), seed_ncm
    if seed_ngl is None:
        seed_ngl = max(1, int(n_layers * 0.4))     # nothing better to go on
    lo = max(0, seed_ngl - NGL_BELOW)
    hi = min(n_layers, seed_ngl + NGL_ABOVE)
    return list(range(lo, hi + 1)), seed_ngl

STAGE_C_UB = [256, 512, 1024, 2048]

STAGE_D_SPEC = [("none", 0), ("draft-mtp", 1), ("draft-mtp", 2), ("draft-mtp", 3),
                ("draft-mtp", 5), ("ngram-mod", 0), ("ngram-cache", 0),
                ("ngram-simple", 0)]


def grid_context(facts, mmproj=None, base=None, model_path=None, carried=None):
    """(baseline config, n_layers, is_moe, stage-A ladder) - what every stage needs.

    Split out of build_speed_grid() so a chained campaign can rebuild it between
    stages against a baseline it has actually MEASURED, rather than against the
    planner's opening guess.

    `carried` is such a baseline. When given it supplies the starting config and
    re-centres the ladder on the split that won, instead of on the planner's seed.
    That matters most in a second round: with speculation switched on, the draft
    cache costs VRAM the planner priced under different assumptions, so the wall
    genuinely moves and the ladder has to move with it."""
    b = dict(base or SPEED_BASE)
    if mmproj:
        b["mmproj"] = mmproj
    nl = facts.get("n_layers") or 65
    # A context past what the model was trained on is not a config, it is a bad
    # request - and a staged grid carries a default context belonging to whichever
    # model it was last used against.
    trained = facts.get("n_ctx_train") or 0
    if trained and b.get("ctx", 0) > trained:
        b["ctx"] = trained
    is_moe = bool(facts.get("is_moe"))
    rungs, seed = ngl_ladder(model_path, b, nl, is_moe) if model_path else (
        [min(v, nl) for v in range(max(1, nl // 3), min(nl, nl // 2 + 4))], None)
    if is_moe:
        # Every block on the GPU; the expert split is what varies. Stages C/D then
        # sit at the planner's n_cpu_moe rather than at some layer count.
        b["ngl"] = nl
        if seed is not None:
            b["ncmoe"] = min(seed, nl)
    elif seed:
        b["ngl"] = min(seed, nl)      # stages C/D sit at the planner's split
    if carried:
        for k, v in carried.items():
            if k != "stage":
                b[k] = v
        # Same asymmetry as ngl_ladder(): on an MoE the wall is the SMALLEST
        # n_cpu_moe that fits, so the ladder reaches further down than up.
        if is_moe:
            c0 = int(carried.get("ncmoe") or 0)
            rungs = list(range(max(0, c0 - NGL_ABOVE), min(nl, c0 + NGL_BELOW) + 1))
        else:
            n0 = int(carried.get("ngl") or 0)
            rungs = list(range(max(0, n0 - NGL_BELOW), min(nl, n0 + NGL_ABOVE) + 1))
    return b, nl, is_moe, rungs


def stage_configs(letter, b, nl, is_moe, rungs, mmproj=None, facts=None):
    """The configs one stage varies, from the baseline it is handed.

    One knob at a time, deliberately - the same reasoning as sweep.build_grid,
    where a confounded design fitted a coefficient at 9x its prior. Here the cost
    is worse than a bad fit: a cross product of these axes is hundreds of loads at
    roughly two minutes each."""
    facts = facts or {}
    out = []

    def add(**kw):
        c = dict(b)
        c.update(kw)
        c["ngl"] = max(0, min(c["ngl"], nl))
        out.append(c)

    if letter == "a":
        for v in rungs:
            add(ncmoe=v, stage="A") if is_moe else add(ngl=v, stage="A")
    elif letter == "b" and mmproj:
        # The projector can sit in system RAM instead. Text decode is unaffected,
        # so those megabytes come back as GPU blocks without giving up images -
        # which means the ceiling moves up, so the ladder has to reach higher.
        if is_moe:
            # The freed megabytes buy back EXPERTS here, not blocks, so the ladder
            # walks down from the seed rather than up.
            for v in range(max(0, rungs[0] - NGL_ABOVE), rungs[-1] + 1):
                add(ncmoe=v, mmproj_offload=False, stage="B")
        else:
            for v in range(rungs[0], min(nl, rungs[-1] + NGL_ABOVE) + 1):
                add(ngl=v, mmproj_offload=False, stage="B")
    elif letter == "c":
        for v in STAGE_C_UB:
            add(ub=v, stage="C")
    elif letter == "d":
        # draft-mtp needs nextn blocks in the file. Without them llama.cpp has
        # nothing to draft from, so those rows are four wasted loads that all fail
        # the same way - and the n-gram variants, which need nothing, are the only
        # speculation such a model can use.
        has_mtp = bool(facts.get("n_mtp_layers"))
        for sp, nmax in STAGE_D_SPEC:
            if sp == "draft-mtp" and not has_mtp:
                continue
            add(spec=sp, spec_n_max=nmax, stage="D")
    return out


def _dedupe(cfgs, seen=None):
    """Drop configs already produced, by full value. `seen` carries across calls."""
    seen = set() if seen is None else seen
    out = []
    for c in cfgs:
        k = tuple(sorted((str(x), str(y)) for x, y in c.items()))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def build_speed_grid(facts, mmproj=None, base=None, stages="abcd", model_path=None):
    """The whole staged grid, one knob at a time from one fixed baseline.

    Stages C and D are pinned at the planner's ngl rather than at whatever stage A
    turns out to find, because nothing here can know stage A's result before stage
    A has run. That is what `chain=True` in speed_sweep() fixes, by building the
    stages one at a time instead of all at once; this function stays as the
    unchained grid and as the size the estimate is computed from."""
    b, nl, is_moe, rungs = grid_context(facts, mmproj, base, model_path)
    cfgs, seen = [], set()
    for letter in "abcd":
        if letter in stages:
            cfgs.extend(_dedupe(
                stage_configs(letter, b, nl, is_moe, rungs, mmproj, facts), seen))
    return cfgs


# ---------------------------------------------------------------------------
# Chaining: what one stage hands to the next
# ---------------------------------------------------------------------------
# A challenger has to beat the incumbent by more than this to take over as the
# baseline. tok_s is a median of `repeat` passes, so a 1% lead is jitter, and
# rebasing on jitter makes the campaign's PATH depend on noise rather than on
# anything it measured - the same grid would then explore different configs on
# two runs of the same machine.
CHAIN_MARGIN = 0.02

# Neutral value per config key, so a row that omits one groups with a row that
# sets it explicitly to its default. Anything absent here defaults to None.
_CONFIG_DEFAULTS = {"ncmoe": 0, "spec_n_max": 0, "fill": 0, "seq": 1,
                    "temp": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                    "rep_pen": 1.0, "pres_pen": 0.0}

# What carries forward. Not ctx / kv / fill / seq / fa: those are frozen by the
# form and are the campaign's definition rather than any of its results. Not
# `stage`, which is a label.
CARRY_KEYS = ("ngl", "ncmoe", "ub", "mmproj_offload", "spec", "spec_n_max")

# Below this fraction of unique 8-word windows the output is repetition, not
# work. One constant because the run-time table and the trustworthy() gate must
# agree: a row warned about while running and then kept for a conclusion - or
# dropped without ever having been flagged - would be worse than either alone.
LOOP_RATIO = 0.5

# Above this fraction of 8-word windows that also appear in the prompt, the
# output is a verbatim copy of the context - the second way a model can look
# healthy while generating nothing new. distinct_ratio stays near 1.0 for a
# copy, so the two gates are complements, and they share the marker-and-gate
# rule above for the same reason.
COPY_RATIO = 0.5


def trustworthy(r):
    """May a conclusion be drawn from this row?

    A spilled row loaded and reported `ok`: WDDM put part of it in shared system
    memory instead of failing, so its speed is off a cliff for a reason that has
    nothing to do with the setting under test. A looping row generated the same
    eight-word window over and over. A copying row stopped generating and is
    echoing its context back - distinct_ratio cannot see it (a copy's windows
    are all distinct), but it inflates speculative acceptance exactly as much
    as looping does.

    All three still belong in the TABLE - they are evidence about where the
    wall is - but none may be the thing a baseline or an effect size is
    computed from. Carrying a spilled row forward as a chained baseline would
    bend every stage after it in the same direction, silently.

    A row recorded before the copy gate existed has no copyback_ratio and is
    judged on the gates it does carry - history is not rewritten, it is just
    labelled (see sweep_index's n_ungated)."""
    if r.get("status") != "ok" or not r.get("tok_s"):
        return False
    if r.get("spilled"):
        return False
    dr = r.get("distinct_ratio")
    if dr is not None and dr < LOOP_RATIO:
        return False
    cb = r.get("copyback_ratio")
    # Strict, mirroring LOOP_RATIO's: a row exactly at the line is not (yet) a copy.
    if cb is not None and cb > COPY_RATIO:
        return False
    return True


def comparable(r, model, base, n_predict=None, repeat=None, prompt_id=None):
    """Was this row measured under the same conditions as the campaign?

    Speed is conditional on all of these, so a row taken at another depth or with
    another KV quant is not a slower config - it is a different experiment, and
    treating it as a rival would silently rewrite what the campaign is measuring.
    A 2k-fill row must never set the baseline for a 32k campaign, and neither
    must a greedy row set it for a campaign sweeping real sampler settings -
    greedy is speculation's best case, so those are two experiments and not two
    configs.

    `prompt_id` is the frozen corpus a campaign is measuring against. A row from
    another corpus - or from before the corpus was frozen at all - is a
    different experiment for exactly the same reason, so when the campaign
    passes its own id, the row must carry that same id."""
    if model and r.get("model") != model:
        return False
    if n_predict is not None and r.get("n_predict") != n_predict:
        return False
    if repeat is not None and r.get("repeat") != repeat:
        return False
    if prompt_id is not None and r.get("prompt_id") != prompt_id:
        return False
    c = r.get("config") or {}
    if not (c.get("ctx") == base.get("ctx")
            and c.get("kv") == base.get("kv")
            and (c.get("fill") or 0) == (base.get("fill") or 0)
            and (c.get("seq") or 1) == (base.get("seq") or 1)
            and bool(c.get("fa")) == bool(base.get("fa"))):
        return False
    for k in ("temp", "top_k", "top_p", "min_p", "rep_pen", "pres_pen"):
        d = _CONFIG_DEFAULTS[k]
        if (c.get(k) if c.get(k) is not None else d) != \
                (base.get(k) if base.get(k) is not None else d):
            return False
    return True


def best_config(rows, model, base, n_predict=None, repeat=None,
                incumbent_tok_s=None, margin=CHAIN_MARGIN, prompt_id=None):
    """The baseline for the next stage: (config, row) or (None, None).

    Reads rows that are already on DISK, not just the ones this process has in
    memory, so a campaign stopped after stage A and restarted tomorrow picks its
    winner back up instead of falling back to the planner's guess."""
    cand = [r for r in rows
            if trustworthy(r) and comparable(r, model, base, n_predict, repeat,
                                             prompt_id)]
    if not cand:
        return None, None
    win = max(cand, key=lambda r: r["tok_s"])
    if incumbent_tok_s and win["tok_s"] <= incumbent_tok_s * (1.0 + margin):
        return None, None
    c = dict(base)
    wc = win.get("config") or {}
    for k in CARRY_KEYS:
        if k in wc:
            c[k] = wc[k]
    c.pop("stage", None)
    return c, win


def resolve_search(axes, chain):
    """(chain, note) - which search actually runs when both were asked for.

    Chaining rebuilds each STAGE against the previous stage's winner, and an
    explicit --speed-axes ladder has no stages, so the two cannot both apply.
    The ladder is the more specific instruction and wins.

    Pulled out as its own function because the failure it prevents is silent:
    with both set, the campaign would take exactly the same hours and measure
    the staged grid instead of the ladder that was asked for, with nothing in
    the output to say so. A precedence rule worth stating out loud is worth
    being able to test."""
    if axes and chain:
        return False, ("note    : --speed-axes given, so chaining is off - an "
                       "explicit ladder has no stages to chain")
    return chain, None


def _carry_summary(c):
    return ("ngl %s ncmoe %s ub %s spec %s"
            % (c.get("ngl"), c.get("ncmoe") or 0, c.get("ub"),
               c.get("spec") or "none"))


def verify_config(win_c, overrides=None):
    """The exact config one verification load measures: the winner plus overrides.

    The winner carries the knobs the search settled on; the overrides carry
    everything that makes the config you actually run different - the spec, the
    draft depth, the samplers. Pure, so the campaign and the self-test agree on
    what gets loaded."""
    c = dict(win_c or {})
    c.pop("stage", None)
    for k, vals in (overrides or {}).items():
        if vals:
            c[k] = vals[-1]
    return c


def _verify_step(backend, model_path, facts, name, base, path, pid, gpu,
                 overrides=None, port=BENCH_PORT, timeout=420.0,
                 n_predict=N_PREDICT, repeat=N_REPEAT, log=print):
    """Load the campaign's winner at the PRODUCTION config, exactly once.

    The staged search measures each knob at the config the grid asked for, and
    the winner is the fastest row that fitted THOSE settings. The config a
    person actually launches can differ - the MTP draft cache, real samplers, a
    deeper fill - and a split that fitted one does not necessarily fit the
    other; a launcher carrying the winner's ngl into a config that OOMs on it
    is the quiet failure this exists for. One verification load settles it, and
    the row lands in the same file with the same prompt_id, so the certified
    answer is recorded data rather than a claim."""
    win_c, win_r = best_config(load_rows(path), name, base, n_predict=n_predict,
                               repeat=repeat, prompt_id=pid)
    if win_c is None:
        log("verify  : nothing to certify - no trustworthy row at this campaign's settings")
        return None
    c = verify_config(win_c, overrides)
    log("verify  : the winner (%s) does not know the config you actually run -" % _carry_summary(win_c))
    log("          loading %s%s once"
        % (_carry_summary(c),
           "  fill %s" % (c.get("fill") or 0)
           if c.get("fill") else ""))
    row = bench_one(backend, model_path, c, port=port, timeout=timeout,
                    n_predict=n_predict, repeat=repeat, log=log)
    row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                "when": int(time.time()), "gpu": gpu})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    good = row.get("status") == "ok" and trustworthy(row)
    if good:
        log("verify  : OK - the winner loads at your production config and is certified")
    else:
        log("verify  : FAILED - the winner does not fit your production config (%s%s)"
            % (row.get("status"), row.get("gen_error") or row.get("error") or ""))
        log("          the tok/s and VRAM evidence above apply to the config as MEASURED,")
        log("          not to this one. Lower ngl / draft depth, or drop the projector")
        log("          offload, before launching.")
    return {"ok": good, "row": _slim(row), "config": c}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def find_mmproj_for(model_path):
    d = os.path.dirname(os.path.abspath(model_path))
    for fn in sorted(os.listdir(d)):
        if fn.lower().startswith("mmproj") and fn.lower().endswith(".gguf"):
            return os.path.join(d, fn)
    return None


def preflight_state(need_free_pct=0.85):
    """Is the card free enough to measure against, and what is holding it?

    Split out from preflight() so the web UI can render the same verdict without
    capturing printed text - and so it can NAME the offender, which is the part
    that makes the block actionable rather than annoying."""
    g = gpu_list(fresh=True)
    st = {"ok": True, "gpu": "", "free_mib": None, "total_mib": None,
          "need_free_pct": need_free_pct, "holders": [], "reason": ""}
    if not g:
        # No GPU reading is not a reason to refuse: the sweep can still run, and
        # a missing nvidia-smi is not evidence that something holds the card.
        return st
    st["gpu"] = g[0].get("name") or ""
    free, total = g[0].get("free_mib") or 0, g[0].get("total_mib") or 0
    st["free_mib"], st["total_mib"] = free, total
    if not (total and free < total * need_free_pct):
        return st
    procs = get_gpu_processes()
    if isinstance(procs, list):
        st["holders"] = [{"name": p.get("name") or "?", "pid": p.get("pid"),
                          "mib": p.get("mib"), "is_engine": bool(p.get("is_engine"))}
                         for p in sorted(procs, key=lambda p: -(p.get("mib") or 0))]
    who = ", ".join("%s (%.0f MiB)" % (h["name"], h["mib"] or 0)
                    for h in st["holders"][:3])
    st["ok"] = False
    st["reason"] = ("Only %.0f of %.0f MiB VRAM is free (%.0f%%). Something is holding "
                    "the card%s. Every row would be measured against a budget that is "
                    "not yours." % (free, total, 100.0 * free / total,
                                    " - %s" % who if who else
                                    " - LM Studio with a model loaded, most likely"))
    return st


def preflight(log=print, need_free_pct=0.85):
    """Refuse to measure against a card someone else is already using.

    LM Studio holding a model resident does not make a run fail - it makes every
    row in the campaign wrong in the same direction, which is worse, because the
    numbers still look like numbers."""
    st = preflight_state(need_free_pct)
    if not st["ok"]:
        log(st["reason"] + "\nClose it and re-run.")
    return st["ok"]


def speed_sweep(models=None, backend=None, dry_run=False, timeout=420.0,
                port=BENCH_PORT, limit=None, axes=None, stages="abcd", fill=None,
                ctx=None, kv=None,
                n_predict=N_PREDICT, repeat=N_REPEAT, log=print, skip_preflight=False,
                on_row=None, should_stop=None, on_total=None,
                chain=False, rounds=1, verify=False, verify_overrides=None):
    """Run the speed grid and append one row per config. Resumable like --sweep.

    `on_row` is called with each finished row, `on_total` when the number of
    configs to run becomes known, and `should_stop` is checked before each config.
    All three default to None, so the CLI path is unchanged. They exist for the
    web UI: it needs structured rows as they land rather than a transcript to
    scrape, and it needs a way to end a two-hour campaign early.

    Stopping is deliberately BETWEEN configs. Killing a server mid-measurement
    would write a half-row, and there is no need: rows are resume-keyed by config
    AND by how they were measured, so a stopped campaign restarted later picks up
    exactly where it left off instead of re-measuring what it already has.

    `chain=True` turns the grid from one fixed plan into coordinate descent: each
    stage is built after the previous one has run, against the fastest row
    measured so far rather than against the planner's opening guess. Same number
    of loads, so the same hours - but stage C then measures ubatch at the layer
    split that actually won, instead of at one nothing has confirmed.

    `rounds` re-runs the stages from the winner. It is cheap by construction and
    needs no special case: _key() does not include the stage letter, so every
    config a later round revisits unchanged is already recorded and is skipped."""
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
    if ctx is not None:
        base["ctx"] = ctx
    if kv is not None:
        base["kv"] = kv
    cfgs = build_speed_grid(facts, mmproj=mmproj, base=base, stages=stages,
                            model_path=mp)
    if axes:
        # An explicit ladder replaces the staged grid: this is how a stage gets
        # re-run at the ngl the previous stage actually settled on.
        combos = [dict(base, **({"mmproj": mmproj} if mmproj else {}))]
        for k, vals in axes.items():
            combos = [dict(c, **{k: v}) for c in combos for v in vals]
        cfgs = combos
    chain, note = resolve_search(axes, chain)
    if note:
        log(note)

    gpu = (gpu_list(fresh=True) or [{}])[0].get("name") or ""
    path = bench_path(gpu, b["build"])
    # The frozen corpus every row of this campaign measures against. A row
    # recorded under a different prompt_id - another corpus, or before the
    # corpus was frozen at all - is a different experiment, and resuming it as
    # already done would mix two prompts into one campaign with no sign.
    pid = prompt_identity()
    # A missing snapshot degrades to "different experiment", not to corruption:
    # the rebuild hashes differently, so these rows refuse to merge with any
    # other checkout's. But it must be said out loud - silently re-basing a
    # campaign's identity on the working tree is the exact failure the freeze
    # exists to catch.
    if not os.path.isfile(_CORPUS_PATH):
        log("corpus  : _corpus.txt is MISSING - rebuilt from this working tree, "
            "so this campaign is its own experiment and its rows will not merge "
            "with another clone's. Commit the snapshot (python -m vram_planner "
            "--refresh-corpus).")
    # Resume must also match how the row was MEASURED, not just what was
    # configured. A row taken at n_predict 32 / repeat 1 - a smoke test - is not
    # the same measurement as one taken at 128 / 3, and silently accepting it as
    # already-done would put a noisier number into the comparison than every
    # other row and give no sign it had happened.
    done = {_key(r["model"], r["config"]) for r in load_rows(path)
            if r.get("config") and r.get("status") in ("ok", "oom")
            and r.get("n_predict") == n_predict and r.get("repeat") == repeat
            and r.get("prompt_id") == pid}
    plan = [c for c in cfgs if _key(os.path.basename(mp), c) not in done]
    skipped = len(cfgs) - len(plan)
    if limit:
        plan = plan[:limit]

    log("backend : %s" % b["build"])
    log("gpu     : %s" % (gpu or "unknown"))
    log("model   : %s  (%d blocks, %s)" % (os.path.basename(mp), facts["n_layers"],
                                           facts["arch"]))
    log("mmproj  : %s" % (os.path.basename(mmproj) if mmproj else "none"))
    log("prompt  : %s (frozen corpus; rows are keyed on it, so a refresh re-measures)" % pid[:12])
    log("output  : %s" % path)
    log("configs : %d to run, %d already recorded" % (len(plan), skipped))
    log("estimate: ~%.1f h  (load + %d warm + %d x %d tokens per config)"
        % (len(plan) * (60.0 + repeat * n_predict / 4.0) / 3600.0, 16, repeat, n_predict))
    if chain:
        log("search  : chained - each stage is built against the fastest row measured")
        log("          so far, not against the planner's guess%s"
            % ("" if rounds < 2 else
               ". %d rounds; a round that revisits a config already recorded skips it"
               % rounds))
    if dry_run:
        # ctx and kv are printed because they are the settings people FREEZE, and a
        # frozen setting that silently is not what you think is the whole failure
        # mode this listing exists to prevent.
        shown = plan
        if chain:
            # Only the first stage can be listed honestly. Its successors are
            # built from a baseline that does not exist yet, so printing values
            # for them would be a guess dressed as a plan - and the guess it
            # would print is exactly the frozen one this mode exists to escape.
            # The COUNTS are still exact: a ladder's length and the stage C/D
            # lists do not depend on the baseline, only their values do.
            first = (stages or "a")[0]
            shown = [c for c in plan if (c.get("stage") or "").lower() == first]
        for c in shown:
            log("  %-2s ctx %-7d kv %-5s ngl %-3d ncmoe %-3d ub %-5d fill %-7d "
                "spec %-13s nmax %-2d mmproj %s"
                % (c.get("stage", "-"), c["ctx"], c["kv"], c["ngl"],
                   c.get("ncmoe") or 0, c["ub"],
                   c.get("fill") or 0, c.get("spec") or "none",
                   c.get("spec_n_max") or 0,
                   "vram" if c.get("mmproj_offload") is not False else "ram"))
        sizes = {}
        for c in plan:
            k = (c.get("stage") or "-").lower()
            sizes[k] = sizes.get(k, 0) + 1
        if chain and len(shown) < len(plan):
            log("  .. and %d more in later stages, built from what the ones above find"
                % (len(plan) - len(shown)))
        if verify:
            log("verify  : after the campaign, the winner is loaded once more at the")
            extra = (" (" + " ".join("%s=%s" % (k, ",".join(map(str, v)))
                                     for k, v in sorted(verify_overrides.items()))
                     + ")") if verify_overrides else ""
            log("          production config%s - a split that fits the grid is not"
                " automatically one that fits what you actually run" % extra)
        # The config list rides along so the UI can show the same preview the CLI
        # prints, rather than parsing the lines above back out of a log.
        return {"planned": len(plan), "path": path, "dry_run": True,
                "configs": shown, "skipped": skipped, "model": os.path.basename(mp),
                "backend": b["build"], "gpu": gpu, "arch": facts["arch"],
                "n_layers": facts["n_layers"],
                "mmproj": os.path.basename(mmproj) if mmproj else None,
                "chained": bool(chain), "rounds": int(rounds or 1),
                "provisional": bool(chain) and len(shown) < len(plan),
                "stage_sizes": sizes,
                "verify": bool(verify),
                "estimate_h": round(len(plan) * (60.0 + repeat * n_predict / 4.0)
                                    / 3600.0, 2)}
    if not skip_preflight and not preflight(log=log):
        return None

    log("")
    log("%-2s %-4s %-5s %-5s %-8s %-13s %-4s %-6s | %8s %9s %7s %6s %6s"
        % ("st", "ngl", "ncmoe", "ub", "fill", "spec", "nmax", "mmproj",
           "tok/s", "prefill", "VRAM", "accept", "distin"))
    out, stopped = [], False
    total = [len(plan)]                 # a list so run_group can revise it
    if on_total:
        on_total(total[0])
    name = os.path.basename(mp)

    with open(path, "a", encoding="utf-8") as fh:

        def run_group(cfgs):
            """Measure a list of configs. Returns False if asked to stop."""
            for c in cfgs:
                if should_stop and should_stop():
                    log("stopped after %d of %d configs. The rows already written "
                        "are keyed, so re-running resumes here."
                        % (len(out), total[0]))
                    return False
                row = bench_one(b, mp, c, port=port, timeout=timeout,
                                n_predict=n_predict, repeat=repeat, log=log)
                row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                            "when": int(time.time()), "gpu": gpu})
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                out.append(row)
                # Keep `done` current so a later round skips what this one just
                # measured, the same way a restarted campaign would.
                if row.get("status") in ("ok", "oom"):
                    done.add(_key(name, c))
                log(_fmt_row(c, row, len(out), total[0]))
                if on_row:
                    on_row(row)
            return True

        if not chain:
            stopped = not run_group(plan)
        else:
            # Coordinate descent. `carried` is the config the fastest trustworthy
            # comparable row used; `best_tok` is what it did, so a challenger has
            # to beat it by CHAIN_MARGIN rather than by any amount at all.
            carried, best_tok = None, None
            budget = limit
            seen = set()
            for rd in range(1, max(1, int(rounds or 1)) + 1):
                for letter in "abcd":
                    if letter not in stages:
                        continue
                    if budget is not None and budget <= 0:
                        break
                    gb, gnl, gmoe, grungs = grid_context(
                        facts, mmproj=mmproj, base=base, model_path=mp,
                        carried=carried)
                    todo = [c for c in _dedupe(
                                stage_configs(letter, gb, gnl, gmoe, grungs,
                                              mmproj, facts), seen)
                            if _key(name, c) not in done]
                    if budget is not None:
                        todo = todo[:budget]
                        budget -= len(todo)
                    tag = "stage %s" % letter.upper() + (
                        " round %d" % rd if (rounds or 1) > 1 else "")
                    if not todo:
                        log("%s: nothing new to measure at this baseline" % tag)
                        continue
                    log("%s: %d config%s at %s"
                        % (tag, len(todo), "" if len(todo) == 1 else "s",
                           _carry_summary(gb)))
                    total[0] = len(out) + len(todo)
                    if on_total:
                        on_total(total[0])
                    if not run_group(todo):
                        stopped = True
                        break
                    # Re-read from DISK, not from `out`: a campaign resumed after a
                    # stop has rows this process never saw, and they are exactly
                    # the ones that say where the previous stage got to.
                    nxt, win = best_config(load_rows(path), name, base,
                                           n_predict=n_predict, repeat=repeat,
                                           incumbent_tok_s=best_tok, prompt_id=pid)
                    if nxt is None:
                        log("  baseline unchanged: nothing beat %s by more than %.0f%%"
                            % ("%.2f tok/s" % best_tok if best_tok
                               else "any trustworthy row", 100 * CHAIN_MARGIN))
                    else:
                        carried, best_tok = nxt, win["tok_s"]
                        log("  baseline -> %s   (%.2f tok/s, stage %s)"
                            % (_carry_summary(carried), best_tok,
                               (win.get("config") or {}).get("stage", "?")))
                if stopped or (budget is not None and budget <= 0):
                    break
    log("")
    log("%d rows -> %s" % (len(out), path))
    verified = None
    if verify and not dry_run and not stopped:
        # Cancellation lands between configs: the verify step is a config load,
        # and a campaign the user stopped mid-way must not start loading again.
        verified = _verify_step(b, mp, facts, name, base, path, pid, gpu,
                                overrides=verify_overrides, port=port,
                                timeout=timeout, n_predict=n_predict,
                                repeat=repeat, log=log)
    return {"rows": out, "path": path, "stopped": stopped, "planned": total[0],
            "chained": bool(chain), "verified": verified}


def _fmt_row(c, row, i=None, n=None):
    head = ("%-2s %-4d %-5d %-5d %-8d %-13s %-4d %-6s | "
            % (c.get("stage", "-"), c["ngl"], c.get("ncmoe") or 0, c["ub"],
               c.get("fill") or 0,
               c.get("spec") or "none", c.get("spec_n_max") or 0,
               "vram" if c.get("mmproj_offload") is not False else "ram"))
    if row.get("status") != "ok":
        return head + "%s %s" % (row["status"].upper(),
                                 row.get("gen_error") or row.get("error") or "")
    dr = row.get("distinct_ratio")
    return head + "%8.2f %9.1f %7.0f %6s %6s%s" % (
        row.get("tok_s") or 0.0, row.get("prefill_tok_s") or 0.0,
        row.get("proc_vram_mib") or 0.0,
        ("%.0f%%" % (100 * row["accept_rate"])) if row.get("accept_rate") else "-",
        ("%.2f" % dr) if dr is not None else "-",
        ("  SPILLED" if row.get("spilled") else "")
        + ("  [filler repeats - speculative rate inflated]"
           if row.get("corpus_repeated") and row["config"].get("spec") not in (None, "none")
           else "")
        + _degenerate_note(row))


def _degenerate_note(row):
    """The warning for a row that measured a model talking to itself.

    distinct_ratio and copyback_ratio are already recorded and already gate
    trustworthy(), so a degenerate row is silently dropped from every conclusion
    later. Printing the numbers at run time is what makes that visible while the
    campaign is still worth stopping - a whole grid can otherwise complete, look
    ordinary, and contribute nothing.

    The two symptoms point opposite ways on the distinct_ratio scale: a loop
    repeats a few windows, while a copy of the context produces nothing but
    distinct ones - which is exactly why the copy went unread until the second
    gate existed. The speculative case gets its own sentence in both because the
    symptom misreads the same way: an acceptance rate near 100% looks like the
    draft model excelling, when repeated output is precisely what makes any
    draft trivially correct. High acceptance ON degenerate text is evidence of
    the degeneration, not of speculation working."""
    dr = row.get("distinct_ratio")
    cb = row.get("copyback_ratio")
    spec = (row.get("config") or {}).get("spec") not in (None, "none")
    loop = dr is not None and dr < LOOP_RATIO
    copy = cb is not None and cb > COPY_RATIO
    # An untemplated row is the KNOWN-BAD mode rather than a symptom of it: the
    # backend had no /apply-template, so the model was handed raw text with
    # nothing marking it as a request. Said even when the output happens to look
    # fine, because the failure is in how the row was produced.
    raw = "  RAW - no chat template on this build, so the model was asked " \
          "nothing and merely continued the text" if row.get("templated") is False else ""
    if not (loop or copy):
        return raw
    if copy:
        what = "COPYING - %.0f%% of the output is a verbatim copy of the prompt, so" \
               % (100 * cb)
    else:
        what = "LOOPING - output is %.0f%% repetition, so" % (100 * (1 - dr))
    if spec and (row.get("accept_rate") or 0) >= 0.95:
        return ("  %s the %.0f%% acceptance is the %s, not the drafter; "
                "excluded from conclusions%s"
                % (what, 100 * row["accept_rate"], "copy" if copy else "loop", raw))
    return "  %s excluded from conclusions%s" % (what, raw)


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------
def load_speed_rows(path=None):
    paths = [path] if path else (
        [os.path.join(bench_dir(), f) for f in sorted(os.listdir(bench_dir()))
         if f.endswith(".jsonl")] if os.path.isdir(bench_dir()) else [])
    rows = []
    for p in paths:
        f = os.path.basename(p)
        for r in load_rows(p):
            # The backend BUILD lives only in the filename - nothing writes it
            # into the row - so without this nothing downstream can tell two
            # llama.cpp versions apart, and an insight would happily average
            # across them. Read-only: nothing writes these rows back.
            r["_file"] = f
            rows.append(r)
    return rows


def rank_rows(rows):
    """Measured rows, fastest first.

    Ordering only - no formatting - so the CLI report and the web UI rank the same
    way by construction rather than by two people remembering to."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("tok_s")]
    return sorted(ok, key=lambda r: -(r.get("tok_s") or 0))


# ---------------------------------------------------------------------------
# Turning rows into findings
#
# A ranked list says which config won. It does not say what the campaign LEARNED,
# and those are different: "1024 is the fastest ubatch" is worth much less than
# "ubatch is worth 6% and speculation is worth 41%", because only the second tells
# you where the next two hours should go.
#
# Everything below is pure - no I/O, no subprocess - so --speed-report and the
# browser reach the same conclusions by construction rather than by two people
# remembering to keep them in step.
# ---------------------------------------------------------------------------

# Every config field that identifies WHICH experiment a row belongs to.
_CONFIG_KEYS = ("ctx", "kv", "fa", "seq", "ub", "ngl", "ncmoe", "spec",
                "spec_n_max", "mmproj_offload", "fill",
                # Samplers belong here for the same reason _key() carries them:
                # greedy is speculation's BEST case, so two rows taken under
                # different sampler settings are not two configs, they are two
                # experiments. Leaving them out let axis_effects average a greedy
                # row against a sampled one and call the difference an effect of
                # whatever axis happened to differ - the exact confusion the
                # corpus_repeated warning exists to prevent, arriving by a
                # different door.
                "temp", "top_k", "top_p", "min_p", "rep_pen", "pres_pen")

# (name, label, the config keys this axis owns, natural reference value).
# The reference is the value the question is really asked against: what
# speculation bought over NOT speculating is the useful number even on the rare
# grid where "none" is not the slowest. Where no such value exists - a layer
# count has no natural zero - the slowest measured value is used instead and is
# reported as such.
EFFECT_AXES = (
    ("ngl", "GPU layers", ("ngl",), None),
    ("ncmoe", "CPU expert layers", ("ncmoe",), None),
    ("ub", "ubatch", ("ub",), 512),
    ("spec", "speculation", ("spec", "spec_n_max"), "none"),
    ("projector", "projector", ("mmproj_offload",), "VRAM"),
    ("kv", "KV quant", ("kv",), "f16"),
)


def _axis_value(r, axis):
    c = r.get("config") or {}
    if axis == "spec":
        s = c.get("spec") or "none"
        n = c.get("spec_n_max") or 0
        return "%s/%d" % (s, n) if s != "none" else "none"
    if axis == "projector":
        return "RAM" if c.get("mmproj_offload") is False else "VRAM"
    if axis in ("ncmoe", "fill"):
        return c.get(axis) or 0
    return c.get(axis)


def _control(r, owned=()):
    """Everything held constant for a comparison along the owned axis.

    This is the whole correctness of axis_effects(). A row from another GPU,
    another llama.cpp build, another context depth or a different n_predict is
    not a slower configuration - it is a different experiment - and putting the
    two in one comparison would manufacture an effect out of the difference
    between the machines. Rows only ever meet inside an identical key."""
    c = r.get("config") or {}
    ctl = [("model", r.get("model")), ("gpu", r.get("gpu")),
           ("file", r.get("_file")), ("n_predict", r.get("n_predict")),
           ("repeat", r.get("repeat")), ("prompt_id", r.get("prompt_id"))]
    for k in _CONFIG_KEYS:
        if k in owned:
            continue
        if k == "mmproj_offload":
            ctl.append((k, c.get(k) is not False))
        elif k == "spec":
            ctl.append((k, c.get(k) or "none"))
        elif k == "fa":
            ctl.append((k, bool(c.get(k))))
        elif k in _CONFIG_DEFAULTS:
            v = c.get(k)
            ctl.append((k, _CONFIG_DEFAULTS[k] if v is None else v))
        else:
            ctl.append((k, c.get(k)))
    return tuple(ctl)


def _slim(r):
    """A row cut down to what a finding needs to show or a script needs to run."""
    return {"model": r.get("model"), "gpu": r.get("gpu"), "file": r.get("_file"),
            "config": r.get("config"), "tok_s": r.get("tok_s"),
            "prefill_tok_s": r.get("prefill_tok_s"),
            "proc_vram_mib": r.get("proc_vram_mib"),
            "accept_rate": r.get("accept_rate"), "draft_n": r.get("draft_n"),
            "spilled": r.get("spilled"), "corpus_repeated": r.get("corpus_repeated"),
            "distinct_ratio": r.get("distinct_ratio"),
            "copyback_ratio": r.get("copyback_ratio"),
            "prompt_id": r.get("prompt_id"), "when": r.get("when"),
            "n_predict": r.get("n_predict"), "repeat": r.get("repeat"),
            "status": r.get("status")}


def sweep_index(rows):
    """One entry per (model, GPU, backend build) - the browsable list of campaigns.

    Split by build as well as by card, because a llama.cpp version bump moves
    these numbers and merging two builds into one campaign would hide that."""
    groups = {}
    for r in rows:
        k = (r.get("model") or "?", r.get("gpu") or "", r.get("_file") or "")
        groups.setdefault(k, []).append(r)
    out = []
    for (model, gpu, f), rs in groups.items():
        ok = [r for r in rs if trustworthy(r)]
        when = [r.get("when") for r in rs if r.get("when")]
        fills = sorted({(r.get("config") or {}).get("fill") or 0 for r in rs})
        stages = sorted({(r.get("config") or {}).get("stage") for r in rs
                         if (r.get("config") or {}).get("stage")})
        best = max(ok, key=lambda r: r["tok_s"]) if ok else None
        out.append({
            "model": model, "gpu": gpu, "file": f,
            "backend": re.sub(r"\.jsonl$", "", f).split("__")[-1],
            "n_rows": len(rs), "n_ok": len(ok),
            "n_failed": sum(1 for r in rs if r.get("status") != "ok"),
            "n_untrusted": sum(1 for r in rs
                               if r.get("status") == "ok" and not trustworthy(r)),
            # Rows recorded before the copy-back gate existed: they carry no
            # copyback_ratio, so a verbatim-copying model could not be detected
            # in them. The campaign is real; it is just not fully gated.
            "n_ungated": sum(1 for r in rs if r.get("status") == "ok"
                             and r.get("copyback_ratio") is None),
            "prompt_id": next((r.get("prompt_id") for r in rs
                               if r.get("prompt_id")), None),
            "first": min(when) if when else None,
            "last": max(when) if when else None,
            "fills": fills, "stages": stages,
            "best_tok_s": best["tok_s"] if best else None,
            "best": _slim(best) if best else None,
        })
    # Most recent first: the campaign you ran this morning is the one you want.
    return sorted(out, key=lambda g: -(g["last"] or 0))


def axis_effects(rows):
    """What each knob was worth, from controlled comparisons only.

    For every axis, rows are bucketed by everything else; the bucket with the
    most distinct values of that axis is the comparison the campaign actually
    ran, and it is the one reported. Other buckets are counted, not merged -
    averaging across them is exactly the mistake _control() exists to prevent.

    An axis measured at only one value is reported as such rather than dropped,
    because "we never varied this" is itself the finding most likely to be
    actionable."""
    usable = [r for r in rows if trustworthy(r)]
    dropped = sum(1 for r in rows if r.get("status") == "ok" and not trustworthy(r))
    out = []
    for axis, label, owned, ref in EFFECT_AXES:
        groups = {}
        for r in usable:
            if _axis_value(r, axis) is None:
                continue
            groups.setdefault(_control(r, owned), []).append(r)
        if not groups:
            continue
        ranked = sorted(groups.values(),
                        key=lambda g: (len({_axis_value(r, axis) for r in g}), len(g)),
                        reverse=True)
        grp = ranked[0]
        # Best row per distinct value, so a value measured twice is represented
        # by its better run rather than by whichever came last.
        by_val = {}
        for r in grp:
            v = _axis_value(r, axis)
            if v not in by_val or r["tok_s"] > by_val[v]["tok_s"]:
                by_val[v] = r
        vals = sorted(by_val.items(), key=lambda kv: -kv[1]["tok_s"])
        if len(vals) < 2:
            out.append({"axis": axis, "label": label, "n_values": 1,
                        "values": [{"value": v, "tok_s": r["tok_s"]} for v, r in vals],
                        "single": True})
            continue
        best_v, best_r = vals[0]
        worst_v, worst_r = vals[-1]
        base_v, base_r = (ref, by_val[ref]) if ref in by_val else (worst_v, worst_r)
        # A repeated filler corpus is not a reason to throw the comparison away -
        # the rows are real - but it is a reason not to quote the number flat.
        # Speculation drafts from what it has already seen, so text that loops
        # back on itself is its best case and nothing like a real conversation.
        # This is the axis that caveat exists for, so it is carried on the axis.
        repeated = sum(1 for r in grp if r.get("corpus_repeated"))
        out.append({
            "axis": axis, "label": label, "single": False,
            "n_values": len(vals), "n_rows": len(grp),
            "n_other_groups": len(ranked) - 1,
            "n_corpus_repeated": repeated,
            "inflated": bool(repeated) and axis == "spec",
            "values": [{"value": v, "tok_s": r["tok_s"],
                        "vram_mib": r.get("proc_vram_mib"),
                        "config": r.get("config")} for v, r in vals],
            "best": best_v, "best_tok_s": best_r["tok_s"],
            "reference": base_v,
            "reference_tok_s": base_r["tok_s"],
            "reference_is_natural": ref in by_val,
            "gain_pct": (100.0 * (best_r["tok_s"] - base_r["tok_s"]) / base_r["tok_s"]
                         if base_r["tok_s"] else None),
            "best_config": best_r.get("config"),
        })
    # Biggest lever first - that is the order the next campaign should read it in.
    out.sort(key=lambda e: -(e.get("gain_pct") or -1))
    return {"effects": out, "n_rows": len(usable), "n_excluded": dropped}


def depth_curve(rows):
    """The same config measured at more than one context depth.

    Decode re-reads the KV cache every token, so speed falls as the context
    fills. A headline taken at 2k says very little about the 40k conversation it
    will actually be used in, and this is the only thing in the tool that can
    show the slope rather than assert it."""
    usable = [r for r in rows if trustworthy(r)]
    groups = {}
    for r in usable:
        groups.setdefault(_control(r, ("fill",)), []).append(r)
    out = []
    for g in groups.values():
        by_fill = {}
        for r in g:
            f = (r.get("config") or {}).get("fill") or 0
            if f not in by_fill or r["tok_s"] > by_fill[f]["tok_s"]:
                by_fill[f] = r
        if len(by_fill) < 2:
            continue
        pts = [{"fill": f, "tok_s": r["tok_s"], "prefill_tok_s": r.get("prefill_tok_s")}
               for f, r in sorted(by_fill.items())]
        out.append({"points": pts, "config": g[0].get("config"),
                    "model": g[0].get("model"), "gpu": g[0].get("gpu"),
                    "drop_pct": (100.0 * (pts[0]["tok_s"] - pts[-1]["tok_s"])
                                 / pts[0]["tok_s"]) if pts[0]["tok_s"] else None})
    return sorted(out, key=lambda d: -(d["points"][-1]["fill"]))


def pareto(rows):
    """Rows nothing else beats on BOTH speed and VRAM.

    "Fastest" and "fastest that still leaves the desktop a card to draw on" are
    different questions, and a ranking by tok/s alone can only answer the first.
    A row 2% slower for 3 GiB less is often the one worth running."""
    cand = [r for r in rows if trustworthy(r) and r.get("proc_vram_mib")]
    out = []
    for r in cand:
        beaten = False
        for o in cand:
            if o is r:
                continue
            if (o["tok_s"] >= r["tok_s"] and o["proc_vram_mib"] <= r["proc_vram_mib"]
                    and (o["tok_s"] > r["tok_s"]
                         or o["proc_vram_mib"] < r["proc_vram_mib"])):
                beaten = True
                break
        if not beaten:
            out.append(r)
    return sorted(out, key=lambda r: -r["tok_s"])


def insights(rows):
    """Everything derivable from a set of rows, in one payload."""
    return {"index": sweep_index(rows),
            "axes": axis_effects(rows),
            "depth": depth_curve(rows),
            "pareto": [_slim(r) for r in pareto(rows)],
            "ranked": [_slim(r) for r in rank_rows(rows)]}


def report_insights(path=None, log=print):
    """The findings behind the ranking: what each knob was worth, and at what depth."""
    rows = load_speed_rows(path)
    if not rows:
        log("No speed rows recorded yet. Run --speed-sweep.")
        return False
    log("CAMPAIGNS")
    for g in sweep_index(rows):
        span = ""
        if g["first"] and g["last"]:
            f = datetime.datetime.fromtimestamp(g["first"]).strftime("%Y-%m-%d")
            t = datetime.datetime.fromtimestamp(g["last"]).strftime("%Y-%m-%d")
            span = f if f == t else "%s..%s" % (f, t)
        log("  %-34s %-24s %-8s %3d rows (%d ok)  best %s  %s"
            % (g["model"][:34], (g["gpu"] or "?")[:24], g["backend"][-8:],
               g["n_rows"], g["n_ok"],
               ("%.2f tok/s" % g["best_tok_s"]) if g["best_tok_s"] else "-", span))
    ung = [g for g in sweep_index(rows) if g["n_ungated"]]
    if ung:
        log("  %d campaign(s) were recorded before the copy-back gate: their rows carry")
        log("  no copyback_ratio, so a model copying its context back could not be")
        log("  detected in them. Deep-fill numbers there are not usable for tuning.")

    ax = axis_effects(rows)
    log("")
    log("WHAT EACH KNOB WAS WORTH   (controlled comparisons only: rows differing in")
    log("model, GPU, backend, context, KV quant, depth or pass count never meet)")
    if ax["n_excluded"]:
        log("  %d row%s excluded from every conclusion below - spilled into shared "
            "memory, looping, or copying the prompt back"
            % (ax["n_excluded"], "" if ax["n_excluded"] == 1 else "s"))
    for e in ax["effects"]:
        if e.get("single"):
            log("  %-20s only one value ever tried (%s) - nothing to compare"
                % (e["label"], e["values"][0]["value"]))
            continue
        log("  %-20s %+7.1f%%  best %-14s vs %-10s (%d values, %d rows)%s"
            % (e["label"], e["gain_pct"] or 0.0, str(e["best"]), str(e["reference"]),
               e["n_values"], e["n_rows"],
               "" if e["reference_is_natural"] else "  [ref = slowest measured]"))
        for v in e["values"]:
            log("      %-16s %7.2f tok/s" % (str(v["value"]), v["tok_s"]))
        if e.get("inflated"):
            log("      !! %d of these rows needed the filler corpus to REPEAT to reach"
                % e["n_corpus_repeated"])
            log("         that depth. Speculation drafts from what it has already seen,")
            log("         so repeated text is its best case and this gain is an upper")
            log("         bound, not what a real conversation will give you.")

    dc = depth_curve(rows)
    if dc:
        log("")
        log("SPEED VS CONTEXT DEPTH   (the same config, measured at more than one fill)")
        for d in dc:
            c = d["config"] or {}
            log("  %-30s ngl %-3s ub %-5s spec %-12s  -%.0f%%"
                % (d["model"][:30], c.get("ngl"), c.get("ub"),
                   c.get("spec") or "none", d["drop_pct"] or 0))
            for p in d["points"]:
                log("      %9s tokens filled  %7.2f tok/s" % ("{:,}".format(p["fill"]),
                                                              p["tok_s"]))

    pf = pareto(rows)
    if pf:
        log("")
        log("SPEED VS VRAM   (nothing else is both faster AND smaller than these)")
        log("  %8s %9s  %-30s" % ("tok/s", "VRAM", "config"))
        for r in pf:
            c = r["config"]
            log("  %8.2f %7.0f M  %-34s %s"
                % (r["tok_s"], r["proc_vram_mib"] or 0,
                   "ngl %s ncmoe %s ub %s" % (c.get("ngl"), c.get("ncmoe") or 0,
                                              c.get("ub")),
                   r["model"][:24]))
    return True


def report(path=None, log=print):
    """Every measured config, fastest first."""
    rows = rank_rows(load_speed_rows(path))
    if not rows:
        log("No speed rows recorded yet. Run --speed-sweep.")
        return False
    log("%-30s %-4s %-5s %-5s %-8s %-13s %-4s %-6s %8s %9s %7s %7s"
        % ("model", "ngl", "ncmoe", "ub", "fill", "spec", "nmax", "mmproj", "tok/s",
           "prefill", "VRAM", "accept"))
    for r in rows:
        c = r["config"]
        log("%-30s %-4d %-5d %-5d %-8d %-13s %-4d %-6s %8.2f %9.1f %7.0f %7s%s"
            % (r["model"][:30], c["ngl"], c.get("ncmoe") or 0, c["ub"],
               c.get("fill") or 0,
               c.get("spec") or "none", c.get("spec_n_max") or 0,
               "vram" if c.get("mmproj_offload") is not False else "ram",
               r["tok_s"], r.get("prefill_tok_s") or 0.0,
               r.get("proc_vram_mib") or 0.0,
               ("%.0f%%" % (100 * r["accept_rate"])) if r.get("accept_rate") else "-",
               # Ranked fastest-first, and a looping row can WIN that ranking:
               # repetition is cheap to generate, and so is copying the prompt
               # back. Marked here for the same reason SPILLED is - the row is
               # real evidence, it just is not evidence about the setting in its
               # own columns.
               ("  SPILLED" if r.get("spilled") else "") + _degenerate_note(r)))
    return True
