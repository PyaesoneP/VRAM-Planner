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
import datetime, hashlib, json, math, os, re, socket, statistics, time, urllib.error, urllib.request
from .gpu import get_gpu_processes, gpu_list, gpu_shared_mib
from .lmstudio import default_models_dir
from .paths import _data_dir
from .sweep import (build_argv, classify_drafter, discover_models,
                    find_drafter_for, finish_row, gpu_side, model_facts,
                    pick_backend, backends_dir, serve, sweep_path,
                    _drafter_block_size, _key, load_rows, unsound_reason)


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


def template_identity(path=None, kwargs_json=None, reasoning=None,
                      reasoning_preserve=None):
    """The hash of the chat template a row was measured under, or None.

    Sibling of prompt_identity(), and for the same reason: what the model was
    asked is half of what a tok/s number means. A thinking template that emits a
    reasoning block generates a different number of tokens per answer than one
    that does not, so two rows measured under different templates are different
    experiments even when every flag matches.

    The FILE'S BYTES are hashed, not its path. Editing a template in place must
    invalidate the rows measured against the old one - a path would not notice,
    and the campaign would silently mix both halves. None means no template was
    pinned and the GGUF's own metadata one was used, which is its own answer and
    groups separately from every pinned template.

    The thinking flags belong to the same identity: --reasoning off makes one
    template produce a different answer of a different length, which is a
    different measurement. They are appended only when SET, so a campaign that
    does not use them hashes to the same bytes it did before they existed and
    its rows keep resuming."""
    if not (path or kwargs_json or reasoning or reasoning_preserve):
        return None
    h = hashlib.sha256()
    if path:
        with open(path, "rb") as fh:
            h.update(fh.read())
    h.update(b"\x00")
    h.update((kwargs_json or "").encode("utf-8"))
    if reasoning or reasoning_preserve:
        h.update(("\x00%s\x00%s" % (reasoning or "",
                                    reasoning_preserve or "")).encode("utf-8"))
    return h.hexdigest()[:16]


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
# Campaign-level, not points in the grid. They ride in the config dict because
# build_argv reads from it, and are stripped back out of the stored row: an
# absolute path is not a knob anyone sweeps over, and every comparison in this
# module groups on config equality. template_id carries them instead.
_TEMPLATE_KEYS = ("chat_template_file", "chat_template_kwargs",
                  "reasoning", "reasoning_preserve")

# The sampler names a sweep CONFIG uses - what sampling_of() reads and _key()
# hashes. The launch-script form spells the last two out in full; web.py maps.
_SWEEP_SAMPLER_KEYS = ("temp", "top_k", "top_p", "min_p", "rep_pen", "pres_pen")


def _skip_stamp(row, skip_count, base):
    """Was this run abandoned because the user asked to skip it?

    `skip_count` is a callable returning how many skips the controlling layer
    (terminal keypress, web button) has requested so far; `base` is what it
    returned when the run started. A press lands on EXACTLY the run it was
    made during: the count only grows, so a press that misses a completed row
    is consumed by the comparison and never bleeds into the next config.

    `ok` and `oom` rows are complete measurements and stay what they are - an
    OOM row in particular is the wall the ladder is read from, and overwriting
    it with a skip would be throwing the one result the run produced. Every
    other status means the run was interrupted mid-way, which is what a skip
    does, so the interruption is named honestly: the row is recorded, keyed,
    and never re-measured - by this campaign, or by a resumed one."""
    if skip_count is None or skip_count() <= base:
        return False
    return row.get("status") not in ("ok", "oom")


def bench_one(backend, model_path, c, port=BENCH_PORT, timeout=420.0,
              n_predict=N_PREDICT, repeat=N_REPEAT, gen_timeout=1800, log=print,
              on_server=None, skip_count=None, abort_floor=None,
              should_abort=None):
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
    # The template is a property of the CAMPAIGN, not of one point in the grid,
    # so it is stamped on the row beside prompt_id rather than left in the config
    # dict: an absolute path is not a setting anyone is sweeping over, and every
    # comparison in this file groups on config equality.
    stored = {k: v for k, v in c.items() if k not in _TEMPLATE_KEYS}
    base_skips = skip_count() if skip_count else 0
    row = {"status": "error", "config": stored,
           "model": os.path.basename(model_path), "backend": backend["build"],
           "speed": True, "n_predict": n_predict, "repeat": repeat}
    tf, tk = c.get("chat_template_file"), c.get("chat_template_kwargs")
    rea, rea_p = c.get("reasoning"), c.get("reasoning_preserve")
    if tf or tk or rea or rea_p:
        row["template_id"] = template_identity(tf, tk, rea, rea_p)
        row["chat_template"] = os.path.basename(tf) if tf else None
        row["template_kwargs"] = tk
        row["reasoning"] = rea
        row["reasoning_preserve"] = rea_p
    # Which frozen corpus this row measured against. Everything a deep-fill
    # number means lives or dies on this: two rows from different corpora are
    # different experiments, and resume and every comparison below treat them
    # that way. Stamped BEFORE the server starts, so a row that never loads -
    # OOM, EXIT, genfail - still belongs to the campaign: resume must be able
    # to skip a wall it already paid to discover.
    row["prompt_id"] = prompt_identity()
    cfg = dict(c)
    cfg["warmup"] = True          # a speed run should not pay for lazy init
    # Resolved per config, not once per campaign: the collision only appears
    # BETWEEN rows, when the previous server's socket is still winding down.
    port = free_port(port)
    # `on_server` goes in as serve()'s on_proc so the process is published the
    # instant Popen returns. A hard stop works by killing the server the worker
    # is blocked on, and it can only kill what it has been told about - so
    # publishing after the context manager yields (which is what this used to
    # do) left the process invisible for the whole LOAD, 20-120s normally and up
    # to `timeout` when something is wrong. `should_abort` ends the wait for the
    # ready line for the same reason.
    with serve(backend, model_path, cfg, port=port, timeout=timeout,
               log_dir=log_dir, log_name="bench.log",
               on_proc=on_server, should_abort=should_abort) as srv:
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
                samp = sampling_of(c)
                row["sampling"] = samp
                # One throwaway pass on a TINY prompt, before anything is timed.
                #
                # llama-server's own --warmup does not cover what this does. The
                # measured shape says so: at ncmoe 32-35 the ready line came
                # 21-49s after launch and the first real request then spent
                # 106-158s on a 2,065-token prompt - while the same model at
                # 119,099 tokens managed 573s. A rate cannot do that. Solving
                # the two gives ~253 tok/s plus ~100s of FIXED cost, and the
                # fixed part is anti-correlated with load time: when loading
                # took 21s the first request took 158s, when it took 49s the
                # request took 116s, and the sum barely moved. That is paging.
                # --n-cpu-moe puts ~20 GB of experts in system RAM, the plan
                # already needs more RAM than is free, and whatever the load did
                # not fault in the first forward pass does.
                #
                # bench_one already knew the first pass is contaminated - it
                # throws the cold pass's DECODE away for exactly this reason,
                # and the numbers agree, cold decode running 0.56-0.82x the warm
                # figure. It just kept the same pass's PREFILL. So prefill_tok_s
                # at shallow fill was measuring page faults.
                #
                # This costs nothing: the fault-in is paid once either way. It
                # moves out of the number instead of being added to the run.
                generate(srv.url, INSTRUCTION.strip(), 1, gen_timeout,
                         cache_prompt=False, sampling=samp, seed=999)
                # Rows recorded before this are not comparable on prefill and
                # must not be silently averaged with these.
                row["prefill_warm"] = True
                cold = generate(srv.url, prompt, min(16, n_predict), gen_timeout,
                                cache_prompt=False, sampling=samp, seed=1000)
                row["cold"] = cold
                # The abort floor: a pass at a fraction of what this model
                # normally delivers is not a slower config, it is a card that
                # stopped being usable - WDDM spilled the process into shared
                # memory, or the machine is busy. Measuring it to the end is the
                # waste the floor exists to stop: the passes crawl, and the
                # number, when it lands, is garbage that has to be explained
                # away later. The row is then recorded with status `spilled` -
                # honest, keyed, and never re-measured - instead of `ok` with a
                # plausible-looking slow number. See _abort_floor().
                aborted = None
                if abort_floor:
                    t = cold["timings"].get("predicted_per_second")
                    if t and t < abort_floor:
                        aborted = t
                runs = []
                if aborted is None:
                    for i in range(max(1, repeat)):
                        r = generate(srv.url, prompt, n_predict, gen_timeout,
                                     cache_prompt=True, sampling=samp,
                                     seed=1000 + i)
                        runs.append(r)
                        if abort_floor:
                            t = r["timings"].get("predicted_per_second")
                            if t and t < abort_floor:
                                aborted = t
                                break
                row["runs"] = runs
                if aborted is not None:
                    row["status"] = "spilled"
                    row["abort_floor"] = abort_floor
                    row["gen_error"] = ("aborted mid-measurement: decode at %.2f "
                                        "tok/s, below the %.2f tok/s this model "
                                        "normally does - WDDM spilled it into "
                                        "shared memory, or the card is busy"
                                        % (aborted, abort_floor))
                else:
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
            # Read INSIDE the serve() block, while the process still exists:
            # a demotion is a property of the running process and the counter
            # instance disappears with it. gpu_free_after_mib is read after
            # teardown and is identical on every row for exactly that reason.
            try:
                row["shared_mib"] = (None if (should_abort and should_abort())
                                     else gpu_shared_mib(srv.proc.pid))
            except Exception:
                row["shared_mib"] = None
    if on_server:
        on_server(None)             # torn down; nothing left to kill
    # An abandoned run's row is DISCARDED by run_group rather than written, so
    # every reading taken for it is time the person who pressed Stop spends
    # waiting for a number nobody will see. gpu_shared_mib is a PowerShell
    # Get-Counter and finish_row is two nvidia-smi calls; together they are
    # tens of seconds of "why has it not stopped yet".
    aborted = bool(should_abort and should_abort())
    finish_row(row, srv, log_dir=log_dir, aborted=aborted)
    # finish_row takes its verdict from the server, which came up fine; a
    # generation that then failed is still a failed measurement.
    if row.get("gen_error") and row["status"] == "ok":
        row["status"] = "genfail"
    elif row["status"] == "ok" and not row.get("tok_s"):
        row["status"] = "genfail"
        row["gen_error"] = "server reported no predicted_per_second"
    # The controlling layer (terminal S key, web Skip button) kills the server
    # to end the run promptly - a hung config would otherwise sit out its whole
    # generation timeout - and the row is stamped with the reason instead of a
    # lie. A stamped row is RECORDED, so it keys and a resumed campaign skips
    # it, which is the entire point: a skip is a judgement, and re-measuring a
    # config the user already judged wastes the same load it just wasted.
    if not aborted and _skip_stamp(row, skip_count, base_skips):
        row["status"] = "skipped"
        row["gen_error"] = "skipped by user"
    # The Windows failure mode is not an OOM. WDDM spills past dedicated VRAM
    # into system RAM and the load SUCCEEDS, so the row says "ok" and only the
    # numbers give it away: a negative floor, and decode falling off a cliff.
    #
    # Only suspect_reason() can answer that from ONE row. The counter reading
    # cannot - it needs the rest of the ladder to be excess over (see demoted())
    # - so `demoted` is False here by construction and the verdict is re-decided
    # in _infer_demotion() on the way back out of the store, where the ladder
    # exists. Written anyway, so a row is never missing the field.
    # unsound_reason(), NOT the stored `suspect`: `suspect` is the ALLOCATION
    # FIT's verdict and excludes every -ot row on principle, which is now every
    # row of every dense campaign. A speed row is judged on whether its numbers
    # can be believed, which is a different question.
    row["spilled"] = (row.get("status") == "ok"
                      and (bool(unsound_reason(row)) or demoted(row)))
    return row


# Excess shared memory over a ladder's own baseline, above which the driver has
# moved something. NOT an absolute reading - see demoted().
SHARED_SPILL_MIB = 64.0

# The fraction of a campaign's own median tok/s below which a pass is not a
# slower config but a card that stopped being usable. The wall being found
# moves a ladder by tens of percent; a WDDM demotion or a busy machine moves
# it by 6x or more, and no real knob in this grid does that. Shared by the
# mid-run abort (_abort_floor) and the recorded-rows verdict (_infer_demotion),
# so a row judged too slow to finish measuring is judged the same way after
# the fact.
SLOW_FRAC = 0.15


def _demotion_group(row, cfg=None, model=None):
    """What a row's tok/s and floor may be compared against.

    Everything that legitimately moves either number, held constant, so what is
    left over is the thing being detected. The placement axes - ngl, ncmoe,
    n_cpu_ffn - are deliberately ABSENT: they are what the ladder varies, and a
    detector needs a ladder to be an anomaly within.

    `fill` is the one that matters most for tok/s and was missing entirely from
    _abort_floor's reckoning. Decode falls ~6x from 2k to 120k on the same
    config (9.45 -> 1.49 on a real campaign), so a median taken across fills sits
    far above a legitimate deep row - which then trips a floor set at 15% of it,
    aborts mid-measurement, and is recorded `spilled`. `spec` and `spec_n_max`
    are the same story pointing the other way: speculative rows run far faster
    and drag the median up.

    Taken from `cfg` when the row does not exist yet - which is how the mid-run
    abort asks the question, before its own row has been written."""
    c = row.get("config") if row is not None else cfg
    c = c or {}
    r = row or {}
    return (model if row is None else r.get("model"),
            r.get("gpu"), r.get("_file"), r.get("prompt_id"), r.get("template_id"),
            c.get("ctx"), c.get("kv"), c.get("ub"),
            c.get("mmproj_offload") is not False,
            c.get("spec") or "none", c.get("spec_n_max") or 0,
            c.get("fill"))


def _abort_floor(rows, cfg, model):
    """The slowest decode this config may plausibly run at, or None.

    A row at a tiny fraction of what the same model normally delivers on this
    card is not a slower config, it is a measurement of a card that stopped
    being usable - WDDM spilled the process into shared memory, or the machine
    is busy with something else. Measuring it is the waste the campaign exists
    to avoid: the passes run at a crawl, and the number, when it lands, is
    garbage that has to be explained away later.

    The reference is the median of the healthy rows measured under the SAME
    CONDITIONS - _demotion_group, the same twelve keys _infer_demotion() groups
    on. It used to be every healthy row for the model, filtered on the name
    alone, which is how a legitimate deep-fill row ended up judged against a
    median drawn from 2k rows six times faster than anything it could reach.
    Two detectors that claim to make the same judgement have to make it from
    the same rows, or a row can be too slow to finish measuring and perfectly
    normal afterwards.

    No rows, no floor: a config with nothing to be judged against is measured
    to the end."""
    want = _demotion_group(None, cfg=cfg, model=model)
    cand = []
    for r in rows:
        if r.get("status") != "ok" or not r.get("tok_s"):
            continue
        if r.get("model") != model:
            continue
        # gpu / _file / prompt_id / template_id are stamped on stored rows and
        # absent from the config in hand, so the comparison is on the config
        # half of the key. The model name and this campaign's own store are
        # what the caller has already narrowed to.
        if _demotion_group(r)[5:] != want[5:]:
            continue
        if r.get("spilled") or not trustworthy(r):
            continue
        cand.append(r["tok_s"])
    if len(cand) < 2:
        return None
    cand.sort()
    mid = (cand[len(cand) // 2] if len(cand) % 2
           else (cand[len(cand) // 2 - 1] + cand[len(cand) // 2]) / 2.0)
    return round(mid * SLOW_FRAC, 3)


def demoted(row):
    """Did WDDM move part of this process into system RAM?

    Only ever answers from `shared_excess`, which is measured against the row's
    own ladder. The raw counter cannot answer it, and the first version of this
    function - `shared_mib > 64` - was wrong for a reason worth writing down,
    because it is the whole difference between the two.

    `\\GPU Process Memory\\Shared Usage` counts every byte of host memory mapped
    for GPU access. That includes memory that is in system RAM BY CHOICE:
    llama.cpp's CUDA_Host pinned staging buffers, and `--no-mmproj-offload`,
    which is a config knob this grid deliberately sweeps. So the counter has a
    large baseline on a process that fits perfectly:

        ngl 26/27/28 + draft-mtp 2, projector in RAM   shared 474.0 MiB, all three
        ngl 31       + no spec,     projector in RAM   shared 238.0 MiB

    Two things there. The reading does not move with ngl - 474.0 exactly, while
    proc_vram climbs 10660 -> 11164 -> 11410 - and demotion under pressure is
    precisely the thing that would. And the 474 MiB row at ngl 28 is the FASTEST
    row ever measured on this model at this depth, 3.95 against 3.25 for the 238
    MiB one. A threshold of 64 called all four spilled, which put every row of
    the campaign outside trustworthy() and left best_config() with nothing to
    pick - a detector that fires on everything reports nothing.

    What separates a demotion from a deliberate placement is that a demotion
    RESPONDS TO PRESSURE. So the signal is the excess over what the rest of the
    ladder carries, exactly like the floor collapse in _infer_demotion(), and it
    needs the same grouping to mean anything.

    None means unmeasured, which is not evidence of absence: an unmeasured row
    is not called clean, it is simply not called demoted either."""
    x = row.get("shared_excess")
    return x is not None and x > SHARED_SPILL_MIB


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
# Fixed for this campaign and not swept: context and KV quant are the two the
# user pinned, flash attention is mandatory with a quantised cache, and -np 1
# because every extra sequence buys recurrent state nobody asked for.
SPEED_BASE = {"ctx": 131072, "kv": "q8_0", "fa": True, "seq": 1, "ub": 512,
              "ngl": 26, "ncmoe": 0, "n_cpu_ffn": 0, "fill": 2048,
              "spec": "none", "mmproj_offload": True,
              # The draft cache's quant, frozen like kv is: llama.cpp keeps it
              # at f16 unless -ctkd/-ctvd say otherwise. Stage D measures
              # whatever this says, and the planner prices it the same way.
              "spec_kv": "f16"}

# Stage A finds the wall, and stage B starts from what A found. That ordering is
# the whole design: a split that spills makes every later comparison a
# comparison of spill behaviour.
#
# The wall is found by BISECTION over the whole range, not by walking a ladder
# bracketed around the planner's guess. Two things change with that, and the
# second is the one that matters:
#
#   * cost. A ladder pays one load - about two minutes - per rung. A bisection
#     pays ceil(log2(n)), so 66 layer counts cost seven loads instead of 66, and
#     the old nine-rung bracket costs seven while covering seven times as much.
#   * correctness. Because a bisection can AFFORD the whole range, it no longer
#     matters whether the planner centred the bracket well. A wall more than
#     NGL_ABOVE rungs above the seed used to sit outside the ladder entirely,
#     and the campaign would then report the top of its own bracket as the wall,
#     with nothing in the output to say it had never looked further up.
#
# planner_split() is still asked where the split falls, but only for the PINS -
# which axis is free and what the others are held at. Where the wall is, is now
# measured rather than guessed at.

# The wall search's step on the context axis. ctx is the one axis with no
# natural unit: a layer count is a layer count, but a window runs to six figures
# and nobody wants 65,536 candidate values. So the domain is multiples of
# n_ctx_train / CTX_SEARCH_DIVISIONS, never finer than CTX_STEP.
#
# 64 divisions puts every model at 64 values, which is seven probes. On a
# 262144-trained model that is 4096-token rungs; on a 32768-trained one the
# floor takes over and it is 1024-token rungs, 32 values, six probes. Neither
# granularity is one anyone tunes at - the question stage A answers is "how much
# window fits", and 4096 tokens is 1.5% of the answer on the model where the
# step is coarsest.
CTX_STEP = 1024

CTX_SEARCH_DIVISIONS = 64


def ctx_search_step(n_ctx_train):
    """The step the context domain is built on, snapped to CTX_STEP."""
    if not n_ctx_train:
        return CTX_STEP
    return max(CTX_STEP,
               int(n_ctx_train // CTX_SEARCH_DIVISIONS // CTX_STEP) * CTX_STEP)


def ctx_values(n_ctx_train):
    """Every context the wall search may land on, ascending.

    Ascending is also ascending in VRAM cost here, which is what lets
    _WallSearch treat a window exactly like a layer count.

    The whole trained range, not a bracket around the planner's max_ctx. The
    trained maximum itself is always in the domain even when the step does not
    divide it - it is the one value worth naming, because "the whole window
    fits" is the answer that ends the search and starts phase 2."""
    if not n_ctx_train:
        return []
    step = ctx_search_step(n_ctx_train)
    n = max(1, int(n_ctx_train // step))
    out = [step * i for i in range(1, n + 1)]
    if out[-1] != int(n_ctx_train):
        out.append(int(n_ctx_train))
    return out


def wall_probes(values):
    """Loads a bisection over `values` costs, worst case.

    Named separately from _WallSearch.budget because the ESTIMATE needs it
    before any search exists, and an estimate computed a second way is an
    estimate that drifts from what runs."""
    n = len(values or [])
    return 0 if n <= 0 else int(math.ceil(math.log(n + 1, 2)))


def wall_values(axis, n_layers=0, n_ctx_train=0):
    """The wall search's domain for `axis`, ordered ASCENDING IN VRAM COST.

    That ordering is the whole trick: it turns four different questions - the
    largest window, the most layers, the fewest experts exiled, the least FFN
    exiled - into one question, "the highest index that loads", so one search
    answers all of them and cannot disagree with itself about which end of an
    axis is the good end. axis_direction() decides which lists get reversed,
    over the same tables the OOM pruning walks."""
    if axis == "ctx":
        vals = ctx_values(n_ctx_train)
    else:
        vals = list(range(0, int(n_layers or 0) + 1))
    if axis_direction(axis) == "down":
        vals.reverse()
    return vals


# The basis every campaign is planned against, named once so the browser can be
# handed the same three numbers instead of picking its own. analyze() takes the
# budget and the reserve separately and subtracts one from the other, so the
# helper is asked for a raw total here and the reserve is passed on below.
PLAN_BASIS = "total"
PLAN_RESERVE_MIB = 512
PLAN_SAFETY_PCT = 5

# Plan rows and seeds recorded before the ceiling/fit retag carry the old mode
# names ("speed" / "context"). They map onto the new categories one-to-one; the
# mapping is read-side only - nothing new is ever written in the old spelling.
# bench sits below recommend in the DAG, so the shared helper lives here and
# recommend imports it.
_LEGACY_PLAN_MODE = {"speed": "ceiling", "context": "fit"}


def norm_plan_mode(m):
    """The category a recorded mode belongs to, new spelling or old."""
    return _LEGACY_PLAN_MODE.get(m, m)


def planner_split(model_path, c, plan_mode=None):
    """Where the planner thinks the split falls, for seeding the ladder.

    Returns {ngl, ncmoe, n_cpu_ffn, max_ctx, ceiling_max_ctx, mode} - or None if
    the planner could not answer. `ncmoe` only means anything on an MoE, and on
    an MoE it is the one that matters; `max_ctx` is the answer of the plan that
    was read (the ceiling plan's window). `ceiling_max_ctx` is the ceiling
    plan's answer REGARDLESS of which plan was read - on a non-dense model the
    two are the same number, and it is the key ngl_ladder() keys its coverage
    rule on. See ngl_ladder().

    `plan_mode` picks which of the two dense plans to read. analyze() computes
    both regardless, so this only decides which one lands in result["plan"] -
    passing "ceiling" here is what makes the ctx ladder centre on the ceiling
    plan's window rather than on the fit plan's layer count.

    Imported here rather than at module scope only to keep the cost off the path
    of callers that never build a grid; plan sits above bench in the DAG, so the
    direction is fine."""
    from .plan import analyze, default_vram_budget  # plan is above bench in the DAG
    from .gpu import get_gpus, get_ram
    try:
        g = (get_gpus() or [{}])[0]
        ram = get_ram() or {}
        # TOTAL, not free - and via the shared helper, because the browser used to
        # answer this same question its own way (free VRAM, 0 reserve) and so
        # planned a different config than the one the ladder was centred on. See
        # default_vram_budget() for why total is the right basis here.
        r = analyze(model_path, int(c["ctx"]), c["kv"], int(c["ub"]), bool(c["fa"]),
                    vram_budget_mib=default_vram_budget(g, PLAN_BASIS, 0),
                    ram_budget_mib=float(ram.get("total_mib") or 0),
                    gpu_reserve_mib=PLAN_RESERVE_MIB,
                    compute_override_mib=None, safety_pct=PLAN_SAFETY_PCT,
                    n_seq=int(c.get("seq") or 1),
                    include_mmproj=(c.get("mmproj_offload") is not False),
                    mtp_spec=bool(c.get("spec") == "draft-mtp"),
                    spec_kv=c.get("spec_kv") or "f16",
                    plan_mode=plan_mode)
        pl = r.get("plan") or {}
        n = pl.get("n_gpu_layers")
        mx = pl.get("max_ctx")
        # The ceiling plan's window, from whichever plan was read: ngl_ladder's
        # coverage rule needs it even when the fit plan is the one on screen,
        # and on a non-dense model the ceiling plan is the plan, so this falls
        # back to the read plan's own max_ctx.
        cmx = pl.get("max_ctx")
        plans = r.get("plans") or {}
        if "ceiling" in plans:
            cmx = (plans["ceiling"] or {}).get("max_ctx") or cmx
        return {"ngl": int(n) if n is not None else None,
                "ncmoe": int(pl.get("n_cpu_moe") or 0),
                "n_cpu_ffn": int(pl.get("n_cpu_ffn") or 0),
                "max_ctx": int(mx) if mx else None,
                "ceiling_max_ctx": int(cmx) if cmx else None,
                "mode": r.get("plan_mode")}
    except Exception:
        return None


def ngl_ladder(model_path, c, n_layers, is_moe=False, plan_mode=None, n_ctx_train=0,
               pin_ctx=True, ffn_pin=None):
    """Stage A's axis, its search domain, and what stays pinned while it runs.

    Returns (axis, values, pins) - the config key being searched, every value it
    may land on (ascending in VRAM cost, see wall_values), and the keys that
    must be fixed for the search to mean anything.

    The domain is the WHOLE range now, not a bracket around the planner's seed.
    A bisection covers 66 layer counts in seven loads, which is fewer than the
    nine-rung bracket cost, so there is nothing left to buy by guessing where to
    centre it - and a bracket that guessed wrong used to hide the wall
    completely. planner_split() is still asked, but only for the pins.

    On a DENSE model the axis follows the PLAN MODE, because the two plans ask
    different questions and the wall is a different knob in each:
      * FIT: the context is the thing being held, so whole blocks are what
        is free. The axis is -ngl with every block's dense FFN exiled, and the
        wall is the most blocks that load. If ALL of them load there is VRAM to
        spare, and phase 2 spends it walking the FFN back on - which is the
        question this function used to answer up front, from the planner's seed,
        before anything had been measured.
      * CEILING: every block's attention and KV stays on the GPU (-ngl all) with
        the FFN exiled, and the only thing left free is the window. The axis is
        ctx and the wall is the largest context that still loads. Same phase 2
        after it, for the same reason.
    Searching -ngl under the ceiling plan would measure a config that plan does
    not propose, and searching ctx under the fit plan would sweep the one
    number the user pinned - each is the other's mistake.

    On an MoE it is --n-cpu-moe, and -ngl is pinned at every block. The two
    knobs are not interchangeable and the difference is most of the model:
      * -ngl N        puts the last N blocks on the GPU - attention, KV and
                      experts together.
      * --n-cpu-moe M moves only the ROUTED EXPERTS of the first M blocks to the
                      CPU, leaving their attention and KV in VRAM.
    Experts are the only weights big enough to be worth moving (92% of the file
    on Qwen3.6-35B-A3B) and only n_expert_used of n_expert fire per token, so
    exiling them costs far less per token than exiling whole blocks - while
    whole-block offload drags KV to the CPU with it, which is the expensive
    thing to lose. Searching -ngl on an MoE therefore measures the fallback
    strategy and never finds the good configuration at all. There is no phase 2
    on an MoE: --n-cpu-moe IS its FFN knob, and phase 1 just searched it.

    Lower n_cpu_moe means more experts resident, so there the wall is the
    SMALLEST value that loads - which wall_values() expresses by reversing the
    domain rather than by anything here having to know it.

    `ffn_pin` is a hand-set n_cpu_ffn (--speed-ot, the form's field). It REPLACES
    the every-block pin both dense modes would use, including when it is 0 -
    which is the whole point of typing 0: keep every dense FFN on the card and
    ladder the other axis against that. It is not a default to fall back on, so
    None and 0 are different answers here and the test has to be `is None`."""
    seed = planner_split(model_path, c, plan_mode=(None if is_moe else plan_mode)) or {}
    ffn = n_layers if ffn_pin is None else max(0, min(int(ffn_pin), n_layers))
    if is_moe:
        # Every block on the GPU; the expert split is what varies. The pin still
        # carries the planner's seed because stage B has to sit somewhere until
        # phase 1 promotes, and the seed is a better guess than SPEED_BASE's.
        seed_ncm = seed.get("ncmoe") or max(1, int(n_layers * 0.5))
        return ("ncmoe", wall_values("ncmoe", n_layers),
                {"ngl": n_layers, "ncmoe": seed_ncm})

    pm = norm_plan_mode(plan_mode)
    cmx = seed.get("ceiling_max_ctx")
    if pm is None:
        # auto: the campaign's own coverage rule, kept deliberately SEPARATE
        # from the report's auto (analyze's default pick, which prices both
        # plans at the requested context and takes the faster). Stage A ladders
        # the axis its plan leaves free, and the ceiling plan leaves ctx free
        # exactly when its window reaches the requested context - the same test
        # the old auto rule ran against the speed plan, now read off the
        # ceiling plan's window instead of the report's chosen one.
        req = int(c.get("ctx") or 0)
        pm = "ceiling" if (cmx is not None and cmx >= req) else "fit"
    if pm == "ceiling":
        # The ceiling plan pins BOTH placements at maximum - every block on the
        # GPU, every block's FFN off it - and the window is what is left free.
        pins = {"n_cpu_ffn": ffn, "ngl": n_layers}
        values = wall_values("ctx", n_layers, n_ctx_train)
        if not values:
            # No trained context to build a domain from. One rung is still a
            # measurement, and it is the one the caller asked about.
            values = [int(c.get("ctx") or 0)] if c.get("ctx") else []
        # Stage B sits at the window the mode PROPOSES until phase 1 promotes,
        # not at a default context belonging to whichever model SPEED_BASE was
        # last used against. pin_ctx is False when the user froze the context by
        # hand, which is the one case where their number outranks the planner's.
        # The seed's max_ctx is the window of the plan the REPORT read - its auto
        # usually answers fit, whose "window" is just the requested context - so
        # the pin takes the ceiling window first and the read plan's only as the
        # fallback the old code got for free when report and campaign agreed.
        if pin_ctx and values:
            pins["ctx"] = cmx or seed.get("max_ctx") or values[-1]
        return "ctx", values, pins

    # The fit plan. Whole blocks are the axis and the FFN stays fully exiled
    # while they are searched; whether any of it can come back is phase 2's
    # question, answered from what phase 1 measured rather than from the seed.
    seed_ngl = seed.get("ngl")
    if seed_ngl is None:
        seed_ngl = max(1, int(n_layers * 0.4))     # nothing better to go on
    return ("ngl", wall_values("ngl", n_layers),
            {"n_cpu_ffn": ffn, "ngl": min(seed_ngl, n_layers)})


# The stages a campaign runs, in order. There are two.
#
#   A  the wall - which value of the mode's own axis is the last one that loads,
#      then how little dense FFN has to stay exiled at it (phase 2)
#   B  speculation - which draft scheme and depth is fastest at what A found
#
# The letters that are gone, and why, because "where did stage C go" is a fair
# question to ask of a store full of stage-C rows:
#
#   B (old)  swept the projector's placement, which is a DECISION rather than a
#            measurement: say where it goes on the plan form and both the plan
#            and the campaign use that. The letter is now stage D's.
#   C        swept ubatch. It is a PREFILL knob and decode barely moves with it -
#            this tool's own insights put it at +1.3% against +37% for
#            speculation and +17% for the expert split - so it spent four loads
#            an hour apiece to re-derive that. It is a form field now, frozen
#            for the campaign the way ctx and kv are. --speed-axes "ub=..."
#            still sweeps it by hand for anyone who wants the number again.
#   E        swept the dense-FFN -ot pin, which both plan modes fix at every
#            block - that pin is what MAKES them the modes they are. Stage A's
#            phase 2 now answers the part of it that was a real question:
#            given the wall, how much FFN can come back onto the card.
#
# Renaming D to B is not cosmetic: with C gone the campaign is A-then-speed, and
# a letter that skips two places reads as two stages that failed to run. Rows
# already on disk keep the letter they were measured under - _key() never
# included the stage, so nothing about resume, dedupe or promotion depends on
# it, and rewriting history to tidy a label is not a trade worth making.
STAGES = "ab"

# The MTP depth ladder, dense through the region where the optimum actually
# falls. It used to be 1, 2, 3, 5 - which SKIPPED 4, and 4 is where the only
# depth curve ever measured here peaked: on gemma-4-12B, 67.06 tok/s at depth 1
# rising to 78.84 at 4 (+18%), holding at 78.56 through 5, then falling away -
# 74.97 at 6, 71.49 at 7, 52.05 at 15, as acceptance decays from 0.822 to 0.215.
#
# So the ladder is dense from 1 to 6 and then samples the tail at 8 and 10,
# rather than spending a load on every depth to watch a curve that is already
# descending. Deeper is not free - the draft cache grows with depth - but it is
# also self-limiting: spec_n_max is monotone up, so the first OOM prunes every
# deeper rung in the same family, and stage B's wall walk resumes the ladder at
# a split that fits instead of abandoning it.
STAGE_B_SPEC = [("none", 0), ("draft-mtp", 1), ("draft-mtp", 2), ("draft-mtp", 3),
                ("draft-mtp", 4), ("draft-mtp", 5), ("draft-mtp", 6),
                ("draft-mtp", 8), ("draft-mtp", 10),
                ("ngram-mod", 0), ("ngram-cache", 0), ("ngram-simple", 0)]

# The old name, kept so a caller written against it still works. The list is the
# same object, not a copy.
STAGE_D_SPEC = STAGE_B_SPEC


def grid_context(facts, mmproj=None, base=None, model_path=None, carried=None,
                 drafter=None, plan_mode=None, pin_ctx=True, ffn_pin=None):
    """(baseline, n_layers, is_moe, stage-A axis, stage-A domain) - what stages need.

    Split out of build_speed_grid() so a chained campaign can rebuild it between
    stages against a baseline it has actually MEASURED, rather than against the
    planner's opening guess.

    `carried` is such a baseline: given, it supplies the starting config. It no
    longer re-centres anything. It used to, and had to, because the search was a
    bracket around a seed and a second round with speculation on genuinely moves
    the wall out from under it. The domain is the whole range now, so the wall
    cannot move outside it - and a second round re-probes almost nothing anyway,
    since every value it revisits is already recorded and answers for free.

    `drafter` is the DFlash drafter {path, block_size} found next to the model;
    its path rides the baseline as `md` so the whole grid carries it.

    `ffn_pin` is the hand-set n_cpu_ffn, forwarded to ngl_ladder(). It cannot be
    left to ride in on `base`: the pins are applied OVER the baseline below, so a
    hand-set value sitting in `base` was overwritten by the mode's every-block
    pin and the campaign measured the split the user had just overridden."""
    b = dict(base or SPEED_BASE)
    if mmproj:
        b["mmproj"] = mmproj
    if drafter:
        b["md"] = drafter["path"]
    nl = facts.get("n_layers") or 65
    # A context past what the model was trained on is not a config, it is a bad
    # request - and a staged grid carries a default context belonging to whichever
    # model it was last used against.
    trained = facts.get("n_ctx_train") or 0
    if trained and b.get("ctx", 0) > trained:
        b["ctx"] = trained
    is_moe = bool(facts.get("is_moe"))
    if model_path:
        axis, rungs, pins = ngl_ladder(model_path, b, nl, is_moe,
                                       plan_mode=plan_mode, n_ctx_train=trained,
                                       pin_ctx=pin_ctx, ffn_pin=ffn_pin)
    else:
        # No model to ask the planner about, so no pins - only ever used by the
        # size estimate and by tests. The domain is the same one a real campaign
        # would search, because it does not depend on the planner at all.
        axis, pins = "ngl", {}
        rungs = wall_values("ngl", nl)
    # The pins ARE the baseline for stage B: it must sit at the split the mode
    # proposes, not at a layer count belonging to whichever model SPEED_BASE was
    # last written against.
    for k, v in pins.items():
        b[k] = min(v, nl) if k in ("ngl", "ncmoe", "n_cpu_ffn") else v
    # The pins can carry a context of their own, so the trained-context clamp
    # has to run again after them - a plan's max_ctx is not automatically one
    # the model was trained for.
    if trained and b.get("ctx", 0) > trained:
        b["ctx"] = trained
    if carried:
        for k, v in carried.items():
            if k != "stage":
                b[k] = v
    return b, nl, is_moe, axis, rungs


# ---------------------------------------------------------------------------
# Monotone walls: what one OOM proves about the configs still queued
# ---------------------------------------------------------------------------
# VRAM use is monotone in these axes - each block adds weights and its own KV,
# a bigger ubatch buys a bigger compute buffer, a deeper draft buys a bigger
# draft cache - so the first hard `oom` at a value proves every worse value
# fails the same way, and the loads that would re-prove it are waste. Nothing
# else does that: a genfail or timeout at one rung says nothing about the next,
# and a `spilled` row loaded and ran (badly), which is a measurement, not a
# wall. So only `oom` prunes, and only along a TAGGED axis within the SAME
# family - every other knob equal - because changing any of them can move the
# wall.
_MONOTONE_AXES = {"ngl": "up", "ub": "up", "ctx": "up", "spec_n_max": "up"}

# On an MoE the expert split is monotone the other way: more experts on the CPU
# means less VRAM, so the wall is the SMALLEST n_cpu_moe that fits and worse
# runs downward. ngl stays monotone-up everywhere - more whole blocks always
# cost more - which is why it is never architecture-gated.
_MONOTONE_MOE = {"ncmoe": "down"}

# On a DENSE model the FFN tensor pin is monotone the same way as the expert
# split: more blocks' dense FFN on the CPU means less VRAM, so the wall is the
# SMALLEST n_cpu_ffn that fits. It is the dense counterpart of _MONOTONE_MOE -
# the two knobs are each other's architecture - so it is gated the other way.
_MONOTONE_FFN = {"n_cpu_ffn": "down"}


def _wall_val(c, axis):
    """The axis value, normalised the way _key() normalises optional keys."""
    if axis in ("ncmoe", "spec_n_max", "n_cpu_ffn"):
        return c.get(axis) or 0
    return c.get(axis)


def _same_family(a, b, axis):
    """Do two configs differ only on `axis`? An OOM only speaks for configs
    that share everything else - changing any other knob can move the wall."""
    for k, v in a.items():
        if k == axis or k == "_wall":
            continue
        if b.get(k) != v:
            return False
    for k, v in b.items():
        if k == axis or k == "_wall":
            continue
        if a.get(k) != v:
            return False
    return True


def _worse(oom_cfg, q, axis, direction):
    ov, qv = _wall_val(oom_cfg, axis), _wall_val(q, axis)
    return (qv > ov) if direction == "up" else (qv < ov)


def _prune_queue(queue, walls, i, name, oom_cfg):
    """Drop pending configs whose OOM the row just measured already proves.

    `i` is the index of the OOM row inside `queue`; entries before it have run
    or are running. `walls` maps each queued config's resume key to the axes it
    was tagged with when generated. Returns (kept, dropped)."""
    kept, dropped = queue[:i + 1], []
    tags = walls.get(_key(name, oom_cfg)) or ()
    for q in queue[i + 1:]:
        if any(_same_family(oom_cfg, q, axis) and _worse(oom_cfg, q, axis, d)
               for axis, d in tags):
            dropped.append(q)
        else:
            kept.append(q)
    return kept, dropped


def _resume_stage(cfgs, recorded, name, facts, tries, drafter, axis, log):
    """A stage's list with the rows already on disk READ, not merely dropped.

    A resumed campaign used to drop every config it had measured before, which
    threw away two things the original run had done with those rows.

    First, the OOM that STARTS a draft family's wall walk. The walk is driven by
    measuring an OOM, and a recorded config never reaches _spec_retry - so on a
    resume the walk did not happen at all, and stage B re-reported the same
    "speculation does not fit" it reported last time, having measured nothing.

    Second, the pruning that OOM did. Pruned rows are deliberately never
    written, so every config the first run ruled out came BACK on the next one
    and was measured individually, each load re-proving the same wall. Thirteen
    of them, on the campaign that prompted this.

    Reading the rows in order fixes both: it is the loop run_group runs, with
    the loads replaced by lookups. Returns the configs still worth measuring,
    the walk's next probe last."""
    todo, walked, ruled_out = [], [], []
    for c in cfgs:
        r = recorded.get(_key(name, c))
        if r is None:
            # Already ruled out by a recorded OOM in the same family - the first
            # run pruned it and never wrote it down.
            if any(_same_family(o, c, ax) and _worse(o, c, ax, d)
                   for o, tags in ruled_out for ax, d in tags):
                continue
            todo.append(c)
            continue
        log(_fmt_row(c, r) + "   (recorded earlier)")
        if r.get("status") == "oom":
            ruled_out.append((c, c.get("_wall") or ()))
        nxt = _spec_retry(c, r, facts, tries, drafter=drafter, axis=axis)
        # A walk whose rungs are ALSO on disk - a campaign resumed twice - keeps
        # replaying for free until it reaches one nobody has measured.
        while nxt is not None:
            r2 = recorded.get(_key(name, nxt))
            if r2 is None:
                break
            log(_fmt_row(nxt, r2) + "   (recorded earlier)")
            nxt = _spec_retry(nxt, r2, facts, tries, drafter=drafter, axis=axis)
        if nxt is not None:
            walked.append(nxt)
    return todo + walked


class _WallSearch(object):
    """The highest index in `values` that LOADS, found by bisection.

    `values` is ordered ascending in VRAM cost (wall_values), so "it loaded" is
    a monotone predicate: if index i fits then every index below it fits, and
    the first index that does not rules out everything above. That is the same
    fact _MONOTONE_AXES encodes for the OOM pruning - used here to CHOOSE the
    next probe rather than to discard queued ones.

    "Fits" is `status == "ok"` and nothing else. `oom` obviously does not fit.
    Neither do `genfail`, `timeout`, `exit` or `skipped` - not because any of
    them proves a wall, but because a bisection needs a TOTAL predicate and this
    is the safe direction: it keeps more exiled and less allocated. It is also
    exactly the rule best_config() promotes by, so the value this converges on
    and the value the campaign carries forward cannot disagree. And it is what
    makes a skipped config unwinnable, which is the whole point of skipping it.

    Determinism is the other half. The probe sequence depends only on the
    statuses fed back, so the same campaign over the same machine takes the same
    path - and a resumed one takes it for free, because a probe whose config is
    already recorded is answered from disk without loading anything.
    """

    def __init__(self, axis, values, base, stage="A", known_fits=None,
                 wall=None, max_probes=None):
        self.axis = axis
        # --limit / "Stop after N" is a load budget, so it has to be able to
        # end a search part-way. Enforced in next() rather than by trimming the
        # domain: a bisection's cost is not in its length, and cutting the
        # domain would silently change WHICH wall it looks for.
        self.max_probes = None if max_probes is None else max(0, int(max_probes))
        self.values = list(values)
        self.base = dict(base)
        self.stage = stage
        self.wall = list(wall) if wall else [(axis, axis_direction(axis))]
        self.lo = 0
        self.hi = len(self.values) - 1
        self.best = None
        self.probed = []                  # (value, status), in the order taken
        self._pending = None
        if known_fits is not None and known_fits in self.values:
            # A value an earlier phase already proved. It is the FLOOR of the
            # search rather than a probe: bisecting down to re-measure a load
            # that has already happened is the one waste resume cannot save,
            # because the config differs from the one on disk in another knob.
            i = self.values.index(known_fits)
            self.best = i
            self.lo = i + 1

    @property
    def budget(self):
        """Probes still to take, at most. Exact enough to be a denominator.

        A bisection over n candidates takes ceil(log2(n + 1)) probes in the
        worst case, and the worst case is what a progress bar has to promise.
        Under by a probe or two is the shape of the error, which is the right
        direction for a bar to be wrong in."""
        n = wall_probes(range(self.lo, self.hi + 1))
        if self.max_probes is not None:
            n = min(n, max(0, self.max_probes - len(self.probed)))
        return n

    def next(self):
        """The next config to measure, or None once the wall is found."""
        if self._pending is not None:
            return self._pending[1]       # issued and not yet fed
        if self.lo > self.hi:
            return None
        if self.max_probes is not None and len(self.probed) >= self.max_probes:
            return None
        mid = (self.lo + self.hi) // 2
        self._pending = (mid, self.config_for(self.values[mid]))
        return self._pending[1]

    def config_for(self, value):
        """The config this search would measure at `value`.

        One place builds them, so a caller that wants the SET of configs a
        search could land on - the self-test, a preview - cannot construct them
        a second, subtly different way."""
        c = dict(self.base)
        c[self.axis] = value
        c["stage"] = self.stage
        c["_wall"] = list(self.wall)
        return c

    def feed(self, row):
        """Answer the outstanding probe with the row it produced."""
        if self._pending is None:
            return
        mid = self._pending[0]
        self._pending = None
        status = (row or {}).get("status")
        self.probed.append((self.values[mid], status))
        if status == "ok":
            self.best = mid
            self.lo = mid + 1
        else:
            self.hi = mid - 1

    @property
    def winner(self):
        """The value at the wall, or None if nothing in the domain loaded."""
        return None if self.best is None else self.values[self.best]

    @property
    def topped_out(self):
        """Did the LARGEST value in the domain load?

        The question phase 2 turns on. A winner in the middle of the domain
        means the wall is real and there is nothing spare; a winner at the top
        means the axis ran out of room to spend VRAM on before the card ran out
        of VRAM, so there is some left and the dense FFN can come back onto it.
        """
        return self.best is not None and self.best == len(self.values) - 1

    def summary(self):
        """One line naming what the search did, for the log."""
        if not self.probed:
            return "%s: nothing probed" % self.axis
        return ("%s: %d probe%s, %s -> %s" % (
            self.axis, len(self.probed), "" if len(self.probed) == 1 else "s",
            " ".join("%s=%s" % (v, st or "?") for v, st in self.probed),
            self.winner if self.winner is not None else "nothing loaded"))


def _tag_axes(cfgs, axes, is_moe):
    """Tag an explicit --speed-axes ladder with its monotone axes, in place.

    Only the axes whose VRAM use is provably monotone can prune. A ladder over
    `spec` or `temp` cannot - the schemes and samplers have no ordering - so
    those axes stay untagged and their rows are all measured, exactly as typed.
    Cross-products get one tag per monotone axis; each prunes along its own
    axis within a family that fixes everything else, so a ngl x spec ladder
    prunes ngl rows within each spec."""
    tags = [(a, d) for a, d in _MONOTONE_AXES.items() if a in axes]
    if is_moe:
        tags.extend((a, d) for a, d in _MONOTONE_MOE.items() if a in axes)
    else:
        tags.extend((a, d) for a, d in _MONOTONE_FFN.items() if a in axes)
    if not tags:
        return
    for c in cfgs:
        c["_wall"] = list(tags)


def _depth_extras(out, base, facts, extra_fills, axis, direction):
    """The configs that re-measure the top stage-A rungs at deeper fills.

    The wall does not move with fill - the allocation is decided by the frozen
    ctx at load time, and fill only decides how much of it is used - so the
    rungs the ladder already proved fit are the rungs a deep-fill row can use,
    without paying for the wall again. Rungs are ranked by the wall axis: the
    most layers on the GPU is the config worth re-measuring at depth, and the
    top three are what the docs tell you to run anyway.

    Rows at another fill never compete for the chained baseline - comparable()
    gates on fill, so these are a separate experiment on purpose: the depth
    slope §8 keeps telling you to run, made part of the campaign instead of a
    command line after it."""
    if not extra_fills:
        return []
    got = []
    for r in out:
        c = r.get("config") or {}
        if (r.get("status") == "ok" and c.get("stage") == "A"
                and (c.get("fill") or 0) == (base.get("fill") or 0)
                and c.get(axis) is not None):
            got.append(c)
    rungs = sorted({c.get(axis) for c in got}, reverse=(direction == "up"))
    cfgs = []
    for f in extra_fills:
        for v in rungs[:3]:
            c = dict(base)
            c[axis] = v
            c["fill"] = f
            c["stage"] = "A"
            c["_wall"] = [(axis, direction)]
            cfgs.append(c)
    return cfgs


# There used to be an _AXIS_DIRECTION table here, holding the same four facts
# as _MONOTONE_AXES / _MONOTONE_MOE / _MONOTONE_FFN. It is gone: axis_direction()
# reads those tables directly and every caller now goes through it, so there is
# no second copy left to drift. Two tables that agree "by construction" only
# agree until somebody edits one.


def stage_configs(letter, b, nl, is_moe, axis, rungs, mmproj=None, facts=None,
                  drafter=None):
    """The configs one stage varies, from the baseline it is handed.

    One knob at a time, deliberately - the same reasoning as sweep.build_grid,
    where a confounded design fitted a coefficient at 9x its prior. Here the cost
    is worse than a bad fit: a cross product of these axes is hundreds of loads at
    roughly two minutes each."""
    facts = facts or {}
    out = []

    def add(wall=(), **kw):
        c = dict(b)
        c.update(kw)
        c["ngl"] = max(0, min(c["ngl"], nl))
        if wall:
            c["_wall"] = list(wall)
        out.append(c)

    if letter == "a":
        # Nothing. Stage A is a SEARCH, not a list - see _WallSearch. Its
        # configs are chosen one at a time from what the previous one did, so
        # there is no list to return and returning the whole domain would be a
        # lie about the cost: 64 values, seven loads. speed_sweep drives it
        # directly and wall_probes() is what the estimate counts.
        return out
    if letter == "b":
        # draft-mtp needs nextn blocks somewhere: in the target file itself, or
        # in an external drafter the user named (a separate MTP GGUF - e.g. a
        # Qwen with a standalone *-MTP file). Without either, llama.cpp has
        # nothing to draft from, so those rows are wasted loads that all fail
        # the same way - and the n-gram variants, which need nothing, are the
        # only speculation such a model can use.
        has_mtp = bool(facts.get("n_mtp_layers"))
        # A drafter without a `kind` is the DFlash drafter by construction - the
        # pre-picker shape only ever carried {path, block_size}.
        mtp_drafter = bool(drafter and drafter.get("kind") == "mtp")
        for sp, nmax in STAGE_B_SPEC:
            if sp == "draft-mtp":
                if mtp_drafter:
                    # The drafter rides every row the way DFlash's does. Depth
                    # is NOT capped at its nextn_predict_layers - see
                    # _draft_depths(): --spec-draft-n-max is a draft run length,
                    # and a drafter reporting one nextn layer still measures
                    # differently at every depth.
                    add(spec=sp, spec_n_max=nmax, stage="B",
                        md=drafter["path"], wall=[("spec_n_max", "up")])
                elif not has_mtp:
                    continue
                else:
                    add(spec=sp, spec_n_max=nmax, stage="B",
                        wall=[("spec_n_max", "up")])
                continue
            add(spec=sp, spec_n_max=nmax, stage="B",
                wall=[("spec_n_max", "up")])
        # DFlash is the same gate a different way: the drafter is a SEPARATE file,
        # so its presence next to the model decides whether the scheme exists -
        # not a flag on the target, and nothing the model's own facts can say.
        # Depths come from the drafter's trained block size and the ladder walks
        # the whole of it: the draft cache grows with depth, so the first depth
        # that OOMs prunes every deeper one at the same split - the pruned ones
        # are exactly the rows a hand-run --speed-axes ladder used to measure.
        if drafter and drafter.get("kind") != "mtp":
            bs = int(drafter.get("block_size") or 0) or 16
            for nmax in range(1, bs + 1):
                add(spec="draft-dflash", spec_n_max=nmax, stage="B",
                    md=drafter["path"], wall=[("spec_n_max", "up")])
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


def build_speed_grid(facts, mmproj=None, base=None, stages=STAGES, model_path=None,
                     drafter=None, plan_mode=None, pin_ctx=True):
    """The whole staged grid, one knob at a time from one fixed baseline.

    Stages C and D are pinned at the planner's ngl rather than at whatever stage A
    turns out to find, because nothing here can know stage A's result before stage
    A has run. That is what `chain=True` in speed_sweep() fixes, by building the
    stages one at a time instead of all at once; this function stays as the
    unchained grid and as the size the estimate is computed from."""
    b, nl, is_moe, axis, rungs = grid_context(facts, mmproj, base, model_path,
                                              drafter=drafter, plan_mode=plan_mode,
                                              pin_ctx=pin_ctx)
    cfgs, seen = [], set()
    for letter in STAGES:
        if letter in stages:
            cfgs.extend(_dedupe(
                stage_configs(letter, b, nl, is_moe, axis, rungs, mmproj, facts,
                              drafter), seen))
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
_CONFIG_DEFAULTS = {"ncmoe": 0, "spec_n_max": 0, "n_cpu_ffn": 0, "fill": 0,
                    "seq": 1, "temp": 0.0, "top_k": 0, "top_p": 1.0,
                    "min_p": 0.0, "rep_pen": 1.0, "pres_pen": 0.0,
                    "spec_kv": "f16"}

# What carries forward. Not ctx / kv / spec_kv / fill / seq / fa / ub: those are
# frozen by the form and are the campaign's definition rather than any of its
# results. Not `stage`, which is a label.
#
# `ub` moved across that line when stage C was retired, and it had to move in
# BOTH lists at once - out of here and into comparable(). The two lists are each
# other's complement on purpose: an axis a stage can win must be carried, or the
# winner is found and thrown away; a frozen setting must be GATED, or rows
# measured at another value of it compete as though they were the same
# experiment. Doing only half of it would have let the ub-1024 rows every old
# stage C left on disk set the baseline for a campaign frozen at 512.
CARRY_KEYS = ("ngl", "ncmoe", "n_cpu_ffn", "mmproj_offload", "spec",
              "spec_n_max", "md")


# There used to be a STAGE_EXTREME_SLACK here: a rung running more than 5%
# below the fastest rung on its ladder was dropped before the extreme was
# taken. It is gone, and stage A now means exactly what it says - the largest
# value that LOADED.
#
# The slack was a speed judgement inside a stage that does not ask a speed
# question, and it could decide the answer. A large context measured a little
# slow is still the large context that fits, which is the thing the ceiling
# plan exists to find; refusing it because a smaller rung read 6% faster hands back
# the window to buy noise. Decode moves 4.2% across a DOUBLING of context on a
# real 27B (74752 -> 148480 at fill 2048), so the signal it was reading is
# almost all jitter to begin with.
#
# What it was really guarding against - a rung that loads and then thrashes,
# because WDDM spilled it into shared memory - is a SPILL, and there are three
# detectors for that (unsound_reason, demoted, _infer_demotion) which answer it
# from the memory counters instead of inferring it from a speed ranking. Stage
# B still gates on them through trustworthy(). Stage A deliberately does not:
# "did it load" is a question a spilled row answers yes to, and truthfully.


def stage_objective(letter, stage_axis):
    """(axis, objective) for the stage that just ran.

    The two stages ask different questions and there is no rule that answers
    both:

      * "extreme" - the value at the WALL: the largest that still loads, or the
        smallest on an axis where less means more resident. Stage A only. The
        axis is the plan's own - context under the dense ceiling plan, -ngl
        under the fit plan, --n-cpu-moe on an MoE - or n_cpu_ffn in phase 2,
        which
        the caller passes explicitly.
      * "fastest" - highest tok/s. Stage B only, and it is the whole point
        there: a draft scheme's worth is its acceptance rate, which nothing
        about the config predicts and no ordering of depths implies.

    The distinction matters most where it was doing damage: ranking stage A by
    tok/s under the ceiling plan promoted the SMALLEST context on the ladder, because
    a smaller window is a hair faster and the margin never saw the difference as
    real."""
    if letter == "a":
        return stage_axis, "extreme"
    return None, "fastest"


def carry_keys(swept=""):
    """What a stage's winner hands to the next stage.

    CARRY_KEYS is the fixed part: knobs a stage RESULT can move. ctx, kv,
    spec_kv, fill, seq and fa are deliberately absent - they are the campaign's
    definition, and a stage that changed one of them would be answering a
    different question than the one being asked.

    ctx moves across that line under the dense ceiling plan, and only there:
    stage A sweeps it, so it stops being the question and becomes the answer."""
    return CARRY_KEYS + (("ctx",) if swept == "ctx" else ())

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


# None is a MEANINGFUL template_id - "no template pinned, the GGUF's own was
# used" - so it cannot double as "do not filter on this". Hence a sentinel.
_ANY = object()


def comparable(r, model, base, n_predict=None, repeat=None, prompt_id=None,
               template_id=_ANY, swept=""):
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
    passes its own id, the row must carry that same id.

    `swept` names the axis stage A is LADDERING, and exists for exactly one case.
    Context is normally part of what a campaign IS - two contexts are two
    experiments, not two configs - which is why it is gated here. Under the dense
    ceiling plan it is instead what stage A goes looking for, and gating it there
    rejects every rung of the ladder but the one that happens to equal the
    opening baseline: the campaign would measure a wall and then promote the
    planner's guess. Only an axis stage A actually sweeps may be un-gated, and
    only ctx is ever both gated here and absent from CARRY_KEYS."""
    if model and r.get("model") != model:
        return False
    if n_predict is not None and r.get("n_predict") != n_predict:
        return False
    if repeat is not None and r.get("repeat") != repeat:
        return False
    if prompt_id is not None and r.get("prompt_id") != prompt_id:
        return False
    # A template changes the ANSWER, so it changes tok/s: a thinking template
    # spends tokens on a reasoning block before it says anything. Filtered in
    # both directions - a campaign that pinned none must not inherit a baseline
    # from one that did, which is why None here means "no template" and the
    # sentinel means "do not filter".
    if template_id is not _ANY and r.get("template_id") != template_id:
        return False
    c = r.get("config") or {}
    if swept != "ctx" and c.get("ctx") != base.get("ctx"):
        return False
    if not (c.get("kv") == base.get("kv")
            # The draft cache's quant is frozen like the target's: a row
            # measured with a q8_0 draft cache is not a slower or faster
            # version of an f16 one - they are two experiments. Absent means
            # llama.cpp's f16 default on both sides, so rows recorded before
            # the knob existed still compare.
            and (c.get("spec_kv") or "f16") == (base.get("spec_kv") or "f16")
            # ubatch, since stage C was retired and it became a form field. It
            # sizes the compute buffer, so it moves the wall as well as prefill
            # - and the store is full of ub 256/1024/2048 rows that stage C
            # measured back when it was an axis. Ungated, those would compete
            # with a campaign frozen at 512 as though nothing differed.
            and (c.get("ub") or 0) == (base.get("ub") or 0)
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


def axis_direction(axis):
    """Which way `axis` runs toward the wall: "up" if larger costs more VRAM.

    One lookup for every caller, over the same tables the OOM pruning walks, so
    the ladder, the promotion and the recommendation cannot disagree about which
    end of an axis is the good end. "down" is the pair of knobs that measure how
    much has been EXILED - --n-cpu-moe and -ot - where fewer blocks moved off the
    card is more resident and the wall is the minimum.
    """
    return (_MONOTONE_AXES.get(axis) or _MONOTONE_MOE.get(axis)
            or _MONOTONE_FFN.get(axis) or "up")


def pick_extreme(cand, axis, direction=None):
    """The row at the wall along `axis`, ties broken toward the faster one.

    Split out of best_config() because the recommendation asks the same question
    of the same rows: on an axis that is monotone in VRAM the value worth having
    is the one at the wall, and ranking those rows by tok/s ranks jitter - a
    27B dense model moves 4.2% of tok/s across a DOUBLING of context, and
    downward, so fastest-wins promotes the SMALLEST window in the one mode whose
    whole purpose is the largest.

    No slack, no threshold, and no tie-break that can overturn the axis: the
    largest value wins - the smallest, on an axis where less means more
    resident - and tok/s only settles a tie between rows at the SAME value.
    Deliberately boring, because this has to agree with the bisection that
    found the wall, and the bisection knows nothing about speed.

    Callers decide which rows are candidates: stage A passes everything that
    loaded, the recommendation card passes what it would be willing to run.
    """
    if not cand:
        return None
    direction = direction or axis_direction(axis)

    def val(r):
        return (r.get("config") or {}).get(axis) or 0

    def tok(r):
        return r.get("tok_s") or 0.0
    return (max(cand, key=lambda r: (val(r), tok(r))) if direction == "up"
            else min(cand, key=lambda r: (val(r), -tok(r))))


def best_config(rows, model, base, n_predict=None, repeat=None,
                incumbent_tok_s=None, margin=CHAIN_MARGIN, prompt_id=None,
                template_id=_ANY, swept="", axis=None, objective="fastest"):
    """The baseline for the next stage: (config, row) or (None, None).

    Reads rows that are already on DISK, not just the ones this process has in
    memory, so a campaign stopped after stage A and restarted tomorrow picks its
    winner back up instead of falling back to the planner's guess.

    `swept` is the axis the stage searched. It widens BOTH gates together, and
    that pairing is the point: an axis the campaign is free to discover must be
    comparable across its values (or the winner is never a candidate) and must
    then be carried (or the winner is found and thrown away). Doing either alone
    is worse than neither - the first promotes a config nothing measured, the
    second reports a baseline the next stage does not run at.

    `objective` is what "best" means for the stage that just ran, and the two
    stages do not share a rule - see stage_objective(). It decides the RANKING
    and the CANDIDATE GATE together, because the two answer to the same
    question:

      * "fastest" (stage B) ranks by tok/s, so a row only counts if its tok/s
        can be believed. trustworthy() is what says so: a spilled row's speed is
        off a cliff for a reason unrelated to the setting under test, and a
        looping or copying row inflates speculative acceptance to 100% by
        generating nothing. The CHAIN_MARGIN applies, because a 1% lead over a
        median of three noisy passes is jitter and rebasing on jitter makes the
        campaign's path depend on noise.

      * "extreme" (stage A) ranks by the axis, so the only thing a row has to
        have done is LOAD - status == "ok" - at the same draft scheme the
        baseline carries, since a draft model changes what "loads" means by
        several hundred MiB. Not trustworthy():
        it demands a believable tok/s, and stage A is not reading tok/s - so a
        row wrongly flagged as spilled would silently remove a rung from a
        search that never asked about speed. Not the margin either: "which value
        loaded" is not a measurement of speed, so there is no noise floor for it
        to sit above.

    Anything that is not `ok` - oom, genfail, timeout, exit, skipped - is not a
    candidate under either objective. That is the same predicate _WallSearch
    bisects on, deliberately: the value the search converges on and the value
    the campaign carries forward have to be the same value, and they are only
    guaranteed to be if both read `ok` the same way."""
    def gate(r):
        if not comparable(r, model, base, n_predict, repeat, prompt_id,
                          template_id, swept):
            return False
        if objective == "extreme" and axis:
            # ...and carries the same draft scheme. A wall search asks "which
            # value of this axis loads", and a row running a DIFFERENT
            # speculator answers that about a different allocation - a draft
            # model's weights and its f16 KV cache are several hundred MiB the
            # search never priced.
            #
            # Ungated, a RESUMED campaign promoted a stage-B row into stage A:
            # the fastest row at the winning context was an n-gram one, so
            # phase 2 re-ran the FFN bisection under a speculator it had not
            # measured, re-proving on six loads a wall already sitting on disk.
            # `spec` is a CARRY key - carried BETWEEN stages, held fixed WITHIN
            # one - and this is the "held fixed" half. A second --speed-rounds
            # pass still re-walks the wall with the winning speculation on,
            # because by then it is the BASELINE's spec that the rows match.
            return (r.get("status") == "ok"
                    and ((r.get("config") or {}).get("spec") or "none")
                    == (base.get("spec") or "none"))
        return trustworthy(r)

    cand = [r for r in rows if gate(r)]
    if not cand:
        return None, None
    if objective == "extreme" and axis:
        win = pick_extreme(cand, axis)
    else:
        win = max(cand, key=lambda r: r["tok_s"])
        if incumbent_tok_s and win["tok_s"] <= incumbent_tok_s * (1.0 + margin):
            return None, None
    c = dict(base)
    wc = win.get("config") or {}
    for k in carry_keys(swept):
        if k in wc:
            c[k] = wc[k]
    c.pop("stage", None)
    return c, win


def resolve_stages(stages):
    """(letters, note) - the stages to run, from whatever spelling was asked for.

    The campaign used to be "acd" and is now "ab". Both old letters still turn
    up: in a saved command line, in a bookmarked URL, in the muscle memory of
    the person who has been typing --speed-stages acd for months. Neither can
    be left to mean nothing silently, which is the failure resolve_search() and
    resolve_rounds() exist to prevent:

      * `d` IS stage B - the same speculation stage under its new letter - so it
        is translated and said.
      * `c` is the retired ubatch sweep. There is no stage to map it to, so it
        is dropped and said, and the note names the flag that still does it.

    An empty result falls back to the full campaign rather than running nothing:
    "--speed-stages c" asked for a stage, and answering with silence and zero
    loads is the worst of the available readings."""
    want = "".join(sorted(set((stages or "").lower()) & set("abcd")))
    notes = []
    out = ""
    for letter in STAGES:                    # in campaign order, not typing order
        if letter in want:
            out += letter
    if "d" in want and "b" not in out:
        out += "b"
        notes.append("note    : --speed-stages d is stage B now - same speculation "
                     "stage, renamed when the ubatch stage was retired")
    if "c" in want:
        notes.append("note    : --speed-stages c was the ubatch sweep, which is "
                     "retired - ubatch is frozen from the form now. Sweep it by "
                     'hand with --speed-axes "ub=256,512,1024,2048"')
    if not out:
        # Only worth a note when letters WERE given and none of them survived.
        # An empty string is the caller declining to choose, and the answer to
        # that is the default campaign, quietly.
        if want:
            notes.append("note    : nothing left to run after that, so the whole "
                         "campaign runs")
        out = STAGES
    return "".join(sorted(set(out), key=STAGES.index)), notes


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


def resolve_rounds(chain, rounds):
    """Rounds only means something with chaining on. Returns (rounds, note).

    A round re-runs the stages from the WINNER, and without chaining there is no
    winner to re-run them from - every stage is built off the same fixed
    baseline, so round two re-derives an identical grid and skips all of it as
    already recorded. The setting was accepted in that state and did precisely
    nothing, with nothing said. Same reasoning as resolve_search(): a knob that
    silently does not apply is worse than one that is refused."""
    rounds = max(1, int(rounds or 1))
    if rounds > 1 and not chain:
        return 1, ("note    : rounds=%d ignored - a round re-runs the stages from "
                   "the winner, and without chaining there is no winner to re-run "
                   "them from" % rounds)
    return rounds, None


def _carry_summary(c):
    # `ctx` leads because it is a wall axis like the rest - stage A's own under
    # the dense ceiling plan, and one of the two the draft walk gives back. Without
    # it the walk printed the same line at every rung it reached, the context
    # being the only thing that had changed.
    s = ("ctx %s ngl %s ncmoe %s ub %s spec %s"
         % (c.get("ctx"), c.get("ngl"), c.get("ncmoe") or 0, c.get("ub"),
            c.get("spec") or "none"))
    # The draft depth, whenever there is a draft. Without it the wall line for a
    # pruned depth ladder prints the same fifteen words fifteen times - the rows
    # differ ONLY by the depth, which was the one thing not being said.
    if (c.get("spec") or "none") != "none" and c.get("spec_n_max"):
        s += "/%s" % c["spec_n_max"]
    # The FFN pin is the one placement that does not show in ngl/ncmoe, so a
    # carried baseline that uses it would read as the plain split.
    if c.get("n_cpu_ffn"):
        s += " ffn-cpu %s" % c.get("n_cpu_ffn")
    # Same for the draft cache's quant: f16 is llama.cpp's default and needs no
    # saying, anything else changes what a speculative baseline actually IS.
    if c.get("spec_kv") and c.get("spec_kv") != "f16":
        s += " spec-kv %s" % c.get("spec_kv")
    return s


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
                 n_predict=N_PREDICT, repeat=N_REPEAT, log=print,
                 template=(None, None, None, None), template_id=_ANY,
                 on_server=None, skip_count=None, swept=""):
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
                               repeat=repeat, prompt_id=pid,
                               template_id=template_id, swept=swept)
    if win_c is None:
        log("verify  : nothing to certify - no trustworthy row at this campaign's settings")
        return None
    c = verify_config(win_c, overrides)
    # The winner came back off DISK, where template keys are deliberately not
    # stored. Re-attach them, or the one load that certifies the campaign would
    # be the only load in it measured against a different template.
    for k, v in zip(_TEMPLATE_KEYS, template):
        if v:
            c[k] = v
    log("verify  : the winner (%s) does not know the config you actually run -" % _carry_summary(win_c))
    log("          loading %s%s once"
        % (_carry_summary(c),
           "  fill %s" % (c.get("fill") or 0)
           if c.get("fill") else ""))
    row = bench_one(backend, model_path, c, port=port, timeout=timeout,
                    n_predict=n_predict, repeat=repeat, log=log,
                    on_server=on_server, skip_count=skip_count)
    row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                "when": int(time.time()), "gpu": gpu})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    good = row.get("status") == "ok" and trustworthy(row)
    if good:
        log("verify  : OK - the winner loads at your production config and is certified")
    elif row.get("status") == "skipped":
        # A skip is a judgement, not a verdict: the verification load was
        # abandoned on request, so nothing was certified - and nothing was
        # disproved either.
        log("verify  : SKIPPED - the verification load was abandoned on request.")
        log("          No verdict, favourable or not. Run the sweep again and let")
        log("          the verify row finish to certify the winner.")
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


def _resume_recorded(rows, n_predict, repeat, pid, tmpl_id):
    """The rows of a store a resumed campaign may treat as already done.

    `ok` and `oom` are measurements - the rungs a ladder is read from. `skipped`
    and `spilled` are judgements, by the user and by the machine respectively,
    and a judgement is exactly what a resumed campaign must not re-litigate:
    re-measuring a config the user already skipped wastes the same load it just
    wasted, and a config that spilled into shared memory (or ran while the card
    was busy) re-spends the minutes the abort saved. Rows of any other status
    failed for a reason that may not hold tomorrow - a transient exit, a
    generation error - so they are re-measured, not carried.

    `n_predict`, `repeat`, `pid` and `tmpl_id` gate what counts as the SAME
    measurement: a row taken at another depth or under another template is a
    different experiment, not this campaign's work already done."""
    recorded = {}
    for r in rows:
        if (r.get("config") and r.get("status") in ("ok", "oom", "skipped", "spilled")
                and r.get("n_predict") == n_predict and r.get("repeat") == repeat
                and r.get("prompt_id") == pid
                # ...and under the same chat template. A thinking template
                # answers at a different length than a terse one, so a row
                # measured without one is not this campaign's row already done.
                and r.get("template_id") == tmpl_id):
            recorded[_key(r["model"], r["config"])] = r
    return recorded


def speed_sweep(models=None, backend=None, dry_run=False, timeout=420.0,
                port=BENCH_PORT, limit=None, axes=None, stages=STAGES, fill=None,
                fills=None, ctx=None, kv=None, ub=None,
                n_predict=N_PREDICT, repeat=N_REPEAT, log=print, skip_preflight=False,
                on_row=None, should_stop=None, on_total=None,
                should_abort=None, on_server=None, skip_count=None,
                chain=False, rounds=1, verify=False, verify_overrides=None,
                chat_template_file=None, chat_template_kwargs=None,
                reasoning=None, reasoning_preserve=None, sampling=None,
                mmproj_offload=None, drafter=None, ot=None, spec_kv=None,
                plan_mode=None):
    """Run the speed grid and append one row per config. Resumable like --sweep.

    `on_row` is called with each finished row, `on_total` when the number of
    configs to run becomes known, and `should_stop` is checked before each config.
    All three default to None, so the CLI path is unchanged. They exist for the
    web UI: it needs structured rows as they land rather than a transcript to
    scrape, and it needs a way to end a two-hour campaign early.

    `skip_count` is how a config in flight gets ABANDONED without ending the
    campaign: the controlling layer (terminal S key, web Skip button) kills the
    server and bumps its counter; the run then fails promptly instead of sitting
    out its generation timeout, and bench_one() stamps the row `skipped`. The
    row is recorded and keyed, so a resumed campaign never re-measures it - the
    user already judged it once, and making it pay for that judgement twice is
    exactly what the skip exists to prevent. A press that lands between runs is
    harmless: nothing is in flight to kill, and the counter comparison in
    bench_one() binds each press to the run it was made during.

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
    # The DFlash drafter, if the model has one next to it. Stage D then measures
    # the scheme whose cost the planner could only derive, which is the whole
    # point of the stage - and the drafter's block size sets the depth ladder.
    # `drafter` is the campaign's own: None or "auto" discovers as before,
    # "none" means no drafter at all (own-MTP and n-gram rows only), and a path
    # names a specific file - an MTP GGUF, which is the scheme a plan can now
    # price but no discovery can find on its own.
    dp = find_drafter_for(mp)
    if drafter in (None, "auto"):
        drafter = ({"path": dp, "block_size": _drafter_block_size(dp) or 16,
                    "kind": "dflash"}
                   if dp else None)
    elif drafter == "none":
        drafter = None
    else:
        drafter = classify_drafter(drafter)

    # Validated before a single server is launched. A bad kwargs string is a
    # four-hour campaign that dies on config one, or worse - a template path with
    # a typo is not an error at all, just a silent fallback to the GGUF's own.
    from .launch import template_args, reasoning_args
    try:
        tmpl_f, tmpl_k = template_args(chat_template_file, chat_template_kwargs)
        rea, rea_p = reasoning_args(reasoning, reasoning_preserve)
    except ValueError as e:
        log("template: %s" % e)
        return None
    if tmpl_f and not os.path.isfile(tmpl_f):
        log("template: no such file - %s" % tmpl_f)
        return None
    # The thinking flags are part of the template's identity, not separate from
    # it: --reasoning off makes the same template produce a different answer of
    # a different length, which is a different measurement.
    tmpl_id = template_identity(tmpl_f, tmpl_k, rea, rea_p)

    # Which stages, in campaign order, translating the letters the campaign used
    # to have. Resolved before anything reads `stages` - the extra-fills check
    # and the ceiling-mode ctx note both do, and both would answer for a stage
    # list nobody asked for.
    stages, snotes = resolve_stages(stages)
    for n in snotes:
        log(n)

    base = dict(SPEED_BASE)
    if tmpl_f:
        base["chat_template_file"] = tmpl_f
    if tmpl_k:
        base["chat_template_kwargs"] = tmpl_k
    if rea:
        base["reasoning"] = rea
    if rea_p:
        base["reasoning_preserve"] = rea_p
    # Samplers, frozen across the campaign rather than swept - the same standing
    # as ctx and kv. They belong in the config dict because sampling_of() reads
    # them from there and _key() includes them, so a campaign re-run at real
    # settings does not resume greedy rows as though they were the same
    # measurement. Which they are not, and the difference is largest exactly
    # where it is least obvious: greedy makes the target's token deterministic,
    # so it is speculation's best case, and an acceptance rate measured there is
    # an upper bound rather than a result.
    for k, v in (sampling or {}).items():
        if k in _SWEEP_SAMPLER_KEYS and v is not None and v != "":
            base[k] = float(v) if k != "top_k" else int(v)
    # Where the projector goes. This is a PLAN input now rather than a swept
    # axis: the user says whether it belongs in VRAM or in system RAM, and the
    # campaign measures at that placement instead of spending a stage proving
    # which one is faster for a decision that was never really about speed.
    if mmproj_offload is not None:
        base["mmproj_offload"] = bool(mmproj_offload)
    if fill is not None:
        base["fill"] = fill
    # --speed-fills: the first value is the campaign fill, the rest are DEEPER
    # fills that re-measure the top stage-A rungs once the wall is known (see
    # _depth_extras). They are part of the campaign rather than a command line
    # run afterwards, so the depth slope §8 keeps telling you to measure is
    # collected while the grid is already loaded - and at rungs chosen by what
    # actually fit, not by the planner's guess.
    extra_fills = None
    if fills:
        base["fill"] = fills[0]
        extra_fills = fills[1:]
        if extra_fills and (axes or "a" not in (stages or "")):
            log("note    : --speed-fills re-measures the top stage-A rungs at deeper "
                "fills, but %s, so the extra fills were dropped"
                % ("an explicit --speed-axes ladder replaces the staged grid"
                   if axes else "stage A is not in --speed-stages"))
            extra_fills = None
    if ctx is not None:
        base["ctx"] = ctx
    if kv is not None:
        base["kv"] = kv
    # ubatch, frozen like ctx and kv rather than swept. Stage C used to spend
    # four loads on it and found +1.3%; it is a prefill knob, and the campaign
    # measures decode. The form's value is what every config runs at, and
    # --speed-axes "ub=..." is still there for anyone who wants the ladder back.
    if ub is not None:
        base["ub"] = int(ub)
    # The draft cache's quant, frozen like the target's: stage D then measures
    # speculation at the cache it will actually run with, and the planner seeds
    # the ladder (planner_split) under the same assumption.
    if spec_kv is not None:
        base["spec_kv"] = spec_kv
    # Stage B IS the projector sweep. Pinning the placement and then running it
    # anyway would re-measure the axis that was just fixed - the configs would
    # all carry the pinned value, so the stage would spend loads proving one
    # thing four times. Dropped, and said, rather than silently wasting them.
    # An MoE has no dense FFN tensors to pin: stage A's --n-cpu-moe ladder is that
    # model's FFN knob already, so -ot names nothing there.
    if facts.get("is_moe") and ot is not None:
        log("note    : n_cpu_ffn pins DENSE FFN tensors, and this model routes "
            "its experts - the pin was dropped")
        ot = None
    if ot is not None:
        # Both dense plan modes pin the FFN split at EVERY block, which is what
        # makes them the modes they are. A hand-set count overrides that, and is
        # said out loud - it is a different config from the one the planner and
        # the mode-aware ladder were built around.
        base["n_cpu_ffn"] = ot
        log("note    : dense FFN pinned to %d blocks on CPU by hand - the plan "
            "modes pin every block, so this measures a different split" % ot
            if ot else
            "note    : dense FFN pinned to 0 blocks on CPU by hand - every "
            "block's FFN stays on the card, and stage A's phase 2 is off "
            "because that pin is the answer it would have searched for")
    # The ceiling plan's whole job is to FIND the largest context that loads, so
    # a frozen context contradicts it. Refused out loud rather than silently
    # sweeping one rung, the way resolve_search() refuses chain-plus-axes.
    if norm_plan_mode(plan_mode) == "ceiling" and ctx is not None and "a" in (stages or ""):
        log("note    : --speed-mode ceiling sweeps CONTEXT to find the wall, so "
            "--speed-ctx %d cannot freeze stage A" % int(ctx))
        log("          A ladders the window anyway; %d is what the later stages "
            "sit at. Planning FOR a fixed context is --speed-mode fit"
            % int(ctx))
    gb0, gnl0, gmoe0, stage_axis, gvals0 = grid_context(
        facts, mmproj=mmproj, base=base, model_path=mp, drafter=drafter,
        plan_mode=plan_mode, pin_ctx=(ctx is None), ffn_pin=ot)
    # Stage A costs a known NUMBER of loads and an unknown set of configs - it
    # bisects, so which value it probes second depends on what the first one
    # did. Everything downstream that wants a count gets wall_probes(); nothing
    # gets a config list, because there is not one to give.
    #
    # Phase 2 is counted here as though it will run. It only runs when phase 1
    # tops out - the whole domain loaded, so there is VRAM the axis could not
    # spend - which cannot be known before phase 1 has run. An estimate that
    # under-counts the work is worse than one that over-counts it, so the
    # upper bound is what is quoted, and the header says so.
    a_phase2 = bool("a" in stages and not axes and not gmoe0 and gnl0
                    and ot is None)
    a_probes = 0
    if "a" in stages and not axes:
        a_probes = wall_probes(gvals0)
        if a_phase2:
            a_probes += wall_probes(wall_values("n_cpu_ffn", gnl0))
    cfgs, _seen0 = [], set()
    for letter in STAGES:
        if letter in stages:
            cfgs.extend(_dedupe(
                stage_configs(letter, gb0, gnl0, gmoe0, stage_axis, gvals0,
                              mmproj, facts, drafter), _seen0))
    if axes:
        log("note    : explicit ladder, so the draft wall walk is off - an OOM "
            "is recorded as measured and no extra rungs are invented")
        # An explicit ladder replaces the staged grid: this is how a stage gets
        # re-run at the ngl the previous stage actually settled on.
        #
        # It starts from the RESOLVED base, not from SPEED_BASE. Those differ in
        # the one place it matters most: SPEED_BASE carries ngl 26, a value that
        # belongs to no model, while grid_context() asks the planner where the
        # layers actually land - ngl 41 on the MoE this was found on. A ladder
        # over ncmoe anchored at ngl 26 puts fifteen whole blocks on the CPU and
        # measures a config nobody asked about, next to stage rows that used 41.
        # It also clamps ctx to what the model was trained on, which the staged
        # path has always done and this one silently did not.
        gbase = grid_context(facts, mmproj=mmproj, base=base, model_path=mp,
                             drafter=drafter, plan_mode=plan_mode, ffn_pin=ot)[0]
        combos = [dict(gbase, **({"mmproj": mmproj} if mmproj else {}))]
        for k, vals in axes.items():
            combos = [dict(c, **{k: v}) for c in combos for v in vals]
        cfgs = combos
        a_probes, a_phase2 = 0, False
        # An explicit ladder prunes the same way a staged one does: a hard OOM
        # at one rung proves every worse rung in the same family fails the same
        # way, on the monotone axes (see _tag_axes). Ladders over `spec` or
        # `temp` have no ordering, so their rows are all measured, exactly as
        # typed.
        _tag_axes(cfgs, set(axes), bool(facts.get("is_moe")))
    chain, note = resolve_search(axes, chain)
    if note:
        log(note)
    rounds, rnote = resolve_rounds(chain, rounds)
    if rnote:
        log(rnote)

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
    # Keep the ROW, not just the key. A resumed config is not a gap in the
    # ladder, it is a rung that was already climbed - and printing only a count
    # of them turned a complete six-rung ladder into three rows with the OOM
    # boundary missing, which reads as the tool ignoring what was asked for.
    recorded = _resume_recorded(load_rows(path), n_predict, repeat, pid, tmpl_id)
    done = set(recorded)
    plan = [c for c in cfgs if _key(os.path.basename(mp), c) not in done]
    skipped = len(cfgs) - len(plan)
    if limit:
        # Stage A runs FIRST and spends the budget first, so the cap has to
        # apply to it before anything is left over for stage B - the same order
        # the campaign runs in, and the same order _WallSearch's max_probes
        # enforces at run time. Truncating only `plan` reported 18 loads for a
        # --limit 5 that would correctly have run 5.
        a_probes = min(a_probes, limit)
        plan = plan[:max(0, limit - a_probes)]

    log("backend : %s" % b["build"])
    log("gpu     : %s" % (gpu or "unknown"))
    log("model   : %s  (%d blocks, %s)" % (os.path.basename(mp), facts["n_layers"],
                                           facts["arch"]))
    log("mmproj  : %s" % (os.path.basename(mmproj) if mmproj else "none"))
    log("prompt  : %s (frozen corpus; rows are keyed on it, so a refresh re-measures)" % pid[:12])
    if tmpl_id:
        log("template: %s %s  (%s, hashed by content - editing it re-measures)"
            % (os.path.basename(tmpl_f) if tmpl_f else "(kwargs only)",
               tmpl_k or "", tmpl_id[:12]))
    else:
        log("template: none pinned - the GGUF's own metadata template")
    if rea or rea_p:
        log("thinking: --reasoning %s, preserve history %s"
            % (rea or "auto", rea_p or "template default"))
    # Said out loud either way. Greedy is the default and it is speculation's
    # best case, so a campaign that never mentions its samplers is the one whose
    # acceptance rate is most likely to be read as a result rather than a bound.
    samp_set = [(k, base[k]) for k in _SWEEP_SAMPLER_KEYS if k in base]
    log("sampling: %s" % (" ".join("%s %s" % kv for kv in samp_set) if samp_set
                          else "greedy (temp 0) - reproducible across configs, and "
                               "speculation's BEST case, so read acceptance as an "
                               "upper bound"))
    # The draft cache's quant is frozen like kv, and f16 is llama.cpp's default,
    # so it is only said when it is not the thing everyone would assume.
    if (base.get("spec_kv") or "f16") != "f16":
        log("draftkv : %s (-ctkd/-ctvd) - stage B measures the draft cache at this"
            % base["spec_kv"])
    log("output  : %s" % path)
    # Stage A's loads are counted, not listed. Everything else in the campaign
    # is a config that exists before it runs; stage A's are decided as it goes.
    total_loads = len(plan) + a_probes
    if a_probes:
        log("stage A : %d probe%s at most - a bisection over %d value%s of %s%s"
            % (a_probes, "" if a_probes == 1 else "s", len(gvals0),
               "" if len(gvals0) == 1 else "s", stage_axis,
               ", then the dense FFN walk" if a_phase2 else ""))
    log("configs : %d to run, %d already recorded" % (total_loads, skipped))
    log("estimate: ~%.1f h  (load + %d warm + %d x %d tokens per config)"
        % (total_loads * (60.0 + repeat * n_predict / 4.0) / 3600.0, 16, repeat,
           n_predict))
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
        if chain and a_probes:
            # Stage B is built from a baseline that does not exist yet, so
            # printing values for it would be a guess dressed as a plan - and
            # the guess it would print is exactly the frozen one this mode
            # exists to escape. The COUNT is still exact: the spec list does not
            # depend on the baseline, only its ngl/ctx columns do.
            shown = []
        for c in shown:
            log("  %-2s ctx %-7d kv %-5s ngl %-3d ncmoe %-3d ot %-3d ub %-5d "
                "fill %-7d spec %-13s nmax %-2d mmproj %s"
                % (c.get("stage", "-"), c["ctx"], c["kv"], c["ngl"],
                   c.get("ncmoe") or 0, c.get("n_cpu_ffn") or 0, c["ub"],
                   c.get("fill") or 0, c.get("spec") or "none",
                   c.get("spec_n_max") or 0,
                   "vram" if c.get("mmproj_offload") is not False else "ram"))
        sizes = {}
        for c in plan:
            k = (c.get("stage") or "-").lower()
            sizes[k] = sizes.get(k, 0) + 1
        if a_probes:
            sizes["a"] = a_probes
        if len(shown) < len(plan):
            log("  .. and %d more in stage B, built from what stage A finds"
                % (len(plan) - len(shown)))
        if extra_fills:
            log("depth   : after stage A, the top rungs are re-measured at fill%s %s "
                "- not listed here, since they depend on what stage A finds"
                % ("s" if len(extra_fills) > 1 else "",
                   " ".join(str(f) for f in extra_fills)))
        if verify:
            log("verify  : after the campaign, the winner is loaded once more at the")
            extra = (" (" + " ".join("%s=%s" % (k, ",".join(map(str, v)))
                                     for k, v in sorted(verify_overrides.items()))
                     + ")") if verify_overrides else ""
            log("          production config%s - a split that fits the grid is not"
                " automatically one that fits what you actually run" % extra)
        # The config list rides along so the UI can show the same preview the CLI
        # prints, rather than parsing the lines above back out of a log.
        return {"planned": total_loads, "path": path, "dry_run": True,
                "configs": shown, "skipped": skipped, "model": os.path.basename(mp),
                "backend": b["build"], "gpu": gpu, "arch": facts["arch"],
                "n_layers": facts["n_layers"],
                "mmproj": os.path.basename(mmproj) if mmproj else None,
                "chained": bool(chain), "rounds": int(rounds or 1),
                "provisional": len(shown) < len(plan) or bool(a_probes),
                "stage_sizes": sizes,
                # What stage A will search, so the preview can say what the
                # campaign is about to go looking for without pretending to know
                # which values it will land on.
                "wall_axis": stage_axis if a_probes else None,
                "wall_domain": len(gvals0) if a_probes else 0,
                "wall_probes": a_probes,
                "wall_phase2": bool(a_phase2),
                "verify": bool(verify),
                "estimate_h": round(total_loads * (60.0 + repeat * n_predict / 4.0)
                                    / 3600.0, 2)}
    if not skip_preflight and not preflight(log=log):
        return None

    log("")
    log("%-2s %-7s %-4s %-5s %-4s %-5s %-8s %-13s %-4s %-6s | %8s %9s %7s %6s %6s"
        % ("st", "ctx", "ngl", "ncmoe", "ot", "ub", "fill", "spec", "nmax",
           "mmproj", "tok/s", "prefill", "VRAM", "accept", "distin"))
    # The rungs already climbed, printed in place before the new ones. Resume
    # exists so a stopped campaign is not re-paid for, but a ladder is read as a
    # ladder: where it OOMs and where it starts working is the whole finding,
    # and half of it missing looks like the request was ignored rather than
    # already answered. Marked "was" so nothing here reads as measured just now.
    name = os.path.basename(mp)
    for c in cfgs:
        r = recorded.get(_key(name, c))
        if r is not None:
            log(_fmt_row(c, r) + "   (recorded earlier)")
    out, stopped = [], False
    total = [len(plan) + a_probes]      # a list so run_group can revise it
    # The campaign's own estimate, which is NOT the same number as the bar's
    # denominator: the campaign measures one stage at a time and only the
    # running one has a known size. Reported separately rather than reconciled,
    # because they are answers to two different questions and picking one to
    # show made the other look like a mistake.
    planned = total[0]
    if on_total:
        on_total(total[0], stage=None, planned=planned)

    with open(path, "a", encoding="utf-8") as fh:

        def run_group(cfgs, stage=None, extra_fills=None, search=None,
                      tries=None):
            """Measure a list of configs. Returns False if asked to stop.

            `stage` names the stage for the depth-extras hook (see _depth_extras);
            `extra_fills` is the list of deeper fills to re-measure the top stage-A
            rungs at, once the wall is known.

            `search` is a _WallSearch driving stage A. It is the opposite shape
            from `cfgs`: instead of a list decided in advance, it hands over one
            config at a time and is told what each one did. A probe whose config
            is already RECORDED never becomes a load - the stored status answers
            it, which is what makes a resumed campaign and a second round cost
            almost nothing, and what keeps the search on the same path it took
            the first time."""
            queue = list(cfgs)
            # The draft families already walking. Handed in when the caller read
            # recorded rows first (see _resume_stage), so a resumed stage picks
            # the walk up where the last run left it instead of starting over.
            tries = {} if tries is None else tries
            walls = {}
            extras_done = [False]
            i = -1
            while True:
                i += 1
                if i >= len(queue):
                    # The search picks its next probe from what the last one
                    # did, so it can only be asked once the queue has drained.
                    # A probe already on disk never becomes a load: the stored
                    # status answers it, which is what makes a resumed campaign
                    # and a second round nearly free and keeps the search on the
                    # path it took the first time.
                    if search is not None:
                        nxt = search.next()
                        while nxt is not None:
                            r = recorded.get(_key(name, nxt))
                            if r is None:
                                break
                            log(_fmt_row(nxt, r) + "   (recorded earlier)")
                            search.feed(r)
                            nxt = search.next()
                        if nxt is not None:
                            queue.append(nxt)
                            total[0] = len(out) + search.budget
                            if on_total:
                                on_total(total[0],
                                         planned=max(planned, total[0]))
                            # Step back one, because the top of the loop steps
                            # forward: without this `i` lands PAST the config
                            # just queued, the queue looks drained again, and
                            # the same unfed probe is appended forever.
                            i -= 1
                            continue
                    # Stage A found the wall and measured the fitting rungs. If
                    # deeper fills were asked for, the top rungs are re-measured
                    # at them now: the wall is free to reuse (the allocation is
                    # decided by ctx at load, not by fill), and rows at another
                    # fill are a separate experiment that cannot carry the
                    # chained baseline (comparable() gates on fill).
                    if (stage == "a" and extra_fills and not extras_done[0]
                            and limit is None):
                        extras_done[0] = True
                        # The axis stage A actually searched. Hardcoding
                        # ngl/ncmoe here was wrong under the ceiling plan,
                        # where ngl is PINNED at every block: every "rung" read the same
                        # value and the top three collapsed to one, so a
                        # --speed-fills campaign measured one depth row instead
                        # of three.
                        axis = stage_axis
                        direction = axis_direction(axis)
                        # `recorded` supplies the rungs of a RESUMED campaign:
                        # its stage-A rows live on disk, not in `out`.
                        ex = [c for c in _depth_extras(
                                out + list(recorded.values()), base, facts,
                                extra_fills, axis, direction)
                              if _key(name, c) not in done]
                        if ex:
                            log("depth   : re-measuring the top stage-A rungs at "
                                "fill%s %s - the wall is already known, so these "
                                "rows do not pay for it again"
                                % ("s" if len(extra_fills) > 1 else "",
                                   " ".join(str(f) for f in extra_fills)))
                            queue.extend(ex)
                            total[0] = len(out) + len(queue) - i - 1
                            if on_total:
                                on_total(total[0], planned=max(planned, total[0]))
                            continue
                    return True
                c = queue[i]
                # The wall tag rides the config to the queue and is stripped
                # before the row is measured: it is search bookkeeping, not a
                # setting, and it must never reach a row, a resume key or disk.
                walls[_key(name, c)] = c.pop("_wall", None) or ()
                if should_stop and should_stop():
                    log("stopped after %d of %d configs. The rows already written "
                        "are keyed, so re-running resumes here."
                        % (len(out), total[0]))
                    return False
                # A config in flight can also end because it is UNUSABLE, not
                # because anyone asked: a pass at a fraction of what this model
                # normally delivers means the process was spilled into shared
                # memory or the card is busy, and measuring it to the end is the
                # waste the floor exists to stop. The floor is this campaign's
                # own median so far (recorded rows included) scaled down - see
                # _abort_floor() - and None until there is something to judge
                # against.
                floor = _abort_floor(out + list(recorded.values()), c, name)
                row = bench_one(b, mp, c, port=port, timeout=timeout,
                                n_predict=n_predict, repeat=repeat, log=log,
                                on_server=on_server, skip_count=skip_count,
                                abort_floor=floor, should_abort=should_abort)
                row.update({"arch": facts["arch"], "n_layers": facts["n_layers"],
                            "when": int(time.time()), "gpu": gpu})
                # A hard stop kills the server mid-request, so this row failed
                # because it was ABANDONED, not because the config cannot run.
                # Writing it would record a fabricated wall - and worse, a
                # `genfail` here is indistinguishable on disk from a real one,
                # so the next campaign would carry the lie forward. Dropped, so
                # the config is simply re-measured.
                if should_abort and should_abort() and row.get("status") != "ok":
                    log("abandoned %s - not recorded, so it will be re-measured"
                        % _carry_summary(c))
                    return False
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                out.append(row)
                # Keep `done` current so a later round skips what this one just
                # measured, the same way a restarted campaign would. `skipped`
                # counts: the user judged the config and a chained round must
                # not re-propose it. `spilled` counts for the same reason: a
                # config that spilled into shared memory (or ran while the card
                # was busy) was judged by the machine, and re-measuring it just
                # wastes the load again.
                if row.get("status") in ("ok", "oom", "skipped", "spilled"):
                    done.add(_key(name, c))
                log(_fmt_row(c, row, len(out), total[0]))
                if on_row:
                    on_row(row)
                if search is not None:
                    search.feed(row)
                # A hard OOM is monotone evidence: it proves every pending config
                # in the same family that is worse on a tagged axis fails the same
                # way, so those loads would re-prove the wall instead of measuring
                # anything. A drafter load failure with no headroom for the
                # drafter's weights is the same evidence wearing a file's clothes -
                # _draft_no_room(): the base filled the card, which is a fact
                # about the split, not the file. Everything else says nothing
                # about the next rung: a genfail, exit or timeout, and `spilled`
                # means it LOADED (and ran badly), which is a measurement; a
                # loadfail WITH headroom is a fact about the file, and the walk's
                # own gate at _spec_retry() keeps it from being retried. The wall
                # row stays: it is the ladder's reading point. Pruned rows are
                # never recorded, so a resumed campaign after a card change
                # re-measures them naturally.
                if row.get("status") == "oom" or _draft_no_room(c, row, drafter):
                    kept, dropped = _prune_queue(queue, walls, i, name, c)
                    if dropped:
                        queue = kept
                        log("wall    : %s %s; %d later config%s provably worse, "
                            "skipped: %s"
                            % (_carry_summary(c),
                               "OOMs" if row.get("status") == "oom"
                               else "leaves no room for the drafter",
                               len(dropped),
                               "" if len(dropped) == 1 else "s",
                               ", ".join(_carry_summary(d) for d in dropped)))
                        total[0] = len(out) + (len(queue) - i - 1)
                        if on_total:
                            on_total(total[0], planned=max(planned, total[0]))
                # A draft family that OOMs walks one rung freer, repeatedly -
                # the walk is per family, with the depth ladder nested inside
                # each rung (see _spec_retry).
                #
                # This is the fix for a stage that could not succeed. Stages A
                # and B choose the fastest split that FITS, which is by
                # construction the one with the least headroom left. Stage D
                # then asks for a draft KV cache that llama.cpp keeps at f16
                # whatever -ctk says, unless the campaign pinned a draft quant
                # (-ctkd/-ctvd, --speed-spec-kv). So the speculative rows OOM,
                # and the campaign concludes speculation does not work on this
                # model.
                #
                # It concluded that twice this week and was wrong both times.
                # draft-mtp OOMed at ngl 31 and was the best config measured at
                # ngl 28 (3.95 vs 3.25); it OOMed at ncmoe 29 and was the best
                # config measured at ncmoe 34 (54.87 vs 47.17). A stage whose
                # design guarantees the answer "no" is not measuring anything.
                #
                # An OOM is cheap - it fails during load, before a single token
                # - so walking a few rungs costs far less than the finding is
                # worth. Bounded, because an OOM that is NOT about the draft
                # cache would otherwise walk the whole ladder.
                #
                # NOT in an explicit --speed-axes run. There the configs are the
                # instruction, not a starting point: the walk answers an OOM by
                # inventing rungs along an axis the ladder never mentioned, and
                # on a ladder that PINS that axis it silently overrides the pin.
                # Asked for eight draft depths at ngl 65, it spent five loads on
                # ngl 64, 63, 62, 61, 60 - none of them a config anyone asked
                # for. Same precedence rule as resolve_search(): the explicit
                # ladder is the more specific instruction and wins.
                if axes:
                    nxt = None
                elif (c.get("spec") or "none") in ("draft-mtp", "draft-dflash"):
                    nxt = _spec_retry(c, row, facts, tries, drafter=drafter,
                                      axis=stage_axis)
                else:
                    nxt = None
                # A probe whose config is already recorded answers from disk
                # and the walk carries straight on. Without this it STALLS
                # there: the rung is skipped as done, no probe is queued, and a
                # walk one rung short of a rung it has already measured simply
                # ends. Same rule the _WallSearch replay follows.
                while nxt is not None and _key(name, nxt) in done:
                    prev = recorded.get(_key(name, nxt))
                    if prev is None:
                        break        # measured this run; the row is in `out`
                    log(_fmt_row(nxt, prev) + "   (recorded earlier)")
                    nxt = _spec_retry(nxt, prev, facts, tries, drafter=drafter,
                                      axis=stage_axis)
                if nxt is not None and _key(name, nxt) not in done:
                    # The prefix used to be "the draft cache does not fit here"
                    # unconditionally, which is plainly false on the OK rows -
                    # where the cache fitted and the walk is simply carrying on
                    # to the next depth. Two situations, two sentences.
                    log("          ^ %s: %s"
                        % (("the drafter does not fit here, walking one rung "
                            "freer" if (row.get("status") == "oom"
                                        or _draft_no_room(c, row, drafter))
                            else "it fits, so the depth ladder continues here"),
                           _carry_summary(nxt)))
                    queue.append(nxt)
                    total[0] = len(out) + (len(queue) - i - 1)
                    if on_total:
                        on_total(total[0], planned=max(planned, total[0]))

        def promote(gb_, s_axis, s_obj, cur, cur_tok):
            """Fold a finished stage into the baseline. Returns (config, tok_s).

            Reads rows from DISK, not from `out`: a campaign resumed after a
            stop has rows this process never saw, and they are exactly the ones
            that say where the previous stage got to.

            `gb_`, NOT `base`. comparable() gates ctx, and under the dense
            ceiling plan nobody pinned one - the campaign opens at SPEED_BASE's context
            and stage A goes looking for the real wall. `base` still says the
            opening value afterwards; `gb_` is the baseline this stage actually
            ran at, so it carries what A found. Comparing measured rows against
            the opening guess made every later stage see zero candidates: stage
            A promoted only because swept= switches that gate off, and from the
            next stage on it is back on and rejects everything."""
            nxt, win = best_config(load_rows(path), name, gb_,
                                   n_predict=n_predict, repeat=repeat,
                                   incumbent_tok_s=cur_tok, prompt_id=pid,
                                   template_id=tmpl_id, swept=(s_axis or ""),
                                   axis=s_axis, objective=s_obj)
            if nxt is None:
                if s_obj == "extreme":
                    log("  baseline unchanged: nothing on the %s search loaded"
                        % s_axis)
                else:
                    log("  baseline unchanged: nothing beat %s by more than %.0f%%"
                        % ("%.2f tok/s" % cur_tok if cur_tok
                           else "any trustworthy row", 100 * CHAIN_MARGIN))
                return cur, cur_tok
            wc = win.get("config") or {}
            why = ("%s %s that loads"
                   % ("largest" if axis_direction(s_axis) == "up" else "smallest",
                      s_axis)
                   if s_obj == "extreme" else "fastest")
            log("  baseline -> %s   (%s: %s, %.2f tok/s, stage %s)"
                % (_carry_summary(nxt), why, wc.get(s_axis) if s_axis else "-",
                   win["tok_s"], wc.get("stage", "?")))
            return nxt, win["tok_s"]

        # Bound outside every branch so the verify step can read it. None means
        # "certify against the campaign's own baseline", which is what an
        # unchained campaign wants.
        carried = None
        if axes:
            # An explicit ladder REPLACES the staged search. resolve_search()
            # has already turned chaining off for it, so there is nothing to
            # promote between rungs and the whole thing is one group.
            stopped = not run_group(plan)
        else:
            # `carried` is the config the last stage promoted; `best_tok` is
            # what it did, so a stage-B challenger has to beat it by
            # CHAIN_MARGIN rather than by any amount at all.
            best_tok = None
            budget = limit
            seen = set()
            for rd in range(1, max(1, int(rounds or 1)) + 1):
                for letter in STAGES:
                    if letter not in stages:
                        continue
                    if budget is not None and budget <= 0:
                        break
                    gb, gnl, gmoe, gaxis, gvals = grid_context(
                        facts, mmproj=mmproj, base=base, model_path=mp,
                        carried=carried, drafter=drafter, plan_mode=plan_mode,
                        pin_ctx=(ctx is None), ffn_pin=ot)
                    tag = "stage %s" % letter.upper() + (
                        " round %d" % rd if (rounds or 1) > 1 else "")

                    # ---- stage A: two bisections, one after the other -------
                    if letter == "a":
                        if not gvals:
                            log("%s: no %s values to search" % (tag, gaxis))
                            continue
                        # Phase 1 always promotes, chained or not. That is not
                        # chaining - it is stage A knowing its own answer, and
                        # phase 2 has to run AT that answer or it measures a
                        # config the mode never proposed. What `chain` decides
                        # is whether stage A's result escapes to stage B, which
                        # is the same thing it has always decided.
                        srch = _WallSearch(gaxis, gvals, gb, stage="A",
                                           max_probes=budget)
                        log("%s: bisecting %s over %d value%s, %d probe%s at "
                            "most, at %s"
                            % (tag, gaxis, len(gvals),
                               "" if len(gvals) == 1 else "s", srch.budget,
                               "" if srch.budget == 1 else "s",
                               _carry_summary(gb)))
                        total[0] = len(out) + srch.budget
                        planned = max(planned, total[0])
                        if on_total:
                            on_total(total[0], stage=tag, planned=planned)
                        ok = run_group([], stage="a", search=srch,
                                       extra_fills=extra_fills)
                        log("  wall    : %s" % srch.summary())
                        if budget is not None:
                            budget -= len(srch.probed)
                        if not ok:
                            stopped = True
                            break
                        a_c, a_tok = promote(gb, gaxis, "extreme", carried, None)

                        # ---- phase 2: walk the dense FFN back on ------------
                        # Only when phase 1 TOPPED OUT: the largest value in the
                        # whole domain loaded, so the axis ran out of room to
                        # spend VRAM on before the card ran out of VRAM. That is
                        # what "there is some left" means, and it is measured
                        # rather than deduced from a plan.
                        #
                        # A winner in the MIDDLE of the domain means the wall is
                        # real and nothing is spare - walking the FFN back on
                        # there would just re-prove the same wall one knob over.
                        #
                        # Not on an MoE: --n-cpu-moe IS its dense-FFN knob, and
                        # phase 1 has just searched it.
                        # `ot is None` because phase 2's axis IS n_cpu_ffn: a
                        # hand-set pin is an answer to phase 2's question, so
                        # bisecting it would spend loads walking away from the
                        # split that was asked for.
                        if (srch.topped_out and not gmoe and a_c and ot is None
                                and (budget is None or budget > 0)):
                            gb2 = grid_context(
                                facts, mmproj=mmproj, base=base, model_path=mp,
                                carried=a_c, drafter=drafter,
                                plan_mode=plan_mode, pin_ctx=(ctx is None),
                                ffn_pin=ot)[0]
                            fvals = wall_values("n_cpu_ffn", gnl)
                            # gnl is known to fit - every phase-1 probe ran with
                            # the FFN fully exiled - so it seeds the search
                            # instead of being re-measured as its first probe.
                            f2 = _WallSearch("n_cpu_ffn", fvals, gb2, stage="A",
                                             known_fits=gnl, max_probes=budget)
                            if f2.budget:
                                log("%s: %s fits whole, so there is VRAM left - "
                                    "bisecting n_cpu_ffn over %d values, %d "
                                    "probe%s at most, at %s"
                                    % (tag, gaxis, len(fvals), f2.budget,
                                       "" if f2.budget == 1 else "s",
                                       _carry_summary(gb2)))
                                total[0] = len(out) + f2.budget
                                planned = max(planned, total[0])
                                if on_total:
                                    on_total(total[0], stage=tag, planned=planned)
                                ok = run_group([], stage=None, search=f2)
                                log("  wall    : %s" % f2.summary())
                                if budget is not None:
                                    budget -= len(f2.probed)
                                if not ok:
                                    stopped = True
                                    break
                                a_c, a_tok = promote(gb2, "n_cpu_ffn", "extreme",
                                                     a_c, a_tok)
                        if chain:
                            carried, best_tok = a_c, a_tok
                        continue

                    # ---- stage B: a list, ranked by tok/s -------------------
                    b_tries = {}
                    todo = _resume_stage(
                        _dedupe(stage_configs(letter, gb, gnl, gmoe, gaxis,
                                              gvals, mmproj, facts, drafter),
                                seen),
                        recorded, name, facts, b_tries, drafter, gaxis, log)
                    if budget is not None:
                        todo = todo[:budget]
                        budget -= len(todo)
                    if not todo:
                        log("%s: nothing new to measure at this baseline" % tag)
                        continue
                    log("%s: %d config%s at %s"
                        % (tag, len(todo), "" if len(todo) == 1 else "s",
                           _carry_summary(gb)))
                    total[0] = len(out) + len(todo)
                    # The campaign estimate was built before any stage ran, so a
                    # rebased stage can outgrow it. Never let it claim fewer
                    # configs than have already been measured.
                    planned = max(planned, total[0])
                    if on_total:
                        on_total(total[0], stage=tag, planned=planned)
                    if not run_group(todo, stage=letter, tries=b_tries):
                        stopped = True
                        break
                    _spec_wall_note(out, todo, gb, log, facts=facts)
                    s_axis, s_obj = stage_objective(letter, gaxis)
                    nxt_c, nxt_tok = promote(gb, s_axis, s_obj, carried, best_tok)
                    if chain:
                        carried, best_tok = nxt_c, nxt_tok
                if stopped or (budget is not None and budget <= 0):
                    break
    log("")
    log("%d rows -> %s" % (len(out), path))
    verified = None
    if verify and not dry_run and not stopped:
        # Cancellation lands between configs: the verify step is a config load,
        # and a campaign the user stopped mid-way must not start loading again.
        # Same reason as the promotion above: certify against the baseline the
        # campaign ENDED at, not the one it opened at. `carried` is None only
        # when nothing chained, and then `base` is still the right answer.
        verified = _verify_step(b, mp, facts, name, carried or base, path, pid, gpu,
                                overrides=verify_overrides, port=port,
                                timeout=timeout, n_predict=n_predict,
                                repeat=repeat, log=log,
                                template=(tmpl_f, tmpl_k, rea, rea_p), template_id=tmpl_id,
                                on_server=on_server, skip_count=skip_count,
                                swept=stage_axis)
    return {"rows": out, "path": path, "stopped": stopped, "planned": total[0],
            "chained": bool(chain), "verified": verified}


# The walk has NO rung budget. It goes until a rung fits or the axis runs out,
# because that is the only stopping rule that matches the question: "does
# speculation fit anywhere in this plan" has an answer, and a budget that stops
# short reports "no" while meaning "I did not look".
#
# It was five, then seven. Both were guesses sitting at the edge of the
# evidence, and seven was wrong on the first model that tested it:
# Muse-Glimmer-30B's dflash drafter fits at n_cpu_ffn 34, and the seven-rung
# walk stopped at 31 - three short, having spent every load it was allowed to
# conclude the opposite of the truth. No constant could have known: the drafter
# wants ~2 GiB against the 1556 MiB its file suggests, because a 202048-token
# vocabulary puts ~394 MiB of logits buffer on the card at ub 512.
#
# Affordable because a rung that OOMs fails during LOAD - 11 to 21 s on that
# model, not one generated token - so the whole dense-FFN axis is a couple of
# minutes and the context axis after it a few more. The loads that cost real
# time are the ones that FIT, and a rung that fits ends the walk.
#
# This walk is deliberately NOT a bisection, and it is the one search in the
# campaign that is not. Every other wall search is looking for a BOUNDARY and
# nothing else, so halving the range is pure saving. This one is looking for
# ROOM, and every rung it reaches is immediately useful: the depth ladder
# resumes there and measures real configs. A rung step costs one OOM, which
# fails during load in seconds rather than paying for a generation, so walking
# is nearly free - while a bisection would leap several rungs further than it
# had to and then measure the whole depth ladder at a context it never needed
# to give up. Fewer loads, worse answers.

# Under the dense ceiling plan the walk frees VRAM by giving back CONTEXT, and a
# context rung cannot be "one less" the way a layer can - the axis runs to six
# figures. Each rung gives back this share of the window, snapped to
# CTX_STEP so the rungs read as sizes a person would type.
#
# A tenth is chosen to match what the walk is buying: the draft cache is a few
# hundred MiB against a KV cache of a few thousand, so two or three rungs at a
# tenth each covers it - and since nothing cuts the walk short, the axis runs
# all the way down to CTX_STEP if that is what it takes. Coarser would overshoot - the walk would hand back gigabytes of
# context to buy back hundreds of MiB and call the result speculation working.
SPEC_RETRY_CTX_FRAC = 0.10


def _spec_ctx_rung(cur):
    """One rung freer on the context axis, or None at the floor."""
    cur = int(cur or 0)
    step = max(CTX_STEP,
               int(round(cur * SPEC_RETRY_CTX_FRAC / CTX_STEP)) * CTX_STEP)
    # Snapped DOWN, not rounded: the walk exists to free VRAM, so a rung must
    # never hand back less than the step it was asked for. The starting context
    # is the plan's own max_ctx and is not a round number, so snapping the value
    # (not just the step) is what puts the whole ladder on the 1024 grid.
    nxt = ((cur - step) // CTX_STEP) * CTX_STEP
    return nxt if nxt >= CTX_STEP else None


def _spec_axis(facts, plan_mode):
    """The three-way axis rule, for a caller that has a mode but no resolved axis.

    The walk has to move the knob the plan left free, or it measures a config the
    plan never proposed. Under the dense CEILING plan -ngl is pinned at every
    block - that pin IS the plan - so walking it down would trade the whole
    regime away to fit a draft cache, and report the result as speculation
    working here.

    speed_sweep() does NOT use this: it resolves the axis once from grid_context()
    and passes it down. Under --speed-mode auto the mode is whichever plan the
    PLANNER chose, so a second derivation from `plan_mode=None` would answer "ngl"
    while the ladder that ran swept ctx - and the walk would free a knob the
    campaign never varied."""
    if facts.get("is_moe"):
        return "ncmoe"
    return "ctx" if norm_plan_mode(plan_mode) == "ceiling" else "ngl"


def _draft_depths(spec, drafter=None):
    """The depth ladder one draft family walks, ascending.

    dflash depths come from the drafter's trained block size, which really is a
    block count: the drafter emits that many tokens per pass and asking for more
    is the same row twice.

    draft-mtp is the list STAGE_B_SPEC uses, and it is NOT capped at an external
    drafter's nextn_predict_layers. It used to be, on the theory that llama.cpp
    clamps a deeper request - and the measured rows say otherwise. On
    gemma-4-12B with an external MTP drafter, depths 1..15 gave fifteen
    different results with acceptance decaying smoothly (0.822 at depth 1,
    0.540 at 4, 0.215 at 15) and the best throughput at depth 4, 18% above
    depth 1. A clamp would have made all fifteen identical. --spec-draft-n-max
    is a draft RUN LENGTH, not a count of prediction heads.

    The cap was doing real damage: the Qwen3.8-27B drafter reports one nextn
    layer, so the ladder was [1], and stage D's wall walk - which had just spent
    two loads finding a context where the draft cache fits - stopped the moment
    depth 1 fitted, with the depth question entirely unmeasured."""
    if spec == "draft-dflash":
        bs = int((drafter or {}).get("block_size") or 0) or 16
        return list(range(1, bs + 1))
    return [n for s, n in STAGE_B_SPEC if s == "draft-mtp"]


def _freer_rung(axis, rung, n_layers):
    """One rung freer along `axis`, or None at the end of it.

    Which way is "freer" is not per-axis knowledge, it is the monotone direction
    the OOM pruning already uses: an axis tagged "up" costs more VRAM as it
    rises, so freeing means going down; "down" axes (n_cpu_moe, n_cpu_ffn) free
    VRAM by going up. Reading it from the same table is what keeps the walk and
    the pruning from ever disagreeing."""
    if axis == "ctx":
        return _spec_ctx_rung(rung)
    if axis_direction(axis) == "down":
        r = int(rung or 0) + 1
        return r if r <= n_layers else None
    r = int(rung or 0) - 1
    return r if r >= 1 else None


def _spec_give_back(c, facts, axis):
    """Which knob the wall walk spends FIRST, ahead of stage A's own axis.

    Stage A phase 2 spends whatever VRAM phase 1 left over on pulling dense FFN
    blocks back onto the card, so wherever it ran, those blocks ARE the
    headroom. They are also the cheapest thing in the config to give back: they
    were bought with the leftover, and buying them was worth a few per cent.
    The context is not cheap - under the ceiling plan it is what the campaign went
    looking for, the number the whole thing reports.

    Measured, Muse-Glimmer-30B at ngl 52 on a 5070 Ti Laptop with 11.5 GiB free.
    One FFN block is ~214 MiB. One context rung at 131072 against a q8_0 cache
    is ~66 MiB. The campaign walked context down all seven rungs - 131072 to
    62464, less than half the window - and freed 463 MiB, which is less than the
    whole KV cache and nowhere near the 1556 MiB dflash drafter it was trying to
    fit. Three FFN blocks would have freed more than the entire context walk
    could; five would have fitted the drafter. The walk could not have succeeded
    on the axis it was walking, and it gave back the answer to fail.

    So this is not a preference between two levers, it is the difference between
    a walk that can find room and one that provably cannot: the context axis
    caps out at the size of the KV cache, and a drafter is usually bigger.

    MoE models are not covered - --n-cpu-moe IS their FFN knob and is already
    the axis, so the rule would only ever name what the caller passed."""
    nl = int(facts.get("n_layers") or 0)
    ffn = c.get("n_cpu_ffn")
    if not facts.get("is_moe") and nl and ffn is not None and int(ffn) < nl:
        return "n_cpu_ffn"
    return axis


def _draft_probe(c, axis, rung, depth):
    """One probe of a draft family's wall walk: one rung freer, one depth."""
    d = dict(c)
    d.pop("_wall", None)
    d["spec_n_max"] = depth
    d[axis] = rung
    return d


_DRAFTER_DET = {}


def _drafter_floor_mib(drafter, md, c=None):
    """The VRAM a card must leave a drafter before its load can even try, in MiB.

    Weights alone proved too low a bar: gemma's MTP head carries 252 MiB of
    tensors yet needs ~1.5 GiB at a 131k context, because the draft keeps
    full-context state and reserves a compute buffer that grows with it -
    measured 88 MiB + 9216 B per cell on the burst that taught this, against
    the drafter's own kv_layer_dims f16 accounting of 8192 B/cell. The floor
    is therefore weights + those cells priced from the file's geometry at this
    config's context and draft-cache quant + a reservation constant. A dflash
    drafter has no such table and holds its cache at block depth regardless,
    so it floors at its weights - which is what caught the Muse burst.

    Taken from the campaign's drafter detail when it carries the numbers, else
    read once from the file the config points at and cached by path. None when
    the file cannot be read, which reads as "cannot tell" and the row stays a
    plain loadfail rather than being waved through the walk on a guess."""
    det = drafter if isinstance(drafter, dict) and drafter.get("cfg") else None
    v = (drafter or {}).get("tensor_bytes")
    if md:
        try:
            key = os.path.normcase(os.path.abspath(md))
        except TypeError:
            return None
        if key not in _DRAFTER_DET:
            try:
                from .plan import load_drafter   # deferred: plan is above bench
                _DRAFTER_DET[key] = load_drafter(md)
            except Exception:
                _DRAFTER_DET[key] = None
        d = _DRAFTER_DET[key]
        if d is not None:
            # the file's own geometry is the truth; a campaign detail only
            # fills in when the file cannot be read
            det = d
            if v is None:
                v = d.get("tensor_bytes")
    elif det is None:
        return None
    if v is None:
        return None
    floor = v / (1024.0 * 1024.0)
    dims = ((det or {}).get("cfg") or {}).get("kv_layer_dims") or []
    ctx = (c or {}).get("ctx")
    if dims and ctx:
        try:
            from .kv import KV_TYPE_BYTES   # deferred: plan/kv sit above bench
            elems = sum(float(kd) + float(vd) for kd, vd, _ in dims)
            bpe = KV_TYPE_BYTES.get(c.get("spec_kv") or "f16", 2.0) / 2.0
            floor += elems * 2.0 * bpe * float(ctx) / (1024.0 * 1024.0)
        except Exception:
            pass
    from .compute import MTP_SPEC_CONST_MIB
    return floor + MTP_SPEC_CONST_MIB


def _draft_no_room(c, row, drafter):
    """A drafter load failure that is a lack of room wearing the file's clothes.

    When the target model has squeezed the draft out, llama.cpp dies while
    loading it with the same "error loading model: invalid vector subscript"
    a broken file produces (ggml-org/llama.cpp#27454: a device split computed
    from memory that is not there), so the log line cannot tell the two apart.
    The row can: the base model's own buffers are in the parsed log, the free
    memory it started with is on the row, and the drafter's floor is a number.
    If the headroom the base left is under that floor - weights plus the
    full-context state and compute reservation a draft context needs, which
    the gemma burst measured at ~6x its weight bytes - the file is innocent:
    nothing it contains could have loaded. That row is the OOM the walk exists
    for. (A clean allocation failure inside the draft's context creation never
    gets here: RE_OOM stamps it oom and the ordinary OOM walk takes it.)

    The reverse stays loadfail: headroom to spare, drafter still refused - the
    broken MTP head the walk must not re-refuse at seven progressively freer
    splits. Missing data (no free reading, no buffers, unreadable drafter)
    also stays loadfail: the walk walks on numbers, not on sympathy."""
    if row.get("status") != "loadfail":
        return False
    lg = row.get("log") or {}
    if not lg.get("load_error"):
        return False
    md = (c.get("md") or "").strip()
    if not md:
        return False
    free = row.get("gpu_free_before_mib")
    if not free:
        return False
    headroom = free - gpu_side(lg.get("buffers") or {})[0]
    if headroom <= 0:
        return True
    floor = _drafter_floor_mib(drafter, md, c)
    return floor is not None and headroom < floor


# A generation whose SERVER died mid-request: urllib surfaces a dead peer as
# a connection-level error, while a server still standing answers with an
# HTTPError carrying its status. The distinction is the walk's: a crashed
# process is plausibly the card running out (Windows aborts the process on a
# failed allocation instead of printing one), an answered error is not.
_RE_GENDIED = re.compile(
    r"ConnectionReset|RemoteDisconnected|IncompleteRead|WinError 10054", re.I)


def _gen_died(row):
    """True when a genfail row's request ended because the process went away."""
    if row.get("status") != "genfail":
        return False
    err = row.get("gen_error") or ""
    return bool(_RE_GENDIED.search(err))


def _spec_retry(c, row, facts, tries, drafter=None, axis=None):
    """The next config a draft family's wall walk measures, or None.

    A draft model needs its own KV cache, and llama.cpp keeps it at f16 unless
    the campaign pinned a draft quant (-ctkd/-ctvd, --speed-spec-kv) - so
    speculation costs several hundred MiB the split it is being tried at was
    never chosen to leave room for. Stage D only
    ever tries it at the split stage A/B settled on - by construction the fastest
    one that fits, usually the one with the least headroom - so the draft rows
    OOM and the campaign reads "speculation does not work" when what it measured
    was "speculation does not fit at this particular split".

    A drafter load failure is walked the same way when the row proves there was
    no room for it - see _draft_no_room. llama.cpp's draft loader cannot report
    a full card as an OOM (ggml-org/llama.cpp#27454): it dies in load_tensors
    with the file-loader's error, and the row reads loadfail. The walk is free
    to treat such a row as the OOM it is, provided the headroom the base left is
    less than the drafter's weights; a drafter refused DESPITE headroom is a
    broken file and stays loadfail, because the walk exists to free VRAM, not to
    re-ask a file for a second chance.

    The walk settles which it is. It belongs to the SPEC FAMILY, not to one
    (spec, depth) pair: one walk per draft scheme, rung by rung in the direction
    that frees VRAM.

    WHICH knob it gives back is _spec_give_back(): the dense FFN blocks stage A
    phase 2 pulled onto the card, wherever it pulled any, because those are the
    leftover VRAM in a form that can be handed straight back. Only when every
    block is exiled again does the walk fall through to stage A's own axis -
    higher n_cpu_moe on an MoE, lower ngl for a dense fit plan, lower
    CONTEXT for a dense ceiling plan, whose -ngl is pinned at every block and is
    not the campaign's to spend.

    The depth ladder is nested inside each rung, ascending:

      * the first depth that OOMs at a rung proves every deeper one does too
        (the draft cache grows with depth), so the ladder restarts one rung
        freer at the depth that just failed - the depths below it fit at a
        stricter rung and are dominated there;
      * a rung where a depth FITS continues the ladder at that same rung until
        it fails to run - an OOM, or a genfail whose process died generating -
        or the ladder is exhausted - the rungs below are dominated,
        because they carry less on the GPU (fewer layers, or less context) and
        nothing fit there that did not fit at the richer rung.

    That makes the walk cheap where the old per-depth walks were wasteful: three
    dflash depths used to re-prove the same wall three times over, once each."""
    # Only the two draft schemes walk. The n-gram speculators build their drafts
    # from the context that is already there and allocate no second cache, so an
    # OOM from one of them is about the model, not about speculation, and walking
    # would only prove it more slowly. This is an exact allow-list, not a prefix
    # match: "draft-gram-l2" is llama.cpp's CLI spelling and must not be walked
    # as though it were draft-mtp.
    spec = (c.get("spec") or "none")
    if spec not in ("draft-mtp", "draft-dflash"):
        return None
    # A drafter load failure with no headroom for the drafter's weights is the
    # OOM that #27454 misreports - the walk treats it as one (see
    # _draft_no_room). A row that LOADED and then lost its connection is the
    # same shortage arriving later: Windows builds abort the whole process on
    # an allocator that comes up short while running, the request dies with a
    # connection reset, and freeing VRAM is the plausible cure. A genfail from
    # a server still standing - it answered with an error status - is about
    # the request, not the card, and stays put; so does any other scheme.
    no_room = _draft_no_room(c, row, drafter)
    died = _gen_died(row)
    if row.get("status") not in ("ok", "oom") and not (no_room or died):
        return None
    depths = _draft_depths(spec, drafter)
    if not depths:
        return None
    is_moe = bool(facts.get("is_moe"))
    nl = facts.get("n_layers") or 0
    # The campaign resolves stage A's axis once and hands it down. Absent one -
    # a direct caller, or a test - fall back to the mode-free reading.
    axis = axis or _spec_axis(facts, None)
    st = tries.get(spec)
    if st is None:
        # The first OOM - or a genfail whose process died running the config -
        # starts the walk one rung freer, resuming the ladder at the depth
        # that just failed.
        if row.get("status") != "oom" and not (no_room or died):
            return None          # the plan rows cover the baseline rung's ladder
        d = int(c.get("spec_n_max") or 0)
        idx = min(range(len(depths)), key=lambda i: abs(depths[i] - d))
        give = _spec_give_back(c, facts, axis)
        st = {"axis": give, "rung": int(c.get(give) or 0), "idx": idx}
        tries[spec] = st
    if row.get("status") == "ok":
        # Continue the depth ladder at the rung that fits. Exhausted means the
        # whole rung fits and the walk is over - nothing is left to find.
        if st["idx"] + 1 >= len(depths):
            return None
        st["idx"] += 1
        return _draft_probe(c, st["axis"], st["rung"], depths[st["idx"]])
    # Out of room - an OOM, or a genfail whose process died running the depth
    # it loaded: walk one rung freer and resume the ladder at the depth that
    # failed.
    r = _freer_rung(st["axis"], st["rung"], nl)
    if r is None and st["axis"] != axis:
        # Every FFN block is back on the CPU and the cache still does not fit.
        # What is left is exactly the config phase 1 promoted, so the walk
        # carries on down phase 1's own axis from there - the rungs it would
        # have walked had phase 2 never run. Nothing is lost by trying the FFN
        # first: this is where the old walk STARTED.
        #
        # Nothing bounds the pair except the axes themselves: n_cpu_ffn ends
        # at every block exiled, context ends at CTX_STEP. Between them that is
        # a finite walk of cheap OOMs, and the end of it is a real answer -
        # "not at any split this plan can reach" - rather than a budget
        # running out.
        st["axis"] = axis
        st["rung"] = int(c.get(axis) or 0)
        r = _freer_rung(axis, st["rung"], nl)
    if r is None:
        return None
    st["rung"] = r
    return _draft_probe(c, st["axis"], r, depths[st["idx"]])


def _spec_wall_note(out, todo, base, log, facts=None):
    """Say what an all-OOM speculation stage actually means.

    A draft model needs its own KV cache, and llama.cpp keeps it at f16 unless
    the campaign pinned a draft quant (-ctkd/-ctvd, --speed-spec-kv) - so
    speculation costs several hundred MiB that the split it is being tried at
    was never chosen to leave room for.

    Stage D only ever tries it at the split stage A/B settled on, and that split
    is by construction the FASTEST one that fits, which usually means the one
    with the least headroom left. So every draft row OOMs, and the campaign
    reads as "speculation does not work on this model" when what it measured is
    "speculation does not fit at this particular split". Those are different
    findings and only one of them is true.

    The remedy is an interaction the staged grid cannot express - vary the split
    AND the speculation together - so this prints the ladder to run rather than
    leaving it to be deduced from four OOM lines."""
    keys = {(c.get("spec") or "none") for c in todo}
    drafts = [c for c in todo if (c.get("spec") or "none").startswith("draft")]
    if not drafts or len(keys) < 2:
        return
    ran = [r for r in out if (r.get("config") or {}).get("spec", "").startswith("draft")]
    if not ran or any(r.get("status") == "ok" for r in ran):
        return
    moe = base.get("ncmoe") is not None and base.get("ncmoe") != 0
    # The ladder to suggest is the one the WALK would have spent, or the advice
    # contradicts the eight OOM lines printed directly above it. On a dense
    # config carrying FFN blocks on the card, that is -ot: under the ceiling
    # plan -ngl is pinned at every block, so an ngl ladder there proposes
    # leaving the plan rather than making room inside it.
    give = _spec_give_back(base, facts or {}, "ncmoe" if moe else "ngl")
    axis = give if give == "n_cpu_ffn" else ("ncmoe" if moe else "ngl")
    cur = base.get(axis)
    if cur is None:
        return
    # More ncmoe or n_cpu_ffn means MORE on the CPU and less in VRAM; more ngl
    # means the opposite. Either way, walk in the direction that frees memory.
    up = axis_direction(axis) == "down"
    rungs = [cur + i for i in range(1, 5)] if up else \
            [cur - i for i in range(1, 5) if cur - i >= 0]
    log("")
    log("note    : every draft-* row OOMed at %s %s. That is not a verdict on"
        % (axis, cur))
    log("          speculation - the draft KV cache is %s whatever -ctk says, and"
        % (base.get("spec_kv") or "f16"))
    log("          this split was picked as the fastest that FITS, so it had no room")
    log("          spare. Vary the split and the speculation together:")
    log("            --speed-axes %s=%s spec=draft-mtp spec_n_max=2%s"
        % (axis, ",".join(str(v) for v in sorted(rungs)),
           " mmproj_offload=false" if base.get("mmproj_offload") is False else ""))
    log("          or paste that into 'Sweep exact values instead of the stages'.")
    log("          (A q8_0 draft cache --speed-spec-kv q8_0 also halves what")
    log("          speculation costs, if the acceptance rate survives it.)")
    log("")


def _fmt_row(c, row, i=None, n=None):
    # `ctx` is in the table because under the dense ceiling plan it is stage A's
    # AXIS - the one number the whole campaign is looking for. Without it every
    # probe of that search printed an identical line, and the ladder read as the
    # same config measured seven times.
    head = ("%-2s %-7d %-4d %-5d %-4d %-5d %-8d %-13s %-4d %-6s | "
            % (c.get("stage", "-"), c.get("ctx") or 0, c["ngl"],
               c.get("ncmoe") or 0,
               c.get("n_cpu_ffn") or 0, c["ub"], c.get("fill") or 0,
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
        # _degenerate_note carries the spill line now, with the megabytes and
        # the evidence behind them, so a bare "SPILLED" here would repeat it
        # while saying strictly less.
        ("  [filler repeats - speculative rate inflated]"
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
    raw = ("  RAW - no chat template on this build, so the model was asked "
           "nothing and merely continued the text"
           if row.get("templated") is False else "")
    if not (loop or copy):
        return spill_note(row) + raw
    if copy:
        what = "COPYING - %.0f%% of the output is a verbatim copy of the prompt, so" \
               % (100 * cb)
    else:
        what = "LOOPING - output is %.0f%% repetition, so" % (100 * (1 - dr))
    if spec and (row.get("accept_rate") or 0) >= 0.95:
        return ("  %s the %.0f%% acceptance is the %s, not the drafter; "
                "excluded from conclusions%s%s"
                % (what, 100 * row["accept_rate"], "copy" if copy else "loop",
                   spill_note(row), raw))
    return "  %s excluded from conclusions%s%s" % (what, spill_note(row), raw)


def spill_note(row):
    """Say WHICH kind of spill, and on what evidence.

    A measured reading and an inference from the campaign's shape are both worth
    printing, but they are not the same claim, and a row that says "SPILLED"
    without saying why invites the reader to trust a deduction as a
    measurement."""
    x = row.get("shared_excess")
    if x is not None and x > SHARED_SPILL_MIB:
        return ("  SPILLED - %.0f MiB more host memory than the rest of this "
                "ladder, so every token that touches it crosses PCIe" % x)
    if row.get("spill_inferred"):
        return ("  SPILLED - floor fell %.0f MiB below the rest of this campaign, "
                "which is memory the driver moved out of VRAM"
                % row["spill_inferred"])
    if row.get("collapse_inferred"):
        return ("  SPILLED - %.2fx the speed of this campaign's own median, which "
                "is not a slower config: the card was busy or the process was "
                "demoted mid-run" % row["collapse_inferred"])
    sh = row.get("shared_mib")
    if x is None and sh is not None and sh > SHARED_SPILL_MIB:
        # Mid-campaign there is no ladder yet, so the counter has no baseline to
        # be excess OVER. Report the number as a fact and pass no verdict: every
        # llama.cpp process carries hundreds of MiB here by design, and calling
        # that a spill is the mistake this whole function was rewritten to stop
        # making. It resolves into a verdict once the group exists.
        return ("  host memory %.0f MiB (no ladder yet to compare it against)" % sh)
    return ""


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
    _infer_demotion(rows)
    return rows


def campaign_match(row, model=None, gpu=None, file=None, prompt_id=_ANY,
                   template_id=_ANY):
    """Is this row part of the named campaign?

    The same five keys sweep_index() groups on, so "delete what that line is
    showing me" removes exactly the rows behind that line and nothing else.

    prompt_id and template_id use the _ANY sentinel rather than None because ""
    is a MEANING here and not an absence: a campaign that pinned no template is
    identified BY its empty template_id, and treating absent-as-any would widen
    a delete from one campaign to every campaign of that model. Same reasoning
    as the insights query, and the stakes are higher on this side.
    """
    if model is not None and (row.get("model") or "?") != model:
        return False
    if gpu is not None and (row.get("gpu") or "") != gpu:
        return False
    if file is not None and (row.get("_file") or "") != file:
        return False
    if prompt_id is not _ANY and (row.get("prompt_id") or "") != (prompt_id or ""):
        return False
    if template_id is not _ANY and (row.get("template_id") or "") != (template_id or ""):
        return False
    return True


def deleted_dir():
    return os.path.join(bench_dir(), "deleted")


def delete_campaign(model=None, gpu=None, file=None, prompt_id=_ANY,
                    template_id=_ANY, backup=True):
    """Remove one campaign's rows from the store. Returns a report dict.

    Three things this is careful about, because hours of GPU time are on the
    other end of it:

      * It NEVER deletes the file. A .jsonl is one GPU and one llama.cpp build,
        so it holds every campaign ever measured on that pair - unlinking it to
        remove one model's rows would take the rest with it.
      * The removed rows are written to speed/deleted/ first, so the operation
        is recoverable by moving one file back. A campaign is two hours of
        measurement; a confirm dialog is not enough protection on its own.
      * The rewrite is atomic - full file to a temp beside it, then replace -
        so an interrupted delete cannot leave a half-written store. A truncated
        JSONL loses far more than the campaign that was being removed.

    A caller that can see a running job should refuse before reaching here (the
    web server does, against its own JOB). But it can only see ITS OWN process:
    a --forget-sweep in a terminal knows nothing about a campaign appending from
    a browser, and neither knows about the other. So the last word is here - the
    file is fingerprinted before it is read and again before it is replaced, and
    a delete that would land on top of an append is abandoned instead. That
    covers every writer, including the ones no guard could have known about.
    """
    if not file:
        return {"ok": False, "error": "no campaign file named"}
    path = os.path.join(bench_dir(), os.path.basename(file))
    if not os.path.isfile(path):
        return {"ok": False, "error": "no such campaign file: %s" % file}
    before = _fingerprint(path)

    keep, drop = [], []
    for line in _read_lines(path):
        try:
            row = json.loads(line)
        except ValueError:
            keep.append(line)          # unparseable, but not ours to discard
            continue
        # campaign_match reads _file, which is stamped by load_speed_rows() and
        # is not in the line itself.
        row["_file"] = os.path.basename(file)
        (drop if campaign_match(row, model, gpu, file, prompt_id,
                                template_id) else keep).append(line)

    if not drop:
        return {"ok": False, "error": "no rows matched that campaign",
                "removed": 0, "kept": len(keep)}

    saved = None
    if backup:
        body = "".join(l if l.endswith("\n") else l + "\n" for l in drop)
        stem = re.sub(r"\.jsonl$", "", os.path.basename(file))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            os.makedirs(deleted_dir(), exist_ok=True)
            # The stamp is only second-granular, and one file holds many
            # campaigns - forgetting two of them in the same second put the
            # second backup on top of the first and took an hour of measurement
            # with it. Opened "x" rather than checked-then-written, so the name
            # is claimed by the same call that finds it free.
            for n in range(200):
                saved = os.path.join(deleted_dir(), "%s.%s%s.jsonl"
                                     % (stem, stamp, "" if not n else "-%d" % n))
                try:
                    f = open(saved, "x", encoding="utf-8", newline="\n")
                except FileExistsError:
                    continue
                with f:
                    f.write(body)
                break
            else:
                raise OSError("200 backups already stamped %s" % stamp)
        except OSError as e:
            # A delete that cannot be undone is a different operation from the
            # one that was asked for, so it does not happen by accident.
            return {"ok": False, "error": "could not write the backup, so nothing "
                                          "was deleted: %s" % e}

    # Last look before the file is replaced. Anything that landed since the read
    # is a row this rewrite does not contain, so replacing now would delete a
    # measurement nobody asked to forget.
    if _fingerprint(path) != before:
        _unlink(saved)
        return {"ok": False, "error": "%s changed while it was being read - a "
                                      "campaign is probably still measuring into "
                                      "it. Nothing was deleted; stop the campaign "
                                      "and try again." % os.path.basename(file)}

    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(l if l.endswith("\n") else l + "\n" for l in keep))
        os.replace(tmp, path)
    except OSError as e:
        _unlink(tmp)
        _unlink(saved)
        return {"ok": False, "error": "could not rewrite %s: %s" % (file, e)}
    return {"ok": True, "removed": len(drop), "kept": len(keep),
            "file": os.path.basename(file), "backup": saved}


def _read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [l for l in f if l.strip()]
    except OSError:
        return []


def _fingerprint(path):
    """Enough of a file's identity to notice an append under a rewrite.

    Size alone catches every append a campaign makes; mtime catches an in-place
    edit that happened to keep the length. A missing file fingerprints as None,
    which compares unequal to any real reading - the right answer, since a file
    that vanished mid-delete is also not one to replace.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _unlink(path):
    """Remove a file we wrote ourselves, if it is still there."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


# How far a row's floor must fall below its own campaign's before the gap is a
# demotion rather than allocator noise. The observed collapse was 246 MiB
# against neighbours that agreed within 12 MiB of each other, so this sits well
# clear of the noise and well under the signal.
FLOOR_DROP_MIB = 100.0


def _infer_demotion(rows):
    """Mark pre-counter rows whose floor collapsed against their own campaign.

    `floor` is the CUDA context plus whatever the allocator holds that llama.cpp
    does not report, and within one campaign it is nearly constant - it moved by
    12 MiB across five rungs of the ngl ladder. When the process hits the
    driver's dedicated-memory budget, the next rung cannot grow: alloc_gpu rises
    by a layer while proc_vram does NOT, and the difference comes out of floor.
    That collapse is the demotion, visible without any counter.

    The counter reading gets the SAME treatment, for the reason set out in
    demoted(): `Shared Usage` counts deliberate host placement - pinned staging
    buffers, `--no-mmproj-offload` - as well as demotion, so its absolute value
    says nothing on its own. Only its excess over the ladder's own baseline
    does. Two signals, one grouping, because both mean "this rung is carrying
    something the others are not".

    What the grouping has to hold constant is everything that moves the floor
    for a legitimate reason, and speculation is the big one: llama.cpp does not
    report the draft KV cache in alloc_gpu, so it lands in floor and puts a
    draft-mtp row near 1050 MiB where its non-speculative twin sits at 230. Mix
    them and the median lands between, and EVERY ordinary row reads as a 380 MiB
    collapse - which is exactly what the first version of this did. The
    projector placement is the same story at ~1100 MiB, and the draft depth
    scales the cache, so spec_n_max belongs here too.

    prompt_id is in the key for the same reason it is in comparable(): a median
    has to come from ONE campaign. Without it a fill-32768 group merged rows
    from three separate runs - pre-freeze, the raw-continuation campaign, and a
    smoke test - and took its baseline from all of them. floor is a memory fact
    rather than a prompt fact, so it survives that better than tok/s would, but
    "better" is not the standard the rest of this module holds. Note this still
    does not isolate two campaigns that share a prompt_id on different days;
    driver state can move the floor between them.

    In memory only - nothing is written back to the store."""
    groups = {}
    for r in rows:
        if r.get("status") != "ok" or r.get("floor_mib") is None:
            continue
        groups.setdefault(_demotion_group(r), []).append(r)
    def _median(vals):
        v = sorted(vals)
        return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1]
                                                  + v[len(v) // 2]) / 2.0

    for grp in groups.values():
        if len(grp) < 3:            # two rows cannot say which one is anomalous
            continue
        # The counter, made relative. A row that reads the SAME as its ladder is
        # carrying the baseline every rung carries and has been demoted by
        # nothing; only the excess is the driver moving something. Recorded as 0
        # rather than dropped, because "measured, and it was normal" is a
        # different statement from "not measured".
        sh = [x["shared_mib"] for x in grp if x.get("shared_mib") is not None]
        if len(sh) >= 3:
            base_sh = _median(sh)
            for r in grp:
                if r.get("shared_mib") is not None:
                    r["shared_excess"] = round(
                        max(0.0, r["shared_mib"] - base_sh), 1)
        # ...and the floor collapse, which is the only signal a row recorded
        # before the counter existed can offer.
        mid = _median([x["floor_mib"] for x in grp])
        for r in grp:
            if r.get("shared_mib") is None and mid - r["floor_mib"] > FLOOR_DROP_MIB:
                # Annotated, deliberately NOT gated. Two reasons, and the second
                # one is the important one.
                #
                # It is the weaker evidence: a deduction from three floors, not
                # a reading. And it can be right about the memory while being
                # wrong about the row - the ngl 26 draft-mtp row here shows the
                # exact collapse signature and is still the FASTEST row in its
                # group, because one more layer on the GPU bought more than the
                # displaced 240 MiB cost. Gating it would have thrown away the
                # best config in the campaign.
                #
                # More generally, a ladder that slows down at its top rung is
                # the wall being FOUND. Excluding those rows would hide the very
                # thing an ngl sweep exists to locate.
                r["spill_inferred"] = round(mid - r["floor_mib"], 1)

    # Every row that carries a counter reading gets its verdict re-decided, and
    # the ones OUTSIDE any usable group matter most. Rows written by the first
    # version of demoted() carry a stored `spilled: true` that came from the raw
    # counter clearing 64 MiB - which every llama.cpp process does - and a lone
    # measured row has no ladder to be excess over, so nothing in the loop above
    # would ever reach it. It would keep a verdict from a detector that no
    # longer exists, permanently outside trustworthy(). That is what happened to
    # the ngl 31 reference row: one measured row in its group, still flagged.
    #
    # demoted() answers from shared_excess, which is None here, so the verdict
    # falls back to suspect_reason() alone - the one test a single row supports.
    for r in rows:
        if r.get("status") == "ok" and r.get("shared_mib") is not None:
            r["spilled"] = bool(unsound_reason(r)) or demoted(r)

    # The RATE is the third signal, and it catches what the two memory signals
    # cannot: a process demoted by a BUSY machine shows no excess shared usage
    # and no floor collapse - the counters never move, the speed does. A row at
    # a fraction of its group's median tok/s is not a slower config - the wall
    # being found moves a ladder by tens of percent, never by 6x (SLOW_FRAC) -
    # it is a measurement of a card that stopped being usable, and left
    # unflagged it would read as a trustworthy number and sit in the corpus
    # forever. Flagged the same way a spill is: excluded from every conclusion,
    # but still a row in the table. Runs AFTER the loops above so a verdict
    # from a memory signal is never overwritten; the groups are the same ones
    # the memory signals grouped on.
    for grp in groups.values():
        ts = [x["tok_s"] for x in grp if x.get("tok_s")]
        if len(ts) < 3:
            continue
        mid = _median(ts)
        for r in grp:
            if r.get("tok_s") and r["tok_s"] < mid * SLOW_FRAC:
                r["spilled"] = True
                r["collapse_inferred"] = round(r["tok_s"] / mid, 3)


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
_CONFIG_KEYS = ("ctx", "kv", "fa", "seq", "ub", "ngl", "ncmoe", "n_cpu_ffn",
                "spec", "spec_n_max", "spec_kv", "mmproj_offload", "fill",
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
    ("n_cpu_ffn", "CPU FFN blocks", ("n_cpu_ffn",), 0),
    ("ub", "ubatch", ("ub",), 512),
    ("spec", "speculation", ("spec", "spec_n_max"), "none"),
    ("projector", "projector", ("mmproj_offload",), "VRAM"),
    ("kv", "KV quant", ("kv",), "f16"),
    # The draft cache's own quant is an axis like the target's: an explicit
    # ladder over it measures the acceptance-rate price of a smaller cache, and
    # an insight reports what q8_0 bought relative to the f16 reference.
    ("spec_kv", "draft KV quant", ("spec_kv",), "f16"),
)


def _axis_value(r, axis):
    c = r.get("config") or {}
    if axis == "spec":
        s = c.get("spec") or "none"
        n = c.get("spec_n_max") or 0
        return "%s/%d" % (s, n) if s != "none" else "none"
    if axis == "projector":
        return "RAM" if c.get("mmproj_offload") is False else "VRAM"
    if axis in ("ncmoe", "fill", "n_cpu_ffn"):
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
           ("repeat", r.get("repeat")), ("prompt_id", r.get("prompt_id")),
           # What the model was ASKED is held constant too. A thinking template
           # spends tokens reasoning before it answers, so a row measured under
           # one is not a faster or slower version of a row measured without -
           # it is a different question.
           ("template_id", r.get("template_id"))]
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
            "shared_mib": r.get("shared_mib"),
            "shared_excess": r.get("shared_excess"),
            "spill_inferred": r.get("spill_inferred"),
            "templated": r.get("templated"),
            "prefill_warm": r.get("prefill_warm"),
            "template_id": r.get("template_id"),
            "chat_template": r.get("chat_template"),
            "template_kwargs": r.get("template_kwargs"),
            "distinct_ratio": r.get("distinct_ratio"),
            "copyback_ratio": r.get("copyback_ratio"),
            "prompt_id": r.get("prompt_id"), "when": r.get("when"),
            "n_predict": r.get("n_predict"), "repeat": r.get("repeat"),
            "status": r.get("status")}


def sweep_index(rows):
    """One entry per EXPERIMENT - the browsable list of campaigns.

    Split by build as well as by card, because a llama.cpp version bump moves
    these numbers and merging two builds into one campaign would hide that.

    And split by prompt_id and template_id, for the reason the rest of this
    module already splits on them: they are what the model was ASKED. Keying on
    (model, gpu, build) alone collapsed 145 rows spanning three prompts and two
    templates into a single line, headlined by whichever prompt_id happened to
    come first and a best_tok_s taken from a 2k-fill row of an experiment nobody
    was looking at. The campaign run that morning was inside it and could not be
    found - which is the whole job of a browsable index.

    The tradeoff is that a campaign whose corpus was refreshed mid-run now shows
    as two entries. That is the honest shape: it WAS two experiments."""
    groups = {}
    for r in rows:
        k = (r.get("model") or "?", r.get("gpu") or "", r.get("_file") or "",
             r.get("prompt_id") or "", r.get("template_id") or "")
        groups.setdefault(k, []).append(r)
    out = []
    for (model, gpu, f, pid, tid), rs in groups.items():
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
            # Now a property of the GROUP rather than of whichever row came
            # first, because the group is keyed on it.
            "prompt_id": pid or None,
            "template_id": tid or None,
            "chat_template": next((r.get("chat_template") for r in rs
                                   if r.get("chat_template")), None),
            "template_kwargs": next((r.get("template_kwargs") for r in rs
                                     if r.get("template_kwargs")), None),
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
    log("%-30s %-4s %-5s %-4s %-5s %-8s %-13s %-4s %-6s %8s %9s %7s %7s"
        % ("model", "ngl", "ncmoe", "ot", "ub", "fill", "spec", "nmax",
           "mmproj", "tok/s", "prefill", "VRAM", "accept"))
    for r in rows:
        c = r["config"]
        log("%-30s %-4d %-5d %-4d %-5d %-8d %-13s %-4d %-6s %8.2f %9.1f %7.0f %7s%s"
            % (r["model"][:30], c["ngl"], c.get("ncmoe") or 0,
               c.get("n_cpu_ffn") or 0, c["ub"], c.get("fill") or 0,
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
               _degenerate_note(r)))
    return True
