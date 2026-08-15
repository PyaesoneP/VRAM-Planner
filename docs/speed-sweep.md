# Running a speed sweep

`--sweep` answers *"does this config fit, and what does the allocator reserve for it"*.
This is the question left over: **of the configs that fit, which one is fastest** — and it
cannot be derived from the memory model, because the two settings most worth tuning are
invisible to it.

- **Speculative decoding.** `speed.py` is a bandwidth roofline with no acceptance-rate
  term, and `per_token_bytes()` skips MTP blocks because they do not run during ordinary
  decode. The planner can price what speculation *costs* — an f16 draft cache, whatever
  `--cache-type-k` says — and nothing of what it saves. On one 27B hybrid the roofline
  predicted MTP would **lose 4%**; measured, it **won by 41%**.
- **Prompt processing**, which is not modelled anywhere in this tool, on purpose.

Everything below is a real command. The same harness also runs from the web UI — see
[§2](#or-from-the-browser) for the mapping and [§11](#11-the-launch-script) for turning a
result into something you can launch — but nothing here *requires* it.

---

## 1. Before you start

**The GPU must be free.** `preflight()` refuses to run if less than 85% of VRAM is
available, because a resident model does not make a sweep *fail* — it makes every row
wrong in the same direction, which is worse, since the numbers still look like numbers.
Close LM Studio, stop any `llama-server` you launched yourself.

> `--dry-run` skips preflight. It is safe to plan a sweep while something is loaded; it
> is not safe to run one.

You also need a `llama.cpp` build. The sweep finds one under LM Studio's
`extensions/backends`, newest CUDA first, and puts its shared vendor package on `PATH` —
without which the binary dies with `STATUS_DLL_NOT_FOUND` and no message at all.

```
python -m vram_planner --speed-sweep --models MODEL --dry-run
```

Always dry-run first. It prints every config and a time estimate, and runs nothing.

---

## 2. Quick start

```
# plan it
python -m vram_planner --speed-sweep --models Qwen3.8 --speed-ctx 131072 --speed-kv q8_0 --dry-run

# run it (hours; resumable, safe to interrupt)
python -m vram_planner --speed-sweep --models Qwen3.8 --speed-ctx 131072 --speed-kv q8_0

# read it back, fastest first
python -m vram_planner --speed-report
```

Rows are appended one at a time and flushed, and any config already recorded is skipped,
so an interrupted sweep resumes where it stopped.

### …or from the browser

`python -m vram_planner`, analyse a model, then use the **Measure real speed** card. It
drives the same harness, so everything in this guide still applies — the differences are
only in the driving:

| | CLI | browser |
|---|---|---|
| context / KV quant | `--speed-ctx`, `--speed-kv` | taken from the form and frozen |
| stages | `--speed-stages abcd` | tick boxes |
| preview | `--dry-run` | **Preview grid** |
| progress | printed rows | live table, ordered as measured |
| stopping | Ctrl-C | **Stop**, which lands between configs |
| chaining | `--speed-chain`, off by default | **Chain the stages**, ticked by default |
| rounds | `--speed-rounds N` | the **Rounds** field |
| reading back | `--speed-report` | **Measured results**, the ranked table |
| findings | `--speed-report --insights` | **Past sweeps**, expandable per campaign |
| the payoff | copy the flags by hand | **Launch script**, §11 |

Two things the browser does that the CLI does not. It refuses to start while anything
else holds the GPU and *names the process*, rather than printing a number you have to
interpret. And **Stop** finishes the config in flight before ending, so its row is
complete rather than half-written — which costs up to a couple of minutes and loses
nothing, because rows are keyed and restarting resumes from there.

**Past sweeps needs none of this.** It reads campaigns already on disk, so it is what the
page shows before you have analysed anything, and every one of its rows can be turned into
a launch script without a model being loaded or a GPU being touched.

---

## 3. The staged design

The grid is **one knob at a time from a baseline**, not a cross product — a cross product
of these axes is hundreds of loads at roughly two minutes each. Stages run in order, and
each is meant to be re-run at the previous stage's winner — which `--speed-chain` now does
for you (§3.1).

| stage | what it varies | why it is first/last |
|---|---|---|
| **A** | `-ngl` (dense) or `--n-cpu-moe` (MoE) | Finds the wall. Every later comparison is meaningless at a config that spills. |
| **B** | projector in VRAM vs system RAM | Only if the model ships an `mmproj`. Moves the ceiling, so it re-walks stage A's ladder. |
| **C** | `-ub` (physical batch) | Mostly a *prefill* knob; costs VRAM, so it interacts with the wall. |
| **D** | `--spec-type` and draft depth | Needs the winner of A–C to be settled first. |

Stage A's ladder is **seeded from the planner** — `plan.analyze()` is asked where the
split falls for this model, this card and this context, and the ladder brackets that.
You do not hand-pick rungs.

**Depth is not a stage.** It is the `--speed-fill` axis, and it matters more than any
single setting (see §8). Run the top two or three configs again at a realistic fill:

```
python -m vram_planner --speed-sweep --models MODEL --speed-fill 32768 \
    --speed-axes "ngl=28 spec=draft-mtp spec_n_max=2 mmproj_offload=0"
```

---

## 3.1 Chaining: stop measuring at a baseline nothing confirmed

Without chaining, stages C and D are pinned at **the planner's** `-ngl`, not at the one
stage A found. That is not an oversight — nothing can know stage A's result before stage A
has run — but it means ubatch and speculation get measured at a split you have no evidence
for. The historical fix was to re-run by hand with `--speed-axes` at the previous winner.

```
python -m vram_planner --speed-sweep --models MODEL --speed-chain
python -m vram_planner --speed-sweep --models MODEL --speed-chain --speed-rounds 2
```

Each stage is built *after* the previous one finishes, from the fastest trustworthy row so
far. **The number of loads does not change**, so neither do the hours: this is the same
search with the baseline kept honest, not a wider one.

Between stages the log says what it decided, including when it decided nothing:

```
stage C round 1: 4 configs at ngl 31 ncmoe 0 ub 512 spec none
  baseline -> ngl 31 ncmoe 0 ub 1024 spec none   (8.42 tok/s, stage C)
stage D round 1: 8 configs at ngl 31 ncmoe 0 ub 1024 spec none
  baseline unchanged: nothing beat 8.42 tok/s by more than 2%
```

Three rules make it safe:

- **Only what was measured under the same conditions competes.** Same model, GPU, backend
  build, context, KV quant, fill depth, `n_predict` and `repeat`. A 2k-fill row must never
  set the baseline for a 32k campaign — that is not a slower config, it is a different
  experiment.
- **Spilled and looping rows never carry forward.** They stay in the table, because they
  are evidence about where the wall is; they just cannot be the thing a baseline is drawn
  from, since a bad baseline bends every stage after it in one direction, silently.
- **A challenger must win by more than 2%.** `tok_s` is a median of `repeat` passes, so a
  1% lead is jitter — and rebasing on jitter would make the campaign explore different
  configs on two runs of the same machine.

The baseline is re-read **from disk**, not from memory, so a campaign stopped after stage A
and restarted tomorrow recovers stage A's winner instead of falling back to the planner.

`--speed-rounds N` re-runs the stages from the winner; it is cheap because the resume key
does not include the stage letter, so anything a later round revisits unchanged is already
recorded and skipped. A second round is worth it mainly when speculation wins in stage D:
the draft cache costs VRAM the planner priced under different assumptions, so the wall
genuinely moves and stage A's ladder is rebuilt around the new split.

A chained dry-run can only list the **first** stage. The stages after it are built from a
baseline that does not yet exist, and printing values for them would be a guess dressed up
as a plan — the guess being exactly the frozen one this mode exists to escape. The
*counts* are still exact (a ladder's length does not depend on where it is centred), so the
hour estimate is not a guess.

---

## 4. Freezing settings vs chasing speed

**Frozen** — you have non-negotiables (a context length, a KV quant):

```
python -m vram_planner --speed-sweep --models MODEL --speed-ctx 131072 --speed-kv q8_0
```

`--speed-ctx` is clamped to the model's trained context, so it can never ask for more
than the file supports. Both values are printed on every dry-run line, because a frozen
setting that silently is not what you think is the failure mode the listing exists to
prevent.

**Unfrozen** — you want the fastest configuration that exists. Then **context is your
biggest lever**, because KV is what crowds layers off the GPU. On one 12 GB card the
planner's split moved from `ngl 25–33` at 131k to `ngl 35–43` at 32k — same model, same
card. Run stage A at two or three contexts, pick the shortest you can live with, then
tune everything else there.

```
python -m vram_planner --speed-sweep --models MODEL --speed-ctx 32768  --speed-stages a
python -m vram_planner --speed-sweep --models MODEL --speed-ctx 131072 --speed-stages a
```

Quantising the KV cache buys layers the same way (`q8_0` is about half of `f16`) and
needs flash attention, which the grid already has on.

---

## 5. Dense vs MoE — the axis inverts

The harness detects this from the file; you do not pass a flag. But know what changes:

|  | dense | MoE |
|---|---|---|
| stage A axis | `-ngl` | `--n-cpu-moe`, at `-ngl` = all blocks |
| better direction | **higher** | **lower** |
| the wall is | the largest value that fits | the smallest value that fits |

`-ngl N` puts the last N blocks on the GPU — attention, **KV** and experts together.
`--n-cpu-moe M` moves only the *routed experts* of the first M blocks to the CPU, leaving
their attention and KV in VRAM. Experts are the only weights big enough to be worth
moving (92% of the file on one 35B-A3B) and only `n_expert_used` of `n_expert` fire per
token — while whole-block offload drags KV to the CPU with it, which is the expensive
thing to lose. **Laddering `-ngl` on an MoE measures the fallback strategy and never
finds the good configuration at all.**

Stage D also skips `draft-mtp` automatically when the file has no `nextn` blocks — there
is nothing to draft from, so those rows would be identical failures. Check before you
plan a campaign around speculation:

```
python -c "from vram_planner.gguf import load_gguf; from vram_planner.model import extract_config; \
c=extract_config(load_gguf(r'PATH.gguf')); print('MTP blocks:', c['n_mtp_layers'])"
```

If that prints `0`, only the `ngram-*` variants apply, and they are worth far less.

---

## 6. Flags and axes

### Command-line

| flag | meaning |
|---|---|
| `--speed-sweep` | run the staged grid |
| `--dry-run` | print the configs and estimate, run nothing |
| `--models NAME` | substring match on the file name |
| `--speed-stages abcd` | which stages to run |
| `--speed-ctx N` / `--speed-kv TYPE` | freeze context / KV quant |
| `--speed-fill N` | prompt length to measure at |
| `--speed-axes "k=v,v …"` | explicit ladder **instead of** the staged grid |
| `--n-predict N` / `--repeat N` | tokens per measured pass / passes per config |
| `--limit N` | stop after N configs |
| `--backend BUILD` | pick a specific llama.cpp build |
| `--sweep-timeout SECONDS` | per-load timeout (raise it for deep fills) |
| `--speed-chain` | build each stage from the fastest row so far (§3.1) |
| `--speed-rounds N` | with `--speed-chain`: re-run the stages from the winner |
| `--speed-report` | every recorded row, fastest first |
| `--speed-report --insights` | what the campaigns *found* rather than the ranking (§8.1) |

### `--speed-axes` names

Values are comma-separated; multiple axes form a cross product, so keep it small.

| axis | type | notes |
|---|---|---|
| `ctx`, `ngl`, `ub`, `seq`, `ncmoe`, `fill` | int | `seq` is `-np` |
| `kv` | string | `f16`, `q8_0`, … |
| `fa`, `mmproj_offload`, `warmup` | bool | `0`/`off`/`false` are false |
| `spec` | string | `none`, `draft-mtp`, `ngram-mod`, `ngram-cache`, `ngram-simple` |
| `spec_n_max`, `spec_n_min` | int | draft depth |
| `temp`, `top_p`, `min_p`, `rep_pen`, `pres_pen` | float | see §7 |
| `top_k` | int | |

```
python -m vram_planner --speed-sweep --models MODEL \
    --speed-axes "ctx=131072 kv=q8_0 ngl=28 mmproj_offload=0 spec=draft-mtp spec_n_max=1,2,3"
```

---

## 7. Measurement discipline

**This section is worth more than the commands.** Each item below exists because it
produced a wrong answer that looked completely plausible.

### Bracket every campaign with a repeated control

Machines drift. One laptop measured **±7% within a session**, and once returned **0.58
tok/s for a config that had twice measured 5.54** — a 5× collapse that had nothing to do
with any setting. Re-run one identical config at the start and end of a campaign. If the
two disagree by more than a few percent, throw the run away.

Never compare a number from today against one from yesterday. Re-measure the baseline
*in the same session* as whatever you are comparing it to.

### Greedy is speculation's best case

The harness defaults to `temperature 0` so the token stream is reproducible and two rows
differ by the setting under test and nothing else. But llama.cpp accepts a draft token
when the target's own sampled token matches it, and under greedy that comparison is
deterministic. **An acceptance rate measured greedy is an upper bound.**

Measured on one 27B hybrid: greedy showed **+85%** over no speculation; at the model's
recommended sampler settings it was **+41%**. Same hardware, same session. If you care
about speculation, sweep at your real settings:

```
--speed-axes "… temp=1 top_k=20 top_p=0.95 min_p=0"
```

The non-speculative baseline is unaffected by sampling — bandwidth does not care what the
tokens say — which makes it a useful control.

### Check `distinct_ratio` on speculative rows

Every row records a text sample and the fraction of unique 8-word windows in it.
`temperature 0` with `ignore_eos` can put a model in a repetition loop, and a loop is
exactly what n-gram speculation predicts perfectly. On one model the ratio fell from
**1.00** at 32k fill to **0.18** at 64k — where acceptance duly read **100%**. Those rows
flatter speculation and must not be believed.

Healthy prose is near 1.0. Well below that, discard the speculative comparison.

### Acceptance rate is conditional on having drafted

A drafter that fires rarely and is always right reports 100% acceptance and buys almost
nothing. One `ngram-mod` row showed 100% acceptance while drafting **15 of 128 tokens** —
a 7% gain. Read `draft_n` next to `accept_rate`, never the rate alone.

### Long fills repeat the filler

The filler prompt is this repository's own README and sources — real prose and code,
because a prompt built by repeating one paragraph hands n-gram speculation a result it
could never reproduce on real work. Past the length of that corpus (~98k tokens) it must
start repeating, and the row is flagged `corpus_repeated`. A repeat is trivially
predictable, so speculative rows above that length are optimistic.

---

## 8. Reading the results

```
st ngl  ncmoe ub    fill     spec          nmax mmproj |    tok/s   prefill    VRAM accept
A  28   0     512   2048     draft-mtp     2    ram    |     9.45     296.9   11410    80%
```

| column | meaning |
|---|---|
| `tok/s` | **median of `--repeat` cache-warm passes** — pure decode |
| `prefill` | from one cold pass, `cache_prompt` off |
| `VRAM` | per-process dedicated VRAM, from the OS counter |
| `accept` | `draft_n_accepted / draft_n`, speculative rows only |

Two passes per config, deliberately: one **cold** pass measures prefill and is discarded
for decode (llama.cpp loads CUDA kernels lazily, so the opening pass pays for machinery
every later pass gets free), then `--repeat` **cache-warm** passes measure decode at the
real context depth. Splitting them is what makes deep fills affordable at all.

### Statuses

| status | meaning |
|---|---|
| `ok` | measured |
| `oom` | did not allocate — a hard upper bound, recorded rather than dropped |
| `exit` / `timeout` | died, or never printed a ready line |
| `genfail` | loaded, then the generation failed — e.g. a draft depth whose context could not allocate |

`spilled` marks a row that loaded but whose floor computed negative — on Windows, WDDM
spills past dedicated VRAM instead of failing, so the load *succeeds* and only the numbers
give it away.

### Depth beats every setting

A headline number at 2k fill is the least representative number in the set. From one real
campaign, same config throughout:

| context filled | tok/s |
|---|---|
| 2k | 9.45 |
| 32k | 4.81 |
| 64k | 2.53 |
| 120k | 1.49 |

If you configured a long context you will *live* at depth, so quote yourself the number at
the depth you actually work at. Speculation's advantage decays the same way — +71% at 2k,
+29% at 32k, level by 120k — because as KV comes to dominate decode, the blocks the draft
cache displaced start to cost more than speculation saves.

---

## 8.1 What the campaign found

A ranking answers "which config won". It does not answer "what did I learn", and the
second is the one that tells you where the next two hours should go. `--speed-report
--insights` — and the **Past sweeps** card in the browser — derive four things from rows
already on disk, measuring nothing new:

```
python -m vram_planner --speed-report --insights
```

| | |
|---|---|
| **campaigns** | one entry per (model, GPU, backend build), with row counts and dates |
| **what each knob was worth** | best value vs its natural reference, with the effect size |
| **speed vs context depth** | the same config at more than one fill — the slope of §8, measured |
| **speed vs VRAM** | the rows nothing beats on *both* |

The effect sizes are the point, and their correctness rests entirely on one rule: rows are
bucketed by **everything except the axis in question**, plus the model, GPU, backend file,
`n_predict` and `repeat`. Rows from different experiments never meet, because averaging
across a GPU change or a depth change manufactures an effect out of the difference between
the runs rather than out of the knob. Where more than one such bucket exists, the one with
the most distinct values is reported and the others are *counted, not merged*.

Two smaller rules, both learned the hard way in §7:

- Spilled and looping rows are excluded from every conclusion and the count of exclusions
  is printed. They stay in the row table — a spilled row is how you find the wall.
- The speculation axis carries a warning when its rows needed the filler corpus to repeat.
  Speculation drafts from what it has already seen, so repeated text is its best case and
  the gain is an upper bound rather than anything a conversation will give you.

An axis measured at only one value is reported as "only one value ever tried" rather than
dropped. That is usually the most actionable line in the output: `kv` shows up that way
until you actually sweep it, and KV quant is one of the biggest levers there is (§4).

**Speed vs VRAM** exists because "the fastest" and "the fastest that still leaves the
desktop a card to draw on" are different questions, and a ranking by tok/s can only answer
the first. A row 2% slower for 3 GiB less is often the one worth running.

None of this needs a model analyzed or touches the GPU, and any row in it can be turned
straight into a launch script — including for a model that has since been moved or deleted,
in which case the script is still written in full and its header says the path did not
resolve.

---

## 9. Where the data lives

```
%LOCALAPPDATA%\vram-planner\speed\<gpu>__<build>.jsonl     # rows
%LOCALAPPDATA%\vram-planner\sweep-logs\                    # last load's log, and failures
```

**Rows go to `speed/`, never `sweeps/`, and that separation is load-bearing.**
`fit.load_sweep()` globs every `.jsonl` under `sweeps/` and fits anything
`suspect_reason()` accepts. A speculative row would sail through that filter, and
`compute.py` has no term for a draft cache — those megabytes would be absorbed into
`floor` and `ctx` and quietly corrupt every future plan. Verify the boundary holds:

```
python -m vram_planner --fit      # must be unchanged by any speed sweep
```

Resume keys on the full config **and** on `n_predict`/`repeat`, so a quick smoke row at
`--n-predict 32 --repeat 1` is never mistaken for a real measurement.

---

## 10. Troubleshooting

**Every row OOMs.** The ladder is seeded from the planner against *total* VRAM. If real
free VRAM is much lower, something is holding the card — that is what preflight is for.

**A flag seems to do nothing.** Check it is not deprecated. In recent builds
`--draft-max` → `--spec-draft-n-max`, and `--no-mmap`/`--mlock` → `--load-mode`. The old
names are accepted and then **ignored with a deprecation line**, which looks exactly like
a setting that had no effect. Confirm with `llama-server --help` for *your* build.

**`genfail` on deep draft settings.** The draft context allocates its own compute
buffers, which grow with draft depth; depth 5 and 8 failed where 1–3 fitted. Not a bug —
a VRAM ceiling.

**Numbers changed and no setting did.** Re-run your control. See §7.

---

## 11. The launch script

The point of all this is a command line you can live with, and the **Launch script**
block writes the whole launcher — for the row selected in the ranked table, or for the
planner's predicted split when nothing has been measured yet, in which case the script's
own header says `PREDICTED by the planner, not measured`.

A measured script carries its evidence in the header: decode and prefill rates, VRAM, the
acceptance rate with its denominator, the **fill depth it was measured at**, and a
`SPILLED` warning if that row was over-committed. That last group matters most — a
generated script that claimed 11 tok/s without saying it was measured at 2,048 tokens of
context would be repeating the exact mistake §7 exists to prevent.

Choose PowerShell or bash; the native one for your machine is preselected.

### What it handles for you

| trap | what the script does |
|---|---|
| `llama-server` is not on PATH | resolves it under LM Studio's backends folder |
| CUDA runtime lives in a separate *vendor* package; without it the process dies with `STATUS_DLL_NOT_FOUND` (exit `-1073741515`) **and no message** | reads the backend manifest and puts every vendor directory on `PATH` / `LD_LIBRARY_PATH` |
| a pinned backend path silently stops existing when LM Studio updates | matches the backend *family* and picks the highest version — version-aware, because by name `2.9.0` sorts above `2.10.0` |
| nothing is logged when launched directly | `--log-file` with a dated name per launch (llama.cpp does not rotate it) |
| `--draft-max` removed, `--no-mmap`/`--mlock` deprecated | emits `--spec-draft-n-max` and `--load-mode` |
| `--jinja` must precede `--chat-template-file`, or only built-in template *names* are accepted | emits `--jinja` first whenever a template is set |
| a missing `--chat-template-file` is **not** an error — llama-server silently falls back to the GGUF's own template | checks the path and refuses to start |
| no backend installed at all | falls back to a `llama-server` already on `PATH` |

Every setting is a parameter, so you can override without regenerating:
`-Ngl 25` in PowerShell, `NGL=25 ./run-model.sh` in bash.

### Two defaults worth knowing

**`--load-mode none`**, not llama.cpp's `mmap`. Measured at 13.81 GiB resident versus
17.94 GiB for `mmap+mlock` on a 9.5 GiB working set: mlock pins the whole mapped file
*including the blocks already resident in VRAM*. Right for CPU-only inference, wasteful
under heavy GPU offload. Plain `mmap` is lower pressure still, but those pages are
file-backed and can be evicted, which is what makes decode stutter late in a session.

**Sampling is blank.** No sampler flags are written unless you fill the fields in. A
made-up default is worse than none, because llama.cpp's own (temp 0.80, top-k 40,
min-p 0.05) are not what every model card asks for. Whatever you do set becomes a
*server* default: any client that sends its own values — Open WebUI, aider, most chat
UIs — overrides them per request.

### Chat template

`--chat-template-file` and `--chat-template-kwargs` are set in the same block as the
samplers, and behave the same way: blank writes nothing at all.

```powershell
-ChatTemplateFile 'C:\models\qwen3-tools.jinja' -ChatTemplateKwargs '{"enable_thinking":false}'
```
```bash
CHATTEMPLATEKWARGS='{"enable_thinking":false}' ./run-model.sh
```

Four things worth knowing:

- **The kwargs are validated as a JSON *object* when the script is generated.** Not a
  list, not a bare scalar. llama-server does not find out otherwise until it is starting
  up, and the difference between a message in the browser and a service that does not come
  back after a reboot is exactly this check.
- **The key names belong to the template, not to llama.cpp.** `enable_thinking` is Qwen3's
  spelling. A model that does not use that variable *ignores* it rather than rejecting it,
  so a typo here is silent — there is no preset list offered because there is nothing to
  verify it against.
- **The flags are assembled at runtime**, not written into the command. `--chat-template-file ''`
  is an error, not a no-op, so an empty parameter has to produce *no flag* — the same
  pattern `--log-file` already uses.
- **Windows PowerShell 5.1 drops the quotes.** Building a native command line, it
  re-quotes each argument and does *not* escape double quotes already inside a value, so
  `{"a":1}` reaches the exe as `{a:1}` and llama-server answers with
  `parse error at line 1, column 2 ... last read: '{e'` — which points at your JSON when
  the JSON was never the problem. The generated script escapes them as `\"`, and skips
  that under PowerShell 7.3+ (`$PSNativeCommandArgumentPassing`), where the argument is
  passed through intact and the backslashes would arrive literally. bash needs none of
  this: `execve` takes argv directly and nothing re-parses it.
- These are **server defaults**, so a client sending its own `chat_template_kwargs` in the
  request wins for that request.

### Where it goes

**save next to the model** writes it into the model's folder and reports the path, which
beats a browser download landing in `Downloads/` with a `.ps1` that then trips execution
policy. **download** and **copy** are there too.
