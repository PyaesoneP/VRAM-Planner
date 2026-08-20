<p align="center">
  <img src="docs/logo.svg" alt="VRAM Planner" width="120">
</p>

# VRAM Planner

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Zero dependencies](https://img.shields.io/badge/dependencies-none-2dd4bf?style=flat-square)
![GGUF parsed directly](https://img.shields.io/badge/GGUF-parsed%20directly-2dd4bf?style=flat-square)
![License](https://img.shields.io/github/license/PyaesoneP/VRAM-Planner?style=flat-square)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square)

A self-contained Python package (standard library only). It parses any GGUF
**directly** (reads the real byte size of every tensor — the same info as
`npx @huggingface/gguf --show-tensor`, but with no npx/Node dependency), reads
your live free VRAM + RAM, and tells you exactly how a model fits at any context
length and KV-cache quant:

- **Dense models** → how many layers go on the GPU (`-ngl` / LM Studio "GPU Layers").
- **MoE models** → how many layers' experts to keep on CPU (`--n-cpu-moe` / `-ot`),
  keeping attention + router + shared experts on the GPU.
- Max context that still fits fully on the GPU, a KV-cache-vs-context table, and a
  full memory breakdown.
- The one estimated term (the compute buffer) can be re-fitted to **your** machine
  with a `--sweep`/`--fit` harness — see **Measuring it yourself**.

It serves its own web UI, so you never type these commands by hand. The UI opens on
a single recommended config, says whether that came from **measurement** or from the
planner's **estimate**, and — when the two disagree — names every reason why. See
**"Largest that fits" is not "fastest"**.

## Using the interface

![The VRAM Planner interface: the control rail, the three steps, and the single answer block](docs/ui-overview.png)

Four controls and a button get you an answer. Everything else is behind two
disclosures, and the results column shows **one thing at a time**.

**Pick a model and the controls it has appear.** Selecting a model reads its header
(`/api/probe` — the same few kilobytes `analyze()` opens with, and it warms the same
cache) and shows what that model actually offers: the projector placement and image
sizing for a multimodal one, the MTP tick box for a model shipping those blocks,
`--n-cpu-moe` for an MoE against `-ot` for a dense model. A line under the picker says
what was found — `qwen35moe · MoE · 41 layers · 35.51 B · vision projector · 1 MTP
block · 256k native ctx` — so a form that changes shape explains why.

These used to be revealed by the *answer*: they were unhidden by the analyze response,
so every setting that depended on the model was one you discovered after it had already
been decided for you, and using it meant running the whole thing again.

**One answer, once.** The config to run, whether it fits, the tok/s if it has been
measured, and where the memory goes are a single block, badged `● measured` or
`estimated`. A trustworthy measured row supersedes the planner's arithmetic; rows that
spilled into shared memory, looped, or copied the prompt back are never eligible however
fast they read. When the measured answer and the estimate disagree, the block names each
reason rather than leaving you to spot it — see
[**"Largest that fits" is not "fastest"**](docs/speed.md#largest-that-fits-is-not-fastest--and-the-two-used-to-disagree-silently).

**Three steps, in the order they depend on each other.** Only the open one renders; the
other two carry their answer on the tab, so nothing is hidden that you would have to go
looking for. The open one does *not* repeat its answer on its own tab — that answer is
on screen directly beneath it.

| step | answers | tab reads when closed |
|---|---|---|
| **1 · Does it fit** | the verdict, the memory bars, and behind one disclosure everything the planner *derives* — suggested settings, the `llama-server` command, the speed estimate, the KV-vs-context table, the full breakdown | `fits, barely · 10.75 GiB` |
| **2 · Measure real speed** | the grid, a preview of what it will cost in hours, live rows as they land, and **Past sweeps** — every campaign ever recorded on this machine | `63.64 tok/s best` |
| **3 · Launch script** | a complete `.ps1`/`.sh` for the selected row, with template, sampler and load-mode controls | `ngl 41 · draft-mtp` |

Steps 2 and 3 need **no model analyzed and touch no GPU** — reading a recorded campaign
and building a script from one of its rows is what the page offers before you have
pressed Analyze at all.

**Three tiers of control.** *Model* holds the four things every plan needs. *Tune* holds
ubatch, sequences, flash attention, the vision projector and speculative decoding.
*Expert* holds the budgets, the layer overrides, the bandwidth figures and calibration.
Both tiers stay shut and say what they are holding for *this* model — `Tune · batch,
sequences, vision projector, MTP` — rather than springing open and undoing whatever you
had already set. Nothing was removed: the long explanations that used to sit under every
field are one click away under a dotted **why**, so the page answers first and explains
on request.

**Stored model cards**, under the picker. A card is a GGUF's header, kept so a model
that is no longer on this disk stays plannable — they accumulate as a side effect of
every analyze. The list shows each one with its architecture, layers, parameters and
size, and **✕** forgets it after a confirmation. A card for a file still on disk is
rebuilt by the next analyze; for one that has gone, this is how it leaves the picker.
(`--cards` and `--forget-card` do the same from the CLI.)

**A glossary** under steps 2 and 3 defines the six abbreviations that head every row
table (`ngl`, `ncmoe`, `ub`, `fill`, `spec`, `proj`); the same text is the `title` on
each column header. Step 1 has no such table and does not show it.

The page is theme-aware: it follows your OS setting, and the ◐ button in the header
overrides it in both directions.

**On the look.** Colour means one thing here: it names a *memory term*. Weights, KV
cache, compute buffer, recurrent state, projector and draft cache each own a hue, and
the memory bar, its legend and the breakdown table are built from the same objects so
they cannot drift apart. Everything structural — panels, rules, tabs, chips, the primary
button, the selected state of anything — is ink, paper and hairline. The only other
coloured things on the page are the fit verdict and the `measured` badge, which are
also facts rather than furniture.

**Checking a plan against reality.** A configuration loaded in LM Studio, with Task
Manager showing what the engine actually allocates:

![The same config running in LM Studio](docs/demo-lmstudio.gif)

## Supported platforms

| | status |
|---|---|
| **Windows + NVIDIA** | validated — all measurements below were taken here |
| **Linux + NVIDIA** | should work (`nvidia-smi` + `/proc/meminfo` paths exist), untested |
| macOS / Metal | **unvalidated** — the tool warns and keeps running |
| AMD / ROCm, Intel | **unvalidated** — same warning |

The split matters because only *part* of the tool is hardware-dependent. Weights, KV
cache, recurrent state and the projector are read from your GGUF and are exact
everywhere. The **compute-buffer estimate** was fitted against CUDA on Windows, and
Metal/ROCm allocate their graphs differently — so on those the total is indicative
until you calibrate it. The tool detects this and says so in the terminal and the UI
rather than quietly reporting a confident wrong number.

Per-process VRAM measurement (**Measure running model**, under *Expert*) additionally
needs `nvidia-smi` or Windows GPU performance counters, and reading LM Studio's
*resolved* config needs its Windows log path.

## Requirements
- Python 3.8+ (standard library only — nothing to `pip install`).
- NVIDIA driver on PATH for **live** VRAM (`nvidia-smi`, installed with your driver).
  If it's missing, everything still works — just type your VRAM budget manually.

## Run
```powershell
python -m vram_planner
# or, if that isn't found:
py -m vram_planner
```
Run it from the folder containing the `vram_planner/` package.
A browser opens at http://localhost:8121. Point the **Models folder** at your
LM Studio models dir (defaults to `%USERPROFILE%\.lmstudio\models`), pick a model,
set context + KV quant, and press **Analyze fit**.

Calibration, model cards and measured rows live in `%LOCALAPPDATA%\vram-planner\`
(`~/.local/share/vram-planner/` elsewhere), not next to the script.

## Documentation

This file is the front door: what it is, how to run it, and how the interface works.
Everything else is one level down, by the question it answers.

| | |
|---|---|
| [**Speed**](docs/speed.md) | The bandwidth roofline, why **"largest that fits" is not "fastest"**, and how a measured row supersedes the estimate. Start here if the planner and your stopwatch disagree. |
| [**Accuracy**](docs/accuracy.md) | What is computed exactly from the tensor table, what is estimated, how far off the one estimated term is, and how to re-fit it to your machine. Known issues live here too. |
| [**Model families**](docs/models.md) | Sliding-window, hybrid, multimodal and MoE models each break a different assumption. Also stored model cards — planning a model you no longer have on disk. |
| [**Command line**](docs/cli.md) | Every flag, the generated `.ps1`/`.sh` launcher, and the tips worth knowing. |
| [**Speed sweep guide**](docs/speed-sweep.md) | The full measurement campaign: the staged design, chaining, measurement discipline, and reading the results back. |
| [**Internals**](docs/internals.md) | The module graph, and how these screenshots are regenerated. |
| [**Qwen3.8-27B findings**](docs/findings-qwen3.8-27b-ud-q4-k-xl.md) | A worked campaign on one model, end to end. |

## License

MIT — see [LICENSE](LICENSE).
