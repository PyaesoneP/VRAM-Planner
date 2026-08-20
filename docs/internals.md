# Internals

The module graph the package is built on, and how the screenshots in these docs are
produced.

[← back to the README](../README.md)

---

## Layout
Each module imports only from those above it, so the import graph is a DAG and
every number has one home:

| module | what lives there |
|---|---|
| `const` | version, byte units |
| `gguf` | the GGUF binary reader |
| `model` | config extraction, tensor classification |
| `kv` | cache geometry — exact, read from metadata |
| `compute` | the compute buffer — the one fitted term |
| `paths` | user data locations |
| `gpu` | live hardware probes |
| `lmstudio` | models, runtime config, server logs, backends |
| `speed` | bandwidth roofline |
| `calib` | fitting `compute` to this machine's measurements |
| `plan` | `analyze()` and the layer-split planners |
| `sweep` | driving `llama-server` across a config grid, recording what it allocates |
| `bench` | driving it across a grid and recording how fast it **generates** |
| `_corpus.txt` | the frozen benchmark filler, committed on purpose — never delete (see *Measuring speed*) |
| `fit` | scoring `compute` against sweep data, held out |
| `recommend` | one config to run — reconciling `plan`'s estimate against `bench`'s measured rows, and naming every reason the two differ |
| `launch` | turning a config into a runnable `llama-server` launch script |
| `job` | the one background campaign the web UI can start and watch |
| `web` | JSON endpoints and static file serving |
| `ui/` | `index.html`, `app.css`, `app.js` — the front end, as real files |
| `selftest` | synthetic GGUFs and the test suite |
| `cli` | entry point |

`sweep`, `bench` and `fit` are the evidence base, not part of a plan — nothing above
imports them and the tool works without ever running any of them. See **Measuring it
yourself** and **Measuring speed**.

Three edges run backwards and are imported inside the function that needs them:
`compute.compute_buffer_terms` reads `calib.calib_coeffs`, `calib.record_calibration`
calls `plan.analyze`, and `calib.migrate_calibration` calls it too when re-deriving
stored rows after a schema change. All three are marked at the call site.

`import vram_planner as v` still re-exports the whole public surface, so
`v.analyze()`, `v.load_gguf()` and friends work unchanged.


## Regenerating the screenshots

`docs/ui-*.png` are captured from the running app rather than drawn, so they cannot
drift into describing an interface that no longer exists — but they do have to be
retaken after a UI change. The capture drives headless Chrome over the DevTools
protocol; Node 22+ ships a WebSocket client, so it needs nothing installed:

1. `python -m vram_planner --port 8140 --no-browser`
2. `chrome --headless --remote-debugging-port=9222 --user-data-dir=<tmp> about:blank`
3. connect to `http://127.0.0.1:9222/json`, drive the page with `Runtime.evaluate`,
   and capture with `Page.captureScreenshot` (`captureBeyondViewport: true` and a clip
   from `Page.getLayoutMetrics` for a full-page shot).

The six of them, and what each is a picture of:

| file | shows | how it is framed |
|---|---|---|
| `ui-overview.png` | the whole page after an analyze, with *Tune* open | full page |
| `ui-recommendation.png` | the answer block alone | clip to `.card.rec` |
| `ui-sweep.png` | the measurement form before a campaign starts | clip to `#sweepbody` |
| `ui-script.png` | the launcher pane with a generated script | clip to `#sweepscript`, after `sweepGen(false)` |
| `ui-insights.png` | a campaign's findings | clip to `.campopen` |
| `ui-forget.png` | the delete confirmation | clip to `.camp:has(.campconfirm)` |

Clip to an element rather than to a pixel rectangle — read
`getBoundingClientRect()` plus `scrollX/scrollY` and pass that as the `clip`, at
`scale: 2`. A hard-coded rectangle silently reframes the moment anything above it
changes height, which is exactly what a UI change does. Start the clip *at* the
element and not a few pixels above it: the block above the answer is the sticky
stepper, and a sliver of it in frame reads as a cropped tab rather than as margin.

Two things worth knowing if you write that script. `app.js` is a classic script, so its
top-level `let` bindings — `LAST`, `SWEEP`, `REC` — live in the global *lexical* scope
and are reachable as bare identifiers but **never** as `window.LAST`. And wait on what
is painted, not on the state behind it: the step tabs are redrawn by a later callback
than the fetch that feeds them, so polling the data captures a loading label.
