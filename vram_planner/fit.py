"""Score the compute-buffer model against sweep data, out of sample.

The number this project used to publish was a fit residual: the coefficients in
compute.py were solved on the same loads the self-test then scored them against.
That measures how well a curve can be drawn through points, not how well it
predicts the next load, and those are very different when there are four
coefficients and 36 rows.

This module only ever reports held-out error. Two splits, because they fail
differently:

  * leave-one-MODEL-out - fit on every architecture but one, score on the one
    left out. This is the question a user actually asks: I am about to load a
    model you have never seen, how wrong will you be?
  * leave-one-AXIS-TOP-out - fit with the top of a knob's range removed, score
    there. This catches a model that interpolates well and extrapolates badly,
    which is the failure mode that matters at long context.

A row is only fitted if it measured what it claims to. sweep.suspect_reason()
rejects the ones taken with the card at the wall, where WDDM spilled to shared
memory and the counter reports the cap rather than the need.
"""
import json, os, re
from .compute import CB_DEFAULTS, compute_buffer_terms, graph_is_split
from .paths import _data_dir
from .sweep import suspect_reason


def sweep_dir():
    return os.path.join(_data_dir(), "sweeps")


def load_sweep(path=None, gpu=None, backend=None):
    """Every recorded row, from one file or from all of them."""
    paths = [path] if path else [
        os.path.join(sweep_dir(), f) for f in sorted(os.listdir(sweep_dir()))
        if f.endswith(".jsonl")] if os.path.isdir(sweep_dir()) else []
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                r["_file"] = os.path.basename(p)
                if gpu and r.get("gpu") != gpu:
                    continue
                if backend and r.get("backend") != backend:
                    continue
                rows.append(r)
    return rows


def usable(rows):
    """Rows that can be fitted, and why the rest cannot."""
    keep, dropped = [], {}
    for r in rows:
        if r.get("status") != "ok":
            dropped[r.get("status", "?")] = dropped.get(r.get("status", "?"), 0) + 1
            continue
        # recomputed rather than trusted: rows written before suspect_reason()
        # existed carry no flag, and the inputs it needs are all stored
        why = r.get("suspect") or suspect_reason(r)
        if why:
            k = why.split(" - ")[0][:40]
            dropped[k] = dropped.get(k, 0) + 1
            continue
        if r["log"]["buffers"].get("CUDA0.compute") is None:
            dropped["no CUDA0.compute reported"] = dropped.get("no CUDA0.compute reported", 0) + 1
            continue
        keep.append(r)
    return keep, dropped


# ---------------------------------------------------------------------------
# What a row says about the model that produced it
# ---------------------------------------------------------------------------
# Taken from llama.cpp's own print_info rather than from the GGUF, so a sweep file
# is self-contained: it can be scored on a machine that does not have the models.
def _int(v, default=0):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


_CFG_CACHE = {}


def _cfg_from_file(name):
    """Fall back to the model file when a row carries no print_info block.

    Rows recorded by an older parser have an empty `info`, and dropping them would
    throw away good measurements over a parsing bug that has since been fixed.
    Matched on basename, like calib._model_facts()."""
    if name in _CFG_CACHE:
        return _CFG_CACHE[name]
    from .gguf import load_gguf
    from .model import extract_config
    from .lmstudio import default_models_dir
    found = None
    for dp, _dn, fns in os.walk(default_models_dir() or ""):
        if name in fns:
            found = os.path.join(dp, name)
            break
    cfg = None
    if found:
        try:
            cfg = extract_config(load_gguf(found))
        except Exception:
            cfg = None
    _CFG_CACHE[name] = cfg
    return cfg


def row_cfg(row):
    """A cfg dict shaped like model.extract_config()'s for this row's model.

    The GGUF is preferred when the file is present, because llama.cpp's print_info
    block does not carry everything: it prints `n_ff = 0` for a MoE and never prints
    the expert FFN width at all, so a model scored from the log alone gets a zero
    where its largest activation tensor should be. print_info remains the fallback,
    which keeps a sweep file scoreable on a machine that has no models on it."""
    cfg = _cfg_from_file(row.get("model") or "")
    if cfg:
        return cfg
    info = (row.get("log") or {}).get("info") or {}
    n_layers = _int(info.get("n_layer"), row.get("n_layers") or 0)
    swa = set((row.get("log") or {}).get("swa_layers") or [])
    return {
        "arch": info.get("arch") or row.get("arch") or "",
        "n_layers": n_layers,
        "hidden": _int(info.get("n_embd")),
        "n_head": _int(info.get("n_head")),
        "n_vocab": _int(info.get("n_vocab")),
        "ffn_len": _int(info.get("n_ff")),
        "n_expert": _int(info.get("n_expert")),
        "n_expert_used": _int(info.get("n_expert_used")),
        "n_swa": _int(info.get("n_swa")),
        "head_dim_k": _int(info.get("n_embd_head_k")),
        "head_dim_v": _int(info.get("n_embd_head_v")),
        "swa_layers": sorted(swa),
        # compute_buffer_terms() only checks this for emptiness
        "attn_layers": [i for i in range(n_layers)],
    }


def observed(row):
    """The target: what the allocator put in VRAM for the compute graph."""
    return row["log"]["buffers"]["CUDA0.compute"]


def observed_host(row):
    return row["log"]["buffers"].get("CUDA_Host.compute", 0.0)


def observed_floor(row):
    """Process VRAM the allocator never reported: CUDA context, driver, kernels."""
    return row.get("floor_mib")


def layers_on_gpu(row):
    """How many blocks actually landed on the GPU, from llama.cpp's own placement
    lines - not from the -ngl we asked for, which it is free to override."""
    layers = (row.get("log") or {}).get("layers") or {}
    if layers:
        return sum(1 for d in layers.values() if re.match(r"(CUDA|ROCm|Vulkan|Metal|SYCL)\d", d))
    return min(row["config"]["ngl"], row.get("n_layers") or row["config"]["ngl"])


def predict_graph(row, coeffs=None):
    """The shipped model's GPU-side compute buffer for this row, in MiB.

    `graph` excludes the floor by construction, which is what makes it comparable:
    the floor is BY DEFINITION the part the allocator never reports - CUDA context,
    driver, lazily-loaded kernel modules - while CUDA0.compute is what the allocator
    does report. Charging the model for a term the measurement cannot contain is,
    on a small model, most of the answer."""
    c = row["config"]
    cfg = row_cfg(row)
    terms = compute_buffer_terms(cfg, c["ctx"], c["ub"], c["fa"], c["seq"], c["kv"],
                                 coeffs=coeffs)
    split = graph_is_split(cfg["n_layers"], c["ngl"], c.get("ncmoe"))
    return terms["graph"] + (terms["split_extra"] if split else 0.0)


def predict_floor(row, coeffs=None):
    """The floor this row should have shown, for scoring against floor_mib."""
    c = row["config"]
    terms = compute_buffer_terms(row_cfg(row), c["ctx"], c["ub"], c["fa"], c["seq"],
                                 c["kv"], coeffs=coeffs)
    return terms["floor_base"] + terms["floor_per_layer"] * layers_on_gpu(row)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(rows, predict_fn, target_fn=observed):
    """Mean and worst absolute percentage error, plus the worst offender."""
    errs, worst, worst_row = [], 0.0, None
    for r in rows:
        t = target_fn(r)
        if t is None or t <= 0:
            continue
        e = abs(predict_fn(r) - t) / t * 100.0
        errs.append(e)
        if e > worst:
            worst, worst_row = e, r
    if not errs:
        return {"n": 0, "mean": 0.0, "worst": 0.0, "worst_label": ""}
    return {"n": len(errs), "mean": sum(errs) / len(errs), "worst": worst,
            "worst_label": label(worst_row) if worst_row else ""}


_PATHS = {}


def model_path(name):
    """Locate the GGUF a sweep row came from, by basename."""
    if not _PATHS:
        from .lmstudio import default_models_dir
        for dp, _dn, fns in os.walk(default_models_dir() or ""):
            for fn in fns:
                if fn.endswith(".gguf"):
                    _PATHS.setdefault(fn, os.path.join(dp, fn))
    return _PATHS.get(name)


def predict_total(row):
    """The planner's total GPU VRAM for this row's exact config, in MiB.

    This is the end-to-end number - weights + KV + recurrent + compute + floor, the
    thing a user reads off the screen - scored against the process counter the sweep
    recorded. The compute buffer is only ever a slice of it, and a model can be
    right about the slice while being wrong about the total, which is precisely what
    a compensating pair of errors looks like."""
    from .plan import analyze
    p = model_path(row.get("model") or "")
    if not p:
        return None
    c = row["config"]
    r = analyze(p, c["ctx"], c["kv"], c["ub"], c["fa"],
                vram_budget_mib=1 << 20, ram_budget_mib=1 << 20, gpu_reserve_mib=0,
                compute_override_mib=None, safety_pct=0, n_seq=c["seq"],
                gpu_layers_override=c["ngl"], include_mmproj=False,
                n_cpu_moe_override=(c.get("ncmoe") or None))
    return r["plan"].get("vram_used_mib")


def label(row):
    c = row["config"]
    return "%s ctx %d ngl %d ub %d np %d fa %d %s%s" % (
        row["model"][:26], c["ctx"], c["ngl"], c["ub"], c["seq"], c["fa"], c["kv"],
        " ncmoe %d" % c["ncmoe"] if c.get("ncmoe") else "")


def cross_validate(rows, fit_fn, predict_fn, group_fn, group_name):
    """Hold out one group at a time; fit on the rest; score on the held-out group.

    fit_fn(train_rows) -> coeffs, predict_fn(row, coeffs) -> MiB. Returns the
    pooled held-out score plus a per-group breakdown, because a mean over groups
    of very different sizes hides the one that is badly wrong."""
    groups = {}
    for r in rows:
        groups.setdefault(group_fn(r), []).append(r)
    if len(groups) < 2:
        return {"name": group_name, "n_groups": len(groups), "skipped":
                "need at least 2 groups to hold one out", "per_group": []}
    pooled, per_group = [], []
    for g, held in sorted(groups.items()):
        train = [r for r in rows if group_fn(r) != g]
        if not train or not held:
            continue
        coeffs = fit_fn(train)
        s = score(held, lambda r: predict_fn(r, coeffs))
        per_group.append({"group": str(g), "n": s["n"], "mean": s["mean"],
                          "worst": s["worst"], "worst_label": s["worst_label"]})
        for r in held:
            t = observed(r)
            if t and t > 0:
                pooled.append(abs(predict_fn(r, coeffs) - t) / t * 100.0)
    return {"name": group_name, "n_groups": len(groups),
            "n": len(pooled),
            "mean": (sum(pooled) / len(pooled)) if pooled else 0.0,
            "worst": max(pooled) if pooled else 0.0,
            "per_group": per_group}


def group_by_axis_top(rows, axis):
    """Split rows into "at the top of this axis" and "below it", for extrapolation.

    Holding out the top of a range asks the question interpolation never does:
    fitted only on contexts up to 131k, what does the model say about 262k?"""
    vals = sorted({r["config"][axis] for r in rows})
    if len(vals) < 3:
        return None
    top = vals[-1]
    return lambda r: ("top" if r["config"][axis] == top else "below")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(rows=None, fit_fn=None, predict_fn=None, log=print):
    """Print the honest scorecard for the current compute model."""
    all_rows = rows if rows is not None else load_sweep()
    if not all_rows:
        log("No sweep data. Run: python -m vram_planner --sweep")
        return None
    keep, dropped = usable(all_rows)
    log("sweep rows      : %d total, %d usable" % (len(all_rows), len(keep)))
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        log("  dropped %4d  : %s" % (v, k))
    if not keep:
        return None
    models = sorted({r["model"] for r in keep})
    archs = sorted({row_cfg(r)["arch"] for r in keep})
    log("models          : %d (%s)" % (len(models), ", ".join(archs)))

    # The shipped model, scored in sample. Reported only to be compared against
    # the held-out numbers below - on its own it means very little.
    fit_fn = fit_fn or (lambda train: dict(CB_DEFAULTS))
    predict_fn = predict_fn or (lambda r, c: predict_graph(r, c))
    ins = score(keep, lambda r: predict_fn(r, fit_fn(keep)))
    log("")
    log("compute buffer (CUDA0.compute), %d rows" % ins["n"])
    log("  in sample     : %5.1f%% mean  %6.1f%% worst   (%s)"
        % (ins["mean"], ins["worst"], ins["worst_label"]))

    cv = cross_validate(keep, fit_fn, predict_fn,
                        lambda r: row_cfg(r)["arch"] or r["model"], "leave-one-model-out")
    if cv.get("skipped"):
        log("  held out      : %s" % cv["skipped"])
        log("                  ONE ARCHITECTURE IS NOT A VALIDATION SET. Sweep another "
            "model before believing any of this.")
    else:
        log("  held out      : %5.1f%% mean  %6.1f%% worst   (%s, %d groups)"
            % (cv["mean"], cv["worst"], cv["name"], cv["n_groups"]))
        for g in cv["per_group"]:
            log("      %-22s n=%-4d %5.1f%% mean  %6.1f%% worst"
                % (g["group"], g["n"], g["mean"], g["worst"]))

    # The floor, scored in MiB rather than percent: it is an additive term of a few
    # hundred MiB, so a percentage of it is not the quantity anyone cares about -
    # what matters is how many MiB of the budget it misplaces.
    fr = [r for r in keep if r.get("floor_mib") is not None]
    if fr:
        # Explicitly the shipped prior, never the user's store: this is a scorecard
        # for the model as distributed. Scoring it under a local calibration would
        # report how well that machine's own measurements describe themselves.
        errs = sorted(((abs(predict_floor(r, dict(CB_DEFAULTS)) - r["floor_mib"]), r)
                       for r in fr), key=lambda x: -x[0])
        log("")
        log("floor (process VRAM the allocator never reports), %d rows" % len(fr))
        log("  mean |error| : %5.1f MiB   worst %5.1f MiB   (%s)"
            % (sum(e for e, _ in errs) / len(errs), errs[0][0], label(errs[0][1])))
        log("  observed     : %5.1f .. %5.1f MiB"
            % (min(r["floor_mib"] for r in fr), max(r["floor_mib"] for r in fr)))

    # The end-to-end number, against the process counter. This is the one a user
    # actually reads, and the one a pair of compensating errors hides inside.
    tr = [r for r in keep if r.get("proc_vram_mib")]
    if tr:
        errs, by_model = [], {}
        for r in tr:
            pred = predict_total(r)
            if pred is None:
                continue
            e = abs(pred - r["proc_vram_mib"]) / r["proc_vram_mib"] * 100.0
            errs.append((e, r, pred))
            by_model.setdefault(r["model"], []).append(e)
        if errs:
            errs.sort(key=lambda x: -x[0])
            log("")
            log("TOTAL GPU VRAM (what the planner reports), %d rows" % len(errs))
            log("  mean %5.1f%%  worst %6.1f%%   (%s: predicted %.0f, measured %.0f)"
                % (sum(e for e, _, _ in errs) / len(errs), errs[0][0],
                   label(errs[0][1]), errs[0][2], errs[0][1]["proc_vram_mib"]))
            for m in sorted(by_model):
                es = by_model[m]
                log("      %-28s n=%-4d %5.1f%% mean  %6.1f%% worst"
                    % (m[:28], len(es), sum(es) / len(es), max(es)))

    for axis in ("ctx", "ub", "ngl"):
        gf = group_by_axis_top(keep, axis)
        if not gf:
            continue
        cva = cross_validate(keep, fit_fn, predict_fn, gf, "top-of-%s held out" % axis)
        top = next((g for g in cva.get("per_group") or [] if g["group"] == "top"), None)
        if top:
            log("  extrapolate %-4s: %5.1f%% mean  %6.1f%% worst on the top %s alone (n=%d)"
                % (axis, top["mean"], top["worst"], axis, top["n"]))
    return {"in_sample": ins, "cv": cv, "n": len(keep)}
