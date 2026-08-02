"""Fitting the compute-buffer model to measurements from this machine."""
import json, os, time
from .const import _mib
from .gguf import load_gguf
from .model import extract_config
from .kv import is_swa_layer, kv_bytes_per_token_layer, swa_cache_len
from .compute import CB_CONST_PER_KHID, CB_CTX_QUANT_BYTES, CB_DEFAULTS, CB_MASK_PER_UB_TOK, CB_SPLIT_GRAPH_MIB, CB_SPLIT_PER_TOKEN, graph_is_split, output_head_on_gpu
from .paths import _user_file
from .gpu import gpu_list
from .lmstudio import current_backend, default_models_dir

_CALIB_CACHE = {}                     # (gpu, backend) key -> fitted coefficient dict

_CALIB_LOADED = False                 # the store is read once, lazily


def _ensure_calibration():
    """Load the store on first use. Without this, calibration only ever applied
    inside serve(), so importing the module and calling analyze() silently fell
    back to the shipped defaults."""
    global _CALIB_LOADED
    if not _CALIB_LOADED:
        _CALIB_LOADED = True          # set first: a failed load must not retry forever
        try:
            refresh_calibration()
        except Exception:
            pass


def calib_coeffs(gpu=None):
    """Coefficients to use right now: the fit for this machine if one exists, else
    the shipped prior. Never a partial dict - unfitted terms keep their default."""
    _ensure_calibration()
    c = dict(CB_DEFAULTS)
    c.update((_CALIB_CACHE.get(gpu if gpu is not None else _active_gpu())
              or {}).get("coeffs") or {})
    return c


def _active_gpu():
    g = gpu_list()
    return (g[0].get("name") or "") if g else ""


# ---------------------------------------------------------------------------
# Per-machine calibration of the compute-buffer coefficients
# ---------------------------------------------------------------------------
# Weights and KV are read from the GGUF and need no calibration. The four
# overhead coefficients do: they vary with GPU, driver and llama.cpp build. Each
# Measure yields one training row (measured process VRAM minus the exact terms),
# so the tool refits from the user's own runs instead of asking anyone to
# understand the constants.
#
# The fit is PROGRESSIVE: it only frees as many coefficients as the data can
# actually identify, and leaves the rest at the shipped prior. One measurement
# moves the constant (the dominant per-machine term); several spread over context
# lengths also move the ctx slope; varied ubatch or model width moves the
# activation slope; a flash-attention-off run moves the score term. Fitting four
# coefficients to two points would be worse than not fitting at all.
CALIB_TERMS = ["const", "ctx", "act", "nofa"]   # freed in this order

CALIB_MIN_SPREAD = 0.25                          # min relative range to free a term


def _calib_store():
    return _user_file("vram_calibration.json")


def load_calibration():
    try:
        return json.load(open(_calib_store(), encoding="utf-8"))
    except Exception:
        return {"rows": []}


def save_calibration(data):
    try:
        json.dump(data, open(_calib_store(), "w", encoding="utf-8"), indent=1)
    except Exception:
        pass


CALIB_SCHEMA = 2                      # 2 adds n_layers + n_vocab to every row


# A measurement taken with the card full is a ceiling, not a reading: llama.cpp
# spills the remainder to shared system memory and the counter reports the cap.
# Detecting that needs the free VRAM at measure time, which only rows recorded by
# this version carry - a high process figure alone proves nothing, since a 12 GiB
# card really does hand out 11.2 GiB to one process.
CALIB_MIN_FREE_MIB = 192.0

# The same condition seen from inside the data, and the only test available to
# rows recorded before free VRAM was stored: a group whose measured VRAM barely
# moves while its exact terms move a lot.
CALIB_FLAT_RATIO   = 0.25             # measured span vs exact span

CALIB_FLAT_MIN_MIB = 200.0            # ...only when the exact span is real


def mark_unreliable(rows, gpu_totals=None):
    """Flag rows whose measured_mib cannot be read as "what this config needed".

    `overhead_mib` is measured minus exact, so a measurement that did not respond
    to the config does not produce a weak observation - it produces a fictitious
    overhead that moves the opposite way from the truth. Five rows of one model
    here spanned 2458 MiB of exact terms against 17 MiB of measured VRAM, which
    fitted as the compute buffer *shrinking* by 2.4 GiB as layers were added.

    Sets or clears row["unreliable"]. Nothing is deleted: re-measuring on a card
    with room rescues the row, and the reason is worth showing to whoever ran it."""
    groups = {}
    for r in rows:
        r.pop("unreliable", None)
        free = r.get("gpu_free_mib")
        if free is not None and free < CALIB_MIN_FREE_MIB:
            r["unreliable"] = ("only %.0f MiB of VRAM was free at measure time - the card "
                               "was at the wall, so the reading is the cap, not the need"
                               % free)
        groups.setdefault((r.get("gpu"), r.get("model"), r.get("ctx"), r.get("ub"),
                           r.get("fa"), r.get("kv_type")), []).append(r)
    for rs in groups.values():
        if len(rs) < 2:
            continue
        span = lambda f: max(x.get(f, 0.0) for x in rs) - min(x.get(f, 0.0) for x in rs)
        exact_span = span("exact_mib")
        if exact_span > CALIB_FLAT_MIN_MIB and \
                span("measured_mib") < CALIB_FLAT_RATIO * exact_span:
            for r in rs:
                r.setdefault("unreliable",
                             "measured VRAM moved %.0f MiB while the exact terms moved "
                             "%.0f - these rows are not independent measurements"
                             % (span("measured_mib"), exact_span))
    return rows


def _model_facts(name, roots=None, _cache={}):
    """(n_layers, n_vocab) read from the named model file, or None if it is gone.

    Matched on basename: the store records that and nothing else, and a model is
    not going to be two different architectures under one file name."""
    if name in _cache:
        return _cache[name]
    found = None
    for root in (roots if roots is not None else [default_models_dir()]):
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if name in files:
                found = os.path.join(dirpath, name)
                break
        if found:
            break
    facts = None
    if found:
        try:
            cfg = extract_config(load_gguf(found))
            facts = (cfg.get("n_layers") or 0, cfg.get("n_vocab") or 0)
        except Exception:
            facts = None
    _cache[name] = facts
    return facts


def migrate_calibration(data):
    """Backfill n_layers / n_vocab onto rows written before they were stored, in
    place. Returns True if anything changed.

    Sources in descending authority: the model file if it is still on disk, then
    another row for the same file name, then another row with the same
    architecture AND the same residual width - the same base model at a different
    quant, which cannot differ in either field. Rows that resolve to none of
    these keep their data and are skipped by the fit rather than deleted: a
    re-measure, or the file reappearing, rescues them later."""
    rows = data.get("rows") or []
    if not rows:
        return False
    snap = lambda: [(r.get("n_layers"), r.get("n_vocab"), r.get("unreliable"))
                    for r in rows]
    before = snap()

    for r in rows:
        if r.get("n_layers") and r.get("n_vocab"):
            continue
        facts = _model_facts(r.get("model") or "")
        if facts:
            r["n_layers"] = r.get("n_layers") or facts[0]
            r["n_vocab"] = r.get("n_vocab") or facts[1]

    # ...then propagate across rows, by file name first and architecture second.
    for key in (lambda r: (r.get("model"),),
                lambda r: (r.get("arch"), r.get("hidden"))):
        known = {}
        for r in rows:
            k = key(r)
            for f in ("n_layers", "n_vocab"):
                if r.get(f):
                    known.setdefault(k, {}).setdefault(f, r[f])
        for r in rows:
            for f, val in (known.get(key(r)) or {}).items():
                if not r.get(f):
                    r[f] = val

    mark_unreliable(rows)
    data["schema"] = CALIB_SCHEMA
    return snap() != before


def _design(row):
    """Regressors for one observation, in MiB, matching compute_buffer_terms.

    The ctx regressor is the raw context length, not the KV bytes it implies: the
    compute buffer was measured to scale with context flat, independently of how
    big the cache is. Rows written before this change carry a `kv_tok_ctx` field
    that is simply ignored now."""
    return {"const": 1.0,
            "ctx":   _mib(row["ctx"]),
            "act":   _mib(row["hidden"] * row["ub"]),
            "nofa":  0.0 if row["fa"] else _mib(row["n_head"] * row["ub"] * row["ctx"])}


def _row_flags(row):
    """Where the graph ran for one stored observation, and whether the row can be
    fitted at all.

    A row is usable only if its placement is knowable. `n_layers` decides both
    predicates, so a row without it cannot be fitted - guessing "unsplit" is what
    let the split surcharge hide in `const`. `n_vocab` only matters when the head
    was on the GPU, which is why partial-offload rows recorded before it was
    stored stay usable: their logits tensor was never charged to VRAM."""
    n_layers, ngl = row.get("n_layers"), row.get("ngl")
    split = graph_is_split(n_layers, ngl, row.get("n_cpu_moe"))
    if row.get("unreliable"):
        return {"ok": False, "why": row["unreliable"], "split": split,
                "output_on_gpu": output_head_on_gpu(n_layers, ngl),
                "n_vocab": row.get("n_vocab") or 0}
    if not n_layers or ngl is None:
        return {"ok": False, "why": "no n_layers recorded",
                "split": split, "output_on_gpu": False, "n_vocab": 0}
    on_gpu = output_head_on_gpu(n_layers, ngl)
    n_vocab = row.get("n_vocab") or 0
    if on_gpu and not n_vocab:
        return {"ok": False, "why": "head was on the GPU but no n_vocab recorded",
                "split": split, "output_on_gpu": True, "n_vocab": 0}
    return {"ok": True, "why": "", "split": split,
            "output_on_gpu": on_gpu, "n_vocab": n_vocab}


def _struct_offset(row):
    """Parts of the observation that are structural rather than per-machine: the
    attention mask, the quantised-cache surcharge, the split-graph penalty, and
    the logits tensor. They are not fitted, so they come off the observation
    before the additive base and the remaining slopes are solved for.

    Every term the planner charges to VRAM must appear either here or in
    _design(). `overhead_mib` is measured VRAM minus the exact terms, so it
    already contains split_extra and logits; anything charged at prediction time
    and not subtracted here gets absorbed into `const` and then billed twice."""
    flags = _row_flags(row)
    off = (CB_CONST_PER_KHID * row["hidden"] / 1000.0
           + _mib(CB_MASK_PER_UB_TOK * row["ub"] * row["ctx"]))
    if row.get("kv_type", "f16") != "f16":
        off += _mib(CB_CTX_QUANT_BYTES * row["ctx"])
    if flags["split"]:
        off += CB_SPLIT_GRAPH_MIB + _mib(CB_SPLIT_PER_TOKEN * row["ctx"])
    if flags["output_on_gpu"]:
        off += _mib(4.0 * row["ub"] * flags["n_vocab"])
    return off


def _solve(A, y):
    """Least squares by normal equations. Returns None if ill-conditioned."""
    n = len(A[0])
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)]
         + [sum(A[k][i] * y[k] for k in range(len(A)))] for i in range(n)]
    scale = [max(abs(M[i][i]), 1e-12) for i in range(n)]
    for i in range(n):
        pv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pv][i]) < 1e-9 * scale[i]:
            return None
        M[i], M[pv] = M[pv], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_calibration(rows, prior=None):
    """Fit what the data supports. Returns {coeffs, free, n, residual_pct} or None."""
    prior = dict(prior or CB_DEFAULTS)
    rows = [r for r in rows if r.get("overhead_mib") is not None and r.get("hidden")]
    # A row whose backend placement cannot be reconstructed is not a weak
    # observation, it is an unreadable one - fitting it means fitting a term that
    # may or may not be in the measurement. Skip, do not guess.
    usable = [r for r in rows if _row_flags(r)["ok"]]
    skipped = len(rows) - len(usable)
    rows = usable
    if len(rows) < 1:
        return None
    des = [_design(r) for r in rows]

    # a term is identifiable only if its regressor actually varies across the rows
    # A term is identifiable only if the knob it belongs to actually varied - and
    # the knob, not the regressor. The activation column is hidden*ubatch, so a
    # handful of different models measured at one ubatch makes it look varied when
    # all that changed was hidden size, which is confounded with the per-hidden
    # structural constant. Fitting it from that produced a coefficient 9x the
    # prior. So require spread in the knob a user can actually turn.
    knob = {"ctx": lambda r: float(r["ctx"]),
            "act": lambda r: float(r["ub"]),
            "nofa": lambda r: 0.0 if r["fa"] else 1.0}
    free = ["const"]
    for t in CALIB_TERMS[1:]:
        vals = [knob[t](r) for r in rows]
        lo, hi = min(vals), max(vals)
        if hi > 0 and (hi - lo) / hi >= CALIB_MIN_SPREAD:
            free.append(t)
    # never free more terms than we have rows to support (plus a degree of freedom)
    while len(free) > 1 and len(rows) < len(free) + 1:
        free.pop()

    coeffs = dict(prior)
    while free:
        fixed = [t for t in CALIB_TERMS if t not in free]
        A = [[d[t] for t in free] for d in des]
        y = [r["overhead_mib"] - _struct_offset(r) - sum(prior[t] * d[t] for t in fixed)
             for r, d in zip(rows, des)]
        x = _solve(A, y)
        # A fit is only accepted if it is finite and physically possible: these
        # terms are byte counts, so a negative one means the solver absorbed error
        # rather than measured anything. Slopes are also held within an order of
        # magnitude of the prior - beyond that the data is being extrapolated, and
        # the shipped default is the better answer.
        good = x is not None and all(v == v and abs(v) < 1e9 for v in x)
        if good:
            cand = dict(zip(free, x))
            for t, val in cand.items():
                if t == "const":
                    # An additive offset, not a byte count on its own: the physical
                    # pool is const + the structural per-hidden part, so a small
                    # negative const is a legitimate fit that trims a slightly
                    # over-predicting prior. Only reject nonsense magnitudes, and
                    # check the total stays positive on the rows we actually have.
                    if abs(val) > 20000 or any(val + _struct_offset(r) < 0 for r in rows):
                        good = False
                elif val < 0 or (prior[t] > 0 and not (0.1 <= val / prior[t] <= 10.0)):
                    good = False
        if good:
            coeffs = dict(prior)
            coeffs.update(cand)
            break
        free.pop()                       # drop the least-identifiable term and retry
    if not free:
        return None

    # in-sample residual: honest label, it is a fit quality not a prediction error
    errs = []
    for r, d in zip(rows, des):
        pred = _struct_offset(r) + sum(coeffs[t] * d[t] for t in CALIB_TERMS)
        base = max(1.0, r["overhead_mib"] + r.get("exact_mib", 0.0))
        errs.append(abs(pred - r["overhead_mib"]) / base * 100.0)
    return {"coeffs": coeffs, "free": free, "n": len(rows),
            "skipped_rows": skipped,
            "residual_pct": round(sum(errs) / len(errs), 1)}


def refresh_calibration(gpu=None):
    """Refit every GPU present in the store and publish the results.

    Rows are also filtered by llama.cpp build: a backend upgrade changes graph
    allocation, so mixing builds fits a curve through two different machines. If
    the current build is known and any row matches it, only those rows are used;
    otherwise everything for that GPU is used and the mismatch is reported."""
    data = load_calibration()
    if migrate_calibration(data):
        save_calibration(data)
    cur = current_backend()
    by_gpu = {}
    for r in data.get("rows", []):
        by_gpu.setdefault(r.get("gpu") or "", []).append(r)
    _CALIB_CACHE.clear()
    for g, rows in by_gpu.items():
        # Only drop a row when its build is KNOWN and different. Rows recorded
        # before builds were tracked carry no backend; discarding those would
        # throw away good measurements from this very machine.
        used  = [r for r in rows if not cur or not r.get("backend") or r["backend"] == cur]
        stale = len(rows) - len(used)
        f = fit_calibration(used)
        if f:
            f["stale_rows"] = stale
            f["backend"] = cur
            _CALIB_CACHE[g] = f
    return _CALIB_CACHE.get(gpu if gpu is not None else _active_gpu())


def calibration_status(gpu=None):
    _ensure_calibration()
    g = gpu if gpu is not None else _active_gpu()
    f = _CALIB_CACHE.get(g)
    if not f:
        return {"calibrated": False, "gpu": g, "n": 0, "coeffs": dict(CB_DEFAULTS),
                "free": [], "residual_pct": None, "stale_rows": 0,
                "skipped_rows": 0, "backend": current_backend()}
    return {"calibrated": True, "gpu": g, "n": f["n"], "coeffs": f["coeffs"],
            "free": f["free"], "residual_pct": f["residual_pct"],
            "stale_rows": f.get("stale_rows", 0),
            "skipped_rows": f.get("skipped_rows", 0), "backend": f.get("backend", "")}


def record_calibration(path, ctx, kv_type, n_ubatch, n_seq, flash_attn, ngl,
                       include_mmproj, measured_mib, gpu="", n_cpu_moe=0):
    """Turn one measurement into a training row and refit. The exact terms are
    recomputed here rather than trusted from the client, so a stale plan on screen
    cannot poison the store."""
    from .plan import analyze         # deferred: plan reads calibration_status()
    r = analyze(path, ctx, kv_type, n_ubatch, flash_attn,
                vram_budget_mib=1 << 20, ram_budget_mib=1 << 20, gpu_reserve_mib=0,
                compute_override_mib=0.001, safety_pct=0, n_seq=n_seq,
                gpu_layers_override=ngl, include_mmproj=include_mmproj,
                n_cpu_moe_override=(n_cpu_moe or None))
    cfg, p = r["config"], r["plan"]
    exact = (p.get("gpu_weights_mib", 0.0) + p.get("gpu_kv_mib", 0.0)
             + p.get("gpu_recurrent_mib", 0.0) + p.get("mmproj_mib", 0.0))
    overhead = measured_mib - exact
    if overhead <= 0:
        return {"error": "Measured %.0f MiB is below the exact terms (%.0f MiB) - the "
                         "loaded model is not the one analysed." % (measured_mib, exact)}
    attn = cfg["attn_layers"]
    kvg = sum(kv_bytes_per_token_layer(cfg, "f16", i) for i in attn if not is_swa_layer(cfg, i))
    kvs = sum(kv_bytes_per_token_layer(cfg, "f16", i) for i in attn if is_swa_layer(cfg, i))
    # Record what the card had, so a reading taken at the wall can be recognised
    # as one later - by mark_unreliable() here, and by any future rule.
    gpu_name = gpu or _active_gpu()
    card = next((g for g in gpu_list(fresh=True) if g.get("name") == gpu_name), {})
    row = {"when": int(time.time()), "gpu": gpu_name,
           "gpu_total_mib": card.get("total_mib") or 0.0,
           "gpu_free_mib": card.get("free_mib"),
           "backend": current_backend(fresh=True),
           "model": os.path.basename(path), "arch": cfg["arch"],
           "ctx": ctx, "ub": n_ubatch, "n_seq": n_seq, "fa": bool(flash_attn),
           "ngl": ngl, "n_cpu_moe": n_cpu_moe, "kv_type": kv_type,
           "hidden": cfg["hidden"] or 4096, "n_head": cfg["n_head"] or 32,
           "n_layers": cfg["n_layers"] or 0, "n_vocab": cfg.get("n_vocab") or 0,
           "kv_tok_ctx": kvg * ctx + kvs * swa_cache_len(cfg, ctx, n_ubatch, n_seq, flash_attn),
           "exact_mib": round(exact, 1), "measured_mib": round(measured_mib, 1),
           "overhead_mib": round(overhead, 1)}
    data = load_calibration()
    rows = data.setdefault("rows", [])
    # One row per distinct config. n_cpu_moe belongs in the key: with expert
    # offload ngl is pinned at n_layers, so without it every --n-cpu-moe sweep of
    # one model collapsed onto a single row and each measurement threw away the
    # last - exactly the sweep that identifies the expert-layer slope.
    def _key(x):
        return (x.get("gpu"), x.get("model"), x.get("ctx"), x.get("ub"),
                x.get("n_seq"), x.get("fa"), x.get("ngl"), x.get("kv_type"),
                x.get("n_cpu_moe") or 0)
    key = _key(row)
    rows[:] = [x for x in rows if _key(x) != key]
    rows.insert(0, row)
    del rows[400:]
    data["schema"] = CALIB_SCHEMA
    mark_unreliable(rows)
    save_calibration(data)
    refresh_calibration()
    return {"ok": True, "row": row, "status": calibration_status(row["gpu"]),
            "unreliable": row.get("unreliable", "")}
