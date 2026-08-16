"""One recommended config, and why it is not the other one.

The planner and the speed sweep answer different questions and, until this
module, said so nowhere. The planner returns the LARGEST SPLIT THAT FITS - both
_plan_dense and _plan_moe search for the first config whose VRAM total lands
under the budget and stop there. The sweep returns the FASTEST ROW MEASURED.
Those coincide only if tok/s is monotone in offload, and it is not: a spilled
row loads and reports ok while running off a cliff, the MTP draft cache moves
the wall a whole rung, and ubatch and speculation move throughput without
moving any number the planner can compute.

So the two disagreeing is not a bug to be papered over - it is the honest state
of affairs, and the useful thing is to say which one is being recommended and
name every reason they differ.

Pure: no I/O, no subprocess, no GPU. Same rule as bench.py's findings code, and
for the same reason - the CLI and the browser reach one conclusion by
construction rather than by two people remembering to keep them in step.
"""
from .bench import comparable, rank_rows, trustworthy, _ANY

# Knobs a measured row can carry that a plan has no field for. The plan's
# vocabulary is ngl / ncmoe / ctx / kv; a row also records how the batch was
# shaped and whether it was speculating. When a winner differs on one of these
# the planner did not "get it wrong" - it was never asked.
UNPLANNABLE = (
    ("ub", "ubatch", 512),
    ("spec", "speculation", "none"),
    ("spec_n_max", "draft depth", 0),
    ("mmproj_offload", "projector placement", True),
)

# Below this the two configs are the same answer and the deltas are noise.
NGL_SAME = 0


def _cfg_of(row):
    return dict((row or {}).get("config") or {})


def plan_config(plan_result):
    """The planner's answer in a measured row's vocabulary, so the two compare.

    Everything the plan cannot express is left absent rather than defaulted -
    an absent key means "the planner had no opinion", which is exactly the
    distinction the `axis` delta reports.
    """
    plan = (plan_result or {}).get("plan") or {}
    inp = (plan_result or {}).get("inputs") or {}
    c = {"ngl": plan.get("n_gpu_layers"), "ncmoe": plan.get("n_cpu_moe") or 0}
    if inp.get("context") is not None:
        c["ctx"] = inp["context"]
    if inp.get("kv_type"):
        c["kv"] = inp["kv_type"]
    if inp.get("n_ubatch") is not None:
        c["ub"] = inp["n_ubatch"]
    if inp.get("n_seq") is not None:
        c["seq"] = inp["n_seq"]
    if inp.get("flash_attn") is not None:
        c["fa"] = bool(inp["flash_attn"])
    return c


def _conditions_match(row, planned):
    """Does this row answer the question currently on screen?

    Only the keys a PLAN has an opinion about - context, KV quant, sequences,
    flash attention. Deliberately not fill, n_predict, repeat or the samplers:
    those are properties of how a campaign was run and a plan has no value for
    any of them, so demanding they agree would reject every row.
    """
    c = row.get("config") or {}
    for key in ("ctx", "kv", "seq", "fa"):
        if key not in planned or key not in c:
            continue
        a, b = c[key], planned[key]
        if key == "fa":
            a, b = bool(a), bool(b)
        if a != b:
            return False
    return True


def _fmt_mib(v):
    return "-" if v is None else "{:,.0f} MiB".format(float(v))


def _budget_delta(plan_result, sweep_budget_mib):
    """Was the plan priced against a different amount of VRAM than the rows?

    This was the single largest source of disagreement and the one nobody could
    see: the browser prefilled its budget from FREE VRAM at page load with a
    zero reserve, while the sweep seeded its ladder from TOTAL with a 512 MiB
    reserve. Same model, same question, several -ngl rungs apart.
    """
    inp = (plan_result or {}).get("inputs") or {}
    have = inp.get("vram_budget_mib")
    if have is None or not sweep_budget_mib:
        return None
    if abs(float(have) - float(sweep_budget_mib)) <= 64:      # same basis
        return None
    return {"kind": "budget",
            "text": "The plan was priced against %s of VRAM; the measurements were "
                    "taken with %s. A budget that differs by that much moves the "
                    "split on its own, before any question of speed."
                    % (_fmt_mib(have), _fmt_mib(sweep_budget_mib))}


def _axis_deltas(won, planned):
    """Knobs the winner sets that a plan cannot predict."""
    out = []
    for key, label, default in UNPLANNABLE:
        wv = won.get(key, default)
        if wv is None:
            wv = default
        pv = planned.get(key, default)
        if pv is None:
            pv = default
        if wv == pv:
            continue
        if key == "mmproj_offload":
            wv = "VRAM" if wv is not False else "system RAM"
            pv = "VRAM" if pv is not False else "system RAM"
        out.append((label, "%s (the plan assumes %s)" % (wv, pv)))
    if not out:
        return None
    return {"kind": "axis",
            "text": "Knobs no plan can predict: "
                    + "; ".join("%s %s" % (a, b) for a, b in out)
                    + ". Speculation has no acceptance rate until it runs, and "
                      "prompt processing is compute bound and is not modelled at all."}


def _stale_delta(won, planned):
    """Does the winning row answer the question currently on screen?"""
    bad = []
    for key, label in (("ctx", "context"), ("kv", "KV quant"),
                       ("seq", "sequences"), ("fa", "flash attention")):
        if key not in planned or key not in won:
            continue
        if won[key] != planned[key]:
            bad.append("%s %s vs %s" % (label, won[key], planned[key]))
    if not bad:
        return None
    return {"kind": "stale",
            "text": "This row was measured at " + ", ".join(bad) +
                    ". Speed is conditional on every one of those, so it is a "
                    "different experiment rather than a faster config."}


def recommend(plan_result, rows, sweep_budget_mib=None, strict=True):
    """The one config to run, and why it is not the other one.

    `rows` are recorded speed rows for this model (bench.load_speed_rows or the
    /api/speed/rows payload). `sweep_budget_mib` is the VRAM basis those rows
    were measured under, for the budget delta.

    `strict` filters rows to the ones comparable() calls the same experiment as
    the plan on screen. Off, the fastest trustworthy row wins whatever it was
    measured at and a `stale` delta says so - which is the right behaviour for a
    page that has just changed its context box and would otherwise blank out.

    Returns:
      source   "measured" | "predicted" | "none"
      config   the config to run, in a row's vocabulary
      row      the winning row, or None
      predicted the planner's own config, always
      deltas   [{kind, text}], empty when the two agree
    """
    planned = plan_config(plan_result)
    out = {"source": "predicted", "config": planned, "row": None,
           "predicted": planned, "tok_s": None, "vram_mib": None,
           "deltas": [], "n_rows": 0, "n_trusted": 0}
    plan = (plan_result or {}).get("plan") or {}
    if plan.get("n_gpu_layers") is None and not plan.get("fits_fully"):
        out["source"] = "none"

    rows = [r for r in (rows or []) if r]
    out["n_rows"] = len(rows)
    # A conclusion drawn from a spilled, looping or copying row is the exact
    # failure the flags on those rows exist to prevent, so they never win here
    # however fast they read.
    cand = [r for r in rank_rows(rows) if trustworthy(r)]
    out["n_trusted"] = len(cand)
    depth_note = None
    if strict and cand:
        # Two passes, because a plan and a row do not have the same vocabulary.
        #
        # First the conditions a PLAN has an opinion about. comparable() cannot
        # be handed this job alone: it also gates on `fill`, and a plan has no
        # fill at all - allocating 32k of context says nothing about how much of
        # it is occupied. Asking comparable() for a fill the plan never had
        # rejected every row, strict narrowing silently emptied, and the fastest
        # row at ANY context won - the exact confusion this function exists to
        # prevent, arriving through the one gate meant to prevent it.
        near = [r for r in cand if _conditions_match(r, planned)]
        if near:
            # Then pick ONE depth and rank inside it. Decode re-reads the KV
            # cache every token, so a 2k row beats a 32k row on nothing but
            # being shallower; ranking across depths would hand the answer to
            # whichever config happened to be measured least deeply. The
            # campaign's own depth is the one with the most rows, deepest wins a
            # tie.
            fills = {}
            for r in near:
                fills.setdefault((r.get("config") or {}).get("fill") or 0, []).append(r)
            fill = sorted(fills, key=lambda f: (len(fills[f]), f))[-1]
            if len(fills) > 1:
                depth_note = {"kind": "depth",
                              "text": "Ranked among the %d row%s measured at %s filled tokens. "
                                      "Rows at other depths are not slower configs - decode "
                                      "re-reads the KV cache every token, so they are different "
                                      "experiments."
                                      % (len(fills[fill]), "" if len(fills[fill]) == 1 else "s",
                                         "{:,}".format(fill))}
            base = dict(planned)
            base["fill"] = fill
            same = [r for r in fills[fill]
                    if comparable(r, r.get("model"), base, prompt_id=None,
                                  template_id=_ANY)]
            cand = same or fills[fill]
    if not cand:
        if rows:
            out["deltas"].append({
                "kind": "untrusted",
                "text": "%d recorded row%s, none of them usable as a conclusion - "
                        "they spilled into shared memory, looped, or copied the "
                        "prompt back. The estimate stands until one is re-measured."
                        % (len(rows), "" if len(rows) == 1 else "s")})
        return out

    win = cand[0]
    won = _cfg_of(win)
    out.update({"source": "measured", "config": won, "row": win,
                "tok_s": win.get("tok_s"), "vram_mib": win.get("proc_vram_mib")})

    same = (won.get("ngl") == planned.get("ngl")
            and (won.get("ncmoe") or 0) == (planned.get("ncmoe") or 0))
    if not same:
        out["deltas"].append({
            "kind": "objective",
            "text": "The planner returns the largest split that FITS - it stops at "
                    "the first config under the budget. This row is the one that "
                    "was FASTEST. Those are different questions, and only the "
                    "second one was measured."})
    for d in (_budget_delta(plan_result, sweep_budget_mib),
              _axis_deltas(won, planned),
              _stale_delta(won, planned),
              depth_note):
        if d:
            out["deltas"].append(d)
    return out
