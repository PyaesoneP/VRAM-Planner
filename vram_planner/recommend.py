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

"Fastest row measured" is itself only half the story, and which half depends on
the question being asked. A dense model that does not fit has two plans, and
each one pins a placement and leaves exactly one knob free - the context in the
ceiling plan, the -ot exile (then -ngl) in the fit plan. Along a knob that is
monotone in VRAM the value worth having is the one at the WALL, not the one that
read fastest: measured rows move 4.2% of tok/s across a doubling of context, and
downward, so ranking a ceiling campaign by tok/s recommends the smallest window
in the plan whose whole purpose is the largest one. mode_axis() below decides
which knob a category leaves free and recommend() ranks along it, using the same
pick_extreme() the campaign promotes stages with - so the card and the sweep
cannot land on different rows. Off the two-plan regime - an MoE, a model that
fits whole - nothing is left free and fastest-wins stands unchanged.

Pure: no I/O, no subprocess, no GPU. Same rule as bench.py's findings code, and
for the same reason - the CLI and the browser reach one conclusion by
construction rather than by two people remembering to keep them in step.
"""
from .bench import (axis_direction, comparable, norm_plan_mode, pick_extreme,
                    rank_rows, trustworthy, _ANY)

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


# ---------------------------------------------------------------------------
# What "best" means, per category
#
# A dense model that does not fit has one answer per question, and the two
# questions do not share an objective. Each mode pins one placement and leaves
# exactly one knob free, and along that knob the value worth having is the one
# at the wall - not the one that read fastest. Measured rows move 4.2% of tok/s
# across a DOUBLING of context, and downward, so ranking a ceiling campaign by
# tok/s recommends the SMALLEST window in the one plan whose whole purpose is
# the largest.
#
# So the recommendation ranks the way the campaign PROMOTES: extreme along the
# mode's own axis, with tok/s only settling a tie between rows at the SAME
# value. No slack, no threshold - the largest that loaded wins even when it is
# the slowest row on the ladder, because that is what the ceiling plan is FOR and
# because decode moves 4.2% across a doubling of the window anyway. A row whose
# numbers cannot be believed is removed by the spill gates (trustworthy()),
# which read the memory counters, rather than by inferring a spill from a speed
# ranking. Off the two-plan regime - an MoE, or a model that fits whole -
# nothing is left free and fastest-wins stands, which is what it always was.
# ---------------------------------------------------------------------------

# The phrase the card puts on the criterion, per axis and direction.
AXIS_GOAL = {
    ("ctx", "up"):          "the largest context that loaded",
    ("ngl", "up"):          "the most layers on the GPU that loaded",
    ("n_cpu_ffn", "down"):  "the least dense FFN exiled to RAM that loaded",
    ("ncmoe", "down"):      "the fewest experts exiled to RAM that loaded",
}


def mode_axis(plan_result):
    """(axis, direction) the selected category leaves free, or (None, None).

    Must agree with bench.ngl_ladder(), which decides the same thing for the
    campaign - it reads the planner's SEED where this reads the finished plan,
    but the rule is one rule and a card that ranked by another would recommend a
    config the sweep would never promote:

      * the CEILING plan pins both placements at maximum - every block on the
        GPU, every block's dense FFN off it - so the only thing left free is the
        window.
      * the FIT plan holds the window and pays for it in the cheapest currency
        first: while every block still fits, the free knob is the -ot exile and
        less exiled is better; once a full exile is not enough, whole blocks
        start leaving and the free knob is -ngl. The campaign reaches the same
        two answers in the other order - it searches -ngl first and only walks
        the -ot exile back when every block turned out to fit - because it can
        MEASURE which case it is in, where this has to read it off the plan.

    A plan outside the two-plan regime has no mode and nothing left free: an
    MoE's --n-cpu-moe and a fits-whole plan are single answers, so they get
    (None, None) and the fastest row wins as before.
    """
    mode = norm_plan_mode((plan_result or {}).get("plan_mode"))
    plan = (plan_result or {}).get("plan") or {}
    if mode not in ("ceiling", "fit") or norm_plan_mode(plan.get("mode")) != mode:
        return None, None
    if mode == "ceiling":
        return "ctx", axis_direction("ctx")
    n_layers = ((plan_result or {}).get("config") or {}).get("n_layers") or 0
    if n_layers and (plan.get("n_gpu_layers") or 0) >= n_layers:
        return "n_cpu_ffn", axis_direction("n_cpu_ffn")
    return "ngl", axis_direction("ngl")


def axis_goal(axis, direction):
    """The criterion in words, for the card and the CLI report."""
    return AXIS_GOAL.get((axis, direction)) or ("the largest %s that loaded" % axis)


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
    c = {"ngl": plan.get("n_gpu_layers"), "ncmoe": plan.get("n_cpu_moe") or 0,
          # The -ot pin is what DEFINES both dense plans, so it has to survive
          # into the launch script; without it the script asks for the layers of
          # a ceiling plan and none of the tensor split that made them fit.
         "n_cpu_ffn": plan.get("n_cpu_ffn") or 0}
    # A ceiling plan's answer IS its context - the largest that fits at ngl=all
    # with the FFN exiled - and that is generally not the number the user typed
    # in. Take the plan's own where it has one, or every ceiling row is judged
    # against a context the plan never proposed and rejected as answering
    # another question.
    if plan.get("max_ctx") is not None:
        c["ctx"] = int(plan["max_ctx"])
    elif inp.get("context") is not None:
        c["ctx"] = inp["context"]
    if inp.get("mmproj_place") in ("vram", "ram"):
        c["mmproj_offload"] = inp["mmproj_place"] == "vram"
    if inp.get("kv_type"):
        c["kv"] = inp["kv_type"]
    if inp.get("n_ubatch") is not None:
        c["ub"] = inp["n_ubatch"]
    if inp.get("n_seq") is not None:
        c["seq"] = inp["n_seq"]
    if inp.get("flash_attn") is not None:
        c["fa"] = bool(inp["flash_attn"])
    # Speculation the plan actually priced must survive into the launch script.
    # An external drafter names the file (md) and the scheme follows its kind;
    # the model's own MTP blocks are a scheme without a file. Absent either,
    # spec stays absent - a plan without speculation must not invent one.
    if inp.get("drafter"):
        c["spec"] = "draft-mtp" if inp.get("drafter_kind") == "mtp" else "draft-dflash"
        if inp.get("drafter_depth"):
            c["spec_n_max"] = inp["drafter_depth"]
        c["md"] = inp["drafter"]
    elif inp.get("mtp_depth"):
        c["spec"] = "draft-mtp"
        c["spec_n_max"] = inp["mtp_depth"]
    return c


def _conditions_match(row, planned, free_axis=None):
    """Does this row answer the question currently on screen?

    Only the keys a PLAN has an opinion about - context, KV quant, sequences,
    flash attention. Deliberately not fill, n_predict, repeat or the samplers:
    those are properties of how a campaign was run and a plan has no value for
    any of them, so demanding they agree would reject every row.

    `free_axis` is the knob the selected category is free to move, and it is
    exempted for the same reason comparable() exempts stage A's swept axis: in
    the CEILING plan the context is not part of what is being asked, it is the
    answer, and gating it keeps only the rungs that happen to equal the
    planner's own guess - so the campaign measures a wall and the card then
    recommends the guess. Only ctx is ever both gated here and free; ngl and
    n_cpu_ffn are not conditions at all.
    """
    c = row.get("config") or {}
    for key in ("ctx", "kv", "seq", "fa"):
        if key == free_axis:
            continue
        if key not in planned or key not in c:
            continue
        a, b = c[key], planned[key]
        if key == "fa":
            a, b = bool(a), bool(b)
        if a != b:
            return False
    return True


# The placement a category PINS. Each dense mode fixes both of these and then
# frees exactly one - the mode's own axis - so the other is what makes a row a
# member of this category at all.
PIN_KEYS = ("ngl", "n_cpu_ffn")


def _pins_match(row, planned, axis):
    """Was this row measured in the layout the selected category proposes?

    Without this the extreme is trivially won by the wrong regime. "The largest
    context that loaded" in the SPEED category was answered on a real store by a
    row at -ngl 28 - a CONTEXT campaign's row, where most of the model is in RAM
    and a 128k window naturally fits. The speed category is DEFINED by -ngl all
    plus -ot all; a row at another placement is the other plan's answer wearing
    this plan's badge.

    The free axis is exempt, because it is the thing being ranked. Only ngl and
    n_cpu_ffn are gated: --n-cpu-moe belongs to a regime that has no categories,
    and everything else a row records is either a condition (settled by
    _conditions_match) or a knob no plan predicts (reported by _axis_deltas).
    """
    c = row.get("config") or {}
    for k in PIN_KEYS:
        if k == axis or planned.get(k) is None or k not in c:
            continue
        if (c.get(k) or 0) != (planned.get(k) or 0):
            return False
    return True


def _pins_text(planned):
    """The pinned placement in llama.cpp's own flags, for a delta to name."""
    bits = []
    if planned.get("ngl") is not None:
        bits.append("-ngl %s" % planned["ngl"])
    if planned.get("n_cpu_ffn"):
        bits.append("-ot on %s blocks" % planned["n_cpu_ffn"])
    return ", ".join(bits) or "this split"


def _fmt_mib(v):
    return "-" if v is None else "{:,.0f} MiB".format(float(v))


def plan_effective_budget(plan_result):
    """What the plan actually had to spend, after the reserve and the margin.

    The same arithmetic analyze() does, deliberately restated rather than read
    back out of `inputs.eff_vram_mib`: that field has the projector, the MTP
    draft cache and the vision peak already held out of it, and the sweep's
    basis has none of those subtracted. Comparing the two would report a
    difference that is a holdout, not a budget.
    """
    inp = (plan_result or {}).get("inputs") or {}
    have = inp.get("vram_budget_mib")
    if have is None:
        return None
    reserve = float(inp.get("gpu_reserve_mib") or 0.0)
    safety = float(inp.get("safety_pct") or 0.0)
    return max(0.0, (float(have) - reserve) * (1.0 - safety / 100.0))


def _budget_delta(plan_result, sweep_budget_mib):
    """Was the plan priced against a different amount of VRAM than the rows?

    This was the single largest source of disagreement and the one nobody could
    see: the browser prefilled its budget from FREE VRAM at page load with a
    zero reserve, while the sweep seeded its ladder from TOTAL with a 512 MiB
    reserve. Same model, same question, several -ngl rungs apart.

    Both sides are compared AFTER the reserve and the safety margin, because
    that is the number a split is actually chosen against. Comparing the raw
    budget field to the sweep's effective one reported a delta on every plan
    ever made - including the default one, where the two agree exactly and the
    whole gap was the 512 MiB reserve the sweep had already taken off.
    """
    have = plan_effective_budget(plan_result)
    if have is None or not sweep_budget_mib:
        return None
    if abs(float(have) - float(sweep_budget_mib)) <= 64:      # same basis
        return None
    return {"kind": "budget",
            "text": "The plan was priced against %s of usable VRAM; the measurements "
                    "were taken with %s. Both are after the driver reserve and the "
                    "safety margin. A budget that differs by that much moves the "
                    "split on its own, before any question of speed."
                    % (_fmt_mib(have), _fmt_mib(sweep_budget_mib))}


# What comparable() falls back to when the plan has no opinion. A plan has no
# sampler settings at all, so the comparison assumes these - and a row measured
# at real sampler settings is a different experiment, not a slower config.
SAMPLERS = (("temp", "temperature", 0.0), ("top_k", "top-k", 0),
            ("top_p", "top-p", 1.0), ("min_p", "min-p", 0.0),
            ("rep_pen", "repeat penalty", 1.0), ("pres_pen", "presence penalty", 0.0))


def _sampler_delta(won):
    """The winner came from a fallback that comparable() had rejected."""
    bits = []
    for key, label, default in SAMPLERS:
        v = won.get(key)
        if v is None:
            v = default
        if v != default:
            bits.append("%s %s" % (label, v))
    return {"kind": "samplers",
            "text": "A plan has no sampler settings, so the comparison assumes the "
                    "defaults - greedy. No row here was measured that way%s, so the "
                    "ranking fell back to the rows there are. Greedy is speculation's "
                    "best case, so a sampled row and a greedy plan are two experiments "
                    "rather than two configs."
                    % ((": these ran at " + ", ".join(bits)) if bits else "")}


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


def _stale_delta(won, planned, free_axis=None):
    """Does the winning row answer the question currently on screen?

    `free_axis` is exempt on the same grounds it is exempt from
    _conditions_match: a ceiling plan did not ASK for a context, it proposed one,
    so a row that found a larger window is the answer improving rather than a
    stale experiment.
    """
    bad = []
    for key, label in (("ctx", "context"), ("kv", "KV quant"),
                       ("seq", "sequences"), ("fa", "flash attention")):
        if key == free_axis:
            continue
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
      mode     "ceiling" | "fit" | None - the category this answers
      axis     the knob that category leaves free, or None
      objective "extreme" (at the wall along `axis`) or "fastest"
      goal     the criterion in words, for the card to print
    """
    planned = plan_config(plan_result)
    # Which category is on screen, and therefore what "best" means among the
    # rows. Absent a two-plan regime this is (None, None) and fastest wins.
    axis, direction = mode_axis(plan_result)
    out = {"source": "predicted", "config": planned, "row": None,
           "predicted": planned, "tok_s": None, "vram_mib": None,
            "deltas": [], "n_rows": 0, "n_trusted": 0,
            "mode": norm_plan_mode((plan_result or {}).get("plan_mode")),
           "axis": axis, "direction": direction,
           "objective": "extreme" if axis else "fastest",
           "goal": axis_goal(axis, direction) if axis else "the fastest row measured"}
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
    regime_note = None
    sampler_fallback = False
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
        # The category's own PLACEMENT comes first, ahead of the conditions,
        # because it decides which QUESTION a row was answering rather than
        # under what settings. Ranking along a free axis only means anything
        # among rows that pinned everything else the way this category does -
        # and unlike the conditions it must hold even in the fallback below,
        # since "the largest context that loaded" over rows from the other
        # regime is exactly the misreading it was added to stop.
        if axis:
            pinned = [r for r in cand if _pins_match(r, planned, axis)]
            if pinned:
                cand = pinned
            else:
                regime_note = {
                    "kind": "regime",
                    "text": "No recorded row was measured in this plan's own layout (%s). "
                            "The rows there are answer the other question, so they are "
                            "ranked here for want of better - a row at a different split "
                            "is not this category's answer, it is the other category's."
                            % _pins_text(planned)}
        near = [r for r in cand if _conditions_match(r, planned, axis)]
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
            # `swept` un-gates ctx inside comparable() as well, and the pairing
            # is the point: an axis the category is free to move must be
            # comparable across its rungs or the winner is never a candidate.
            # Only ctx is gated there, so any other free axis passes through.
            same = [r for r in fills[fill]
                    if comparable(r, r.get("model"), base, prompt_id=None,
                                  template_id=_ANY, swept=(axis or ""))]
            # Everything comparable() still gates on here is the samplers - the
            # conditions were settled above and `base` carries the group's own
            # fill. So an empty narrowing means the campaign swept real sampler
            # settings, and falling back silently would recommend a sampled row
            # against a greedy assumption without saying so - the same silence
            # this module exists to end. Fall back, and name it.
            cand = same or fills[fill]
            sampler_fallback = not same
    if not cand:
        if rows:
            out["deltas"].append({
                "kind": "untrusted",
                "text": "%d recorded row%s, none of them usable as a conclusion - "
                        "they spilled into shared memory, looped, or copied the "
                        "prompt back. The estimate stands until one is re-measured."
                        % (len(rows), "" if len(rows) == 1 else "s")})
        return out

    # rank_rows() left `cand` fastest-first, which is the answer when nothing is
    # left free. With a category selected the winner is the row at the wall
    # along its own axis instead - see mode_axis().
    win = pick_extreme(cand, axis, direction) if axis else cand[0]
    won = _cfg_of(win)
    out.update({"source": "measured", "config": won, "row": win,
                "tok_s": win.get("tok_s"), "vram_mib": win.get("proc_vram_mib")})

    same = (won.get("ngl") == planned.get("ngl")
            and (won.get("ncmoe") or 0) == (planned.get("ncmoe") or 0)
            and (axis is None
                 or (won.get(axis) or 0) == (planned.get(axis) or 0)))
    if not same:
        out["deltas"].append({
            "kind": "objective",
            "text": ("The planner returns the largest split that FITS - it stops at "
                     "the first config under the budget. This row is %s. Those are "
                     "different questions, and only the second one was measured."
                     % out["goal"]) if axis else
                    ("The planner returns the largest split that FITS - it stops at "
                     "the first config under the budget. This row is the one that "
                     "was FASTEST. Those are different questions, and only the "
                     "second one was measured.")})
    for d in (_budget_delta(plan_result, sweep_budget_mib),
              _axis_deltas(won, planned),
              _stale_delta(won, planned, axis),
              _sampler_delta(won) if sampler_fallback else None,
              regime_note, depth_note):
        if d:
            out["deltas"].append(d)
    return out
