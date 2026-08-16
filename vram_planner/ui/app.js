"use strict";
/* VRAM Planner UI.
 *
 * Two rules hold this together:
 *
 *   1. All markup is built with the `h` tagged template, which escapes every
 *      interpolation by default. Anything already-safe HTML must be wrapped in
 *      raw(). The old version escaped a handful of sites by hand and missed the
 *      rest - model names, file paths and llama.cpp error strings all reach the
 *      page, and any of them can contain < or &.
 *   2. No inline onclick. One delegated listener dispatches on [data-action], so
 *      dynamically rendered buttons need no globals and the page keeps working
 *      under a strict CSP.
 */

const $ = id => document.getElementById(id);
let SYS = null, MODELS = [], LAST = null, CTX_MAX = null;

const CALIB_TERMS = ["floor", "ctx", "act", "nofa"];

/* ---------------------------------------------------------------- escaping */
const RAW = Symbol("raw");
const raw = s => ({ [RAW]: String(s) });

function esc(x){
  if(x === null || x === undefined) return "";
  if(typeof x === "object" && RAW in x) return x[RAW];
  return String(x).replace(/[&<>"']/g, c => (
    { "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]
  ));
}

/** Tagged template that escapes interpolations. Arrays are joined. */
function h(strings, ...vals){
  let out = strings[0];
  for(let i = 0; i < vals.length; i++){
    const v = vals[i];
    out += (Array.isArray(v) ? v.map(esc).join("") : esc(v)) + strings[i + 1];
  }
  return out;
}

/* ------------------------------------------------------------- formatting */
function fmt(m){ return (m == null || isNaN(m)) ? "-" : Math.round(m).toLocaleString() + " MiB"; }
function fmtG(m){ return (m == null || isNaN(m)) ? "-" : (m / 1024).toFixed(2) + " GiB"; }
function fmtGB(m){ return (m == null || isNaN(m)) ? "-" : (m * 1048576 / 1e9).toFixed(2) + " GB"; }
function B(n){ return (n / 1e9).toFixed(2) + " B"; }
function num(n){ return (n == null) ? "-" : Number(n).toLocaleString(); }
function clampPct(x){ return Math.max(0, Math.min(100, x)); }

/* ------------------------------------------------------------------ theme */
function initTheme(){
  const btn = $("themeBtn");
  if(btn) btn.setAttribute("aria-pressed", document.documentElement.dataset.theme === "light");
}
function toggleTheme(){
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try{ localStorage.setItem("vramplanner-theme", next); }catch(e){}
  initTheme();
}

/* ------------------------------------------------------------------- boot */
async function boot(){
  initTheme();
  try{
    const s = await (await fetch("/api/system")).json();
    SYS = s;
    if(!$("dir").value) $("dir").value = s.default_dir || "";
    if(s.version) $("ver").textContent = " · v" + s.version;
    renderPlatform(s.platform);
    renderSys(s);
    prefillBudgets(s);
    if($("dir").value) scanModels();
  }catch(e){
    $("sys").innerHTML = h`<span class="muted small">system read failed: ${e}</span>`;
  }
  buildCtxChips();
  loadBandwidth();
  showPastSweeps();
}

/** The empty state, before anything has been analyzed.
 *
 *  A tool holding two hours of measurements should not open looking empty. The
 *  whole Past sweeps path - browse a campaign, read what it found, build a script
 *  from one of its rows - needs no model analyzed and touches no GPU, so there is
 *  no reason to hide it behind Analyze. */
async function showPastSweeps(){
  if(LAST) return;                       // a plan arrived first; render() owns the page
  SWEEP = sweepDefaults();
  await Promise.all([loadHistory(), attachRunningJob()]);
  if(LAST) return;
  const n = (SWEEP.campaigns || []).length;
  // A campaign started in another tab is worth showing even with nothing else
  // recorded yet - that is exactly when the page would otherwise look idle
  // while the GPU is busy.
  if(!n && !(SWEEP.status && SWEEP.status.status === "running")) return;
  renderIdle();
}

/** The page with no model analyzed. Same three steps, with step 1 asking for the
 *  one thing it needs instead of being empty - and steps 2 and 3 fully usable,
 *  because browsing a recorded campaign and building a script from one of its
 *  rows needs no plan and touches no GPU. */
function renderIdle(){
  if(LAST) return;
  const busy = !!(SWEEP && SWEEP.status && SWEEP.status.status === "running");
  const step = stepNow();
  let panel;
  if(step === "fit"){
    panel = h`<section class="card lead">
      <p class="placeholder">Pick a model on the left and press <b>Analyze fit</b>.</p>
      <p class="note">Steps 2 and 3 work without this: a recorded campaign can be read and a
        launch script built from any of its rows with nothing analyzed and the GPU untouched.</p>
    </section>`;
  }else if(step === "measure"){
    panel = renderSweep() + renderHistory();
  }else{
    panel = renderScriptStep();
  }
  $("out").innerHTML = renderStepper(null) + renderGlossary() + panel;
  drawSweep({ grid: busy, results: busy, script: true, history: true });
}

function renderPlatform(p){
  const el = $("platwarn");
  if(!el) return;
  el.innerHTML = (!p || p.supported) ? "" : h`<div class="card warns" style="margin-bottom:14px">
      <div style="color:var(--warn);font-weight:600;margin-bottom:4px">&#9888; Unvalidated platform</div>
      <div>${p.reason}</div></div>`;
}

/** The VRAM budget the plan is built against.
 *
 *  Server-side now, and deliberately: this page used to prefill from FREE VRAM
 *  with a zero reserve while the speed sweep seeded its ladder from TOTAL minus
 *  512. Neither number was wrong; having two of them meant the planner and the
 *  campaign recommended different splits with nothing on the page to say why.
 *  /api/system computes both bases and names which one the sweep uses. */
function budgetFor(basis){
  const b = (SYS && SYS.vram_budget) || null;
  if(!b) return null;
  const v = basis === "free" ? b.on_free : b.on_total;
  return (v == null) ? null : Math.round(v);
}

function prefillBudgets(s){
  const b = s.vram_budget || {};
  const sel = $("vrambasis");
  if(sel && b.basis) sel.value = b.basis;
  if(!$("vram").value && b.default != null) $("vram").value = Math.round(b.default);
  if($("reserve").value === "" && b.reserve != null) $("reserve").value = b.reserve;
  // RAM budget defaults to TOTAL installed (minus a small OS reserve): a model can load
  // into standby/paged memory, so "free right now" understates what will actually load.
  if(s.ram && s.ram.total_mib && !$("ram").value)
    $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
  markBasis();
}

/** Refill the budget from whichever basis is selected, and say what changed.
 *  Switching the basis with a stale number in the box is the divergence this
 *  whole control exists to remove, so it always rewrites the field. */
function setBasis(){
  const v = budgetFor($("vrambasis").value);
  if(v != null) $("vram").value = v;
  markBasis();
}

function markBasis(){
  const b = (SYS && SYS.vram_budget) || {};
  const el = $("basishint");
  if(!el || b.on_total == null) return;
  const basis = $("vrambasis").value;
  const off = basis !== (b.basis || "total");
  el.innerHTML = h`<b>${fmt(budgetFor(basis))}</b> of ${fmt(b.on_total)} on the card${
      b.on_free != null ? h` · ${fmt(b.on_free)} free right now` : ""}. ` +
    (off
      ? h`<span style="color:var(--warn)">Speed campaigns are planned against <b>card total</b>,
          so a measured row and this plan are now priced differently — the card at the top will
          say so.</span> `
      : "Speed campaigns are planned against this same basis. ") +
    h`<details class="why"><summary>why total is the default</summary>
      A speed campaign refuses to start unless the card is essentially empty, so free VRAM at
      planning time is transient state the measurements will never be taken under. Planning
      against it produces a split no recorded row can match.</details>`;
}

function meter(name, freeMib, totalMib){
  const usedPct = clampPct((totalMib - freeMib) / totalMib * 100);
  return h`<div class="g">
    <div class="lbl"><span>${name}</span>
      <span><b>${num(Math.round(freeMib))}</b> / ${num(Math.round(totalMib))} MiB free</span></div>
    <div class="meter" role="progressbar" aria-label="${name} used"
         aria-valuenow="${Math.round(usedPct)}" aria-valuemin="0" aria-valuemax="100">
      <i style="width:${usedPct}%"></i></div></div>`;
}

function renderSys(s){
  let out = "";
  if(Array.isArray(s.gpus) && s.gpus.length){
    out += s.gpus.map(g => meter(g.name, g.free_mib, g.total_mib)).join("");
  }else{
    out += h`<div class="g"><div class="lbl"><span>GPU</span>
      <span class="muted">nvidia-smi not found</span></div><div class="meter"></div></div>`;
  }
  if(s.ram) out += meter("System RAM", s.ram.free_mib, s.ram.total_mib);
  out += h`<button class="ghost" type="button" data-action="refresh-sys">&#8635; refresh</button>`;
  $("sys").innerHTML = out;
}

async function loadBandwidth(){
  try{
    const b = await (await fetch("/api/bandwidth")).json();
    if(b.vram_gbs && !$("bwv").value) $("bwv").value = b.vram_gbs;
    if(b.ram_gbs && !$("bwr").value) $("bwr").value = b.ram_gbs;
    $("bwhint").innerHTML = (b.notes || []).map(esc).join("<br>");
  }catch(e){ $("bwhint").textContent = "bandwidth auto-detect failed: " + e; }
}

async function refreshSys(){
  const s = await (await fetch("/api/system?fresh=1")).json();
  SYS = s;
  renderSys(s);
  const v = budgetFor($("vrambasis").value);
  if(v != null) $("vram").value = v;
  if(s.ram) $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
  markBasis();
}

/* ------------------------------------------------------------- ctx presets */
function ctxLabel(v){ return v >= 1024 ? (v / 1024) + "k" : String(v); }

function buildCtxChips(){
  const vals = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
  $("ctxchips").innerHTML = vals.map(v =>
    h`<button type="button" class="chip" data-action="set-ctx" data-v="${v}"
              aria-pressed="false">${ctxLabel(v)}</button>`).join("");
  markCtx();
}

function setCtxMax(nctx){
  CTX_MAX = nctx;
  $("ctx").max = nctx;
  let chip = $("ctxmax");
  if(!chip){
    chip = document.createElement("button");
    chip.id = "ctxmax";
    chip.type = "button";
    chip.className = "chip";
    chip.dataset.action = "set-ctx";
    $("ctxchips").appendChild(chip);
  }
  chip.textContent = "max " + ctxLabel(nctx);
  chip.dataset.v = String(nctx);
  markCtx();
}

function setCtx(v){ $("ctx").value = v; markCtx(); }

function markCtx(){
  const cur = parseInt($("ctx").value) || 0;
  document.querySelectorAll("#ctxchips .chip").forEach(c => {
    c.setAttribute("aria-pressed", String(parseInt(c.dataset.v) === cur));
  });
}

/* ----------------------------------------------------------------- models */
async function scanModels(){
  const dir = $("dir").value.trim();
  $("scanhint").textContent = "scanning…";
  try{
    const r = await (await fetch("/api/models?dir=" + encodeURIComponent(dir))).json();
    MODELS = r.models || [];
    $("model").innerHTML =
      h`<option value="">&mdash; ${MODELS.length} models found &mdash;</option>` +
      MODELS.map((m, i) => h`<option value="${i}">${m.from_card ? "○ " : ""}${m.name}  (${
        fmtG(m.size_mix || m.size_mib)})${
        m.from_card ? "  · stored card, not on disk"
                    : (m.n_ctx_train ? "  · " + ctxLabel(m.n_ctx_train) + " ctx" : "")
        }</option>`).join("");
    const onDisk = MODELS.length - (r.n_cards || 0);
    $("scanhint").textContent = MODELS.length
      ? onDisk + " GGUF found in folder"
        + (r.n_cards ? " + " + r.n_cards + " stored card"
                       + (r.n_cards == 1 ? "" : "s") + " (model not on disk)" : "")
      : "no .gguf found here";
  }catch(e){ $("scanhint").textContent = "scan failed: " + e; }
}

function onPick(){
  const i = $("model").value;
  if(i === "") return;
  const m = MODELS[parseInt(i)];
  if(!m) return;
  $("path").value = "";
  if(m.n_ctx_train) setCtxMax(m.n_ctx_train);
}

/* -------------------------------------------------------------- analysis */
function currentPath(){
  const typed = $("path").value.trim();
  if(typed) return typed;
  const i = $("model").value;
  return (i !== "" && MODELS[parseInt(i)]) ? MODELS[parseInt(i)].path : "";
}

async function run(){
  const path = currentPath();
  if(!path){ alert("Pick a model from the list or paste a .gguf path."); return; }
  const body = {
    path: path,
    context: parseInt($("ctx").value) || 8192,
    kv_type: $("kv").value,
    n_ubatch: parseInt($("ubatch").value) || 512,
    n_seq: parseInt($("nseq").value) || 1,
    include_mmproj: $("mmproj").checked,
    mtp_spec: $("mtpspec").checked,
    n_cpu_moe_override: $("ncpumoe").value === "" ? null : parseInt($("ncpumoe").value),
    bw_vram_gbs: parseFloat($("bwv").value) || 0,
    bw_ram_gbs: parseFloat($("bwr").value) || 0,
    ram_eff: parseFloat($("rameff").value) || 0,
    ctx_fill: $("ctxfill").value === "" ? null : parseInt($("ctxfill").value),
    flash_attn: $("fa").checked,
    vram_budget_mib: parseFloat($("vram").value) || 0,
    ram_budget_mib: parseFloat($("ram").value) || 0,
    gpu_reserve_mib: parseFloat($("reserve").value) || 0,
    compute_override_mib: parseFloat($("compute").value) || 0,
    safety_pct: parseFloat($("safety").value) || 0,
    kv_on_gpu: $("kvgpu").checked,
    gpu_layers_override: $("ngl").value.trim() === "" ? null : parseInt($("ngl").value),
    ram_free_mib: (SYS && SYS.ram) ? SYS.ram.free_mib : null,
    // Only sent when "Plan for images" is ticked. Absent means a text-only plan,
    // which the server warns about rather than guessing a size on the user's behalf.
    image_w: $("visionplan").checked ? (parseInt($("imagew").value) || 1024) : null,
    image_h: $("visionplan").checked ? (parseInt($("imageh").value) || 1024) : null,
    vision_flash_attn: $("visionfa").checked
  };
  const btn = $("goBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>analyzing';
  $("out").innerHTML = '<div class="card"><p class="placeholder"><span class="spin"></span>parsing GGUF…</p></div>';
  try{
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const r = await res.json();
    if(!r.ok) $("out").innerHTML = h`<div class="card warns"><b>Error:</b> ${r.error || "unknown"}</div>`;
    else {
      // Pressing Analyze is a question about FIT, so answer that one - even if
      // the last thing looked at was a script. Without this the step stuck
      // wherever it was left and Analyze appeared to do nothing at all.
      STEP = null;
      render(r);
    }
  }catch(e){
    $("out").innerHTML = h`<div class="card warns"><b>Request failed:</b> ${e}</div>`;
  }
  btn.disabled = false;
  btn.textContent = "Analyze fit";
}

/* ------------------------------------------------------------- fragments */
function kvItem(k, v){
  return h`<div class="kvi"><div class="k">${k}</div><div class="v">${v == null ? "-" : raw(v)}</div></div>`;
}
function brow(k, v){ return h`<tr><td>${raw(k)}</td><td>${raw(v)}</td></tr>`; }

function kvBadge(c){
  return c.is_moe
    ? h`<span class="badge moe">MoE &middot; ${c.n_expert_used}/${c.n_expert} experts</span>`
    : '<span class="badge dense">DENSE</span>';
}

function bar(title, capMib, used, segs){
  const sum = segs.reduce((a, s) => a + (s.mib || 0), 0);
  const scale = Math.max(capMib, sum, 1);
  const inner = segs.map(s => h`<span class="${s.cls}" style="width:${clampPct(s.mib / scale * 100)}%;background:${s.color}"
      title="${s.label}: ${fmt(s.mib)}"></span>`).join("");
  const over = sum > capMib;
  const utilTxt = capMib ? (used / capMib * 100).toFixed(0) + "%" : "-";
  const legend = segs.filter(s => s.mib > 0.5).map(s =>
    h`<span><i style="background:${s.color}"></i>${s.label} <span class="n">${fmt(s.mib)}</span></span>`).join("");
  return h`<div class="barwrap">
    <div class="bar-top">
      <span>${title} &middot; using <b>${fmt(used)}</b> of ${fmt(capMib)}</span>
      <span class="${over ? "over" : ""}">${utilTxt}${over ? " OVER" : ""}</span>
    </div>
    <div class="bar" role="img" aria-label="${title}: ${fmt(used)} of ${fmt(capMib)} used">
      ${raw(inner)}<span class="cap" data-l="${fmt(capMib)}" style="left:${clampPct(capMib / scale * 100)}%"></span>
    </div>
    <div class="legend">${raw(legend)}</div></div>`;
}

/* ------------------------------------------------------- the recommendation
 *
 * The one thing the page is for: what should I actually run. It sits above the
 * steps because it is the answer all three of them are working towards, and
 * because the two sources of that answer used to be three clicks apart with
 * nothing anywhere saying they disagreed.
 *
 * The planner returns the LARGEST SPLIT THAT FITS. The sweep returns the
 * FASTEST ROW MEASURED. Those are different questions. When they give different
 * answers this card names every reason, rather than leaving the user to notice
 * that step 1 said ngl 41 and step 2's best row says ngl 47. */
let REC = null, REC_FOR = null, REC_SEQ = 0;

const REC_KIND = {
  objective: "objective",
  budget:    "budget",
  axis:      "knobs",
  stale:     "conditions",
  samplers:  "samplers",
  depth:     "depth",
  untrusted: "no usable rows"
};

/** Config in one line, in the vocabulary the launcher and the row tables use. */
function cfgLine(c){
  if(!c) return "—";
  const bits = [];
  if(c.ngl != null) bits.push("ngl " + c.ngl);
  if(c.ncmoe) bits.push("ncmoe " + c.ncmoe);
  if(c.ub) bits.push("ub " + c.ub);
  if(c.spec && c.spec !== "none")
    bits.push(c.spec + (c.spec_n_max ? "/" + c.spec_n_max : ""));
  if(c.mmproj_offload === false) bits.push("projector in RAM");
  return bits.join(" · ") || "—";
}

/** Fetch the reconciliation for this plan.
 *
 *  Guarded on the plan object itself, because render() runs on every tab switch
 *  and the answer cannot change between two clicks of the same plan. Without it
 *  each switch cost a round trip and blanked the card while it flew.
 *
 *  Sequenced as well as guarded: REC_FOR alone only rejects a REPEAT of the
 *  plan in flight, so two analyses in quick succession could land out of order
 *  and leave the card describing plan A beside plan B's deltas. It is also
 *  cleared to null and refetched for the SAME plan after rows are measured or
 *  forgotten, where the object identity is unchanged and only the counter can
 *  tell the two answers apart. Last request asked for wins; everything else is
 *  dropped rather than drawn. */
async function loadRecommendation(r){
  if(!r || !r.plan) return;
  if(REC_FOR === r) return;
  REC_FOR = r;
  const seq = ++REC_SEQ;
  REC = null;
  drawRecommendation();
  let out;
  try{
    out = await (await fetch("/api/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan: r, model: (currentPath().split(/[\\/]/).pop() || "") })
    })).json();
  }catch(e){ out = { ok: false, error: String(e) }; }
  if(seq !== REC_SEQ) return;          // superseded while in flight
  REC = out;
  drawRecommendation();
}

function drawRecommendation(){
  const el = $("rec");
  if(el) el.innerHTML = renderRecommendation();
}

function renderRecommendation(){
  if(!LAST) return "";
  const v = (LAST.plan && LAST.plan.verdict) || {};
  if(!REC) return h`<section class="card rec"><p class="muted small">reconciling against
    recorded measurements&hellip;</p></section>`;
  if(REC.ok === false) return "";

  const measured = REC.source === "measured";
  const cfg = REC.config || {};
  const badge = measured
    ? '<span class="pill ok">● measured</span>'
    : '<span class="pill">estimated</span>';

  // The headline number differs by source on purpose: a measured row has a real
  // tok/s and a real process VRAM reading, and an estimate has neither - showing
  // the planner's arithmetic in the same slot would dress one up as the other.
  const figures = measured
    ? h`<span class="rec-big">${(REC.tok_s || 0).toFixed(2)}</span><span class="rec-unit">tok/s</span>
        <span class="rec-sep"></span>
        <span class="rec-num">${fmt(REC.vram_mib)}</span><span class="rec-unit">VRAM, measured</span>`
    : h`<span class="rec-big ${v.state === "fits" ? "" : "warn"}">${v.word || "—"}</span>
        <span class="rec-sep"></span>
        <span class="rec-num">${fmt(v.vram_mib)}</span><span class="rec-unit">of ${
          fmt(v.vram_budget_mib)} VRAM, estimated</span>`;

  const deltas = (REC.deltas || []).map(d =>
    h`<div class="delta"><span class="dk">${REC_KIND[d.kind] || d.kind}</span><span>${d.text}</span></div>`
  ).join("");

  const other = measured && REC.predicted
    ? h`<p class="note rec-alt">The estimate on its own says <b class="mono">${
        cfgLine(REC.predicted)}</b>.</p>`
    : "";

  const foot = measured
    ? h`<p class="note">From the fastest recorded row this campaign can stand behind — rows that
        spilled, looped or copied the prompt back are never eligible, however fast they read.
        ${REC.n_trusted} of ${REC.n_rows} recorded row${REC.n_rows === 1 ? "" : "s"} qualified.</p>`
    : h`<p class="note">Nothing measured for this model yet, so this is the planner's arithmetic:
        the <b>largest split that fits</b>, which is not the same question as the fastest.
        Step 2 measures the difference.</p>`;

  return h`<section class="card rec ${measured ? "is-measured" : ""}">
    <div class="rec-head"><h2>Run this</h2>${raw(badge)}</div>
    <p class="rec-cfg mono">${cfgLine(cfg)}</p>
    <p class="rec-figs">${raw(figures)}</p>
    <p class="rec-cond mono">${num(LAST.inputs.context)} ctx · ${LAST.inputs.kv_type}${
      LAST.inputs.n_seq > 1 ? " · " + LAST.inputs.n_seq + " seqs" : ""}</p>
    ${raw(deltas ? h`<div class="deltas"><p class="sublabel">WHY THE TWO ANSWERS DIFFER</p>${
      raw(deltas)}</div>` : "")}
    ${raw(other)}
    ${raw(foot)}
    <div class="actions">
      <button class="ghost" type="button" data-action="rec-script">Build a launch script &rarr;</button>
      <button class="ghost" type="button" data-action="set-step" data-step="measure">${
        measured ? "See the measurements" : "Measure it for real"}</button>
    </div>
  </section>`;
}

/** Take the recommendation to step 3 with its own row selected.
 *  Goes through the existing pick path rather than adding a second way to set
 *  the same state - two paths to one selection is how the page once managed to
 *  show one row checked and build the script from another. */
function recToScript(){
  if(REC && REC.row){
    const key = rowKey(REC.row);
    PICKABLE[key] = REC.row;
    SWEEP.pickKey = key;
    SWEEP.script = null;
  }
  setStep("script");
}

/* ------------------------------------------------------------ result cards */
function renderVerdict(r){
  const c = r.config, p = r.plan, s = r.sizes_mib, inp = r.inputs;
  // The state comes from the server now. The browser used to re-derive its own
  // reading of the same fields, and the step tab's copy of that logic read two
  // keys this API has never returned - so it announced "fits" for every plan,
  // including the ones that did not.
  const vd = p.verdict || {};
  const vcls = { fits: "ok", tight: "warn", spills: "bad", no_fit: "bad" }[vd.state] || "warn";

  const vramSegs = [
    { cls:"s-wt",  color:"var(--wt)",   label:"weights (GPU)",     mib:p.gpu_weights_mib || 0 },
    { cls:"s-kv",  color:"var(--kv)",   label:"KV cache (GPU)",    mib:p.gpu_kv_mib || 0 },
    { cls:"s-rec", color:"var(--rec)",  label:"recurrent state",   mib:p.gpu_recurrent_mib || 0 },
    { cls:"s-prj", color:"var(--proj)", label:"vision projector",  mib:p.mmproj_mib || 0 },
    { cls:"s-spc", color:"var(--spec)", label:"MTP draft cache",   mib:p.spec_mib || 0 },
    { cls:"s-cmp", color:"var(--cmp)",  label:"compute buffer",    mib:p.compute_mib || 0 },
    { cls:"s-rsv", color:"",            label:"driver reserve",    mib:inp.gpu_reserve_mib || 0 }
  ];
  const ramSegs = [
    { cls:"s-wt",  color:"var(--wt)",  label:"weights (CPU/RAM)",      mib:p.cpu_weights_mib || 0 },
    { cls:"s-kv",  color:"var(--kv)",  label:"KV cache (RAM)",         mib:p.cpu_kv_mib || 0 },
    { cls:"s-rec", color:"var(--rec)", label:"recurrent state (RAM)",  mib:p.cpu_recurrent_mib || 0 },
    { cls:"s-cmp", color:"var(--cmp)", label:"compute buffer (CPU)",   mib:p.cpu_compute_mib || 0 }
  ];
  const vramUsed = vramSegs.reduce((a, x) => a + x.mib, 0);
  const ramUsed = ramSegs.reduce((a, x) => a + x.mib, 0);

  const cal = r.calibration && r.calibration.calibrated
    ? h`The compute buffer is <b style="color:var(--kv)">calibrated for your GPU</b> from ${
        r.calibration.n} measurement${r.calibration.n == 1 ? "" : "s"} (${
        r.calibration.free.join(", ")} fitted, in-sample ${r.calibration.residual_pct}%)${
        r.calibration.when ? ", fitted " + new Date(r.calibration.when * 1000)
          .toLocaleDateString(undefined, {year:"numeric", month:"short", day:"numeric"}) : ""
        }. These coefficients are frozen until you press <b>Measure running model</b> again, so
        the same plan always gives the same numbers. ` +
      (r.calibration.outdated
        ? h`<b style="color:var(--warn)">The stored fit may not match this machine:</b> ${
            r.calibration.outdated}. ` : "")
    : "The compute buffer uses shipped defaults, fitted to 146 measured llama.cpp loads over 5 " +
      "models. Held out by architecture it scores 22.5% mean / 85.6% worst on the buffer alone, " +
      "and the whole plan lands at 7.6% mean / 39.6% worst against the process counter. It is " +
      "the only term that depends on your hardware rather than the model &mdash; press " +
      "<b>Measure running model</b> with a model loaded to calibrate it for yours. ";

  const split = p.cpu_compute_mib > 0.5
    ? h`Split across backends here: ${fmt(p.compute_mib)} on the GPU, ${fmt(p.cpu_compute_mib)
        } in RAM. llama.cpp gives every backend running part of the graph its own scratch pool. The ${
        fmt(s.compute_output)} output tensor (4 bytes × ${num(c.n_vocab || 0)
        } vocab) is host memory wherever the layers run.`
    : h`All of it (${fmt(p.compute_mib)}) is on the GPU at this split.`;

  return h`<section class="card lead">
    <h2>Does it fit</h2>
    <div class="verdict ${vcls}">
      <p class="vword">${vd.word || "—"}</p>
      <p class="h">${p.headline}</p>
    </div>
    ${raw(bar("VRAM", inp.vram_budget_mib, vramUsed, vramSegs))}
    ${raw(ramUsed > 0.5 ? bar("System RAM", inp.ram_budget_mib, ramUsed, ramSegs) : "")}
    <p class="note">This is the <b>largest split that fits</b> — the planner stops at the first
      config under the budget. It is not a claim about speed, and on a model with recorded
      measurements it is usually not the fastest config either. Weights and KV are computed
      exactly from the GGUF tensor table.
      <details class="why"><summary>how the compute buffer is arrived at</summary>
        ${raw(cal)}${raw(split)}</details></p>
  </section>`;
}

function renderWarnings(r){
  if(!r.warnings || !r.warnings.length) return "";
  return h`<section class="card warns"><h2>Notes</h2>${
    raw(r.warnings.map(w => h`<div>&#9888; ${w}</div>`).join(""))}</section>`;
}

function renderSettings(r){
  const p = r.plan;
  let tail = "";
  if(p.max_ctx_kv_gpu != null)
    tail = h`<p class="note">Max context with <b>all KV on GPU</b> (FFN on CPU): ~<b>${num(p.max_ctx_kv_gpu)}</b> tokens.</p>`;
  else if(p.max_ctx_gpu != null)
    tail = h`<p class="note">Max context fully on GPU at this quant: ~<b>${num(p.max_ctx_gpu)}</b> tokens.</p>`;

  return h`<section class="card">
    <h2>Predicted settings</h2>
    <p class="note">Where the planner thinks the split falls, from the model&rsquo;s exact
      weights and KV size. Good enough to load with &mdash; but it cannot know what
      speculative decoding will accept or how fast prefill runs, so once
      <b>Measured results</b> above has a row, that row is the better answer.</p>
    <div style="margin-bottom:12px">
      <p class="sublabel">LM STUDIO &middot; advanced load settings</p>
      ${raw((p.lmstudio || []).map(x => h`<div class="setrow">${x}</div>`).join(""))}
    </div>
    <div class="cmdhead">
      <p class="sublabel" style="margin:0">LLAMA.CPP</p>
      <button class="ghost" type="button" data-action="copy-cmd">copy</button>
    </div>
    <pre class="cmd" id="cmd">${p.llama_cmd || ""}</pre>
    ${raw(tail)}</section>`;
}

function renderSpeed(r){
  const sp = r.speed;
  if(!sp || sp.error) return "";
  const num_ = sp.calibrated
    ? h`<span class="hero ok">${sp.tok_s.toFixed(1)}</span> tok/s`
    : h`<span class="hero">${sp.tok_s_lo.toFixed(0)}&ndash;${sp.tok_s_hi.toFixed(0)}</span> tok/s`;
  return h`<section class="card">
    <h2>Generation speed</h2>
    <p style="margin:2px 0 14px">${raw(num_)}
      <span class="muted small" style="margin-left:10px">${
        sp.calibrated ? "calibrated · RAM at " + (sp.ram_eff * 100).toFixed(0) + "% of peak"
                      : "uncalibrated bracket · measure once to collapse it"}</span></p>
    <div class="tablewrap"><table><tbody>
      ${raw(brow("Read from VRAM per token", fmt(sp.gpu_mib)))}
      ${raw(brow("Read from system RAM per token", fmt(sp.cpu_mib) +
        (sp.cpu_mib > sp.gpu_mib / 4 ? '  <span style="color:var(--warn)">&larr; the bottleneck</span>' : "")))}
      ${raw(sp.expert_frac < 1 ? brow("Experts active per token",
        (sp.expert_frac * 100).toFixed(2) + "% of expert weights") : "")}
      ${raw(brow("Context assumed filled", num(sp.ctx_fill) + " tokens"))}
      ${raw(brow("Bandwidth used", Math.round(sp.bw_vram_gbs) + " GB/s VRAM · " +
        sp.bw_ram_gbs.toFixed(1) + " GB/s RAM"))}
    </tbody></table></div>
    <p class="note">Generation is memory-bandwidth bound: every token reads each active weight
      once. Byte counts are exact; the bandwidths are not, which is the whole width of the
      bracket. Prompt processing is compute bound and is <b>not</b> modelled here.</p>
    <div class="actions">
      <button class="ghost" type="button" data-action="bench">&#9654; Benchmark the loaded model</button>
      <span class="muted small">runs one short generation on the LM Studio server
        (localhost:1234) &mdash; do it while your agent is idle</span>
    </div>
    <div id="benchout"></div>
    <div id="spdhist"><p class="muted small">looking for past measurements…</p></div>
  </section>`;
}

function renderSummary(r){
  const c = r.config, s = r.sizes_mib, inp = r.inputs;
  return h`<section class="card">
    <h2>Model</h2>
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px">
      <span class="mono" style="font-size:var(--t-lead)">${r.model_name || "?"}</span>${raw(kvBadge(r))}
    </div>
    <div class="summary-grid">
      ${raw(kvItem("architecture", esc(c.arch)))}
      ${raw(kvItem("params (total)", B(r.params_total)))}
      ${raw(r.is_moe ? kvItem("params (active)", B(r.active_params)) : "")}
      ${raw(kvItem("quant", r.bpw.toFixed(2) + " bpw"))}
      ${raw(kvItem("layers", c.n_layers))}
      ${raw(kvItem("hidden size", c.hidden))}
      ${raw(kvItem("attn heads", esc(c.n_head + " / kv " + c.n_head_kv)))}
      ${raw(r.hybrid && r.hybrid.is_hybrid
        ? kvItem("KV-bearing layers", r.hybrid.n_attn_layers + " of " + c.n_layers +
            " (hybrid; " + r.hybrid.n_ssm_layers + " linear/SSM)") : "")}
      ${raw(r.swa && r.swa.enabled
        ? kvItem("sliding window", num(r.swa.n_swa) + " tok · " + r.swa.n_swa_layers +
            " windowed / " + r.swa.n_global_layers + " global") : "")}
      ${raw(c.n_mtp_layers
        ? kvItem("multi-token pred.", c.n_mtp_layers + " block" + (c.n_mtp_layers == 1 ? "" : "s") +
            ' <span class="muted">(' + (inp.mtp_spec ? "drafting: KV counted" : "idle: weights only") + ')</span>') : "")}
      ${raw(kvItem("head dim", (c.head_dim_k || "-") +
        ((r.swa && r.swa.enabled && r.swa.head_dim !== r.swa.head_dim_global)
          ? '  <span class="muted">/ ' + esc(r.swa.head_dim) + ' swa</span>' : "")))}
      ${raw(kvItem("native ctx", c.n_ctx_train ? num(c.n_ctx_train) : "-"))}
      ${raw(kvItem("weights", fmtG(s.weights)))}
    </div></section>`;
}

function renderKvTable(r){
  const c = r.config, s = r.sizes_mib, inp = r.inputs;
  const rows = (r.kv_table || []).map(t => {
    const cur = t.ctx === inp.context;
    return h`<tr class="${cur ? "cur" : ""}"><td>${num(t.ctx)}${cur ? "  ←" : ""}</td>
      <td>${fmt(t.kv_mib)}</td><td>${fmtG(t.kv_mib)}</td></tr>`;
  }).join("");

  const swa = (r.swa && r.swa.enabled)
    ? h`<p class="note">This model uses <b>sliding-window attention</b>, so the table is not a
        straight line. ${r.swa.n_swa_layers} layers only ever cache ${num(r.swa.window_cache_tokens)
        } tokens (a ${num(r.swa.n_swa)}-token window × ${inp.n_seq} seq + one ubatch, padded)
        &mdash; ${fmt(s.kv_swa)}, flat no matter how long the context gets. Only the ${
        r.swa.n_global_layers} full-attention layers grow with context (${fmt(s.kv_global)} here, ${
        s.kv_grow_per_token_kib.toFixed(1)} KiB/token). Detected from <code>${r.swa.source}</code>.</p>`
    : "";

  return h`<section class="card">
    <h2>KV cache vs context (${inp.kv_type})</h2>
    <div class="tablewrap"><table>
      <thead><tr><th>context</th><th>KV size</th><th></th></tr></thead>
      <tbody>${raw(rows)}</tbody></table></div>
    <p class="note">Per token: ${s.kv_per_token_kib.toFixed(1)} KiB across ${
      (r.hybrid && r.hybrid.is_hybrid)
        ? r.hybrid.n_attn_layers + " KV-bearing of " + c.n_layers + " layers — this is a hybrid model, the other " +
          r.hybrid.n_ssm_layers + " layers use a fixed recurrent state"
        : "all " + c.n_layers + " layers"}.</p>
    ${raw(swa)}</section>`;
}

function renderBreakdown(r){
  const c = r.config, p = r.plan, s = r.sizes_mib, inp = r.inputs;
  return h`<section class="card">
    <h2>Memory breakdown</h2>
    <div class="tablewrap"><table><tbody>
      ${raw(brow("Model weights (exact)", fmt(s.weights) + "  (" + fmtG(s.weights) + ")"))}
      ${raw(brow("&nbsp;&nbsp;token embeddings", fmt(s.embed)))}
      ${raw(brow("&nbsp;&nbsp;output / head", fmt(s.output)))}
      ${raw(r.is_moe ? brow("&nbsp;&nbsp;routed experts (offloadable)", fmt(s.expert_total)) : "")}
      ${raw(brow("Per-layer weight (avg)", fmt(s.per_layer_mean)))}
      ${raw(brow("KV cache @ " + num(inp.context) + " ctx", fmt(s.kv_total) +
        ((r.hybrid && r.hybrid.is_hybrid)
          ? '  <span class="muted">(' + r.hybrid.n_attn_layers + " attn layers)</span>" : "")))}
      ${raw((r.swa && r.swa.enabled)
        ? brow("&nbsp;&nbsp;full-attention layers (" + r.swa.n_global_layers + ")", fmt(s.kv_global)) +
          brow("&nbsp;&nbsp;sliding-window layers (" + r.swa.n_swa_layers + ", fixed)", fmt(s.kv_swa))
        : "")}
      ${raw(s.recurrent_total > 0.5
        ? brow("Recurrent state (fixed, x" + inp.n_seq + " seq)", fmt(s.recurrent_total)) : "")}
      ${raw(brow("Compute buffer (est.)", fmt(s.compute)))}
      ${raw(brow("&nbsp;&nbsp;graph scratch + attn mask", fmt(s.compute_graph)))}
      ${raw(brow("&nbsp;&nbsp;CUDA context + kernel modules", fmt(s.compute_floor)))}
      ${raw(brow("&nbsp;&nbsp;output tensor (4B x " + num(c.n_vocab || 0) + " vocab x " +
        inp.n_seq + " slot" + (inp.n_seq == 1 ? "" : "s") + ")",
        fmt(s.compute_output) +
        '  <span class="muted">host memory, not VRAM</span>'))}
      ${raw(r.mmproj ? brow("Vision projector (" + esc(r.mmproj.name) + ")",
        fmt(r.mmproj.mib) + (r.mmproj.included ? "" : '  <span class="muted">not loaded</span>')) : "")}
      ${raw(r.vision && r.vision.peak ? brow(
        "Vision encoder peak (" + num(r.vision.grid.width) + "&times;" + num(r.vision.grid.height) +
        ", " + num(r.vision.grid.n_patches) + " patches)",
        fmt(r.vision.peak.total_mib) +
        '  <span class="muted">transient, derived not measured</span>') : "")}
      ${raw(r.vision && r.vision.peak ? brow(
        "&nbsp;&nbsp;attention scores (quadratic in patches)",
        fmt(r.vision.peak.scores_mib) + (r.vision.peak.flash_attn
          ? '  <span class="muted">fused away</span>'
          : '  <span class="muted">' +
            Math.round(100 * r.vision.peak.scores_mib / Math.max(1, r.vision.peak.total_mib)) +
            '% of the peak</span>')) : "")}
      ${raw(r.vision && r.vision.peak ? brow("&nbsp;&nbsp;activations + FFN",
        fmt(r.vision.peak.act_mib + r.vision.peak.ffn_mib)) : "")}
      ${raw(r.vision && r.vision.peak ? brow("&nbsp;&nbsp;image tokens added to context",
        num(r.vision.peak.image_tokens) + " tok") : "")}
      ${raw(brow("File on disk", fmtG(s.file_on_disk) + "  (" + fmtGB(s.file_on_disk) + ")"))}
      ${raw(r.mmproj ? brow("&nbsp;&nbsp;+ projector = LM Studio's &quot;model size&quot;",
        fmtG(s.bundle_on_disk) + "  (" + fmtGB(s.bundle_on_disk) + ")") : "")}
    </tbody></table></div></section>`;
}

/** Two tiers, in the order the questions actually get asked.
 *
 *  MEASURED comes first because it is the point: the verdict gates everything
 *  (there is no sense measuring a config that will not load), and then the sweep,
 *  its results and the script it produces. PREDICTED follows - the planner's
 *  estimate is what you use before you have spent the two hours, and what the
 *  measurements supersede once you have. */
function render(r){
  LAST = r;
  const c = r.config;
  $("mmprojfield").hidden = !r.mmproj;
  $("ncpumoefield").hidden = !r.is_moe;
  $("mtprow").hidden = !c.n_mtp_layers;
  if(r.mmproj){
    $("mmprojhint").innerHTML = h`${r.mmproj.name} &middot; ${fmt(r.mmproj.mib)
      } of VRAM. LM Studio loads it with the model and includes it in the size it shows.`;
  }
  // The image controls only mean anything for a projector with a VISION tower -
  // an audio-only mmproj has no patch grid to size.
  $("visionrow").hidden = !(r.vision && r.vision.config);
  $("visioninputs").hidden = !$("visionplan").checked;
  // The projector controls live in Tune, which is closed by default. A model
  // that ships one has a real decision to make there and would otherwise have
  // no sign of it, so the tier says what it is holding rather than springing
  // open and undoing whatever the user had set.
  const tn = $("tuneNote");
  if(tn){
    tn.textContent = r.mmproj ? "batch, sequences, vision projector"
                              : "batch, sequences";
    tn.classList.toggle("flagged", !!r.mmproj);
  }
  const step = stepNow();
  let panel;
  if(step === "fit"){
    // The gate, and then everything the planner DERIVED - which is reference
    // material, not a next action, so it opens closed.
    panel = renderVerdict(r) + renderWarnings(r) +
      tier("PREDICTED", "calculated from the model's own metadata — step 2 supersedes it") +
      h`<details class="adv derived"><summary>Settings the planner suggests, its speed
        estimate, and where the memory goes</summary>` +
      renderSettings(r) + renderSpeed(r) + renderSummary(r) +
      renderKvTable(r) + renderBreakdown(r) +
      h`</details>`;
  }else if(step === "measure"){
    panel = renderSweep() + renderHistory();
  }else{
    panel = renderScriptStep();
  }
  $("out").innerHTML = h`<div id="rec"></div>` + renderStepper(r) +
                       renderGlossary() + panel;
  drawRecommendation();
  // Needs the recorded rows and a round trip, and the plan is what the button
  // was pressed for - so it fills in after the page rather than holding it up.
  loadRecommendation(r);
  if(r.speed && !r.speed.error) loadSpeedHistory(r);
  // Filled in asynchronously: it needs a live GPU reading and the recorded rows,
  // and neither should hold up the plan the user actually pressed the button for.
  initSweep(r);
}

/* ------------------------------------------------------------- benchmark */
async function benchNow(){
  const out = $("benchout");
  out.innerHTML = '<p class="note">generating… this occupies the GPU for a few seconds</p>';
  const r = LAST;
  let q = "?max_tokens=128";
  if(r && r.inputs) q += "&ctx=" + r.inputs.context;
  if(r && r.plan && r.plan.n_gpu_layers != null) q += "&ngl=" + r.plan.n_gpu_layers;
  let d;
  try{ d = await (await fetch("/api/benchmark" + q)).json(); }
  catch(e){ out.innerHTML = h`<p class="note" style="color:var(--warn)">benchmark failed: ${e}</p>`; return; }
  if(d.error || !d.tok_s){
    out.innerHTML = h`<p class="note" style="color:var(--warn)">${d.error || "no throughput returned"}
      <br>Is a model loaded and the server running on localhost:1234?</p>`;
    return;
  }
  const fill = (d.prompt_tokens || 0) + Math.round((d.predicted_tokens || 0) / 2);
  out.innerHTML = h`<p class="note"><b style="color:var(--kv)">${d.tok_s.toFixed(2)} tok/s</b>
    measured on <b>${d.model}</b> &middot; ${d.predicted_tokens} tokens${
      d.ttft_s != null ? " · TTFT " + Number(d.ttft_s).toFixed(2) + "s" : ""} &middot; source: ${d.source}
    <button class="ghost" type="button" style="margin-left:8px" data-action="use-measured"
            data-toks="${d.tok_s}" data-fill="${fill}">calibrate from this</button>
    <br>Saved to speed_history.json, so it shows up in the list below from now on.</p>`;
  loadSpeedHistory(LAST);
}

async function loadSpeedHistory(r){
  const box = $("spdhist");
  if(!box) return;
  let hist;
  try{
    hist = await (await fetch("/api/speedhistory?model=" + encodeURIComponent(r.model_name || "") +
      "&path=" + encodeURIComponent($("path").value || $("model").value || ""))).json();
  }catch(e){ box.innerHTML = h`<p class="muted small">history read failed: ${e}</p>`; return; }
  if(hist.error){ box.innerHTML = h`<p class="muted small">no measurement history: ${hist.error}</p>`; return; }

  const ms = hist.matches || [];
  if(!ms.length){
    const n = (hist.all || []).length;
    box.innerHTML = h`<p class="note">No past measurements for <b>${r.model_name || "this model"}</b>
      in LM Studio’s saved chats${n ? " (" + n + " record" + (n == 1 ? "" : "s") + " for other models)" : ""}.
      Generate once in LM Studio, then re-run &mdash; the tok/s it reports gets saved with the chat
      and shows up here to calibrate against.</p>`;
    return;
  }
  const rows = ms.slice(0, 8).map(m => {
    const fill = (m.prompt_tokens || 0) + Math.round((m.predicted_tokens || 0) / 2);
    return h`<tr><td>${m.tok_s.toFixed(2)} tok/s</td>
      <td class="muted">ngl ${m.n_gpu_layers == null ? "?" : m.n_gpu_layers} &middot; ctx ${
        m.ctx ? num(m.ctx) : "?"} &middot; ~${num(fill)} filled</td>
      <td style="text-align:right"><button class="ghost" type="button" data-action="use-measured"
        data-toks="${m.tok_s}" data-fill="${fill}">calibrate</button></td></tr>`;
  }).join("");

  box.innerHTML = h`<div style="margin-top:16px">
    <p class="sublabel">MEASURED IN LM STUDIO${
      ms[0].match < 3 ? " (approximate model match — check the config)" : ""}</p>
    <div class="tablewrap"><table><tbody>${raw(rows)}</tbody></table></div>
    <p class="note">Picking one sets <b>Context filled</b> to that run and solves for the real RAM
      efficiency, then re-runs. Only calibrate against a run whose layer split matches the plan
      above &mdash; otherwise you are fitting the wrong configuration.</p>
    </div>` + srvLogPanel(hist);
}

function srvLogPanel(hist){
  const sl = hist.server_log;
  if(!Array.isArray(sl) || !sl.length) return "";
  const n = Math.min(sl.length, 20), recent = sl.slice(0, n);
  const pre = recent.reduce((a, x) => a + x.prefill_s, 0);
  const dec = recent.reduce((a, x) => a + x.decode_s, 0);
  if(pre + dec <= 0) return "";
  const pct = Math.round(pre / (pre + dec) * 100);
  return h`<div style="margin-top:16px">
    <p class="sublabel">SERVER MODE &middot; LAST ${n} RESPONSES</p>
    <div class="tablewrap"><table><tbody>
      ${raw(brow("Prompt processing (prefill)", pre.toFixed(0) + " s total &middot; <b>" + pct + "%</b> of the time"))}
      ${raw(brow("Token generation (decode)", dec.toFixed(0) + " s total &middot; " + (100 - pct) + "%"))}
    </tbody></table></div>
    <p class="note">Wall times parsed from <span class="mono">~/.lmstudio/server-logs</span>. The
      succinct log has no token counts, so this cannot give tok/s &mdash; but it does show where
      your time actually goes. ${raw(pct >= 50
        ? "<b>Prefill dominates here</b>, and prefill is compute bound, so layer-split tuning will " +
          "not fix it &mdash; a shorter prompt or a reused prefix will."
        : "Decode dominates, which is what the bandwidth model above predicts.")}</p>
    </div>`;
}

async function useMeasured(tokS, fill){
  $("ctxfill").value = Math.max(0, Math.round(fill));
  $("rameff").value = "";              // clear so the solve is not anchored to an old value
  const r = LAST;
  if(!r || !r.speed) return;
  // solve: t_total = gpu_bytes/(BWv*GPU_EFF) + cpu_bytes/(BWr*eff)
  const sp = r.speed, GPU_EFF = 0.85;
  const gpuB = sp.gpu_mib * 1048576, cpuB = sp.cpu_mib * 1048576;
  const tTot = 1 / tokS, tGpu = gpuB / (sp.bw_vram_gbs * GPU_EFF * 1e9), tCpu = tTot - tGpu;
  if(cpuB <= 0 || tCpu <= 0){
    $("calhint").innerHTML = '<b style="color:var(--warn)">That measurement is faster than the GPU ' +
      'term alone allows &mdash; it was a different config (or a different context fill). Not calibrating.</b>';
    return;
  }
  $("rameff").value = (cpuB / (tCpu * sp.bw_ram_gbs * 1e9)).toFixed(3);
  run();
}

/* ----------------------------------------------------------- calibration */
function calibHint(pending){
  const tips = {
    ctx: "measure again at a very different context length",
    act: "measure a model with a different FFN width, or change ubatch",
    nofa: "measure once with flash attention off"
  };
  return pending.map(t => tips[t]).filter(Boolean).join("; ") + ".";
}

async function calibrate(){
  const box = $("calhint");
  if(!LAST){
    box.innerHTML = '<b style="color:var(--warn)">Press Analyze fit first, so the exact terms are known.</b>';
    return;
  }
  box.textContent = "reading GPU processes…";
  let r;
  try{ r = await (await fetch("/api/gpuprocs")).json(); }
  catch(e){ box.textContent = "could not read GPU processes: " + e; return; }

  const procs = r.procs;
  if(!Array.isArray(procs) || !procs.length){
    box.innerHTML = h`<b style="color:var(--warn)">No GPU processes readable${
      procs && procs.error ? " (" + procs.error + ")" : ""}.</b>`;
    return;
  }
  const eng = procs.find(p => p.is_engine) || procs[0];
  const p = LAST.plan, inp = LAST.inputs;

  // A measurement is only meaningful against a plan for the SAME config. LM Studio
  // silently overrides its own GPU Offload slider to respect its VRAM cap, so read
  // what it actually loaded and refuse to back-solve against a different split -
  // otherwise every layer of difference lands in the compute buffer as error.
  const rt = r.runtime;
  if(rt){
    const bad = [];
    if(rt.n_gpu_layers != null && rt.n_gpu_layers !== p.n_gpu_layers)
      bad.push(h`GPU layers: engine is running <b>${rt.n_gpu_layers}</b>, this plan is for <b>${
        p.n_gpu_layers}</b>${p.forced_ngl ? "" : " (auto-picked)"}`);
    if(rt.context != null && rt.context !== inp.context)
      bad.push(h`Context: engine is running <b>${num(rt.context)}</b>, this plan is for <b>${num(inp.context)}</b>`);
    if(bad.length){
      box.innerHTML = h`<b style="color:var(--warn)">Config mismatch — not measuring.</b><br>${
        raw(bad.join("<br>"))}<br>Set <b>GPU layers</b> to ${rt.n_gpu_layers}${
        rt.context != null ? " and context to " + num(rt.context) : ""}, press Analyze fit, then
        Measure again. <button class="ghost" type="button" data-action="match-runtime"
          data-ngl="${rt.n_gpu_layers}" data-ctx="${rt.context || 0}">do it for me</button>`;
      return;
    }
  }

  // Record it as a calibration sample rather than a one-off override: the server
  // recomputes the exact terms, stores the row, and refits the coefficients for
  // this GPU. That makes every future plan better, not just this one.
  box.textContent = "measuring and refitting…";
  let res;
  try{
    res = await (await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: currentPath(),
        context: parseInt($("ctx").value) || 8192,
        kv_type: $("kv").value,
        n_ubatch: parseInt($("ubatch").value) || 512,
        n_seq: parseInt($("nseq").value) || 1,
        flash_attn: $("fa").checked,
        include_mmproj: $("mmproj").checked
      })
    })).json();
  }catch(e){ box.textContent = "calibration failed: " + e; return; }

  if(!res.ok){ box.innerHTML = h`<b style="color:var(--warn)">${res.error || "failed"}</b>`; return; }

  const row = res.row, st = res.status;
  const pending = CALIB_TERMS.filter(t => st.free.indexOf(t) < 0);
  const fit = st.calibrated
    ? h`<b style="color:var(--kv)">Calibrated</b> from ${st.n} measurement${st.n == 1 ? "" : "s"}
        on this GPU &mdash; fitted: ${st.free.join(", ")} (in-sample ${st.residual_pct}%).` +
      h`<br><span class="muted">Saved &mdash; these coefficients are now frozen and will not
        change on their own. Measuring again is the only thing that refits them.</span>` +
      (st.skipped_rows ? h`<br><span class="muted">${st.skipped_rows} stored measurement${
        st.skipped_rows == 1 ? " was" : "s were"} left out: the reading did not respond to the
        config, or the layer count was never recorded. Re-measure with VRAM to spare to bring
        them back.</span>` : "") +
      (st.outdated ? h`<br><span style="color:var(--warn)">${st.outdated}</span>` : "")
    : '<b style="color:var(--warn)">Recorded, but not fitted yet.</b> The measurement is saved; ' +
      'it did not produce a usable fit on its own, so the shipped defaults still apply. ' +
      'Measure once more at a different context length.';

  box.innerHTML = h`<b style="color:var(--kv)">${eng.name}</b> is using <b>${fmt(row.measured_mib)}</b>
    at ngl ${row.ngl} / ctx ${num(row.ctx)} <span class="muted">(read from LM Studio’s log)</span>.<br>
    minus exact terms ${fmt(row.exact_mib)} &rarr; overhead <b>${fmt(row.overhead_mib)}</b>.<br>
    ${raw(fit)}${raw(pending.length
      ? h`<br><span class="muted">Still on defaults: ${pending.join(", ")}. ${calibHint(pending)}</span>`
      : "")}<br>Press Analyze fit again.`;
  run();
}

async function matchRuntime(ngl, ctx){
  $("ngl").value = ngl;
  if(ctx){ $("ctx").value = ctx; markCtx(); }
  await run();
  calibrate();
}

function copyCmd(btn){
  const text = $("cmd").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const old = btn.textContent;
    btn.textContent = "copied";
    setTimeout(() => { btn.textContent = old; }, 1200);
  });
}

/* ------------------------------------------------------------ speed sweep */
/* The planner models memory. It cannot model speculative decoding (no
 * acceptance-rate term) or prompt processing (compute bound, not modelled at
 * all), so those can only be measured. This card drives that measurement and
 * turns whatever wins into a launcher. */

let SWEEP = null;

function sweepDefaults(){
  // rowsLoaded distinguishes "no rows recorded" from "the fetch has not landed
  // yet". Without it the step-2 tab asserted "nothing measured yet" for the
  // second or two before /api/speed/rows returned - on a model with 65 recorded
  // rows, which is exactly the claim the tab exists to make and exactly the one
  // it was getting wrong.
  return { path: "", model: "", preflight: null, rows: [], rowsLoaded: false, plan: null,
           status: null, since: 0, timer: null, script: null, pickKey: null,
           shell: (SYS && SYS.shell) || "bash", busy: "",
           campaigns: null, insights: null, openCampaign: null,
           templates: [], speeddir: "",
           // Which campaign is being asked about, which is mid-delete, and what
           // the last delete did. Deleting takes two clicks and the second one
           // reports what it removed and where it put it.
           delKey: null, delBusy: null, delNote: "" };
}

/* Four cards, not one. They are rewritten on different clocks: the grid and the
 * results follow the 1.5s poll, the launch script must NOT - it holds a dozen
 * text inputs, and rebuilding it under the poll destroyed whatever was being
 * typed into them once every second and a half. */
function renderSweep(){
  return h`<section class="card lead" id="sweepcard">
    <p class="note">Two of the settings that matter most to tokens/second cannot be calculated:
      speculative decoding has no acceptance rate until you run it, and prompt processing is
      compute bound and is not modelled at all. This drives
      <span class="mono">llama-server</span> across a grid and records what it really does.</p>
    <div id="sweepbody"><p class="muted small">checking the GPU&hellip;</p></div>
  </section>
  <section class="card" id="sweepresultcard">
    <h2>Measured results</h2>
    <div id="sweepresults"><p class="muted small">looking for recorded rows&hellip;</p></div>
  </section>`;
}

/** Step 3 on its own. Kept out of renderSweep() because it must NOT be rewritten
 *  on the 1.5s poll - it holds a dozen text inputs, and redrawing it under the
 *  poll destroyed whatever was being typed once every second and a half. */
function renderScriptStep(){
  return h`<section class="card lead" id="sweepscriptcard">
    <div id="sweepscript"></div>
  </section>` + renderHistory();
}

function renderHistory(){
  return h`<section class="card" id="sweephistcard">
    <h2>Past sweeps</h2>
    <p class="note">Every campaign ever run on this machine, and what it found. Rows live in
      <span class="mono">speed/</span> keyed by GPU and llama.cpp build, so a run from last
      month is as usable as one from this hour &mdash; including for building a script.</p>
    <div id="sweephistory"><p class="muted small">reading recorded campaigns&hellip;</p></div>
  </section>`;
}

function tier(label, note){
  return h`<div class="tier"><span>${label}</span><i>${note}</i></div>`;
}

/* --------------------------------------------------------------- glossary
 *
 * Six abbreviations head every row table on this page and none of them was
 * defined anywhere. They are llama.cpp's own spellings, which is the right
 * choice - they are what you type at the command line - but a column header is
 * not a place to learn a vocabulary from. Defined once here and reused as the
 * `title` on each <th>, so the two cannot drift. */
const TERMS = [
  ["ngl", "GPU layers", "How many transformer blocks live in VRAM. llama.cpp offloads the LAST n, so this is a count and not a list."],
  ["ncmoe", "CPU expert layers", "On an MoE only: the routed experts of the first n blocks are pushed to system RAM, leaving their attention and KV in VRAM. A different knob from ngl, and usually the one that matters — experts are most of the file and only a few of them fire per token."],
  ["ub", "ubatch", "Physical batch size: how many tokens are pushed through the graph at once during prompt processing. Sizes the compute buffer."],
  ["fill", "context filled", "How many tokens of context were actually present when the row was measured. Decode re-reads the KV cache every token, so a number taken at 2k says little about 40k."],
  ["spec", "speculation", "Speculative decoding: a cheap draft proposes tokens and the real model checks them. draft-mtp uses the model's own multi-token-prediction blocks; the ngram variants need nothing extra."],
  ["proj", "projector", "Where the vision projector's weights sit — VRAM or system RAM. Text decode is unaffected by moving it, so those megabytes come back as GPU layers."],
  ["accepted", "acceptance rate", "The share of drafted tokens the target model kept, and how many were drafted at all. Read them together: 100% of 15 drafted out of 128 generated is a 7% gain."]
];

function termTitle(k){
  const t = TERMS.find(x => x[0] === k);
  return t ? t[1] + " — " + t[2] : "";
}

function renderGlossary(){
  return h`<details class="card glossary"><summary>What the column names mean</summary>
    <dl>${raw(TERMS.map(([k, name, desc]) =>
      h`<dt class="mono">${k}</dt><dd><b>${name}.</b> ${desc}</dd>`).join(""))}</dl>
  </details>`;
}

/* ---------------------------------------------------------------- the steps
 *
 *  Three things anyone actually comes here to do, in the order they depend on
 *  each other: does it FIT, how FAST is it really, and what do I RUN.
 *
 *  They used to be twelve stacked sections in one scrolling column - verdict,
 *  notes, measure, results, script, past sweeps, predicted settings, speed
 *  estimate, model, KV table, memory breakdown - with no signal about which
 *  ones were answers and which were controls. Every one of them was visible at
 *  once, so none of them was the next thing to do.
 *
 *  Only the active step renders. The others collapse into the stepper's own
 *  labels, which carry their answer - "FITS · 11.4 GB", "MEASURED · 3.95 tok/s"
 *  - so nothing is hidden that you would have to go looking for. */
const STEPS = [
  ["fit",     "Does it fit"],
  ["measure", "Measure real speed"],
  ["script",  "Launch script"],
];

let STEP = null;          // null = follow the default for the current state

function stepNow(){
  if(STEP) return STEP;
  // A campaign running is the thing you opened the page for, whatever else is
  // on it. Otherwise start at the gate: measuring a config that cannot load is
  // an hour spent proving it cannot load.
  if(SWEEP && SWEEP.status && SWEEP.status.status === "running") return "measure";
  return LAST ? "fit" : "measure";
}

function stepSummary(id, r){
  if(id === "fit"){
    if(!r) return "no model analyzed";
    // r.verdict and r.totals never existed on this API. `r.verdict || {}` was
    // therefore always {}, `{}.fits === false` always false, and this tab said
    // "fits" for every plan ever analyzed - including the ones that spilled.
    // The state is computed once, server-side, in plan._verdict().
    const v = (r.plan && r.plan.verdict) || {};
    const word = { fits: "fits", tight: "fits, barely",
                   spills: "spills to shared", no_fit: "does not fit" }[v.state] || "planned";
    return word + (v.vram_mib ? " · " + fmtG(v.vram_mib) : "");
  }
  if(id === "measure"){
    const st = SWEEP && SWEEP.status;
    if(st && st.status === "running")
      return "measuring " + st.done + "/" + (st.total || "?");
    // "nothing measured yet" is a claim, so it waits until the rows are in.
    if(!(SWEEP && SWEEP.rowsLoaded)) return "reading recorded rows…";
    const rows = sweepAllRows ? sweepAllRows() : [];
    const best = rows.find(x => x.tok_s);
    return best ? best.tok_s.toFixed(2) + " tok/s best" : "nothing measured yet";
  }
  const row = (typeof sweepPickedRow === "function") ? sweepPickedRow() : null;
  const c = row ? row.config : (SWEEP && SWEEP.predicted);
  return c ? "ngl " + c.ngl + (c.spec && c.spec !== "none" ? " · " + c.spec : "")
           : "not ready";
}

function renderStepper(r){
  const now = stepNow();
  const tabs = STEPS.map(([id, label], i) => {
    const on = id === now;
    const busy = id === "measure" && SWEEP && SWEEP.status
                 && SWEEP.status.status === "running";
    return h`<button class="step ${on ? "on" : ""} ${busy ? "busy" : ""}" type="button"
      data-action="set-step" data-step="${id}" aria-pressed="${on ? "true" : "false"}">
      <b>${String(i + 1)}</b><span>${label}</span><i>${stepSummary(id, r)}</i></button>`;
  }).join("");
  return h`<nav class="steps" id="stepper" aria-label="What do you want to do">${raw(tabs)}</nav>`;
}

/** Refresh only the tab labels.
 *
 *  They carry the answer of every step that is NOT open - "3.95 tok/s best",
 *  "measuring 4/9" - which is the entire reason collapsing the other two is
 *  acceptable. Left out of the poll they went stale immediately: rows load
 *  after the first paint, so step 2 sat on "nothing measured yet" for a model
 *  with 145 recorded rows. */
function drawStepper(){
  const el = $("stepper");
  if(el) el.outerHTML = renderStepper(LAST);
}

function setStep(id){
  STEP = id;
  if(LAST) render(LAST); else renderIdle();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function initSweep(r){
  if(SWEEP && SWEEP.timer) clearInterval(SWEEP.timer);
  SWEEP = sweepDefaults();
  SWEEP.path = currentPath();
  // Speed rows are keyed by FILE name. r.model_name is the display name out of
  // the GGUF metadata ("Qwen3.8-27B"), which is not the same string and silently
  // matches nothing.
  SWEEP.model = SWEEP.path.split(/[\\/]/).pop();
  SWEEP.mmproj = !!(r && r.mmproj);
  // The planner's own answer is the fallback config: a script is useful before
  // anyone has spent two hours measuring, it just has to say that it is a guess.
  if(r && r.plan) SWEEP.predicted = {
    ctx: r.inputs ? r.inputs.context : parseInt($("ctx").value),
    kv: $("kv").value, fa: $("fa").checked || $("kv").value !== "f16",
    seq: parseInt($("nseq").value) || 1, ub: parseInt($("ubatch").value) || 512,
    ngl: r.plan.n_gpu_layers, ncmoe: r.plan.n_cpu_moe || 0
  };
  PICKABLE = {};
  // The row and campaign loads refresh the tabs as soon as THEY land, rather
  // than when the whole batch does. loadPreflight() shells out to nvidia-smi and
  // enumerates GPU processes, so it is seconds slower than the rest - and step
  // 2's tab was sitting on "reading recorded rows…" for that whole time with the
  // rows already in hand.
  await Promise.all([loadPreflight(),
                     loadSweepRows().then(drawStepper),
                     loadHistory().then(drawStepper),
                     loadTemplates(), attachRunningJob()]);
  // Pre-select this model's campaign, so Past sweeps opens on what was just
  // analyzed instead of on whatever ran most recently.
  // `mine.length === 1` used to be right, because a model had exactly one
  // campaign entry. Splitting the index by prompt and template means a model
  // that has been measured more than once now has several, and an equality on
  // 1 quietly stopped opening anything at all. Campaigns arrive newest first,
  // so the most recent one is the one to open - which is the same intent.
  const mine = (SWEEP.campaigns || []).filter(g => g.model === SWEEP.model);
  if(mine.length) openCampaign(campaignId(mine[0]));
  drawSweepAll();
}

/** Pick a campaign that is already running back up.
 *
 *  The job lives in the server, not the page, so a reload - or opening a second
 *  tab - used to leave a two-hour sweep running with nothing watching it: no
 *  progress, no log, no Stop, and a new Start refused as "already running" with
 *  no visible reason. Ask once on load, and resume polling if there is one. */
async function attachRunningJob(){
  let st;
  try{ st = await (await fetch("/api/speed/status?since=0")).json(); }
  catch(e){ return; }
  if(!st || st.status !== "running") return;
  SWEEP.status = st;
  SWEEP.since = st.log_next || 0;
  if(SWEEP.timer) clearInterval(SWEEP.timer);
  SWEEP.timer = setInterval(pollSweep, 1500);
}

async function loadPreflight(){
  try{ SWEEP.preflight = await (await fetch("/api/speed/preflight")).json(); }
  catch(e){ SWEEP.preflight = null; }
}

async function loadTemplates(){
  if(!SWEEP.path){ SWEEP.templates = []; return; }
  try{
    const d = await (await fetch("/api/templates?path=" +
                                encodeURIComponent(SWEEP.path))).json();
    SWEEP.templates = d.templates || [];
  }catch(e){ SWEEP.templates = []; }
}

async function loadSweepRows(){
  try{
    const d = await (await fetch("/api/speed/rows?model=" +
                                encodeURIComponent(SWEEP.model))).json();
    SWEEP.rows = d.rows || [];
  }catch(e){ SWEEP.rows = []; }
  SWEEP.rowsLoaded = true;
}

/** Redraw the sweep panes.
 *
 *  `what` names which ones. The launch-script pane is excluded by default and is
 *  the reason this function takes an argument at all: it is rebuilt from state on
 *  every call, so redrawing it under the 1.5s poll wiped the port, bind host, load
 *  mode, chat-template path and all six sampler fields while a two-hour campaign
 *  ran. Nothing in it depends on the poll, so nothing in it should follow it. */
function drawSweep(what){
  what = what || {};
  // Each pane is guarded on its own. Before anything is analyzed the page shows
  // only Past sweeps and the launch script, so a single early return on a missing
  // grid would leave both of those unfilled.
  const b = $("sweepbody");
  if(what.grid !== false && b){
    const st = SWEEP.status;
    b.innerHTML = (st && st.status === "running")
      ? sweepRunning(st)
      : sweepIdle() + (st && st.status === "failed"
          ? h`<p class="note" style="color:var(--warn)">Sweep failed: ${st.error}</p>` : "");
  }
  if(what.results !== false){
    const rr = $("sweepresults");
    if(rr) rr.innerHTML = sweepResults();
  }
  if(what.script){
    // The pane is rebuilt from state, so anything living only in the DOM is
    // about to be thrown away. Pressing Generate is not a request to lose the
    // template path you just typed, or to have the section you opened fold shut.
    captureScriptForm();
    const sc = $("sweepscript");
    if(sc) sc.innerHTML = sweepScript();
  }
  if(what.history){
    const hh = $("sweephistory");
    if(hh) hh.innerHTML = sweepHistory();
  }
  drawStepper();
}

/** Everything, including the panes that hold live inputs. Only safe when the
 *  user just acted on one of them, or when the card is being built fresh. */
function drawSweepAll(){
  drawSweep({ grid: true, results: true, script: true, history: true });
}

/* -- 1. blocked, or ready ------------------------------------------------- */
function sweepIdle(){
  const pf = SWEEP.preflight;
  if(pf && !pf.ok){
    const who = (pf.holders || []).map(x =>
      h`<li><b>${x.name}</b> &middot; pid ${x.pid} &middot; ${fmt(x.mib)}${
        x.is_engine ? " (an inference engine)" : ""}</li>`).join("");
    return h`<div class="warns" style="margin-bottom:14px">
      <div>&#9888; <b>The GPU is not free, so measuring is blocked.</b></div>
      <p class="note" style="margin:8px 0 6px">${pf.reason}</p>
      ${raw(who ? "<ul class='muted small' style='margin:6px 0 6px 18px'>" + who + "</ul>" : "")}
      <p class="note" style="margin:6px 0 0">A model held resident does not make the sweep
        <i>fail</i> &mdash; it makes every row wrong in the same direction, which is worse,
        because the numbers still look like numbers. Close it, then
        <button class="ghost" type="button" data-action="sweep-recheck">re-check</button></p>
    </div>` + sweepFormDisabled();
  }
  return sweepForm(pf);
}

function sweepFormDisabled(){
  return h`<p class="muted small">The grid controls appear once the card is free.</p>`;
}

function sweepForm(pf){
  const free = pf && pf.total_mib
    ? h`<span class="muted small">${fmt(pf.free_mib)} of ${fmt(pf.total_mib)} free</span>` : "";
  return h`<div class="sweepform">
    <p class="sublabel">GRID &middot; context and KV quant are taken from the form above and
      frozen, not swept ${raw(free)}</p>
    <div class="chips" role="group" aria-label="Stages">
      ${raw(sweepStage("a", "A &middot; layer wall", "How many blocks fit before it spills"))}
      ${raw(sweepStage("b", "B &middot; projector", "Vision tower in VRAM or in system RAM"))}
      ${raw(sweepStage("c", "C &middot; ubatch", "Physical batch size"))}
      ${raw(sweepStage("d", "D &middot; speculation", "MTP draft depths and the n-gram types"))}
    </div>
    <div class="row" style="margin-top:10px">
      <div class="field"><label for="swfill">Context filled (tokens)</label>
        <input type="number" id="swfill" value="2048" step="1024" min="0"></div>
      <div class="field"><label for="swpred">Tokens per pass</label>
        <input type="number" id="swpred" value="128" step="32" min="16"></div>
      <div class="field"><label for="swrep">Passes (median)</label>
        <input type="number" id="swrep" value="3" step="1" min="1"></div>
      <div class="field"><label for="swlimit">Stop after (blank = all)</label>
        <input type="number" id="swlimit" step="1" min="1" placeholder="all"></div>
    </div>
    <div class="field" style="max-width:24em">
      <label for="swmmproj">Vision projector</label>
      <select id="swmmproj">
        <option value="">sweep it (that is stage B)</option>
        <option value="vram">keep in VRAM</option>
        <option value="ram">move to system RAM</option>
      </select>
      <p class="hint">Pin it when you already know where you want it and are sweeping something else.
        <details class="why"><summary>why it is worth an axis</summary>
          Where the projector lives is a real axis &mdash; it was worth ~900 MiB of VRAM on one
          model, which bought back whole expert layers. Sweeping it is stage B. Before this
          control the only way to pin it was to write
          <span class="mono">mmproj_offload=false</span> into the axes box.</details></p>
    </div>
    <p class="hint">Measure where you actually work: decode slows as the context fills, so a
      number taken at 2k is not the speed you feel at 40k. A deeper fill costs real time
      &mdash; the prompt is processed once per config.</p>
    <div class="field">
      <label class="check"><input type="checkbox" id="swchain" checked> Chain the stages
        &mdash; measure each knob at the split that won</label>
      <p class="hint">Same number of loads, so the same hours.
        <details class="why"><summary>what chaining changes</summary>
          Off, every stage runs from one fixed baseline, so ubatch and speculation
          are measured at a layer count <i>stage A has not confirmed</i>. On, each stage is built
          after the last one finishes, from the fastest trustworthy row so far. A row that
          spilled into shared memory or that caught the model looping is never carried forward
          &mdash; it would bend every later stage the same way, silently. A new baseline also has
          to win by more than 2%, because tok/s is a median of a few passes and rebasing on
          jitter would make the campaign&rsquo;s path depend on noise.</details></p>
      <div class="field" id="swroundsfield" style="max-width:22em">
        <label for="swrounds">Rounds</label>
        <input type="number" id="swrounds" value="1" min="1" max="4" step="1">
        <p class="hint">Runs A&ndash;D again from the winner. Cheap: anything unchanged is skipped.
          <details class="why"><summary>why a second round finds anything</summary>
            A and B choose the memory split <i>before</i> D turns speculation on, and speculation
            costs VRAM &mdash; so the wall A found has moved by the time D is done. A second round
            re-walks the split with the speculation that won already switched on.
            <b>The preview cannot price it</b>: which configs a later round adds depends on what
            this one finds. It is bounded by construction though &mdash; anything a round revisits
            unchanged is already recorded and is skipped, so only new combinations cost
            time.</details></p>
      </div>
    </div>
    <div class="field">
      <label class="check"><input type="checkbox" id="swverify"> Verify the winner with
        the production config</label>
      <p class="hint">One extra load: the winning config plus the knobs below.
        <details class="why"><summary>why the winner is not the config you run</summary>
          The row that wins is often not the row you run &mdash; MTP&rsquo;s draft cache alone
          moved the OOM wall one <span class="mono">ngl</span> rung on one model &mdash; so this
          measures what you would actually launch, e.g.
          <span class="mono">spec=draft-mtp spec_n_max=2</span>, and says if it is
          trustworthy.</details></p>
      <div class="field" id="swverifyfield" style="max-width:28em">
        <label for="swverifyoverrides">Overrides (same grammar as axes)</label>
        <input type="text" id="swverifyoverrides" placeholder="spec=draft-mtp spec_n_max=2">
      </div>
    </div>
    <details class="adv" id="swaxesbox"><summary>Sweep exact values instead of the stages</summary>
      <p class="hint">The staged grid moves one knob at a time from a baseline, which cannot
        answer a question about an <b>interaction</b>. Stage D only ever tries speculation at the
        split stage B settled on &mdash; so if that split is already at the memory ceiling, every
        speculative row OOMs and the campaign reads as &ldquo;speculation does not work here&rdquo;
        when the truth is &ldquo;speculation needs one more rung of offload&rdquo;.</p>
      <div class="field">
        <label for="swaxes">Axes</label>
        <input type="text" id="swaxes"
               placeholder="ncmoe=30,31,32,33 spec=draft-mtp spec_n_max=2 mmproj_offload=false">
        <p class="hint">Space-separated <span class="mono">key=v,v,v</span>, run as a full cross
          product. Anything left out keeps the value from the form above.
          <b>This replaces the stages and the chaining</b> &mdash; a ladder you wrote down is
          already the search, so there is nothing left to chain.</p>
      </div>
    </details>
    ${raw(sweepAskSection())}
    <div class="actions">
      <button class="ghost" type="button" data-action="sweep-plan">Preview grid</button>
      <button class="go" type="button" style="width:auto" data-action="sweep-start">&#9654; Start measuring</button>
    </div>
    <div id="sweepplan"></div>
  </div>`;
}

/** How the model is ASKED, while it is being measured.
 *
 *  These are not axes. They are frozen for the campaign, exactly like context
 *  and KV quant, and they are here because leaving them out made every campaign
 *  measure something other than what the launcher would go on to run:
 *
 *    - no template, so an instruction-tuned model was handed raw text and
 *      continued the document instead of answering it
 *    - greedy sampling, which is speculation's BEST case, so an acceptance
 *      rate measured here is an upper bound and not a result
 *
 *  The generated script now warns when it is launching a config that differs
 *  from the row it cites. This is the other half: the way to make it not
 *  differ. */
function sweepAskSection(){
  const tl = (SWEEP && SWEEP.templates) || [];
  const num = (id, label, ph) => h`<div class="field"><label for="${id}">${label}</label>
    <input type="number" id="${id}" step="0.01" placeholder="${ph}"></div>`;
  return h`<details class="adv" id="swaskbox"><summary>How the model is asked while measuring
      &mdash; template, thinking, samplers</summary>
    <p class="hint">Frozen for the campaign, not swept. Set these to <b>what you will actually
      run</b> &mdash; rows are keyed on the template&rsquo;s content hash and on the samplers, so
      changing anything here re-measures rather than resuming.</p>
    <div class="field">
      <label for="swtmplfile">Chat template file</label>
      <input type="text" id="swtmplfile" list="tmpllist2"
             placeholder="&mdash; the model&rsquo;s own &mdash;">
      <datalist id="tmpllist2">${raw(tl.map(t => h`<option value="${t}"></option>`).join(""))}</datalist>
      <p class="hint">Without one the benchmark still asks through the server&rsquo;s
        <span class="mono">/apply-template</span>, so it uses whatever the GGUF carries. Pin the
        file here when the launcher will pin it, or the two are measuring different prompts.
        Quotes around a pasted Windows path are stripped.</p>
    </div>
    <div class="field">
      <label for="swtmplkw">Template keyword arguments (JSON object)</label>
      <input type="text" id="swtmplkw" placeholder='{"reasoning_effort":"xhigh"}'>
    </div>
    <div class="row">
      <div class="field"><label for="swreason">Thinking</label>
        <select id="swreason"><option value="auto">auto</option>
          <option value="on">on</option><option value="off">off</option></select></div>
      <div class="field"><label for="swreasonpre">Preserve thinking</label>
        <select id="swreasonpre"><option value="default">template default</option>
          <option value="on">on</option><option value="off">off</option></select></div>
    </div>
    <p class="sublabel" style="margin-top:14px">SAMPLERS</p>
    <p class="hint">Blank means <b>greedy</b> (temp 0) &mdash; the right default for comparing
      configs, and the wrong one for believing an acceptance rate.
      <details class="why"><summary>why greedy flatters speculation</summary>
        Greedy makes the token stream reproducible, so two rows differ by the knob under test and
        nothing else. But llama.cpp accepts a draft token when the target&rsquo;s own sampled
        token matches it, and under greedy that comparison is deterministic &mdash; greedy is
        speculation&rsquo;s best case. Fill these in before drawing a speculation
        conclusion.</details></p>
    <div class="row">${raw(num("swtemp", "temp", "0 (greedy)") + num("swtopk", "top-k", "0")
      + num("swtopp", "top-p", "1.0") + num("swminp", "min-p", "0"))}</div>
    <div class="row">${raw(num("swreppen", "repeat-penalty", "1.0")
      + num("swprespen", "presence-penalty", "0"))}</div>
  </details>`;
}

function sweepStage(letter, label, hint){
  return h`<label class="chip" title="${hint}"><input type="checkbox" class="swstage"
    value="${letter}" checked> ${raw(label)}</label>`;
}

function sweepStages(){
  return Array.from(document.querySelectorAll(".swstage"))
    .filter(x => x.checked).map(x => x.value).join("") || "a";
}

function sweepBody(){
  const v = id => { const el = $(id); return el && el.value !== "" ? parseInt(el.value) : null; };
  const chain = $("swchain") ? $("swchain").checked : true;
  const verify = $("swverify") ? $("swverify").checked : false;
  return { path: SWEEP.path, stages: sweepStages(), context: parseInt($("ctx").value),
           kv_type: $("kv").value, fill: v("swfill"), n_predict: v("swpred"),
           repeat: v("swrep"), limit: v("swlimit"),
           chain: chain, rounds: chain ? (v("swrounds") || 1) : 1,
           verify: verify,
           verify_overrides: verify ? ($("swverifyoverrides").value.trim() || null) : null,
           axes: ($("swaxes") && $("swaxes").value.trim()) || null,
           // "" = sweep it (stage B's job). Otherwise pinned for the campaign,
           // which is a different thing from absent and has to survive as one.
           mmproj_offload: (($("swmmproj") && $("swmmproj").value) || "") === ""
             ? null : $("swmmproj").value === "ram" ? false : true,
           ...sweepAskBody() };
}

/** The "how the model is asked" fields, in the shapes the server expects.
 *  Sampler names are the launch-script card's, so one vocabulary reaches both
 *  and web.py does the single mapping to a sweep config's shorter spelling. */
function sweepAskBody(){
  const t = id => { const el = $(id); return (el && el.value.trim()) || null; };
  const n = id => { const el = $(id); return el && el.value.trim() !== ""
                                      ? parseFloat(el.value) : null; };
  const sel = id => { const el = $(id); return el ? el.value : null; };
  const sampling = {};
  [["swtemp", "temp"], ["swtopk", "top_k"], ["swtopp", "top_p"],
   ["swminp", "min_p"], ["swreppen", "repeat_penalty"],
   ["swprespen", "presence_penalty"]].forEach(([id, k]) => {
     const v = n(id); if(v !== null) sampling[k] = v;
   });
  return {
    chat_template_file: t("swtmplfile"),
    chat_template_kwargs: t("swtmplkw"),
    // auto/default are llama.cpp's own answers, sent as null so no flag is
    // emitted rather than one restating a default that could later move
    reasoning: sel("swreason") === "auto" ? null : sel("swreason"),
    reasoning_preserve: sel("swreasonpre") === "default" ? null : sel("swreasonpre"),
    sampling: sampling
  };
}

async function sweepPlan(){
  const box = $("sweepplan");
  box.innerHTML = '<p class="muted small">building the grid…</p>';
  let d;
  try{
    d = await (await fetch("/api/speed/plan", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sweepBody()) })).json();
  }catch(e){ box.innerHTML = h`<p class="note">preview failed: ${e}</p>`; return; }
  if(!d.ok){ box.innerHTML = h`<p class="note" style="color:var(--warn)">${d.error}</p>`; return; }
  const rows = (d.configs || []).map(c => h`<tr>
    <td>${c.stage || "-"}</td><td>${c.ngl}</td><td>${c.ncmoe || 0}</td><td>${c.ub}</td>
    <td>${num(c.fill || 0)}</td><td class="mono">${c.spec || "none"}</td>
    <td>${c.spec_n_max || 0}</td>
    <td>${c.mmproj_offload === false ? "RAM" : "VRAM"}</td></tr>`).join("");
  const sizes = d.stage_sizes || {};
  const later = Object.keys(sizes).sort().filter(k => k !== "-")
    .map(k => k.toUpperCase() + " " + sizes[k]).join(" &middot; ");
  box.innerHTML = h`<p class="note"><b>${d.planned}</b> config${d.planned == 1 ? "" : "s"}
      to run, ~<b>${d.estimate_h}</b> h${raw(d.skipped
        ? h`. ${d.skipped} already measured and will be skipped &mdash; rows are keyed by
            config <i>and</i> by how they were measured, so this resumes rather than repeats.`
        : ".")}</p>` +
    (d.provisional ? h`<p class="note">Chained, so only the <b>first stage</b> can be listed:
        the ones after it are built from a baseline that does not exist yet, and printing
        values for them would be a guess dressed up as a plan. The <i>counts</i> are exact
        &mdash; a ladder&rsquo;s length does not depend on where it is centred &mdash; so the
        estimate above is not a guess. Stages: ${raw(later)}.</p>` : "") +
    /* The estimate covers round 1 only, and said nothing about it - the preview
       was byte-identical for rounds 1 and rounds 4, so the control changed
       nothing anyone could inspect before committing the hours. */
    ((d.rounds || 1) > 1 ? h`<p class="note"><b>${String(d.rounds)} rounds.</b> The count and
        the estimate above are for <b>round 1</b>. What a later round adds cannot be listed or
        priced here &mdash; it is built from a winner that does not exist yet &mdash; but it is
        bounded: anything a round revisits unchanged is already recorded and is skipped, so a
        round only pays for combinations round 1 never tried.</p>` : "") +
    (d.planned ? h`<div class="tablewrap"><table>
      <thead><tr><th>stage</th><th>ngl</th><th>ncmoe</th><th>ub</th><th>fill</th>
        <th>spec</th><th>n-max</th><th>projector</th></tr></thead>
      <tbody>${raw(rows)}</tbody></table></div>` : "");
}

async function sweepStart(){
  let d;
  try{
    d = await (await fetch("/api/speed/start", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sweepBody()) })).json();
  }catch(e){ alert("could not start: " + e); return; }
  if(d.preflight) SWEEP.preflight = d.preflight;
  if(!d.ok){ drawSweep({ results: false }); alert(d.error || "could not start"); return; }
  SWEEP.since = 0;
  pollSweep();
  if(SWEEP.timer) clearInterval(SWEEP.timer);
  SWEEP.timer = setInterval(pollSweep, 1500);
}

async function sweepStop(){
  try{ await fetch("/api/speed/stop", { method: "POST" }); }catch(e){}
  pollSweep();
}

async function pollSweep(){
  let st;
  try{ st = await (await fetch("/api/speed/status?since=" + SWEEP.since)).json(); }
  catch(e){ return; }
  const prev = SWEEP.status;
  // The log ships incrementally, so keep what we already have and append.
  const kept = (prev && prev.status === st.status) ? (prev.log || []) : [];
  st.log = kept.concat(st.log || []);
  SWEEP.since = st.log_next;
  SWEEP.status = st;
  if(st.status !== "running"){
    if(SWEEP.timer){ clearInterval(SWEEP.timer); SWEEP.timer = null; }
    if(prev && prev.status === "running"){
      loadPreflight();
      // A finished campaign changes the history too, so refresh it once here -
      // once, not per poll.
      loadHistory();
      loadSweepRows().then(() => drawSweep({ history: true }));
      // New rows can change which config is recommended - that is the whole
      // point of having run the campaign - so the reconciliation is refetched
      // rather than left showing what was true two hours ago.
      REC_FOR = null;
      if(LAST) loadRecommendation(LAST);
    }
  }
  // Grid and results only. See drawSweep(): the script pane holds live inputs.
  drawSweep();
}

/* -- 2. running ----------------------------------------------------------- */
function sweepRunning(st){
  const pct = st.total ? clampPct(100 * st.done / st.total) : 0;
  const mins = st.elapsed_s != null ? (st.elapsed_s / 60).toFixed(1) : "?";
  const tail = (st.log || []).slice(-14).join("\n");
  /* Under a chained search the bar's denominator is THIS STAGE, not the
     campaign: a later stage's configs are built from a baseline that does not
     exist yet, so only the running one has a known list. The header meanwhile
     prints the campaign's estimate. Showing one and not the other made the
     other look like a mistake - "1 of 9" under a log saying "36 to run". */
  const chained = !!st.stage;
  const head = chained
    ? h`<b>Measuring</b> &mdash; ${st.stage} &middot; ${st.done} of ${st.total || "?"}
        &middot; ~${st.planned || "?"} planned for the campaign, ${mins} min elapsed`
    : h`<b>Measuring</b> &mdash; ${st.done} of ${st.total || "?"} configs, ${mins} min elapsed`;
  return h`<div class="running">
    <p>${raw(head)}
      ${raw(st.cancelling ? '<span style="color:var(--warn)">&middot; stopping</span>' : "")}</p>
    ${raw(chained ? h`<p class="hint" style="margin:-2px 0 8px">The bar tracks the stage that is
      running. Chained, the stages after it are built from a baseline that has not been measured
      yet, so their contents are not known in advance &mdash; the campaign figure is an estimate
      from the unchained grid and can move.</p>` : "")}
    <div class="prog"><i style="width:${pct}%"></i></div>
    <div class="actions">
      <button class="ghost" type="button" data-action="sweep-stop" ${
        st.aborting ? "disabled" : ""}>&#9632; ${st.cancelling && !st.aborting
          ? "Stop now &mdash; abandon this config" : "Stop"}</button>
      <span class="muted small">${raw(st.aborting
        ? "Abandoning the config in flight. Its row is discarded rather than recorded, so it " +
          "is re-measured next time rather than kept as a failure it never really was."
        : st.cancelling
          ? "Finishing the config in flight so its row is complete rather than half-written. " +
            "That can be minutes at a deep fill &mdash; nearly all of a config's time is one " +
            "blocking request to the server. <b>Press again</b> to kill it now instead."
          : "Stopping finishes the config in flight first, so its row is complete rather than " +
            "half-written. Nothing is lost either way &mdash; every row is already on disk and " +
            "keyed, so starting again resumes from here.")}</span>
    </div>
    ${raw(st.log_dropped ? h`<p class="muted small">${st.log_dropped} earlier log line${
      st.log_dropped == 1 ? "" : "s"} dropped</p>` : "")}
    <pre class="logbox">${tail}</pre>
  </div>`;
}

/* -- 3. results ----------------------------------------------------------- */
/** Rows measured in this job, merged over the recorded history. */
function sweepAllRows(){
  const live = (SWEEP.status && SWEEP.status.rows) || [];
  const ok = live.filter(r => r.status === "ok" && r.tok_s);
  const seen = new Set(ok.map(r => JSON.stringify(r.config)));
  const rest = SWEEP.rows.filter(r => !seen.has(JSON.stringify(r.config)));
  return ok.concat(rest).sort((a, b) => (b.tok_s || 0) - (a.tok_s || 0));
}

/** The caveats that make a headline number a lie. Every one of these was a wrong
 *  conclusion at some point before it was a badge. */
function rowFlags(r){
  const f = [];
  if(r.spilled) f.push(["spilled", "Loaded, but over-committed: WDDM spilled into shared " +
    "system memory instead of failing. The row says ok and the speed is off a cliff." +
    /* The EXCESS over the rest of the ladder, never the raw counter. Shared Usage
       counts host memory this process holds on purpose - pinned staging buffers,
       --no-mmproj-offload - so every healthy row reads in the hundreds of MiB. */
    (r.shared_excess != null ? " " + Math.round(r.shared_excess) +
      " MiB more host memory than the rest of its ladder." : "")]);
  /* Deduced from the campaign's own floors rather than read from a counter, so it is
     shown but NOT treated as untrustworthy: the fastest row in one real campaign carried
     this signature, because the extra layer bought more than the displaced memory cost. */
  else if(r.spill_inferred) f.push(["at the wall", "This config could not grow: " +
    Math.round(r.spill_inferred) + " MiB moved out of VRAM into system RAM, deduced from a " +
    "floor that fell that far below the rest of the campaign. Still a real measurement - a " +
    "ladder slowing at its top rung is the wall being found."]);
  /* Not a bad row - the decode figure is fine, it comes from the warm passes.
     Only PREFILL is unusable: the pass it was taken from also paid the one-time
     cost of faulting CPU-resident weights in, which a shallow prompt cannot
     amortise. Measured, ~100s of it on an ncmoe config at fill 2048. */
  if(r.prefill_warm !== true && r.prefill_tok_s)
    f.push(["prefill unwarmed", "The prefill figure on this row includes one-time load cost " +
      "- lazily-faulted CPU-resident weights and kernel init - because it was taken from the " +
      "first request after the server came up. At a shallow fill that dominates it: 2,065 " +
      "tokens took ~110s where the rate implies ~8s. Decode is unaffected; it comes from the " +
      "warm passes. Re-measure for a prefill number you can compare."]);
  if(r.templated === false) f.push(["no template", "This backend had no /apply-template, so " +
    "the model was handed raw text with nothing marking it as a request and merely continued " +
    "the document. Not comparable with templated rows."]);
  if(r.corpus_repeated && r.config && r.config.spec && r.config.spec !== "none")
    f.push(["filler repeats", "The prompt had to repeat to reach this depth, which inflates " +
      "any speculative acceptance rate. Compare only against other rows at this depth."]);
  if(r.distinct_ratio != null && r.distinct_ratio < 0.5)
    f.push(["looping", "Only " + Math.round(100 * r.distinct_ratio) + "% of 8-word windows " +
      "were distinct - the model was repeating itself, so this speed is not real work."]);
  if(r.copyback_ratio != null && r.copyback_ratio > 0.5)
    f.push(["copying", Math.round(100 * r.copyback_ratio) + "% of the output was a verbatim " +
      "copy of the prompt - the model stopped generating and echoed its context back, which " +
      "distinct_ratio cannot see and which inflates speculative acceptance exactly like " +
      "looping. Not real work."]);
  return f;
}

const ROW_HEAD = h`<thead><tr><th></th><th>tok/s</th><th>prefill</th>
  <th title="${termTitle("ngl")}">ngl</th><th title="${termTitle("ncmoe")}">ncmoe</th>
  <th title="${termTitle("ub")}">ub</th><th title="${termTitle("fill")}">fill</th>
  <th title="${termTitle("spec")}">spec</th><th title="${termTitle("proj")}">proj</th>
  <th title="${termTitle("accepted")}">accepted</th><th>VRAM</th><th></th>
  </tr></thead>`;

/** Identity of a measured row, for "which one did you pick".
 *  Not a list index: rows are now offered from two different tables over two
 *  different row sets, and an index into one of them means nothing in the other. */
function rowKey(r){
  return (r.model || "") + "|" + JSON.stringify(r.config || {});
}

/* key -> row, refilled as pickable rows render, so a click can find its row
 * whichever table it came from. */
let PICKABLE = {};

function sweepRow(r, i, pickable){
  const c = r.config || {}, flags = rowFlags(r);
  const failed = r.status && r.status !== "ok";
  const key = rowKey(r);
  if(pickable){
    PICKABLE[key] = r;
    // The FIRST pickable row to render adopts the selection outright, instead
    // of merely drawing itself checked and leaving SWEEP.pickKey null.
    //
    // Drawing it checked without recording it was two bugs at once. Pressing
    // Generate with nothing clicked fell through to sweepAllRows()[0], which
    // before any Analyze is empty - so a page showing a plainly selected row
    // answered "Analyze a model first". And once a model HAD been analyzed,
    // two tables each drew their own row 0 checked under one radio name, so
    // the browser showed the campaign's row selected while the script was
    // built from the results table's - silently the wrong row, which is worse.
    //
    // Now exactly one radio is ever checked, and it is the one that is used.
    if(SWEEP.pickKey == null) SWEEP.pickKey = key;
  }
  const on = pickable && SWEEP.pickKey === key;
  return h`<tr class="${on && pickable ? "best" : ""}">
    <td>${raw(pickable ? h`<input type="radio" name="swpick" data-action="sweep-pick"
      data-key="${key}" ${on ? "checked" : ""}>` : "")}</td>
    <td>${raw(failed ? h`<span style="color:var(--warn)">${r.status}</span>`
                     : h`<b>${(r.tok_s || 0).toFixed(2)}</b>`)}</td>
    <td>${(r.prefill_tok_s || 0).toFixed(0)}</td>
    <td>${c.ngl}</td><td>${c.ncmoe || 0}</td><td>${c.ub}</td>
    <td>${num(c.fill || 0)}</td>
    <td class="mono">${c.spec || "none"}${c.spec_n_max ? "/" + c.spec_n_max : ""}</td>
    <td>${c.mmproj_offload === false ? "RAM" : "VRAM"}</td>
    <td>${raw(r.accept_rate != null
      ? h`${Math.round(100 * r.accept_rate)}% <span class="muted small">of ${
          num(r.draft_n)}</span>` : "-")}</td>
    <td>${fmt(r.proc_vram_mib)}</td>
    <td>${raw(flags.map(f => h`<span class="flag" title="${f[1]}">${f[0]}</span>`).join(" "))}</td>
  </tr>`;
}

function sweepResults(){
  // Two tables, because they answer different questions. The run in progress is
  // read as a LADDER, in the order measured - that is how you see where the wall
  // is - and a slow row is the most informative one there. Ranking it against
  // every historical row would bury exactly that.
  // PICKABLE is deliberately NOT cleared here. This pane redraws on every poll
  // while the history pane does not, so clearing would drop a row picked out of
  // Past sweeps 1.5 seconds later. Entries are keyed by content, so a stale one
  // still maps to the row it named.
  const live = (SWEEP.status && SWEEP.status.rows) || [];
  const ranked = sweepAllRows();
  let out = "";
  if(live.length){
    out += h`<div style="margin-top:18px">
      <p class="sublabel">THIS RUN &middot; in the order measured</p>
      <div class="tablewrap"><table class="rowtable">${raw(ROW_HEAD)}
        <tbody>${raw(live.map((r, i) => sweepRow(r, i, false)).join(""))}</tbody></table></div>
    </div>`;
  }
  if(!ranked.length){
    return out || h`<p class="muted small" style="margin-top:14px">No speed rows recorded for
      this model yet.</p>`;
  }
  const best = ranked[0], shown = ranked.slice(0, 24);
  const fill = (best.config || {}).fill || 0;
  return out + h`<div style="margin-top:18px">
    <p class="sublabel">ALL RECORDED &middot; fastest first &middot; pick the one to build a
      script from ${ranked.length > 24 ? h`(showing 24 of ${ranked.length})` : ""}</p>
    <div class="tablewrap"><table class="rowtable">${raw(ROW_HEAD)}
      <tbody>${raw(shown.map((r, i) => sweepRow(r, i, true)).join(""))}</tbody></table></div>
    <p class="note">Read <b>accepted</b> together with the count beside it: 100% of 15 drafted
      tokens out of 128 generated is a 7% gain, not a miracle. And every figure here is
      conditional on the <b>fill</b> column &mdash; ${(best.tok_s || 0).toFixed(2)} tok/s at
      ${num(fill)} filled tokens says nothing about what you get at ${num(fill * 8)}.
      Hover any flag for what makes that row untrustworthy.</p>
  </div>`;
}

/* -- 4. the launcher ------------------------------------------------------ */
const SAMPLERS = ["temp", "top_k", "top_p", "min_p", "repeat_penalty",
                  "presence_penalty"];
const SCRIPT_FIELDS = ["swport", "swhost", "swload", "sm_tmplfile", "sm_tmplkw",
                       "sm_reason", "sm_reasonpre"]
  .concat(SAMPLERS.map(k => "sm_" + k));

/** Read the launch-script pane's inputs into SWEEP.form.
 *
 *  This pane is rendered from state, so state has to be where the values live.
 *  Without this, every redraw of it - pressing Generate, picking another row,
 *  switching shell - silently emptied every field the user had filled in. */
function captureScriptForm(){
  if(!$("sweepscript")) return;
  const f = SWEEP.form || (SWEEP.form = {});
  SCRIPT_FIELDS.forEach(id => { const el = $(id); if(el) f[id] = el.value; });
  f.open = Array.from(document.querySelectorAll("#sweepscript details"))
                .map(d => d.open);
}

/** A remembered field value, or the default the pane opens with. */
function fv(id, dflt){
  const f = SWEEP.form || {};
  return (f[id] === undefined || f[id] === null) ? (dflt == null ? "" : dflt) : f[id];
}
function fopen(i){ return ((SWEEP.form || {}).open || [])[i] ? "open" : ""; }

function sweepPickedRow(){
  if(SWEEP.pickKey && PICKABLE[SWEEP.pickKey]) return PICKABLE[SWEEP.pickKey];
  const rows = sweepAllRows();
  return rows.length ? rows[0] : null;
}

function sweepScript(){
  const row = sweepPickedRow();
  const shells = ((SYS && SYS.shells) || ["powershell", "bash"]).map(s =>
    h`<label class="chip"><input type="radio" name="swshell" value="${s}" ${
      SWEEP.shell === s ? "checked" : ""} data-action="sweep-shell"> ${
      s === "powershell" ? "PowerShell (.ps1)" : "bash (.sh)"}</label>`).join("");
  const modes = ((SYS && SYS.load_modes) || ["none", "mmap", "mlock", "mmap+mlock", "dio"])
    .map(m => h`<option value="${m}" ${m === fv("swload", "none") ? "selected" : ""}>${m}</option>`).join("");
  const src = row
    ? h`the <b>measured</b> row selected above (${(row.tok_s || 0).toFixed(2)} tok/s)`
    : h`the planner's <b>predicted</b> split &mdash; nothing has been measured for this model yet,
        and the script will say so`;
  /* No heading here. This renders INTO the "Launch script" card, which already
     has one - the sublabel was left behind when the block moved out of the
     "Measure real speed" card into its own, and printed the title twice. */
  return h`<div>
    <p class="note">Built from ${raw(src)}.</p>
    <details class="adv"><summary>What the script takes care of</summary>
      <p class="hint">Resolves the llama.cpp backend fresh at every launch, so a pinned path
        cannot stop existing when LM Studio updates. Puts the vendor CUDA libraries on the path
        &mdash; without them the process dies with <span class="mono">STATUS_DLL_NOT_FOUND</span>
        and no message at all. Writes a dated log file, because llama.cpp does not rotate one.
        Uses <span class="mono">--load-mode</span> and
        <span class="mono">--spec-draft-n-max</span> rather than the deprecated spellings that
        are accepted and then silently ignored.</p>
    </details>
    <div class="chips">${raw(shells)}</div>
    <div class="row" style="margin-top:10px">
      <div class="field"><label for="swport">Port</label>
        <input type="number" id="swport" value="${fv("swport", 8080)}" step="1"></div>
      <div class="field"><label for="swhost">Listen on</label>
        <select id="swhost">
          <option value="127.0.0.1" ${fv("swhost", "127.0.0.1") === "127.0.0.1" ? "selected" : ""
            }>127.0.0.1 (this machine only)</option>
          <option value="0.0.0.0" ${fv("swhost", "127.0.0.1") === "0.0.0.0" ? "selected" : ""
            }>0.0.0.0 (also WSL and the LAN)</option>
        </select>
        <p class="hint">WSL2 is a separate VM behind NAT, so Windows&rsquo; loopback is not its
          loopback &mdash; 127.0.0.1 is unreachable from there.</p></div>
      <div class="field"><label for="swload">Weight loading</label>
        <select id="swload">${raw(modes)}</select>
        <p class="hint"><span class="mono">none</span> reads each tensor straight to its final
          home. Avoid <span class="mono">mmap+mlock</span> under heavy GPU offload: mlock pins
          the whole mapped file, including blocks already resident in VRAM.</p></div>
    </div>
    <details class="adv" ${raw(fopen(0))}><summary>Chat template (optional)</summary>
      <p class="hint">Overrides the template baked into the GGUF.
        <details class="why"><summary>what the script does about a bad path</summary>
          Useful when the conversion predates a fixed template, or when the model card ships a
          patched one for tool calls. A path that does not exist is <b>not</b> an error to
          llama-server &mdash; it falls back to the built-in template without saying so &mdash; so
          the generated script checks it and refuses to start. It also passes
          <span class="mono">--jinja</span> <i>before</i> the template flags, because without it a
          build accepts only its built-in template <i>names</i> and rejects a path
          outright.</details></p>
      <div class="field">
        <label for="sm_tmplfile">Template file</label>
        <input type="text" id="sm_tmplfile" list="tmpllist" value="${fv("sm_tmplfile")}"
               placeholder="&mdash; use the model&rsquo;s own &mdash;">
        <datalist id="tmpllist">${raw(((SWEEP && SWEEP.templates) || [])
          .map(t => h`<option value="${t}"></option>`).join(""))}</datalist>
        <p class="hint" id="tmplhint">${(SWEEP && SWEEP.templates && SWEEP.templates.length)
          ? SWEEP.templates.length + " .jinja file(s) found next to the model — pick one or paste a path"
          : "No .jinja files next to the model; paste a full path."}</p>
      </div>
      <div class="field">
        <label for="sm_tmplkw">Template keyword arguments (JSON object)</label>
        <input type="text" id="sm_tmplkw" value="${fv('sm_tmplkw')}"
               placeholder='{"enable_thinking": false}'>
        <p class="hint">Checked for valid JSON before the script is written. Server defaults &mdash;
          a client sending its own wins for that request.
          <details class="why"><summary>why a typo here is silent</summary>
            The key names belong to <b>the template</b>, not to llama.cpp:
            <span class="mono">enable_thinking</span> is Qwen3&rsquo;s spelling and is
            <i>ignored, not rejected</i>, by a model that does not use that variable.</details></p>
      </div>
      <div class="row">
        <div class="field">
          <label for="sm_reason">Thinking</label>
          <select id="sm_reason">
            <option value="auto"${fv('sm_reason') === 'on' || fv('sm_reason') === 'off' ? "" : " selected"}>auto — detect from the template</option>
            <option value="on"${fv('sm_reason') === 'on' ? " selected" : ""}>on</option>
            <option value="off"${fv('sm_reason') === 'off' ? " selected" : ""}>off</option>
          </select>
          <p class="hint">Replaces <span class="mono">enable_thinking</span> in the kwargs above,
            which current builds accept and then warn about:
            <i>&ldquo;Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use
            --reasoning on / --reasoning off instead.&rdquo;</i></p>
        </div>
        <div class="field">
          <label for="sm_reasonpre">Preserve thinking across history</label>
          <select id="sm_reasonpre">
            <option value="default"${fv('sm_reasonpre') === 'on' || fv('sm_reasonpre') === 'off' ? "" : " selected"}>template default</option>
            <option value="on"${fv('sm_reasonpre') === 'on' ? " selected" : ""}>on</option>
            <option value="off"${fv('sm_reasonpre') === 'off' ? " selected" : ""}>off</option>
          </select>
          <p class="hint">Keeps the reasoning trace for the <b>whole</b> history, not just the
            last assistant message.
            <details class="why"><summary>why the kwargs above cannot do this</summary>
              This one <b>cannot</b> be set from the kwargs even when the template has a variable
              for it (Qwen3 spells it <span class="mono">preserve_thinking</span>): llama-server
              strips <span class="mono">&lt;think&gt;</span> out of the history <i>before</i>
              rendering, so by the time the template reads the variable there is nothing left to
              preserve. Only shows from the second turn on &mdash; which is how it survives a
              benchmark and then quietly loses the trace in daily use.</details></p>
        </div>
      </div>
    </details>
    <details class="adv" ${raw(fopen(1))}><summary>Sampling (optional)</summary>
      <p class="hint">Left blank, no sampler flags are written at all &mdash; a made-up default is
        worse than none. llama.cpp&rsquo;s own are temp 0.80, top-k 40, min-p 0.05, which several
        model cards do <i>not</i> want, so fill these in from yours. Server defaults: a client
        sending its own overrides them per request.</p>
      <div class="row">
        ${raw(SAMPLERS
          .map(k => h`<div class="field"><label for="sm_${k}">${k.replace(/_/g, " ")}</label>
            <input type="number" id="sm_${k}" step="0.01" value="${fv("sm_" + k)}"
                   placeholder="&mdash;"></div>`).join(""))}
      </div>
    </details>
    <div class="actions">
      <button class="ghost" type="button" data-action="sweep-gen">Generate script</button>
      ${raw(SWEEP.script ? h`
        <button class="ghost" type="button" data-action="sweep-copy">copy</button>
        <button class="ghost" type="button" data-action="sweep-save">save next to the model</button>
        <button class="ghost" type="button" data-action="sweep-dl">download</button>` : "")}
      <span class="muted small" id="swscripthint">${SWEEP.busy}</span>
    </div>
    ${raw(SWEEP.script ? h`<pre class="cmd" id="swscript">${SWEEP.script.text}</pre>` : "")}
  </div>`;
}

function sweepScriptBody(){
  const row = sweepPickedRow();
  const sampling = {};
  SAMPLERS.forEach(k => {
    const el = $("sm_" + k);
    if(el && el.value !== "") sampling[k] = parseFloat(el.value);
  });
  const cfg = row ? row.config : SWEEP.predicted;
  const val = id => { const el = $(id); return (el && el.value.trim()) || null; };
  return { path: SWEEP.path,
           // A row picked out of Past sweeps may belong to a model this page has
           // never analyzed, and rows record a basename rather than a path. The
           // server resolves it, and says so when it cannot.
           model_name: (row && row.model) || SWEEP.model || null,
           config: cfg, shell: SWEEP.shell,
           mmproj: (cfg && cfg.mmproj) ? cfg.mmproj : (SWEEP.mmproj || false),
           sampling: sampling,
           chat_template_file: val("sm_tmplfile"),
           chat_template_kwargs: val("sm_tmplkw"),
           // "auto"/"default" are llama.cpp's own answers, so they are sent as
           // null - meaning no flag at all rather than a flag that restates the
           // default and would then be wrong if the default ever moved.
           reasoning: val("sm_reason") === "auto" ? null : val("sm_reason"),
           reasoning_preserve: val("sm_reasonpre") === "default"
             ? null : val("sm_reasonpre"),
           port: parseInt(($("swport") || {}).value || 8080),
           bind_host: ($("swhost") || {}).value || "127.0.0.1",
           load_mode: ($("swload") || {}).value || "none",
           measured: row || null };
}

async function sweepGen(save){
  const body = sweepScriptBody();
  if(!body.config){ alert("Analyze a model first."); return; }
  SWEEP.busy = save ? "saving…" : "generating…";
  drawSweep({ grid: false, results: false, script: true });
  let d;
  try{
    d = await (await fetch(save ? "/api/script/save" : "/api/script", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) })).json();
  }catch(e){ SWEEP.busy = "failed: " + e; drawSweep({ grid: false, results: false, script: true }); return; }
  if(!d.ok){ SWEEP.busy = d.error || "failed"; drawSweep({ grid: false, results: false, script: true }); return; }
  SWEEP.script = d;
  SWEEP.busy = d.saved ? "saved to " + d.saved : "";
  drawSweep({ grid: false, results: false, script: true });
}

/* -- 5. past sweeps -------------------------------------------------------- */
async function loadHistory(){
  try{
    const d = await (await fetch("/api/speed/history")).json();
    SWEEP.campaigns = d.campaigns || [];
    SWEEP.speeddir = d.dir || "";
  }catch(e){ SWEEP.campaigns = []; }
}

/* prompt_id and template_id are part of the identity, not decoration: the same
   model on the same build can hold several campaigns that asked different
   questions, and they must not open each other. */
function campaignId(g){
  return [g.model, g.gpu, g.file, g.prompt_id || "", g.template_id || ""]// A separator that survives an HTML attribute round trip. This id goes out
  // as data-id and comes back through el.dataset.id, and a control
  // character does NOT make that trip: it is dropped, the returned id stops
  // matching the one campaignRow() computes, and every campaign silently
  // refuses to open.
  .join("~");
}

async function openCampaign(id){
  if(SWEEP.openCampaign === id){         // second click closes it
    SWEEP.openCampaign = null;
    drawSweep({ grid: false, results: false, history: true });
    return;
  }
  SWEEP.openCampaign = id;
  SWEEP.insights = null;
  drawSweep({ grid: false, results: false, history: true });
  const g = (SWEEP.campaigns || []).find(x => campaignId(x) === id);
  if(!g) return;
  const q = "model=" + encodeURIComponent(g.model) + "&gpu=" + encodeURIComponent(g.gpu) +
            "&file=" + encodeURIComponent(g.file) +
            // always sent, even empty - "" narrows to the rows that have none,
            // where omitting the key would widen to all of them
            "&pid=" + encodeURIComponent(g.prompt_id || "") +
            "&tid=" + encodeURIComponent(g.template_id || "");
  try{ SWEEP.insights = await (await fetch("/api/speed/insights?" + q)).json(); }
  catch(e){ SWEEP.insights = { ok: false, error: String(e) }; }
  drawSweep({ grid: false, results: false, history: true });
}

function when(ts){
  return ts ? new Date(ts * 1000).toISOString().slice(0, 10) : "?";
}

function campaignRow(g){
  const id = campaignId(g), open = SWEEP.openCampaign === id;
  const span = (g.first && g.last && when(g.first) !== when(g.last))
    ? when(g.first) + " → " + when(g.last) : when(g.last);
  return h`<div class="camp ${open ? "open" : ""}">
    <div class="camphead">
    <button class="camprow" type="button" data-action="sweep-open" data-id="${id}">
      <span class="mono">${g.model}</span>
      <span class="muted small">${g.gpu || "?"} &middot; ${g.backend} &middot; ${span}
        &middot; ${raw(g.prompt_id ? "prompt " + h`${g.prompt_id.slice(0, 8)}`
                       : "<i>pre-freeze prompt</i>")}
        &middot; ${raw(g.chat_template
                       ? "template " + h`${g.chat_template}`
                       : (g.template_id ? "template " + h`${g.template_id.slice(0, 8)}`
                          : "<i>model&rsquo;s own template</i>"))}</span>
      <span class="campnum">${g.n_ok}/${g.n_rows} ok${raw(
        g.n_untrusted ? " &middot; " + g.n_untrusted + " flagged" : "")}${raw(
        g.n_ungated ? " &middot; " + g.n_ungated + " pre-copy-gate" : "")}</span>
      <span class="campbest">${g.best_tok_s ? g.best_tok_s.toFixed(2) + " tok/s" : "—"}</span>
      <span class="campcaret">${open ? "▾" : "▸"}</span>
    </button>
    <button class="ghost campdel" type="button" data-action="sweep-del" data-id="${id}"
            title="Forget this campaign" aria-label="Forget this campaign">&#10005;</button>
    </div>
    ${raw(SWEEP.delKey === id ? campaignConfirm(g) : "")}
    ${raw(open ? campaignBody() : "")}</div>`;
}

/** The second click. Deleting is the one action here that destroys hours of GPU
 *  time, so it says exactly what goes and where it goes first. A campaign is
 *  not recoverable by re-running it cheaply - it IS the two hours. */
function campaignConfirm(g){
  const busy = SWEEP.delBusy === campaignId(g);
  return h`<div class="campconfirm">
    <p><b>Forget ${g.n_rows} row${g.n_rows === 1 ? "" : "s"}</b> measured for
      <span class="mono">${g.model}</span> on ${g.gpu || "?"} ${g.backend}${
      g.best_tok_s ? h`, best ${g.best_tok_s.toFixed(2)} tok/s` : ""}?</p>
    <p class="note">Only this campaign goes. <span class="mono">${g.file}</span> is one GPU
      and one llama.cpp build and holds every other campaign measured on that pair — those
      stay. The removed rows are written to <span class="mono">speed/deleted/</span> first,
      so this is undone by moving one file back.</p>
    <div class="actions">
      <button class="ghost danger" type="button" data-action="sweep-del-yes" data-id="${
        campaignId(g)}" ${busy ? "disabled" : ""}>${busy ? "deleting…" : "Forget it"}</button>
      <button class="ghost" type="button" data-action="sweep-del-no">Cancel</button>
    </div>
  </div>`;
}

function askDelete(id){
  SWEEP.delKey = (SWEEP.delKey === id) ? null : id;
  drawSweep({ grid: false, results: false, history: true });
}

async function doDelete(id){
  const g = (SWEEP.campaigns || []).find(x => campaignId(x) === id);
  if(!g) return;
  SWEEP.delBusy = id;
  drawSweep({ grid: false, results: false, history: true });
  let d;
  try{
    d = await (await fetch("/api/speed/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true, model: g.model, gpu: g.gpu, file: g.file,
                             pid: g.prompt_id || "", tid: g.template_id || "" })
    })).json();
  }catch(e){ d = { ok: false, error: String(e) }; }
  SWEEP.delBusy = null;
  SWEEP.delKey = null;
  SWEEP.delNote = d.ok
    ? h`Forgot ${d.removed} row${d.removed === 1 ? "" : "s"} from <span class="mono">${
        d.file}</span>${d.backup ? h` — saved to <span class="mono">${d.backup}</span>` : ""}.`
    : h`<span style="color:var(--warn)">Could not delete: ${d.error || "unknown"}</span>`;
  if(d.ok){
    // The campaign that was open may be the one just removed, and a row picked
    // out of it is no longer backed by anything on disk.
    if(SWEEP.openCampaign === id){ SWEEP.openCampaign = null; SWEEP.insights = null; }
    SWEEP.pickKey = null;
    SWEEP.script = null;
    await Promise.all([loadHistory(), loadSweepRows()]);
    // Fewer rows can change which config is recommended - including back to the
    // planner's estimate, if what just went was the only trustworthy campaign.
    REC_FOR = null;
    if(LAST) loadRecommendation(LAST);
  }
  drawSweep({ grid: false, results: true, script: true, history: true });
}

function campaignBody(){
  const ins = SWEEP.insights;
  if(!ins) return '<div class="campopen"><p class="muted small">reading rows…</p></div>';
  if(!ins.ok) return h`<div class="campopen"><p class="note" style="color:var(--warn)">${
    ins.error || "could not read this campaign"}</p></div>`;
  return '<div class="campopen">' + axisPanel(ins.axes) + depthPanel(ins.depth) +
         paretoPanel(ins.pareto) + campaignRows(ins.ranked) + '</div>';
}

/** What each knob was worth. The headline of the whole feature: a ranking says
 *  which config won, this says where the next two hours should go. */
function axisPanel(ax){
  if(!ax || !ax.effects || !ax.effects.length) return "";
  const body = ax.effects.map(e => {
    if(e.single){
      return h`<tr class="dim"><td>${e.label}</td><td colspan="3" class="muted small">only
        one value ever tried (${String(e.values[0].value)}) &mdash; nothing to compare</td></tr>`;
    }
    const vals = e.values.map(v => h`<span class="vchip ${
      String(v.value) === String(e.best) ? "win" : ""}">${String(v.value)}
      <i>${v.tok_s.toFixed(2)}</i></span>`).join("");
    return h`<tr>
      <td><b>${e.label}</b></td>
      <td class="gain ${e.gain_pct > 1 ? "up" : ""}">${e.gain_pct == null ? "—"
        : (e.gain_pct >= 0 ? "+" : "") + e.gain_pct.toFixed(1) + "%"}</td>
      <td>${raw(vals)}</td>
      <td class="muted small">vs ${String(e.reference)}${
        e.reference_is_natural ? "" : " (slowest)"} &middot; ${e.n_rows} rows${
        raw(e.n_other_groups ? " &middot; " + e.n_other_groups + " other comparison(s) not merged" : "")}
        ${raw(e.inflated ? h`<br><span style="color:var(--warn)">${e.n_corpus_repeated} of these
          needed the filler to repeat &mdash; speculation drafts from what it has already seen,
          so this is an upper bound, not a conversation.</span>` : "")}</td>
    </tr>`;
  }).join("");
  return h`<p class="sublabel">WHAT EACH KNOB WAS WORTH</p>
    <p class="note">Controlled comparisons only. Rows that differ in GPU, llama.cpp build,
      context, KV quant, fill depth or pass count are different experiments and never meet in
      one comparison &mdash; putting them together would manufacture an effect out of the
      difference between the runs.${raw(ax.n_excluded ? h` ${ax.n_excluded} row(s) are excluded
      from every number here: they spilled into shared memory, looped, or copied the prompt
      back verbatim.` : "")}</p>
    <div class="tablewrap"><table class="axtable"><tbody>${raw(body)}</tbody></table></div>`;
}

function depthPanel(depth){
  if(!depth || !depth.length) return "";
  const body = depth.map(d => {
    const c = d.config || {};
    const pts = d.points.map(p => h`<span class="vchip">${num(p.fill)}
      <i>${p.tok_s.toFixed(2)}</i></span>`).join(" → ");
    return h`<tr><td class="mono small">ngl ${c.ngl} ub ${c.ub} ${c.spec || "none"}</td>
      <td>${raw(pts)}</td>
      <td class="gain ${d.drop_pct > 20 ? "down" : ""}">&minus;${(d.drop_pct || 0).toFixed(0)}%</td>
      </tr>`;
  }).join("");
  return h`<p class="sublabel" style="margin-top:16px">SPEED VS CONTEXT DEPTH</p>
    <p class="note">The same config measured at more than one fill. Decode re-reads the KV
      cache every token, so this is the only thing here that <i>shows</i> the slope rather
      than asserting it &mdash; and it is why a headline taken at 2k says so little about the
      long conversation you will actually have.</p>
    <div class="tablewrap"><table class="axtable"><tbody>${raw(body)}</tbody></table></div>`;
}

function paretoPanel(pf){
  if(!pf || pf.length < 2) return "";
  const body = pf.map(r => { const c = r.config || {};
    return h`<tr><td><b>${(r.tok_s || 0).toFixed(2)}</b></td><td>${fmt(r.proc_vram_mib)}</td>
      <td class="mono small">ngl ${c.ngl} ncmoe ${c.ncmoe || 0} ub ${c.ub} ${
        c.spec || "none"}</td></tr>`; }).join("");
  return h`<p class="sublabel" style="margin-top:16px">SPEED VS VRAM</p>
    <p class="note">Nothing measured is both faster <i>and</i> smaller than these. &ldquo;The
      fastest&rdquo; and &ldquo;the fastest that still leaves the desktop a card to draw on&rdquo;
      are different questions, and a ranking by tok/s can only answer the first &mdash; a row 2%
      slower for 3 GiB less is often the one worth running.</p>
    <div class="tablewrap"><table class="axtable">
      <thead><tr><th>tok/s</th><th>VRAM</th><th>config</th></tr></thead>
      <tbody>${raw(body)}</tbody></table></div>`;
}

function campaignRows(ranked){
  if(!ranked || !ranked.length) return "";
  const shown = ranked.slice(0, 24);
  return h`<p class="sublabel" style="margin-top:16px">EVERY ROW &middot; fastest first &middot;
      pick one to build a script from${ranked.length > 24
        ? h` (showing 24 of ${ranked.length})` : ""}</p>
    <div class="tablewrap"><table class="rowtable">${raw(ROW_HEAD)}
      <tbody>${raw(shown.map((r, i) => sweepRow(r, i, true)).join(""))}</tbody></table></div>`;
}

function sweepHistory(){
  const cs = SWEEP.campaigns;
  const note = SWEEP.delNote
    ? h`<p class="note delnote">${raw(SWEEP.delNote)}</p>` : "";
  if(cs == null) return '<p class="muted small">reading recorded campaigns…</p>';
  if(!cs.length){
    return note + h`<p class="note">No campaigns recorded yet. Analyze a model and press
      <b>Start measuring</b> above, or run
      <span class="mono">python -m vram_planner --speed-sweep</span>.</p>`;
  }
  return note + h`<div class="camps">${raw(cs.map(campaignRow).join(""))}</div>
    <p class="note">Rows in <span class="mono">${SWEEP.speeddir || "speed/"}</span>, one file
      per GPU and llama.cpp build. They are split that way on purpose: a version bump moves
      these numbers, and merging two builds into one campaign would hide it. <b>&#10005;</b>
      forgets a campaign — its rows are moved to <span class="mono">speed/deleted/</span>
      rather than dropped, so it is undone by moving one file back.</p>`;
}

function sweepDownload(){
  if(!SWEEP.script) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([SWEEP.script.text], { type: "text/plain" }));
  a.download = SWEEP.script.filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function sweepCopy(btn){
  if(!SWEEP.script) return;
  navigator.clipboard.writeText(SWEEP.script.text).then(() => {
    const old = btn.textContent;
    btn.textContent = "copied";
    setTimeout(() => { btn.textContent = old; }, 1200);
  });
}

/* -------------------------------------------------------------- wiring */
const ACTIONS = {
  "theme":       () => toggleTheme(),
  "scan":        () => scanModels(),
  "refresh-sys": () => refreshSys(),
  "set-ctx":     el => setCtx(parseInt(el.dataset.v)),
  "measure":     () => calibrate(),
  "bench":       () => benchNow(),
  "copy-cmd":    el => copyCmd(el),
  "use-measured": el => useMeasured(parseFloat(el.dataset.toks), parseFloat(el.dataset.fill)),
  "match-runtime": el => matchRuntime(parseInt(el.dataset.ngl), parseInt(el.dataset.ctx)),
  "sweep-recheck": () => loadPreflight().then(() => drawSweep({ results: false })),
  "sweep-plan":  () => sweepPlan(),
  "sweep-start": () => sweepStart(),
  "sweep-stop":  () => sweepStop(),
  // Picking a row IS the step-3 question - "build me this one" - so it goes
  // there rather than leaving you to find the tab yourself.
  "sweep-pick":  el => { SWEEP.pickKey = el.dataset.key; SWEEP.script = null;
                         if(stepNow() !== "script"){ setStep("script"); return; }
                         drawSweep({ grid: false, script: true, history: true }); },
  "set-step":    el => setStep(el.dataset.step),
  "sweep-open":  el => openCampaign(el.dataset.id),
  "sweep-del":     el => askDelete(el.dataset.id),
  "sweep-del-yes": el => doDelete(el.dataset.id),
  "sweep-del-no":  () => { SWEEP.delKey = null;
                           drawSweep({ grid: false, results: false, history: true }); },
  "sweep-shell": el => { SWEEP.shell = el.value; SWEEP.script = null;
                         drawSweep({ grid: false, results: false, script: true }); },
  "sweep-gen":   () => sweepGen(false),
  "sweep-save":  () => sweepGen(true),
  "sweep-copy":  el => sweepCopy(el),
  "sweep-dl":    () => sweepDownload(),
  "rec-script":  () => recToScript()
};

document.addEventListener("click", ev => {
  const el = ev.target.closest("[data-action]");
  if(!el) return;
  const fn = ACTIONS[el.dataset.action];
  if(!fn) return;
  ev.preventDefault();
  fn(el);
});

$("controls").addEventListener("submit", ev => { ev.preventDefault(); run(); });
$("ctx").addEventListener("input", markCtx);
// Reveal the image-size inputs as soon as the box is ticked, before the next plan
// runs - the controls appearing only after a re-plan reads as the tick not working.
$("visionplan").addEventListener("change", () => {
  $("visioninputs").hidden = !$("visionplan").checked;
});
$("model").addEventListener("change", onPick);
// Switching the basis with a stale number in the budget box is exactly the
// divergence the control exists to remove, so it rewrites the field.
$("vrambasis").addEventListener("change", setBasis);
$("dir").addEventListener("keydown", ev => { if(ev.key === "Enter"){ ev.preventDefault(); scanModels(); } });

boot();
