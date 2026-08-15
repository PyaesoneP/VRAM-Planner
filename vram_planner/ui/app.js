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
  const busy = !!(SWEEP.status && SWEEP.status.status === "running");
  $("out").innerHTML = busy
    // A sweep is running from another tab or from before a reload. Show the
    // whole measured column so it can be watched and stopped, not just listed.
    ? renderSweep() + renderHistory()
    : h`<div class="card"><p class="placeholder">Pick a model and press Analyze fit.</p></div>` +
      renderHistory() +
      h`<section class="card" id="sweepscriptcard">
          <h2>Launch script</h2><div id="sweepscript"></div></section>`;
  drawSweep({ grid: busy, results: busy, script: true, history: true });
}

function renderPlatform(p){
  const el = $("platwarn");
  if(!el) return;
  el.innerHTML = (!p || p.supported) ? "" : h`<div class="card warns" style="margin-bottom:14px">
      <div style="color:var(--warn);font-weight:600;margin-bottom:4px">&#9888; Unvalidated platform</div>
      <div>${p.reason}</div></div>`;
}

function prefillBudgets(s){
  let vfree = 0;
  if(Array.isArray(s.gpus) && s.gpus.length) vfree = s.gpus[0].free_mib;
  if(vfree && !$("vram").value) $("vram").value = Math.round(vfree);
  // RAM budget defaults to TOTAL installed (minus a small OS reserve): a model can load
  // into standby/paged memory, so "free right now" understates what will actually load.
  if(s.ram && s.ram.total_mib && !$("ram").value)
    $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
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
  const s = await (await fetch("/api/system")).json();
  SYS = s;
  renderSys(s);
  if(Array.isArray(s.gpus) && s.gpus.length) $("vram").value = Math.round(s.gpus[0].free_mib);
  if(s.ram) $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
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
    else render(r);
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

/* ------------------------------------------------------------ result cards */
function renderVerdict(r){
  const c = r.config, p = r.plan, s = r.sizes_mib, inp = r.inputs;
  let vcls = "warn";
  if(p.fits_fully) vcls = "ok";
  else if(p.attention_overflow || p.kv_overflow || p.ram_ok === false || p.vram_ok === false) vcls = "bad";

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
    <h2>Verdict</h2>
    <div class="verdict ${vcls}"><p class="h">${p.headline}</p></div>
    ${raw(bar("VRAM", inp.vram_budget_mib, vramUsed, vramSegs))}
    ${raw(ramUsed > 0.5 ? bar("System RAM", inp.ram_budget_mib, ramUsed, ramSegs) : "")}
    <p class="note">Weights and KV cache are computed exactly from the GGUF tensor table. ${raw(cal)}${raw(split)}</p>
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
  $("out").innerHTML =
    renderVerdict(r) + renderWarnings(r) +
    renderSweep() + renderHistory() +
    tier("PREDICTED", "calculated, not measured — the cards above supersede these once you have rows") +
    renderSettings(r) + renderSpeed(r) + renderSummary(r) +
    renderKvTable(r) + renderBreakdown(r);
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
  return { path: "", model: "", preflight: null, rows: [], plan: null,
           status: null, since: 0, timer: null, script: null, pickKey: null,
           shell: (SYS && SYS.shell) || "bash", busy: "",
           campaigns: null, insights: null, openCampaign: null,
           templates: [], speeddir: "" };
}

/* Four cards, not one. They are rewritten on different clocks: the grid and the
 * results follow the 1.5s poll, the launch script must NOT - it holds a dozen
 * text inputs, and rebuilding it under the poll destroyed whatever was being
 * typed into them once every second and a half. */
function renderSweep(){
  return h`<section class="card" id="sweepcard">
    <h2>Measure real speed</h2>
    <p class="note">Everything above is calculated. Two of the settings that matter most to
      tokens/second are <b>not calculable</b>: speculative decoding has no acceptance rate
      until you run it, and prompt processing is compute bound and is not modelled here at
      all. This drives <span class="mono">llama-server</span> across a grid and records what
      it actually does &mdash; then writes the launch script for whatever wins.</p>
    <div id="sweepbody"><p class="muted small">checking the GPU&hellip;</p></div>
  </section>
  <section class="card" id="sweepresultcard">
    <h2>Measured results</h2>
    <div id="sweepresults"><p class="muted small">looking for recorded rows&hellip;</p></div>
  </section>
  <section class="card" id="sweepscriptcard">
    <h2>Launch script</h2>
    <div id="sweepscript"></div>
  </section>`;
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
  await Promise.all([loadPreflight(), loadSweepRows(), loadHistory(), loadTemplates(),
                     attachRunningJob()]);
  // Pre-select this model's campaign, so Past sweeps opens on what was just
  // analyzed instead of on whatever ran most recently.
  const mine = (SWEEP.campaigns || []).filter(g => g.model === SWEEP.model);
  if(mine.length === 1) openCampaign(campaignId(mine[0]));
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
    <p class="hint" style="margin-top:-4px">Measure where you actually work: decode slows as
      the context fills, so a number taken at 2k is not the speed you feel at 40k. A deeper
      fill costs real time though &mdash; the prompt has to be processed once per config.</p>
    <div class="field">
      <label class="check"><input type="checkbox" id="swchain" checked> Chain the stages
        &mdash; measure each knob at the split that won, not at the planner&rsquo;s guess</label>
      <p class="hint">Off, every stage runs from one fixed baseline, so ubatch and speculation
        are measured at a layer count <i>stage A has not confirmed</i>. On, each stage is built
        after the last one finishes, from the fastest trustworthy row so far. <b>Same number of
        loads, so the same hours.</b> A row that spilled into shared memory or that caught the
        model looping is never carried forward &mdash; it would bend every later stage the same
        way, silently. A new baseline also has to win by more than 2%, because tok/s is a median
        of a few passes and rebasing on jitter would make the campaign&rsquo;s path depend on noise.</p>
      <div class="field" id="swroundsfield" style="max-width:16em">
        <label for="swrounds">Rounds</label>
        <input type="number" id="swrounds" value="1" min="1" max="4" step="1">
        <p class="hint">Re-run the stages from the winner. Cheap: a config a later round
          revisits unchanged is already recorded and is skipped, so only genuinely new
          combinations cost time.</p>
      </div>
    </div>
    <div class="actions">
      <button class="ghost" type="button" data-action="sweep-plan">Preview grid</button>
      <button class="go" type="button" style="width:auto" data-action="sweep-start">&#9654; Start measuring</button>
    </div>
    <div id="sweepplan"></div>
  </div>`;
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
  return { path: SWEEP.path, stages: sweepStages(), context: parseInt($("ctx").value),
           kv_type: $("kv").value, fill: v("swfill"), n_predict: v("swpred"),
           repeat: v("swrep"), limit: v("swlimit"),
           chain: chain, rounds: chain ? (v("swrounds") || 1) : 1 };
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
  return h`<div class="running">
    <p><b>Measuring</b> &mdash; ${st.done} of ${st.total || "?"} configs, ${mins} min elapsed
      ${raw(st.cancelling ? '<span style="color:var(--warn)">&middot; stopping</span>' : "")}</p>
    <div class="prog"><i style="width:${pct}%"></i></div>
    <div class="actions">
      <button class="ghost" type="button" data-action="sweep-stop" ${
        st.cancelling ? "disabled" : ""}>&#9632; Stop</button>
      <span class="muted small">Stopping finishes the config in flight first, so its row is
        complete rather than half-written. Nothing is lost either way &mdash; every row is
        already on disk and keyed, so starting again resumes from here.</span>
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
    "system memory instead of failing. The row says ok and the speed is off a cliff."]);
  if(r.corpus_repeated && r.config && r.config.spec && r.config.spec !== "none")
    f.push(["filler repeats", "The prompt had to repeat to reach this depth, which inflates " +
      "any speculative acceptance rate. Compare only against other rows at this depth."]);
  if(r.distinct_ratio != null && r.distinct_ratio < 0.5)
    f.push(["looping", "Only " + Math.round(100 * r.distinct_ratio) + "% of 8-word windows " +
      "were distinct - the model was repeating itself, so this speed is not real work."]);
  return f;
}

const ROW_HEAD = h`<thead><tr><th></th><th>tok/s</th><th>prefill</th><th>ngl</th><th>ncmoe</th>
  <th>ub</th><th>fill</th><th>spec</th><th>proj</th><th>accepted</th><th>VRAM</th><th></th>
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
  if(pickable) PICKABLE[key] = r;
  const on = SWEEP.pickKey == null ? (pickable && i === 0) : SWEEP.pickKey === key;
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
const SCRIPT_FIELDS = ["swport", "swhost", "swload", "sm_tmplfile", "sm_tmplkw"]
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
  return h`<div style="margin-top:20px">
    <p class="sublabel">LAUNCH SCRIPT</p>
    <p class="note">Built from ${raw(src)}. It resolves the llama.cpp backend fresh at every
      launch, puts the vendor CUDA libraries on the path (without them the process dies with
      <span class="mono">STATUS_DLL_NOT_FOUND</span> and no message), writes a dated log file,
      and uses <span class="mono">--load-mode</span> and
      <span class="mono">--spec-draft-n-max</span> rather than the deprecated and removed
      spellings that are accepted and then silently ignored.</p>
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
          the whole mapped file, including the blocks already resident in VRAM.</p></div>
    </div>
    <details class="adv" ${raw(fopen(0))}><summary>Chat template (optional)</summary>
      <p class="hint">Overrides the template baked into the GGUF. Useful when the conversion
        predates a fixed template, or when the model card ships a patched one for tool calls.
        A path that does not exist is <b>not</b> an error to llama-server &mdash; it falls back
        to the built-in template without saying so &mdash; so the generated script checks it and
        refuses to start. It also passes <span class="mono">--jinja</span> <i>before</i> the
        template flags, because without it a build accepts only its built-in template
        <i>names</i> and rejects a path outright.</p>
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
        <p class="hint">The key names belong to <b>the template</b>, not to llama.cpp:
          <span class="mono">enable_thinking</span> is Qwen3&rsquo;s spelling and is
          <i>ignored, not rejected</i>, by a model that does not use that variable &mdash; so a
          typo here is silent. Checked for valid JSON before the script is written. These are
          <b>server defaults</b>: a client that sends its own
          <span class="mono">chat_template_kwargs</span> wins for that request.</p>
      </div>
    </details>
    <details class="adv" ${raw(fopen(1))}><summary>Sampling (optional)</summary>
      <p class="hint">Left blank, no sampler flags are written at all &mdash; a made-up default
        is worse than none. llama.cpp&rsquo;s own defaults are temp 0.80, top-k 40, min-p 0.05,
        which several model cards do <i>not</i> want, so fill these in from yours. These become
        <b>server defaults</b>: any client that sends its own values overrides them per request.</p>
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

function campaignId(g){ return g.model + " " + g.gpu + " " + g.file; }

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
            "&file=" + encodeURIComponent(g.file);
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
    <button class="camprow" type="button" data-action="sweep-open" data-id="${id}">
      <span class="mono">${g.model}</span>
      <span class="muted small">${g.gpu || "?"} &middot; ${g.backend} &middot; ${span}</span>
      <span class="campnum">${g.n_ok}/${g.n_rows} ok${raw(
        g.n_untrusted ? " &middot; " + g.n_untrusted + " flagged" : "")}</span>
      <span class="campbest">${g.best_tok_s ? g.best_tok_s.toFixed(2) + " tok/s" : "—"}</span>
      <span class="campcaret">${open ? "▾" : "▸"}</span>
    </button>
    ${raw(open ? campaignBody() : "")}</div>`;
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
      from every number here: they spilled into shared memory, or caught the model looping.` : "")}</p>
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
  if(cs == null) return '<p class="muted small">reading recorded campaigns…</p>';
  if(!cs.length){
    return h`<p class="note">No campaigns recorded yet. Analyze a model and press
      <b>Start measuring</b> above, or run
      <span class="mono">python -m vram_planner --speed-sweep</span>.</p>`;
  }
  return h`<div class="camps">${raw(cs.map(campaignRow).join(""))}</div>
    <p class="note">Rows in <span class="mono">${SWEEP.speeddir || "speed/"}</span>, one file
      per GPU and llama.cpp build. They are split that way on purpose: a version bump moves
      these numbers, and merging two builds into one campaign would hide it.</p>`;
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
  "sweep-pick":  el => { SWEEP.pickKey = el.dataset.key; SWEEP.script = null;
                         drawSweep({ grid: false, script: true, history: true }); },
  "sweep-open":  el => openCampaign(el.dataset.id),
  "sweep-shell": el => { SWEEP.shell = el.value; SWEEP.script = null;
                         drawSweep({ grid: false, results: false, script: true }); },
  "sweep-gen":   () => sweepGen(false),
  "sweep-save":  () => sweepGen(true),
  "sweep-copy":  el => sweepCopy(el),
  "sweep-dl":    () => sweepDownload()
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
$("dir").addEventListener("keydown", ev => { if(ev.key === "Enter"){ ev.preventDefault(); scanModels(); } });

boot();
