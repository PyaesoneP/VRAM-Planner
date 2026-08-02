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

const CALIB_TERMS = ["const", "ctx", "act", "nofa"];

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
      MODELS.map((m, i) => h`<option value="${i}">${m.name}  (${fmtG(m.size_mix || m.size_mib)})${
        m.n_ctx_train ? "  · " + ctxLabel(m.n_ctx_train) + " ctx" : ""}</option>`).join("");
    $("scanhint").textContent = MODELS.length
      ? MODELS.length + " GGUF found in folder" : "no .gguf found here";
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
    ram_free_mib: (SYS && SYS.ram) ? SYS.ram.free_mib : null
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
        r.calibration.free.join(", ")} fitted, in-sample ${r.calibration.residual_pct}%). `
    : "The compute buffer uses shipped defaults, fitted to measured llama.cpp runs across 4 " +
      "architectures (4.8% mean, 12% worst on one CUDA card). It is the only term that depends " +
      "on your hardware rather than the model &mdash; press <b>Measure running model</b> with a " +
      "model loaded to calibrate it for yours. ";

  const split = p.cpu_compute_mib > 0.5
    ? h`Split across backends here: ${fmt(p.compute_mib)} on the GPU, ${fmt(p.cpu_compute_mib)
        } in RAM. llama.cpp gives every backend running part of the graph its own scratch pool, and the ${
        fmt(s.compute_logits)} logits tensor (${inp.n_ubatch} × ${num(c.n_vocab || 0)
        } vocab) belongs to whichever side holds the output head.`
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
    <h2>Settings to use</h2>
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
      ${raw(brow("&nbsp;&nbsp;logits (" + inp.n_ubatch + " x " + num(c.n_vocab || 0) + " vocab)",
        fmt(s.compute_logits) + '  <span class="muted">' +
        ((p.cpu_compute_mib > 0.5) ? "on CPU &mdash; the output head is not offloaded" : "on GPU") + "</span>"))}
      ${raw(r.mmproj ? brow("Vision projector (" + esc(r.mmproj.name) + ")",
        fmt(r.mmproj.mib) + (r.mmproj.included ? "" : '  <span class="muted">not loaded</span>')) : "")}
      ${raw(brow("File on disk", fmtG(s.file_on_disk) + "  (" + fmtGB(s.file_on_disk) + ")"))}
      ${raw(r.mmproj ? brow("&nbsp;&nbsp;+ projector = LM Studio's &quot;model size&quot;",
        fmtG(s.bundle_on_disk) + "  (" + fmtGB(s.bundle_on_disk) + ")") : "")}
    </tbody></table></div></section>`;
}

/** Results are ordered by what the user came for: the answer, anything alarming
 *  about it, the settings that produce it, then the evidence behind it. */
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
  $("out").innerHTML = renderVerdict(r) + renderWarnings(r) + renderSettings(r) +
                       renderSpeed(r) + renderSummary(r) + renderKvTable(r) + renderBreakdown(r);
  if(r.speed && !r.speed.error) loadSpeedHistory(r);
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
      (st.skipped_rows ? h`<br><span class="muted">${st.skipped_rows} stored measurement${
        st.skipped_rows == 1 ? " was" : "s were"} left out: the reading did not respond to the
        config, or the layer count was never recorded. Re-measure with VRAM to spare to bring
        them back.</span>` : "")
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
  "match-runtime": el => matchRuntime(parseInt(el.dataset.ngl), parseInt(el.dataset.ctx))
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
$("model").addEventListener("change", onPick);
$("dir").addEventListener("keydown", ev => { if(ev.key === "Enter"){ ev.preventDefault(); scanModels(); } });

boot();
