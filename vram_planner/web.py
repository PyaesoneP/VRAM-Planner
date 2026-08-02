"""The local web UI and its JSON endpoints."""
import json, os, webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs
from urllib.parse import urlparse
from .const import __version__
from .gguf import load_gguf
from .model import extract_config
from .paths import _data_dir
from .gpu import get_bandwidth, get_gpu_processes, get_gpus, get_ram, platform_support
from .lmstudio import benchmark_server, default_models_dir, load_benchmarks, match_speed_history, read_lmstudio_runtime, resolve_runtime_ngl, save_benchmark, scan_models, scan_server_logs, scan_speed_history
from .calib import _active_gpu, calibration_status, record_calibration, refresh_calibration
from .plan import analyze


# ---------------------------------------------------------------------------
# Embedded web UI
# ---------------------------------------------------------------------------
HTML_PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRAM Planner</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121a23; --panel2:#0e141b; --line:#25313d;
  --ink:#e7eef5; --muted:#8ba0b3; --dim:#5b6b7b;
  --ok:#37d5b0; --warn:#f2b64c; --bad:#ff6a6a;
  --wt:#5aa9ff; --kv:#37d5b0; --cmp:#b98bff; --rsv:#3a4756;
  --mono:ui-monospace,"Cascadia Code","JetBrains Mono",Consolas,"DejaVu Sans Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:
   radial-gradient(1200px 600px at 80% -10%, #14202c 0%, transparent 60%),
   var(--bg); color:var(--ink); font-family:var(--sans); font-size:14px; line-height:1.45}
a{color:var(--wt)}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:14px 22px;justify-content:space-between;
   padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:20px}
.brand h1{font-family:var(--mono);font-weight:600;font-size:20px;letter-spacing:.14em;margin:0}
.brand .tag{color:var(--muted);font-size:12px;letter-spacing:.05em;margin-top:3px}
.brand .accent{color:var(--ok)}
.sys{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
.sys .g{min-width:190px}
.sys .lbl{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.08em;
   display:flex;justify-content:space-between;gap:10px}
.sys .lbl b{color:var(--ink);font-weight:600}
.meter{height:7px;border-radius:4px;background:#0c1219;border:1px solid var(--line);
   margin-top:5px;overflow:hidden}
.meter > i{display:block;height:100%;background:linear-gradient(90deg,var(--wt),var(--kv))}
button{font-family:var(--sans);cursor:pointer;border:1px solid var(--line);background:var(--panel2);
   color:var(--ink);border-radius:7px;padding:7px 12px;font-size:13px}
button:hover{border-color:#39485a}
button.ghost{background:transparent;color:var(--muted);padding:5px 9px;font-size:12px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:18px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px}
.card h2{font-family:var(--mono);font-size:12px;letter-spacing:.1em;color:var(--muted);
   text-transform:uppercase;margin:0 0 12px;font-weight:600}
.field{margin-bottom:13px}
.field label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
.field .hint{color:var(--dim);font-size:11px;margin-top:4px}
input[type=text],input[type=number],select{width:100%;background:var(--panel2);border:1px solid var(--line);
   color:var(--ink);border-radius:7px;padding:8px 10px;font-family:var(--mono);font-size:13px}
input:focus,select:focus{outline:none;border-color:var(--wt)}
.row{display:flex;gap:8px}
.row > *{flex:1}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
.chip{font-family:var(--mono);font-size:12px;padding:4px 9px;border:1px solid var(--line);
   border-radius:20px;background:var(--panel2);color:var(--muted);cursor:pointer}
.chip:hover{color:var(--ink)}
.chip.on{border-color:var(--kv);color:var(--kv)}
.check{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink);cursor:pointer}
.check input{width:16px;height:16px;accent-color:var(--kv)}
details.adv{margin-top:6px;border-top:1px dashed var(--line);padding-top:12px}
details.adv summary{cursor:pointer;color:var(--muted);font-size:12px;font-family:var(--mono);
   letter-spacing:.06em;list-style:none;margin-bottom:10px}
details.adv summary::-webkit-details-marker{display:none}
.go{width:100%;background:linear-gradient(90deg,#1c3a4f,#1a4a44);border-color:#2b5a52;
   color:#eafff9;font-weight:600;padding:11px;font-size:14px;letter-spacing:.03em;margin-top:6px}
.go:hover{filter:brightness(1.12)}
.results{display:flex;flex-direction:column;gap:16px;min-width:0}
.placeholder{color:var(--dim);font-size:13px;text-align:center;padding:40px 10px;
   font-family:var(--mono);letter-spacing:.05em}
.badge{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.06em;
   padding:2px 8px;border-radius:5px;border:1px solid var(--line)}
.badge.moe{color:var(--cmp);border-color:#4a3a63}
.badge.dense{color:var(--wt);border-color:#28405a}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px 16px}
.kv{font-family:var(--mono)}
.kv .k{color:var(--muted);font-size:11px;letter-spacing:.04em}
.kv .v{color:var(--ink);font-size:15px;margin-top:2px}
.verdict{border-left:3px solid var(--rsv);padding-left:12px}
.verdict.ok{border-color:var(--ok)} .verdict.warn{border-color:var(--warn)} .verdict.bad{border-color:var(--bad)}
.verdict .h{font-size:15px;font-weight:600}
.verdict.ok .h{color:var(--ok)} .verdict.warn .h{color:var(--warn)} .verdict.bad .h{color:var(--bad)}
.barwrap{margin-top:14px}
.bar-top{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;
   color:var(--muted);margin-bottom:5px}
.bar-top b{color:var(--ink)}
.bar{position:relative;height:26px;border-radius:6px;background:#0a1017;border:1px solid var(--line);
   overflow:hidden;display:flex}
.bar > span{height:100%;transition:width .5s cubic-bezier(.2,.7,.2,1)}
.bar .s-wt{background:var(--wt)} .bar .s-kv{background:var(--kv)}
.bar .s-cmp{background:var(--cmp)} .bar .s-rsv{background:repeating-linear-gradient(45deg,#2c3742,#2c3742 5px,#232d38 5px,#232d38 10px)}
.cap{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.85}
.cap:after{content:attr(data-l);position:absolute;top:-15px;left:50%;transform:translateX(-50%);
   font-family:var(--mono);font-size:9px;color:var(--muted);white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:9px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.legend .n{color:var(--ink)}
.setrow{font-family:var(--mono);font-size:13px;padding:5px 0;border-bottom:1px dashed var(--line);color:var(--ink)}
.setrow:last-child{border-bottom:none}
pre.cmd{background:#080c11;border:1px solid var(--line);border-radius:8px;padding:12px;
   font-family:var(--mono);font-size:12.5px;color:#d6e6f2;overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:0}
.cmdhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.04em}
td:first-child,th:first-child{text-align:left}
.warns{color:var(--warn);font-size:12.5px}
.warns div{padding:4px 0}
.muted{color:var(--muted)} .small{font-size:12px}
.note{color:var(--dim);font-size:11px;margin-top:8px;line-height:1.5}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--dim);border-top-color:var(--ok);
   border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <h1>VRAM<span class="accent">//</span>PLANNER</h1>
      <div class="tag">exact GGUF weights + KV cache &middot; live VRAM/RAM &middot; layer &amp; expert offload
        <span class="muted" id="ver"></span></div>
    </div>
    <div class="sys" id="sys"><span class="muted small">reading hardware...</span></div>
  </header>
  <div id="platwarn"></div>

  <div class="grid">
    <div class="card">
      <h2>Model &amp; run settings</h2>
      <div class="field">
        <label>Models folder</label>
        <div class="row">
          <input type="text" id="dir" placeholder="path to .gguf folder">
          <button class="ghost" style="flex:0 0 auto" onclick="scanModels()">Scan</button>
        </div>
        <div class="hint" id="scanhint"></div>
      </div>
      <div class="field">
        <label>Model</label>
        <select id="model" onchange="onPick()"><option value="">-- scan or paste a path --</option></select>
      </div>
      <div class="field">
        <label>...or full path to a .gguf</label>
        <input type="text" id="path" placeholder="C:\\...\\model.gguf">
      </div>
      <div class="field">
        <label>Context length (tokens)</label>
        <input type="number" id="ctx" value="8192" min="256" step="256">
        <div class="chips" id="ctxchips"></div>
      </div>
      <div class="row">
        <div class="field">
          <label>KV cache quant</label>
          <select id="kv">
            <option value="f16">f16 (2.00 B)</option>
            <option value="q8_0">q8_0 (1.06 B)</option>
            <option value="q5_1">q5_1 (0.75 B)</option>
            <option value="q4_0">q4_0 (0.56 B)</option>
          </select>
        </div>
        <div class="field">
          <label>ubatch = LM Studio "Physical Batch Size"</label>
          <input type="number" id="ubatch" value="512" min="16" step="16">
        </div>
        <div class="field">
          <label>seqs = "Max Concurrent Predictions"</label>
          <input type="number" id="nseq" value="1" min="1" step="1">
        </div>
      </div>
      <div class="hint">Match seqs to LM Studio's "Max Concurrent Predictions" (default 4). It
        sizes the recurrent state on hybrid models and the sliding-window cache on SWA models
        (Gemma, Mistral, gpt-oss) &mdash; llama.cpp allocates one window per sequence slot.</div>
      <div class="field">
        <label class="check"><input type="checkbox" id="fa"> Flash attention (needed for quant KV)</label>
      </div>
      <div class="field" id="mmprojfield" style="display:none">
        <label class="check"><input type="checkbox" id="mmproj" checked> Load vision projector (mmproj)</label>
        <div id="mtprow" style="display:none">
        <label class="check"><input type="checkbox" id="mtpspec" checked> MTP speculative decoding</label>
        <div class="hint">This model ships multi-token-prediction blocks. LM Studio turns this on by
          default for them (<span class="mono">Speculative Decoding: MTP</span>, llama.cpp
          <span class="mono">--spec-type draft-mtp</span>). It gives the draft blocks a KV cache back
          &mdash; at f16 regardless of your KV quant &mdash; plus a per-sequence slot. Untick only if
          you set Speculative Decoding to Off.</div></div>
        <div class="hint" id="mmprojhint"></div>
      </div>
      <div class="field">
        <label class="check"><input type="checkbox" id="kvgpu"> Force ALL layers' KV onto GPU (offload FFN)</label>
        <div class="hint">Advanced, llama.cpp <b>-ot</b> only. Different from LM Studio's "Offload KV
          Cache to GPU" (which only keeps <i>GPU-layer</i> KV in VRAM). This pushes FFN weights to CPU so
          every layer's KV sits in VRAM; dense generation gets slower. Leave OFF to model a normal
          LM Studio GPU-offload split.</div>
      </div>
      <details class="adv">
        <summary>Advanced budgets &amp; overrides</summary>
        <div class="field">
          <label>GPU layers (blank = auto; set to match/verify an LM Studio run)</label>
          <input type="number" id="ngl" step="1" min="0" placeholder="auto">
          <div class="hint">Enter LM Studio's "GPU Offload" number to see the exact VRAM/RAM that
            config uses, then compare against Task Manager. <b>Check the real value in the log</b>
            (<span class="mono">%APPDATA%\LM Studio\logs\main.log</span>, "Resolved GPU config
            options") &mdash; LM Studio silently adjusts the slider to respect its VRAM cap.</div>
        </div>
        <div class="field" id="ncpumoefield" style="display:none">
          <label>CPU expert layers (MoE; blank = auto)</label>
          <input type="number" id="ncpumoe" step="1" min="0" placeholder="auto">
          <div class="hint">LM Studio's "Number of layers to keep experts on CPU" /
            llama.cpp <b>--n-cpu-moe</b>. Different knob from GPU Offload: it moves only the
            routed experts of the first N blocks, leaving their attention and KV on the GPU.</div>
        </div>
        <div class="row">
          <div class="field"><label>VRAM budget (MiB)</label><input type="number" id="vram" step="64"></div>
          <div class="field"><label>RAM budget (MiB)</label><input type="number" id="ram" step="256"></div>
        </div>
        <div class="hint" style="margin:-6px 0 12px">RAM budget defaults to total installed, since a
          model can load into standby/paged memory (LM Studio pushes RAM to ~100%). The header shows
          what's free right now; the plan warns if it exceeds that.</div>
        <div class="row">
          <div class="field"><label>GPU reserve (MiB)</label><input type="number" id="reserve" value="0" step="64">
            <div class="hint">Extra headroom on top of the plan. Defaults to 0 because the CUDA
              context and driver overhead are already inside the compute-buffer estimate (they were
              fitted together against measured loads). Add to this only if you want the plan to
              leave room for something else on the GPU, e.g. your desktop.</div></div>
          <div class="field"><label>Safety %</label><input type="number" id="safety" value="5" step="1" min="0" max="40"></div>
        </div>
        <div class="row">
          <div class="field"><label>VRAM bandwidth (GB/s)</label>
            <input type="number" id="bwv" step="1" placeholder="auto"></div>
          <div class="field"><label>RAM bandwidth (GB/s)</label>
            <input type="number" id="bwr" step="0.1" placeholder="auto"></div>
        </div>
        <div class="hint" id="bwhint" style="margin:-6px 0 12px"></div>
        <div class="row">
          <div class="field"><label>Context filled (tokens)</label>
            <input type="number" id="ctxfill" step="256" placeholder="8192">
            <div class="hint">KV is re-read every token, so speed depends on how full the
              context actually is &mdash; not on the size you allocated.</div></div>
          <div class="field"><label>RAM efficiency (0 = bracket)</label>
            <input type="number" id="rameff" step="0.01" min="0" max="1" placeholder="0">
            <div class="hint">Set by Calibrate. Fraction of peak RAM bandwidth actually reached.</div></div>
        </div>
        <div class="field">
          <label>Compute buffer override (MiB, 0 = estimate)</label>
          <input type="number" id="compute" value="0" step="16">
          <div class="hint">Weights &amp; KV are exact. The compute buffer is estimated. You do not
            need the load log &mdash; if the model is loaded right now, press Measure and the tool reads
            the engine process's real VRAM and back-solves the true overhead.</div>
          <button class="ghost" style="margin-top:8px" onclick="calibrate()">&#9673; Measure running model</button>
          <div class="hint" id="calhint"></div>
        </div>
      </details>
      <button class="go" id="goBtn" onclick="run()">Analyze fit</button>
    </div>

    <div class="results" id="out">
      <div class="card"><div class="placeholder">Pick a model and press Analyze fit.</div></div>
    </div>
  </div>
</div>
<script src="/app.js"></script>
</body>
</html>'''


APP_JS = '''
const $ = id => document.getElementById(id);
let SYS = null, MODELS = [], LAST = null;

function fmt(m){ if(m==null||isNaN(m)) return "-"; return Math.round(m).toLocaleString()+" MiB"; }
function fmtG(m){ if(m==null||isNaN(m)) return "-"; return (m/1024).toFixed(2)+" GiB"; }
function fmtGB(m){ if(m==null||isNaN(m)) return "-"; return (m*1048576/1e9).toFixed(2)+" GB"; }
function B(n){ return (n/1e9).toFixed(2)+" B"; }
function clampPct(x){ return Math.max(0, Math.min(100, x)); }

function renderPlatform(p){
  const el = $("platwarn");
  if(!el) return;
  if(!p || p.supported){ el.innerHTML = ""; return; }
  el.innerHTML = '<div class="card warns" style="margin-bottom:14px">'+
    '<div style="color:var(--warn);font-weight:600;margin-bottom:4px">'+
    '&#9888; Unvalidated platform</div><div>'+escapeHtml(p.reason)+'</div></div>';
}

async function boot(){
  try{
    const s = await (await fetch("/api/system")).json();
    SYS = s;
    if(!$("dir").value) $("dir").value = s.default_dir || "";
    if(s.version) $("ver").textContent = " · v"+s.version;
    renderPlatform(s.platform);
    renderSys(s);
    prefillBudgets(s);
    if($("dir").value) scanModels();
  }catch(e){ $("sys").innerHTML = '<span class="muted small">system read failed: '+e+'</span>'; }
  buildCtxChips();
  loadBandwidth();
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

function renderSys(s){
  let h = "";
  if(Array.isArray(s.gpus) && s.gpus.length){
    s.gpus.forEach(g=>{
      const usedPct = clampPct(g.used_mib/g.total_mib*100);
      h += '<div class="g"><div class="lbl"><span>'+g.name+'</span>'+
           '<span><b>'+Math.round(g.free_mib).toLocaleString()+'</b> / '+Math.round(g.total_mib).toLocaleString()+' MiB free</span></div>'+
           '<div class="meter"><i style="width:'+usedPct+'%"></i></div></div>';
    });
  } else {
    h += '<div class="g"><div class="lbl"><span>GPU</span><span class="muted">nvidia-smi not found</span></div>'+
         '<div class="meter"></div></div>';
  }
  if(s.ram){
    const usedPct = clampPct((s.ram.total_mib-s.ram.free_mib)/s.ram.total_mib*100);
    h += '<div class="g"><div class="lbl"><span>System RAM</span>'+
         '<span><b>'+Math.round(s.ram.free_mib).toLocaleString()+'</b> / '+Math.round(s.ram.total_mib).toLocaleString()+' MiB free</span></div>'+
         '<div class="meter"><i style="width:'+usedPct+'%"></i></div></div>';
  }
  h += '<button class="ghost" onclick="refreshSys()">&#8635; refresh</button>';
  $("sys").innerHTML = h;
}

async function loadBandwidth(){
  try{
    const b = await (await fetch("/api/bandwidth")).json();
    if(b.vram_gbs && !$("bwv").value) $("bwv").value = b.vram_gbs;
    if(b.ram_gbs  && !$("bwr").value) $("bwr").value = b.ram_gbs;
    $("bwhint").innerHTML = (b.notes||[]).join("<br>");
  }catch(e){ $("bwhint").textContent = "bandwidth auto-detect failed: "+e; }
}

async function refreshSys(){
  const s = await (await fetch("/api/system")).json(); SYS = s;
  renderSys(s);
  if(Array.isArray(s.gpus)&&s.gpus.length) $("vram").value = Math.round(s.gpus[0].free_mib);
  if(s.ram) $("ram").value = Math.max(1024, Math.round(s.ram.total_mib - 2048));
}

function buildCtxChips(){
  const vals=[2048,4096,8192,16384,32768,65536,131072];
  $("ctxchips").innerHTML = vals.map(v=>'<span class="chip" data-v="'+v+'" onclick="setCtx('+v+')">'+
     (v>=1024?(v/1024)+"k":v)+'</span>').join("");
  markCtx();
}
function setCtx(v){ $("ctx").value=v; markCtx(); }
function markCtx(){
  const cur=parseInt($("ctx").value)||0;
  document.querySelectorAll("#ctxchips .chip").forEach(c=>{
    c.classList.toggle("on", parseInt(c.dataset.v)===cur);
  });
}
$("ctx") && $("ctx").addEventListener("input", markCtx);

async function scanModels(){
  const dir=$("dir").value.trim();
  $("scanhint").textContent="scanning...";
  try{
    const r = await (await fetch("/api/models?dir="+encodeURIComponent(dir))).json();
    MODELS = r.models||[];
    const sel=$("model");
    sel.innerHTML = '<option value="">-- '+MODELS.length+' models found --</option>'+
      MODELS.map((m,i)=>'<option value="'+i+'">'+
        m.name+'  ('+fmtG(m.size_mix||m.size_mib)+')'+
        (m.n_ctx_train?('  &middot; '+(m.n_ctx_train>=1024?(m.n_ctx_train/1024)+'k':m.n_ctx_train)+' ctx'):'')+
        '</option>').join("");
    $("scanhint").textContent = MODELS.length? (MODELS.length+" GGUF found in folder")
                                             : "no .gguf found here";
  }catch(e){ $("scanhint").textContent="scan failed: "+e; }
}
function onPick(){
  const i=$("model").value;
  if(i==="") return;
  const m=MODELS[parseInt(i)];
  if(!m) return;
  $("path").value="";
  if(m.n_ctx_train){ setCtxMax(m.n_ctx_train); }
}
let CTX_MAX = null;
function setCtxMax(nctx){
  CTX_MAX = nctx;
  $("ctx").max = nctx;
  // add/refresh a "max" chip
  let chip=document.getElementById("ctxmax");
  const label = (nctx>=1024?(nctx/1024)+"k":nctx);
  if(!chip){
    chip=document.createElement("span");
    chip.id="ctxmax"; chip.className="chip";
    $("ctxchips").appendChild(chip);
  }
  chip.textContent="max "+label;
  chip.setAttribute("data-v", nctx);
  chip.onclick=()=>setCtx(nctx);
  markCtx();
}

async function run(){
  let path = $("path").value.trim();
  if(!path){
    const i=$("model").value;
    if(i!=="" && MODELS[parseInt(i)]) path = MODELS[parseInt(i)].path;
  }
  if(!path){ alert("Pick a model from the list or paste a .gguf path."); return; }
  const body = {
    path: path,
    context: parseInt($("ctx").value)||8192,
    kv_type: $("kv").value,
    n_ubatch: parseInt($("ubatch").value)||512,
    n_seq: parseInt($("nseq").value)||1,
    include_mmproj: $("mmproj").checked,
    mtp_spec: $("mtpspec").checked,
    n_cpu_moe_override: $("ncpumoe").value===""? null : parseInt($("ncpumoe").value),
    bw_vram_gbs: parseFloat($("bwv").value)||0,
    bw_ram_gbs: parseFloat($("bwr").value)||0,
    ram_eff: parseFloat($("rameff").value)||0,
    ctx_fill: $("ctxfill").value===""? null : parseInt($("ctxfill").value),
    flash_attn: $("fa").checked,
    vram_budget_mib: parseFloat($("vram").value)||0,
    ram_budget_mib: parseFloat($("ram").value)||0,
    gpu_reserve_mib: parseFloat($("reserve").value)||0,
    compute_override_mib: parseFloat($("compute").value)||0,
    safety_pct: parseFloat($("safety").value)||0,
    kv_on_gpu: $("kvgpu").checked,
    gpu_layers_override: ($("ngl").value.trim()===""? null : parseInt($("ngl").value)),
    ram_free_mib: (SYS && SYS.ram ? SYS.ram.free_mib : null),
  };
  const btn=$("goBtn"); btn.disabled=true; const old=btn.textContent;
  btn.innerHTML='<span class="spin"></span>analyzing';
  $("out").innerHTML='<div class="card"><div class="placeholder"><span class="spin"></span>parsing GGUF...</div></div>';
  try{
    const res = await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const r = await res.json();
    if(!r.ok){ $("out").innerHTML='<div class="card warns"><b>Error:</b> '+(r.error||"unknown")+'</div>'; }
    else render(r);
  }catch(e){ $("out").innerHTML='<div class="card warns"><b>Request failed:</b> '+e+'</div>'; }
  btn.disabled=false; btn.textContent=old;
}

function kvBadge(c){ return c.is_moe
  ? '<span class="badge moe">MoE &middot; '+c.n_expert_used+'/'+c.n_expert+' experts</span>'
  : '<span class="badge dense">DENSE</span>'; }

function bar(title, capMib, used, segs){
  const sum = segs.reduce((a,s)=>a+(s.mib||0),0);
  const scale = Math.max(capMib, sum, 1);
  let inner = segs.map(s=>'<span class="'+s.cls+'" style="width:'+clampPct(s.mib/scale*100)+'%" title="'+s.label+': '+fmt(s.mib)+'"></span>').join("");
  const capPos = clampPct(capMib/scale*100);
  const over = sum>capMib;
  const utilTxt = capMib? (used/capMib*100).toFixed(0)+'%' : '-';
  let legend = segs.filter(s=>s.mib>0.5).map(s=>'<span><i style="background:'+s.color+'"></i>'+
      s.label+' <span class="n">'+fmt(s.mib)+'</span></span>').join("");
  return '<div class="barwrap"><div class="bar-top"><span>'+title+' &middot; using <b>'+fmt(used)+'</b> of '+fmt(capMib)+
     '</span><span style="color:'+(over?'var(--bad)':'var(--muted)')+'">'+utilTxt+(over?' OVER':'')+'</span></div>'+
     '<div class="bar">'+inner+'<span class="cap" data-l="'+fmt(capMib)+'" style="left:'+capPos+'%"></span></div>'+
     '<div class="legend">'+legend+'</div></div>';
}

function render(r){
  LAST = r;
  const c=r.config, p=r.plan, s=r.sizes_mib, inp=r.inputs;
  $("mmprojfield").style.display = r.mmproj? "" : "none";
  $("ncpumoefield").style.display = r.is_moe? "" : "none";
  { const row=$("mtprow"); if(row) row.style.display = c.n_mtp_layers ? "" : "none"; }
  if(r.mmproj) $("mmprojhint").innerHTML = r.mmproj.name+" &middot; "+fmt(r.mmproj.mib)+
    " of VRAM. LM Studio loads it with the model and includes it in the size it shows.";
  // verdict class
  let vcls="warn";
  if(p.fits_fully) vcls="ok";
  else if(p.attention_overflow || p.kv_overflow || p.ram_ok===false || p.vram_ok===false) vcls="bad";

  // ---- summary ----
  let sum = '<div class="card"><h2>Model</h2>'+
    '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px">'+
    '<div style="font-family:var(--mono);font-size:15px">'+ (r.model_name||"?") +'</div>'+ kvBadge(r) +'</div>'+
    '<div class="summary-grid">'+
      kvItem("architecture", c.arch)+
      kvItem("params (total)", B(r.params_total))+
      (r.is_moe? kvItem("params (active)", B(r.active_params)) : "")+
      kvItem("quant", (r.bpw).toFixed(2)+" bpw")+
      kvItem("layers", c.n_layers)+
      kvItem("hidden size", c.hidden)+
      kvItem("attn heads", c.n_head + " / kv " + c.n_head_kv)+
      (r.hybrid && r.hybrid.is_hybrid
        ? kvItem("KV-bearing layers", r.hybrid.n_attn_layers + " of " + c.n_layers +
            " (hybrid; " + r.hybrid.n_ssm_layers + " linear/SSM)")
        : "")+
      (r.swa && r.swa.enabled
        ? kvItem("sliding window", r.swa.n_swa.toLocaleString() + " tok &middot; " +
            r.swa.n_swa_layers + " windowed / " + r.swa.n_global_layers + " global")
        : "")+
      (c.n_mtp_layers
        ? kvItem("multi-token pred.", c.n_mtp_layers + " block" + (c.n_mtp_layers==1?"":"s") +
            ' <span class="muted">(' + (inp.mtp_spec ? "drafting: KV counted" : "idle: weights only") + ')</span>')
        : "")+
      kvItem("head dim", (c.head_dim_k||"-") +
        ((r.swa && r.swa.enabled && r.swa.head_dim!==r.swa.head_dim_global)
          ? '  <span class="muted">/ '+r.swa.head_dim+' swa</span>' : ""))+
      kvItem("native ctx", (c.n_ctx_train? c.n_ctx_train.toLocaleString():"-"))+
      kvItem("weights", fmtG(s.weights))+
    '</div></div>';

  // ---- verdict + bars ----
  const vramSegs=[
    {cls:"s-wt",  color:"var(--wt)",  label:"weights (GPU)", mib:p.gpu_weights_mib||0},
    {cls:"s-kv",  color:"var(--kv)",  label:"KV cache (GPU)", mib:p.gpu_kv_mib||0},
    {cls:"s-kv",  color:"#2aa88c",     label:"recurrent state", mib:p.gpu_recurrent_mib||0},
    {cls:"s-wt",  color:"#7d6cff",     label:"vision projector", mib:p.mmproj_mib||0},
    {cls:"s-kv",  color:"#e0a03a",     label:"MTP draft cache", mib:p.spec_mib||0},
    {cls:"s-cmp", color:"var(--cmp)", label:"compute buffer", mib:p.compute_mib||0},
    {cls:"s-rsv", color:"#2c3742",    label:"driver reserve", mib:inp.gpu_reserve_mib||0},
  ];
  const vramUsed=vramSegs.reduce((a,x)=>a+x.mib,0);
  const ramSegs=[
    {cls:"s-wt", color:"var(--wt)", label:"weights (CPU/RAM)", mib:p.cpu_weights_mib||0},
    {cls:"s-kv", color:"var(--kv)", label:"KV cache (RAM)", mib:p.cpu_kv_mib||0},
    {cls:"s-kv", color:"#2aa88c",   label:"recurrent state (RAM)", mib:p.cpu_recurrent_mib||0},
    {cls:"s-cmp",color:"var(--cmp)",label:"compute buffer (CPU)", mib:p.cpu_compute_mib||0},
  ];
  const ramUsed=ramSegs.reduce((a,x)=>a+x.mib,0);

  let verdict='<div class="card"><h2>Verdict</h2><div class="verdict '+vcls+'"><div class="h">'+
    p.headline+'</div></div>'+
    bar("VRAM", inp.vram_budget_mib, vramUsed, vramSegs);
  if(ramUsed>0.5) verdict += bar("System RAM", inp.ram_budget_mib, ramUsed, ramSegs);
  verdict += '<div class="note">Weights and KV cache are computed exactly from the GGUF tensor '+
    'table. '+
    ((r.calibration && r.calibration.calibrated)
      ? 'The compute buffer is <b style="color:var(--kv)">calibrated for your GPU</b> from '+
        r.calibration.n+' measurement'+(r.calibration.n==1?'':'s')+' ('+
        r.calibration.free.join(", ")+' fitted, in-sample '+r.calibration.residual_pct+'%). '
      : 'The compute buffer uses shipped defaults, fitted to measured llama.cpp runs across 4 '+
        'architectures (4.8% mean, 12% worst on one CUDA card). It is the only term that depends '+
        'on your hardware rather than the model &mdash; press <b>Measure running model</b> with a '+
        'model loaded to calibrate it for yours. ')+
    ((p.cpu_compute_mib>0.5)
      ? 'Split across backends here: '+fmt(p.compute_mib)+' on the GPU, '+fmt(p.cpu_compute_mib)+
        ' in RAM. llama.cpp gives every backend running part of the graph its own scratch pool, '+
        'and the '+fmt(s.compute_logits)+' logits tensor ('+inp.n_ubatch+' &times; '+
        (c.n_vocab||0).toLocaleString()+' vocab) belongs to whichever side holds the output head.'
      : 'All of it ('+fmt(p.compute_mib)+') is on the GPU at this split.')+
    '</div></div>';

  // ---- settings ----
  let setrows = (p.lmstudio||[]).map(x=>'<div class="setrow">'+x+'</div>').join("");
  let settings='<div class="card"><h2>Settings to use</h2>'+
    '<div style="margin-bottom:12px"><div class="k muted small" style="margin-bottom:6px;font-family:var(--mono);letter-spacing:.06em">LM STUDIO &middot; advanced load settings</div>'+
    setrows+'</div>'+
    '<div class="cmdhead"><div class="k muted small" style="font-family:var(--mono);letter-spacing:.06em">LLAMA.CPP</div>'+
    '<button class="ghost" onclick="copyCmd(this)">copy</button></div>'+
    '<pre class="cmd" id="cmd">'+escapeHtml(p.llama_cmd||"")+'</pre>';
  if(p.max_ctx_kv_gpu!=null)
    settings += '<div class="note">Max context with <b style="color:var(--ink)">all KV on GPU</b> '+
      '(FFN on CPU): ~<b style="color:var(--ink)">'+p.max_ctx_kv_gpu.toLocaleString()+'</b> tokens.</div>';
  else if(p.max_ctx_gpu!=null)
    settings += '<div class="note">Max context fully on GPU at this quant: ~<b style="color:var(--ink)">'+
      p.max_ctx_gpu.toLocaleString()+'</b> tokens.</div>';
  settings += '</div>';

  // ---- KV table ----
  let rows=(r.kv_table||[]).map(t=>{
    const cur = t.ctx===inp.context;
    return '<tr'+(cur?' style="color:var(--kv)"':'')+'><td>'+t.ctx.toLocaleString()+
      (cur?'  &larr;':'')+'</td><td>'+fmt(t.kv_mib)+'</td><td>'+fmtG(t.kv_mib)+'</td></tr>';
  }).join("");
  let kvcard='<div class="card"><h2>KV cache vs context ('+inp.kv_type+')</h2>'+
    '<table><thead><tr><th>context</th><th>KV size</th><th></th></tr></thead><tbody>'+rows+
    '</tbody></table><div class="note">Per token: '+s.kv_per_token_kib.toFixed(1)+' KiB across '+
    ((r.hybrid&&r.hybrid.is_hybrid)? r.hybrid.n_attn_layers+' KV-bearing of '+c.n_layers+' layers &mdash; '+
      'this is a hybrid model, the other '+r.hybrid.n_ssm_layers+' layers use a fixed recurrent state'
      : 'all '+c.n_layers+' layers')+'.</div>'+
    ((r.swa && r.swa.enabled)
      ? '<div class="note">This model uses <b style="color:var(--ink)">sliding-window attention</b>, so '+
        'the table is not a straight line. '+r.swa.n_swa_layers+' layers only ever cache '+
        r.swa.window_cache_tokens.toLocaleString()+' tokens (a '+r.swa.n_swa.toLocaleString()+
        '-token window &times; '+inp.n_seq+' seq + one ubatch, padded) &mdash; '+fmt(s.kv_swa)+
        ', flat no matter how long the context gets. Only the '+r.swa.n_global_layers+
        ' full-attention layers grow with context ('+fmt(s.kv_global)+' here, '+
        s.kv_grow_per_token_kib.toFixed(1)+' KiB/token). Detected from <code>'+r.swa.source+'</code>.</div>'
      : "")+'</div>';

  // ---- breakdown ----
  let brk='<div class="card"><h2>Memory breakdown</h2><table><tbody>'+
    brow("Model weights (exact)", fmt(s.weights)+"  ("+fmtG(s.weights)+")")+
    brow("&nbsp;&nbsp;token embeddings", fmt(s.embed))+
    brow("&nbsp;&nbsp;output / head", fmt(s.output))+
    (r.is_moe? brow("&nbsp;&nbsp;routed experts (offloadable)", fmt(s.expert_total)) : "")+
    brow("Per-layer weight (avg)", fmt(s.per_layer_mean))+
    brow("KV cache @ "+inp.context.toLocaleString()+" ctx", fmt(s.kv_total)+
      (r.hybrid && r.hybrid.is_hybrid
        ? '  <span class="muted">('+r.hybrid.n_attn_layers+' attn layers)</span>' : ""))+
    ((r.swa && r.swa.enabled)
      ? brow("&nbsp;&nbsp;full-attention layers ("+r.swa.n_global_layers+")", fmt(s.kv_global))+
        brow("&nbsp;&nbsp;sliding-window layers ("+r.swa.n_swa_layers+", fixed)", fmt(s.kv_swa))
      : "")+
    (s.recurrent_total>0.5? brow("Recurrent state (fixed, x"+inp.n_seq+" seq)", fmt(s.recurrent_total)) : "")+
    brow("Compute buffer (est.)", fmt(s.compute))+
    brow("&nbsp;&nbsp;graph scratch + attn mask", fmt(s.compute_graph))+
    brow("&nbsp;&nbsp;logits ("+inp.n_ubatch+" x "+(c.n_vocab||0).toLocaleString()+" vocab)",
      fmt(s.compute_logits)+'  <span class="muted">'+
      ((p.cpu_compute_mib>0.5)?'on CPU &mdash; the output head is not offloaded':'on GPU')+'</span>')+
    (r.mmproj? brow("Vision projector ("+r.mmproj.name+")",
        fmt(r.mmproj.mib)+(r.mmproj.included?"":'  <span class="muted">not loaded</span>')) : "")+
    brow("File on disk", fmtG(s.file_on_disk)+"  ("+fmtGB(s.file_on_disk)+")")+
    (r.mmproj? brow("&nbsp;&nbsp;+ projector = LM Studio's &quot;model size&quot;",
        fmtG(s.bundle_on_disk)+"  ("+fmtGB(s.bundle_on_disk)+")") : "")+
    '</tbody></table></div>';

  let warns="";
  if(r.warnings && r.warnings.length)
    warns='<div class="card warns"><h2 style="color:var(--warn)">Notes</h2>'+
      r.warnings.map(w=>'<div>&#9888; '+w+'</div>').join("")+'</div>';

  // ---- speed ----
  let speed = "";
  const sp = r.speed;
  if(sp && !sp.error){
    const num = sp.calibrated
      ? '<span style="color:var(--kv);font-size:26px;font-weight:600">'+sp.tok_s.toFixed(1)+'</span> tok/s'
      : '<span style="color:var(--ink);font-size:26px;font-weight:600">'+sp.tok_s_lo.toFixed(0)+
        '&ndash;'+sp.tok_s_hi.toFixed(0)+'</span> tok/s';
    speed = '<div class="card"><h2>Generation speed</h2>'+
      '<div style="margin:2px 0 14px">'+num+
      '<span class="muted small" style="margin-left:10px">'+
      (sp.calibrated? 'calibrated &middot; RAM at '+(sp.ram_eff*100).toFixed(0)+'% of peak'
                    : 'uncalibrated bracket &middot; measure once to collapse it')+
      '</span></div>'+
      '<table><tbody>'+
      brow("Read from VRAM per token", fmt(sp.gpu_mib))+
      brow("Read from system RAM per token", fmt(sp.cpu_mib)+
        (sp.cpu_mib>sp.gpu_mib/4? '  <span style="color:var(--warn)">&larr; the bottleneck</span>':""))+
      (sp.expert_frac<1? brow("Experts active per token",
          (sp.expert_frac*100).toFixed(2)+"% of expert weights") : "")+
      brow("Context assumed filled", sp.ctx_fill.toLocaleString()+" tokens")+
      brow("Bandwidth used", Math.round(sp.bw_vram_gbs)+" GB/s VRAM &middot; "+
           sp.bw_ram_gbs.toFixed(1)+" GB/s RAM")+
      '</tbody></table>'+
      '<div class="note">Generation is memory-bandwidth bound: every token reads each active '+
      'weight once. Byte counts are exact; the bandwidths are not, which is the whole width of '+
      'the bracket. Prompt processing is compute bound and is <b>not</b> modelled here.</div>'+
      '<div style="margin-top:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'+
      '<button class="ghost" onclick="benchNow()">&#9654; Benchmark the loaded model</button>'+
      '<span class="muted small">runs one short generation on the LM Studio server '+
      '(localhost:1234) &mdash; do it while your agent is idle</span></div>'+
      '<div id="benchout"></div>'+
      '<div id="spdhist"><div class="muted small">looking for past measurements...</div></div></div>';
  }

  $("out").innerHTML = sum + verdict + settings + speed + kvcard + brk + warns;
  if(sp && !sp.error) loadSpeedHistory(r);
}

async function benchNow(){
  const out = $("benchout");
  out.innerHTML = '<div class="note">generating... this occupies the GPU for a few seconds</div>';
  const r = LAST;
  let q = "?max_tokens=128";
  if(r && r.inputs) q += "&ctx="+r.inputs.context;
  if(r && r.plan && r.plan.n_gpu_layers!=null) q += "&ngl="+r.plan.n_gpu_layers;
  let d;
  try{ d = await (await fetch("/api/benchmark"+q)).json(); }
  catch(e){ out.innerHTML='<div class="note" style="color:var(--warn)">benchmark failed: '+e+'</div>'; return; }
  if(d.error || !d.tok_s){
    out.innerHTML='<div class="note" style="color:var(--warn)">'+(d.error||"no throughput returned")+
      '<br>Is a model loaded and the server running on localhost:1234?</div>'; return; }
  const fill = (d.prompt_tokens||0) + Math.round((d.predicted_tokens||0)/2);
  out.innerHTML = '<div class="note"><b style="color:var(--kv)">'+d.tok_s.toFixed(2)+' tok/s</b> measured on <b>'+
    d.model+'</b> &middot; '+d.predicted_tokens+' tokens'+
    (d.ttft_s!=null? ' &middot; TTFT '+Number(d.ttft_s).toFixed(2)+'s':'')+
    ' &middot; source: '+d.source+
    ' <button class="ghost" style="margin-left:8px" onclick="useMeasured('+d.tok_s+','+fill+')">calibrate from this</button>'+
    '<br>Saved to speed_history.json, so it shows up in the list below from now on.</div>';
  loadSpeedHistory(LAST);
}

async function loadSpeedHistory(r){
  const box = $("spdhist"); if(!box) return;
  let h;
  try{
    h = await (await fetch("/api/speedhistory?model="+encodeURIComponent(r.model_name||"")+
                           "&path="+encodeURIComponent(($("path").value||$("model").value||"")))).json();
  }catch(e){ box.innerHTML='<div class="muted small">history read failed: '+e+'</div>'; return; }
  if(h.error){ box.innerHTML='<div class="muted small">no measurement history: '+h.error+'</div>'; return; }
  const ms = h.matches||[];
  if(!ms.length){
    const n=(h.all||[]).length;
    box.innerHTML='<div class="note">No past measurements for <b>'+(r.model_name||"this model")+
      '</b> in LM Studio’s saved chats'+(n? ' ('+n+' record'+(n==1?'':'s')+' for other models)':'')+
      '. Generate once in LM Studio, then re-run &mdash; the tok/s it reports gets saved with the '+
      'chat and shows up here to calibrate against.</div>';
    return;
  }
  let rows = ms.slice(0,8).map((m,i)=>{
    const fill = (m.prompt_tokens||0) + Math.round((m.predicted_tokens||0)/2);
    return '<tr><td>'+m.tok_s.toFixed(2)+' tok/s</td>'+
      '<td class="muted">ngl '+(m.n_gpu_layers==null?"?":m.n_gpu_layers)+
        ' &middot; ctx '+(m.ctx? m.ctx.toLocaleString():"?")+
        ' &middot; ~'+fill.toLocaleString()+' filled</td>'+
      '<td style="text-align:right"><button class="ghost" onclick="useMeasured('+
        m.tok_s+','+fill+')">calibrate</button></td></tr>';
  }).join("");
  box.innerHTML = '<div style="margin-top:16px"><div class="k muted small" style="font-family:var(--mono);'+
    'letter-spacing:.06em;margin-bottom:6px">MEASURED IN LM STUDIO'+
    (ms[0].match<3? ' (approximate model match — check the config)':'')+'</div>'+
    '<table><tbody>'+rows+'</tbody></table>'+
    '<div class="note">Picking one sets <b>Context filled</b> to that run and solves for the real '+
    'RAM efficiency, then re-runs. Only calibrate against a run whose layer split matches the plan '+
    'above &mdash; otherwise you are fitting the wrong configuration.</div></div>' + srvLogPanel(h);
}

function srvLogPanel(h){
  const sl = h.server_log;
  if(!Array.isArray(sl) || !sl.length) return "";
  const n = Math.min(sl.length, 20), recent = sl.slice(0, n);
  const pre = recent.reduce((a,x)=>a+x.prefill_s,0), dec = recent.reduce((a,x)=>a+x.decode_s,0);
  if(pre+dec <= 0) return "";
  const pct = Math.round(pre/(pre+dec)*100);
  return '<div style="margin-top:16px"><div class="k muted small" style="font-family:var(--mono);'+
    'letter-spacing:.06em;margin-bottom:6px">SERVER MODE &middot; LAST '+n+' RESPONSES</div>'+
    '<table><tbody>'+
    brow("Prompt processing (prefill)", pre.toFixed(0)+" s total &middot; <b>"+pct+"%</b> of the time")+
    brow("Token generation (decode)", dec.toFixed(0)+" s total &middot; "+(100-pct)+"%")+
    '</tbody></table><div class="note">Wall times parsed from '+
    '<span class="mono">~/.lmstudio/server-logs</span>. The succinct log has no token counts, so this '+
    'cannot give tok/s &mdash; but it does show where your time actually goes. '+
    (pct>=50? '<b>Prefill dominates here</b>, and prefill is compute bound, so layer-split tuning '+
              'will not fix it &mdash; a shorter prompt or a reused prefix will.'
            : 'Decode dominates, which is what the bandwidth model above predicts.')+
    '</div></div>';
}

async function useMeasured(tokS, fill){
  $("ctxfill").value = Math.max(0, Math.round(fill));
  $("rameff").value = "";              // clear so the solve is not anchored to an old value
  const r = LAST; if(!r || !r.speed) return;
  // solve: t_total = gpu_bytes/(BWv*GPU_EFF) + cpu_bytes/(BWr*eff)
  const sp = r.speed, GPU_EFF = 0.85;
  const gpuB = sp.gpu_mib*1048576, cpuB = sp.cpu_mib*1048576;
  const tTot = 1/tokS, tGpu = gpuB/(sp.bw_vram_gbs*GPU_EFF*1e9), tCpu = tTot-tGpu;
  if(cpuB<=0 || tCpu<=0){
    $("calhint").innerHTML='<b style="color:var(--warn)">That measurement is faster than the GPU '+
      'term alone allows &mdash; it was a different config (or a different context fill). Not calibrating.</b>';
    return;
  }
  $("rameff").value = (cpuB/(tCpu*sp.bw_ram_gbs*1e9)).toFixed(3);
  run();
}

async function calibrate(){
  const h=$("calhint");
  if(!LAST){ h.innerHTML='<b style="color:var(--warn)">Press Analyze fit first, so the exact terms are known.</b>'; return; }
  h.textContent="reading GPU processes...";
  let r;
  try{ r = await (await fetch("/api/gpuprocs")).json(); }
  catch(e){ h.textContent="could not read GPU processes: "+e; return; }
  const procs = r.procs;
  if(!Array.isArray(procs) || !procs.length){
    h.innerHTML='<b style="color:var(--warn)">No GPU processes readable'+
      (procs&&procs.error? " ("+procs.error+")":"")+'.</b>'; return; }
  const eng = procs.find(p=>p.is_engine) || procs[0];
  const s=LAST.sizes_mib, p=LAST.plan, inp=LAST.inputs;

  // A measurement is only meaningful against a plan for the SAME config. LM Studio
  // silently overrides its own GPU Offload slider to respect its VRAM cap, so read
  // what it actually loaded and refuse to back-solve against a different split -
  // otherwise every layer of difference lands in the compute buffer as error.
  const rt = r.runtime;
  if(rt){
    let bad=[];
    if(rt.n_gpu_layers!=null && rt.n_gpu_layers!==p.n_gpu_layers)
      bad.push("GPU layers: engine is running <b>"+rt.n_gpu_layers+"</b>, this plan is for <b>"+
        p.n_gpu_layers+"</b>"+(inp&&LAST.plan.forced_ngl?"":" (auto-picked)"));
    if(rt.context!=null && rt.context!==inp.context)
      bad.push("Context: engine is running <b>"+rt.context.toLocaleString()+
        "</b>, this plan is for <b>"+inp.context.toLocaleString()+"</b>");
    if(bad.length){
      h.innerHTML='<b style="color:var(--warn)">Config mismatch - not measuring.</b><br>'+
        bad.join("<br>")+'<br>Set <b>GPU layers</b> to '+rt.n_gpu_layers+
        (rt.context!=null?' and <b>context</b> to '+rt.context.toLocaleString():'')+
        ', press Analyze fit, then Measure again. '+
        '<button class="ghost" onclick="matchRuntime('+rt.n_gpu_layers+','+(rt.context||0)+')">'+
        'do it for me</button>';
      return;
    }
  }

  // Record it as a calibration sample rather than a one-off override: the server
  // recomputes the exact terms, stores the row, and refits the coefficients for
  // this GPU. That makes every future plan better, not just this one.
  h.textContent="measuring and refitting...";
  let path = $("path").value.trim();
  if(!path){ const i=$("model").value; if(i!=="" && MODELS[parseInt(i)]) path = MODELS[parseInt(i)].path; }
  let res;
  try{
    res = await (await fetch("/api/calibrate",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({path:path, context:parseInt($("ctx").value)||8192,
        kv_type:$("kv").value, n_ubatch:parseInt($("ubatch").value)||512,
        n_seq:parseInt($("nseq").value)||1, flash_attn:$("fa").checked,
        include_mmproj:$("mmproj").checked})})).json();
  }catch(e){ h.textContent="calibration failed: "+e; return; }
  if(!res.ok){ h.innerHTML='<b style="color:var(--warn)">'+escapeHtml(res.error||"failed")+'</b>'; return; }

  const row=res.row, st=res.status;
  const pending = CALIB_TERMS.filter(t=>st.free.indexOf(t)<0);
  h.innerHTML = '<b style="color:var(--kv)">'+eng.name+'</b> is using <b>'+fmt(row.measured_mib)+
    '</b> at ngl '+row.ngl+' / ctx '+row.ctx.toLocaleString()+
    ' <span class="muted">(read from LM Studio\\'s log)</span>.<br>'+
    'minus exact terms '+fmt(row.exact_mib)+' &rarr; overhead <b>'+fmt(row.overhead_mib)+'</b>.<br>'+
    (st.calibrated
      ? '<b style="color:var(--kv)">Calibrated</b> from '+st.n+' measurement'+(st.n==1?'':'s')+
        ' on this GPU &mdash; fitted: '+st.free.join(", ")+
        ' (in-sample '+st.residual_pct+'%).'+
        (st.skipped_rows? '<br><span class="muted">'+st.skipped_rows+' stored measurement'+
          (st.skipped_rows==1?' was':'s were')+' left out: the reading did not respond to '+
          'the config, or the layer count was never recorded. Re-measure with VRAM to '+
          'spare to bring them back.</span>' : '')
      : '<b style="color:var(--warn)">Recorded, but not fitted yet.</b> The measurement is '+
        'saved; it did not produce a usable fit on its own, so the shipped defaults still '+
        'apply. Measure once more at a different context length.')+
    (pending.length? '<br><span class="muted">Still on defaults: '+pending.join(", ")+
      '. '+calibHint(pending)+'</span>' : '')+
    '<br>Press Analyze fit again.';
  run();
}

const CALIB_TERMS=["const","ctx","act","nofa"];
function calibHint(pending){
  const tips={ctx:"measure again at a very different context length",
              act:"measure a model with a different FFN width, or change ubatch",
              nofa:"measure once with flash attention off"};
  return pending.map(t=>tips[t]).filter(Boolean).join("; ")+".";
}

async function matchRuntime(ngl, ctx){
  $("ngl").value = ngl;
  if(ctx){ $("ctx").value = ctx; markCtx(); }
  await run();
  calibrate();
}

function kvItem(k,v){ return '<div class="kv"><div class="k">'+k+'</div><div class="v">'+(v==null?"-":v)+'</div></div>'; }
function brow(k,v){ return '<tr><td>'+k+'</td><td>'+v+'</td></tr>'; }
function escapeHtml(x){ return x.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function copyCmd(btn){
  const t=$("cmd").textContent;
  navigator.clipboard.writeText(t).then(()=>{ const o=btn.textContent; btn.textContent="copied"; setTimeout(()=>btn.textContent=o,1200); });
}
boot();
'''


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _calibrate(self, data):
        """Record one Measure as a calibration row. Everything that matters is
        determined server-side: the engine's real VRAM, and the config LM Studio
        actually loaded (which is not always the one on screen)."""
        path = data.get("path") or ""
        if not os.path.exists(path):
            return {"ok": False, "error": "file not found: %s" % path}
        procs = get_gpu_processes()
        if isinstance(procs, dict):
            return {"ok": False, "error": "could not read GPU processes: %s"
                                          % procs.get("error", "?")}
        eng = next((p for p in procs if p.get("is_engine")), None)
        if not eng:
            return {"ok": False, "error": "No inference process on the GPU. Load the "
                                          "model in LM Studio first."}
        rt = read_lmstudio_runtime()
        try:
            n_layers = extract_config(load_gguf(path))["n_layers"] or 0
        except Exception as e:
            return {"ok": False, "error": "could not read model: %s" % e}
        ngl = resolve_runtime_ngl(rt, n_layers)
        ctx = (rt or {}).get("context") or int(data.get("context") or 0)
        if ngl is None:
            return {"ok": False, "error": "Could not read LM Studio's resolved GPU config "
                                          "from its log, so the layer count is unknown. "
                                          "Measurement skipped rather than guessed."}
        try:
            out = record_calibration(
                path, ctx, data.get("kv_type") or "f16",
                int(data.get("n_ubatch") or 512), int(data.get("n_seq") or 1),
                bool(data.get("flash_attn", True)), int(ngl),
                bool(data.get("include_mmproj", True)), float(eng["mib"]),
                gpu=_active_gpu(), n_cpu_moe=(rt or {}).get("n_cpu_moe") or 0)
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        if out.get("error"):
            return {"ok": False, "error": out["error"]}
        out.update({"ok": True, "engine": eng,
                    "runtime": {"n_gpu_layers": ngl, "context": ctx,
                                "n_cpu_moe": (rt or {}).get("n_cpu_moe") or 0,
                                "all_layers": bool((rt or {}).get("all_layers"))}})
        return out

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        if u.path == "/app.js":
            return self._send(200, APP_JS, "application/javascript; charset=utf-8")
        if u.path == "/api/benchmark":
            q = parse_qs(u.query)
            res = benchmark_server(
                base_url=(q.get("url") or ["http://localhost:1234"])[0],
                max_tokens=int((q.get("max_tokens") or ["128"])[0]))
            if res.get("tok_s"):
                res["ctx"] = (q.get("ctx") or [None])[0]
                res["n_gpu_layers"] = (q.get("ngl") or [None])[0]
                save_benchmark(res)
            return self._send(200, res)
        if u.path == "/api/bandwidth":
            return self._send(200, get_bandwidth())
        if u.path == "/api/speedhistory":
            q = parse_qs(u.query)
            recs = scan_speed_history()
            if isinstance(recs, dict):
                recs = []
            for bm in load_benchmarks():          # server-mode measurements
                if bm.get("tok_s"):
                    recs.append({"model": bm.get("model") or "?", "tok_s": bm["tok_s"],
                                 "n_gpu_layers": bm.get("n_gpu_layers"),
                                 "ttft_s": bm.get("ttft_s"),
                                 "prompt_tokens": bm.get("prompt_tokens") or 0,
                                 "predicted_tokens": bm.get("predicted_tokens") or 0,
                                 "ctx": bm.get("ctx"), "when": bm.get("when"),
                                 "conversation": "benchmark", "benchmark": True})
            name = (q.get("model") or [""])[0]
            path = (q.get("path") or [""])[0]
            return self._send(200, {"matches": match_speed_history(recs, name, path),
                                    "all": recs, "server_log": scan_server_logs(60)})
        if u.path == "/api/gpuprocs":
            return self._send(200, {"procs": get_gpu_processes(),
                                    "runtime": read_lmstudio_runtime(),
                                    "calibration": calibration_status()})
        if u.path == "/api/system":
            fresh = parse_qs(u.query).get("fresh", ["0"])[0] == "1"
            return self._send(200, {"gpus": get_gpus(fresh=fresh), "ram": get_ram(),
                                    "default_dir": default_models_dir(),
                                    "version": __version__,
                                    "platform": platform_support()})
        if u.path == "/api/models":
            q = parse_qs(u.query)
            d = (q.get("dir", [""])[0]) or default_models_dir()
            try:
                return self._send(200, {"models": scan_models(d), "dir": d})
            except Exception as e:
                return self._send(200, {"models": [], "error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/api/analyze", "/api/calibrate"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._send(200, {"ok": False, "error": "bad request: %s" % e})
        if u.path == "/api/calibrate":
            return self._send(200, self._calibrate(data))
        try:
            path = data["path"]
            if not os.path.exists(path):
                return self._send(200, {"ok": False, "error": "file not found: %s" % path})
            res = analyze(
                path=path,
                ctx=int(data.get("context", 8192)),
                kv_type=data.get("kv_type", "f16"),
                n_ubatch=int(data.get("n_ubatch", 512)),
                n_seq=int(data.get("n_seq", 1) or 1),
                include_mmproj=bool(data.get("include_mmproj", True)),
                mtp_spec=bool(data.get("mtp_spec", True)),
                flash_attn=bool(data.get("flash_attn", False)),
                vram_budget_mib=float(data.get("vram_budget_mib", 0)),
                ram_budget_mib=float(data.get("ram_budget_mib", 0)),
                gpu_reserve_mib=float(data.get("gpu_reserve_mib", 512)),
                compute_override_mib=float(data.get("compute_override_mib", 0)),
                safety_pct=float(data.get("safety_pct", 5)),
                kv_on_gpu=bool(data.get("kv_on_gpu", False)),
                bw_vram_gbs=float(data.get("bw_vram_gbs") or 0) or None,
                bw_ram_gbs=float(data.get("bw_ram_gbs") or 0) or None,
                ram_eff=float(data.get("ram_eff") or 0) or None,
                ctx_fill=(int(data["ctx_fill"]) if data.get("ctx_fill") not in (None,"",False) else None),
                n_cpu_moe_override=(int(data["n_cpu_moe_override"])
                                    if data.get("n_cpu_moe_override") not in (None, "", False)
                                    else None),
                gpu_layers_override=(int(data["gpu_layers_override"])
                                     if data.get("gpu_layers_override") not in (None, "", False)
                                     else None),
                ram_free_mib=(float(data["ram_free_mib"])
                              if data.get("ram_free_mib") else None),
            )
            return self._send(200, res)
        except Exception as e:
            import traceback
            return self._send(200, {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                                    "trace": traceback.format_exc()})


def serve(host, port, open_browser):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0") else host, port)
    refresh_calibration()
    st = calibration_status()
    plat = platform_support()
    print("\n  VRAM Planner %s  running at  %s" % (__version__, url))
    print("  models folder default :  %s" % default_models_dir())
    print("  user data             :  %s" % _data_dir())
    print("  compute-buffer model  :  %s"
          % ("calibrated from %d measurement(s) on %s (fitted: %s, in-sample %.1f%%)"
             % (st["n"], st["gpu"] or "this GPU", ", ".join(st["free"]), st["residual_pct"])
             if st["calibrated"] else
             "shipped defaults - press Measure on a loaded model to calibrate"))
    if st.get("skipped_rows"):
        print("  measurements skipped  :  %d (reading did not respond to the config, or "
              "no layer count recorded)" % st["skipped_rows"])
    if not plat["supported"]:
        print("\n  !! UNVALIDATED PLATFORM\n     %s" % plat["reason"])
    print("  press Ctrl+C to stop\n")
    if open_browser:
        try: webbrowser.open(url)
        except Exception: pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.server_close()
