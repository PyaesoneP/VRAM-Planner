"""Turn a config into a launch script someone can actually live with.

The payoff of a speed campaign is a command line, and a command line is the part
that gets lost. This module writes the whole launcher instead - the flags, the
backend resolution, the library path, the log file - so that none of the traps
below have to be rediscovered:

  * `llama-server` is not on PATH. It ships inside LM Studio's backends folder.
  * It will not run from that folder alone: the CUDA runtime lives in a separate
    shared "vendor" package, and without it the process dies with
    STATUS_DLL_NOT_FOUND (exit -1073741515) and no error message at all.
  * A pinned backend path silently stops existing when LM Studio updates, so the
    generated script resolves the newest build of the same family every launch.
  * `--draft-max` was REMOVED; the knob is `--spec-draft-n-max`.
  * `--no-mmap` and `--mlock` are DEPRECATED in favour of `--load-mode`. The old
    names are accepted and then ignored, which looks exactly like a setting that
    did nothing.
  * `--jinja` has to be passed BEFORE `--chat-template-file`, or the build
    accepts only its built-in template NAMES and rejects a path. It is
    default-on in current builds and was not in older ones - and the backend is
    resolved fresh every launch, so that default can move underneath you.
  * A `--chat-template-file` that does not exist is not an error: llama-server
    falls back to the template baked into the GGUF without saying so.
  * Windows PowerShell 5.1 DROPS double quotes that are already inside an
    argument when it builds a native command line, so a valid
    `{"a":1}` arrives at the exe as `{a:1}`. Only `--chat-template-kwargs`
    carries quotes, and the resulting error names the JSON rather than the
    shell that mangled it.
  * Launched directly, llama-server logs to the console and nothing is kept.

Flag spelling is NOT re-implemented here. sweep.build_argv() is the single source
of truth, and this module calls it with probe=False and then substitutes script
variables in for the handful of values worth parameterising. A second copy of the
flag list would drift, and the flags most worth getting right are precisely the
ones that have changed names.

Pure text generation - no subprocess work, nothing launched - so plan.py and
web.py can import it freely.
"""
import datetime, json, os, re

from .sweep import backends_dir, build_argv, find_backends

SHELLS = ("powershell", "bash")

# sweep.EXE_NAME is right for the HOST, and the script is written for whichever
# shell the user asked for - which is not always the same thing. Generating a bash
# script on Windows must not go looking for llama-server.exe.
EXE_BY_SHELL = {"powershell": "llama-server.exe", "bash": "llama-server"}

# --load-mode replaced --no-mmap/--mlock. `none` is the default here and it is not
# llama.cpp's: measured at 13.81 GiB resident versus 17.94 GiB for mmap+mlock on a
# 9.5 GiB working set, because mlock pins the whole mapped file INCLUDING the
# blocks already resident in VRAM. Right for CPU-only inference, wasteful under
# heavy GPU offload.
LOAD_MODES = ("none", "mmap", "mlock", "mmap+mlock", "dio")

# Sampler keys we know how to emit, in the order a person reads them, each with the
# type its PARAMETER must have. The type belongs to the flag, not to the value that
# happened to arrive: `temp: 1` is a perfectly ordinary JSON integer, and inferring
# from it would declare [int]$Temp - which then silently truncates -Temp 0.7 to 0.
# top-k is the only genuinely integral one. Nothing here has a default: see
# sampler_args().
SAMPLER_FLAGS = (
    ("temp", "--temp", "double"),
    ("top_k", "--top-k", "int"),
    ("top_p", "--top-p", "double"),
    ("min_p", "--min-p", "double"),
    ("repeat_penalty", "--repeat-penalty", "double"),
    ("presence_penalty", "--presence-penalty", "double"),
)

# Which argv values become script parameters, keyed by the flag in front of them.
# Everything not listed is emitted literally.
_PARAM_BY_FLAG = {
    "-m": "Model", "--mmproj": "Mmproj", "-ngl": "Ngl", "-c": "Ctx",
    "-ub": "Ubatch", "--n-cpu-moe": "NCpuMoe", "--spec-draft-n-max": "DraftMax",
    "--host": "BindHost", "--port": "Port",
}
# Model and mmproj are resolved paths, not tunables: they get a variable for
# readability but no param block entry.
_NOT_A_PARAM = ("Model", "Mmproj")


def default_shell():
    return "powershell" if os.name == "nt" else "bash"


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------
def ps_quote(s):
    """Single-quoted PowerShell literal: no expansion, ' doubled."""
    return "'%s'" % str(s).replace("'", "''")


def sh_quote(s):
    """Single-quoted POSIX literal: no expansion, ' spliced out and back."""
    return "'%s'" % str(s).replace("'", "'\\''")


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
def sampler_args(sampling):
    """(flag, value, kind) for each sampler the caller actually set.

    Blank means blank. A fabricated default would be worse than none: llama.cpp's
    own defaults (temp 0.80, top-k 40, min-p 0.05) are not what several model
    cards recommend, so a value printed here has to be one somebody chose. The
    caller passing nothing gets a script with no sampler flags, which is the
    honest representation of "I did not decide this"."""
    out = []
    for key, flag, kind in SAMPLER_FLAGS:
        v = (sampling or {}).get(key)
        if v is None or v == "":
            continue
        # Render to the flag's own type, so a double-valued sampler that arrived
        # as a whole number still reads (and declares) as a double.
        try:
            v = int(float(v)) if kind == "int" else float(v)
        except (TypeError, ValueError):
            continue
        out.append((flag, v, kind))
    return out


def template_args(chat_template_file=None, chat_template_kwargs=None):
    """Validate the chat-template pair, returning (path, kwargs_json).

    llama-server's own help is explicit that --chat-template-kwargs "must be a
    valid json object string", and it does not find out otherwise until it is
    starting up. Checking here is the difference between a message in the browser
    and a service that will not come back after a reboot, so this raises rather
    than passing anything doubtful through.

    A dict is accepted as well as a string, because the web layer has already
    parsed the request body and re-serialising to make this re-parse it would be
    a round trip that could only lose. Keys are sorted so regenerating the same
    script twice produces the same bytes."""
    f = (chat_template_file or "").strip() or None
    kw = chat_template_kwargs
    if kw is None or (isinstance(kw, str) and not kw.strip()):
        return f, None
    if isinstance(kw, dict):
        obj = kw
    else:
        try:
            obj = json.loads(kw)
        except (TypeError, ValueError) as e:
            raise ValueError("--chat-template-kwargs must be valid JSON: %s" % e)
    if not isinstance(obj, dict):
        # A list or a bare scalar parses fine and is still wrong: the flag names
        # template VARIABLES, so the top level has to be an object.
        raise ValueError('--chat-template-kwargs must be a JSON object, e.g. '
                         '{"enable_thinking": false} - got %s'
                         % type(obj).__name__)
    return f, json.dumps(obj, separators=(",", ":"), sort_keys=True)


def backend_match(build):
    """A wildcard matching every version of one backend family.

    The build directory name ends in a version - strip it and the rest names the
    family (OS, arch, GPU framework, instruction set). Matching on the family
    rather than the exact directory is what lets the script survive an LM Studio
    update; matching on nothing at all would let it pick a CPU build."""
    return re.sub(r"[-_.]?\d+\.\d+\.\d+\s*$", "", build or "") + "*"


def command_lines(text):
    """The lines a shell would execute, with comments and blanks dropped.

    A generated script NAMES the deprecated flags in its comments, on purpose -
    "`--no-mmap` is ignored, use `--load-mode`" is the useful part. So any check
    for "does this script pass a dead flag" has to look at what runs, not at what
    the file contains, or the explanation trips the guard that the explanation
    exists to make unnecessary. Both shells use # for comments."""
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(ln)
    return out


def script_name(model_path, shell):
    stem = re.sub(r"\.gguf$", "", os.path.basename(model_path or "model"), flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "model"
    return "run-%s.%s" % (stem.lower(), "ps1" if shell == "powershell" else "sh")


def _lines_from_argv(argv, var):
    """Group argv (past the exe) into one readable line per setting.

    `var(name)` renders a script variable reference. -ctk/-ctv are held on one
    line because they are one decision, not two."""
    lines, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-"):          # defensive; build_argv never does this
            lines.append([tok]); i += 1; continue
        val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else None
        if val is None:
            lines.append([tok]); i += 1; continue
        param = _PARAM_BY_FLAG.get(tok)
        rendered = var(param) if param else val
        if tok == "-ctv" and lines and lines[-1][0] == "-ctk":
            lines[-1] += [tok, rendered]
        else:
            lines.append([tok, rendered])
        i += 2
    return lines


def _params_used(argv):
    """Which parameters this config actually reaches, in argv order.

    A config with no speculation should not carry a $DraftMax that goes nowhere -
    a knob that does nothing is worse than a missing one."""
    seen = []
    for i, tok in enumerate(argv):
        p = _PARAM_BY_FLAG.get(tok)
        if p and p not in _NOT_A_PARAM and p not in seen:
            seen.append(p)
    return seen


def _provenance(model_path, c, measured, backend, tmpl=(None, None),
                path_resolved=True):
    """Where these numbers came from - measured, or the planner's estimate."""
    out = ["Model    : %s" % os.path.basename(model_path or "?")]
    if not path_resolved:
        # Speed rows record the model's BASENAME, so a script built from an old
        # campaign can name a file this machine no longer has. Emitting the flags
        # anyway is right - they are the valuable part - but silently writing a
        # path that does not resolve would look like a working script.
        out.append("           !! this file was NOT found on disk. Every flag below is")
        out.append("              correct; fix the -m path before running it.")
    if backend:
        out.append("Backend  : %s (resolved fresh at launch, see below)" % backend["build"])
    if tmpl[0] or tmpl[1]:
        bits = []
        if tmpl[0]:
            bits.append(os.path.basename(tmpl[0]))
        if tmpl[1]:
            n = len(json.loads(tmpl[1]))
            bits.append("%d kwarg%s" % (n, "" if n == 1 else "s"))
        out.append("Template : %s (overrides the one baked into the GGUF)"
                   % " + ".join(bits))
    if not measured:
        out.append("Settings : PREDICTED by the planner, not measured. Run the speed")
        out.append("           sweep to replace these with numbers off your own card.")
        return out
    bits = []
    if measured.get("tok_s"):
        bits.append("%.2f tok/s decode" % measured["tok_s"])
    if measured.get("prefill_tok_s"):
        bits.append("%.0f tok/s prefill" % measured["prefill_tok_s"])
    if measured.get("proc_vram_mib"):
        bits.append("%.0f MiB VRAM" % measured["proc_vram_mib"])
    out.append("Settings : MEASURED - %s" % (", ".join(bits) or "recorded"))
    fill = (measured.get("config") or {}).get("fill") or c.get("fill")
    if fill:
        out.append("           measured at %s tokens of context filled. Decode slows as"
                   % "{:,}".format(int(fill)))
        out.append("           the context fills, so expect less than this when deep in")
        out.append("           a long conversation - the figure is a ranking, not a promise.")
    if measured.get("accept_rate") is not None and measured.get("draft_n"):
        out.append("           speculation accepted %.0f%% of %d drafted tokens"
                   % (100 * measured["accept_rate"], measured["draft_n"]))
    if measured.get("spilled"):
        out.append("           !! this row SPILLED into shared memory - it loaded, but the")
        out.append("              GPU was over-committed. Treat the speed as suspect.")
    if measured.get("when"):
        out.append("           recorded %s"
                   % datetime.datetime.fromtimestamp(measured["when"]).strftime("%Y-%m-%d"))
    return out


# ---------------------------------------------------------------------------
# The two shells
# ---------------------------------------------------------------------------
def _powershell(model_path, c, argv, backend, mmproj, sampling, port, bind_host,
                load_mode, log_dir, measured, params, tmpl, path_resolved):
    exe_name = EXE_BY_SHELL["powershell"]
    # PowerShell is case-insensitive about variables, but $model reading as $Model
    # in one place and not the other just looks like a bug to whoever edits this.
    var = lambda n: "$" + (n.lower() if n in _NOT_A_PARAM else n)
    body = _lines_from_argv(argv, var)
    has_tmpl = bool(tmpl[0] or tmpl[1])
    L = []
    A = L.append

    for line in _provenance(model_path, c, measured, backend, tmpl, path_resolved):
        A("# " + line)
    A("#")
    A("# Usage:  powershell -ExecutionPolicy Bypass -File .\\%s"
      % script_name(model_path, "powershell"))
    A("#         (every setting below is a -Parameter, e.g. -Ngl 25)")
    A("")
    A("param(")
    decl = {
        "Ngl": ("[int]", str(c.get("ngl", 0))),
        "Ctx": ("[int]", str(c.get("ctx", 4096))),
        "Ubatch": ("[int]", str(c.get("ub", 512))),
        "NCpuMoe": ("[int]", str(c.get("ncmoe") or 0)),
        "DraftMax": ("[int]", str(c.get("spec_n_max") or 0)),
        "Port": ("[int]", str(port)),
    }
    # Pad on the whole "[type]$Name" token, not on the name: the type prefixes
    # differ in length, so aligning the names alone leaves the = signs ragged.
    used = [p for p in params if p in decl]
    w = max([len(decl[p][0]) + len(p) + 1 for p in used] or [0])
    for p in used:
        t, d = decl[p]
        A("    %-*s = %s," % (w, t + "$" + p, d))
    A("")
    A("    # 127.0.0.1 is loopback-only and is NOT reachable from WSL2 - that is a")
    A("    # separate VM behind NAT, so Windows' loopback is not its loopback. Use")
    A("    # 0.0.0.0 to accept the WSL adapter (and the LAN, so pair it with a")
    A("    # firewall rule). Note: $Host is a reserved variable, hence $BindHost.")
    A("    [string]$BindHost = %s," % ps_quote(bind_host))
    A("")
    A("    # llama.cpp does NOT rotate --log-file, so one long-lived name would grow")
    A("    # without bound; a dated name per launch keeps each session greppable.")
    A("    # Pass -LogFile '' to log to the console only, which keeps nothing.")
    A("    [string]$LogFile = (Join-Path $HOME (%s -f (Get-Date -Format 'yyyyMMdd-HHmmss'))),"
      % ps_quote(os.path.join(log_dir or "llama-logs",
                             re.sub(r"^run-|\.ps1$", "", script_name(model_path, "powershell"))
                             + "-{0}.log")))
    A("")
    A("    # --no-mmap and --mlock are DEPRECATED in favour of --load-mode; passing")
    A("    # the old names is accepted and then ignored with a deprecation line,")
    A("    # which looks exactly like a setting that did nothing.")
    A("    #   none        no mmap - each tensor is read straight to its final home,")
    A("    #               so host RAM keeps only the CPU-resident blocks and not the")
    A("    #               part that already lives in VRAM. Slower first load.")
    A("    #   mmap        llama.cpp's default. Lowest RAM pressure, but the")
    A("    #               CPU-resident pages are file-backed and CAN be evicted,")
    A("    #               which is what makes decode stutter in a long session.")
    A("    #   mmap+mlock  AVOID under heavy GPU offload: mlock pins the whole mapped")
    A("    #               file, including the blocks already resident in VRAM.")
    A("    [ValidateSet(%s)]" % ",".join(ps_quote(m) for m in LOAD_MODES))
    A("    [string]$LoadMode = %s%s"
      % (ps_quote(load_mode), "," if (has_tmpl or sampling) else ""))
    if has_tmpl:
        A("")
        A("    # A chat template the GGUF's own metadata does not carry - a patched")
        A("    # tool-call template, or one from the model card that the conversion")
        A("    # predates. A path that does not exist is NOT an error to")
        A("    # llama-server: it falls back to the baked-in template silently, so")
        A("    # the check below is the only thing that would tell you.")
        A("    [string]$ChatTemplateFile = %s," % ps_quote(tmpl[0] or ""))
        A("")
        A("    # Extra variables handed to the template, as a JSON OBJECT. The key")
        A("    # names belong to the TEMPLATE, not to llama.cpp:")
        A("    # '{\"enable_thinking\":false}' is Qwen3's spelling and means nothing")
        A("    # to a model that does not use that variable - it is ignored, not")
        A("    # rejected, so a typo here is silent. These are SERVER defaults: a")
        A("    # client sending chat_template_kwargs in the request wins for that")
        A("    # request.")
        A("    [string]$ChatTemplateKwargs = %s%s"
          % (ps_quote(tmpl[1] or ""), "," if sampling else ""))
    if sampling:
        A("")
        A("    # Sampling. These are SERVER DEFAULTS - a client that sends its own")
        A("    # temperature/top_p (Open WebUI, aider, most chat UIs do) overrides")
        A("    # them per request. Passed explicitly because llama.cpp's own defaults")
        A("    # (temp 0.80, top-k 40, min-p 0.05) are not always what a model wants.")
        decls = []
        for flag, v, kind in sampling:
            name = "".join(p.capitalize() for p in flag.lstrip("-").split("-"))
            decls.append(("[%s]$%s" % (kind, name), v))
        sw = max(len(d) for d, _ in decls)
        for i, (d, v) in enumerate(decls):
            A("    %-*s = %s%s" % (sw, d, v, "," if i < len(decls) - 1 else ""))
    A(")")
    A("")
    A("$ErrorActionPreference = 'Stop'")
    A("")
    A("$model  = %s" % ps_quote(model_path))
    if mmproj:
        A("$mmproj = %s" % ps_quote(mmproj))
    A("foreach ($p in @($model%s)) {" % (", $mmproj" if mmproj else ""))
    A("    if (-not (Test-Path $p)) { throw \"Missing: $p\" }")
    A("}")
    A("")
    A("# Resolve the backend fresh every launch. A hardcoded path silently stops")
    A("# existing when LM Studio updates; matching the family wildcard and taking the")
    A("# highest version keeps working. Sorting is version-aware on purpose - by name,")
    A("# 2.9.0 sorts above 2.10.0.")
    A("$backends = %s" % ps_quote(backends_dir()))
    A("$exe = $null")
    A("if (Test-Path $backends) {")
    A("    $backend = Get-ChildItem $backends -Directory |")
    A("        Where-Object { $_.Name -like %s -and" % ps_quote(backend_match(backend["build"]) if backend else "*"))
    A("                       (Test-Path (Join-Path $_.FullName %s)) } |" % ps_quote(exe_name))
    A("        Sort-Object @{Expression = {")
    A("            if ($_.Name -match '(\\d+)\\.(\\d+)\\.(\\d+)\\s*$') {")
    A("                [version]('{0}.{1}.{2}' -f $Matches[1], $Matches[2], $Matches[3])")
    A("            } else { [version]'0.0.0' } }} |")
    A("        Select-Object -Last 1")
    A("    if ($backend) {")
    A("        $exe = Join-Path $backend.FullName %s" % ps_quote(exe_name))
    A("        # The CUDA runtime DLLs live in a SEPARATE shared vendor package, named")
    A("        # by the backend's own manifest. Without it on PATH the process dies")
    A("        # with STATUS_DLL_NOT_FOUND (exit -1073741515) and prints nothing.")
    A("        $libs = @($backend.FullName)")
    A("        $manifest = Join-Path $backend.FullName 'backend-manifest.json'")
    A("        if (Test-Path $manifest) {")
    A("            $man = Get-Content $manifest -Raw | ConvertFrom-Json")
    A("            foreach ($pkg in @($man.vendor_lib_package_names)) {")
    A("                $v = Join-Path $backends (Join-Path 'vendor' $pkg)")
    A("                if (Test-Path $v) { $libs += $v }")
    A("            }")
    A("        }")
    A("        $env:PATH = (($libs + $env:PATH) -join ';')")
    A("        Write-Host \"backend : $($backend.Name)\"")
    A("    }")
    A("}")
    A("# Fall back to a llama-server already on PATH - a self-built one, say.")
    A("if (-not $exe) {")
    A("    $onPath = Get-Command 'llama-server' -ErrorAction SilentlyContinue")
    A("    if (-not $onPath) { throw \"No llama-server found under $backends or on PATH\" }")
    A("    $exe = $onPath.Source")
    A("    Write-Host \"backend : $exe (from PATH)\"")
    A("}")
    A("")
    A("$logArgs = @()")
    A("if ($LogFile) {")
    A("    $dir = Split-Path $LogFile -Parent")
    A("    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }")
    A("    $logArgs = @('--log-file', $LogFile, '--log-timestamps')")
    A("}")
    A("")
    if has_tmpl:
        # Built at runtime rather than written into the command, because
        # `--chat-template-file ''` is an error and not a no-op: an empty
        # parameter has to produce NO flag, not an empty one.
        A("$tmplArgs = @()")
        A("if ($ChatTemplateFile -or $ChatTemplateKwargs) {")
        A("    # --jinja MUST come before the template flags. Without it, builds")
        A("    # accept only the built-in template NAMES and reject a path outright.")
        A("    # It is default-ON from ~2.28, but the backend above is resolved fresh")
        A("    # every launch, so an update in either direction can change that")
        A("    # default underneath you. Passing it explicitly costs nothing.")
        A("    $tmplArgs += '--jinja'")
        A("}")
        A("if ($ChatTemplateFile) {")
        A("    if (-not (Test-Path $ChatTemplateFile)) {")
        A("        throw \"No such chat template: $ChatTemplateFile\"")
        A("    }")
        A("    $tmplArgs += @('--chat-template-file', $ChatTemplateFile)")
        A("}")
        A("if ($ChatTemplateKwargs) {")
        A("    # Windows PowerShell 5.1 builds a native command line by joining the")
        A("    # arguments and re-quoting them, and it does NOT escape double quotes")
        A("    # already inside a value - they are simply dropped. So a perfectly")
        A("    # valid '{\"a\":1}' arrives at the exe as {a:1}, and llama-server")
        A("    # rejects it with")
        A("    #   parse error at line 1, column 2 ... last read: '{e'")
        A("    # which reads like the JSON is wrong when the JSON was never the")
        A("    # problem. Escaping them as \\\" is the fix, and it must NOT be applied")
        A("    # under PowerShell 7.3+, where the argument is passed through intact")
        A("    # and the backslashes would then arrive literally.")
        A("    $kwargs = $ChatTemplateKwargs")
        A("    if ($PSNativeCommandArgumentPassing -notin @('Standard', 'Windows')) {")
        A("        $kwargs = $kwargs -replace '\"', '\\\"'")
        A("    }")
        A("    $tmplArgs += @('--chat-template-kwargs', $kwargs)")
        A("}")
        A("")
    A("Write-Host \"model   : $(Split-Path $model -Leaf)\"")
    if has_tmpl:
        A("if ($ChatTemplateFile) { Write-Host \"template: $(Split-Path $ChatTemplateFile -Leaf)\" }")
        A("if ($ChatTemplateKwargs) { Write-Host \"tmpl-kw : $ChatTemplateKwargs\" }")
    A("Write-Host \"log     : $(if ($LogFile) { $LogFile } else { 'console only (not saved)' })\"")
    A("Write-Host \"serving : http://${BindHost}:$Port  (OpenAI-compatible at /v1)\"")
    A("if ($BindHost -eq '0.0.0.0') {")
    A("    $wsl = (Get-NetIPAddress -InterfaceAlias 'vEthernet (WSL*)' -AddressFamily IPv4 `")
    A("            -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress")
    A("    if ($wsl) { Write-Host \"from WSL: http://${wsl}:$Port/v1  (changes on reboot in NAT mode)\" }")
    A("}")
    A("Write-Host ''")
    A("")
    tail = [["--load-mode", "$LoadMode"]]
    for flag, _v, _k in sampling:
        name = "".join(w.capitalize() for w in flag.lstrip("-").split("-"))
        tail.append([flag, "$" + name])
    A("& $exe `")
    for ln in body + tail:
        A("    %s `" % " ".join(ln))
    if has_tmpl:
        A("    @tmplArgs `")
    A("    @logArgs")
    return "\n".join(L) + "\n"


def _bash(model_path, c, argv, backend, mmproj, sampling, port, bind_host,
          load_mode, log_dir, measured, params, tmpl, path_resolved):
    exe_name = EXE_BY_SHELL["bash"]
    has_tmpl = bool(tmpl[0] or tmpl[1])
    # Paths come from the host, which may be Windows. Backslashes are an escape
    # character in sh, and every shell that runs this on Windows (Git Bash, MSYS)
    # takes forward slashes anyway.
    q = lambda p: sh_quote(str(p).replace("\\", "/"))
    var = lambda n: '"$%s"' % _SNAKE(n)
    body = _lines_from_argv(argv, var)
    L = []
    A = L.append

    A("#!/usr/bin/env bash")
    A("# " + "-" * 68)
    for line in _provenance(model_path, c, measured, backend, tmpl, path_resolved):
        A("# " + line)
    A("#")
    A("# Every setting below is an environment variable override, e.g.")
    A("#   NGL=25 ./%s" % script_name(model_path, "bash"))
    A("# " + "-" * 68)
    A("set -euo pipefail")
    A("")
    defaults = {
        "NGL": c.get("ngl", 0), "CTX": c.get("ctx", 4096), "UBATCH": c.get("ub", 512),
        "NCPUMOE": c.get("ncmoe") or 0, "DRAFTMAX": c.get("spec_n_max") or 0,
        "PORT": port,
    }
    for p in params:
        n = _SNAKE(p)
        if n in defaults:
            A('%s="${%s:-%s}"' % (n, n, defaults[n]))
    A("")
    A("# 127.0.0.1 is loopback-only. Use 0.0.0.0 to accept connections from other")
    A("# hosts - pair that with a firewall rule rather than leaving it open.")
    A('BINDHOST="${BINDHOST:-%s}"' % bind_host)
    A("")
    A("# --no-mmap and --mlock are DEPRECATED in favour of --load-mode; the old names")
    A("# are accepted and then ignored, which looks like a setting that did nothing.")
    A("#   none        no mmap; host RAM keeps only the CPU-resident blocks.")
    A("#   mmap        llama.cpp's default; those pages are file-backed and CAN be")
    A("#               evicted under pressure, which makes decode stutter later.")
    A("#   mmap+mlock  AVOID under heavy GPU offload - mlock pins the whole mapped")
    A("#               file, including the blocks already resident in VRAM.")
    A('LOADMODE="${LOADMODE:-%s}"' % load_mode)
    if has_tmpl:
        A("")
        A("# A chat template the GGUF's own metadata does not carry. A path that does")
        A("# not exist is NOT an error to llama-server - it falls back to the baked-in")
        A("# template silently, so the check below is the only thing that would say so.")
        A("#")
        A("# The defaults are held in their own single-quoted variables first because")
        A("# they are not safe to inline: a template path can contain spaces and the")
        A("# kwargs default is JSON, so it is full of double quotes. Both then use")
        A("# ${VAR-default} rather than ${VAR:-default}, so exporting an EMPTY value")
        A("# turns the flag off instead of quietly restoring the default.")
        A("_tmpl_file_default=%s" % q(tmpl[0] or ""))
        A('CHATTEMPLATEFILE="${CHATTEMPLATEFILE-$_tmpl_file_default}"')
        A("")
        A("# Extra variables for the template, as a JSON OBJECT. The key names belong")
        A('# to the TEMPLATE, not to llama.cpp: \'{"enable_thinking":false}\' is Qwen3\'s')
        A("# spelling and is ignored - not rejected - by a model that does not use it,")
        A("# so a typo here is silent. SERVER defaults: a client sending")
        A("# chat_template_kwargs in the request wins for that request.")
        A("_tmpl_kwargs_default=%s" % sh_quote(tmpl[1] or ""))
        A('CHATTEMPLATEKWARGS="${CHATTEMPLATEKWARGS-$_tmpl_kwargs_default}"')
    if sampling:
        A("")
        A("# Sampling. These are SERVER DEFAULTS - a client that sends its own")
        A("# temperature/top_p overrides them per request. Passed explicitly because")
        A("# llama.cpp's defaults (temp 0.80, top-k 40, min-p 0.05) are not always")
        A("# what a given model wants.")
        for flag, v, _k in sampling:
            n = flag.lstrip("-").replace("-", "_").upper()
            A('%s="${%s:-%s}"' % (n, n, v))
    A("")
    A("# llama.cpp does not rotate --log-file, so a dated name per launch keeps each")
    A('# session bounded. Set LOGFILE="" to log to the console only, which keeps')
    A("# nothing. That is ${LOGFILE-...} and not ${LOGFILE:-...} on purpose: the")
    A("# colon form treats an explicitly-empty value as unset and would hand back")
    A("# the default, so there would be no way to turn logging off.")
    A('LOGFILE="${LOGFILE-$HOME/%s/%s-$(date +%%Y%%m%%d-%%H%%M%%S).log}"'
      % ((log_dir or "llama-logs"),
         re.sub(r"^run-|\.sh$", "", script_name(model_path, "bash"))))
    A("")
    A("MODEL=%s" % q(model_path))
    if mmproj:
        A("MMPROJ=%s" % q(mmproj))
    A('for f in "$MODEL"%s; do' % (' "$MMPROJ"' if mmproj else ""))
    A('  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }')
    A("done")
    A("")
    A("# Resolve the backend fresh every launch: a pinned path silently stops existing")
    A("# when LM Studio updates. sort -V is version-aware - by name, 2.9.0 sorts above")
    A("# 2.10.0, which would pick the older build.")
    A("BACKENDS=%s" % q(backends_dir()))
    A("EXE=\"\"")
    A('if [ -d "$BACKENDS" ]; then')
    A('  backend=$(for d in "$BACKENDS"/*/; do')
    A('      [ -x "${d}%s" ] || continue' % exe_name)
    A('      case "$(basename "$d")" in %s) printf \'%%s\\n\' "${d%%/}";; esac'
      % (backend_match(backend["build"]) if backend else "*"))
    A("    done | sort -V | tail -1)")
    A('  if [ -n "$backend" ]; then')
    A('    EXE="$backend/%s"' % exe_name)
    A("    # The GPU runtime libraries live in a SEPARATE shared vendor package. Every")
    A("    # vendor package present is added rather than only the ones this backend's")
    A("    # manifest names, so that no JSON parser is needed in a shell script; the")
    A("    # extra entries are harmless.")
    A('    libs="$backend"')
    A('    for v in "$BACKENDS"/vendor/*/; do [ -d "$v" ] && libs="$libs:${v%/}"; done')
    A('    export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"')
    A('    echo "backend : $(basename "$backend")"')
    A("  fi")
    A("fi")
    A("# Fall back to a llama-server already on PATH - a self-built one, say.")
    A('if [ -z "$EXE" ]; then')
    A('  EXE=$(command -v %s || true)' % exe_name)
    A('  [ -n "$EXE" ] || { echo "No %s under $BACKENDS or on PATH" >&2; exit 1; }' % exe_name)
    A('  echo "backend : $EXE (from PATH)"')
    A("fi")
    A("")
    A("log_args=()")
    A('if [ -n "$LOGFILE" ]; then')
    A('  mkdir -p "$(dirname "$LOGFILE")"')
    A('  log_args=(--log-file "$LOGFILE" --log-timestamps)')
    A("fi")
    A("")
    if has_tmpl:
        # Built at runtime rather than written into the command, because
        # `--chat-template-file ''` is an error and not a no-op: an empty value
        # has to produce NO flag, not an empty one.
        A("tmpl_args=()")
        A('if [ -n "$CHATTEMPLATEFILE" ] || [ -n "$CHATTEMPLATEKWARGS" ]; then')
        A("  # --jinja MUST come before the template flags. Without it, builds accept")
        A("  # only the built-in template NAMES and reject a path outright. It is")
        A("  # default-ON from ~2.28, but the backend above is resolved fresh every")
        A("  # launch, so an update in either direction can move that default")
        A("  # underneath you. Passing it explicitly costs nothing.")
        A("  tmpl_args+=(--jinja)")
        A("fi")
        A('if [ -n "$CHATTEMPLATEFILE" ]; then')
        A('  [ -f "$CHATTEMPLATEFILE" ] || {')
        A('    echo "No such chat template: $CHATTEMPLATEFILE" >&2; exit 1; }')
        A('  tmpl_args+=(--chat-template-file "$CHATTEMPLATEFILE")')
        A("fi")
        A('if [ -n "$CHATTEMPLATEKWARGS" ]; then')
        A('  tmpl_args+=(--chat-template-kwargs "$CHATTEMPLATEKWARGS")')
        A("fi")
        A("")
    A('echo "model   : $(basename "$MODEL")"')
    if has_tmpl:
        # `[ -n "$X" ] && echo ...` would be shorter and would ABORT the script
        # under `set -e` whenever X is empty, because the whole list then exits
        # non-zero. An if is the version that survives its own default.
        A('if [ -n "$CHATTEMPLATEFILE" ]; then')
        A('  echo "template: $(basename "$CHATTEMPLATEFILE")"')
        A("fi")
        A('if [ -n "$CHATTEMPLATEKWARGS" ]; then')
        A('  echo "tmpl-kw : $CHATTEMPLATEKWARGS"')
        A("fi")
    A('echo "log     : ${LOGFILE:-console only (not saved)}"')
    A('echo "serving : http://$BINDHOST:$PORT  (OpenAI-compatible at /v1)"')
    A("echo")
    A("")
    tail = [["--load-mode", '"$LOADMODE"']]
    for flag, _v, _k in sampling:
        tail.append([flag, '"$%s"' % flag.lstrip("-").replace("-", "_").upper()])
    A('exec "$EXE" \\')
    for ln in body + tail:
        A("  %s \\" % " ".join(ln))
    # Guarded expansion: under `set -u`, bash before 4.4 treats an empty array as
    # an unset variable and aborts here rather than passing nothing.
    if has_tmpl:
        A('  ${tmpl_args[@]+"${tmpl_args[@]}"} \\')
    A('  ${log_args[@]+"${log_args[@]}"}')
    return "\n".join(L) + "\n"


def _SNAKE(name):
    """PowerShell parameter name -> shell variable name. NCpuMoe -> NCPUMOE."""
    return name.upper()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def launch_script(model_path, c, backend=None, mmproj=None, shell=None,
                  sampling=None, port=8080, bind_host="127.0.0.1",
                  load_mode="none", log_dir=None, measured=None,
                  chat_template_file=None, chat_template_kwargs=None,
                  path_resolved=True):
    """The text of a launcher for one config.

    `c` is a sweep/bench config dict (ctx, kv, fa, seq, ub, ngl, and optionally
    ncmoe / spec / spec_n_max / mmproj_offload). `measured` is the speed row it
    came from, if any - used only for the provenance header, so a generated
    script says what evidence produced it. `sampling` holds only the samplers the
    caller actually chose; see sampler_args(). The chat-template pair is
    validated here rather than at launch; see template_args().

    `path_resolved=False` says the model file was not found on disk - a script
    built from an old speed row, which records a basename and not a path. The
    flags are still worth having, so it is generated and the header says so."""
    shell = shell or default_shell()
    if shell not in SHELLS:
        raise ValueError("unknown shell: %s" % shell)
    if load_mode not in LOAD_MODES:
        raise ValueError("unknown load mode: %s" % load_mode)
    if backend is None:
        backend = (find_backends() or [None])[0]

    c = dict(c or {})
    c.setdefault("ctx", 4096); c.setdefault("ub", 512); c.setdefault("seq", 1)
    c.setdefault("ngl", 0); c.setdefault("kv", "f16"); c.setdefault("fa", True)
    if mmproj:
        c["mmproj"] = mmproj
    else:
        c.pop("mmproj", None)
    mmproj = c.get("mmproj")

    # probe=False: no -v, no --cache-ram 0, no --no-warmup. Those three make a run
    # measurable and would be wrong in something you use every day.
    argv = build_argv("<exe>", model_path, c, port, probe=False, host=bind_host)[1:]
    samp = sampler_args(sampling)
    tmpl = template_args(chat_template_file, chat_template_kwargs)
    params = _params_used(argv)
    fn = _powershell if shell == "powershell" else _bash
    return fn(model_path, c, argv, backend, mmproj, samp, port, bind_host,
              load_mode, log_dir, measured, params, tmpl, path_resolved)
