#!/usr/bin/env pwsh
<#
Launcher for slAUdshow (Windows PowerShell)
- Localhost only
- Project virtual environment
- Requirements installation
- Optional webrtcvad-wheels, required pywhispercpp
- Preflight checks for Ollama and ComfyUI
- Start FastAPI app via uvicorn (app:app)
#>

param(
  # Server bind (localhost only for privacy)
  [string]$HostIP    = $(if ($env:HOST) { $env:HOST } else { "127.0.0.1" }),
  [int]$Port         = $(if ($env:PORT) { [int]$env:PORT } else { 8080 }),

  # Python and venv configuration
  [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
  [string]$VenvDir   = $(if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }),
  [string]$ReqFile   = $(if ($env:REQ_FILE) { $env:REQ_FILE } else { "requirements.txt" }),

  # Uvicorn reload for development and pip quiet mode
  [int]$DevReload    = $(if ($env:DEV_RELOAD) { [int]$env:DEV_RELOAD } else { 1 }),
  [int]$QuietPip     = $(if ($env:QUIET_PIP) { [int]$env:QUIET_PIP } else { 0 })
)

$ErrorActionPreference = "Stop"

# Simple log helpers for consistent colored output
function Write-Info([string]$msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err([string]$msg)  { Write-Error $msg }
function Write-Wrn([string]$msg)  { Write-Warning $msg }

# Helper: run small Python snippets by piping code on stdin (avoids shell quoting issues)
function PyRunStdin([string]$code) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $PythonBin
  $psi.Arguments = "-"
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $proc = New-Object System.Diagnostics.Process
  $proc.StartInfo = $psi
  [void]$proc.Start()
  $proc.StandardInput.WriteLine($code)
  $proc.StandardInput.Close()
  $out = $proc.StandardOutput.ReadToEnd()
  $err = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  if ($proc.ExitCode -ne 0) {
    throw "python - (stdin) failed: $err"
  }
  return $out.Trim()
}

# Enforce localhost binding for privacy and GDPR compliance
if ($HostIP -ne "127.0.0.1" -and $HostIP -ne "localhost") {
  Write-Err ("For privacy, HostIP must be 127.0.0.1 or localhost. Given: {0}" -f $HostIP)
  exit 1
}

# Check Python availability and version (>= 3.9)
try {
  $ver = PyRunStdin @'
import sys
print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))
'@
  if (-not $ver) { throw "Python version detection returned empty output." }
  Write-Info ("Python {0} detected via '{1}'" -f $ver, $PythonBin)
  $maj = [int]$ver.Split('.')[0]; $min = [int]$ver.Split('.')[1]
  if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 9)) { throw "Python >= 3.9 required" }
} catch {
  Write-Err ("Python not found or unsuitable: {0}" -f $_)
  exit 1
}

# Detect if the current Python is already inside the desired venv
$alreadyInVenv = $false
try {
  $pyinfo = PyRunStdin @'
import sys
print(sys.executable)
print(sys.prefix)
print(sys.base_prefix)
'@
  $lines = $pyinfo -split "`r?`n"
  $exePath = $lines[0]; $sysPrefix = $lines[1]; $basePrefix = $lines[2]
  if ($sysPrefix -ne $basePrefix -and ($exePath -like "*$VenvDir*")) {
    $alreadyInVenv = $true
    Write-Info ("Existing venv detected: {0}" -f $exePath)
  }
} catch {
  # Non-fatal; continue with venv creation/activation below
}

# Create and activate virtual environment if not already active
if (-not $alreadyInVenv) {
  if (-not (Test-Path $VenvDir)) {
    Write-Info ("Creating venv at '{0}'..." -f $VenvDir)
    & $PythonBin -m venv $VenvDir
  } else {
    Write-Info ("Using existing venv at '{0}'" -f $VenvDir)
  }
  $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
  if (-not (Test-Path $activate)) {
    Write-Err ("Could not activate venv (missing {0})." -f $activate)
    exit 1
  }
  # Dot-source the venv activation for the current PowerShell session
  . $activate
  $venvPy = PyRunStdin @'
import sys
print(sys.executable)
'@
  Write-Ok ("Activated venv: {0}" -f $venvPy)
} else {
  Write-Ok "Keeping current active venv."
}

# Upgrade pip inside the venv, optionally in quiet mode
if ($QuietPip -eq 1) {
  python -m pip install --upgrade pip -q
} else {
  python -m pip install --upgrade pip
}

# Install base requirements if a requirements.txt is present
if (Test-Path $ReqFile) {
  Write-Info ("Installing requirements from {0}..." -f $ReqFile)
  if ($QuietPip -eq 1) {
    pip install -r $ReqFile -q
  } else {
    pip install -r $ReqFile
  }
} else {
  Write-Wrn ("No {0} found; skipping base dependency install." -f $ReqFile)
}

# Optional: attempt to install webrtcvad wheels (not critical; used for VAD speed)
try {
  $hasVad = PyRunStdin @'
import importlib.util
print('yes' if importlib.util.find_spec('webrtcvad') else 'no')
'@
} catch { $hasVad = "no" }
if ($hasVad -ne "yes") {
  Write-Info "Attempting install: webrtcvad-wheels (optional)"
  try {
    pip install --no-cache-dir webrtcvad-wheels | Out-Null
  } catch {
    Write-Wrn "webrtcvad-wheels failed - continuing without WebRTC VAD."
  }
} else {
  Write-Info "webrtcvad already installed"
}

# Ensure pywhispercpp is available (required for realtime transcription)
try {
  $hasWhisper = PyRunStdin @'
import importlib.util
print('yes' if importlib.util.find_spec('pywhispercpp') else 'no')
'@
} catch { $hasWhisper = "no" }
if ($hasWhisper -ne "yes") {
  Write-Info "Installing pywhispercpp..."
  try {
    pip install --no-cache-dir pywhispercpp
  } catch {
    Write-Err "pywhispercpp could not be installed. See https://github.com/absadiki/pywhispercpp"
    exit 2
  }
} else {
  Write-Info "pywhispercpp already installed"
}

# Ensure outputs directory exists for generated images
try {
  $null = New-Item -ItemType Directory -Force -Path "outputs\images" | Out-Null
} catch {
  # Non-fatal, continue
}

# Port reachability test without colliding with PowerShell's $Host variable
function Test-Port {
  param(
    [string]$Name,
    [string]$TargetHost,
    [int]$TargetPort
  )
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($TargetHost, $TargetPort, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(1000, $false)
    if ($ok -and $client.Connected) {
      $client.EndConnect($iar); $client.Close()
      Write-Ok ("{0} reachable at {1}:{2}" -f $Name, $TargetHost, $TargetPort)
      return $true
    } else {
      $client.Close()
      Write-Wrn ("{0} NOT reachable at {1}:{2} - continuing anyway." -f $Name, $TargetHost, $TargetPort)
      return $false
    }
  } catch {
    Write-Wrn ("{0} NOT reachable at {1}:{2} - continuing anyway." -f $Name, $TargetHost, $TargetPort)
    return $false
  }
}

# Resolve local service endpoints (kept to 127.0.0.1 by default)
$OllamaHost = if ($env:APP_OLLAMA_HOST) { $env:APP_OLLAMA_HOST } else { "127.0.0.1" }
$OllamaPort = if ($env:APP_OLLAMA_PORT) { [int]$env:APP_OLLAMA_PORT } else { 11434 }
$ComfyHost  = if ($env:APP_COMFY_HOST)  { $env:APP_COMFY_HOST } else { "127.0.0.1" }
$ComfyPort  = if ($env:APP_COMFY_PORT)  { [int]$env:APP_COMFY_PORT } else { 8188 }

# Non-fatal preflight checks: helpful hints if services are not yet up
$ollamaUp = Test-Port -Name "Ollama" -TargetHost $OllamaHost -TargetPort $OllamaPort
$comfyUp  = Test-Port -Name "ComfyUI" -TargetHost $ComfyHost  -TargetPort $ComfyPort

if (-not $ollamaUp) {
  Write-Wrn "Tip: Start Ollama first (ollama serve) and pull a small model, e.g., 'ollama pull llama3.2:3b'."
}
if (-not $comfyUp) {
  Write-Wrn "Tip: Start ComfyUI: python main.py --listen 127.0.0.1 --port 8188 --lowvram"
}

# Start FastAPI via uvicorn on localhost
$extra = @()
if ($DevReload -eq 1) { $extra += "--reload" }

Write-Host ("Starting server at http://{0}:{1}" -f $HostIP, $Port)
python -m uvicorn app:app --host $HostIP --port $Port @extra
