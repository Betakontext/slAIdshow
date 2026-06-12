#!/usr/bin/env pwsh
param(
  [string]$HostIP = $(if ($env:HOST) { $env:HOST } else { "127.0.0.1" }),
  [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8080 }),
  [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
  [string]$VenvDir = $(if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }),
  [string]$ReqFile = $(if ($env:REQ_FILE) { $env:REQ_FILE } else { "requirements.txt" }),
  [int]$DevReload = $(if ($env:DEV_RELOAD) { [int]$env:DEV_RELOAD } else { 1 }),
  [int]$QuietPip = $(if ($env:QUIET_PIP) { [int]$env:QUIET_PIP } else { 0 })
)

$ErrorActionPreference = "Stop"

# ---- Check Python version (>= 3.9) ----
try {
  $ver = & $PythonBin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  Write-Host "[INFO] Python $ver detected"
  $maj = [int]$ver.Split('.')[0]
  $min = [int]$ver.Split('.')[1]
  if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 9)) {
    throw "Python >= 3.9 required"
  }
} catch {
  Write-Error "[ERROR] Python not found or unsuitable: $_"
  exit 1
}

# ---- Create/activate virtual environment ----
if (-not (Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}
$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  Write-Error "[ERROR] Could not activate venv ($activate missing)."
  exit 1
}
. $activate

# ---- Upgrade pip ----
if ($QuietPip -eq 1) {
  python -m pip install --upgrade pip -q
} else {
  python -m pip install --upgrade pip
}

# ---- Install requirements (if present) ----
if (Test-Path $ReqFile) {
  if ($QuietPip -eq 1) {
    pip install -r $ReqFile -q
  } else {
    pip install -r $ReqFile
  }
}

# ---- Optional: webrtcvad wheels (non-fatal) ----
# If install fails, we continue; app can rely on RMS-based VAD.
$hasVad = python - << 'PY'
import importlib, sys
try:
    importlib.import_module("webrtcvad")
    print("yes")
except Exception:
    print("no")
PY
if (-not $hasVad.Trim().Equals("yes")) {
  Write-Host "[INFO] attempting install: webrtcvad-wheels"
  try { pip install --no-cache-dir webrtcvad-wheels | Out-Null } catch { Write-Warning "webrtcvad-wheels failed – continuing without WebRTC VAD." }
} else {
  Write-Host "[INFO] webrtcvad already installed"
}

# ---- pywhispercpp (if not already available) ----
$hasWhisper = python - << 'PY'
import importlib
try:
    importlib.import_module("pywhispercpp")
    print("yes")
except Exception:
    print("no")
PY
if (-not $hasWhisper.Trim().Equals("yes")) {
  Write-Host "[INFO] installing pywhispercpp"
  try { pip install --no-cache-dir pywhispercpp } catch {
    Write-Error "pywhispercpp could not be installed. See https://github.com/absadiki/pywhispercpp"
    exit 2
  }
} else {
  Write-Host "[INFO] pywhispercpp already installed"
}

# ---- Ensure output directory exists ----
$null = New-Item -ItemType Directory -Force -Path "outputs\images"

# ---- Preflight: local port checks via TcpClient ----
function Test-Port {
  param([string]$Name,[string]$Host,[int]$Port)
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($Host, $Port, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(1000, $false)
    if ($ok -and $client.Connected) {
      $client.EndConnect($iar)
      $client.Close()
      Write-Host "[OK] $Name reachable at $Host`:$Port"
    } else {
      $client.Close()
      Write-Warning "[WARN] $Name NOT reachable at $Host`:$Port – continuing anyway."
    }
  } catch {
    Write-Warning "[WARN] $Name NOT reachable at $Host`:$Port – continuing anyway."
  }
}

# Hosts/ports aligned with app defaults
$OllamaHost = if ($env:APP_OLLAMA_HOST) { $env:APP_OLLAMA_HOST } else { "127.0.0.1" }
$OllamaPort = if ($env:APP_OLLAMA_PORT) { [int]$env:APP_OLLAMA_PORT } else { 11434 }
$ComfyHost  = if ($env:APP_COMFY_HOST)  { $env:APP_COMFY_HOST } else { "127.0.0.1" }
$ComfyPort  = if ($env:APP_COMFY_PORT)  { [int]$env:APP_COMFY_PORT } else { 8188 }

Test-Port -Name "Ollama" -Host $OllamaHost -Port $OllamaPort
Test-Port -Name "ComfyUI" -Host $ComfyHost -Port $ComfyPort

# ---- Start Uvicorn ----
$extra = @()
if ($DevReload -eq 1) { $extra += "--reload" }

Write-Host "Starting server at http://$HostIP`:$Port"
python -m uvicorn app:app --host $HostIP --port $Port @extra
