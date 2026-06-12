#!/usr/bin/env bash
set -euo pipefail

# ---- Settings ----
HOST="${HOST:-127.0.0.1}"         # Bind strictly to localhost
PORT="${PORT:-8080}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
DEV_RELOAD="${DEV_RELOAD:-1}"     # 1 = uvicorn --reload, 0 = no reload
QUIET_PIP="${QUIET_PIP:-0}"       # 1 = quieter pip output

# ---- Check Python version (>= 3.9) ----
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} not found." >&2
  exit 1
fi
PY_VER="$("${PYTHON_BIN}" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
MAJ="$("${PYTHON_BIN}" -c 'import sys;print(sys.version_info.major)')"
MIN="$("${PYTHON_BIN}" -c 'import sys;print(sys.version_info.minor)')"
echo "[INFO] Python ${PY_VER} detected"
if [ "$MAJ" -lt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -lt 9 ]; }; then
  echo "[ERROR] Python >= 3.9 required." >&2
  exit 1
fi

# ---- Create/activate virtual environment ----
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# ---- Upgrade pip ----
if [[ "${QUIET_PIP}" == "1" ]]; then
  python -m pip install --upgrade pip -q
else
  python -m pip install --upgrade pip
fi

# ---- Install requirements (if present) ----
if [[ -f "${REQ_FILE}" ]]; then
  if [[ "${QUIET_PIP}" == "1" ]]; then
    pip install -r "${REQ_FILE}" -q
  else
    pip install -r "${REQ_FILE}"
  fi
fi

# ---- Optional: webrtcvad fallback wheels (non-fatal) ----
# If installation fails we proceed; app can use RMS-VAD.
if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("webrtcvad")
PY
then
  echo "[INFO] webrtcvad already installed"
else
  echo "[INFO] attempting install: webrtcvad-wheels"
  if ! pip install --no-cache-dir webrtcvad-wheels >/dev/null 2>&1; then
    echo "[WARN] webrtcvad-wheels install failed – continuing without WebRTC VAD."
  fi
fi

# ---- pywhispercpp (if not already available) ----
if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("pywhispercpp")
PY
then
  echo "[INFO] pywhispercpp already installed"
else
  echo "[INFO] installing pywhispercpp (prefer wheels)"
  if ! pip install --no-cache-dir pywhispercpp; then
    echo "[ERROR] pywhispercpp could not be installed. Please verify platform/wheels."
    echo "        See: https://github.com/absadiki/pywhispercpp"
    exit 2
  fi
fi

# ---- Ensure output directory exists ----
mkdir -p outputs/images

# ---- Preflight: local port checks via Python sockets (no /dev/tcp needed) ----
function check_local() {
  local name="$1" host="$2" port="$3"
  if python - <<PY >/dev/null 2>&1
import socket
s=socket.socket()
s.settimeout(1.0)
ok=False
try:
    s.connect(("${host}", int(${port})))
    ok=True
except Exception:
    ok=False
finally:
    s.close()
print("ok" if ok else "no")
PY
  then
    echo "[OK] ${name} reachable at ${host}:${port}"
  else
    echo "[WARN] ${name} NOT reachable at ${host}:${port} – continuing anyway."
  fi
}

# Ollama & ComfyUI hosts/ports, matching app defaults
OLLAMA_HOST="${APP_OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${APP_OLLAMA_PORT:-11434}"
COMFY_HOST="${APP_COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${APP_COMFY_PORT:-8188}"

check_local "Ollama" "${OLLAMA_HOST}" "${OLLAMA_PORT}"
check_local "ComfyUI" "${COMFY_HOST}" "${COMFY_PORT}"

# ---- Start Uvicorn ----
EXTRA_FLAGS=()
if [[ "${DEV_RELOAD}" == "1" ]]; then
  EXTRA_FLAGS+=(--reload)
fi

echo "Starting server at http://${HOST}:${PORT}"
exec uvicorn app:app --host "${HOST}" --port "${PORT}" "${EXTRA_FLAGS[@]}"
