#!/usr/bin/env bash
set -euo pipefail

# ---- Einstellungen ----
HOST="${HOST:-127.0.0.1}"         # Nur lokal binden
PORT="${PORT:-8080}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
DEV_RELOAD="${DEV_RELOAD:-1}"     # 1 = uvicorn --reload, 0 = ohne reload
QUIET_PIP="${QUIET_PIP:-0}"       # 1 = weniger Pip-Ausgabe

# ---- .env laden (falls vorhanden) ----
# if [[ -f ".env" ]]; then
#   set -a
  # shellcheck disable=SC1090
#  . "./.env"
#  set +a
# fi

# ---- Python-Version prüfen (mindestens 3.9 empfohlen) ----
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} nicht gefunden." >&2
  exit 1
fi
PY_VER=$("${PYTHON_BIN}" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python ${PY_VER} erkannt"

# ---- venv anlegen/aktivieren ----
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# ---- Pip aktualisieren ----
if [[ "${QUIET_PIP}" == "1" ]]; then
  python -m pip install --upgrade pip -q
else
  python -m pip install --upgrade pip
fi

# ---- Requirements installieren (falls vorhanden) ----
if [[ -f "${REQ_FILE}" ]]; then
  if [[ "${QUIET_PIP}" == "1" ]]; then
    pip install -r "${REQ_FILE}" -q
  else
    pip install -r "${REQ_FILE}"
  fi
fi

# ---- Optional: webrtcvad Wheels (Fallback-freundlich) ----
# Wenn Installation scheitert, einfach weiter (wir nutzen RMS-VAD in der App)
if python -c "import webrtcvad" >/dev/null 2>&1; then
  echo "[INFO] webrtcvad bereits installiert"
else
  echo "[INFO] versuche Installation: webrtcvad-wheels"
  if ! pip install --no-cache-dir webrtcvad-wheels >/dev/null 2>&1; then
    echo "[WARN] webrtcvad-wheels Installation fehlgeschlagen – fahre ohne WebRTC VAD fort."
  fi
fi

# ---- pywhispercpp (falls nicht schon in requirements) ----
if python -c "import pywhispercpp" >/dev/null 2>&1; then
  echo "[INFO] pywhispercpp bereits installiert"
else
  echo "[INFO] installiere pywhispercpp (Wheel bevorzugt)"
  if ! pip install --no-cache-dir pywhispercpp; then
    echo "[ERROR] pywhispercpp konnte nicht installiert werden. Bitte prüfe Plattform/Wheels."
    echo "        Siehe: https://github.com/absadiki/pywhispercpp"
    exit 2
  fi
fi

# ---- Ausgabeverzeichnis anlegen ----
mkdir -p outputs/images

# ---- Preflight-Checks (optional, ohne Abbruch) ----
function check_local() {
  local name="$1" host="$2" port="$3"
  if timeout 1 bash -c "</dev/tcp/${host}/${port}" 2>/dev/null; then
    echo "[OK] ${name} erreichbar auf ${host}:${port}"
  else
    echo "[WARN] ${name} NICHT erreichbar auf ${host}:${port} – die App startet trotzdem."
  fi
}
# Ollama und ComfyUI Host/Port aus ENV (Default wie in app.py)
OLLAMA_HOST="${APP_OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${APP_OLLAMA_PORT:-11434}"
COMFY_HOST="${APP_COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${APP_COMFY_PORT:-8188}"

check_local "Ollama" "${OLLAMA_HOST}" "${OLLAMA_PORT}"
check_local "ComfyUI" "${COMFY_HOST}" "${COMFY_PORT}"

# ---- Uvicorn Start ----
EXTRA_FLAGS=()
if [[ "${DEV_RELOAD}" == "1" ]]; then
  EXTRA_FLAGS+=(--reload)
fi

echo "Starte Server auf http://${HOST}:${PORT}"
exec uvicorn app:app --host "${HOST}" --port "${PORT}" "${EXTRA_FLAGS[@]}"
