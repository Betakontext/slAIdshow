#!/usr/bin/env bash
set -euo pipefail

# =========================
# slAIdshow bootstrap script (project-folder agnostic)
# =========================

# ---- Settings (overridable via env) ----
HOST="${HOST:-127.0.0.1}"          # Bind strictly to localhost
PORT="${PORT:-8080}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
DEV_RELOAD="${DEV_RELOAD:-1}"      # 1 = uvicorn --reload
QUIET_PIP="${QUIET_PIP:-0}"        # 1 = quieter pip output
AUTO_APT="${AUTO_APT:-1}"          # 1 = try to install OS prereqs on Debian/Ubuntu

# ---- Normalize working directory to the script location ----
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

log()  { echo -e "[INFO] $*"; }
warn() { echo -e "[WARN] $*" >&2; }
err()  { echo -e "[ERROR] $*" >&2; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

is_debian_like() {
  if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    case "${ID:-}${ID_LIKE:-}" in
      *debian*|*ubuntu*|*linuxmint*|*elementary*|*pop*|*neon*) return 0 ;;
    esac
  fi
  return 1
}

maybe_install_apt_prereqs() {
  if [[ "${AUTO_APT}" != "1" ]]; then
    log "Skipping OS prereqs (AUTO_APT=0)."
    return 0
  fi
  if ! is_debian_like; then
    log "Non-Debian/Ubuntu system – skipping apt prereqs."
    return 0
  fi
  if ! have_cmd sudo || ! have_cmd apt; then
    warn "sudo/apt not available – skipping apt prereqs."
    return 0
  fi
  log "Installing Debian/Ubuntu prereqs (requires sudo)..."
  set +e
  sudo apt update
  sudo apt install -y build-essential cmake pkg-config python3-dev libportaudio2 libasound2-dev
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    warn "apt prereqs installation failed (code $rc). Continuing anyway."
  else
    log "apt prereqs OK."
  fi
}

# Unset any pre-existing venv to avoid PATH pollution from other projects
unset VIRTUAL_ENV || true

# --- Validate or create venv ---
create_venv() {
  local sys_py=""
  if have_cmd python3; then
    sys_py="$(command -v python3)"
  elif have_cmd python; then
    sys_py="$(command -v python)"
  else
    err "Kein Python 3 gefunden. Installiere python3 (z.B. 'sudo apt install -y python3')."
    exit 1
  fi
  log "Creating virtual environment at ${VENV_DIR} using ${sys_py}"
  "${sys_py}" -m venv "${VENV_DIR}"
}

validate_or_recreate_venv() {
  # Fall 1: venv fehlt → erstellen
  if [[ ! -d "${VENV_DIR}" ]]; then
    create_venv
    return
  fi

  # Fall 2: venv existiert, prüfe .venv/bin/python
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    warn "Existing venv is missing bin/python → recreating."
    rm -rf "${VENV_DIR}"
    create_venv
    return
  fi

  # Fall 3: Prüfe, ob Pip/Shebang auf einen falschen Pfad zeigt
  if [[ -f "${VENV_DIR}/bin/pip" ]]; then
    if head -n1 "${VENV_DIR}/bin/pip" | grep -q "speechtoimage_ai/.venv/bin/python"; then
      warn "Detected stale shebang in pip (points to old project path). Recreating venv."
      rm -rf "${VENV_DIR}"
      create_venv
      return
    fi
  fi

  # Fall 4: Zusätzlicher Integritätscheck: Python aus venv starten
  if ! "${VENV_DIR}/bin/python" -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
    warn "Venv python is not runnable → recreating."
    rm -rf "${VENV_DIR}"
    create_venv
    return
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  echo "[DEBUG] which python after activate: $(command -v python || echo 'not found')"
  echo "[DEBUG] PATH: $PATH"
}

ensure_requirements() {
  local PY_BIN="${VENV_DIR}/bin/python"
  local PIP_BIN="${VENV_DIR}/bin/pip"

  local quiet_flag=()
  if [[ "${QUIET_PIP}" == "1" ]]; then
    quiet_flag+=(-q)
  fi

  "${PY_BIN}" -m pip install --upgrade pip setuptools wheel "${quiet_flag[@]}"

  if [[ -f "${REQ_FILE}" ]]; then
    log "Installing Python requirements from ${REQ_FILE}"
    "${PIP_BIN}" install -r "${REQ_FILE}" "${quiet_flag[@]}"
  else
    warn "No requirements.txt found – installing minimal core deps"
    "${PIP_BIN}" install httpx pydantic fastapi 'uvicorn[standard]' "${quiet_flag[@]}"
  fi
}

ensure_webrtcvad() {
  local PY_BIN="${VENV_DIR}/bin/python"
  local PIP_BIN="${VENV_DIR}/bin/pip"
  if "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import importlib; importlib.import_module("webrtcvad")
PY
  then
    log "webrtcvad already present"
  else
    log "Attempting optional install: webrtcvad-wheels"
    if ! "${PIP_BIN}" install --no-cache-dir webrtcvad-wheels >/dev/null 2>&1; then
      warn "webrtcvad-wheels install failed – continuing without WebRTC VAD."
    fi
  fi
}

ensure_pywhispercpp() {
  local PY_BIN="${VENV_DIR}/bin/python"
  local PIP_BIN="${VENV_DIR}/bin/pip"
  if "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import importlib; importlib.import_module("pywhispercpp")
PY
  then
    log "pywhispercpp already present"
  else
    log "Installing pywhispercpp (prefer wheels)"
    if ! "${PIP_BIN}" install --no-cache-dir pywhispercpp; then
      err "pywhispercpp could not be installed. Please verify platform/wheels."
      err "See: https://github.com/absadiki/pywhispercpp"
      exit 2
    fi
  fi
}

check_local() {
  local name="$1" host="$2" port="$3"
  local PY_BIN="${VENV_DIR}/bin/python"
  if "${PY_BIN}" - <<PY >/dev/null 2>&1
import socket
h="${host}"; p=int("${port}")
s=socket.socket(); s.settimeout(1.0)
ok=False
try:
    s.connect((h,p)); ok=True
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

main() {
  maybe_install_apt_prereqs

  validate_or_recreate_venv
  activate_venv

  # Explizit Python-Version aus venv prüfen
  local PY_BIN="${VENV_DIR}/bin/python"
  local VER MAJ MIN
  VER="$("${PY_BIN}" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  MAJ="$("${PY_BIN}" -c 'import sys;print(sys.version_info.major)')"
  MIN="$("${PY_BIN}" -c 'import sys;print(sys.version_info.minor)')"
  echo "[INFO] Python ${VER} (venv) detected"
  if [ "$MAJ" -lt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -lt 9 ]; }; then
    err "Python >= 3.9 required."
    exit 1
  fi

  ensure_requirements
  ensure_webrtcvad
  ensure_pywhispercpp

  mkdir -p outputs/images

  local OLLAMA_HOST="${APP_OLLAMA_HOST:-127.0.0.1}"
  local OLLAMA_PORT="${APP_OLLAMA_PORT:-11434}"
  local COMFY_HOST="${APP_COMFY_HOST:-127.0.0.1}"
  local COMFY_PORT="${APP_COMFY_PORT:-8188}"
  check_local "Ollama" "${OLLAMA_HOST}" "${OLLAMA_PORT}"
  check_local "ComfyUI" "${COMFY_HOST}" "${COMFY_PORT}"

  local EXTRA_FLAGS=()
  if [[ "${DEV_RELOAD}" == "1" ]]; then
    EXTRA_FLAGS+=(--reload)
  fi

  log "Starting server at http://${HOST}:${PORT}"
  exec "${VENV_DIR}/bin/uvicorn" app:app --host "${HOST}" --port "${PORT}" "${EXTRA_FLAGS[@]}"
}

main "$@"
