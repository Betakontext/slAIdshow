#!/usr/bin/env bash
set -euo pipefail

# Nur lokal binden
HOST="127.0.0.1"
PORT="${PORT:-8080}"

# .env laden, falls vorhanden (macht ENV-Variablen global verfügbar)
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

# venv anlegen/aktivieren
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Python-Pip updaten + Basispakete installieren
pip install --upgrade pip
pip install -r requirements.txt

# Zusätzliche Wheels, wie in deinem Setup
# webrtcvad: Wheel-Paket statt Source-Build
pip uninstall -y webrtcvad || true
pip install --no-cache-dir webrtcvad-wheels

# pywhispercpp: passendes Wheel (CPU/GPU-Variante je nach Plattform)
pip install --no-cache-dir pywhispercpp

# Ordnerstruktur für Ausgaben
mkdir -p outputs/images

echo "Starte Server auf http://${HOST}:${PORT}"
exec uvicorn app:app --host "${HOST}" --port "${PORT}" --reload
