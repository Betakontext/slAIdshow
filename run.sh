#!/usr/bin/env bash
set -euo pipefail

# Nur lokal binden
HOST="127.0.0.1"
PORT="${PORT:-8080}"

# .env laden, falls vorhanden
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

# venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Abhängigkeiten
pip install --upgrade pip
pip install -r requirements.txt

# Ordnerstruktur
mkdir -p outputs/images

echo "Starte Server auf http://${HOST}:${PORT}"
exec uvicorn app:app --host "${HOST}" --port "${PORT}" --reload
