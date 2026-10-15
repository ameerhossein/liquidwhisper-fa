#!/usr/bin/env bash
set -e

# Load .env if present (repo root next to this script, else CWD).
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# RunPod default proxy port is 7860 or 8000
PORT=${PORT:-7860}
HOST=${HOST:-0.0.0.0}

echo "=========================================================="
echo " Launching Persian ASR RunPod Cockpit"
echo " Web UI & Live Telemetry Server"
echo " Host: ${HOST} | Port: ${PORT}"
echo "=========================================================="

python3 -m uvicorn server.app:app --host "${HOST}" --port "${PORT}"
