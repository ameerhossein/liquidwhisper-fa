#!/usr/bin/env bash
set -e

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
