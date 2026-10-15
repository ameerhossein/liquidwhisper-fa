#!/usr/bin/env bash
set -e

# Load .env if present (repo root next to this script, else CWD).
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

PORT=${PORT:-7860}

echo "=========================================================="
echo " Launching Persian ASR Local Web UI Demo"
echo " (Runs on Standard Python 3 - Zero GPU/Pip required)"
echo " Open in your browser: http://127.0.0.1:${PORT}"
echo "=========================================================="

python3 scripts/demo_local.py
