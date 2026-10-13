#!/usr/bin/env bash
set -e

PORT=${PORT:-7860}

echo "=========================================================="
echo " Launching Persian ASR Local Web UI Demo"
echo " (Runs on Standard Python 3 - Zero GPU/Pip required)"
echo " Open in your browser: http://127.0.0.1:${PORT}"
echo "=========================================================="

python3 scripts/demo_local.py
