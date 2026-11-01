#!/usr/bin/env bash
set -e

# Run from project root
cd "$(dirname "$0")"

# Load .env if present
if [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=".:${PYTHONPATH}"
exec python3 scripts/runpod_l40s_watch.py "$@"
