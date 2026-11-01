#!/usr/bin/env bash
set -e

# Load .env if present
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH=".:${PYTHONPATH}"
exec python3 scripts/runpod_l40s_watch.py "$@"
