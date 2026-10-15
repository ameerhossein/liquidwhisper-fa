#!/usr/bin/env bash
set -e

# Load .env if present (repo root next to this script, else CWD).
if [ -f "$(dirname "$0")/.env" ]; then
    set -a; . "$(dirname "$0")/.env"; set +a
elif [ -f ".env" ]; then
    set -a; . "./.env"; set +a
fi

# Activate virtualenv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Running Baseline Benchmark on Neyshekar Test Split"
echo " Comparing: openai/whisper-large-v3 vs. BuzzASR/persian"
echo "=========================================================="

# Pass optional arguments like --max_samples 100 for smoke testing
python scripts/benchmark_baselines.py \
    --models "openai/whisper-large-v3" "BuzzASR/persian" \
    --dataset_name "shekar-ai/Neyshekar" \
    --split "test" \
    --batch_size 16 \
    --output_dir "results/baselines" \
    "$@"
