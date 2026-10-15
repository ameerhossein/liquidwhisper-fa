#!/usr/bin/env bash
set -e

# Activate virtualenv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Launching Persian ASR LoRA Fine-Tuning (BuzzASR + Neyshekar)"
echo " Tuned for NVIDIA L40S (BF16, batch 16, grad accum 2)"
echo " Default output: /workspace/stt-outputs/... on RunPod, else outputs/..."
echo " Override: ASR_OUTPUT_DIR=/path/to/dir ./run_train.sh ..."
echo "   or: ./run_train.sh --output_dir /path/to/dir"
echo " Fresh start (clears prior checkpoints): ./run_train.sh --fresh_start"
echo "=========================================================="

# Persistent-volume-aware default (env wins, then /workspace, then local).
# Explicit --output_dir in "$@" still wins (argparse: last flag wins).
DEFAULT_OUTPUT_DIR="${ASR_OUTPUT_DIR:-}"
if [ -z "${DEFAULT_OUTPUT_DIR}" ]; then
    if [ -d "/workspace" ]; then
        DEFAULT_OUTPUT_DIR="/workspace/stt-outputs/buzzasr-neyshekar-lora"
    else
        DEFAULT_OUTPUT_DIR="outputs/buzzasr-neyshekar-lora"
    fi
fi

python scripts/train_lora.py \
    --base_model "BuzzASR/persian" \
    --dataset_name "shekar-ai/Neyshekar" \
    --output_dir "${DEFAULT_OUTPUT_DIR}" \
    --lora_r 64 \
    --lora_alpha 128 \
    --batch_size 16 \
    --grad_accum 2 \
    --learning_rate 1e-4 \
    --epochs 4 \
    --warmup_steps 500 \
    --eval_steps 500 \
    --save_steps 500 \
    --num_proc 4 \
    "$@"
