#!/usr/bin/env bash
set -e

# Activate virtualenv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Launching Persian ASR LoRA Fine-Tuning (BuzzASR + Neyshekar)"
echo " Tuned for NVIDIA L40S (BF16, batch 16, grad accum 2)"
echo "=========================================================="

python scripts/train_lora.py \
    --base_model "BuzzASR/persian" \
    --dataset_name "shekar-ai/Neyshekar" \
    --output_dir "outputs/buzzasr-neyshekar-lora" \
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
