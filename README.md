# High-Precision Persian Speech-to-Text (ASR) Pipeline

Fine-tuning and evaluation suite designed to overcome high Word Error Rates (WER) in Persian ASR. Built around **`BuzzASR/persian`** (Whisper Large v3 architecture with a native monolingual Persian BPE tokenizer) and fine-tuned on the **`shekar-ai/Neyshekar`** (v6) corpus.

Optimized for **NVIDIA L40S (48 GB VRAM)** using unquantized **Bfloat16 LoRA (PEFT)**.

---

## Table of Contents
1. [⚡ Quickstart: Local UI Demo (No GPU Needed)](#-quickstart-local-ui-demo-no-gpu-needed)
2. [Why This Architecture?](#why-this-architecture)
3. [Project Structure](#project-structure)
4. [Transferring to the GPU Server](#transferring-to-the-gpu-server)
5. [Environment Setup on GPU Server](#environment-setup-on-gpu-server)
6. [Step 1: Baseline Benchmarking](#step-1-baseline-benchmarking)
7. [Step 2: Training on NVIDIA L40S](#step-2-training-on-nvidia-l40s)
8. [Step 3: Post-Training Evaluation & Diagnostics](#step-3-post-training-evaluation--diagnostics)
9. [Step 4: Merging Adapter Weights for Deployment](#step-4-merging-adapter-weights-for-deployment)
10. [Step 5: Transcribing Custom Audio (CLI)](#step-5-transcribing-custom-audio-cli)

---

## ⚡ Quickstart: Local UI Demo (No GPU Needed)

You can launch and explore the full Web Cockpit right now on your local machine using standard Python 3:

```bash
./run_local_demo.sh
```

Then open `http://127.0.0.1:7860` in your browser. You can click **Launch Run** to simulate active Persian ASR training, watch the live loss trajectory, view simulated epoch benchmarks updating in real time, and inspect the canary Persian sentences.

---

## Why This Architecture?

* **Native Persian Tokenizer:** OpenAI's standard Whisper Large v3 uses a multilingual byte-level BPE tokenizer, taking 5–12 tokens per Persian word. `BuzzASR/persian` replaces this with a Persian BPE tokenizer (~3.3x compression), preventing looping hallucinations and speeding up generation.
* **Neyshekar Dataset (v5/v6):** Provides ~80–100 hours of curated native Persian speech with strict **speaker-disjoint** and **text-disjoint** splits, containing both formal Persian and ~25–35% colloquial/informal Persian.
* **Full BF16 LoRA on L40S:** With 48 GB VRAM, training runs in full 16-bit precision without 4-bit quantization latency, taking ~3 to 5 hours for 4 epochs with fused AdamW and SDPA attention.

---

## Project Structure

```
.
├── configs/
│   ├── l40s_train_config.yaml     # Recommended training hyperparameters for L40S
│   └── eval_config.yaml           # Benchmark settings
├── src/
│   ├── normalizer.py              # Persian text normalizer (Hazm, ZWNJ, Arabic-to-Persian)
│   ├── metrics.py                 # WER / CER calculator (raw, normalized, diagnostic, splits)
│   ├── collator.py                # Speech Seq2Seq collator with padding & -100 label mask
│   └── data.py                    # Audio resampling & feature extraction pipeline
├── scripts/
│   ├── benchmark_baselines.py     # Evaluates Whisper v3 vs. BuzzASR on Neyshekar test set
│   ├── train_lora.py              # BF16 LoRA training loop on L40S
│   ├── evaluate_model.py          # Detailed evaluation of trained checkpoint & error inspection
│   ├── merge_lora.py              # Merges LoRA adapter into base model for standalone inference
│   └── transcribe.py              # Production CLI for audio transcription
├── tests/
│   └── test_normalizer_metrics.py # Offline unit tests
├── requirements.txt               # Dependencies
├── setup_server.sh                # Automated server installation script
├── run_baseline.sh                # One-click baseline benchmark launcher
└── run_train.sh                   # One-click training launcher
```

---

## Transferring to the GPU Server

From your local machine, sync this project to your GPU server:

```bash
# Using rsync (replace user and server_ip with your remote details):
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'outputs' \
  /home/ameer/Projects/stt/ user@server_ip:~/stt/

# Or push to a private Git repository and clone on the server:
git init
git add .
git commit -m "Initial Persian ASR pipeline"
git remote add origin <your-git-remote>
git push -u origin main
```

---

## Environment Setup on GPU Server

SSH into the GPU server and run the automated installer:

```bash
cd ~/stt
chmod +x setup_server.sh run_baseline.sh run_train.sh
./setup_server.sh
```

This will:
* Create a `.venv` virtual environment.
* Install PyTorch with CUDA 12 support.
* Install Hugging Face, Peft, Hazm, and JiWER.
* Verify GPU acceleration and native BF16 compatibility on the L40S.

---

## Step 1: Baseline Benchmarking

Establish baseline performance on the Neyshekar test split to compare `openai/whisper-large-v3` against `BuzzASR/persian`:

```bash
# Quick smoke-test (first 50 samples):
./run_baseline.sh --max_samples 50

# Full benchmark across the complete Neyshekar test set:
./run_baseline.sh
```

Results are printed as a comparison table and saved to `results/baselines/benchmark_summary.json`.

---

## Step 2: Training on NVIDIA L40S

Run the training pipeline in a `tmux` session or in the background:

```bash
tmux new -s train_asr

# Launch training with default L40S configuration:
./run_train.sh
```

### Key Training Flags:
* `--base_model`: Base Hugging Face model (`BuzzASR/persian` by default).
* `--dataset_name`: Hugging Face dataset (`shekar-ai/Neyshekar`).
* `--batch_size 16 --grad_accum 2`: Effective batch size of 32.
* `--learning_rate 1e-4`: Optimal for LoRA.
* `--epochs 4`: 4 full epochs with cosine scheduler and validation checkpointing.
* `--config configs/l40s_train_config.yaml`: YAML defaults (CLI flags override file).
* `--eval_batch_size 16 --max_benchmark_samples 300`: epoch benchmark guard (`0` = full 2,149-sample test set; default 300 avoids L40S OOM).
* `--no-enable-telemetry / --no-run-epoch-benchmark`: disable live telemetry or epoch eval.
* Training keeps punctuation in targets; eval strips it for standardized WER (intentional, see `src/data.py`).
* CPU inference runs without autocast (CUDA-only AMP); >30s audio is truncated to 30s, <0.5s filtered.
* Baseline numbers live in `configs/baselines.json` (single source for trainer, server, demo).

### Monitor & Control in Real-Time via Web Interface (RunPod Cockpit):
Launch the zero-dependency web interface on port 7860 (compatible with RunPod HTTP proxy):
```bash
./runpod_web.sh
```
Features available in your browser:
- Start/stop training directly from the UI.
- Real-time VRAM allocation, loss curve, learning rate, and ETA.
- Automatic **Epoch Benchmark Comparison Matrix** (default 300-sample subset per epoch; `--max_benchmark_samples 0` for all 2,149 test recordings; plots delta improvements vs. Whisper Large v3).
- **Canary Persian Orthography Diff** showing ZWNJ and spoken contractions side-by-side.
- Note: 2,149 test-sample count comes from `configs/baselines.json` / Neyshekar test split metadata; verify with `load_dataset("shekar-ai/Neyshekar", split="test")` if the dataset version changes.

---

## Step 3: Post-Training Evaluation & Diagnostics

Evaluate the fine-tuned adapter against the Neyshekar test set:

```bash
source .venv/bin/activate

python scripts/evaluate_model.py \
    --base_model BuzzASR/persian \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --dataset_name shekar-ai/Neyshekar \
    --split test \
    --batch_size 16
```

This generates:
1. Overall Normalized WER & CER.
2. Formal vs. Informal subset accuracy.
3. Diagnostic analysis showing the top 5 worst transcription errors to inspect remaining phonetic or dialect edge cases.

---

## Step 4: Merging Adapter Weights for Deployment

To run inference without requiring `peft`, merge the adapter back into the base Whisper weights:

```bash
python scripts/merge_lora.py \
    --base_model BuzzASR/persian \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --output_dir models/buzzasr-persian-fine-tuned
```

---

## Step 5: Transcribing Custom Audio (CLI)

Transcribe any local audio file or folder using either the adapter or the merged model:

```bash
# Using the LoRA adapter:
python scripts/transcribe.py \
    --audio sample.wav \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --normalize

# Batch transcribe an entire directory of audio recordings:
python scripts/transcribe.py \
    --audio_dir ./recordings/ \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --output_file results/my_transcriptions.json \
    --normalize
```
