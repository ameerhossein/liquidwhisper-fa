# Persian ASR GPU Server Execution Guide

A concise, step-by-step guide to transferring and running the Persian Speech-to-Text pipeline on your remote GPU server (NVIDIA L40S, 48 GB VRAM), plus a local zero-dependency UI demo.

---

## ⚡ Try the Interactive Web UI Demo Locally (No GPU Needed)

You can launch and explore the full Web Cockpit right now on your local machine using standard Python 3 (zero packages to install):

```bash
./run_local_demo.sh
```

Open **`http://127.0.0.1:7860`** in your browser:
* Click **Launch Run** to simulate active Persian ASR training.
* Watch the live loss curve descent, VRAM telemetry, and step throughput.
* Watch the **Epoch Benchmark Matrix** update as simulated epochs complete (calculating normalized WER, CER, and emerald improvement badges vs. Whisper Large v3).
* Inspect the **Canary Audio Orthography Diff** cards showing ZWNJ and colloquial contractions.

---

## 1. Transfer Project to the GPU Server

From your local machine terminal, copy the project directory to your remote GPU server:

```bash
# Using rsync (replace 'user' and 'your_server_ip' with your remote server credentials)
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'outputs' --exclude 'results' \
  /home/ameer/Projects/stt/ user@your_server_ip:~/stt/
```

*Alternative via Git:*
```bash
# On local machine:
cd /home/ameer/Projects/stt
git init
git add .
git commit -m "Persian ASR setup"
git remote add origin <your-git-remote-url>
git push -u origin main

# On GPU server:
git clone <your-git-remote-url> ~/stt
```

---

## 2. Environment Setup on GPU Server

SSH into your GPU server and run the automated setup script:

```bash
ssh user@your_server_ip
cd ~/stt

# Make scripts executable
chmod +x setup_server.sh run_baseline.sh run_train.sh

# Run installation
./setup_server.sh
```

**What this does automatically:**
- Creates a Python virtual environment (`.venv`).
- Installs PyTorch with CUDA 12 support.
- Installs all dependencies (`transformers`, `peft`, `datasets`, `hazm`, `jiwer`, etc.).
- Verifies GPU hardware, memory, and native Bfloat16 support on the L40S.

---

## 3. Run Baseline Benchmarks

Before fine-tuning, benchmark `openai/whisper-large-v3` against `BuzzASR/persian` on the Neyshekar test set to obtain baseline metrics:

```bash
# Smoke test on the first 50 samples:
./run_baseline.sh --max_samples 50

# Full benchmark across all test samples:
./run_baseline.sh
```

**Output:**
- Generates a side-by-side comparison table reporting:
  - Raw WER & CER
  - Normalized WER & CER (ZWNJ preserved)
  - Diagnostic WER (ZWNJ converted to space)
  - Formal vs. Informal subset accuracy
- Saves full predictions to `results/baselines/benchmark_summary.json`.

---

## 4. Fine-Tuning on NVIDIA L40S

Run the training pipeline in a detached `tmux` session so disconnection won't stop the job:

```bash
# Start a new tmux session
tmux new -s train_asr

# Activate environment
source .venv/bin/activate

# Launch training with the L40S-optimized configuration
./run_train.sh
```

*(To detach from the session at any time, press `Ctrl+B` then `D`. To reattach later, run `tmux attach -t train_asr`.)*

### Training Details:
| Parameter | Value |
| :--- | :--- |
| **Base Model** | `BuzzASR/persian` |
| **Dataset** | `shekar-ai/Neyshekar` (v6 official release, ~100 hours) |
| **Precision** | Native Bfloat16 (`bf16`) |
| **LoRA Configuration** | $r=64$, $\alpha=128$, target: `q_proj, v_proj, k_proj, out_proj, fc1, fc2` |
| **Batch Size** | 16 per device $\times$ 2 gradient accumulation = **32 effective batch size** |
| **Optimizer** | `adamw_torch_fused` |
| **Duration** | **~3 to 5 hours** for 4 epochs |
| **Checkpoint Output** | `outputs/buzzasr-neyshekar-lora/best_adapter` |
| **Config File** | `configs/l40s_train_config.yaml` via `--config` (CLI overrides file) |
| **Epoch Benchmark** | 300-sample subset by default (`--max_benchmark_samples 0` = full 2,149) |

### Live Monitoring & Management (RunPod Web Cockpit):
Launch the web interface (port 7860, RunPod HTTP proxy compatible):
```bash
./runpod_web.sh
```
Open your RunPod HTTP proxy URL in your browser to view:
- **Real-time Telemetry:** GPU VRAM allocation gauge, Loss curve, LR, and ETA.
- **Epoch Benchmark Comparison Matrix:** Evaluates a 300-sample subset per epoch by default (use `--max_benchmark_samples 0` for the complete 2,149-sample test split) and records normalized WER, CER, formal vs. informal breakdown, and emerald delta badges vs. Whisper Large v3.
- **Canary Audio Inspector:** Live orthography comparison for ZWNJ and colloquial contractions.
- Set `ASR_COCKPIT_TOKEN` env to require a token on train start/stop (public proxy hardening).

---

## 5. Evaluate the Fine-Tuned Model

Once training finishes, evaluate the best adapter checkpoint on the held-out test split:

```bash
source .venv/bin/activate

python scripts/evaluate_model.py \
    --base_model BuzzASR/persian \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --dataset_name shekar-ai/Neyshekar \
    --split test \
    --batch_size 16
```

**Features:**
- Compares post-training WER and CER against baseline results.
- Breaks down accuracy on formal vs. colloquial/informal Persian.
- Displays the top 5 worst errors for qualitative inspection.

---

## 6. Merge Adapter for Standalone Deployment

Merge the LoRA adapter back into base Whisper weights for zero-overhead inference:

```bash
python scripts/merge_lora.py \
    --base_model BuzzASR/persian \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --output_dir models/persian-whisper-standalone
```

---

## 7. Transcribe Audio Files (CLI)

Use the transcription CLI tool with your trained model:

```bash
# Transcribe a single audio file:
python scripts/transcribe.py \
    --audio sample.wav \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --normalize

# Transcribe an entire folder of recordings and save JSON output:
python scripts/transcribe.py \
    --audio_dir ./recordings/ \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --output_file results/transcriptions.json \
    --normalize
```
