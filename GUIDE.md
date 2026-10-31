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
* Watch the **Epoch Benchmark Matrix** update as simulated epochs complete (calculating normalized WER, CER, emerald `Δ clean vs base` / `Δ phone vs base` badges from the simulated BuzzASR baseline, plus a `Reference` tag on the simulated Whisper row).
* Inspect the **Canary Audio Orthography Diff** cards showing ZWNJ and colloquial contractions.

The demo baseline values, including phone WER, are simulated only (`DEMO_SIMULATED_BASELINES`); training and the server never read them.

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

### Configuration via `.env` (optional)

Copy `.env.example` to `.env` and set tokens/paths there instead of exporting
them in every shell. Python entrypoints and `run_*.sh` / `runpod_web.sh` load
it automatically (exported variables still win):

```bash
cp .env.example .env
# edit .env: HF_TOKEN, ASR_OUTPUT_DIR, ASR_RESULTS_DIR, ASR_RUNS_DIR,
# ASR_FAKE_TRAINER, PORT, ASR_COCKPIT_TOKEN, ...
```

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

### Automatic Training Preflight (Mandatory)

Training always runs a preflight before the model loads. It measures `BuzzASR/persian` (the authoritative baseline) and `openai/whisper-large-v3` (reference-only) on the first 1,000 test rows — the same rows the epoch matrix evaluates. Cockpit runs cache the shared result at `<runs root>/_shared/baselines/preflight.json` and copy it to `<run>/baselines/preflight.json`; CLI training defaults to `$ASR_RESULTS_DIR/baselines/preflight.json`. A matching dataset/split/sample-count/phone fingerprint reuses the cache; pass `--overwrite_preflight` to force a fresh measurement. The former `configs/` baseline file (`baselines.json`) is gone; baselines are always measured.

For a smoke run, keep both evaluations aligned with matching small values:

```bash
./run_train.sh --preflight_samples 20 --max_benchmark_samples 20
```

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
| **Batch Size** | 8 per device $\times$ 4 gradient accumulation = **32 effective batch size**, gradient checkpointing off |
| **Telephony Mix** | 50% clean / 50% customer-channel simulation (`--telephony_prob 0.5`) |
| **Optimizer** | `adamw_torch_fused` |
| **Duration** | **~10 hours** for 4 epochs (estimated from benchmark step time; not yet measured end to end) |
| **Validation / Checkpoint** | Every 1,000 optimizer steps (`--eval_steps 1000 --save_steps 1000`) |
| **Checkpoint Output** | `outputs/buzzasr-neyshekar-lora/best_adapter` |
| **Config File** | `configs/l40s_train_config.yaml` via `--config` (CLI overrides file) |
| **Preflight Baseline** | Mandatory; cockpit runs cache at `<runs root>/_shared/baselines/preflight.json` (CLI default `$ASR_RESULTS_DIR/baselines/preflight.json`); first 1,000 test rows (BuzzASR authoritative, Whisper reference-only) |
| **Epoch Benchmark** | 1,000-sample subset by default, matching the preflight (`--preflight_samples 0 --max_benchmark_samples 0` = full 2,149) |

### Live Monitoring & Management (RunPod Web Cockpit):
Launch the web interface (port 7860, RunPod HTTP proxy compatible):
```bash
./runpod_web.sh
```
Open your RunPod HTTP proxy URL in your browser to view:
- **Real-time Telemetry:** GPU VRAM allocation gauge, Loss curve, LR, and ETA.
- **Epoch Benchmark Comparison Matrix:** Evaluates a 1,000-sample subset per epoch by default, matching the preflight rows (use `--preflight_samples 0 --max_benchmark_samples 0` for the complete 2,149-sample test split; keep both values aligned). It records normalized WER, CER, formal vs. informal breakdown, and `Δ clean vs base` / `Δ phone vs base` deltas in percentage points (pp) against the measured BuzzASR baseline.
- **Canary Audio Inspector:** Live orthography comparison for ZWNJ and colloquial contractions.
- **Final Full-Test Report:** Generated by the training pipeline at the end of the run. It shows BuzzASR deltas only when the preflight and full-test row counts match; the default 1,000-row preflight against the 2,149-row test split reports `n/a (no matching full-test baseline)`.
- Set `ASR_COCKPIT_TOKEN` env to require a token on train start/stop (public proxy hardening).

### Persistent Run Ledger (Start, Reload, Resume, Merge)

Every cockpit **Start** creates a self-contained directory under the runs root.
The default is `ASR_RESULTS_DIR/runs` (`/workspace/stt-outputs/results/runs` on
RunPod when `/workspace` exists, otherwise repo-local `results/runs`);
`ASR_RUNS_DIR` overrides it.

```
<runs root>/_shared/baselines/preflight.json
<runs root>/<run_id>/            # e.g. 20261031-142530-1a2b3c
    run.json                     # config, stages, attempts, status
    steps.jsonl                  # append-only optimizer-step records
    events.jsonl                 # append-only stage/status events
    live_telemetry.json
    loss_history.json
    epoch_benchmarks.json
    baselines/preflight.json     # preflight copy used by this run
    training_live.log
    output/                      # checkpoints, best_adapter, final report
    merged/                      # created by Merge
```

- **Start** = preflight → training → post-eval. Merge is a later, optional
  stage; `merge` stays pending until you trigger it.
- **Reload/restart:** on page load the cockpit restores the latest run id,
  config, stepper, and saved steps. Server startup reconciles the ledger: a run
  whose trainer PID is alive is reattached; one that disappeared without a
  recorded exit becomes `interrupted` (Resume available); older runs remain in
  the run picker.
- **Resume** (`POST /api/runs/<run_id>/resume`) works for `interrupted`,
  `stopped`, and `failed` runs in the same directory with the original
  settings. It resumes from the training checkpoint; when training already
  completed but post-eval did not, it runs post-eval only (`--post_eval_only`).
- **Merge** (`POST /api/runs/<run_id>/merge`) requires completed training and
  `output/best_adapter/adapter_config.json`, and writes `<run>/merged/`.
- A non-persistent runs root shows this warning in the cockpit:
  `Runs are not on persistent storage (<path>). Set ASR_RUNS_DIR to a /workspace path.`

### GPU-Free Development (CPU Only)

Set `ASR_FAKE_TRAINER=1` to run the cockpit end to end without CUDA. The server
uses `scripts/dev_fake_trainer.py` and `scripts/dev_fake_merge.py`, marks the run
`source: "fake"`, and writes placeholder artifacts.

```bash
export ASR_RESULTS_DIR=/tmp/asr-demo/results
export ASR_FAKE_TRAINER=1
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 7860

# Another shell:
curl -s -X POST localhost:7860/api/train/start \
  -H 'Content-Type: application/json' -d '{}'
curl -s localhost:7860/api/runs/latest
curl -s localhost:7860/api/status
```

Wait for `run.status == "completed"` with `steps.count > 0`, then restart the
server and confirm `/api/runs/latest` still returns the run. Use Stop, Resume,
and Merge from the cockpit (or the matching API endpoints) to exercise the rest
of the lifecycle. `ASR_FAKE_STEPS` (default 200) and `ASR_FAKE_STEP_DELAY`
(default 0.05 s) shorten or lengthen the fake run.

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
- Prints post-training WER and CER.
- Breaks down accuracy on formal vs. colloquial/informal Persian.
- Displays the top 5 worst errors for qualitative inspection.

This standalone script prints metrics only; baseline comparison is performed by the training pipeline's final report (see Section 4), not by `evaluate_model.py`.

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
