# High-Precision Persian Speech-to-Text (ASR) Pipeline

Fine-tuning and evaluation suite designed to overcome high Word Error Rates (WER) in Persian ASR. Built around **`BuzzASR/persian`** (Whisper Large v3 architecture with a native monolingual Persian BPE tokenizer) and fine-tuned on the **`shekar-ai/Neyshekar`** (v6) corpus.

Optimized for **NVIDIA L40S (48 GB VRAM)** using unquantized **Bfloat16 LoRA (PEFT)**.

---

## Table of Contents
1. [⚡ Quickstart: Local UI Demo (No GPU Needed)](#-quickstart-local-ui-demo-no-gpu-needed)
2. [Persistent Runs & Cockpit Ledger](#persistent-runs--cockpit-ledger)
3. [Durable HF Run Mirror](#durable-hf-run-mirror)
4. [Why This Architecture?](#why-this-architecture)
5. [Project Structure](#project-structure)
6. [Transferring to the GPU Server](#transferring-to-the-gpu-server)
7. [Environment Setup on GPU Server](#environment-setup-on-gpu-server)
8. [Step 1: Baseline Benchmarking](#step-1-baseline-benchmarking)
9. [Step 2: Training on NVIDIA L40S](#step-2-training-on-nvidia-l40s)
10. [Step 3: Post-Training Evaluation & Diagnostics](#step-3-post-training-evaluation--diagnostics)
11. [Step 4: Merging Adapter Weights for Deployment](#step-4-merging-adapter-weights-for-deployment)
12. [Step 5: Transcribing Custom Audio (CLI)](#step-5-transcribing-custom-audio-cli)

---

## ⚡ Quickstart: Local UI Demo (No GPU Needed)

You can launch and explore the full Web Cockpit right now on your local machine using standard Python 3:

```bash
./run_local_demo.sh
```

Then open `http://127.0.0.1:7860` in your browser. You can click **Launch Run** to simulate active Persian ASR training, watch the live loss trajectory, view simulated epoch benchmarks updating in real time, and inspect the canary Persian sentences. The demo's baseline values are simulated only (`DEMO_SIMULATED_BASELINES`); training and the server never read them.

---

## Persistent Runs & Cockpit Ledger

Cockpit training is backed by a persistent run ledger. Every **Start** creates a
self-contained run directory; the web UI and the `/api/runs/*` endpoints read
from it, so a page reload or server restart does not lose the run.

The default runs root is `ASR_RESULTS_DIR/runs`:
`/workspace/stt-outputs/results/runs` on RunPod when `/workspace` exists,
otherwise the repo-local `results/runs`. `ASR_RUNS_DIR` overrides it.

```
/workspace/stt-outputs/results/runs/
├── _shared/baselines/preflight.json     # shared fingerprint-keyed preflight cache
└── <run_id>/                            # e.g. 20261031-142530-1a2b3c
    ├── run.json                         # config, stage/status state, attempts
    ├── steps.jsonl                      # append-only optimizer-step records
    ├── events.jsonl                     # append-only stage/status events
    ├── live_telemetry.json              # latest telemetry snapshot
    ├── loss_history.json                # capped chart cache
    ├── epoch_benchmarks.json            # this run's benchmark matrix
    ├── baselines/preflight.json         # preflight copy used by this run
    ├── training_live.log                # trainer stdout/stderr
    ├── output/                          # checkpoints, best_adapter, final report
    └── merged/                          # created by the Merge stage
```

* **Start** runs **preflight → training → post-eval**. The `merge` stage stays
  pending: it is a separate, optional stage you trigger after training has
  completed. `POST /api/train/start` creates the run; the cockpit polls
  `/api/runs/latest`, `/api/runs/<run_id>`, and `/api/runs/<run_id>/steps`.
* **Reload / server restart.** On load the cockpit restores the latest run —
  run id, config, stepper, and saved steps. The saved-steps line is exact for
  the active run: it reports the live telemetry `step_count`, which increments
  once per new optimizer step (a resumed or replayed step is not
  double-counted); historical runs fall back to the deduplicated manifest
  count. On startup the server reconciles
  the ledger: a run whose trainer PID is still alive is reattached; a trainer
  that disappeared without a recorded exit is marked `interrupted` (Resume
  available); older runs remain browsable from the run picker.
* **Resume.** `POST /api/runs/<run_id>/resume` continues an `interrupted`,
  `stopped`, or `failed` run in the same directory with its original settings.
  If a training checkpoint exists it auto-resumes; if training already
  completed but post-eval did not, it runs post-eval only
  (`--post_eval_only`) instead of retraining.
* **Merge.** Once the **training stage is completed** and
  `output/best_adapter/adapter_config.json` exists, the cockpit offers Merge
  even when post-eval failed or was stopped (the overall run status may be
  `completed`, `failed`, `stopped`, or `interrupted`). The overall run must not
  still be `starting`/`running`, no stage may still be running, and a run whose
  merge stage already completed is not offered again. The Merge action (or
  `POST /api/runs/<run_id>/merge`) writes the standalone merged model to
  `<run_dir>/merged/` (`config.json`, `model.safetensors`, tokenizer files).

A runs root outside `/workspace` or `/runpod-volume` is **not persistent**: the
cockpit shows `Runs are not on persistent storage (<path>). Set ASR_RUNS_DIR to
a /workspace path.` Repo-local runs survive a server restart but not a pod
restart.

**Upgrading an existing deployment:** finish any legacy (pre-ledger) training
run before upgrading. The cockpit resumes and reconciles runs from `run.json`;
a process with only the legacy global results PID file is still detected, but
it has no attempt history to resume or merge.

### GPU-Free Development (CPU Only)

Set `ASR_FAKE_TRAINER=1` to exercise the full cockpit lifecycle without CUDA.
The server spawns `scripts/dev_fake_trainer.py` and `scripts/dev_fake_merge.py`
instead of the real trainer/merge; runs are marked `source: "fake"` and write
placeholder artifacts only. Fake runs use `<runs root>/_shared-fake` for their
preflight cache, so the real `_shared/baselines/preflight.json` is never
overwritten. Fake runs mirror the real lifecycle: `--fresh_start` clears prior
fake telemetry, benchmarks, stop signal, and best adapter (and is ignored with
`--post_eval_only`, like the real trainer), and a directly launched fake
trainer owns its attempt and `<run>/training.pid`, closing it on exit, while a
cockpit-spawned fake trainer leaves the server-owned attempt alone.

```bash
# Isolated ledger; fake trainer, no GPU.
export ASR_RESULTS_DIR=/tmp/asr-demo/results
export ASR_FAKE_TRAINER=1
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 7860

# In another shell:
curl -s -X POST localhost:7860/api/train/start -H 'Content-Type: application/json' -d '{}'
curl -s localhost:7860/api/runs/latest
curl -s localhost:7860/api/status
```

Wait for `run.status == "completed"` and `steps.count > 0`, restart the server,
and confirm `/api/runs/latest` still returns the same run. `ASR_FAKE_STEPS`
(default 200) and `ASR_FAKE_STEP_DELAY` (default 0.05 s) control the fake run's
length. `./run_local_demo.sh` is different: it serves the zero-dependency
simulated UI and does not create ledger runs.

---

## Durable HF Run Mirror

Set `ASR_HF_RUN_REPO=<owner>/<name>` (for example `user/stt-experiment`) and
authenticate with `HF_TOKEN` to mirror the run ledger and artifacts to a
Hugging Face repo. The repo is created **private by default**. For new runs,
mirroring is off when the variable is unset; an existing run keeps the repo
recorded in its `run.json` (`hf.repo_id`) and a cockpit merge still reuses it
(remove that field to opt an old run out). Uploads are best effort: a Hugging
Face outage does not fail training or merging, although background uploads
may keep a training/merge process alive while they retry and drain (up to
about 120 seconds on close).

Each run is self-contained under `runs/<run_id>/`:

* `checkpoint-<step>/` is uploaded when a complete checkpoint is written, and
  the run's `hf_state.json` records the high-water step used on restore.
* `adapter_snapshots/step-<step>/` is uploaded every
  `--hf_adapter_snapshot_steps` steps (default 250; `0` disables snapshots).
* The ledger is refreshed at stage transitions: `run.json`, `steps.jsonl`,
  `events.jsonl`, telemetry, benchmark, loss, and preflight files.
* `best_adapter/`, the final reports, and the cockpit's `merged/` model are
  uploaded when available. When a repo is configured, the cockpit merge passes
  `--run_dir` and `--hf_run_repo`; a manual merge needs both
  `--run_dir <run_dir>` and `--hf_run_repo <owner/name>` (or `ASR_HF_RUN_REPO`
  set) to upload the merged model with a status-`merge` ledger.
* `latest.json` points at the most recently active run.

Restore by run id on a fresh pod (the repo comes from `ASR_HF_RUN_REPO`, or
pass `--repo`):

```bash
python scripts/hf_restore.py --run 20261031-142530-1a2b3c
# Then resume:
python scripts/train_lora.py --hf_resume 20261031-142530-1a2b3c
```

The restore CLI exits `0` with a complete checkpoint, `2` when only the ledger
was restored (no resume is possible), and `1` on failure. Add
`--include-merged` to fetch `merged/` too.

In the cockpit, **Recover** lists remote runs and offers **Restore only** or
**Restore & resume**. Completed runs are restore-only, and a run without a
complete checkpoint cannot resume.

Remote retention is **manual**: the mirror never deletes old runs or merged
models. Remove obsolete `runs/<run_id>/` folders in the Hugging Face repo (UI
or `hf` CLI) to reclaim storage. Tokens are read from the environment only and
are never written to run files or logs; `HF_TOKEN` is shared with
`scripts/upload_to_hf.py`.

---

## Why This Architecture?

* **Native Persian Tokenizer:** OpenAI's standard Whisper Large v3 uses a multilingual byte-level BPE tokenizer, taking 5–12 tokens per Persian word. `BuzzASR/persian` replaces this with a Persian BPE tokenizer (~3.3x compression), preventing looping hallucinations and speeding up generation.
* **Neyshekar Dataset (v5/v6):** Provides ~80–100 hours of curated native Persian speech with strict **speaker-disjoint** and **text-disjoint** splits, containing both formal Persian and ~25–35% colloquial/informal Persian.
* **Full BF16 LoRA on L40S:** With 48 GB VRAM, training runs in full 16-bit precision without 4-bit quantization latency, taking ~10 hours for the default 2 paired epochs (same optimizer steps as 4 disjoint epochs; prep is ~2× the first disjoint run) with fused AdamW and SDPA attention.

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
│   ├── data.py                    # Audio resampling & feature extraction pipeline
│   └── run_state.py               # Persistent run ledger (manifest, steps, stages)
├── server/
│   ├── app.py                     # FastAPI cockpit + run/resume/merge APIs
│   └── static/                    # Zero-build web UI
├── scripts/
│   ├── benchmark_baselines.py     # Evaluates Whisper v3 vs. BuzzASR on Neyshekar test set
│   ├── train_lora.py              # BF16 LoRA training loop on L40S
│   ├── evaluate_model.py          # Detailed evaluation of trained checkpoint & error inspection
│   ├── merge_lora.py              # Merges LoRA adapter into base model for standalone inference
│   ├── dev_fake_trainer.py        # GPU-free fake trainer for cockpit development
│   ├── dev_fake_merge.py          # GPU-free fake merge for cockpit development
│   └── transcribe.py              # Production CLI for audio transcription
├── tests/                         # Offline unit, API, and UI-contract tests
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

Optional: copy `.env.example` to `.env` to set `HF_TOKEN`, `ASR_OUTPUT_DIR`,
`ASR_RESULTS_DIR`, `ASR_RUNS_DIR`, `ASR_FAKE_TRAINER`, `PORT`,
`ASR_COCKPIT_TOKEN`, etc. Python entrypoints and `run_*.sh` / `runpod_web.sh`
load it automatically (exported vars win).

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

# Smoke-run wiring check: keep preflight and epoch evaluation on the same rows.
./run_train.sh --preflight_samples 20 --max_benchmark_samples 20
```

### Key Training Flags:
* `--base_model`: Base Hugging Face model (`BuzzASR/persian` by default).
* `--dataset_name`: Hugging Face dataset (`shekar-ai/Neyshekar`).
* `--batch_size 8 --grad_accum 4`: Effective batch size of 32; the measured fast L40S configuration.
* `--gradient_checkpointing` / `--no-gradient_checkpointing`: **off by default**. Off is about 35% faster; enable it only if a larger micro-batch requires the VRAM savings.
* `--eval_steps 1000 --save_steps 1000`: evaluation and checkpoint interval. `save_steps` must be a multiple of `eval_steps`.
* `--telephony_mode paired` (default): every utterance once clean and once phone-simulated, 116,488 rows; `--epochs 2` is compute-matched to 4 disjoint epochs.
* `--telephony_mode disjoint`: the old 50/50 coin (`--telephony_prob 0.5`); use `--epochs 4`.
* `--telephony_mode clean`: no phone simulation; use `--epochs 4`.
* `--epochs 2`: 2 paired epochs (same optimizer steps as 4 disjoint epochs).
* `--learning_rate 1e-4`: Optimal for LoRA.
* `--config configs/l40s_train_config.yaml`: YAML defaults (CLI flags override file).
* Prepared-data cache: direct Arrow shards in `prep_cache/train` and `prep_cache/validation`, published atomically with a validated `meta.json` (format, fingerprint, shard rows/sizes, feature schema); no `save_to_disk` duplicate. Paired training needs ~170 GB steady and peak; the writer guards ~180 GB free (current free plus the stale cache it replaces), leaving room on the ~250 GB overlay for raw audio/model weights (~25 GB).
* `--run_dir <dir>`: self-contained CLI run ledger. It must be a canonical run directory directly under the resolved runs root (`<runs-root>/YYYYMMDD-HHMMSS-xxxxxx`); anything else exits with an error telling you to use `ASR_RUNS_DIR`/a canonical id. When `--results_dir` is not given it defaults to the run directory, and a default `--output_dir` becomes `<run>/output`. The trainer records its PID/attempt, writes `<run>/training.pid`, tees stdout/stderr to `<run>/training_live.log`, and closes the attempt at exit; the cockpit detects the live CLI run from the pid file. An explicit `--output_dir` still wins. An explicit `--results_dir` is honored for the run's telemetry (`live_telemetry.json`), benchmarks, loss history, and stop signal; the log and PID file always stay under the run directory. `--fresh_start` clears prior checkpoints, adapters, reports, and telemetry history first, including for a plain CLI run with no `--run_dir`.
* `--eval_batch_size 16 --preflight_samples 1000 --max_benchmark_samples 1000`: the mandatory preflight and the per-epoch matrix both evaluate the first 1,000 test rows. Set both to `0` (`--preflight_samples 0 --max_benchmark_samples 0`) for the full 2,149-sample test set; keep them aligned.
* `--no-enable-telemetry / --no-run-epoch-benchmark`: disable live telemetry or epoch eval.
* Training keeps punctuation in targets; eval strips it for standardized WER (intentional, see `src/data.py`).
* CPU inference runs without autocast (CUDA-only AMP); >30s audio is truncated to 30s, <0.5s filtered.
* Before training starts, a mandatory preflight measures `BuzzASR/persian` and `openai/whisper-large-v3` on the same test rows. Cockpit runs cache the shared result at `<runs root>/_shared/baselines/preflight.json` and copy it to `<run>/baselines/preflight.json`; CLI training defaults to `$ASR_RESULTS_DIR/baselines/preflight.json`. A matching fingerprint reuses the cache; `--overwrite_preflight` forces re-measurement. BuzzASR is the authoritative baseline and Whisper is reference-only. The former `configs/` baseline file (`baselines.json`) is gone; baselines are always measured.

### Monitor & Control in Real-Time via Web Interface (RunPod Cockpit):
Launch the zero-dependency web interface on port 7860 (compatible with RunPod HTTP proxy):
```bash
./runpod_web.sh
```
Features available in your browser:
- Start/stop training directly from the UI.
- Real-time VRAM allocation, loss curve, learning rate, and ETA.
- Automatic **Epoch Benchmark Comparison Matrix** (default 1,000-row subset per epoch, the same rows as the preflight; use `--preflight_samples 0 --max_benchmark_samples 0` for all 2,149 test recordings, keeping both aligned). Deltas are `Δ clean vs base` and `Δ phone vs base`, in percentage points (pp), against the measured BuzzASR preflight baseline; rows evaluated on a different sample count show `—`.
- **Canary Persian Orthography Diff** showing ZWNJ and spoken contractions side-by-side.
- The training pipeline's **final full-test report** performs the baseline comparison: it shows BuzzASR deltas only when a baseline was measured on the same full-test row count. With the default 1,000-row preflight it reports `n/a (no matching full-test baseline)` against the 2,149-row test split.
- Note: the 2,149 test-sample count comes from the Neyshekar test split metadata; verify with `load_dataset("shekar-ai/Neyshekar", split="test")` if the dataset version changes.

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

This standalone script prints metrics only:
1. Overall Normalized WER & CER.
2. Formal vs. Informal subset accuracy.
3. Diagnostic analysis showing the top 5 worst transcription errors to inspect remaining phonetic or dialect edge cases.

Baseline deltas are produced by the training pipeline's final report (see Step 2), not by `evaluate_model.py`.

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
