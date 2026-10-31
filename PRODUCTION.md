# Production Notes: Checkpoints, Adapters, Merged Models, Storage

## 1. The 3 Artifacts

| Artifact | How created | Contents | Use for |
|---|---|---|---|
| `checkpoint-*/` | Auto-saved every `--save_steps 1000` during training (`scripts/train_lora.py`). Only last 2 kept (`--save_total_limit 2`). | LoRA weights + optimizer + scheduler + `trainer_state.json` | Crash / pod-restart resume only. Not for inference. |
| `best_adapter/` | Auto-saved once at end: `trainer.model.save_pretrained(best_adapter)` (`scripts/train_lora.py`). Selected by `load_best_model_at_end=True, metric_for_best_model="wer"`. | `adapter_model.safetensors` + `adapter_config.json` + processor. No optimizer. | `evaluate_model.py`, `transcribe.py --adapter_path`, `merge_lora.py --adapter_path`. Requires base `BuzzASR/persian` + `peft`. |
| Merged model | **Manual**: `scripts/merge_lora.py` (`merge_and_unload()`), or the cockpit's Merge action (`POST /api/runs/<run_id>/merge`) after training completes. Nothing auto-runs it. | Full Whisper weights (`W0 + BA`) + processor. No `peft` needed. | Production inference. |

## 2. Production Use (Merged Model)

```bash
# 1. Merge (required, manual, --output_dir has no default)
python scripts/merge_lora.py \
  --base_model BuzzASR/persian \
  --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
  --output_dir models/buzzasr-persian-fine-tuned

# 2a. Single file
python scripts/transcribe.py \
  --audio sample.wav \
  --model_path models/buzzasr-persian-fine-tuned \
  --normalize

# 2b. Batch dir -> JSON
python scripts/transcribe.py \
  --audio_dir ./recordings/ \
  --model_path models/buzzasr-persian-fine-tuned \
  --output_file results/my_transcriptions.json \
  --normalize
```

Direct integration:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
processor = WhisperProcessor.from_pretrained("models/buzzasr-persian-fine-tuned")
model = WhisperForConditionalGeneration.from_pretrained(
    "models/buzzasr-persian-fine-tuned", dtype="auto").to("cuda").eval()
```

Inference details (`scripts/transcribe.py`): 16 kHz mono resample, `max_new_tokens=225`, CUDA = BF16/FP16 autocast, CPU = FP32, `--normalize` applies `PersianTextNormalizer` (Hazm/ZWNJ).

## 3. Persistent vs Ephemeral (RunPod)

Only `/workspace` survives pod restarts. Container disk (`~/stt`, `/tmp`, `~/.cache`) does not.

Persistent by default (when `/workspace` exists):
- `ASR_OUTPUT_DIR` default: `/workspace/stt-outputs/buzzasr-neyshekar-lora` (`scripts/train_lora.py`, `run_train.sh`, `configs/l40s_train_config.yaml`). CLI training writes `checkpoint-*/`, `best_adapter/`, `final_evaluation_report.json/md` there. Cockpit runs are run-scoped: `output/` lives inside the run directory unless the start payload sets an explicit `output_dir` (recorded in `run.json`).
- `ASR_RESULTS_DIR` default: `/workspace/stt-outputs/results`. Top-level `live_telemetry.json`, `loss_history.json`, `epoch_benchmarks.json`, `training_live.log`, and `training.pid` remain for legacy/no-ledger views and the CLI preflight cache at `baselines/preflight.json` (plus per-model `baselines/preflight/` evaluation details). Cockpit runs keep the same files inside their run directory. A CLI `--run_dir` with an explicit `--results_dir` splits them: telemetry, benchmarks, loss history, and the stop signal live in `--results_dir`, while `training_live.log` and `training.pid` stay under the run directory.
- **Run ledger** (`ASR_RUNS_DIR`, default `$ASR_RESULTS_DIR/runs`):

  ```
  results/runs/<run_id>/          # run.json, steps.jsonl, events.jsonl,
                                  # telemetry, benchmarks, log, output/, merged/
  results/runs/_shared/baselines/preflight.json
  ```

  On RunPod this resolves to `/workspace/stt-outputs/results/runs`; the local
  fallback is repo-local `results/runs`. Each `<run_id>/` is self-contained:
  `run.json` (atomic manifest with config, stages, attempts, status),
  append-only `steps.jsonl` and `events.jsonl`, `live_telemetry.json`,
  `loss_history.json`, `epoch_benchmarks.json`, `baselines/preflight.json`,
  `training_live.log`, `training.pid`, `stop_training.signal`, `output/`,
  `merged/`, `merge.log`, and `merge.pid`. The cockpit restores, resumes, and
  merges from this directory, including after a server restart.

Ephemeral (re-downloaded / lost on restart):
- `~/stt/.venv/`, repo-local `outputs/`, `results/`, `models/`
- HF caches (base model + Neyshekar raw + `map` temps) and `prep_cache`
  (prepared Arrows): auto-routed by `src/cache.py` to the roomiest disk
  (`~/.cache/huggingface`, `/tmp/hf_cache`), never `/workspace`.
  Overrides: `ASR_HF_CACHE_DIR` / `HF_DATASETS_CACHE` / `ASR_PREP_CACHE_DIR`.
  A template-default `HF_HOME` pointing at `/workspace` is overridden with
  a warning (`ASR_HONOR_HF_HOME=1` keeps it).
- Any merge output under `models/...` unless you point it at `/workspace`.

Persistent merge:

```bash
python scripts/merge_lora.py \
  --base_model BuzzASR/persian \
  --adapter_path /workspace/stt-outputs/buzzasr-neyshekar-lora/best_adapter \
  --output_dir /workspace/stt-outputs/buzzasr-persian-merged
```

The cockpit's Merge action is persistent automatically: it writes to
`<runs_root>/<run_id>/merged/` under the same `ASR_RUNS_DIR` root.

Check: `echo $ASR_OUTPUT_DIR; ls -d /workspace/stt-outputs/buzzasr-neyshekar-lora/checkpoint-*`

## 4. Continuing Training Later

- Have `checkpoint-*/`? Resume (preserves optimizer/scheduler):
  ```bash
  ./run_train.sh  # auto-resumes latest checkpoint in --output_dir
  ```
- In the cockpit, `interrupted`, `stopped`, and `failed` runs show **Resume**.
  It reuses the same run directory and original settings, auto-resumes from the
  checkpoint, and switches to `--post_eval_only` when training already
  completed but post-eval is still pending.
- Only have `best_adapter`/merged? Start round 2 with merged as new base (use lower LR to avoid forgetting). Do not stack adapter-on-adapter:
  ```bash
  python scripts/train_lora.py \
    --base_model /workspace/stt-outputs/buzzasr-persian-merged \
    --output_dir outputs/buzzasr-round2-lora \
    --learning_rate 5e-5
  ```

## 5. Baselines and Preflight

The former `configs/` baseline file (`baselines.json`) is gone. Every training run performs a mandatory preflight before the model/CUDA load:

- Both the preflight and the epoch benchmark matrix use the first 1,000 test rows by default (`--preflight_samples 1000 --max_benchmark_samples 1000`; `0` = full 2,149-row test split).
- **BuzzASR/persian is the authoritative baseline; Whisper Large v3 is reference-only.** A failed Whisper preflight omits its row instead of inventing values or blocking training.
- The measured preflight is cached at `<runs root>/_shared/baselines/preflight.json` for cockpit runs (CLI default: `$ASR_RESULTS_DIR/baselines/preflight.json`) and copied to `<run>/baselines/preflight.json`. A matching fingerprint (dataset, split, sample count, phone seed/profiles, telephony ffmpeg encoding, eval batch size, model ids, and a hash of the evaluator/metric code) reuses the cache across restarts; `--overwrite_preflight` forces re-measurement. Dataset/model revisions are not pinned: bump `PREFLIGHT_VERSION` in `src/preflight.py` and clear (or overwrite) the cache after changing them.
- Epoch deltas are displayed as `Δ clean vs base` and `Δ phone vs base`, in percentage points (pp), and only when the row and baseline have the same sample count.
- The final full-test report shows no delta without a matching full-test baseline; the default 1,000-row preflight against the 2,149-row test split reports `n/a (no matching full-test baseline)`.
- For smoke runs, pass matching small values so the comparison stays valid: `--preflight_samples 20 --max_benchmark_samples 20`.
