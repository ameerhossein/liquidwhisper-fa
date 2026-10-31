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
- HF caches (base model + Neyshekar raw) and the **prep cache** (prepared Arrow
  shards): auto-routed by `src/cache.py` to the roomiest disk
  (`~/.cache/huggingface`, `/tmp/hf_cache`), never `/workspace`.
  Overrides: `ASR_HF_CACHE_DIR` / `HF_DATASETS_CACHE` / `ASR_PREP_CACHE_DIR`;
  `--prep_cache_dir <output_dir>/prep_cache` keeps the prep cache on the
  persistent volume. A template-default `HF_HOME` pointing at `/workspace` is
  overridden with a warning (`ASR_HONOR_HF_HOME=1` keeps it).
- Any merge output under `models/...` unless you point it at `/workspace`.

### Prep cache (direct Arrow shards)

`scripts/train_lora.py` maps prepared features directly into
`<prep cache>/train` and `<prep cache>/validation` with
`datasets.map(cache_file_name=...)`; there is no `Dataset.save_to_disk()`
copy, so peak and steady state are the same size. The default paired train
split is 116,488 rows and takes ~170 GB.

A cache slot holds the Arrow shards plus `meta.json` (format `shards_v1`, the
full fingerprint, ordered shard files with row counts/sizes, total rows, and
the feature schema). Loading validates all of it and treats any mismatch as a
miss. Publication maps into `<name>.staging-<pid>-<random>`, writes
`meta.json` last, removes the old slot, and atomically renames the staging
directory into place; a crash leaves only the staging directory, which the
next attempt cleans up. Before the first map the trainer requires ~180 GB
projected free space (current free plus the stale train cache it will remove)
and aborts with a clear error otherwise. The ~250 GB RunPod overlay leaves
headroom for raw audio/model weights (~25 GB on the same disk).

Cache slots are keyed by the prep fingerprint (dataset, split, sample cap,
telephony mode/prob/profiles/seed/ffmpeg, label ids), so switching modes
re-preps. The validation slot is mode-independent and survives train-mode
changes. Existing `save_to_disk` caches are not readable; the new fingerprint
and `meta.json` format replace them on the first run. To keep both paired and
disjoint train caches, use a different `--prep_cache_dir` (or
`ASR_PREP_CACHE_DIR`) per mode; only one train slot exists per directory.

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
- **Resume with the original telephony mode.** A checkpoint trained before
  paired became the default (or any disjoint run) must resume with
  `--telephony_mode disjoint` and the old epoch count (typically
  `--epochs 4`), or start fresh with `--fresh_start`. The mode is part of the
  prep fingerprint, so resuming an old run as paired re-preps different data
  than the checkpoint saw. Cockpit Resume reuses the stored mode and injects
  `disjoint` for legacy manifests that do not record one.
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

## 6. Durable HF Run Mirror

Enable mirroring with `ASR_HF_RUN_REPO=<owner>/<name>` and `HF_TOKEN`
(`HF_TOKEN` is the same token used by `scripts/upload_to_hf.py`). Unset means
off. The repo is created **private by default**, and uploads are best effort
with retries, so a Hugging Face outage never blocks training or merging.

The remote layout is one self-contained folder per run:

```
latest.json
runs/<run_id>/run.json
runs/<run_id>/hf_state.json
runs/<run_id>/checkpoint-<step>/
runs/<run_id>/adapter_snapshots/step-<step>/
runs/<run_id>/best_adapter/
runs/<run_id>/reports/
runs/<run_id>/merged/
```

- A complete `checkpoint-<step>/` is uploaded as it is written. `hf_state.json`
  is the authoritative high-water mark, and `latest.json` points at the most
  recently active run.
- `adapter_snapshots/step-<step>/` is uploaded every
  `--hf_adapter_snapshot_steps` steps (default 250; `0` disables). The trainer
  keeps the newest two snapshots locally.
- The ledger (`run.json`, `steps.jsonl`, `events.jsonl`, `live_telemetry.json`,
  `loss_history.json`, `epoch_benchmarks.json`, and
  `baselines/preflight.json`) is refreshed at stage transitions.
- `best_adapter/`, the final reports, and the merged model are uploaded when
  available. The cockpit merge passes `--run_dir`/`--hf_run_repo` to
  `scripts/merge_lora.py`; a manual merge can add `--run_dir <run_dir>`.

Restore by run id on a fresh pod (repo from `ASR_HF_RUN_REPO`, or `--repo`):

```bash
python scripts/hf_restore.py --run <run_id>            # add --include-merged for merged/
python scripts/train_lora.py --hf_resume <run_id>
```

Exit codes: `0` restored with a complete checkpoint, `2` ledger only (cannot
resume), `1` failure.

In the cockpit, the **Recover** dialog lists remote runs and offers **Restore
only** or **Restore & resume**. Completed runs are restore-only; a run without
a complete checkpoint cannot resume. Restore progress survives a page reload.

**Retention is manual.** The mirror never prunes remote runs, checkpoints,
snapshots, or merged models. Delete old `runs/<run_id>/` folders in the
Hugging Face repo (UI or `hf` CLI) when storage needs to be reclaimed. Tokens
are read from the environment only and are never written into run files or
logs.
