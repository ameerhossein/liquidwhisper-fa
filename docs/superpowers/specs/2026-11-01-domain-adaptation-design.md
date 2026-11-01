# Domain Adaptation on Merged Standalone Model — Design

- Date: 2026-11-01
- Status: Draft for review
- Scope: domain data pipeline, formal/informal replay sampling, local merged model support, single-command CLI, cockpit UI/API, tests

---

## 1. Intent

Fine-tune an existing merged Persian Whisper model (`BuzzASR/persian` merged with a Neyshekar LoRA adapter via `scripts/merge_lora.py`) on small domain-specific custom audio (typically 2–3 hours). 

Training exclusively on 2–3 hours of custom speech causes severe catastrophic forgetting and acoustic overfitting. To anchor general Persian capability while adapting to domain jargon and acoustics, this system:
1. Ingests custom domain audio and transcripts directly from disk.
2. Extracts a balanced 50/50 replay buffer of formal and informal Persian utterances from Neyshekar (`shekar-ai/Neyshekar`).
3. Interleaves custom domain data and the replay buffer into a unified training split.
4. Trains a lightweight second LoRA adapter ($r=16, \alpha=32$) on top of the merged standalone base model.
5. Exposes both a single-command CLI and full Cockpit WebUI support.

---

## 2. Goals

- **Existing Merged Model Required:** Target base model must be an existing local merged standalone model (or local directory) containing `model.safetensors`, `config.json`, `generation_config.json`, and processor files.
- **Custom Disk Ingestion:** Ingest custom speech datasets from disk supporting:
  - Audio folder with `metadata.csv` or `metadata.jsonl` (`file_name,text`).
  - Paired audio and transcript files (`*.wav` / `*.mp3` / `*.flac` + `*.txt`).
  - Pre-split (`train/`, `val/`) or automatic 90/10 train-eval split.
- **Balanced Replay Anchor:** Partition Neyshekar into formal and informal pools using `is_informal` and `PersianTextNormalizer.is_colloquial()`, and sample an exact 50/50 balance (default: 500 formal + 500 informal = 1,000 replay clips).
- **Audio Validation:** Filter all audio clips strictly to $0.5\text{s} \le \text{duration} \le 30.0\text{s}$ (no truncating). Resample to 16 kHz mono.
- **Interleaving:** Blend domain custom data and the replay buffer at a configurable ratio (default: 75% domain, 25% replay).
- **Local Model Label Support:** Update `scripts/train_lora.py`'s `_label_format()` so local model directories load their `generation_config.json` directly from disk, preserving BuzzASR's Persian prompt token (`<|fa|>`) and EOS token IDs.
- **Single-Command CLI:** Provide `scripts/train_domain.py` that prepares the data and invokes the trainer in one command.
- **Cockpit WebUI Support:** Add domain training fields to the web dashboard and `/api/train/start` endpoint, enabling users to launch domain runs directly from the browser with live telemetry and logs.
- **Comprehensive Tests:** Cover data ingestion, formal/informal balancing, local label resolution, CLI flags, server endpoint validation, and UI integration.

---

## 3. Non-goals

- Retraining Neyshekar from scratch.
- Merging the second domain LoRA automatically inside the training loop (users can still merge via `scripts/merge_lora.py` if desired).
- Web-based file uploads of large audio archives (paths on disk are used).
- Modifying telephony profiles or changing the underlying `PersianASRDataPipeline` feature extraction schema (`input_features`, `labels`, `is_informal`).

---

## 4. Anti-goals

- Overwriting or modifying the base merged model weights during training.
- Training on custom domain data alone without general Persian replay (guaranteeing catastrophic forgetting).
- Imbalanced replay sampling (e.g., drawing only formal or only informal Neyshekar samples).
- Falling back to wrong tokenizer token IDs when training from a local BuzzASR merged model.
- Truncating audio clips over 30 seconds rather than dropping them (which causes Whisper hallucination).

---

## 5. Constraints

- **Compute/Hardware:** NVIDIA GPU (e.g. L40S 48 GB) or CUDA/CPU test environment.
- **Model Architecture:** Whisper Large v3 / BuzzASR architecture (128 mel bins).
- **LoRA Size:** Tuned for small 2–3 hour datasets: $r=16, \alpha=32, \text{dropout}=0.05$ (unlike $r=64, \alpha=128$ for full Neyshekar).
- **Base Model Verification:** If `--base_model` is not an existing directory or is missing weights/config, halt with an explicit actionable error.
- **Disk Caching:** Dataset preparation must respect `prep_cache` fingerprinting or output staging to prevent corrupt cache collisions.

---

## 6. Approaches Considered

### Approach A: Two-Step Pipeline (Dataset Builder + Trainer)
Run a script to build an Arrow dataset, then pass `--dataset_name <dir>` to `scripts/train_lora.py`.
- *Pros:* Modular.
- *Cons:* Multi-step friction for users; does not fulfill the requirement for a single command or easy WebUI trigger.

### Approach B: Single-Command Ingestion & Training with WebUI Support *(Selected)*
1. Core ingestion & replay module `src/domain_data.py`.
2. Local model label format support in `scripts/train_lora.py`.
3. Standalone orchestrator CLI `scripts/train_domain.py`.
4. WebUI backend `/api/train/start` payload expansion and frontend fields.
- *Why Selected:* Combines modularity in code with zero-friction UX in both CLI and Web dashboard, ensuring reproducibility and safety.

---

## 7. Detailed Component Design

### 7.1. Domain Data Engine (`src/domain_data.py`)

#### Custom Ingestion:
1. `scan_custom_audio_dir(data_path: Path) -> List[Dict[str, Any]]`:
   - Checks for `metadata.csv` / `metadata.jsonl` (reads columns `file_name` / `audio` and `text` / `transcript`).
   - If not found, scans directory for paired files (`name.wav` + `name.txt`).
   - Resolves full audio paths, casts audio array, and validates $0.5\text{s} \le \text{duration} \le 30.0\text{s}$.
2. Split strategy:
   - If data directory contains explicit `train/` and `val/` or `validation/` subfolders, respect them.
   - Otherwise, partitions custom data into 85% train, 15% evaluation.

#### Balanced Neyshekar Replay Buffer:
1. `sample_balanced_neyshekar(sample_count: int = 1000, seed: int = 42) -> Dataset`:
   - Loads `shekar-ai/Neyshekar` train split.
   - Determines informality via `informal` / `is_informal` column or `PersianTextNormalizer.is_colloquial(text)`.
   - Filters into `formal_pool` and `informal_pool`.
   - Samples `sample_count // 2` from each pool deterministically.
   - Returns concatenated balanced replay dataset.

#### Interleaving & Output:
1. `build_domain_dataset(custom_path: str, replay_samples: int = 1000, replay_ratio: float = 0.25, output_dir: Optional[str] = None, seed: int = 42) -> DatasetDict`:
   - Interleaves custom train and replay train using `interleave_datasets(..., probabilities=[1 - replay_ratio, replay_ratio], seed=seed, stopping_strategy="all_exhausted")`.
   - Builds `validation` split containing custom val + a proportional 50/50 sample from Neyshekar validation.
   - Supplies `test` split: Evaluates against the standard `shekar-ai/Neyshekar` `test` split. This ensures epoch benchmarks and post-training evaluation directly monitor whether the general Persian baseline regressed (catastrophic forgetting guard).
   - Standardizes split keys to `"train"`, `"validation"`, and `"test"`.
   - If `output_dir` provided, saves to disk (`DatasetDict.save_to_disk`).

---

### 7.2. Trainer Local Merged Model Support (`scripts/train_lora.py` & `src/data.py`)

#### Transparent Local Dataset Loading (`src/data.py`):
Update `PersianASRDataPipeline.load_raw_dataset()`:
```python
if os.path.isdir(self.dataset_name):
    # Local DatasetDict saved via save_to_disk
    ds_dict = load_from_disk(self.dataset_name)
    ds = ds_dict[split]
else:
    ds = load_dataset(self.dataset_name, split=split, **_ld_kwargs)
```
This guarantees any local dataset path works natively without needing new CLI arguments or changing downstream pipeline contracts.

#### Local Generation Config & Token IDs (`scripts/train_lora.py`):
Updated logic:
```python
def _label_format(base_model: str):
    # Support local merged directories as well as Hub identifier
    gen_cfg_path = None
    if os.path.isdir(base_model):
        candidate = os.path.join(base_model, "generation_config.json")
        if os.path.exists(candidate):
            gen_cfg_path = candidate
    elif base_model == "BuzzASR/persian":
        from huggingface_hub import hf_hub_download
        gen_cfg_path = hf_hub_download(base_model, "generation_config.json")

    if gen_cfg_path and os.path.exists(gen_cfg_path):
        with open(gen_cfg_path, encoding="utf-8") as f:
            return persian_label_ids(json.load(f))
    return None, None
```
This guarantees that any model merged from BuzzASR (or containing its generation config) uses the correct token IDs `((51908, 51968, 51972), 51865)`.

---

### 7.3. Single-Command CLI (`scripts/train_domain.py`)

Arguments:
- `--base_model`: Path to merged standalone model directory (required, validated).
- `--custom_data_path`: Path to custom audio directory on disk (required, validated).
- `--output_dir`: Output directory for checkpoints & adapter (default: `outputs/buzzasr-domain-lora`).
- `--results_dir`: Results/telemetry directory.
- `--replay_samples`: Number of Neyshekar replay samples (default: 1000).
- `--replay_ratio`: Replay mixing probability (default: 0.25).
- `--epochs`: Training epochs (default: 3).
- `--learning_rate`: Peak LR (default: 2e-5).
- `--lora_r`: LoRA rank (default: 16).
- `--lora_alpha`: LoRA alpha (default: 32).
- `--batch_size`: Batch size (default: 8).
- `--grad_accum`: Gradient accumulation (default: 4).
- `--telephony_mode`: Telephony mode (`paired`, `disjoint`, or `clean`; default: `paired`).
- `--fresh_start`: Start fresh.

Flow:
1. Validates base model directory (`model.safetensors` or `pytorch_model.bin`, `config.json`).
2. Generates/loads prepared domain dataset into a run-scoped directory via `src/domain_data.py`.
3. Invokes `scripts/train_lora.py` with `--base_model <local_dir>`, `--dataset_name <prepared_dir>`, and domain hyperparameters.

---

### 7.4. WebUI Cockpit Integration (`server/app.py`, `server/static/`)

#### Backend (`server/app.py`):
1. Expand `_normalize_start_config`:
   - Accept optional `domain_adaptation: bool` (default False).
   - If true:
     - `custom_data_path`: Must be a valid existing local directory.
     - `base_model`: Must be a valid existing local directory (default `models/buzzasr-neyshekar-standalone`).
     - `replay_samples`: Integer between 100 and 5000 (default 1000).
     - Default `learning_rate` to `2e-5`, `epochs` to 3, `lora_r` to 16, `lora_alpha` to 32.
2. In `_start_new_run`:
   - Non-blocking execution: `_start_new_run` launches `scripts/train_domain.py` as the child process rather than running data preparation synchronously in the HTTP handler.
   - The subprocess immediately writes to `training_live.log`, sets the run stage to `"preflight"` (with detail `"preparing domain dataset"`), and returns the HTTP 200 response immediately.
   - If `ASR_FAKE_TRAINER=1` (test mode), spawns the fake trainer as normal.

#### Frontend (`server/static/index.html` & `app.js`):
1. Add a collapsible/toggle section: **"Domain Adaptation (2–3h Custom Data)"**.
2. Inputs:
   - Checkbox: "Enable Domain Adaptation"
   - Input: "Custom audio path" (e.g. `/data/my_clips`)
   - Input: "Merged model path" (default `models/buzzasr-neyshekar-standalone`)
   - Input: "Replay samples" (default `1000`)
3. When checked, updates default LR to `0.00002` and Epochs to `3`.
4. Submits payload to `/api/train/start`.

---

## 8. Verification & Testing

1. `tests/test_domain_data.py`:
   - Test scanning metadata CSV, metadata JSONL, and paired WAV+TXT files.
   - Test duration filtering drops $<0.5\text{s}$ and $>30\text{s}$ clips.
   - Test formal/informal partitioning produces equal 50/50 replay splits.
   - Test dataset interleaving maintains correct label and feature schemas.
2. `tests/test_train_local_model.py`:
   - Test `_label_format` reads local directory `generation_config.json` correctly.
   - Test error handling when local `base_model` path does not exist.
3. `tests/test_train_domain_cli.py`:
   - Test `scripts/train_domain.py` argument parsing and preflight checks.
4. `tests/test_server_domain_train.py`:
   - Test `/api/train/start` validates `custom_data_path` and `base_model` paths.
   - Test cockpit payload handles domain adaptation config.

---

## 9. What's Left to the Builder

- Standard UI CSS styling consistent with existing Cockpit theme.
- Log message formatting for dataset preparation steps.
- Temporary directory cleanup during dataset creation.
