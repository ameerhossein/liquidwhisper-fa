# Domain Adaptation on Merged Standalone Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a single-command domain adaptation workflow and Cockpit WebUI interface that trains a lightweight second LoRA adapter on 2–3 hours of custom audio targeting an existing merged BuzzASR+Neyshekar model, using a balanced formal/informal Neyshekar replay buffer to prevent catastrophic forgetting.

**Architecture:** 
A dedicated domain data ingestion engine (`src/domain_data.py`) scans custom audio and transcripts from disk, samples an exact 50/50 mix of formal and informal clips from Neyshekar, and interleaves them into a combined dataset. `scripts/train_lora.py` and `src/data.py` are enhanced to load local datasets and read BuzzASR generation config token IDs from local model directories. A CLI orchestrator (`scripts/train_domain.py`) and WebUI Cockpit integration (`server/app.py`, `server/static/`) allow running the full pipeline in one command or from the browser.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face `transformers`, `datasets`, `peft`, FastAPI, vanilla JS / WebSockets.

**Spec:** `docs/superpowers/specs/2026-11-01-domain-adaptation-design.md`

## Global Constraints

- Target base model must be an existing local merged standalone model containing `model.safetensors` (or `pytorch_model.bin`), `config.json`, and `generation_config.json`.
- Ingested audio must be resampled to 16 kHz mono and strictly filtered to $0.5\text{s} \le \text{duration} \le 30.0\text{s}$ (clips exceeding 30s are dropped, not truncated).
- The Neyshekar replay buffer must sample an exact 50/50 balance of formal and informal clips using the `is_informal` column with `PersianTextNormalizer.is_colloquial()` as fallback.
- Standard split keys must be `"train"`, `"validation"`, and `"test"`, with `"test"` pointing to Neyshekar's test split to monitor general Persian retention.
- Domain LoRA hyperparameters default to $r=16, \alpha=32, \text{lr}=2\times 10^{-5}, \text{epochs}=3$.
- WebUI dataset preparation must run asynchronously in the child process so the HTTP request returns immediately and streams logs.

## Review Focus

1. **Missing or invalid `--base_model` directory:** If `--base_model` does not exist or lacks `config.json`, the CLI and server must halt immediately with an explicit actionable error message rather than failing deep inside PyTorch/HF. (Covered in Task 4 & Task 5).
2. **Audio clip duration bounds:** Any audio clip $<0.5\text{s}$ or $>30.0\text{s}$ in custom data must be dropped and counted, never passed to the trainer. (Covered in Task 2).
3. **Empty or missing transcripts:** Custom metadata missing transcript entries or containing empty text must be caught and excluded without crashing the scanner. (Covered in Task 2).
4. **Token ID mismatch on local BuzzASR models:** `_label_format()` must correctly locate `generation_config.json` in local directories and load BuzzASR prompt IDs `((51908, 51968, 51972), 51865)` rather than falling back to standard Whisper tokenizer IDs. (Covered in Task 1).
5. **WebUI non-blocking startup:** Starting domain training from the WebUI must return HTTP 200 within 1 second while dataset preparation progress streams to `training_live.log`. (Covered in Task 5).

---

### Task 1: Local Model Label Format & Transparent Local Dataset Loading

**Files:**
- Modify: `src/data.py:305-335`
- Modify: `scripts/train_lora.py:345-355`
- Test: `tests/test_local_model_and_dataset.py`

**Interfaces:**
- Consumes: `src.data.persian_label_ids`, `datasets.load_from_disk`, `datasets.load_dataset`
- Produces:
  - `_label_format(base_model: str) -> Tuple[Optional[Tuple[int, ...]], Optional[int]]` (accepts local dir or hub ID)
  - `PersianASRDataPipeline.load_raw_dataset(split: Optional[str] = None, max_samples: Optional[int] = None) -> Any` (supports local directories via `load_from_disk`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_local_model_and_dataset.py`:
```python
import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_label_format_reads_local_generation_config(tmp_path):
    from scripts.train_lora import _label_format
    
    # Create mock local model directory with generation_config.json
    gen_cfg = {
        "lang_to_id": {"<|fa|>": 51908},
        "task_to_id": {"transcribe": 51968},
        "no_timestamps_token_id": 51972,
        "eos_token_id": 51865,
    }
    model_dir = tmp_path / "mock_merged_model"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(json.dumps(gen_cfg), encoding="utf-8")
    
    prefix_ids, eos_id = _label_format(str(model_dir))
    assert prefix_ids == (51908, 51968, 51972)
    assert eos_id == 51865

def test_label_format_nonexistent_returns_none():
    from scripts.train_lora import _label_format
    prefix_ids, eos_id = _label_format("/nonexistent/model/path")
    assert prefix_ids is None
    assert eos_id is None

def test_load_raw_dataset_branches_to_load_from_disk(tmp_path):
    from src.data import PersianASRDataPipeline
    
    dataset_dir = tmp_path / "mock_dataset_dir"
    dataset_dir.mkdir()
    
    pipeline = PersianASRDataPipeline(processor=MagicMock(), dataset_name=str(dataset_dir))
    
    with patch("src.data.load_from_disk") as mock_load_from_disk:
        mock_ds_dict = {"train": MagicMock(), "validation": MagicMock()}
        mock_load_from_disk.return_value = mock_ds_dict
        
        res = pipeline.load_raw_dataset(split="train")
        mock_load_from_disk.assert_called_once_with(str(dataset_dir))
        assert res == mock_ds_dict["train"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_local_model_and_dataset.py -v`
Expected: FAIL because `_label_format` returns `(None, None)` for local directories, and `src.data.load_from_disk` is not imported or used.

- [ ] **Step 3: Implement local model label reading and disk dataset loading**

In `scripts/train_lora.py`:
Update `_label_format(base_model: str)`:
```python
def _label_format(base_model: str):
    """(prefix_ids, eos_id) for training labels, or (None, None) for tokenizer labels."""
    gen_cfg_path = None
    if os.path.isdir(base_model):
        candidate = os.path.join(base_model, "generation_config.json")
        if os.path.exists(candidate):
            gen_cfg_path = candidate
    elif base_model == "BuzzASR/persian":
        from huggingface_hub import hf_hub_download
        try:
            gen_cfg_path = hf_hub_download(base_model, "generation_config.json")
        except Exception:
            gen_cfg_path = None

    if gen_cfg_path and os.path.exists(gen_cfg_path):
        try:
            with open(gen_cfg_path, encoding="utf-8") as f:
                return persian_label_ids(json.load(f))
        except Exception:
            return None, None
    return None, None
```

In `src/data.py`:
Add `load_from_disk` to datasets import (lines 28–35).
In `PersianASRDataPipeline.load_raw_dataset()`:
```python
        if os.path.isdir(self.dataset_name):
            print(f"Loading local dataset from disk: '{self.dataset_name}' (split={split})...")
            if not _DATASETS_AVAILABLE or load_from_disk is None:
                raise ImportError("datasets package not installed; cannot load disk dataset.")
            ds_loaded = load_from_disk(self.dataset_name)
            if split:
                if isinstance(ds_loaded, (DatasetDict, dict)):
                    ds = ds_loaded[split]
                else:
                    ds = ds_loaded
            else:
                ds = ds_loaded
        else:
            print(f"Loading dataset '{self.dataset_name}' (split={split})...")
            _ld_kwargs: Dict[str, Any] = {}
            if _cache_dir:
                _ld_kwargs["cache_dir"] = _cache_dir
            try:
                ds = load_dataset(self.dataset_name, split=split, **_ld_kwargs)
            except Exception as e:
                print(f"Error loading {self.dataset_name}: {e}. Retrying with trust_remote_code=True...")
                ds = load_dataset(self.dataset_name, split=split, trust_remote_code=True, **_ld_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_local_model_and_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data.py scripts/train_lora.py tests/test_local_model_and_dataset.py
git commit -m "feat: support local model generation config and load_from_disk datasets"
```

---

### Task 2: Custom Audio Directory Scanner & Duration Filter

**Files:**
- Create: `src/domain_data.py`
- Test: `tests/test_domain_data_scan.py`

**Interfaces:**
- Consumes: `soundfile` / `wave` / `torchaudio` or stdlib `wave` for duration checking
- Produces:
  - `scan_custom_audio_dir(data_path: Union[str, Path]) -> List[Dict[str, Any]]`
  - `split_custom_dataset(records: List[Dict[str, Any]], eval_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_domain_data_scan.py`:
```python
import csv
import json
import wave
import pytest
from pathlib import Path

def _create_mock_wav(path: Path, duration_seconds: float, sample_rate: int = 16000):
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)

def test_scan_csv_metadata_filters_duration(tmp_path):
    from src.domain_data import scan_custom_audio_dir
    
    audio_dir = tmp_path / "audio_csv"
    audio_dir.mkdir()
    
    # 1 valid (3.0s), 1 too short (0.2s), 1 too long (35.0s)
    _create_mock_wav(audio_dir / "clip1.wav", 3.0)
    _create_mock_wav(audio_dir / "short.wav", 0.2)
    _create_mock_wav(audio_dir / "long.wav", 35.0)
    
    meta_file = audio_dir / "metadata.csv"
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerow(["clip1.wav", "سلام دنیا"])
        writer.writerow(["short.wav", "کوتاه"])
        writer.writerow(["long.wav", "طولانی"])
    
    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["audio"] == str(audio_dir / "clip1.wav")
    assert records[0]["text"] == "سلام دنیا"
    assert 2.9 <= records[0]["duration"] <= 3.1

def test_scan_paired_wav_txt(tmp_path):
    from src.domain_data import scan_custom_audio_dir
    
    audio_dir = tmp_path / "audio_paired"
    audio_dir.mkdir()
    
    _create_mock_wav(audio_dir / "sample1.wav", 2.0)
    (audio_dir / "sample1.txt").write_text("متن نمونه اول", encoding="utf-8")
    
    _create_mock_wav(audio_dir / "sample2.wav", 4.0)
    (audio_dir / "sample2.txt").write_text("متن نمونه دوم", encoding="utf-8")
    
    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 2
    texts = {r["text"] for r in records}
    assert "متن نمونه اول" in texts
    assert "متن نمونه دوم" in texts

def test_split_custom_dataset():
    from src.domain_data import split_custom_dataset
    
    dummy_records = [{"audio": f"f{i}.wav", "text": f"t{i}"} for i in range(100)]
    train_recs, val_recs = split_custom_dataset(dummy_records, eval_ratio=0.15, seed=42)
    assert len(train_recs) == 85
    assert len(val_recs) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_domain_data_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.domain_data'`

- [ ] **Step 3: Implement scanner and split functions in `src/domain_data.py`**

In `src/domain_data.py`:
- Use `wave` (or `soundfile`/`torchaudio` fallback) to read duration without loading entire audio arrays into memory.
- Implement `_get_audio_duration(path: Path) -> float`.
- Implement `scan_custom_audio_dir(data_path: Union[str, Path], min_duration: float = 0.5, max_duration: float = 30.0) -> List[Dict[str, Any]]`:
  1. Check for `metadata.csv` or `metadata.jsonl`. If found, parse rows: detect audio column (`file_name`, `audio`, `path`) and text column (`text`, `transcript`, `sentence`). Resolve relative audio paths against `data_path`.
  2. If no metadata file, scan for audio files (`*.wav`, `*.mp3`, `*.flac`, `*.ogg`, `*.m4a`) with matching `.txt` file with same stem.
  3. Filter duration $0.5\text{s} \le \text{dur} \le 30.0\text{s}$ and non-empty text.
- Implement `split_custom_dataset(records: List[Dict[str, Any]], eval_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`:
  - Deterministic shuffle using `random.Random(seed)`.
  - Partition into `train` and `val`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_domain_data_scan.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain_data.py tests/test_domain_data_scan.py
git commit -m "feat: add custom audio directory scanning and duration filtering"
```

---

### Task 3: Balanced Neyshekar Replay Buffer & Interleaving

**Files:**
- Modify: `src/domain_data.py`
- Test: `tests/test_domain_replay_interleave.py`

**Interfaces:**
- Consumes: `PersianTextNormalizer`, `datasets.Dataset`, `datasets.interleave_datasets`, `datasets.DatasetDict`
- Produces:
  - `sample_balanced_neyshekar(neyshekar_ds: Dataset, sample_count: int = 1000, seed: int = 42) -> Dataset`
  - `build_domain_dataset(custom_path: Union[str, Path], replay_samples: int = 1000, replay_ratio: float = 0.25, output_dir: Optional[Union[str, Path]] = None, seed: int = 42) -> DatasetDict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_domain_replay_interleave.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset

def test_sample_balanced_neyshekar():
    from src.domain_data import sample_balanced_neyshekar
    
    # 20 formal rows, 20 informal rows
    data = []
    for i in range(20):
        data.append({"audio": {"path": f"formal_{i}.wav"}, "text": f"متن رسمی شماره {i}", "is_informal": False})
    for i in range(20):
        data.append({"audio": {"path": f"informal_{i}.wav"}, "text": f"می‌خوام برم خونه شماره {i}", "is_informal": True})
        
    ds = Dataset.from_list(data)
    
    # Sample 10 total: should get 5 formal and 5 informal
    sampled = sample_balanced_neyshekar(ds, sample_count=10, seed=42)
    assert len(sampled) == 10
    
    informal_count = sum(1 for row in sampled if row["is_informal"])
    formal_count = len(sampled) - informal_count
    assert informal_count == 5
    assert formal_count == 5

def test_build_domain_dataset_structure(tmp_path):
    from src.domain_data import build_domain_dataset
    
    # Mock scan_custom_audio_dir and load_dataset
    mock_records = [{"audio": f"custom_{i}.wav", "text": f"سفارشی {i}", "duration": 2.0} for i in range(20)]
    mock_neyshekar_train = Dataset.from_list([
        {"audio": {"path": f"n_{i}.wav"}, "text": f"متن {i}", "is_informal": (i % 2 == 0)} for i in range(40)
    ])
    mock_neyshekar_val = Dataset.from_list([
        {"audio": {"path": f"nv_{i}.wav"}, "text": f"ارزیابی {i}", "is_informal": False} for i in range(10)
    ])
    mock_neyshekar_test = Dataset.from_list([
        {"audio": {"path": f"nt_{i}.wav"}, "text": f"آزمون {i}", "is_informal": False} for i in range(10)
    ])
    
    with patch("src.domain_data.scan_custom_audio_dir", return_value=mock_records), \
         patch("src.domain_data.load_dataset") as mock_ld:
        def fake_load_dataset(name, split=None, **kwargs):
            if split == "train":
                return mock_neyshekar_train
            if split == "validation":
                return mock_neyshekar_val
            return mock_neyshekar_test
        mock_ld.side_effect = fake_load_dataset
        
        out_dir = tmp_path / "domain_dataset_out"
        ds_dict = build_domain_dataset(
            custom_path=str(tmp_path / "custom"),
            replay_samples=10,
            replay_ratio=0.25,
            output_dir=out_dir,
            seed=42,
        )
        assert "train" in ds_dict
        assert "validation" in ds_dict
        assert "test" in ds_dict
        assert out_dir.is_dir()
        assert (out_dir / "dataset_dict.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_domain_replay_interleave.py -v`
Expected: FAIL with `ImportError: cannot import name 'sample_balanced_neyshekar'`

- [ ] **Step 3: Implement balanced replay sampling and dataset building**

In `src/domain_data.py`:
- `sample_balanced_neyshekar(neyshekar_ds: Dataset, sample_count: int = 1000, seed: int = 42) -> Dataset`:
  - Initialize `normalizer = PersianTextNormalizer()`
  - Partition into indices `formal_indices` and `informal_indices`. Use `row["informal"]` / `row["is_informal"]` if present, else `normalizer.is_colloquial(row.get("text", ""))`.
  - Draw `target_each = sample_count // 2` from each pool using `random.Random(seed).sample`.
  - Combine selected rows into single `Dataset`.
- `build_domain_dataset(...) -> DatasetDict`:
  - Scan custom directory with `scan_custom_audio_dir`. Split 85/15 train/val.
  - Convert custom records to Hugging Face `Dataset` with `Audio(sampling_rate=16000)`.
  - Load `shekar-ai/Neyshekar` train, validation, and test splits.
  - Draw balanced replay samples from Neyshekar train.
  - Interleave custom train and replay train with `interleave_datasets(..., probabilities=[1 - replay_ratio, replay_ratio], seed=seed, stopping_strategy="all_exhausted")`.
  - Form validation split: concatenate custom validation with a proportional small sample of Neyshekar validation.
  - Form test split: keep Neyshekar test split intact.
  - Cast audio columns to `Audio(sampling_rate=16000)`.
  - Assemble `DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})`.
  - If `output_dir` provided, call `ds_dict.save_to_disk(str(output_dir))`. Return `ds_dict`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_domain_replay_interleave.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain_data.py tests/test_domain_replay_interleave.py
git commit -m "feat: implement balanced formal/informal replay sampling and domain dataset interleaving"
```

---

### Task 4: Single-Command CLI Orchestrator (`scripts/train_domain.py`)

**Files:**
- Create: `scripts/train_domain.py`
- Test: `tests/test_train_domain_cli.py`

**Interfaces:**
- Consumes: `src.domain_data.build_domain_dataset`, `scripts.train_lora.build_parser`, `scripts.train_lora._run_training`
- Produces: Single CLI entrypoint `python scripts/train_domain.py [options]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_domain_cli.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_train_domain_cli_validates_base_model(tmp_path):
    from scripts.train_domain import parse_args, validate_inputs
    
    # Missing base model should raise SystemExit or ValueError
    with pytest.raises((SystemExit, ValueError)):
        validate_inputs(
            base_model="/nonexistent/model",
            custom_data_path=str(tmp_path),
        )

def test_train_domain_cli_validates_custom_path(tmp_path):
    from scripts.train_domain import validate_inputs
    
    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    
    with pytest.raises((SystemExit, ValueError)):
        validate_inputs(
            base_model=str(model_dir),
            custom_data_path="/nonexistent/custom/path",
        )

def test_train_domain_cli_orchestration(tmp_path):
    from scripts.train_domain import run_domain_adaptation
    
    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    
    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()
    
    with patch("scripts.train_domain.build_domain_dataset") as mock_build, \
         patch("scripts.train_domain._run_training_subprocess") as mock_run:
        
        run_domain_adaptation(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
            output_dir=str(tmp_path / "out"),
            results_dir=str(tmp_path / "res"),
            replay_samples=500,
            replay_ratio=0.25,
            epochs=3,
            learning_rate=2e-5,
            lora_r=16,
            lora_alpha=32,
        )
        mock_build.assert_called_once()
        mock_run.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_train_domain_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.train_domain'`

- [ ] **Step 3: Implement `scripts/train_domain.py`**

In `scripts/train_domain.py`:
- Implement `parse_args()` using `argparse.ArgumentParser`:
  - `--base_model` (required, path to merged standalone model)
  - `--custom_data_path` (required, path to custom audio directory)
  - `--output_dir` (default: `outputs/buzzasr-domain-lora`)
  - `--results_dir` (default: `results/domain-lora`)
  - `--run_dir` (optional persistent run directory)
  - `--replay_samples` (default: 1000)
  - `--replay_ratio` (default: 0.25)
  - `--epochs` (default: 3)
  - `--learning_rate` (default: 2e-5)
  - `--lora_r` (default: 16)
  - `--lora_alpha` (default: 32)
  - `--batch_size` (default: 8)
  - `--grad_accum` (default: 4)
  - `--telephony_mode` (default: `paired`)
  - `--fresh_start` (flag)
- Implement `validate_inputs(base_model: str, custom_data_path: str)`:
  - Check `Path(base_model).is_dir()` and `(Path(base_model) / "config.json").exists()`. If invalid, raise `ValueError` with clear guidance.
  - Check `Path(custom_data_path).is_dir()`. If invalid, raise `ValueError`.
- Implement `run_domain_adaptation(args)`:
  - Prepare dataset into `<output_dir>/prepared_domain_dataset` (or `<run_dir>/dataset`).
  - Assemble CLI arguments for `scripts/train_lora.py` with `--base_model <base_model>`, `--dataset_name <prepared_dataset_path>`, and domain hyperparameters.
  - Invoke `train_lora.py` directly (or via subprocess when launched from CLI).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_train_domain_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_domain.py tests/test_train_domain_cli.py
git commit -m "feat: implement single-command train_domain.py CLI orchestrator"
```

---

### Task 5: Cockpit WebUI Backend Integration

**Files:**
- Modify: `server/app.py:2209-2280`
- Modify: `server/app.py:2455-2560`
- Modify: `scripts/dev_fake_trainer.py`
- Test: `tests/test_server_domain_train.py`

**Interfaces:**
- Consumes: `_normalize_start_config`, `_start_new_run`, `scripts.train_domain`
- Produces: API support for `domain_adaptation` in `POST /api/train/start`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_domain_train.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

def test_normalize_start_config_domain_adaptation(tmp_path):
    from server.app import _normalize_start_config
    
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    
    cfg = {
        "domain_adaptation": True,
        "base_model": str(model_dir),
        "custom_data_path": str(custom_dir),
        "replay_samples": 800,
    }
    norm = _normalize_start_config(cfg)
    assert norm["domain_adaptation"] is True
    assert norm["base_model"] == str(model_dir)
    assert norm["custom_data_path"] == str(custom_dir)
    assert norm["replay_samples"] == 800
    assert norm["learning_rate"] == 2e-5
    assert norm["epochs"] == 3

def test_normalize_start_config_domain_rejects_missing_paths():
    from server.app import _normalize_start_config
    from fastapi import HTTPException
    
    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": "/nonexistent/model",
            "custom_data_path": "/nonexistent/custom",
        })
    assert exc.value.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_server_domain_train.py -v`
Expected: FAIL because `domain_adaptation` is not recognized and defaults are not applied.

- [ ] **Step 3: Implement domain adaptation configuration handling in `server/app.py`**

In `server/app.py`:
1. In `_normalize_start_config(config: Dict[str, Any])`:
   - Detect `domain_mode = bool(config.get("domain_adaptation", False))`.
   - If `domain_mode`:
     - Validate `base_model = str(config.get("base_model", "models/buzzasr-neyshekar-standalone")).strip()`. Must be an existing directory with `config.json`.
     - Validate `custom_data_path = str(config.get("custom_data_path", "")).strip()`. Must be an existing directory.
     - Parse `replay_samples = int(config.get("replay_samples", 1000))` (range 100..5000).
     - Default `learning_rate` to `2e-5` (if not explicitly overridden).
     - Default `epochs` to `3` (if not explicitly overridden).
     - Default `lora_r` to 16, `lora_alpha` to 32.
2. In `_start_new_run(config: Dict[str, Any])`:
   - If `values.get("domain_adaptation")`:
     - Build command pointing to `scripts/train_domain.py` (or `scripts/dev_fake_trainer.py` when `_fake_trainer_enabled()`).
     - Pass `--base_model`, `--custom_data_path`, `--replay_samples`, `--run_dir`, and `--results_dir`.
   - In `scripts/dev_fake_trainer.py`:
     - Add argument parsing support for `--custom_data_path`, `--replay_samples`, `--replay_ratio` so fake runs in tests succeed without error.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_server_domain_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/app.py scripts/dev_fake_trainer.py tests/test_server_domain_train.py
git commit -m "feat: integrate domain adaptation training into server backend"
```

---

### Task 6: Cockpit WebUI Frontend Controls

**Files:**
- Modify: `server/static/index.html:114-165`
- Modify: `server/static/app.js:1768-1815`
- Test: `tests/test_ui_domain_adaptation.py`

**Interfaces:**
- Consumes: `POST /api/train/start` payload
- Produces: Cockpit UI elements (`#chkDomainAdaptation`, `#fieldDomainSection`, `#inputCustomDataPath`, `#inputMergedModelPath`, `#inputReplaySamples`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ui_domain_adaptation.py`:
```python
from pathlib import Path

def test_index_html_has_domain_adaptation_elements():
    html_file = Path("server/static/index.html")
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    
    assert 'id="chkDomainAdaptation"' in content
    assert 'id="inputCustomDataPath"' in content
    assert 'id="inputMergedModelPath"' in content
    assert 'id="inputReplaySamples"' in content

def test_app_js_includes_domain_adaptation_in_payload():
    js_file = Path("server/static/app.js")
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")
    
    assert "domain_adaptation" in content
    assert "custom_data_path" in content
    assert "base_model" in content
    assert "replay_samples" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_ui_domain_adaptation.py -v`
Expected: FAIL because HTML and JS lack domain adaptation elements.

- [ ] **Step 3: Implement HTML and JS controls**

In `server/static/index.html`:
Inside `<form class="config">`:
Add a collapsible or styled subsection:
```html
<div class="domain-config-toggle">
  <label class="field-checkbox">
    <input type="checkbox" id="chkDomainAdaptation">
    <span class="field-label-inline">Domain Adaptation (2–3h custom data)</span>
  </label>
</div>
<div id="domainSection" class="domain-section" hidden>
  <label class="field">
    <span class="field-label">Custom audio path (on disk)</span>
    <input type="text" id="inputCustomDataPath" class="field-input" placeholder="/path/to/custom_audio">
  </label>
  <label class="field">
    <span class="field-label">Merged base model path</span>
    <input type="text" id="inputMergedModelPath" class="field-input" value="models/buzzasr-neyshekar-standalone">
  </label>
  <label class="field">
    <span class="field-label">Replay samples (Neyshekar)</span>
    <input type="number" id="inputReplaySamples" class="field-input" value="1000" min="100" max="5000">
  </label>
</div>
```

In `server/static/app.js`:
- Bind `#chkDomainAdaptation` change event:
  - When checked: reveal `#domainSection`, set `el.inputLearningRate.value = "0.00002"`, set `el.inputEpochs.value = "3"`.
  - When unchecked: hide `#domainSection`, set `el.inputLearningRate.value = "0.0001"`, set `el.inputEpochs.value = "2"`.
- In `startRun(freshStart)`:
  - If `chkDomainAdaptation.checked`:
    - Include `domain_adaptation: true`, `custom_data_path: el.inputCustomDataPath.value`, `base_model: el.inputMergedModelPath.value`, `replay_samples: parseInt(el.inputReplaySamples.value, 10) || 1000` in the JSON payload sent to `/api/train/start`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_ui_domain_adaptation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/static/index.html server/static/app.js tests/test_ui_domain_adaptation.py
git commit -m "feat: add domain adaptation controls to Cockpit WebUI"
```

---

### Task 7: End-to-End Integration Verification

**Files:**
- Test: `tests/test_domain_adaptation_e2e.py`

**Interfaces:**
- Verifies the full pipeline end-to-end using a mock custom audio directory and lightweight simulation.

- [ ] **Step 1: Write the end-to-end integration test**

Create `tests/test_domain_adaptation_e2e.py`:
```python
import os
import wave
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

def _create_mock_wav(path: Path, duration_seconds: float = 2.0, sample_rate: int = 16000):
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)

def test_domain_adaptation_end_to_end(tmp_path):
    # 1. Setup mock merged model
    model_dir = tmp_path / "mock_merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "generation_config.json").write_text(json.dumps({
        "lang_to_id": {"<|fa|>": 51908},
        "task_to_id": {"transcribe": 51968},
        "no_timestamps_token_id": 51972,
        "eos_token_id": 51865,
    }), encoding="utf-8")
    
    # 2. Setup mock custom dataset
    custom_dir = tmp_path / "mock_custom"
    custom_dir.mkdir()
    _create_mock_wav(custom_dir / "sample1.wav", 2.0)
    (custom_dir / "sample1.txt").write_text("اصطلاح تخصصی حوزه کاری", encoding="utf-8")
    
    # 3. Test CLI validate_inputs succeeds
    from scripts.train_domain import validate_inputs
    validate_inputs(base_model=str(model_dir), custom_data_path=str(custom_dir))
```

- [ ] **Step 2: Run all domain tests**

Run: `PYTHONPATH=. pytest tests/test_local_model_and_dataset.py tests/test_domain_data_scan.py tests/test_domain_replay_interleave.py tests/test_train_domain_cli.py tests/test_server_domain_train.py tests/test_ui_domain_adaptation.py tests/test_domain_adaptation_e2e.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_domain_adaptation_e2e.py
git commit -m "test: add end-to-end integration test for domain adaptation"
```
