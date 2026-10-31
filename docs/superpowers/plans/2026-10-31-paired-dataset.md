# Paired Clean + Phone Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every training utterance appear once clean and once through the customer phone channel, default to paired mode with 2 epochs, and store the prepared data in a direct-shard cache that peaks at ~170 GB.

**Architecture:** `src/data.py` gains `prepare_paired_dataset`: one duration filter, a clean `datasets.map` pass (`telephony_prob=0.0`), a phone `datasets.map` pass (`telephony_prob=1.0`, strict failures), then `concatenate_datasets`. Prepared shards are written directly into a staging directory by `datasets.map(cache_file_name=...)`; `src/prep_cache.py` validates and atomically publishes them with a metadata file, and `scripts/train_lora.py` loads them by concatenating `Dataset.from_file` shards. `--telephony_mode {paired,disjoint,clean}` defaults to `paired`; epochs default to 2; the cockpit, server, config, and launcher carry the mode.

**Tech Stack:** Python 3.12, `datasets` 3.6, Hugging Face `transformers`, PyTorch, FastAPI, vanilla JS, pytest, Node `assert` tests.

**Spec:** `docs/superpowers/specs/2026-10-31-paired-dataset-design.md`

## Global Constraints

- Work on `main`; do not push.
- All implementation and review subagents use `opencode-go/deepseek-v4.1-flash#max` unless the user explicitly overrides it.
- TDD: write the test first, run it and see it fail for the expected reason, implement, run it and see it pass.
- `--telephony_mode` choices are exactly `paired`, `disjoint`, `clean`; default is `paired`.
- Default epochs is 2 for paired; disjoint/clean use 4 for equal compute.
- Paired train rows are exactly `2 ×` the post-filter count (116,488 on the full Neyshekar v6 train split).
- Validation stays clean (`telephony_prob=0.0`); preflight, per-epoch benchmark, and final evaluation stay clean + phone and are not changed.
- Prep cache format is `shards_v1`; the direct shard cache must peak at ~170 GB, not ~340 GB.
- Free-space guard requires at least 180 GB projected free (current free + reclaimable stale cache) before the first paired map; the target overlay has ~250 GB+ free.
- Paired preparation fails closed: no phone module, or any phone augmentation error, aborts rather than caching clean rows.
- No production fallback baseline values; do not touch `configs/baselines.json` (it is already deleted).
- Do not modify unrelated untracked files (`docs/superpowers/specs/2026-10-30-stateful-cockpit-design.md`).
- Run the full suite before every commit: `.venv/bin/python -m pytest -q`.

## Review Focus

These are the input classes most likely to bite a real user and are not fully exercised by the obvious tests. Each has a test in the owning task.

1. A crashed or interrupted map leaves shard files but no valid metadata — loading must miss and the staging directory must be cleaned, never partially loaded. (Task 3 tests `test_load_missing_meta_is_miss` and `test_staging_cleanup`.)
2. A stale disjoint cache exists when the first paired run starts — the free-space guard must count its reclaimable bytes, and the old cache must be removed before mapping. (Task 3 `test_ensure_space_counts_reclaimable`; the Task 5 flow removes the cache after the guard and before the map.)
3. Metadata is valid but a shard is truncated or replaced — loading must miss, not train on corrupt features. (Task 3 `test_load_wrong_shard_size_is_miss`.)
4. `--telephony_prob` changes while mode is `paired` — the paired fingerprint must not change, so the 170 GB cache is not rebuilt for an ignored flag. (Task 2 `test_paired_fingerprint_ignores_telephony_prob`.)
5. The telephone module is unavailable or one row fails degradation — paired preparation must abort, not silently cache a clean row in the phone half. (Task 1 `test_prepare_paired_dataset_requires_telephone` and `test_prepare_paired_dataset_fails_closed_on_degradation_error`.)

---

### Task 1: Paired dataset construction

**Files:**
- Modify: `src/data.py` (`PersianASRDataPipeline`, around L225-517)
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Consumes: `PersianASRDataPipeline._prepare_split`, `_filter_by_duration`, `prepare_sample`, `_TELEPHONE_AVAILABLE`, `apply_random_telephone`.
- Produces:
  - `prepare_paired_dataset(self, dataset: Dataset, remove_columns: Optional[list] = None) -> Dataset`
  - `_prepare_split(..., filter_durations: bool = True)`
  - `self.strict_telephony: bool` (default `False`)
  - `prepare_dataset_split` behavior is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_prep.py`:

```python
def _pairing_dataset(n=6):
    datasets = pytest.importorskip("datasets")
    return datasets.Dataset.from_dict({
        "audio": [_clip() for _ in range(n)],
        "text": [f"سلام {i}" for i in range(n)],
    })


def test_prepare_paired_dataset_pairs_each_utterance():
    ds = _pairing_dataset(6)
    out = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    assert len(out) == 12
    assert out["labels"][:6] == out["labels"][6:]
    assert out["is_informal"][:6] == out["is_informal"][6:]
    assert any(
        np.asarray(a).tobytes() != np.asarray(b).tobytes()
        for a, b in zip(out["input_features"][:6], out["input_features"][6:])
    )


def test_prepare_paired_dataset_clean_half_matches_clean_pipeline():
    ds = _pairing_dataset(4)
    paired = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    clean = _pipeline(prob=0.0).prepare_dataset_split(ds)
    for a, b in zip(paired["input_features"][:4], clean["input_features"]):
        np.testing.assert_array_equal(a, b)


def test_prepare_paired_dataset_deterministic_across_num_proc():
    datasets = pytest.importorskip("datasets")
    ds = _pairing_dataset(8)
    datasets.disable_caching()
    try:
        a = _pipeline(num_proc=1).prepare_paired_dataset(ds)
        b = _pipeline(num_proc=2).prepare_paired_dataset(ds)
    finally:
        datasets.enable_caching()
    for x, y in zip(a["input_features"][8:], b["input_features"][8:]):
        np.testing.assert_array_equal(x, y)


def test_prepare_paired_dataset_filters_once():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({
        "audio": [_clip(dur=0.1), _clip(dur=1.0), _clip(dur=1.0)],
        "text": ["a", "b", "c"],
    })
    out = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    assert len(out) == 4


def test_prepare_paired_dataset_requires_telephone(monkeypatch):
    ds = _pairing_dataset(2)
    monkeypatch.setattr("src.data._TELEPHONE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="telephone"):
        _pipeline(prob=0.5).prepare_paired_dataset(ds)


def test_prepare_paired_dataset_fails_closed_on_degradation_error(monkeypatch):
    ds = _pairing_dataset(2)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated codec failure")

    monkeypatch.setattr("src.data.apply_random_telephone", boom)
    with pytest.raises(RuntimeError, match="simulated codec failure"):
        _pipeline(prob=0.5).prepare_paired_dataset(ds)


class _MismatchedPipeline(PersianASRDataPipeline):
    def prepare_sample(self, batch, index=None):
        out = super().prepare_sample(batch, index=index)
        if self.telephony_prob >= 1.0:
            out["input_features"] = "not-a-feature"
        return out


def test_prepare_paired_dataset_rejects_schema_mismatch():
    pipe = _MismatchedPipeline(
        processor=_StubProcessor(),
        normalizer=_StubNormalizer(),
        telephony_prob=0.5,
        telephony_profiles=["mobile"],
        telephony_use_ffmpeg=False,
    )
    with pytest.raises(ValueError, match="schema"):
        pipe.prepare_paired_dataset(_pairing_dataset(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -q`
Expected: FAIL with `AttributeError: 'PersianASRDataPipeline' object has no attribute 'prepare_paired_dataset'`.

- [ ] **Step 3: Implement the paired preparation**

In `src/data.py`:

1. Add `import copy` and import `concatenate_datasets` from `datasets`.
2. Add `self.strict_telephony = False` in `PersianASRDataPipeline.__init__`.
3. In `prepare_sample`, when `not _TELEPHONE_AVAILABLE` and `self.strict_telephony`, raise `RuntimeError("telephone augmentation unavailable")` instead of warning. In the `except Exception` around `apply_random_telephone`, when `self.strict_telephony`, raise `RuntimeError(f"Telephone augmentation failed for row {index}: {e}") from e`; otherwise keep the warn-and-fallback behavior.
4. Extract the effective-`num_proc` resolution (the current CUDA-fork guard) into `_resolve_num_proc(self) -> int`. Keep the existing `_set_torch_threads(1)` / restore wrapper.
5. Add `filter_durations: bool = True` to `_prepare_split` and wrap the existing `_filter_by_duration` call in that flag.
6. Add `prepare_paired_dataset(self, dataset, remove_columns=None) -> Dataset`:
   - Raise `RuntimeError("paired mode requires the telephone module")` if `not _TELEPHONE_AVAILABLE`.
   - Resolve the effective process count, set the torch thread cap, and restore it in `finally` exactly like `prepare_dataset_split`.
   - Filter durations once: `filtered = self._filter_by_duration(dataset, effective_proc)`.
   - Build `clean_pipe = copy.copy(self)`; set `clean_pipe.telephony_prob = 0.0`; set `clean_pipe.strict_telephony = False`.
   - Build `phone_pipe = copy.copy(self)`; set `phone_pipe.telephony_prob = 1.0`; set `phone_pipe.strict_telephony = True`.
   - Map each with `_prepare_split(filtered, remove_columns, effective_proc, filter_durations=False)`.
   - If `clean.features != phone.features`, raise `ValueError("clean and phone feature schemas differ")`.
   - Return `concatenate_datasets([clean, phone])`.
7. Do not add `cache_dir` in this task; that is Task 4.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data_prep.py
git commit -m "feat: add paired clean+phone dataset preparation"
```

---

### Task 2: Telephony mode, defaults, and paired selection

**Files:**
- Modify: `scripts/train_lora.py` (`parse_args` around L85-160, `_apply_yaml_config` around L154-234, `_prep_fingerprint` around L281-300, `_cli_run_config` around L572-608, prep flow around L1169-1202)
- Modify: `configs/l40s_train_config.yaml`
- Modify: `run_train.sh`
- Test: `tests/test_script_defaults.py`, `tests/test_train_lora_run_stages.py`

**Interfaces:**
- Consumes: Task 1's `prepare_paired_dataset`; existing `prepare_dataset_split`.
- Produces:
  - `--telephony_mode {paired,disjoint,clean}` (default `paired`).
  - `--epochs` default `2`.
  - `_prep_fingerprint(args, split: str, max_samples, telephony_mode: str, telephony_prob: float) -> Dict[str, Any]`; the dict includes `"telephony_mode"` and `"telephony_prob": None` when mode is `paired`.
  - `_cli_run_config` includes `telephony_mode`.

**Intermediate-state note:** this task keeps the existing `save_to_disk` prep cache so the mode/defaults can be reviewed independently. Task 5 replaces it with the direct-shard cache. Do not run a full paired prep on the 250 GB target until Task 5 is complete; it would still hit the old ~340 GB peak.

- [ ] **Step 1: Update the failing tests**

In `tests/test_script_defaults.py` (keep the existing batch/grad/eval/save/gradient/max-benchmark/preflight assertions; these snippets show the new or changed assertions):

```python
def test_train_lora_parser_matches_fast_l40s_defaults(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    assert args.epochs == 2
    assert args.telephony_mode == "paired"
    assert args.telephony_prob == 0.5
    assert args.lr_scheduler_type == "cosine"


def test_l40s_config_matches_fast_defaults():
    import yaml
    path = ROOT / "configs" / "l40s_train_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["training"]["epochs"] == 2
    assert cfg["telephony"]["mode"] == "paired"
    assert cfg["telephony"]["prob"] == 0.5


def test_run_train_script_matches_fast_defaults():
    text = (ROOT / "run_train.sh").read_text(encoding="utf-8")
    for expected in ("--epochs 2", "--telephony_mode paired"):
        assert expected in text
```

Add:

```python
def test_prep_fingerprint_includes_mode(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _prep_fingerprint, parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    paired = _prep_fingerprint(args, "train", None, "paired", 0.5)
    disjoint = _prep_fingerprint(args, "train", None, "disjoint", 0.5)
    clean = _prep_fingerprint(args, "train", None, "clean", 0.0)
    assert paired["telephony_mode"] == "paired"
    assert paired["telephony_prob"] is None
    assert paired != disjoint and disjoint != clean and paired != clean

    args.telephony_mode = "paired"
    validation_a = _prep_fingerprint(args, "validation", None, "clean", 0.0)
    args.telephony_mode = "disjoint"
    validation_b = _prep_fingerprint(args, "validation", None, "clean", 0.0)
    assert validation_a == validation_b


def test_paired_fingerprint_ignores_telephony_prob(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _prep_fingerprint, parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    assert _prep_fingerprint(args, "train", None, "paired", 0.5) == _prep_fingerprint(
        args, "train", None, "paired", 0.9
    )
```

In `tests/test_train_lora_run_stages.py`, add `telephony_mode="paired"` to the test `args` namespace and add a `prepare_paired_dataset` method to `_FakePipeline` that raises `AssertionError("cached prep must skip re-prep")` (same as `prepare_dataset_split`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py tests/test_train_lora_run_stages.py -q`
Expected: FAIL on `args.telephony_mode`, `args.epochs == 2`, config mode, launcher flags.

- [ ] **Step 3: Implement the mode and defaults**

In `scripts/train_lora.py`:

1. In `parse_args`, add:
   ```python
   parser.add_argument(
       "--telephony_mode",
       choices=("paired", "disjoint", "clean"),
       default="paired",
       help="paired=clean+phone rows (2 epochs), disjoint=50/50 coin, clean=no augmentation",
   )
   ```
   Change `--epochs` default from `4` to `2`.
2. In `_apply_yaml_config`, add `"mode": "telephony_mode"` and `"telephony_mode": "telephony_mode"` to `key_map`.
3. In `_cli_run_config`, add `"telephony_mode"` to `keys`.
4. Change `_prep_fingerprint` to:
   ```python
   def _prep_fingerprint(args, split: str, max_samples, telephony_mode: str, telephony_prob: float) -> Dict[str, Any]:
   ```
   Keep the existing fields; add `"telephony_mode": telephony_mode` and set `"telephony_prob": None if telephony_mode == "paired" else float(telephony_prob or 0.0)`.
5. In the prep flow, set `mode = args.telephony_mode`; build the train pipeline with `telephony_prob=0.0 if mode != "disjoint" else float(args.telephony_prob or 0.0)`.
6. Update the fingerprint calls: `train_fp = _prep_fingerprint(args, "train", args.max_train_samples, mode, args.telephony_prob)` and `eval_fp = _prep_fingerprint(args, "validation", args.max_eval_samples, "clean", 0.0)`.
7. In the `if need_train` block, choose:
   ```python
   if mode == "paired":
       train_dataset = train_pipeline.prepare_paired_dataset(raw_train)
   else:
       train_dataset = train_pipeline.prepare_dataset_split(raw_train)
   ```
8. Log the composition:
   - paired: `Paired train set: {len(train_dataset)//2} clean + {len(train_dataset)//2} phone = {len(train_dataset)} rows`.
   - disjoint/clean: `Train set: {len(train_dataset)} rows (mode={mode})`.
9. Update the reused-cache note to say the cached draw (both halves in paired mode) is frozen.
10. In `configs/l40s_train_config.yaml`, set `training.epochs: 2`, add `mode: paired` under `telephony`, and comment that `prob` applies only to `disjoint`.
11. In `run_train.sh`, change `--epochs 4` to `--epochs 2` and add `--telephony_mode paired` before `"$@"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py tests/test_train_lora_run_stages.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_lora.py configs/l40s_train_config.yaml run_train.sh tests/test_script_defaults.py tests/test_train_lora_run_stages.py
git commit -m "feat: make paired clean+phone the training default"
```

---

### Task 3: Direct-shard prep cache module

**Files:**
- Create: `src/prep_cache.py`
- Test: `tests/test_prep_cache.py`

**Interfaces:**
- Consumes: `datasets.Dataset`, `datasets.concatenate_datasets`, `Dataset.from_file`, `hashlib`, `json`, `os`, `shutil`, `uuid`, `pathlib.Path`.
- Produces:
  - `PREP_CACHE_FORMAT = "shards_v1"`
  - `class PrepCacheSpaceError(RuntimeError)`
  - `staging_dir(cache_dir: Path) -> Path`
  - `cleanup_stale_staging(cache_dir: Path) -> None`
  - `estimate_feature_bytes(rows: int, copies: int = 1) -> int`
  - `dir_size(path: Path) -> int`
  - `ensure_space(cache_dir: Path, required_bytes: int, reclaimable_bytes: int = 0) -> None`
  - `publish_dataset(dataset, staging: Path, cache_dir: Path, fingerprint: dict) -> dict`
  - `load_prep_cache(cache_dir: Path, fingerprint: dict) -> Optional[Dataset]`
  - `remove_cache(cache_dir: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prep_cache.py`:

```python
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
from datasets import Dataset

from src import prep_cache


def _mapped_dataset(cache_dir: Path, num_proc=2, n=10):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict({"x": list(range(n))})
    return ds.map(
        _double,
        with_indices=True,
        num_proc=num_proc,
        cache_file_name=str(cache_dir / "data.arrow"),
    )


def _double(row, index):
    return {"y": row["x"] * 2}


def test_publish_and_load_roundtrip(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=2)
    meta = prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert (tmp_path / "train" / "meta.json").is_file()
    assert meta["total_rows"] == 10
    loaded = prep_cache.load_prep_cache(tmp_path / "train", {"mode": "paired"})
    assert loaded is not None
    assert len(loaded) == 10
    assert loaded[0]["y"] == 0


def test_load_missing_meta_is_miss(tmp_path):
    (tmp_path / "train").mkdir()
    assert prep_cache.load_prep_cache(tmp_path / "train", {}) is None


def test_load_wrong_fingerprint_is_miss(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "disjoint"}) is None


def test_load_wrong_shard_size_is_miss(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    shard = next((tmp_path / "train").glob("*.arrow"))
    shard.write_bytes(shard.read_bytes()[:-1])
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "paired"}) is None


def test_publish_removes_unreferenced_files(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    (staging / "stray.tmp").write_text("junk")
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert not (tmp_path / "train" / "stray.tmp").exists()


def test_staging_cleanup(tmp_path):
    stale = tmp_path / "train.staging-old"
    stale.mkdir()
    (stale / "junk").write_text("x")
    prep_cache.cleanup_stale_staging(tmp_path / "train")
    assert not stale.exists()


def test_ensure_space_counts_reclaimable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda path: SimpleNamespace(total=100, used=95, free=5),
    )
    prep_cache.ensure_space(tmp_path / "train", required_bytes=12, reclaimable_bytes=10)
    with pytest.raises(prep_cache.PrepCacheSpaceError):
        prep_cache.ensure_space(tmp_path / "train", required_bytes=30, reclaimable_bytes=10)


def test_single_process_layout(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    assert (staging / "data.arrow").is_file()
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "clean"})
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "clean"}) is not None


def test_disable_caching_honors_explicit_cache_file(tmp_path):
    import datasets
    staging = tmp_path / "stage"
    staging.mkdir()
    datasets.disable_caching()
    try:
        ds = _mapped_dataset(staging, num_proc=2)
    finally:
        datasets.enable_caching()
    assert (staging / "data_00000_of_00002.arrow").is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prep_cache.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.prep_cache'`.

- [ ] **Step 3: Implement `src/prep_cache.py`**

Implement the interfaces above. Key behaviors:

- `staging_dir(cache_dir)`: return `cache_dir.with_name(f"{cache_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}")`.
- `cleanup_stale_staging(cache_dir)`: remove all `cache_dir.name + ".staging-*"` siblings with `shutil.rmtree(..., ignore_errors=True)`.
- `estimate_feature_bytes(rows, copies=1)`: return `int(rows * 128 * 3000 * 4 * copies)`.
- `dir_size(path)`: sum `p.stat().st_size` for files under `path.rglob("*")`; return 0 if missing.
- `ensure_space(cache_dir, required_bytes, reclaimable_bytes=0)`: `cache_dir.parent.mkdir(parents=True, exist_ok=True)`; `free = shutil.disk_usage(cache_dir.parent).free`; if `free + reclaimable_bytes < required_bytes`, raise `PrepCacheSpaceError` with both values in GB.
- `publish_dataset(dataset, staging, cache_dir, fingerprint)`:
  - Resolve `staging` and `cache_dir`.
  - Collect `dataset.cache_files` filenames; require all under `staging`; require at least one.
  - Delete any file in `staging` not referenced by the shard list (keep only the shards and then write meta).
  - For each shard, load `Dataset.from_file` to get `num_rows`, record its `name`, `rows`, and `bytes`.
  - Build `meta = {"format": PREP_CACHE_FORMAT, "fingerprint": fingerprint, "total_rows": len(dataset), "features": dataset.features.to_dict(), "shards": [...]}`.
  - Write `meta.json` atomically (`.tmp` + `os.replace`).
  - `remove_cache(cache_dir)`; `os.replace(staging, cache_dir)`; return meta.
- `load_prep_cache(cache_dir, fingerprint)`:
  - Read `meta.json`; on any error return `None`.
  - Require `format == PREP_CACHE_FORMAT`, `fingerprint == fingerprint`, `total_rows > 0`, non-empty shards.
  - For each shard: `Path(entry["file"]).name == entry["file"]`; file exists; size matches `entry["bytes"]`; `Dataset.from_file` rows match `entry["rows"]`; `features.to_dict() == meta["features"]`.
  - Concatenate parts; require `len(combined) == total_rows`; return it.
  - Print a one-line HIT or the reason for a miss.
- `remove_cache(cache_dir)`: `shutil.rmtree(cache_dir, ignore_errors=True)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prep_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prep_cache.py tests/test_prep_cache.py
git commit -m "feat: add direct-shard prep cache with atomic metadata"
```

---

### Task 4: Direct-shard writes in the data pipeline

**Files:**
- Modify: `src/data.py` (`prepare_dataset_split`, `prepare_paired_dataset`, `_prepare_split`)
- Test: `tests/test_data_prep.py`

**Interfaces:**
- Consumes: Task 1's paired method; `datasets.map(cache_file_name=...)`.
- Produces:
  - `prepare_dataset_split(self, dataset, remove_columns=None, cache_dir: Optional[str] = None) -> Dataset`
  - `prepare_paired_dataset(self, dataset, remove_columns=None, cache_dir: Optional[str] = None) -> Dataset`
  - `_prepare_split(..., cache_file_name: Optional[str] = None)`
  - Multi-process maps write `data_00000_of_0000N.arrow` (or `clean_...`/`phone_...`); single-process writes the exact `data.arrow` / `clean.arrow` / `phone.arrow`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_prep.py`:

```python
def test_prepare_dataset_split_writes_cache_dir(tmp_path):
    ds = _pairing_dataset(4)
    out = _pipeline(prob=0.0, num_proc=1).prepare_dataset_split(
        ds, cache_dir=str(tmp_path / "stage")
    )
    files = [Path(c["filename"]).resolve() for c in out.cache_files]
    assert files
    assert all(tmp_path.resolve() in f.parents for f in files)
    assert all(f.is_file() for f in files)


def test_prepare_paired_dataset_writes_clean_and_phone_shards(tmp_path):
    ds = _pairing_dataset(4)
    out = _pipeline(prob=0.5, num_proc=1).prepare_paired_dataset(
        ds, cache_dir=str(tmp_path / "stage")
    )
    names = [Path(c["filename"]).name for c in out.cache_files]
    assert any(n.startswith("clean") for n in names)
    assert any(n.startswith("phone") for n in names)
    assert len(out) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -q`
Expected: FAIL with `TypeError: prepare_dataset_split() got an unexpected keyword argument 'cache_dir'`.

- [ ] **Step 3: Implement the cache plumbing**

In `src/data.py`:

1. Add `cache_file_name: Optional[str] = None` to `_prepare_split`; add it to `map_kwargs` only when not `None`.
2. Add `cache_dir: Optional[str] = None` to `prepare_dataset_split`; when set, `os.makedirs(cache_dir, exist_ok=True)` and pass `cache_file_name=os.path.join(cache_dir, "data.arrow")`.
3. Add `cache_dir: Optional[str] = None` to `prepare_paired_dataset`; when set, create it and pass `cache_file_name=os.path.join(cache_dir, "clean.arrow")` / `"phone.arrow"` to the two `_prepare_split` calls.
4. Keep the existing multiprocess-to-single-process retry; the explicit `cache_file_name` is honored in both paths.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data_prep.py
git commit -m "feat: write prepared shards directly into the prep cache"
```

---

### Task 5: Wire the direct-shard cache into training

**Files:**
- Modify: `scripts/train_lora.py` (prep flow around L1169-1202; remove `_try_load_prep_cache` / `_save_prep_cache` around L303-340)
- Modify: `tests/test_train_lora_run_stages.py`
- Test: `tests/test_train_lora_run_stages.py` (plus Task 2's tests)

**Interfaces:**
- Consumes: Task 2's `_prep_fingerprint` and mode; Task 1/4's pipeline methods; Task 3's `prep_cache`.
- Produces the final flow: `<prep_cache_dir>/train` and `<prep_cache_dir>/validation` hold shard caches with `meta.json`; no `save_to_disk`.

- [ ] **Step 1: Update the failing tests**

In `tests/test_train_lora_run_stages.py`:

- The cached-prep test must monkeypatch the new cache functions:
  ```python
  monkeypatch.setattr(train_lora.prep_cache, "load_prep_cache", lambda *a, **k: SimpleNamespace())
  monkeypatch.setattr(train_lora.prep_cache, "publish_dataset", lambda *a, **k: None)
  ```
- The fake pipeline must have both `prepare_dataset_split` and `prepare_paired_dataset` raise `AssertionError("cached prep must skip re-prep")`.
- Add a test that a prep failure never publishes:
  ```python
  def test_prep_failure_does_not_publish(monkeypatch, ...):
      # load returns None; pipeline prepare_paired_dataset raises RuntimeError
      # monkeypatch ensure_space/remove_cache/staging_dir/cleanup_stale_staging/publish_dataset
      # assert publish_dataset was not called and the run is failed at preflight
  ```
  Use the existing run-stage test fixture for the run dir and error handling.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_lora_run_stages.py -q`
Expected: FAIL because the train script still calls the old cache helpers.

- [ ] **Step 3: Implement the staging/publish flow**

In `scripts/train_lora.py`:

1. Import `from src import prep_cache` and `from pathlib import Path` (Path is likely already imported).
2. Remove `_try_load_prep_cache` and `_save_prep_cache`.
3. Replace the prep-cache section with:
   ```python
   mode = args.telephony_mode
   train_cache = Path(prep_cache_dir) / "train"
   eval_cache = Path(prep_cache_dir) / "validation"
   load_cache = reuse_prep and not overwrite_prep
   if load_cache:
       train_dataset = prep_cache.load_prep_cache(train_cache, train_fp)
       eval_dataset = prep_cache.load_prep_cache(eval_cache, eval_fp)
   ```
4. Before loading raw data, determine `need_train = train_dataset is None` and `need_eval = eval_dataset is None`.
5. When `need_train`:
   - `prep_cache.cleanup_stale_staging(train_cache)`.
   - `required = prep_cache.estimate_feature_bytes(len(raw_train), copies=2 if mode == "paired" else 1)`; multiply by 1.05 and cast to int.
   - `reclaimable = prep_cache.dir_size(train_cache)`.
   - `prep_cache.ensure_space(Path(prep_cache_dir), required, reclaimable)`.
   - `prep_cache.remove_cache(train_cache)`.
   - `staging = prep_cache.staging_dir(train_cache)`.
   - In a `try`: call `prepare_paired_dataset(raw_train, cache_dir=str(staging))` for paired, else `prepare_dataset_split(raw_train, cache_dir=str(staging))`.
   - `prep_cache.publish_dataset(train_dataset, staging, train_cache, train_fp)`.
   - `train_dataset = prep_cache.load_prep_cache(train_cache, train_fp)`; if `None`, raise `RuntimeError("prep cache publish/load failed")`.
   - In `except`: `shutil.rmtree(staging, ignore_errors=True); raise`.
6. Do the same for the validation cache with `eval_pipeline`, `copies=1`, and `eval_fp`.
7. Keep the `if need_train or need_eval or args.run_epoch_benchmark` raw-load block; `raw_test` stays for the callback.
8. Keep the `del raw_train, raw_eval` and the preflight call after prep.
9. Remove the old "reused train prep freezes phone augmentation" message; print the mode and row count after a cache hit or fresh prep.
10. Ensure the free-space guard runs before `remove_cache`; the projected free includes the old cache.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_lora_run_stages.py tests/test_script_defaults.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_lora.py tests/test_train_lora_run_stages.py
git commit -m "feat: use the direct-shard prep cache in training"
```

---

### Task 6: Server and smoke wiring

**Files:**
- Modify: `server/app.py` (`_TRAINER_VALUE_ARGS` around L1760-1795, `_normalize_start_config` around L1845-1930, `_start_new_run` around L2110-2225, `_start_resume_run` around L2233-2320, smoke command around L1680-1700)
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: `telephony_mode` in the start payload.
- Produces:
  - `_normalize_start_config` returns `telephony_mode` (default `"paired"`) and defaults epochs to 2.
  - `_TRAINER_VALUE_ARGS` includes `"telephony_mode"`.
  - `trainer_args`, `run_config`, and `response_config` include the mode.
  - Resume of a run whose `trainer_args` lack the mode uses `"disjoint"`.
  - Smoke command includes `--telephony_mode <mode>`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_server_train.py`:

```python
def test_start_defaults_to_paired_and_two_epochs(client):
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200
    flags = _flags(client.launched[0])
    assert flags["--telephony_mode"] == "paired"
    assert flags["--epochs"] == "2"


def test_start_rejects_unknown_telephony_mode(client):
    res = client.post("/api/train/start", json={"telephony_mode": "satellite"})
    assert res.status_code == 400
    assert client.launched == []


def test_resume_legacy_run_defaults_to_disjoint(client):
    start = client.post("/api/train/start", json={}).json()
    run_id = start["run_id"]
    # Stop the fake child so the resume is allowed, as in the existing
    # test_resume_uses_manifest_run_dir test.
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None

    def strip_mode(run):
        run["trainer_args"].pop("telephony_mode", None)
        run["config"].pop("telephony_mode", None)
        run["status"] = "failed"
        return run

    run_state.update_run(client.mod.RUNS_DIR / run_id, strip_mode)
    res = client.post(f"/api/runs/{run_id}/resume")
    assert res.status_code == 200, res.text
    flags = _flags(client.launched[-1])
    assert flags["--telephony_mode"] == "disjoint"


def test_smoke_command_includes_paired_mode(client):
    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 200
    assert _flags(client.launched[0])["--telephony_mode"] == "paired"
```

Also update `test_defaults_use_small_batch_with_accumulation_and_sparse_eval` and `test_start_creates_self_contained_run` to assert the new mode/epochs.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`
Expected: FAIL on missing `--telephony_mode` and epochs `2`.

- [ ] **Step 3: Implement the server changes**

1. In `_TRAINER_VALUE_ARGS`, add `"telephony_mode"` after `"telephony_prob"`.
2. In `_normalize_start_config`, default `epochs_i` to 2; read `mode = str(config.get("telephony_mode", "paired")).strip().lower()`; reject anything outside `{"paired","disjoint","clean"}` with a 400; return `"telephony_mode": mode`.
3. In `_start_new_run`, add `"telephony_mode": values["telephony_mode"]` to `trainer_args`, `run_config`, and `response_config`.
4. In `_start_resume_run`, after `trainer_args = dict(run.get("trainer_args") or {})`, add `trainer_args.setdefault("telephony_mode", "disjoint")`; add `"telephony_mode": trainer_args["telephony_mode"]` to `response_config`.
5. In the smoke endpoint, read and validate `telephony_mode` (default `"paired"`) and add `--telephony_mode <mode>` to the smoke command.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: carry telephony mode through the cockpit API"
```

---

### Task 7: Cockpit training-data selector

**Files:**
- Modify: `server/static/index.html` (Epochs field around L108-113)
- Modify: `server/static/app.js` (`DEFAULT_EPOCHS` L7, `el` L20-60, `restoreRunConfig` L146-164, `startRun` L1232-1263)
- Modify: `server/static/run_view.js` (`formValuesFromConfig` L112-125, exports L144-157)
- Test: `tests/test_run_view.js`, `tests/test_web_ui_contract.py`

**Interfaces:**
- Produces:
  - `<select id="inputTelephonyMode">` with paired selected.
  - `DEFAULT_EPOCHS = 2`.
  - `epochsForTelephonyMode(mode)` in `run_view.js`: `2` for `paired`, `4` for `disjoint`/`clean`.
  - `formValuesFromConfig` maps `telephony_mode`.
  - `startRun` sends `telephony_mode`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_run_view.js`:

```javascript
const { epochsForTelephonyMode } = require("../server/static/run_view.js");
assert.strictEqual(epochsForTelephonyMode("paired"), 2);
assert.strictEqual(epochsForTelephonyMode("disjoint"), 4);
assert.strictEqual(epochsForTelephonyMode("clean"), 4);
assert.strictEqual(epochsForTelephonyMode(undefined), 2);
assert.strictEqual(
  formValuesFromConfig({ ...fullConfig, telephony_mode: "clean" }).telephony_mode,
  "clean",
);
```

In `tests/test_web_ui_contract.py`, add:

```python
def test_cockpit_exposes_paired_training_default():
    html = (ROOT / "server" / "static" / "index.html").read_text()
    js = (ROOT / "server" / "static" / "app.js").read_text()
    view = (ROOT / "server" / "static" / "run_view.js").read_text()
    assert 'id="inputTelephonyMode"' in html
    assert 'value="paired" selected' in html
    assert 'id="inputEpochs"' in html and 'value="2"' in html
    assert "telephony_mode" in js
    assert "epochsForTelephonyMode" in js
    assert "epochsForTelephonyMode" in view
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node tests/test_run_view.js && .venv/bin/python -m pytest tests/test_web_ui_contract.py -q`
Expected: FAIL — `epochsForTelephonyMode` is not exported; the select is missing.

- [ ] **Step 3: Implement the UI**

1. `index.html`: change the Epochs input to `value="2"`; add after it:
   ```html
   <label class="field">
     <span class="field-label">Training data</span>
     <select id="inputTelephonyMode" class="field-input">
       <option value="paired" selected>Paired clean + phone (2 epochs)</option>
       <option value="disjoint">Disjoint 50/50 (4 epochs)</option>
       <option value="clean">Clean only (4 epochs)</option>
     </select>
   </label>
   ```
   Bump the cache-busting queries on the changed scripts: `run_view.js?v=2` → `?v=3` and `app.js?v=11` → `?v=12`.
2. `run_view.js`: add `telephony_mode` to the `formValuesFromConfig` key list; add:
   ```javascript
   function epochsForTelephonyMode(mode) {
     return String(mode || "paired") === "paired" ? 2 : 4;
   }
   ```
   Export it in `module.exports`.
3. `app.js`: set `DEFAULT_EPOCHS = 2`; add `inputTelephonyMode: document.getElementById("inputTelephonyMode")` to `el`; in `restoreRunConfig`, add `[el.inputTelephonyMode, values.telephony_mode]` to the fields; add a change listener that sets `el.inputEpochs.value = epochsForTelephonyMode(el.inputTelephonyMode.value)`; in `startRun`, add `telephony_mode: el.inputTelephonyMode ? el.inputTelephonyMode.value : "paired"`.
4. Keep the smoke-test button unchanged; it uses the server default paired.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --check server/static/app.js && node --check server/static/run_view.js && node tests/test_run_view.js && .venv/bin/python -m pytest tests/test_web_ui_contract.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/run_view.js tests/test_run_view.js tests/test_web_ui_contract.py
git commit -m "feat: add paired training selector to the cockpit"
```

---

### Task 8: Documentation and final verification

**Files:**
- Modify: `README.md` (key flags around L245-255)
- Modify: `GUIDE.md` (training table around L143; cache/benchmark notes)
- Modify: `PRODUCTION.md` (cache and resume sections)
- Modify: `configs/l40s_train_config.yaml` comments (if not already done in Task 2)

**Interfaces:**
- Consumes: the final behavior from Tasks 1-7.
- Produces: docs that match the default paired mode, 2 epochs, direct-shard cache, free-space requirement, and resume rule.

- [ ] **Step 1: Update the docs**

- `README.md`:
  - Replace the `--telephony_prob 0.5` and `--epochs 4` bullets with:
    - `--telephony_mode paired` (default): every utterance once clean and once phone-simulated, 116,488 rows; `--epochs 2` is compute-matched to 4 disjoint epochs.
    - `--telephony_mode disjoint`: the old 50/50 coin (`--telephony_prob 0.5`); use `--epochs 4`.
    - `--telephony_mode clean`: no phone simulation; use `--epochs 4`.
    - `--epochs 2`: 2 paired epochs (same optimizer steps as 4 disjoint epochs).
  - Add the direct-shard cache note: `~170 GB` steady/peak, `meta.json` validation, ~180 GB free-space guard, ~250 GB overlay, raw/model ~25 GB.
- `GUIDE.md`:
  - Training table row: `Paired clean + phone (1:1, 2 epochs)`; update the duration note (same steps as 4 disjoint, ~10 h; prep ~2×) and the cache note.
  - Add a line that the cockpit selector auto-fills 2/4/4 epochs.
- `PRODUCTION.md`:
  - Cache section: direct shard cache, `meta.json`, staging publish, free-space guard, old `save_to_disk` caches are replaced.
  - Resume section: an old disjoint run must resume with `--telephony_mode disjoint` (and the old epoch count) or start fresh; cockpit resume injects disjoint for legacy manifests.
- `configs/l40s_train_config.yaml`: comment that `prob` applies only to `disjoint`.

- [ ] **Step 2: Verify no stale claims**

Run: `git grep -n "340\\|cleanup_cache_files\\|save_to_disk" README.md GUIDE.md PRODUCTION.md configs/l40s_train_config.yaml run_train.sh`
Expected: only intentional references to the old format, or no output.

- [ ] **Step 3: Run the full verification**

Run:
```bash
.venv/bin/python -m pytest -q
node --check server/static/app.js
node --check server/static/run_view.js
node tests/test_run_view.js
node tests/test_benchmark_view.js
bash -n run_train.sh
git diff --check
```
Expected: all tests pass; Node checks pass; `run_train.sh` parses; no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add README.md GUIDE.md PRODUCTION.md configs/l40s_train_config.yaml
git commit -m "docs: document paired training and the shard cache"
```

---

## Manual L40S Smoke (after the plan is complete)

Run on the L40S:

```bash
./run_train.sh \
  --max_train_samples 20 \
  --max_eval_samples 10 \
  --max_benchmark_samples 20 \
  --preflight_samples 20 \
  --no-run_epoch_benchmark \
  --no-run_post_test_eval \
  --fresh_start
```

Confirm in the log:

- `Paired train set: 20 clean + 20 phone = 40 rows`.
- The prep cache contains `train/meta.json` and shard `*.arrow` files.
- No `save_to_disk` duplicate and no `data/` directory.
- The free-space log/guard sees the expected ~250 GB overlay.

Then start the full run with the defaults (paired, 2 epochs, 1,000-row preflight).
