# Automatic Preflight Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every training run measure a mandatory BuzzASR base preflight on the first 1,000 test rows, use it for clean/phone WER deltas, and remove all hardcoded baseline values.

**Architecture:** A new `src/preflight.py` checks a fingerprint, invokes the existing `scripts/evaluate_model.py` for BuzzASR and Whisper, parses the two `evaluation_details.json` files, and atomically writes `results/baselines/preflight.json`. `scripts/train_lora.py` calls it after data prep and before model load, passes the metrics to `WebTelemetryCallback`, and the server/UI render measured deltas only when sample counts match.

**Tech Stack:** Python 3.12, stdlib `subprocess`/`json`/`pathlib`, pytest, FastAPI, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-10-30-preflight-baseline-design.md`

## Global Constraints

- Preflight and epoch benchmark both use the **first 1,000 rows** of the `test` split.
- `--preflight_samples` defaults to `max_benchmark_samples`; `0` means the full 2,149 rows.
- BuzzASR base is mandatory; Whisper is reference-only.
- No production fallback metrics. `configs/baselines.json` is deleted.
- Deltas are percentage points; negative is improvement. Deltas are `null` unless `total_samples` matches.
- Use `telephony_profiles=["customer"]`, `phone_seed=1234`, and the run's `telephony_use_ffmpeg` value.
- Do not change training defaults: batch 8 × grad accum 4, no gradient checkpointing, eval/save 1000.
- Do not import `transformers`/`torch` in `src/preflight.py`; it orchestrates subprocesses.
- Work on `main`; do not push.

## Review Focus

- Preflight must run **after** `datasets.map` preparation and before any training model/CUDA load, or forked workers deadlock.
- A cache fingerprint that omits sample count, phone seed, profile, or ffmpeg flag can silently reuse the wrong baseline.
- A 1,000-row preflight compared with a full-test or 300-row epoch row must produce `null` deltas, never a number.
- BuzzASR failure must abort after one retry; Whisper failure must warn and omit its row without blocking training.
- No preflight must produce an explicit empty state in the API/UI, and the final full-test report must not claim deltas against a mismatched baseline.

---

### Task 1: Preflight fingerprint, cache, parsing, and delta helper

**Files:**
- Create: `src/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Produces: `PREFLIGHT_VERSION = "1"`
- Produces: `DEFAULT_PREFLIGHT_MODELS = {"buzzasr_base": "BuzzASR/persian", "whisper_v3": "openai/whisper-large-v3"}`
- Produces: `build_fingerprint(*, dataset_name: str, split: str, sample_count: int, phone_seed: int, telephony_profiles: Sequence[str], telephony_use_ffmpeg: bool, eval_batch_size: int, models: Mapping[str, str] = DEFAULT_PREFLIGHT_MODELS) -> Dict[str, Any]`
- Produces: `load_cached_preflight(path: Path, fingerprint: Mapping[str, Any]) -> Optional[Dict[str, Any]]`
- Produces: `parse_evaluation_details(payload: Mapping[str, Any], *, name: str, epoch: str, source: str = "preflight", is_reference: bool = False, phone_seed: Optional[int] = None, phone_profiles: Optional[Sequence[str]] = None) -> Dict[str, Any]`
- Produces: `compute_deltas(clean_wer: float, phone_wer: Optional[float], total_samples: int, baseline: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]`

- [ ] **Step 1: Write the failing tests**

```python
def test_fingerprint_changes_with_sample_count_and_seed():
    a = build_fingerprint(dataset_name="d", split="test", sample_count=1000,
                          phone_seed=1234, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    b = build_fingerprint(dataset_name="d", split="test", sample_count=2149,
                          phone_seed=1234, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    c = build_fingerprint(dataset_name="d", split="test", sample_count=1000,
                          phone_seed=9999, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    assert a["sample_count"] == 1000
    assert a != b and a != c


def test_cached_preflight_requires_matching_fingerprint(tmp_path):
    fp = {"sample_count": 1000, "phone_seed": 1234}
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"fingerprint": fp, "models": {"buzzasr_base": {}}}))
    assert load_cached_preflight(path, fp)["fingerprint"] == fp
    assert load_cached_preflight(path, {**fp, "phone_seed": 1}) is None


def test_parse_evaluation_details_extracts_clean_and_phone():
    row = parse_evaluation_details({
        "metrics": {"total_samples": 1000,
                    "overall": {"normalized_wer": 17.2, "normalized_cer": 4.8,
                                "diagnostic_wer_zwnj_space": 15.9, "raw_wer": 18.0, "raw_cer": 5.0},
                    "formal": {"normalized_wer": 15.1},
                    "informal": {"normalized_wer": 22.4}},
        "phone_metrics": {"overall": {"normalized_wer": 24.1, "normalized_cer": 7.2},
                          "formal": {"normalized_wer": 21.0},
                          "informal": {"normalized_wer": 29.0}},
        "phone_seed": 1234, "phone_profiles": ["customer"],
    }, name="BuzzASR Base", epoch="BuzzASR Base (preflight, 1,000)")
    assert row["total_samples"] == 1000
    assert row["normalized_wer"] == 17.2
    assert row["phone_normalized_wer"] == 24.1
    assert row["phone_seed"] == 1234


def test_compute_deltas_requires_matching_sample_count():
    baseline = {"total_samples": 1000, "normalized_wer": 17.2, "phone_normalized_wer": 24.1}
    assert compute_deltas(15.0, 20.0, 1000, baseline) == {
        "delta_vs_buzzasr": -2.2, "delta_phone_vs_buzzasr": -4.1}
    assert compute_deltas(15.0, 20.0, 2149, baseline) == {
        "delta_vs_buzzasr": None, "delta_phone_vs_buzzasr": None}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`  
Expected: FAIL because `src.preflight` does not exist.

- [ ] **Step 3: Implement the pure helpers in `src/preflight.py`**

`build_fingerprint` returns a plain dict with every argument normalized: `preflight_version`, profiles sorted, `sample_count` and `eval_batch_size` as `int`, `telephony_use_ffmpeg` as `bool`, and the model mapping copied. `load_cached_preflight` returns the parsed dict only when `data["fingerprint"] == dict(fingerprint)` and `data["models"]["buzzasr_base"]` exists. `parse_evaluation_details` reads `payload["metrics"]` and optional `payload["phone_metrics"]`, rounds metrics to two decimals, and copies `phone_seed`/`phone_profiles`. `compute_deltas` returns `None` deltas unless `baseline["total_samples"] == total_samples`; clean delta subtracts `baseline["normalized_wer"]`, phone delta subtracts `baseline["phone_normalized_wer"]` when both are non-null.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/preflight.py tests/test_preflight.py
git commit -m "feat: add preflight fingerprint and metric parsing"
```

---

### Task 2: Preflight runner with retry, cache, and atomic write

**Files:**
- Modify: `src/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `build_fingerprint`, `load_cached_preflight`, `parse_evaluation_details`
- Produces: `preflight_path(results_dir: Path) -> Path` returning `<results_dir>/baselines/preflight.json`
- Produces: `run_preflight(*, results_dir: Path, dataset_name: str, split: str, sample_count: int, phone_seed: int, telephony_profiles: Sequence[str], telephony_use_ffmpeg: bool, eval_batch_size: int, overwrite: bool = False) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing runner tests**

Define this helper above the tests:

```python
def _fake_details(model: str):
    clean = {"total_samples": 1000,
             "overall": {"normalized_wer": 17.2, "normalized_cer": 4.8,
                         "diagnostic_wer_zwnj_space": 15.9, "raw_wer": 18.0, "raw_cer": 5.0},
             "formal": {"normalized_wer": 15.1},
             "informal": {"normalized_wer": 22.4}}
    phone = {"overall": {"normalized_wer": 24.1, "normalized_cer": 7.2},
             "formal": {"normalized_wer": 21.0},
             "informal": {"normalized_wer": 29.0}}
    return {"metrics": clean, "phone_metrics": phone,
            "phone_seed": 1234, "phone_profiles": ["customer"]}
```

```python
def test_run_preflight_writes_unified_json_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1])
        out.mkdir(parents=True)
        model = cmd[cmd.index("--base_model") + 1]
        payload = _fake_details(model)
        (out / "evaluation_details.json").write_text(json.dumps(payload))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert preflight_path(tmp_path).exists()
    assert result["models"]["buzzasr_base"]["phone_normalized_wer"] == 24.1
    assert result["models"]["whisper_v3"]["is_reference"] is True
    assert len(calls) == 2
    run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                  sample_count=1000, phone_seed=1234,
                  telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                  eval_batch_size=16)
    assert len(calls) == 2


def test_run_preflight_retries_buzzasr_then_raises(tmp_path, monkeypatch):
    def always_fail(cmd, check, cwd):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("src.preflight.subprocess.run", always_fail)
    with pytest.raises(RuntimeError, match="BuzzASR"):
        run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                      sample_count=1000, phone_seed=1234,
                      telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                      eval_batch_size=16)


def test_run_preflight_omits_whisper_on_failure(tmp_path, monkeypatch):
    def fake_run(cmd, check, cwd):
        if "whisper" in " ".join(cmd).lower():
            raise subprocess.CalledProcessError(1, cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1]); out.mkdir(parents=True)
        (out / "evaluation_details.json").write_text(json.dumps(_fake_details("BuzzASR/persian")))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert "whisper_v3" not in result["models"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`  
Expected: FAIL because `run_preflight` and `preflight_path` do not exist.

- [ ] **Step 3: Implement `run_preflight`**

Create `results_dir/baselines/preflight/` and each per-model output directory. Use `sys.executable` and `<repo>/scripts/evaluate_model.py`. For each model in the order `("buzzasr_base", "whisper_v3")`, build:

```text
--base_model <id> --dataset_name <name> --split <split>
--max_samples <sample_count> --batch_size <eval_batch_size>
--output_dir <results_dir>/baselines/preflight/<slug>
--phone_eval --phone_seed <seed> --telephony_profiles <comma-joined>
```

Add `--telephony_use_ffmpeg` only when true; otherwise `--no-telephony_use_ffmpeg`. Retry BuzzASR once; after a second failure raise `RuntimeError("BuzzASR preflight failed: ...")`. Catch Whisper failure, print a warning, and omit its row. Parse each `evaluation_details.json` with `parse_evaluation_details`. Write `{"version": "1", "created_at": time.time(), "fingerprint": fp, "models": models}` to a `.tmp` sibling, then `replace` the final path. Return the cached file when `overwrite` is false and the fingerprint matches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/preflight.py tests/test_preflight.py
git commit -m "feat: run cached BuzzASR and Whisper preflight"
```

---

### Task 3: Wire preflight into training and use 1,000 rows everywhere

**Files:**
- Modify: `scripts/train_lora.py`
- Modify: `configs/l40s_train_config.yaml`
- Modify: `run_train.sh`
- Modify: `tests/test_script_defaults.py`

**Interfaces:**
- Consumes: `run_preflight`, `compute_deltas`
- Produces: `--preflight_samples` and `--overwrite_preflight` CLI flags
- Produces: `_run_post_training_evaluation(..., baseline_benchmark: Optional[Dict[str, Any]] = None)`

- [ ] **Step 1: Update the consistency tests to the new defaults**

In `tests/test_script_defaults.py`, assert:

```python
assert args.max_benchmark_samples == 1000
assert args.preflight_samples is None
assert cfg["training"]["max_benchmark_samples"] == 1000
assert cfg["training"]["preflight_samples"] == 1000
assert "--max_benchmark_samples 1000" in run_train_text
assert "--preflight_samples 1000" in run_train_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py -q`  
Expected: FAIL on `1000` vs the current `300`.

- [ ] **Step 3: Add the flags and defaults**

Add to `parse_args()`:

```python
parser.add_argument("--preflight_samples", type=int, default=None,
                    help="Preflight rows; default follows --max_benchmark_samples (0 = full test).")
parser.add_argument("--overwrite_preflight", "--overwrite-preflight",
                    action=argparse.BooleanOptionalAction, default=False)
```

Change `--max_benchmark_samples` default to `1000`. Add `preflight_samples` and `overwrite_preflight` to `_apply_yaml_config`'s key map and boolean keys. Add `preflight_samples: 1000` to `configs/l40s_train_config.yaml` and change `max_benchmark_samples` to `1000`. Add `--max_benchmark_samples 1000` and `--preflight_samples 1000` to `run_train.sh`.

- [ ] **Step 4: Insert the preflight after data prep**

Immediately after `assert train_dataset is not None and eval_dataset is not None` and before the CUDA-info block:

```python
preflight_count = args.preflight_samples
if preflight_count is None:
    preflight_count = args.max_benchmark_samples
preflight = run_preflight(
    results_dir=Path(args.results_dir),
    dataset_name=args.dataset_name,
    split="test",
    sample_count=int(preflight_count or 0),
    phone_seed=int(args.phone_seed),
    telephony_profiles=args.telephony_profiles,
    telephony_use_ffmpeg=bool(args.telephony_use_ffmpeg),
    eval_batch_size=int(args.eval_batch_size),
    overwrite=bool(args.overwrite_preflight),
)
baseline_benchmark = preflight["models"]
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Remove `load_baselines` from the `src.trainer_callbacks` import and add `from src.preflight import compute_deltas, run_preflight` near the other `src` imports. Pass `baseline_benchmark=baseline_benchmark` to `WebTelemetryCallback` and to `_run_post_training_evaluation`. Do not delete the preflight cache in `_clear_previous_state`; `--overwrite_preflight` is the explicit invalidation path.

- [ ] **Step 5: Update the final report to use matching baselines only**

In `_run_post_training_evaluation`, add `baseline_benchmark: Optional[Dict[str, Any]] = None` and replace the `load_baselines()` block with:

```python
buzz = (baseline_benchmark or {}).get("buzzasr_base") or {}
deltas = compute_deltas(
    norm_wer,
    phone_block.get("normalized_wer") if phone_block else None,
    total_samples,
    buzz,
)
delta_vs_buzzasr = deltas["delta_vs_buzzasr"]
delta_phone_vs_buzzasr = deltas["delta_phone_vs_buzzasr"]
```

Remove `delta_vs_whisper`. In JSON, write `delta_vs_buzzasr_base`, `delta_phone_vs_buzzasr_base`, and `baseline_comparison` with `source`, `sample_count`, and `evaluated_samples`. In Markdown/console, print `n/a (no matching full-test baseline)` when the delta is `None`. Change the epoch table to show `Δ clean vs base` and `Δ phone vs base`, formatting `None` as `--`.

- [ ] **Step 6: Run the targeted tests**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py tests/test_preflight.py -q`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_lora.py configs/l40s_train_config.yaml run_train.sh tests/test_script_defaults.py
git commit -m "feat: run mandatory 1k preflight before training"
```

---

### Task 4: Callback baseline rows and phone deltas

**Files:**
- Modify: `src/trainer_callbacks.py`
- Create: `tests/test_trainer_callbacks.py`

**Interfaces:**
- Consumes: `compute_deltas`
- Produces: `WebTelemetryCallback(..., baseline_benchmark: Optional[Dict[str, Any]] = None)` with initial baseline rows from preflight
- Produces: epoch records containing `delta_vs_buzzasr` and `delta_phone_vs_buzzasr` (`float | None`)

- [ ] **Step 1: Write the failing callback tests**

```python
def test_initial_rows_include_preflight_phone_metrics(tmp_path):
    cb = WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"epoch": "BuzzASR Base (preflight, 1,000)", "source": "preflight",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
        "whisper_v3": {"epoch": "Whisper (preflight)", "source": "preflight",
                       "is_reference": True, "total_samples": 1000,
                       "normalized_wer": 28.5, "phone_normalized_wer": 35.0},
    })
    rows = json.loads((tmp_path / "epoch_benchmarks.json").read_text())
    by_epoch = {r["epoch"]: r for r in rows}
    assert set(by_epoch) == {"BuzzASR Base (preflight, 1,000)", "Whisper (preflight)"}
    assert by_epoch["BuzzASR Base (preflight, 1,000)"]["phone_normalized_wer"] == 24.1


def test_epoch_record_computes_clean_and_phone_deltas(tmp_path):
    cb = WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1}})
    cb._record_epoch_benchmark(1, {
        "total_samples": 1000,
        "overall": {"normalized_wer": 15.0, "normalized_cer": 4.0,
                    "diagnostic_wer_zwnj_space": 14.0},
        "formal": {"normalized_wer": 13.0}, "informal": {"normalized_wer": 19.0},
        "phone_metrics": {"overall": {"normalized_wer": 20.0, "normalized_cer": 6.0},
                          "formal": {"normalized_wer": 18.0},
                          "informal": {"normalized_wer": 23.0}},
        "phone_seed": 1234, "phone_profiles": ["customer"],
    })
    row = json.loads((tmp_path / "epoch_benchmarks.json").read_text())[-1]
    assert row["delta_vs_buzzasr"] == -2.2
    assert row["delta_phone_vs_buzzasr"] == -4.1
    assert "delta_vs_whisper" not in row
```

Add a third test where `results["total_samples"] == 2149` and assert both deltas are `None`. Add a fourth test that pre-seeds `epoch_benchmarks.json` with an old baseline row plus `Epoch 1`; after `WebTelemetryCallback(...)`, the old baseline row is gone, the new preflight row is present, and `Epoch 1` is preserved.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trainer_callbacks.py -q`  
Expected: FAIL because the callback still reads `configs/baselines.json` and has no phone delta.

- [ ] **Step 3: Implement the callback changes**

Delete `_default_baselines()` and `load_baselines()`; remove the now-unused `Path` import. In `__init__`, set `self.baseline_benchmark = baseline_benchmark or {}` and build `initial_benchmarks` from `self.baseline_benchmark.get(key)` for `buzzasr_base` then `whisper_v3`; prepend `type="baseline"`, `is_baseline=True`, `canary_transcriptions=[]`. If `epoch_benchmarks.json` already exists, load it, drop rows with `is_baseline`, and write the new baseline rows followed by the preserved epoch rows. In `_record_epoch_benchmark`, after the phone block, call:

```python
record.update(compute_deltas(
    norm_wer,
    record.get("phone_normalized_wer"),
    record["total_samples"],
    self.baseline_benchmark.get("buzzasr_base"),
))
```

Remove `delta_vs_whisper`. Import `compute_deltas` from `src.preflight`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_trainer_callbacks.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainer_callbacks.py tests/test_trainer_callbacks.py
git commit -m "feat: show measured clean and phone deltas in benchmark rows"
```

---

### Task 5: Server serves preflight baselines only

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_server_train.py`

**Interfaces:**
- Produces: `PREFLIGHT_FILE = RESULTS_DIR / "baselines" / "preflight.json"`
- Produces: `get_default_benchmarks() -> List[Dict[str, Any]]` returning preflight rows or `[]`
- Changes: `/api/benchmarks` returns `[]` when neither preflight nor epoch data exists

- [ ] **Step 1: Write the failing server tests**

```python
def test_default_benchmarks_load_preflight_rows(client, tmp_path, monkeypatch):
    preflight = tmp_path / "baselines" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"models": {
        "buzzasr_base": {"name": "BuzzASR Base", "epoch": "BuzzASR Base (preflight, 1,000)",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
        "whisper_v3": {"name": "Whisper", "epoch": "Whisper (preflight)",
                       "is_reference": True, "total_samples": 1000,
                       "normalized_wer": 28.5, "phone_normalized_wer": 35.0},
    }}))
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", preflight)
    by_epoch = {r["epoch"]: r for r in client.mod.get_default_benchmarks()}
    assert by_epoch["BuzzASR Base (preflight, 1,000)"]["phone_normalized_wer"] == 24.1
    assert by_epoch["Whisper (preflight)"]["is_reference"] is True


def test_default_benchmarks_empty_without_preflight(client, tmp_path, monkeypatch):
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", tmp_path / "missing.json")
    assert client.mod.get_default_benchmarks() == []


def test_benchmarks_endpoint_drops_unmeasured_baselines(client, tmp_path, monkeypatch):
    bench = tmp_path / "epoch_benchmarks.json"
    bench.write_text(json.dumps([
        {"epoch": "Old hardcoded base", "is_baseline": True, "normalized_wer": 17.2},
        {"epoch": "Epoch 1", "is_baseline": False, "normalized_wer": 15.0},
    ]))
    monkeypatch.setattr(client.mod, "BENCHMARKS_FILE", bench)
    rows = client.get("/api/benchmarks").json()
    assert not any(r.get("is_baseline") for r in rows)
    assert any(r.get("epoch") == "Epoch 1" for r in rows)
```

Expose the module on the fixture (`client.mod = mod`) so the tests can call `get_default_benchmarks()` and monkeypatch `PREFLIGHT_FILE`. Update the existing start-training assertion to `--max_benchmark_samples == "1000"` and `--preflight_samples == "1000"`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`  
Expected: FAIL because the server still uses `configs/baselines.json` and hardcoded rows.

- [ ] **Step 3: Implement the server changes**

Delete `_HARDCODED_BASELINES` and `_load_shared_baselines`. Add `PREFLIGHT_FILE`. Implement `_load_preflight_benchmarks()` reading `PREFLIGHT_FILE`, iterating `("buzzasr_base", "whisper_v3")`, and returning rows with `epoch_number=0`, `type="baseline"`, `is_baseline=True`, and `canary_transcriptions=[]` merged with the preflight row. `get_default_benchmarks()` returns that list. In `start_training`, change the `max_bench` default to `1000` and append `--preflight_samples str(max_bench_i)` to the command. In `/api/benchmarks`, after reading a non-empty list, drop `is_baseline` rows whose `source` is not `"preflight"`; if the filtered list is empty, fall back to `get_default_benchmarks()`. Write defaults only when the list is non-empty.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: serve preflight baselines in the cockpit API"
```

---

### Task 6: Web UI phone-delta columns, summary, and empty state

**Files:**
- Modify: `server/static/index.html`
- Modify: `server/static/app.js`
- Modify: `server/static/style.css` (only if a summary style is needed)
- Create: `tests/test_web_ui_contract.py`

**Interfaces:**
- Consumes: preflight baseline rows with `total_samples`, `normalized_wer`, `phone_normalized_wer`, `is_reference`
- Produces: table columns `Δ clean vs base` and `Δ phone vs base`
- Produces: summary text `Phone WER <base>% → <best>% (<delta> pp)`

- [ ] **Step 1: Write the failing UI contract tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_columns_use_measured_deltas():
    html = (ROOT / "server/static/index.html").read_text(encoding="utf-8")
    assert "Δ clean vs base" in html
    assert "Δ phone vs base" in html
    assert "vs Whisper v3" not in html
    assert 'id="benchmarkSummary"' in html


def test_ui_has_no_hardcoded_baseline_fallbacks():
    js = (ROOT / "server/static/app.js").read_text(encoding="utf-8")
    assert "WHISPER_WER_FALLBACK" not in js
    assert "BUZZASR_WER_FALLBACK" not in js
    assert "No measured baseline yet" in js
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q`  
Expected: FAIL on the old column labels and fallback constants.

- [ ] **Step 3: Update the HTML**

Change the subtitle to `<p id="benchmarkSubtitle">Held-out test split. Lower is better.</p>` and add `<p id="benchmarkSummary" class="muted"></p>` under the section head. Replace the header row with:

```html
<th scope="col">Checkpoint</th>
<th scope="col">Clean WER</th>
<th scope="col" class="num">Δ clean vs base</th>
<th scope="col" class="num">Phone WER</th>
<th scope="col" class="num">Δ phone vs base</th>
<th scope="col" class="num opt">CER</th>
<th scope="col" class="num opt">Formal</th>
<th scope="col" class="num opt">Informal</th>
<th scope="col" class="num opt">ZWNJ → space</th>
```

Keep the empty-row `colspan="9"`.

- [ ] **Step 4: Update `updateBenchmarks`**

Remove the two fallback constants and `findBaseline`. Add `benchmarkSubtitle` and `benchmarkSummary` to `el`. On an empty list, render one row with `No measured baseline yet. Start a run to generate the preflight.` and return. Find the BuzzASR baseline row; compute clean and phone deltas only when `row.total_samples === baseline.total_samples`. Use `const scaleMax = Math.max(1, ...list.map((r) => Number(r.normalized_wer) || 0)) * 1.08;` with no fallback baseline. Render the nine cells above. Tag baseline rows `Baseline` or `Reference`; tag the best clean row `Best`. Build the summary from the best non-baseline phone row and the baseline phone WER. Change `formatDelta` to use `pp` instead of `pts`.

- [ ] **Step 5: Verify syntax and run the UI contract tests**

Run: `node --check server/static/app.js`  
Expected: no output, exit 0.

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/style.css tests/test_web_ui_contract.py
git commit -m "feat: show measured phone WER improvement in the cockpit"
```

---

### Task 7: Delete hardcoded baselines and update docs/demo

**Files:**
- Delete: `configs/baselines.json`
- Modify: `scripts/demo_local.py`
- Modify: `tests/test_normalizer_metrics.py`
- Modify: `tests/test_script_defaults.py`
- Modify: `README.md`
- Modify: `GUIDE.md`
- Modify: `PRODUCTION.md`

**Interfaces:**
- Produces: no production module reads `configs/baselines.json`
- Produces: demo-only `DEMO_SIMULATED_BASELINES` constant clearly labeled
- Produces: docs describing the automatic 1,000-row preflight

- [ ] **Step 1: Write the failing no-hardcoded-baseline test**

```python
def test_no_production_baseline_file_or_fallbacks():
    assert not (ROOT / "configs" / "baselines.json").exists()
    for rel in ("src/trainer_callbacks.py", "server/app.py",
                "server/static/app.js", "scripts/train_lora.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "28.5" not in text and "17.2" not in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py::test_no_production_baseline_file_or_fallbacks -q`  
Expected: FAIL because `configs/baselines.json` still exists.

- [ ] **Step 3: Remove the production baseline source**

Delete `configs/baselines.json` with `git rm`. In `scripts/demo_local.py`, replace `_load_demo_baselines()` with a `DEMO_SIMULATED_BASELINES` tuple and a comment: `# Simulated local-demo values only; never read by training or the server.` Keep the existing demo metrics. Remove `test_baselines_config` and its call from `tests/test_normalizer_metrics.py`.

- [ ] **Step 4: Update docs**

Confirm Task 3 already set `max_benchmark_samples: 1000` and `preflight_samples: 1000` in `configs/l40s_train_config.yaml` and `run_train.sh`. Update `README.md`, `GUIDE.md`, and `PRODUCTION.md`:

- preflight is mandatory and cached at `results/baselines/preflight.json`;
- both preflight and epoch matrix use the first 1,000 test rows;
- BuzzASR is the authoritative baseline, Whisper is reference-only;
- `configs/baselines.json` is gone;
- epoch deltas are `Δ clean vs base` and `Δ phone vs base`, in pp;
- the final full-test report shows no delta without a matching full baseline;
- for smoke runs, pass matching small values such as `--preflight_samples 20 --max_benchmark_samples 20`.

- [ ] **Step 5: Run the no-hardcoded test and the full suite**

Run: `.venv/bin/python -m pytest tests/test_script_defaults.py tests/test_normalizer_metrics.py -q`  
Expected: PASS.

Run: `.venv/bin/python -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove hardcoded baselines; require preflight"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run the full Python suite**

Run: `.venv/bin/python -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 2: Verify JS and shell syntax**

Run: `node --check server/static/app.js`  
Run: `bash -n run_train.sh`  
Expected: both exit 0 with no output.

- [ ] **Step 3: Verify no production baseline values remain**

Run: `git grep -n "28\.5\|17\.2\|configs/baselines.json" -- src scripts server README.md GUIDE.md PRODUCTION.md configs ':!scripts/demo_local.py'`  
Expected: no matches.

- [ ] **Step 4: Inspect the final diff**

Run: `git diff --check`  
Run: `git status --short`  
Expected: clean whitespace check; only intended files changed.

- [ ] **Step 5: Commit any verification fixes**

```bash
git add -A
git commit -m "test: verify preflight baseline integration"
```
