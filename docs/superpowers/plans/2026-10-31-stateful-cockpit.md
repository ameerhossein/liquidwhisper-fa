# Stateful Cockpit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every cockpit launch a self-contained persistent run with a durable config, per-optimizer-step records, stage history, restart recovery, resume, and merge, while remaining fully developable on a CPU-only machine.

**Architecture:** Add a stdlib `src/run_state.py` that owns one directory per run (`run.json`, `events.jsonl`, `steps.jsonl`, telemetry, benchmarks, logs). The FastAPI server creates/loads runs, scans all run directories for live trainer/merge PIDs, and exposes run APIs. The trainer records preflight/training/post-eval stages and appends steps. A fake trainer and fake merge make the whole cockpit testable without CUDA.

**Tech Stack:** Python 3, FastAPI, `fcntl`/`json`/`pathlib` (stdlib), PyTorch/HuggingFace only inside the real trainer, vanilla JS, Node for pure-helper tests, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-stateful-cockpit-design.md`

## Global Constraints

- `/workspace` (or `/runpod-volume`) is the only RunPod persistent root; the local fallback is repo-local `results/runs`.
- One full run, one smoke test, and one merge may be active at a time; the ledger keeps unlimited historical runs.
- The active run owns the live trainer PID and may be older than the newest run. Scan all run directories; do not assume newest = active.
- Every optimizer step appends one `steps.jsonl` record; model weights remain checkpointed only at `--save_steps`.
- `run.json` writes are atomic (temp + `os.replace`); read-modify-write updates hold `fcntl.flock` on `<run_dir>/.run.lock`.
- A cockpit run must reject an explicit `results_dir`. Its default `output_dir` is `<run_dir>/output`.
- `trainer_args` stores the full effective argparse namespace, not only the six form fields.
- `interrupted` means no observable exit; a definite nonzero/killed exit is `failed`; a requested halt is `stopped`.
- A failed merge does not change an otherwise completed run.
- Development and CI must pass with CPU-only torch; no test may require CUDA. Real training is a RunPod acceptance step.
- Keep existing `/api/status`, `/api/benchmarks`, `/api/loss-history`, `/api/logs`, `/api/train/start`, `/api/train/stop`, smoke endpoints, and `/ws/live` fields backward-compatible.
- Do not modify smoke-test isolation or include smoke runs in the ledger.
- Use the existing managed paths in `src/paths.py`; do not hardcode `/workspace` outside it.
- Before Tasks 7–8, load the `interface-design` skill and follow the existing cockpit visual language.

## Review Focus

- Server restarts while the trainer is alive, including when an older run owns the PID: the cockpit must attach to that run and keep streaming it.
- Pod/server restart with a dead PID and a partially written last JSONL line: the run must become `interrupted`, the partial line ignored, and Resume offered.
- Halt during preflight or post-eval: the run must become `stopped`, remaining stages `skipped`, and the grace-period kill must not become `failed`.
- Resume after post-eval failure: it must use `--post_eval_only`, never replay optimizer steps, and never duplicate `steps.jsonl`.
- A non-persistent runs root or an explicit `results_dir`: the first must be visible as a warning, the second rejected, never silently accepted.

---

## Task 1: Run-state ledger and managed paths

**Files:**
- Create: `src/run_state.py`
- Modify: `src/paths.py`
- Test: `tests/test_run_state.py`

**Interfaces:**
- Produces:
  - `new_run_id(now: float | None = None) -> str`
  - `run_id_valid(run_id: str) -> bool`
  - `create_run(runs_root: str | Path, *, config: dict, trainer_args: dict, paths: dict | None = None, source: str = "cockpit", run_id: str | None = None) -> dict`
  - `load_run(run_dir: str | Path) -> dict | None`
  - `list_runs(runs_root: str | Path, limit: int = 50) -> list[dict]`
  - `update_run(run_dir: str | Path, mutator: Callable[[dict], dict | None]) -> dict`
  - `set_status(run_dir, status: str, **fields) -> dict`
  - `set_stage(run_dir, name: str, status: str, detail: str | None = None, **fields) -> dict`
  - `set_progress(run_dir, *, step: int, step_count: int, last_step_at: float) -> dict`
  - `append_event(run_dir, event: dict) -> None`
  - `append_step(run_dir, record: dict) -> None`
  - `read_steps(run_dir, tail: int = 2000) -> list[dict]`
  - `step_count(run_dir) -> int`
  - `start_attempt(run_dir, pid: int, reason: str = "initial") -> dict`
  - `finish_attempt(run_dir, exit_code: int | None, reason: str) -> dict`
  - `reconcile(run_dir, *, pid_alive: bool, exit_code: int | None = None, reason: str | None = None) -> dict`
  - `reconcile_all(runs_root, *, is_alive: Callable[[int], bool]) -> dict` returning
    `{"active_run": dict | None, "latest_run": dict | None, "merge_run": dict | None,
    "runs": list[dict]}`
  - `storage_info(runs_root) -> dict`
  - `write_preflight_copy(run_dir, source_path) -> Path`
  - `STAGES`, `STAGE_STATUSES`, `RUN_STATUSES`
- `src/paths.py` produces `default_runs_dir() -> str` and `resolve_runs_dir(raw: str | None = None) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_state.py` with tests for:

```python
def test_new_run_id_format_and_validation():
    rid = run_state.new_run_id(now=1761846420.0)
    assert re.match(r"^\d{8}-\d{6}-[0-9a-f]{6}$", rid)
    assert run_state.run_id_valid(rid)
    assert not run_state.run_id_valid("../escape")


def test_create_run_writes_manifest_and_pending_stages(tmp_path):
    run = run_state.create_run(
        tmp_path, config={"epochs": 4}, trainer_args={"epochs": 4},
        paths={"output_dir": str(tmp_path / "run" / "output")},
        run_id="20261030-194700-1a2b3c",
    )
    assert run["status"] == "starting"
    assert [s["name"] for s in run["stages"]] == ["preflight", "training", "post_eval", "merge"]
    assert all(s["status"] == "pending" for s in run["stages"])
    assert run_state.load_run(tmp_path / run["run_id"])["config"]["epochs"] == 4


def test_list_runs_is_newest_first(tmp_path):
    # create two fixed-id runs with different created_at
    ...


def test_update_run_preserves_fields_across_writers(tmp_path):
    # two updates: one sets config, one sets last_error; both survive
    ...


def test_set_stage_tracks_current_stage_and_timestamps(tmp_path):
    # preflight running -> completed; training running -> current_stage training
    ...


def test_append_step_reads_tail_and_skips_bad_trailing_line(tmp_path):
    # append 5 steps, append "{bad", read tail=3 => last 3 valid, count=5
    ...


def test_read_steps_keeps_last_record_per_step(tmp_path):
    # append step 1 twice; read returns one step 1 (the later record)
    ...


def test_attempt_lifecycle_and_reconcile_stopped(tmp_path):
    # start_attempt; write stop_requested_at; reconcile dead => stopped + stage stopped
    ...


def test_reconcile_classifies_failed_and_interrupted(tmp_path):
    # exit_code=1 => failed; exit_code=None => interrupted
    ...


def test_reconcile_leaves_a_fresh_starting_run_alone(tmp_path):
    # create_run status=starting, no attempts, created_at=now; reconcile dead
    # => still starting
    ...


def test_reconcile_all_finds_older_live_run(tmp_path):
    # run A older pid=101 alive; run B newer pid=202 dead; active_run_id == A
    ...


def test_storage_info_marks_workspace_persistent(tmp_path, monkeypatch):
    # /workspace path => persistent True; tmp_path => False
    ...


def test_write_preflight_copy_is_atomic_and_readable(tmp_path):
    ...
```

Also extend `tests/test_script_defaults.py` (or add `tests/test_paths.py`) for:

```python
def test_runs_dir_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    assert paths.resolve_runs_dir(None) == str(tmp_path / "runs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_run_state.py -q`
Expected: FAIL with `ModuleNotFoundError: src.run_state` / missing `resolve_runs_dir`.

- [ ] **Step 3: Implement `src/run_state.py`**

Implement the interfaces above:

- `STAGES = ("preflight", "training", "post_eval", "merge")`.
- `create_run` creates `<runs_root>/<run_id>/`, `output/`, `baselines/`, writes `run.json` with `schema_version: 1`, `status: "starting"`, default path entries, pending stages, empty attempts, `step_count: 0`.
- `update_run` opens `<run_dir>/.run.lock` with `a+`, takes `fcntl.flock(LOCK_EX)`, loads `run.json`, applies the mutator, updates `updated_at`, writes `run.json.tmp`, and `os.replace`s it. Use a no-op fallback when `fcntl` is unavailable.
- `set_stage` validates name/status, sets `started_at` on `running`, `ended_at` on terminal statuses, updates `current_stage`, and appends a `stage` event.
- `set_status` validates status, sets `ended_at` for terminal statuses, and appends an `event`; accept `failed_stage`, `last_error`, `stop_requested_at`.
- `append_event`/`append_step` write one JSON line with `ts` defaulted to `time.time()`, flush every line, and `fsync` only in `append_event`.
- `read_steps` reads all lines, ignores malformed lines, deduplicates by `step` keeping the last, sorts by step, returns the tail. `step_count` returns unique-step count.
- `reconcile` only acts when `status in ("starting", "running")` and `pid_alive`
  is false. A `starting` run with no attempts and `created_at` newer than 15
  seconds is left alone, so the monitor cannot race a spawn that is still in
  progress. Classification:
  - stop requested -> `stopped`; running stage `stopped`; later stages `skipped`.
  - `exit_code == 0` -> `completed` only when training is completed and
    post-eval is completed/skipped; otherwise `failed` with `failed_stage` set
    to `post_eval` when training completed, else `training`.
  - `exit_code not in (None, 0)` -> `failed`; `failed_stage` is the last
    running stage (or `training`).
  - `exit_code is None` -> `interrupted` (and running stage `interrupted`).
  - Finish the open attempt with the matching reason.
- `start_attempt` appends the attempt and also sets top-level `current_pid` to
  the PID; `finish_attempt` clears `current_pid`.
- `reconcile_all` loads all runs; the active run is the first with a live
  `current_pid`; also reports a live `merge_pid`; reconciles dead
  starting/running runs except those with live PIDs.

- [ ] **Step 4: Add managed runs paths**

In `src/paths.py`:

```python
RUNS_DIR_PERSISTENT = "/workspace/stt-outputs/results/runs"
RUNS_DIR_LOCAL = "results/runs"

def default_runs_dir() -> str:
    env = _env("ASR_RUNS_DIR")
    if env:
        return env
    return str(Path(default_results_dir()) / "runs")

def resolve_runs_dir(raw: Optional[str] = None) -> str:
    raw = (raw or "").strip() or default_runs_dir()
    return _fallback_if_persistent_missing(raw, RUNS_DIR_LOCAL, "runs")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_run_state.py tests/test_script_defaults.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/run_state.py src/paths.py tests/test_run_state.py tests/test_script_defaults.py
git commit -m "feat: add persistent run ledger and managed runs path"
```

---

## Task 2: Preflight cache dir, stop hook, and trainer run stages

**Files:**
- Modify: `src/preflight.py`
- Modify: `scripts/train_lora.py`
- Test: `tests/test_preflight.py`
- Test: `tests/test_train_lora_run_stages.py`

**Interfaces:**
- `run_preflight(..., cache_dir: Path | None = None, should_stop: Callable[[], bool] | None = None) -> dict`
  - Cached result returns `{**cached, "cached": True}`; fresh result returns `{**result, "cached": False}`.
  - `should_stop()` is checked before each model attempt; if true, raise `RuntimeError("halt requested")`.
- `scripts/train_lora.py` gains:
  - `--run_dir PATH`
  - `--preflight_cache_dir PATH`
  - `--post_eval_only` (BooleanOptionalAction, default False)
  - helpers `_copy_preflight_to_run(run_dir, cache_dir)`, `_mark_run_stage(run_dir, name, status, detail=None)`, `_record_trainer_args(run_dir, args)`.
- Consumes `src.run_state` from Task 1.

- [ ] **Step 1: Write the failing preflight tests**

Extend `tests/test_preflight.py`:

```python
def test_run_preflight_uses_explicit_cache_dir(tmp_path, monkeypatch):
    # results_dir has no cache; cache_dir has a matching cached file
    # assert run_preflight(..., cache_dir=cache_dir)["cached"] is True
    # and no subprocess.run call was made


def test_run_preflight_stops_before_first_model(tmp_path, monkeypatch):
    calls = []
    def stop():
        calls.append(1)
        return True
    with pytest.raises(RuntimeError, match="halt requested"):
        run_preflight(..., should_stop=stop)
    assert calls
```

- [ ] **Step 2: Run preflight tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: FAIL with unexpected keyword argument `cache_dir` / `should_stop`.

- [ ] **Step 3: Implement preflight cache/stop**

Modify `run_preflight`:

```python
def run_preflight(*, results_dir: Path, cache_dir: Optional[Path] = None, ...,
                  should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    path = preflight_path(Path(cache_dir or results_dir).resolve())
    if should_stop and should_stop():
        raise RuntimeError("halt requested")
    if not overwrite:
        cached = load_cached_preflight(path, fingerprint)
        if cached is not None:
            return {**cached, "cached": True}
    ...
    for key in (...):
        if should_stop and should_stop():
            raise RuntimeError("halt requested")
        ...
    return {**result, "cached": False}
```

- [ ] **Step 4: Write the failing trainer stage tests**

Create `tests/test_train_lora_run_stages.py`:

```python
def test_parser_accepts_run_dir_and_preflight_cache(tmp_path, monkeypatch):
    from scripts.train_lora import parse_args
    monkeypatch.setattr(sys, "argv", [
        "train_lora.py", "--run_dir", str(tmp_path / "run"),
        "--preflight_cache_dir", str(tmp_path / "cache"),
    ])
    args = parse_args()
    assert args.run_dir == str(tmp_path / "run")
    assert args.preflight_cache_dir == str(tmp_path / "cache")


def test_copy_preflight_to_run_writes_manifest_copy(tmp_path):
    from scripts.train_lora import _copy_preflight_to_run
    cache = tmp_path / "cache"
    (cache / "baselines").mkdir(parents=True)
    src = cache / "baselines" / "preflight.json"
    src.write_text('{"models": {}}')
    run = tmp_path / "run"
    dest = _copy_preflight_to_run(run, cache)
    assert dest == run / "baselines" / "preflight.json"
    assert dest.read_text() == '{"models": {}}'


def test_record_trainer_args_stores_full_namespace(tmp_path):
    from scripts.train_lora import _record_trainer_args
    run = run_state.create_run(tmp_path, config={}, trainer_args={}, run_id="20261030-194700-1a2b3c")
    _record_trainer_args(tmp_path / run["run_id"], {"lora_r": 64, "warmup_steps": 500})
    saved = run_state.load_run(tmp_path / run["run_id"])
    assert saved["trainer_args"]["lora_r"] == 64
```

- [ ] **Step 5: Implement trainer args, copy, and stage helpers**

In `scripts/train_lora.py`:

- Add the three parser arguments.
- Add `_copy_preflight_to_run(run_dir: Path, cache_dir: Path) -> Path` using `run_state.write_preflight_copy`.
- Add `_mark_run_stage(run_dir: str | None, name: str, status: str, detail: str | None = None)` that no-ops when `run_dir` is falsy, otherwise calls `run_state.set_stage`.
- Add `_record_trainer_args(run_dir, args)` that writes `vars(args)` (JSON-safe: convert `Path` to `str`; keep lists) to `run.json.trainer_args`.
- In `main()`:
  - Resolve `run_dir = Path(args.run_dir).resolve()` when provided.
  - If `args.post_eval_only`: call `_run_post_eval_only(args)` and return.
  - Mark `preflight` running with detail `preparing datasets` before data preparation.
  - Wrap data preparation in `try/except`: on exception mark preflight failed, status failed, `failed_stage="preflight"`, re-raise.
  - Before `run_preflight`, mark preflight detail `measuring baseline`.
  - Use `cache_dir = Path(args.preflight_cache_dir) if args.preflight_cache_dir else Path(args.results_dir)`.
  - Pass `should_stop=lambda: os.path.exists(os.path.join(args.results_dir, "stop_training.signal"))`.
  - On success copy the preflight and mark preflight completed with `detail="cache hit"` or `detail=f"measured, n={...}"`.
  - Mark training running before `trainer.train()`; on return mark completed unless `telemetry_cb.halt_requested`, then mark `stopped` and skip post-eval.
  - Mark post_eval running/completed/skipped; on failure mark failed and status failed with `failed_stage="post_eval"`.
  - Mark overall completed only when training completed and post-eval completed/skipped.
  - Record `trainer_args` through `_record_trainer_args`.
  - Extend `_clear_previous_state` to remove `steps.jsonl` and `events.jsonl`
    when `--fresh_start` is used, so a fresh CLI run cannot append to an old
    step log. Keep the existing `loss_history.json` handling.

- [ ] **Step 6: Add the halt check to data prep/preflight**

Use the existing stop signal path `<results_dir>/stop_training.signal`. If it exists before data prep, mark preflight stopped, post_eval skipped, status stopped, remove the signal, and return. Inside `run_preflight`, the `should_stop` callback covers the preflight subprocess boundaries.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_preflight.py tests/test_train_lora_run_stages.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/preflight.py scripts/train_lora.py tests/test_preflight.py tests/test_train_lora_run_stages.py
git commit -m "feat: record preflight and trainer stages in the run ledger"
```

---

## Task 3: Per-step JSONL, progress, halt, and post-eval-only mode

**Files:**
- Modify: `src/trainer_callbacks.py`
- Modify: `scripts/train_lora.py`
- Test: `tests/test_trainer_steps.py`
- Test: `tests/test_train_lora_run_stages.py`

**Interfaces:**
- `WebTelemetryCallback.__init__(..., run_dir: str | None = None, attempt: int = 1)`.
- `WebTelemetryCallback` appends `steps.jsonl` to `results_dir` on every `on_step_end`, calls `run_state.set_progress` every 100 steps and at `on_train_end`.
- `_run_post_eval_only(args)` loads `best_adapter`, optional `trainer_state.json`, calls `_run_post_training_evaluation`, and updates stages/status.

- [ ] **Step 1: Write the failing callback tests**

Create `tests/test_trainer_steps.py`:

```python
def test_on_step_end_appends_one_record_per_step(tmp_path):
    # skip if torch/transformers missing
    cb = WebTelemetryCallback(
        results_dir=str(tmp_path), run_dir=str(tmp_path),
        test_dataset=None, processor=None, metrics_calculator=None,
    )
    state = SimpleNamespace(global_step=1, max_steps=10, epoch=0.1,
                            log_history=[{"step": 1, "loss": 0.5, "learning_rate": 1e-4}])
    args = SimpleNamespace()
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_step_end(args, state, control)
    lines = (tmp_path / "steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 1
    assert record["loss"] == 0.5
    assert record["loss_step"] == 1


def test_steps_file_is_uncapped_and_resumes(tmp_path):
    # append 600 records through the helper; loss_history stays <=500
    # create a second callback with the same results_dir; next step appends
```

- [ ] **Step 2: Run callback tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_trainer_steps.py -q`
Expected: FAIL because `steps.jsonl` is not written.

- [ ] **Step 3: Implement step records and progress**

In `WebTelemetryCallback`:

- Add `run_dir`/`attempt` fields.
- `self.steps_file = os.path.join(self.results_dir, "steps.jsonl")`.
- Set `self.step_count = run_state.step_count(self.results_dir)` in `__init__` (0 if absent).
- In `on_step_end`, after the telemetry write:

```python
run_state.append_step(self.results_dir, {
    "step": state.global_step,
    "epoch": round(state.epoch or 0.0, 2),
    "loss": latest_loss,
    "loss_step": latest_loss_step,
    "learning_rate": latest_lr,
    "speed_s_per_step": round(avg_step_dt, 3),
    "elapsed_seconds": int(now - self.start_time),
    "eta_seconds": eta_seconds,
    "attempt": self.attempt,
})
self.step_count += 1
if self.run_dir and (self.step_count % 100 == 0):
    run_state.set_progress(self.run_dir, step=state.global_step,
                           step_count=self.step_count, last_step_at=now)
```

Track `latest_loss_step` while scanning `state.log_history` (the `step` key of the entry that supplied the loss).

- `on_train_end` calls `set_progress` with the final count when `run_dir` is set.

- [ ] **Step 4: Write the failing post-eval-only test**

In `tests/test_train_lora_run_stages.py`:

```python
def test_post_eval_only_calls_report_without_training(tmp_path, monkeypatch):
    # create run dir with output/best_adapter/adapter_config.json
    # monkeypatch processor/model loaders and _run_post_training_evaluation
    # call _run_post_eval_only(args)
    # assert report called once, no trainer.train
```

- [ ] **Step 5: Implement `_run_post_eval_only`**

In `scripts/train_lora.py`:

```python
def _run_post_eval_only(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir)
    adapter_path = output_dir / "best_adapter"
    if not (adapter_path / "adapter_config.json").is_file():
        raise SystemExit("post-eval-only: best_adapter/adapter_config.json not found")
    _mark_run_stage(run_dir, "post_eval", "running", detail="post-eval only")
    # load processor from adapter_path, base model, PeftModel.from_pretrained
    # load trainer_state.json from the newest complete checkpoint if available
    # load run baselines/preflight.json for baseline_benchmark
    # call _run_post_training_evaluation(...)
    _mark_run_stage(run_dir, "post_eval", "completed")
    run_state.set_status(run_dir, "completed")
```

It must not call `trainer.train()`, must not append steps, and must use `args.results_dir`/`args.output_dir` from the run.

- [ ] **Step 6: Wire post-eval-only into the parser/main**

`--post_eval_only` branches immediately after path resolution and before processor/data prep. It does not run preflight or training.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_trainer_steps.py tests/test_train_lora_run_stages.py -q`
Expected: PASS (or skip only if torch/transformers are absent).

- [ ] **Step 8: Commit**

```bash
git add src/trainer_callbacks.py scripts/train_lora.py tests/test_trainer_steps.py tests/test_train_lora_run_stages.py
git commit -m "feat: persist every optimizer step and resume post-eval only"
```

---

## Task 4: GPU-free fake trainer and fake merge

**Files:**
- Create: `scripts/dev_fake_trainer.py`
- Create: `scripts/dev_fake_merge.py`
- Test: `tests/test_fake_trainer.py`

**Interfaces:**
- `dev_fake_trainer.py` accepts `--run_dir`, `--results_dir`, `--output_dir`, `--preflight_cache_dir`, plus unknown real-trainer flags via `parse_known_args`.
- Environment: `ASR_FAKE_STEPS` (default 200), `ASR_FAKE_STEP_DELAY` (default 0.05), `ASR_FAKE_FAIL_STAGE` (`preflight|training|post_eval`), `ASR_FAKE_HANG=1`.
- `dev_fake_merge.py` accepts `--adapter_path`, `--output_dir`, writes `config.json`, `model.safetensors`, and `tokenizer_config.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fake_trainer.py`:

```python
def test_fake_trainer_completes_and_writes_steps(tmp_path):
    # subprocess.run([sys.executable, "scripts/dev_fake_trainer.py",
    #   "--run_dir", str(run_dir), "--results_dir", str(run_dir),
    #   "--output_dir", str(run_dir / "output"),
    #   "--preflight_cache_dir", str(tmp_path / "cache")],
    #   env={**os.environ, "ASR_FAKE_STEPS": "12", "ASR_FAKE_STEP_DELAY": "0"})
    # assert returncode == 0
    # run = run_state.load_run(run_dir)
    # assert run["status"] == "completed"
    # assert all(stage["status"] == "completed" for stage in run["stages"])
    # assert run_state.step_count(run_dir) == 12
    # assert (run_dir / "live_telemetry.json").exists()


def test_fake_trainer_forced_failure(tmp_path):
    # ASR_FAKE_FAIL_STAGE=training => returncode 1, status failed, failed_stage training


def test_fake_trainer_halt_marks_stopped(tmp_path):
    # start process with ASR_FAKE_STEPS=1000, ASR_FAKE_STEP_DELAY=0.01
    # wait for run.json current_stage == training, write stop_training.signal
    # wait; assert status stopped and post_eval skipped


def test_fake_merge_writes_artifacts(tmp_path):
    # run dev_fake_merge; assert merged/config.json and model.safetensors
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fake_trainer.py -q`
Expected: FAIL with missing script.

- [ ] **Step 3: Implement the fake trainer**

Use only stdlib + `src.run_state`:

- Create/load the run with `source="fake"`.
- `preflight`: set running, write a fake `preflight.json` to `run_dir/baselines/` and `preflight_cache_dir/baselines/`, set completed.
- `training`: set running; loop `ASR_FAKE_STEPS`; write `live_telemetry.json` atomically (step, max_steps, epoch, loss, lr, speed, eta), append `loss_history.json` capped at 500, append `steps.jsonl` via `run_state.append_step`; call `set_progress` every 10 steps; sleep `ASR_FAKE_STEP_DELAY`; check `stop_training.signal`.
- If stopped: mark training stopped, post_eval skipped, status stopped, remove signal, exit 0.
- `post_eval`: set running, write a fake benchmark row to `epoch_benchmarks.json`, write `output/final_evaluation_report.json`, set completed.
- Create `output/best_adapter/adapter_config.json` before post_eval.
- Forced failure: set the named stage failed, status failed, `failed_stage`, exit 1.
- Hang: after entering training, sleep in a loop until the stop signal appears (for Halt testing).

- [ ] **Step 4: Implement the fake merge**

Write placeholder files under `--output_dir`; exit 0. It must not import torch.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_fake_trainer.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/dev_fake_trainer.py scripts/dev_fake_merge.py tests/test_fake_trainer.py
git commit -m "test: add GPU-free fake trainer and merge"
```

---

## Task 5: Server run creation, resume, and stop

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_server_train.py`

**Interfaces:**
- `RUNS_DIR`, `ACTIVE_RUN_ID`, `LATEST_RUN`, `_refresh_runs()`, `_current_run()`, `_active_run()`, `_run_paths(run)`.
- `_start_run(*, config: dict, resume_run_id: str | None = None) -> dict`.
- `POST /api/train/start` creates a run; `POST /api/runs/{run_id}/resume` resumes it; `POST /api/train/stop` targets the active run.
- Consumes `src.run_state` and `src.paths.resolve_runs_dir`.

- [ ] **Step 1: Update the test fixture for real PIDs**

In `tests/test_server_train.py`:

- Add `from src import run_state`.
- Set `_FakeProcess.pid = os.getpid()` so `_is_process_running` sees a live process.
- Add a helper `_run_json(client, run_id)` and `_write_run(client, **fields)`.
- Keep existing smoke tests unchanged.

- [ ] **Step 2: Write the failing start tests**

```python
def test_start_creates_self_contained_run(client):
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    run_dir = client.mod.RUNS_DIR / run_id
    run = run_state.load_run(run_dir)
    assert run["status"] == "running"
    assert run["paths"]["results_dir"] == str(run_dir)
    assert run["paths"]["output_dir"] == str(run_dir / "output")
    flags = _flags(client.launched[0])
    assert flags["--run_dir"] == str(run_dir)
    assert flags["--preflight_cache_dir"] == str(client.mod.RUNS_DIR / "_shared")
    assert flags["--output_dir"] == str(run_dir / "output")
    assert flags["--results_dir"] == str(run_dir)
    assert run["trainer_args"]["epochs"] == 4


def test_start_rejects_explicit_results_dir(client):
    res = client.post("/api/train/start", json={"results_dir": "/tmp/elsewhere"})
    assert res.status_code == 400
    assert "results_dir" in res.json()["detail"]
    assert client.launched == []


def test_status_includes_run_and_storage(client):
    client.post("/api/train/start", json={})
    body = client.get("/api/status").json()
    assert body["run"]["run_id"]
    assert body["active_run_id"] == body["run"]["run_id"]
    assert body["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)
    assert body["steps"]["count"] == 0
```

- [ ] **Step 3: Write the failing resume/stop tests**

```python
def test_resume_reuses_paths_and_appends_attempt(client):
    start = client.post("/api/train/start", json={}).json()
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    run = run_state.load_run(client.mod.RUNS_DIR / start["run_id"])
    assert len(run["attempts"]) == 2
    flags = _flags(client.launched[-1])
    assert flags["--output_dir"] == run["paths"]["output_dir"]
    assert "--auto_resume" in client.launched[-1]


def test_resume_after_training_completed_uses_post_eval_only(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "failed")
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    assert "--post_eval_only" in client.launched[-1]


def test_stop_writes_run_stop_signal(client):
    start = client.post("/api/train/start", json={}).json()
    res = client.post("/api/train/stop", json={})
    assert res.status_code == 200
    assert (client.mod.RUNS_DIR / start["run_id"] / "stop_training.signal").exists()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q -k "run or stop"`
Expected: FAIL with missing `RUNS_DIR` / `run_id`.

- [ ] **Step 5: Implement run resolution and start**

In `server/app.py`:

- Import `run_state`; resolve `RUNS_DIR = Path(_managed_paths.resolve_runs_dir(None))`; resolve relative to `BASE_DIR`; `mkdir`.
- Extend `_sanitize_output_dir` to allow paths inside `RUNS_DIR` even when the
  runs root is outside `/workspace`, the repo, and `ASR_ALLOWED_OUTPUT_ROOTS`.
- Add `ACTIVE_RUN_ID: str | None = None`, `LATEST_RUN: dict | None = None`, `CURRENT_RUN: dict | None = None`.
- `_refresh_runs()`:
  - `summary = run_state.reconcile_all(RUNS_DIR, is_alive=_is_process_running)`.
  - Set `ACTIVE_RUN_ID` / `CURRENT_RUN` / `LATEST_RUN` from the result.
  - Reconcile dead starting/running runs with `exit_code=None`.
- `_current_run()` returns the active run, else the latest run, else `None`.
- `_run_paths(run)` returns the six per-run file paths, falling back to the legacy constants when `run is None`.
- `_start_run(config, resume_run_id=None)`:
  - Validate as today; reject `results_dir` for a new cockpit run.
  - For a new run: `run_id = run_state.new_run_id()`; `run_dir = RUNS_DIR / run_id`; default `output_dir = run_dir / "output"`.
  - Build `trainer_args` from the normalized values. At minimum include:
    `base_model`, `dataset_name`, `output_dir`, `results_dir`, `epochs`,
    `batch_size`, `grad_accum`, `learning_rate`, `warmup_steps` (500),
    `lr_scheduler_type` ("cosine"), `eval_steps`, `save_steps`,
    `save_total_limit` (2), `eval_batch_size`, `max_benchmark_samples`,
    `preflight_samples`, `num_proc`, `telephony_prob`, `telephony_profiles`
    (["customer"]), `telephony_seed` (42), `telephony_use_ffmpeg` (true),
    `phone_benchmark` (true), `phone_seed` (1234), `gradient_checkpointing`,
    `run_post_test_eval` (true), `run_epoch_benchmark` (true),
    `enable_telemetry` (true), `lora_r` (64), `lora_alpha` (128),
    `lora_dropout` (0.05), `auto_resume`, `fresh_start`, `run_dir`. The trainer
    overwrites this with the full argparse namespace once it starts.
  - `run_state.create_run(RUNS_DIR, config=..., trainer_args=..., paths=...,
    run_id=run_id, source="fake" if ASR_FAKE_TRAINER else "cockpit")`.
  - Write initial telemetry and log.
  - Build the command with `--run_dir`, `--preflight_cache_dir`, run-scoped
    output/results. When `ASR_FAKE_TRAINER=1`, use
    `scripts/dev_fake_trainer.py` instead of `scripts/train_lora.py`; the fake
    script tolerates the real trainer flags.
  - For resume: load the run, build `trainer_args` from `run["trainer_args"]`,
    use the same paths, force `fresh_start=False` and `auto_resume=True`, drop
    any stored `resume_from_checkpoint`, and append `--post_eval_only` when
    training is completed and post_eval is not.
  - Spawn; on success `run_state.start_attempt(run_dir, pid, reason)`, write `training.pid`, set status running.
  - Return `{run_id, pid, config, paths, storage}`.
- Keep the `training_process`/`last_exit_code` globals for the spawned child.

- [ ] **Step 6: Implement resume/stop and route the start endpoint**

- `POST /api/runs/{run_id}/resume` validates the run status is one of
  `interrupted`, `stopped`, or `failed`, rejects while any job is active, and
  calls `_start_run(config, resume_run_id=run_id)`.
- `POST /api/train/start` calls `_start_run(config)`.
- `POST /api/train/stop` resolves the active run, writes its stop signal, sets `stop_requested_at`, and escalates against the PID.

- [ ] **Step 7: Route status/log/benchmark/loss to the current run**

- `_refresh_runs()` at the start of `/api/status`, `/api/benchmarks`, `/api/loss-history`, `/api/logs`, and the WebSocket loop.
- If a run exists, read/write that run's files; otherwise use the legacy constants.
- Extend `_derive_status(telemetry, is_running, run=None)` to use the run's
  `status`, `stop_requested_at`, and stage state instead of the in-memory
  `halt_requested` when a run is present. Preserve the legacy path when no run
  exists.
- Add `run`, `active_run_id`, `steps`, and `storage` to `/api/status`.
- Keep the existing fields unchanged.

- [ ] **Step 8: Run tests**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`
Expected: PASS; existing smoke tests still pass.

- [ ] **Step 9: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: create and resume self-contained cockpit runs"
```

---

## Task 6: Server recovery, monitor, merge, and run APIs

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_server_train.py`

**Interfaces:**
- `_monitor_loop()`, `_merge_running()`, `_start_merge(run_id)`, `_run_payload(run)`.
- `GET /api/runs?limit=20`, `GET /api/runs/latest`, `GET /api/runs/{run_id}`, `GET /api/runs/{run_id}/steps?tail=2000`, `POST /api/runs/{run_id}/merge`.
- Consumes `run_state.reconcile_all`, `read_steps`, `step_count`, `storage_info`.

- [ ] **Step 1: Write the failing recovery tests**

Add a test helper before the tests:

```python
def _create_run(client, *, pid, status="running", run_id="20261030-194700-aaaaaa"):
    run = run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_state.start_attempt(client.mod.RUNS_DIR / run_id, pid, "test")
    run_state.set_status(client.mod.RUNS_DIR / run_id, status)
    return run
```

```python
def test_refresh_finds_older_live_run(client):
    old = _create_run(client, pid=os.getpid(), run_id="20261030-194700-aaaaaa")
    new = _create_run(client, pid=999999, run_id="20261030-195700-bbbbbb")
    client.mod._refresh_runs()
    assert client.mod.ACTIVE_RUN_ID == old["run_id"]
    assert client.mod.LATEST_RUN["run_id"] == new["run_id"]


def test_refresh_marks_dead_run_interrupted(client):
    run = _create_run(client, pid=999999, status="running")
    client.mod._refresh_runs()
    assert run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["status"] == "interrupted"


def test_refresh_does_not_rewrite_completed_run(client):
    run = _create_run(client, pid=999999, status="completed")
    client.mod._refresh_runs()
    assert run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["status"] == "completed"
```

- [ ] **Step 2: Write the failing API/merge tests**

```python
def test_runs_endpoints_list_latest_and_steps(client):
    start = client.post("/api/train/start", json={}).json()
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == start["run_id"]
    latest = client.get("/api/runs/latest").json()
    assert latest["run"]["run_id"] == start["run_id"]
    steps = client.get(f"/api/runs/{start['run_id']}/steps?tail=5").json()
    assert steps["run_id"] == start["run_id"]
    assert steps["total"] == 0


def test_merge_starts_and_marks_stage(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "output" / "best_adapter").mkdir(parents=True)
    (run_dir / "output" / "best_adapter" / "adapter_config.json").write_text("{}")
    run_state.set_stage(run_dir, "training", "completed")
    res = client.post(f"/api/runs/{start['run_id']}/merge")
    assert res.status_code == 200, res.text
    assert "--adapter_path" in client.launched[-1]
    assert run_state.load_run(run_dir)["stages"][3]["status"] == "running"


def test_merge_rejected_before_training_completed(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "output" / "best_adapter").mkdir(parents=True)
    (run_dir / "output" / "best_adapter" / "adapter_config.json").write_text("{}")
    res = client.post(f"/api/runs/{start['run_id']}/merge")
    assert res.status_code == 400
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q -k "refresh or runs_endpoint or merge"`
Expected: FAIL with missing endpoints/merge.

- [ ] **Step 4: Implement `_refresh_runs`, monitor, and merge**

- `_refresh_runs` calls `run_state.reconcile_all`; sets `ACTIVE_RUN_ID`, `CURRENT_RUN`, `LATEST_RUN`.
- `_monitor_loop` calls `_training_running()` and `_merge_running()` first (so a
  known child exit code is applied), then `_refresh_runs()`, every 2 seconds.
  Start it from a FastAPI startup event; guard against duplicate tasks in tests.
- `_merge_running` polls `merge_process`; on exit 0 check `merged/config.json` + `model.safetensors`; update the merge stage; record exit code. If no child handle, read `merge.pid`; a dead PID with complete artifacts becomes completed, otherwise failed.
- `_start_merge(run_id)` validates training completed and adapter exists, sets merge running, spawns `dev_fake_merge.py` when `ASR_FAKE_TRAINER=1` else `merge_lora.py`, opens `merge.log`, writes `merge.pid`, records the PID.
- Extend `_any_job_running()` with `_merge_running()`.

- [ ] **Step 5: Implement run APIs and payload**

- `_run_payload(run)` returns `{run, telemetry, benchmarks, loss_history, steps, storage,
  process_running, adapter_exists}`; `adapter_exists` is true when
  `<output_dir>/best_adapter/adapter_config.json` exists.
- `GET /api/runs` returns summaries `{run_id, created_at, status, current_stage, step_count, source, output_dir}`.
- `GET /api/runs/latest` uses `_current_run()`; `run: null` when none.
- `GET /api/runs/{run_id}` validates then returns `_run_payload`.
- `GET /api/runs/{run_id}/steps` reads `run_state.read_steps` and `step_count`; cap `tail` at 10000.
- Include `storage` from `run_state.storage_info(RUNS_DIR)` in all run payloads and `/api/status`.
- Add `run`/`steps` to the WebSocket payload.

- [ ] **Step 6: Implement exit classification**

- In `_training_running`, when the child exits, call `run_state.reconcile(run_dir, pid_alive=False, exit_code=code, reason="process_exit")`.
- If only a PID file exists and the PID is dead, call `reconcile(..., exit_code=None)`; `reconcile` decides stopped/failed/interrupted from `stop_requested_at` and stage state.
- Ensure a requested stop wins over a signal-derived failure.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat: recover active runs, monitor exits, and merge from the cockpit"
```

---

## Task 7: UI run header, stepper, and state restoration

**Files:**
- Create: `server/static/run_view.js`
- Modify: `server/static/index.html`
- Modify: `server/static/app.js`
- Modify: `server/static/style.css`
- Test: `tests/test_run_view.js`
- Test: `tests/test_web_ui_contract.py`

**Interfaces:**
- `run_view.js` exports `STAGE_ORDER`, `stageMap(run)`, `stageLabel(name)`, `runStatusLabel(run)`, `savedStepsLabel(steps)`, `isMergeAvailable(run, adapterExists = false)`.
- DOM ids: `runSelect`, `runIdentity`, `runCreated`, `stageStepper`, `stepSaved`, `storageWarning`, `btnResume`, `btnMerge`, `mergeResult`, `mergeLog`.
- `app.js` loads `/api/runs/latest`, populates the form from `run.config`, renders the stepper, and updates from the WebSocket `run` block.

- [ ] **Step 1: Write the failing helper tests**

Create `tests/test_run_view.js` in the style of `tests/test_benchmark_view.js`:

```javascript
const assert = require("node:assert");
const {
  STAGE_ORDER, stageMap, runStatusLabel, savedStepsLabel, isMergeAvailable,
} = require("../server/static/run_view.js");

assert.deepStrictEqual(STAGE_ORDER, ["preflight", "training", "post_eval", "merge"]);
const run = {
  status: "completed",
  stages: [
    { name: "preflight", status: "completed" },
    { name: "training", status: "completed" },
    { name: "post_eval", status: "completed" },
    { name: "merge", status: "pending" },
  ],
};
assert.strictEqual(stageMap(run).get("post_eval").status, "completed");
assert.strictEqual(runStatusLabel(run), "Run completed");
assert.strictEqual(savedStepsLabel({ count: 1234 }), "1,234 steps saved");
assert.strictEqual(isMergeAvailable(run, true), true);
assert.strictEqual(isMergeAvailable({ ...run, status: "running" }, true), false);
assert.strictEqual(isMergeAvailable(run, false), false);
console.log("run_view behavior tests passed");
```

- [ ] **Step 2: Write the failing UI contract tests**

Extend `tests/test_web_ui_contract.py`:

```python
def test_stateful_run_controls_are_present():
    html = (ROOT / "server/static/index.html").read_text()
    js = (ROOT / "server/static/app.js").read_text()
    for element in ("runSelect", "runIdentity", "stageStepper", "stepSaved",
                    "storageWarning", "btnResume", "btnMerge", "mergeResult", "mergeLog"):
        assert f'id="{element}"' in html
    assert "/static/run_view.js" in html
    assert "/api/runs/latest" in js
    assert "stageStepper" in js
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q && node tests/test_run_view.js`
Expected: FAIL with missing elements/file.

- [ ] **Step 4: Implement `run_view.js`**

Pure helpers only, with the same CommonJS export pattern as `benchmark_view.js`. Normalize missing stages to `pending`; labels: `Preflight`, `Training`, `Post-eval`, `Merge`; overall labels: `starting`, `running`, `completed`, `stopped`, `failed`, `interrupted`.

- [ ] **Step 5: Add the run header markup**

In `index.html`, above the existing run section:

- A topbar `select id="runSelect"` with a "Latest" option.
- A run identity line with `runIdentity` and `runCreated`.
- A `stageStepper` containing four stage blocks with dots and labels.
- A `stepSaved` line.
- A hidden `storageWarning` banner.
- `btnResume` and `btnMerge` in the action row, hidden by default.
- `mergeResult`/`mergeLog` below the actions.
- Load `run_view.js` before `app.js`, bump the cache-busting version.

- [ ] **Step 6: Implement load/restore and rendering**

In `app.js`:

- Add element references.
- `async function loadRunState()`: fetch `/api/runs/latest`; if `run` is null or the request fails, hide the run-only UI and keep legacy behavior; otherwise render the run, populate form fields from `config`, load benchmarks/loss/log for that run.
- `renderRun(run, steps, storage)`: update run identity, created time, stage stepper via `stageMap`, saved-steps line, storage banner, Resume/Merge visibility, and the Dev simulation badge when `run.source === "fake"`.
- In the WebSocket handler, if `data.run` and the selected run is "latest", render it; if a historical run is selected, ignore live updates.
- `pollTelemetryFallback` also calls `loadRunState` when `data.run` is present, or use a separate 2-second poll for run state.
- Keep the existing telemetry/benchmark/canary/log/smoke rendering.

- [ ] **Step 7: Style the stepper and banner**

Add CSS for `.run-identity`, `.stage-stepper`, `.stage`, `.stage-dot` with status colors, `.stage-label`, `.stage-detail`, `.step-saved`, `.storage-warning`, and the hidden/visible states. Reuse the existing color tokens and typography.

- [ ] **Step 8: Run tests**

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q`
Run: `node tests/test_run_view.js && node tests/test_benchmark_view.js`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add server/static/run_view.js server/static/index.html server/static/app.js server/static/style.css tests/test_run_view.js tests/test_web_ui_contract.py
git commit -m "feat: restore and render persistent run state in the cockpit"
```

---

## Task 8: UI resume, merge, run history, and storage warning behavior

**Files:**
- Modify: `server/static/app.js`
- Modify: `server/static/index.html`
- Modify: `server/static/style.css`
- Test: `tests/test_web_ui_contract.py`
- Test: `tests/test_run_view.js`

**Interfaces:**
- `selectRun(runId)`, `resumeRun()`, `mergeRun()`, `refreshRunList()`, `renderMergeStatus(payload)` in `app.js`.
- Uses `POST /api/runs/{run_id}/resume`, `POST /api/runs/{run_id}/merge`, and `/api/runs?limit=20`.

- [ ] **Step 1: Write the failing action tests**

In `tests/test_run_view.test.js` add:

```javascript
assert.strictEqual(isMergeAvailable(
  { status: "completed", stages: [{ name: "training", status: "completed" }] },
  false,
), false, "merge requires an adapter artifact");
assert.strictEqual(runStatusLabel({ status: "interrupted" }), "Run interrupted");
```

In `tests/test_web_ui_contract.py`:

```python
def test_run_actions_are_wired():
    js = (ROOT / "server/static/app.js").read_text()
    assert "/resume" in js
    assert "/merge" in js
    assert "/api/runs?limit=" in js
    assert "storageWarning" in js
    assert "Dev simulation" in js
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q && node tests/test_run_view.js`
Expected: FAIL on missing action wiring.

- [ ] **Step 3: Implement run selection and actions**

- `refreshRunList()` fetches `/api/runs?limit=20`, fills `runSelect` with `Latest` plus each run id/status, and preserves the current selection.
- `selectRun("latest")` loads `/api/runs/latest`; any other id loads `/api/runs/{id}`. Historical selection sets `viewingHistorical = true`; live WebSocket updates are ignored until "Latest" is chosen.
- `resumeRun()` posts `/api/runs/{selectedRunId}/resume`, shows errors as an alert, and refreshes state.
- `mergeRun()` posts `/api/runs/{selectedRunId}/merge`; polls the selected run every 2 seconds while `merge` is running; shows the merge log tail on failure.
- `renderRun` sets `btnResume.hidden = !["interrupted","stopped","failed"].includes(run.status)` and `btnMerge.hidden = !isMergeAvailable(run, !!run.adapter_exists)`.
- The run payload from the server must include `adapter_exists` (computed from `<output_dir>/best_adapter/adapter_config.json`).
- The storage warning text: `Runs are not on persistent storage (`<runs_dir>`). Set ASR_RUNS_DIR to a /workspace path.`
- The Dev simulation badge appears when `run.source === "fake"`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_web_ui_contract.py -q`
Run: `node tests/test_run_view.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/static/app.js server/static/index.html server/static/style.css tests/test_web_ui_contract.py tests/test_run_view.js
git commit -m "feat: resume, merge, and browse runs from the cockpit"
```

---

## Task 9: Docs, GPU-free smoke verification, and full test pass

**Files:**
- Modify: `README.md`
- Modify: `GUIDE.md`
- Modify: `PRODUCTION.md`
- Modify: `.env.example`
- Test: full suite + node + fake-run smoke

- [ ] **Step 1: Document the run layout and resume/merge behavior**

Add to `README.md` and `GUIDE.md`:

- `/workspace/stt-outputs/results/runs/<run_id>/` layout.
- Start = preflight → training → post-eval; Merge is a later stage.
- Reload/restart restores the latest run; Resume continues from checkpoint; Merge writes `<run>/merged/`.
- `ASR_RUNS_DIR` override and the persistent-storage warning.
- CPU-only development with `ASR_FAKE_TRAINER=1`.

Add to `PRODUCTION.md` persistent artifacts:

```
results/runs/<run_id>/          # run.json, steps.jsonl, events.jsonl,
                                # telemetry, benchmarks, log, output/, merged/
results/runs/_shared/baselines/preflight.json
```

Add to `.env.example`:

```bash
# Optional: override the persistent run ledger root (default: ASR_RESULTS_DIR/runs)
# ASR_RUNS_DIR=/workspace/stt-outputs/results/runs

# Local CPU-only development: launch the fake trainer instead of train_lora.py
# ASR_FAKE_TRAINER=1
```

- [ ] **Step 2: Add the GPU-free smoke procedure**

Document and verify:

```bash
ASR_FAKE_TRAINER=1 .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
curl -s -X POST localhost:7860/api/train/start -H 'Content-Type: application/json' -d '{}'
curl -s localhost:7860/api/runs/latest
curl -s localhost:7860/api/status
```

The run must reach `completed`, `steps.count` must be > 0, and the run must still be listed after restarting the server.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (GPU-only tests may skip).

Run: `node tests/test_benchmark_view.js && node tests/test_run_view.js`
Expected: both print `... passed`.

- [ ] **Step 4: Run the fake-run verification manually**

Start the server with `ASR_FAKE_TRAINER=1`, start a run, then verify:

- page reload restores the run id, config, stepper, and saved steps;
- server restart while the fake run is active reattaches to the same run;
- Stop marks `stopped` and skips post-eval;
- Resume continues the same run;
- Merge writes `<run>/merged/config.json`;
- killing the fake trainer mid-run leaves `interrupted` with Resume available.

Record the commands and observed statuses in the task notes/commit message.

- [ ] **Step 5: Commit**

```bash
git add README.md GUIDE.md PRODUCTION.md .env.example
git commit -m "docs: document persistent runs and GPU-free development"
```

---

## Final Verification

- `.venv/bin/python -m pytest -q` passes.
- `node tests/test_benchmark_view.js` passes.
- `node tests/test_run_view.js` passes.
- A fake run completes, survives a server restart, resumes, stops, and merges.
- No test requires CUDA; `torch.cuda.is_available()` may be false.
- Existing preflight, smoke-test, benchmark, and final-report behavior is unchanged.
