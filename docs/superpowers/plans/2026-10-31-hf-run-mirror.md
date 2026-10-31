# Durable HF Run Mirror Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror every run's durable artifacts to a private Hugging Face repo at checkpoint and stage boundaries, restore a run by ID into the local run ledger, and expose recovery in the cockpit.

**Architecture:** A stdlib-only `src/hf_sync.py` owns config, remote discovery, restore, and a background uploader. The Trainer callback enqueues checkpoints and adapter snapshots; `train_lora.py` wires the callback and stage uploads; `scripts/hf_restore.py` restores by run ID; FastAPI endpoints and a cockpit dialog expose the same path. HF failures are logged and never fail training.

**Tech Stack:** Python 3.10+, `huggingface_hub>=0.23.0` (lazy import), FastAPI, vanilla JS, pytest, Node for pure JS tests.

**Spec:** `docs/superpowers/specs/2026-10-31-hf-run-mirror-design.md`

## Global Constraints

- Feature is off when neither `--hf_run_repo` nor `ASR_HF_RUN_REPO` is set; no HF import or network call is made in that case.
- Repos are private by default; `--no-hf_private` is an explicit opt-out.
- No per-optimizer-step weight uploads. Full checkpoints go at Trainer save boundaries (`save_steps=1000`); adapter-only snapshots every 250 steps; stage artifacts at stage boundaries.
- HF upload failures, auth failures, and missing `huggingface_hub` never fail training.
- Never write `HF_TOKEN` or any token into `run.json`, `hf_sync.json`, logs, or uploaded files.
- A checkpoint is complete only when it contains `trainer_state.json` and adapter weights (`adapter_model.safetensors` or `adapter_model.bin`).
- Remote ledgers are append-only: never upload a ledger with a lower `last_step` than the per-run `hf_state.json` high-water mark.
- `--fresh_start` always suppresses remote auto-repair and is mutually exclusive with `--hf_resume`.
- Restoring never silently starts a fresh run. No complete checkpoint means restore-only plus a non-zero exit.
- `huggingface_hub>=0.23.0`; `CommitOperationAdd` and `create_commit` are available.
- All tests run offline; no `HF_TOKEN` and no network. Use a fake API/downloader.
- Keep the existing local-first behavior and the manual `scripts/upload_to_hf.py` publish path.

## Review Focus

- A pod restored from an older volume snapshot must not overwrite a newer remote `steps.jsonl`/`events.jsonl`. Tested by the high-water tests in Task 3 and 6.
- A pod dying during checkpoint upload must leave the previous committed checkpoint usable, never a partial one. Tested by validation and fallback tests in Tasks 2 and 3.
- A fresh pod with a run ID but no repo/token must fail clearly before any download or training. Tested in Tasks 2, 5, and 6.
- A completed run must restore artifacts without retraining, whether selected by ID or `latest`. Tested in Tasks 2, 5, and 7.
- An HF outage during training must leave training running and surface an error badge. Tested in Tasks 3, 4, and 7.
- Restoring into an existing non-empty run directory must be refused, not overwritten. Tested in Tasks 2 and 6.

---

### Task 1: Config, Repo Layout, Validation, and Remote Discovery

**Files:**
- Create: `src/hf_sync.py`
- Test: `tests/test_hf_sync.py`

**Interfaces:**
- Produces:
  - `HFSyncConfig(repo_id: str, private: bool = True, adapter_snapshot_steps: int = 250, auto_resume: bool = True)`
  - `resolve_config(args=None, environ=None) -> HFSyncConfig | None`
  - `validate_repo_id(repo_id: str) -> None`
  - `repo_run_path(run_id: str) -> str`, `repo_checkpoint_path(run_id: str, step: int) -> str`
  - `is_complete_checkpoint(path: Path | str) -> bool`
  - `validate_adapter(path: Path | str) -> tuple[bool, str]`
  - `validate_merged(path: Path | str) -> tuple[bool, str]`
  - `select_checkpoint_step(files: Iterable[str], run_id: str) -> int | None`
  - `list_remote_runs(api, repo_id: str, *, download=_default_download, limit: int = 50) -> list[dict]`
  - `resolve_resume_run(api, repo_id: str, resume: str, *, download=_default_download) -> str`
  - Exceptions: `HFSyncError`, `HFRestoreError`, `HFRestoreConflict`, `HFNoCheckpointError`, `HFCompletedRunError`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_sync.py
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.hf_sync import (
    HFSyncConfig,
    HFCompletedRunError,
    is_complete_checkpoint,
    list_remote_runs,
    repo_checkpoint_path,
    repo_run_path,
    resolve_config,
    resolve_resume_run,
    select_checkpoint_step,
    validate_adapter,
    validate_merged,
    validate_repo_id,
)

RUN = "20261031-120000-abcdef"
OTHER = "20261030-120000-aaaaaa"


def test_config_is_off_without_a_repo(monkeypatch):
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    assert resolve_config(SimpleNamespace(), environ={}) is None


def test_config_prefers_cli_over_env():
    args = SimpleNamespace(
        hf_run_repo="me/run",
        hf_private=False,
        hf_adapter_snapshot_steps=0,
        hf_auto_resume=False,
    )
    cfg = resolve_config(args, environ={"ASR_HF_RUN_REPO": "me/env"})
    assert cfg == HFSyncConfig(
        repo_id="me/run",
        private=False,
        adapter_snapshot_steps=0,
        auto_resume=False,
    )


def test_config_reads_env_and_rejects_bad_repo():
    cfg = resolve_config(SimpleNamespace(), environ={"ASR_HF_RUN_REPO": "me/run"})
    assert cfg.repo_id == "me/run"
    with pytest.raises(ValueError):
        validate_repo_id("not-a-repo")
    with pytest.raises(ValueError):
        validate_repo_id("a/b/c")


def test_repo_paths_are_run_scoped():
    assert repo_run_path(RUN) == f"runs/{RUN}"
    assert repo_checkpoint_path(RUN, 1000) == f"runs/{RUN}/checkpoint-1000"


def test_checkpoint_requires_state_and_adapter(tmp_path):
    ckpt = tmp_path / "checkpoint-1000"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}")
    assert is_complete_checkpoint(ckpt) is False
    (ckpt / "adapter_model.safetensors").write_text("")
    assert is_complete_checkpoint(ckpt) is True


def test_adapter_and_merged_validation(tmp_path):
    adapter = tmp_path / "best_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_text("")
    assert validate_adapter(adapter)[0] is True

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_text("")
    assert validate_merged(merged)[0] is True
    assert validate_merged(tmp_path / "missing")[0] is False


def test_select_checkpoint_step_ignores_incomplete_and_bad_names():
    files = [
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
        f"runs/{RUN}/checkpoint-2000/trainer_state.json",
        f"runs/{RUN}/checkpoint-nope/trainer_state.json",
    ]
    assert select_checkpoint_step(files, RUN) == 1000


def test_list_remote_runs_reads_manifest_and_state():
    blobs = {
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "running", "updated_at": 20.0}
        ).encode(),
        f"runs/{RUN}/hf_state.json": json.dumps({"last_step": 2000}).encode(),
        f"runs/{OTHER}/run.json": json.dumps(
            {"run_id": OTHER, "status": "completed", "updated_at": 10.0}
        ).encode(),
    }
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/hf_state.json",
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
        f"runs/{OTHER}/run.json",
    ]

    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return list(files)

    runs = list_remote_runs(
        FakeApi(), "me/run", download=lambda repo, path: blobs[path]
    )
    assert [r["run_id"] for r in runs] == [RUN, OTHER]
    assert runs[0]["last_step"] == 2000
    assert runs[0]["has_checkpoint"] is True
    assert runs[1]["has_checkpoint"] is False


def test_resolve_latest_rejects_completed_run():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [
                "latest.json",
                f"runs/{RUN}/run.json",
                f"runs/{RUN}/checkpoint-1000/trainer_state.json",
                f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
            ]

    blobs = {
        "latest.json": json.dumps({"run_id": RUN}).encode(),
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "completed", "updated_at": 20.0}
        ).encode(),
    }
    with pytest.raises(HFCompletedRunError):
        resolve_resume_run(
            FakeApi(), "me/run", "latest", download=lambda repo, path: blobs[path]
        )


def test_resolve_explicit_run_works_even_when_completed():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [f"runs/{RUN}/run.json"]

    blobs = {
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "completed", "updated_at": 20.0}
        ).encode()
    }
    assert resolve_resume_run(
        FakeApi(), "me/run", RUN, download=lambda repo, path: blobs[path]
    ) == RUN
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hf_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.hf_sync'`.

- [ ] **Step 3: Implement the module**

Create `src/hf_sync.py` with only stdlib imports at module scope. Import `huggingface_hub` inside `_default_download` and uploader methods. Define `RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")`, `CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")`, the dataclass, validators, path helpers, and the exceptions above. `resolve_config` reads `args.hf_run_repo` then `ASR_HF_RUN_REPO`; clamps `adapter_snapshot_steps` to `max(0, int(...))`; returns `None` when no repo. `_default_download(repo_id, remote_path)` lazily imports `hf_hub_download`, downloads to a temp file, and returns its bytes. `list_remote_runs` finds every `runs/<id>/run.json`, reads `hf_state.json` when present, computes `last_step` and `has_checkpoint`, and sorts by `updated_at` then `run_id`, descending. `resolve_resume_run` treats `latest` as the pointer target (or newest run ID when `latest.json` is absent) and raises `HFCompletedRunError` only for `latest`; an explicit ID is returned even when completed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hf_sync.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hf_sync.py tests/test_hf_sync.py
git commit -m "feat: add HF sync config and remote discovery"
```

---

### Task 2: Restore Engine, Path Rewrite, and Restore CLI

**Files:**
- Modify: `src/hf_sync.py`
- Create: `scripts/hf_restore.py`
- Test: `tests/test_hf_restore.py`

**Interfaces:**
- Consumes: Task 1 exceptions, path helpers, `select_checkpoint_step`, `is_complete_checkpoint`.
- Produces:
  - `make_api() -> Any` (lazy `HfApi`; clear error when `huggingface_hub` is missing)
  - `restore_run(api, repo_id, run_id, runs_root, *, include_merged=False, force=False, download=_default_download, status=lambda state, message: None) -> dict`
  - Result keys: `run_id`, `run_dir`, `output_dir`, `checkpoint_step` (int or `None`), `downloaded` (list of repo paths), `run` (restored manifest dict).
  - `rewrite_run_paths(run: dict, run_dir: Path) -> dict`
  - `normalize_restored_run(run: dict) -> dict`
  - `apply_restored_args(args, run: dict, *, run_dir: Path, explicit: set[str] | None = None) -> args`
  - `scripts/hf_restore.py` exit codes: `0` restored with a checkpoint, `2` restored without a checkpoint, `1` failure.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_restore.py
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.hf_sync import (
    HFRestoreConflict,
    apply_restored_args,
    normalize_restored_run,
    restore_run,
)

RUN = "20261031-120000-abcdef"
OUTPUT = "runs/{}/checkpoint-{}"


def _api(files):
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return list(files)

    return FakeApi()


def _blobs(checkpoint_step=1000, status="running"):
    b = {
        f"runs/{RUN}/run.json": json.dumps(
            {
                "run_id": RUN,
                "status": status,
                "current_pid": 999999,
                "paths": {
                    "run_dir": "/workspace/old/run",
                    "output_dir": "/workspace/old/run/output",
                    "results_dir": "/workspace/old/run",
                    "merged_dir": "/workspace/old/run/merged",
                    "preflight_cache": "/workspace/old/run/baselines/preflight.json",
                },
                "trainer_args": {
                    "run_dir": "/workspace/old/run",
                    "output_dir": "/workspace/old/run/output",
                    "results_dir": "/workspace/old/run",
                    "preflight_cache_dir": "/workspace/old/run",
                    "epochs": 4,
                    "batch_size": 8,
                    "resume_from_checkpoint": None,
                    "fresh_start": False,
                },
                "stages": [
                    {"name": "preflight", "status": "completed"},
                    {"name": "training", "status": "running"},
                    {"name": "post_eval", "status": "pending"},
                ],
                "attempts": [{"pid": 999999, "ended_at": None}],
            }
        ).encode(),
        f"runs/{RUN}/steps.jsonl": b'{"step": 1000}\n',
        f"runs/{RUN}/events.jsonl": b'{"type": "stage"}\n',
        f"runs/{RUN}/best_adapter/adapter_config.json": b"{}",
        f"runs/{RUN}/best_adapter/adapter_model.safetensors": b"",
        f"runs/{RUN}/hf_state.json": json.dumps(
            {"last_step": checkpoint_step}
        ).encode(),
    }
    if checkpoint_step is not None:
        b[f"runs/{RUN}/checkpoint-{checkpoint_step}/trainer_state.json"] = b"{}"
        b[f"runs/{RUN}/checkpoint-{checkpoint_step}/adapter_model.safetensors"] = b""
    return b


def _files(checkpoint_step=1000):
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/steps.jsonl",
        f"runs/{RUN}/events.jsonl",
        f"runs/{RUN}/best_adapter/adapter_config.json",
        f"runs/{RUN}/best_adapter/adapter_model.safetensors",
        f"runs/{RUN}/hf_state.json",
    ]
    if checkpoint_step is not None:
        files += [
            f"runs/{RUN}/checkpoint-{checkpoint_step}/trainer_state.json",
            f"runs/{RUN}/checkpoint-{checkpoint_step}/adapter_model.safetensors",
        ]
    return files


def _download(blobs):
    return lambda repo, path: blobs[path]


def test_restore_downloads_ledger_and_newest_checkpoint(tmp_path):
    blobs = _blobs(1000)
    result = restore_run(
        _api(_files(1000)),
        "me/run",
        RUN,
        tmp_path,
        download=_download(blobs),
    )
    run_dir = tmp_path / RUN
    assert result["checkpoint_step"] == 1000
    assert (run_dir / "steps.jsonl").read_text() == '{"step": 1000}\n'
    assert (
        run_dir / "output" / "checkpoint-1000" / "trainer_state.json"
    ).is_file()
    assert (run_dir / "output" / "best_adapter" / "adapter_config.json").is_file()

    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "interrupted"
    assert run["current_pid"] is None
    assert run["paths"]["run_dir"] == str(run_dir)
    assert run["paths"]["output_dir"] == str(run_dir / "output")
    assert run["trainer_args"]["epochs"] == 4
    assert run["trainer_args"]["output_dir"] == str(run_dir / "output")


def test_restore_skips_incomplete_newest_checkpoint(tmp_path):
    blobs = _blobs(2000)
    files = _files(2000) + [
        f"runs/{RUN}/checkpoint-3000/trainer_state.json",
        f"runs/{RUN}/checkpoint-3000/optimizer.pt",
    ]
    result = restore_run(
        _api(files), "me/run", RUN, tmp_path, download=_download(blobs)
    )
    assert result["checkpoint_step"] == 2000


def test_restore_refuses_nonempty_local_run(tmp_path):
    (tmp_path / RUN).mkdir()
    (tmp_path / RUN / "run.json").write_text("{}")
    with pytest.raises(HFRestoreConflict):
        restore_run(
            _api(_files(1000)),
            "me/run",
            RUN,
            tmp_path,
            download=_download(_blobs(1000)),
        )


def test_restore_without_checkpoint_returns_partial_state(tmp_path):
    result = restore_run(
        _api(_files(None)),
        "me/run",
        RUN,
        tmp_path,
        download=_download(_blobs(None)),
    )
    assert result["checkpoint_step"] is None
    assert (tmp_path / RUN / "run.json").is_file()
    assert (tmp_path / RUN / "steps.jsonl").is_file()


def test_rewrite_and_normalize_keep_metrics_and_clear_dead_state(tmp_path):
    run = json.loads(
        _blobs(1000)[f"runs/{RUN}/run.json"].decode()
    )
    run["final_full_test_metrics"] = {"normalized_wer": 12.5}
    run["stages"][1]["status"] = "running"
    run["merge_pid"] = 123
    run["attempts"] = [{"pid": 999999, "ended_at": None}]

    normalize_restored_run(run)
    assert run["status"] == "interrupted"
    assert run["current_pid"] is None
    assert run["merge_pid"] is None
    assert run["attempts"][0]["ended_at"] is not None
    assert run["stages"][1]["status"] == "interrupted"
    assert run["final_full_test_metrics"]["normalized_wer"] == 12.5


def test_apply_restored_args_saved_values_and_explicit_overrides():
    run = {
        "trainer_args": {
            "epochs": 4,
            "batch_size": 16,
            "output_dir": "/old/output",
            "resume_from_checkpoint": "checkpoint-1",
            "fresh_start": True,
        },
        "stages": [
            {"name": "training", "status": "completed"},
            {"name": "post_eval", "status": "failed"},
        ],
    }
    args = SimpleNamespace(
        epochs=2,
        batch_size=8,
        output_dir="/cli/output",
        results_dir=None,
        run_dir=None,
        preflight_cache_dir=None,
        resume_from_checkpoint=None,
        fresh_start=True,
        auto_resume=False,
        post_eval_only=False,
        hf_resume="latest",
        hf_run_repo="me/run",
    )
    applied = apply_restored_args(
        args, run, explicit={"batch_size"}, run_dir=Path("/new/run")
    )
    assert applied.epochs == 4
    assert applied.batch_size == 8  # explicit beat saved
    assert applied.output_dir == "/new/run/output"
    assert applied.results_dir == "/new/run"
    assert applied.run_dir == "/new/run"
    assert applied.fresh_start is False
    assert applied.post_eval_only is True
    assert applied.resume_from_checkpoint is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hf_restore.py -v`
Expected: FAIL because `restore_run`, `normalize_restored_run`, and `apply_restored_args` do not exist.

- [ ] **Step 3: Implement the restore engine**

`restore_run` must:
- validate `run_id` and require `runs/<run_id>/run.json` in the remote file list;
- refuse an existing non-empty `runs_root/<run_id>` unless `force`, then `shutil.rmtree` it;
- call `status("listing", ...)`, `status("ledger", ...)`, `status("checkpoint", ...)`, `status("verify", ...)`;
- download `run.json`, `steps.jsonl`, `events.jsonl`, `live_telemetry.json`, `loss_history.json`, `epoch_benchmarks.json`, `baselines/preflight.json`, `hf_state.json` when present;
- download `best_adapter/*` into `<run_dir>/output/best_adapter/`;
- download `reports/*` into `<run_dir>/output/` and `<run_dir>/`;
- select the highest complete checkpoint with `select_checkpoint_step`, download its files into `<run_dir>/output/checkpoint-<step>/`, verify `is_complete_checkpoint`, and fall back to the next lower complete checkpoint if verification fails;
- download the newest `adapter_snapshots/step-*` into `<run_dir>/output/adapter_snapshots/step-*/` when present;
- optionally download `merged/*` into `<run_dir>/merged/`;
- rewrite paths with `rewrite_run_paths`, normalize with `normalize_restored_run`, remove any local `training.pid`, write `run.json` atomically, and return the result dict.

`rewrite_run_paths` sets `paths.run_dir`, `paths.output_dir`, `paths.results_dir`, `paths.merged_dir`, `paths.preflight_cache`; it also rewrites `trainer_args.run_dir`, `trainer_args.output_dir`, `trainer_args.results_dir`, `trainer_args.preflight_cache_dir`, and removes `trainer_args.merged_dir` if present. `normalize_restored_run` sets `starting`/`running` to `interrupted`, clears `current_pid`/`merge_pid`/`training.pid` references, ends open attempts, marks `running` stages `interrupted`, and preserves terminal statuses, metrics, stages, and config. `apply_restored_args` takes `run_dir` as a keyword only argument (the test above passes it); it applies saved `trainer_args` except for `RESET_KEYS = {"run_dir", "output_dir", "results_dir", "preflight_cache_dir", "resume_from_checkpoint", "fresh_start", "auto_resume", "hf_resume", "hf_run_repo", "post_eval_only"}` and names in `explicit`; then sets the new local paths, `fresh_start=False`, `auto_resume=True`, `hf_resume="none"`, and derives `post_eval_only` from `training == completed` and `post_eval not in (completed, skipped)`.

Add `make_api()` to `src/hf_sync.py`: lazily import `HfApi` and return an instance; raise `HFSyncError` with the install/auth remediation when the import fails.

Then create `scripts/hf_restore.py`:
- `--repo` default `ASR_HF_RUN_REPO`, `--run` required, `--runs-root` default `src.paths.resolve_runs_dir()`, `--include-merged`, `--force`, `--status-file`.
- loads `.env` via `src.dotenv`; uses `make_api()`; fails with a clear message when `huggingface_hub` is missing.
- calls `resolve_resume_run` and `restore_run`; writes a status JSON at each milestone with `state`, `run_id`, `message`, `updated_at`, `pid`, `checkpoint_step`.
- prints the restored run directory and `python scripts/train_lora.py --hf_resume <run_id>`; exits `2` when `checkpoint_step` is `None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hf_sync.py tests/test_hf_restore.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hf_sync.py scripts/hf_restore.py tests/test_hf_restore.py
git commit -m "feat: restore HF runs by id"
```

---

### Task 3: Background Uploader and Per-Run High-Water Mark

**Files:**
- Modify: `src/hf_sync.py`
- Test: `tests/test_hf_mirror.py`

**Interfaces:**
- Consumes: Task 1 `HFSyncConfig`, `is_complete_checkpoint`, `repo_*_path`.
- Produces:
  - `SyncJob(kind: str, path: Path | None = None, step: int | None = None, label: str = "")`
  - `HFRunMirror(config, *, run_id, run_dir, output_dir, results_dir, api=None, manifest_update=None, download=_default_download, log=print, start_worker=True, max_attempts=3, sleep=time.sleep)`
  - `start()`, `auto_repair() -> bool`, `enqueue_checkpoint(path, step)`, `enqueue_adapter_snapshot(path, step)`, `enqueue_adapter(path, label="best_adapter")`, `enqueue_ledger(status="running", *, reports=False)`, `enqueue_merged(path)`, `enqueue_latest()`, `process_pending() -> int`, `drain(timeout=120.0) -> bool`, `close()`, `status() -> dict`
  - `hf_sync.json` schema from the spec.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_mirror.py
import json
from pathlib import Path

from src.hf_sync import HFSyncConfig, HFRunMirror

RUN = "20261031-120000-abcdef"


class FakeApi:
    def __init__(self, files=None):
        self.files = list(files or [])
        self.created = []
        self.folders = []
        self.commits = []
        self.files_uploaded = []
        self.fail_folder = 0

    def create_repo(self, repo_id, private=True, exist_ok=True):
        self.created.append((repo_id, private, exist_ok))

    def list_repo_files(self, repo_id, revision="main"):
        return list(self.files)

    def upload_folder(self, **kwargs):
        if self.fail_folder:
            self.fail_folder -= 1
            raise RuntimeError("transient")
        self.folders.append(kwargs)

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)

    def upload_file(self, **kwargs):
        self.files_uploaded.append(kwargs)


def _mirror(tmp_path, api, **kwargs):
    run_dir = tmp_path / RUN
    output = run_dir / "output"
    output.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": RUN,
                "status": "running",
                "last_step": 1000,
                "paths": {"output_dir": str(output)},
            }
        )
    )
    (run_dir / "steps.jsonl").write_text('{"step": 1000}\n')
    (run_dir / "events.jsonl").write_text('{"type": "stage"}\n')
    kwargs.setdefault("download", lambda repo, path: b"{}")
    return HFRunMirror(
        HFSyncConfig(repo_id="me/run"),
        run_id=RUN,
        run_dir=run_dir,
        output_dir=output,
        results_dir=run_dir,
        api=api,
        start_worker=False,
        sleep=lambda _: None,
        **kwargs,
    )


def _checkpoint(output, step):
    path = output / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}")
    (path / "adapter_model.safetensors").write_text("")
    return path


def test_checkpoint_upload_uses_run_scoped_path(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1
    assert api.folders[0]["path_in_repo"] == f"runs/{RUN}/checkpoint-1000"


def test_incomplete_checkpoint_is_not_uploaded(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    path = mirror.output_dir / "checkpoint-1000"
    path.mkdir()
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 0
    assert api.folders == []


def test_ledger_upload_contains_state_and_reports(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 1
    paths = [op.path_in_repo for op in api.commits[0]["operations"]]
    assert f"runs/{RUN}/run.json" in paths
    assert f"runs/{RUN}/steps.jsonl" in paths
    assert f"runs/{RUN}/hf_state.json" in paths


def test_high_water_blocks_older_ledger(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.download = lambda repo, path: json.dumps({"last_step": 2000}).encode()
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 0
    assert api.commits == []
    assert "2000" in mirror.status()["last_error"]


def test_transient_failure_retries_and_records_success(tmp_path):
    api = FakeApi()
    api.fail_folder = 2
    mirror = _mirror(tmp_path, api)
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1
    assert len(api.folders) == 1
    assert mirror.status()["last_success_step"] == 1000


def test_adapter_snapshots_coalesce_to_newest(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    for step in (250, 500, 750):
        path = mirror.output_dir / "adapter_snapshots" / f"step-{step:06d}"
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_text("")
        mirror.enqueue_adapter_snapshot(path, step)
    assert mirror.process_pending() == 1
    assert api.folders[0]["path_in_repo"].endswith("step-000750")


def test_failure_writes_error_status_without_raising(tmp_path):
    api = FakeApi()
    api.fail_folder = 3
    mirror = _mirror(tmp_path, api)
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 0
    status = mirror.status()
    assert status["status"] == "error"
    assert "transient" in status["last_error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hf_mirror.py -v`
Expected: FAIL because `HFRunMirror` is missing.

- [ ] **Step 3: Implement the uploader**

Implement `HFRunMirror` with a `threading.Condition`, a `collections.deque`, and a daemon worker. `start()` calls `ensure_repo` (via `api.create_repo`), creates a static `README.md` at the repo root when it is missing (layout plus the restore command), writes/updates `run.json` through `manifest_update` with `{"hf": {...}}`, loads `hf_sync_pending.json` and re-enqueues checkpoint/adapter jobs whose paths still exist, backfills local complete checkpoints newer than the remote high-water mark plus `best_adapter/`, then starts the worker when `start_worker` is true. The remote high-water mark is `runs/<id>/hf_state.json.last_step` when present, otherwise the highest complete remote `checkpoint-<step>` found in `list_repo_files`. After a successful adapter-snapshot upload, keep the newest two local `adapter_snapshots/step-*` directories and delete older ones; never delete a snapshot whose upload has not succeeded. Implement `auto_repair()`: compare the highest complete local checkpoint with the highest complete remote checkpoint; when the remote is newer, download it into `output_dir` using the same temp-directory-then-rename rule as `restore_run`, and return `True`. Return `False` when local is current or remote has nothing newer. `enqueue_adapter_snapshot` removes any pending adapter job before appending, so only the newest snapshot is uploaded. `process_pending()` pops and uploads jobs until the deque is empty; it returns the number of jobs successfully handled. `_upload_job` validates checkpoint/adapter/merged paths, uses `upload_folder` for directories, `create_commit` with `CommitOperationAdd` for ledger files, and `upload_file` for `latest.json`. The ledger snapshot includes `run.json`, `steps.jsonl`, `events.jsonl`, telemetry/loss/benchmark files when present, `baselines/preflight.json`, `training_live.log` when under 5 MB, `hf_state.json`, and reports when `reports=True`. Before a ledger commit, read remote `runs/<id>/hf_state.json` via `download`; if its `last_step` is higher, log an error, set `hf_sync.json` to `error`, and drop the job. On success update `hf_sync.json` (atomic), `last_step`, `latest.json`, and the pending file. On failure retry `max_attempts` times with `sleep(2 ** attempt)`; after the last failure record `status="error"` and `last_error`, append an `{"type": "hf_sync", "status": "error", ...}` line to `run_dir/events.jsonl`, but never raise. `drain(timeout)` waits on the condition until the deque is empty and no job is active; `close()` drains then stops the worker.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hf_mirror.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hf_sync.py tests/test_hf_mirror.py
git commit -m "feat: add background HF artifact uploader"
```

---

### Task 4: Trainer Callback for Checkpoints and Adapter Snapshots

**Files:**
- Modify: `src/trainer_callbacks.py`
- Test: `tests/test_trainer_hf_callback.py`

**Interfaces:**
- Consumes: `HFRunMirror` from Task 3.
- Produces: `HFMirrorCallback(mirror, *, model, processor, output_dir, snapshot_steps=250, save_steps=1000)` with `on_train_begin`, `on_step_end`, `on_save`, `on_train_end`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trainer_hf_callback.py
from pathlib import Path
from types import SimpleNamespace

from src.trainer_callbacks import HFMirrorCallback


class FakeMirror:
    def __init__(self):
        self.jobs = []

    def enqueue_checkpoint(self, path, step):
        self.jobs.append(("checkpoint", step, Path(path)))

    def enqueue_adapter_snapshot(self, path, step):
        self.jobs.append(("adapter", step, Path(path)))

    def enqueue_ledger(self, status="running"):
        self.jobs.append(("ledger", status, None))


class FakeModel:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        (Path(path) / "adapter_model.safetensors").write_text("")


class FakeProcessor:
    def save_pretrained(self, path):
        (Path(path) / "preprocessor_config.json").write_text("{}")


def test_snapshot_every_250_steps_and_skip_checkpoint_steps(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    cb.on_step_end(None, SimpleNamespace(global_step=250), None)
    assert mirror.jobs == [
        ("adapter", 250, tmp_path / "adapter_snapshots" / "step-000250")
    ]
    assert (
        tmp_path / "adapter_snapshots" / "step-000250" / "adapter_config.json"
    ).is_file()

    cb.on_step_end(None, SimpleNamespace(global_step=1000), None)
    assert [job[0] for job in mirror.jobs] == ["adapter"]


def test_on_save_enqueues_checkpoint(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    ckpt = tmp_path / "checkpoint-1000"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}")
    cb.on_save(None, SimpleNamespace(global_step=1000), None)
    assert mirror.jobs == [("checkpoint", 1000, ckpt)]


def test_train_end_enqueues_ledger(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
    )
    cb.on_train_end(None, SimpleNamespace(global_step=500), None)
    assert mirror.jobs == [("ledger", "training", None)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trainer_hf_callback.py -v`
Expected: FAIL because `HFMirrorCallback` is missing.

- [ ] **Step 3: Implement the callback**

Add `HFMirrorCallback` to `src/trainer_callbacks.py` (it already imports `TrainerCallback`, `shutil`, `os`, `time`). In `on_step_end`, return unless `snapshot_steps > 0`, `step > 0`, `step % snapshot_steps == 0`, and (`save_steps <= 0` or `step % save_steps != 0`); save to `<output_dir>/adapter_snapshots/.staging-step-<n>` (remove any previous staging directory), call `model.save_pretrained` and `processor.save_pretrained`, then `os.replace` to `step-<n>` and enqueue. Wrap the save in `try/except Exception` and log a warning; a snapshot failure must not stop training. In `on_save`, build `<output_dir>/checkpoint-<state.global_step>` and enqueue it only if it exists. In `on_train_begin` and `on_train_end`, enqueue a ledger with status `"training"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trainer_hf_callback.py tests/test_trainer_callbacks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainer_callbacks.py tests/test_trainer_hf_callback.py
git commit -m "feat: enqueue HF checkpoints and adapter snapshots from trainer"
```

---

### Task 5: `train_lora.py` Flags, Run Auto-Creation, Restore Mode, and Stage Uploads

**Files:**
- Modify: `scripts/train_lora.py`
- Test: `tests/test_train_lora_hf.py`

**Interfaces:**
- Consumes: `HFSyncConfig`, `resolve_config`, `resolve_resume_run`, `restore_run`, `apply_restored_args`, `HFRunMirror`, `HFMirrorCallback`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser` (the existing parser construction extracted; `parse_args()` calls it)
  - `_explicit_cli_keys(parser, argv) -> set[str]`
  - `_maybe_create_hf_run_dir(args) -> Path | None`
  - `_build_hf_mirror(args, run_dir, run=None) -> HFRunMirror | None`
  - New flags: `--hf_run_repo`, `--hf_private/--no-hf_private`, `--hf_adapter_snapshot_steps`, `--hf_auto_resume/--no-hf_auto_resume`, `--hf_resume {none,latest,<run_id>}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_lora_hf.py
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))

from src import run_state

RUN = "20261031-120000-abcdef"


def _train_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_lora
    return train_lora


def test_parser_accepts_hf_flags(monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lora.py",
            "--hf_run_repo", "me/run",
            "--no-hf_private",
            "--hf_adapter_snapshot_steps", "0",
            "--hf_resume", RUN,
        ],
    )
    args = train_lora.parse_args()
    assert args.hf_run_repo == "me/run"
    assert args.hf_private is False
    assert args.hf_adapter_snapshot_steps == 0
    assert args.hf_resume == RUN


def test_fresh_start_and_hf_resume_are_mutually_exclusive(monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_lora.py", "--fresh_start", "--hf_resume", "latest"],
    )
    with pytest.raises(SystemExit):
        train_lora.parse_args()


def test_mirror_auto_creates_canonical_run_dir(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    args = SimpleNamespace(
        hf_run_repo="me/run",
        run_dir=None,
        post_eval_only=False,
        output_dir=str(tmp_path / "output"),
        results_dir=None,
    )
    run_dir = train_lora._maybe_create_hf_run_dir(args)
    assert run_dir is not None
    assert run_dir.parent == (tmp_path / "runs").resolve()
    assert run_state.run_id_valid(run_dir.name)


def test_mirror_does_not_create_run_without_repo(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    args = SimpleNamespace(
        hf_run_repo=None, run_dir=None, post_eval_only=False,
        output_dir=str(tmp_path / "output"), results_dir=None,
    )
    assert train_lora._maybe_create_hf_run_dir(args) is None


def test_hf_resume_restores_and_applies_saved_args(tmp_path, monkeypatch):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    run = run_state.create_run(
        runs_root,
        run_id=RUN,
        config={},
        trainer_args={"epochs": 4, "batch_size": 8},
    )
    run_dir = runs_root / RUN
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "failed")

    args = SimpleNamespace(
        hf_resume=RUN,
        hf_run_repo="me/run",
        run_dir=None,
        output_dir=str(tmp_path / "default-output"),
        results_dir=None,
        post_eval_only=False,
        epochs=2,
        batch_size=8,
        fresh_start=False,
        auto_resume=False,
        resume_from_checkpoint=None,
        preflight_cache_dir=None,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "resolve_resume_run",
        lambda *a, **k: RUN,
    )
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: {"run_id": RUN, "checkpoint_step": 1000},
    )
    observed = {}

    def fake_run(parsed, current_run_dir):
        observed["run_dir"] = str(current_run_dir)
        observed["post_eval_only"] = parsed.post_eval_only
        observed["output_dir"] = parsed.output_dir

    monkeypatch.setattr(train_lora, "_run_training", fake_run)
    train_lora.main()

    assert observed["run_dir"] == str(run_dir.resolve())
    assert observed["post_eval_only"] is True
    assert observed["output_dir"] == str(run_dir / "output")


def test_hf_resume_completed_run_does_not_train(tmp_path, monkeypatch):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    run_state.create_run(
        runs_root, run_id=RUN, config={}, trainer_args={"epochs": 4}
    )
    run_dir = runs_root / RUN
    run_state.set_status(run_dir, "completed")
    args = SimpleNamespace(
        hf_resume=RUN,
        hf_run_repo="me/run",
        run_dir=None,
        output_dir=str(tmp_path / "default-output"),
        results_dir=None,
        post_eval_only=False,
        epochs=2,
        batch_size=8,
        fresh_start=False,
        auto_resume=False,
        resume_from_checkpoint=None,
        preflight_cache_dir=None,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    monkeypatch.setattr(train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: {"run_id": RUN, "checkpoint_step": 1000},
    )
    called = []
    monkeypatch.setattr(
        train_lora, "_run_training", lambda parsed, current: called.append(current)
    )
    train_lora.main()
    assert called == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_train_lora_hf.py -v`
Expected: FAIL because the flags and helpers do not exist.

- [ ] **Step 3: Implement the integration**

- Extract the current parser construction into `build_parser()` and have `parse_args()` call it; keep every existing default and validation identical. Add the HF flags with defaults `hf_run_repo=None`, `hf_private=True`, `hf_adapter_snapshot_steps=250`, `hf_auto_resume=True`, `hf_resume="none"`. After parsing, reject `fresh_start=True` with `hf_resume != "none"` via `parser.error`.
- Add `_explicit_cli_keys(parser, argv)` scanning option strings, mapping `--no-foo` back to `foo`, and returning the set of `action.dest` values. Set `args._explicit_keys` in `parse_args()`.
- Add `_maybe_create_hf_run_dir(args)`: return `None` when no `hf_run_repo`/`ASR_HF_RUN_REPO`, `post_eval_only`, or an existing `run_dir`; otherwise create `<resolve_runs_dir()>/<run_state.new_run_id()>` and set `args.run_dir` to it. Do this in `main()` before the existing `run_dir` handling, so run-scoped output/results defaults apply.
- Add restore mode at the top of `main()`: when `hf_resume != "none"`, build `HFSyncConfig`, call `hf_sync.make_api()` and `resolve_resume_run`, set `args.run_dir`, call `restore_run` (skip if the local run already has a complete checkpoint), load the restored `run.json`, and call `apply_restored_args(args, run, explicit=getattr(args, "_explicit_keys", set()), run_dir=run_dir)`. Do not call `_record_trainer_args` in restore mode. If the restored run is completed, print the restored path and return 0 before `_ensure_cli_run`, status changes, or training; if `checkpoint_step is None`, print a clear error and raise `SystemExit(2)`.
- Add `_build_hf_mirror(args, run_dir, run=None)`: return `None` without a repo; otherwise create `HFRunMirror` with `manifest_update=lambda mutate: run_state.update_run(run_dir, mutate)`, `output_dir=Path(args.output_dir)`, `results_dir=Path(args.results_dir)`, and call `start()`. When resuming an existing run with `auto_resume` on and `fresh_start` off, call `mirror.auto_repair()` before `_resolve_resume_checkpoint` so a missing or stale local checkpoint is pulled from HF. The repo ID is `run.get("hf", {}).get("repo_id")` when present, otherwise `args.hf_run_repo`/`ASR_HF_RUN_REPO`; a differing env/CLI repo on an existing run is logged and ignored. Catch exceptions, log, and return `None` so training continues.
- In restore mode, `--config` is ignored with a warning; the saved `trainer_args` already contain the YAML-resolved values, and explicit individual CLI flags are the override mechanism.
- In `_run_training`, build the mirror before the `post_eval_only` early return (so post-eval-only resumes can upload their report), and wrap the body in `try/finally`. In `finally`, enqueue a final ledger using the run's terminal status when `run.json` exists, then call `mirror.close()`. Append `HFMirrorCallback(mirror, model=model, processor=processor, output_dir=args.output_dir, snapshot_steps=args.hf_adapter_snapshot_steps, save_steps=args.save_steps)` after the telemetry callback. After the preflight stage completes, enqueue a ledger with `"preflight"`. After `best_adapter` is saved, enqueue it. After `_run_post_eval_stage` returns true, or after `_run_post_eval_only` finishes, enqueue a ledger with `"post_eval"` and `reports=True`.
- Add `hf_run_repo`, `hf_private`, `hf_adapter_snapshot_steps`, `hf_auto_resume` to `_cli_run_config` so the cockpit can show them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_train_lora_hf.py tests/test_train_lora_run_stages.py tests/test_script_defaults.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_lora.py tests/test_train_lora_hf.py
git commit -m "feat: mirror training checkpoints to HF"
```

---

### Task 6: Server Restore Endpoints and Run Sync Payload

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_hf_restore_server.py`

**Interfaces:**
- Consumes: `list_remote_runs`, `resolve_resume_run`, `restore_run`, `HFSyncConfig`.
- Produces:
  - `GET /api/hf/config` -> `{"enabled": bool, "repo_id": str, "runs_root": str}`
  - `GET /api/hf/runs?repo=<id>` -> `{"runs": [...], "repo_id": str}`
  - `POST /api/hf/restore` body `{"run_id": str, "repo": str | None, "include_merged": bool}` -> `{"state": "running", "run_id": str}`
  - `GET /api/hf/restore/status` -> `{"state", "run_id", "message", "updated_at", "log_tail"}`
  - `_hf_restore_status()`, `_hf_restore_running()`, `_start_hf_restore(run_id, repo, include_merged=False)`.
  - `_run_payload` gains `"hf": run.get("hf") or {}` and `"hf_sync": read_json_safely(_run_dir(run) / "hf_sync.json", {})`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_restore_server.py
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = "20261031-120000-abcdef"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ASR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ASR_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("ASR_HF_RUN_REPO", "me/run")
    monkeypatch.delenv("ASR_COCKPIT_TOKEN", raising=False)
    spec = importlib.util.spec_from_file_location("server_hf_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_hf_app"] = mod
    spec.loader.exec_module(mod)
    launched = []

    class FakeProcess:
        pid = os.getpid()

        def poll(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or FakeProcess())
    client = TestClient(mod.app)
    client.mod = mod
    client.launched = launched
    yield client
    mod._close_hf_restore_log_fd()


def test_hf_config_redacts_token(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    data = client.get("/api/hf/config").json()
    assert data["enabled"] is True
    assert data["repo_id"] == "me/run"
    assert "hf_secret" not in json.dumps(data)


def test_restore_rejects_invalid_run_id(client):
    assert client.post("/api/hf/restore", json={"run_id": "../bad"}).status_code == 400


def test_restore_refuses_existing_local_run(client):
    run_dir = client.mod.RUNS_DIR / RUN
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}")
    res = client.post("/api/hf/restore", json={"run_id": RUN})
    assert res.status_code == 409


def test_restore_spawns_script_and_writes_initial_status(client):
    res = client.post(
        "/api/hf/restore",
        json={"run_id": RUN, "repo": "me/run", "include_merged": True},
    )
    assert res.status_code == 200, res.text
    cmd = client.launched[-1]
    assert "--run" in cmd and RUN in cmd
    assert "--repo" in cmd and "me/run" in cmd
    assert "--include-merged" in cmd
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "running"
    assert status["run_id"] == RUN


def test_restore_status_reads_file_and_log_tail(client):
    client.mod._hf_restore_status_file().write_text(
        json.dumps(
            {
                "state": "completed",
                "run_id": RUN,
                "message": "done",
                "checkpoint_step": 1000,
                "updated_at": 1.0,
            }
        )
    )
    client.mod._hf_restore_log_file().write_text("line 1\nline 2\n")
    data = client.get("/api/hf/restore/status").json()
    assert data["state"] == "completed"
    assert data["log_tail"] == ["line 1", "line 2"]


def test_run_payload_includes_hf_sync(client):
    from src import run_state

    run_state.create_run(client.mod.RUNS_DIR, run_id=RUN, config={}, trainer_args={})
    (client.mod.RUNS_DIR / RUN / "hf_sync.json").write_text(
        json.dumps({"status": "idle", "last_success_step": 1000})
    )
    data = client.get(f"/api/runs/{RUN}").json()
    assert data["hf_sync"]["last_success_step"] == 1000


def test_stale_running_restore_status_becomes_interrupted(client, monkeypatch):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "running", "run_id": RUN, "pid": 999999})
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    assert client.mod._hf_restore_running() is False
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "interrupted"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hf_restore_server.py -v`
Expected: FAIL with `404`/missing attributes.

- [ ] **Step 3: Implement the server endpoints**

Define `HF_RESTORE_STATUS_FILE = RUNS_DIR / "_hf_restore.status.json"`, `HF_RESTORE_LOG_FILE = RUNS_DIR / "_hf_restore.log"`, and module globals `hf_restore_process`, `hf_restore_log_fd`, `hf_restore_run_id`. Add `_hf_restore_status()` (read JSON, default `{"state": "idle"}`), `_hf_restore_log_file()`, `_hf_restore_status_file()`, `_close_hf_restore_log_fd()`, and `_hf_restore_running()` (poll the child, close the log, mark a stale `running` status with a dead PID as `interrupted`). Add `_start_hf_restore` that validates the run ID, refuses while `_any_job_running()` or `_hf_restore_running()` is true, refuses a non-empty local run dir, resolves the repo from the request or `ASR_HF_RUN_REPO`, opens the log, writes the initial status, and spawns `[sys.executable, "-u", "scripts/hf_restore.py", "--run", run_id, "--repo", repo, "--runs-root", str(RUNS_DIR), "--status-file", str(status_file)]` plus `--include-merged`; use `start_new_session=True` and `cwd=str(BASE_DIR)`.

If the restore child cannot be spawned, write `state="failed"` with the error to the status file and return `500`; never leave a stale `running` status. Add the four endpoints. `GET /api/hf/config` returns `enabled` from `src.hf_sync` availability (lazy `importlib.util.find_spec("huggingface_hub") is not None`), `repo_id` from env (empty when unset), and `runs_root`. `GET /api/hf/runs` validates the repo, builds `HfApi`, calls `list_remote_runs`, caches the result per repo for 30 seconds, and returns `502` with a short error on failure. `POST /api/hf/restore` calls `_check_auth(payload)` and `_start_hf_restore`. `GET /api/hf/restore/status` returns the status plus `log_tail` (last 40 lines). Add `("startup HF restore recovery", _hf_restore_running)` to `_recover_runs_on_startup`. Add `hf`/`hf_sync` to `_run_payload` and to the WebSocket payload.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hf_restore_server.py tests/test_server_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_hf_restore_server.py
git commit -m "feat: add cockpit HF restore endpoints"
```

---

### Task 7: Cockpit Recover Dialog, Remote Run List, and Sync Badge

**Files:**
- Modify: `server/static/index.html`
- Modify: `server/static/app.js`
- Modify: `server/static/run_view.js`
- Modify: `server/static/style.css`
- Test: `tests/test_web_ui_contract.py`
- Test: `tests/test_run_view.js`

**Interfaces:**
- Consumes: Task 6 endpoints and `_run_payload.hf_sync`.
- Produces:
  - `recoverActionState(remoteRun) -> {canRestore, canResume, reason}`
  - `hfSyncLabel(sync) -> string`
  - `remoteRunOptionLabel(run) -> string`
  - DOM ids: `btnRecover`, `hfRecoverDialog`, `inputHfRepo`, `inputHfRunId`, `btnHfList`, `hfRunList`, `btnHfRestore`, `btnHfRestoreResume`, `hfRestoreStatus`, `hfRestoreLog`, `btnHfClose`, `hfSyncBadge`.
  - JS functions: `loadHfConfig`, `openRecoverDialog`, `closeRecoverDialog`, `loadRemoteRuns`, `renderRemoteRuns`, `startRestore`, `pollRestore`, `renderHfBadge`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web_ui_contract.py`:

```python
def test_hf_recover_ui_is_present_and_wired():
    html = (ROOT / "server/static/index.html").read_text()
    js = (ROOT / "server/static/app.js").read_text()
    for element in (
        "btnRecover",
        "hfRecoverDialog",
        "inputHfRepo",
        "inputHfRunId",
        "btnHfList",
        "hfRunList",
        "btnHfRestore",
        "btnHfRestoreResume",
        "hfRestoreStatus",
        "hfRestoreLog",
        "btnHfClose",
        "hfSyncBadge",
    ):
        assert f'id="{element}"' in html
    for function in (
        "loadHfConfig",
        "openRecoverDialog",
        "closeRecoverDialog",
        "loadRemoteRuns",
        "renderRemoteRuns",
        "startRestore",
        "pollRestore",
        "renderHfBadge",
    ):
        assert f"function {function}(" in js
    for endpoint in (
        "/api/hf/config",
        "/api/hf/runs",
        "/api/hf/restore",
        "/api/hf/restore/status",
    ):
        assert endpoint in js
    assert "hfSyncLabel" in js
    assert "recoverActionState" in js
```

Add `recoverActionState`, `hfSyncLabel`, and `remoteRunOptionLabel` to the top-of-file `require` destructure in `tests/test_run_view.js`, then add before the export checks:

```js
assert.deepStrictEqual(
  recoverActionState({ run_id: "r", status: "running", has_checkpoint: true }),
  { canRestore: true, canResume: true, reason: "" },
);
assert.strictEqual(
  recoverActionState({ run_id: "r", status: "completed", has_checkpoint: true })
    .canResume,
  false,
);
assert.match(
  recoverActionState({ run_id: "r", status: "completed", has_checkpoint: true })
    .reason,
  /completed/i,
);
assert.strictEqual(
  recoverActionState({ run_id: "r", status: "running", has_checkpoint: false })
    .canResume,
  false,
);
assert.strictEqual(
  hfSyncLabel({ enabled: true, status: "idle", last_success_step: 2000 }),
  "HF: synced 2,000",
);
assert.strictEqual(hfSyncLabel({ enabled: true, status: "uploading" }), "HF: uploading…");
assert.match(hfSyncLabel({ enabled: true, status: "error" }), /error/i);
assert.strictEqual(hfSyncLabel({ enabled: false }), "");
assert.strictEqual(
  remoteRunOptionLabel({ run_id: "20261031-120000-abcdef", status: "running", last_step: 1000 }),
  "20261031-120000-abcdef · running · step 1,000",
);
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_web_ui_contract.py -v && node tests/test_run_view.js`
Expected: FAIL on the missing ids/functions and helpers.

- [ ] **Step 3: Implement the UI**

- Add `btnRecover` to the top bar next to the run picker. Add a `<dialog id="hfRecoverDialog">` with the IDs above. Put `hfSyncBadge` in the run header meta line. Update the static asset query strings to `style.css?v=12`, `run_view.js?v=4`, `app.js?v=13`.
- In `run_view.js`, implement the three pure helpers and export them. `recoverActionState` returns `canRestore=True` for any valid `run_id`; `canResume` only when status is not `completed` and `has_checkpoint` is truthy; `reason` explains the disabled case. `hfSyncLabel` returns `""` when `enabled === false`, `HF: uploading…` when `status === "uploading"`, `HF: error — retrying` when `status === "error"`, `HF: waiting for first upload` when no step, and `HF: synced <step>` otherwise (append ` · <local time>` when `last_success_at` is valid). `remoteRunOptionLabel` returns `<id> · <status> · step <n>`.
- In `app.js`, add the element handles and functions. `loadHfConfig` fetches `/api/hf/config`, unhides `btnRecover` when `enabled`, and stores the default repo. `openRecoverDialog` shows the dialog, fills the repo, and focuses the run ID. `loadRemoteRuns` fetches `/api/hf/runs?repo=...`, renders a table with status/step/checkpoint, and fills the run ID on row click. `startRestore(resume)` posts to `/api/hf/restore`, disables both action buttons, and starts `pollRestore`. `pollRestore` fetches `/api/hf/restore/status` every 2 seconds, updates text/log, and on `completed` closes the dialog, refreshes `/api/runs`, selects the restored run, and calls `resumeRun()` when the resume flag was set; on `failed`/`interrupted` it re-enables the buttons and shows the error. `renderHfBadge` sets `hfSyncBadge.textContent` from `hfSyncLabel(run.hf_sync)` and hides the badge when the label is empty. Call it from `renderRun`, and in `applyLiveRun` pass `{...data.run, hf_sync: data.hf_sync}` so the WebSocket payload updates the badge.
- On page load, call `loadHfConfig()` and also check `/api/hf/restore/status`; if a restore is still `running` (for example after a page reload), open the dialog and resume polling instead of losing the progress view.
- In `style.css`, style the dialog, the remote list/table, the progress/log area, and the `.hf-sync-badge` idle/uploading/error states. Respect `prefers-reduced-motion` already handled globally.
- Disable `btnRecover` while a restore is pending; hide it when `loadHfConfig` reports `enabled: false`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_web_ui_contract.py -v && node tests/test_run_view.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/run_view.js server/static/style.css tests/test_web_ui_contract.py tests/test_run_view.js
git commit -m "feat: add cockpit HF recovery dialog"
```

---

### Task 8: Merge Upload, Documentation, and End-to-End Verification

**Files:**
- Modify: `scripts/merge_lora.py`
- Modify: `scripts/dev_fake_merge.py`
- Modify: `server/app.py`
- Modify: `README.md`
- Modify: `PRODUCTION.md`
- Modify: `.env.example`
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: `HFRunMirror`, `HFSyncConfig`.
- Produces: `merge_lora.py --run_dir <dir> --hf_run_repo <id>` uploads `<run_dir>/merged/` and a ledger with status `"merge"` when both are provided.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server_train.py`:

```python
def test_merge_command_passes_hf_context(client, monkeypatch):
    monkeypatch.setenv("ASR_HF_RUN_REPO", "me/run")
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-abcdef"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 200
    cmd = client.launched[-1]
    assert "--run_dir" in cmd
    assert "--hf_run_repo" in cmd
    assert "me/run" in cmd
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_server_train.py::test_merge_command_passes_hf_context -v`
Expected: FAIL because the command does not include the new flags.

- [ ] **Step 3: Implement merge upload and docs**

- Add `--run_dir` and `--hf_run_repo` (default `ASR_HF_RUN_REPO`) to `scripts/merge_lora.py` and `scripts/dev_fake_merge.py`; the fake accepts and ignores them.
- After `merged_model.save_pretrained` and `processor.save_pretrained`, when both `run_dir` and `hf_run_repo` are set, build `HFRunMirror` (run ID from `Path(run_dir).name`, output/results `run_dir/output` and `run_dir`), call `start()`, `enqueue_merged(Path(output_dir))`, `enqueue_ledger("merge")`, and `close()` in `finally`. Catch exceptions and log; a merge must still succeed if upload fails.
- In `server/app.py` `_start_merge`, append `--run_dir str(run_dir_path)` and `--hf_run_repo <id>` where the ID is `run.get("hf", {}).get("repo_id")` or `ASR_HF_RUN_REPO`; omit the HF flag when neither is set.
- Update `.env.example` with a commented `ASR_HF_RUN_REPO=user/stt-<experiment>` and a note that `HF_TOKEN` is shared with `upload_to_hf.py`.
- Add a "Durable HF run mirror" section to `README.md` and `PRODUCTION.md` covering: enable with `ASR_HF_RUN_REPO`; checkpoint/stage/snapshot behavior; restore with `python scripts/hf_restore.py --run <id>`; the cockpit Recover dialog; private-by-default; retention is manual.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_server_train.py -v`
Expected: PASS.

Then run the full suite:

Run: `pytest -q`
Expected: PASS (or only pre-existing skips).

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_lora.py scripts/dev_fake_merge.py server/app.py README.md PRODUCTION.md .env.example tests/test_server_train.py
git commit -m "feat: mirror merged models and document HF recovery"
```

---

## Self-Review

- **Spec coverage:** Config/repo layout (Task 1), restore by run ID and path/normalization (Task 2), uploader/high-water/retries (Task 3), adapter snapshots (Task 4), trainer flags/auto-run/stage uploads (Task 5), server endpoints/status payload (Task 6), cockpit dialog/badge (Task 7), merge upload/docs (Task 8).
- **Type consistency:** `restore_run` returns `checkpoint_step` as `int | None`; Task 2 CLI and Task 5 `train_lora` both check it. `HFRunMirror` methods used by Task 4 and Task 5 match Task 3. `_run_payload["hf_sync"]` is consumed by Task 7's `hfSyncLabel`.
- **Review focus coverage:** high-water (Tasks 3, 6), partial checkpoint (Tasks 2, 3), missing repo/run (Tasks 2, 5, 6), completed run (Tasks 2, 5, 7), HF outage (Tasks 3, 4, 7), existing local run (Tasks 2, 6).
- **No open TBDs:** every flag, file, endpoint, and exit code above is named; the builder can choose internal helper names and exact HTML copy.
