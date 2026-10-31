"""Hugging Face run-mirror configuration, layout, validation, and discovery.

This module is the stdlib-only foundation for mirroring run artifacts to a
private Hugging Face repo and restoring them by run id. It deliberately has no
module-level dependency on ``huggingface_hub``: the feature can be disabled (or
the CLI imported) on a machine without that package installed. The uploader and
restore machinery import ``huggingface_hub`` lazily when they actually run.

Repo layout (one self-contained folder per run)::

    latest.json
    runs/<run_id>/run.json
    runs/<run_id>/hf_state.json
    runs/<run_id>/checkpoint-<step>/
    runs/<run_id>/adapter_snapshots/step-<step>/
    runs/<run_id>/best_adapter/
    runs/<run_id>/merged/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

LATEST_FILE = "latest.json"
MANIFEST_FILE = "run.json"
STATE_FILE = "hf_state.json"
SYNC_FILE = "hf_sync.json"
PENDING_FILE = "hf_sync_pending.json"
README_FILE = "README.md"
TRAINING_LOG_FILE = "training_live.log"
MAX_TRAINING_LOG_BYTES = 5 * 1024 * 1024

LEDGER_ROOT_FILES = (
    MANIFEST_FILE,
    "steps.jsonl",
    "events.jsonl",
    "live_telemetry.json",
    "loss_history.json",
    "epoch_benchmarks.json",
)
LEDGER_PREFLIGHT_FILE = "baselines/preflight.json"
REPORT_FILES = ("final_evaluation_report.json", "final_evaluation_report.md")

README_CONTENT = """\
# ASR run mirror

Private mirror of ASR training runs. Each run is self-contained so it can be
restored on a fresh pod by run id.

## Layout

```
latest.json
runs/<run_id>/
  run.json
  steps.jsonl
  events.jsonl
  live_telemetry.json
  loss_history.json
  epoch_benchmarks.json
  baselines/preflight.json
  hf_state.json
  checkpoint-<step>/
  adapter_snapshots/step-<step>/
  best_adapter/
  reports/
  merged/
  training_live.log
```

`latest.json` points at the most recently active run. `runs/<run_id>/hf_state.json`
is the per-run high-water mark and the authoritative state for an older run.

## Restore

```bash
python scripts/hf_restore.py --run <run_id>
python scripts/train_lora.py --hf_resume <run_id>
```

The repository is private by default.
"""

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
MERGED_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

DEFAULT_ADAPTER_SNAPSHOT_STEPS = 250
DEFAULT_LIMIT = 50


class HFSyncError(Exception):
    """Base class for mirror-related failures."""


class HFRestoreError(HFSyncError):
    """A restore request cannot be satisfied."""


class HFRestoreConflict(HFRestoreError):
    """Restoring would overwrite newer local or remote state."""


class HFNoCheckpointError(HFRestoreError):
    """The selected run has no complete checkpoint to restore."""


class HFCompletedRunError(HFRestoreError):
    """``latest`` resolved to a completed run, which must be named explicitly."""


@dataclass(frozen=True)
class HFSyncConfig:
    """Effective mirror configuration for one training process."""

    repo_id: str
    private: bool = True
    adapter_snapshot_steps: int = DEFAULT_ADAPTER_SNAPSHOT_STEPS
    auto_resume: bool = True

    def __post_init__(self) -> None:
        validate_repo_id(self.repo_id)
        object.__setattr__(self, "private", bool(self.private))
        object.__setattr__(
            self,
            "adapter_snapshot_steps",
            max(0, int(self.adapter_snapshot_steps)),
        )
        object.__setattr__(self, "auto_resume", bool(self.auto_resume))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def resolve_config(
    args: Any = None,
    environ: Mapping[str, str] | None = None,
) -> HFSyncConfig | None:
    """Resolve the mirror config from CLI args, then the environment.

    A CLI repo wins over ``ASR_HF_RUN_REPO``. With neither set the feature is
    off and ``None`` is returned, so callers make no network calls.
    """
    env = os.environ if environ is None else environ

    raw_repo = getattr(args, "hf_run_repo", None) if args is not None else None
    repo_id = str(raw_repo).strip() if raw_repo else ""
    if not repo_id:
        repo_id = str(env.get("ASR_HF_RUN_REPO") or "").strip()
    if not repo_id:
        return None
    validate_repo_id(repo_id)

    raw_private = getattr(args, "hf_private", None) if args is not None else None
    private = True if raw_private is None else bool(raw_private)

    raw_steps = (
        getattr(args, "hf_adapter_snapshot_steps", None) if args is not None else None
    )
    if raw_steps is None:
        steps = DEFAULT_ADAPTER_SNAPSHOT_STEPS
    else:
        try:
            steps = max(0, int(raw_steps))
        except (TypeError, ValueError):
            steps = DEFAULT_ADAPTER_SNAPSHOT_STEPS

    raw_auto = getattr(args, "hf_auto_resume", None) if args is not None else None
    auto_resume = True if raw_auto is None else bool(raw_auto)

    return HFSyncConfig(
        repo_id=repo_id,
        private=private,
        adapter_snapshot_steps=steps,
        auto_resume=auto_resume,
    )


def validate_repo_id(repo_id: str) -> None:
    """Raise ``ValueError`` unless *repo_id* is a safe ``owner/name`` id."""
    if not isinstance(repo_id, str):
        raise ValueError(f"invalid Hugging Face repo id: {repo_id!r}")
    candidate = repo_id.strip()
    if (
        candidate != repo_id
        or not candidate
        or ".." in candidate
        or REPO_ID_RE.fullmatch(candidate) is None
    ):
        raise ValueError(
            f"invalid Hugging Face repo id {repo_id!r}; "
            "expected '<owner>/<name>' with no extra path segments"
        )


# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------


def _require_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(
            f"invalid run id {run_id!r}; expected YYYYMMDD-HHMMSS-<6 hex chars>"
        )
    return run_id


def repo_run_path(run_id: str) -> str:
    """Repo-relative folder for one run."""
    return f"runs/{_require_run_id(run_id)}"


def repo_checkpoint_path(run_id: str, step: int) -> str:
    """Repo-relative folder for one run's ``checkpoint-<step>``."""
    _require_run_id(run_id)
    try:
        value = int(step)
    except (TypeError, ValueError):
        raise ValueError(f"invalid checkpoint step: {step!r}") from None
    if value < 0:
        raise ValueError(f"invalid checkpoint step: {step!r}")
    return f"runs/{run_id}/checkpoint-{value}"


# ---------------------------------------------------------------------------
# Local artifact validation
# ---------------------------------------------------------------------------


def _has_any_file(directory: Path, names: tuple[str, ...]) -> bool:
    return any((directory / name).is_file() for name in names)


def is_complete_checkpoint(path: Path | str) -> bool:
    """A checkpoint is complete with trainer state and adapter weights."""
    directory = Path(path)
    return (
        directory.is_dir()
        and (directory / "trainer_state.json").is_file()
        and _has_any_file(directory, ADAPTER_WEIGHT_FILES)
    )


def validate_adapter(path: Path | str) -> tuple[bool, str]:
    """Validate a PEFT adapter directory; return ``(ok, reason)``."""
    directory = Path(path)
    if not directory.is_dir():
        return False, f"adapter directory not found: {directory}"
    if not (directory / ADAPTER_CONFIG_FILE).is_file():
        return False, f"missing {ADAPTER_CONFIG_FILE}"
    if not _has_any_file(directory, ADAPTER_WEIGHT_FILES):
        return False, "missing adapter weights (adapter_model.safetensors/.bin)"
    return True, ""


def validate_merged(path: Path | str) -> tuple[bool, str]:
    """Validate a standalone merged model directory; return ``(ok, reason)``."""
    directory = Path(path)
    if not directory.is_dir():
        return False, f"merged model directory not found: {directory}"
    if not (directory / "config.json").is_file():
        return False, "missing config.json"
    if not _has_any_file(directory, MERGED_WEIGHT_FILES):
        return False, "missing model weights"
    return True, ""


def select_checkpoint_step(files: Iterable[str], run_id: str) -> int | None:
    """Highest complete remote checkpoint step for *run_id*, or ``None``.

    A checkpoint is complete only when the listing contains both
    ``trainer_state.json`` and an adapter weight file in the same folder.
    """
    _require_run_id(run_id)
    prefix = f"runs/{run_id}/"
    state_steps: set[int] = set()
    weight_steps: set[int] = set()

    for raw in files:
        if not isinstance(raw, str):
            continue
        path = raw.strip("/")
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix) :]
        folder, _, filename = rest.partition("/")
        match = CHECKPOINT_RE.fullmatch(folder)
        if match is None or not filename:
            continue
        step = int(match.group(1))
        if filename == "trainer_state.json":
            state_steps.add(step)
        elif filename in ADAPTER_WEIGHT_FILES:
            weight_steps.add(step)

    complete = state_steps & weight_steps
    return max(complete) if complete else None


# ---------------------------------------------------------------------------
# Remote discovery
# ---------------------------------------------------------------------------


def _json_payload(raw: Any) -> dict | None:
    """Decode a downloaded JSON object, tolerating bytes and bad input."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(raw, str):
        text = raw
    else:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _download_json(download: Any, repo_id: str, remote_path: str) -> dict | None:
    return _json_payload(download(repo_id, remote_path))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default  # reject NaN


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _remote_run_ids(files: Iterable[str]) -> list[str]:
    """Run ids with a ``runs/<run_id>/run.json`` manifest, oldest first."""
    ids: set[str] = set()
    for raw in files:
        if not isinstance(raw, str):
            continue
        path = raw.strip("/")
        if not path.startswith("runs/") or not path.endswith(f"/{MANIFEST_FILE}"):
            continue
        run_id = path[len("runs/") : -len(f"/{MANIFEST_FILE}")]
        if RUN_ID_RE.fullmatch(run_id) is not None:
            ids.add(run_id)
    return sorted(ids)


def _default_download(repo_id: str, remote_path: str) -> bytes:
    """Download one repo file and return its bytes.

    ``huggingface_hub`` is imported lazily so this module (and ``--help``)
    works on machines where the optional dependency is absent.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - lazy on purpose

    with tempfile.TemporaryDirectory(prefix="hf-sync-") as tmp:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=tmp,
        )
        return Path(local_path).read_bytes()


def list_remote_runs(
    api: Any,
    repo_id: str,
    *,
    download: Any = _default_download,
    limit: int | None = DEFAULT_LIMIT,
) -> list[dict]:
    """Return remote run summaries, newest activity first.

    Each summary carries ``run_id``, ``status``, ``last_step``, ``updated_at``,
    and ``has_checkpoint``. ``hf_state.json`` is authoritative for
    ``last_step``; the run manifest is the fallback.
    """
    validate_repo_id(repo_id)
    files = list(api.list_repo_files(repo_id, revision="main"))
    file_set = set(files)
    rows: list[dict] = []

    for run_id in _remote_run_ids(files):
        manifest_path = f"runs/{run_id}/{MANIFEST_FILE}"
        try:
            manifest = _download_json(download, repo_id, manifest_path)
        except Exception:
            # One unreadable manifest must not hide every other run.
            continue
        if manifest is None:
            continue

        status = str(manifest.get("status") or "unknown")
        updated_at = _as_float(manifest.get("updated_at"))
        last_step = _as_int(manifest.get("last_step"))

        state_path = f"runs/{run_id}/{STATE_FILE}"
        if state_path in file_set:
            try:
                state = _download_json(download, repo_id, state_path)
            except Exception:
                state = None
            if state is not None:
                state_step = _as_int(state.get("last_step"))
                if state_step is not None:
                    last_step = state_step

        checkpoint_step = select_checkpoint_step(files, run_id)
        rows.append(
            {
                "run_id": run_id,
                "status": status,
                "last_step": last_step,
                "updated_at": updated_at,
                "has_checkpoint": checkpoint_step is not None,
            }
        )

    rows.sort(key=lambda row: (row["updated_at"], row["run_id"]), reverse=True)
    if limit is None:
        return rows
    try:
        capped = max(0, int(limit))
    except (TypeError, ValueError):
        capped = DEFAULT_LIMIT
    return rows[:capped]


def _available_runs_message(available: Iterable[str]) -> str:
    listed = ", ".join(sorted(available, reverse=True)[:10])
    return listed or "none"


def resolve_resume_run(
    api: Any,
    repo_id: str,
    resume: str,
    *,
    download: Any = _default_download,
) -> str:
    """Resolve ``latest`` or an explicit id to a run that exists remotely.

    ``latest`` follows ``latest.json`` (falling back to the newest run id when
    the pointer is absent) and refuses a completed run unless the caller names
    that run explicitly. An explicit id is returned even when completed.
    """
    validate_repo_id(repo_id)
    target = str(resume or "").strip()
    if not target or target == "none":
        raise HFRestoreError("no run was requested for restore")

    files = list(api.list_repo_files(repo_id, revision="main"))
    available = _remote_run_ids(files)

    if target != "latest":
        if RUN_ID_RE.fullmatch(target) is None:
            raise HFRestoreError(
                f"invalid run id {target!r}; expected YYYYMMDD-HHMMSS-<6 hex chars>"
            )
        if target not in available:
            raise HFRestoreError(
                f"run {target} was not found in {repo_id}; "
                f"available runs: {_available_runs_message(available)}"
            )
        return target

    if LATEST_FILE in files:
        try:
            pointer = _download_json(download, repo_id, LATEST_FILE)
        except Exception as exc:
            raise HFRestoreError(
                f"could not read {LATEST_FILE} from {repo_id}: {exc}"
            ) from exc
        if pointer is None:
            raise HFRestoreError(
                f"{LATEST_FILE} in {repo_id} is not a JSON object"
            )
        target = str(pointer.get("run_id") or "").strip()
        if RUN_ID_RE.fullmatch(target) is None:
            raise HFRestoreError(
                f"{LATEST_FILE} in {repo_id} has an invalid run_id: {target!r}"
            )
    else:
        # Run ids embed a UTC timestamp and sort chronologically as strings.
        if not available:
            raise HFRestoreError(f"no runs were found in {repo_id}")
        target = max(available)

    if target not in available:
        raise HFRestoreError(
            f"run {target} was not found in {repo_id}; "
            f"available runs: {_available_runs_message(available)}"
        )

    try:
        manifest = _download_json(
            download, repo_id, f"runs/{target}/{MANIFEST_FILE}"
        )
    except Exception as exc:
        raise HFRestoreError(
            f"could not read the manifest for run {target} in {repo_id}: {exc}"
        ) from exc
    if manifest is None:
        raise HFRestoreError(
            f"the manifest for run {target} in {repo_id} is not a JSON object"
        )

    if str(manifest.get("status") or "") == "completed":
        raise HFCompletedRunError(
            f"latest run {target} is completed; pass --hf_resume {target} "
            "to restore it explicitly, or start a fresh run"
        )
    return target


# ---------------------------------------------------------------------------
# Hugging Face API and restore
# ---------------------------------------------------------------------------

RESTORE_LEDGER_FILES = (
    MANIFEST_FILE,
    "steps.jsonl",
    "events.jsonl",
    "live_telemetry.json",
    "loss_history.json",
    "epoch_benchmarks.json",
    "baselines/preflight.json",
    STATE_FILE,
)

# Saved argparse values that never carry over to a restored local run.
RESET_KEYS = frozenset(
    {
        "run_dir",
        "output_dir",
        "results_dir",
        "preflight_cache_dir",
        "resume_from_checkpoint",
        "fresh_start",
        "auto_resume",
        "hf_resume",
        "hf_run_repo",
        "post_eval_only",
    }
)

_ADAPTER_SNAPSHOT_RE = re.compile(r"^step-(\d+)$")
_ACTIVE_RUN_STATUSES = frozenset(("starting", "running"))


def make_api() -> Any:
    """Return a Hugging Face ``HfApi``, imported only when needed.

    Raises:
        HFSyncError: when ``huggingface_hub`` is not installed, with install
            and authentication remediation.
    """
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - lazy on purpose
    except Exception as exc:
        raise HFSyncError(
            "huggingface_hub is required for Hugging Face run mirroring; "
            "install it with `pip install huggingface_hub` and authenticate "
            "with `hf auth login` (or set HF_TOKEN)"
        ) from exc
    return HfApi()


def _payload_bytes(raw: Any, remote_path: str) -> bytes:
    """Normalize a downloader result to bytes."""
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    raise HFRestoreError(
        f"download of {remote_path} returned {type(raw).__name__}, expected bytes"
    )


def _fetch(download: Any, repo_id: str, remote_path: str) -> bytes:
    try:
        raw = download(repo_id, remote_path)
    except Exception as exc:
        raise HFRestoreError(
            f"could not download {remote_path} from {repo_id}: {exc}"
        ) from exc
    return _payload_bytes(raw, remote_path)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _relative_remote_path(remote_path: str, prefix: str) -> str:
    """Return the safe path below *prefix*, rejecting traversal."""
    relative = remote_path[len(prefix) :]
    parts = Path(relative).parts
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or ".." in parts
    ):
        raise HFRestoreError(f"unsafe path in remote listing: {remote_path!r}")
    return relative


def _listed_files(api: Any, repo_id: str) -> list[str]:
    raw = api.list_repo_files(repo_id, revision="main")
    files: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        path = item.strip("/")
        if path:
            files.append(path)
    return files


def _run_dir_is_nonempty(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        return True
    return any(run_dir.iterdir())


def _refuse_existing_run(run_dir: Path, *, force: bool) -> None:
    if not force and _run_dir_is_nonempty(run_dir):
        raise HFRestoreConflict(
            f"local run directory already exists: {run_dir}; "
            "pass --force to replace it"
        )


def _remove_path(path: Path) -> None:
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _make_staging_run_dir(runs_root: Path, run_id: str) -> Path:
    """Create a hidden sibling directory that becomes the run when complete."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.restore-", dir=str(runs_root))
    )
    (staging / "output").mkdir(exist_ok=True)
    return staging


def _install_staged_run(staging: Path, run_dir: Path, *, force: bool) -> None:
    """Move a fully restored staging tree into the final run directory."""
    _refuse_existing_run(run_dir, force=force)
    if not run_dir.exists():
        os.replace(staging, run_dir)
        return
    if run_dir.is_dir() and not any(run_dir.iterdir()):
        run_dir.rmdir()
        os.replace(staging, run_dir)
        return

    # Keep the old run until the staged replacement is ready, then swap.
    backup = run_dir.with_name(
        f".{run_dir.name}.restore-old-{os.getpid()}-{time.time_ns()}"
    )
    os.replace(run_dir, backup)
    try:
        os.replace(staging, run_dir)
    except Exception:
        try:
            os.replace(backup, run_dir)
        except Exception:
            pass
        raise
    try:
        _remove_path(backup)
    except OSError:
        # The new run is already installed; a leftover backup is harmless.
        pass


def _download_tree(
    download: Any,
    repo_id: str,
    remote_paths: Iterable[str],
    prefix: str,
    target: Path,
) -> list[str]:
    """Download every path under *prefix* into *target*; return repo paths."""
    downloaded: list[str] = []
    for remote in sorted(remote_paths):
        relative = _relative_remote_path(remote, prefix)
        data = _fetch(download, repo_id, remote)
        _write_bytes_atomic(Path(target) / relative, data)
        downloaded.append(remote)
    return downloaded


def _download_checkpoint(
    download: Any,
    repo_id: str,
    run_id: str,
    files: Iterable[str],
    output_dir: Path,
) -> tuple[int | None, list[str]]:
    """Download the newest checkpoint that verifies locally.

    A listing-complete checkpoint can still fail local verification (for
    example, a remote listing that disagrees with the downloaded tree). Each
    attempt lands in a hidden temporary directory and is only moved into
    ``output/checkpoint-<step>`` after ``is_complete_checkpoint`` accepts it.
    A failed local verification falls back to the next lower complete
    checkpoint; download and write errors propagate to the caller.
    """
    prefix = f"{repo_run_path(run_id)}/"
    candidates = [
        path for path in files if path.startswith(prefix)
    ]
    step = select_checkpoint_step(candidates, run_id)
    while step is not None:
        checkpoint_prefix = f"{prefix}checkpoint-{step}/"
        checkpoint_files = [
            path for path in candidates if path.startswith(checkpoint_prefix)
        ]
        temp_dir = output_dir / f".restore-checkpoint-{step}.tmp"
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            downloaded = _download_tree(
                download, repo_id, checkpoint_files, checkpoint_prefix, temp_dir
            )
            if is_complete_checkpoint(temp_dir):
                destination = output_dir / f"checkpoint-{step}"
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(temp_dir, destination)
                return step, downloaded
        finally:
            # Verification failure falls through to the next lower complete
            # checkpoint. Download/write errors propagate after temp cleanup.
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        candidates = [
            path for path in candidates if not path.startswith(checkpoint_prefix)
        ]
        step = select_checkpoint_step(candidates, run_id)
    return None, []


def _download_best_adapter(
    download: Any,
    repo_id: str,
    files: Iterable[str],
    run_prefix: str,
    output_dir: Path,
) -> list[str]:
    prefix = f"{run_prefix}/best_adapter/"
    remote_paths = [path for path in files if path.startswith(prefix)]
    return _download_tree(
        download, repo_id, remote_paths, prefix, output_dir / "best_adapter"
    )


def _download_reports(
    download: Any,
    repo_id: str,
    files: Iterable[str],
    run_prefix: str,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Restore report files into both the results and output folders."""
    prefix = f"{run_prefix}/reports/"
    downloaded: list[str] = []
    for remote in sorted(path for path in files if path.startswith(prefix)):
        relative = _relative_remote_path(remote, prefix)
        data = _fetch(download, repo_id, remote)
        _write_bytes_atomic(output_dir / relative, data)
        _write_bytes_atomic(run_dir / relative, data)
        downloaded.append(remote)
    return downloaded


def _download_newest_adapter_snapshot(
    download: Any,
    repo_id: str,
    files: Iterable[str],
    run_prefix: str,
    output_dir: Path,
) -> list[str]:
    prefix = f"{run_prefix}/adapter_snapshots/"
    # Keep the literal remote folder name for each snapshot. Snapshots are
    # zero-padded by the trainer callback (``step-000250``), so reconstructing
    # the folder from the parsed integer (``step-250``) would build the wrong
    # remote prefix and a nested local target.
    snapshots: dict[int, dict[str, list[str]]] = {}
    for remote in files:
        if not remote.startswith(prefix):
            continue
        relative = remote[len(prefix) :]
        folder, separator, remainder = relative.partition("/")
        if not separator or not remainder:
            continue
        match = _ADAPTER_SNAPSHOT_RE.fullmatch(folder)
        if match is not None:
            step = int(match.group(1))
            snapshots.setdefault(step, {}).setdefault(folder, []).append(remote)

    if not snapshots:
        return []
    newest = max(snapshots)
    folders = snapshots[newest]
    # A listing should only contain one spelling per step; pick
    # deterministically if it does not.
    folder = max(folders)
    snapshot_prefix = f"{prefix}{folder}/"
    return _download_tree(
        download,
        repo_id,
        folders[folder],
        snapshot_prefix,
        output_dir / "adapter_snapshots" / folder,
    )


def _download_merged(
    download: Any,
    repo_id: str,
    files: Iterable[str],
    run_prefix: str,
    run_dir: Path,
) -> list[str]:
    prefix = f"{run_prefix}/merged/"
    remote_paths = [path for path in files if path.startswith(prefix)]
    return _download_tree(download, repo_id, remote_paths, prefix, run_dir / "merged")


def rewrite_run_paths(run: dict, run_dir: Path) -> dict:
    """Point a restored manifest at its new local folder."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "output"
    paths = run.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        run["paths"] = paths
    paths["run_dir"] = str(run_dir)
    paths["output_dir"] = str(output_dir)
    paths["results_dir"] = str(run_dir)
    paths["merged_dir"] = str(run_dir / "merged")
    paths["preflight_cache"] = str(run_dir / "baselines" / "preflight.json")

    trainer_args = run.get("trainer_args")
    if isinstance(trainer_args, dict):
        trainer_args["run_dir"] = str(run_dir)
        trainer_args["output_dir"] = str(output_dir)
        trainer_args["results_dir"] = str(run_dir)
        trainer_args["preflight_cache_dir"] = str(run_dir)
        trainer_args.pop("merged_dir", None)
    return run


def normalize_restored_run(run: dict) -> dict:
    """Clear dead process state while preserving results and config."""
    now = time.time()
    current_status = run.get("status")
    was_active = (
        isinstance(current_status, str)
        and current_status in _ACTIVE_RUN_STATUSES
    )
    if was_active:
        run["status"] = "interrupted"
        run["ended_at"] = now

    run["current_pid"] = None
    run["merge_pid"] = None
    run.pop("training.pid", None)
    run.pop("training_pid", None)

    training = run.get("training")
    if isinstance(training, dict):
        training.pop("pid", None)

    attempts = run.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("ended_at") is None:
                attempt["ended_at"] = now

    stages = run.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict) and stage.get("status") == "running":
                stage["status"] = "interrupted"
                stage["ended_at"] = now
    return run


def _stage_status(run: dict, name: str) -> str:
    for stage in run.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name") == name:
            return str(stage.get("status") or "pending")
    return "pending"


def apply_restored_args(
    args: Any,
    run: dict,
    *,
    run_dir: Path,
    explicit: set[str] | None = None,
) -> Any:
    """Apply saved trainer args, then the restored run's local paths/flags."""
    saved = run.get("trainer_args")
    skip = set(RESET_KEYS) | set(explicit or ())
    if isinstance(saved, dict):
        for key, value in saved.items():
            if key in skip:
                continue
            setattr(args, key, value)

    run_dir = Path(run_dir)
    output_dir = run_dir / "output"
    args.run_dir = str(run_dir)
    args.output_dir = str(output_dir)
    args.results_dir = str(run_dir)
    args.preflight_cache_dir = str(run_dir)
    args.fresh_start = False
    args.auto_resume = True
    args.hf_resume = "none"

    training_done = _stage_status(run, "training") == "completed"
    post_eval_status = _stage_status(run, "post_eval")
    args.post_eval_only = bool(
        training_done and post_eval_status not in ("completed", "skipped")
    )
    return args


def restore_run(
    api: Any,
    repo_id: str,
    run_id: str,
    runs_root: Path | str,
    *,
    include_merged: bool = False,
    force: bool = False,
    download: Any = _default_download,
    status: Any = lambda state, message: None,
) -> dict:
    """Restore one mirrored run into ``runs_root/<run_id>``.

    The ledger and newest complete checkpoint are downloaded, the manifest is
    rewritten to local paths, and dead process state is cleared. A run with no
    complete checkpoint is still restored (ledger-only); the caller decides
    whether that is enough to resume.
    """
    validate_repo_id(repo_id)
    run_prefix = repo_run_path(run_id)
    run_dir = Path(runs_root) / run_id

    status("listing", f"Listing {repo_id} for run {run_id}")
    files = _listed_files(api, repo_id)
    file_set = set(files)

    manifest_path = f"{run_prefix}/{MANIFEST_FILE}"
    if manifest_path not in file_set:
        raise HFRestoreError(
            f"run {run_id} was not found in {repo_id}; "
            f"expected {manifest_path}"
        )

    # Refuse a conflicting local run before downloading anything, but keep it
    # in place until the replacement has been fully staged and validated.
    _refuse_existing_run(run_dir, force=force)
    staging_dir = _make_staging_run_dir(Path(runs_root), run_id)
    output_dir = staging_dir / "output"

    try:
        status("ledger", f"Downloading the run ledger for {run_id}")
        manifest_payload: dict | None = None
        downloaded: list[str] = []
        for relative in RESTORE_LEDGER_FILES:
            remote_path = f"{run_prefix}/{relative}"
            if remote_path not in file_set:
                continue
            data = _fetch(download, repo_id, remote_path)
            _write_bytes_atomic(staging_dir / relative, data)
            downloaded.append(remote_path)
            if relative == MANIFEST_FILE:
                manifest_payload = _json_payload(data)
        if manifest_payload is None:
            raise HFRestoreError(
                f"the manifest for run {run_id} in {repo_id} is not a JSON object"
            )

        status(
            "checkpoint",
            f"Restoring the newest complete checkpoint for {run_id}",
        )
        checkpoint_step, checkpoint_files = _download_checkpoint(
            download, repo_id, run_id, files, output_dir
        )
        downloaded.extend(checkpoint_files)

        downloaded.extend(
            _download_best_adapter(
                download, repo_id, files, run_prefix, output_dir
            )
        )
        downloaded.extend(
            _download_reports(
                download, repo_id, files, run_prefix, staging_dir, output_dir
            )
        )
        downloaded.extend(
            _download_newest_adapter_snapshot(
                download, repo_id, files, run_prefix, output_dir
            )
        )
        if include_merged:
            downloaded.extend(
                _download_merged(
                    download, repo_id, files, run_prefix, staging_dir
                )
            )

        run = manifest_payload
        rewrite_run_paths(run, run_dir)
        normalize_restored_run(run)
        run["run_id"] = run_id
        run["updated_at"] = time.time()

        try:
            (staging_dir / "training.pid").unlink(missing_ok=True)
        except OSError:
            pass
        _write_json_atomic(staging_dir / MANIFEST_FILE, run)
        _install_staged_run(staging_dir, run_dir, force=force)
    except BaseException:
        # Never leave a partial staging tree behind, and never remove the old
        # local run unless the staged replacement was installed successfully.
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    if checkpoint_step is None:
        status(
            "verify",
            f"Restored {run_id} without a complete checkpoint (ledger only)",
        )
    else:
        status("verify", f"Restored {run_id} at checkpoint-{checkpoint_step}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_dir": str(run_dir / "output"),
        "checkpoint_step": checkpoint_step,
        "downloaded": downloaded,
        "run": run,
    }


# ---------------------------------------------------------------------------
# Background uploader
# ---------------------------------------------------------------------------


@dataclass
class SyncJob:
    """One unit of upload work for :class:`HFRunMirror`."""

    kind: str
    path: Path | None = None
    step: int | None = None
    label: str = ""
    reports: bool = False


class _DropJob(Exception):
    """A job is invalid or disallowed; retrying it cannot help."""

    def __init__(self, message: str, *, record_error: bool = False) -> None:
        super().__init__(message)
        self.record_error = record_error


def _job_to_dict(job: SyncJob) -> dict:
    return {
        "kind": job.kind,
        "path": str(job.path) if job.path is not None else None,
        "step": job.step,
        "label": job.label,
        "reports": bool(job.reports),
    }


def _job_from_dict(raw: Any) -> SyncJob | None:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    path: Path | None = None
    raw_path = raw.get("path")
    if raw_path:
        try:
            path = Path(str(raw_path))
        except (TypeError, ValueError):
            return None
    label = raw.get("label")
    return SyncJob(
        kind=kind,
        path=path,
        step=_as_int(raw.get("step")),
        label=str(label) if label is not None else "",
        reports=bool(raw.get("reports")),
    )


def _path_size(path: Path | None) -> int:
    """Total bytes of a file or directory tree, best effort."""
    if path is None:
        return 0
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                try:
                    if child.is_file():
                        total += int(child.stat().st_size)
                except OSError:
                    continue
            return total
    except OSError:
        return 0
    return 0


def _read_json_file(path: Path) -> dict | None:
    try:
        return _json_payload(Path(path).read_bytes())
    except (OSError, ValueError):
        return None


class HFRunMirror:
    """Background, best-effort mirror of one training run to a HF repo.

    Every upload is wrapped so a Hugging Face failure can only move
    ``hf_sync.json`` to ``error``; it never propagates into training.
    """

    def __init__(
        self,
        config: HFSyncConfig,
        *,
        run_id: str,
        run_dir: Path | str,
        output_dir: Path | str,
        results_dir: Path | str,
        api: Any = None,
        manifest_update: Any = None,
        download: Any = _default_download,
        log: Any = print,
        start_worker: bool = True,
        max_attempts: int = 3,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.run_id = _require_run_id(run_id)
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.api = api if api is not None else make_api()
        self.manifest_update = manifest_update
        self.download = download
        self.log = log
        self.start_worker = bool(start_worker)
        try:
            self.max_attempts = max(1, int(max_attempts))
        except (TypeError, ValueError):
            self.max_attempts = 3
        self.sleep = sleep

        self._condition = threading.Condition()
        self._pending: deque[SyncJob] = deque()
        self._active: SyncJob | None = None
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._started = False
        self._remote_files: list[str] | None = None
        self._last_step = 0
        self._uploaded_checkpoints: set[int] = set()

        self._state: dict[str, Any] = {
            "enabled": True,
            "repo_id": self.config.repo_id,
            "status": "idle",
            "last_success_at": None,
            "last_success_step": None,
            "last_error": None,
            "pending": 0,
            "bytes_uploaded": 0,
        }

    # -- status and persistence ------------------------------------------------

    def status(self) -> dict:
        """Return a snapshot of the local ``hf_sync.json`` state."""
        with self._condition:
            payload = dict(self._state)
            payload["pending"] = len(self._pending) + (
                1 if self._active is not None else 0
            )
            return payload

    def _write_state_locked(self) -> None:
        payload = dict(self._state)
        payload["pending"] = len(self._pending) + (
            1 if self._active is not None else 0
        )
        try:
            _write_json_atomic(self.run_dir / SYNC_FILE, payload)
        except Exception as exc:  # noqa: BLE001 - status is best effort
            self.log(f"[hf-sync] could not write {SYNC_FILE}: {exc}")

    def _record_error(self, message: str) -> None:
        with self._condition:
            self._state["status"] = "error"
            self._state["last_error"] = str(message)
            self._write_state_locked()

    def _record_failure(self, job: SyncJob, exc: BaseException) -> None:
        message = str(exc) or type(exc).__name__
        with self._condition:
            self._state["status"] = "error"
            self._state["last_error"] = message
            self._write_state_locked()
        self.log(f"[hf-sync] upload failed for {job.kind}: {message}")
        self._append_error_event(job, message)

    def _append_error_event(self, job: SyncJob, message: str) -> None:
        payload = {
            "type": "hf_sync",
            "status": "error",
            "kind": job.kind,
            "step": job.step,
            "error": message,
            "at": time.time(),
        }
        try:
            path = self.run_dir / "events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - never fail training
            self.log(f"[hf-sync] could not append the error event: {exc}")

    def _persist_pending_locked(self) -> None:
        jobs: list[dict] = []
        if self._active is not None:
            jobs.append(_job_to_dict(self._active))
        jobs.extend(_job_to_dict(job) for job in self._pending)
        try:
            _write_json_atomic(self.run_dir / PENDING_FILE, {"jobs": jobs})
        except Exception as exc:  # noqa: BLE001 - never fail training
            self.log(f"[hf-sync] could not write {PENDING_FILE}: {exc}")

    def _load_pending(self) -> None:
        payload = _read_json_file(self.run_dir / PENDING_FILE)
        if payload is None:
            return
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
        if not isinstance(raw_jobs, list):
            return
        loaded: list[SyncJob] = []
        for raw in raw_jobs:
            job = _job_from_dict(raw)
            if job is None:
                continue
            if job.kind in ("checkpoint", "adapter", "merged"):
                if job.path is None or not job.path.exists():
                    continue
            loaded.append(job)
        with self._condition:
            for job in loaded:
                self._append_job_locked(job, persist=False)
            self._persist_pending_locked()

    # -- enqueue -------------------------------------------------------------

    def _append_job_locked(self, job: SyncJob, *, persist: bool = True) -> None:
        if job.kind == "adapter" and job.label in ("", "snapshot"):
            self._pending = deque(
                pending
                for pending in self._pending
                if not (
                    pending.kind == "adapter"
                    and pending.label in ("", "snapshot")
                )
            )
        elif job.kind == "latest":
            self._pending = deque(
                pending for pending in self._pending if pending.kind != "latest"
            )
        self._pending.append(job)
        if persist:
            self._persist_pending_locked()
            self._write_state_locked()
        self._condition.notify_all()

    def _enqueue(self, job: SyncJob) -> None:
        try:
            with self._condition:
                self._append_job_locked(job)
        except Exception as exc:  # noqa: BLE001 - enqueue never raises
            self.log(f"[hf-sync] could not enqueue {job.kind}: {exc}")

    def enqueue_checkpoint(self, path: Path | str, step: int) -> None:
        self._enqueue(
            SyncJob("checkpoint", Path(path), _as_int(step))
        )

    def enqueue_adapter_snapshot(self, path: Path | str, step: int) -> None:
        self._enqueue(
            SyncJob("adapter", Path(path), _as_int(step), label="snapshot")
        )

    def enqueue_adapter(
        self, path: Path | str, label: str = "best_adapter"
    ) -> None:
        self._enqueue(SyncJob("adapter", Path(path), label=str(label)))

    def enqueue_ledger(self, status: str = "running", *, reports: bool = False) -> None:
        self._enqueue(
            SyncJob(
                "ledger",
                None,
                label=str(status or "running"),
                reports=bool(reports),
            )
        )

    def enqueue_merged(self, path: Path | str) -> None:
        self._enqueue(SyncJob("merged", Path(path)))

    def enqueue_latest(self) -> None:
        self._enqueue(SyncJob("latest"))

    def _is_pending(self, kind: str, path: Path) -> bool:
        with self._condition:
            if (
                self._active is not None
                and self._active.kind == kind
                and self._active.path == path
            ):
                return True
            return any(
                job.kind == kind and job.path == path for job in self._pending
            )

    # -- start and backfill ---------------------------------------------------

    def start(self) -> None:
        """Prepare the repo and enqueue recovery/backfill work once."""
        with self._condition:
            if self._started:
                return
            self._started = True

        files: list[str] | None = None
        try:
            self._ensure_repo()
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"could not create or verify the repository: {exc}")
            self.log(f"[hf-sync] could not create or verify the repository: {exc}")

        try:
            files = self._list_files(force=True)
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"could not list the repository: {exc}")
            self.log(f"[hf-sync] could not list the repository: {exc}")

        try:
            self._ensure_readme(files)
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"could not create {README_FILE}: {exc}")
            self.log(f"[hf-sync] could not create {README_FILE}: {exc}")

        try:
            self._update_manifest()
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"could not update {MANIFEST_FILE}: {exc}")
            self.log(f"[hf-sync] could not update {MANIFEST_FILE}: {exc}")

        try:
            self._load_pending()
        except Exception as exc:  # noqa: BLE001 - never fail training
            self.log(f"[hf-sync] could not load {PENDING_FILE}: {exc}")

        try:
            self._backfill(files)
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"could not backfill local artifacts: {exc}")
            self.log(f"[hf-sync] could not backfill local artifacts: {exc}")

        with self._condition:
            self._write_state_locked()

        if self.start_worker:
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"hf-sync-{self.run_id}",
                daemon=True,
            )
            with self._condition:
                self._worker = worker
            worker.start()

    def _ensure_repo(self) -> None:
        create = getattr(self.api, "create_repo", None)
        if create is None:
            return
        create(
            self.config.repo_id,
            private=self.config.private,
            exist_ok=True,
        )

    def _ensure_readme(self, files: list[str] | None) -> None:
        if files is None:
            files = self._list_files()
        if README_FILE in files:
            return
        self.api.upload_file(
            path_or_fileobj=README_CONTENT.encode("utf-8"),
            path_in_repo=README_FILE,
            repo_id=self.config.repo_id,
            commit_message="Add the run mirror README",
        )

    def _update_manifest(self) -> None:
        hf = {
            "enabled": True,
            "repo_id": self.config.repo_id,
            "private": self.config.private,
            "adapter_snapshot_steps": self.config.adapter_snapshot_steps,
            "auto_resume": self.config.auto_resume,
            "run_id": self.run_id,
            "layout": 1,
        }

        def mutate(run: dict) -> None:
            run["hf"] = dict(hf)

        if callable(self.manifest_update):
            self.manifest_update(mutate)
            return
        path = self.run_dir / MANIFEST_FILE
        run = _read_json_file(path)
        if run is None:
            return
        mutate(run)
        run["updated_at"] = time.time()
        _write_json_atomic(path, run)

    def _list_files(self, *, force: bool = False) -> list[str]:
        with self._condition:
            cached = self._remote_files
        if cached is not None and not force:
            return list(cached)
        files = _listed_files(self.api, self.config.repo_id)
        with self._condition:
            self._remote_files = list(files)
        return files

    def _remember_remote_paths(self, paths: Iterable[str]) -> None:
        """Add committed paths to the cached listing when we have one."""
        with self._condition:
            if self._remote_files is None:
                return
            merged = set(self._remote_files)
            merged.update(paths)
            self._remote_files = sorted(merged)

    def _remote_high_water(self, files: list[str] | None) -> int | None:
        state_path = f"{repo_run_path(self.run_id)}/{STATE_FILE}"
        if files is None or state_path in files:
            try:
                state = _json_payload(
                    self.download(self.config.repo_id, state_path)
                )
            except Exception:  # noqa: BLE001 - a missing state is not fatal
                state = None
            if state is not None:
                step = _as_int(state.get("last_step"))
                if step is not None:
                    return step
        if files is None:
            return None
        return select_checkpoint_step(files, self.run_id)

    def _backfill(self, files: list[str] | None) -> None:
        remote_step = self._remote_high_water(files)
        for step, path in self._local_checkpoints():
            if remote_step is not None and step <= remote_step:
                continue
            if not self._is_pending("checkpoint", path):
                self.enqueue_checkpoint(path, step)

        best = self.output_dir / "best_adapter"
        if best.is_dir() and validate_adapter(best)[0]:
            if not self._is_pending("adapter", best):
                self.enqueue_adapter(best)

    def _local_checkpoints(self) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        try:
            entries = list(self.output_dir.iterdir())
        except OSError:
            return found
        for path in entries:
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match is not None and is_complete_checkpoint(path):
                found.append((int(match.group(1)), path))
        found.sort(key=lambda item: item[0])
        return found

    def _highest_local_checkpoint(self) -> int | None:
        checkpoints = self._local_checkpoints()
        return checkpoints[-1][0] if checkpoints else None

    # -- repair ---------------------------------------------------------------

    def auto_repair(self) -> bool:
        """Pull a newer remote checkpoint into ``output_dir`` when one exists."""
        try:
            local_step = self._highest_local_checkpoint()
            files = self._list_files(force=True)
            remote_step = select_checkpoint_step(files, self.run_id)
            if remote_step is None:
                return False
            if local_step is not None and remote_step <= local_step:
                return False
            step, _downloaded = _download_checkpoint(
                self.download,
                self.config.repo_id,
                self.run_id,
                files,
                self.output_dir,
            )
            if step is None:
                return False
            if local_step is not None and step <= local_step:
                return False
            self.log(
                f"[hf-sync] repaired the local run with checkpoint-{step}"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - never fail training
            self._record_error(f"auto repair failed: {exc}")
            self.log(f"[hf-sync] auto repair failed: {exc}")
            return False

    # -- worker and draining --------------------------------------------------

    def process_pending(self) -> int:
        """Upload every queued job synchronously; return successes."""
        handled = 0
        while True:
            job = self._take_job()
            if job is None:
                break
            success = self._handle_job(job)
            with self._condition:
                self._active = None
                self._persist_pending_locked()
                if not self._pending and self._state["status"] != "error":
                    self._state["status"] = "idle"
                self._write_state_locked()
                self._condition.notify_all()
            if success:
                handled += 1
        return handled

    def _take_job(self) -> SyncJob | None:
        with self._condition:
            if not self._pending:
                return None
            job = self._pending.popleft()
            self._active = job
            self._state["status"] = "uploading"
            self._persist_pending_locked()
            self._write_state_locked()
            return job

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait()
                if self._stopped and not self._pending:
                    return
                job = self._pending.popleft()
                self._active = job
                self._state["status"] = "uploading"
                self._persist_pending_locked()
                self._write_state_locked()
            self._handle_job(job)
            with self._condition:
                self._active = None
                self._persist_pending_locked()
                if not self._pending and self._state["status"] != "error":
                    self._state["status"] = "idle"
                self._write_state_locked()
                self._condition.notify_all()

    def drain(self, timeout: float = 120.0) -> bool:
        try:
            wait_seconds = max(0.0, float(timeout))
        except (TypeError, ValueError):
            wait_seconds = 120.0
        deadline = time.monotonic() + wait_seconds
        with self._condition:
            while self._pending or self._active is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout: float = 120.0) -> None:
        with self._condition:
            worker = self._worker
        if worker is not None and worker.is_alive():
            self.drain(timeout)
        else:
            self.process_pending()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if worker is not None and worker.is_alive():
            try:
                worker.join(timeout=min(5.0, max(0.0, float(timeout))))
            except (TypeError, ValueError):
                worker.join(timeout=5.0)

    # -- uploads --------------------------------------------------------------

    def _handle_job(self, job: SyncJob) -> bool:
        for attempt in range(self.max_attempts):
            try:
                step, uploaded_bytes = self._upload_job(job)
                self._record_success(job, step, uploaded_bytes)
                return True
            except _DropJob as exc:
                if exc.record_error:
                    self._record_error(str(exc))
                self.log(f"[hf-sync] dropped {job.kind} job: {exc}")
                return False
            except Exception as exc:  # noqa: BLE001 - retry, then record
                if attempt + 1 < self.max_attempts:
                    self.log(f"[hf-sync] upload failed ({exc}), retrying")
                    try:
                        self.sleep(2**attempt)
                    except Exception:  # noqa: BLE001 - a bad sleep hook is harmless
                        pass
                    continue
                self._record_failure(job, exc)
                return False
        return False

    def _record_success(
        self, job: SyncJob, step: int | None, uploaded_bytes: int
    ) -> None:
        if job.kind == "checkpoint" and step is not None:
            self._uploaded_checkpoints.add(int(step))
        self._write_local_hf_state(step)
        latest_bytes = self._upload_latest()
        with self._condition:
            self._state["last_success_at"] = time.time()
            if step is not None:
                self._state["last_success_step"] = int(step)
            self._state["last_error"] = None
            self._state["bytes_uploaded"] = (
                _as_int(self._state.get("bytes_uploaded")) or 0
            ) + int(uploaded_bytes) + int(latest_bytes)
            self._state["status"] = (
                "uploading" if self._pending or self._active is not None else "idle"
            )
            self._write_state_locked()

    def _upload_job(self, job: SyncJob) -> tuple[int | None, int]:
        kind = job.kind
        if kind == "checkpoint":
            return self._upload_checkpoint(job)
        if kind == "adapter":
            return self._upload_adapter(job)
        if kind == "ledger":
            step, size = self._commit_ledger(
                status=job.label or "running",
                step=job.step,
                reports=job.reports,
            )
            return step, size
        if kind == "merged":
            return self._upload_merged(job)
        if kind == "latest":
            return None, self._upload_latest()
        raise _DropJob(f"unknown job kind: {kind!r}")

    def _upload_checkpoint(self, job: SyncJob) -> tuple[int | None, int]:
        path = job.path
        if path is None or not is_complete_checkpoint(path):
            raise _DropJob(f"incomplete checkpoint: {path}")
        step = job.step
        if step is None:
            match = CHECKPOINT_RE.fullmatch(path.name)
            step = int(match.group(1)) if match is not None else None
        if step is None:
            raise _DropJob(f"checkpoint has no step: {path}")
        size = _path_size(path)
        self._upload_folder(
            path,
            repo_checkpoint_path(self.run_id, step),
            f"Upload checkpoint-{step}",
        )
        _ledger_step, ledger_size = self._commit_ledger(
            status=self._manifest_status(),
            step=step,
            reports=False,
            checkpoint_step=step,
        )
        return step, size + ledger_size

    def _upload_adapter(self, job: SyncJob) -> tuple[int | None, int]:
        path = job.path
        if path is None:
            raise _DropJob("adapter job has no path")
        ok, reason = validate_adapter(path)
        if not ok:
            raise _DropJob(f"invalid adapter {path}: {reason}")
        label = job.label or "snapshot"
        run_prefix = repo_run_path(self.run_id)
        snapshot_step: int | None = None
        if label == "best_adapter" or path.name == "best_adapter":
            path_in_repo = f"{run_prefix}/best_adapter"
        else:
            match = _ADAPTER_SNAPSHOT_RE.fullmatch(path.name)
            if match is None:
                raise _DropJob(
                    f"adapter snapshot has an invalid folder name: {path.name}"
                )
            snapshot_step = int(match.group(1))
            path_in_repo = f"{run_prefix}/adapter_snapshots/{path.name}"
        size = _path_size(path)
        self._upload_folder(
            path,
            path_in_repo,
            f"Upload {path.name}",
        )
        if snapshot_step is not None:
            self._retain_adapter_snapshots(snapshot_step)
        return job.step, size

    def _upload_merged(self, job: SyncJob) -> tuple[int | None, int]:
        path = job.path
        if path is None:
            raise _DropJob("merged job has no path")
        ok, reason = validate_merged(path)
        if not ok:
            raise _DropJob(f"invalid merged model {path}: {reason}")
        size = _path_size(path)
        self._upload_folder(
            path,
            f"{repo_run_path(self.run_id)}/merged",
            "Upload the merged model",
        )
        return None, size

    def _upload_folder(self, path: Path, path_in_repo: str, message: str) -> None:
        self.api.upload_folder(
            folder_path=str(path),
            path_in_repo=path_in_repo,
            repo_id=self.config.repo_id,
            commit_message=message,
        )
        committed: list[str] = []
        try:
            for child in path.rglob("*"):
                if child.is_file():
                    relative = child.relative_to(path).as_posix()
                    committed.append(f"{path_in_repo}/{relative}")
        except OSError:
            pass
        if committed:
            self._remember_remote_paths(committed)

    def _retain_adapter_snapshots(self, newest_step: int) -> None:
        """Keep the newest two snapshots; delete only superseded ones."""
        root = self.output_dir / "adapter_snapshots"
        try:
            entries = list(root.iterdir())
        except OSError:
            return
        snapshots: list[tuple[int, Path]] = []
        for path in entries:
            match = _ADAPTER_SNAPSHOT_RE.fullmatch(path.name)
            if match is not None and path.is_dir():
                snapshots.append((int(match.group(1)), path))
        snapshots.sort(key=lambda item: item[0], reverse=True)
        for step, path in snapshots[2:]:
            if step >= newest_step:
                # Never remove an unsucceeded snapshot that is newer than the
                # one whose upload just succeeded.
                continue
            try:
                shutil.rmtree(path)
                self.log(f"[hf-sync] removed old adapter snapshot {path.name}")
            except OSError as exc:
                self.log(f"[hf-sync] could not remove {path}: {exc}")

    # -- ledger and latest ----------------------------------------------------

    def _manifest(self) -> dict:
        return _read_json_file(self.run_dir / MANIFEST_FILE) or {}

    def _manifest_status(self) -> str:
        return str(self._manifest().get("status") or "running")

    def _local_hf_state(self) -> dict:
        return _read_json_file(self.run_dir / STATE_FILE) or {}

    def _find_ledger_file(self, name: str) -> Path | None:
        for base in (self.run_dir, self.results_dir):
            path = base / name
            if path.is_file():
                return path
        return None

    def _report_files(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for base in (self.run_dir, self.output_dir):
            for name in REPORT_FILES:
                path = base / name
                if path.is_file():
                    found.setdefault(name, path)
            reports_dir = base / "reports"
            if reports_dir.is_dir():
                for path in sorted(reports_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    try:
                        relative = path.relative_to(reports_dir).as_posix()
                    except ValueError:
                        continue
                    found[relative] = path
        return found

    def _ledger_files(self, *, reports: bool) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for name in LEDGER_ROOT_FILES:
            path = self._find_ledger_file(name)
            if path is not None:
                files[name] = path
        preflight = self._find_ledger_file(LEDGER_PREFLIGHT_FILE)
        if preflight is not None:
            files[LEDGER_PREFLIGHT_FILE] = preflight
        log_file = self.run_dir / TRAINING_LOG_FILE
        if not log_file.is_file():
            log_file = self.results_dir / TRAINING_LOG_FILE
        try:
            if log_file.is_file() and log_file.stat().st_size < MAX_TRAINING_LOG_BYTES:
                files[TRAINING_LOG_FILE] = log_file
        except OSError:
            pass
        if reports:
            for relative, path in self._report_files().items():
                files[f"reports/{relative}"] = path
        return files

    def _write_local_hf_state(self, step: int | None) -> None:
        manifest = self._manifest()
        current = self._local_hf_state()
        checkpoints = {
            int(value)
            for value in current.get("checkpoints") or []
            if _as_int(value) is not None
        }
        checkpoints.update(self._uploaded_checkpoints)
        last_step = max(
            int(step or 0),
            _as_int(manifest.get("last_step")) or 0,
            _as_int(current.get("last_step")) or 0,
            self._last_step,
        )
        payload = {
            "last_step": last_step,
            "updated_at": time.time(),
            "checkpoints": sorted(checkpoints),
            "status": str(manifest.get("status") or "running"),
        }
        _write_json_atomic(self.run_dir / STATE_FILE, payload)
        self._last_step = last_step

    def _remote_ledger_step(self) -> int | None:
        state_path = f"{repo_run_path(self.run_id)}/{STATE_FILE}"
        try:
            state = _json_payload(self.download(self.config.repo_id, state_path))
        except Exception:  # noqa: BLE001 - verify presence before falling back
            state = None
        if state is not None:
            step = _as_int(state.get("last_step"))
            if step is not None:
                return step

        # A state file that exists but could not be read must not be replaced
        # by the lower checkpoint listing. The listing is refreshed so commits
        # made by this process are visible to the next conflict check.
        try:
            files = self._list_files(force=True)
        except Exception as exc:  # noqa: BLE001 - refuse an unverifiable ledger
            raise _DropJob(
                f"cannot verify the remote high-water mark: {exc}",
                record_error=True,
            ) from exc
        if state_path in files:
            raise _DropJob(
                "cannot verify the remote high-water mark: "
                f"{state_path} exists but could not be read",
                record_error=True,
            )
        return select_checkpoint_step(files, self.run_id)

    def _commit_ledger(
        self,
        *,
        status: str,
        step: int | None,
        reports: bool,
        checkpoint_step: int | None = None,
    ) -> tuple[int, int]:
        manifest = self._manifest()
        current = self._local_hf_state()
        local_step = max(
            int(step or 0),
            _as_int(manifest.get("last_step")) or 0,
            _as_int(current.get("last_step")) or 0,
            self._last_step,
        )
        remote_step = self._remote_ledger_step()
        if remote_step is not None and remote_step > local_step:
            raise _DropJob(
                "refusing to upload ledger at last_step "
                f"{local_step}: remote high-water mark is {remote_step}",
                record_error=True,
            )

        checkpoints = {
            int(value)
            for value in current.get("checkpoints") or []
            if _as_int(value) is not None
        }
        checkpoints.update(self._uploaded_checkpoints)
        if checkpoint_step is not None:
            checkpoints.add(int(checkpoint_step))
        self._write_hf_state(local_step, checkpoints, status)

        files = self._ledger_files(reports=reports)
        state_path = self.run_dir / STATE_FILE
        files[STATE_FILE] = state_path

        try:
            from huggingface_hub import CommitOperationAdd  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - treated as an upload failure
            raise HFSyncError(
                "huggingface_hub is required to upload ledger snapshots"
            ) from exc

        prefix = repo_run_path(self.run_id)
        operations = []
        for relative, path in files.items():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/{relative}",
                    path_or_fileobj=str(path),
                )
            )
        self.api.create_commit(
            repo_id=self.config.repo_id,
            operations=operations,
            commit_message=f"Sync {self.run_id} ledger ({status})",
        )
        self._remember_remote_paths(
            f"{prefix}/{relative}" for relative in files
        )
        return local_step, sum(_path_size(path) for path in files.values())

    def _write_hf_state(
        self, last_step: int, checkpoints: set[int], status: str
    ) -> None:
        payload = {
            "last_step": int(last_step),
            "updated_at": time.time(),
            "checkpoints": sorted(int(value) for value in checkpoints),
            "status": str(status or "running"),
        }
        _write_json_atomic(self.run_dir / STATE_FILE, payload)
        self._last_step = max(self._last_step, int(last_step))

    def _upload_latest(self) -> int:
        manifest = self._manifest()
        payload = {
            "run_id": self.run_id,
            "status": str(manifest.get("status") or "running"),
            "last_step": self._last_step,
            "updated_at": time.time(),
            "layout": 1,
        }
        data = (
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.api.upload_file(
            path_or_fileobj=data,
            path_in_repo=LATEST_FILE,
            repo_id=self.config.repo_id,
            commit_message=f"Update {LATEST_FILE} for {self.run_id}",
        )
        return len(data)
