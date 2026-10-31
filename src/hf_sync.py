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
import time
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


def _prepare_run_dir(runs_root: Path, run_id: str, *, force: bool) -> Path:
    """Create the local run folder, refusing to overwrite unless forced."""
    run_dir = Path(runs_root) / run_id
    if run_dir.exists():
        is_dir = run_dir.is_dir()
        nonempty = not is_dir or any(run_dir.iterdir())
        if nonempty and not force:
            raise HFRestoreConflict(
                f"local run directory already exists: {run_dir}; "
                "pass --force to replace it"
            )
        if nonempty:
            if is_dir:
                shutil.rmtree(run_dir)
            else:
                run_dir.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(exist_ok=True)
    return run_dir


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

    A listing-complete checkpoint can still fail to materialize (network error
    or a truncated remote folder). Each attempt lands in a hidden temporary
    directory and is only moved into ``output/checkpoint-<step>`` after
    ``is_complete_checkpoint`` accepts it, so a failed attempt never leaves a
    half checkpoint behind.
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
        except Exception:
            # Fall through to the next lower complete checkpoint.
            pass
        finally:
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
    snapshots: dict[int, list[str]] = {}
    for remote in files:
        if not remote.startswith(prefix):
            continue
        relative = remote[len(prefix) :]
        folder, separator, remainder = relative.partition("/")
        if not separator or not remainder:
            continue
        match = _ADAPTER_SNAPSHOT_RE.fullmatch(folder)
        if match is not None:
            snapshots.setdefault(int(match.group(1)), []).append(remote)

    if not snapshots:
        return []
    newest = max(snapshots)
    snapshot_prefix = f"{prefix}step-{newest}/"
    return _download_tree(
        download,
        repo_id,
        snapshots[newest],
        snapshot_prefix,
        output_dir / "adapter_snapshots" / f"step-{newest}",
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

    status("listing", f"Listing {repo_id} for run {run_id}")
    files = _listed_files(api, repo_id)
    file_set = set(files)

    manifest_path = f"{run_prefix}/{MANIFEST_FILE}"
    if manifest_path not in file_set:
        raise HFRestoreError(
            f"run {run_id} was not found in {repo_id}; "
            f"expected {manifest_path}"
        )

    # Only touch the local filesystem after confirming the remote run exists,
    # so a bad id never deletes an existing local run or leaves an empty one.
    run_dir = _prepare_run_dir(Path(runs_root), run_id, force=force)
    output_dir = run_dir / "output"

    status("ledger", f"Downloading the run ledger for {run_id}")
    manifest_payload: dict | None = None
    downloaded: list[str] = []
    for relative in RESTORE_LEDGER_FILES:
        remote_path = f"{run_prefix}/{relative}"
        if remote_path not in file_set:
            continue
        data = _fetch(download, repo_id, remote_path)
        _write_bytes_atomic(run_dir / relative, data)
        downloaded.append(remote_path)
        if relative == MANIFEST_FILE:
            manifest_payload = _json_payload(data)
    if manifest_payload is None:
        raise HFRestoreError(
            f"the manifest for run {run_id} in {repo_id} is not a JSON object"
        )

    status("checkpoint", f"Restoring the newest complete checkpoint for {run_id}")
    checkpoint_step, checkpoint_files = _download_checkpoint(
        download, repo_id, run_id, files, output_dir
    )
    downloaded.extend(checkpoint_files)

    downloaded.extend(
        _download_best_adapter(download, repo_id, files, run_prefix, output_dir)
    )
    downloaded.extend(
        _download_reports(
            download, repo_id, files, run_prefix, run_dir, output_dir
        )
    )
    downloaded.extend(
        _download_newest_adapter_snapshot(
            download, repo_id, files, run_prefix, output_dir
        )
    )
    if include_merged:
        downloaded.extend(
            _download_merged(download, repo_id, files, run_prefix, run_dir)
        )

    run = manifest_payload
    rewrite_run_paths(run, run_dir)
    normalize_restored_run(run)
    run["run_id"] = run_id
    run["updated_at"] = time.time()

    try:
        (run_dir / "training.pid").unlink(missing_ok=True)
    except OSError:
        pass
    _write_json_atomic(run_dir / MANIFEST_FILE, run)

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
        "output_dir": str(output_dir),
        "checkpoint_step": checkpoint_step,
        "downloaded": downloaded,
        "run": run,
    }
