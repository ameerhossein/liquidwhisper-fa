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
import tempfile
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
