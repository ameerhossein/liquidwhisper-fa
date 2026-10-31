"""Persistent run ledger for the training cockpit and CLI runs.

One directory per run, rooted at the managed runs directory:

    <runs_root>/<run_id>/run.json      atomic manifest
    <runs_root>/<run_id>/steps.jsonl   append-only per-step records
    <runs_root>/<run_id>/events.jsonl  append-only stage/status events
    <runs_root>/<run_id>/output/       checkpoints and adapter (created here)
    <runs_root>/<run_id>/merged/       merge output (created by the merge step)

The module is intentionally stdlib-only: the FastAPI server, the trainer, and
tests on a CPU-only machine can all share it without importing torch.

`run.json` reads and writes are safe across processes.  Every mutation takes an
exclusive ``fcntl.flock`` on ``<run_dir>/.run.lock``, applies the mutator to a
freshly read manifest, and replaces the manifest atomically.  On platforms
without ``fcntl`` the lock becomes a no-op.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # POSIX; the fallback keeps the module importable on non-POSIX hosts.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

try:
    from .paths import PERSISTENT_ROOTS as _PERSISTENT_ROOTS
except Exception:  # pragma: no cover - direct module import fallback
    _PERSISTENT_ROOTS = ("workspace", "runpod-volume")


STAGES = ("preflight", "training", "post_eval", "merge")
STAGE_STATUSES = (
    "pending",
    "running",
    "completed",
    "stopped",
    "failed",
    "skipped",
    "interrupted",
)
RUN_STATUSES = (
    "starting",
    "running",
    "completed",
    "stopped",
    "failed",
    "interrupted",
)

_TERMINAL_RUN_STATUSES = frozenset(
    ("completed", "stopped", "failed", "interrupted")
)
_TERMINAL_STAGE_STATUSES = frozenset(
    ("completed", "stopped", "failed", "skipped", "interrupted")
)
_ACTIVE_RUN_STATUSES = frozenset(("starting", "running"))

_RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_STARTING_GRACE_SECONDS = 15.0
_UNSET = object()


# ---------------------------------------------------------------------------
# Run ids
# ---------------------------------------------------------------------------


def new_run_id(now: float | None = None) -> str:
    """Return a path-safe, chronologically sortable run id.

    Format: ``YYYYMMDD-HHMMSS-<6 hex chars>``.  UTC keeps ids stable across
    machines and daylight-saving changes.
    """
    timestamp = time.time() if now is None else float(now)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(timestamp))
    return f"{stamp}-{secrets.token_hex(3)}"


def run_id_valid(run_id: str) -> bool:
    """True when *run_id* matches the only safe run-directory name shape."""
    return isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id) is not None


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------


def _manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run.json"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_file_pid(run_dir: str | Path) -> int | None:
    """Read ``<run_dir>/training.pid`` when a manifest has no ``current_pid``."""
    try:
        text = (Path(run_dir) / "training.pid").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return None
    return _as_pid(text)


def _run_training_pid(run_dir: str | Path, run: dict) -> int | None:
    """The PID owning *run*: manifest ``current_pid`` or the run PID file."""
    pid = _as_pid(run.get("current_pid"))
    if pid is not None:
        return pid
    return _pid_file_pid(run_dir)


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write *payload* to *path* via a sibling temp file and ``os.replace``."""
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
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict | None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


@contextmanager
def _locked(run_dir: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``<run_dir>/.run.lock`` for one mutation."""
    lock_path = Path(run_dir) / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError:  # pragma: no cover - filesystems without flock
            locked = False
    try:
        yield
    finally:
        if fcntl is not None and locked:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:  # pragma: no cover
                pass
        handle.close()


def _append_json_line(path: Path, payload: dict, *, fsync: bool) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Manifest read/write
# ---------------------------------------------------------------------------


def create_run(
    runs_root: str | Path,
    *,
    config: dict,
    trainer_args: dict,
    paths: dict | None = None,
    source: str = "cockpit",
    run_id: str | None = None,
) -> dict:
    """Create a new run directory and its initial manifest."""
    root = Path(runs_root)
    if run_id is None:
        run_id = new_run_id()
    elif not run_id_valid(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")

    run_dir = root / run_id
    manifest = _manifest_path(run_dir)
    if manifest.exists():
        raise FileExistsError(f"run already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(exist_ok=True)
    (run_dir / "baselines").mkdir(exist_ok=True)

    now = time.time()
    path_values: dict[str, str] = {
        "run_dir": str(run_dir),
        "output_dir": str(run_dir / "output"),
        "results_dir": str(run_dir),
        "merged_dir": str(run_dir / "merged"),
        "preflight_cache": str(root / "_shared" / "baselines" / "preflight.json"),
    }
    for key, value in (paths or {}).items():
        path_values[key] = str(value)

    run = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now,
        "updated_at": now,
        "status": "starting",
        "current_stage": None,
        "failed_stage": None,
        "ended_at": None,
        "config": dict(config or {}),
        "trainer_args": dict(trainer_args or {}),
        "paths": path_values,
        "stages": [{"name": name, "status": "pending"} for name in STAGES],
        "attempts": [],
        "current_pid": None,
        "merge_pid": None,
        "step_count": 0,
        "last_step": None,
        "last_step_at": None,
        "last_error": None,
        "stop_requested_at": None,
        "source": source,
    }
    _write_json_atomic(manifest, run)
    return run


def load_run(run_dir: str | Path) -> dict | None:
    """Load one run manifest, or ``None`` when it is missing/unreadable."""
    return _read_json(_manifest_path(run_dir))


def _load_run_entries(runs_root: str | Path) -> list[tuple[Path, dict]]:
    """Return ``(scanned_directory, manifest)`` pairs newest first."""
    root = Path(runs_root)
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name, reverse=True)
    except OSError:
        return []

    runs: list[tuple[Path, dict]] = []
    for entry in entries:
        if not entry.is_dir() or not run_id_valid(entry.name):
            continue
        run = load_run(entry)
        if run is not None:
            runs.append((entry, run))
    runs.sort(
        key=lambda pair: (
            _as_float(pair[1].get("created_at")),
            str(pair[1].get("run_id") or ""),
        ),
        reverse=True,
    )
    return runs


def _load_all_runs(runs_root: str | Path) -> list[dict]:
    return [run for _run_dir, run in _load_run_entries(runs_root)]


def list_runs(runs_root: str | Path, limit: int = 50) -> list[dict]:
    """Return run manifests newest first, capped at *limit*."""
    runs = _load_all_runs(runs_root)
    if limit is None:
        return runs
    try:
        capped = max(0, int(limit))
    except (TypeError, ValueError):
        capped = 50
    return runs[:capped]


def update_run(
    run_dir: str | Path,
    mutator: Callable[[dict], dict | None],
) -> dict:
    """Locked read-modify-write of one ``run.json``.

    *mutator* receives the freshly read manifest. It may mutate it in place and
    return ``None`` (or the same dict), or return a replacement dict.
    """
    run_dir = Path(run_dir)
    with _locked(run_dir):
        run = load_run(run_dir)
        if run is None:
            raise FileNotFoundError(f"run manifest not found: {run_dir}")
        updated = mutator(run)
        result = updated if isinstance(updated, dict) else run
        result["updated_at"] = time.time()
        _write_json_atomic(_manifest_path(run_dir), result)
        return result


# ---------------------------------------------------------------------------
# Status, stages, progress
# ---------------------------------------------------------------------------


def set_status(run_dir: str | Path, status: str, **fields: Any) -> dict:
    """Set the overall run status, timestamps, and optional top-level fields."""
    if status not in RUN_STATUSES:
        raise ValueError(f"invalid run status: {status!r}")

    def mutate(run: dict) -> dict:
        for key, value in fields.items():
            run[key] = value
        run["status"] = status
        if status in _TERMINAL_RUN_STATUSES:
            run["ended_at"] = time.time()
        else:
            run["ended_at"] = None
            if "failed_stage" not in fields:
                run["failed_stage"] = None
        return run

    run = update_run(run_dir, mutate)
    event: dict[str, Any] = {"type": "status", "status": status}
    event.update(fields)
    append_event(run_dir, event)
    return run


def set_stage(
    run_dir: str | Path,
    name: str,
    status: str,
    detail: Any = _UNSET,
    **fields: Any,
) -> dict:
    """Update one pipeline stage and append a ``stage`` event.

    Omit *detail* to leave the existing detail untouched, or pass ``None`` to
    clear stale failure text when a resumed stage finishes successfully.
    """
    if name not in STAGES:
        raise ValueError(f"invalid stage name: {name!r}")
    if status not in STAGE_STATUSES:
        raise ValueError(f"invalid stage status: {status!r}")

    def mutate(run: dict) -> dict:
        stages = run.setdefault("stages", [])
        stage = next((item for item in stages if item.get("name") == name), None)
        if stage is None:
            stage = {"name": name}
            stages.append(stage)

        stage["status"] = status
        if detail is not _UNSET:
            if detail is None:
                stage.pop("detail", None)
            else:
                stage["detail"] = detail
        for key, value in fields.items():
            stage[key] = value

        if status == "running":
            stage["started_at"] = time.time()
            stage["ended_at"] = None
        elif status in _TERMINAL_STAGE_STATUSES:
            stage["ended_at"] = time.time()
        else:  # pending
            stage["started_at"] = None
            stage["ended_at"] = None

        run["current_stage"] = name
        return run

    run = update_run(run_dir, mutate)
    event: dict[str, Any] = {"type": "stage", "stage": name, "status": status}
    if detail is not _UNSET and detail is not None:
        event["detail"] = detail
    event.update(fields)
    append_event(run_dir, event)
    return run


def set_progress(
    run_dir: str | Path,
    *,
    step: int,
    step_count: int,
    last_step_at: float,
) -> dict:
    """Store the latest saved step and unique-step count in the manifest."""

    def mutate(run: dict) -> dict:
        run["step_count"] = int(step_count)
        run["last_step"] = int(step)
        run["last_step_at"] = last_step_at
        return run

    return update_run(run_dir, mutate)


# ---------------------------------------------------------------------------
# Append-only logs
# ---------------------------------------------------------------------------


def append_event(run_dir: str | Path, event: dict) -> None:
    """Append one JSON event line, flushed and fsynced."""
    payload = dict(event)
    payload.setdefault("ts", time.time())
    _append_json_line(Path(run_dir) / "events.jsonl", payload, fsync=True)


def append_step(run_dir: str | Path, record: dict) -> None:
    """Append one JSON step line, flushed but not fsynced (hot path)."""
    payload = dict(record)
    payload.setdefault("ts", time.time())
    _append_json_line(Path(run_dir) / "steps.jsonl", payload, fsync=False)


def _read_step_records(run_dir: str | Path) -> dict[int, dict]:
    path = Path(run_dir) / "steps.jsonl"
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return {}

    records: dict[int, dict] = {}
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            step = record.get("step")
            if isinstance(step, bool):
                continue
            try:
                key = int(step)
            except (TypeError, ValueError):
                continue
            records[key] = record
    return records


def read_steps(run_dir: str | Path, tail: int = 2000) -> list[dict]:
    """Return deduplicated step records, sorted by step, as the last *tail*."""
    records = _read_step_records(run_dir)
    ordered = [records[key] for key in sorted(records)]
    try:
        count = int(tail)
    except (TypeError, ValueError):
        count = 2000
    if count <= 0:
        return []
    return ordered[-count:]


def step_count(run_dir: str | Path) -> int:
    """Number of unique optimizer steps recorded in ``steps.jsonl``."""
    return len(_read_step_records(run_dir))


# ---------------------------------------------------------------------------
# Attempts and reconciliation
# ---------------------------------------------------------------------------


def _finish_open_attempt(run: dict, exit_code: int | None, reason: str) -> None:
    now = time.time()
    attempts = run.get("attempts") or []
    for attempt in reversed(attempts):
        if attempt.get("ended_at") is None:
            attempt["ended_at"] = now
            attempt["exit_code"] = exit_code
            attempt["reason"] = reason
            break


def _attempt_reason(
    run: dict,
    exit_code: int | None,
    requested: str | None = None,
) -> str:
    """Classify why an attempt ended.

    A user halt always wins. Otherwise a known exit code is authoritative;
    *requested* only labels an attempt with no observable exit (for example a
    process-lost scan or a caller-supplied classification).
    """
    if run.get("stop_requested_at") or run.get("status") == "stopped":
        return "halted"
    if requested == "halted":
        return "halted"
    if exit_code == 0:
        return "completed"
    if exit_code is not None:
        return "nonzero_exit"
    return requested or "process_lost"


def start_attempt(
    run_dir: str | Path,
    pid: int,
    reason: str = "initial",
) -> dict:
    """Append a new attempt and expose its PID at the run top level."""

    def mutate(run: dict) -> dict:
        attempts = run.setdefault("attempts", [])
        number = 1
        if attempts:
            try:
                number = int(attempts[-1].get("attempt") or 0) + 1
            except (TypeError, ValueError):
                number = len(attempts) + 1
        attempts.append(
            {
                "attempt": number,
                "pid": int(pid),
                "started_at": time.time(),
                "ended_at": None,
                "exit_code": None,
                "reason": reason,
            }
        )
        run["current_pid"] = int(pid)
        return run

    return update_run(run_dir, mutate)


def finish_attempt(
    run_dir: str | Path,
    exit_code: int | None,
    reason: str,
) -> dict:
    """Close the open attempt and clear the run's current PID."""

    def mutate(run: dict) -> dict:
        _finish_open_attempt(run, exit_code, reason)
        run["current_pid"] = None
        return run

    return update_run(run_dir, mutate)


def finalize_attempt(
    run_dir: str | Path,
    *,
    exit_code: int | None = None,
    reason: str | None = None,
) -> dict:
    """Close the open attempt without changing a terminal run status.

    The server calls this after observing a child exit even when the trainer
    already wrote ``completed``/``stopped``/``failed``.  A requested halt always
    wins; otherwise the exit code maps to ``completed``/``nonzero_exit`` and an
    unobserved exit becomes ``process_lost``.
    """

    def mutate(run: dict) -> dict:
        final_reason = _attempt_reason(run, exit_code, reason)
        _finish_open_attempt(run, exit_code, final_reason)
        run["current_pid"] = None
        return run

    return update_run(run_dir, mutate)


def _find_stage(stages: list[dict], name: str) -> dict | None:
    return next((stage for stage in stages if stage.get("name") == name), None)


def _last_running_stage(stages: list[dict]) -> tuple[int | None, dict | None]:
    for index in range(len(stages) - 1, -1, -1):
        if stages[index].get("status") == "running":
            return index, stages[index]
    return None, None


def _mark_stage(stages: list[dict], name: str, status: str, now: float) -> None:
    stage = _find_stage(stages, name)
    if stage is None:
        return
    if stage.get("status") not in ("pending", "running"):
        return
    stage["status"] = status
    if status in _TERMINAL_STAGE_STATUSES:
        stage["ended_at"] = now


def _skip_stages_after(stages: list[dict], index: int | None, now: float) -> None:
    if index is None:
        return
    for stage in stages[index + 1 :]:
        if stage.get("status") == "pending":
            stage["status"] = "skipped"
            stage["ended_at"] = now


def _within_starting_grace(run: dict, now: float) -> bool:
    return (
        run.get("status") == "starting"
        and not (run.get("attempts") or [])
        and now - _as_float(run.get("created_at")) < _STARTING_GRACE_SECONDS
    )


def _reconcile_run(
    run: dict,
    *,
    exit_code: int | None,
    reason: str | None,
    now: float,
) -> dict:
    status = run.get("status")
    if status not in _ACTIVE_RUN_STATUSES or _within_starting_grace(run, now):
        return run

    stages = run.get("stages") or []
    running_index, running_stage = _last_running_stage(stages)
    training = _find_stage(stages, "training")
    post_eval = _find_stage(stages, "post_eval")
    training_status = training.get("status") if training else "pending"
    post_eval_status = post_eval.get("status") if post_eval else "pending"

    failed_stage: str | None = None
    stop_requested = bool(run.get("stop_requested_at"))
    if stop_requested:
        new_status = "stopped"
        if running_stage is not None:
            running_stage["status"] = "stopped"
            running_stage["ended_at"] = now
        skip_from = running_index
        if skip_from is None and run.get("current_stage") in STAGES:
            skip_from = STAGES.index(run["current_stage"])
        _skip_stages_after(stages, skip_from, now)
    elif exit_code == 0:
        if training_status == "completed" and post_eval_status in (
            "completed",
            "skipped",
        ):
            new_status = "completed"
        else:
            new_status = "failed"
            failed_stage = "post_eval" if training_status == "completed" else "training"
            _mark_stage(stages, failed_stage, "failed", now)
    elif exit_code is not None:
        new_status = "failed"
        if running_stage is not None:
            failed_stage = str(running_stage.get("name") or "training")
        else:
            failed_stage = "training"
        _mark_stage(stages, failed_stage, "failed", now)
    else:
        new_status = "interrupted"
        if running_stage is not None:
            running_stage["status"] = "interrupted"
            running_stage["ended_at"] = now

    run["status"] = new_status
    run["failed_stage"] = failed_stage
    run["ended_at"] = now
    run["current_pid"] = None
    final_reason = _attempt_reason(run, exit_code, reason)
    _finish_open_attempt(run, exit_code, final_reason)
    return run


def reconcile(
    run_dir: str | Path,
    *,
    pid_alive: bool,
    exit_code: int | None = None,
    reason: str | None = None,
) -> dict:
    """Terminalize a dead ``starting``/``running`` run.

    A fresh ``starting`` run with no attempts is left alone for 15 seconds so
    the server monitor cannot race a launch that is still in progress.
    """
    run = load_run(run_dir)
    if run is None:
        raise FileNotFoundError(f"run manifest not found: {run_dir}")
    if pid_alive or run.get("status") not in _ACTIVE_RUN_STATUSES:
        return run
    if _within_starting_grace(run, time.time()):
        return run

    return update_run(
        run_dir,
        lambda current: _reconcile_run(
            current,
            exit_code=exit_code,
            reason=reason,
            now=time.time(),
        ),
    )


def reconcile_all(
    runs_root: str | Path,
    *,
    is_alive: Callable[[int], bool],
) -> dict:
    """Scan every run for live trainer/merge PIDs and reconcile dead runs."""
    entries = _load_run_entries(runs_root)
    alive_cache: dict[int, bool] = {}

    def alive(pid: int) -> bool:
        if pid not in alive_cache:
            try:
                alive_cache[pid] = bool(is_alive(pid))
            except Exception:
                # An unavailable process probe is unknown, not proof of death.
                # Treat the PID as alive so a probe failure never terminalizes
                # a run whose process may still be working.
                alive_cache[pid] = True
        return alive_cache[pid]

    active_index: int | None = None
    merge_index: int | None = None
    live_indices: list[int] = []
    for index, (run_dir, run) in enumerate(entries):
        # Only a starting/running run can be active. A terminal run may keep a
        # stale current_pid; that PID can be reused by an unrelated process.
        if run.get("status") in _ACTIVE_RUN_STATUSES:
            pid = _run_training_pid(run_dir, run)
            if pid is not None and alive(pid):
                live_indices.append(index)
                if active_index is None:
                    active_index = index
        merge_pid = _as_pid(run.get("merge_pid"))
        if merge_index is None and merge_pid is not None and alive(merge_pid):
            merge_index = index

    refreshed: list[dict] = []
    for run_dir, run in entries:
        if run.get("status") in _ACTIVE_RUN_STATUSES:
            pid = _run_training_pid(run_dir, run)
            if pid is None or not alive(pid):
                try:
                    run = reconcile(run_dir, pid_alive=False, exit_code=None)
                except FileNotFoundError:
                    pass
        refreshed.append(run)

    latest_run = refreshed[0] if refreshed else None
    duplicates = [
        str(entries[index][1].get("run_id") or "")
        for index in live_indices
        if index != active_index
    ]
    duplicates = [run_id for run_id in duplicates if run_id]
    return {
        "active_run": refreshed[active_index] if active_index is not None else None,
        "latest_run": latest_run,
        "merge_run": refreshed[merge_index] if merge_index is not None else None,
        "runs": refreshed,
        "duplicates": duplicates,
    }


# ---------------------------------------------------------------------------
# Storage and provenance helpers
# ---------------------------------------------------------------------------


def storage_info(runs_root: str | Path) -> dict:
    """Report the runs root and whether it lives on a persistent volume."""
    raw = str(runs_root)
    persistent = False
    try:
        resolved = Path(runs_root).expanduser().resolve()
        parts = resolved.parts
        persistent = len(parts) > 1 and parts[1] in _PERSISTENT_ROOTS
    except Exception:  # pragma: no cover - defensive
        persistent = False
    return {"runs_dir": raw, "persistent": bool(persistent)}


def write_preflight_copy(
    run_dir: str | Path,
    source_path: str | Path,
) -> Path:
    """Atomically copy a shared ``preflight.json`` into the run's baselines."""
    run_dir = Path(run_dir)
    source = Path(source_path)
    dest = run_dir / "baselines" / "preflight.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        data = source.read_bytes()
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return dest
