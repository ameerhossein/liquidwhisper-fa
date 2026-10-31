"""
RunPod Web Management Server for Persian ASR Training & Benchmarking.
FastAPI application delivering:
- Real-time telemetry WebSocket (/ws/live)
- REST endpoints for training control, benchmarks, and loss curves
- Static UI delivery on Port 7860
"""

import importlib.util
import os
import sys
import json
import time
import signal
import shutil
import subprocess
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Paths: managed via src/paths.py (single source of truth).
# Persistent volume on RunPod (/workspace/...), else repo-local. Env overrides:
# ASR_OUTPUT_DIR, ASR_RESULTS_DIR, ASR_MERGED_DIR.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
# Load .env (repo root / CWD) before any os.environ reads. Stdlib-only.
try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass
from src import paths as _managed_paths  # noqa: E402
from src import run_state  # noqa: E402
from src import hf_sync  # noqa: E402


def _default_results_dir() -> Path:
    resolved = _managed_paths.resolve_results_dir(None)
    p = Path(resolved)
    return p if p.is_absolute() else BASE_DIR / p


def _default_output_dir() -> str:
    """Managed default (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.default_output_dir()


RESULTS_DIR = _default_results_dir()
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# One self-contained directory per run. The default follows ASR_RESULTS_DIR,
# so it inherits the RunPod persistent-volume behavior; ASR_RUNS_DIR overrides.
RUNS_DIR = Path(_managed_paths.resolve_runs_dir(None)).expanduser()
if not RUNS_DIR.is_absolute():
    RUNS_DIR = BASE_DIR / RUNS_DIR
RUNS_DIR = RUNS_DIR.resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_FILE = RESULTS_DIR / "live_telemetry.json"
BENCHMARKS_FILE = RESULTS_DIR / "epoch_benchmarks.json"
PREFLIGHT_FILE = RESULTS_DIR / "baselines" / "preflight.json"
LOSS_HISTORY_FILE = RESULTS_DIR / "loss_history.json"
LOG_FILE = RESULTS_DIR / "training_live.log"
STOP_SIGNAL_FILE = RESULTS_DIR / "stop_training.signal"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Recover the run ledger and start the background process monitor."""
    _recover_runs_on_startup()
    _ensure_monitor_task()
    try:
        yield
    finally:
        await _stop_monitor_task()


app = FastAPI(
    title="Persian ASR RunPod Cockpit",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_to_json(request: Request, exc: Exception):
    """Always return JSON (never plain-text "Internal Server Error").

    The Web UI calls ``await res.json()`` on error responses; a plain-text
    500 body surfaces there as ``SyntaxError: Unexpected token 'I',
    "Internal S..." is not valid JSON`` and hides the real cause. This
    handler logs the traceback server-side and returns the message as JSON.
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    try:
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    except Exception:
        pass
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})

# Global process manager
training_process: Optional[subprocess.Popen] = None
training_log_fd = None
connected_websockets: List[WebSocket] = []
PID_FILE = RESULTS_DIR / "training.pid"
# HF restore child state. The status file is durable across page reloads and
# server restarts; the log is the full child output.
HF_RESTORE_STATUS_FILE = RUNS_DIR / "_hf_restore.status.json"
HF_RESTORE_LOG_FILE = RUNS_DIR / "_hf_restore.log"
last_exit_code: Optional[int] = None
halt_requested = False
SMOKE_DIR_NAME = "smoke-test"
smoke_process: Optional[subprocess.Popen] = None
smoke_log_fd = None
smoke_last_exit_code: Optional[int] = None
ACTIVE_STATUSES = ("initializing", "training", "evaluating", "evaluating_benchmark")
HALT_GRACE_SECONDS = int(os.environ.get("ASR_HALT_GRACE_SECONDS", "900") or 900)
MONITOR_INTERVAL_SECONDS = 2.0

# Run-ledger view: refreshed by the API/WebSocket paths and before every
# start/resume/stop so a live run is found after a server restart.
ACTIVE_RUN_ID: Optional[str] = None
LATEST_RUN: Optional[Dict[str, Any]] = None
CURRENT_RUN: Optional[Dict[str, Any]] = None

# Merge child state. The ledger's ``merge_pid`` is the durable copy; these
# in-memory handles let the monitor observe the real exit code while the
# server is up.
merge_process: Optional[subprocess.Popen] = None
merge_log_fd = None
merge_run_id: Optional[str] = None
training_run_id: Optional[str] = None
hf_restore_process: Optional[subprocess.Popen] = None
hf_restore_log_fd = None
hf_restore_run_id: Optional[str] = None
_hf_restore_last_state: str = "idle"
# Restore status states that do not represent in-flight work. Every other
# state is a phase reported by the child while it is still restoring, so a
# restart must keep treating it as active.
_HF_RESTORE_TERMINAL_STATES = frozenset(
    ("idle", "completed", "failed", "interrupted")
)
_monitor_task: Optional[asyncio.Task] = None

# Short TTL caches for the 1s status/WebSocket path. The full ledger scan
# (reconcile_all) and the merge no-child scan both walk every run directory;
# caching them keeps the hot path cheap while lifecycle actions and the
# 2s monitor force a fresh scan.
_REFRESH_TTL_SECONDS = 1.5
_MERGE_SCAN_TTL_SECONDS = 2.0
_refresh_cache_at: float = 0.0
_refresh_cache_summary: Optional[Dict[str, Any]] = None
_merge_scan_at: float = 0.0
_merge_scan_result: bool = False
_log_exception_state: Dict[str, list] = {}
_LOG_EXCEPTION_WINDOW_SECONDS = 60.0
_LOG_EXCEPTION_MAX_CONTEXTS = 128
_DUPLICATE_WARN_TTL_SECONDS = 60.0
_duplicate_warn_at: float = 0.0

# Remote run listing cache: one list_repo_files/download scan per repo every
# 30s, shared by every cockpit client.
_HF_RUNS_CACHE_TTL_SECONDS = 30.0
_hf_runs_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _load_preflight_benchmarks(
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Baseline rows from the measured preflight cache, empty when absent.

    The trainer writes ``results/baselines/preflight.json`` after evaluating
    BuzzASR and Whisper on the same rows. There is no hardcoded fallback: a
    missing or malformed cache means the API has no measured baseline yet.
    """
    preflight_file = Path(path) if path is not None else PREFLIGHT_FILE
    data = read_json_safely(preflight_file, {})
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for key in ("buzzasr_base", "whisper_v3"):
        row = models.get(key)
        if not isinstance(row, dict) or not row:
            continue
        merged = {
            "epoch_number": 0,
            "type": "baseline",
            "is_baseline": True,
            "canary_transcriptions": [],
            **row,
        }
        # The cache is the measured preflight by definition; force the source
        # so a row can never be mistaken for a legacy hardcoded baseline and
        # filtered out on the next read.
        merged["source"] = "preflight"
        rows.append(merged)
    return rows


def _is_process_running(pid: Optional[int] = None) -> bool:
    """Check PID liveness (for restart recovery via PID file)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _terminate_child(pid: Optional[int]) -> None:
    """Terminate a spawned child without ever signalling our own group.

    A stale/reused PID must not take the cockpit down: refuse the whole-process
    group when it is the server's own, and fall back to a direct signal when the
    target is not a group leader.
    """
    try:
        pid_i = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid_i = None
    if pid_i is None or pid_i == os.getpid():
        return
    if sys.platform == "win32":
        try:
            os.kill(pid_i, signal.SIGTERM)
        except Exception:
            pass
        return
    try:
        target_pgid = os.getpgid(pid_i)
        own_pgid = os.getpgid(0)
    except Exception:
        target_pgid = None
        own_pgid = None
    if target_pgid is not None and own_pgid is not None:
        if target_pgid == own_pgid:
            # Never killpg the server's own process group.
            return
        if target_pgid == pid_i:
            try:
                os.killpg(target_pgid, signal.SIGTERM)
                return
            except Exception:
                pass
    try:
        os.kill(pid_i, signal.SIGTERM)
    except Exception:
        pass


def _check_auth(config: Optional[Dict[str, Any]]) -> None:
    """Optional token gate: set ASR_COCKPIT_TOKEN env to require it.

    Public RunPod proxy URLs are unauthenticated by default; setting the
    env var requires callers to send {"token": "..."} or X-Auth-Token.
    """
    required = os.environ.get("ASR_COCKPIT_TOKEN", "")
    if not required:
        return
    provided = ""
    if isinstance(config, dict):
        provided = str(config.get("token", "") or "")
    # FastAPI header check happens in endpoint via Request; body fallback here.
    if provided != required:
        raise HTTPException(status_code=401, detail="Invalid or missing cockpit token.")


def get_default_benchmarks(
    preflight_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    return _load_preflight_benchmarks(preflight_file)


def read_json_safely(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _remember_log_exception_context(context: str, now: float) -> None:
    """Record an emission, evicting the oldest context when over the cap.

    Dynamic contexts (e.g. per-run ``clear merge_pid`` ids) must not grow
    the table without bound.
    """
    _log_exception_state[context] = [now, 0]
    while len(_log_exception_state) > _LOG_EXCEPTION_MAX_CONTEXTS:
        try:
            oldest = min(
                _log_exception_state,
                key=lambda key: float(_log_exception_state[key][0]),
            )
        except Exception:
            break
        if oldest == context:
            break
        _log_exception_state.pop(oldest, None)


def _log_exception(context: str) -> None:
    """Print one monitor/recovery failure, rate-limited per context.

    A persistent failure (e.g. a broken run directory polled every 2s) would
    otherwise print a full traceback on every tick. Only the first failure
    per context per 60s is printed; repeats are counted and reported with
    the next emission.
    """
    global _log_exception_state
    try:
        now = time.monotonic()
    except Exception:
        now = time.time()
    entry = _log_exception_state.get(context)
    if entry is not None:
        try:
            last_emit, suppressed = entry
        except (TypeError, ValueError):
            last_emit, suppressed = now, 0
        if (now - float(last_emit)) < _LOG_EXCEPTION_WINDOW_SECONDS:
            try:
                entry[1] = int(suppressed) + 1
            except Exception:
                pass
            return
        pending = 0
        try:
            pending = int(suppressed)
        except (TypeError, ValueError):
            pending = 0
        _remember_log_exception_context(context, now)
        try:
            import traceback
            suffix = (
                f" (and {pending} similar suppressed)"
                if pending
                else ""
            )
            print(f"[cockpit] {context} failed{suffix}:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        return
    _remember_log_exception_context(context, now)
    try:
        import traceback
        print(f"[cockpit] {context} failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    except Exception:
        pass


def _run_for_training_exit() -> Optional[Dict[str, Any]]:
    """The run that owned the just-exited child, else the current run view."""
    if training_run_id and run_state.run_id_valid(training_run_id):
        run = run_state.load_run(RUNS_DIR / training_run_id)
        if run is not None:
            return run
    return _active_run() or _current_run()


def _unlink_quietly(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _invalidate_refresh_cache() -> None:
    """Force the next cached scan to rescan after a ledger mutation."""
    global _refresh_cache_at, _refresh_cache_summary
    _refresh_cache_at = 0.0
    _refresh_cache_summary = None


def _note_merge_scan(result: bool) -> None:
    """Record a merge-scan outcome without walking the ledger.

    Used when the outcome is already known (a merge just started, or its
    launch failed): a non-forced ``_merge_running()`` must not report the
    previous cached value after that.
    """
    global _merge_scan_at, _merge_scan_result
    try:
        _merge_scan_at = time.monotonic()
    except Exception:
        _merge_scan_at = time.time()
    _merge_scan_result = result


def _reconcile_training_exit(exit_code: Optional[int]) -> None:
    """Apply the observed child exit (or a dead PID) to the run ledger.

    ``run_state.reconcile`` owns the status classification: a requested halt
    wins, a known exit code distinguishes failed/completed, and ``None`` means
    the process was lost (interrupted) when no terminal writer ran.  The
    attempt is finalized even when the trainer already wrote a terminal status,
    so a graceful exit never leaves ``current_pid`` behind for PID reuse.
    """
    global training_run_id
    run = _run_for_training_exit()
    if run is not None:
        _unlink_quietly(_run_paths(run)["pid_file"])
        try:
            run_state.reconcile(
                _run_dir(run),
                pid_alive=False,
                exit_code=exit_code,
                reason="process_exit" if exit_code is not None else None,
            )
            run_state.finalize_attempt(
                _run_dir(run),
                exit_code=exit_code,
                reason="process_exit" if exit_code is not None else None,
            )
        except FileNotFoundError:
            pass
        except Exception:
            # A broken manifest must never make the monitor crash.
            _log_exception(f"reconcile training exit for {run.get('run_id')}")
        finally:
            _invalidate_refresh_cache()
    training_run_id = None


def _training_running() -> bool:
    """Poll the training child process, recording its exit code for status."""
    global training_process, last_exit_code
    if training_process is not None:
        code = training_process.poll()
        if code is None:
            return True
        last_exit_code = code
        training_process = None
        _close_log_fd()
        _reconcile_training_exit(code)
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text(encoding="utf-8").strip() or "0")
            if _is_process_running(pid):
                return True
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    # After a server restart there is no child handle. Classify a dead
    # run-scoped PID file here too, so the 2-second monitor does not need to
    # wait for the full ledger rescan to notice a lost process.
    try:
        run = _active_run()
        if run is not None:
            pid_file = _run_paths(run)["pid_file"]
            if pid_file.exists():
                pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
                if _is_process_running(pid):
                    return True
                pid_file.unlink(missing_ok=True)
                current_pid = run.get("current_pid")
                try:
                    current_pid_i = int(current_pid) if current_pid is not None else None
                except (TypeError, ValueError):
                    current_pid_i = None
                if current_pid_i is None or current_pid_i == pid:
                    run_state.reconcile(
                        _run_dir(run), pid_alive=False, exit_code=None
                    )
                    _invalidate_refresh_cache()
    except Exception:
        pass
    return False


def _fake_trainer_enabled() -> bool:
    return (os.environ.get("ASR_FAKE_TRAINER", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _trainer_script_path() -> Path:
    name = "dev_fake_trainer.py" if _fake_trainer_enabled() else "train_lora.py"
    return BASE_DIR / "scripts" / name


def _run_dir(run: Optional[Dict[str, Any]]) -> Path:
    """Directory backing *run*, preferring the manifest's recorded path."""
    raw = ((run or {}).get("paths") or {}).get("run_dir")
    if raw:
        return Path(str(raw))
    run_id = str((run or {}).get("run_id") or "")
    return RUNS_DIR / run_id


def _run_paths(run: Optional[Dict[str, Any]]) -> Dict[str, Path]:
    """Per-run files, or the legacy global files when no ledger run exists.

    A run may point ``paths.results_dir`` away from its run directory (CLI
    runs with an explicit ``--results_dir``). Telemetry, benchmarks, loss
    history, and the stop signal live under that directory; the log and PID
    stay under the run directory so one run never overwrites another's log.
    """
    if not run:
        return {
            "telemetry_file": TELEMETRY_FILE,
            "benchmarks_file": BENCHMARKS_FILE,
            "loss_history_file": LOSS_HISTORY_FILE,
            "log_file": LOG_FILE,
            "pid_file": PID_FILE,
            "stop_signal_file": STOP_SIGNAL_FILE,
        }
    run_dir = _run_dir(run)
    raw_results = ((run or {}).get("paths") or {}).get("results_dir")
    if raw_results:
        results_dir = Path(str(raw_results))
        if not results_dir.is_absolute():
            results_dir = BASE_DIR / results_dir
    else:
        results_dir = run_dir
    return {
        "telemetry_file": results_dir / "live_telemetry.json",
        "benchmarks_file": results_dir / "epoch_benchmarks.json",
        "loss_history_file": results_dir / "loss_history.json",
        "log_file": run_dir / "training_live.log",
        "pid_file": run_dir / "training.pid",
        "stop_signal_file": results_dir / "stop_training.signal",
    }


def _refresh_runs() -> Dict[str, Any]:
    """Reconcile every run and refresh the active/latest/current view.

    This is the forced path used by lifecycle actions, startup recovery,
    and the background monitor. Read-only status paths should prefer
    :func:`_refresh_runs_cached`.
    """
    global ACTIVE_RUN_ID, LATEST_RUN, CURRENT_RUN
    global _refresh_cache_at, _refresh_cache_summary
    try:
        summary = run_state.reconcile_all(RUNS_DIR, is_alive=_is_process_running)
    except Exception:
        # A broken run directory must never take status/start/stop down.
        try:
            runs = run_state.list_runs(RUNS_DIR)
        except Exception:
            runs = []
        summary = {
            "active_run": None,
            "latest_run": runs[0] if runs else None,
            "merge_run": None,
            "runs": runs,
            "duplicates": [],
        }
    if "duplicates" not in summary:
        summary["duplicates"] = []
    duplicates = summary.get("duplicates") or []
    if duplicates:
        global _duplicate_warn_at
        try:
            warn_now = time.monotonic()
        except Exception:
            warn_now = time.time()
        if (warn_now - _duplicate_warn_at) >= _DUPLICATE_WARN_TTL_SECONDS:
            _duplicate_warn_at = warn_now
            try:
                print(
                    f"[cockpit] warning: duplicate live trainer PIDs for runs: "
                    f"{', '.join(str(run_id) for run_id in duplicates)}",
                    file=sys.stderr,
                )
            except Exception:
                pass
    # `_merge_running()` owns live-merge busy gating: it also has the child
    # handle and the artifact checks. The summary's "merge_run" is advisory.
    active = summary.get("active_run")
    latest = summary.get("latest_run")
    ACTIVE_RUN_ID = str(active.get("run_id")) if active else None
    LATEST_RUN = latest
    CURRENT_RUN = active or latest
    try:
        _refresh_cache_at = time.monotonic()
    except Exception:
        _refresh_cache_at = time.time()
    _refresh_cache_summary = summary
    return summary


def _refresh_runs_cached() -> Dict[str, Any]:
    """Return the cached run scan when fresh, else do a forced refresh.

    The returned mapping is a shallow copy; nested run dicts are shared and
    must be treated as read-only so callers cannot mutate the cache.
    """
    global _refresh_cache_summary
    try:
        now = time.monotonic()
    except Exception:
        now = time.time()
    if (
        _refresh_cache_summary is not None
        and (now - _refresh_cache_at) < _REFRESH_TTL_SECONDS
    ):
        return dict(_refresh_cache_summary)
    return dict(_refresh_runs())


def _current_run() -> Optional[Dict[str, Any]]:
    """The active run, else the latest run, else ``None`` (legacy mode)."""
    return CURRENT_RUN if CURRENT_RUN is not None else LATEST_RUN


def _active_run() -> Optional[Dict[str, Any]]:
    if ACTIVE_RUN_ID is None:
        return None
    run = CURRENT_RUN
    if run is not None and str(run.get("run_id")) == ACTIVE_RUN_ID:
        return run
    if LATEST_RUN is not None and str(LATEST_RUN.get("run_id")) == ACTIVE_RUN_ID:
        return LATEST_RUN
    return None


def _run_pid(run: Optional[Dict[str, Any]]) -> Optional[int]:
    """The run's trainer PID, falling back to ``<run>/training.pid``.

    CLI runs and older attempts may have no ``current_pid`` in the manifest.
    """
    if not run:
        return None
    try:
        pid = int(run.get("current_pid"))
        if pid > 0:
            return pid
    except (TypeError, ValueError):
        pass
    try:
        pid_file = _run_paths(run)["pid_file"]
        pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
        return pid if pid > 0 else None
    except Exception:
        return None


def _active_run_process_running() -> bool:
    """True when a run recovered from the ledger owns a live PID.

    Used after a server restart, when there is no child handle to poll.
    ``last_exit_code`` distinguishes a just-exited child from an unobserved
    process: once the server has seen an exit, PID liveness is not consulted.
    """
    if training_process is not None or last_exit_code is not None:
        return False
    run = _active_run()
    if run is None or str(run.get("status") or "") not in ("starting", "running"):
        return False
    pid = _run_pid(run)
    return pid is not None and _is_process_running(pid)


def _steps_summary(
    run: Optional[Dict[str, Any]],
    telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not run:
        return {"count": 0, "last_step": None, "last_saved_at": None}
    # The active run reports its exact per-step count from live telemetry;
    # historical runs report the deduplicated manifest count.
    try:
        if ACTIVE_RUN_ID is not None and str(run.get("run_id")) == ACTIVE_RUN_ID:
            data = telemetry if isinstance(telemetry, dict) else None
            if data is None:
                data = read_json_safely(
                    _run_paths(run)["telemetry_file"], {}
                )
            if isinstance(data, dict) and data.get("step_count") is not None:
                try:
                    active_count = int(data["step_count"])
                except (TypeError, ValueError):
                    active_count = None
                if active_count is not None:
                    last_step = data.get("step", run.get("last_step"))
                    try:
                        last_step_i = int(last_step) if last_step is not None else None
                    except (TypeError, ValueError):
                        last_step_i = run.get("last_step")
                    return {
                        "count": active_count,
                        "last_step": last_step_i,
                        "last_saved_at": run.get("last_step_at")
                        or data.get("updated_at"),
                    }
    except Exception:
        pass
    try:
        count = int(run.get("step_count") or 0)
    except (TypeError, ValueError):
        count = 0
    return {
        "count": count,
        "last_step": run.get("last_step"),
        "last_saved_at": run.get("last_step_at"),
    }


def _abs_output_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def _run_output_dir(run: Optional[Dict[str, Any]]) -> Path:
    """The run's effective output directory (``<run_dir>/output`` default)."""
    raw = ((run or {}).get("paths") or {}).get("output_dir")
    if raw:
        return _abs_output_path(str(raw))
    return _run_dir(run) / "output"


def _run_adapter_exists(run: Optional[Dict[str, Any]]) -> bool:
    if not run:
        return False
    return (
        _run_output_dir(run) / "best_adapter" / "adapter_config.json"
    ).is_file()


def _run_process_running(run: Optional[Dict[str, Any]]) -> bool:
    if not run or str(run.get("status") or "") not in ("starting", "running"):
        return False
    pid = _run_pid(run)
    return pid is not None and _is_process_running(pid)


def _run_payload(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The full per-run payload used by the REST run endpoints."""
    if not run:
        return {
            "run": None,
            "telemetry": read_json_safely(TELEMETRY_FILE, {}),
            "benchmarks": _get_benchmark_rows(),
            "loss_history": read_json_safely(LOSS_HISTORY_FILE, []),
            "steps": _steps_summary(None),
            "storage": run_state.storage_info(RUNS_DIR),
            "process_running": False,
            "adapter_exists": False,
            "hf": {},
            "hf_sync": {},
        }
    paths = _run_paths(run)
    telemetry = read_json_safely(paths["telemetry_file"], {})
    return {
        "run": run,
        "telemetry": telemetry,
        "benchmarks": _get_benchmark_rows(
            benchmarks_file=paths["benchmarks_file"],
            preflight_file=_run_preflight_file(run),
        ),
        "loss_history": read_json_safely(paths["loss_history_file"], []),
        "steps": _steps_summary(run, telemetry),
        "storage": run_state.storage_info(RUNS_DIR),
        "process_running": _run_process_running(run),
        "adapter_exists": _run_adapter_exists(run),
        "hf": run.get("hf") or {},
        "hf_sync": read_json_safely(_run_dir(run) / "hf_sync.json", {}),
    }


async def _monitor_loop() -> None:
    """Tick the trainer and merge children, then rescan the ledger."""
    while True:
        try:
            # Poll known children first so a real exit code is applied before
            # the ledger rescan (which only knows PID liveness).
            _training_running()
            _merge_running(force=True)
            _hf_restore_running()
            _refresh_runs()
        except asyncio.CancelledError:
            raise
        except Exception:
            _log_exception("monitor loop")
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)


def _ensure_monitor_task() -> Optional[asyncio.Task]:
    """Start the monitor once per event loop; return the live task."""
    global _monitor_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = _monitor_task
    if task is not None and not task.done():
        return task
    _monitor_task = loop.create_task(_monitor_loop())
    return _monitor_task


async def _stop_monitor_task() -> None:
    global _monitor_task
    task = _monitor_task
    _monitor_task = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


def _recover_runs_on_startup() -> None:
    """Startup recovery: apply child exits, merge recovery, ledger rescan."""
    def _forced_merge_scan() -> None:
        _merge_running(force=True)

    steps = (
        ("startup training poll", _training_running),
        ("startup merge recovery", _forced_merge_scan),
        ("startup HF restore recovery", _hf_restore_running),
        ("startup ledger refresh", _refresh_runs),
    )
    for context, step in steps:
        try:
            step()
        except Exception:
            _log_exception(context)


def _smoke_output_dir() -> str:
    production = _abs_output_path(
        _managed_paths.resolve_output_dir(_default_output_dir())
    ).resolve()
    candidates = [
        Path(_default_output_dir()).parent / SMOKE_DIR_NAME,
        BASE_DIR / "outputs" / SMOKE_DIR_NAME,
        BASE_DIR / "outputs" / f"{SMOKE_DIR_NAME}-check",
    ]
    for candidate in candidates:
        effective = _managed_paths.resolve_output_dir(str(candidate))
        if _abs_output_path(effective).resolve() != production:
            return _sanitize_output_dir(effective)
    raise HTTPException(
        status_code=400,
        detail="Smoke output would collide with the production output; "
               "set ASR_OUTPUT_DIR to a different directory.",
    )


def _smoke_results_dir() -> Path:
    production = _default_results_dir().resolve()
    candidates = [
        RESULTS_DIR / SMOKE_DIR_NAME,
        BASE_DIR / "results" / SMOKE_DIR_NAME,
        BASE_DIR / "results" / f"{SMOKE_DIR_NAME}-check",
    ]
    for candidate in candidates:
        effective = Path(_managed_paths.resolve_results_dir(str(candidate)))
        if not effective.is_absolute():
            effective = BASE_DIR / effective
        effective = effective.resolve()
        if effective != production:
            return effective
    raise HTTPException(
        status_code=400,
        detail="Smoke results would collide with production results; "
               "set ASR_RESULTS_DIR to a different directory.",
    )


def _smoke_log_file() -> Path:
    return _smoke_results_dir() / "training_live.log"


def _smoke_pid_file() -> Path:
    return _smoke_results_dir() / "training.pid"


def _smoke_attempt_file() -> Path:
    return _smoke_results_dir() / "smoke_attempt.json"


def _record_smoke_result(exit_code: int) -> None:
    try:
        path = _smoke_attempt_file()
        data = read_json_safely(path, {})
        if not isinstance(data, dict):
            data = {}
        data["exit_code"] = int(exit_code)
        data["finished_at"] = time.time()
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _close_smoke_log_fd() -> None:
    global smoke_log_fd
    try:
        if smoke_log_fd is not None:
            try:
                smoke_log_fd.flush()
            except Exception:
                pass
            try:
                smoke_log_fd.close()
            except Exception:
                pass
    finally:
        smoke_log_fd = None


def _smoke_running() -> bool:
    global smoke_process, smoke_last_exit_code
    if smoke_process is not None:
        code = smoke_process.poll()
        if code is None:
            return True
        smoke_last_exit_code = code
        smoke_process = None
        _close_smoke_log_fd()
        _record_smoke_result(code)
        try:
            _smoke_pid_file().unlink(missing_ok=True)
        except Exception:
            pass
    try:
        pid_file = _smoke_pid_file()
        if pid_file.exists():
            pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
            if _is_process_running(pid):
                return True
            pid_file.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def _read_log_tail(path: Path, tail: int = 30) -> List[str]:
    try:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip() for line in f.readlines()[-tail:]]
    except Exception:
        return []


def _merge_paths(run: Dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """``(run_dir, merged_dir, merge_log, merge_pid_file)`` for *run*."""
    run_dir = _run_dir(run)
    raw_merged = (run.get("paths") or {}).get("merged_dir") or (run_dir / "merged")
    merged_dir = Path(str(raw_merged))
    if not merged_dir.is_absolute():
        merged_dir = BASE_DIR / merged_dir
    return run_dir, merged_dir, run_dir / "merge.log", run_dir / "merge.pid"


def _merge_stage(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return next(
        (
            stage
            for stage in run.get("stages") or []
            if stage.get("name") == "merge"
        ),
        None,
    )


_MERGE_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.bin")
_TERMINAL_STAGE_STATUSES = (
    "completed",
    "stopped",
    "failed",
    "skipped",
    "interrupted",
)


def _merge_artifact_paths(
    run: Dict[str, Any]
) -> Optional[tuple[Path, Path]]:
    """The merged model's ``(config, weights)`` when both are present."""
    _run_dir_path, merged_dir, _log, _pid = _merge_paths(run)
    config = merged_dir / "config.json"
    if not config.is_file():
        return None
    for name in _MERGE_WEIGHT_NAMES:
        weights = merged_dir / name
        if weights.is_file():
            return config, weights
    return None


def _merge_artifacts_complete(run: Dict[str, Any]) -> bool:
    """True when the merged model has its config and a model weights file."""
    return _merge_artifact_paths(run) is not None


def _merge_artifacts_fresh(run: Dict[str, Any], stage: Dict[str, Any]) -> bool:
    """True when both merged artifacts were written at/after the stage start.

    Used only for no-PID recovery: a crash before ``merge.pid`` was recorded
    must not inherit a previous attempt's complete output.
    """
    artifacts = _merge_artifact_paths(run)
    if artifacts is None:
        return False
    try:
        since = float(stage.get("started_at"))
    except (TypeError, ValueError):
        return False
    try:
        return all(path.stat().st_mtime >= since for path in artifacts)
    except OSError:
        return False


def _merge_stages_terminal(run: Dict[str, Any]) -> bool:
    """R12: a target trainer may still exit only when its run is finished."""
    stages = {
        str(stage.get("name")): str(stage.get("status") or "pending")
        for stage in run.get("stages") or []
    }
    if stages.get("training") != "completed":
        return False
    if stages.get("post_eval") not in ("completed", "skipped"):
        return False
    for name in ("preflight", "training", "post_eval"):
        if stages.get(name, "pending") not in _TERMINAL_STAGE_STATUSES:
            return False
    return not any(
        str(stage.get("status")) == "running"
        for stage in run.get("stages") or []
    )


def _close_merge_log_fd() -> None:
    global merge_log_fd
    try:
        if merge_log_fd is not None:
            try:
                merge_log_fd.flush()
            except Exception:
                pass
            try:
                merge_log_fd.close()
            except Exception:
                pass
    finally:
        merge_log_fd = None


def _finish_merge(
    run_id: str,
    exit_code: Optional[int],
    *,
    pid_known: bool = False,
) -> None:
    """Classify a finished/recovered merge from its exit code and artifacts."""
    global merge_run_id
    run = None
    if run_state.run_id_valid(run_id):
        try:
            run = run_state.load_run(RUNS_DIR / run_id)
        except Exception:
            run = None
    if run is not None:
        run_dir, merged_dir, log_file, _pid_file = _merge_paths(run)
        stage = _merge_stage(run)
        if stage is not None and stage.get("status") == "running":
            complete = _merge_artifacts_complete(run)
            completed = complete and (
                exit_code == 0
                or (
                    exit_code is None
                    and (
                        pid_known
                        or _merge_artifacts_fresh(run, stage)
                    )
                )
            )
            if completed:
                run_state.set_stage(
                    run_dir,
                    "merge",
                    "completed",
                    detail=f"merged to {merged_dir}",
                    exit_code=exit_code,
                )
            else:
                tail = _read_log_tail(log_file, 30)
                if exit_code == 0:
                    detail = "merge exited 0 without complete artifacts"
                elif exit_code is None and complete:
                    try:
                        missing_started = float(stage.get("started_at"))
                    except (TypeError, ValueError):
                        missing_started = None
                    if missing_started is None:
                        detail = "merge stage start time is missing"
                    else:
                        detail = "merge artifacts predate the merge attempt"
                elif exit_code is None:
                    detail = "merge process ended without artifacts"
                else:
                    detail = f"merge exited {exit_code}"
                run_state.set_stage(
                    run_dir,
                    "merge",
                    "failed",
                    detail=detail,
                    exit_code=exit_code,
                    log_tail=tail,
                )
            try:
                run_state.update_run(
                    run_dir, lambda current: {**current, "merge_pid": None}
                )
            except Exception:
                _log_exception(f"clear merge_pid for {run_id}")
            finally:
                _invalidate_refresh_cache()
    merge_run_id = None


def _merge_pid_from_run(run: Dict[str, Any]) -> Optional[int]:
    raw = run.get("merge_pid")
    if raw not in (None, ""):
        try:
            pid = int(raw)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            return pid
    try:
        _run_dir_path, _merged, _log, pid_file = _merge_paths(run)
        if pid_file.is_file():
            pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
            return pid if pid > 0 else None
    except Exception:
        pass
    return None


def _merge_running(force: bool = False) -> bool:
    """Poll the merge child, or recover a merge PID from the ledger.

    With a live child handle the real exit code is observed. Without one (a
    server restart), every run's ``merge`` stage is inspected: a live PID
    means the merge continues, a dead PID is classified from its artifacts.
    The no-child scan is cached for ~2s; lifecycle actions and the monitor
    pass ``force=True``.
    """
    global merge_process, merge_run_id
    global _merge_scan_at, _merge_scan_result
    if merge_process is not None:
        code = merge_process.poll()
        if code is None:
            return True
        run_id = merge_run_id
        merge_process = None
        _close_merge_log_fd()
        if run_id:
            _finish_merge(run_id, code, pid_known=True)
        try:
            _merge_scan_at = time.monotonic()
        except Exception:
            _merge_scan_at = time.time()
        _merge_scan_result = False
        return False

    if not force:
        try:
            now = time.monotonic()
        except Exception:
            now = time.time()
        if (now - _merge_scan_at) < _MERGE_SCAN_TTL_SECONDS:
            return _merge_scan_result

    try:
        runs = run_state.list_runs(RUNS_DIR, limit=None)
    except Exception:
        _log_exception("scan runs for merge recovery")
        runs = []
    result = False
    for run in runs:
        stage = _merge_stage(run)
        if stage is None or stage.get("status") != "running":
            continue
        pid = _merge_pid_from_run(run)
        if pid is not None and _is_process_running(pid):
            merge_run_id = str(run.get("run_id") or "") or None
            result = True
            break
        _finish_merge(
            str(run.get("run_id") or ""),
            None,
            pid_known=pid is not None,
        )
    try:
        _merge_scan_at = time.monotonic()
    except Exception:
        _merge_scan_at = time.time()
    _merge_scan_result = result
    return result


def _any_job_running() -> bool:
    """Shared busy check: full run, smoke test, merge, or HF restore."""
    if (
        _training_running()
        or _smoke_running()
        or _merge_running(force=True)
        or _hf_restore_running()
    ):
        return True
    _refresh_runs()
    return _active_run() is not None


def _any_training_running() -> bool:
    """Backward-compatible alias for :func:`_any_job_running`."""
    return _any_job_running()


def _hf_restore_status_file() -> Path:
    return HF_RESTORE_STATUS_FILE


def _hf_restore_log_file() -> Path:
    return HF_RESTORE_LOG_FILE


def _hf_restore_status() -> Dict[str, Any]:
    """The persisted restore status, or an idle placeholder when absent."""
    data = read_json_safely(_hf_restore_status_file(), {})
    return data if isinstance(data, dict) and data else {"state": "idle"}


def _redact_secret_text(value: Any) -> str:
    """Remove configured secrets from text returned to a cockpit client."""
    text = str(value if value is not None else "")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ASR_COCKPIT_TOKEN"):
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, "***")
    return text


def _write_hf_restore_status(payload: Dict[str, Any]) -> None:
    """Atomically persist the restore status consumed by reloads/restarts."""
    body = dict(payload)
    body["updated_at"] = time.time()
    path = _hf_restore_status_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(body, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _record_hf_restore_failure(run_id: str, message: str) -> None:
    try:
        _write_hf_restore_status(
            {
                "state": "failed",
                "run_id": run_id,
                "message": _redact_secret_text(message),
                "pid": None,
                "checkpoint_step": None,
            }
        )
    except Exception:
        pass


def _close_hf_restore_log_fd() -> None:
    global hf_restore_log_fd
    try:
        if hf_restore_log_fd is not None:
            try:
                hf_restore_log_fd.flush()
            except Exception:
                pass
            try:
                hf_restore_log_fd.close()
            except Exception:
                pass
    finally:
        hf_restore_log_fd = None


def _hf_restore_running() -> bool:
    """Poll the restore child and reconcile a stale persisted status.

    After a restart there is no child handle, so progress is recovered from
    the status file. Every non-terminal state (``running``, ``resolving``,
    ``listing``, ``ledger``, ``checkpoint``, ``verify``, and any future phase)
    means a restore may still be active. If its recorded PID is dead (or was
    never recorded), the status becomes ``interrupted`` and stops blocking new
    work.
    """
    global hf_restore_process, hf_restore_run_id, _hf_restore_last_state
    exited = False
    if hf_restore_process is not None:
        code = hf_restore_process.poll()
        if code is None:
            return True
        hf_restore_process = None
        hf_restore_run_id = None
        _close_hf_restore_log_fd()
        exited = True

    status = _hf_restore_status()
    state = str(status.get("state") or "idle")
    if state not in _HF_RESTORE_TERMINAL_STATES:
        pid = status.get("pid")
        try:
            pid_int = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid_int = None
        # With no child handle, a live PID means the restore survived the
        # server restart. With a handle, poll() has already proven the child
        # exited, so even a reused PID must not keep the phase alive.
        if not exited and pid_int is not None and _is_process_running(pid_int):
            hf_restore_run_id = (
                str(status.get("run_id") or "") or hf_restore_run_id
            )
            _hf_restore_last_state = state
            return True

        status["state"] = "interrupted"
        status["message"] = (
            f"restore process is no longer running; phase '{state}' was "
            "marked interrupted"
        )
        try:
            _write_hf_restore_status(status)
        except Exception:
            pass
        state = "interrupted"

    if exited:
        # A completed restore adds a run directory; any terminal exit should
        # make the next run-list scan see the final on-disk state.
        _invalidate_refresh_cache()
        _hf_restore_last_state = state
    elif state != _hf_restore_last_state:
        if state in ("completed", "failed", "interrupted"):
            _invalidate_refresh_cache()
        _hf_restore_last_state = state
    return False


def _start_hf_restore(
    run_id: str,
    repo: Optional[str] = None,
    include_merged: bool = False,
) -> Dict[str, Any]:
    """Spawn ``scripts/hf_restore.py`` and persist its initial status."""
    global hf_restore_process, hf_restore_log_fd, hf_restore_run_id
    global _hf_restore_last_state

    run_id = str(run_id or "").strip()
    if not run_state.run_id_valid(run_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "invalid run id; expected YYYYMMDD-HHMMSS-<6 hex chars>."
            ),
        )

    if _hf_restore_running():
        raise HTTPException(
            status_code=400, detail="A Hugging Face restore is already running."
        )
    if _any_job_running():
        raise HTTPException(
            status_code=400,
            detail=(
                "Training, smoke test, merge, or Hugging Face restore is "
                "already running."
            ),
        )

    run_dir = RUNS_DIR / run_id
    try:
        occupied = run_dir.exists() and (
            not run_dir.is_dir() or any(run_dir.iterdir())
        )
    except OSError:
        occupied = True
    if occupied:
        raise HTTPException(
            status_code=409,
            detail=(
                f"local run directory already exists: {run_dir}; "
                "remove it or choose another run"
            ),
        )

    repo_id = str(repo or "").strip() or (
        os.environ.get("ASR_HF_RUN_REPO", "") or ""
    ).strip()
    if not repo_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "no Hugging Face repo configured; set ASR_HF_RUN_REPO or "
                "pass repo"
            ),
        )
    try:
        hf_sync.validate_repo_id(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status_file = _hf_restore_status_file()
    log_file = _hf_restore_log_file()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    _close_hf_restore_log_fd()
    try:
        hf_restore_log_fd = open(log_file, "w", encoding="utf-8")
        hf_restore_log_fd.write(
            f"=== Hugging Face restore launched at {time.ctime()} ===\n"
        )
        hf_restore_log_fd.write(f"Run: {run_id}\nRepo: {repo_id}\n\n")
        hf_restore_log_fd.flush()
    except OSError as exc:
        _close_hf_restore_log_fd()
        _record_hf_restore_failure(
            run_id, f"Failed to open the restore log: {exc}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to open the restore log: {exc}"
        ) from exc

    cmd = [
        sys.executable,
        "-u",
        "scripts/hf_restore.py",
        "--run", run_id,
        "--repo", repo_id,
        "--runs-root", str(RUNS_DIR),
        "--status-file", str(status_file),
    ]
    if include_merged:
        cmd.append("--include-merged")

    hf_restore_process = None
    try:
        hf_restore_process = subprocess.Popen(
            cmd,
            stdout=hf_restore_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            start_new_session=True,
        )
    except Exception as exc:
        try:
            hf_restore_log_fd.write(
                f"\n!!! Failed to launch the restore: {exc}\n"
            )
            hf_restore_log_fd.flush()
        except Exception:
            pass
        _close_hf_restore_log_fd()
        _record_hf_restore_failure(
            run_id, f"Failed to launch the restore: {exc}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to launch the restore: {exc}"
        ) from exc

    hf_restore_run_id = run_id
    _hf_restore_last_state = "running"
    pid = hf_restore_process.pid
    try:
        _write_hf_restore_status(
            {
                "state": "running",
                "run_id": run_id,
                "message": f"Restoring {run_id} from {repo_id}",
                "pid": pid,
                "checkpoint_step": None,
            }
        )
    except Exception as exc:
        _terminate_child(pid)
        hf_restore_process = None
        hf_restore_run_id = None
        _close_hf_restore_log_fd()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record the restore status: {exc}",
        ) from exc

    return {"state": "running", "run_id": run_id}


def _derive_status(
    telemetry: Dict[str, Any],
    is_running: bool,
    run: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map raw telemetry plus process state to the status shown in the UI."""
    if run is not None:
        status = telemetry.get("status") or "idle"
        run_status = str(run.get("status") or "")
        if run_status in ("starting", "running"):
            stages = {
                str(stage.get("name")): str(stage.get("status") or "pending")
                for stage in run.get("stages") or []
            }
            post_eval_running = stages.get("post_eval") == "running"
            training_running = stages.get("training") == "running"
            if post_eval_running or status in ("evaluating", "evaluating_benchmark"):
                status = "evaluating_benchmark"
            elif status in ("idle", "running"):
                status = "training"
            elif status in ("completed", "stopped"):
                # A terminal telemetry snapshot must never mask a live run:
                # on_train_end writes before post-eval starts.
                if training_running:
                    status = "training"
                elif stages.get("training") == "completed":
                    status = "evaluating_benchmark"
                else:
                    status = "training"
            if run.get("stop_requested_at") and status in ACTIVE_STATUSES:
                status = "stopping"
        elif run_status in ("completed", "stopped", "failed", "interrupted"):
            status = run_status
        telemetry["status"] = status
        return telemetry

    # Legacy path (no ledger run): keep the in-memory halt behavior.
    status = telemetry.get("status") or "idle"
    if is_running:
        if status in ("idle", "running"):
            status = "training"
        if halt_requested and status in ACTIVE_STATUSES:
            status = "stopping"
    elif status in ACTIVE_STATUSES:
        status = "stopped" if halt_requested else "failed"
    if not is_running and not halt_requested and last_exit_code not in (None, 0):
        status = "failed"
    telemetry["status"] = status
    return telemetry


def get_gpu_info() -> Dict[str, Any]:
    """System-wide GPU info: prefer nvidia-smi, fall back to torch.

    The previous torch-only check reported "GPU unavailable" whenever the
    *server* process lacked a CUDA torch build (or CUDA_VISIBLE_DEVICES hid
    the device), even on an L40S pod, and torch's memory_allocated() only
    reflects this process (~0 GB), not the training job. nvidia-smi reports
    machine-wide usage so the header meter is accurate.
    """
    # 1) nvidia-smi: works without torch and reflects the training process.
    try:
        import shutil
        if shutil.which("nvidia-smi"):
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                if len(parts) >= 4:
                    name, used, total, util = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                    alloc_gb = round(used / 1024.0, 2)
                    total_gb = round(total / 1024.0, 2)
                    return {
                        "available": True,
                        "device_name": name,
                        "allocated_gb": alloc_gb,
                        "reserved_gb": alloc_gb,
                        "total_gb": total_gb,
                        "utilization_pct": round(util, 1),
                        "source": "nvidia-smi",
                    }
    except Exception:
        pass
    # 2) torch fallback (server-process-local view).
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            res = torch.cuda.memory_reserved() / (1024**3)
            tot = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "allocated_gb": round(alloc, 2),
                "reserved_gb": round(res, 2),
                "total_gb": round(tot, 2),
                "utilization_pct": round(alloc / tot * 100.0, 1) if tot > 0 else 0.0,
                "source": "torch",
            }
    except Exception:
        pass
    return {
        "available": False,
        "device_name": "N/A",
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "total_gb": 0.0,
        "utilization_pct": 0.0,
        "source": "none",
    }


def _close_log_fd():
    global training_log_fd
    try:
        if training_log_fd is not None:
            try:
                training_log_fd.flush()
            except Exception:
                pass
            try:
                training_log_fd.close()
            except Exception:
                pass
    finally:
        training_log_fd = None


def _sanitize_output_dir(raw: str) -> str:
    """Confine output_dir to repo or persistent-volume roots.

    Allows relative paths under BASE_DIR (legacy behavior) plus absolute
    paths under /workspace, /runpod-volume, /mnt, colon-separated
    ASR_ALLOWED_OUTPUT_ROOTS, or the managed runs root. A self-contained run's
    output lives under the runs root even when that root is elsewhere.
    Returns absolute path as string so the trainer can write checkpoints
    directly to the persistent volume (auto-resumable).
    """
    raw = (raw or _default_output_dir()).strip() or _default_output_dir()
    extra_roots = [r for r in os.environ.get("ASR_ALLOWED_OUTPUT_ROOTS", "").split(":") if r.strip()]
    allowed_abs = [Path("/workspace"), Path("/runpod-volume"), Path("/mnt")]
    for r in extra_roots:
        try:
            allowed_abs.append(Path(r).resolve())
        except Exception:
            continue
    try:
        allowed_abs.append(RUNS_DIR.resolve())
    except Exception:
        pass
    if os.path.isabs(raw):
        p = Path(raw).resolve()
        if p == BASE_DIR.resolve() or BASE_DIR.resolve() in p.parents:
            return str(p)
        for root in allowed_abs:
            try:
                if p == root or root in p.parents:
                    return str(p)
            except Exception:
                continue
        raise HTTPException(status_code=400, detail="output_dir must be inside project or /workspace, /runpod-volume, /mnt (or ASR_ALLOWED_OUTPUT_ROOTS).")
    p = (BASE_DIR / raw).resolve()
    try:
        p.relative_to(BASE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="output_dir must stay inside project directory.")
    return str(p.relative_to(BASE_DIR.resolve()))


def _check_audio_backend_or_400() -> None:
    """Fail /api/train/start fast if datasets cannot decode audio.

    datasets 4.x decodes the ``audio`` column via
    ``torchcodec.decoders.AudioDecoder``, but torchcodec==0.2.1 (cu124)
    only ships VideoDecoder -> ImportError at first filter/map. datasets<4
    uses the soundfile backend and needs no torchcodec. Check the right
    backend for the installed datasets version so the Web UI surfaces the
    real fix instead of a wasted launch cycle.
    """
    try:
        import datasets as _ds
        _ver = str(getattr(_ds, "__version__", "0") or "0")
        _major = int(_ver.split(".")[0] or 0)
    except Exception:
        _major = 0
    if _major >= 4:
        try:
            from torchcodec.decoders import AudioDecoder  # noqa: F401
            return
        except Exception as e:
            reason = f" (import failed{': ' + str(e).strip()[:300] if str(e).strip() else ''})"
            raise HTTPException(
                status_code=500,
                detail=(
                    "Incompatible audio stack: datasets>=4 needs "
                    "'torchcodec.decoders.AudioDecoder', but installed torchcodec "
                    "does not provide it (0.2.1/cu124 has only VideoDecoder). Pick ONE: "
                    "(1) downgrade to the pinned backend: source .venv/bin/activate && "
                    "pip install 'datasets>=2.19,<4' (keeps torch 2.6/cu124), OR "
                    "(2) upgrade torch+torchcodec together: pip install -U torch torchaudio "
                    "torchcodec --index-url https://download.pytorch.org/whl/cu128 "
                    "(needs CUDA 12.8 driver). Do NOT upgrade torchcodec alone."
                    f"{reason}"
                ),
            )
    try:
        import soundfile  # noqa: F401
        return
    except Exception as e:
        reason = f" (import failed{': ' + str(e).strip()[:300] if str(e).strip() else ''})"
        raise HTTPException(
            status_code=500,
            detail=(
                "Audio backend missing: datasets<4 decodes 'audio' via 'soundfile'. "
                "On the server run: source .venv/bin/activate && pip install soundfile librosa "
                "(or pip install -U 'datasets[audio]'), then restart training. "
                f"Fresh pods: ./setup_server.sh installs it automatically.{reason}"
            ),
        )


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "RunPod Persian ASR Cockpit active. UI index.html not found."}


@app.get("/api/status")
async def get_status():
    # Apply a known child exit before refreshing the ledger so the response
    # never reports a stale ``running`` run with ``process_running: false``.
    training_live = _training_running()
    _merge_running()
    _refresh_runs_cached()
    run = _current_run()
    paths = _run_paths(run)
    is_running = training_live or _active_run_process_running()

    telemetry = read_json_safely(paths["telemetry_file"], {
        "status": "running" if is_running else "idle",
        "step": 0,
        "max_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "learning_rate": 0.0,
        "speed_s_per_step": 0.0,
        "eta_seconds": 0,
        "gpu": get_gpu_info()
    })

    telemetry = _derive_status(telemetry, is_running, run)
    telemetry["process_running"] = is_running
    telemetry["gpu"] = telemetry.get("gpu") or get_gpu_info()
    telemetry["run"] = run
    telemetry["active_run_id"] = ACTIVE_RUN_ID
    telemetry["steps"] = _steps_summary(run, telemetry)
    telemetry["storage"] = run_state.storage_info(RUNS_DIR)
    telemetry["adapter_exists"] = _run_adapter_exists(run)
    return telemetry


@app.get("/api/train/smoke/status")
async def get_smoke_status():
    running = _smoke_running()
    adapter_path = None
    try:
        output_dir = _smoke_output_dir()
        adapter_path = (
            _abs_output_path(output_dir) / "best_adapter" / "adapter_config.json"
        )
        adapter_exists = adapter_path.is_file()
    except Exception:
        output_dir = ""
        adapter_exists = False

    try:
        attempt = read_json_safely(_smoke_attempt_file(), {})
    except Exception:
        attempt = {}
    attempt_exists = isinstance(attempt, dict) and bool(attempt)
    persisted_exit = attempt.get("exit_code") if attempt_exists else None
    effective_exit = (
        smoke_last_exit_code if smoke_last_exit_code is not None else persisted_exit
    )

    adapter_is_new = False
    if adapter_exists and adapter_path is not None:
        started_at = attempt.get("started_at") if attempt_exists else None
        try:
            adapter_is_new = (
                started_at is None
                or adapter_path.stat().st_mtime >= float(started_at) - 1.0
            )
        except Exception:
            adapter_is_new = False

    reason = None
    if running:
        status = "running"
    elif not attempt_exists:
        status = "idle"
    elif effective_exit == 0 and adapter_exists:
        status = "passed"
    elif effective_exit is None and adapter_exists and adapter_is_new:
        status = "passed"
    else:
        status = "failed"
        if effective_exit is not None and effective_exit != 0:
            reason = "nonzero_exit"
        elif effective_exit == 0 and not adapter_exists:
            reason = "missing_adapter"
        else:
            reason = "ended_without_result"

    payload: Dict[str, Any] = {
        "status": status,
        "running": running,
        "exit_code": effective_exit,
        "output_dir": output_dir,
        "adapter_exists": adapter_exists,
    }
    if reason:
        payload["reason"] = reason
    if status == "failed":
        try:
            payload["log_tail"] = _read_log_tail(_smoke_log_file())
        except Exception:
            payload["log_tail"] = []
    return payload


def _drop_stale_baselines(rows: Any) -> List[Dict[str, Any]]:
    """Keep every row except legacy baselines with no measured preflight source.

    Old hardcoded/config rows are ``is_baseline`` but carry no ``source``;
    showing them would present numbers we cannot reproduce.
    """
    if not isinstance(rows, list):
        return []
    return [
        row for row in rows
        if not (
            isinstance(row, dict)
            and row.get("is_baseline")
            and row.get("source") != "preflight"
        )
    ]


def _get_benchmark_rows(
    *,
    persist_defaults: bool = False,
    benchmarks_file: Optional[Path] = None,
    preflight_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return benchmark rows shared by the REST endpoint and the live socket."""
    path = Path(benchmarks_file) if benchmarks_file is not None else BENCHMARKS_FILE
    rows = _drop_stale_baselines(read_json_safely(path, None))
    if rows:
        return rows
    rows = get_default_benchmarks(preflight_file)
    if rows and persist_defaults:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


@app.get("/api/benchmarks")
async def get_benchmarks():
    _refresh_runs_cached()
    run = _current_run()
    paths = _run_paths(run)
    return _get_benchmark_rows(
        persist_defaults=True,
        benchmarks_file=paths["benchmarks_file"],
        preflight_file=_run_preflight_file(run),
    )


@app.get("/api/loss-history")
async def get_loss_history():
    _refresh_runs_cached()
    paths = _run_paths(_current_run())
    return read_json_safely(paths["loss_history_file"], [])


@app.get("/api/logs")
async def get_logs(tail: int = 150):
    _refresh_runs_cached()
    paths = _run_paths(_current_run())
    log_file = paths["log_file"]
    if not log_file.exists():
        return {"logs": ["Log buffer empty. Training has not started."]}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return {"logs": [line.rstrip() for line in lines[-tail:]]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}


@app.post("/api/train/smoke")
async def start_smoke_test(config: Dict[str, Any] = None):
    global smoke_process, smoke_log_fd, smoke_last_exit_code

    _check_auth(config)
    _check_audio_backend_or_400()
    if _any_job_running():
        raise HTTPException(status_code=400, detail="Training or smoke test is already running.")

    config = config or {}
    try:
        batch_i = int(config.get("batch_size", 8))
        accum_i = int(config.get("grad_accum", 4))
        lr_f = float(config.get("learning_rate", 1e-4))
        eval_bs_i = int(config.get("eval_batch_size", 16))
        num_proc_i = int(config.get("num_proc", 8))
        tele_f = float(config.get("telephony_prob", 0.5))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="batch_size/grad_accum/eval_batch_size/num_proc must be integers; "
                   "learning_rate/telephony_prob numeric.",
        )
    if not (1 <= batch_i <= 64):
        raise HTTPException(status_code=400, detail="batch_size must be 1..64.")
    if not (1 <= accum_i <= 16):
        raise HTTPException(status_code=400, detail="grad_accum must be 1..16.")
    if not (1e-7 <= lr_f <= 1e-2):
        raise HTTPException(status_code=400, detail="learning_rate must be 1e-7..1e-2.")
    if not (1 <= eval_bs_i <= 64):
        raise HTTPException(status_code=400, detail="eval_batch_size must be 1..64.")
    if not (1 <= num_proc_i <= 16):
        raise HTTPException(status_code=400, detail="num_proc must be 1..16.")
    if not (0.0 <= tele_f <= 1.0):
        raise HTTPException(status_code=400, detail="telephony_prob must be 0..1.")
    mode = str(config.get("telephony_mode", "paired")).strip().lower()
    if mode not in ("paired", "disjoint", "clean"):
        raise HTTPException(
            status_code=400,
            detail="telephony_mode must be one of paired, disjoint, clean.",
        )

    ckpt_raw = config.get("gradient_checkpointing", False)
    gradient_ckpt = (
        str(ckpt_raw).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(ckpt_raw, str)
        else bool(ckpt_raw)
    )
    if batch_i > 8 and not gradient_ckpt:
        raise HTTPException(
            status_code=400,
            detail="batch_size above 8 needs gradient_checkpointing on; "
                   "without it batch 16 runs out of memory on a 48 GB GPU.",
        )

    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar")).strip() or "shekar-ai/Neyshekar"
    if len(dataset_name) > 128 or any(c in dataset_name for c in ";&|$` \n"):
        raise HTTPException(status_code=400, detail="Invalid dataset_name.")

    output_dir = _smoke_output_dir()
    smoke_results_dir = _smoke_results_dir()
    smoke_results_dir.mkdir(parents=True, exist_ok=True)
    _abs_output_path(output_dir).mkdir(parents=True, exist_ok=True)

    # A new attempt must never inherit a previous pass.
    adapter_dir = _abs_output_path(output_dir) / "best_adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    if adapter_dir.exists():
        raise HTTPException(
            status_code=500,
            detail="Could not remove the previous smoke adapter; refusing to start.",
        )
    _smoke_pid_file().unlink(missing_ok=True)
    _smoke_attempt_file().parent.mkdir(parents=True, exist_ok=True)
    _smoke_attempt_file().write_text(
        json.dumps({"started_at": time.time()}), encoding="utf-8"
    )
    smoke_last_exit_code = None

    cmd = [
        sys.executable,
        "-u",
        str(BASE_DIR / "scripts" / "train_lora.py"),
        "--base_model", "BuzzASR/persian",
        "--dataset_name", dataset_name,
        "--output_dir", output_dir,
        "--results_dir", str(smoke_results_dir),
        "--epochs", "1",
        "--batch_size", str(batch_i),
        "--grad_accum", str(accum_i),
        "--learning_rate", str(lr_f),
        "--eval_batch_size", str(eval_bs_i),
        "--max_train_samples", "200",
        "--max_eval_samples", "50",
        "--max_benchmark_samples", "25",
        "--preflight_samples", "25",
        "--num_proc", str(num_proc_i),
        "--telephony_prob", str(tele_f),
        "--telephony_mode", mode,
        "--warmup_steps", "2",
        "--eval_steps", "5",
        "--save_steps", "5",
        "--enable_telemetry",
        "--run_epoch_benchmark",
        "--no-run-post-test-eval",
        "--fresh_start",
        "--no-auto_resume",
        "--gradient_checkpointing" if gradient_ckpt else "--no-gradient_checkpointing",
    ]

    _close_smoke_log_fd()
    smoke_log_fd = open(_smoke_log_file(), "w", encoding="utf-8")
    smoke_log_fd.write(f"=== Smoke test launched at {time.ctime()} ===\n")
    smoke_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    smoke_log_fd.flush()

    try:
        smoke_process = subprocess.Popen(
            cmd,
            stdout=smoke_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
    except Exception as e:
        try:
            smoke_log_fd.write(f"\n!!! Failed to launch smoke test: {e}\n")
            smoke_log_fd.flush()
        except Exception:
            pass
        _close_smoke_log_fd()
        smoke_process = None
        raise HTTPException(status_code=500, detail=f"Failed to launch smoke test: {e}")

    try:
        _smoke_pid_file().write_text(str(smoke_process.pid), encoding="utf-8")
    except Exception as e:
        _terminate_child(smoke_process.pid)
        _close_smoke_log_fd()
        raise HTTPException(
            status_code=500, detail=f"Failed to record smoke PID: {e}"
        )

    return {
        "status": "started",
        "mode": "smoke",
        "pid": smoke_process.pid,
        "output_dir": output_dir,
        "results_dir": str(smoke_results_dir),
    }


_TRAINER_VALUE_ARGS = (
    "config",
    "base_model",
    "dataset_name",
    "epochs",
    "batch_size",
    "grad_accum",
    "learning_rate",
    "warmup_steps",
    "lr_scheduler_type",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "eval_batch_size",
    "max_benchmark_samples",
    "preflight_samples",
    "num_proc",
    "telephony_prob",
    "telephony_mode",
    "telephony_seed",
    "phone_seed",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "max_train_samples",
    "max_eval_samples",
    "prep_cache_dir",
)
_TRAINER_LIST_ARGS = ("telephony_profiles",)
_TRAINER_BOOL_ARGS = (
    "telephony_use_ffmpeg",
    "phone_benchmark",
    "run_post_test_eval",
    "run_epoch_benchmark",
    "enable_telemetry",
    "gradient_checkpointing",
    "auto_resume",
    "fresh_start",
    "overwrite_preflight",
    "reuse_prep",
    "overwrite_prep_cache",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _build_trainer_command(
    *,
    trainer_args: Dict[str, Any],
    run_dir: Path,
    output_dir: str,
    results_dir: str,
    preflight_cache_dir: Path,
    post_eval_only: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> List[str]:
    """Rebuild the trainer command from a run's stored argument mapping."""
    cmd = [
        sys.executable,
        "-u",
        str(_trainer_script_path()),
        "--run_dir", str(run_dir),
        "--preflight_cache_dir", str(preflight_cache_dir),
        "--output_dir", str(output_dir),
        "--results_dir", str(results_dir),
    ]
    for key in _TRAINER_VALUE_ARGS:
        value = trainer_args.get(key)
        if value is None or value == "":
            continue
        cmd += [f"--{key}", str(value)]
    for key in _TRAINER_LIST_ARGS:
        value = trainer_args.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            value = ",".join(str(item) for item in value)
        cmd += [f"--{key}", str(value)]
    for key in _TRAINER_BOOL_ARGS:
        if key not in trainer_args:
            continue
        cmd.append(f"--{key}" if _as_bool(trainer_args.get(key)) else f"--no-{key}")
    if resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", str(resume_from_checkpoint)]
    if post_eval_only:
        cmd.append("--post_eval_only")
    return cmd


def _normalize_start_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a new-run payload (legacy rules unchanged)."""
    try:
        epochs_i = int(config.get("epochs", 2))
        batch_i = int(config.get("batch_size", 8))
        accum_i = int(config.get("grad_accum", 4))
        eval_steps_i = int(config.get("eval_steps", 1000))
        lr_f = float(config.get("learning_rate", 1e-4))
        eval_bs_i = int(config.get("eval_batch_size", 16))
        max_bench = config.get("max_benchmark_samples", 1000)
        max_bench_i = int(max_bench) if max_bench not in (None, "") else 1000
        # Preprocessing workers for datasets.map (fork-based). Default 8.
        num_proc_i = int(config.get("num_proc", 8))
        tele_f = float(config.get("telephony_prob", 0.5))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="epochs/batch_size/grad_accum/eval_batch_size/num_proc must be integers; learning_rate/telephone_prob numeric.")
    if not (1 <= epochs_i <= 20):
        raise HTTPException(status_code=400, detail="epochs must be 1..20.")
    if not (1 <= batch_i <= 64):
        raise HTTPException(status_code=400, detail="batch_size must be 1..64.")
    if not (1 <= accum_i <= 16):
        raise HTTPException(status_code=400, detail="grad_accum must be 1..16.")
    if not (1e-7 <= lr_f <= 1e-2):
        raise HTTPException(status_code=400, detail="learning_rate must be 1e-7..1e-2.")
    if not (1 <= eval_bs_i <= 64):
        raise HTTPException(status_code=400, detail="eval_batch_size must be 1..64.")
    if not (0 <= max_bench_i <= 10000):
        raise HTTPException(status_code=400, detail="max_benchmark_samples must be 0..10000 (0=full).")
    if not (1 <= num_proc_i <= 16):
        raise HTTPException(status_code=400, detail="num_proc must be 1..16.")
    if not (0.0 <= tele_f <= 1.0):
        raise HTTPException(status_code=400, detail="telephony_prob must be 0..1.")
    mode = str(config.get("telephony_mode", "paired")).strip().lower()
    if mode not in ("paired", "disjoint", "clean"):
        raise HTTPException(
            status_code=400,
            detail="telephony_mode must be one of paired, disjoint, clean.",
        )
    if not (1 <= eval_steps_i <= 10000):
        raise HTTPException(status_code=400, detail="eval_steps must be 1..10000.")
    ckpt_raw = config.get("gradient_checkpointing", False)
    gradient_ckpt = (
        str(ckpt_raw).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(ckpt_raw, str)
        else bool(ckpt_raw)
    )
    if batch_i > 8 and not gradient_ckpt:
        raise HTTPException(status_code=400, detail="batch_size above 8 needs gradient_checkpointing on; without it batch 16 runs out of memory on a 48 GB GPU.")
    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar")).strip() or "shekar-ai/Neyshekar"
    if len(dataset_name) > 128 or any(c in dataset_name for c in ";&|$` \n"):
        raise HTTPException(status_code=400, detail="Invalid dataset_name.")
    # Resume controls: default auto-resume ON so pod restarts continue from
    # checkpoints. Send {"fresh_start": true} for a clean run, or
    # {"resume_from_checkpoint": "<path|last>"} to force a checkpoint.
    fresh_start = bool(config.get("fresh_start", config.get("fresh", False)))
    resume_explicit = str(config.get("resume_from_checkpoint", "") or "").strip()
    auto_resume = config.get("auto_resume", True)
    auto_resume_on = (_as_bool(auto_resume) if isinstance(auto_resume, str) else bool(auto_resume))
    if fresh_start:
        auto_resume_on = False
    return {
        "epochs": epochs_i,
        "batch_size": batch_i,
        "grad_accum": accum_i,
        "eval_steps": eval_steps_i,
        "learning_rate": lr_f,
        "eval_batch_size": eval_bs_i,
        "max_benchmark_samples": max_bench_i,
        "num_proc": num_proc_i,
        "telephony_prob": tele_f,
        "telephony_mode": mode,
        "gradient_checkpointing": gradient_ckpt,
        "dataset_name": dataset_name,
        "fresh_start": fresh_start,
        "resume_explicit": resume_explicit,
        "auto_resume": auto_resume_on,
    }


def _write_initial_telemetry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "initializing",
        "step": 0,
        "max_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "learning_rate": 0.0,
        "speed_s_per_step": 0.0,
        "eta_seconds": 0,
    }), encoding="utf-8")


def _spawn_trainer(
    *,
    run_id: str,
    run_dir: Path,
    paths: Dict[str, Any],
    trainer_args: Dict[str, Any],
    post_eval_only: bool,
    reason: str,
    response_config: Dict[str, Any],
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Spawn one trainer attempt and record it in the run ledger."""
    global training_process, training_log_fd, last_exit_code, halt_requested
    global training_run_id

    file_paths = _run_paths({"run_id": run_id, "paths": paths})
    log_file = Path(file_paths["log_file"])
    pid_file = Path(file_paths["pid_file"])
    stop_signal = Path(file_paths["stop_signal_file"])
    telemetry_file = Path(file_paths["telemetry_file"])

    # A resumed run must not consume a stale halt request.
    try:
        stop_signal.unlink(missing_ok=True)
    except OSError:
        pass

    _close_log_fd()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    training_log_fd = open(log_file, "w", encoding="utf-8")
    training_log_fd.write(f"=== Persian ASR Training Job Launched at {time.ctime()} ===\n")
    training_log_fd.write(f"Run: {run_id} ({reason})\n")
    training_log_fd.flush()

    last_exit_code = None
    halt_requested = False
    training_run_id = None
    _write_initial_telemetry(telemetry_file)

    cmd = _build_trainer_command(
        trainer_args=trainer_args,
        run_dir=run_dir,
        output_dir=str(paths["output_dir"]),
        results_dir=str(paths["results_dir"]),
        preflight_cache_dir=_preflight_cache_dir_for_paths(paths),
        post_eval_only=post_eval_only,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    training_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    training_log_fd.flush()

    training_process = None
    try:
        training_process = subprocess.Popen(
            cmd,
            stdout=training_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env={**os.environ, "ASR_COCKPIT_SPAWNED": "1"},
            preexec_fn=os.setsid if sys.platform != "win32" else None
        )
    except Exception as e:
        try:
            training_log_fd.write(f"\n!!! Failed to launch training process: {e}\n")
            training_log_fd.flush()
        except Exception:
            pass
        _close_log_fd()
        try:
            run_state.set_status(
                run_dir, "failed", failed_stage="training", last_error=str(e)
            )
        except Exception:
            pass
        _invalidate_refresh_cache()
        raise HTTPException(status_code=500, detail=f"Failed to launch training: {e}")

    training_run_id = run_id

    # Record the attempt before any other bookkeeping so the trainer's
    # callback (which may start immediately) sees the correct attempt number.
    try:
        run_state.start_attempt(run_dir, training_process.pid, reason)
    except Exception as e:
        try:
            training_log_fd.write(f"\n!!! Failed to record run attempt: {e}\n")
            training_log_fd.flush()
        except Exception:
            pass
        _terminate_child(training_process.pid)
        _close_log_fd()
        training_run_id = None
        try:
            run_state.set_status(
                run_dir, "failed", failed_stage="training", last_error=str(e)
            )
        except Exception:
            pass
        _invalidate_refresh_cache()
        raise HTTPException(
            status_code=500, detail=f"Failed to record run attempt: {e}"
        )

    try:
        pid_file.write_text(str(training_process.pid), encoding="utf-8")
    except Exception:
        pass
    run_state.set_status(run_dir, "running", stop_requested_at=None)
    _invalidate_refresh_cache()

    return {
        "status": "started",
        "run_id": run_id,
        "pid": training_process.pid,
        "config": response_config,
        "paths": {key: str(value) for key, value in paths.items()},
        "storage": run_state.storage_info(RUNS_DIR),
    }


def _shared_preflight_cache_dir() -> Path:
    """Shared preflight cache root; fake runs never touch the real cache."""
    name = "_shared-fake" if _fake_trainer_enabled() else "_shared"
    return RUNS_DIR / name


def _preflight_cache_dir_for_paths(paths: Dict[str, Any]) -> Path:
    raw = (paths or {}).get("preflight_cache")
    if raw:
        p = Path(str(raw))
        if p.name == "preflight.json":
            return p.parent.parent
        return p
    return _shared_preflight_cache_dir()


def _run_preflight_file(run: Optional[Dict[str, Any]]) -> Path:
    """Run-local preflight copy, else the shared cache, else the legacy file."""
    if not run:
        return PREFLIGHT_FILE
    run_copy = _run_dir(run) / "baselines" / "preflight.json"
    if run_copy.is_file():
        return run_copy
    shared = (
        _preflight_cache_dir_for_paths(run.get("paths") or {})
        / "baselines"
        / "preflight.json"
    )
    if shared.is_file():
        return shared
    return PREFLIGHT_FILE


def _start_new_run(config: Dict[str, Any]) -> Dict[str, Any]:
    _check_audio_backend_or_400()
    if _any_job_running():
        raise HTTPException(status_code=400, detail="Training or smoke test is already running.")

    results_cfg = (config.get("results_dir", "") or "")
    if str(results_cfg).strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "results_dir is managed per run for a cockpit run; "
                "omit it or relocate the runs root with ASR_RUNS_DIR."
            ),
        )

    values = _normalize_start_config(config)
    run_id = run_state.new_run_id()
    run_dir = RUNS_DIR / run_id
    output_dir = str(run_dir / "output")
    explicit_output = str(config.get("output_dir", "") or "").strip()
    if explicit_output:
        output_dir = _sanitize_output_dir(explicit_output)

    try:
        smoke_output = _abs_output_path(_smoke_output_dir()).resolve()
    except Exception:
        smoke_output = None
    if smoke_output is not None and _abs_output_path(output_dir).resolve() == smoke_output:
        raise HTTPException(status_code=400, detail="output_dir must not be the smoke-test directory.")

    preflight_cache_dir = _shared_preflight_cache_dir()
    preflight_cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_dir": str(run_dir),
        "output_dir": output_dir,
        "results_dir": str(run_dir),
        "merged_dir": str(run_dir / "merged"),
        "preflight_cache": str(preflight_cache_dir / "baselines" / "preflight.json"),
    }
    trainer_args = {
        "base_model": "BuzzASR/persian",
        "dataset_name": values["dataset_name"],
        "output_dir": output_dir,
        "results_dir": str(run_dir),
        "epochs": values["epochs"],
        "batch_size": values["batch_size"],
        "grad_accum": values["grad_accum"],
        "learning_rate": values["learning_rate"],
        "warmup_steps": 500,
        "lr_scheduler_type": "cosine",
        "eval_steps": values["eval_steps"],
        "save_steps": values["eval_steps"],
        "save_total_limit": 2,
        "eval_batch_size": values["eval_batch_size"],
        "max_benchmark_samples": values["max_benchmark_samples"],
        "preflight_samples": values["max_benchmark_samples"],
        "num_proc": values["num_proc"],
        "telephony_prob": values["telephony_prob"],
        "telephony_mode": values["telephony_mode"],
        "telephony_profiles": ["customer"],
        "telephony_seed": 42,
        "telephony_use_ffmpeg": True,
        "phone_benchmark": True,
        "phone_seed": 1234,
        "gradient_checkpointing": values["gradient_checkpointing"],
        "run_post_test_eval": True,
        "run_epoch_benchmark": True,
        "enable_telemetry": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "auto_resume": values["auto_resume"],
        "fresh_start": values["fresh_start"],
        "run_dir": str(run_dir),
    }
    if values["resume_explicit"]:
        trainer_args["resume_from_checkpoint"] = values["resume_explicit"]
    run_config = {
        "base_model": "BuzzASR/persian",
        "dataset_name": values["dataset_name"],
        "epochs": values["epochs"],
        "batch_size": values["batch_size"],
        "grad_accum": values["grad_accum"],
        "learning_rate": values["learning_rate"],
        "eval_batch_size": values["eval_batch_size"],
        "eval_steps": values["eval_steps"],
        "save_steps": values["eval_steps"],
        "max_benchmark_samples": values["max_benchmark_samples"],
        "preflight_samples": values["max_benchmark_samples"],
        "num_proc": values["num_proc"],
        "telephony_prob": values["telephony_prob"],
        "telephony_mode": values["telephony_mode"],
        "gradient_checkpointing": values["gradient_checkpointing"],
        "auto_resume": values["auto_resume"],
        "fresh_start": values["fresh_start"],
        "resume_from_checkpoint": values["resume_explicit"] or None,
        "output_dir": output_dir,
        "results_dir": str(run_dir),
    }
    run = run_state.create_run(
        RUNS_DIR,
        config=run_config,
        trainer_args=trainer_args,
        paths=paths,
        run_id=run_id,
        source="fake" if _fake_trainer_enabled() else "cockpit",
    )
    response_config = {
        "epochs": str(values["epochs"]),
        "batch_size": str(values["batch_size"]),
        "learning_rate": str(values["learning_rate"]),
        "grad_accum": str(values["grad_accum"]),
        "eval_steps": str(values["eval_steps"]),
        "gradient_checkpointing": values["gradient_checkpointing"],
        "num_proc": str(values["num_proc"]),
        "telephony_prob": str(values["telephony_prob"]),
        "telephony_mode": values["telephony_mode"],
        "output_dir": output_dir,
        "results_dir": str(run_dir),
        "auto_resume": values["auto_resume"],
        "fresh_start": values["fresh_start"],
        "resume_from_checkpoint": values["resume_explicit"] or None,
    }
    return _spawn_trainer(
        run_id=run_id,
        run_dir=run_dir,
        paths=run["paths"],
        trainer_args=trainer_args,
        post_eval_only=False,
        reason="initial",
        response_config=response_config,
        resume_from_checkpoint=values["resume_explicit"] or None,
    )


def _start_resume_run(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    if not run_state.run_id_valid(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    manifest_dir = RUNS_DIR / run_id
    run = run_state.load_run(manifest_dir)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    # Reconcile a dead running/starting attempt before deciding whether the
    # run is resumable; completed runs are still rejected below.
    _refresh_runs()
    run = run_state.load_run(manifest_dir) or run
    status = str(run.get("status") or "")
    if status not in ("interrupted", "stopped", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} cannot resume from status '{status or 'unknown'}'.",
        )

    _check_audio_backend_or_400()
    if _any_job_running():
        raise HTTPException(status_code=400, detail="Training or smoke test is already running.")

    run_dir = _run_dir(run)
    stored_paths = dict(run.get("paths") or {})
    output_dir = str(stored_paths.get("output_dir") or (run_dir / "output"))
    results_dir = str(stored_paths.get("results_dir") or run_dir)
    preflight_cache_dir = _preflight_cache_dir_for_paths(stored_paths)
    paths = {
        "run_dir": str(run_dir),
        "output_dir": output_dir,
        "results_dir": results_dir,
        "merged_dir": str(stored_paths.get("merged_dir") or (run_dir / "merged")),
        "preflight_cache": str(preflight_cache_dir / "baselines" / "preflight.json"),
    }

    trainer_args = dict(run.get("trainer_args") or {})
    # Pre-change runs (and malformed null/empty values) were disjoint; never
    # let a resumed checkpoint fall through to the trainer's paired default.
    if not trainer_args.get("telephony_mode"):
        trainer_args["telephony_mode"] = "disjoint"
    trainer_args.pop("resume_from_checkpoint", None)
    trainer_args.update({
        "run_dir": str(run_dir),
        "output_dir": output_dir,
        "results_dir": results_dir,
        "fresh_start": False,
        "auto_resume": True,
    })

    stages = {str(item.get("name")): item for item in run.get("stages") or []}
    training_status = str((stages.get("training") or {}).get("status") or "pending")
    post_eval_status = str((stages.get("post_eval") or {}).get("status") or "pending")
    post_eval_only = training_status == "completed" and post_eval_status != "completed"

    response_config = {
        "epochs": str(trainer_args.get("epochs", "")),
        "batch_size": str(trainer_args.get("batch_size", "")),
        "learning_rate": str(trainer_args.get("learning_rate", "")),
        "grad_accum": str(trainer_args.get("grad_accum", "")),
        "eval_steps": str(trainer_args.get("eval_steps", "")),
        "gradient_checkpointing": _as_bool(trainer_args.get("gradient_checkpointing", False)),
        "num_proc": str(trainer_args.get("num_proc", "")),
        "telephony_prob": str(trainer_args.get("telephony_prob", "")),
        "telephony_mode": trainer_args["telephony_mode"],
        "output_dir": output_dir,
        "results_dir": results_dir,
        "auto_resume": True,
        "fresh_start": False,
        "resume_from_checkpoint": None,
    }
    preflight_cache_dir.mkdir(parents=True, exist_ok=True)
    return _spawn_trainer(
        run_id=run_id,
        run_dir=run_dir,
        paths=paths,
        trainer_args=trainer_args,
        post_eval_only=post_eval_only,
        reason="resume",
        response_config=response_config,
        resume_from_checkpoint=None,
    )


def _start_merge(run_id: str) -> Dict[str, Any]:
    """Start the optional merge stage for a completed-training run."""
    global merge_process, merge_log_fd, merge_run_id

    if not run_state.run_id_valid(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    run = run_state.load_run(RUNS_DIR / run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    run_dir = _run_dir(run)

    # Apply any known child exits and a possibly-dead merge attempt before
    # deciding whether a new merge may start.
    training_busy = _training_running()
    merge_busy = _merge_running(force=True)
    _refresh_runs()
    run = run_state.load_run(RUNS_DIR / run_id) or run
    run_dir = _run_dir(run)

    stages = {
        str(item.get("name")): item for item in run.get("stages") or []
    }
    training_status = str((stages.get("training") or {}).get("status") or "pending")
    if training_status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Merge requires the training stage to be completed.",
        )
    if not _run_adapter_exists(run):
        raise HTTPException(
            status_code=400,
            detail="Merge requires best_adapter/adapter_config.json.",
        )

    # R12: only the target's own still-exiting trainer may overlap a merge,
    # and only when every target stage is terminal. Any other live trainer,
    # including a legacy PID-file process with no tracked child, blocks it.
    target_trainer_live = bool(training_busy and training_run_id == run_id)
    if training_busy and not target_trainer_live:
        raise HTTPException(
            status_code=400, detail="Training, smoke test, or merge is already running."
        )
    if target_trainer_live and not _merge_stages_terminal(run):
        raise HTTPException(
            status_code=400, detail="Training, smoke test, or merge is already running."
        )
    if _smoke_running() or merge_busy:
        raise HTTPException(
            status_code=400, detail="Training, smoke test, or merge is already running."
        )
    active = _active_run()
    if (
        active is not None
        and str(active.get("run_id")) != run_id
        and str(active.get("status")) in ("starting", "running")
    ):
        raise HTTPException(
            status_code=400, detail="Training, smoke test, or merge is already running."
        )
    # Any target stage still running (e.g. post-eval) blocks the merge.
    if any(
        str(stage.get("status")) == "running"
        for stage in run.get("stages") or []
    ):
        raise HTTPException(
            status_code=400, detail="Training, smoke test, or merge is already running."
        )

    run_config = run.get("config") or {}
    trainer_args = run.get("trainer_args") or {}
    base_model = str(
        run_config.get("base_model")
        or trainer_args.get("base_model")
        or "BuzzASR/persian"
    )
    adapter_dir = _run_output_dir(run) / "best_adapter"
    run_dir_path, merged_dir, log_file, pid_file = _merge_paths(run)
    script_name = "dev_fake_merge.py" if _fake_trainer_enabled() else "merge_lora.py"
    hf_repo = ""
    hf_section = run.get("hf")
    if isinstance(hf_section, dict):
        hf_repo = str(hf_section.get("repo_id") or "").strip()
    if not hf_repo:
        hf_repo = (os.environ.get("ASR_HF_RUN_REPO", "") or "").strip()
    cmd = [
        sys.executable,
        "-u",
        str(BASE_DIR / "scripts" / script_name),
        "--base_model", base_model,
        "--adapter_path", str(adapter_dir),
        "--output_dir", str(merged_dir),
        "--run_dir", str(run_dir_path),
    ]
    if hf_repo:
        cmd.extend(["--hf_run_repo", hf_repo])

    run_dir_path.mkdir(parents=True, exist_ok=True)
    # A crashed retry must never read the previous attempt's PID.
    _unlink_quietly(pid_file)
    _close_merge_log_fd()
    try:
        merge_log_fd = open(log_file, "w", encoding="utf-8")
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to open merge log: {e}"
        )
    merge_log_fd.write(f"=== LoRA merge started at {time.ctime()} ===\n")
    merge_log_fd.write(f"Run: {run_id}\n")
    merge_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    merge_log_fd.flush()

    def _reset_merge_stage(current: dict) -> dict:
        stage = next(
            (
                item
                for item in current.get("stages") or []
                if item.get("name") == "merge"
            ),
            None,
        )
        if stage is not None:
            for key in ("exit_code", "log_tail"):
                stage.pop(key, None)
        current["merge_pid"] = None
        return current

    run_state.update_run(run_dir, _reset_merge_stage)
    run_state.set_stage(run_dir, "merge", "running", detail="merging")
    _invalidate_refresh_cache()
    # The merge stage is running from here on: keep a non-forced scan from
    # reporting a stale False (or reaping the stage before the PID lands).
    _note_merge_scan(True)

    merge_process = None
    try:
        merge_process = subprocess.Popen(
            cmd,
            stdout=merge_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
    except Exception as e:
        try:
            merge_log_fd.write(f"\n!!! Failed to launch merge: {e}\n")
            merge_log_fd.flush()
        except Exception:
            pass
        _close_merge_log_fd()
        run_state.set_stage(
            run_dir,
            "merge",
            "failed",
            detail=f"Failed to launch merge: {e}",
            exit_code=None,
            log_tail=_read_log_tail(log_file),
        )
        run_state.update_run(
            run_dir, lambda current: {**current, "merge_pid": None}
        )
        _note_merge_scan(False)
        raise HTTPException(status_code=500, detail=f"Failed to launch merge: {e}")

    merge_run_id = run_id
    pid = merge_process.pid
    try:
        pid_file.write_text(str(pid), encoding="utf-8")
    except OSError as e:
        _terminate_child(pid)
        merge_process = None
        merge_run_id = None
        _close_merge_log_fd()
        run_state.set_stage(
            run_dir,
            "merge",
            "failed",
            detail=f"Failed to record merge PID: {e}",
            exit_code=None,
        )
        run_state.update_run(
            run_dir, lambda current: {**current, "merge_pid": None}
        )
        _note_merge_scan(False)
        raise HTTPException(
            status_code=500, detail=f"Failed to record merge PID: {e}"
        )

    run_state.update_run(
        run_dir, lambda current: {**current, "merge_pid": int(pid)}
    )
    _invalidate_refresh_cache()
    # The merge child is live: a non-forced scan must report it as running.
    _note_merge_scan(True)
    return {
        "status": "started",
        "run_id": run_id,
        "pid": pid,
        "merged_dir": str(merged_dir),
        "log_file": str(log_file),
    }


def _start_run(
    *,
    config: Dict[str, Any],
    resume_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = config or {}
    _check_auth(config)
    if resume_run_id is not None:
        return _start_resume_run(config, resume_run_id)
    return _start_new_run(config)


@app.post("/api/train/start")
async def start_training(config: Dict[str, Any] = None):
    return _start_run(config=config or {})


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, config: Dict[str, Any] = None):
    return _start_run(config=config or {}, resume_run_id=run_id)


def _run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    try:
        step_count = int(run.get("step_count") or 0)
    except (TypeError, ValueError):
        step_count = 0
    return {
        "run_id": run.get("run_id"),
        "created_at": run.get("created_at"),
        "status": run.get("status"),
        "current_stage": run.get("current_stage"),
        "step_count": step_count,
        "source": run.get("source"),
        "output_dir": str(_run_output_dir(run)),
    }


@app.get("/api/runs")
async def list_runs(limit: int = 20):
    summary = _refresh_runs_cached()
    try:
        capped = max(0, int(limit))
    except (TypeError, ValueError):
        capped = 20
    capped = min(capped, 200)
    runs = summary.get("runs") or []
    return {
        "runs": [_run_summary(run) for run in runs[:capped]],
        "storage": run_state.storage_info(RUNS_DIR),
    }


@app.get("/api/runs/latest")
async def latest_run():
    _refresh_runs_cached()
    return _run_payload(_current_run())


@app.get("/api/runs/{run_id}/steps")
async def get_run_steps(run_id: str, tail: int = 2000):
    if not run_state.run_id_valid(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    run = run_state.load_run(RUNS_DIR / run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        count = int(tail)
    except (TypeError, ValueError):
        count = 2000
    count = max(0, min(count, 10000))
    steps_dir = _run_paths(run)["telemetry_file"].parent
    return {
        "run_id": run_id,
        "total": run_state.step_count(steps_dir),
        "steps": run_state.read_steps(steps_dir, tail=count),
    }


@app.get("/api/runs/{run_id}/logs")
async def get_run_logs(run_id: str, tail: int = 80):
    """Read the selected run's log without touching the live-run buffer."""
    if not run_state.run_id_valid(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    run = run_state.load_run(RUNS_DIR / run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    log_file = _run_paths(run)["log_file"]
    if not log_file.exists():
        return {"logs": ["Log buffer empty. Training has not started."]}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}
    try:
        count = int(tail)
    except (TypeError, ValueError):
        count = 80
    count = max(0, min(count, 2000))
    selected = [] if count == 0 else lines[-count:]
    return {"logs": [line.rstrip() for line in selected]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    if not run_state.run_id_valid(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    _refresh_runs_cached()
    run = run_state.load_run(RUNS_DIR / run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _run_payload(run)


@app.post("/api/runs/{run_id}/merge")
async def merge_run(run_id: str, config: Dict[str, Any] = None):
    _check_auth(config)
    return _start_merge(run_id)


@app.post("/api/hf/restore")
async def start_hf_restore(payload: Dict[str, Any] = None):
    payload = payload or {}
    _check_auth(payload)
    return _start_hf_restore(
        str(payload.get("run_id") or ""),
        str(payload.get("repo") or "").strip() or None,
        include_merged=_as_bool(payload.get("include_merged", False)),
    )


@app.get("/api/hf/restore/status")
async def get_hf_restore_status():
    _hf_restore_running()
    status = _hf_restore_status()
    data: Dict[str, Any] = {
        "state": "idle",
        "run_id": None,
        "message": "",
        "updated_at": None,
    }
    data.update(status)
    data["message"] = _redact_secret_text(data.get("message"))
    data["log_tail"] = [
        _redact_secret_text(line)
        for line in _read_log_tail(_hf_restore_log_file(), 40)
    ]
    return data


@app.get("/api/hf/config")
async def get_hf_config():
    try:
        enabled = importlib.util.find_spec("huggingface_hub") is not None
    except Exception:
        enabled = False
    return {
        "enabled": enabled,
        "repo_id": (os.environ.get("ASR_HF_RUN_REPO", "") or "").strip(),
        "runs_root": str(RUNS_DIR),
    }


@app.get("/api/hf/runs")
def list_hf_runs(repo: Optional[str] = None):
    # Deliberately synchronous: FastAPI runs a plain ``def`` path operation in
    # its threadpool, so the blocking HF listing cannot stall the event loop
    # (and therefore the 1s WebSocket telemetry tick).
    repo_id = str(repo or "").strip() or (
        os.environ.get("ASR_HF_RUN_REPO", "") or ""
    ).strip()
    if not repo_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "no Hugging Face repo configured; set ASR_HF_RUN_REPO or "
                "pass repo"
            ),
        )
    try:
        hf_sync.validate_repo_id(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        now = time.monotonic()
    except Exception:
        now = time.time()
    cached = _hf_runs_cache.get(repo_id)
    if cached is not None and (now - cached[0]) < _HF_RUNS_CACHE_TTL_SECONDS:
        return {"runs": cached[1], "repo_id": repo_id}

    try:
        runs = hf_sync.list_remote_runs(hf_sync.make_api(), repo_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=_redact_secret_text(
                f"Could not list Hugging Face runs: {exc}"
            )[:300],
        ) from exc
    if not isinstance(runs, list):
        runs = list(runs or [])
    _hf_runs_cache[repo_id] = (now, runs)
    return {"runs": runs, "repo_id": repo_id}


def _escalation_run(run_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if run_dir is None:
        return None
    try:
        return run_state.load_run(run_dir)
    except Exception:
        return None


def _run_is_terminal(run: Optional[Dict[str, Any]]) -> bool:
    return bool(run) and str(run.get("status") or "") in (
        "completed",
        "stopped",
        "failed",
        "interrupted",
    )


def _halt_target_finished(
    telemetry_file: Optional[Path],
    run: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when the halt target already finished.

    A ledger run decides from its own stage/status: telemetry ``completed`` is
    written before post-eval, so it must not end the grace escalation while the
    run is still active. Only the legacy (no-ledger) path trusts telemetry.
    """
    if run is not None:
        return _run_is_terminal(run)
    telemetry_path = Path(telemetry_file) if telemetry_file is not None else TELEMETRY_FILE
    return read_json_safely(telemetry_path, {}).get("status") in (
        "stopped",
        "completed",
    )


async def _escalate_halt(
    proc: subprocess.Popen,
    telemetry_file: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> None:
    """Wait for the trainer to checkpoint and exit; SIGTERM only after the grace period."""
    deadline = time.time() + HALT_GRACE_SECONDS
    while proc.poll() is None:
        if time.time() >= deadline:
            run = _escalation_run(run_dir)
            if run is not None:
                if _run_is_terminal(run):
                    return
                current_pid = _run_pid(run)
                if current_pid is not None and int(current_pid) != int(proc.pid):
                    return
            elif _halt_target_finished(telemetry_file):
                return
            if sys.platform == "win32":
                try:
                    proc.terminate()
                except Exception:
                    pass
            else:
                _terminate_child(proc.pid)
            return
        await asyncio.sleep(2)


async def _escalate_halt_pid(
    pid: int,
    telemetry_file: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> None:
    """Escalate against a run PID recovered from the ledger after a restart."""
    deadline = time.time() + HALT_GRACE_SECONDS
    while _is_process_running(pid):
        if time.time() >= deadline:
            run = _escalation_run(run_dir)
            if run is not None:
                if _run_is_terminal(run):
                    return
                current_pid = _run_pid(run)
                if current_pid is None or int(current_pid) != int(pid):
                    # The run no longer owns this PID; do not signal a reuse.
                    return
                if str(run.get("status") or "") not in ("starting", "running"):
                    return
            elif _halt_target_finished(telemetry_file):
                return
            _terminate_child(pid)
            return
        await asyncio.sleep(2)


@app.post("/api/train/stop")
async def stop_training(payload: Dict[str, Any] = None):
    global halt_requested
    _check_auth(payload)
    _refresh_runs()
    active = _active_run()

    if active is not None:
        run_id = str(active.get("run_id"))
        run_dir = _run_dir(active)
        paths = _run_paths(active)
        try:
            paths["stop_signal_file"].parent.mkdir(parents=True, exist_ok=True)
            paths["stop_signal_file"].write_text("stop", encoding="utf-8")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to write stop signal: {e}"
            )

        def mark_stop(run: dict) -> dict:
            run["stop_requested_at"] = time.time()
            current = run.get("current_stage")
            for stage in run.get("stages") or []:
                if (
                    stage.get("name") == current
                    and stage.get("status") == "running"
                ):
                    stage["detail"] = "halt requested"
            return run

        try:
            run_state.update_run(run_dir, mark_stop)
        except FileNotFoundError:
            pass
        _invalidate_refresh_cache()
        halt_requested = True
        pid = _run_pid(active)
        proc = None
        if training_process is not None:
            try:
                if pid is None or int(training_process.pid) == int(pid):
                    proc = training_process
            except (TypeError, ValueError):
                proc = None
        if proc is not None:
            asyncio.create_task(
                _escalate_halt(proc, paths["telemetry_file"], run_dir)
            )
        elif pid is not None:
            try:
                asyncio.create_task(
                    _escalate_halt_pid(
                        int(pid), paths["telemetry_file"], run_dir
                    )
                )
            except (TypeError, ValueError):
                pass
        return {"status": "stopping", "run_id": run_id}

    # Legacy fallback: no ledger run, keep the pre-run behavior.
    if not _training_running():
        return {"status": "not_running"}
    halt_requested = True
    with open(STOP_SIGNAL_FILE, "w") as f:
        f.write("stop")
    if training_process is not None:
        asyncio.create_task(_escalate_halt(training_process))
    return {"status": "stopping"}


@app.websocket("/ws/live")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Poll status, telemetry and send to client (1s cadence to save bandwidth)
            training_live = _training_running()
            _merge_running()
            _refresh_runs_cached()
            run = _current_run()
            paths = _run_paths(run)
            telemetry = read_json_safely(paths["telemetry_file"], {})
            benchmarks = _get_benchmark_rows(
                benchmarks_file=paths["benchmarks_file"],
                preflight_file=_run_preflight_file(run),
            )
            loss_hist = read_json_safely(paths["loss_history_file"], [])

            is_running = training_live or _active_run_process_running()
            telemetry = _derive_status(telemetry, is_running, run)
            telemetry["process_running"] = is_running
            # WS clients otherwise never get a fresh GPU reading (the trainer
            # only rewrites telemetry on steps); mirror /api/status behavior.
            telemetry["gpu"] = telemetry.get("gpu") or get_gpu_info()

            payload = {
                "telemetry": telemetry,
                "benchmarks": benchmarks,
                "loss_history": loss_hist[-100:],  # last 100 points for crisp SVG chart
                "run": run,
                "steps": _steps_summary(run, telemetry),
                "storage": run_state.storage_info(RUNS_DIR),
                "adapter_exists": _run_adapter_exists(run),
                "hf": (run or {}).get("hf") or {},
                "hf_sync": read_json_safely(
                    _run_dir(run) / "hf_sync.json", {}
                ),
                "timestamp": time.time()
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    # Default to RunPod HTTP proxy port 7860
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f" Persian ASR RunPod Telemetry Cockpit")
    print(f" Listening on http://{host}:{port}")
    print(f"=======================================================\n")
    uvicorn.run("server.app:app", host=host, port=port, reload=False)
