#!/usr/bin/env python3
"""GPU-free stand-in for ``scripts/train_lora.py``.

The cockpit server can spawn this script when ``ASR_FAKE_TRAINER=1`` so the
whole run lifecycle (ledger, stage transitions, telemetry, stop signal) can be
developed on a CPU-only machine. It writes the same run files as the real
trainer but never imports torch, transformers, peft, or CUDA code.

Environment:
    ASR_FAKE_STEPS        optimizer steps to simulate (default 200)
    ASR_FAKE_STEP_DELAY   seconds slept per step (default 0.05)
    ASR_FAKE_FAIL_STAGE   preflight|training|post_eval to fail on purpose
    ASR_FAKE_HANG=1       idle in training until stop_training.signal appears

The merge stage is deliberately left ``pending``: the server owns it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_run_state():
    """Load ``src/run_state.py`` without importing the ``src`` package.

    ``src/__init__.py`` eagerly imports the data/collator modules, which import
    torch when the ML stack is installed. The fake trainer must stay importable
    (and GPU-free) without that stack, so execute the stdlib-only ledger module
    directly.
    """
    path = _ROOT / "src" / "run_state.py"
    spec = importlib.util.spec_from_file_location("asr_fake_run_state", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load run_state from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_state = _load_run_state()

FAIL_STAGES = ("preflight", "training", "post_eval")
MAX_LOSS_HISTORY = 500
PREFLIGHT_SAMPLES = 1000

_PREFLIGHT_MODELS = {
    "buzzasr_base": {
        "name": "BuzzASR Base",
        "epoch": "BuzzASR Base (preflight, 1,000)",
        "source": "preflight",
        "is_reference": False,
        "total_samples": PREFLIGHT_SAMPLES,
        "normalized_wer": 17.2,
        "normalized_cer": 5.3,
        "diagnostic_wer": 16.4,
        "formal_wer": 14.1,
        "informal_wer": 22.0,
        "phone_normalized_wer": 24.1,
        "phone_normalized_cer": 7.8,
        "phone_formal_wer": 20.0,
        "phone_informal_wer": 30.0,
    },
    "whisper_v3": {
        "name": "Whisper",
        "epoch": "Whisper (preflight)",
        "source": "preflight",
        "is_reference": True,
        "total_samples": PREFLIGHT_SAMPLES,
        "normalized_wer": 28.5,
        "normalized_cer": 8.9,
        "diagnostic_wer": 26.1,
        "formal_wer": 24.0,
        "informal_wer": 36.8,
        "phone_normalized_wer": 35.0,
        "phone_normalized_cer": 12.0,
        "phone_formal_wer": 30.0,
        "phone_informal_wer": 42.0,
    },
}


# ---------------------------------------------------------------------------
# Small stdlib helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write *payload* as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from exc


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


class _Logger:
    """Mirror progress to stdout and ``training_live.log``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[fake-trainer] {time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# Run ledger and fake artifacts
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-free fake trainer for the stateful cockpit"
    )
    parser.add_argument("--run_dir", required=True, help="Persistent run ledger dir")
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Telemetry/results dir (default: the run dir)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Checkpoint/adapter dir (default: <run_dir>/output)",
    )
    parser.add_argument(
        "--preflight_cache_dir",
        default=None,
        help="Shared preflight cache dir (default: the run dir)",
    )
    parser.add_argument(
        "--post_eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only rerun post-eval from the existing best_adapter",
    )
    parser.add_argument(
        "--fresh_start",
        "--fresh-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Start fresh: clear prior fake results (loss history, steps, "
            "benchmarks, telemetry, stop signal) and the fake best_adapter."
        ),
    )
    # The real trainer passes many more flags. The fake does not need them, but
    # it must accept them so the server can build one command shape.
    args, _unknown = parser.parse_known_args(argv)
    return args


def _paths_for(
    run_dir: Path,
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> dict:
    return {
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "merged_dir": str(run_dir / "merged"),
        "preflight_cache": str(cache_dir / "baselines" / "preflight.json"),
    }


def _create_run_at(
    run_dir: Path,
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    trainer_args: dict,
) -> dict:
    """Create a ledger at *run_dir*, tolerating a non-run-id directory name.

    ``run_state.create_run`` only accepts canonical run ids. Server-created runs
    always use one, but manual/dev invocations may point at any directory. For
    those, create a canonical sibling and move it into place.
    """
    paths = _paths_for(run_dir, results_dir, output_dir, cache_dir)
    if run_state.run_id_valid(run_dir.name):
        return run_state.create_run(
            run_dir.parent,
            config={},
            trainer_args=trainer_args,
            paths=paths,
            source="fake",
            run_id=run_dir.name,
        )

    temporary_id = run_state.new_run_id()
    created = run_state.create_run(
        run_dir.parent,
        config={},
        trainer_args=trainer_args,
        paths=paths,
        source="fake",
        run_id=temporary_id,
    )
    generated = run_dir.parent / temporary_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(generated, run_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(generated, ignore_errors=True)

    def repoint(run: dict) -> dict:
        run["run_id"] = run_dir.name
        run["paths"] = {**(run.get("paths") or {}), **paths}
        return run

    run_state.update_run(run_dir, repoint)
    created = run_state.load_run(run_dir) or created
    return created


def _ensure_run(
    run_dir: Path,
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    trainer_args: dict,
) -> dict:
    run = run_state.load_run(run_dir)
    if run is None:
        return _create_run_at(
            run_dir, results_dir, output_dir, cache_dir, trainer_args
        )

    paths = _paths_for(run_dir, results_dir, output_dir, cache_dir)

    def update(run: dict) -> dict:
        merged = (run.get("paths") or {}).get("merged_dir")
        run["source"] = "fake"
        run["paths"] = {
            **(run.get("paths") or {}),
            **paths,
            "merged_dir": merged or paths["merged_dir"],
        }
        return run

    return run_state.update_run(run_dir, update)


def _fake_preflight_payload() -> dict:
    return {
        "version": "1",
        "created_at": time.time(),
        "fingerprint": {"fake": True},
        "models": {key: dict(row) for key, row in _PREFLIGHT_MODELS.items()},
        "fake": True,
    }


def _seed_benchmarks(results_dir: Path, preflight: dict) -> None:
    path = results_dir / "epoch_benchmarks.json"
    existing = _read_json(path, [])
    preserved = []
    if isinstance(existing, list):
        preserved = [
            row
            for row in existing
            if isinstance(row, dict) and not row.get("is_baseline")
        ]
    rows = []
    for key in ("buzzasr_base", "whisper_v3"):
        model = (preflight.get("models") or {}).get(key)
        if not model:
            continue
        rows.append(
            {
                "type": "baseline",
                "is_baseline": True,
                "canary_transcriptions": [],
                **model,
            }
        )
    _write_json(path, rows + preserved)


def _final_benchmark() -> dict:
    clean_wer = 15.4
    phone_wer = 21.5
    baseline = _PREFLIGHT_MODELS["buzzasr_base"]
    return {
        "epoch": "Epoch 1",
        "epoch_number": 1,
        "type": "checkpoint",
        "is_baseline": False,
        "normalized_wer": clean_wer,
        "normalized_cer": 4.6,
        "diagnostic_wer": 14.7,
        "formal_wer": 12.9,
        "informal_wer": 19.8,
        "phone_normalized_wer": phone_wer,
        "phone_normalized_cer": 7.0,
        "phone_formal_wer": 18.0,
        "phone_informal_wer": 26.5,
        "phone_seed": 1234,
        "phone_profiles": ["customer"],
        "total_samples": PREFLIGHT_SAMPLES,
        "canary_transcriptions": [],
        "delta_vs_buzzasr": round(
            clean_wer - baseline["normalized_wer"], 2
        ),
        "delta_phone_vs_buzzasr": round(
            phone_wer - baseline["phone_normalized_wer"], 2
        ),
        "timestamp": time.time(),
        "fake": True,
    }


def _record_final_benchmark(results_dir: Path) -> tuple[dict, list]:
    row = _final_benchmark()
    path = results_dir / "epoch_benchmarks.json"
    rows = _read_json(path, [])
    if not isinstance(rows, list):
        rows = []
    rows = [
        item
        for item in rows
        if isinstance(item, dict) and item.get("epoch") != row["epoch"]
    ]
    rows.append(row)
    _write_json(path, rows)
    return row, rows


def _fake_report(row: dict, benchmarks: list) -> dict:
    metrics = {
        "normalized_wer": row["normalized_wer"],
        "normalized_cer": row["normalized_cer"],
        "diagnostic_wer": row["diagnostic_wer"],
        "formal_wer": row["formal_wer"],
        "informal_wer": row["informal_wer"],
        "delta_vs_buzzasr_base": row["delta_vs_buzzasr"],
        "delta_phone_vs_buzzasr_base": row["delta_phone_vs_buzzasr"],
        "details": {
            "total_samples": row["total_samples"],
            "overall": {
                "normalized_wer": row["normalized_wer"],
                "normalized_cer": row["normalized_cer"],
                "diagnostic_wer_zwnj_space": row["diagnostic_wer"],
            },
            "formal": {"normalized_wer": row["formal_wer"], "count": 500},
            "informal": {"normalized_wer": row["informal_wer"], "count": 500},
        },
        "phone": {
            "seed": row["phone_seed"],
            "normalized_wer": row["phone_normalized_wer"],
            "normalized_cer": row["phone_normalized_cer"],
            "profiles": row["phone_profiles"],
            "gap_vs_clean_pp": round(
                row["phone_normalized_wer"] - row["normalized_wer"], 2
            ),
        },
    }
    return {
        "dataset_name": "fake/dataset",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": row["total_samples"],
        "total_audio_duration_seconds": 125.0,
        "total_inference_time_seconds": 5.0,
        "rtfx_speedup": 25.0,
        "final_full_test_metrics": metrics,
        "baseline_comparison": {
            "source": "preflight",
            "sample_count": PREFLIGHT_SAMPLES,
            "evaluated_samples": row["total_samples"],
        },
        "all_epoch_benchmarks": benchmarks,
        "trainer_state_summary": {
            "best_metric": row["normalized_wer"],
            "global_step": row["epoch_number"] * 100,
            "train_logs": [],
            "eval_logs": [],
        },
        "fake": True,
    }


def _telemetry(
    *,
    status: str,
    step: int,
    max_steps: int,
    epoch: float,
    loss: float | None,
    learning_rate: float | None,
    speed: float,
    eta_seconds: int,
    elapsed_seconds: int,
) -> dict:
    return {
        "status": status,
        "step": int(step),
        "max_steps": int(max_steps),
        "epoch": round(float(epoch), 2),
        "loss": loss,
        "learning_rate": learning_rate,
        "speed_s_per_step": round(float(speed), 3),
        "elapsed_seconds": int(elapsed_seconds),
        "eta_seconds": int(eta_seconds),
        "gpu": {
            "available": False,
            "device_name": "CPU (fake)",
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0,
            "utilization_pct": 0.0,
        },
        "updated_at": time.time(),
    }


def _attempt_number(run_dir: Path) -> int:
    run = run_state.load_run(run_dir) or {}
    attempts = run.get("attempts") or []
    if not attempts:
        return 1
    try:
        return int(attempts[-1].get("attempt") or 1)
    except (TypeError, ValueError):
        return 1


def _resume_step(run_dir: Path, results_dir: Path) -> int:
    """Return the last recorded optimizer step so a resume never duplicates it."""
    run = run_state.load_run(run_dir) or {}
    try:
        last_step = int(run.get("last_step") or 0)
    except (TypeError, ValueError):
        last_step = 0
    tail = run_state.read_steps(results_dir, tail=1)
    if tail:
        try:
            last_step = max(last_step, int(tail[-1].get("step") or 0))
        except (TypeError, ValueError):
            pass
    return max(0, last_step)


def _remove_signal(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


_FRESH_RESULT_FILES = (
    "loss_history.json",
    "steps.jsonl",
    "events.jsonl",
    "epoch_benchmarks.json",
    "live_telemetry.json",
)


def _is_cockpit_spawned() -> bool:
    """True when the server already owns the log/PID/attempt for this process."""
    return (os.environ.get("ASR_COCKPIT_SPAWNED", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pid_alive(pid: int) -> bool:
    """True when *pid* refers to a running process (or an unkillable one)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _server_owns_attempt(run_dir: Path) -> bool:
    """True when a live foreign process already holds this run's open attempt.

    Fallback for detecting a server-owned run when ``ASR_COCKPIT_SPAWNED`` is
    unavailable: the server records its child's PID before spawning, so an open
    attempt whose PID belongs to another live process means this invocation
    must stay hands-off instead of double-closing the attempt.
    """
    ours = os.getpid()
    run = run_state.load_run(run_dir) or {}
    attempts = run.get("attempts") or []
    for item in reversed(attempts):
        if not isinstance(item, dict) or item.get("ended_at") is not None:
            continue
        try:
            pid = int(item.get("pid") or 0)
        except (TypeError, ValueError):
            return True
        return pid != ours and _pid_alive(pid)
    return False


def _clear_fake_previous_state(
    output_dir: Path, results_dir: Path
) -> list[str]:
    """Delete prior fake results so ``--fresh_start`` truly starts clean.

    Removes the fake telemetry history and stop signal from the results dir
    plus the fake best adapter. Mirrors the real trainer's fresh-start
    clearing; the run ledger itself (``run.json``) is preserved.
    """
    removed: list[str] = []
    for name in (*_FRESH_RESULT_FILES, "stop_training.signal"):
        try:
            target = results_dir / name
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
                removed.append(name)
        except OSError:
            continue
    try:
        adapter = output_dir / "best_adapter"
        if adapter.is_dir() and not adapter.is_symlink():
            shutil.rmtree(adapter, ignore_errors=True)
            removed.append("best_adapter")
    except OSError:
        pass
    return removed


def _reset_fake_progress(run_dir: Path) -> None:
    """Restart step numbering after a fresh start on a reused run dir."""
    run_state.set_progress(
        run_dir, step=0, step_count=0, last_step_at=time.time()
    )

    def clear(run: dict) -> dict:
        run["stop_requested_at"] = None
        return run

    run_state.update_run(run_dir, clear)


def _ensure_fake_attempt(run_dir: Path) -> None:
    """Record this process as an attempt unless it already owns one."""
    pid = os.getpid()
    run = run_state.load_run(run_dir) or {}
    attempts = run.get("attempts") or []
    open_attempts = [
        item
        for item in attempts
        if isinstance(item, dict) and item.get("ended_at") is None
    ]
    if open_attempts:
        try:
            same_pid = int(open_attempts[-1].get("pid") or 0) == pid
        except (TypeError, ValueError):
            same_pid = False
        if same_pid:
            try:
                current = int(run.get("current_pid") or 0)
            except (TypeError, ValueError):
                current = 0
            if current != pid:
                run_state.update_run(
                    run_dir,
                    lambda current_run: {**current_run, "current_pid": pid},
                )
            return
        # A dead earlier attempt must not stay open forever.
        run_state.finalize_attempt(run_dir, exit_code=None, reason="process_lost")
        run = run_state.load_run(run_dir) or {}
        attempts = run.get("attempts") or []
    run_state.start_attempt(
        run_dir, pid, reason="initial" if not attempts else "resume"
    )


def _write_fake_pid(run_dir: Path) -> None:
    try:
        (run_dir / "training.pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write {run_dir / 'training.pid'}: {exc}")


def _abort_fake_setup(run_dir: Path) -> None:
    """Close any attempt opened during setup and drop the PID file."""
    if run_state.load_run(run_dir) is not None:
        try:
            run_state.finalize_attempt(run_dir, exit_code=1, reason="nonzero_exit")
        except Exception as exc:
            print(
                f"Warning: could not finalize run attempt: {exc}", file=sys.stderr
            )
    try:
        (Path(run_dir) / "training.pid").unlink(missing_ok=True)
    except OSError:
        pass


def _finalize_fake_attempt(run_dir: Path, exit_code: int) -> None:
    """Close the owned attempt and remove its PID file on every exit path."""
    try:
        run = run_state.load_run(run_dir)
        if run is not None:
            if run.get("stop_requested_at") or run.get("status") == "stopped":
                reason = "halted"
            elif exit_code == 0:
                reason = "completed"
            else:
                reason = "nonzero_exit"
            run_state.finalize_attempt(
                run_dir, exit_code=exit_code, reason=reason
            )
    except Exception as exc:
        print(f"Warning: could not finalize run attempt: {exc}", file=sys.stderr)
    try:
        (run_dir / "training.pid").unlink(missing_ok=True)
    except OSError:
        pass


def _system_exit_code(exc: SystemExit) -> int:
    """Map a ``SystemExit`` to a process exit code.

    ``SystemExit`` with code ``None`` (bare ``raise SystemExit``) means a
    clean exit, just like code ``0``; anything non-integer is a failure.
    The finalizer uses this so the recorded attempt matches the real exit.
    """
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _fail(
    run_dir: Path,
    results_dir: Path,
    stage: str,
    message: str,
    log: _Logger,
) -> int:
    log(f"{stage} failed: {message}")
    payload = _read_json(results_dir / "live_telemetry.json", {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "status": "failed",
            "failed_stage": stage,
            "error": message,
            "updated_at": time.time(),
        }
    )
    _write_json(results_dir / "live_telemetry.json", payload)
    run_state.set_stage(run_dir, stage, "failed", detail=message)
    run_state.set_status(
        run_dir, "failed", failed_stage=stage, last_error=message
    )
    return 1


def _halt(
    run_dir: Path,
    results_dir: Path,
    signal_path: Path,
    log: _Logger,
    *,
    step: int,
    max_steps: int,
    epoch: float,
    loss: float | None,
    learning_rate: float | None,
    elapsed: float,
) -> int:
    _remove_signal(signal_path)
    log("stop requested; training halted")
    _write_json(
        results_dir / "live_telemetry.json",
        _telemetry(
            status="stopped",
            step=step,
            max_steps=max_steps,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            speed=0.0,
            eta_seconds=0,
            elapsed_seconds=int(elapsed),
        ),
    )
    # Mark the halted stage last so ``current_stage`` points at what stopped
    # (the real trainer's _halt_run does the same).
    run_state.set_stage(run_dir, "post_eval", "skipped", detail="training halted")
    run_state.set_stage(run_dir, "training", "stopped", detail="halt requested")
    run_state.set_status(
        run_dir, "stopped", failed_stage=None, last_error=None
    )
    return 0


def _hang_until_stopped(
    run_dir: Path,
    results_dir: Path,
    signal_path: Path,
    log: _Logger,
    *,
    start_step: int,
    max_steps: int,
    started_at: float,
) -> int:
    """Pretend training is stuck until the cockpit writes a stop signal."""
    log(f"hanging in training until {signal_path.name} appears")
    while not signal_path.exists():
        time.sleep(0.1)
    return _halt(
        run_dir,
        results_dir,
        signal_path,
        log,
        step=start_step,
        max_steps=max_steps,
        epoch=0.0,
        loss=None,
        learning_rate=None,
        elapsed=time.time() - started_at,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else run_dir
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "output"
    )
    cache_dir = (
        Path(args.preflight_cache_dir).expanduser().resolve()
        if args.preflight_cache_dir
        else run_dir
    )
    for directory in (run_dir, results_dir, output_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    post_eval_only = bool(args.post_eval_only)
    fresh_start = bool(getattr(args, "fresh_start", False))
    if fresh_start and not post_eval_only:
        for name in _clear_fake_previous_state(output_dir, results_dir):
            print(f"[fake-trainer] fresh start: cleared {name}", flush=True)

    trainer_args = {
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "preflight_cache_dir": str(cache_dir),
    }
    run = _ensure_run(run_dir, results_dir, output_dir, cache_dir, trainer_args)
    if fresh_start and not post_eval_only:
        _reset_fake_progress(run_dir)
    # The server records the attempt before spawning its child; a standalone
    # invocation owns its own attempt and PID file.
    owned = not _is_cockpit_spawned() and not _server_owns_attempt(run_dir)
    if owned:
        try:
            _ensure_fake_attempt(run_dir)
            _write_fake_pid(run_dir)
        except Exception:
            _abort_fake_setup(run_dir)
            raise

    exit_code = 0
    try:
        exit_code = _run(args, run, run_dir, results_dir, output_dir, cache_dir)
    except SystemExit as exc:
        exit_code = _system_exit_code(exc)
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        if owned:
            _finalize_fake_attempt(run_dir, exit_code)
    return exit_code


def _run(
    args: argparse.Namespace,
    run: dict,
    run_dir: Path,
    results_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> int:
    steps = max(0, _env_int("ASR_FAKE_STEPS", 200))
    delay = max(0.0, _env_float("ASR_FAKE_STEP_DELAY", 0.05))
    fail_stage = (os.environ.get("ASR_FAKE_FAIL_STAGE") or "").strip().lower()
    if fail_stage and fail_stage not in FAIL_STAGES:
        raise SystemExit(
            "ASR_FAKE_FAIL_STAGE must be one of "
            f"{', '.join(FAIL_STAGES)}, got {fail_stage!r}"
        )
    hang = _env_flag("ASR_FAKE_HANG")

    source = run.get("source") or "fake"
    log = _Logger(run_dir / "training_live.log")
    if bool(args.post_eval_only) and bool(getattr(args, "fresh_start", False)):
        log("Note: --fresh_start is ignored with --post_eval_only.")
    signal_path = results_dir / "stop_training.signal"
    telemetry_path = results_dir / "live_telemetry.json"
    started_at = time.time()
    attempt = _attempt_number(run_dir)
    post_eval_only = bool(args.post_eval_only)
    start_step = _resume_step(run_dir, results_dir)
    total_steps = start_step + steps

    # The ledger is authoritative even if the caller left it "starting".
    run_state.set_status(
        run_dir, "running", failed_stage=None, last_error=None
    )
    if not post_eval_only:
        (results_dir / "steps.jsonl").touch(exist_ok=True)
    _write_json(
        telemetry_path,
        _telemetry(
            status="initializing",
            step=start_step,
            max_steps=total_steps,
            epoch=0.0,
            loss=None,
            learning_rate=None,
            speed=0.0,
            eta_seconds=0,
            elapsed_seconds=0,
        ),
    )
    log(
        f"started source={source} steps={steps} delay={delay:g}s "
        f"post_eval_only={post_eval_only} run_dir={run_dir}"
    )

    loss = None
    lr = 1e-4
    epoch = 0.0
    last_completed_step = start_step

    if not post_eval_only:
        # --------------------------------------------------------------
        # Preflight
        # --------------------------------------------------------------
        run_state.set_stage(
            run_dir, "preflight", "running", detail="fake baseline"
        )
        log("preflight started")
        if fail_stage == "preflight":
            return _fail(
                run_dir,
                results_dir,
                "preflight",
                "ASR_FAKE_FAIL_STAGE forced preflight failure",
                log,
            )

        preflight = _fake_preflight_payload()
        _write_json(run_dir / "baselines" / "preflight.json", preflight)
        _write_json(cache_dir / "baselines" / "preflight.json", preflight)
        _seed_benchmarks(results_dir, preflight)
        run_state.set_stage(
            run_dir, "preflight", "completed", detail="fake baseline (measured)"
        )
        log("preflight completed")

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        run_state.set_stage(run_dir, "training", "running", detail="fake LoRA")
        _write_json(
            telemetry_path,
            _telemetry(
                status="training",
                step=start_step,
                max_steps=total_steps,
                epoch=0.0,
                loss=None,
                learning_rate=1e-4,
                speed=0.0,
                eta_seconds=int(steps * delay),
                elapsed_seconds=0,
            ),
        )
        log(
            f"training started ({steps} steps, "
            f"steps {start_step + 1}-{total_steps})"
        )
        if fail_stage == "training":
            return _fail(
                run_dir,
                results_dir,
                "training",
                "ASR_FAKE_FAIL_STAGE forced training failure",
                log,
            )

        if hang:
            return _hang_until_stopped(
                run_dir,
                results_dir,
                signal_path,
                log,
                start_step=start_step,
                max_steps=total_steps,
                started_at=started_at,
            )

        loss_history = _read_json(results_dir / "loss_history.json", [])
        if not isinstance(loss_history, list):
            loss_history = []
        loss_history = [
            point for point in loss_history if isinstance(point, dict)
        ][-MAX_LOSS_HISTORY:]
        _write_json(results_dir / "loss_history.json", loss_history)

    if not post_eval_only:
        halted = False
        for step in range(start_step + 1, total_steps + 1):
            now = time.time()
            progress = (step - start_step) / max(1, steps)
            epoch = progress * 3.0
            loss = round(
                max(
                    0.05,
                    2.4 * math.exp(-3.2 * progress)
                    + 0.12
                    + 0.03 * math.sin(step / 3.0),
                ),
                4,
            )
            lr = round(1e-4 * 0.5 * (1.0 + math.cos(math.pi * progress)), 10)
            elapsed = now - started_at
            eta_seconds = int(max(0, total_steps - step) * delay)

            _write_json(
                telemetry_path,
                _telemetry(
                    status="training",
                    step=step,
                    max_steps=total_steps,
                    epoch=epoch,
                    loss=loss,
                    learning_rate=lr,
                    speed=delay,
                    eta_seconds=eta_seconds,
                    elapsed_seconds=int(elapsed),
                ),
            )

            loss_history.append(
                {
                    "step": step,
                    "loss": loss,
                    "epoch": round(epoch, 2),
                    "lr": lr,
                }
            )
            loss_history = loss_history[-MAX_LOSS_HISTORY:]
            _write_json(results_dir / "loss_history.json", loss_history)

            last_completed_step = step
            run_state.append_step(
                results_dir,
                {
                    "step": step,
                    "epoch": round(epoch, 2),
                    "loss": loss,
                    "loss_step": step,
                    "learning_rate": lr,
                    "speed_s_per_step": round(delay, 3),
                    "elapsed_seconds": int(elapsed),
                    "eta_seconds": eta_seconds,
                    "attempt": attempt,
                },
            )
            if step % 10 == 0 or step == total_steps:
                run_state.set_progress(
                    run_dir,
                    step=step,
                    step_count=run_state.step_count(results_dir),
                    last_step_at=now,
                )
                log(
                    f"step {step}/{total_steps} loss={loss:.4f} "
                    f"lr={lr:.2e} elapsed={int(elapsed)}s"
                )

            if signal_path.exists():
                halted = True
                break
            if delay:
                time.sleep(delay)

        # Flush the final partial progress interval even when steps % 10 != 0.
        run_state.set_progress(
            run_dir,
            step=last_completed_step,
            step_count=run_state.step_count(results_dir),
            last_step_at=time.time(),
        )

        if halted:
            return _halt(
                run_dir,
                results_dir,
                signal_path,
                log,
                step=last_completed_step,
                max_steps=total_steps,
                epoch=epoch,
                loss=loss,
                learning_rate=lr,
                elapsed=time.time() - started_at,
            )

        run_state.set_stage(run_dir, "training", "completed", detail="fake LoRA")
        log("training completed")

        # --------------------------------------------------------------
        # Best adapter (saved before post-eval so failures are resumable)
        # --------------------------------------------------------------
        adapter_dir = output_dir / "best_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            adapter_dir / "adapter_config.json",
            {
                "peft_type": "LORA",
                "task_type": "SEQ_2_SEQ_LM",
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "base_model_name_or_path": "BuzzASR/persian",
                "fake": True,
            },
        )
        log(f"saved fake adapter to {adapter_dir}")
    else:
        log("post-eval-only: skipping preflight and training")
        adapter_config = output_dir / "best_adapter" / "adapter_config.json"
        if not adapter_config.is_file():
            run_state.set_stage(
                run_dir, "post_eval", "running", detail="post-eval only"
            )
            return _fail(
                run_dir,
                results_dir,
                "post_eval",
                f"post-eval-only: {adapter_config} not found",
                log,
            )

    # ------------------------------------------------------------------
    # Post-eval (also the only stage in --post_eval_only mode)
    # ------------------------------------------------------------------
    run_state.set_stage(run_dir, "post_eval", "running", detail="fake benchmark")
    post_eval_step = start_step if post_eval_only else last_completed_step
    _write_json(
        telemetry_path,
        _telemetry(
            status="evaluating_benchmark",
            step=post_eval_step,
            max_steps=post_eval_step,
            epoch=epoch,
            loss=loss,
            learning_rate=0.0,
            speed=delay,
            eta_seconds=0,
            elapsed_seconds=int(time.time() - started_at),
        ),
    )
    log("post-eval started")
    if fail_stage == "post_eval":
        return _fail(
            run_dir,
            results_dir,
            "post_eval",
            "ASR_FAKE_FAIL_STAGE forced post_eval failure",
            log,
        )

    row, benchmarks = _record_final_benchmark(results_dir)
    _write_json(
        output_dir / "final_evaluation_report.json",
        _fake_report(row, benchmarks),
    )
    run_state.set_stage(run_dir, "post_eval", "completed", detail="fake report")
    finished_step = start_step if post_eval_only else total_steps
    _write_json(
        telemetry_path,
        _telemetry(
            status="completed",
            step=finished_step,
            max_steps=finished_step,
            epoch=epoch,
            loss=loss,
            learning_rate=0.0,
            speed=delay,
            eta_seconds=0,
            elapsed_seconds=int(time.time() - started_at),
        ),
    )
    # Merge stays pending: the server owns the merge stage.
    run_state.set_status(
        run_dir, "completed", failed_stage=None, last_error=None
    )
    log("run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
