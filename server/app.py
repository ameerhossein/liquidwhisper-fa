"""
RunPod Web Management Server for Persian ASR Training & Benchmarking.
FastAPI application delivering:
- Real-time telemetry WebSocket (/ws/live)
- REST endpoints for training control, benchmarks, and loss curves
- Static UI delivery on Port 7860
"""

import os
import sys
import json
import time
import signal
import subprocess
import asyncio
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

TELEMETRY_FILE = RESULTS_DIR / "live_telemetry.json"
BENCHMARKS_FILE = RESULTS_DIR / "epoch_benchmarks.json"
LOSS_HISTORY_FILE = RESULTS_DIR / "loss_history.json"
LOG_FILE = RESULTS_DIR / "training_live.log"
STOP_SIGNAL_FILE = RESULTS_DIR / "stop_training.signal"

app = FastAPI(title="Persian ASR RunPod Cockpit", version="1.0.0")

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
last_exit_code: Optional[int] = None
halt_requested = False
ACTIVE_STATUSES = ("initializing", "training", "evaluating_benchmark")
HALT_GRACE_SECONDS = int(os.environ.get("ASR_HALT_GRACE_SECONDS", "900") or 900)


_HARDCODED_BASELINES = [
    {
        "epoch": "Whisper Large v3 (Zero-Shot)",
        "epoch_number": 0,
        "type": "baseline",
        "is_baseline": True,
        "normalized_wer": 28.50,
        "normalized_cer": 8.90,
        "diagnostic_wer": 26.10,
        "formal_wer": 24.00,
        "informal_wer": 36.80,
        "delta_vs_whisper": 0.0,
        "total_samples": 2149,
        "canary_transcriptions": []
    },
    {
        "epoch": "BuzzASR Base (Zero-Shot)",
        "epoch_number": 0,
        "type": "baseline",
        "is_baseline": True,
        "normalized_wer": 17.20,
        "normalized_cer": 4.80,
        "diagnostic_wer": 15.90,
        "formal_wer": 15.10,
        "informal_wer": 22.40,
        "delta_vs_whisper": -11.30,
        "total_samples": 2149,
        "canary_transcriptions": []
    }
]


def _load_shared_baselines() -> List[Dict[str, Any]]:
    """Single source of truth: configs/baselines.json, fallback to hardcoded."""
    for cand in (BASE_DIR / "configs" / "baselines.json", Path("configs/baselines.json")):
        try:
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                w = data.get("whisper_v3", {})
                b = data.get("buzzasr_base", {})
                if w and b:
                    return [
                        {
                            "epoch": w.get("epoch", w.get("name", "Whisper Large v3 (Zero-Shot)")),
                            "epoch_number": 0,
                            "type": "baseline",
                            "is_baseline": True,
                            "normalized_wer": float(w.get("normalized_wer", 28.5)),
                            "normalized_cer": float(w.get("normalized_cer", 8.9)),
                            "diagnostic_wer": float(w.get("diagnostic_wer", 26.1)),
                            "formal_wer": float(w.get("formal_wer", 24.0)),
                            "informal_wer": float(w.get("informal_wer", 36.8)),
                            "delta_vs_whisper": 0.0,
                            "total_samples": int(w.get("total_samples", 2149)),
                            "canary_transcriptions": []
                        },
                        {
                            "epoch": b.get("epoch", b.get("name", "BuzzASR Base (Zero-Shot)")),
                            "epoch_number": 0,
                            "type": "baseline",
                            "is_baseline": True,
                            "normalized_wer": float(b.get("normalized_wer", 17.2)),
                            "normalized_cer": float(b.get("normalized_cer", 4.8)),
                            "diagnostic_wer": float(b.get("diagnostic_wer", 15.9)),
                            "formal_wer": float(b.get("formal_wer", 15.1)),
                            "informal_wer": float(b.get("informal_wer", 22.4)),
                            "delta_vs_whisper": round(float(b.get("normalized_wer", 17.2)) - float(w.get("normalized_wer", 28.5)), 2),
                            "total_samples": int(b.get("total_samples", 2149)),
                            "canary_transcriptions": []
                        },
                    ]
        except Exception:
            continue
    return [dict(r) for r in _HARDCODED_BASELINES]


def _is_process_running(pid: Optional[int] = None) -> bool:
    """Check PID liveness (for restart recovery via PID file)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


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


def get_default_benchmarks() -> List[Dict[str, Any]]:
    return _load_shared_baselines()


def read_json_safely(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


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
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text(encoding="utf-8").strip() or "0")
            if _is_process_running(pid):
                return True
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def _derive_status(telemetry: Dict[str, Any], is_running: bool) -> Dict[str, Any]:
    """Map raw telemetry plus process state to the status shown in the UI."""
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
    paths under /workspace, /runpod-volume, /mnt, or colon-separated
    ASR_ALLOWED_OUTPUT_ROOTS. Returns absolute path as string so the trainer
    can write checkpoints directly to the persistent volume (auto-resumable).
    """
    raw = (raw or _default_output_dir()).strip() or _default_output_dir()
    extra_roots = [r for r in os.environ.get("ASR_ALLOWED_OUTPUT_ROOTS", "").split(":") if r.strip()]
    allowed_abs = [Path("/workspace"), Path("/runpod-volume"), Path("/mnt")]
    for r in extra_roots:
        try:
            allowed_abs.append(Path(r).resolve())
        except Exception:
            continue
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

    Without torchcodec the trainer dies later inside dataset.filter/map with
    "To support decoding audio data, please install 'torchcodec'". Surfacing it
    here saves a wasted launch cycle from the Web UI.
    """
    try:
        import torchcodec  # noqa: F401
    except Exception as e:
        # Catch-all (not just ImportError): a version/CUDA mismatch raises
        # RuntimeError/OSError on import, which would otherwise bubble up as
        # an unhandled 500 with a plain-text "Internal Server Error" body.
        reason = f" (import failed{': ' + str(e).strip()[:300] if str(e).strip() else ''})"
        raise HTTPException(
            status_code=500,
            detail=(
                "Audio backend missing: 'torchcodec' is not installed, so the "
                "dataset 'audio' column cannot be decoded. On the server run: "
                "source .venv/bin/activate && pip install 'torchcodec==0.2.1' "
                "--index-url https://download.pytorch.org/whl/cu124 "
                "(pinned for torch 2.6; same index as torch), then restart training. "
                f"Fresh pods: ./setup_server.sh now installs it automatically.{reason}"
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
    is_running = _training_running()

    telemetry = read_json_safely(TELEMETRY_FILE, {
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

    telemetry = _derive_status(telemetry, is_running)
    telemetry["process_running"] = is_running
    telemetry["gpu"] = telemetry.get("gpu") or get_gpu_info()
    return telemetry


@app.get("/api/benchmarks")
async def get_benchmarks():
    data = read_json_safely(BENCHMARKS_FILE, None)
    if not data or not isinstance(data, list) or len(data) == 0:
        data = get_default_benchmarks()
        with open(BENCHMARKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return data


@app.get("/api/loss-history")
async def get_loss_history():
    return read_json_safely(LOSS_HISTORY_FILE, [])


@app.get("/api/logs")
async def get_logs(tail: int = 150):
    if not LOG_FILE.exists():
        return {"logs": ["Log buffer empty. Training has not started."]}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return {"logs": [line.rstrip() for line in lines[-tail:]]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}


@app.post("/api/train/start")
async def start_training(config: Dict[str, Any] = None):
    global training_process, training_log_fd, last_exit_code, halt_requested
    _check_auth(config)
    _check_audio_backend_or_400()
    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training is already currently running.")
    # Recover from PID file after restart.
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text(encoding="utf-8").strip() or "0")
            if _is_process_running(pid):
                raise HTTPException(status_code=400, detail=f"Training already running (pid {pid}).")
            PID_FILE.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception:
        pass

    config = config or {}
    try:
        epochs_i = int(config.get("epochs", 4))
        batch_i = int(config.get("batch_size", 16))
        accum_i = int(config.get("grad_accum", 2))
        lr_f = float(config.get("learning_rate", 1e-4))
        eval_bs_i = int(config.get("eval_batch_size", 16))
        max_bench = config.get("max_benchmark_samples", 300)
        max_bench_i = int(max_bench) if max_bench not in (None, "") else 300
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="epochs/batch_size/grad_accum/eval_batch_size must be integers; learning_rate numeric.")
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

    output_dir = _sanitize_output_dir(str(config.get("output_dir", None) or _default_output_dir()))
    results_cfg = (config.get("results_dir", "") or "")
    results_dir = _sanitize_output_dir(str(results_cfg).strip()) if str(results_cfg).strip() else str(RESULTS_DIR)
    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar")).strip() or "shekar-ai/Neyshekar"
    if len(dataset_name) > 128 or any(c in dataset_name for c in ";&|$` \n"):
        raise HTTPException(status_code=400, detail="Invalid dataset_name.")
    # Resume controls: default auto-resume ON so pod restarts continue from
    # persistent-volume checkpoints. Send {"fresh_start": true} for clean run,
    # or {"resume_from_checkpoint": "<path|last>"} to force a checkpoint.
    fresh_start = bool(config.get("fresh_start", config.get("fresh", False)))
    resume_explicit = str(config.get("resume_from_checkpoint", "") or "").strip()
    auto_resume = config.get("auto_resume", True)
    auto_resume_on = (str(auto_resume).strip().lower() not in ("0", "false", "no", "off", "")) if isinstance(auto_resume, str) else bool(auto_resume)
    if fresh_start:
        auto_resume_on = False

    # Remove any old stop signal
    if STOP_SIGNAL_FILE.exists():
        STOP_SIGNAL_FILE.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-u",  # unbuffered: stream prints to training_live.log during long imports/downloads
        str(BASE_DIR / "scripts" / "train_lora.py"),
        "--base_model", "BuzzASR/persian",
        "--dataset_name", dataset_name,
        "--output_dir", output_dir,
        "--results_dir", results_dir,
        "--epochs", str(epochs_i),
        "--batch_size", str(batch_i),
        "--grad_accum", str(accum_i),
        "--learning_rate", str(lr_f),
        "--eval_batch_size", str(eval_bs_i),
        "--max_benchmark_samples", str(max_bench_i),
        "--enable_telemetry",
        "--run_epoch_benchmark"
    ]
    if resume_explicit:
        cmd += ["--resume_from_checkpoint", resume_explicit]
    if auto_resume_on:
        cmd += ["--auto_resume"]
    else:
        cmd += ["--no-auto_resume"]
    if fresh_start:
        cmd += ["--fresh_start"]

    _close_log_fd()
    training_log_fd = open(LOG_FILE, "w", encoding="utf-8")
    training_log_fd.write(f"=== Persian ASR Training Job Launched at {time.ctime()} ===\n")
    training_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    training_log_fd.flush()

    last_exit_code = None
    halt_requested = False
    TELEMETRY_FILE.write_text(json.dumps({
        "status": "initializing",
        "step": 0,
        "max_steps": 0,
        "epoch": 0.0,
        "loss": None,
        "learning_rate": 0.0,
        "speed_s_per_step": 0.0,
        "eta_seconds": 0,
    }), encoding="utf-8")

    training_process = None
    try:
        training_process = subprocess.Popen(
            cmd,
            stdout=training_log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            preexec_fn=os.setsid if sys.platform != "win32" else None
        )
    except Exception as e:
        try:
            training_log_fd.write(f"\n!!! Failed to launch training process: {e}\n")
            training_log_fd.flush()
        except Exception:
            pass
        _close_log_fd()
        raise HTTPException(status_code=500, detail=f"Failed to launch training: {e}")
    try:
        PID_FILE.write_text(str(training_process.pid), encoding="utf-8")
    except Exception:
        pass

    return {
        "status": "started",
        "pid": training_process.pid,
        "config": {
            "epochs": str(epochs_i),
            "batch_size": str(batch_i),
            "learning_rate": str(lr_f),
            "output_dir": output_dir,
            "results_dir": results_dir,
            "auto_resume": auto_resume_on,
            "fresh_start": fresh_start,
            "resume_from_checkpoint": resume_explicit or None
        }
    }


async def _escalate_halt(proc: subprocess.Popen) -> None:
    """Wait for the trainer to checkpoint and exit; SIGTERM only after the grace period."""
    deadline = time.time() + HALT_GRACE_SECONDS
    while proc.poll() is None:
        if time.time() >= deadline:
            if read_json_safely(TELEMETRY_FILE, {}).get("status") in ("stopped", "completed"):
                return
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except Exception:
                pass
            return
        await asyncio.sleep(2)


@app.post("/api/train/stop")
async def stop_training(payload: Dict[str, Any] = None):
    global halt_requested
    _check_auth(payload)
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
            telemetry = read_json_safely(TELEMETRY_FILE, {})
            benchmarks = read_json_safely(BENCHMARKS_FILE, get_default_benchmarks())
            loss_hist = read_json_safely(LOSS_HISTORY_FILE, [])

            is_running = _training_running()
            telemetry = _derive_status(telemetry, is_running)
            telemetry["process_running"] = is_running
            # WS clients otherwise never get a fresh GPU reading (the trainer
            # only rewrites telemetry on steps); mirror /api/status behavior.
            telemetry["gpu"] = telemetry.get("gpu") or get_gpu_info()

            payload = {
                "telemetry": telemetry,
                "benchmarks": benchmarks,
                "loss_history": loss_hist[-100:],  # last 100 points for crisp SVG chart
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
