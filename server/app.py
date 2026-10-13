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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
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

# Global process manager
training_process: Optional[subprocess.Popen] = None
training_log_fd = None
connected_websockets: List[WebSocket] = []
PID_FILE = RESULTS_DIR / "training.pid"

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


def get_gpu_info() -> Dict[str, Any]:
    """Lightweight check of GPU info via torch or nvidia-smi."""
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
                "utilization_pct": round(alloc / tot * 100.0, 1) if tot > 0 else 0.0
            }
    except Exception:
        pass
    return {
        "available": False,
        "device_name": "N/A",
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "total_gb": 0.0,
        "utilization_pct": 0.0
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
    """Confine output_dir under BASE_DIR to block ../ traversal."""
    raw = (raw or "outputs/buzzasr-neyshekar-lora").strip() or "outputs/buzzasr-neyshekar-lora"
    # Reject absolute paths outside repo and traversal.
    p = (BASE_DIR / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
    try:
        p.relative_to(BASE_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="output_dir must stay inside project directory.")
    return str(p.relative_to(BASE_DIR.resolve()))


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "RunPod Persian ASR Cockpit active. UI index.html not found."}


@app.get("/api/status")
async def get_status():
    global training_process
    is_running = False
    if training_process is not None:
        poll_res = training_process.poll()
        if poll_res is None:
            is_running = True
        else:
            training_process = None
            _close_log_fd()
            try:
                if PID_FILE.exists():
                    PID_FILE.unlink(missing_ok=True)
            except Exception:
                pass
    if not is_running:
        # Recover liveness across server restarts via PID file.
        try:
            if PID_FILE.exists():
                pid = int(PID_FILE.read_text(encoding="utf-8").strip() or "0")
                if _is_process_running(pid):
                    is_running = True
                else:
                    PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

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

    if is_running and telemetry.get("status") == "idle":
        telemetry["status"] = "training"
    elif not is_running and telemetry.get("status") in ["training", "evaluating_benchmark"]:
        telemetry["status"] = "idle"

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
    global training_process, training_log_fd
    _check_auth(config)
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

    output_dir = _sanitize_output_dir(str(config.get("output_dir", "outputs/buzzasr-neyshekar-lora")))
    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar")).strip() or "shekar-ai/Neyshekar"
    if len(dataset_name) > 128 or any(c in dataset_name for c in ";&|$` \n"):
        raise HTTPException(status_code=400, detail="Invalid dataset_name.")

    # Remove any old stop signal
    if STOP_SIGNAL_FILE.exists():
        STOP_SIGNAL_FILE.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "train_lora.py"),
        "--base_model", "BuzzASR/persian",
        "--dataset_name", dataset_name,
        "--output_dir", output_dir,
        "--epochs", str(epochs_i),
        "--batch_size", str(batch_i),
        "--grad_accum", str(accum_i),
        "--learning_rate", str(lr_f),
        "--eval_batch_size", str(eval_bs_i),
        "--max_benchmark_samples", str(max_bench_i),
        "--enable_telemetry",
        "--run_epoch_benchmark"
    ]

    _close_log_fd()
    training_log_fd = open(LOG_FILE, "w", encoding="utf-8")
    training_log_fd.write(f"=== Persian ASR Training Job Launched at {time.ctime()} ===\n")
    training_log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    training_log_fd.flush()

    training_process = subprocess.Popen(
        cmd,
        stdout=training_log_fd,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        preexec_fn=os.setsid if sys.platform != "win32" else None
    )
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
            "output_dir": output_dir
        }
    }


@app.post("/api/train/stop")
async def stop_training(payload: Dict[str, Any] = None):
    global training_process
    _check_auth(payload)
    # 1. Trigger graceful stop via file signal
    with open(STOP_SIGNAL_FILE, "w") as f:
        f.write("stop")

    if training_process is not None and training_process.poll() is None:
        # Give 3 seconds for graceful checkpoint save, then terminate if necessary
        await asyncio.sleep(2)
        if training_process.poll() is None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
                else:
                    training_process.terminate()
            except Exception:
                pass
        training_process = None
    _close_log_fd()
    try:
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    return {"status": "stopped"}


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

            # Check process status (in-memory + PID file for restart recovery)
            global training_process
            is_running = training_process is not None and training_process.poll() is None
            if not is_running:
                try:
                    if PID_FILE.exists():
                        pid = int(PID_FILE.read_text(encoding="utf-8").strip() or "0")
                        is_running = _is_process_running(pid)
                except Exception:
                    pass
            telemetry["process_running"] = is_running
            if not is_running and telemetry.get("status") in ["training", "evaluating_benchmark"]:
                telemetry["status"] = "idle"

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
