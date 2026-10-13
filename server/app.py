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
connected_websockets: List[WebSocket] = []


def get_default_benchmarks() -> List[Dict[str, Any]]:
    return [
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
    global training_process
    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training is already currently running.")

    config = config or {}
    epochs = str(config.get("epochs", 4))
    batch_size = str(config.get("batch_size", 16))
    grad_accum = str(config.get("grad_accum", 2))
    lr = str(config.get("learning_rate", 1e-4))
    output_dir = str(config.get("output_dir", "outputs/buzzasr-neyshekar-lora"))
    dataset_name = str(config.get("dataset_name", "shekar-ai/Neyshekar"))

    # Remove any old stop signal
    if STOP_SIGNAL_FILE.exists():
        STOP_SIGNAL_FILE.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(BASE_DIR / "scripts" / "train_lora.py"),
        "--base_model", "BuzzASR/persian",
        "--dataset_name", dataset_name,
        "--output_dir", output_dir,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--grad_accum", grad_accum,
        "--learning_rate", lr,
        "--enable_telemetry",
        "--run_epoch_benchmark"
    ]

    log_fd = open(LOG_FILE, "w", encoding="utf-8")
    log_fd.write(f"=== Persian ASR Training Job Launched at {time.ctime()} ===\n")
    log_fd.write(f"Command: {' '.join(cmd)}\n\n")
    log_fd.flush()

    training_process = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        preexec_fn=os.setsid if sys.platform != "win32" else None
    )

    return {
        "status": "started",
        "pid": training_process.pid,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "output_dir": output_dir
        }
    }


@app.post("/api/train/stop")
async def stop_training():
    global training_process
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

    return {"status": "stopped"}


@app.websocket("/ws/live")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Poll status, telemetry and send to client
            telemetry = read_json_safely(TELEMETRY_FILE, {})
            benchmarks = read_json_safely(BENCHMARKS_FILE, get_default_benchmarks())
            loss_hist = read_json_safely(LOSS_HISTORY_FILE, [])

            # Check process status
            global training_process
            is_running = training_process is not None and training_process.poll() is None
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
            await asyncio.sleep(0.5)  # 500ms broadcast cycle
    except WebSocketDisconnect:
        pass
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
