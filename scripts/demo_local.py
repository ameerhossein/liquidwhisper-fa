"""
Zero-Dependency Local Demo Server for Persian ASR Telemetry Cockpit.
Runs completely on Python 3 Standard Library (no pip dependencies required).
Simulates realistic training, loss descent, GPU telemetry, and epoch benchmark metrics.
"""

import os
import sys
import json
import time
import math
import random
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "server" / "static"

# Load .env (repo root / CWD) before any os.environ reads. Stdlib-only,
# so the local demo keeps its zero-dependency (no pip) property.
sys.path.insert(0, str(BASE_DIR))
try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass


def _load_demo_baselines():
    """Load shared baselines so demo matches server/training numbers."""
    defaults = (28.50, 8.90, 26.10, 24.00, 36.80, 17.20, 4.80, 15.90, 15.10, 22.40)
    try:
        with open(BASE_DIR / "configs" / "baselines.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        w, b = data.get("whisper_v3", {}), data.get("buzzasr_base", {})
        return (
            float(w.get("normalized_wer", defaults[0])),
            float(w.get("normalized_cer", defaults[1])),
            float(w.get("diagnostic_wer", defaults[2])),
            float(w.get("formal_wer", defaults[3])),
            float(w.get("informal_wer", defaults[4])),
            float(b.get("normalized_wer", defaults[5])),
            float(b.get("normalized_cer", defaults[6])),
            float(b.get("diagnostic_wer", defaults[7])),
            float(b.get("formal_wer", defaults[8])),
            float(b.get("informal_wer", defaults[9])),
        )
    except Exception:
        return defaults


_WHISPER_WER, _WHISPER_CER, _WHISPER_DIAG, _WHISPER_FORMAL, _WHISPER_INFORMAL, \
    _BUZZ_WER, _BUZZ_CER, _BUZZ_DIAG, _BUZZ_FORMAL, _BUZZ_INFORMAL = _load_demo_baselines()

DEMO_FAIL_AT_STEP = int(os.environ.get("DEMO_FAIL_AT_STEP", "0") or 0)

# Simulation State
sim_state = {
    "is_running": False,
    "status": "idle",
    "step": 0,
    "max_steps": 7280,
    "epoch": 0.0,
    "current_epoch_int": 0,
    "loss": 0.4850,
    "learning_rate": 1e-4,
    "speed_s_per_step": 0.12,
    "elapsed_seconds": 0,
    "eta_seconds": 1800,
    "gpu": {
        "available": True,
        "device_name": "NVIDIA L40S (Simulation Mode)",
        "allocated_gb": 21.4,
        "reserved_gb": 24.0,
        "total_gb": 48.0,
        "utilization_pct": 44.6
    }
}

sim_benchmarks = [
    {
        "epoch": "Whisper Large v3 (Zero-Shot)",
        "epoch_number": 0,
        "type": "baseline",
        "is_baseline": True,
        "normalized_wer": _WHISPER_WER,
        "normalized_cer": _WHISPER_CER,
        "diagnostic_wer": _WHISPER_DIAG,
        "formal_wer": _WHISPER_FORMAL,
        "informal_wer": _WHISPER_INFORMAL,
        "delta_vs_whisper": 0.0,
        "total_samples": 2149,
        "canary_transcriptions": [
            {
                "category": "ZWNJ / نیم‌فاصله",
                "ref": "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
                "pred": "دانش اموزان کتابهای خود را در کیف می گذارند.",
                "is_informal": False
            },
            {
                "category": "محاوره‌ای / Spoken",
                "ref": "من الان می‌خوام برم خونه چون خیلی خسته‌ام و کار دارم.",
                "pred": "من الان می خواهم بروم به خانه چون خیلی خسته ام و کار دارم.",
                "is_informal": True
            },
            {
                "category": "معیار / Formal",
                "ref": "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد.",
                "pred": "بر اساس آخرین گزارشهای اقتصادی نرخ تورم اعلام شد.",
                "is_informal": False
            }
        ]
    },
    {
        "epoch": "BuzzASR Base (Zero-Shot)",
        "epoch_number": 0,
        "type": "baseline",
        "is_baseline": True,
        "normalized_wer": _BUZZ_WER,
        "normalized_cer": _BUZZ_CER,
        "diagnostic_wer": _BUZZ_DIAG,
        "formal_wer": _BUZZ_FORMAL,
        "informal_wer": _BUZZ_INFORMAL,
        "delta_vs_whisper": round(_BUZZ_WER - _WHISPER_WER, 2),
        "total_samples": 2149,
        "canary_transcriptions": []
    }
]

sim_loss_history = []
sim_logs = [
    "[Demo Engine] Local simulation initialized.",
    "[Demo Engine] Hardware: NVIDIA L40S 48GB (Simulated)",
    "[Demo Engine] Ready. Click 'Launch Run' to simulate active training."
]


def simulation_worker():
    """Background thread that advances the simulation when active."""
    epoch_targets = [
        # (wer, cer, diag, formal, informal, canary_preds)
        (13.80, 3.90, 12.90, 12.20, 17.50, [
            "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
            "من الان می‌خواهم برم خونه چون خسته‌ام و کار دارم.",
            "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد."
        ]),
        (10.60, 2.80, 9.80, 9.50, 13.10, [
            "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
            "من الان می‌خوام برم خونه چون خیلی خسته‌ام و کار دارم.",
            "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد."
        ]),
        (8.90, 2.10, 8.10, 7.90, 10.80, [
            "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
            "من الان می‌خوام برم خونه چون خیلی خسته‌ام و کار دارم.",
            "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد."
        ]),
        (7.95, 1.85, 7.30, 7.10, 9.60, [
            "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
            "من الان می‌خوام برم خونه چون خیلی خسته‌ام و کار دارم.",
            "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد."
        ])
    ]

    while True:
        time.sleep(0.5)
        if sim_state["status"] == "stopping":
            time.sleep(2.0)  # Simulated checkpoint save before halt
            sim_state.update({"is_running": False, "status": "stopped"})
            sim_logs.append("[Demo Engine] >>> Checkpoint saved. Halted by user. (Launch resumes.)")
            continue
        if sim_state["status"] == "initializing":
            time.sleep(2.0)  # Simulated model and dataset loading
            sim_state["status"] = "training"
            sim_logs.append("[Demo Engine] >>> Model and data ready. Training started.")
            continue
        if not sim_state["is_running"]:
            continue

        sim_state["step"] += 25
        sim_state["elapsed_seconds"] += 1
        sim_state["eta_seconds"] = max(0, sim_state["eta_seconds"] - 3)

        if DEMO_FAIL_AT_STEP and sim_state["step"] >= DEMO_FAIL_AT_STEP:
            sim_state.update({"is_running": False, "status": "failed"})
            sim_logs.append("Traceback (most recent call last):")
            sim_logs.append("RuntimeError: CUDA out of memory (simulated failure, DEMO_FAIL_AT_STEP)")
            continue

        # Smooth loss curve decay
        current_step = sim_state["step"]
        decay_factor = math.exp(-current_step / 2500)
        sim_state["loss"] = round(0.08 + 0.38 * decay_factor + random.uniform(-0.005, 0.005), 4)
        sim_state["learning_rate"] = max(1e-6, 1e-4 * (1 - current_step / sim_state["max_steps"]))

        sim_loss_history.append({
            "step": current_step,
            "loss": sim_state["loss"],
            "epoch": round(current_step / (sim_state["max_steps"] / 4), 2),
            "lr": sim_state["learning_rate"]
        })

        if current_step % 100 == 0:
            sim_logs.append(f"Step {current_step}/{sim_state['max_steps']} | Loss: {sim_state['loss']:.4f} | LR: {sim_state['learning_rate']:.2e} | VRAM: 21.4 GB")
            if len(sim_logs) > 200:
                sim_logs.pop(0)

        # Trigger epoch evaluation every ~1800 steps
        epoch_idx = current_step // 1800
        if epoch_idx > sim_state["current_epoch_int"] and epoch_idx <= len(epoch_targets):
            sim_state["current_epoch_int"] = epoch_idx
            sim_state["status"] = "evaluating_benchmark"
            sim_logs.append(f"\n>>> [Epoch {epoch_idx}] Evaluation Started on held-out test split (2,149 samples)...")
            # Halt may arrive during the pause below; the status must not be overwritten.
            time.sleep(2.0)  # Pause to show evaluation status

            metrics = epoch_targets[epoch_idx - 1]
            canary_cards = [
                {
                    "category": "ZWNJ / نیم‌فاصله",
                    "ref": "دانش‌آموزان کتاب‌های خود را در کیف می‌گذارند.",
                    "pred": metrics[5][0],
                    "is_informal": False
                },
                {
                    "category": "محاوره‌ای / Spoken",
                    "ref": "من الان می‌خوام برم خونه چون خیلی خسته‌ام و کار دارم.",
                    "pred": metrics[5][1],
                    "is_informal": True
                },
                {
                    "category": "معیار / Formal",
                    "ref": "بر اساس آخرین گزارش‌های اقتصادی نرخ تورم اعلام شد.",
                    "pred": metrics[5][2],
                    "is_informal": False
                }
            ]

            delta_whisper = round(metrics[0] - _WHISPER_WER, 2)
            delta_buzz = round(metrics[0] - _BUZZ_WER, 2)
            sim_benchmarks.append({
                "epoch": f"Epoch {epoch_idx}",
                "epoch_number": epoch_idx,
                "type": "checkpoint",
                "is_baseline": False,
                "normalized_wer": metrics[0],
                "normalized_cer": metrics[1],
                "diagnostic_wer": metrics[2],
                "formal_wer": metrics[3],
                "informal_wer": metrics[4],
                "delta_vs_whisper": delta_whisper,
                "delta_vs_buzzasr": delta_buzz,
                "total_samples": 2149,
                "canary_transcriptions": canary_cards,
                "timestamp": time.time()
            })
            sim_logs.append(f">>> [Epoch {epoch_idx}] Benchmark Finished! Normalized WER: {metrics[0]}% (Improvement: {delta_whisper}% vs Whisper v3, {delta_buzz}% vs BuzzASR)")
            if sim_state["status"] == "evaluating_benchmark":
                sim_state["status"] = "training"

        sim_state["epoch"] = round(current_step / 1820, 2)

        # Auto-stop at max_steps so a second Launch starts a fresh run.
        if current_step >= sim_state["max_steps"]:
            sim_state["is_running"] = False
            sim_state["status"] = "completed"
            sim_logs.append("[Demo Engine] >>> Simulation complete. Launch again for a fresh run.")


class LocalDemoHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence verbose logging
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(STATIC_DIR / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        if path.startswith("/static/"):
            rel_path = path.replace("/static/", "", 1)
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                if rel_path.endswith(".css"):
                    self.send_header("Content-Type", "text/css")
                elif rel_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            payload = dict(sim_state)
            payload["process_running"] = sim_state["is_running"]
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        if path == "/api/benchmarks":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(sim_benchmarks).encode("utf-8"))
            return

        if path == "/api/loss-history":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(sim_loss_history[-100:]).encode("utf-8"))
            return

        if path.startswith("/api/logs"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"logs": sim_logs[-80:]}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/api/train/start":
            # Fresh-run semantics: if previous simulation completed, reset.
            # Otherwise stop→start resumes (documented cumulative behavior).
            if sim_state.get("step", 0) >= sim_state.get("max_steps", 7280):
                sim_state.update({
                    "step": 0, "epoch": 0.0, "current_epoch_int": 0,
                    "loss": 0.4850, "learning_rate": 1e-4,
                    "elapsed_seconds": 0, "eta_seconds": 1800,
                })
                sim_loss_history.clear()
                # Keep the two baselines, drop epoch checkpoints.
                del sim_benchmarks[2:]
                sim_logs.append("[Demo Engine] >>> Previous run completed; state reset for fresh run.")
            sim_state["is_running"] = True
            sim_state["status"] = "initializing"
            sim_logs.append("[Demo Engine] >>> Training initiated by user. Initializing model and data...")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "pid": 9999}).encode("utf-8"))
            return

        if path == "/api/train/stop":
            if sim_state["is_running"]:
                sim_state["status"] = "stopping"
                sim_logs.append("[Demo Engine] >>> Halt requested. Saving checkpoint...")
                result = "stopping"
            else:
                result = "not_running"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": result}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def main():
    port = int(os.environ.get("PORT", 7860))
    host = "127.0.0.1"

    # Start simulation background thread
    t = threading.Thread(target=simulation_worker, daemon=True)
    t.start()

    server = HTTPServer((host, port), LocalDemoHandler)
    print("\n" + "=" * 62)
    print(" Persian ASR Cockpit - Local Simulation Demo")
    print(f" Web UI running at: http://{host}:{port}")
    print(" Zero external dependencies required (Pure Python 3)")
    print(" Press Ctrl+C in this terminal to stop.")
    print("=" * 62 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server...")
        server.server_close()


if __name__ == "__main__":
    main()
