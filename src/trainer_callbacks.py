"""
Custom Hugging Face Trainer Callbacks for Persian ASR Web Dashboard & Telemetry.
Handles:
- Real-time step telemetry (Loss, LR, Step, ETA, VRAM)
- Live log streaming to JSON state file
- End-of-epoch full benchmark evaluation on Neyshekar test set
- Canary audio sentence inspection
- Graceful stop signal handling from web dashboard
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class WebTelemetryCallback(TrainerCallback):
    """
    Captures live training telemetry and persists it for the web dashboard.
    Evaluates benchmark test set at the end of each epoch and computes deltas.
    """

    def __init__(
        self,
        output_dir: str = "outputs/buzzasr-neyshekar-lora",
        results_dir: str = "results",
        test_dataset: Optional[Any] = None,
        processor: Optional[Any] = None,
        metrics_calculator: Optional[Any] = None,
        normalizer: Optional[Any] = None,
        canary_samples: Optional[List[Dict[str, Any]]] = None,
        baseline_benchmark: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.test_dataset = test_dataset
        self.processor = processor
        self.metrics_calculator = metrics_calculator
        self.normalizer = normalizer
        self.canary_samples = canary_samples or []
        
        # Baselines to compare against
        self.baseline_benchmark = baseline_benchmark or {
            "whisper_v3": {
                "name": "Whisper Large v3 (Zero-Shot)",
                "normalized_wer": 28.50,
                "normalized_cer": 8.90,
                "diagnostic_wer": 26.10,
                "formal_wer": 24.00,
                "informal_wer": 36.80,
            },
            "buzzasr_base": {
                "name": "BuzzASR Base (Zero-Shot)",
                "normalized_wer": 17.20,
                "normalized_cer": 4.80,
                "diagnostic_wer": 15.90,
                "formal_wer": 15.10,
                "informal_wer": 22.40,
            }
        }

        os.makedirs(self.results_dir, exist_ok=True)
        self.telemetry_file = os.path.join(self.results_dir, "live_telemetry.json")
        self.benchmarks_file = os.path.join(self.results_dir, "epoch_benchmarks.json")
        self.stop_signal_file = os.path.join(self.results_dir, "stop_training.signal")
        self.loss_history_file = os.path.join(self.results_dir, "loss_history.json")

        # Runtime trackers
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.step_durations: List[float] = []
        self.loss_history: List[Dict[str, Any]] = []

        # Initialize telemetry file
        self._write_telemetry({
            "status": "initializing",
            "step": 0,
            "max_steps": 0,
            "epoch": 0.0,
            "loss": None,
            "learning_rate": 0.0,
            "speed_s_per_step": 0.0,
            "eta_seconds": 0,
            "gpu": self._get_gpu_stats(),
            "updated_at": time.time(),
        })

        # Initialize benchmarks file if not existing
        if not os.path.exists(self.benchmarks_file):
            initial_benchmarks = [
                {
                    "epoch": "Whisper Large v3",
                    "type": "baseline",
                    "is_baseline": True,
                    **self.baseline_benchmark["whisper_v3"]
                },
                {
                    "epoch": "BuzzASR Base",
                    "type": "baseline",
                    "is_baseline": True,
                    **self.baseline_benchmark["buzzasr_base"]
                }
            ]
            self._write_json(self.benchmarks_file, initial_benchmarks)

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Fetch current CUDA VRAM and device metadata."""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_name": "CPU",
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "utilization_pct": 0.0
            }

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        util_pct = (allocated / total * 100.0) if total > 0 else 0.0

        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "utilization_pct": round(util_pct, 1)
        }

    def _write_json(self, path: str, data: Any):
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"[TelemetryCallback] Warning: Failed to write {path}: {e}")

    def _write_telemetry(self, data: Dict[str, Any]):
        self._write_json(self.telemetry_file, data)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        self.last_step_time = time.time()
        print("[TelemetryCallback] Training started. Initializing live telemetry stream...")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 1. Check for web stop signal
        if os.path.exists(self.stop_signal_file):
            print("\n[TelemetryCallback] Stop signal received from Web Dashboard! Halting training...")
            control.should_training_stop = True
            try:
                os.remove(self.stop_signal_file)
            except OSError:
                pass
            return control

        # 2. Timing calculation
        now = time.time()
        step_dt = now - self.last_step_time
        self.last_step_time = now
        
        self.step_durations.append(step_dt)
        if len(self.step_durations) > 50:
            self.step_durations.pop(0)

        avg_step_dt = sum(self.step_durations) / len(self.step_durations)
        remaining_steps = max(0, state.max_steps - state.global_step)
        eta_seconds = int(remaining_steps * avg_step_dt)

        # 3. Extract latest logged values
        latest_loss = None
        latest_lr = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    latest_loss = float(entry["loss"])
                if "learning_rate" in entry:
                    latest_lr = float(entry["learning_rate"])
                if latest_loss is not None and latest_lr is not None:
                    break

        if latest_loss is not None and (not self.loss_history or self.loss_history[-1].get("step") != state.global_step):
            self.loss_history.append({
                "step": state.global_step,
                "loss": round(latest_loss, 4),
                "epoch": round(state.epoch or 0.0, 2),
                "lr": latest_lr
            })
            self._write_json(self.loss_history_file, self.loss_history[-500:])

        # 4. Save live snapshot
        telemetry_payload = {
            "status": "training",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch or 0.0, 2),
            "loss": latest_loss,
            "learning_rate": latest_lr,
            "speed_s_per_step": round(avg_step_dt, 3),
            "elapsed_seconds": int(now - self.start_time),
            "eta_seconds": eta_seconds,
            "gpu": self._get_gpu_stats(),
            "updated_at": now
        }
        self._write_telemetry(telemetry_payload)
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        Triggered when an epoch completes.
        Runs full benchmark evaluation and updates the historical comparison table.
        """
        current_epoch = int(round(state.epoch or 1.0))
        print(f"\n=======================================================")
        print(f" [TelemetryCallback] Epoch {current_epoch} Complete!")
        print(f" Running Benchmark Evaluation across held-out Test Set...")
        print(f"=======================================================\n")

        # Update telemetry status to "evaluating"
        self._write_telemetry({
            "status": "evaluating_benchmark",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": current_epoch,
            "eval_epoch": current_epoch,
            "gpu": self._get_gpu_stats(),
            "updated_at": time.time()
        })

        epoch_results = self._evaluate_benchmark(model, current_epoch)
        self._record_epoch_benchmark(current_epoch, epoch_results)

        # Restore telemetry status
        self._write_telemetry({
            "status": "training",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": current_epoch,
            "gpu": self._get_gpu_stats(),
            "updated_at": time.time()
        })
        return control

    def _evaluate_benchmark(self, model, epoch_num: int) -> Dict[str, Any]:
        """Runs inference over the test dataset split and computes metrics."""
        if not self.test_dataset or not self.processor or not self.metrics_calculator:
            print("[TelemetryCallback] Test dataset or calculator not provided, skipping benchmark eval.")
            return {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32

        model.eval()
        total_test = len(self.test_dataset)
        batch_size = 16
        predictions = []
        references = [sample.get("text", "") or "" for sample in self.test_dataset]

        is_informal = []
        for s in self.test_dataset:
            if "informal" in s:
                is_informal.append(bool(s["informal"]))
            elif "is_informal" in s:
                is_informal.append(bool(s["is_informal"]))
            else:
                is_informal.append(self.normalizer.is_colloquial(s.get("text", "")))

        with torch.inference_mode():
            for i in range(0, total_test, batch_size):
                batch = self.test_dataset[i : i + batch_size]
                audios = batch["audio"] if isinstance(batch["audio"], list) else [batch["audio"]]
                waveforms = [a["array"] for a in audios]

                inputs = self.processor(
                    waveforms,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)

                with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                    pred_ids = model.generate(inputs, max_new_tokens=225)

                batch_preds = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
                predictions.extend(batch_preds)

        # Compute full metrics
        metrics = self.metrics_calculator.evaluate_predictions(
            references=references,
            predictions=predictions,
            is_informal_flags=is_informal
        )

        # Evaluate Canary Sentences (if provided)
        canary_evals = []
        for canary in self.canary_samples:
            ref = canary.get("text", "")
            audio_arr = canary.get("audio_array")
            if audio_arr is not None:
                inp = self.processor(
                    audio_arr, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                    c_pred_ids = model.generate(inp, max_new_tokens=225)
                c_pred = self.processor.batch_decode(c_pred_ids, skip_special_tokens=True)[0]
                canary_evals.append({
                    "label": canary.get("label", "Canary"),
                    "category": canary.get("category", "General"),
                    "ref": ref,
                    "pred": c_pred,
                    "is_informal": canary.get("is_informal", False)
                })

        metrics["canary_transcriptions"] = canary_evals
        model.train()
        return metrics

    def _record_epoch_benchmark(self, epoch_num: int, results: Dict[str, Any]):
        """Persists epoch metrics and calculates deltas vs. baselines."""
        overall = results.get("overall", {})
        formal = results.get("formal", {})
        informal = results.get("informal", {})

        norm_wer = overall.get("normalized_wer", 0.0)
        norm_cer = overall.get("normalized_cer", 0.0)
        diag_wer = overall.get("diagnostic_wer_zwnj_space", 0.0)
        formal_wer = formal.get("normalized_wer", 0.0)
        informal_wer = informal.get("normalized_wer", 0.0)

        # Baseline comparisons
        whisper_wer = self.baseline_benchmark["whisper_v3"]["normalized_wer"]
        delta_vs_whisper = round(norm_wer - whisper_wer, 2)  # negative is improvement

        buzzasr_wer = self.baseline_benchmark["buzzasr_base"]["normalized_wer"]
        delta_vs_buzzasr = round(norm_wer - buzzasr_wer, 2)

        record = {
            "epoch": f"Epoch {epoch_num}",
            "epoch_number": epoch_num,
            "type": "checkpoint",
            "is_baseline": False,
            "normalized_wer": round(norm_wer, 2),
            "normalized_cer": round(norm_cer, 2),
            "diagnostic_wer": round(diag_wer, 2),
            "formal_wer": round(formal_wer, 2),
            "informal_wer": round(informal_wer, 2),
            "delta_vs_whisper": delta_vs_whisper,
            "delta_vs_buzzasr": delta_vs_buzzasr,
            "total_samples": results.get("total_samples", 0),
            "canary_transcriptions": results.get("canary_transcriptions", []),
            "timestamp": time.time()
        }

        # Read existing records
        records = []
        if os.path.exists(self.benchmarks_file):
            try:
                with open(self.benchmarks_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []

        # Remove duplicate epoch entry if re-evaluated
        records = [r for r in records if r.get("epoch") != f"Epoch {epoch_num}"]
        records.append(record)
        self._write_json(self.benchmarks_file, records)
        print(f"[TelemetryCallback] Epoch {epoch_num} benchmarks saved to {self.benchmarks_file}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._write_telemetry({
            "status": "completed",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch or 0.0, 2),
            "elapsed_seconds": int(time.time() - self.start_time),
            "gpu": self._get_gpu_stats(),
            "updated_at": time.time()
        })
        print("\n[TelemetryCallback] Training complete. Web telemetry finalized.")
