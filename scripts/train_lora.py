"""
Fine-tune BuzzASR/persian on Neyshekar using 16-bit LoRA (PEFT).
Optimized for NVIDIA L40S (48 GB VRAM):
- Bfloat16 training
- LoRA on attention & feed-forward projections (r=64, alpha=128)
- Fused AdamW optimizer
- Scaled Dot-Product Attention / FlashAttention
- Native Persian text normalization for validation WER
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import evaluate
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalizer import PersianTextNormalizer
from src.metrics import ASRMetricsCalculator
from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.data import PersianASRDataPipeline
from src.trainer_callbacks import WebTelemetryCallback, load_baselines, _resolve_infer_dtype, _should_autocast
from src import paths as _managed_paths


def _default_output_dir() -> str:
    """Managed default (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.default_output_dir()


def _default_results_dir() -> str:
    """Managed default (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.default_results_dir()


def _resolve_results_dir(raw: Optional[str]) -> str:
    """Managed resolver (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.resolve_results_dir(raw)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BuzzASR with LoRA on Persian Speech")
    # Model & Data
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. configs/l40s_train_config.yaml). CLI flags override file.")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--dataset_name", type=str, default="shekar-ai/Neyshekar", help="Dataset identifier")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(), help="Output directory for checkpoints (default: /workspace/... on RunPod, else outputs/...)")
    parser.add_argument("--results_dir", type=str, default=None, help="Telemetry/benchmark directory (default: /workspace/stt-outputs/results on RunPod, else results/). Env ASR_RESULTS_DIR also works.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (smoke test)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples")

    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training Hyperparameters (L40S tuned)
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch: batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for LoRA")
    parser.add_argument("--epochs", type=int, default=4, help="Total training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Explicit checkpoint to resume from: path, or 'last' for latest in output_dir. Overrides --auto_resume.")
    parser.add_argument("--auto_resume", "--auto-resume", action=argparse.BooleanOptionalAction, default=True, help="Automatically resume from latest checkpoint in output_dir (for pod restarts). Use --no-auto_resume for a fresh run.")
    parser.add_argument("--fresh_start", "--fresh-start", action=argparse.BooleanOptionalAction, default=False, help="Start fresh: delete prior checkpoint-*, best_adapter, reports and telemetry history in output_dir/results before training.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of preprocessing workers")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device eval + benchmark batch size")
    parser.add_argument("--max_benchmark_samples", type=int, default=500, help="Test subset for end-of-epoch benchmark, decoded twice (500 clean + same 500 phone). 0/None = full set (2149).")
    parser.add_argument("--run_post_test_eval", "--run-post-test-eval", action=argparse.BooleanOptionalAction, default=True, help="Run full test evaluation on best model post-training and save report")
    parser.add_argument("--enable_telemetry", "--enable-telemetry", action=argparse.BooleanOptionalAction, default=True, help="Enable real-time telemetry for Web UI (--no-enable-telemetry to disable)")
    parser.add_argument("--run_epoch_benchmark", "--run-epoch-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Run test benchmark after each epoch (--no-run-epoch-benchmark to disable)")
    parser.add_argument("--telephony_prob", type=float, default=0.0, help="Share of train samples phone-simulated on-the-fly (0=clean only, 0.5=mixed). YAML telephony.prob enables 0.5 on L40S.")
    parser.add_argument("--telephony_profiles", type=str, default="mobile,pstn", help="Comma-separated phone profiles: mobile,pstn (MCI/Irancell cellular vs TCI landline)")
    parser.add_argument("--telephony_seed", type=int, default=42, help="RNG seed for training-time phone augmentation")
    parser.add_argument("--telephony_use_ffmpeg", "--telephony-use-ffmpeg", action=argparse.BooleanOptionalAction, default=True, help="Use true ffmpeg AMR-NB/G.711 codecs when available (fallback to lite otherwise)")
    parser.add_argument("--phone_benchmark", "--phone-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Dual clean + phone eval each epoch (--no-phone-benchmark for clean only)")
    parser.add_argument("--phone_seed", type=int, default=1234, help="Deterministic seed for phone-simulated eval")
    args = parser.parse_args()
    # YAML config overlay: file provides defaults, CLI wins only if explicitly passed.
    if args.config:
        if not _YAML_AVAILABLE:
            print("Warning: PyYAML not installed; ignoring --config. pip install pyyaml")
        else:
            _apply_yaml_config(parser, args, args.config)
    return args


def _apply_yaml_config(parser: argparse.ArgumentParser, args: argparse.Namespace, path: str):
    """Overlay YAML file onto parsed args. Only overrides defaults, not explicit CLI flags."""
    import sys as _sys
    cli_flags = {a.split("=")[0] for a in _sys.argv[1:] if a.startswith("--")}
    # Map --no-xxx back to base flag so explicit disables are not clobbered.
    normalized_cli = set()
    for f in cli_flags:
        if f.startswith("--no-"):
            normalized_cli.add("--" + f[5:].replace("-", "_"))
        else:
            normalized_cli.add(f)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load config {path}: {e}")
        return
    flat: Dict[str, Any] = {}
    for section in ("model", "lora", "training", "benchmark", "telephony"):
        sub = cfg.get(section, {})
        if isinstance(sub, dict):
            flat.update(sub)
    # Flat top-level keys also allowed.
    for k, v in cfg.items():
        if not isinstance(v, dict):
            flat[k] = v
    key_map = {
        "base_model": "base_model", "dataset_name": "dataset_name",
        "output_dir": "output_dir", "r": "lora_r", "lora_r": "lora_r",
        "alpha": "lora_alpha", "lora_alpha": "lora_alpha",
        "dropout": "lora_dropout", "lora_dropout": "lora_dropout",
        "batch_size": "batch_size", "grad_accum": "grad_accum",
        "learning_rate": "learning_rate", "epochs": "epochs",
        "num_train_epochs": "epochs", "warmup_steps": "warmup_steps",
        "eval_steps": "eval_steps", "save_steps": "save_steps",
        "save_total_limit": "save_total_limit", "num_workers": "num_proc",
        "num_proc": "num_proc", "eval_batch_size": "eval_batch_size",
        "max_benchmark_samples": "max_benchmark_samples",
        "prob": "telephony_prob", "telephony_prob": "telephony_prob",
        "profiles": "telephony_profiles", "telephony_profiles": "telephony_profiles",
        "seed": "telephony_seed", "telephony_seed": "telephony_seed",
        "use_ffmpeg": "telephony_use_ffmpeg", "telephony_use_ffmpeg": "telephony_use_ffmpeg",
        "phone_benchmark": "phone_benchmark", "phone_seed": "phone_seed",
        "resume_from_checkpoint": "resume_from_checkpoint",
        "auto_resume": "auto_resume", "auto-resume": "auto_resume",
        "fresh_start": "fresh_start", "fresh-start": "fresh_start",
        "results_dir": "results_dir", "results-dir": "results_dir",
    }
    _BOOL_KEYS = {"auto_resume", "fresh_start", "phone_benchmark"}
    for yk, ak in key_map.items():
        if yk in flat:
            cli_name = "--" + ak
            if cli_name not in normalized_cli and ("--" + ak.replace("_", "-")) not in normalized_cli:
                if ak == "telephony_profiles":
                    # Keep list from YAML as-is; CLI string normalized later.
                    setattr(args, ak, flat[yk])
                    continue
                try:
                    if ak in _BOOL_KEYS:
                        v = flat[yk]
                        if isinstance(v, str):
                            setattr(args, ak, v.strip().lower() in ("1", "true", "yes", "y", "on"))
                        else:
                            setattr(args, ak, bool(v))
                    else:
                        setattr(args, ak, type(getattr(args, ak))(flat[yk]) if getattr(args, ak) is not None else flat[yk])
                except Exception:
                    setattr(args, ak, flat[yk])
    # Target modules are code-fixed; warn if config disagrees.
    if isinstance(cfg.get("lora", {}).get("target_modules"), list):
        print(f"Note: lora.target_modules from {path} ignored (code-fixed).")


def _parse_telephony_profiles(value) -> list:
    if value is None:
        return ["mobile", "pstn"]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _resolve_output_dir(raw: str) -> str:
    """Managed resolver (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.resolve_output_dir(raw)


def _clear_previous_state(output_dir: str, results_dir: Optional[str] = None) -> None:
    """Delete prior run state so --fresh_start truly starts clean.

    Removes only known training artifacts inside output_dir
    (checkpoint-*, best_adapter, final reports, runs/) and the live
    telemetry history in results_dir (loss_history, epoch benchmarks are
    rebuilt with baselines by the callback). Never deletes output_dir itself.
    """
    import shutil
    removed = []
    rdir = Path(_resolve_results_dir(results_dir))
    try:
        base = Path(output_dir)
        if base.is_dir():
            for ckpt in sorted(base.glob("checkpoint-*")):
                try:
                    if ckpt.is_dir():
                        shutil.rmtree(ckpt, ignore_errors=True)
                        removed.append(ckpt.name)
                except Exception:
                    continue
            for name in ("best_adapter", "runs"):
                try:
                    t = base / name
                    if t.is_dir():
                        shutil.rmtree(t, ignore_errors=True)
                        removed.append(name)
                except Exception:
                    continue
            for name in ("final_evaluation_report.json", "final_evaluation_report.md", "trainer_state.json"):
                try:
                    t = base / name
                    if t.is_file():
                        t.unlink(missing_ok=True)
                        removed.append(name)
                except Exception:
                    continue
    except Exception as e:
        print(f"Warning: fresh-start output cleanup failed: {e}")
    try:
        for name in ("loss_history.json", "live_telemetry.json", "training_live.log"):
            try:
                t = rdir / name
                if t.is_file():
                    t.unlink(missing_ok=True)
                    removed.append(f"results/{name}")
            except Exception:
                continue
        # Reset epoch benchmarks to baselines-only (callback rebuilds if missing).
        try:
            bfile = rdir / "epoch_benchmarks.json"
            if bfile.is_file():
                bfile.unlink(missing_ok=True)
                removed.append("results/epoch_benchmarks.json")
        except Exception:
            pass
        # Clear stale stop signal so the fresh run is not halted immediately.
        try:
            sfile = rdir / "stop_training.signal"
            if sfile.is_file():
                sfile.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: fresh-start results cleanup failed: {e}")
    if removed:
        print(f"Fresh start: cleared {', '.join(removed)}")


def _find_last_checkpoint(output_dir: str) -> Optional[str]:
    """Managed lookup (see src/paths.find_last_checkpoint). Wrapper for backward compat."""
    return _managed_paths.find_last_checkpoint(output_dir)


def _resolve_resume_checkpoint(args) -> Optional[bool | str]:
    """Decide what to pass as trainer.train(resume_from_checkpoint=...).

    Returns None (fresh), True (let Trainer pick latest), or a path.
    Explicit --resume_from_checkpoint wins; otherwise --auto_resume picks up
    the latest checkpoint in --output_dir (which should live on the persistent
    volume, e.g. /workspace/stt-outputs/...). --fresh_start disables resume.
    """
    if bool(getattr(args, "fresh_start", False)):
        return None
    explicit = (getattr(args, "resume_from_checkpoint", None) or "").strip() if isinstance(getattr(args, "resume_from_checkpoint", None), str) else getattr(args, "resume_from_checkpoint", None)
    if explicit:
        low = str(explicit).strip().lower()
        if low in ("last", "true", "yes", "1", "auto"):
            last = _find_last_checkpoint(args.output_dir)
            if last:
                print(f"Resuming from latest checkpoint: {last}")
                return last
            print("No checkpoint found in output_dir; starting fresh.")
            return None
        p = Path(str(explicit))
        if p.is_dir():
            print(f"Resuming from explicit checkpoint: {p}")
            return str(p)
        print(f"Warning: --resume_from_checkpoint '{explicit}' not found; starting fresh.")
        return None
    if bool(getattr(args, "auto_resume", True)):
        last = _find_last_checkpoint(args.output_dir)
        if last:
            print(f"Auto-resume: found checkpoint {last}")
            return last
    return None


def main():
    args = parse_args()
    args.telephony_profiles = _parse_telephony_profiles(getattr(args, "telephony_profiles", "mobile,pstn"))
    args.output_dir = _resolve_output_dir(args.output_dir)
    args.results_dir = _resolve_results_dir(getattr(args, "results_dir", None))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    if bool(getattr(args, "fresh_start", False)):
        _clear_previous_state(args.output_dir, results_dir=args.results_dir)

    print("=======================================================")
    print(" Persian ASR Fine-Tuning: BuzzASR + Neyshekar LoRA")
    print(f" Base Model:      {args.base_model}")
    print(f" Dataset:         {args.dataset_name}")
    print(f" Output Dir:      {args.output_dir}")
    print(f" Results Dir:     {args.results_dir}")
    print(f" LoRA Rank:       r={args.lora_r}, alpha={args.lora_alpha}")
    print(f" Batch Size:      {args.batch_size} (accum: {args.grad_accum})")
    print(f" Telephony:       prob={getattr(args, 'telephony_prob', 0.0)} profiles={args.telephony_profiles} ffmpeg={getattr(args, 'telephony_use_ffmpeg', True)}")
    print(f" Phone Benchmark: {getattr(args, 'phone_benchmark', True)} (seed={getattr(args, 'phone_seed', 1234)})")
    print(f" CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f" BF16 Supported:  {torch.cuda.is_bf16_supported()}")
    print("=======================================================\n")

    # 1. Load Processor
    print(f"Loading processor for '{args.base_model}'...")
    processor = WhisperProcessor.from_pretrained(args.base_model)
    normalizer = PersianTextNormalizer()

    # 2. Load Base Model in BF16 (fallback to fp32 on CPU)
    try:
        _bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        _bf16_ok = False
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if _bf16_ok else torch.float16
    else:
        dtype = torch.float32
    print(f"Loading base model in {dtype}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # Disable cache when training with gradient checkpointing
    model.config.use_cache = False

    # 3. Configure LoRA
    print("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare Data (train gets phone augmentation, eval stays clean)
    train_pipeline = PersianASRDataPipeline(
        processor=processor,
        dataset_name=args.dataset_name,
        normalizer=normalizer,
        num_proc=args.num_proc,
        telephony_prob=float(getattr(args, "telephony_prob", 0.0) or 0.0),
        telephony_profiles=args.telephony_profiles,
        telephony_seed=int(getattr(args, "telephony_seed", 42) or 42),
        telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
    )
    eval_pipeline = PersianASRDataPipeline(
        processor=processor,
        dataset_name=args.dataset_name,
        normalizer=normalizer,
        num_proc=args.num_proc,
        telephony_prob=0.0,
    )

    raw_train = train_pipeline.load_raw_dataset(split="train", max_samples=args.max_train_samples)
    raw_eval = eval_pipeline.load_raw_dataset(split="validation", max_samples=args.max_eval_samples)
    raw_test = None
    if args.run_epoch_benchmark:
        print("Loading full test benchmark split for end-of-epoch evaluation...")
        raw_test = eval_pipeline.load_raw_dataset(split="test")

    train_dataset = train_pipeline.prepare_dataset_split(raw_train)
    eval_dataset = eval_pipeline.prepare_dataset_split(raw_eval)

    # 5. Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. Evaluation Metric Function & Callbacks
    wer_metric = evaluate.load("wer")
    metrics_calculator = ASRMetricsCalculator(normalizer=normalizer)

    callbacks = []
    if args.enable_telemetry:
        telemetry_cb = WebTelemetryCallback(
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            test_dataset=raw_test,
            processor=processor,
            metrics_calculator=metrics_calculator,
            normalizer=normalizer,
            eval_batch_size=args.eval_batch_size,
            max_benchmark_samples=args.max_benchmark_samples,
            phone_benchmark=bool(getattr(args, "phone_benchmark", True)),
            phone_seed=int(getattr(args, "phone_seed", 1234) or 1234),
            telephony_profiles=args.telephony_profiles,
            telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
        )
        callbacks.append(telemetry_cb)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize with Persian text normalizer
        norm_preds = [normalizer.normalize(p, remove_punctuation=True, zwnj_mode="keep") for p in pred_str]
        norm_labels = [normalizer.normalize(l, remove_punctuation=True, zwnj_mode="keep") for l in label_str]

        wer = 100 * wer_metric.compute(predictions=norm_preds, references=norm_labels)
        return {"wer": wer}

    # 7. Training Arguments (Configured for L40S)
    try:
        use_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16 and torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_proc,
        remove_unused_columns=False,
    )

    # 8. Initialize Trainer (transformers>=4.46 uses processing_class)
    _trainer_common = dict(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    try:
        trainer = Seq2SeqTrainer(processing_class=processor, **_trainer_common)
    except TypeError:
        # Older transformers: fall back to tokenizer kwarg.
        try:
            trainer = Seq2SeqTrainer(tokenizer=processor.tokenizer, **_trainer_common)
        except Exception:
            trainer = Seq2SeqTrainer(tokenizer=processor, **_trainer_common)

    # 9. Execute Training (auto-resumable from persistent-volume checkpoints)
    resume_ckpt = _resolve_resume_checkpoint(args)
    if resume_ckpt:
        print(f"\nResuming training from: {resume_ckpt}")
    else:
        print("\nStarting training loop (fresh)...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 10. Save Best Adapter & Processor
    best_save_path = os.path.join(args.output_dir, "best_adapter")
    print(f"\nTraining completed! Saving best LoRA adapter to: {best_save_path}")
    trainer.model.save_pretrained(best_save_path)
    processor.save_pretrained(best_save_path)
    print("Done saving best adapter.")

    # 11. Run Full Post-Training Test Benchmark & Generate Final Report
    if args.run_post_test_eval:
        print("\n=======================================================")
        print(" Running Post-Training Full Benchmark on Complete Test Set...")
        print("=======================================================")
        _run_post_training_evaluation(
            model=trainer.model,
            processor=processor,
            metrics_calculator=metrics_calculator,
            normalizer=normalizer,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            eval_batch_size=args.eval_batch_size,
            trainer_state=trainer.state,
            phone_benchmark=bool(getattr(args, "phone_benchmark", True)),
            phone_seed=int(getattr(args, "phone_seed", 1234) or 1234),
            telephony_profiles=args.telephony_profiles,
            telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
        )


def _run_post_training_evaluation(
    model,
    processor,
    metrics_calculator: ASRMetricsCalculator,
    normalizer: PersianTextNormalizer,
    dataset_name: str,
    output_dir: str,
    results_dir: Optional[str] = None,
    eval_batch_size: int = 16,
    trainer_state: Optional[Any] = None,
    phone_benchmark: bool = True,
    phone_seed: int = 1234,
    telephony_profiles: Optional[list] = None,
    telephony_use_ffmpeg: bool = True,
):
    import time

    phone_profiles = list(telephony_profiles or ["mobile", "pstn"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_infer_dtype(device)
    use_amp = _should_autocast(device)

    # 1. Load full test dataset
    pipeline = PersianASRDataPipeline(processor=processor, dataset_name=dataset_name)
    test_ds = pipeline.load_raw_dataset(split="test")
    total_samples = len(test_ds)
    print(f"Loaded full test split: {total_samples} samples.")

    references = [s.get("text", "") or "" for s in test_ds]
    is_informal = []
    for s in test_ds:
        if "informal" in s:
            is_informal.append(bool(s["informal"]))
        elif "is_informal" in s:
            is_informal.append(bool(s["is_informal"]))
        else:
            is_informal.append(normalizer.is_colloquial(s.get("text", "")))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _transcribe(label: str, degrade_fn=None):
        preds = []
        audio_seconds = 0.0
        print(f"Transcribing {total_samples} samples ({label}, batch size = {eval_batch_size})...")
        for i in range(0, total_samples, eval_batch_size):
            batch = test_ds[i : i + eval_batch_size]
            audios = batch["audio"] if isinstance(batch["audio"], list) else [batch["audio"]]
            waveforms = []
            for j, a in enumerate(audios):
                arr = a.get("array")
                sr = a.get("sampling_rate", 16000)
                if arr is not None and len(arr) > 0:
                    audio_seconds += len(arr) / float(sr)
                if degrade_fn is not None:
                    arr = degrade_fn(arr, index=i + j, seed_base=phone_seed,
                                     profiles=phone_profiles, use_ffmpeg=telephony_use_ffmpeg)
                waveforms.append(arr)

            feats = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            if use_amp:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                    pred_ids = model.generate(input_features=feats, max_new_tokens=225)
            else:
                with torch.inference_mode():
                    pred_ids = model.generate(input_features=feats, max_new_tokens=225)

            preds.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
        return preds, audio_seconds

    model.eval()
    start_eval_time = time.time()
    predictions, total_audio_seconds = _transcribe("clean")
    elapsed_eval_time = max(0.001, time.time() - start_eval_time)
    rtfx = round(total_audio_seconds / elapsed_eval_time, 2) if total_audio_seconds > 0 else 0.0

    # 2. Compute full metrics
    eval_results = metrics_calculator.evaluate_predictions(
        references=references,
        predictions=predictions,
        is_informal_flags=is_informal,
    )

    # 2b. Phone-simulated second pass on the same test split (deterministic)
    phone_block = None
    if phone_benchmark:
        try:
            from src.telephone import degrade_for_eval
            phone_predictions, _ = _transcribe("phone-simulated", degrade_fn=degrade_for_eval)
            phone_results = metrics_calculator.evaluate_predictions(
                references=references,
                predictions=phone_predictions,
                is_informal_flags=is_informal,
            )
            print(metrics_calculator.format_results_table("BuzzASR + LoRA (Final Full Test, phone)", phone_results))
            p_overall = phone_results.get("overall", {})
            phone_block = {
                "normalized_wer": round(p_overall.get("normalized_wer", 0.0), 2),
                "normalized_cer": round(p_overall.get("normalized_cer", 0.0), 2),
                "formal_wer": round(phone_results.get("formal", {}).get("normalized_wer", 0.0), 2),
                "informal_wer": round(phone_results.get("informal", {}).get("normalized_wer", 0.0), 2),
                "gap_vs_clean_pp": round(p_overall.get("normalized_wer", 0.0) - eval_results.get("overall", {}).get("normalized_wer", 0.0), 2),
                "seed": phone_seed,
                "profiles": phone_profiles,
                "details": phone_results,
            }
        except Exception as e:
            print(f"[Final Report] Phone pass failed ({e}); report will be clean-only.")

    # 3. Read epoch benchmark history & trainer logs
    epoch_benchmarks = []
    benchmarks_path = os.path.join(_resolve_results_dir(results_dir), "epoch_benchmarks.json")
    if os.path.exists(benchmarks_path):
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                epoch_benchmarks = json.load(f)
        except Exception:
            epoch_benchmarks = []

    # Read training loss & step history from trainer_state
    epoch_training_summary = []
    if trainer_state and getattr(trainer_state, "log_history", None):
        # Group log_history by epoch or summarize evaluation/loss points
        eval_logs = [log for log in trainer_state.log_history if "eval_wer" in log or "eval_loss" in log]
        train_logs = [log for log in trainer_state.log_history if "loss" in log]
        epoch_training_summary = {
            "best_metric": trainer_state.best_metric,
            "best_model_checkpoint": trainer_state.best_model_checkpoint,
            "global_step": trainer_state.global_step,
            "train_logs": train_logs,
            "eval_logs": eval_logs,
        }

    # 4. Compare vs Baselines
    baselines = load_baselines()
    norm_wer = eval_results.get("overall", {}).get("normalized_wer", 0.0)
    norm_cer = eval_results.get("overall", {}).get("normalized_cer", 0.0)
    diag_wer = eval_results.get("overall", {}).get("diagnostic_wer_zwnj_space", 0.0)
    formal_wer = eval_results.get("formal", {}).get("normalized_wer", 0.0)
    informal_wer = eval_results.get("informal", {}).get("normalized_wer", 0.0)

    whisper_wer = baselines.get("whisper_v3", {}).get("normalized_wer", 28.5)
    buzzasr_wer = baselines.get("buzzasr_base", {}).get("normalized_wer", 17.2)

    delta_vs_whisper = round(norm_wer - whisper_wer, 2)
    delta_vs_buzzasr = round(norm_wer - buzzasr_wer, 2)

    final_report = {
        "dataset_name": dataset_name,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_samples,
        "total_audio_duration_seconds": round(total_audio_seconds, 2),
        "total_inference_time_seconds": round(elapsed_eval_time, 2),
        "rtfx_speedup": rtfx,
        "final_full_test_metrics": {
            "normalized_wer": round(norm_wer, 2),
            "normalized_cer": round(norm_cer, 2),
            "diagnostic_wer": round(diag_wer, 2),
            "formal_wer": round(formal_wer, 2),
            "informal_wer": round(informal_wer, 2),
            "raw_wer": round(eval_results.get("overall", {}).get("raw_wer", 0.0), 2),
            "raw_cer": round(eval_results.get("overall", {}).get("raw_cer", 0.0), 2),
            "delta_vs_whisper_v3": delta_vs_whisper,
            "delta_vs_buzzasr_base": delta_vs_buzzasr,
            "details": eval_results,
            "phone": phone_block,
        },
        "baselines_comparison": {
            "whisper_large_v3": baselines.get("whisper_v3"),
            "buzzasr_base": baselines.get("buzzasr_base"),
        },
        "all_epoch_benchmarks": epoch_benchmarks,
        "trainer_state_summary": epoch_training_summary,
    }

    # Save JSON report next to final model
    json_report_path = os.path.join(output_dir, "final_evaluation_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\n[Final Report] Saved JSON report to: {json_report_path}")

    # Generate Markdown report
    md_lines = [
        "# Final Persian ASR Evaluation Report",
        f"- **Dataset**: `{dataset_name}` (Full Test Split)",
        f"- **Total Test Samples**: {total_samples}",
        f"- **Total Audio Processed**: {round(total_audio_seconds / 60.0, 2)} minutes",
        f"- **Inference Speed**: {rtfx}x Real-Time (RTFX)",
        f"- **Evaluation Timestamp**: {final_report['evaluated_at']}",
        "",
        "## 1. Post-Training Full Test Benchmark (Best Adapter)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Normalized WER** | **{norm_wer:.2f}%** |",
        f"| **Normalized CER** | {norm_cer:.2f}% |",
    ]
    if phone_block:
        md_lines.append(
            f"| **Phone Normalized WER** (MCI/Irancell + TCI sim, seed={phone_block['seed']}) | "
            f"**{phone_block['normalized_wer']:.2f}%** ({phone_block['gap_vs_clean_pp']:+.2f} pp vs clean) |"
        )
    md_lines += [
        f"| **Diagnostic WER (ZWNJ $\\to$ Space)** | {diag_wer:.2f}% |",
        f"| **Formal Subset WER** | {formal_wer:.2f}% ({eval_results.get('formal', {}).get('count', 0)} samples) |",
        f"| **Informal Subset WER** | {informal_wer:.2f}% ({eval_results.get('informal', {}).get('count', 0)} samples) |",
        f"| **$\\Delta$ vs. Whisper Large v3** | **{delta_vs_whisper:+.2f}%** |",
        f"| **$\\Delta$ vs. BuzzASR Base** | **{delta_vs_buzzasr:+.2f}%** |",
        "",
        "## 2. Benchmark History Across All Epochs",
        "",
    ]

    if epoch_benchmarks:
        md_lines.append("| Epoch | Samples | Norm WER (%) | Phone WER (%) | Norm CER (%) | Formal WER (%) | Informal WER (%) | $\\Delta$ vs Whisper | $\\Delta$ vs BuzzASR |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        for ep in epoch_benchmarks:
            phone_wer = ep.get("phone_normalized_wer")
            phone_str = f"{phone_wer:.2f}%" if phone_wer is not None else "--"
            md_lines.append(
                f"| {ep.get('epoch', 'N/A')} | {ep.get('total_samples', '-')} | "
                f"{ep.get('normalized_wer', 0.0):.2f}% | {phone_str} | {ep.get('normalized_cer', 0.0):.2f}% | "
                f"{ep.get('formal_wer', 0.0):.2f}% | {ep.get('informal_wer', 0.0):.2f}% | "
                f"{ep.get('delta_vs_whisper', 0.0):+.2f}% | {ep.get('delta_vs_buzzasr', 0.0):+.2f}% |"
            )
        md_lines.append("")
    else:
        md_lines.append("*No per-epoch benchmark telemetry found.*\n")

    md_report_path = os.path.join(output_dir, "final_evaluation_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Final Report] Saved Markdown report to: {md_report_path}")

    # Print summary table to console
    print("\n" + metrics_calculator.format_results_table("BuzzASR + LoRA (Final Full Test)", eval_results))
    print(f"Improvement vs. Whisper v3: {delta_vs_whisper:+.2f}%")
    print(f"Improvement vs. BuzzASR Base: {delta_vs_buzzasr:+.2f}%")
    if phone_block:
        print(f"Clean WER: {norm_wer:.2f}% | Phone WER: {phone_block['normalized_wer']:.2f}% | Gap: {phone_block['gap_vs_clean_pp']:+.2f}pp\n")


if __name__ == "__main__":
    main()
