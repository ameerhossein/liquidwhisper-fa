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
from typing import Dict, Any
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
from src.trainer_callbacks import WebTelemetryCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train BuzzASR with LoRA on Persian Speech")
    # Model & Data
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. configs/l40s_train_config.yaml). CLI flags override file.")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--dataset_name", type=str, default="shekar-ai/Neyshekar", help="Dataset identifier")
    parser.add_argument("--output_dir", type=str, default="outputs/buzzasr-neyshekar-lora", help="Output directory for checkpoints")
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
    parser.add_argument("--num_proc", type=int, default=4, help="Number of preprocessing workers")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device eval + benchmark batch size")
    parser.add_argument("--max_benchmark_samples", type=int, default=300, help="Max test samples for end-of-epoch benchmark (0/None = full set). Default 300 guards L40S OOM.")
    parser.add_argument("--enable_telemetry", "--enable-telemetry", action=argparse.BooleanOptionalAction, default=True, help="Enable real-time telemetry for Web UI (--no-enable-telemetry to disable)")
    parser.add_argument("--run_epoch_benchmark", "--run-epoch-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Run test benchmark after each epoch (--no-run-epoch-benchmark to disable)")
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
    for section in ("model", "lora", "training", "benchmark"):
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
    }
    for yk, ak in key_map.items():
        if yk in flat:
            cli_name = "--" + ak
            if cli_name not in normalized_cli and ("--" + ak.replace("_", "-")) not in normalized_cli:
                try:
                    setattr(args, ak, type(getattr(args, ak))(flat[yk]) if getattr(args, ak) is not None else flat[yk])
                except Exception:
                    setattr(args, ak, flat[yk])
    # Target modules are code-fixed; warn if config disagrees.
    if isinstance(cfg.get("lora", {}).get("target_modules"), list):
        print(f"Note: lora.target_modules from {path} ignored (code-fixed).")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=======================================================")
    print(" Persian ASR Fine-Tuning: BuzzASR + Neyshekar LoRA")
    print(f" Base Model:      {args.base_model}")
    print(f" Dataset:         {args.dataset_name}")
    print(f" Output Dir:      {args.output_dir}")
    print(f" LoRA Rank:       r={args.lora_r}, alpha={args.lora_alpha}")
    print(f" Batch Size:      {args.batch_size} (accum: {args.grad_accum})")
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

    # 4. Prepare Data
    data_pipeline = PersianASRDataPipeline(
        processor=processor,
        dataset_name=args.dataset_name,
        normalizer=normalizer,
        num_proc=args.num_proc
    )

    raw_train = data_pipeline.load_raw_dataset(split="train", max_samples=args.max_train_samples)
    raw_eval = data_pipeline.load_raw_dataset(split="validation", max_samples=args.max_eval_samples)
    raw_test = None
    if args.run_epoch_benchmark:
        print("Loading full test benchmark split for end-of-epoch evaluation...")
        raw_test = data_pipeline.load_raw_dataset(split="test")

    train_dataset = data_pipeline.prepare_dataset_split(raw_train)
    eval_dataset = data_pipeline.prepare_dataset_split(raw_eval)

    # 5. Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. Evaluation Metric Function & Callbacks
    wer_metric = evaluate.load("wer")
    metrics_calculator = ASRMetricsCalculator(normalizer=normalizer)

    callbacks = []
    if args.enable_telemetry:
        telemetry_cb = WebTelemetryCallback(
            output_dir=args.output_dir,
            results_dir="results",
            test_dataset=raw_test,
            processor=processor,
            metrics_calculator=metrics_calculator,
            normalizer=normalizer,
            eval_batch_size=args.eval_batch_size,
            max_benchmark_samples=args.max_benchmark_samples,
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

    # 9. Execute Training
    print("\nStarting training loop...")
    trainer.train()

    # 10. Save Best Adapter & Processor
    best_save_path = os.path.join(args.output_dir, "best_adapter")
    print(f"\nTraining completed! Saving best LoRA adapter to: {best_save_path}")
    trainer.model.save_pretrained(best_save_path)
    processor.save_pretrained(best_save_path)
    print("Done! You can now benchmark or transcribe using this adapter.")


if __name__ == "__main__":
    main()
