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
    parser.add_argument("--enable_telemetry", action="store_true", default=True, help="Enable real-time telemetry for Web UI")
    parser.add_argument("--run_epoch_benchmark", action="store_true", default=True, help="Run test benchmark after each epoch")
    return parser.parse_args()


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

    # 2. Load Base Model in BF16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
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

    # 8. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

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
