"""
Merge trained LoRA adapter back into base Whisper weights to produce a standalone model.
Usage:
  python scripts/merge_lora.py \
    --base_model BuzzASR/persian \
    --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
    --output_dir models/buzzasr-neyshekar-standalone
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Load .env (repo root / CWD) before any os.environ reads. Stdlib-only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with Base Whisper Model")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged standalone model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for merging (default: cuda if available, else cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=======================================================")
    print(" Merging LoRA Adapter into Base Whisper Weights")
    print(f" Base Model:   {args.base_model}")
    print(f" Adapter Path: {args.adapter_path}")
    print(f" Output Dir:   {args.output_dir}")
    print("=======================================================\n")

    print(f"Loading processor from '{args.adapter_path}'...")
    processor = WhisperProcessor.from_pretrained(args.adapter_path)

    print(f"Loading base model '{args.base_model}'...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(args.device)

    print(f"Loading PEFT adapter from '{args.adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path).to(args.device)

    print("Merging weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\nMerge complete! The standalone model is ready for zero-overhead inference.")


if __name__ == "__main__":
    main()
