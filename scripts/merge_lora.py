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
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory to mirror to Hugging Face after the merge",
    )
    parser.add_argument(
        "--hf_run_repo",
        type=str,
        default=None,
        help="Hugging Face repo id for the run mirror (default: auto-resolved or ASR_HF_RUN_REPO)",
    )
    parser.add_argument(
        "--no_hf_sync",
        "--no-hf-sync",
        action="store_true",
        help="Disable Hugging Face mirroring completely even if authenticated or ASR_HF_RUN_REPO is set.",
    )
    return parser.parse_args()


def _upload_merged_to_hf(args) -> None:
    """Best-effort mirror of the merged model; never fail the merge.

    A mirror is only attempted when both a run directory and a repo are
    configured. Upload failures are logged and swallowed, so the standalone
    merged model remains usable even when Hugging Face is unavailable.
    """
    run_dir = str(getattr(args, "run_dir", None) or "").strip()
    if not run_dir:
        return

    mirror = None
    try:
        from src import hf_sync

        config = hf_sync.resolve_config(args)
        if config is None:
            return

        run_path = Path(run_dir).expanduser().resolve()
        mirror = hf_sync.HFRunMirror(
            config,
            run_id=run_path.name,
            run_dir=run_path,
            output_dir=run_path / "output",
            results_dir=run_path,
            log=print,
        )
        mirror.start()
        mirror.enqueue_merged(Path(args.output_dir))
        mirror.enqueue_ledger("merge")
    except Exception as exc:
        print(f"[hf-sync] merged model upload skipped: {exc}")
    finally:
        if mirror is not None:
            try:
                mirror.close()
            except Exception as exc:
                print(f"[hf-sync] merged model upload failed: {exc}")


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
    _upload_merged_to_hf(args)


if __name__ == "__main__":
    main()
