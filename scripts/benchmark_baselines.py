"""
Benchmark Baseline Models on Persian ASR (Neyshekar Test Split).
Evaluates models such as:
- openai/whisper-large-v3
- BuzzASR/persian
- (Optional) hosseinzr/neyshekar-whisper-large-v3-lora

Generates side-by-side comparison tables covering:
- Raw WER / CER
- Normalized WER / CER (ZWNJ preserved)
- Diagnostic WER (ZWNJ -> space)
- Formal vs. Informal utterance breakdown
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalizer import PersianTextNormalizer
from src.metrics import ASRMetricsCalculator
from src.data import PersianASRDataPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Persian ASR Baseline Models")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. configs/eval_config.yaml). CLI flags override file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/whisper-large-v3", "BuzzASR/persian"],
        help="List of model IDs or paths to benchmark"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="shekar-ai/Neyshekar",
        help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (e.g., test or validation)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples (useful for quick smoke-test)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baselines",
        help="Directory to save predictions and metric reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device"
    )
    args = parser.parse_args()
    if args.config:
        _apply_yaml_config(args, args.config)
    return args


def _apply_yaml_config(args: argparse.Namespace, path: str):
    """Overlay eval YAML (configs/eval_config.yaml) onto args; CLI wins."""
    import sys as _sys
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed; ignoring --config.")
        return
    cli_flags = {a.split("=")[0] for a in _sys.argv[1:] if a.startswith("--")}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load config {path}: {e}")
        return
    bench = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    mapping = {
        "dataset_name": "dataset_name", "split": "split",
        "batch_size": "batch_size", "output_dir": "output_dir",
        "models": "models",
    }
    for yk, ak in mapping.items():
        if yk in bench and ("--" + ak) not in cli_flags:
            setattr(args, ak, bench[yk])


def _resolve_infer_dtype(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _should_autocast(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def transcribe_batch(
    model,
    processor,
    audios: List[Dict[str, Any]],
    model_id: str,
    device: str
) -> List[str]:
    """Transcribes a batch of 16kHz audio waveforms."""
    sampling_rate = 16000
    waveforms = [a["array"] for a in audios]

    feats = processor(
        waveforms,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features.to(device)

    # Autocast to bfloat16/float16 only on CUDA; plain inference on CPU.
    if _should_autocast(device):
        dtype = _resolve_infer_dtype(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            if "buzzasr" in model_id.lower():
                predicted_ids = model.generate(input_features=feats, max_new_tokens=225)
            else:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="persian", task="transcribe")
                predicted_ids = model.generate(
                    input_features=feats,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=225,
                )
    else:
        with torch.inference_mode():
            # Monolingual BuzzASR already bakes Persian prompt into its generation config
            if "buzzasr" in model_id.lower():
                predicted_ids = model.generate(input_features=feats, max_new_tokens=225)
            else:
                # For standard multilingual Whisper, force Persian transcription
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="persian", task="transcribe")
                predicted_ids = model.generate(
                    input_features=feats,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=225,
                )

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


def run_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    normalizer = PersianTextNormalizer()
    metrics_calc = ASRMetricsCalculator(normalizer=normalizer)

    print(f"\n=======================================================")
    print(f" Persian ASR Benchmark Harness")
    print(f" Target Dataset: {args.dataset_name} ({args.split} split)")
    print(f" Models to evaluate: {args.models}")
    print(f" Running on device: {args.device}")
    print(f"=======================================================\n")

    # Load dataset audio and references once
    # Dummy processor just to instantiate the pipeline for dataset loading
    dummy_proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    pipeline = PersianASRDataPipeline(processor=dummy_proc, dataset_name=args.dataset_name)
    test_ds = pipeline.load_raw_dataset(split=args.split, max_samples=args.max_samples)

    total_samples = len(test_ds)
    references = [sample.get("text", "") or "" for sample in test_ds]
    
    # Extract informal flags if available
    is_informal = []
    for s in test_ds:
        if "informal" in s:
            is_informal.append(bool(s["informal"]))
        elif "is_informal" in s:
            is_informal.append(bool(s["is_informal"]))
        else:
            is_informal.append(normalizer.is_colloquial(s.get("text", "")))

    all_model_results = {}

    for model_id in args.models:
        print(f"\n---> Loading model: {model_id}...")
        
        # Load processor
        processor = WhisperProcessor.from_pretrained(model_id)
        
        # Load model weights
        dtype = _resolve_infer_dtype(args.device)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(args.device)
        model.eval()

        predictions = []
        batch_size = args.batch_size

        print(f"---> Generating transcriptions for {total_samples} samples...")
        for i in tqdm(range(0, total_samples, batch_size), desc=f"Evaluating {model_id}"):
            batch_slice = test_ds[i : i + batch_size]
            # When slicing HF Dataset, 'audio' is a list of audio dicts
            batch_audios = batch_slice["audio"] if isinstance(batch_slice["audio"], list) else [batch_slice["audio"]]
            
            batch_preds = transcribe_batch(
                model=model,
                processor=processor,
                audios=batch_audios,
                model_id=model_id,
                device=args.device
            )
            predictions.extend(batch_preds)

        # Calculate metrics
        eval_metrics = metrics_calc.evaluate_predictions(
            references=references,
            predictions=predictions,
            is_informal_flags=is_informal
        )
        all_model_results[model_id] = eval_metrics

        # Print single-model formatted summary
        print("\n" + metrics_calc.format_results_table(model_id, eval_metrics) + "\n")

        # Save model predictions
        safe_model_name = model_id.replace("/", "_")
        pred_file = os.path.join(args.output_dir, f"preds_{safe_model_name}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "ref": r,
                        "pred": p,
                        "is_informal": inf
                    }
                    for r, p, inf in zip(references, predictions, is_informal)
                ],
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"Saved predictions to: {pred_file}")

        # Free VRAM before next model
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary report
    report_file = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    print(f"\nAll benchmark results saved to: {report_file}")


if __name__ == "__main__":
    run_benchmark()
