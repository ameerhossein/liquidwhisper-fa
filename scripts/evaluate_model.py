"""
Evaluate a Trained Persian ASR Model or LoRA Adapter on Benchmark Split.
Produces:
- Corpus WER and CER (raw, normalized, diagnostic)
- Formal vs. Informal breakdown
- Qualitative error inspector (identifies the most severe error cases)
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
    parser = argparse.ArgumentParser(description="Evaluate Trained Persian ASR Model or LoRA Adapter")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to standalone/merged model directory")
    parser.add_argument("--dataset_name", type=str, default="shekar-ai/Neyshekar")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phone_eval", "--phone-eval", action=argparse.BooleanOptionalAction, default=True, help="Also evaluate on phone-simulated copy (MCI/Irancell/TCI) for dual clean+phone report")
    parser.add_argument("--phone_seed", type=int, default=1234, help="Deterministic seed for phone-simulated eval")
    parser.add_argument("--telephony_profiles", type=str, default="mobile,pstn", help="Comma-separated phone profiles for eval")
    parser.add_argument("--telephony_use_ffmpeg", "--telephony-use-ffmpeg", action=argparse.BooleanOptionalAction, default=True, help="Use true ffmpeg codecs in phone eval when available")
    return parser.parse_args()


def _parse_profiles(value: str):
    return [p.strip() for p in str(value or "").split(",") if p.strip()] or ["mobile", "pstn"]


def transcribe_split(model, processor, test_ds, batch_size: int, device: str,
                     use_amp, dtype, degrade_phone: bool = False,
                     phone_seed: int = 1234, profiles=None,
                     use_ffmpeg: bool = True) -> List[str]:
    """Transcribe a split; degrade_phone=True applies deterministic phone sim."""
    from tqdm import tqdm as _tqdm
    degrade_fn = None
    if degrade_phone:
        try:
            from src.telephone import degrade_for_eval as _degrade
            degrade_fn = _degrade
        except ImportError:
            print("Warning: src.telephone missing; phone eval falls back to clean.")
            degrade_fn = None
    predictions: List[str] = []
    total = len(test_ds)
    for i in _tqdm(range(0, total, batch_size), desc="Evaluating (phone)" if degrade_phone else "Evaluating (clean)"):
        batch_slice = test_ds[i: i + batch_size]
        batch_audios = batch_slice["audio"] if isinstance(batch_slice["audio"], list) else [batch_slice["audio"]]
        waveforms = []
        for j, a in enumerate(batch_audios):
            wav = a["array"]
            if degrade_phone and degrade_fn is not None:
                wav = degrade_fn(wav, index=i + j, seed_base=phone_seed,
                                 profiles=profiles or ["mobile", "pstn"],
                                 use_ffmpeg=use_ffmpeg)
            waveforms.append(wav)

        feats = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        if use_amp:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                predicted_ids = model.generate(input_features=feats, max_new_tokens=225)
        else:
            with torch.inference_mode():
                predicted_ids = model.generate(input_features=feats, max_new_tokens=225)

        batch_preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)
    return predictions


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    normalizer = PersianTextNormalizer()
    metrics_calc = ASRMetricsCalculator(normalizer=normalizer)

    print("=======================================================")
    print(" Persian ASR Model Evaluation & Diagnostics")
    print(f" Dataset:      {args.dataset_name} ({args.split})")
    print(f" Base Model:   {args.base_model}")
    print(f" LoRA Adapter: {args.adapter_path}")
    print(f" Device:       {args.device}")
    print("=======================================================\n")

    # 1. Load Processor
    load_source = args.adapter_path or args.model_path or args.base_model
    print(f"Loading processor from '{load_source}'...")
    processor = WhisperProcessor.from_pretrained(load_source)

    # 2. Load Model
    def _resolve_dtype(device: str):
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        return torch.float32

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    dtype = _resolve_dtype(args.device)
    if args.model_path:
        print(f"Loading merged/standalone model from '{args.model_path}'...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device)
    else:
        print(f"Loading base model from '{args.base_model}'...")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
        if args.adapter_path:
            print(f"Loading LoRA adapter from '{args.adapter_path}'...")
            model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)

    model.eval()

    # 3. Load Evaluation Data
    pipeline = PersianASRDataPipeline(processor=processor, dataset_name=args.dataset_name)
    test_ds = pipeline.load_raw_dataset(split=args.split, max_samples=args.max_samples)

    total_samples = len(test_ds)
    references = [sample.get("text", "") or "" for sample in test_ds]

    is_informal = []
    for s in test_ds:
        if "informal" in s:
            is_informal.append(bool(s["informal"]))
        elif "is_informal" in s:
            is_informal.append(bool(s["is_informal"]))
        else:
            is_informal.append(normalizer.is_colloquial(s.get("text", "")))

    profiles = _parse_profiles(args.telephony_profiles)

    # 4. Run Batch Inference (clean pass)
    print(f"Transcribing {total_samples} samples (clean)...")
    predictions = transcribe_split(
        model, processor, test_ds, args.batch_size, args.device,
        use_amp, dtype, degrade_phone=False,
    )

    # 5. Compute Metrics (clean)
    results = metrics_calc.evaluate_predictions(
        references=references,
        predictions=predictions,
        is_informal_flags=is_informal
    )

    eval_label = args.adapter_path or args.model_path or args.base_model
    print("\n" + metrics_calc.format_results_table(f"{eval_label} [clean]", results) + "\n")

    # 5b. Optional phone-simulated second pass (deterministic, same subset)
    phone_results = None
    phone_predictions: List[str] = []
    if args.phone_eval:
        print(f"Transcribing {total_samples} samples (phone-simulated, seed={args.phone_seed})...")
        # Honor --telephony_use_ffmpeg if telephone module supports the flag.
        phone_predictions = transcribe_split(
            model, processor, test_ds, args.batch_size, args.device,
            use_amp, dtype, degrade_phone=True,
            phone_seed=args.phone_seed, profiles=profiles,
            use_ffmpeg=bool(args.telephony_use_ffmpeg),
        )
        phone_results = metrics_calc.evaluate_predictions(
            references=references,
            predictions=phone_predictions,
            is_informal_flags=is_informal,
        )
        print("\n" + metrics_calc.format_results_table(f"{eval_label} [phone]", phone_results) + "\n")
        try:
            clean_wer = results.get("overall", {}).get("normalized_wer", 0.0)
            phone_wer = phone_results.get("overall", {}).get("normalized_wer", 0.0)
            print(f"Clean WER: {clean_wer:.2f}% | Phone WER: {phone_wer:.2f}% | Gap: {phone_wer - clean_wer:+.2f}pp\n")
        except Exception:
            pass

    # 6. Save Detailed Predictions & Diagnostics
    output_records = []
    for idx, (ref, pred, inf) in enumerate(zip(references, predictions, is_informal)):
        norm_ref = normalizer.normalize(ref, remove_punctuation=True, zwnj_mode="keep")
        norm_pred = normalizer.normalize(pred, remove_punctuation=True, zwnj_mode="keep")
        sample_wer = metrics_calc.compute_wer([norm_ref], [norm_pred])
        output_records.append({
            "id": idx,
            "ref_raw": ref,
            "pred_raw": pred,
            "ref_norm": norm_ref,
            "pred_norm": norm_pred,
            "is_informal": inf,
            "sample_wer": sample_wer
        })

    eval_json = os.path.join(args.output_dir, "evaluation_details.json")
    payload: Dict[str, Any] = {"metrics": results, "samples": output_records}
    if phone_results is not None:
        payload["phone_metrics"] = phone_results
        payload["phone_seed"] = args.phone_seed
        payload["phone_profiles"] = profiles
        payload["phone_predictions"] = phone_predictions
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Print Top 5 Hardest Errors for Qualitative Diagnosis
    sorted_by_err = sorted(output_records, key=lambda x: x["sample_wer"], reverse=True)
    print("--- Top 5 Error Cases for Qualitative Inspection ---")
    for item in sorted_by_err[:5]:
        print(f"Sample #{item['id']} [WER: {item['sample_wer']:.1f}% | Informal: {item['is_informal']}]:")
        print(f"   REF:  {item['ref_norm']}")
        print(f"   PRED: {item['pred_norm']}")
        print("-----------------------------------------------------")

    print(f"\nDetailed report written to: {eval_json}")


if __name__ == "__main__":
    main()
