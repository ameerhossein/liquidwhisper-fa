"""
Inference CLI: Transcribe Persian audio files using BuzzASR / Fine-Tuned Whisper.
Usage:
  python scripts/transcribe.py --audio sample.mp3
  python scripts/transcribe.py --audio sample.wav --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter
  python scripts/transcribe.py --audio_dir ./recordings/ --output_file transcriptions.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalizer import PersianTextNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe Persian audio files")
    parser.add_argument("--audio", type=str, default=None, help="Path to a single audio file")
    parser.add_argument("--audio_dir", type=str, default=None, help="Path to directory of audio files")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to merged standalone model directory")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save JSON transcriptions")
    parser.add_argument("--normalize", action="store_true", help="Apply Hazm/Persian text normalization on output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_audio(file_path: str, target_sr: int = 16000):
    """Loads and resamples audio to 16 kHz mono."""
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return waveform


def main():
    args = parse_args()

    if not args.audio and not args.audio_dir:
        print("Error: You must provide either --audio <path> or --audio_dir <path>")
        sys.exit(1)

    normalizer = PersianTextNormalizer()

    # Collect audio files
    audio_files: List[Path] = []
    if args.audio:
        p = Path(args.audio)
        if not p.is_file():
            print(f"Error: File not found: {args.audio}")
            sys.exit(1)
        audio_files.append(p)
    elif args.audio_dir:
        p = Path(args.audio_dir)
        if not p.is_dir():
            print(f"Error: Directory not found: {args.audio_dir}")
            sys.exit(1)
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac"):
            audio_files.extend(p.glob(ext))
        audio_files = sorted(audio_files)

    if not audio_files:
        print("No audio files found to process.")
        sys.exit(0)

    # Load processor & model
    load_source = args.adapter_path or args.model_path or args.base_model
    print(f"Loading processor from: {load_source}...")
    processor = WhisperProcessor.from_pretrained(load_source)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if args.device == "cuda" else torch.float32
    if args.model_path:
        print(f"Loading merged model from: {args.model_path}...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device)
    else:
        print(f"Loading base model: {args.base_model}...")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=dtype).to(args.device)
        if args.adapter_path:
            print(f"Applying LoRA adapter from: {args.adapter_path}...")
            model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)

    model.eval()

    print(f"\nProcessing {len(audio_files)} audio file(s)...")
    results = []

    for file_path in audio_files:
        print(f"\nTranscribing: {file_path.name}")
        waveform = load_audio(str(file_path), target_sr=16000)

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(args.device)

        with torch.inference_mode(), torch.autocast(device_type=args.device.split(":")[0], dtype=dtype):
            predicted_ids = model.generate(inputs, max_new_tokens=225)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        if args.normalize:
            transcription = normalizer.normalize(transcription, remove_punctuation=False, zwnj_mode="keep")

        print(f"Output: {transcription}")
        results.append({
            "file": str(file_path),
            "filename": file_path.name,
            "transcription": transcription
        })

    if args.output_file:
        out_p = Path(args.output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nAll transcriptions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
