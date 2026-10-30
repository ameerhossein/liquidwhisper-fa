#!/usr/bin/env python3
"""Paired legacy-synth vs current-synth telephone WER smoke test.

IMPORTANT: the six ``FakePhoneCalls`` files are **not** real carrier
recordings.  The repository's own analysis labels them "Synth mobile" /
"Synth PSTN" and describes them as single utterances passed through an older
telephone simulator.  Their filenames are labels, not carrier evidence.

This script runs one ASR model on the clean source, the legacy synth output,
and several seeded current-simulator outputs for the same content.  It
measures whether the current simulator is harsher or easier than the legacy
baseline; it does **not** validate either against a real network.

A true real-vs-simulated WER is not possible from this corpus: the 107 real
call-center recordings have no transcripts, and the 77 gated real recordings
have no clean counterparts.  If human transcripts for the real gated calls
become available, pass ``--real-gated-manifest`` to add a ``real_gated`` row.

Usage:
  python RecordingTesting/analysis/smoke_wer.py --list-only
  python RecordingTesting/analysis/smoke_wer.py --model BuzzASR/persian
  python RecordingTesting/analysis/smoke_wer.py --dtype bfloat16 --seeds 1 2 3

The first run downloads the model (about 3.1 GB for BuzzASR/persian).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import ASRMetricsCalculator  # noqa: E402
from src.normalizer import PersianTextNormalizer  # noqa: E402
from src.telephone import simulate_telephone  # noqa: E402

FAKE_RE = re.compile(
    r"^spk_(\d+)_clip(\d+)_id(\d+)_(mobile|pstn)-", re.IGNORECASE
)
CONDITION_ORDER = [
    "clean",
    "legacy_mobile",
    "sim_mobile",
    "legacy_pstn",
    "sim_pstn",
    "sim_gated",
    "real_gated",
]


def parse_legacy_name(path: Path) -> Optional[Dict[str, object]]:
    match = FAKE_RE.match(path.name)
    if not match:
        return None
    speaker, clip, dataset_id, profile = match.groups()
    return {
        "speaker": speaker,
        "clip": clip,
        "id": int(dataset_id),
        "profile": profile.lower(),
        "base": f"spk_{speaker}_clip{clip}_id{dataset_id}",
        "transcript_key": f"{speaker}_{dataset_id}",
    }


def read_audio(path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_sr is not None and sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly

        factor = gcd(int(sr), int(target_sr))
        audio = resample_poly(
            audio, target_sr // factor, sr // factor
        ).astype(np.float32)
        sr = int(target_sr)
    return audio, int(sr)


def active_rms_db(audio: np.ndarray, sr: int, frame_s: float = 0.02) -> float:
    """Median active-frame RMS (dBFS), same relative VAD as the report."""
    frame = max(1, int(round(frame_s * sr)))
    n = len(audio) // frame
    if n == 0:
        return 10.0 * np.log10(float(np.mean(audio ** 2)) + 1e-20)
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    low, high = np.percentile(energy, [10.0, 95.0])
    active = energy > max(low + 7.0, high - 30.0)
    if not np.any(active):
        return float(np.median(energy))
    return float(np.median(energy[active]))


def level_match(audio: np.ndarray, sr: int, target_db: float,
                max_gain_db: float = 15.0) -> Tuple[np.ndarray, float]:
    """Match active speech to a common level, avoiding clipping."""
    current = active_rms_db(audio, sr)
    gain_db = float(np.clip(target_db - current, -max_gain_db, max_gain_db))
    out = (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        scale = 0.98 / peak
        out = (out * scale).astype(np.float32)
        gain_db += 20.0 * np.log10(scale)
    return out, float(gain_db)


def collect_pairs(legacy_dir: Path, clips_dir: Path,
                  transcripts: Dict[str, str]) -> List[Dict[str, object]]:
    pairs = []
    for path in sorted(legacy_dir.glob("*.wav")):
        info = parse_legacy_name(path)
        if info is None:
            continue
        clean = clips_dir / f"spk_{info['speaker']}" / f"clip{info['clip']}_id{info['id']}.wav"
        text = transcripts.get(str(info["transcript_key"]))
        pairs.append({
            **info,
            "legacy_path": path,
            "clean_path": clean,
            "text": text,
            "ok": clean.is_file() and bool(text),
        })
    return pairs


def load_transcripts(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def build_rows(pairs: Iterable[Dict[str, object]], clips_dir: Path,
               seeds: Sequence[int], profiles: Sequence[str],
               use_ffmpeg: bool, real_gated_manifest: Optional[Path],
               level_match_enabled: bool = True,
               target_active_dbfs: float = -26.0):
    rows: List[Dict[str, object]] = []
    paired_bases = set()
    seen_clean = set()
    seen_sim = set()

    def add_row(condition: str, base: str, seed: int, file: str,
                audio: np.ndarray, sr: int, ref: str) -> None:
        input_db = active_rms_db(audio, sr)
        gain_db = 0.0
        if level_match_enabled:
            audio, gain_db = level_match(audio, sr, target_active_dbfs)
        rows.append({
            "condition": condition, "base": base, "seed": seed,
            "file": file, "audio": audio, "sr": sr, "ref": ref,
            "input_active_db": input_db, "gain_db": gain_db,
        })

    for pair in pairs:
        if not pair["ok"]:
            print(f"SKIP {pair['legacy_path'].name}: missing clean source or transcript")
            continue
        base = str(pair["base"])
        paired_bases.add(base)
        ref = str(pair["text"])
        clean, clean_sr = read_audio(Path(pair["clean_path"]), target_sr=16000)
        if base not in seen_clean:
            seen_clean.add(base)
            add_row("clean", base, 0, str(pair["clean_path"]),
                    clean, clean_sr, ref)
        legacy, legacy_sr = read_audio(
            Path(pair["legacy_path"]), target_sr=16000
        )
        add_row(f"legacy_{pair['profile']}", base, 0,
                str(pair["legacy_path"]), legacy, legacy_sr, ref)
        for profile in profiles:
            for seed in seeds:
                key = (base, profile, int(seed))
                if key in seen_sim:
                    continue
                seen_sim.add(key)
                degraded = simulate_telephone(
                    clean, sr=16000, profile=profile, seed=int(seed),
                    use_ffmpeg=use_ffmpeg, output_sr=16000,
                )
                add_row(f"sim_{profile}", base, int(seed),
                        f"sim_{profile}_seed{seed}", degraded, 16000, ref)

    if real_gated_manifest is not None:
        manifest = json.loads(real_gated_manifest.read_text(encoding="utf-8"))
        items = manifest.items() if isinstance(manifest, dict) else [
            (item["path"], item["text"]) for item in manifest
        ]
        for path_str, text in items:
            path = Path(path_str)
            if not path.is_file():
                print(f"SKIP real gated {path}: file not found")
                continue
            audio, sr = read_audio(path, target_sr=16000)
            add_row("real_gated", path.stem, 0, str(path), audio, sr, str(text))
    return rows, paired_bases


def condition_sort_key(name: str) -> Tuple[int, str]:
    try:
        return CONDITION_ORDER.index(name), name
    except ValueError:
        return len(CONDITION_ORDER), name


def normalized(text: str, normalizer: PersianTextNormalizer,
               zwnj_mode: str = "keep") -> str:
    return normalizer.normalize(
        text, remove_punctuation=True, zwnj_mode=zwnj_mode
    )


def summarize(rows: Sequence[Dict[str, object]], metrics: ASRMetricsCalculator,
              normalizer: PersianTextNormalizer) -> Dict[str, object]:
    by_condition: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    conditions: Dict[str, object] = {}
    for condition in sorted(by_condition, key=condition_sort_key):
        items = by_condition[condition]
        refs = [str(item["ref"]) for item in items]
        preds = [str(item["pred"]) for item in items]
        overall = metrics.evaluate_predictions(refs, preds)["overall"]
        clip_wer_keep: Dict[str, List[float]] = defaultdict(list)
        clip_wer_diag: Dict[str, List[float]] = defaultdict(list)
        clip_cer_diag: Dict[str, List[float]] = defaultdict(list)
        for item in items:
            ref_keep = normalized(str(item["ref"]), normalizer, "keep")
            pred_keep = normalized(str(item["pred"]), normalizer, "keep")
            ref_diag = normalized(str(item["ref"]), normalizer, "to_space")
            pred_diag = normalized(str(item["pred"]), normalizer, "to_space")
            item["wer"] = float(metrics.compute_wer([ref_keep], [pred_keep]))
            item["wer_zwnj_space"] = float(
                metrics.compute_wer([ref_diag], [pred_diag])
            )
            item["cer"] = float(metrics.compute_cer([ref_keep], [pred_keep]))
            item["cer_zwnj_space"] = float(
                metrics.compute_cer([ref_diag], [pred_diag])
            )
            base = str(item["base"])
            clip_wer_keep[base].append(item["wer"])
            clip_wer_diag[base].append(item["wer_zwnj_space"])
            clip_cer_diag[base].append(item["cer_zwnj_space"])
        mean_clip_wer = float(np.mean([
            float(np.mean(values)) for values in clip_wer_keep.values()
        ])) if clip_wer_keep else 0.0
        mean_clip_wer_diag = float(np.mean([
            float(np.mean(values)) for values in clip_wer_diag.values()
        ])) if clip_wer_diag else 0.0
        mean_clip_cer_diag = float(np.mean([
            float(np.mean(values)) for values in clip_cer_diag.values()
        ])) if clip_cer_diag else 0.0
        conditions[condition] = {
            "n": len(items),
            "n_clips": len(clip_wer_keep),
            "corpus_wer": float(overall["normalized_wer"]),
            "corpus_wer_zwnj_space": float(
                overall["diagnostic_wer_zwnj_space"]
            ),
            "corpus_cer": float(overall["normalized_cer"]),
            "mean_clip_wer": mean_clip_wer,
            "mean_clip_wer_zwnj_space": mean_clip_wer_diag,
            "mean_clip_cer_zwnj_space": mean_clip_cer_diag,
        }

    deltas = {}
    for profile in ("mobile", "pstn"):
        legacy = conditions.get(f"legacy_{profile}")
        sim = conditions.get(f"sim_{profile}")
        if legacy and sim:
            deltas[profile] = {
                "sim_minus_legacy_mean_clip_wer_zwnj_space": (
                    sim["mean_clip_wer_zwnj_space"]
                    - legacy["mean_clip_wer_zwnj_space"]
                ),
                "sim_minus_legacy_corpus_wer_zwnj_space": (
                    sim["corpus_wer_zwnj_space"]
                    - legacy["corpus_wer_zwnj_space"]
                ),
            }
    return {"conditions": conditions, "deltas": deltas}


def write_outputs(rows: Sequence[Dict[str, object]], summary: Dict[str, object],
                  output_dir: Path, meta: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "condition", "base", "seed", "file", "duration_s",
            "input_active_db", "gain_db",
            "wer", "wer_zwnj_space", "cer", "cer_zwnj_space",
            "reference", "prediction",
        ])
        for row in rows:
            writer.writerow([
                row["condition"], row["base"], row["seed"], row["file"],
                round(len(row["audio"]) / float(row["sr"]), 3),
                round(float(row.get("input_active_db", -200.0)), 2),
                round(float(row.get("gain_db", 0.0)), 2),
                round(float(row.get("wer", 0.0)), 3),
                round(float(row.get("wer_zwnj_space", 0.0)), 3),
                round(float(row.get("cer", 0.0)), 3),
                round(float(row.get("cer_zwnj_space", 0.0)), 3),
                row["ref"], row.get("pred", ""),
            ])

    payload = {"meta": meta, **summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Real vs simulated telephone WER smoke test",
        "",
        f"Model: `{meta['model']}`  ",
        f"Dtype: `{meta['dtype']}`  ",
        f"Seeds: `{', '.join(str(s) for s in meta['seeds'])}`  ",
        f"Simulated profiles: `{', '.join(meta['profiles'])}`  ",
        (
            f"Level matching: `{meta['target_active_dbfs']:.0f} dBFS "
            "active speech`"
            if meta.get("level_match") else "Level matching: `disabled`"
        ),
        "",
        "`clip WER` and `ZWNJ->space` use the diagnostic normalization; "
        "the repo-standard normalized WER is shown separately.",
        "",
        "| condition | n | clips | corpus WER | corpus WER (ZWNJ->space) | clip WER (ZWNJ->space) | corpus CER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, data in summary["conditions"].items():
        lines.append(
            f"| {condition} | {data['n']} | {data['n_clips']} | "
            f"{data['corpus_wer']:.2f}% | "
            f"{data['corpus_wer_zwnj_space']:.2f}% | "
            f"{data['mean_clip_wer_zwnj_space']:.2f}% | "
            f"{data['corpus_cer']:.2f}% |"
        )
    if summary["deltas"]:
        lines += ["", "Current synth minus legacy synth "
                  "(positive = current synth harder):"]
        for profile, data in summary["deltas"].items():
            lines.append(
                f"- **{profile}**: mean clip WER "
                f"{data['sim_minus_legacy_mean_clip_wer_zwnj_space']:+.2f} pp; "
                f"corpus WER "
                f"{data['sim_minus_legacy_corpus_wer_zwnj_space']:+.2f} pp"
            )
    lines += [
        "",
        "The six `FakePhoneCalls` files are **legacy synth outputs**, not real",
        "carrier recordings. This table compares the current simulator to that",
        "older synth baseline; it does **not** validate either against a real",
        "network.",
        "",
        "The source clips come from the Neyshekar **train** split (the only split",
        "with matching clean sources and legacy outputs), so absolute WER is not",
        "a held-out benchmark. Use the paired current-vs-legacy deltas.",
        "",
        "A true real-vs-sim WER needs real recordings with transcripts: the 107",
        "real call-center files have none, and the 77 real gated calls have no",
        "clean counterparts. Real gated transcripts can be supplied with",
        "`--real-gated-manifest`.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def collect_transcripts(args) -> None:
    """Recover gold text for the local Neyshekar source clips via streaming."""
    from datasets import load_dataset

    wanted: Dict[str, set] = defaultdict(set)
    for path in sorted(args.clips_dir.glob("spk_*/*.wav")):
        match = re.match(r"spk_(\d+)/clip\d+_id(\d+)\.wav$", str(
            path.relative_to(args.clips_dir)
        ))
        if match:
            wanted[match.group(1)].add(int(match.group(2)))
    if not wanted:
        raise SystemExit(f"No spk_*/*.wav clips found under {args.clips_dir}")

    ds = load_dataset(
        args.dataset, split="train", streaming=True, revision=args.revision
    )
    iterator = iter(ds)
    found: Dict[Tuple[str, int], str] = {}
    target = sum(len(ids) for ids in wanted.values())
    for scanned, row in enumerate(iterator, 1):
        speaker = str(row["speaker_id"])
        dataset_id = int(row["id"])
        if speaker in wanted and dataset_id in wanted[speaker]:
            found[(speaker, dataset_id)] = str(row["text"])
        if len(found) >= target:
            print(f"Recovered {len(found)}/{target} transcripts after {scanned} rows.")
            break
        if scanned % 500 == 0:
            print(f"scanned {scanned}; recovered {len(found)}/{target}", flush=True)
    missing = [
        f"{speaker}_{dataset_id}"
        for speaker, ids in wanted.items()
        for dataset_id in ids
        if (speaker, dataset_id) not in found
    ]
    payload = {
        f"{speaker}_{dataset_id}": text
        for (speaker, dataset_id), text in sorted(found.items())
    }
    args.transcripts.parent.mkdir(parents=True, exist_ok=True)
    args.transcripts.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {args.transcripts} ({len(payload)} transcripts)")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    # datasets' streaming iterator can abort during interpreter finalization;
    # the output is already flushed, so exit before that cleanup runs.
    sys.stdout.flush()
    os._exit(0 if not missing else 1)


def detect_dtype(requested: str, device: str):
    import torch

    if requested != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[requested]
    if device == "cuda":
        return torch.float16
    flags = ""
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    if "avx512_bf16" in flags or "amx_bf16" in flags:
        return torch.bfloat16
    return torch.float32


def transcribe(model, processor, audio: np.ndarray, device: str, dtype,
               max_new_tokens: int) -> str:
    import torch

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=dtype)
    with torch.inference_mode():
        ids = model.generate(features, max_new_tokens=max_new_tokens)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-dir", type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "FakePhoneCalls",
        help="Legacy synth outputs to compare against current synth.",
    )
    parser.add_argument(
        "--clips-dir", type=Path,
        default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"),
    )
    parser.add_argument(
        "--transcripts", type=Path,
        default=HERE / "neyshekar_transcripts.json",
    )
    parser.add_argument("--real-gated-manifest", type=Path, default=None)
    parser.add_argument("--model", default="BuzzASR/persian")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--profiles", nargs="+", default=["gated", "mobile", "pstn"],
        choices=["gated", "mobile", "pstn"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--use-ffmpeg", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "telephone_smoke_wer",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--level-match", action=argparse.BooleanOptionalAction, default=True,
        help="Match all ASR inputs to --target-active-dbfs (default: on).",
    )
    parser.add_argument("--target-active-dbfs", type=float, default=-26.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--collect-transcripts", action="store_true")
    parser.add_argument("--dataset", default="shekar-ai/neyshekar")
    parser.add_argument(
        "--revision", default="30696ada746a66bb3047c984744b5ecfd60fe4e9"
    )
    args = parser.parse_args()

    if args.collect_transcripts:
        collect_transcripts(args)
        return

    transcripts = load_transcripts(args.transcripts)
    pairs = collect_pairs(args.legacy_dir, args.clips_dir, transcripts)
    if not pairs:
        raise SystemExit(f"No matching legacy/clean pairs under {args.legacy_dir}")
    for pair in pairs:
        status = "ok" if pair["ok"] else "MISSING"
        print(
            f"{pair['base']:<24} {pair['profile']:<6} {status} "
            f"legacy={pair['legacy_path'].name}"
        )
    if args.list_only:
        return

    rows, _ = build_rows(
        pairs, args.clips_dir, args.seeds, args.profiles,
        args.use_ffmpeg, args.real_gated_manifest,
        level_match_enabled=args.level_match,
        target_active_dbfs=args.target_active_dbfs,
    )
    if not rows:
        raise SystemExit("No rows to evaluate.")

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = detect_dtype(args.dtype, args.device)
    dtype_name = str(dtype).replace("torch.", "")
    print(f"Loading {args.model} on {args.device} as {dtype_name} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()

    started = time.time()
    for index, row in enumerate(rows, 1):
        row["pred"] = transcribe(
            model, processor, row["audio"], args.device, dtype,
            args.max_new_tokens,
        )
        elapsed = time.time() - started
        eta = elapsed / index * (len(rows) - index)
        print(
            f"[{index:>3}/{len(rows)}] {row['condition']:<12} "
            f"{row['base']:<24} seed={row['seed']} "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
            flush=True,
        )

    metrics = ASRMetricsCalculator(normalizer=PersianTextNormalizer())
    summary = summarize(rows, metrics, metrics.normalizer)
    meta = {
        "model": args.model,
        "dtype": dtype_name,
        "device": args.device,
        "seeds": list(args.seeds),
        "profiles": list(args.profiles),
        "use_ffmpeg": bool(args.use_ffmpeg),
        "level_match": bool(args.level_match),
        "target_active_dbfs": float(args.target_active_dbfs),
        "legacy_dir": str(args.legacy_dir),
        "clips_dir": str(args.clips_dir),
        "transcripts": str(args.transcripts),
        "elapsed_s": round(time.time() - started, 1),
    }
    write_outputs(rows, summary, args.output_dir, meta)

    print("\ncondition      n  clips  corpusWER  corpusWER*  clipWER*   CER")
    for condition, data in summary["conditions"].items():
        print(
            f"{condition:<14} {data['n']:>3} {data['n_clips']:>6} "
            f"{data['corpus_wer']:>9.2f}% "
            f"{data['corpus_wer_zwnj_space']:>10.2f}% "
            f"{data['mean_clip_wer_zwnj_space']:>7.2f}% "
            f"{data['corpus_cer']:>6.2f}%"
        )
    print("* ZWNJ->space diagnostic normalization")
    for profile, data in summary["deltas"].items():
        print(
            f"sim - legacy {profile:<7} mean clip WER* "
            f"{data['sim_minus_legacy_mean_clip_wer_zwnj_space']:+.2f} pp "
            f"(corpus* {data['sim_minus_legacy_corpus_wer_zwnj_space']:+.2f} pp)"
        )
    print(f"\nWrote {args.output_dir / 'summary.md'}")
    print(
        "NOTE: the six FakePhoneCalls are legacy synth, not real network "
        "recordings; this does not validate against real carriers."
    )


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
