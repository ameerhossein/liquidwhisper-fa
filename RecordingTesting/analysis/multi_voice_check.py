#!/usr/bin/env python3
"""Multi-voice validation for the telephone profiles.

Runs the gated/mobile/PSTN profiles on independent Persian speaker clips and
compares the speech-band metrics with the real gated/noisy corpora.  This is
the check that separates source-voice colour from channel colour: if the
mobile/PSTN output tracks the source HF content, a fixed high shelf is not
justified.

Example:
    python RecordingTesting/analysis/multi_voice_check.py \
        --voice-dir /home/ameer/Hobbies/voiceprint/fa_clips \
        --seeds 1 2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RecordingTesting" / "analysis"))

import compare_telephone as C  # noqa: E402
from src.telephone import simulate_telephone  # noqa: E402

PROFILES = ("gated", "mobile", "pstn")
FIELDS = [
    "speech_rms_db", "noise_rms_db", "line_noise_db", "line_snr_db",
    "spec_centroid", "spec_roll95", "hf_1_2k_rel_db", "hf_2_3k_rel_db",
    "hf_3_4k_rel_db", "speech_frac",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice-dir", type=Path,
                   default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/tmp/opencode/voicecheck"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--min-duration", type=float, default=2.0)
    return p.parse_args()


def scalar_row(row):
    return {
        k: (float(row[k]) if row.get(k) is not None and np.isfinite(row[k])
            else float("nan"))
        for k in FIELDS
    }


def med_iqr(values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(vals):
        return [float("nan")] * 3
    return [float(np.median(vals)), float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75))]


def load_real_targets():
    summary = json.loads(
        (ROOT / "RecordingTesting" / "analysis" / "summary.json").read_text()
    )
    out = {}
    for key in ("real_gated", "real_noisy"):
        out[key] = {}
        for name, vals in summary[key]["metrics"].items():
            if name in FIELDS:
                out[key][name] = {
                    "median": float(vals[0]),
                    "iqr": [float(vals[1]), float(vals[2])],
                }
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    for path in sorted(args.voice_dir.glob("*/*.wav")):
        info = sf.info(path)
        if info.frames / info.samplerate >= args.min_duration:
            sources.append(path)
    if not sources:
        raise SystemExit(f"no source clips >= {args.min_duration}s in {args.voice_dir}")

    rows = []
    for seed in args.seeds:
        out_dir = args.out_dir / f"synth_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(sources, 1):
            speaker = path.parent.name
            x, sr = sf.read(path, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            src = scalar_row(C.analyze_file(path, speaker))
            for profile in PROFILES:
                y = simulate_telephone(
                    x, sr=16000, profile=profile, seed=seed,
                    use_ffmpeg=True, output_sr=16000,
                )
                out_path = out_dir / f"{speaker}_{path.stem}_{profile}.wav"
                sf.write(out_path, y, 16000, subtype="PCM_16")
                out = scalar_row(C.analyze_file(out_path, profile))
                row = {"speaker": speaker, "clip": path.stem,
                       "profile": profile, "seed": seed}
                for name in FIELDS:
                    row[f"src_{name}"] = src[name]
                    row[f"out_{name}"] = out[name]
                    row[f"delta_{name}"] = out[name] - src[name]
                rows.append(row)
            if i % 10 == 0 or i == len(sources):
                print(f"seed {seed}: {i}/{len(sources)} clips", flush=True)

    files_csv = args.out_dir / "multi_voice_files.csv"
    with files_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {files_csv} ({len(rows)} synth rows)")

    real = load_real_targets()
    summary = {"n_sources": len(sources), "seeds": args.seeds, "profiles": {}}
    for profile in PROFILES:
        pr = [r for r in rows if r["profile"] == profile]
        summary["profiles"][profile] = {
            name: med_iqr([r[f"out_{name}"] for r in pr])
            for name in FIELDS
        }

    summary["speakers"] = {}
    for speaker in sorted({r["speaker"] for r in rows}):
        summary["speakers"][speaker] = {}
        for profile in ("source",) + PROFILES:
            if profile == "source":
                pr = [r for r in rows if r["speaker"] == speaker
                      and r["profile"] == "gated"]
                prefix = "src_"
            else:
                pr = [r for r in rows if r["speaker"] == speaker
                      and r["profile"] == profile]
                prefix = "out_"
            summary["speakers"][speaker][profile] = {
                name: med_iqr([r[f"{prefix}{name}"] for r in pr])
                for name in ("spec_centroid", "spec_roll95",
                             "hf_1_2k_rel_db", "hf_2_3k_rel_db",
                             "hf_3_4k_rel_db", "speech_rms_db")
            }

    summary["channel_delta"] = {}
    for profile in PROFILES:
        pr = [r for r in rows if r["profile"] == profile]
        summary["channel_delta"][profile] = {}
        for name in ("hf_1_2k_rel_db", "hf_2_3k_rel_db", "hf_3_4k_rel_db"):
            src = np.asarray([r[f"src_{name}"] for r in pr], dtype=float)
            out = np.asarray([r[f"out_{name}"] for r in pr], dtype=float)
            delta = np.asarray([r[f"delta_{name}"] for r in pr], dtype=float)
            ok = np.isfinite(src) & np.isfinite(out)
            summary["channel_delta"][profile][name] = {
                "median_delta_db": float(np.nanmedian(delta)),
                "corr": (float(np.corrcoef(src[ok], out[ok])[0, 1])
                         if np.sum(ok) > 2 else float("nan")),
            }

    summary["overlap_real_noisy"] = {}
    for profile in PROFILES:
        pr = [r for r in rows if r["profile"] == profile]
        summary["overlap_real_noisy"][profile] = {}
        for name in ("hf_1_2k_rel_db", "hf_2_3k_rel_db", "hf_3_4k_rel_db"):
            lo, hi = real["real_noisy"][name]["iqr"]
            vals = np.asarray([r[f"out_{name}"] for r in pr], dtype=float)
            summary["overlap_real_noisy"][profile][name] = {
                "inside_fraction": float(np.mean((vals >= lo) & (vals <= hi))),
                "real_noisy_iqr": [lo, hi],
            }

    summary_path = args.out_dir / "multi_voice_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")

    print("\nprofile medians (real noisy in brackets)")
    for profile in PROFILES:
        m = summary["profiles"][profile]
        print(f"\n{profile}")
        for name in ("spec_centroid", "spec_roll95", "hf_1_2k_rel_db",
                     "hf_2_3k_rel_db", "hf_3_4k_rel_db", "speech_rms_db"):
            vals = m[name]
            print(f"  {name:20} {vals[0]:8.2f} "
                  f"[{vals[1]:8.2f}, {vals[2]:8.2f}]")
        for name in ("hf_1_2k_rel_db", "hf_2_3k_rel_db", "hf_3_4k_rel_db"):
            target = real["real_noisy"][name]["median"]
            inside = summary["overlap_real_noisy"][profile][name]["inside_fraction"]
            print(f"  {name:20} real-noisy median={target:7.2f} "
                  f"inside={inside*100:5.1f}%")

    print("\nchannel delta (output - source)")
    for profile in PROFILES:
        for name, v in summary["channel_delta"][profile].items():
            print(f"  {profile:6} {name:20} "
                  f"median={v['median_delta_db']:7.2f} dB  r={v['corr']: .3f}")


if __name__ == "__main__":
    main()
