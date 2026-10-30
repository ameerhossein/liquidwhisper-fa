#!/usr/bin/env python3
"""Fit the customer 1/3-octave EQ to the answered-speech target.

The script measures the customer chain on clean voices, compares per-file
normalized 1/3-octave shapes with the committed answered-customer band table,
and iteratively updates the peaking gains. It is an offline analysis tool; the
runtime simulator uses the constants pasted into ``src/telephone.py``.

By default the fit starts from ``CUSTOMER_EQ_INITIAL_GAINS_DB`` (the measured
target-minus-gated delta) and uses the documented 8 iterations / 0.5 damping,
so the JSON history is reproducible. Use ``--initial-gains final`` only to
inspect the shipped constants. Use ``--holdout-speakers`` to fit on the
remaining speakers and report a true speaker-held-out residual table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import compare_telephone as ct  # noqa: E402
import gated_spectrum_check as gs  # noqa: E402
from src import telephone as tel  # noqa: E402

PRIMARY = [315.0, 400.0, 500.0, 630.0, 800.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0]


def read_bands(path: Path) -> dict[float, float]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Normalize each file to its own 1 kHz band before taking the median.
    # The old difference-of-medians method was biased by between-file level
    # correlation and over-cut the high bands.
    shapes = []
    for row in rows:
        ref = float(row["hz_1000"])
        shapes.append([
            float(row[f"hz_{center}"]) - ref for center in gs.CENTERS
        ])
    med = np.median(np.asarray(shapes, dtype=np.float64), axis=0)
    return {center: float(value) for center, value in zip(gs.CENTERS, med)}


def synth_levels(voice_dir: Path, seeds, n_sources: int,
                 gains: dict[float, float],
                 speakers: Optional[set[str]] = None,
                 exclude: Optional[set[str]] = None) -> dict[float, float]:
    psds = []
    for path in sorted(voice_dir.glob("*/*.wav"))[:n_sources]:
        speaker = path.parent.name
        if speakers is not None and speaker not in speakers:
            continue
        if exclude and speaker in exclude:
            continue
        x = gs.read_source(path, tel.TARGET_SR)
        for seed in seeds:
            y = tel.simulate_customer(
                x, seed=int(seed), output_sr=tel.PHONE_SR,
                use_ffmpeg=False, eq_gains_db=gains,
            )
            psd = gs.loud_psd_for_audio(y)
            if psd is not None:
                psds.append(psd)
    freqs = ct.frame_psd(np.zeros(1024))[1]
    levels = gs.octave_shape(psds, freqs)
    return dict(zip(gs.CENTERS, map(float, levels)))


def print_table(title: str, target: dict[float, float],
                measured: dict[float, float]) -> dict[float, float]:
    print(f"\n{title}")
    print("Hz     target  measured  residual")
    residual = {}
    for center in PRIMARY:
        residual[center] = target[center] - measured[center]
        print(
            f"{center:5.0f} {target[center]:8.2f} {measured[center]:9.2f} "
            f"{residual[center]:9.2f}"
        )
    return residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bands", type=Path,
        default=HERE / "answered_customer_bands.csv",
    )
    parser.add_argument(
        "--voice-dir", type=Path,
        default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"),
    )
    parser.add_argument("--n-sources", type=int, default=46)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument(
        "--initial-gains", choices=["initial", "final"], default="initial",
        help="Start from the measured delta (default) or the shipped constants.",
    )
    parser.add_argument(
        "--holdout-speakers", nargs="+", default=[],
        help="Fit on the other speakers and report this held-out residual.",
    )
    parser.add_argument(
        "--exclude-speakers", nargs="+", default=[],
        help="Additional speakers to exclude from the fit.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencode/customer_eq_fit"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target = read_bands(args.bands)
    initial = (
        tel.CUSTOMER_EQ_INITIAL_GAINS_DB
        if args.initial_gains == "initial" else tel.CUSTOMER_EQ_GAINS_DB
    )
    gains = {float(k): float(v) for k, v in initial.items()}
    fit_exclude = set(args.exclude_speakers) | set(args.holdout_speakers)
    history = []

    for iteration in range(args.iterations + 1):
        measured = synth_levels(
            args.voice_dir, args.seeds, args.n_sources, gains,
            exclude=fit_exclude,
        )
        residual = print_table(f"Iteration {iteration}", target, measured)
        history.append({
            "iteration": iteration,
            "gains": dict(gains),
            "measured": {str(k): round(v, 3) for k, v in measured.items()},
            "residual": {str(k): round(v, 3) for k, v in residual.items()},
        })
        if iteration == args.iterations:
            break
        for center in PRIMARY:
            gains[center] = float(np.clip(
                gains[center] + args.damping * residual[center], -20.0, 15.0
            ))

    result = {
        "initial_gains": {str(k): round(v, 3) for k, v in initial.items()},
        "gains": {str(k): round(v, 3) for k, v in gains.items()},
        "history": history,
    }
    if args.holdout_speakers:
        holdout = synth_levels(
            args.voice_dir, args.seeds, args.n_sources, gains,
            speakers=set(args.holdout_speakers),
        )
        holdout_residual = print_table(
            f"Held-out speakers: {', '.join(args.holdout_speakers)}",
            target, holdout,
        )
        result["holdout_speakers"] = list(args.holdout_speakers)
        result["holdout_measured"] = {
            str(k): round(v, 3) for k, v in holdout.items()
        }
        result["holdout_residual"] = {
            str(k): round(v, 3) for k, v in holdout_residual.items()
        }

    out = args.out_dir / "customer_eq_gains.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("Final gains:")
    for center in PRIMARY:
        print(f"  {center:5.0f}: {gains[center]:+.2f} dB")


if __name__ == "__main__":
    main()
