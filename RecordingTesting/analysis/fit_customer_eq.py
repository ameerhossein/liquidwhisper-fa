#!/usr/bin/env python3
"""Fit the customer 1/3-octave EQ to the answered-speech target.

The script measures the baseline customer chain (no EQ) on clean voices,
compares it with the committed answered-customer band table, and iteratively
updates the peaking gains. It is an offline analysis tool; the runtime
simulator uses the constants pasted into ``src/telephone.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
    med = {
        center: float(np.median([float(r[f"hz_{center}"]) for r in rows]))
        for center in gs.CENTERS
    }
    ref = med[1000]
    return {center: med[center] - ref for center in gs.CENTERS}


def synth_levels(voice_dir: Path, seeds, n_sources: int,
                 gains: dict[float, float]) -> dict[float, float]:
    psds = []
    for path in sorted(voice_dir.glob("*/*.wav"))[:n_sources]:
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
    levels = gs.octave_levels(psds, freqs)
    return dict(zip(gs.CENTERS, map(float, levels)))


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
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencode/customer_eq_fit"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target = read_bands(args.bands)
    gains = {float(k): float(v) for k, v in tel.CUSTOMER_EQ_GAINS_DB.items()}
    history = []

    for iteration in range(args.iterations + 1):
        measured = synth_levels(args.voice_dir, args.seeds, args.n_sources, gains)
        residual = {
            center: target[center] - measured[center] for center in gs.CENTERS
        }
        history.append({
            "iteration": iteration,
            "gains": dict(gains),
            "measured": {str(k): round(v, 3) for k, v in measured.items()},
            "residual": {str(k): round(v, 3) for k, v in residual.items()},
        })
        print(f"\nIteration {iteration}")
        print("Hz     target  measured  residual")
        for center in PRIMARY:
            print(
                f"{center:5.0f} {target[center]:8.2f} {measured[center]:9.2f} "
                f"{residual[center]:9.2f}"
            )
        if iteration == args.iterations:
            break
        for center in PRIMARY:
            gains[center] = float(np.clip(
                gains[center] + args.damping * residual[center], -20.0, 15.0
            ))

    out = args.out_dir / "customer_eq_gains.json"
    out.write_text(
        json.dumps({"gains": {str(k): round(v, 3) for k, v in gains.items()},
                    "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    print("Final gains:")
    for center in PRIMARY:
        print(f"  {center:5.0f}: {gains[center]:+.2f} dB")


if __name__ == "__main__":
    main()
