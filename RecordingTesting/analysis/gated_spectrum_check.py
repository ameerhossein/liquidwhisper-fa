#!/usr/bin/env python3
"""Compare real gated calls with simulated gated audio in 1/3-octave bands.

The older report used band ratios relative to the 0.3-1 kHz sum.  That
denominator is dominated by the real corpus's 400-500 Hz peak, so it can
report a match while the 1 kHz-relative high band is 12 dB low.  This script
plots the loud-frame spectrum relative to 1 kHz, which exposes that shape.

Usage:
  python RecordingTesting/analysis/gated_spectrum_check.py \
      --voice-dir /home/ameer/Hobbies/voiceprint/fa_clips \
      --out-dir /tmp/opencode/gated_spectrum_check
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as ss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import compare_telephone as ct  # noqa: E402
from src import telephone as tel  # noqa: E402

CENTERS = [125, 160, 200, 250, 315, 400, 500, 630, 800,
           1000, 1250, 1600, 2000, 2500, 3150, 4000]


def octave_levels(psds, freqs, centers=CENTERS, ref_hz=1000.0):
    """Median 1/3-octave band levels in dB, normalized at ``ref_hz``."""
    if not psds:
        raise ValueError("no spectra to summarize")
    levels = []
    for center in centers:
        lo = center / 2.0 ** (1.0 / 6.0)
        hi = center * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        levels.append(float(np.median([
            10.0 * np.log10(float(np.sum(psd[mask])) + 1e-30) for psd in psds
        ])))
    levels = np.asarray(levels, dtype=np.float64)
    return levels - levels[int(np.argmin(np.abs(np.asarray(centers) - ref_hz)))]


def loud_psd_for_audio(x8):
    """Loud-frame normalized PSD, or None when the clip has no speech."""
    speech, _, _ = ct.energy_vad(x8)
    if not np.any(speech):
        return None
    sample_mask = ct.mask_to_samples(speech, n=len(x8))
    spec, _ = ct.spectral_summary(x8, sample_mask, exclude_silence=True)
    return spec.get("loud_psd_norm")


def real_gated_psds(real_dir: Path):
    psds = []
    for path in sorted(real_dir.glob("*.wav")):
        row = ct.analyze_file(path, "real")
        if row and row.get("is_gated") and row.get("_loud_psd_norm") is not None:
            psds.append(row["_loud_psd_norm"])
    return psds


def synth_gated_psds(voice_dir: Path, seeds, n_sources=48,
                     band_low=300.0, band_high=3400.0,
                     low_shelf_db=4.0, low_shelf_hz=500.0):
    psds = []
    sources = sorted(voice_dir.glob("*/*.wav"))[:n_sources]
    if not sources:
        raise FileNotFoundError(f"no .wav clips under {voice_dir}")
    for path in sources:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != tel.TARGET_SR:
            x = ss.resample_poly(x, tel.TARGET_SR, sr).astype(np.float32)
        for seed in seeds:
            y = tel.simulate_gated(
                x, sr=tel.TARGET_SR, seed=int(seed), output_sr=tel.PHONE_SR,
                band_low_hz=band_low, band_high_hz=band_high,
                low_shelf_db=low_shelf_db, low_shelf_hz=low_shelf_hz,
            )
            psd = loud_psd_for_audio(y)
            if psd is not None:
                psds.append(psd)
    return psds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-dir", type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "RealRecordedPhoneCalls",
    )
    parser.add_argument(
        "--voice-dir", type=Path,
        default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-sources", type=int, default=48)
    parser.add_argument("--band-low", type=float, default=300.0)
    parser.add_argument("--band-high", type=float, default=3400.0)
    parser.add_argument("--low-shelf-db", type=float, default=4.0)
    parser.add_argument("--low-shelf-hz", type=float, default=500.0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/opencode/gated_spectrum_check")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    freqs = ct.frame_psd(np.zeros(1024))[1]
    real = real_gated_psds(args.real_dir)
    synth = synth_gated_psds(
        args.voice_dir, args.seeds, args.n_sources,
        args.band_low, args.band_high, args.low_shelf_db, args.low_shelf_hz,
    )
    real_levels = octave_levels(real, freqs)
    synth_levels = octave_levels(synth, freqs)

    print(f"real gated files: {len(real)}; synth outputs: {len(synth)}")
    print("Hz   real_dB  synth_dB  diff_dB")
    for center, real_db, synth_db in zip(CENTERS, real_levels, synth_levels):
        print(f"{center:5d} {real_db:8.1f} {synth_db:8.1f} {synth_db - real_db:7.1f}")

    with open(args.out_dir / "gated_octave_levels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hz", "real_db_re_1k", "synth_db_re_1k", "diff_db"])
        for center, real_db, synth_db in zip(CENTERS, real_levels, synth_levels):
            writer.writerow([center, round(real_db, 2), round(synth_db, 2),
                             round(synth_db - real_db, 2)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CENTERS))
    ax.plot(x, real_levels, "o-", label="Real gated calls")
    ax.plot(x, synth_levels, "s-", label="Simulated gated")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CENTERS], rotation=45)
    ax.set_xlabel("1/3-octave center (Hz)")
    ax.set_ylabel("Loud-frame PSD (dB re 1 kHz)")
    ax.set_title("Gated channel: real vs simulated")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "gated_octave_levels.png", dpi=140)
    print(f"\nWrote {args.out_dir / 'gated_octave_levels.csv'}")
    print(f"Wrote {args.out_dir / 'gated_octave_levels.png'}")


if __name__ == "__main__":
    main()
