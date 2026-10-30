#!/usr/bin/env python3
"""Compare real gated calls with simulated gated audio in 1/3-octave bands.

The older report used band ratios relative to the 0.3-1 kHz sum.  That
denominator is dominated by the real corpus's 400-500 Hz peak, so it can
report a match while the 1 kHz-relative high band is 12 dB low.  This script
plots the loud-frame spectrum relative to 1 kHz, which exposes that shape,
and optionally reproduces the active in-band noise-SNR table.

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


def octave_abs(psd, freqs, centers=CENTERS):
    """Absolute 1/3-octave band levels for one spectrum."""
    levels = []
    for center in centers:
        lo = center / 2.0 ** (1.0 / 6.0)
        hi = center * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        levels.append(10.0 * np.log10(float(np.sum(psd[mask])) + 1e-30))
    return np.asarray(levels, dtype=np.float64)


def octave_levels(psds, freqs, centers=CENTERS, ref_hz=1000.0):
    """Median 1/3-octave band levels in dB, normalized at ``ref_hz``.

    This is the old difference-of-medians statistic: median each band across
    files, then normalize. It is retained for historical comparisons and is
    deprecated for calibration; use :func:`octave_shape` instead.
    """
    if not psds:
        raise ValueError("no spectra to summarize")
    levels = np.median(
        np.asarray([octave_abs(psd, freqs, centers) for psd in psds]),
        axis=0,
    )
    ref_idx = int(np.argmin(np.abs(np.asarray(centers) - ref_hz)))
    return levels - levels[ref_idx]


def octave_shape(psds, freqs, centers=CENTERS, ref_hz=1000.0):
    """Median per-file 1/3-octave shape, normalized at ``ref_hz``.

    Each spectrum is normalized by its own ``ref_hz`` band before the median.
    This is the appropriate statistic for a typical call's channel shape; it
    avoids the between-file level correlation that biases the pooled
    difference-of-medians.
    """
    if not psds:
        raise ValueError("no spectra to summarize")
    ref_idx = int(np.argmin(np.abs(np.asarray(centers) - ref_hz)))
    shapes = []
    for psd in psds:
        levels = octave_abs(psd, freqs, centers)
        shapes.append(levels - levels[ref_idx])
    return np.median(np.asarray(shapes), axis=0)


def loud_psd_for_audio(x8):
    """Loud-frame normalized PSD, or None when the clip has no speech."""
    speech, _, _ = ct.energy_vad(x8)
    if not np.any(speech):
        return None
    sample_mask = ct.mask_to_samples(speech, n=len(x8))
    spec, _ = ct.spectral_summary(x8, sample_mask, exclude_silence=True)
    return spec.get("loud_psd_norm")


def read_source(path: Path, target_sr: int) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = ss.resample_poly(x, target_sr, sr).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def real_gated_psds(real_dir: Path, manifest: Path | None = None,
                    outside_dir: Path | None = None):
    paths = []
    if manifest is not None and manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("calibration", "")).lower() != "true":
                    continue
                if row.get("source") == "outside":
                    if outside_dir is not None:
                        paths.append(outside_dir / row["path"])
                else:
                    paths.append(real_dir / row["file"])
    else:
        paths = sorted(real_dir.glob("*.wav"))
    psds = []
    for path in paths:
        row = ct.analyze_file(path, "real")
        if row and row.get("_loud_psd_norm") is not None:
            psds.append(row["_loud_psd_norm"])
    return psds


def clean_source_psds(voice_dir: Path, n_sources=48):
    psds = []
    for path in sorted(voice_dir.glob("*/*.wav"))[:n_sources]:
        psd = loud_psd_for_audio(read_source(path, tel.PHONE_SR))
        if psd is not None:
            psds.append(psd)
    return psds


def synth_customer_psds(voice_dir: Path, seeds, n_sources=48,
                        band_low=300.0, band_high=3400.0,
                        low_shelf_db=4.0, low_shelf_hz=500.0,
                        eq: bool = True):
    psds = []
    sources = sorted(voice_dir.glob("*/*.wav"))[:n_sources]
    if not sources:
        raise FileNotFoundError(f"no .wav clips under {voice_dir}")
    for path in sources:
        x = read_source(path, tel.TARGET_SR)
        for seed in seeds:
            y = tel.simulate_customer(
                x, sr=tel.TARGET_SR, seed=int(seed), output_sr=tel.PHONE_SR,
                band_low_hz=band_low, band_high_hz=band_high,
                low_shelf_db=low_shelf_db, low_shelf_hz=low_shelf_hz,
                eq_gains_db=None if eq else {},
            )
            psd = loud_psd_for_audio(y)
            if psd is not None:
                psds.append(psd)
    return psds


def band_snr_row(x8):
    """Median active-vs-noise 300-3400 Hz SNR and levels (dBFS)."""
    x8 = np.asarray(x8, dtype=np.float64).reshape(-1)
    speech, energies, _ = ct.energy_vad(x8)
    n = min(len(speech), len(energies))
    if n == 0:
        return None
    speech = speech[:n]
    energies = energies[:n]
    noise_floor = float(np.percentile(energies, 10))
    speech_level = float(np.percentile(energies, 95))
    threshold = max(noise_floor + 7.0, speech_level - 30.0)
    noise_sel = energies < (threshold - 4.0)
    if np.sum(noise_sel) < 5:
        noise_sel = energies <= np.percentile(energies, 10)
    sos = ss.butter(5, [300.0, 3400.0], btype="band", fs=tel.PHONE_SR, output="sos")
    band = ss.sosfilt(sos, x8)
    frames = band[:n * 160].reshape(n, 160)
    band_energy = 10.0 * np.log10(
        np.mean(frames ** 2, axis=1) + 1e-20
    )
    return {
        "band_snr_db": float(np.median(band_energy[speech])
                             - np.median(band_energy[noise_sel])),
        "full_noise_db": float(np.median(energies[noise_sel])),
        "band_noise_db": float(np.median(band_energy[noise_sel])),
        "speech_frac": float(np.mean(speech)),
    }


def real_noisy_snr_rows(real_dir: Path):
    rows = []
    for path in sorted(real_dir.glob("*.wav")):
        row = ct.analyze_file(path, "real")
        if row and not row.get("is_gated"):
            result = band_snr_row(ct.load_audio(path)[2])
            if result is not None:
                rows.append(result)
    return rows


def synth_noise_snr_rows(voice_dir: Path, seeds, profile: str,
                         n_sources=12):
    rows = []
    for path in sorted(voice_dir.glob("*/*.wav"))[:n_sources]:
        x = read_source(path, tel.TARGET_SR)
        for seed in seeds:
            y = tel.simulate_telephone(
                x, sr=tel.TARGET_SR, profile=profile, seed=int(seed),
                output_sr=tel.PHONE_SR, use_ffmpeg=False,
            )
            result = band_snr_row(y)
            if result is not None:
                rows.append(result)
    return rows


def _median_row(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-dir", type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "RealRecordedPhoneCalls",
    )
    parser.add_argument(
        "--outside-dir", type=Path,
        default=Path("/home/ameer/TerasVoice/Outside"),
    )
    parser.add_argument(
        "--content-manifest", type=Path,
        default=HERE / "answered_customer_manifest.csv",
    )
    parser.add_argument(
        "--voice-dir", type=Path,
        default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"),
    )
    parser.add_argument(
        "--eq", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the measured customer 1/3-octave EQ (default: on).",
    )
    parser.add_argument(
        "--legacy-gated", action="store_true",
        help="Reproduce the old all-gated, no-EQ comparison.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-sources", type=int, default=48)
    parser.add_argument("--band-low", type=float, default=300.0)
    parser.add_argument("--band-high", type=float, default=3400.0)
    parser.add_argument("--low-shelf-db", type=float, default=4.0)
    parser.add_argument("--low-shelf-hz", type=float, default=500.0)
    parser.add_argument(
        "--noise-check", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--noise-sources", type=int, default=12)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/opencode/gated_spectrum_check")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    freqs = ct.frame_psd(np.zeros(1024))[1]
    if args.legacy_gated:
        real = real_gated_psds(args.real_dir)
        synth = synth_customer_psds(
            args.voice_dir, args.seeds, args.n_sources,
            args.band_low, args.band_high, args.low_shelf_db, args.low_shelf_hz,
            eq=False,
        )
    else:
        real = real_gated_psds(args.real_dir, args.content_manifest, args.outside_dir)
        synth = synth_customer_psds(
            args.voice_dir, args.seeds, args.n_sources,
            args.band_low, args.band_high, args.low_shelf_db, args.low_shelf_hz,
            eq=args.eq,
        )
    clean = clean_source_psds(args.voice_dir, args.n_sources)
    real_levels = octave_shape(real, freqs)
    clean_levels = octave_shape(clean, freqs)
    synth_levels = octave_shape(synth, freqs)

    print(f"real customer files: {len(real)}; clean sources: {len(clean)}; "
          f"synth outputs: {len(synth)}")
    print("Hz   real_dB  clean_dB  synth_dB  diff_dB")
    for center, real_db, clean_db, synth_db in zip(
        CENTERS, real_levels, clean_levels, synth_levels
    ):
        print(
            f"{center:5d} {real_db:8.1f} {clean_db:9.1f} "
            f"{synth_db:9.1f} {synth_db - real_db:7.1f}"
        )

    with open(args.out_dir / "gated_octave_levels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "hz", "real_db_re_1k", "clean_db_re_1k",
            "synth_db_re_1k", "diff_db",
        ])
        for center, real_db, clean_db, synth_db in zip(
            CENTERS, real_levels, clean_levels, synth_levels
        ):
            writer.writerow([
                center, round(real_db, 2), round(clean_db, 2),
                round(synth_db, 2), round(synth_db - real_db, 2),
            ])

    if args.noise_check:
        real_noise = real_noisy_snr_rows(args.real_dir)
        groups = [
            ("real_noisy", real_noise),
            ("synth_mobile", synth_noise_snr_rows(
                args.voice_dir, args.noise_seeds, "mobile", args.noise_sources
            )),
            ("synth_pstn", synth_noise_snr_rows(
                args.voice_dir, args.noise_seeds, "pstn", args.noise_sources
            )),
        ]
        print("\nNoise check (median active-vs-noise, 300-3400 Hz):")
        print("group          n   band_snr  full_noise  band_noise  speech_frac")
        noise_rows = []
        for name, rows in groups:
            summary = {
                "group": name,
                "n": len(rows),
                "band_snr_db": _median_row(rows, "band_snr_db"),
                "full_noise_db": _median_row(rows, "full_noise_db"),
                "band_noise_db": _median_row(rows, "band_noise_db"),
                "speech_frac": _median_row(rows, "speech_frac"),
            }
            noise_rows.append(summary)
            print(
                f"{name:<14} {summary['n']:3d} "
                f"{summary['band_snr_db']:9.1f} "
                f"{summary['full_noise_db']:10.1f} "
                f"{summary['band_noise_db']:10.1f} "
                f"{summary['speech_frac']:10.2f}"
            )
        with open(args.out_dir / "noise_snr.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=[
                    "group", "n", "band_snr_db", "full_noise_db",
                    "band_noise_db", "speech_frac",
                ],
            )
            writer.writeheader()
            for row in noise_rows:
                writer.writerow({
                    key: (round(value, 2) if isinstance(value, float) else value)
                    for key, value in row.items()
                })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CENTERS))
    ax.plot(x, real_levels, "o-", label="Real answered customer")
    ax.plot(x, clean_levels, "--", label="Clean source")
    ax.plot(x, synth_levels, "s-", label="Simulated customer")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CENTERS], rotation=45)
    ax.set_xlabel("1/3-octave center (Hz)")
    ax.set_ylabel("Loud-frame PSD (dB re 1 kHz)")
    ax.set_title("Customer channel: real vs simulated")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "gated_octave_levels.png", dpi=140)
    print(f"\nWrote {args.out_dir / 'gated_octave_levels.csv'}")
    print(f"Wrote {args.out_dir / 'gated_octave_levels.png'}")
    if args.noise_check:
        print(f"Wrote {args.out_dir / 'noise_snr.csv'}")


if __name__ == "__main__":
    main()
