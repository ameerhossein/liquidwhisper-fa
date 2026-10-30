#!/usr/bin/env python3
"""Build the answered-customer 1/3-octave target bands.

This is the same loud-frame PSD / octave-level method used by
`gated_spectrum_check.py`, restricted to the manifest's calibration rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import compare_telephone as ct  # noqa: E402
import gated_spectrum_check as gs  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def resolve(manifest_row: dict[str, str], real_dir: Path, outside_dir: Path) -> Path:
    if manifest_row["source"] == "in_repo":
        return real_dir / manifest_row["file"]
    return outside_dir / manifest_row["path"]


def octave_abs(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Absolute 1/3-octave band levels (dB); caller normalizes to 1 kHz."""
    levels = []
    for center in gs.CENTERS:
        lo = center / 2.0 ** (1.0 / 6.0)
        hi = center * 2.0 ** (1.0 / 6.0)
        mask = (freqs >= lo) & (freqs < hi)
        levels.append(10.0 * np.log10(float(np.sum(psd[mask])) + 1e-30))
    return np.asarray(levels, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "answered_customer_manifest.csv",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "RealRecordedPhoneCalls",
    )
    parser.add_argument(
        "--outside-dir",
        type=Path,
        default=Path("/home/ameer/TerasVoice/Outside"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "answered_customer_bands.csv",
    )
    args = parser.parse_args()

    rows = [r for r in read_rows(args.manifest) if r["calibration"].lower() == "true"]
    freqs = ct.frame_psd(np.zeros(1024))[1]

    out_rows: list[dict[str, object]] = []
    psds = []
    for row in rows:
        path = resolve(row, args.real_dir, args.outside_dir)
        _, _, x8 = ct.load_audio(path)
        psd = gs.loud_psd_for_audio(x8)
        if psd is None:
            continue
        psds.append(psd)
        levels = octave_abs(psd, freqs)
        record: dict[str, object] = {
            "file": row["file"],
            "source": row["source"],
        }
        for center, level in zip(gs.CENTERS, levels):
            record[f"hz_{center}"] = round(float(level), 3)
        out_rows.append(record)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {args.out} ({len(out_rows)} rows)")
    band_medians = {
        center: float(np.median([float(r[f"hz_{center}"]) for r in out_rows]))
        for center in gs.CENTERS
    }
    ref = band_medians[1000]
    for center in [315, 400, 500, 3150]:
        print(
            f"  hz_{center}: median {band_medians[center] - ref:.2f} dB re 1 kHz"
        )


if __name__ == "__main__":
    main()
