"""Regression tests for the customer 1/3-octave calibration statistic."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "RecordingTesting" / "analysis"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT))

import compare_telephone as ct  # noqa: E402
import fit_customer_eq as fit  # noqa: E402
import gated_spectrum_check as gs  # noqa: E402


def test_read_bands_uses_median_per_file_shape(tmp_path):
    """median(band - own 1k) must win over median(band) - median(1k)."""
    path = tmp_path / "bands.csv"
    columns = ["file", "source", *[f"hz_{c}" for c in gs.CENTERS]]
    rows = []
    for name, level_1k, level_3150 in [
        ("a.wav", 0.0, -30.0),
        ("b.wav", 0.0, -10.0),
        ("c.wav", 100.0, 50.0),
    ]:
        row = {column: 0.0 for column in columns}
        row.update({"file": name, "source": "outside"})
        row["hz_1000"] = level_1k
        row["hz_3150"] = level_3150
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    target = fit.read_bands(path)

    # Per-file shapes are -30, -10, -50 dB; their median is -30 dB.
    # The old difference-of-medians statistic gives -10 dB.
    assert target[3150.0] == pytest.approx(-30.0)


def test_octave_shape_is_median_of_per_file_shapes():
    freqs = ct.frame_psd(np.zeros(1024))[1]
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
    idx_3150 = int(np.argmin(np.abs(freqs - 3150.0)))

    def make_psd(level_1k, level_3150):
        psd = np.full(len(freqs), 1e-12, dtype=np.float64)
        for center, level in ((1000.0, level_1k), (3150.0, level_3150)):
            lo = center / 2.0 ** (1.0 / 6.0)
            hi = center * 2.0 ** (1.0 / 6.0)
            mask = (freqs >= lo) & (freqs < hi)
            psd[mask] = level / max(1, int(np.sum(mask)))
        return psd, idx_1k, idx_3150

    psd_a, _, _ = make_psd(1.0, 0.01)      # -20 dB shape
    psd_b, _, _ = make_psd(100.0, 10.0)    # -10 dB shape
    shape = gs.octave_shape([psd_a, psd_b], freqs)
    idx = gs.CENTERS.index(3150.0)
    assert shape[idx] == pytest.approx(-15.0, abs=0.15)
