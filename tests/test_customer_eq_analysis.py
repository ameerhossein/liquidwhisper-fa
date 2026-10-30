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

    def make_psd(level_1k_db, level_3150_db):
        psd = np.full(len(freqs), 1e-12, dtype=np.float64)
        for center, level_db in ((1000.0, level_1k_db), (3150.0, level_3150_db)):
            lo = center / 2.0 ** (1.0 / 6.0)
            hi = center * 2.0 ** (1.0 / 6.0)
            mask = (freqs >= lo) & (freqs < hi)
            psd[mask] = (10.0 ** (level_db / 10.0)) / max(1, int(np.sum(mask)))
        return psd

    # Per-file 3150/1k shapes are -30, -10, -50 dB; median -30 dB.
    # Difference-of-medians gives -10 dB, so this fixture distinguishes them.
    psd_a = make_psd(0.0, -30.0)
    psd_b = make_psd(0.0, -10.0)
    psd_c = make_psd(100.0, 50.0)
    shape = gs.octave_shape([psd_a, psd_b, psd_c], freqs)
    idx = gs.CENTERS.index(3150.0)
    assert shape[idx] == pytest.approx(-30.0, abs=0.1)

    pooled = gs.octave_levels([psd_a, psd_b, psd_c], freqs)
    assert pooled[idx] == pytest.approx(-10.0, abs=0.1)


def test_outside_high_band_residuals_are_not_dark():
    from statistics import median

    with open(ANALYSIS / "answered_customer_bands.csv", newline="", encoding="utf-8") as fh:
        bands = [
            row for row in csv.DictReader(fh) if row["source"] == "outside"
        ]
    with open(ANALYSIS / "customer_octave_levels.csv", newline="", encoding="utf-8") as fh:
        synth = {
            row["hz"]: float(row["synth_db_re_1k"])
            for row in csv.DictReader(fh)
        }
    assert bands and synth
    for hz in ("1600", "2000", "2500", "3150"):
        real = median(
            float(row[f"hz_{hz}"]) - float(row["hz_1000"]) for row in bands
        )
        diff = synth[hz] - real
        assert abs(diff) <= 2.5, (hz, diff)
        if hz == "3150":
            assert diff > -2.0
