import csv
from pathlib import Path
from statistics import median

ANALYSIS = Path(__file__).resolve().parent.parent / "RecordingTesting" / "analysis"
MANIFEST = ANALYSIS / "answered_customer_manifest.csv"
BANDS = ANALYSIS / "answered_customer_bands.csv"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_manifest_counts_and_calibration_subset():
    rows = _rows(MANIFEST)
    assert len(rows) == 320
    calib = [r for r in rows if r["calibration"].lower() == "true"]
    assert len(calib) == 67
    assert {r["source"] for r in calib} == {"in_repo", "outside"}
    assert {r["caller_class"] for r in calib} == {"mobile"}
    assert all(r["is_gated"].lower() == "true" for r in calib)
    assert all(r["content_class"] == "speech_only" for r in calib)
    for key, expected in {
        "speech_rms_db": -25.6,
        "noise_rms_db": -71.3,
        "hf_1_2k_rel_db": -15.1,
        "hf_2_3k_rel_db": -27.5,
        "spec_centroid": 459.2,
        "spec_roll95": 687.5,
    }.items():
        assert abs(median(float(r[key]) for r in calib) - expected) < 0.2


def test_answered_bands_have_67_rows_and_16_bands():
    rows = _rows(BANDS)
    assert len(rows) == 67
    band_cols = [c for c in rows[0] if c.startswith("hz_")]
    assert len(band_cols) == 16
    assert band_cols[0] == "hz_125"
    assert band_cols[-1] == "hz_4000"
    med = {c: median(float(r[c]) for r in rows) for c in band_cols}
    ref = med["hz_1000"]
    for col, expected in {
        "hz_315": 4.52,
        "hz_400": 12.99,
        "hz_500": 12.95,
        "hz_3150": -19.77,
    }.items():
        assert abs((med[col] - ref) - expected) < 0.2
