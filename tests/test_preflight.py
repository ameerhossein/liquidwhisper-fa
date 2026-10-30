"""
Offline unit tests for preflight fingerprint, cache, parsing, and deltas.
No torch, transformers, network, or subprocess required.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preflight import (
    build_fingerprint,
    compute_deltas,
    load_cached_preflight,
    parse_evaluation_details,
)


def test_fingerprint_changes_with_sample_count_and_seed():
    a = build_fingerprint(dataset_name="d", split="test", sample_count=1000,
                          phone_seed=1234, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    b = build_fingerprint(dataset_name="d", split="test", sample_count=2149,
                          phone_seed=1234, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    c = build_fingerprint(dataset_name="d", split="test", sample_count=1000,
                          phone_seed=9999, telephony_profiles=["customer"],
                          telephony_use_ffmpeg=True, eval_batch_size=16)
    assert a["sample_count"] == 1000
    assert a != b and a != c


def test_cached_preflight_requires_matching_fingerprint(tmp_path):
    fp = {"sample_count": 1000, "phone_seed": 1234}
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"fingerprint": fp, "models": {"buzzasr_base": {}}}))
    assert load_cached_preflight(path, fp)["fingerprint"] == fp
    assert load_cached_preflight(path, {**fp, "phone_seed": 1}) is None


def test_parse_evaluation_details_extracts_clean_and_phone():
    row = parse_evaluation_details({
        "metrics": {"total_samples": 1000,
                    "overall": {"normalized_wer": 17.2, "normalized_cer": 4.8,
                                "diagnostic_wer_zwnj_space": 15.9, "raw_wer": 18.0, "raw_cer": 5.0},
                    "formal": {"normalized_wer": 15.1},
                    "informal": {"normalized_wer": 22.4}},
        "phone_metrics": {"overall": {"normalized_wer": 24.1, "normalized_cer": 7.2},
                          "formal": {"normalized_wer": 21.0},
                          "informal": {"normalized_wer": 29.0}},
        "phone_seed": 1234, "phone_profiles": ["customer"],
    }, name="BuzzASR Base", epoch="BuzzASR Base (preflight, 1,000)")
    assert row["total_samples"] == 1000
    assert row["normalized_wer"] == 17.2
    assert row["phone_normalized_wer"] == 24.1
    assert row["phone_seed"] == 1234


def test_compute_deltas_requires_matching_sample_count():
    baseline = {"total_samples": 1000, "normalized_wer": 17.2, "phone_normalized_wer": 24.1}
    assert compute_deltas(15.0, 20.0, 1000, baseline) == {
        "delta_vs_buzzasr": -2.2, "delta_phone_vs_buzzasr": -4.1}
    assert compute_deltas(15.0, 20.0, 2149, baseline) == {
        "delta_vs_buzzasr": None, "delta_phone_vs_buzzasr": None}
