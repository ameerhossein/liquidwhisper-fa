"""Regression tests for the post-training report's measured baseline handling."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _train_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_lora

    return train_lora


def test_format_delta_uses_percentage_points():
    train_lora = _train_lora()
    assert train_lora._format_delta(-2.2) == "-2.20 pp"
    assert train_lora._format_delta(1.5) == "+1.50 pp"
    assert "no matching full-test baseline" in train_lora._format_delta(None)


def test_format_epoch_delta_uses_pp_and_dash():
    train_lora = _train_lora()
    assert train_lora._format_epoch_delta(None) == "--"
    assert train_lora._format_epoch_delta(-4.1) == "-4.10 pp"
    assert train_lora._format_epoch_delta(0.0) == "+0.00 pp"


def test_report_drops_legacy_baselines_and_recomputes_epoch_deltas():
    train_lora = _train_lora()
    buzz = {
        "source": "preflight",
        "total_samples": 1000,
        "normalized_wer": 17.2,
        "phone_normalized_wer": 24.1,
    }
    rows = [
        {   # Legacy hardcoded baseline: no measured source, must disappear.
            "epoch": "BuzzASR Base",
            "is_baseline": True,
            "total_samples": 2149,
            "normalized_wer": 17.2,
        },
        {   # Legacy 300-row epoch with stale deltas against the old baseline.
            "epoch": "Epoch 1",
            "is_baseline": False,
            "total_samples": 300,
            "normalized_wer": 10.0,
            "phone_normalized_wer": 12.0,
            "delta_vs_buzzasr": -7.2,
            "delta_phone_vs_buzzasr": -12.1,
        },
        {   # Matching 1,000-row row: stored deltas must be recomputed.
            "epoch": "Epoch 2",
            "is_baseline": False,
            "total_samples": 1000,
            "normalized_wer": 15.0,
            "phone_normalized_wer": 20.0,
            "delta_vs_buzzasr": 99.0,
            "delta_phone_vs_buzzasr": 99.0,
        },
        {   # Measured preflight baseline row stays in the history table.
            "epoch": "BuzzASR Base (preflight, 1,000)",
            "is_baseline": True,
            "source": "preflight",
            "total_samples": 1000,
            "normalized_wer": 17.2,
            "phone_normalized_wer": 24.1,
        },
    ]

    prepared = train_lora._normalize_epoch_benchmarks(rows, buzz)
    by_epoch = {row["epoch"]: row for row in prepared}

    assert "BuzzASR Base" not in by_epoch
    assert set(by_epoch) == {"Epoch 1", "Epoch 2", "BuzzASR Base (preflight, 1,000)"}
    assert by_epoch["Epoch 1"]["delta_vs_buzzasr"] is None
    assert by_epoch["Epoch 1"]["delta_phone_vs_buzzasr"] is None
    assert by_epoch["Epoch 2"]["delta_vs_buzzasr"] == -2.2
    assert by_epoch["Epoch 2"]["delta_phone_vs_buzzasr"] == -4.1
    assert by_epoch["BuzzASR Base (preflight, 1,000)"]["delta_vs_buzzasr"] is None


def test_report_without_baseline_never_trusts_stored_deltas():
    train_lora = _train_lora()
    prepared = train_lora._normalize_epoch_benchmarks(
        [{
            "epoch": "Epoch 1",
            "is_baseline": False,
            "total_samples": 1000,
            "normalized_wer": 15.0,
            "phone_normalized_wer": 20.0,
            "delta_vs_buzzasr": -2.2,
            "delta_phone_vs_buzzasr": -4.1,
        }],
        {},
    )
    assert prepared[0]["delta_vs_buzzasr"] is None
    assert prepared[0]["delta_phone_vs_buzzasr"] is None
