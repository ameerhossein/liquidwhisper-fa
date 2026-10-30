"""
Unit tests for telemetry callback baseline rows and epoch deltas.

No torch model, dataset, or CUDA is required. The callback only writes JSON
files during construction and ``_record_epoch_benchmark``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer_callbacks import WebTelemetryCallback


def test_initial_rows_include_preflight_phone_metrics(tmp_path):
    WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"epoch": "BuzzASR Base (preflight, 1,000)", "source": "preflight",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
        "whisper_v3": {"epoch": "Whisper (preflight)", "source": "preflight",
                       "is_reference": True, "total_samples": 1000,
                       "normalized_wer": 28.5, "phone_normalized_wer": 35.0},
    })
    rows = json.loads((tmp_path / "epoch_benchmarks.json").read_text())
    by_epoch = {r["epoch"]: r for r in rows}
    assert set(by_epoch) == {"BuzzASR Base (preflight, 1,000)", "Whisper (preflight)"}
    assert by_epoch["BuzzASR Base (preflight, 1,000)"]["phone_normalized_wer"] == 24.1


def test_epoch_record_computes_clean_and_phone_deltas(tmp_path):
    cb = WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1}})
    cb._record_epoch_benchmark(1, {
        "total_samples": 1000,
        "overall": {"normalized_wer": 15.0, "normalized_cer": 4.0,
                    "diagnostic_wer_zwnj_space": 14.0},
        "formal": {"normalized_wer": 13.0}, "informal": {"normalized_wer": 19.0},
        "phone_metrics": {"overall": {"normalized_wer": 20.0, "normalized_cer": 6.0},
                          "formal": {"normalized_wer": 18.0},
                          "informal": {"normalized_wer": 23.0}},
        "phone_seed": 1234, "phone_profiles": ["customer"],
    })
    row = json.loads((tmp_path / "epoch_benchmarks.json").read_text())[-1]
    assert row["delta_vs_buzzasr"] == -2.2
    assert row["delta_phone_vs_buzzasr"] == -4.1
    assert "delta_vs_whisper" not in row


def test_epoch_record_omits_deltas_on_sample_count_mismatch(tmp_path):
    cb = WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1}})
    cb._record_epoch_benchmark(1, {
        "total_samples": 2149,
        "overall": {"normalized_wer": 15.0, "normalized_cer": 4.0,
                    "diagnostic_wer_zwnj_space": 14.0},
        "formal": {"normalized_wer": 13.0}, "informal": {"normalized_wer": 19.0},
        "phone_metrics": {"overall": {"normalized_wer": 20.0, "normalized_cer": 6.0},
                          "formal": {"normalized_wer": 18.0},
                          "informal": {"normalized_wer": 23.0}},
        "phone_seed": 1234, "phone_profiles": ["customer"],
    })
    row = json.loads((tmp_path / "epoch_benchmarks.json").read_text())[-1]
    assert row["delta_vs_buzzasr"] is None
    assert row["delta_phone_vs_buzzasr"] is None


def test_initial_rows_replace_stale_baselines_and_preserve_epochs(tmp_path):
    stale = [
        {"epoch": "Whisper Large v3", "type": "baseline", "is_baseline": True,
         "normalized_wer": 28.5},
        {"epoch": "BuzzASR Base", "type": "baseline", "is_baseline": True,
         "normalized_wer": 17.2},
        {"epoch": "Epoch 1", "type": "checkpoint", "is_baseline": False,
         "normalized_wer": 16.0},
    ]
    (tmp_path / "epoch_benchmarks.json").write_text(json.dumps(stale))
    WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"epoch": "BuzzASR Base (preflight, 1,000)", "source": "preflight",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
    })
    rows = json.loads((tmp_path / "epoch_benchmarks.json").read_text())
    epochs = [r["epoch"] for r in rows]
    assert "Whisper Large v3" not in epochs
    assert "BuzzASR Base" not in epochs
    assert "BuzzASR Base (preflight, 1,000)" in epochs
    assert "Epoch 1" in epochs


def test_missing_whisper_baseline_is_tolerated(tmp_path):
    cb = WebTelemetryCallback(results_dir=str(tmp_path), baseline_benchmark={
        "buzzasr_base": {"epoch": "BuzzASR Base (preflight, 1,000)", "source": "preflight",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
    })
    rows = json.loads((tmp_path / "epoch_benchmarks.json").read_text())
    assert [r["epoch"] for r in rows] == ["BuzzASR Base (preflight, 1,000)"]

    cb._record_epoch_benchmark(1, {
        "total_samples": 1000,
        "overall": {"normalized_wer": 15.0},
        "formal": {"normalized_wer": 13.0}, "informal": {"normalized_wer": 19.0},
        "phone_metrics": {"overall": {"normalized_wer": 20.0}},
    })
    row = json.loads((tmp_path / "epoch_benchmarks.json").read_text())[-1]
    assert row["delta_vs_buzzasr"] == -2.2
    assert "delta_vs_whisper" not in row
