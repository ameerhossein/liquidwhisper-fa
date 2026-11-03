"""
Unit tests for telemetry callback baseline rows and epoch deltas.

No torch model, dataset, or CUDA is required. The callback only writes JSON
files during construction and ``_record_epoch_benchmark``.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer_callbacks import WebTelemetryCallback
import pytest


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


@pytest.mark.parametrize("failure", ["exception", "no-profile"])
def test_epoch_phone_degradation_never_falls_back_to_clean(tmp_path, monkeypatch, failure):
    class Rows:
        def __getitem__(self, index):
            return {"audio": [{"array": [0.0] * 160}]}

    def degrade(waveform, **kwargs):
        if failure == "exception":
            raise RuntimeError("codec unavailable")
        return waveform, None

    monkeypatch.setattr("src.telephone.degrade_for_eval_profiled", degrade)
    cb = WebTelemetryCallback(results_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="Phone degradation"):
        cb._transcribe_pass(Rows(), None, "cpu", None, False, 1, 1, degrade_phone=True)


def test_epoch_phone_pass_failure_propagates_and_restores_training_mode(tmp_path, monkeypatch):
    class Rows:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter([{"text": "سلام", "is_informal": False}])

    model = SimpleNamespace(training=True)
    model.eval = lambda: setattr(model, "training", False)
    model.train = lambda: setattr(model, "training", True)
    metrics = SimpleNamespace(evaluate_predictions=lambda **k: {"total_samples": 1})
    cb = WebTelemetryCallback(
        results_dir=str(tmp_path), test_dataset=Rows(), processor=object(),
        metrics_calculator=metrics,
    )

    def transcribe(*args, degrade_phone=False):
        if degrade_phone:
            raise RuntimeError("generation failed")
        return ["سلام"], [None]

    monkeypatch.setattr(cb, "_transcribe_pass", transcribe)
    with pytest.raises(RuntimeError, match="Phone benchmark failed: generation failed"):
        cb._evaluate_benchmark(model, 1)
    assert model.training is True
