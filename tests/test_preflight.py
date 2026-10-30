"""
Offline unit tests for preflight fingerprint, cache, parsing, deltas, and runner.
No torch, transformers, network, or real subprocess is required; the runner test
monkeypatches ``subprocess.run``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preflight import (
    build_fingerprint,
    compute_deltas,
    load_cached_preflight,
    parse_evaluation_details,
    preflight_path,
    run_preflight,
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


def _fake_details(model: str):
    clean = {"total_samples": 1000,
             "overall": {"normalized_wer": 17.2, "normalized_cer": 4.8,
                         "diagnostic_wer_zwnj_space": 15.9, "raw_wer": 18.0, "raw_cer": 5.0},
             "formal": {"normalized_wer": 15.1},
             "informal": {"normalized_wer": 22.4}}
    phone = {"overall": {"normalized_wer": 24.1, "normalized_cer": 7.2},
             "formal": {"normalized_wer": 21.0},
             "informal": {"normalized_wer": 29.0}}
    return {"metrics": clean, "phone_metrics": phone,
            "phone_seed": 1234, "phone_profiles": ["customer"]}


def test_run_preflight_writes_unified_json_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1])
        out.mkdir(parents=True)
        model = cmd[cmd.index("--base_model") + 1]
        payload = _fake_details(model)
        (out / "evaluation_details.json").write_text(json.dumps(payload))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert preflight_path(tmp_path).exists()
    assert result["models"]["buzzasr_base"]["phone_normalized_wer"] == 24.1
    assert result["models"]["whisper_v3"]["is_reference"] is True
    assert len(calls) == 2
    run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                  sample_count=1000, phone_seed=1234,
                  telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                  eval_batch_size=16)
    assert len(calls) == 2


def test_run_preflight_retries_buzzasr_then_raises(tmp_path, monkeypatch):
    def always_fail(cmd, check, cwd):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("src.preflight.subprocess.run", always_fail)
    with pytest.raises(RuntimeError, match="BuzzASR"):
        run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                      sample_count=1000, phone_seed=1234,
                      telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                      eval_batch_size=16)


def test_run_preflight_omits_whisper_on_failure(tmp_path, monkeypatch):
    def fake_run(cmd, check, cwd):
        if "whisper" in " ".join(cmd).lower():
            raise subprocess.CalledProcessError(1, cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1]); out.mkdir(parents=True)
        (out / "evaluation_details.json").write_text(json.dumps(_fake_details("BuzzASR/persian")))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert "whisper_v3" not in result["models"]


def test_run_preflight_retries_buzzasr_on_empty_payload(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1]); out.mkdir(parents=True, exist_ok=True)
        payload = _fake_details(cmd[cmd.index("--base_model") + 1])
        if "buzzasr" in " ".join(cmd).lower() and len(calls) == 1:
            payload["metrics"]["total_samples"] = 0
        (out / "evaluation_details.json").write_text(json.dumps(payload))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert len([c for c in calls if "buzzasr" in " ".join(c).lower()]) == 2
    assert result["models"]["buzzasr_base"]["total_samples"] == 1000


def test_run_preflight_omits_whisper_on_empty_payload(tmp_path, monkeypatch):
    def fake_run(cmd, check, cwd):
        out = Path(cmd[cmd.index("--output_dir") + 1]); out.mkdir(parents=True)
        payload = _fake_details(cmd[cmd.index("--base_model") + 1])
        if "whisper" in " ".join(cmd).lower():
            payload["metrics"]["total_samples"] = 0
        (out / "evaluation_details.json").write_text(json.dumps(payload))

    monkeypatch.setattr("src.preflight.subprocess.run", fake_run)
    result = run_preflight(results_dir=tmp_path, dataset_name="d", split="test",
                           sample_count=1000, phone_seed=1234,
                           telephony_profiles=["customer"], telephony_use_ffmpeg=True,
                           eval_batch_size=16)
    assert "whisper_v3" not in result["models"]
    on_disk = json.loads(preflight_path(tmp_path).read_text())
    assert "whisper_v3" not in on_disk["models"]
