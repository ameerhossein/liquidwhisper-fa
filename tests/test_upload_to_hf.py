"""Tests for HF model-card metric rendering."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.upload_to_hf import build_model_card, load_metrics_summary


def test_load_metrics_summary_omits_null_deltas(tmp_path):
    report = {
        "total_test_samples": 2149,
        "final_full_test_metrics": {
            "normalized_wer": 14.25,
            "normalized_cer": 4.1,
            "formal_wer": 12.0,
            "informal_wer": 17.0,
            "delta_vs_buzzasr_base": None,
            "delta_phone_vs_buzzasr_base": None,
        },
    }
    (tmp_path / "final_evaluation_report.json").write_text(json.dumps(report))

    summary = load_metrics_summary(tmp_path)

    assert summary["normalized_wer"] == 14.25
    assert "delta_vs_whisper_v3" not in summary
    assert "delta_vs_buzzasr_base" not in summary
    assert "delta_phone_vs_buzzasr_base" not in summary


def test_model_card_renders_missing_deltas_as_na():
    card = build_model_card(
        "adapter",
        "user/repo",
        "BuzzASR/persian",
        "shekar-ai/Neyshekar",
        {"normalized_wer": 14.25, "normalized_cer": 4.1, "total_test_samples": 2149},
    )

    assert "None" not in card
    assert "Δ vs Whisper Large v3" not in card
    assert "- Δ vs BuzzASR base: clean n/a | phone n/a" in card
    assert "WER: 14.25 | CER: 4.1" in card


def test_model_card_renders_present_deltas_in_pp():
    card = build_model_card(
        "merged",
        "user/repo",
        "BuzzASR/persian",
        "shekar-ai/Neyshekar",
        {
            "normalized_wer": 14.25,
            "delta_vs_buzzasr_base": -2.95,
            "delta_phone_vs_buzzasr_base": -4.1,
        },
    )

    assert "clean -2.95 pp" in card
    assert "phone -4.10 pp" in card
    assert "None" not in card
