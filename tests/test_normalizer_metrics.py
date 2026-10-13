"""
Unit tests for Persian normalizer and metrics module.
Can run locally on any CPU without external network access.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalizer import PersianTextNormalizer
from src.metrics import ASRMetricsCalculator


def test_character_standardization():
    norm = PersianTextNormalizer(use_hazm=False)
    # Arabic kaf, yeh, tatweel
    arabic_text = "كتاب ياسمين كـار"
    clean = norm.standardize_characters(arabic_text)
    assert "ک" in clean, f"Expected Persian kaf, got: {clean}"
    assert "ی" in clean, f"Expected Persian yeh, got: {clean}"
    assert "ـ" not in clean, f"Expected tatweel stripped, got: {clean}"
    print("✓ Character standardization passed.")


def test_zwnj_handling():
    norm = PersianTextNormalizer(use_hazm=False)
    text_with_zwnj = "می\u200cروم به مدرسه"
    
    kept = norm.normalize_zwnj(text_with_zwnj, mode="keep")
    assert "\u200c" in kept, "ZWNJ should be preserved in 'keep' mode"

    spaced = norm.normalize_zwnj(text_with_zwnj, mode="to_space")
    assert "\u200c" not in spaced and "می روم" in spaced, "ZWNJ should convert to space"

    removed = norm.normalize_zwnj(text_with_zwnj, mode="remove")
    assert "\u200c" not in removed and "میروم" in removed, "ZWNJ should be stripped"
    print("✓ ZWNJ handling passed.")


def test_colloquial_detection():
    norm = PersianTextNormalizer(use_hazm=False)
    assert norm.is_colloquial("من الان می‌خوام برم خونه") == True
    assert norm.is_colloquial("واسه چی دیر اومدی؟") == True
    assert norm.is_colloquial("امروز جلسه‌ای در دانشگاه برگزار شد") == False
    print("✓ Colloquial detection passed.")


def test_metrics_evaluation():
    norm = PersianTextNormalizer(use_hazm=False)
    calc = ASRMetricsCalculator(normalizer=norm)

    refs = [
        "من به مدرسه می‌روم",
        "امروز هوا بارانی است",
        "می‌خوام برم خونه واسه استراحت"
    ]
    preds = [
        "من به مدرسه میروم",
        "امروز هوا بارانی است",
        "میخوام برم خونه واسه خواب"
    ]

    results = calc.evaluate_predictions(refs, preds)
    assert "overall" in results
    assert "normalized_wer" in results["overall"]
    assert "formal" in results
    assert "informal" in results
    print(calc.format_results_table("Test-Model", results))
    print("✓ Metrics calculation passed.")


if __name__ == "__main__":
    test_character_standardization()
    test_zwnj_handling()
    test_colloquial_detection()
    test_metrics_evaluation()
    print("\nAll unit tests passed successfully!")
