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
    # Formal words must NOT trigger colloquial (regression for over-broad regex).
    assert norm.is_colloquial("چرا آمدی؟") == False, "چرا is formal"
    assert norm.is_colloquial("اینجا دانشگاه است") == False, "اینجا is formal"
    assert norm.is_colloquial("چطور این اتفاق افتاد؟") == False, "چطور is formal"
    assert norm.is_colloquial("اونا رفتند") == True, "اونا is colloquial"
    print("✓ Colloquial detection passed.")


def test_zwnj_edges():
    norm = PersianTextNormalizer(use_hazm=False)
    # Duplicate ZWNJ collapses, boundary ZWNJ stripped, spaced ZWNJ -> space.
    assert norm.normalize_zwnj("می\u200c\u200cروم", mode="keep") == "می\u200cروم"
    assert norm.normalize_zwnj("\u200cسلام\u200c", mode="keep") == "سلام"
    assert norm.normalize_zwnj("می \u200cروم", mode="keep") == "می روم"
    assert norm.normalize("  سلام   دنیا  ", zwnj_mode="keep") == "سلام دنیا"
    assert norm.normalize("", zwnj_mode="keep") == ""
    print("✓ ZWNJ edge cases passed.")


def test_ezafe_preservation():
    norm = PersianTextNormalizer(use_hazm=False)
    # "ۀ" must not be silently mapped to "ه" (would delete ezafe).
    assert "ۀ" not in norm.arabic_to_persian_chars, "ۀ mapping removed; Hazm handles ezafe"
    print("✓ Ezafe preservation passed.")


def test_metrics_empty_and_single():
    norm = PersianTextNormalizer(use_hazm=False)
    calc = ASRMetricsCalculator(normalizer=norm)
    assert calc.compute_wer([], []) == 0.0
    assert calc.compute_cer([], []) == 0.0
    assert calc.evaluate_predictions([], []) == {"total_samples": 0}
    # Single perfect + single error.
    assert calc.compute_wer(["سلام دنیا"], ["سلام دنیا"]) == 0.0
    assert calc.compute_wer(["سلام دنیا"], ["سلام"]) > 0.0
    print("✓ Empty/single metrics passed.")


def test_collator_mask():
    try:
        import torch
    except ImportError:
        print("⊘ Collator test skipped (torch not installed).")
        return
    from src.collator import DataCollatorSpeechSeq2SeqWithPadding

    class FakeTokenizer:
        pad_token_id = 0
        bos_token_id = 999  # never matches, so no trim

        def pad(self, features, return_tensors=None):
            import torch as _t
            ids = [f["input_ids"] for f in features]
            maxlen = max(len(x) for x in ids)
            padded, mask = [], []
            for x in ids:
                padlen = maxlen - len(x)
                padded.append(list(x) + [self.pad_token_id] * padlen)
                mask.append([1] * len(x) + [0] * padlen)
            batch = {"input_ids": _t.tensor(padded, dtype=_t.long)}
            # mimic BatchEncoding.attention_mask attribute access
            batch["attention_mask"] = _t.tensor(mask, dtype=_t.long)
            class B(dict):
                @property
                def attention_mask(self):
                    return self["attention_mask"]
            return B(batch)

    class FakeFeat:
        def pad(self, features, return_tensors=None):
            import torch as _t
            import numpy as _np
            feats = [_np.asarray(f["input_features"]) for f in features]
            maxlen = max(f.shape[0] for f in feats)
            dim = feats[0].shape[1] if feats[0].ndim > 1 else 1
            out = _np.zeros((len(feats), maxlen, dim), dtype=_np.float32)
            for i, f in enumerate(feats):
                out[i, : f.shape[0]] = f.reshape(-1, dim)
            return {"input_features": _t.tensor(out)}

    class FakeProc:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.feature_extractor = FakeFeat()

    import numpy as np
    coll = DataCollatorSpeechSeq2SeqWithPadding(processor=FakeProc())
    batch = coll([
        {"input_features": np.ones((10, 4), dtype=np.float32), "labels": [1, 2, 3]},
        {"input_features": np.ones((6, 4), dtype=np.float32), "labels": [1, 2]},
    ])
    assert batch["labels"].shape == (2, 3), f"unexpected labels shape {batch['labels'].shape}"
    assert int(batch["labels"][1, 2]) == -100, "padding must be -100"
    print("✓ Collator -100 mask passed.")


def test_duration_helpers():
    from src.data import PersianASRDataPipeline
    pipe = PersianASRDataPipeline(
        processor=None, max_duration_seconds=30.0, min_duration_seconds=0.5
    )
    short = {"array": [0.0] * 3200, "sampling_rate": 16000}  # 0.2s
    ok = {"array": [0.0] * 80000, "sampling_rate": 16000}  # 5s
    long = {"array": list(range(16000 * 35)), "sampling_rate": 16000}  # 35s
    assert pipe.is_valid_duration(short) is False
    assert pipe.is_valid_duration(ok) is True
    assert pipe.is_valid_duration(long) is True
    trunc = pipe._truncate_audio(dict(long))
    assert len(trunc["array"]) == 16000 * 30, f"truncate to 30s, got {len(trunc['array'])}"
    print("✓ Duration filter/truncate passed.")


def test_baselines_config():
    import json
    p = Path(__file__).resolve().parent.parent / "configs" / "baselines.json"
    assert p.exists(), "configs/baselines.json must exist"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "whisper_v3" in data and "buzzasr_base" in data
    assert data["whisper_v3"]["normalized_wer"] > data["buzzasr_base"]["normalized_wer"]
    print("✓ Baselines config passed.")


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
    test_zwnj_edges()
    test_ezafe_preservation()
    test_metrics_empty_and_single()
    test_metrics_evaluation()
    test_collator_mask()
    test_duration_helpers()
    test_baselines_config()
    print("\nAll unit tests passed successfully!")
