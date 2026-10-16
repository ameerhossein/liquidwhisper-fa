"""
Offline tests for training-time preprocessing (src/data.py).
Dataset-level tests need datasets<4 + soundfile (and torch for the thread check);
they skip gracefully when those are absent.
Run: uv run --with numpy --with scipy --with soundfile --with "datasets>=2.19,<4" --with pytest -- python -m pytest tests/test_data_prep.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import PersianASRDataPipeline


class _StubFeatureExtractor:
    def __call__(self, array, sampling_rate):
        return SimpleNamespace(input_features=[np.asarray(array, dtype=np.float32)])


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [2, 3]
        return SimpleNamespace(input_ids=[1, 7, *ids, 0] if add_special_tokens else ids)


class _StubProcessor:
    feature_extractor = _StubFeatureExtractor()
    tokenizer = _StubTokenizer()


class _StubNormalizer:
    def normalize(self, text, **kwargs):
        return text

    def is_colloquial(self, text):
        return False


def _pipeline(num_proc=1, seed=42, prob=1.0):
    return PersianASRDataPipeline(
        processor=_StubProcessor(),
        normalizer=_StubNormalizer(),
        num_proc=num_proc,
        telephony_prob=prob,
        telephony_profiles=["mobile", "pstn"],
        telephony_seed=seed,
        telephony_use_ffmpeg=False,
    )


def _clip(dur=1.0, sr=16000):
    t = np.arange(int(sr * dur)) / float(sr)
    return {"array": (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).tolist(), "sampling_rate": sr}


def test_augmentation_depends_only_on_seed_and_row_index():
    row = {"audio": _clip(), "text": "سلام"}
    pipe = _pipeline()
    pipe.prepare_sample(dict(row), index=9)
    a = pipe.prepare_sample(dict(row), index=5)
    b = _pipeline().prepare_sample(dict(row), index=5)
    np.testing.assert_array_equal(a["input_features"], b["input_features"])


def test_augmentation_differs_across_row_indices():
    row = {"audio": _clip(), "text": "سلام"}
    pipe = _pipeline()
    outs = [pipe.prepare_sample(dict(row), index=i)["input_features"] for i in range(20)]
    assert len({np.asarray(o).tobytes() for o in outs}) == 20


def test_model_native_labels_use_prompt_and_eos_ids():
    pipe = PersianASRDataPipeline(
        processor=_StubProcessor(),
        normalizer=_StubNormalizer(),
        telephony_prob=0.0,
        label_prefix_ids=(51908, 51968, 51972),
        label_eos_id=51865,
    )
    out = pipe.prepare_sample({"audio": _clip(), "text": "سلام"}, index=0)
    assert out["labels"] == [51908, 51968, 51972, 2, 3, 51865]


def test_tokenizer_labels_kept_without_label_format():
    out = _pipeline(prob=0.0).prepare_sample({"audio": _clip(), "text": "سلام"}, index=0)
    assert out["labels"] == [1, 7, 2, 3, 0]


def test_prepare_split_matches_across_num_proc():
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    ds = datasets.Dataset.from_dict({"audio": [_clip() for _ in range(24)], "text": ["سلام"] * 24})
    datasets.disable_caching()
    try:
        single = _pipeline(num_proc=1).prepare_dataset_split(ds)
        multi = _pipeline(num_proc=2).prepare_dataset_split(ds)
    finally:
        datasets.enable_caching()
    assert len(single) == len(multi) == 24
    assert single["input_features"] == multi["input_features"]
    assert len({tuple(f) for f in single["input_features"]}) == 24


def test_prepare_split_restores_torch_threads():
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("soundfile")
    ds = datasets.Dataset.from_dict({"audio": [_clip() for _ in range(8)], "text": ["سلام"] * 8})
    before = torch.get_num_threads()
    datasets.disable_caching()
    try:
        _pipeline(num_proc=2).prepare_dataset_split(ds)
    finally:
        datasets.enable_caching()
    assert torch.get_num_threads() == before
