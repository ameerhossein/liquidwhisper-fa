"""
Offline tests for training-time preprocessing (src/data.py).
Dataset-level tests need datasets<4 + soundfile (and torch for the thread check);
they skip gracefully when those are absent.
Run: uv run --with numpy --with scipy --with soundfile --with "datasets>=2.19,<4" --with pytest -- python -m pytest tests/test_data_prep.py -q
"""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import PersianASRDataPipeline, persian_label_ids


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


def test_informal_supplement_filters_and_adapts_audio(monkeypatch, tmp_path):
    from datasets import Dataset, Audio
    import src.data as data

    def base(text, informal=False):
        return Dataset.from_dict({"audio": [_clip()], "text": [text],
                                  "informal": [informal]}).cast_column("audio", Audio(sampling_rate=16000))

    train = base("رسمی")
    validation = base("آزمون")
    test = base("تست")
    texts = ["میخوام برم", "میخوام برم", "آزمون", "تست", "رسمی", "", "کوتاه", "طولانی", "خراب"]
    waves = [[0.1] * 44100 for _ in texts]
    waves[6] = []
    waves[7] = [0.1] * (44100 * 31)
    waves[8] = [float("nan")] * 44100
    supplement = Dataset.from_dict({"audio": waves, "transcript": texts,
                                    "samplerate": [44100] * len(texts)})
    calls = []

    def load(name, split=None, **kwargs):
        calls.append((name, split))
        if name == data.INFORMAL_DATASET:
            return supplement
        return {"train": train, "validation": validation, "test": test}[split]

    monkeypatch.setattr(data, "load_dataset", load)
    monkeypatch.setattr("src.cache.default_datasets_cache_dir", lambda: str(tmp_path))
    pipe = _pipeline(prob=0)
    result = pipe.load_raw_dataset(split="train")
    assert len(result) == 2
    assert set(result["text"]) == {"رسمی", "میخوام برم"}
    row = next(row for row in result if row["text"] == "میخوام برم")
    assert row["is_informal"] is True
    assert row["audio"]["sampling_rate"] == 16000
    assert len(row["audio"]["array"]) == 16000
    assert result["is_informal"][result["text"].index("رسمی")] is False
    assert (data.INFORMAL_DATASET, "train") in calls

    calls.clear()
    held_out = pipe.load_raw_dataset(split="validation")
    assert held_out["text"] == ["آزمون"]
    assert calls == [("shekar-ai/Neyshekar", "validation")]

    # A fully rejected supplement must retain a valid training schema.
    supplement = supplement.select([2, 3, 4])
    result = pipe.load_raw_dataset(split="train")
    assert result["text"] == ["رسمی"]
    assert result["is_informal"] == [False]


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


def test_persian_label_ids_read_from_generation_config():
    gen_cfg = {
        "lang_to_id": {"<|fa|>": 51908},
        "task_to_id": {"transcribe": 51968, "translate": 51967},
        "no_timestamps_token_id": 51972,
        "eos_token_id": 51865,
    }
    assert persian_label_ids(gen_cfg) == ((51908, 51968, 51972), 51865)


def test_tokenizer_labels_kept_without_label_format():
    out = _pipeline(prob=0.0).prepare_sample({"audio": _clip(), "text": "سلام"}, index=0)
    assert out["labels"] == [1, 7, 2, 3, 0]


def test_tokenizer_fallback_sets_persian_but_native_label_ids_bypass_it():
    calls = []

    class Tokenizer(_StubTokenizer):
        def set_prefix_tokens(self, **kwargs):
            calls.append(kwargs)

    processor = SimpleNamespace(feature_extractor=_StubFeatureExtractor(), tokenizer=Tokenizer())
    PersianASRDataPipeline(processor=processor, normalizer=_StubNormalizer())
    assert calls == [{"language": "fa", "task": "transcribe"}]
    calls.clear()
    PersianASRDataPipeline(
        processor=processor, normalizer=_StubNormalizer(),
        label_prefix_ids=(51908, 51968, 51972), label_eos_id=51865,
    )
    assert calls == []


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


def _pairing_dataset(n=6):
    datasets = pytest.importorskip("datasets")
    return datasets.Dataset.from_dict({
        "audio": [_clip() for _ in range(n)],
        "text": [f"سلام {i}" for i in range(n)],
    })


def test_prepare_paired_dataset_pairs_each_utterance():
    ds = _pairing_dataset(6)
    out = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    assert len(out) == 12
    assert out["labels"][:6] == out["labels"][6:]
    assert out["is_informal"][:6] == out["is_informal"][6:]
    assert any(
        np.asarray(a).tobytes() != np.asarray(b).tobytes()
        for a, b in zip(out["input_features"][:6], out["input_features"][6:])
    )


def test_prepare_paired_dataset_clean_half_matches_clean_pipeline():
    ds = _pairing_dataset(4)
    paired = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    clean = _pipeline(prob=0.0).prepare_dataset_split(ds)
    for a, b in zip(paired["input_features"][:4], clean["input_features"]):
        np.testing.assert_array_equal(a, b)


def test_prepare_paired_dataset_deterministic_across_num_proc():
    datasets = pytest.importorskip("datasets")
    ds = _pairing_dataset(8)
    datasets.disable_caching()
    try:
        a = _pipeline(num_proc=1).prepare_paired_dataset(ds)
        b = _pipeline(num_proc=2).prepare_paired_dataset(ds)
    finally:
        datasets.enable_caching()
    for x, y in zip(a["input_features"][8:], b["input_features"][8:]):
        np.testing.assert_array_equal(x, y)


def test_prepare_paired_dataset_filters_once():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({
        "audio": [_clip(dur=0.1), _clip(dur=1.0), _clip(dur=1.0)],
        "text": ["a", "b", "c"],
    })
    out = _pipeline(prob=0.5).prepare_paired_dataset(ds)
    assert len(out) == 4


def test_prepare_paired_dataset_requires_telephone(monkeypatch):
    ds = _pairing_dataset(2)
    monkeypatch.setattr("src.data._TELEPHONE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="telephone"):
        _pipeline(prob=0.5).prepare_paired_dataset(ds)


def test_prepare_paired_dataset_fails_closed_on_degradation_error(monkeypatch):
    ds = _pairing_dataset(2)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated codec failure")

    monkeypatch.setattr("src.data.apply_random_telephone", boom)
    with pytest.raises(RuntimeError, match="simulated codec failure"):
        _pipeline(prob=0.5).prepare_paired_dataset(ds)


class _MismatchedPipeline(PersianASRDataPipeline):
    def prepare_sample(self, batch, index=None):
        out = super().prepare_sample(batch, index=index)
        if self.telephony_prob >= 1.0:
            out["input_features"] = "not-a-feature"
        return out


def test_prepare_paired_dataset_rejects_schema_mismatch():
    pipe = _MismatchedPipeline(
        processor=_StubProcessor(),
        normalizer=_StubNormalizer(),
        telephony_prob=0.5,
        telephony_profiles=["mobile"],
        telephony_use_ffmpeg=False,
    )
    with pytest.raises(ValueError, match="schema"):
        pipe.prepare_paired_dataset(_pairing_dataset(2))


def test_prepare_dataset_split_writes_cache_dir(tmp_path):
    ds = _pairing_dataset(4)
    out = _pipeline(prob=0.0, num_proc=1).prepare_dataset_split(
        ds, cache_dir=str(tmp_path / "stage")
    )
    files = [Path(c["filename"]).resolve() for c in out.cache_files]
    assert files
    assert all(tmp_path.resolve() in f.parents for f in files)
    assert all(f.is_file() for f in files)


def test_prepare_paired_dataset_writes_clean_and_phone_shards(tmp_path):
    ds = _pairing_dataset(4)
    out = _pipeline(prob=0.5, num_proc=1).prepare_paired_dataset(
        ds, cache_dir=str(tmp_path / "stage")
    )
    names = [Path(c["filename"]).name for c in out.cache_files]
    assert any(n.startswith("clean") for n in names)
    assert any(n.startswith("phone") for n in names)
    assert len(out) == 8


def test_prepare_paired_dataset_releases_clean_before_phone(monkeypatch, tmp_path):
    """The clean half must not stay memory-mapped while the phone half runs.

    Keeping the clean Dataset alive pins its Arrow pages in the cgroup, so a
    paired prep on a 188 GB pod dies partway through the phone map.
    """
    pipe = _pipeline(prob=0.5, num_proc=1)
    clean_ref = {"ref": None}
    orig_prepare = pipe._prepare_split

    def spy(dataset, remove_columns, effective_proc, filter_durations=True,
            cache_file_name=None):
        out = orig_prepare(
            dataset,
            remove_columns,
            effective_proc,
            filter_durations=filter_durations,
            cache_file_name=cache_file_name,
        )
        name = Path(cache_file_name).name if cache_file_name else ""
        if name.startswith("clean"):
            clean_ref["ref"] = weakref.ref(out)
        elif name.startswith("phone"):
            assert clean_ref["ref"] is not None
            assert clean_ref["ref"]() is None, (
                "clean Dataset was still alive during the phone map"
            )
        return out

    monkeypatch.setattr(pipe, "_prepare_split", spy)
    out = pipe.prepare_paired_dataset(
        _pairing_dataset(4), cache_dir=str(tmp_path / "stage")
    )
    assert len(out) == 8
    assert out["labels"][:4] == out["labels"][4:]
