from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from datasets import Dataset


def test_sample_balanced_neyshekar():
    from src.domain_data import sample_balanced_neyshekar

    # 20 formal rows, 20 informal rows
    data = []
    for i in range(20):
        data.append({"audio": {"path": f"formal_{i}.wav"}, "text": f"متن رسمی شماره {i}", "is_informal": False})
    for i in range(20):
        data.append({"audio": {"path": f"informal_{i}.wav"}, "text": f"می‌خوام برم خونه شماره {i}", "is_informal": True})

    ds = Dataset.from_list(data)

    # Sample 10 total: should get 5 formal and 5 informal
    sampled = sample_balanced_neyshekar(ds, sample_count=10, seed=42)
    assert len(sampled) == 10

    informal_count = sum(1 for row in sampled if row["is_informal"])
    formal_count = len(sampled) - informal_count
    assert informal_count == 5
    assert formal_count == 5


def test_sample_balanced_neyshekar_fallback_colloquial():
    from src.domain_data import sample_balanced_neyshekar

    # Without is_informal column; relies on PersianTextNormalizer.is_colloquial
    data = []
    for i in range(15):
        data.append({"audio": f"f_{i}.wav", "text": f"این یک گزارش رسمی استاندارد است شماره {i}"})
    for i in range(15):
        data.append({"audio": f"inf_{i}.wav", "text": f"می‌خوام الان برم بیرون شماره {i}"})

    ds = Dataset.from_list(data)
    sampled = sample_balanced_neyshekar(ds, sample_count=8, seed=42)
    assert len(sampled) == 8


def test_build_domain_dataset_structure(tmp_path):
    from src.domain_data import build_domain_dataset

    # Mock scan_custom_audio_dir and load_dataset
    mock_records = [{"audio": f"custom_{i}.wav", "text": f"سفارشی {i}", "duration": 2.0} for i in range(20)]
    mock_neyshekar_train = Dataset.from_list([
        {"audio": {"path": f"n_{i}.wav"}, "text": f"متن {i}", "is_informal": (i % 2 == 0)} for i in range(40)
    ])
    mock_neyshekar_val = Dataset.from_list([
        {"audio": {"path": f"nv_{i}.wav"}, "text": f"ارزیابی {i}", "is_informal": False} for i in range(10)
    ])
    mock_neyshekar_test = Dataset.from_list([
        {"audio": {"path": f"nt_{i}.wav"}, "text": f"آزمون {i}", "is_informal": False} for i in range(10)
    ])

    with patch("src.domain_data.scan_custom_audio_dir", return_value=mock_records), \
         patch("src.domain_data.load_dataset") as mock_ld:
        def fake_load_dataset(name, split=None, **kwargs):
            if split == "train":
                return mock_neyshekar_train
            if split == "validation":
                return mock_neyshekar_val
            return mock_neyshekar_test
        mock_ld.side_effect = fake_load_dataset

        out_dir = tmp_path / "domain_dataset_out"
        ds_dict = build_domain_dataset(
            custom_path=str(tmp_path / "custom"),
            replay_samples=10,
            replay_ratio=0.25,
            output_dir=out_dir,
            seed=42,
        )
        assert "train" in ds_dict
        assert "validation" in ds_dict
        assert "test" in ds_dict
        assert out_dir.is_dir()
        assert (out_dir / "dataset_dict.json").exists()


def test_sample_balanced_neyshekar_pool_imbalance():
    from src.domain_data import sample_balanced_neyshekar

    # 15 formal rows, only 2 informal rows
    data = []
    for i in range(15):
        data.append({"audio": f"formal_{i}.wav", "text": f"متن رسمی شماره {i}", "is_informal": False})
    for i in range(2):
        data.append({"audio": f"informal_{i}.wav", "text": f"می‌خوام برم خونه شماره {i}", "is_informal": True})

    ds = Dataset.from_list(data)

    # Request 8 samples: target_informal is 4, but only 2 available, so 2 informal + 6 formal = 8
    sampled = sample_balanced_neyshekar(ds, sample_count=8, seed=42)
    assert len(sampled) == 8

    informal_count = sum(1 for row in sampled if row["is_informal"])
    assert informal_count == 2


def test_sample_balanced_neyshekar_empty_or_zero():
    from src.domain_data import sample_balanced_neyshekar

    data = [{"audio": "a.wav", "text": "تست", "is_informal": False}]
    ds = Dataset.from_list(data)
    assert len(sample_balanced_neyshekar(ds, sample_count=0)) == 0


def test_build_domain_dataset_with_subdirs(tmp_path):
    from src.domain_data import build_domain_dataset

    train_dir = tmp_path / "custom" / "train"
    val_dir = tmp_path / "custom" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    mock_train_records = [{"audio": f"train_{i}.wav", "text": f"آموزش {i}", "duration": 2.0} for i in range(10)]
    mock_val_records = [{"audio": f"val_{i}.wav", "text": f"ارزیابی {i}", "duration": 2.0} for i in range(3)]
    mock_neyshekar = Dataset.from_list([
        {"audio": {"path": f"n_{i}.wav"}, "text": f"متن {i}", "is_informal": False} for i in range(10)
    ])

    def mock_scan(p, **kwargs):
        if Path(p) == train_dir:
            return mock_train_records
        return mock_val_records

    with patch("src.domain_data.scan_custom_audio_dir", side_effect=mock_scan), \
         patch("src.domain_data.load_dataset", return_value=mock_neyshekar):
        ds_dict = build_domain_dataset(
            custom_path=tmp_path / "custom",
            replay_samples=4,
            replay_ratio=0.2,
            seed=42,
        )
        assert len(ds_dict["train"]) > 0
        assert len(ds_dict["validation"]) > 0
        assert len(ds_dict["test"]) == 10


def test_build_domain_dataset_empty_records_raises(tmp_path):
    from src.domain_data import build_domain_dataset

    with patch("src.domain_data.scan_custom_audio_dir", return_value=[]), \
         patch("src.domain_data.load_dataset"):
        with pytest.raises(ValueError, match="No valid custom audio records found"):
            build_domain_dataset(custom_path=tmp_path)

