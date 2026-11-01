import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide mock fallback for optional ML dependencies if running under minimal/system Python.
for _mod in ("torch", "transformers", "peft", "evaluate", "soundfile", "huggingface_hub"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

if "datasets" not in sys.modules:
    try:
        import datasets
    except ImportError:
        _mock_datasets = MagicMock()
        _mock_datasets.DatasetDict = dict
        _mock_datasets.Dataset = object
        sys.modules["datasets"] = _mock_datasets


def test_label_format_reads_local_generation_config(tmp_path):
    from scripts.train_lora import _label_format
    
    # Create mock local model directory with generation_config.json
    gen_cfg = {
        "lang_to_id": {"<|fa|>": 51908},
        "task_to_id": {"transcribe": 51968},
        "no_timestamps_token_id": 51972,
        "eos_token_id": 51865,
    }
    model_dir = tmp_path / "mock_merged_model"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(json.dumps(gen_cfg), encoding="utf-8")
    
    prefix_ids, eos_id = _label_format(str(model_dir))
    assert prefix_ids == (51908, 51968, 51972)
    assert eos_id == 51865


def test_label_format_nonexistent_returns_none():
    from scripts.train_lora import _label_format
    prefix_ids, eos_id = _label_format("/nonexistent/model/path")
    assert prefix_ids is None
    assert eos_id is None


def test_label_format_local_dir_without_generation_config_returns_none(tmp_path):
    from scripts.train_lora import _label_format
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()
    prefix_ids, eos_id = _label_format(str(empty_dir))
    assert prefix_ids is None
    assert eos_id is None


def test_label_format_local_dir_invalid_json_returns_none(tmp_path):
    from scripts.train_lora import _label_format
    model_dir = tmp_path / "corrupt_model"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text("invalid json {", encoding="utf-8")
    prefix_ids, eos_id = _label_format(str(model_dir))
    assert prefix_ids is None
    assert eos_id is None


def test_label_format_hub_persian(tmp_path):
    from scripts.train_lora import _label_format
    gen_cfg = {
        "lang_to_id": {"<|fa|>": 51908},
        "task_to_id": {"transcribe": 51968},
        "no_timestamps_token_id": 51972,
        "eos_token_id": 51865,
    }
    cfg_file = tmp_path / "generation_config.json"
    cfg_file.write_text(json.dumps(gen_cfg), encoding="utf-8")
    with patch("huggingface_hub.hf_hub_download", return_value=str(cfg_file)):
        prefix_ids, eos_id = _label_format("BuzzASR/persian")
        assert prefix_ids == (51908, 51968, 51972)
        assert eos_id == 51865


def test_load_raw_dataset_branches_to_load_from_disk(tmp_path):
    from src.data import PersianASRDataPipeline
    
    dataset_dir = tmp_path / "mock_dataset_dir"
    dataset_dir.mkdir()
    
    pipeline = PersianASRDataPipeline(processor=MagicMock(), dataset_name=str(dataset_dir))
    
    with patch("src.data.load_from_disk") as mock_load_from_disk:
        mock_train = MagicMock()
        mock_train.cast_column.return_value = mock_train
        mock_ds_dict = {"train": mock_train, "validation": MagicMock()}
        mock_load_from_disk.return_value = mock_ds_dict
        
        res = pipeline.load_raw_dataset(split="train")
        mock_load_from_disk.assert_called_once_with(str(dataset_dir))
        assert res == mock_ds_dict["train"]


def test_load_raw_dataset_branches_to_load_from_disk_no_split(tmp_path):
    from src.data import PersianASRDataPipeline
    
    dataset_dir = tmp_path / "mock_dataset_dir"
    dataset_dir.mkdir()
    
    pipeline = PersianASRDataPipeline(processor=MagicMock(), dataset_name=str(dataset_dir))
    
    with patch("src.data.load_from_disk") as mock_load_from_disk:
        mock_ds = MagicMock()
        mock_ds.cast_column.return_value = mock_ds
        mock_load_from_disk.return_value = mock_ds
        
        res = pipeline.load_raw_dataset(split=None)
        mock_load_from_disk.assert_called_once_with(str(dataset_dir))
        assert res == mock_ds


def test_load_raw_dataset_missing_datasets_raises_import_error(tmp_path):
    from src.data import PersianASRDataPipeline
    import src.data as data_mod
    
    dataset_dir = tmp_path / "mock_dataset_dir"
    dataset_dir.mkdir()
    
    pipeline = PersianASRDataPipeline(processor=MagicMock(), dataset_name=str(dataset_dir))
    
    with patch.object(data_mod, "load_from_disk", None):
        with pytest.raises(ImportError, match="datasets package not installed"):
            pipeline.load_raw_dataset(split="train")
