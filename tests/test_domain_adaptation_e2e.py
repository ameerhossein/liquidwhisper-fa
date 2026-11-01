import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import wave

import pytest
from datasets import Audio, Dataset, load_from_disk
from fastapi.testclient import TestClient

from scripts.train_domain import (
    main,
    run_domain_adaptation,
    validate_inputs,
)
from scripts.train_lora import _label_format
from src.data import PersianASRDataPipeline
from src.domain_data import build_domain_dataset


def _create_mock_wav(path: Path, duration_seconds: float = 2.0, sample_rate: int = 16000) -> None:
    """Create a valid PCM mono 16-bit WAV audio file of specified duration."""
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)


def _setup_mock_model(model_dir: Path) -> Path:
    """Populate a mock merged standalone model directory with config, generation_config, and weights."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "generation_config.json").write_text(
        json.dumps({
            "lang_to_id": {"<|fa|>": 51908},
            "task_to_id": {"transcribe": 51968},
            "no_timestamps_token_id": 51972,
            "eos_token_id": 51865,
        }),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    return model_dir


def _setup_mock_custom_dataset(custom_dir: Path, count: int = 8) -> Path:
    """Populate a mock custom audio directory with paired WAV and Persian transcript TXT files."""
    custom_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        wav_path = custom_dir / f"sample{i}.wav"
        txt_path = custom_dir / f"sample{i}.txt"
        _create_mock_wav(wav_path, duration_seconds=1.5 + (i * 0.2))
        txt_path.write_text(f"اصطلاح تخصصی حوزه کاری شماره {i}", encoding="utf-8")
    return custom_dir


def _create_mock_neyshekar_dataset(sample_wav_path: Path, count: int = 20) -> Dataset:
    """Create a mock Neyshekar dataset with formal and informal utterances and Audio feature."""
    records = []
    for i in range(count):
        records.append({
            "audio": str(sample_wav_path),
            "text": f"متن نیشکر نمونه {i}",
            "is_informal": (i % 2 == 1),
        })
    ds = Dataset.from_list(records)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def test_domain_adaptation_end_to_end(tmp_path):
    """End-to-end integration test for domain adaptation workflow.

    Verifies:
    1. Base model validation & generation_config token extraction.
    2. Custom dataset directory scanning and duration filtering.
    3. Domain dataset building with balanced Neyshekar replay interleaving.
    4. Dataset serialization to disk and deserialization via datasets.load_from_disk.
    5. Single-command orchestrator execution simulation and CLI flag assembly.
    """
    # 1. Setup mock merged model
    model_dir = tmp_path / "mock_merged"
    _setup_mock_model(model_dir)

    # 2. Setup mock custom dataset
    custom_dir = tmp_path / "mock_custom"
    _setup_mock_custom_dataset(custom_dir, count=6)

    # 3. Test CLI validate_inputs succeeds
    validate_inputs(base_model=str(model_dir), custom_data_path=str(custom_dir))

    # 4. Verify generation_config label format extraction
    prefix_ids, eos_id = _label_format(str(model_dir))
    assert prefix_ids == (51908, 51968, 51972)
    assert eos_id == 51865

    # 5. Build domain dataset end-to-end with mock Neyshekar replay
    output_dir = tmp_path / "outputs" / "buzzasr-domain-lora"
    results_dir = tmp_path / "results" / "domain-lora"
    prepared_ds_dir = output_dir / "prepared_domain_dataset"

    mock_neyshekar = _create_mock_neyshekar_dataset(custom_dir / "sample1.wav", count=20)

    with patch("src.domain_data.load_dataset", return_value=mock_neyshekar), \
         patch("scripts.train_domain._run_training_subprocess") as mock_subprocess:

        mock_subprocess.return_value = 0

        exit_code = run_domain_adaptation(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            replay_samples=10,
            replay_ratio=0.25,
            epochs=3,
            learning_rate=2e-5,
            lora_r=16,
            lora_alpha=32,
            batch_size=8,
            grad_accum=4,
            telephony_mode="paired",
        )

        assert exit_code == 0
        mock_subprocess.assert_called_once()

        # Check command assembly
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        cwd = call_args[1].get("cwd")

        assert str(Path("scripts/train_lora.py")) in " ".join(cmd)
        assert "--base_model" in cmd
        assert str(model_dir) in cmd
        assert "--dataset_name" in cmd
        assert str(prepared_ds_dir) in cmd
        assert "--output_dir" in cmd
        assert str(output_dir) in cmd
        assert "--results_dir" in cmd
        assert str(results_dir) in cmd
        assert "--epochs" in cmd
        assert "3" in cmd
        assert "--learning_rate" in cmd
        assert "2e-05" in cmd or "2e-5" in cmd
        assert "--lora_r" in cmd
        assert "16" in cmd
        assert "--lora_alpha" in cmd
        assert "32" in cmd
        assert "--batch_size" in cmd
        assert "8" in cmd
        assert "--grad_accum" in cmd
        assert "4" in cmd
        assert "--telephony_mode" in cmd
        assert "paired" in cmd

        # Verify prepared dataset was saved to disk and can be loaded
        assert prepared_ds_dir.is_dir()
        assert (prepared_ds_dir / "dataset_dict.json").is_file()

        disk_ds = load_from_disk(str(prepared_ds_dir))
        assert "train" in disk_ds
        assert "validation" in disk_ds
        assert "test" in disk_ds
        assert len(disk_ds["train"]) > 0
        assert len(disk_ds["validation"]) > 0
        assert len(disk_ds["test"]) > 0

        # Features validation
        for split_name in ("train", "validation", "test"):
            feat = disk_ds[split_name].features["audio"]
            assert isinstance(feat, Audio)
            assert feat.sampling_rate == 16000


def test_domain_adaptation_e2e_cli_entrypoint(tmp_path):
    """Verify that train_domain CLI main entrypoint works end-to-end with arguments."""
    model_dir = tmp_path / "mock_merged"
    _setup_mock_model(model_dir)

    custom_dir = tmp_path / "mock_custom"
    _setup_mock_custom_dataset(custom_dir, count=4)

    mock_neyshekar = _create_mock_neyshekar_dataset(custom_dir / "sample1.wav", count=10)
    output_dir = tmp_path / "out_cli"
    results_dir = tmp_path / "res_cli"

    cli_argv = [
        "--base_model", str(model_dir),
        "--custom_data_path", str(custom_dir),
        "--output_dir", str(output_dir),
        "--results_dir", str(results_dir),
        "--replay_samples", "8",
        "--replay_ratio", "0.2",
        "--epochs", "2",
        "--learning_rate", "3e-5",
    ]

    with patch("src.domain_data.load_dataset", return_value=mock_neyshekar), \
         patch("scripts.train_domain._run_training_subprocess", return_value=0) as mock_subp:
        rc = main(cli_argv)
        assert rc == 0
        mock_subp.assert_called_once()
        cmd = mock_subp.call_args[0][0]
        assert "--base_model" in cmd
        assert str(model_dir) in cmd
        assert "--epochs" in cmd
        assert "2" in cmd
        assert "--learning_rate" in cmd
        assert "3e-05" in cmd or "3e-5" in cmd


def test_domain_adaptation_e2e_data_pipeline_load(tmp_path):
    """Verify PersianASRDataPipeline can load the prepared domain dataset from disk."""
    custom_dir = tmp_path / "mock_custom"
    _setup_mock_custom_dataset(custom_dir, count=4)

    mock_neyshekar = _create_mock_neyshekar_dataset(custom_dir / "sample1.wav", count=10)
    dataset_output = tmp_path / "prepared_dataset"

    with patch("src.domain_data.load_dataset", return_value=mock_neyshekar):
        build_domain_dataset(
            custom_path=str(custom_dir),
            replay_samples=4,
            replay_ratio=0.25,
            output_dir=dataset_output,
            seed=42,
        )

    # Initialize data pipeline with dataset_name pointing to local directory
    mock_processor = MagicMock()
    pipeline = PersianASRDataPipeline(processor=mock_processor, dataset_name=str(dataset_output))

    train_ds = pipeline.load_raw_dataset(split="train")
    val_ds = pipeline.load_raw_dataset(split="validation")

    assert train_ds is not None
    assert val_ds is not None
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert "audio" in train_ds.column_names
    assert "text" in train_ds.column_names


def test_domain_adaptation_e2e_server_workflow(tmp_path, monkeypatch):
    """Verify end-to-end Cockpit WebUI backend flow starting a domain adaptation run."""
    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ASR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ASR_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ASR_COCKPIT_TOKEN", raising=False)

    import server.app as server_mod

    model_dir = tmp_path / "mock_merged"
    _setup_mock_model(model_dir)

    custom_dir = tmp_path / "mock_custom"
    _setup_mock_custom_dataset(custom_dir, count=4)

    launched_cmds = []

    class _FakeProcess:
        pid = 43210
        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        launched_cmds.append(cmd)
        return _FakeProcess()

    monkeypatch.setattr(server_mod, "training_process", None)
    monkeypatch.setattr(server_mod, "smoke_process", None)
    monkeypatch.setattr(server_mod, "_active_run", lambda: None)
    monkeypatch.setattr(server_mod, "_check_audio_backend_or_400", lambda: None)
    monkeypatch.setattr(server_mod.subprocess, "Popen", _fake_popen)

    client = TestClient(server_mod.app)
    resp = client.post(
        "/api/train/start",
        json={
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
            "replay_samples": 600,
            "replay_ratio": 0.3,
            "epochs": 3,
            "learning_rate": 2e-5,
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "started"
    assert "run_id" in data

    assert len(launched_cmds) == 1
    cmd = launched_cmds[0]
    assert any("train_domain.py" in part or "dev_fake_trainer.py" in part for part in cmd)
    assert "--base_model" in cmd
    assert str(model_dir) in cmd
    assert "--custom_data_path" in cmd
    assert str(custom_dir) in cmd
    assert "--replay_samples" in cmd
    assert "600" in cmd


def test_domain_adaptation_e2e_invalid_inputs_error_handling(tmp_path):
    """Verify input validation fails cleanly on invalid inputs before running."""
    # 1. Nonexistent base model
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    with pytest.raises(ValueError, match="Base model directory does not exist"):
        validate_inputs(base_model="/nonexistent/model", custom_data_path=str(custom_dir))

    # 2. Base model missing config.json
    invalid_model = tmp_path / "model_no_config"
    invalid_model.mkdir()
    with pytest.raises(ValueError, match="missing config.json"):
        validate_inputs(base_model=str(invalid_model), custom_data_path=str(custom_dir))

    # 3. Nonexistent custom data directory
    valid_model = tmp_path / "valid_model"
    _setup_mock_model(valid_model)
    with pytest.raises(ValueError, match="Custom data directory does not exist"):
        validate_inputs(base_model=str(valid_model), custom_data_path="/nonexistent/custom")

    # 4. Custom data directory with no valid audio files
    empty_custom = tmp_path / "empty_custom"
    empty_custom.mkdir()
    with pytest.raises(ValueError, match="No valid custom audio records found"):
        build_domain_dataset(
            custom_path=str(empty_custom),
            replay_samples=10,
            replay_ratio=0.25,
            seed=42,
        )
