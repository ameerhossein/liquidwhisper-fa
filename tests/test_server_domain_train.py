import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.app import _normalize_start_config, _build_trainer_command, app


def test_normalize_start_config_domain_adaptation(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    cfg = {
        "domain_adaptation": True,
        "base_model": str(model_dir),
        "custom_data_path": str(custom_dir),
        "replay_samples": 800,
    }
    norm = _normalize_start_config(cfg)
    assert norm["domain_adaptation"] is True
    assert norm["base_model"] == str(model_dir)
    assert norm["custom_data_path"] == str(custom_dir)
    assert norm["replay_samples"] == 800
    assert norm["learning_rate"] == 2e-5
    assert norm["epochs"] == 3
    assert norm["lora_r"] == 16
    assert norm["lora_alpha"] == 32
    assert norm["eval_steps"] == 50
    assert norm["save_steps"] == 50
    assert norm["warmup_steps"] == 25


def test_normalize_start_config_domain_rejects_missing_paths():
    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": "/nonexistent/model",
            "custom_data_path": "/nonexistent/custom",
        })
    assert exc.value.status_code == 400


def test_normalize_start_config_domain_rejects_missing_config_json(tmp_path):
    model_dir = tmp_path / "model_no_json"
    model_dir.mkdir()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
        })
    assert exc.value.status_code == 400


def test_normalize_start_config_domain_replay_samples_out_of_bounds(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    with pytest.raises(HTTPException) as exc_low:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
            "replay_samples": 50,
        })
    assert exc_low.value.status_code == 400

    with pytest.raises(HTTPException) as exc_high:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
            "replay_samples": 6000,
        })
    assert exc_high.value.status_code == 400


def test_normalize_start_config_domain_explicit_overrides(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    cfg = {
        "domain_adaptation": True,
        "base_model": str(model_dir),
        "custom_data_path": str(custom_dir),
        "learning_rate": 5e-5,
        "epochs": 5,
        "lora_r": 32,
        "lora_alpha": 64,
    }
    norm = _normalize_start_config(cfg)
    assert norm["learning_rate"] == 5e-5
    assert norm["epochs"] == 5
    assert norm["lora_r"] == 32
    assert norm["lora_alpha"] == 64


def test_normalize_start_config_domain_default_base_model(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    # When base_model is omitted, it defaults to models/buzzasr-neyshekar-standalone.
    # If that doesn't exist, it should raise 400.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "custom_data_path": str(custom_dir),
        })
    assert exc.value.status_code == 400

    # If the default directory exists with config.json, it should be accepted.
    default_dir = tmp_path / "models" / "buzzasr-neyshekar-standalone"
    default_dir.mkdir(parents=True)
    (default_dir / "config.json").write_text("{}", encoding="utf-8")

    norm = _normalize_start_config({
        "domain_adaptation": True,
        "custom_data_path": str(custom_dir),
    })
    assert norm["base_model"] == "models/buzzasr-neyshekar-standalone"


def test_normalize_start_config_domain_missing_custom_data():
    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "custom_data_path": "",
        })
    assert exc.value.status_code == 400


def test_normalize_start_config_domain_replay_ratio_out_of_bounds(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    with pytest.raises(HTTPException) as exc:
        _normalize_start_config({
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
            "replay_ratio": 1.5,
        })
    assert exc.value.status_code == 400


def test_build_trainer_command_domain_adaptation(tmp_path):
    run_dir = tmp_path / "runs" / "test-run"
    trainer_args = {
        "domain_adaptation": True,
        "base_model": str(tmp_path / "base_model"),
        "custom_data_path": str(tmp_path / "custom_data"),
        "replay_samples": 800,
        "epochs": 3,
        "learning_rate": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "eval_steps": 50,
        "save_steps": 50,
        "warmup_steps": 25,
    }
    cmd = _build_trainer_command(
        trainer_args=trainer_args,
        run_dir=run_dir,
        output_dir=str(run_dir / "output"),
        results_dir=str(run_dir),
        preflight_cache_dir=tmp_path / "preflight",
    )
    assert any("train_domain.py" in part or "dev_fake_trainer.py" in part for part in cmd)
    assert "--base_model" in cmd
    assert str(tmp_path / "base_model") in cmd
    assert "--custom_data_path" in cmd
    assert str(tmp_path / "custom_data") in cmd
    assert "--replay_samples" in cmd
    assert "800" in cmd
    assert "--run_dir" in cmd
    assert str(run_dir) in cmd
    assert "--results_dir" in cmd
    assert str(run_dir) in cmd
    assert "--eval_steps" in cmd
    assert "50" in cmd
    assert "--save_steps" in cmd
    assert "50" in cmd
    assert "--warmup_steps" in cmd
    assert "25" in cmd


def test_start_new_run_domain_adaptation(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ASR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ASR_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("ASR_COCKPIT_TOKEN", raising=False)

    import server.app as mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()

    launched = []

    class _FakeProcess:
        pid = 12345
        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return _FakeProcess()

    monkeypatch.setattr(mod, "_check_audio_backend_or_400", lambda: None)
    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)

    client = TestClient(mod.app)
    response = client.post(
        "/api/train/start",
        json={
            "domain_adaptation": True,
            "base_model": str(model_dir),
            "custom_data_path": str(custom_dir),
            "replay_samples": 500,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "started"
    assert len(launched) == 1
    cmd = launched[0]
    assert any("train_domain.py" in part or "dev_fake_trainer.py" in part for part in cmd)
    assert "--base_model" in cmd
    assert str(model_dir) in cmd
    assert "--custom_data_path" in cmd
    assert str(custom_dir) in cmd
    assert "--replay_samples" in cmd
    assert "500" in cmd


def test_dev_fake_trainer_arg_parsing():
    from scripts.dev_fake_trainer import parse_args
    args = parse_args([
        "--run_dir", "/tmp/fake_run",
        "--base_model", "/tmp/model",
        "--custom_data_path", "/tmp/custom",
        "--replay_samples", "1200",
        "--replay_ratio", "0.3",
    ])
    assert args.run_dir == "/tmp/fake_run"
    assert args.base_model == "/tmp/model"
    assert args.custom_data_path == "/tmp/custom"
    assert args.replay_samples == 1200
    assert args.replay_ratio == 0.3
