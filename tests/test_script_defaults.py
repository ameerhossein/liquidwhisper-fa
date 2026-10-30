from pathlib import Path

import pytest


def test_trainer_default_profiles_customer():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.trainer_callbacks import DEFAULT_TELEPHONY_PROFILES
    assert DEFAULT_TELEPHONY_PROFILES == ["customer"]


def test_train_lora_parser_defaults_to_customer():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _parse_telephony_profiles
    assert _parse_telephony_profiles(None) == ["customer"]


def test_evaluate_model_parser_defaults_to_customer():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts.evaluate_model import _parse_profiles
    assert _parse_profiles("") == ["customer"]


def test_l40s_config_uses_customer_profile():
    import yaml
    path = Path(__file__).resolve().parent.parent / "configs" / "l40s_train_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["telephony"]["profiles"] == ["customer"]
