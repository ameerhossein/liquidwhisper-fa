import pytest


def test_trainer_default_profiles_customer():
    from src.trainer_callbacks import DEFAULT_TELEPHONY_PROFILES
    assert DEFAULT_TELEPHONY_PROFILES == ["customer"]


def test_train_lora_parser_defaults_to_customer():
    pytest.importorskip("torch")
    from scripts.train_lora import _parse_telephony_profiles
    assert _parse_telephony_profiles(None) == ["customer"]


def test_evaluate_model_parser_defaults_to_customer():
    from scripts.evaluate_model import _parse_profiles
    assert _parse_profiles("") == ["customer"]
