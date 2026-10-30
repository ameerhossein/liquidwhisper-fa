import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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
    path = ROOT / "configs" / "l40s_train_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["telephony"]["profiles"] == ["customer"]


def test_train_lora_parser_matches_fast_l40s_defaults(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    assert args.batch_size == 8
    assert args.grad_accum == 4
    assert args.eval_steps == 1000
    assert args.save_steps == 1000
    assert args.gradient_checkpointing is False
    assert args.max_benchmark_samples == 1000
    assert args.preflight_samples is None
    assert args.telephony_prob == 0.5
    assert args.lr_scheduler_type == "cosine"


def test_train_lora_parser_accepts_scheduler_override(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import parse_args

    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", "--lr_scheduler_type", "linear"]
    )
    assert parse_args().lr_scheduler_type == "linear"


@pytest.mark.parametrize("value", ["-1", "10001"])
def test_preflight_samples_out_of_range_is_rejected(monkeypatch, value):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import parse_args

    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", f"--preflight_samples={value}"]
    )
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("value", ["0", "10000"])
def test_preflight_samples_bounds_are_accepted(monkeypatch, value):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import parse_args

    monkeypatch.setattr(
        sys, "argv", ["train_lora.py", f"--preflight_samples={value}"]
    )
    assert parse_args().preflight_samples == int(value)


def test_l40s_config_matches_fast_defaults():
    import yaml
    path = ROOT / "configs" / "l40s_train_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = cfg["training"]
    assert training["batch_size"] == 8
    assert training["grad_accum"] == 4
    assert training["eval_steps"] == 1000
    assert training["save_steps"] == 1000
    assert training["gradient_checkpointing"] is False
    assert training["max_benchmark_samples"] == 1000
    assert training["preflight_samples"] == 1000
    assert training["lr_scheduler_type"] == "cosine"
    assert cfg["telephony"]["prob"] == 0.5


def test_run_train_script_matches_fast_defaults():
    text = (ROOT / "run_train.sh").read_text(encoding="utf-8")
    for expected in (
        "--batch_size 8",
        "--grad_accum 4",
        "--eval_steps 1000",
        "--save_steps 1000",
        "--max_benchmark_samples 1000",
        "--telephony_prob 0.5",
        "--no-gradient_checkpointing",
    ):
        assert expected in text
    # Preflight follows --max_benchmark_samples instead of duplicating the
    # count, so overriding the benchmark size cannot leave a 1,000-row preflight.
    assert "--preflight_samples" not in text


def test_web_ui_eval_default_is_1000():
    html = (ROOT / "server" / "static" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'id="inputEvalSteps"[^>]*value="(\d+)"', html)
    assert match is not None
    assert match.group(1) == "1000"


def test_no_production_baseline_file_or_fallbacks():
    assert not (ROOT / "configs" / "baselines.json").exists()
    for rel in ("src/trainer_callbacks.py", "server/app.py",
                "server/static/app.js", "scripts/train_lora.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "28.5" not in text and "17.2" not in text
