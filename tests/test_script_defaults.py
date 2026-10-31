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
    assert args.epochs == 2
    assert args.telephony_mode == "paired"
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
    assert training["epochs"] == 2
    assert training["eval_steps"] == 1000
    assert training["save_steps"] == 1000
    assert training["gradient_checkpointing"] is False
    assert training["max_benchmark_samples"] == 1000
    assert training["preflight_samples"] == 1000
    assert training["lr_scheduler_type"] == "cosine"
    assert cfg["telephony"]["mode"] == "paired"
    assert cfg["telephony"]["prob"] == 0.5


def test_run_train_script_matches_fast_defaults():
    text = (ROOT / "run_train.sh").read_text(encoding="utf-8")
    for expected in (
        "--batch_size 8",
        "--grad_accum 4",
        "--epochs 2",
        "--eval_steps 1000",
        "--save_steps 1000",
        "--max_benchmark_samples 1000",
        "--telephony_mode paired",
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


def test_runs_dir_env_wins(monkeypatch, tmp_path):
    from src import paths

    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    assert paths.resolve_runs_dir(None) == str(tmp_path / "runs")


def test_runs_dir_follows_results_dir(monkeypatch, tmp_path):
    from src import paths

    monkeypatch.delenv("ASR_RUNS_DIR", raising=False)
    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    expected = str(tmp_path / "results" / "runs")
    assert paths.default_runs_dir() == expected
    assert paths.resolve_runs_dir(None) == expected


def test_runs_dir_explicit_argument_wins(monkeypatch, tmp_path):
    from src import paths

    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "env-runs"))
    assert paths.resolve_runs_dir(str(tmp_path / "explicit")) == str(
        tmp_path / "explicit"
    )


def test_cli_run_config_captures_full_resume_fields(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _cli_run_config, parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    config = _cli_run_config(args)
    expected = {
        "base_model",
        "dataset_name",
        "epochs",
        "batch_size",
        "grad_accum",
        "learning_rate",
        "eval_steps",
        "save_steps",
        "max_benchmark_samples",
        "preflight_samples",
        "num_proc",
        "telephony_mode",
        "telephony_prob",
        "gradient_checkpointing",
        "fresh_start",
        "auto_resume",
        "output_dir",
        "results_dir",
        "eval_batch_size",
        "resume_from_checkpoint",
        "telephony_profiles",
        "telephony_seed",
        "telephony_use_ffmpeg",
        "phone_benchmark",
        "phone_seed",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "warmup_steps",
        "lr_scheduler_type",
        "save_total_limit",
        "run_post_test_eval",
        "run_epoch_benchmark",
        "enable_telemetry",
    }
    assert expected <= set(config.keys())
    assert config["lora_r"] == args.lora_r
    assert config["warmup_steps"] == args.warmup_steps
    assert config["telephony_seed"] == args.telephony_seed
    assert config["telephony_mode"] == "paired"


def test_prep_fingerprint_includes_mode(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _prep_fingerprint, parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    paired = _prep_fingerprint(args, "train", None, "paired", 0.5)
    disjoint = _prep_fingerprint(args, "train", None, "disjoint", 0.5)
    clean = _prep_fingerprint(args, "train", None, "clean", 0.0)
    assert paired["telephony_mode"] == "paired"
    assert paired["telephony_prob"] is None
    assert paired != disjoint and disjoint != clean and paired != clean

    args.telephony_mode = "paired"
    validation_a = _prep_fingerprint(args, "validation", None, "clean", 0.0)
    args.telephony_mode = "disjoint"
    validation_b = _prep_fingerprint(args, "validation", None, "clean", 0.0)
    assert validation_a == validation_b


def test_paired_fingerprint_ignores_telephony_prob(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from scripts.train_lora import _prep_fingerprint, parse_args

    monkeypatch.setattr(sys, "argv", ["train_lora.py"])
    args = parse_args()
    assert _prep_fingerprint(args, "train", None, "paired", 0.5) == _prep_fingerprint(
        args, "train", None, "paired", 0.9
    )
