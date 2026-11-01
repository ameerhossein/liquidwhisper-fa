import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_train_domain_cli_validates_base_model(tmp_path):
    from scripts.train_domain import parse_args, validate_inputs

    # Missing base model should raise SystemExit or ValueError
    with pytest.raises((SystemExit, ValueError)):
        validate_inputs(
            base_model="/nonexistent/model",
            custom_data_path=str(tmp_path),
        )


def test_train_domain_cli_validates_custom_path(tmp_path):
    from scripts.train_domain import validate_inputs

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises((SystemExit, ValueError)):
        validate_inputs(
            base_model=str(model_dir),
            custom_data_path="/nonexistent/custom/path",
        )


def test_train_domain_cli_orchestration(tmp_path):
    from scripts.train_domain import run_domain_adaptation

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    with patch("scripts.train_domain.build_domain_dataset") as mock_build, \
         patch("scripts.train_domain._run_training_subprocess") as mock_run:

        run_domain_adaptation(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
            output_dir=str(tmp_path / "out"),
            results_dir=str(tmp_path / "res"),
            replay_samples=500,
            replay_ratio=0.25,
            epochs=3,
            learning_rate=2e-5,
            lora_r=16,
            lora_alpha=32,
        )
        mock_build.assert_called_once()
        mock_run.assert_called_once()


def test_train_domain_cli_validates_missing_config_json(tmp_path):
    from scripts.train_domain import validate_inputs

    model_dir = tmp_path / "model_without_config"
    model_dir.mkdir()
    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    with pytest.raises((SystemExit, ValueError)) as exc_info:
        validate_inputs(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
        )
    assert "config.json" in str(exc_info.value)


def test_train_domain_cli_parse_args():
    from scripts.train_domain import parse_args

    args = parse_args([
        "--base_model", "models/standalone",
        "--custom_data_path", "data/custom",
    ])
    assert args.base_model == "models/standalone"
    assert args.custom_data_path == "data/custom"
    assert args.output_dir == "outputs/buzzasr-domain-lora"
    assert args.results_dir == "results/domain-lora"
    assert args.replay_samples == 1000
    assert args.replay_ratio == 0.25
    assert args.epochs == 3
    assert args.learning_rate == 2e-5
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.batch_size == 8
    assert args.grad_accum == 4
    assert args.telephony_mode == "paired"
    assert args.fresh_start is False


def test_train_domain_cli_command_assembly(tmp_path):
    from scripts.train_domain import run_domain_adaptation

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    output_dir = tmp_path / "output_test"
    results_dir = tmp_path / "results_test"
    run_dir = tmp_path / "run_test"

    with patch("scripts.train_domain.build_domain_dataset") as mock_build, \
         patch("scripts.train_domain._run_training_subprocess") as mock_run:

        run_domain_adaptation(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            run_dir=str(run_dir),
            replay_samples=800,
            replay_ratio=0.3,
            epochs=4,
            learning_rate=1e-5,
            lora_r=32,
            lora_alpha=64,
            batch_size=4,
            grad_accum=8,
            telephony_mode="disjoint",
            fresh_start=True,
        )

        expected_dataset_dir = run_dir / "dataset"
        mock_build.assert_called_once_with(
            custom_path=str(custom_dir),
            replay_samples=800,
            replay_ratio=0.3,
            output_dir=str(expected_dataset_dir),
            seed=42,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Verify command flags
        assert "--base_model" in cmd
        assert cmd[cmd.index("--base_model") + 1] == str(model_dir)
        assert "--dataset_name" in cmd
        assert cmd[cmd.index("--dataset_name") + 1] == str(expected_dataset_dir)
        assert "--output_dir" in cmd
        assert cmd[cmd.index("--output_dir") + 1] == str(output_dir)
        assert "--results_dir" in cmd
        assert cmd[cmd.index("--results_dir") + 1] == str(results_dir)
        assert "--run_dir" in cmd
        assert cmd[cmd.index("--run_dir") + 1] == str(run_dir)
        assert "--epochs" in cmd
        assert cmd[cmd.index("--epochs") + 1] == "4"
        assert "--learning_rate" in cmd
        assert cmd[cmd.index("--learning_rate") + 1] == "1e-05" or cmd[cmd.index("--learning_rate") + 1] == "1e-5"
        assert "--lora_r" in cmd
        assert cmd[cmd.index("--lora_r") + 1] == "32"
        assert "--lora_alpha" in cmd
        assert cmd[cmd.index("--lora_alpha") + 1] == "64"
        assert "--batch_size" in cmd
        assert cmd[cmd.index("--batch_size") + 1] == "4"
        assert "--grad_accum" in cmd
        assert cmd[cmd.index("--grad_accum") + 1] == "8"
        assert "--telephony_mode" in cmd
        assert cmd[cmd.index("--telephony_mode") + 1] == "disjoint"
        assert "--fresh_start" in cmd


def test_train_domain_cli_main_entrypoint(tmp_path):
    from scripts.train_domain import main

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    with patch("scripts.train_domain.build_domain_dataset") as mock_build, \
         patch("scripts.train_domain._run_training_subprocess") as mock_run:
        mock_run.return_value = 0

        ret = main([
            "--base_model", str(model_dir),
            "--custom_data_path", str(custom_dir),
            "--output_dir", str(tmp_path / "out"),
        ])
        assert ret == 0
        mock_build.assert_called_once()
        mock_run.assert_called_once()


def test_train_domain_cli_require_weights(tmp_path):
    from scripts.train_domain import validate_inputs

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    # Fails when weights are required but absent
    with pytest.raises(ValueError) as exc_info:
        validate_inputs(
            base_model=str(model_dir),
            custom_data_path=str(custom_dir),
            require_weights=True,
        )
    assert "model weights" in str(exc_info.value)

    # Passes when weights file exists
    (model_dir / "model.safetensors").write_bytes(b"dummy")
    validate_inputs(
        base_model=str(model_dir),
        custom_data_path=str(custom_dir),
        require_weights=True,
    )


def test_train_domain_cli_subprocess_execution():
    from scripts.train_domain import _run_training_subprocess

    # Successful command
    code = _run_training_subprocess(["python", "-c", "import sys; sys.exit(0)"])
    assert code == 0

    # Failing command raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        _run_training_subprocess(["python", "-c", "import sys; sys.exit(42)"])
    assert "exit code 42" in str(exc_info.value)


def test_train_domain_cli_namespace_input(tmp_path):
    from scripts.train_domain import parse_args, run_domain_adaptation

    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    custom_dir = tmp_path / "valid_custom"
    custom_dir.mkdir()

    args = parse_args([
        "--base_model", str(model_dir),
        "--custom_data_path", str(custom_dir),
        "--output_dir", str(tmp_path / "out"),
        "--results_dir", str(tmp_path / "res"),
    ])

    with patch("scripts.train_domain.build_domain_dataset") as mock_build, \
         patch("scripts.train_domain._run_training_subprocess") as mock_run:
        mock_run.return_value = 0
        ret = run_domain_adaptation(args)
        assert ret == 0
        mock_build.assert_called_once()
        mock_run.assert_called_once()

