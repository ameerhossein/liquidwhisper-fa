"""Offline tests for trainer run-ledger stages and preflight plumbing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import run_state

RUN_ID = "20261030-194700-1a2b3c"


def _train_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_lora

    return train_lora


def _run(tmp_path, run_id=RUN_ID):
    run = run_state.create_run(
        tmp_path, config={}, trainer_args={}, run_id=run_id
    )
    return Path(tmp_path) / run["run_id"]


def _main_args(tmp_path, run_dir, **overrides):
    """Minimal argparse namespace consumed by main() before/at data prep."""
    values = dict(
        run_dir=str(run_dir),
        output_dir=str(Path(run_dir) / "output"),
        results_dir=str(run_dir),
        fresh_start=False,
        telephony_profiles="customer",
        base_model="BuzzASR/persian",
        dataset_name="test/dataset",
        lora_r=64,
        lora_alpha=128,
        batch_size=8,
        grad_accum=4,
        telephony_prob=0.0,
        telephony_use_ffmpeg=True,
        phone_benchmark=False,
        phone_seed=1234,
        num_proc=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stages(run_dir):
    run = run_state.load_run(run_dir)
    return {stage["name"]: stage for stage in run["stages"]}


def test_parser_accepts_run_dir_and_preflight_cache(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(sys, "argv", [
        "train_lora.py", "--run_dir", str(tmp_path / "run"),
        "--preflight_cache_dir", str(tmp_path / "cache"),
    ])
    args = train_lora.parse_args()
    assert args.run_dir == str(tmp_path / "run")
    assert args.preflight_cache_dir == str(tmp_path / "cache")
    assert args.post_eval_only is False


def test_parser_accepts_post_eval_only_flag(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--post_eval_only"])
    assert train_lora.parse_args().post_eval_only is True
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--no-post_eval_only"])
    assert train_lora.parse_args().post_eval_only is False


def test_copy_preflight_to_run_writes_manifest_copy(tmp_path):
    train_lora = _train_lora()
    cache = tmp_path / "cache"
    (cache / "baselines").mkdir(parents=True)
    src = cache / "baselines" / "preflight.json"
    src.write_text('{"models": {}}')
    run = tmp_path / "run"
    dest = train_lora._copy_preflight_to_run(run, cache)
    assert dest == run / "baselines" / "preflight.json"
    assert dest.read_text() == '{"models": {}}'


def test_record_trainer_args_stores_full_namespace(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    train_lora._record_trainer_args(
        run_dir, {"lora_r": 64, "warmup_steps": 500}
    )
    saved = run_state.load_run(run_dir)
    assert saved["trainer_args"]["lora_r"] == 64
    assert saved["trainer_args"]["warmup_steps"] == 500


def test_record_trainer_args_is_json_safe_for_paths_and_lists(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = SimpleNamespace(
        run_dir=tmp_path / "run",
        telephony_profiles=["customer"],
        lora_r=64,
        warmup_steps=None,
        ratios=[0.5, 1],
    )
    train_lora._record_trainer_args(run_dir, args)
    saved = run_state.load_run(run_dir)
    assert saved["trainer_args"]["run_dir"] == str(tmp_path / "run")
    assert saved["trainer_args"]["telephony_profiles"] == ["customer"]
    assert saved["trainer_args"]["ratios"] == [0.5, 1]
    assert saved["trainer_args"]["warmup_steps"] is None


def test_record_trainer_args_noops_without_run_dir(tmp_path):
    train_lora = _train_lora()
    assert train_lora._record_trainer_args(None, {"lora_r": 64}) is None


def test_mark_run_stage_noops_without_run_dir():
    train_lora = _train_lora()
    assert train_lora._mark_run_stage(None, "preflight", "running") is None


def test_mark_run_stage_updates_stage_detail(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    train_lora._mark_run_stage(
        run_dir, "preflight", "running", detail="preparing datasets"
    )
    stage = _stages(run_dir)["preflight"]
    assert stage["status"] == "running"
    assert stage["detail"] == "preparing datasets"


def test_fresh_start_clears_step_and_event_logs(tmp_path):
    train_lora = _train_lora()
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"
    output_dir.mkdir()
    results_dir.mkdir()
    for name in (
        "loss_history.json",
        "steps.jsonl",
        "events.jsonl",
        "live_telemetry.json",
        "training_live.log",
    ):
        (results_dir / name).write_text("stale", encoding="utf-8")

    train_lora._clear_previous_state(str(output_dir), results_dir=str(results_dir))

    for name in ("loss_history.json", "steps.jsonl", "events.jsonl"):
        assert not (results_dir / name).exists()
    # The server owns the live log/telemetry files; fresh start must not unlink them.
    assert (results_dir / "live_telemetry.json").exists()
    assert (results_dir / "training_live.log").exists()


def test_main_halts_before_data_prep_when_signal_exists(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")
    args = _main_args(tmp_path, run_dir)
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)

    class _NeverCalled:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("data preparation must not start after a halt")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _NeverCalled)
    train_lora.main()

    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    assert run["current_stage"] == "preflight"
    stages = _stages(run_dir)
    assert stages["preflight"]["status"] == "stopped"
    assert stages["training"]["status"] == "skipped"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "skipped"
    assert not (run_dir / "stop_training.signal").exists()


def test_main_marks_preflight_failed_when_data_prep_fails(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = _main_args(tmp_path, run_dir)
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)

    class _BrokenProcessor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("processor boom")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _BrokenProcessor)
    with pytest.raises(RuntimeError, match="processor boom"):
        train_lora.main()

    run = run_state.load_run(run_dir)
    assert run["status"] == "failed"
    assert run["failed_stage"] == "preflight"
    assert run["last_error"] == "processor boom"
    stage = _stages(run_dir)["preflight"]
    assert stage["status"] == "failed"
