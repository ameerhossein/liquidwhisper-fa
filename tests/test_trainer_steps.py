"""Offline tests for per-step JSONL records and run progress updates.

No model, dataset, or CUDA is required: the callback only writes JSON files
during construction and ``on_step_end``/``on_train_end``.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src import run_state
from src.trainer_callbacks import WebTelemetryCallback

RUN_ID = "20261031-010203-abcdef"

STEP_FIELDS = {
    "step",
    "epoch",
    "loss",
    "loss_step",
    "learning_rate",
    "speed_s_per_step",
    "elapsed_seconds",
    "eta_seconds",
    "attempt",
    "ts",
}


def _callback(results_dir, **kwargs):
    return WebTelemetryCallback(
        results_dir=str(results_dir),
        test_dataset=None,
        processor=None,
        metrics_calculator=None,
        **kwargs,
    )


def _state(step, *, max_steps=600, epoch=0.1, loss=0.5, lr=1e-4):
    log_history = []
    if loss is not None:
        log_history.append({"step": step, "loss": loss, "learning_rate": lr})
    return SimpleNamespace(
        global_step=step,
        max_steps=max_steps,
        epoch=epoch,
        log_history=log_history,
    )


def _args_control():
    return SimpleNamespace(), SimpleNamespace(
        should_save=False, should_training_stop=False
    )


def _records(results_dir):
    text = (Path(results_dir) / "steps.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.strip().splitlines()]


def test_on_step_end_appends_one_record_per_step(tmp_path):
    cb = _callback(tmp_path, run_dir=str(tmp_path))
    args, control = _args_control()
    state = SimpleNamespace(
        global_step=1,
        max_steps=10,
        epoch=0.1,
        log_history=[{"step": 1, "loss": 0.5, "learning_rate": 1e-4}],
    )

    cb.on_step_end(args, state, control)

    lines = (tmp_path / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 1
    assert record["loss"] == 0.5
    assert record["loss_step"] == 1
    assert record["attempt"] == 1
    assert record["ts"] > 0
    assert STEP_FIELDS <= set(record)


def test_step_record_carries_loss_forward_with_its_origin_step(tmp_path):
    cb = _callback(tmp_path)
    args, control = _args_control()
    state = _state(50, max_steps=100, loss=0.4, lr=2e-4)
    cb.on_step_end(args, state, control)

    # Same log_history: the loss was produced at step 50, not 51.
    state.global_step = 51
    cb.on_step_end(args, state, control)

    records = _records(tmp_path)
    assert [record["step"] for record in records] == [50, 51]
    assert all(record["loss"] == 0.4 for record in records)
    assert all(record["loss_step"] == 50 for record in records)
    assert all(record["learning_rate"] == 2e-4 for record in records)


def test_step_record_uses_callback_attempt(tmp_path):
    cb = _callback(tmp_path, attempt=3)
    args, control = _args_control()

    cb.on_step_end(args, _state(1), control)

    assert _records(tmp_path)[0]["attempt"] == 3


def test_steps_file_is_uncapped_and_resumes(tmp_path):
    cb = _callback(tmp_path)
    args, control = _args_control()
    for step in range(1, 601):
        cb.on_step_end(args, _state(step), control)

    assert len(_records(tmp_path)) == 600
    history = json.loads(
        (tmp_path / "loss_history.json").read_text(encoding="utf-8")
    )
    assert len(history) == 500

    resumed = _callback(tmp_path)
    resumed.on_step_end(args, _state(601), control)

    records = _records(tmp_path)
    assert len(records) == 601
    assert records[-1]["step"] == 601


def test_progress_updates_every_10_steps_and_on_train_end(tmp_path):
    run_dir = Path(tmp_path) / RUN_ID
    run_state.create_run(tmp_path, config={}, trainer_args={}, run_id=RUN_ID)
    cb = _callback(run_dir, run_dir=str(run_dir))
    args, control = _args_control()

    for step in range(1, 10):
        cb.on_step_end(args, _state(step, max_steps=200), control)
    assert run_state.load_run(run_dir)["step_count"] == 0

    cb.on_step_end(args, _state(10, max_steps=200), control)
    run = run_state.load_run(run_dir)
    assert run["step_count"] == 10
    assert run["last_step"] == 10
    assert run["last_step_at"] is not None

    for step in range(11, 24):
        cb.on_step_end(args, _state(step, max_steps=200), control)
    cb.on_train_end(args, _state(23, max_steps=200), control)

    run = run_state.load_run(run_dir)
    assert run["step_count"] == 23
    assert run["last_step"] == 23


def test_progress_is_skipped_without_run_dir(tmp_path):
    cb = _callback(tmp_path)
    args, control = _args_control()

    cb.on_step_end(args, _state(100), control)
    cb.on_train_end(args, _state(100), control)

    assert len(_records(tmp_path)) == 1


def test_replayed_step_keeps_progress_count_unique(tmp_path):
    run_dir = Path(tmp_path) / RUN_ID
    run_state.create_run(tmp_path, config={}, trainer_args={}, run_id=RUN_ID)
    for step in range(1, 100):
        run_state.append_step(run_dir, {"step": step, "loss": 0.1})

    cb = _callback(run_dir, run_dir=str(run_dir))
    assert cb.step_count == 99
    args, control = _args_control()

    # A resume from checkpoint-90 replays step 90 before reaching new steps.
    # The replay appends a duplicate line but must not inflate the unique
    # count or trigger a manifest flush on its own.
    cb.on_step_end(args, _state(90), control)
    assert cb.step_count == 99
    run = run_state.load_run(run_dir)
    assert run["step_count"] == 0

    cb.on_train_end(args, _state(90), control)
    run = run_state.load_run(run_dir)
    assert run["step_count"] == 99
    assert run["last_step"] == 90


def test_on_step_end_consumes_stop_signal_and_records_halted_step(tmp_path):
    cb = _callback(tmp_path)
    (tmp_path / "stop_training.signal").write_text("stop", encoding="utf-8")
    args, control = _args_control()

    cb.on_step_end(args, _state(7), control)

    assert cb.halt_requested is True
    assert control.should_training_stop is True
    assert control.should_save is True
    assert not (tmp_path / "stop_training.signal").exists()
    # The step that consumed the halt still finished and gets its record.
    assert [record["step"] for record in _records(tmp_path)] == [7]


def test_on_train_end_does_not_claim_run_completed(tmp_path):
    cb = _callback(tmp_path)
    args, control = _args_control()

    cb.on_train_end(args, _state(10, max_steps=10), control)

    telemetry = json.loads(
        (Path(tmp_path) / "live_telemetry.json").read_text(encoding="utf-8")
    )
    # Post-eval may still be ahead; only the main script may mark completion.
    assert telemetry["status"] == "training"

    cb.halt_requested = True
    cb.on_train_end(args, _state(10, max_steps=10), control)
    telemetry = json.loads(
        (Path(tmp_path) / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "stopped"


def test_replayed_step_does_not_inflate_telemetry_count(tmp_path):
    run_dir = Path(tmp_path) / RUN_ID
    run_state.create_run(tmp_path, config={}, trainer_args={}, run_id=RUN_ID)
    for step in range(1, 11):
        run_state.append_step(run_dir, {"step": step, "loss": 0.1})

    cb = _callback(run_dir, run_dir=str(run_dir))
    assert cb.step_count == 10
    args, control = _args_control()
    cb.on_step_end(args, _state(5, max_steps=100), control)

    assert cb.step_count == 10
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["step_count"] == 10


def test_telemetry_snapshots_include_step_count(tmp_path):
    cb = _callback(tmp_path)
    args, control = _args_control()
    cb.on_step_end(args, _state(1, max_steps=10), control)

    telemetry = json.loads(
        (Path(tmp_path) / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["step_count"] == 1

    cb.on_train_end(args, _state(1, max_steps=10), control)
    telemetry = json.loads(
        (Path(tmp_path) / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["step_count"] == 1


def test_progress_flushes_every_10_steps(tmp_path):
    run_dir = Path(tmp_path) / RUN_ID
    run_state.create_run(tmp_path, config={}, trainer_args={}, run_id=RUN_ID)
    cb = _callback(run_dir, run_dir=str(run_dir))
    args, control = _args_control()

    for step in range(1, 10):
        cb.on_step_end(args, _state(step, max_steps=200), control)
    assert run_state.load_run(run_dir)["step_count"] == 0

    cb.on_step_end(args, _state(10, max_steps=200), control)
    run = run_state.load_run(run_dir)
    assert run["step_count"] == 10
    assert run["last_step"] == 10
