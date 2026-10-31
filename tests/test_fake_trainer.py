"""CPU-only end-to-end tests for the fake trainer and fake merge scripts.

Both scripts run in subprocesses so the tests exercise the same CLI contract
the cockpit server will use. No CUDA or heavy ML stack is required.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import run_state

TRAINER = ROOT / "scripts" / "dev_fake_trainer.py"
MERGE = ROOT / "scripts" / "dev_fake_merge.py"
RUN_ID = "20261031-120000-abc123"


def _fake_env(**overrides):
    """Inherit the developer env but never leak ambient ASR_FAKE_* settings."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ASR_FAKE_") and key != "ASR_COCKPIT_SPAWNED"
    }
    env.update({"ASR_FAKE_STEPS": "12", "ASR_FAKE_STEP_DELAY": "0"})
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def _trainer_command(run_dir, cache_dir, extra_args=(), results_dir=None):
    return [
        sys.executable,
        str(TRAINER),
        "--run_dir",
        str(run_dir),
        "--results_dir",
        str(results_dir if results_dir is not None else run_dir),
        "--output_dir",
        str(run_dir / "output"),
        "--preflight_cache_dir",
        str(cache_dir),
        # Real-trainer flags the fake must tolerate via parse_known_args.
        "--base_model",
        "BuzzASR/persian",
        "--epochs",
        "4",
        *extra_args,
    ]


def _run_trainer(run_dir, cache_dir, extra_args=(), results_dir=None, **env):
    return subprocess.run(
        _trainer_command(run_dir, cache_dir, extra_args, results_dir),
        cwd=str(ROOT),
        env=_fake_env(**env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _stages(run_dir):
    run = run_state.load_run(run_dir)
    assert run is not None
    return {stage["name"]: stage for stage in run["stages"]}


def test_fake_trainer_does_not_overwrite_real_shared_cache(tmp_path):
    runs = tmp_path / "runs"
    real_cache = runs / "_shared" / "baselines" / "preflight.json"
    real_cache.parent.mkdir(parents=True)
    sentinel = json.dumps({"models": {"buzzasr_base": {"normalized_wer": 1.0}}})
    real_cache.write_text(sentinel, encoding="utf-8")
    fake_cache = runs / "_shared-fake"
    run_dir = runs / RUN_ID

    result = _run_trainer(run_dir, fake_cache)

    assert result.returncode == 0, result.stderr
    assert real_cache.read_text(encoding="utf-8") == sentinel
    assert (fake_cache / "baselines" / "preflight.json").is_file()
    assert run_state.load_run(run_dir)["status"] == "completed"


def test_fake_trainer_completes_and_writes_steps(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID
    cache_dir = tmp_path / "cache"

    result = _run_trainer(run_dir, cache_dir)

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    assert run["source"] == "fake"
    assert run["failed_stage"] is None

    stages = _stages(run_dir)
    assert stages["preflight"]["status"] == "completed"
    assert stages["training"]["status"] == "completed"
    assert stages["post_eval"]["status"] == "completed"
    # The server owns merge; the trainer must leave it available.
    assert stages["merge"]["status"] == "pending"

    assert run_state.step_count(run_dir) == 12
    assert run["step_count"] == 12
    steps = run_state.read_steps(run_dir)
    assert [record["step"] for record in steps] == list(range(1, 13))
    assert all(record["loss"] is not None for record in steps)

    for name in (
        "run.json",
        "events.jsonl",
        "steps.jsonl",
        "live_telemetry.json",
        "loss_history.json",
        "epoch_benchmarks.json",
        "training_live.log",
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "training_live.log").read_text(encoding="utf-8").strip()

    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "completed"
    assert telemetry["step"] == 12
    assert telemetry["max_steps"] == 12
    assert {
        "step",
        "max_steps",
        "epoch",
        "loss",
        "learning_rate",
        "speed_s_per_step",
        "eta_seconds",
    } <= set(telemetry)

    history = json.loads((run_dir / "loss_history.json").read_text(encoding="utf-8"))
    assert len(history) == 12
    assert [point["step"] for point in history] == list(range(1, 13))

    for preflight_path in (
        run_dir / "baselines" / "preflight.json",
        cache_dir / "baselines" / "preflight.json",
    ):
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        assert "buzzasr_base" in payload["models"]

    adapter_config = run_dir / "output" / "best_adapter" / "adapter_config.json"
    assert adapter_config.is_file()

    benchmarks = json.loads(
        (run_dir / "epoch_benchmarks.json").read_text(encoding="utf-8")
    )
    assert any(row.get("is_baseline") and row.get("source") == "preflight"
               for row in benchmarks)
    assert any(row.get("type") == "checkpoint" for row in benchmarks)

    report = json.loads(
        (run_dir / "output" / "final_evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["fake"] is True
    assert report["final_full_test_metrics"]["normalized_wer"] is not None


def test_fake_trainer_caps_loss_history(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    result = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="505")

    assert result.returncode == 0, result.stderr
    history = json.loads((run_dir / "loss_history.json").read_text(encoding="utf-8"))
    assert len(history) == 500
    assert history[-1]["step"] == 505
    # steps.jsonl is the uncapped authority.
    assert run_state.step_count(run_dir) == 505


@pytest.mark.parametrize("stage", ["preflight", "training", "post_eval"])
def test_fake_trainer_forced_failure(tmp_path, stage):
    run_dir = tmp_path / "runs" / RUN_ID

    result = _run_trainer(
        run_dir, tmp_path / "cache", ASR_FAKE_FAIL_STAGE=stage
    )

    assert result.returncode == 1, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "failed"
    assert run["failed_stage"] == stage
    assert run["last_error"]
    assert _stages(run_dir)[stage]["status"] == "failed"
    if stage == "post_eval":
        # The adapter is saved before post-eval, so a failed benchmark still
        # leaves a resumable adapter for --post_eval_only.
        adapter = run_dir / "output" / "best_adapter" / "adapter_config.json"
        assert adapter.is_file()


def _start_trainer(tmp_path, **env):
    run_dir = tmp_path / "runs" / RUN_ID
    process = subprocess.Popen(
        _trainer_command(run_dir, tmp_path / "cache"),
        cwd=str(ROOT),
        env=_fake_env(**env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, run_dir


def _wait_for_training(run_dir):
    def entered_training():
        run = run_state.load_run(run_dir)
        return bool(
            run
            and run.get("status") == "running"
            and run.get("current_stage") == "training"
        )

    return _wait_until(entered_training)


def _finish(process, timeout=20):
    try:
        output, _ = process.communicate(timeout=timeout)
        return output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_fake_trainer_halt_marks_stopped(tmp_path):
    process, run_dir = _start_trainer(
        tmp_path, ASR_FAKE_STEPS="1000", ASR_FAKE_STEP_DELAY="0.01"
    )
    try:
        assert _wait_for_training(run_dir), "fake trainer never entered training"
        signal = run_dir / "stop_training.signal"
        signal.write_text("stop", encoding="utf-8")
        output = _finish(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, output
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "stopped"
    assert run["failed_stage"] is None
    assert run["current_stage"] == "training"
    stages = _stages(run_dir)
    assert stages["preflight"]["status"] == "completed"
    assert stages["training"]["status"] == "stopped"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "pending"
    assert not signal.exists()
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "stopped"
    assert 0 < run_state.step_count(run_dir) < 1000


def test_fake_trainer_hang_waits_for_stop(tmp_path):
    process, run_dir = _start_trainer(
        tmp_path,
        ASR_FAKE_STEPS="1000",
        ASR_FAKE_STEP_DELAY="0.01",
        ASR_FAKE_HANG="1",
    )
    try:
        assert _wait_for_training(run_dir), "fake trainer never entered training"
        time.sleep(0.2)
        run = run_state.load_run(run_dir)
        assert run["status"] == "running"
        assert run_state.step_count(run_dir) == 0
        signal = run_dir / "stop_training.signal"
        signal.write_text("stop", encoding="utf-8")
        output = _finish(process, timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, output
    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    stages = _stages(run_dir)
    assert stages["training"]["status"] == "stopped"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "pending"


def test_fake_trainer_telemetry_carries_step_count(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    result = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="5")

    assert result.returncode == 0, result.stderr
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "completed"
    assert telemetry["step"] == 5
    assert telemetry["max_steps"] == 5
    assert telemetry["step_count"] == 5
    assert telemetry["step_count"] == run_state.step_count(run_dir)


def test_fake_trainer_telemetry_step_count_current_per_step(tmp_path):
    process, run_dir = _start_trainer(
        tmp_path, ASR_FAKE_STEPS="1000", ASR_FAKE_STEP_DELAY="0.01"
    )
    try:
        assert _wait_for_training(run_dir), "fake trainer never entered training"

        def read_telemetry():
            try:
                return json.loads(
                    (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
                )
            except ValueError:
                return {}

        assert _wait_until(
            lambda: read_telemetry().get("step_count", 0) >= 3
        ), "telemetry step_count never advanced"
        telemetry = read_telemetry()
        assert telemetry["status"] == "training"
        # The snapshot carries the count including the current step (written
        # just before the step record is appended), so it is either the exact
        # unique count or one ahead of it.
        assert telemetry["step_count"] in (
            run_state.step_count(run_dir),
            run_state.step_count(run_dir) + 1,
        )
        (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")
        output = _finish(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, output
    halted = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert halted["status"] == "stopped"
    assert halted["step_count"] == run_state.step_count(run_dir)


def test_fake_trainer_post_eval_only_does_not_replay_steps(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    first = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="5")
    assert first.returncode == 0, first.stderr
    step_count = run_state.step_count(run_dir)
    steps_before = (run_dir / "steps.jsonl").read_text(encoding="utf-8")
    loss_before = (run_dir / "loss_history.json").read_text(encoding="utf-8")

    # Simulate the cockpit resuming a run whose post-eval failed.
    run_state.set_stage(run_dir, "post_eval", "failed", detail="boom")
    run_state.set_status(
        run_dir, "failed", failed_stage="post_eval", last_error="boom"
    )
    report_path = run_dir / "output" / "final_evaluation_report.json"
    report_path.unlink()

    result = _run_trainer(
        run_dir,
        tmp_path / "cache",
        extra_args=("--post_eval_only",),
        ASR_FAKE_STEPS="12",
    )

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run["status"] == "completed"
    assert run["failed_stage"] is None
    # No optimizer step may be replayed or appended.
    assert run_state.step_count(run_dir) == step_count
    assert (run_dir / "steps.jsonl").read_text(encoding="utf-8") == steps_before
    assert (run_dir / "loss_history.json").read_text(
        encoding="utf-8"
    ) == loss_before
    assert _stages(run_dir)["post_eval"]["status"] == "completed"
    assert report_path.is_file()
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["step"] == step_count


def test_fake_trainer_resume_continues_after_stopped_run(tmp_path):
    process, run_dir = _start_trainer(
        tmp_path, ASR_FAKE_STEPS="1000", ASR_FAKE_STEP_DELAY="0.01"
    )
    try:
        assert _wait_for_training(run_dir), "fake trainer never entered training"
        (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")
        output = _finish(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, output
    steps_before = run_state.step_count(run_dir)
    assert 0 < steps_before < 1000

    result = _run_trainer(
        run_dir, tmp_path / "cache", ASR_FAKE_STEPS="5"
    )

    assert result.returncode == 0, result.stderr
    assert run_state.step_count(run_dir) == steps_before + 5
    steps = run_state.read_steps(run_dir)
    assert [record["step"] for record in steps] == list(
        range(1, steps_before + 6)
    )
    assert run_state.load_run(run_dir)["status"] == "completed"


def test_fake_merge_writes_artifacts(tmp_path):
    adapter_path = tmp_path / "best_adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "merged"

    result = subprocess.run(
        [
            sys.executable,
            str(MERGE),
            "--adapter_path",
            str(adapter_path),
            "--output_dir",
            str(output_dir),
            # Tolerated for parity with scripts/merge_lora.py.
            "--base_model",
            "BuzzASR/persian",
            "--run_dir",
            str(tmp_path / "run"),
            "--hf_run_repo",
            "me/run",
        ],
        cwd=str(ROOT),
        env=_fake_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        path = output_dir / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["fake"] is True
    tokenizer = json.loads(
        (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer["fake"] is True


def _write_import_blockers(directory):
    """Make any import of a heavy ML module fail loudly (not ImportError)."""
    directory.mkdir(parents=True)
    for name in ("torch", "transformers", "peft"):
        (directory / f"{name}.py").write_text(
            f"raise RuntimeError('blocked heavy import: {name}')\n",
            encoding="utf-8",
        )


def test_fake_scripts_run_without_gpu_stack(tmp_path):
    blockers = tmp_path / "blockers"
    _write_import_blockers(blockers)
    env = _fake_env(PYTHONPATH=str(blockers))

    run_dir = tmp_path / "runs" / RUN_ID
    trainer = subprocess.run(
        _trainer_command(run_dir, tmp_path / "cache"),
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert trainer.returncode == 0, trainer.stderr

    adapter_path = tmp_path / "best_adapter"
    adapter_path.mkdir()
    output_dir = tmp_path / "merged"
    merge = subprocess.run(
        [
            sys.executable,
            str(MERGE),
            "--adapter_path",
            str(adapter_path),
            "--output_dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert merge.returncode == 0, merge.stderr
    assert (output_dir / "model.safetensors").is_file()


@pytest.mark.parametrize("script", [TRAINER, MERGE])
def test_fake_scripts_do_not_import_heavy_ml_modules(script):
    source = script.read_text(encoding="utf-8")
    for module in ("torch", "transformers", "peft"):
        assert not re.search(
            rf"^\s*(?:import|from)\s+{module}\b", source, re.MULTILINE
        ), f"{script.name} imports {module}"


def test_fake_trainer_failure_telemetry_preserves_progress(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    result = _run_trainer(
        run_dir,
        tmp_path / "cache",
        ASR_FAKE_STEPS="5",
        ASR_FAKE_FAIL_STAGE="post_eval",
    )

    assert result.returncode == 1, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "failed"
    assert run["failed_stage"] == "post_eval"
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "failed"
    assert telemetry["failed_stage"] == "post_eval"
    assert telemetry["error"]
    assert telemetry["step"] == 5
    assert telemetry["max_steps"] == 5
    assert telemetry["step_count"] == 5
    assert telemetry["epoch"] > 0
    assert telemetry["loss"] is not None
    assert telemetry["gpu"]["available"] is False
    # The failed attempt is closed even though the fake owns the run.
    assert run["current_pid"] is None
    assert len(run["attempts"]) == 1
    assert run["attempts"][-1]["ended_at"] is not None
    assert run["attempts"][-1]["exit_code"] == 1
    assert not (run_dir / "training.pid").exists()


def test_fake_trainer_fresh_start_clears_previous_artifacts(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    first = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="5")
    assert first.returncode == 0, first.stderr
    assert run_state.step_count(run_dir) == 5
    assert (run_dir / "output" / "best_adapter").is_dir()
    # A stale stop signal must not halt the fresh run.
    (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")

    second = _run_trainer(
        run_dir,
        tmp_path / "cache",
        extra_args=("--fresh_start",),
        ASR_FAKE_STEPS="3",
    )

    assert second.returncode == 0, second.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    # The reused directory restarts instead of appending to old history.
    assert run_state.step_count(run_dir) == 3
    assert [record["step"] for record in run_state.read_steps(run_dir)] == [1, 2, 3]
    history = json.loads(
        (run_dir / "loss_history.json").read_text(encoding="utf-8")
    )
    assert [point["step"] for point in history] == [1, 2, 3]
    assert not (run_dir / "stop_training.signal").exists()
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["step"] == 3
    assert telemetry["max_steps"] == 3
    assert (run_dir / "output" / "best_adapter" / "adapter_config.json").is_file()


def test_fake_trainer_direct_run_closes_its_attempt(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    result = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="3")

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    assert len(run["attempts"]) == 1
    attempt = run["attempts"][0]
    assert attempt["ended_at"] is not None
    assert attempt["exit_code"] == 0
    assert isinstance(attempt["pid"], int)
    assert run["current_pid"] is None
    assert not (run_dir / "training.pid").exists()


def test_fake_trainer_cockpit_spawned_leaves_attempt_to_server(tmp_path):
    runs = tmp_path / "runs"
    run_state.create_run(
        runs, config={}, trainer_args={}, source="cockpit", run_id=RUN_ID
    )
    run_dir = runs / RUN_ID
    server_pid = 424242
    run_state.start_attempt(run_dir, server_pid, "initial")
    (run_dir / "training.pid").write_text(str(server_pid), encoding="utf-8")

    result = _run_trainer(
        run_dir, tmp_path / "cache", ASR_FAKE_STEPS="3", ASR_COCKPIT_SPAWNED="1"
    )

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    # The server owns the attempt and PID file; the fake must not double-close.
    assert len(run["attempts"]) == 1
    assert run["attempts"][0]["pid"] == server_pid
    assert run["attempts"][0]["ended_at"] is None
    assert run["current_pid"] == server_pid
    assert (run_dir / "training.pid").read_text(encoding="utf-8").strip() == str(
        server_pid
    )


def test_fake_trainer_live_foreign_attempt_is_left_alone(tmp_path):
    runs = tmp_path / "runs"
    run_state.create_run(
        runs, config={}, trainer_args={}, source="cockpit", run_id=RUN_ID
    )
    run_dir = runs / RUN_ID
    # This process is alive for the whole test, so the fake must treat the
    # open attempt as owned by someone else and stay hands-off.
    owner_pid = os.getpid()
    run_state.start_attempt(run_dir, owner_pid, "initial")
    (run_dir / "training.pid").write_text(str(owner_pid), encoding="utf-8")

    result = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="3")

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert len(run["attempts"]) == 1
    assert run["attempts"][0]["pid"] == owner_pid
    assert run["attempts"][0]["ended_at"] is None
    assert run["current_pid"] == owner_pid
    assert (run_dir / "training.pid").read_text(encoding="utf-8").strip() == str(
        owner_pid
    )


def test_fake_trainer_malformed_benchmark_rows_do_not_crash(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    first = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="3")
    assert first.returncode == 0, first.stderr
    path = run_dir / "epoch_benchmarks.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.extend(["junk", 42, None, {"type": "mystery"}])
    path.write_text(json.dumps(rows), encoding="utf-8")

    second = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="2")

    assert second.returncode == 0, second.stderr
    assert run_state.load_run(run_dir)["status"] == "completed"
    final = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(final, list)
    assert all(isinstance(row, dict) for row in final)
    assert any(row.get("is_baseline") for row in final)
    assert any(row.get("type") == "checkpoint" for row in final)
    assert run_state.step_count(run_dir) == 5


def test_fake_trainer_separate_results_dir(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID
    results_dir = tmp_path / "results"

    result = _run_trainer(
        run_dir, tmp_path / "cache", ASR_FAKE_STEPS="4", results_dir=results_dir
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "live_telemetry.json",
        "loss_history.json",
        "steps.jsonl",
        "epoch_benchmarks.json",
    ):
        assert (results_dir / name).is_file(), name
    assert not (results_dir / "run.json").exists()
    assert not (results_dir / "training_live.log").exists()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "training_live.log").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert not (run_dir / "live_telemetry.json").exists()
    assert not (run_dir / "steps.jsonl").exists()
    assert run_state.step_count(results_dir) == 4
    assert [record["step"] for record in run_state.read_steps(results_dir)] == [
        1,
        2,
        3,
        4,
    ]
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    assert run["step_count"] == 4
    assert run["paths"]["results_dir"] == str(results_dir.resolve())
    telemetry = json.loads(
        (results_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "completed"
    assert telemetry["step"] == 4


def _load_fake_trainer_module():
    """Import the fake trainer in-process for white-box exit-path tests."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dev_fake_trainer_under_test", str(TRAINER)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raised", "expected_code", "expected_reason"),
    [
        (SystemExit(), 0, "completed"),
        (SystemExit(0), 0, "completed"),
        (SystemExit("boom"), 1, "nonzero_exit"),
    ],
)
def test_fake_trainer_system_exit_code_finalizes_attempt(
    tmp_path, monkeypatch, raised, expected_code, expected_reason
):
    module = _load_fake_trainer_module()
    run_dir = tmp_path / "runs" / RUN_ID
    monkeypatch.delenv("ASR_COCKPIT_SPAWNED", raising=False)
    for key in [key for key in os.environ if key.startswith("ASR_FAKE_")]:
        monkeypatch.delenv(key, raising=False)

    def blow_up(*args, **kwargs):
        raise raised

    monkeypatch.setattr(module, "_run", blow_up)
    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                "--run_dir",
                str(run_dir),
                "--results_dir",
                str(run_dir),
                "--output_dir",
                str(run_dir / "output"),
                "--preflight_cache_dir",
                str(tmp_path / "cache"),
            ]
        )
    assert exc_info.value.code == raised.code
    run = run_state.load_run(run_dir)
    assert run is not None
    assert len(run["attempts"]) == 1
    assert run["attempts"][0]["exit_code"] == expected_code
    assert run["attempts"][0]["reason"] == expected_reason
    assert run["current_pid"] is None
    assert not (run_dir / "training.pid").exists()


def test_fake_trainer_fresh_start_ignored_with_post_eval_only(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID

    first = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="5")
    assert first.returncode == 0, first.stderr
    steps_before = (run_dir / "steps.jsonl").read_text(encoding="utf-8")
    loss_before = (run_dir / "loss_history.json").read_text(encoding="utf-8")
    adapter = run_dir / "output" / "best_adapter" / "adapter_config.json"
    assert adapter.is_file()

    result = _run_trainer(
        run_dir,
        tmp_path / "cache",
        extra_args=("--post_eval_only", "--fresh_start"),
        ASR_FAKE_STEPS="12",
    )

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    # Prior optimizer state is preserved, not cleared, like the real trainer.
    assert (run_dir / "steps.jsonl").read_text(encoding="utf-8") == steps_before
    assert (run_dir / "loss_history.json").read_text(
        encoding="utf-8"
    ) == loss_before
    assert adapter.is_file()
    assert "Note: --fresh_start is ignored with --post_eval_only." in (
        run_dir / "training_live.log"
    ).read_text(encoding="utf-8")


def test_fake_trainer_stale_attempt_finalized_as_process_lost(tmp_path):
    runs = tmp_path / "runs"
    run_state.create_run(
        runs, config={}, trainer_args={}, source="fake", run_id=RUN_ID
    )
    run_dir = runs / RUN_ID
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = proc.pid
    assert proc.wait(timeout=30) == 0
    run_state.start_attempt(run_dir, dead_pid, "initial")
    (run_dir / "training.pid").write_text(str(dead_pid), encoding="utf-8")

    result = _run_trainer(run_dir, tmp_path / "cache", ASR_FAKE_STEPS="3")

    assert result.returncode == 0, result.stderr
    run = run_state.load_run(run_dir)
    assert run is not None
    assert run["status"] == "completed"
    assert len(run["attempts"]) == 2
    stale, fresh = run["attempts"]
    assert stale["pid"] == dead_pid
    assert stale["ended_at"] is not None
    assert stale["exit_code"] is None
    assert stale["reason"] == "process_lost"
    assert fresh["pid"] != dead_pid
    assert fresh["ended_at"] is not None
    assert fresh["exit_code"] == 0
    assert fresh["reason"] == "completed"
    assert run["current_pid"] is None
    assert not (run_dir / "training.pid").exists()
    assert run_state.step_count(run_dir) == 3
    assert all(
        record.get("attempt") == 2 for record in run_state.read_steps(run_dir)
    )
