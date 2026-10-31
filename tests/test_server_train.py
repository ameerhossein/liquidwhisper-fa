"""Offline tests for server training control and benchmark endpoints.

Training is never launched: the audio-backend check is stubbed and subprocess.Popen
records the command instead. Benchmark tests only read files under tmp_path.
"""

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from src import run_state

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ASR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ASR_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.delenv("ASR_COCKPIT_TOKEN", raising=False)
    monkeypatch.delenv("ASR_RUNS_DIR", raising=False)
    monkeypatch.delenv("ASR_FAKE_TRAINER", raising=False)
    spec = importlib.util.spec_from_file_location("server_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_app"] = mod
    spec.loader.exec_module(mod)

    launched = []

    class _FakeProcess:
        pid = os.getpid()

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        launched.append(cmd)
        return _FakeProcess()

    monkeypatch.setattr(mod, "_check_audio_backend_or_400", lambda: None)
    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
    client = TestClient(mod.app)
    client.mod = mod
    client.launched = launched
    try:
        yield client
    finally:
        for close in (
            mod._close_merge_log_fd,
            mod._close_log_fd,
            mod._close_smoke_log_fd,
        ):
            try:
                close()
            except Exception:
                pass


def _flags(cmd):
    return dict(zip(cmd[:-1], cmd[1:]))


def _run_json(client, run_id):
    return json.loads(
        (client.mod.RUNS_DIR / run_id / "run.json").read_text(encoding="utf-8")
    )


def _write_run(client, run_id=None, **fields):
    """Update the newest run manifest (or *run_id*) and return it."""
    if run_id is None:
        runs = sorted(
            path.name
            for path in client.mod.RUNS_DIR.iterdir()
            if (path / "run.json").is_file()
        )
        assert runs, "no run manifest to update"
        run_id = runs[-1]

    def mutate(run):
        run.update(fields)
        return run

    return run_state.update_run(client.mod.RUNS_DIR / run_id, mutate)


def _create_run(client, *, pid, status="running", run_id="20261030-194700-aaaaaa"):
    run = run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_state.start_attempt(client.mod.RUNS_DIR / run_id, pid, "test")
    run_state.set_status(client.mod.RUNS_DIR / run_id, status)
    return run


def _seed_adapter(client, run_id):
    adapter = client.mod.RUNS_DIR / run_id / "output" / "best_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter


def _write_merge_artifacts(client, run_id, *, weights="model.safetensors"):
    merged = client.mod.RUNS_DIR / run_id / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (merged / weights).write_text("", encoding="utf-8")
    return merged


def test_refresh_finds_older_live_run(client, monkeypatch):
    monkeypatch.setattr(
        client.mod, "_is_process_running", lambda pid=None: pid == os.getpid()
    )
    old = _create_run(client, pid=os.getpid(), run_id="20261030-194700-aaaaaa")
    new = _create_run(client, pid=999999, run_id="20261030-195700-bbbbbb")
    client.mod._refresh_runs()
    assert client.mod.ACTIVE_RUN_ID == old["run_id"]
    assert client.mod.LATEST_RUN["run_id"] == new["run_id"]


def test_refresh_marks_dead_run_interrupted(client, monkeypatch):
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run = _create_run(client, pid=999999, status="running")
    client.mod._refresh_runs()
    assert (
        run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["status"]
        == "interrupted"
    )


def test_refresh_does_not_rewrite_completed_run(client, monkeypatch):
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run = _create_run(client, pid=999999, status="completed")
    client.mod._refresh_runs()
    assert (
        run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["status"]
        == "completed"
    )


def test_runs_endpoints_list_latest_and_steps(client):
    start = client.post("/api/train/start", json={}).json()
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == start["run_id"]
    latest = client.get("/api/runs/latest").json()
    assert latest["run"]["run_id"] == start["run_id"]
    steps = client.get(f"/api/runs/{start['run_id']}/steps?tail=5").json()
    assert steps["run_id"] == start["run_id"]
    assert steps["total"] == 0


def test_run_logs_are_scoped_and_missing_runs_are_404(client):
    first = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-aaaaaa"
    )
    second = _create_run(
        client, pid=999999, status="completed", run_id="20261030-195700-bbbbbb"
    )
    first_dir = client.mod.RUNS_DIR / first["run_id"]
    second_dir = client.mod.RUNS_DIR / second["run_id"]
    (first_dir / "training_live.log").write_text(
        "first line\nsecond line\nthird line\n", encoding="utf-8"
    )
    (second_dir / "training_live.log").write_text("live only\n", encoding="utf-8")

    res = client.get(f"/api/runs/{first['run_id']}/logs?tail=2")
    assert res.status_code == 200, res.text
    assert res.json() == {"logs": ["second line", "third line"]}

    # The live endpoint keeps its current-run behavior.
    live = client.get("/api/logs?tail=5")
    assert live.status_code == 200, live.text
    assert live.json() == {"logs": ["live only"]}

    empty = _create_run(
        client, pid=999999, status="completed", run_id="20261030-200700-cccccc"
    )
    empty_res = client.get(f"/api/runs/{empty['run_id']}/logs")
    assert empty_res.status_code == 200, empty_res.text
    assert empty_res.json() == {
        "logs": ["Log buffer empty. Training has not started."]
    }

    missing = client.get("/api/runs/20990101-000000-abcdef/logs")
    assert missing.status_code == 404
    invalid = client.get("/api/runs/not-a-run/logs")
    assert invalid.status_code == 404


def test_merge_starts_and_marks_stage(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "output" / "best_adapter").mkdir(parents=True)
    (run_dir / "output" / "best_adapter" / "adapter_config.json").write_text("{}")
    run_state.set_stage(run_dir, "preflight", "completed")
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")
    res = client.post(f"/api/runs/{start['run_id']}/merge")
    assert res.status_code == 200, res.text
    assert "--adapter_path" in client.launched[-1]
    assert run_state.load_run(run_dir)["stages"][3]["status"] == "running"


def test_merge_rejected_while_target_trainer_live_and_stages_pending(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    _seed_adapter(client, start["run_id"])
    run_state.set_stage(run_dir, "training", "completed")
    # The fake trainer handle is still live and belongs to this run, but
    # preflight/post_eval are not terminal yet: R12 blocks the merge.
    res = client.post(f"/api/runs/{start['run_id']}/merge")
    assert res.status_code == 400
    assert len(client.launched) == 1


def test_merge_rejected_before_training_completed(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "output" / "best_adapter").mkdir(parents=True)
    (run_dir / "output" / "best_adapter" / "adapter_config.json").write_text("{}")
    res = client.post(f"/api/runs/{start['run_id']}/merge")
    assert res.status_code == 400


def test_defaults_use_small_batch_with_accumulation_and_sparse_eval(client):
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200, res.text
    flags = _flags(client.launched[0])
    assert flags["--batch_size"] == "8"
    assert flags["--grad_accum"] == "4"
    assert flags["--eval_steps"] == "1000"
    assert flags["--save_steps"] == "1000"
    assert flags["--max_benchmark_samples"] == "1000"
    assert flags["--preflight_samples"] == "1000"
    assert flags["--telephony_prob"] == "0.5"
    assert "--no-gradient_checkpointing" in client.launched[0]


def test_gradient_checkpointing_flag_is_passed_through(client):
    res = client.post("/api/train/start", json={"batch_size": 16, "gradient_checkpointing": True})
    assert res.status_code == 200, res.text
    assert "--gradient_checkpointing" in client.launched[0]
    assert res.json()["config"]["gradient_checkpointing"] is True


def test_batch_above_8_without_checkpointing_is_rejected(client):
    res = client.post("/api/train/start", json={"batch_size": 16, "gradient_checkpointing": False})
    assert res.status_code == 400
    assert "gradient_checkpointing" in res.json()["detail"]
    assert client.launched == []


def test_eval_steps_out_of_range_is_rejected(client):
    res = client.post("/api/train/start", json={"eval_steps": 0})
    assert res.status_code == 400
    assert client.launched == []


def test_default_benchmarks_load_preflight_rows(client, tmp_path, monkeypatch):
    preflight = tmp_path / "baselines" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"models": {
        "buzzasr_base": {"name": "BuzzASR Base", "epoch": "BuzzASR Base (preflight, 1,000)",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
        "whisper_v3": {"name": "Whisper", "epoch": "Whisper (preflight)",
                       "is_reference": True, "total_samples": 1000,
                       "normalized_wer": 28.5, "phone_normalized_wer": 35.0},
    }}))
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", preflight)
    by_epoch = {r["epoch"]: r for r in client.mod.get_default_benchmarks()}
    assert by_epoch["BuzzASR Base (preflight, 1,000)"]["phone_normalized_wer"] == 24.1
    assert by_epoch["Whisper (preflight)"]["is_reference"] is True


def test_default_benchmarks_empty_without_preflight(client, tmp_path, monkeypatch):
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", tmp_path / "missing.json")
    assert client.mod.get_default_benchmarks() == []


def test_benchmarks_endpoint_drops_unmeasured_baselines(client, tmp_path, monkeypatch):
    bench = tmp_path / "epoch_benchmarks.json"
    bench.write_text(json.dumps([
        {"epoch": "Old hardcoded base", "is_baseline": True, "normalized_wer": 17.2},
        {"epoch": "Epoch 1", "is_baseline": False, "normalized_wer": 15.0},
    ]))
    monkeypatch.setattr(client.mod, "BENCHMARKS_FILE", bench)
    rows = client.get("/api/benchmarks").json()
    assert not any(r.get("is_baseline") for r in rows)
    assert any(r.get("epoch") == "Epoch 1" for r in rows)


def test_websocket_drops_unmeasured_baselines(client, tmp_path, monkeypatch):
    bench = tmp_path / "epoch_benchmarks.json"
    bench.write_text(json.dumps([
        {"epoch": "Old hardcoded base", "is_baseline": True, "normalized_wer": 17.2},
        {"epoch": "Epoch 1", "is_baseline": False, "normalized_wer": 15.0},
    ]))
    monkeypatch.setattr(client.mod, "BENCHMARKS_FILE", bench)
    monkeypatch.setattr(client.mod, "get_gpu_info", lambda: {})
    with client.websocket_connect("/ws/live") as ws:
        payload = ws.receive_json()
    rows = payload["benchmarks"]
    assert not any(r.get("is_baseline") for r in rows)
    assert any(r.get("epoch") == "Epoch 1" for r in rows)


def test_benchmark_fallback_rows_are_tagged_preflight(client, tmp_path, monkeypatch):
    bench = tmp_path / "epoch_benchmarks.json"
    bench.write_text(json.dumps([
        {"epoch": "Old hardcoded base", "is_baseline": True, "normalized_wer": 17.2},
    ]))
    preflight = tmp_path / "baselines" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"models": {
        "buzzasr_base": {"epoch": "BuzzASR Base (preflight, 1,000)",
                         "total_samples": 1000, "normalized_wer": 17.2,
                         "phone_normalized_wer": 24.1},
    }}))
    monkeypatch.setattr(client.mod, "BENCHMARKS_FILE", bench)
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", preflight)
    rows = client.get("/api/benchmarks").json()
    assert rows and rows[0]["source"] == "preflight"
    persisted = json.loads(bench.read_text())
    assert persisted and all(r["source"] == "preflight" for r in persisted)
    assert client.get("/api/benchmarks").json() == rows


def _write_smoke_attempt(
    client, *, exit_code=None, adapter=True, log_lines=(), started_at=0
):
    out = Path(client.mod._smoke_output_dir())
    if adapter:
        adapter_dir = out / "best_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    results_dir = client.mod._smoke_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    attempt = {"started_at": started_at}
    if exit_code is not None:
        attempt["exit_code"] = exit_code
    client.mod._smoke_attempt_file().write_text(
        json.dumps(attempt), encoding="utf-8"
    )
    client.mod.smoke_last_exit_code = exit_code
    if log_lines:
        client.mod._smoke_log_file().write_text(
            "\n".join(log_lines), encoding="utf-8"
        )


def test_smoke_status_is_idle_before_any_attempt(client):
    res = client.get("/api/train/smoke/status")
    assert res.status_code == 200
    assert res.json()["status"] == "idle"


def test_smoke_status_reports_running_from_live_pid(client):
    client.mod._smoke_results_dir().mkdir(parents=True, exist_ok=True)
    client.mod._smoke_pid_file().write_text(str(os.getpid()), encoding="utf-8")
    res = client.get("/api/train/smoke/status")
    assert res.status_code == 200
    assert res.json()["status"] == "running"
    assert res.json()["running"] is True


def test_smoke_status_passes_on_zero_exit_and_adapter(client):
    _write_smoke_attempt(client, exit_code=0, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "passed"
    assert res.json()["adapter_exists"] is True


def test_smoke_status_passes_after_restart_when_adapter_exists(client):
    _write_smoke_attempt(client, exit_code=None, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "passed"


def test_smoke_status_fails_on_nonzero_exit_even_with_adapter(client):
    _write_smoke_attempt(client, exit_code=1, adapter=True)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["exit_code"] == 1
    assert res.json()["reason"] == "nonzero_exit"


def test_smoke_status_uses_persisted_exit_code_after_restart(client):
    _write_smoke_attempt(client, exit_code=1, adapter=True)
    client.mod.smoke_last_exit_code = None
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["exit_code"] == 1


def test_smoke_status_fails_when_adapter_predates_attempt(client):
    _write_smoke_attempt(
        client, exit_code=None, adapter=True, started_at=time.time() + 60
    )
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"


def test_smoke_status_fails_when_adapter_is_missing(client):
    _write_smoke_attempt(client, exit_code=0, adapter=False)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["adapter_exists"] is False
    assert res.json()["reason"] == "missing_adapter"


def test_smoke_status_fails_when_attempt_has_no_result(client):
    _write_smoke_attempt(client, exit_code=None, adapter=False)
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["reason"] == "ended_without_result"


def test_smoke_status_includes_log_tail_on_failure(client):
    _write_smoke_attempt(
        client,
        exit_code=1,
        adapter=False,
        log_lines=["loading model", "RuntimeError: out of memory"],
    )
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "failed"
    assert res.json()["log_tail"][-1] == "RuntimeError: out of memory"


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class _RunningProcess:
    pid = 999_999

    def poll(self):
        return None


def test_smoke_start_builds_fast_isolated_command(client):
    res = client.post(
        "/api/train/smoke",
        json={
            "batch_size": 4,
            "grad_accum": 8,
            "learning_rate": 2e-4,
            "gradient_checkpointing": False,
        },
    )
    assert res.status_code == 200, res.text
    cmd = client.launched[0]
    assert _flag_value(cmd, "--max_train_samples") == "200"
    assert _flag_value(cmd, "--max_eval_samples") == "50"
    assert _flag_value(cmd, "--epochs") == "1"
    assert _flag_value(cmd, "--eval_steps") == "5"
    assert _flag_value(cmd, "--save_steps") == "5"
    assert _flag_value(cmd, "--warmup_steps") == "2"
    assert _flag_value(cmd, "--max_benchmark_samples") == "25"
    assert _flag_value(cmd, "--batch_size") == "4"
    assert _flag_value(cmd, "--grad_accum") == "8"
    assert _flag_value(cmd, "--learning_rate") == "0.0002"
    assert "--no-run-post-test-eval" in cmd
    assert "--fresh_start" in cmd
    assert "--no-auto_resume" in cmd
    assert _flag_value(cmd, "--preflight_samples") == "25"
    smoke_output = Path(_flag_value(cmd, "--output_dir")).resolve()
    smoke_results = Path(_flag_value(cmd, "--results_dir")).resolve()
    assert smoke_output.name == "smoke-test"
    assert smoke_results.name == "smoke-test"
    assert smoke_output != Path(client.mod._default_output_dir()).resolve()
    assert smoke_results != client.mod.RESULTS_DIR.resolve()
    assert res.json()["mode"] == "smoke"


def test_smoke_output_dir_avoids_persistent_fallback_collision(client, monkeypatch):
    monkeypatch.setattr(
        client.mod,
        "_default_output_dir",
        lambda: "/workspace/stt-outputs/buzzasr-neyshekar-lora",
    )

    def fake_resolve_output_dir(raw=None):
        raw = str(raw or "")
        if raw.startswith("/workspace"):
            return "outputs/buzzasr-neyshekar-lora"
        return raw

    monkeypatch.setattr(
        client.mod._managed_paths, "resolve_output_dir", fake_resolve_output_dir
    )
    effective = Path(client.mod._smoke_output_dir()).resolve()
    production = (
        client.mod.BASE_DIR / "outputs" / "buzzasr-neyshekar-lora"
    ).resolve()
    assert effective != production
    assert effective.name in ("smoke-test", "smoke-test-check")


def test_smoke_start_removes_stale_adapter_and_writes_attempt(client):
    out = Path(client.mod._smoke_output_dir())
    adapter_dir = out / "best_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 200, res.text
    assert not adapter_dir.exists()
    assert client.mod._smoke_attempt_file().is_file()


def test_smoke_start_refuses_when_stale_adapter_cannot_be_removed(client, monkeypatch):
    out = Path(client.mod._smoke_output_dir())
    adapter_dir = out / "best_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(client.mod.shutil, "rmtree", lambda *args, **kwargs: None)

    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 500
    assert "refusing to start" in res.json()["detail"]
    assert client.launched == []


def test_smoke_start_fails_when_pid_cannot_be_written(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        client.mod, "_smoke_pid_file", lambda: tmp_path / "missing" / "training.pid"
    )
    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 500
    assert "Failed to record smoke PID" in res.json()["detail"]
    assert len(client.launched) == 1


def test_smoke_start_rejects_batch_above_8_without_checkpointing(client):
    res = client.post(
        "/api/train/smoke",
        json={"batch_size": 16, "gradient_checkpointing": False},
    )
    assert res.status_code == 400
    assert "gradient_checkpointing" in res.json()["detail"]
    assert client.launched == []


def test_smoke_start_rejects_invalid_accumulation(client):
    res = client.post("/api/train/smoke", json={"grad_accum": 0})
    assert res.status_code == 400
    assert client.launched == []


def test_smoke_start_rejects_invalid_learning_rate(client):
    res = client.post("/api/train/smoke", json={"learning_rate": 1.0})
    assert res.status_code == 400
    assert client.launched == []


def test_smoke_start_rejected_while_full_run_active(client):
    client.mod.training_process = _RunningProcess()
    res = client.post("/api/train/smoke", json={})
    assert res.status_code == 400
    assert client.launched == []


def test_full_start_rejected_while_smoke_active(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    res = client.post("/api/train/start", json={})
    assert res.status_code == 400
    assert len(client.launched) == 1


class _FinishedProcess:
    pid = 999_999

    def poll(self):
        return 0


def test_full_start_works_after_smoke_finishes(client, monkeypatch):
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    assert client.post("/api/train/smoke", json={}).status_code == 200
    client.mod.smoke_process = _FinishedProcess()

    res = client.post("/api/train/start", json={})
    assert res.status_code == 200
    assert len(client.launched) == 2
    assert client.mod.smoke_last_exit_code == 0


def test_full_start_rejects_smoke_output_path(client):
    res = client.post(
        "/api/train/start", json={"output_dir": client.mod._smoke_output_dir()}
    )
    assert res.status_code == 400
    assert client.launched == []


def test_start_creates_self_contained_run(client):
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200, res.text
    run_id = res.json()["run_id"]
    run_dir = client.mod.RUNS_DIR / run_id
    run = run_state.load_run(run_dir)
    assert run["status"] == "running"
    assert run["paths"]["results_dir"] == str(run_dir)
    assert run["paths"]["output_dir"] == str(run_dir / "output")
    flags = _flags(client.launched[0])
    assert flags["--run_dir"] == str(run_dir)
    assert flags["--preflight_cache_dir"] == str(client.mod.RUNS_DIR / "_shared")
    assert flags["--output_dir"] == str(run_dir / "output")
    assert flags["--results_dir"] == str(run_dir)
    assert run["trainer_args"]["epochs"] == 4


def test_start_rejects_explicit_results_dir(client):
    res = client.post(
        "/api/train/start", json={"results_dir": "/tmp/elsewhere"}
    )
    assert res.status_code == 400
    assert "results_dir" in res.json()["detail"]
    assert client.launched == []


def test_resume_unknown_run_checks_manifest_before_audio(client, monkeypatch):
    def broken_audio():
        raise client.mod.HTTPException(
            status_code=500, detail="audio backend unavailable"
        )

    monkeypatch.setattr(client.mod, "_check_audio_backend_or_400", broken_audio)
    res = client.post("/api/runs/20261031-000000-abcdef/resume")
    assert res.status_code == 404


def test_resume_completed_run_checks_status_before_audio(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    _write_run(
        client, start["run_id"], status="completed", current_pid=None
    )

    def broken_audio():
        raise client.mod.HTTPException(
            status_code=500, detail="audio backend unavailable"
        )

    monkeypatch.setattr(client.mod, "_check_audio_backend_or_400", broken_audio)
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 400
    assert "cannot resume" in res.json()["detail"]


def test_resume_rejected_while_live_run_exists(client):
    client.post("/api/train/start", json={})
    other = run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}
    )
    other_dir = client.mod.RUNS_DIR / other["run_id"]
    run_state.set_status(other_dir, "failed")
    res = client.post(f"/api/runs/{other['run_id']}/resume")
    assert res.status_code == 400
    assert "already running" in res.json()["detail"]
    assert len(client.launched) == 1


def test_resume_uses_manifest_run_dir(client, tmp_path):
    start = client.post("/api/train/start", json={}).json()
    source_dir = client.mod.RUNS_DIR / start["run_id"]
    manifest_run_dir = tmp_path / "cli-run"

    def repoint(run):
        run["status"] = "failed"
        run["paths"]["run_dir"] = str(manifest_run_dir)
        return run

    run_state.update_run(source_dir, repoint)
    shutil.copytree(source_dir, manifest_run_dir)
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    flags = _flags(client.launched[-1])
    assert flags["--run_dir"] == str(manifest_run_dir)
    assert run_state.load_run(manifest_run_dir)["status"] == "running"


def test_run_preflight_file_falls_back_to_legacy(client, tmp_path, monkeypatch):
    legacy = tmp_path / "legacy-preflight.json"
    monkeypatch.setattr(client.mod, "PREFLIGHT_FILE", legacy)
    run = {
        "run_id": "20261031-000000-abcdef",
        "paths": {
            "run_dir": str(tmp_path / "missing-run"),
            "preflight_cache": str(
                tmp_path / "missing-shared" / "baselines" / "preflight.json"
            ),
        },
    }
    assert client.mod._run_preflight_file(run) == legacy


def test_escalate_halt_pid_skips_terminate_when_run_completed(
    client, tmp_path, monkeypatch
):
    telemetry = tmp_path / "run-telemetry.json"
    telemetry.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    terminated = []
    monkeypatch.setattr(client.mod, "HALT_GRACE_SECONDS", 0)
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: True)
    monkeypatch.setattr(
        client.mod, "_terminate_child", lambda pid: terminated.append(pid)
    )
    asyncio.run(client.mod._escalate_halt_pid(12345, telemetry))
    assert terminated == []


def test_start_uses_fake_trainer_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ASR_FAKE_TRAINER", "1")
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200, res.text
    cmd = client.launched[0]
    assert Path(cmd[2]).name == "dev_fake_trainer.py"
    run = _run_json(client, res.json()["run_id"])
    assert run["source"] == "fake"


def test_fake_run_uses_isolated_preflight_cache(client, monkeypatch):
    monkeypatch.setenv("ASR_FAKE_TRAINER", "1")
    res = client.post("/api/train/start", json={})
    assert res.status_code == 200, res.text
    flags = _flags(client.launched[0])
    expected = client.mod.RUNS_DIR / "_shared-fake"
    assert Path(flags["--preflight_cache_dir"]) == expected
    run = _run_json(client, res.json()["run_id"])
    assert run["paths"]["preflight_cache"] == str(
        expected / "baselines" / "preflight.json"
    )


def test_resume_forces_auto_resume_and_drops_checkpoint(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]

    def seed_args(run):
        run["trainer_args"].update({
            "fresh_start": True,
            "auto_resume": False,
            "resume_from_checkpoint": "last",
        })
        return run

    run_state.update_run(run_dir, seed_args)
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    cmd = client.launched[-1]
    assert "--auto_resume" in cmd
    assert "--no-auto_resume" not in cmd
    assert "--resume_from_checkpoint" not in cmd
    assert "--fresh_start" not in cmd


def test_sanitize_output_dir_allows_custom_runs_root(
    client, tmp_path, monkeypatch
):
    outside_runs = tmp_path.parent / f"{tmp_path.name}-custom-runs"
    monkeypatch.setattr(client.mod, "RUNS_DIR", outside_runs)
    target = outside_runs / "20261031-000000-abcdef" / "output"
    assert client.mod._sanitize_output_dir(str(target)) == str(target.resolve())


def test_status_includes_run_and_storage(client):
    client.post("/api/train/start", json={})
    body = client.get("/api/status").json()
    assert body["run"]["run_id"]
    assert body["active_run_id"] == body["run"]["run_id"]
    assert body["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)
    assert body["steps"]["count"] == 0


def test_run_scoped_logs_and_status_use_active_run_files(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "training_live.log").write_text(
        "run-scoped log marker\n", encoding="utf-8"
    )
    client.mod.LOG_FILE.write_text("legacy log marker\n", encoding="utf-8")
    (run_dir / "live_telemetry.json").write_text(
        json.dumps({
            "status": "training",
            "step": 77,
            "max_steps": 100,
            "epoch": 1.5,
            "loss": 0.25,
            "learning_rate": 1e-4,
            "speed_s_per_step": 0.5,
            "eta_seconds": 12,
        }),
        encoding="utf-8",
    )
    client.mod.TELEMETRY_FILE.write_text(
        json.dumps({"status": "training", "step": 1}), encoding="utf-8"
    )

    logs = client.get("/api/logs").json()["logs"]
    assert "run-scoped log marker" in logs
    assert "legacy log marker" not in logs
    body = client.get("/api/status").json()
    assert body["step"] == 77
    assert body["run"]["run_id"] == start["run_id"]


def test_legacy_stop_without_run_reports_not_running(client):
    res = client.post("/api/train/stop", json={})
    assert res.status_code == 200
    assert res.json()["status"] == "not_running"
    assert not client.mod.STOP_SIGNAL_FILE.exists()


def test_resume_reuses_paths_and_appends_attempt(client):
    start = client.post("/api/train/start", json={}).json()
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    run = run_state.load_run(client.mod.RUNS_DIR / start["run_id"])
    assert len(run["attempts"]) == 2
    flags = _flags(client.launched[-1])
    assert flags["--output_dir"] == run["paths"]["output_dir"]
    assert "--auto_resume" in client.launched[-1]


def test_resume_after_training_completed_uses_post_eval_only(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "failed")
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 200, res.text
    assert "--post_eval_only" in client.launched[-1]


def test_resume_rejects_completed_run(client):
    start = client.post("/api/train/start", json={}).json()
    client.mod._is_process_running = lambda pid=None: False
    client.mod.training_process = None
    _write_run(
        client, start["run_id"], status="completed", current_pid=None
    )
    res = client.post(f"/api/runs/{start['run_id']}/resume")
    assert res.status_code == 400
    assert "cannot resume" in res.json()["detail"]
    assert len(client.launched) == 1


def test_stop_writes_run_stop_signal(client):
    start = client.post("/api/train/start", json={}).json()
    res = client.post("/api/train/stop", json={})
    assert res.status_code == 200
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    assert (run_dir / "stop_training.signal").exists()
    assert run_state.load_run(run_dir)["stop_requested_at"] is not None


def test_smoke_status_reports_running_after_start(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "running"


def test_runs_endpoint_returns_empty_list_without_runs(client):
    body = client.get("/api/runs").json()
    assert body["runs"] == []
    assert body["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)


def test_latest_run_is_null_without_runs(client):
    assert client.get("/api/runs/latest").json()["run"] is None


def test_latest_run_without_ledger_falls_back_to_legacy_files(client):
    client.mod.TELEMETRY_FILE.write_text(
        json.dumps({"status": "training", "step": 5}), encoding="utf-8"
    )
    client.mod.BENCHMARKS_FILE.write_text(
        json.dumps([{"epoch": "Epoch 1", "normalized_wer": 15.0}]),
        encoding="utf-8",
    )
    client.mod.LOSS_HISTORY_FILE.write_text(
        json.dumps([{"step": 5, "loss": 0.5}]), encoding="utf-8"
    )

    body = client.get("/api/runs/latest").json()

    assert body["run"] is None
    assert body["telemetry"]["step"] == 5
    assert body["benchmarks"][0]["epoch"] == "Epoch 1"
    assert body["loss_history"] == [{"step": 5, "loss": 0.5}]


def test_run_endpoints_reject_invalid_and_missing_ids(client):
    for run_id in ("not-a-run", "20261031-000000-abcdef"):
        assert client.get(f"/api/runs/{run_id}").status_code == 404
        assert client.get(f"/api/runs/{run_id}/steps").status_code == 404
    assert client.post("/api/runs/not-a-run/merge").status_code == 404
    assert (
        client.post("/api/runs/20261031-000000-abcdef/merge").status_code == 404
    )


def test_run_payload_includes_adapter_exists(client):
    start = client.post("/api/train/start", json={}).json()
    payload = client.get("/api/runs/latest").json()
    assert payload["run"]["run_id"] == start["run_id"]
    assert payload["adapter_exists"] is False
    assert payload["process_running"] is True
    assert payload["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)
    assert payload["steps"] == {
        "count": 0,
        "last_step": None,
        "last_saved_at": None,
    }
    assert payload["telemetry"]["status"] == "initializing"
    assert payload["benchmarks"] == []
    assert payload["loss_history"] == []

    _seed_adapter(client, start["run_id"])
    payload = client.get(f"/api/runs/{start['run_id']}").json()
    assert payload["adapter_exists"] is True


def test_run_steps_endpoint_reads_and_caps_tail(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    for step in range(1, 4):
        run_state.append_step(run_dir, {"step": step, "loss": 1.0 / step})

    body = client.get(f"/api/runs/{start['run_id']}/steps?tail=2").json()
    assert body["run_id"] == start["run_id"]
    assert body["total"] == 3
    assert [row["step"] for row in body["steps"]] == [2, 3]

    body = client.get(f"/api/runs/{start['run_id']}/steps?tail=99999").json()
    assert len(body["steps"]) == 3

    body = client.get(f"/api/runs/{start['run_id']}/steps?tail=0").json()
    assert body["steps"] == []


def test_status_and_websocket_include_adapter_and_storage(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    body = client.get("/api/status").json()
    assert body["adapter_exists"] is False
    assert body["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)

    _seed_adapter(client, start["run_id"])
    monkeypatch.setattr(client.mod, "get_gpu_info", lambda: {})
    with client.websocket_connect("/ws/live") as ws:
        payload = ws.receive_json()
    assert payload["run"]["run_id"] == start["run_id"]
    assert payload["steps"]["count"] == 0
    assert payload["storage"]["runs_dir"] == str(client.mod.RUNS_DIR)
    assert payload["adapter_exists"] is True


class _ExitedProcess:
    def __init__(self, code, pid=999_999):
        self._code = code
        self.pid = pid

    def poll(self):
        return self._code


def test_training_nonzero_exit_marks_run_failed(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = _ExitedProcess(7)

    assert client.mod._training_running() is False

    run = run_state.load_run(run_dir)
    assert run["status"] == "failed"
    assert run["failed_stage"] == "training"
    assert run["current_pid"] is None
    assert run["attempts"][-1]["exit_code"] == 7
    assert run["attempts"][-1]["reason"] == "nonzero_exit"
    assert run["stages"][1]["status"] == "failed"
    assert client.mod.last_exit_code == 7


def test_training_exit_zero_with_terminal_stages_marks_completed(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = _ExitedProcess(0)

    assert client.mod._training_running() is False
    assert run_state.load_run(run_dir)["status"] == "completed"


def test_training_exit_after_stop_request_marks_stopped(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    run_state.set_stage(run_dir, "training", "running")
    assert client.post("/api/train/stop", json={}).status_code == 200
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = _ExitedProcess(-15)

    assert client.mod._training_running() is False
    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    assert run["attempts"][-1]["reason"] == "halted"
    assert run["stages"][1]["status"] == "stopped"


def test_terminal_trainer_exit_closes_attempt_and_clears_pid(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")
    # The trainer writes its terminal status before the process exits.
    run_state.set_status(run_dir, "completed")
    assert run_state.load_run(run_dir)["current_pid"] is not None
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = _ExitedProcess(0)

    assert client.mod._training_running() is False

    run = run_state.load_run(run_dir)
    assert run["status"] == "completed"
    assert run["current_pid"] is None
    attempt = run["attempts"][-1]
    assert attempt["ended_at"] is not None
    assert attempt["exit_code"] == 0
    assert attempt["reason"] == "completed"
    assert not (run_dir / "training.pid").exists()


def test_completed_run_with_reused_live_pid_is_not_active(client, monkeypatch):
    run = _create_run(
        client,
        pid=os.getpid(),
        status="completed",
        run_id="20261030-194700-dedede",
    )
    monkeypatch.setattr(
        client.mod, "_is_process_running", lambda pid=None: pid == os.getpid()
    )

    client.mod._refresh_runs()

    assert client.mod.ACTIVE_RUN_ID is None
    assert client.mod._active_run_process_running() is False
    saved = run_state.load_run(client.mod.RUNS_DIR / run["run_id"])
    assert client.mod._run_process_running(saved) is False


def test_refresh_detects_pid_file_only_cli_run(client, monkeypatch):
    run_id = "20261030-194700-c1c1c1"
    run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_dir = client.mod.RUNS_DIR / run_id
    run_state.set_status(run_dir, "running")
    (run_dir / "training.pid").write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(
        client.mod, "_is_process_running", lambda pid=None: pid == os.getpid()
    )

    client.mod._refresh_runs()

    assert client.mod.ACTIVE_RUN_ID == run_id
    assert client.mod._active_run_process_running() is True


def test_derive_status_maps_running_post_eval_to_evaluating(client):
    run = {
        "status": "running",
        "stages": [
            {"name": "preflight", "status": "completed"},
            {"name": "training", "status": "completed"},
            {"name": "post_eval", "status": "running"},
        ],
    }
    derived = client.mod._derive_status({"status": "completed"}, True, run)
    assert derived["status"] == "evaluating_benchmark"


def test_terminate_child_refuses_own_process_group(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client.mod.os, "getpgid", lambda pid: 1234 if pid in (0, 4242) else 9999
    )
    monkeypatch.setattr(
        client.mod.os, "killpg", lambda *args: calls.append(("killpg", args))
    )
    monkeypatch.setattr(
        client.mod.os, "kill", lambda *args: calls.append(("kill", args))
    )

    client.mod._terminate_child(4242)

    assert calls == []


def test_terminate_child_falls_back_for_non_group_leader(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client.mod.os, "getpgid", lambda pid: 1234 if pid == 0 else 9999
    )
    monkeypatch.setattr(
        client.mod.os, "killpg", lambda *args: calls.append(("killpg", args))
    )
    monkeypatch.setattr(
        client.mod.os, "kill", lambda *args: calls.append(("kill", args))
    )

    client.mod._terminate_child(4243)

    assert calls == [("kill", (4243, client.mod.signal.SIGTERM))]


def test_escalate_halt_pid_kills_running_post_eval_despite_completed_telemetry(
    client, tmp_path, monkeypatch
):
    run_id = "20261030-194700-e5e5e5"
    run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_dir = client.mod.RUNS_DIR / run_id
    run_state.start_attempt(run_dir, 12345)
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "running")
    run_state.set_status(run_dir, "running")
    telemetry = tmp_path / "run-telemetry.json"
    telemetry.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    terminated = []
    monkeypatch.setattr(client.mod, "HALT_GRACE_SECONDS", 0)
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: True)
    monkeypatch.setattr(
        client.mod, "_terminate_child", lambda pid: terminated.append(pid)
    )

    asyncio.run(
        client.mod._escalate_halt_pid(12345, telemetry, run_dir)
    )

    assert terminated == [12345]


def test_escalate_halt_kills_running_post_eval_after_grace(client, tmp_path, monkeypatch):
    run_id = "20261030-194700-a7a7a7"
    run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_dir = client.mod.RUNS_DIR / run_id
    run_state.start_attempt(run_dir, 24680)
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "running")
    run_state.set_status(run_dir, "running")
    telemetry = tmp_path / "run-telemetry.json"
    telemetry.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    terminated = []

    class _RunningProc:
        pid = 24680

        def poll(self):
            return None

    monkeypatch.setattr(client.mod, "HALT_GRACE_SECONDS", 0)
    monkeypatch.setattr(
        client.mod, "_terminate_child", lambda pid: terminated.append(pid)
    )

    asyncio.run(client.mod._escalate_halt(_RunningProc(), telemetry, run_dir))

    assert terminated == [24680]


def test_escalate_halt_pid_skips_reused_pid(client, tmp_path, monkeypatch):
    run_id = "20261030-194700-f6f6f6"
    run_state.create_run(
        client.mod.RUNS_DIR, config={}, trainer_args={}, run_id=run_id
    )
    run_dir = client.mod.RUNS_DIR / run_id
    run_state.start_attempt(run_dir, 30001)
    run_state.set_stage(run_dir, "post_eval", "running")
    run_state.set_status(run_dir, "running")
    telemetry = tmp_path / "run-telemetry.json"
    telemetry.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    terminated = []
    monkeypatch.setattr(client.mod, "HALT_GRACE_SECONDS", 0)
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: True)
    monkeypatch.setattr(
        client.mod, "_terminate_child", lambda pid: terminated.append(pid)
    )

    asyncio.run(
        client.mod._escalate_halt_pid(30002, telemetry, run_dir)
    )

    assert terminated == []


def test_status_polls_child_before_refresh(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = _ExitedProcess(1)

    body = client.get("/api/status").json()
    assert body["run"]["run_id"] == start["run_id"]
    assert body["run"]["status"] == "failed"
    assert body["process_running"] is False


def test_websocket_polls_child_before_refresh(client, monkeypatch):
    start = client.post("/api/train/start", json={}).json()
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    monkeypatch.setattr(client.mod, "get_gpu_info", lambda: {})
    client.mod.training_process = _ExitedProcess(1)

    with client.websocket_connect("/ws/live") as ws:
        payload = ws.receive_json()
    assert payload["run"]["run_id"] == start["run_id"]
    assert payload["run"]["status"] == "failed"
    assert payload["telemetry"]["process_running"] is False


def test_training_running_reconciles_dead_run_pid_file(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="running", run_id="20261030-194700-bbbbbc"
    )
    run_dir = client.mod.RUNS_DIR / run["run_id"]
    (run_dir / "training.pid").write_text("999999", encoding="utf-8")
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = None
    client.mod.training_run_id = None
    client.mod.ACTIVE_RUN_ID = run["run_id"]
    client.mod.CURRENT_RUN = run_state.load_run(run_dir)
    client.mod.LATEST_RUN = client.mod.CURRENT_RUN

    assert client.mod._training_running() is False
    assert run_state.load_run(run_dir)["status"] == "interrupted"
    assert not (run_dir / "training.pid").exists()


def test_startup_recovery_reconciles_dead_run(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="running", run_id="20261030-194700-bbbbbd"
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)

    client.mod._recover_runs_on_startup()
    assert run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["status"] == (
        "interrupted"
    )


def test_startup_recovery_logs_failures(client, monkeypatch, capsys):
    def boom():
        raise RuntimeError("startup boom")

    monkeypatch.setattr(client.mod, "_training_running", boom)
    client.mod._recover_runs_on_startup()
    assert "startup training poll failed" in capsys.readouterr().err


def test_lifespan_starts_and_stops_monitor(client):
    from fastapi.testclient import TestClient

    with TestClient(client.mod.app):
        task = client.mod._monitor_task
        assert task is not None
        assert not task.done()
    assert client.mod._monitor_task is None


def _terminal_stages(run_dir):
    """Complete preflight/training/post-eval so R12 permits a live-target merge."""
    run_state.set_stage(run_dir, "preflight", "completed")
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")


def test_merge_completes_on_zero_exit_with_artifacts(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    _seed_adapter(client, start["run_id"])
    _terminal_stages(run_dir)
    assert client.post(f"/api/runs/{start['run_id']}/merge").status_code == 200

    _write_merge_artifacts(client, start["run_id"])
    client.mod.merge_process = _ExitedProcess(0)

    assert client.mod._merge_running() is False
    run = run_state.load_run(run_dir)
    merge_stage = run["stages"][3]
    assert merge_stage["status"] == "completed"
    assert merge_stage["exit_code"] == 0
    assert run["merge_pid"] is None


def test_merge_fails_on_zero_exit_without_artifacts(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    _seed_adapter(client, start["run_id"])
    _terminal_stages(run_dir)
    assert client.post(f"/api/runs/{start['run_id']}/merge").status_code == 200

    client.mod.merge_process = _ExitedProcess(0)

    assert client.mod._merge_running() is False
    run = run_state.load_run(run_dir)
    merge_stage = run["stages"][3]
    assert merge_stage["status"] == "failed"
    assert "artifact" in merge_stage["detail"].lower()
    assert "log_tail" in merge_stage
    assert run["merge_pid"] is None


def test_merge_fails_on_nonzero_exit(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    _seed_adapter(client, start["run_id"])
    _terminal_stages(run_dir)
    assert client.post(f"/api/runs/{start['run_id']}/merge").status_code == 200

    _write_merge_artifacts(client, start["run_id"])
    client.mod.merge_process = _ExitedProcess(3)

    assert client.mod._merge_running() is False
    run = run_state.load_run(run_dir)
    assert run["stages"][3]["status"] == "failed"
    assert run["stages"][3]["exit_code"] == 3


def _mark_merge_running(client, run_id, pid=999999):
    run_dir = client.mod.RUNS_DIR / run_id

    def mutate(run):
        run["merge_pid"] = pid
        return run

    run_state.update_run(run_dir, mutate)
    run_state.set_stage(run_dir, "merge", "running", detail="test")
    return run_dir


def test_merge_recovery_uses_dead_pid_artifacts(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-cccccc"
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run_dir = _mark_merge_running(client, run["run_id"])

    merged = _write_merge_artifacts(client, run["run_id"])
    stage = run_state.load_run(run_dir)["stages"][3]
    # A recorded dead PID is enough even when the files are older than the
    # stage start (only the no-PID path needs the mtime check).
    _age_artifacts(merged, float(stage["started_at"]) - 60.0)
    assert client.mod._merge_running() is False
    assert run_state.load_run(run_dir)["stages"][3]["status"] == "completed"


def test_merge_recovery_marks_dead_pid_failed_without_artifacts(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-dddddd"
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run_dir = _mark_merge_running(client, run["run_id"])

    assert client.mod._merge_running() is False
    assert run_state.load_run(run_dir)["stages"][3]["status"] == "failed"


def _age_artifacts(merged_dir, mtime):
    for name in ("config.json", "model.safetensors"):
        os.utime(merged_dir / name, (mtime, mtime))


def test_merge_recovery_rejects_stale_artifacts_without_pid(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-777777"
    )
    merged = _write_merge_artifacts(client, run["run_id"])
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run_dir = _mark_merge_running(client, run["run_id"], pid=None)
    stage = run_state.load_run(run_dir)["stages"][3]
    _age_artifacts(merged, float(stage["started_at"]) - 60.0)

    assert client.mod._merge_running() is False
    stage = run_state.load_run(run_dir)["stages"][3]
    assert stage["status"] == "failed"
    assert "predate" in stage["detail"]


def test_merge_recovery_accepts_fresh_artifacts_without_pid(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-888888"
    )
    merged = _write_merge_artifacts(client, run["run_id"])
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    run_dir = _mark_merge_running(client, run["run_id"], pid=None)
    stage = run_state.load_run(run_dir)["stages"][3]
    _age_artifacts(merged, float(stage["started_at"]) + 1.0)

    assert client.mod._merge_running() is False
    assert run_state.load_run(run_dir)["stages"][3]["status"] == "completed"


def test_merge_artifacts_reject_adapter_weights(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-999999"
    )
    merged = _write_merge_artifacts(
        client, run["run_id"], weights="adapter_model.safetensors"
    )
    assert (
        client.mod._merge_artifacts_complete(
            run_state.load_run(client.mod.RUNS_DIR / run["run_id"])
        )
        is False
    )
    (merged / "model.safetensors").write_text("", encoding="utf-8")
    assert client.mod._merge_artifacts_complete(
        run_state.load_run(client.mod.RUNS_DIR / run["run_id"])
    )


def test_merge_clears_stale_pid_before_start(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-aaaaab"
    )
    _seed_adapter(client, run["run_id"])
    run_dir = client.mod.RUNS_DIR / run["run_id"]
    run_state.set_stage(run_dir, "training", "completed")
    (run_dir / "merge.pid").write_text("12345", encoding="utf-8")
    run_state.update_run(
        run_dir, lambda current: {**current, "merge_pid": 12345}
    )

    res = client.post(f"/api/runs/{run['run_id']}/merge")
    assert res.status_code == 200, res.text
    body = res.json()
    assert (run_dir / "merge.pid").read_text(encoding="utf-8") == str(body["pid"])
    assert run_state.load_run(run_dir)["merge_pid"] == body["pid"]


def test_merge_rejected_when_legacy_trainer_pid_is_live(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-aaaaac"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    client.mod.training_process = None
    client.mod.training_run_id = None
    client.mod.PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(
        client.mod, "_is_process_running", lambda pid=None: pid == os.getpid()
    )

    res = client.post(f"/api/runs/{run['run_id']}/merge")
    assert res.status_code == 400
    assert client.launched == []


def test_merge_writes_log_pid_and_manifest_pid(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-555555"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    res = client.post(f"/api/runs/{run['run_id']}/merge")
    assert res.status_code == 200, res.text
    body = res.json()
    run_dir = client.mod.RUNS_DIR / run["run_id"]
    assert body["status"] == "started"
    assert body["merged_dir"] == str(run_dir / "merged")
    assert (run_dir / "merge.log").is_file()
    assert (run_dir / "merge.pid").read_text(encoding="utf-8") == str(body["pid"])
    assert run_state.load_run(run_dir)["merge_pid"] == body["pid"]
    cmd = client.launched[-1]
    assert _flag_value(cmd, "--adapter_path") == str(
        run_dir / "output" / "best_adapter"
    )
    assert _flag_value(cmd, "--output_dir") == str(run_dir / "merged")


def test_merge_rejected_without_adapter(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-111111"
    )
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    res = client.post(f"/api/runs/{run['run_id']}/merge")
    assert res.status_code == 400
    assert "best_adapter" in res.json()["detail"]
    assert client.launched == []


def test_merge_rejected_while_target_post_eval_is_running(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-666666"
    )
    run_dir = client.mod.RUNS_DIR / run["run_id"]
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "running")

    res = client.post(f"/api/runs/{run['run_id']}/merge")
    assert res.status_code == 400
    assert client.launched == []


def test_merge_rejected_while_another_run_is_active(client):
    target = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-222222"
    )
    _seed_adapter(client, target["run_id"])
    run_state.set_stage(
        client.mod.RUNS_DIR / target["run_id"], "training", "completed"
    )
    _create_run(
        client, pid=os.getpid(), status="running", run_id="20261030-194700-333333"
    )

    res = client.post(f"/api/runs/{target['run_id']}/merge")
    assert res.status_code == 400
    assert client.launched == []


def test_merge_blocks_start_smoke_resume_and_second_merge(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-eeeeee"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    interrupted = _create_run(
        client, pid=999999, status="interrupted", run_id="20261030-194700-444444"
    )
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 200
    assert len(client.launched) == 1

    assert client.post("/api/train/start", json={}).status_code == 400
    assert client.post("/api/train/smoke", json={}).status_code == 400
    assert (
        client.post(f"/api/runs/{interrupted['run_id']}/resume").status_code
        == 400
    )
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 400
    assert len(client.launched) == 1


def test_merge_uses_real_script_when_fake_disabled(client):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-ffffff"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 200
    assert Path(client.launched[-1][2]).name == "merge_lora.py"
    assert "--base_model" in client.launched[-1]


def test_merge_uses_fake_script_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ASR_FAKE_TRAINER", "1")
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-abcdef"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(client.mod.RUNS_DIR / run["run_id"], "training", "completed")
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 200
    assert Path(client.launched[-1][2]).name == "dev_fake_merge.py"


def test_monitor_task_start_is_idempotent(client):
    async def scenario():
        first = client.mod._ensure_monitor_task()
        second = client.mod._ensure_monitor_task()
        assert first is not None
        assert first is second
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        client.mod._monitor_task = None

    asyncio.run(scenario())


def test_smoke_controls_are_present_in_static_ui():
    html = (ROOT / "server" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "server" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="btnSmoke"' in html
    assert 'id="smokeResult"' in html
    assert 'id="smokeStatus"' in html
    assert 'id="smokeLog"' in html
    assert "/api/train/smoke" in js
    assert "/api/train/smoke/status" in js


def test_smoke_status_transient_failure_keeps_polling():
    js = (ROOT / "server" / "static" / "app.js").read_text(encoding="utf-8")
    assert "Smoke status unavailable" in js
    assert "scheduleSmokePoll(5000)" in js
    assert "if (el.btnSmoke)" in js


def test_refresh_cached_reuses_scan_within_ttl(client, monkeypatch):
    calls = []
    orig = client.mod.run_state.reconcile_all

    def counting(root, **kwargs):
        calls.append(1)
        return orig(root, **kwargs)

    monkeypatch.setattr(client.mod.run_state, "reconcile_all", counting)
    client.mod._refresh_runs_cached()
    client.mod._refresh_runs_cached()
    assert len(calls) == 1
    client.mod._refresh_runs()
    assert len(calls) == 2


def test_spawn_and_stop_invalidate_cached_scan(client, monkeypatch):
    calls = []
    orig = client.mod.run_state.reconcile_all

    def counting(root, **kwargs):
        calls.append(1)
        return orig(root, **kwargs)

    monkeypatch.setattr(client.mod.run_state, "reconcile_all", counting)
    client.mod._refresh_runs_cached()
    assert len(calls) == 1

    # A spawn must force the next cached read to rescan so the new run is
    # visible immediately instead of serving the pre-spawn summary.
    start = client.post("/api/train/start", json={}).json()
    n_after_post = len(calls)  # the busy-check refresh inside POST
    summary = client.mod._refresh_runs_cached()
    assert len(calls) == n_after_post + 1
    assert summary["active_run"]["run_id"] == start["run_id"]
    # A second read inside the TTL is served from the cache.
    assert client.mod._refresh_runs_cached()["active_run"]["run_id"] == (
        start["run_id"]
    )
    assert len(calls) == n_after_post + 1

    # A stop must likewise invalidate: the cached read that follows has to
    # rescan instead of serving the pre-stop summary.
    assert client.post("/api/train/stop", json={}).status_code == 200
    n_after_stop = len(calls)  # stop's own forced refresh
    client.mod._refresh_runs_cached()
    assert len(calls) == n_after_stop + 1
    assert (
        run_state.load_run(client.mod.RUNS_DIR / start["run_id"])[
            "stop_requested_at"
        ]
        is not None
    )


def test_merge_start_invalidates_cached_scan(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-ccccce"
    )
    _seed_adapter(client, run["run_id"])
    run_state.set_stage(
        client.mod.RUNS_DIR / run["run_id"], "training", "completed"
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    client.mod.training_process = None

    calls = []
    orig = client.mod.run_state.reconcile_all

    def counting(root, **kwargs):
        calls.append(1)
        return orig(root, **kwargs)

    monkeypatch.setattr(client.mod.run_state, "reconcile_all", counting)
    client.mod._refresh_runs_cached()
    before = len(calls)
    assert client.post(f"/api/runs/{run['run_id']}/merge").status_code == 200
    assert len(calls) == before + 1  # the pre-flight forced refresh in POST
    client.mod._refresh_runs_cached()
    assert len(calls) == before + 2
    assert (
        run_state.load_run(client.mod.RUNS_DIR / run["run_id"])["stages"][3][
            "status"
        ]
        == "running"
    )


def test_refresh_cached_returns_shallow_copy(client):
    client.post("/api/train/start", json={})
    first = client.mod._refresh_runs_cached()
    first["active_run"] = None
    first["injected"] = True
    second = client.mod._refresh_runs_cached()
    assert "injected" not in second
    assert second["active_run"] is not None


def test_duplicate_warning_is_rate_limited(client, monkeypatch, capsys):
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: True)
    _create_run(
        client, pid=111111, status="running", run_id="20261030-194700-aaaaaa"
    )
    _create_run(
        client, pid=222222, status="running", run_id="20261030-195700-bbbbbb"
    )
    clock = {"now": 5000.0}
    monkeypatch.setattr(client.mod.time, "monotonic", lambda: clock["now"])
    client.mod._refresh_runs()
    assert "duplicate" in capsys.readouterr().err.lower()
    # A second forced rescan inside the TTL must not repeat the warning.
    client.mod._refresh_runs()
    assert "duplicate" not in capsys.readouterr().err.lower()
    clock["now"] += 61.0
    client.mod._refresh_runs()
    assert "duplicate" in capsys.readouterr().err.lower()


def test_log_exception_state_is_bounded(client, capsys):
    for index in range(client.mod._LOG_EXCEPTION_MAX_CONTEXTS + 25):
        client.mod._log_exception(f"bounded-ctx-{index}")
    capsys.readouterr()
    assert len(client.mod._log_exception_state) <= (
        client.mod._LOG_EXCEPTION_MAX_CONTEXTS
    )


def test_merge_scan_cached_reuses_no_child_scan(client, monkeypatch):
    calls = []
    orig_list = client.mod.run_state.list_runs

    def counting(root, limit=None):
        calls.append(1)
        return orig_list(root, limit=limit)

    monkeypatch.setattr(client.mod.run_state, "list_runs", counting)
    client.mod.merge_process = None
    client.mod._merge_running()
    client.mod._merge_running()
    assert len(calls) == 1
    client.mod._merge_running(force=True)
    assert len(calls) == 2


def test_log_exception_rate_limits(client, capsys):
    ctx = "rate-limit-test-ctx-unique-1"
    client.mod._log_exception(ctx)
    client.mod._log_exception(ctx)
    err = capsys.readouterr().err
    assert err.count(ctx) == 1


def test_log_exception_reports_suppressed_count(client, monkeypatch, capsys):
    ctx = "rate-limit-suppressed-ctx-unique-1"
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        client.mod.time, "monotonic", lambda: clock["now"]
    )
    client.mod._log_exception(ctx)
    assert capsys.readouterr().err.count(ctx) == 1
    client.mod._log_exception(ctx)
    assert capsys.readouterr().err.count(ctx) == 0
    clock["now"] += 61.0
    client.mod._log_exception(ctx)
    err = capsys.readouterr().err
    assert ctx in err
    assert "suppress" in err.lower()


def test_duplicate_live_pid_produces_warning(client, monkeypatch, capsys):
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: True)
    _create_run(
        client, pid=111111, status="running", run_id="20261030-194700-aaaaaa"
    )
    _create_run(
        client, pid=222222, status="running", run_id="20261030-195700-bbbbbb"
    )
    summary = client.mod.run_state.reconcile_all(
        client.mod.RUNS_DIR, is_alive=lambda pid: True
    )
    assert "duplicates" in summary
    assert len(summary["duplicates"]) == 1
    capsys.readouterr()
    client.mod._refresh_runs()
    assert "duplicate" in capsys.readouterr().err.lower()


def test_finish_merge_missing_started_at_detail(client, monkeypatch):
    run = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-777778"
    )
    run_dir = _mark_merge_running(client, run["run_id"], pid=None)
    _write_merge_artifacts(client, run["run_id"])

    def clear_started(current):
        for stage in current.get("stages") or []:
            if stage.get("name") == "merge":
                stage.pop("started_at", None)
        return current

    run_state.update_run(run_dir, clear_started)
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)

    assert client.mod._merge_running() is False
    stage = run_state.load_run(run_dir)["stages"][3]
    assert stage["status"] == "failed"
    assert "missing" in stage["detail"].lower() or "start time" in stage["detail"].lower()


def test_run_paths_honors_custom_results_dir(client, tmp_path):
    run_id = "20261030-194700-aaaaab"
    custom = tmp_path / "custom-results"
    custom.mkdir()
    run_state.create_run(
        client.mod.RUNS_DIR,
        config={},
        trainer_args={},
        run_id=run_id,
        paths={"results_dir": str(custom)},
    )
    run = run_state.load_run(client.mod.RUNS_DIR / run_id)
    paths = client.mod._run_paths(run)
    assert paths["telemetry_file"] == custom / "live_telemetry.json"
    assert paths["benchmarks_file"] == custom / "epoch_benchmarks.json"
    assert paths["loss_history_file"] == custom / "loss_history.json"
    assert paths["stop_signal_file"] == custom / "stop_training.signal"
    assert paths["log_file"].parent == client.mod.RUNS_DIR / run_id
    assert paths["pid_file"].parent == client.mod.RUNS_DIR / run_id


def test_steps_summary_uses_active_telemetry_count(client):
    start = client.post("/api/train/start", json={}).json()
    run_dir = client.mod.RUNS_DIR / start["run_id"]
    (run_dir / "live_telemetry.json").write_text(
        json.dumps(
            {
                "status": "training",
                "step": 42,
                "step_count": 42,
                "updated_at": 123.0,
            }
        ),
        encoding="utf-8",
    )
    client.mod._refresh_runs()
    run = run_state.load_run(run_dir)
    summary = client.mod._steps_summary(run)
    assert summary["count"] == 42

    historical = _create_run(
        client, pid=999999, status="completed", run_id="20261030-194700-cccccd"
    )
    historical_dir = client.mod.RUNS_DIR / historical["run_id"]
    run_state.set_progress(historical_dir, step=7, step_count=7, last_step_at=1.0)
    (historical_dir / "live_telemetry.json").write_text(
        json.dumps({"status": "completed", "step": 99, "step_count": 99}),
        encoding="utf-8",
    )
    historical_run = run_state.load_run(historical_dir)
    historical_summary = client.mod._steps_summary(historical_run)
    assert historical_summary["count"] == 7
