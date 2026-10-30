"""Offline tests for server training control and benchmark endpoints.

Training is never launched: the audio-backend check is stubbed and subprocess.Popen
records the command instead. Benchmark tests only read files under tmp_path.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

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
    spec = importlib.util.spec_from_file_location("server_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_app"] = mod
    spec.loader.exec_module(mod)

    launched = []

    class _FakeProcess:
        pid = 999_999

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
    return client


def _flags(cmd):
    return dict(zip(cmd[:-1], cmd[1:]))


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


def test_smoke_status_reports_running_after_start(client):
    assert client.post("/api/train/smoke", json={}).status_code == 200
    res = client.get("/api/train/smoke/status")
    assert res.json()["status"] == "running"


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
