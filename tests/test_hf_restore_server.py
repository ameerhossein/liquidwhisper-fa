# tests/test_hf_restore_server.py
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = "20261031-120000-abcdef"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ASR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ASR_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("ASR_HF_RUN_REPO", "me/run")
    monkeypatch.delenv("ASR_COCKPIT_TOKEN", raising=False)
    spec = importlib.util.spec_from_file_location("server_hf_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_hf_app"] = mod
    spec.loader.exec_module(mod)
    launched = []

    class FakeProcess:
        pid = os.getpid()

        def poll(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or FakeProcess())
    client = TestClient(mod.app)
    client.mod = mod
    client.launched = launched
    yield client
    mod._close_hf_restore_log_fd()


def test_hf_config_redacts_token(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    data = client.get("/api/hf/config").json()
    assert data["enabled"] is True
    assert data["repo_id"] == "me/run"
    assert "hf_secret" not in json.dumps(data)


def test_hf_config_returns_default_repo_when_authenticated(client, monkeypatch):
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "carol")
    data = client.get("/api/hf/config").json()
    assert data["repo_id"] == "carol/liquidwhisper-neyshekar"


def test_restore_rejects_invalid_run_id(client):
    assert client.post("/api/hf/restore", json={"run_id": "../bad"}).status_code == 400


def test_restore_refuses_existing_local_run(client):
    run_dir = client.mod.RUNS_DIR / RUN
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}")
    res = client.post("/api/hf/restore", json={"run_id": RUN})
    assert res.status_code == 409


def test_restore_spawns_script_and_writes_initial_status(client):
    res = client.post(
        "/api/hf/restore",
        json={"run_id": RUN, "repo": "me/run", "include_merged": True},
    )
    assert res.status_code == 200, res.text
    cmd = client.launched[-1]
    assert "--run" in cmd and RUN in cmd
    assert "--repo" in cmd and "me/run" in cmd
    assert "--include-merged" in cmd
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "running"
    assert status["run_id"] == RUN


def test_restore_status_reads_file_and_log_tail(client):
    client.mod._hf_restore_status_file().write_text(
        json.dumps(
            {
                "state": "completed",
                "run_id": RUN,
                "message": "done",
                "checkpoint_step": 1000,
                "updated_at": 1.0,
            }
        )
    )
    client.mod._hf_restore_log_file().write_text("line 1\nline 2\n")
    data = client.get("/api/hf/restore/status").json()
    assert data["state"] == "completed"
    assert data["log_tail"] == ["line 1", "line 2"]


def test_run_payload_includes_hf_sync(client):
    from src import run_state

    run_state.create_run(client.mod.RUNS_DIR, run_id=RUN, config={}, trainer_args={})
    (client.mod.RUNS_DIR / RUN / "hf_sync.json").write_text(
        json.dumps({"status": "idle", "last_success_step": 1000})
    )
    data = client.get(f"/api/runs/{RUN}").json()
    assert data["hf_sync"]["last_success_step"] == 1000


def test_stale_running_restore_status_becomes_interrupted(client, monkeypatch):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "running", "run_id": RUN, "pid": 999999})
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    assert client.mod._hf_restore_running() is False
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "interrupted"


RESTORE_PHASES = (
    "running",
    "resolving",
    "listing",
    "ledger",
    "checkpoint",
    "verify",
    "future_phase",
)


@pytest.mark.parametrize("state", RESTORE_PHASES)
def test_live_restore_phase_is_treated_as_running(client, state):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": state, "run_id": RUN, "pid": os.getpid()})
    )
    assert client.mod._hf_restore_running() is True


@pytest.mark.parametrize("state", RESTORE_PHASES)
def test_dead_restore_phase_becomes_interrupted(client, monkeypatch, state):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": state, "run_id": RUN, "pid": 999999})
    )
    monkeypatch.setattr(client.mod, "_is_process_running", lambda pid=None: False)
    assert client.mod._hf_restore_running() is False
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "interrupted"
    assert status["run_id"] == RUN
    assert status["message"]


def test_restore_refused_while_restart_recovered_phase_is_live(client):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "checkpoint", "run_id": RUN, "pid": os.getpid()})
    )
    assert client.mod._any_job_running() is True
    res = client.post(
        "/api/hf/restore", json={"run_id": "20261031-130000-abcdef"}
    )
    assert res.status_code == 400
    assert client.launched == []


# --- hardening around the global constraints -------------------------------


def test_restore_requires_cockpit_token(client, monkeypatch):
    monkeypatch.setenv("ASR_COCKPIT_TOKEN", "cockpit-secret")
    res = client.post("/api/hf/restore", json={"run_id": RUN})
    assert res.status_code == 401
    assert client.launched == []


def test_restore_refuses_while_another_job_is_running(client, monkeypatch):
    monkeypatch.setattr(client.mod, "_any_job_running", lambda: True)
    res = client.post("/api/hf/restore", json={"run_id": RUN})
    assert res.status_code == 400
    assert client.launched == []


def test_another_restore_is_refused_while_child_is_live(client):
    assert client.post("/api/hf/restore", json={"run_id": RUN}).status_code == 200
    second = client.post(
        "/api/hf/restore", json={"run_id": "20261031-130000-abcdef"}
    )
    assert second.status_code == 400
    assert len(client.launched) == 1


def test_restore_command_points_at_script_runs_root_and_status_file(client):
    assert client.post("/api/hf/restore", json={"run_id": RUN}).status_code == 200
    cmd = client.launched[-1]
    assert cmd[0] == sys.executable
    assert "scripts/hf_restore.py" in cmd
    runs_root = cmd[cmd.index("--runs-root") + 1]
    status_file = cmd[cmd.index("--status-file") + 1]
    assert runs_root == str(client.mod.RUNS_DIR)
    assert status_file == str(client.mod._hf_restore_status_file())


def test_completed_restore_invalidates_run_cache(client):
    client.mod._refresh_runs()
    client.mod._refresh_cache_at = time.monotonic()
    assert client.mod._refresh_cache_summary is not None

    class _Exited:
        pid = 999_999

        def poll(self):
            return 0

    client.mod.hf_restore_process = _Exited()
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "completed", "run_id": RUN})
    )
    assert client.mod._hf_restore_running() is False
    assert client.mod._refresh_cache_summary is None


def test_restore_launch_failure_records_failed_status(client, monkeypatch):
    def _fail(cmd, **kwargs):
        raise OSError("could not exec the restore child")

    monkeypatch.setattr(client.mod.subprocess, "Popen", _fail)
    res = client.post("/api/hf/restore", json={"run_id": RUN})
    assert res.status_code == 500
    status = json.loads(client.mod._hf_restore_status_file().read_text())
    assert status["state"] == "failed"
    assert "could not exec" in status["message"]


def test_restore_status_tail_is_capped_at_40_lines(client):
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "failed", "run_id": RUN})
    )
    client.mod._hf_restore_log_file().write_text(
        "".join(f"line {index}\n" for index in range(60))
    )
    data = client.get("/api/hf/restore/status").json()
    assert len(data["log_tail"]) == 40
    assert data["log_tail"][-1] == "line 59"


def test_restore_status_redacts_token_in_log_tail(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    client.mod._hf_restore_status_file().write_text(
        json.dumps({"state": "failed", "message": "bad hf_secret"})
    )
    client.mod._hf_restore_log_file().write_text("auth with hf_secret failed\n")
    data = client.get("/api/hf/restore/status").json()
    assert "hf_secret" not in json.dumps(data)


def test_hf_runs_lists_and_caches_per_repo(client, monkeypatch):
    rows = [
        {
            "run_id": RUN,
            "status": "completed",
            "last_step": 1000,
            "updated_at": 1.0,
            "has_checkpoint": True,
        }
    ]
    calls = []

    def _list(api, repo_id, **kwargs):
        calls.append(repo_id)
        return rows

    monkeypatch.setattr(client.mod.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(client.mod.hf_sync, "list_remote_runs", _list)
    first = client.get("/api/hf/runs?repo=me/run")
    second = client.get("/api/hf/runs?repo=me/run")
    assert first.status_code == 200, first.text
    assert first.json() == {"runs": rows, "repo_id": "me/run"}
    assert second.json() == first.json()
    assert calls == ["me/run"]


def test_hf_runs_failure_is_502_without_leaking_token(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")

    def _fail(api, repo_id, **kwargs):
        raise RuntimeError("authentication failed for token hf_secret")

    monkeypatch.setattr(client.mod.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(client.mod.hf_sync, "list_remote_runs", _fail)
    res = client.get("/api/hf/runs?repo=me/run")
    assert res.status_code == 502
    assert "hf_secret" not in res.text


def test_hf_runs_rejects_an_invalid_repo(client):
    res = client.get("/api/hf/runs?repo=../bad")
    assert res.status_code == 400


def test_run_payload_includes_hf_metadata(client):
    from src import run_state

    run_state.create_run(client.mod.RUNS_DIR, run_id=RUN, config={}, trainer_args={})
    run_dir = client.mod.RUNS_DIR / RUN
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["hf"] = {"repo_id": "me/run"}
    (run_dir / "run.json").write_text(json.dumps(manifest))
    (run_dir / "hf_sync.json").write_text(json.dumps({"status": "idle"}))
    data = client.get(f"/api/runs/{RUN}").json()
    assert data["hf"] == {"repo_id": "me/run"}
    assert data["hf_sync"] == {"status": "idle"}


def test_websocket_payload_includes_hf_sync(client, monkeypatch):
    from src import run_state

    run_state.create_run(client.mod.RUNS_DIR, run_id=RUN, config={}, trainer_args={})
    (client.mod.RUNS_DIR / RUN / "hf_sync.json").write_text(
        json.dumps({"status": "idle", "last_success_step": 1000})
    )
    monkeypatch.setattr(client.mod, "get_gpu_info", lambda: {})
    with client.websocket_connect("/ws/live") as ws:
        payload = ws.receive_json()
    assert payload["hf_sync"]["last_success_step"] == 1000
