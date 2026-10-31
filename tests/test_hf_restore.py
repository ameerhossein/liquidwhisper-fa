# tests/test_hf_restore.py
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import hf_sync
from src.hf_sync import (
    HFRestoreConflict,
    HFRestoreError,
    HFSyncError,
    apply_restored_args,
    make_api,
    normalize_restored_run,
    restore_run,
    rewrite_run_paths,
)

RUN = "20261031-120000-abcdef"
OUTPUT = "runs/{}/checkpoint-{}"


def _api(files):
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return list(files)

    return FakeApi()


def _blobs(checkpoint_step=1000, status="running"):
    b = {
        f"runs/{RUN}/run.json": json.dumps(
            {
                "run_id": RUN,
                "status": status,
                "current_pid": 999999,
                "paths": {
                    "run_dir": "/workspace/old/run",
                    "output_dir": "/workspace/old/run/output",
                    "results_dir": "/workspace/old/run",
                    "merged_dir": "/workspace/old/run/merged",
                    "preflight_cache": "/workspace/old/run/baselines/preflight.json",
                },
                "trainer_args": {
                    "run_dir": "/workspace/old/run",
                    "output_dir": "/workspace/old/run/output",
                    "results_dir": "/workspace/old/run",
                    "preflight_cache_dir": "/workspace/old/run",
                    "epochs": 4,
                    "batch_size": 8,
                    "resume_from_checkpoint": None,
                    "fresh_start": False,
                },
                "stages": [
                    {"name": "preflight", "status": "completed"},
                    {"name": "training", "status": "running"},
                    {"name": "post_eval", "status": "pending"},
                ],
                "attempts": [{"pid": 999999, "ended_at": None}],
            }
        ).encode(),
        f"runs/{RUN}/steps.jsonl": b'{"step": 1000}\n',
        f"runs/{RUN}/events.jsonl": b'{"type": "stage"}\n',
        f"runs/{RUN}/best_adapter/adapter_config.json": b"{}",
        f"runs/{RUN}/best_adapter/adapter_model.safetensors": b"",
        f"runs/{RUN}/hf_state.json": json.dumps(
            {"last_step": checkpoint_step}
        ).encode(),
    }
    if checkpoint_step is not None:
        b[f"runs/{RUN}/checkpoint-{checkpoint_step}/trainer_state.json"] = b"{}"
        b[f"runs/{RUN}/checkpoint-{checkpoint_step}/adapter_model.safetensors"] = b""
    return b


def _files(checkpoint_step=1000):
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/steps.jsonl",
        f"runs/{RUN}/events.jsonl",
        f"runs/{RUN}/best_adapter/adapter_config.json",
        f"runs/{RUN}/best_adapter/adapter_model.safetensors",
        f"runs/{RUN}/hf_state.json",
    ]
    if checkpoint_step is not None:
        files += [
            f"runs/{RUN}/checkpoint-{checkpoint_step}/trainer_state.json",
            f"runs/{RUN}/checkpoint-{checkpoint_step}/adapter_model.safetensors",
        ]
    return files


def _download(blobs):
    return lambda repo, path: blobs[path]


def test_restore_downloads_ledger_and_newest_checkpoint(tmp_path):
    blobs = _blobs(1000)
    result = restore_run(
        _api(_files(1000)),
        "me/run",
        RUN,
        tmp_path,
        download=_download(blobs),
    )
    run_dir = tmp_path / RUN
    assert result["checkpoint_step"] == 1000
    assert (run_dir / "steps.jsonl").read_text() == '{"step": 1000}\n'
    assert (
        run_dir / "output" / "checkpoint-1000" / "trainer_state.json"
    ).is_file()
    assert (run_dir / "output" / "best_adapter" / "adapter_config.json").is_file()

    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "interrupted"
    assert run["current_pid"] is None
    assert run["paths"]["run_dir"] == str(run_dir)
    assert run["paths"]["output_dir"] == str(run_dir / "output")
    assert run["trainer_args"]["epochs"] == 4
    assert run["trainer_args"]["output_dir"] == str(run_dir / "output")


def test_restore_skips_incomplete_newest_checkpoint(tmp_path):
    blobs = _blobs(2000)
    files = _files(2000) + [
        f"runs/{RUN}/checkpoint-3000/trainer_state.json",
        f"runs/{RUN}/checkpoint-3000/optimizer.pt",
    ]
    result = restore_run(
        _api(files), "me/run", RUN, tmp_path, download=_download(blobs)
    )
    assert result["checkpoint_step"] == 2000


def test_restore_refuses_nonempty_local_run(tmp_path):
    (tmp_path / RUN).mkdir()
    (tmp_path / RUN / "run.json").write_text("{}")
    with pytest.raises(HFRestoreConflict):
        restore_run(
            _api(_files(1000)),
            "me/run",
            RUN,
            tmp_path,
            download=_download(_blobs(1000)),
        )


def test_restore_without_checkpoint_returns_partial_state(tmp_path):
    result = restore_run(
        _api(_files(None)),
        "me/run",
        RUN,
        tmp_path,
        download=_download(_blobs(None)),
    )
    assert result["checkpoint_step"] is None
    assert (tmp_path / RUN / "run.json").is_file()
    assert (tmp_path / RUN / "steps.jsonl").is_file()


def test_rewrite_and_normalize_keep_metrics_and_clear_dead_state(tmp_path):
    run = json.loads(
        _blobs(1000)[f"runs/{RUN}/run.json"].decode()
    )
    run["final_full_test_metrics"] = {"normalized_wer": 12.5}
    run["stages"][1]["status"] = "running"
    run["merge_pid"] = 123
    run["attempts"] = [{"pid": 999999, "ended_at": None}]

    normalize_restored_run(run)
    assert run["status"] == "interrupted"
    assert run["current_pid"] is None
    assert run["merge_pid"] is None
    assert run["attempts"][0]["ended_at"] is not None
    assert run["stages"][1]["status"] == "interrupted"
    assert run["final_full_test_metrics"]["normalized_wer"] == 12.5


def test_apply_restored_args_saved_values_and_explicit_overrides():
    run = {
        "trainer_args": {
            "epochs": 4,
            "batch_size": 16,
            "output_dir": "/old/output",
            "resume_from_checkpoint": "checkpoint-1",
            "fresh_start": True,
        },
        "stages": [
            {"name": "training", "status": "completed"},
            {"name": "post_eval", "status": "failed"},
        ],
    }
    args = SimpleNamespace(
        epochs=2,
        batch_size=8,
        output_dir="/cli/output",
        results_dir=None,
        run_dir=None,
        preflight_cache_dir=None,
        resume_from_checkpoint=None,
        fresh_start=True,
        auto_resume=False,
        post_eval_only=False,
        hf_resume="latest",
        hf_run_repo="me/run",
    )
    applied = apply_restored_args(
        args, run, explicit={"batch_size"}, run_dir=Path("/new/run")
    )
    assert applied.epochs == 4
    assert applied.batch_size == 8  # explicit beat saved
    assert applied.output_dir == "/new/run/output"
    assert applied.results_dir == "/new/run"
    assert applied.run_dir == "/new/run"
    assert applied.fresh_start is False
    assert applied.post_eval_only is True
    assert applied.resume_from_checkpoint is None


def test_rewrite_run_paths_sets_local_layout(tmp_path):
    run = {
        "paths": {
            "run_dir": "/old",
            "output_dir": "/old/output",
            "results_dir": "/old",
            "merged_dir": "/old/merged",
            "preflight_cache": "/old/baselines/preflight.json",
        },
        "trainer_args": {
            "run_dir": "/old",
            "output_dir": "/old/output",
            "results_dir": "/old",
            "preflight_cache_dir": "/old",
            "merged_dir": "/old/merged",
            "epochs": 4,
        },
    }
    target = tmp_path / "new-run"
    rewrite_run_paths(run, target)
    assert run["paths"] == {
        "run_dir": str(target),
        "output_dir": str(target / "output"),
        "results_dir": str(target),
        "merged_dir": str(target / "merged"),
        "preflight_cache": str(target / "baselines" / "preflight.json"),
    }
    assert run["trainer_args"]["run_dir"] == str(target)
    assert run["trainer_args"]["output_dir"] == str(target / "output")
    assert run["trainer_args"]["results_dir"] == str(target)
    assert run["trainer_args"]["preflight_cache_dir"] == str(target)
    assert "merged_dir" not in run["trainer_args"]
    assert run["trainer_args"]["epochs"] == 4


def test_restore_status_milestones(tmp_path):
    states = []
    restore_run(
        _api(_files(1000)),
        "me/run",
        RUN,
        tmp_path,
        download=_download(_blobs(1000)),
        status=lambda state, message: states.append(state),
    )
    assert states[:4] == ["listing", "ledger", "checkpoint", "verify"]


def test_restore_reports_snapshot_and_merged(tmp_path):
    blobs = _blobs(1000)
    blobs[f"runs/{RUN}/reports/final_evaluation_report.json"] = b'{"wer": 1}'
    blobs[f"runs/{RUN}/adapter_snapshots/step-250/adapter_config.json"] = b"{}"
    blobs[f"runs/{RUN}/adapter_snapshots/step-250/adapter_model.safetensors"] = b""
    blobs[f"runs/{RUN}/adapter_snapshots/step-500/adapter_config.json"] = b"{}"
    blobs[f"runs/{RUN}/adapter_snapshots/step-500/adapter_model.safetensors"] = b""
    blobs[f"runs/{RUN}/merged/config.json"] = b"{}"
    blobs[f"runs/{RUN}/merged/model.safetensors"] = b""
    files = _files(1000) + [
        f"runs/{RUN}/reports/final_evaluation_report.json",
        f"runs/{RUN}/adapter_snapshots/step-250/adapter_config.json",
        f"runs/{RUN}/adapter_snapshots/step-250/adapter_model.safetensors",
        f"runs/{RUN}/adapter_snapshots/step-500/adapter_config.json",
        f"runs/{RUN}/adapter_snapshots/step-500/adapter_model.safetensors",
        f"runs/{RUN}/merged/config.json",
        f"runs/{RUN}/merged/model.safetensors",
    ]
    result = restore_run(
        _api(files),
        "me/run",
        RUN,
        tmp_path,
        include_merged=True,
        download=_download(blobs),
    )
    run_dir = tmp_path / RUN
    assert (run_dir / "output" / "final_evaluation_report.json").is_file()
    assert (run_dir / "final_evaluation_report.json").is_file()
    assert (
        run_dir
        / "output"
        / "adapter_snapshots"
        / "step-500"
        / "adapter_config.json"
    ).is_file()
    assert not (
        run_dir / "output" / "adapter_snapshots" / "step-250"
    ).exists()
    assert (run_dir / "merged" / "model.safetensors").is_file()
    assert result["checkpoint_step"] == 1000


def test_restore_propagates_checkpoint_download_failure(tmp_path):
    blobs = _blobs(2000)
    files = _files(2000) + [
        f"runs/{RUN}/checkpoint-3000/trainer_state.json",
        f"runs/{RUN}/checkpoint-3000/adapter_model.safetensors",
    ]
    with pytest.raises(HFRestoreError, match="checkpoint-3000"):
        restore_run(
            _api(files),
            "me/run",
            RUN,
            tmp_path,
            download=_download(blobs),
        )
    assert not (tmp_path / RUN).exists()
    assert not list(tmp_path.glob(f".{RUN}.restore-*"))


def test_restore_falls_back_when_checkpoint_verification_fails(
    tmp_path, monkeypatch
):
    blobs = _blobs(2000)
    blobs[f"runs/{RUN}/checkpoint-3000/trainer_state.json"] = b"{}"
    blobs[f"runs/{RUN}/checkpoint-3000/adapter_model.safetensors"] = b""
    files = _files(2000) + [
        f"runs/{RUN}/checkpoint-3000/trainer_state.json",
        f"runs/{RUN}/checkpoint-3000/adapter_model.safetensors",
    ]
    real_is_complete = hf_sync.is_complete_checkpoint

    def fake_is_complete(path):
        if Path(path).name == ".restore-checkpoint-3000.tmp":
            return False
        return real_is_complete(path)

    monkeypatch.setattr(hf_sync, "is_complete_checkpoint", fake_is_complete)
    result = restore_run(
        _api(files),
        "me/run",
        RUN,
        tmp_path,
        download=_download(blobs),
    )
    assert result["checkpoint_step"] == 2000
    assert (tmp_path / RUN / "output" / "checkpoint-2000").is_dir()
    assert not (tmp_path / RUN / "output" / "checkpoint-3000").exists()


def test_restore_force_replaces_existing_run(tmp_path):
    old_run = tmp_path / RUN
    old_run.mkdir()
    (old_run / "old.txt").write_text("keep me")
    result = restore_run(
        _api(_files(1000)),
        "me/run",
        RUN,
        tmp_path,
        force=True,
        download=_download(_blobs(1000)),
    )
    assert result["checkpoint_step"] == 1000
    assert not (old_run / "old.txt").exists()
    assert not list(tmp_path.glob(f".{RUN}.restore-*"))


def test_force_preserves_existing_run_when_restore_fails(tmp_path):
    local = tmp_path / RUN
    local.mkdir()
    (local / "run.json").write_text('{"local": true}')
    (local / "keep.txt").write_text("keep")
    blobs = _blobs(1000)
    del blobs[f"runs/{RUN}/steps.jsonl"]

    with pytest.raises(HFRestoreError, match="steps.jsonl"):
        restore_run(
            _api(_files(1000)),
            "me/run",
            RUN,
            tmp_path,
            force=True,
            download=_download(blobs),
        )

    assert json.loads((local / "run.json").read_text()) == {"local": True}
    assert (local / "keep.txt").read_text() == "keep"
    assert not list(tmp_path.glob(f".{RUN}.restore-*"))


def test_restore_missing_remote_run_does_not_touch_local(tmp_path):
    local = tmp_path / RUN
    local.mkdir()
    (local / "run.json").write_text('{"local": true}')

    class EmptyApi:
        def list_repo_files(self, repo_id, revision="main"):
            return []

    with pytest.raises(HFRestoreError):
        restore_run(
            EmptyApi(),
            "me/run",
            RUN,
            tmp_path,
            force=True,
            download=lambda repo, path: pytest.fail("must not download"),
        )
    assert json.loads((local / "run.json").read_text()) == {"local": True}
    assert not (local / "output").exists()


def test_make_api_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(HFSyncError, match="huggingface_hub"):
        make_api()


def _script_module():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "hf_restore_script", root / "scripts" / "hf_restore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_usage_errors_exit_one(capsys):
    script = _script_module()
    with pytest.raises(SystemExit) as excinfo:
        script.main([])
    assert excinfo.value.code == 1
    assert "required" in capsys.readouterr().err


def test_cli_help_exits_zero(capsys):
    script = _script_module()
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_writes_completed_status_and_resume_hint(tmp_path, monkeypatch, capsys):
    script = _script_module()
    status_file = tmp_path / "status.json"
    monkeypatch.setattr(script, "make_api", lambda: object())
    monkeypatch.setattr(script, "resolve_resume_run", lambda api, repo, run: RUN)

    def fake_restore(api, repo, run_id, runs_root, **kwargs):
        kwargs["status"]("listing", "listing")
        return {
            "run_id": run_id,
            "run_dir": str(Path(runs_root) / run_id),
            "output_dir": str(Path(runs_root) / run_id / "output"),
            "checkpoint_step": 1000,
            "downloaded": [],
            "run": {},
        }

    monkeypatch.setattr(script, "restore_run", fake_restore)
    code = script.main(
        [
            "--repo",
            "me/run",
            "--run",
            RUN,
            "--runs-root",
            str(tmp_path),
            "--status-file",
            str(status_file),
        ]
    )
    assert code == 0
    status = json.loads(status_file.read_text())
    assert status["state"] == "completed"
    assert status["run_id"] == RUN
    assert status["checkpoint_step"] == 1000
    assert status["pid"] > 0
    out = capsys.readouterr().out
    assert f"--hf_resume {RUN}" in out


def test_cli_exits_two_without_checkpoint(tmp_path, monkeypatch):
    script = _script_module()
    status_file = tmp_path / "status.json"
    monkeypatch.setattr(script, "make_api", lambda: object())
    monkeypatch.setattr(script, "resolve_resume_run", lambda api, repo, run: RUN)
    monkeypatch.setattr(
        script,
        "restore_run",
        lambda api, repo, run_id, runs_root, **kwargs: {
            "run_id": run_id,
            "run_dir": str(Path(runs_root) / run_id),
            "output_dir": str(Path(runs_root) / run_id / "output"),
            "checkpoint_step": None,
            "downloaded": [],
            "run": {},
        },
    )
    code = script.main(
        [
            "--repo",
            "me/run",
            "--run",
            RUN,
            "--runs-root",
            str(tmp_path),
            "--status-file",
            str(status_file),
        ]
    )
    assert code == 2
    status = json.loads(status_file.read_text())
    assert status["state"] == "failed"
    assert status["checkpoint_step"] is None
