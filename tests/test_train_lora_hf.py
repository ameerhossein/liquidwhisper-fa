import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))

from src import run_state

RUN = "20261031-120000-abcdef"


def _train_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_lora
    return train_lora


def test_parser_accepts_hf_flags(monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lora.py",
            "--hf_run_repo", "me/run",
            "--no-hf_private",
            "--hf_adapter_snapshot_steps", "0",
            "--hf_resume", RUN,
        ],
    )
    args = train_lora.parse_args()
    assert args.hf_run_repo == "me/run"
    assert args.hf_private is False
    assert args.hf_adapter_snapshot_steps == 0
    assert args.hf_resume == RUN


def test_fresh_start_and_hf_resume_are_mutually_exclusive(monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_lora.py", "--fresh_start", "--hf_resume", "latest"],
    )
    with pytest.raises(SystemExit):
        train_lora.parse_args()


def test_mirror_auto_creates_canonical_run_dir(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setenv("ASR_RUNS_DIR", str(tmp_path / "runs"))
    args = SimpleNamespace(
        hf_run_repo="me/run",
        run_dir=None,
        post_eval_only=False,
        output_dir=str(tmp_path / "output"),
        results_dir=None,
    )
    run_dir = train_lora._maybe_create_hf_run_dir(args)
    assert run_dir is not None
    assert run_dir.parent == (tmp_path / "runs").resolve()
    assert run_state.run_id_valid(run_dir.name)


def test_mirror_does_not_create_run_without_repo(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    args = SimpleNamespace(
        hf_run_repo=None, run_dir=None, post_eval_only=False,
        output_dir=str(tmp_path / "output"), results_dir=None,
    )
    assert train_lora._maybe_create_hf_run_dir(args) is None


def test_hf_resume_restores_and_applies_saved_args(tmp_path, monkeypatch):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    run = run_state.create_run(
        runs_root,
        run_id=RUN,
        config={},
        trainer_args={"epochs": 4, "batch_size": 8},
    )
    run_dir = runs_root / RUN
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "failed")

    args = SimpleNamespace(
        hf_resume=RUN,
        hf_run_repo="me/run",
        run_dir=None,
        output_dir=str(tmp_path / "default-output"),
        results_dir=None,
        post_eval_only=False,
        epochs=2,
        batch_size=8,
        fresh_start=False,
        auto_resume=False,
        resume_from_checkpoint=None,
        preflight_cache_dir=None,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "resolve_resume_run",
        lambda *a, **k: RUN,
    )
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: {"run_id": RUN, "checkpoint_step": 1000},
    )
    observed = {}

    def fake_run(parsed, current_run_dir):
        observed["run_dir"] = str(current_run_dir)
        observed["post_eval_only"] = parsed.post_eval_only
        observed["output_dir"] = parsed.output_dir

    monkeypatch.setattr(train_lora, "_run_training", fake_run)
    train_lora.main()

    assert observed["run_dir"] == str(run_dir.resolve())
    assert observed["post_eval_only"] is True
    assert observed["output_dir"] == str(run_dir / "output")


def test_hf_resume_completed_run_does_not_train(tmp_path, monkeypatch):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    run_state.create_run(
        runs_root, run_id=RUN, config={}, trainer_args={"epochs": 4}
    )
    run_dir = runs_root / RUN
    run_state.set_status(run_dir, "completed")
    args = SimpleNamespace(
        hf_resume=RUN,
        hf_run_repo="me/run",
        run_dir=None,
        output_dir=str(tmp_path / "default-output"),
        results_dir=None,
        post_eval_only=False,
        epochs=2,
        batch_size=8,
        fresh_start=False,
        auto_resume=False,
        resume_from_checkpoint=None,
        preflight_cache_dir=None,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    monkeypatch.setattr(train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: {"run_id": RUN, "checkpoint_step": 1000},
    )
    downloads = []
    monkeypatch.setattr(
        train_lora,
        "_download_restore_artifacts",
        lambda api, repo, run_id, current: downloads.append(run_id) or 1000,
    )
    called = []
    monkeypatch.setattr(
        train_lora, "_run_training", lambda parsed, current: called.append(current)
    )
    train_lora.main()
    assert called == []
    assert downloads == [RUN]


def test_completed_run_with_local_checkpoint_restores_without_repo(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    run_state.create_run(
        runs_root, run_id=RUN, config={}, trainer_args={}
    )
    run_dir = runs_root / RUN
    run_state.set_status(run_dir, "completed")
    checkpoint = run_dir / "output" / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_text("", encoding="utf-8")
    args = _restore_args(tmp_path, hf_run_repo=None)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "make_api",
        lambda: pytest.fail("a complete local checkpoint needs no HF call"),
    )

    assert train_lora._restore_hf_run(args) is True


def test_completed_run_missing_checkpoint_downloads_without_replacing_ledger(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    _runs_root, run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=2000
    )
    run_state.set_status(run_dir, "completed")
    before = (run_dir / "run.json").read_text(encoding="utf-8")
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: pytest.fail("the local ledger must not be replaced"),
    )
    downloads = []
    monkeypatch.setattr(
        train_lora,
        "_download_restore_artifacts",
        lambda api, repo, run_id, current: downloads.append(run_id) or 1000,
    )

    assert train_lora._restore_hf_run(args) is True
    assert downloads == [RUN]
    assert (run_dir / "run.json").read_text(encoding="utf-8") == before


def test_completed_run_missing_checkpoint_without_remote_exits_2(
    tmp_path, monkeypatch, capsys
):
    train_lora = _train_lora()
    _runs_root, run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=2000
    )
    run_state.set_status(run_dir, "completed")
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: pytest.fail("the local ledger must not be replaced"),
    )
    monkeypatch.setattr(
        train_lora, "_download_restore_artifacts", lambda *a, **k: None
    )

    with pytest.raises(SystemExit) as excinfo:
        train_lora._restore_hf_run(args)
    assert excinfo.value.code == 2
    assert "no complete checkpoint" in capsys.readouterr().err



def test_explicit_cli_keys_maps_no_flags():
    train_lora = _train_lora()
    parser = train_lora.build_parser()
    keys = train_lora._explicit_cli_keys(
        parser,
        ["--no-hf_private", "--epochs=3", "--hf_resume", RUN],
    )
    assert {"hf_private", "epochs", "hf_resume"} <= keys


def test_build_mirror_uses_recorded_repo_and_repairs(monkeypatch, tmp_path):
    train_lora = _train_lora()
    created = {}

    class _FakeMirror:
        def __init__(self, config, **kwargs):
            created["config"] = config
            created["kwargs"] = kwargs
            self.started = False
            self.repaired = False

        def start(self):
            self.started = True

        def auto_repair(self):
            self.repaired = True

    monkeypatch.setattr(train_lora.hf_sync, "HFRunMirror", _FakeMirror)

    def _args(**overrides):
        values = dict(
            hf_run_repo="cli/run",
            hf_private=False,
            hf_adapter_snapshot_steps=0,
            hf_auto_resume=True,
            fresh_start=False,
            output_dir=str(tmp_path / "output"),
            results_dir=str(tmp_path / "results"),
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    mirror = train_lora._build_hf_mirror(
        _args(), tmp_path / "run", run={"hf": {"repo_id": "recorded/run"}}
    )
    assert mirror.started is True
    assert mirror.repaired is True
    assert created["config"].repo_id == "recorded/run"
    assert created["config"].private is False
    assert created["config"].adapter_snapshot_steps == 0

    created.clear()
    mirror = train_lora._build_hf_mirror(
        _args(fresh_start=True), tmp_path / "run", run={"hf": {"repo_id": "recorded/run"}}
    )
    assert mirror.started is True
    assert mirror.repaired is False


def test_build_mirror_is_off_without_repo(monkeypatch, tmp_path):
    train_lora = _train_lora()
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)

    class _MustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError("no mirror may be built without a repo")

    monkeypatch.setattr(train_lora.hf_sync, "HFRunMirror", _MustNotConstruct)
    args = SimpleNamespace(
        hf_run_repo=None,
        hf_private=True,
        hf_adapter_snapshot_steps=250,
        hf_auto_resume=True,
        fresh_start=False,
        output_dir=str(tmp_path / "output"),
        results_dir=str(tmp_path / "results"),
    )
    assert train_lora._build_hf_mirror(args, tmp_path / "run") is None


def test_post_eval_only_uploads_reports_and_terminal_ledger(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run = run_state.create_run(
        tmp_path, run_id=RUN, config={}, trainer_args={}
    )
    run_dir = tmp_path / RUN
    events = []

    class _FakeMirror:
        def enqueue_ledger(self, status, **kwargs):
            events.append(("ledger", status, kwargs))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        train_lora, "_build_hf_mirror", lambda args, path, run=None: _FakeMirror()
    )

    def _fake_post_eval(args):
        run_state.set_status(run_dir, "completed")

    monkeypatch.setattr(train_lora, "_run_post_eval_only", _fake_post_eval)
    args = SimpleNamespace(
        post_eval_only=True,
        fresh_start=False,
        results_dir=str(tmp_path),
        run_dir=str(run_dir),
    )
    train_lora._run_training(args, run_dir)

    assert ("ledger", "post_eval", {"reports": True}) in events
    assert ("ledger", "completed", {}) in events
    assert events[-1] == ("close",)


def _restore_args(tmp_path, **overrides):
    values = dict(
        hf_resume=RUN,
        hf_run_repo="me/run",
        run_dir=None,
        output_dir=str(tmp_path / "default-output"),
        results_dir=None,
        post_eval_only=False,
        epochs=2,
        batch_size=8,
        fresh_start=False,
        auto_resume=True,
        resume_from_checkpoint=None,
        preflight_cache_dir=None,
        telephony_profiles="customer",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare_restore_run(
    tmp_path, monkeypatch, *, last_step, recorded_repo=None
):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("ASR_RUNS_DIR", str(runs_root))
    run_state.create_run(
        runs_root, run_id=RUN, config={}, trainer_args={"epochs": 4}
    )
    run_dir = runs_root / RUN
    if last_step:
        run_state.set_progress(
            run_dir, step=last_step, step_count=last_step, last_step_at=1.0
        )
    if recorded_repo:
        run_state.update_run(
            run_dir,
            lambda run: {**run, "hf": {"repo_id": recorded_repo}},
        )
    return runs_root, run_dir


def test_restore_keeps_newer_local_ledger(tmp_path, monkeypatch):
    train_lora = _train_lora()
    _runs_root, run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=2000
    )
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(train_lora, "_remote_hf_high_water", lambda *a, **k: 1000)
    monkeypatch.setattr(train_lora, "_remote_checkpoint_step", lambda *a, **k: 1000)
    restore_calls = []
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: restore_calls.append(k),
    )

    assert train_lora._restore_hf_run(args) is False
    assert restore_calls == []
    assert args.run_dir == str(run_dir.resolve())
    assert args.output_dir == str(run_dir / "output")
    assert getattr(args, "_hf_repair_pending", False) is True


def test_restore_keeps_local_ledger_and_exits_without_remote_checkpoint(
    tmp_path, monkeypatch, capsys
):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=2000
    )
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(train_lora, "_remote_hf_high_water", lambda *a, **k: 1000)
    monkeypatch.setattr(train_lora, "_remote_checkpoint_step", lambda *a, **k: None)
    restore_calls = []
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: restore_calls.append(k),
    )

    with pytest.raises(SystemExit) as excinfo:
        train_lora._restore_hf_run(args)
    assert excinfo.value.code == 2
    assert restore_calls == []
    assert "no usable remote checkpoint" in capsys.readouterr().err


def test_restore_keeps_local_ledger_on_tie(tmp_path, monkeypatch):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=1000
    )
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(train_lora, "_remote_hf_high_water", lambda *a, **k: 1000)
    monkeypatch.setattr(train_lora, "_remote_checkpoint_step", lambda *a, **k: 1000)
    restore_calls = []
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: restore_calls.append(k),
    )

    assert train_lora._restore_hf_run(args) is False
    assert restore_calls == []
    assert getattr(args, "_hf_repair_pending", False) is True


def test_restore_replaces_local_ledger_when_remote_newer(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path, monkeypatch, last_step=1000
    )
    args = _restore_args(tmp_path)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    monkeypatch.setattr(
        train_lora.hf_sync, "resolve_resume_run", lambda *a, **k: RUN
    )
    monkeypatch.setattr(
        train_lora, "_remote_hf_high_water", lambda *a, **k: 2000
    )
    restore_calls = []

    def _restore(*args, **kwargs):
        restore_calls.append(kwargs)
        return {"run_id": RUN, "checkpoint_step": 2000}

    monkeypatch.setattr(train_lora.hf_sync, "restore_run", _restore)

    assert train_lora._restore_hf_run(args) is False
    assert len(restore_calls) == 1
    assert restore_calls[0]["force"] is True
    assert getattr(args, "_hf_repair_pending", False) is False


def test_restore_uses_recorded_repo_over_cli(tmp_path, monkeypatch):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path,
        monkeypatch,
        last_step=2000,
        recorded_repo="recorded/repo",
    )
    args = _restore_args(tmp_path, hf_run_repo="cli/repo")
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    seen = {}

    def _resolve(api, repo_id, requested):
        seen["resolve"] = repo_id
        return RUN

    def _high_water(api, repo_id, run_id):
        seen["high_water"] = repo_id
        return 1000

    monkeypatch.setattr(train_lora.hf_sync, "resolve_resume_run", _resolve)
    monkeypatch.setattr(train_lora, "_remote_hf_high_water", _high_water)
    monkeypatch.setattr(train_lora, "_remote_checkpoint_step", lambda *a, **k: 1000)
    monkeypatch.setattr(
        train_lora.hf_sync,
        "restore_run",
        lambda *a, **k: pytest.fail("local ledger is newer"),
    )

    assert train_lora._restore_hf_run(args) is False
    assert seen == {"resolve": "recorded/repo", "high_water": "recorded/repo"}
    assert args.hf_run_repo == "recorded/repo"


def test_restore_uses_recorded_repo_without_cli_or_env(tmp_path, monkeypatch):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path,
        monkeypatch,
        last_step=0,
        recorded_repo="recorded/repo",
    )
    args = _restore_args(tmp_path, hf_run_repo=None)
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    monkeypatch.setattr(train_lora.hf_sync, "resolve_config", lambda *a, **k: None)
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    seen = {}

    def _resolve(api, repo_id, requested):
        seen["resolve"] = repo_id
        return RUN

    def _restore(api, repo_id, run_id, runs_root, **kwargs):
        seen["restore"] = repo_id
        return {"run_id": RUN, "checkpoint_step": 1000}

    monkeypatch.setattr(train_lora.hf_sync, "resolve_resume_run", _resolve)
    monkeypatch.setattr(train_lora.hf_sync, "restore_run", _restore)

    assert train_lora._restore_hf_run(args) is False
    assert seen == {"resolve": "recorded/repo", "restore": "recorded/repo"}
    assert args.hf_run_repo == "recorded/repo"


def test_restore_latest_switches_to_recorded_repo(tmp_path, monkeypatch):
    train_lora = _train_lora()
    _runs_root, _run_dir = _prepare_restore_run(
        tmp_path,
        monkeypatch,
        last_step=0,
        recorded_repo="recorded/repo",
    )
    args = _restore_args(tmp_path, hf_resume="latest", hf_run_repo="cli/repo")
    monkeypatch.setattr(train_lora.hf_sync, "make_api", lambda: object())
    seen = {}

    def _resolve(api, repo_id, requested):
        seen["resolve"] = repo_id
        assert requested == "latest"
        return RUN

    def _restore(api, repo_id, run_id, runs_root, **kwargs):
        seen["restore"] = repo_id
        return {"run_id": RUN, "checkpoint_step": 1000}

    monkeypatch.setattr(train_lora.hf_sync, "resolve_resume_run", _resolve)
    monkeypatch.setattr(train_lora.hf_sync, "restore_run", _restore)

    assert train_lora._restore_hf_run(args) is False
    assert seen == {"resolve": "cli/repo", "restore": "recorded/repo"}
    assert args.hf_run_repo == "recorded/repo"


def test_build_mirror_repairs_before_starting_uploader(tmp_path, monkeypatch):
    train_lora = _train_lora()
    events = []
    created = {}

    class _FakeMirror:
        def __init__(self, config, **kwargs):
            events.append("init")
            created.update(kwargs)

        def start(self):
            events.append("start")

        def auto_repair(self):
            events.append("repair")

        def start_uploader(self):
            events.append("uploader")

    monkeypatch.setattr(train_lora.hf_sync, "HFRunMirror", _FakeMirror)
    args = SimpleNamespace(
        hf_run_repo="me/run",
        hf_private=True,
        hf_adapter_snapshot_steps=250,
        hf_auto_resume=True,
        fresh_start=False,
        output_dir=str(tmp_path / "output"),
        results_dir=str(tmp_path / "results"),
    )
    mirror = train_lora._build_hf_mirror(args, tmp_path / "run", run={})

    assert mirror is not None
    assert created["start_worker"] is False
    assert events == ["init", "start", "repair", "uploader"]
