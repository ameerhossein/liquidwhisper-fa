"""Offline tests for optional merged-model mirroring from ``merge_lora.py``.

The real script imports torch/transformers/peft and merges weights; these tests
exercise only the post-save upload helper with a recording stand-in mirror, so
no model download or Hugging Face network call happens.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = "20261031-120000-abcdef"


def _merge_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import merge_lora

    return merge_lora


def _args(tmp_path, *, run_dir=True, repo="me/run"):
    run = tmp_path / RUN
    output = run / "merged"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / "model.safetensors").write_text("", encoding="utf-8")
    return SimpleNamespace(
        run_dir=str(run) if run_dir else None,
        hf_run_repo=repo,
        output_dir=str(output),
    )


class _RecordingMirror:
    instances = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.calls = []
        _RecordingMirror.instances.append(self)

    def start(self):
        self.calls.append("start")

    def enqueue_merged(self, path):
        self.calls.append(("merged", Path(path)))

    def enqueue_ledger(self, status):
        self.calls.append(("ledger", status))

    def close(self):
        self.calls.append("close")


def test_merge_upload_starts_enqueues_and_closes(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    args = _args(tmp_path)
    _RecordingMirror.instances.clear()
    monkeypatch.setattr(hf_sync, "HFRunMirror", _RecordingMirror)

    merge_lora._upload_merged_to_hf(args)

    assert len(_RecordingMirror.instances) == 1
    mirror = _RecordingMirror.instances[0]
    assert mirror.config.repo_id == "me/run"
    assert mirror.config.private is True
    assert mirror.kwargs["run_id"] == RUN
    assert Path(mirror.kwargs["run_dir"]) == tmp_path / RUN
    assert Path(mirror.kwargs["output_dir"]) == tmp_path / RUN / "output"
    assert Path(mirror.kwargs["results_dir"]) == tmp_path / RUN
    assert mirror.calls == [
        "start",
        ("merged", Path(args.output_dir)),
        ("ledger", "merge"),
        "close",
    ]


def test_merge_upload_failure_does_not_raise(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    def broken_mirror(*args, **kwargs):
        raise RuntimeError("Hugging Face is offline")

    monkeypatch.setattr(hf_sync, "HFRunMirror", broken_mirror)

    # A merge must still succeed when the optional mirror cannot start.
    merge_lora._upload_merged_to_hf(_args(tmp_path))


def test_merge_upload_start_failure_still_closes(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    calls = []

    class _StartFails:
        def __init__(self, config, **kwargs):
            calls.append("init")

        def start(self):
            calls.append("start")
            raise RuntimeError("Hugging Face is offline")

        def enqueue_merged(self, path):
            calls.append("merged")

        def enqueue_ledger(self, status):
            calls.append("ledger")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(hf_sync, "HFRunMirror", _StartFails)

    merge_lora._upload_merged_to_hf(_args(tmp_path))

    assert calls == ["init", "start", "close"]


def test_merge_upload_close_failure_does_not_raise(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    calls = []

    class _CloseFails:
        def __init__(self, config, **kwargs):
            calls.append("init")

        def start(self):
            calls.append("start")

        def enqueue_merged(self, path):
            calls.append("merged")

        def enqueue_ledger(self, status):
            calls.append("ledger")

        def close(self):
            calls.append("close")
            raise RuntimeError("drain failed")

    monkeypatch.setattr(hf_sync, "HFRunMirror", _CloseFails)

    merge_lora._upload_merged_to_hf(_args(tmp_path))

    assert calls == ["init", "start", "merged", "ledger", "close"]


def test_merge_upload_is_skipped_without_run_dir_or_repo(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    created = []
    monkeypatch.setattr(
        hf_sync, "HFRunMirror", lambda *args, **kwargs: created.append(1)
    )

    merge_lora._upload_merged_to_hf(_args(tmp_path, run_dir=False))
    merge_lora._upload_merged_to_hf(_args(tmp_path, repo=""))

    assert created == []


def test_merge_upload_auto_resolves_default_repo_when_authenticated(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    monkeypatch.setattr(hf_sync, "get_authenticated_user", lambda api=None: "alice")
    _RecordingMirror.instances.clear()
    monkeypatch.setattr(hf_sync, "HFRunMirror", _RecordingMirror)

    args = _args(tmp_path, repo=None)
    merge_lora._upload_merged_to_hf(args)

    assert len(_RecordingMirror.instances) == 1
    assert _RecordingMirror.instances[0].config.repo_id == "alice/liquidwhisper-neyshekar"


def test_merge_upload_honors_opt_out(tmp_path, monkeypatch):
    merge_lora = _merge_lora()
    from src import hf_sync

    monkeypatch.setattr(hf_sync, "get_authenticated_user", lambda api=None: "alice")
    created = []
    monkeypatch.setattr(hf_sync, "HFRunMirror", lambda *args, **kwargs: created.append(1))

    args = _args(tmp_path, repo="off")
    merge_lora._upload_merged_to_hf(args)

    args_no_sync = _args(tmp_path, repo="alice/liquidwhisper-neyshekar")
    args_no_sync.no_hf_sync = True
    merge_lora._upload_merged_to_hf(args_no_sync)

    assert created == []
