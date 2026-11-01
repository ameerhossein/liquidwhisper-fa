import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.hf_sync import (
    DEFAULT_HF_REPO_NAME,
    DISABLE_HF_VALUES,
    HFSyncConfig,
    HFCompletedRunError,
    HFRestoreError,
    is_complete_checkpoint,
    list_remote_runs,
    repo_checkpoint_path,
    repo_run_path,
    resolve_config,
    resolve_default_repo,
    resolve_resume_run,
    reset_repo,
    get_authenticated_user,
    select_checkpoint_step,
    validate_adapter,
    validate_merged,
    validate_repo_id,
)

RUN = "20261031-120000-abcdef"
OTHER = "20261030-120000-aaaaaa"


def test_config_is_off_without_a_repo(monkeypatch):
    monkeypatch.delenv("ASR_HF_RUN_REPO", raising=False)
    assert resolve_config(SimpleNamespace(), environ={}) is None


def test_config_prefers_cli_over_env():
    args = SimpleNamespace(
        hf_run_repo="me/run",
        hf_private=False,
        hf_adapter_snapshot_steps=0,
        hf_auto_resume=False,
    )
    cfg = resolve_config(args, environ={"ASR_HF_RUN_REPO": "me/env"})
    assert cfg == HFSyncConfig(
        repo_id="me/run",
        private=False,
        adapter_snapshot_steps=0,
        auto_resume=False,
    )


def test_config_reads_env_and_rejects_bad_repo():
    cfg = resolve_config(SimpleNamespace(), environ={"ASR_HF_RUN_REPO": "me/run"})
    assert cfg.repo_id == "me/run"
    with pytest.raises(ValueError):
        validate_repo_id("not-a-repo")
    with pytest.raises(ValueError):
        validate_repo_id("a/b/c")


def test_repo_paths_are_run_scoped():
    assert repo_run_path(RUN) == f"runs/{RUN}"
    assert repo_checkpoint_path(RUN, 1000) == f"runs/{RUN}/checkpoint-1000"


def test_checkpoint_requires_state_and_adapter(tmp_path):
    ckpt = tmp_path / "checkpoint-1000"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}")
    assert is_complete_checkpoint(ckpt) is False
    (ckpt / "adapter_model.safetensors").write_text("")
    assert is_complete_checkpoint(ckpt) is True


def test_adapter_and_merged_validation(tmp_path):
    adapter = tmp_path / "best_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_text("")
    assert validate_adapter(adapter)[0] is True

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_text("")
    assert validate_merged(merged)[0] is True
    assert validate_merged(tmp_path / "missing")[0] is False


def test_merged_index_requires_referenced_shards(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.a": "model-00001-of-00002.safetensors"}}
        )
    )
    ok, reason = validate_merged(merged)
    assert ok is False
    assert "shard" in reason

    (merged / "model-00001-of-00002.safetensors").write_text("")
    assert validate_merged(merged)[0] is True


@pytest.mark.parametrize(
    ("payload", "reason_fragment"),
    [
        ("not json", "not valid JSON"),
        (json.dumps({}), "weight_map"),
        (json.dumps({"weight_map": []}), "weight_map"),
        (json.dumps({"weight_map": {}}), "weight_map"),
        (
            json.dumps({"weight_map": {"model.a": "/abs/model.safetensors"}}),
            "shard",
        ),
        (
            json.dumps({"weight_map": {"model.a": "shard\\model.safetensors"}}),
            "shard",
        ),
        (
            json.dumps({"weight_map": {"model.a": "../model.safetensors"}}),
            "shard",
        ),
    ],
)
def test_merged_index_rejects_invalid_or_unsafe_shards(
    tmp_path, payload, reason_fragment
):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors.index.json").write_text(payload)

    ok, reason = validate_merged(merged)

    assert ok is False
    assert reason_fragment in reason


def test_merged_bin_index_accepts_existing_shards(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {"weight_map": {"model.a": "pytorch_model-00001-of-00001.bin"}}
        )
    )
    (merged / "pytorch_model-00001-of-00001.bin").write_text("")

    assert validate_merged(merged)[0] is True


def test_select_checkpoint_step_ignores_incomplete_and_bad_names():
    files = [
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
        f"runs/{RUN}/checkpoint-2000/trainer_state.json",
        f"runs/{RUN}/checkpoint-nope/trainer_state.json",
    ]
    assert select_checkpoint_step(files, RUN) == 1000


def test_list_remote_runs_reads_manifest_and_state():
    blobs = {
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "running", "updated_at": 20.0}
        ).encode(),
        f"runs/{RUN}/hf_state.json": json.dumps({"last_step": 2000}).encode(),
        f"runs/{OTHER}/run.json": json.dumps(
            {"run_id": OTHER, "status": "completed", "updated_at": 10.0}
        ).encode(),
    }
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/hf_state.json",
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
        f"runs/{OTHER}/run.json",
    ]

    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return list(files)

    runs = list_remote_runs(
        FakeApi(), "me/run", download=lambda repo, path: blobs[path]
    )
    assert [r["run_id"] for r in runs] == [RUN, OTHER]
    assert runs[0]["last_step"] == 2000
    assert runs[0]["has_checkpoint"] is True
    assert runs[1]["has_checkpoint"] is False


def test_resolve_latest_rejects_completed_run():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [
                "latest.json",
                f"runs/{RUN}/run.json",
                f"runs/{RUN}/checkpoint-1000/trainer_state.json",
                f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
            ]

    blobs = {
        "latest.json": json.dumps({"run_id": RUN}).encode(),
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "completed", "updated_at": 20.0}
        ).encode(),
    }
    with pytest.raises(HFCompletedRunError):
        resolve_resume_run(
            FakeApi(), "me/run", "latest", download=lambda repo, path: blobs[path]
        )


def test_resolve_explicit_run_works_even_when_completed():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [f"runs/{RUN}/run.json"]

    blobs = {
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "completed", "updated_at": 20.0}
        ).encode()
    }
    assert resolve_resume_run(
        FakeApi(), "me/run", RUN, download=lambda repo, path: blobs[path]
    ) == RUN


def test_config_clamps_snapshot_steps_and_falls_back_to_env():
    cfg = resolve_config(
        SimpleNamespace(hf_run_repo="", hf_adapter_snapshot_steps=-5),
        environ={"ASR_HF_RUN_REPO": "me/env"},
    )
    assert cfg.repo_id == "me/env"
    assert cfg.adapter_snapshot_steps == 0


def test_config_validates_cli_repo():
    with pytest.raises(ValueError):
        resolve_config(
            SimpleNamespace(hf_run_repo="not-a-repo"), environ={}
        )


def test_resolve_latest_falls_back_to_newest_run_id():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [f"runs/{OTHER}/run.json", f"runs/{RUN}/run.json"]

    blobs = {
        f"runs/{RUN}/run.json": json.dumps(
            {"run_id": RUN, "status": "running", "updated_at": 20.0}
        ).encode(),
        f"runs/{OTHER}/run.json": json.dumps(
            {"run_id": OTHER, "status": "running", "updated_at": 10.0}
        ).encode(),
    }
    assert resolve_resume_run(
        FakeApi(), "me/run", "latest", download=lambda repo, path: blobs[path]
    ) == RUN


def test_resolve_explicit_missing_run_raises_restore_error():
    class FakeApi:
        def list_repo_files(self, repo_id, revision="main"):
            return [f"runs/{OTHER}/run.json"]

    with pytest.raises(HFRestoreError):
        resolve_resume_run(
            FakeApi(),
            "me/run",
            RUN,
            download=lambda repo, path: pytest.fail("missing run must not download"),
        )


def test_reset_repo_deletes_and_creates_repo():
    calls = []

    class FakeApi:
        def delete_repo(self, repo_id, repo_type="model", missing_ok=False):
            calls.append(("delete_repo", repo_id, repo_type, missing_ok))

        def create_repo(self, repo_id, repo_type="model", private=True, exist_ok=False):
            calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    api = FakeApi()
    reset_repo("me/smoke-test", api=api, private=True)

    assert ("delete_repo", "me/smoke-test", "model", True) in calls
    assert ("create_repo", "me/smoke-test", "model", True, True) in calls


def test_reset_repo_falls_back_to_delete_files_on_permission_error():
    calls = []

    class FakeApi:
        def delete_repo(self, repo_id, repo_type="model", missing_ok=False):
            calls.append(("delete_repo", repo_id))
            raise RuntimeError("403 Forbidden: delete_repo not allowed")

        def list_repo_files(self, repo_id, revision="main"):
            calls.append(("list_repo_files", repo_id))
            return [".gitattributes", "runs/run1/adapter.bin", "README.md"]

        def delete_files(self, repo_id, delete_patterns, commit_message=None):
            calls.append(("delete_files", repo_id, delete_patterns, commit_message))

        def create_repo(self, repo_id, repo_type="model", private=True, exist_ok=False):
            calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    api = FakeApi()
    reset_repo("me/smoke-test", api=api)

    assert ("delete_repo", "me/smoke-test") in calls
    assert ("list_repo_files", "me/smoke-test") in calls
    assert ("delete_files", "me/smoke-test", ["runs/run1/adapter.bin", "README.md"], "Reset smoke test repository") in calls
    assert ("create_repo", "me/smoke-test", "model", True, True) in calls


def test_reset_repo_raises_when_fallback_delete_files_fails():
    class FakeApi:
        def delete_repo(self, repo_id, repo_type="model", missing_ok=False):
            raise RuntimeError("403 Forbidden: delete_repo not allowed")

        def list_repo_files(self, repo_id, revision="main"):
            return ["README.md"]

        def delete_files(self, repo_id, delete_patterns, commit_message=None):
            raise RuntimeError("403 Forbidden: delete_files token lacks write permission")

        def create_repo(self, repo_id, repo_type="model", private=True, exist_ok=False):
            pass

    api = FakeApi()
    with pytest.raises(RuntimeError, match="delete_files token lacks write permission"):
        reset_repo("me/smoke-test", api=api)



def test_reset_repo_validates_repo_id():
    with pytest.raises(ValueError):
        reset_repo("invalid_repo_name")


def test_get_authenticated_user_returns_username():
    class FakeApi:
        def whoami(self):
            return {"type": "user", "name": "testuser", "email": "test@example.com"}

    assert get_authenticated_user(FakeApi()) == "testuser"


def test_get_authenticated_user_returns_none_on_error():
    class FakeApi:
        def whoami(self):
            raise RuntimeError("LocalTokenNotFoundError")

    assert get_authenticated_user(FakeApi()) is None
    assert get_authenticated_user(object()) is None


def test_resolve_default_repo_returns_authenticated_user_repo(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "alice")
    assert resolve_default_repo(environ={}) == f"alice/{DEFAULT_HF_REPO_NAME}"


def test_resolve_default_repo_returns_none_when_unauthenticated(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: None)
    assert resolve_default_repo(environ={}) is None


def test_resolve_default_repo_honors_explicit_env():
    assert resolve_default_repo(environ={"ASR_HF_RUN_REPO": "myorg/custom-repo"}) == "myorg/custom-repo"


def test_resolve_default_repo_honors_opt_out_env(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "alice")
    for val in ("off", "none", "false", "0", "disabled", "no"):
        assert resolve_default_repo(environ={"ASR_HF_RUN_REPO": val}) is None


def test_resolve_config_defaults_to_authenticated_user_repo(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "alice")
    cfg = resolve_config(SimpleNamespace(hf_run_repo=None), environ={})
    assert cfg is not None
    assert cfg.repo_id == f"alice/{DEFAULT_HF_REPO_NAME}"


def test_resolve_config_is_off_when_unauthenticated_and_no_repo(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: None)
    assert resolve_config(SimpleNamespace(hf_run_repo=None), environ={}) is None


def test_resolve_config_honors_cli_opt_out(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "alice")
    assert resolve_config(SimpleNamespace(hf_run_repo="off"), environ={}) is None
    assert resolve_config(SimpleNamespace(hf_run_repo="none"), environ={}) is None


def test_resolve_config_honors_no_hf_sync(monkeypatch):
    monkeypatch.setattr("src.hf_sync.get_authenticated_user", lambda api=None: "alice")
    assert resolve_config(SimpleNamespace(no_hf_sync=True, hf_run_repo=None), environ={}) is None
    assert resolve_config(SimpleNamespace(no_hf_sync=True, hf_run_repo="my/repo"), environ={}) is None



