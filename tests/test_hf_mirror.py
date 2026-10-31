import json
import shutil
from pathlib import Path

from src.hf_sync import HFSyncConfig, HFRunMirror

RUN = "20261031-120000-abcdef"


class FakeApi:
    def __init__(self, files=None):
        self.files = list(files or [])
        self.created = []
        self.folders = []
        self.commits = []
        self.files_uploaded = []
        self.fail_folder = 0

    def create_repo(self, repo_id, private=True, exist_ok=True):
        self.created.append((repo_id, private, exist_ok))

    def list_repo_files(self, repo_id, revision="main"):
        return list(self.files)

    def upload_folder(self, **kwargs):
        if self.fail_folder:
            self.fail_folder -= 1
            raise RuntimeError("transient")
        self.folders.append(kwargs)

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)

    def upload_file(self, **kwargs):
        self.files_uploaded.append(kwargs)


def _mirror(tmp_path, api, **kwargs):
    run_dir = tmp_path / RUN
    output = run_dir / "output"
    output.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": RUN,
                "status": "running",
                "last_step": 1000,
                "paths": {"output_dir": str(output)},
            }
        )
    )
    (run_dir / "steps.jsonl").write_text('{"step": 1000}\n')
    (run_dir / "events.jsonl").write_text('{"type": "stage"}\n')
    kwargs.setdefault("download", lambda repo, path: b"{}")
    return HFRunMirror(
        HFSyncConfig(repo_id="me/run"),
        run_id=RUN,
        run_dir=run_dir,
        output_dir=output,
        results_dir=run_dir,
        api=api,
        start_worker=False,
        sleep=lambda _: None,
        **kwargs,
    )


def _checkpoint(output, step):
    path = output / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}")
    (path / "adapter_model.safetensors").write_text("")
    return path


def test_start_uploader_defers_uploads_until_called(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()
    assert mirror._worker is None
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert api.folders == []
    mirror.start_uploader()
    assert mirror.drain(5)
    assert api.folders
    assert api.folders[0]["path_in_repo"] == f"runs/{RUN}/checkpoint-1000"
    mirror.close()


def test_checkpoint_upload_uses_run_scoped_path(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1
    assert api.folders[0]["path_in_repo"] == f"runs/{RUN}/checkpoint-1000"


def test_incomplete_checkpoint_is_not_uploaded(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    path = mirror.output_dir / "checkpoint-1000"
    path.mkdir()
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 0
    assert api.folders == []


def test_ledger_upload_contains_state_and_reports(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 1
    paths = [op.path_in_repo for op in api.commits[0]["operations"]]
    assert f"runs/{RUN}/run.json" in paths
    assert f"runs/{RUN}/steps.jsonl" in paths
    assert f"runs/{RUN}/hf_state.json" in paths


def test_high_water_blocks_older_ledger(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.download = lambda repo, path: json.dumps({"last_step": 2000}).encode()
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 0
    assert api.commits == []
    assert "2000" in mirror.status()["last_error"]


def test_transient_failure_retries_and_records_success(tmp_path):
    api = FakeApi()
    api.fail_folder = 2
    mirror = _mirror(tmp_path, api)
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1
    assert len(api.folders) == 1
    assert mirror.status()["last_success_step"] == 1000


def test_adapter_snapshots_coalesce_to_newest(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    for step in (250, 500, 750):
        path = mirror.output_dir / "adapter_snapshots" / f"step-{step:06d}"
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_text("")
        mirror.enqueue_adapter_snapshot(path, step)
    assert mirror.process_pending() == 1
    assert api.folders[0]["path_in_repo"].endswith("step-000750")


def test_failure_writes_error_status_without_raising(tmp_path):
    api = FakeApi()
    api.fail_folder = 3
    mirror = _mirror(tmp_path, api)
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 0
    status = mirror.status()
    assert status["status"] == "error"
    assert "transient" in status["last_error"]


def _complete_adapter(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_text("")
    return path


def _read_operation_bytes(operation):
    target = operation.path_or_fileobj
    if isinstance(target, bytes):
        return target
    return Path(target).read_bytes()


def test_start_creates_repo_readme_and_manifest_config(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start()

    assert api.created == [("me/run", True, True)]
    readme = [
        call for call in api.files_uploaded if call["path_in_repo"] == "README.md"
    ]
    assert readme
    assert b"scripts/hf_restore.py" in readme[0]["path_or_fileobj"]

    run = json.loads((mirror.run_dir / "run.json").read_text())
    assert run["hf"]["repo_id"] == "me/run"
    assert run["hf"]["private"] is True
    assert (mirror.run_dir / "hf_sync.json").is_file()


def test_manifest_update_callback_receives_hf_section(tmp_path):
    api = FakeApi()
    updates = []
    mirror = _mirror(tmp_path, api, manifest_update=updates.append)
    mirror.start()
    assert updates
    run = {}
    updates[0](run)
    assert run["hf"]["repo_id"] == "me/run"
    assert run["hf"]["enabled"] is True


def test_start_backfills_newer_local_checkpoint_and_best_adapter(tmp_path):
    files = [
        "README.md",
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/hf_state.json",
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
    ]
    api = FakeApi(files)
    download = lambda repo, path: (
        json.dumps({"last_step": 1000}).encode()
        if path.endswith("/hf_state.json")
        else b"{}"
    )
    mirror = _mirror(tmp_path, api, download=download)
    _checkpoint(mirror.output_dir, 1000)
    _checkpoint(mirror.output_dir, 2000)
    _complete_adapter(mirror.output_dir / "best_adapter")

    mirror.start()
    assert mirror.process_pending() == 2
    assert [call["path_in_repo"] for call in api.folders] == [
        f"runs/{RUN}/checkpoint-2000",
        f"runs/{RUN}/best_adapter",
    ]


def test_pending_file_requeues_existing_paths_and_drops_vanished(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    checkpoint = _checkpoint(mirror.output_dir, 1000)
    snapshot = _complete_adapter(
        mirror.output_dir / "adapter_snapshots" / "step-000500"
    )
    mirror.enqueue_checkpoint(checkpoint, 1000)
    mirror.enqueue_adapter_snapshot(snapshot, 500)
    pending = json.loads((mirror.run_dir / "hf_sync_pending.json").read_text())
    assert {job["kind"] for job in pending["jobs"]} == {"checkpoint", "adapter"}

    shutil.rmtree(checkpoint)

    second_api = FakeApi()
    restored = HFRunMirror(
        HFSyncConfig(repo_id="me/run"),
        run_id=RUN,
        run_dir=mirror.run_dir,
        output_dir=mirror.output_dir,
        results_dir=mirror.run_dir,
        api=second_api,
        start_worker=False,
        sleep=lambda _: None,
        download=lambda repo, path: b"{}",
    )
    restored.start()
    assert restored.process_pending() == 1
    assert second_api.folders[0]["path_in_repo"].endswith("step-000500")


def test_auto_repair_downloads_only_a_newer_remote_checkpoint(tmp_path):
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
        f"runs/{RUN}/checkpoint-2000/trainer_state.json",
        f"runs/{RUN}/checkpoint-2000/adapter_model.safetensors",
    ]
    blobs = {
        f"runs/{RUN}/checkpoint-2000/trainer_state.json": b"{}",
        f"runs/{RUN}/checkpoint-2000/adapter_model.safetensors": b"",
    }
    api = FakeApi(files)
    mirror = _mirror(tmp_path, api, download=lambda repo, path: blobs[path])
    _checkpoint(mirror.output_dir, 1000)

    assert mirror.auto_repair() is True
    repaired = mirror.output_dir / "checkpoint-2000"
    assert (repaired / "trainer_state.json").is_file()
    assert (repaired / "adapter_model.safetensors").is_file()
    assert not (mirror.output_dir / ".restore-checkpoint-2000.tmp").exists()

    assert mirror.auto_repair() is False


def test_auto_repair_returns_false_without_a_newer_remote(tmp_path):
    api = FakeApi([f"runs/{RUN}/run.json"])
    mirror = _mirror(tmp_path, api)
    _checkpoint(mirror.output_dir, 1000)
    assert mirror.auto_repair() is False


def test_adapter_retention_keeps_newest_two_after_success(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    paths = {}
    for step in (250, 500, 750):
        paths[step] = _complete_adapter(
            mirror.output_dir / "adapter_snapshots" / f"step-{step:06d}"
        )
    mirror.enqueue_adapter_snapshot(paths[750], 750)
    assert mirror.process_pending() == 1

    root = mirror.output_dir / "adapter_snapshots"
    assert (root / "step-000750").is_dir()
    assert (root / "step-000500").is_dir()
    assert not (root / "step-000250").exists()


def test_merged_upload_uses_run_scoped_path(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    merged = mirror.output_dir / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_text("")

    mirror.enqueue_merged(merged)
    assert mirror.process_pending() == 1
    assert api.folders[0]["path_in_repo"] == f"runs/{RUN}/merged"


def test_ledger_includes_reports_when_requested(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    (mirror.output_dir / "final_evaluation_report.json").write_text("{}")
    (mirror.output_dir / "final_evaluation_report.md").write_text("# report\n")
    reports = mirror.run_dir / "reports" / "nested"
    reports.mkdir(parents=True)
    (reports / "extra.json").write_text("{}")

    mirror.enqueue_ledger("post_eval", reports=True)
    assert mirror.process_pending() == 1
    paths = [op.path_in_repo for op in api.commits[0]["operations"]]
    assert f"runs/{RUN}/reports/final_evaluation_report.json" in paths
    assert f"runs/{RUN}/reports/final_evaluation_report.md" in paths
    assert f"runs/{RUN}/reports/nested/extra.json" in paths


def test_success_uploads_latest_and_hf_state(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1

    latest = [
        call for call in api.files_uploaded if call["path_in_repo"] == "latest.json"
    ]
    assert len(latest) == 1
    payload = json.loads(latest[0]["path_or_fileobj"])
    assert payload["run_id"] == RUN
    assert payload["last_step"] == 1000

    state = json.loads((mirror.run_dir / "hf_state.json").read_text())
    assert state["last_step"] == 1000
    assert state["checkpoints"] == [1000]

    status = json.loads((mirror.run_dir / "hf_sync.json").read_text())
    assert set(status) == {
        "enabled",
        "repo_id",
        "status",
        "last_success_at",
        "last_success_step",
        "last_error",
        "pending",
        "bytes_uploaded",
    }
    assert status["last_success_step"] == 1000
    assert status["last_error"] is None


def test_local_high_water_is_never_lowered(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    (mirror.run_dir / "hf_state.json").write_text(
        json.dumps({"last_step": 2000, "checkpoints": [2000]})
    )
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 1

    operations = api.commits[0]["operations"]
    state_op = next(
        op for op in operations if op.path_in_repo.endswith("/hf_state.json")
    )
    state = json.loads(_read_operation_bytes(state_op))
    assert state["last_step"] == 2000


def test_worker_drains_and_close_stops_it(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    mirror.start_worker = True
    mirror.start()
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)

    assert mirror.drain(5.0) is True
    assert [call["path_in_repo"] for call in api.folders] == [
        f"runs/{RUN}/checkpoint-1000"
    ]
    mirror.close()
    assert mirror.status()["pending"] == 0


def test_ledger_higher_remote_checkpoint_blocks_even_without_state(tmp_path):
    api = FakeApi(
        [
            f"runs/{RUN}/checkpoint-2000/trainer_state.json",
            f"runs/{RUN}/checkpoint-2000/adapter_model.safetensors",
        ]
    )
    blobs = {
        f"runs/{RUN}/checkpoint-2000/trainer_state.json": b"{}",
        f"runs/{RUN}/checkpoint-2000/adapter_model.safetensors": b"",
    }
    mirror = _mirror(tmp_path, api, download=lambda repo, path: blobs[path])
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 0
    assert api.commits == []
    assert "2000" in mirror.status()["last_error"]


def test_latest_upload_failure_is_retried_and_never_raises(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)

    def upload_file(**kwargs):
        if kwargs["path_in_repo"] == "latest.json":
            raise RuntimeError("latest boom")
        api.files_uploaded.append(kwargs)

    api.upload_file = upload_file
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)

    assert mirror.process_pending() == 0
    assert mirror.status()["status"] == "error"
    assert "latest boom" in mirror.status()["last_error"]
    assert len(api.folders) == 3


def test_unreadable_remote_state_blocks_ledger_when_state_exists(tmp_path):
    files = [
        f"runs/{RUN}/run.json",
        f"runs/{RUN}/hf_state.json",
        f"runs/{RUN}/checkpoint-1000/trainer_state.json",
        f"runs/{RUN}/checkpoint-1000/adapter_model.safetensors",
    ]
    api = FakeApi(files)

    def download(repo, path):
        raise RuntimeError("state unavailable")

    mirror = _mirror(tmp_path, api, download=download)
    mirror.start()
    mirror.enqueue_ledger("training")

    assert mirror.process_pending() == 0
    assert api.commits == []
    assert "hf_state.json" in mirror.status()["last_error"]


def test_ledger_uses_a_fresh_listing_instead_of_the_start_cache(tmp_path):
    class TrackingApi(FakeApi):
        def __init__(self):
            super().__init__([])
            self.create_calls = 0

        def create_commit(self, **kwargs):
            super().create_commit(**kwargs)
            self.create_calls += 1
            # A real HF repo exposes the committed hf_state.json next listing.
            self.files = [f"runs/{RUN}/hf_state.json"]

    api = TrackingApi()

    def download(repo, path):
        raise RuntimeError("download unavailable")

    mirror = _mirror(tmp_path, api, download=download)
    mirror.start()
    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 1

    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 0
    assert api.create_calls == 1
    assert "hf_state.json" in mirror.status()["last_error"]


def test_checkpoint_accounting_uses_actual_checkpoint_step(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    run = json.loads((mirror.run_dir / "run.json").read_text())
    run["last_step"] = 2000
    (mirror.run_dir / "run.json").write_text(json.dumps(run))
    path = _checkpoint(mirror.output_dir, 1000)

    mirror.enqueue_checkpoint(path, 1000)
    assert mirror.process_pending() == 1
    assert mirror.status()["last_success_step"] == 1000

    local_state = json.loads((mirror.run_dir / "hf_state.json").read_text())
    assert local_state["last_step"] == 2000
    assert local_state["checkpoints"] == [1000]

    mirror.enqueue_ledger("training")
    assert mirror.process_pending() == 1
    state_op = next(
        op
        for op in api.commits[-1]["operations"]
        if op.path_in_repo.endswith("/hf_state.json")
    )
    uploaded = json.loads(_read_operation_bytes(state_op))
    assert uploaded["last_step"] == 2000
    assert uploaded["checkpoints"] == [1000]


def test_status_is_uploading_while_a_job_is_active(tmp_path):
    api = FakeApi()
    mirror = _mirror(tmp_path, api)
    observed = []
    persisted = []

    def upload_folder(**kwargs):
        observed.append(mirror.status()["status"])
        status_path = mirror.run_dir / "hf_sync.json"
        persisted.append(json.loads(status_path.read_text())["status"])
        api.folders.append(kwargs)

    api.upload_folder = upload_folder
    path = _checkpoint(mirror.output_dir, 1000)
    mirror.enqueue_checkpoint(path, 1000)

    assert mirror.process_pending() == 1
    assert observed == ["uploading"]
    assert persisted == ["uploading"]
    assert mirror.status()["status"] == "idle"
