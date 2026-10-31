import json
import re
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import paths, run_state

RUN_A = "20261030-194700-aaaaaa"
RUN_B = "20261031-194700-bbbbbb"
RUN_C = "20261029-194700-cccccc"


def _create(root, run_id=RUN_A, **kwargs):
    kwargs.setdefault("config", {})
    kwargs.setdefault("trainer_args", {})
    return run_state.create_run(root, run_id=run_id, **kwargs)


def _set_created(run_dir, created_at):
    return run_state.update_run(
        run_dir, lambda run: {**run, "created_at": created_at}
    )


def _stages_by_name(run):
    return {stage["name"]: stage for stage in run["stages"]}


def _events(run_dir):
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_new_run_id_format_and_validation():
    rid = run_state.new_run_id(now=1761846420.0)
    assert re.match(r"^\d{8}-\d{6}-[0-9a-f]{6}$", rid)
    assert run_state.run_id_valid(rid)
    assert not run_state.run_id_valid("../escape")
    assert not run_state.run_id_valid("20261030-194700-1A2B3C")
    assert not run_state.run_id_valid("20261030-194700-1a2b3")
    assert not run_state.run_id_valid("20261030-194700-1a2b3c/../x")
    assert not run_state.run_id_valid(rid + "\n")


def test_create_run_writes_manifest_and_pending_stages(tmp_path):
    run = run_state.create_run(
        tmp_path,
        config={"epochs": 4},
        trainer_args={"epochs": 4},
        paths={"output_dir": str(tmp_path / "run" / "output")},
        run_id=RUN_A,
    )
    run_dir = tmp_path / run["run_id"]
    assert run["status"] == "starting"
    assert run["schema_version"] == 1
    assert [s["name"] for s in run["stages"]] == [
        "preflight",
        "training",
        "post_eval",
        "merge",
    ]
    assert all(s["status"] == "pending" for s in run["stages"])
    assert run["attempts"] == []
    assert run["step_count"] == 0
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "output").is_dir()
    assert (run_dir / "baselines").is_dir()
    assert run["paths"]["run_dir"] == str(run_dir)
    assert run["paths"]["results_dir"] == str(run_dir)
    # An explicit path entry wins, the rest keep their defaults.
    assert run["paths"]["output_dir"] == str(tmp_path / "run" / "output")
    assert run["paths"]["merged_dir"] == str(run_dir / "merged")
    assert run_state.load_run(run_dir)["config"]["epochs"] == 4
    assert run_state.load_run(tmp_path / "missing") is None


def test_create_run_rejects_unsafe_run_id(tmp_path):
    with pytest.raises(ValueError):
        _create(tmp_path, run_id="../escape")


def test_create_run_rejects_duplicate_run_id(tmp_path):
    _create(tmp_path)
    with pytest.raises(FileExistsError):
        _create(tmp_path)


def test_list_runs_is_newest_first(tmp_path):
    root = tmp_path / "runs"
    _set_created(_create(root, RUN_A)["paths"]["run_dir"], 100.0)
    _set_created(_create(root, RUN_B)["paths"]["run_dir"], 200.0)

    runs = run_state.list_runs(root)
    assert [run["run_id"] for run in runs] == [RUN_B, RUN_A]
    assert [run["run_id"] for run in run_state.list_runs(root, limit=1)] == [RUN_B]
    assert run_state.list_runs(tmp_path / "does-not-exist") == []


def test_update_run_preserves_fields_across_writers(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    run_state.update_run(
        run_dir, lambda current: {**current, "config": {"epochs": 9}}
    )
    run_state.update_run(
        run_dir, lambda current: {**current, "last_error": "boom"}
    )

    loaded = run_state.load_run(run_dir)
    assert loaded["config"] == {"epochs": 9}
    assert loaded["last_error"] == "boom"
    assert loaded["updated_at"] >= loaded["created_at"]


def test_update_run_lock_prevents_lost_updates(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.update_run(run_dir, lambda current: {**current, "counter": 0})

    errors = []

    def bump():
        try:
            for _ in range(10):
                run_state.update_run(
                    run_dir,
                    lambda current: {**current, "counter": current["counter"] + 1},
                )
        except Exception as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert run_state.load_run(run_dir)["counter"] == 40


def test_update_run_accepts_in_place_mutators(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    def mutate(current):
        current["last_error"] = "in place"

    returned = run_state.update_run(run_dir, mutate)
    assert returned["last_error"] == "in place"
    assert run_state.load_run(run_dir)["last_error"] == "in place"


def test_set_stage_tracks_current_stage_and_timestamps(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    running = run_state.set_stage(
        run_dir, "preflight", "running", detail="measuring baseline"
    )
    stage = _stages_by_name(running)["preflight"]
    assert stage["status"] == "running"
    assert stage["detail"] == "measuring baseline"
    assert isinstance(stage["started_at"], float)
    assert stage["ended_at"] is None
    assert running["current_stage"] == "preflight"

    completed = run_state.set_stage(
        run_dir, "preflight", "completed", detail="cache hit"
    )
    stage = _stages_by_name(completed)["preflight"]
    assert stage["status"] == "completed"
    assert stage["detail"] == "cache hit"
    assert stage["ended_at"] >= stage["started_at"]

    training = run_state.set_stage(run_dir, "training", "running")
    assert training["current_stage"] == "training"
    assert _stages_by_name(training)["training"]["started_at"] is not None

    events = _events(run_dir)
    assert events[-1]["type"] == "stage"
    assert events[-1]["stage"] == "training"
    assert events[-1]["status"] == "running"
    assert any(
        event.get("type") == "stage" and event.get("stage") == "preflight"
        for event in events
    )

    with pytest.raises(ValueError):
        run_state.set_stage(run_dir, "bogus", "running")
    with pytest.raises(ValueError):
        run_state.set_stage(run_dir, "training", "bogus")


def test_set_status_sets_fields_and_terminal_timestamp(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    running = run_state.set_status(run_dir, "running", stop_requested_at=123.0)
    assert running["status"] == "running"
    assert running["stop_requested_at"] == 123.0
    assert running["ended_at"] is None

    failed = run_state.set_status(
        run_dir, "failed", failed_stage="training", last_error="boom"
    )
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "training"
    assert failed["last_error"] == "boom"
    assert isinstance(failed["ended_at"], float)

    events = _events(run_dir)
    assert events[-1]["type"] == "status"
    assert events[-1]["status"] == "failed"

    with pytest.raises(ValueError):
        run_state.set_status(run_dir, "bogus")


def test_set_progress_updates_run_manifest(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    updated = run_state.set_progress(
        run_dir, step=1234, step_count=1230, last_step_at=1761847000.0
    )
    assert updated["last_step"] == 1234
    assert updated["step_count"] == 1230
    assert updated["last_step_at"] == 1761847000.0
    assert run_state.load_run(run_dir)["step_count"] == 1230


def test_read_steps_missing_file_is_empty(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    assert run_state.read_steps(run_dir) == []
    assert run_state.step_count(run_dir) == 0


def test_append_step_reads_tail_and_skips_bad_trailing_line(tmp_path):
    run = _create(tmp_path)
    run_dir = Path(run["paths"]["run_dir"])

    for step in range(1, 6):
        run_state.append_step(run_dir, {"step": step, "loss": float(step)})

    with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{bad trailing line\n")

    assert [record["step"] for record in run_state.read_steps(run_dir, tail=3)] == [
        3,
        4,
        5,
    ]
    assert run_state.step_count(run_dir) == 5
    assert run_state.read_steps(run_dir, tail=0) == []
    assert len(run_state.read_steps(run_dir)) == 5


def test_read_steps_keeps_last_record_per_step(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    run_state.append_step(run_dir, {"step": 1, "loss": 9.0})
    run_state.append_step(run_dir, {"step": 2, "loss": 5.0})
    run_state.append_step(run_dir, {"step": 1, "loss": 1.0, "attempt": 2})

    records = run_state.read_steps(run_dir)
    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["loss"] == 1.0
    assert records[0]["attempt"] == 2
    assert run_state.step_count(run_dir) == 2


def test_attempt_lifecycle_and_reconcile_stopped(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.set_stage(run_dir, "preflight", "completed")
    run_state.set_stage(run_dir, "training", "running")
    run_state.set_status(run_dir, "running")

    started = run_state.start_attempt(run_dir, 4242, reason="initial")
    attempt = started["attempts"][-1]
    assert started["current_pid"] == 4242
    assert attempt["attempt"] == 1
    assert attempt["pid"] == 4242
    assert attempt["reason"] == "initial"
    assert attempt["ended_at"] is None
    assert attempt["exit_code"] is None

    run_state.update_run(
        run_dir, lambda current: {**current, "stop_requested_at": time.time()}
    )
    stopped = run_state.reconcile(
        run_dir, pid_alive=False, exit_code=None, reason="process_exit"
    )

    assert stopped["status"] == "stopped"
    stages = _stages_by_name(stopped)
    assert stages["training"]["status"] == "stopped"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "skipped"
    assert stopped["current_pid"] is None
    assert stopped["attempts"][-1]["ended_at"] is not None
    assert stopped["attempts"][-1]["reason"] == "halted"


def test_reconcile_stopped_without_running_stage_skips_later_stages(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.set_stage(run_dir, "preflight", "completed")
    run_state.update_run(
        run_dir, lambda current: {**current, "current_stage": "training"}
    )
    run_state.set_status(run_dir, "running")
    run_state.start_attempt(run_dir, 4243)
    run_state.update_run(
        run_dir, lambda current: {**current, "stop_requested_at": time.time()}
    )

    stopped = run_state.reconcile(
        run_dir, pid_alive=False, exit_code=None, reason="process_exit"
    )

    assert stopped["status"] == "stopped"
    stages = _stages_by_name(stopped)
    assert stages["preflight"]["status"] == "completed"
    assert stages["training"]["status"] == "pending"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "skipped"
    assert stopped["attempts"][-1]["reason"] == "halted"


def test_finish_attempt_clears_current_pid(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.start_attempt(run_dir, 5150, reason="resume")

    finished = run_state.finish_attempt(run_dir, 0, "completed")
    attempt = finished["attempts"][-1]
    assert finished["current_pid"] is None
    assert attempt["ended_at"] is not None
    assert attempt["exit_code"] == 0
    assert attempt["reason"] == "completed"


def test_finalize_attempt_closes_terminal_run_without_changing_status(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")
    run_state.set_status(run_dir, "completed")
    run_state.start_attempt(run_dir, 6161)
    run_state.update_run(
        run_dir, lambda current: {**current, "current_pid": 6161}
    )

    finalized = run_state.finalize_attempt(run_dir, exit_code=0)

    attempt = finalized["attempts"][-1]
    assert finalized["status"] == "completed"
    assert finalized["current_pid"] is None
    assert attempt["ended_at"] is not None
    assert attempt["exit_code"] == 0
    assert attempt["reason"] == "completed"


def test_finalize_attempt_maps_exit_and_requested_halt(tmp_path):
    failed_run = _create(tmp_path, run_id=RUN_A)
    failed_dir = failed_run["paths"]["run_dir"]
    run_state.set_status(failed_dir, "failed")
    run_state.start_attempt(failed_dir, 6162)
    failed = run_state.finalize_attempt(failed_dir, exit_code=3)
    assert failed["attempts"][-1]["reason"] == "nonzero_exit"
    assert failed["attempts"][-1]["exit_code"] == 3

    halt_run = _create(tmp_path, run_id=RUN_B)
    halt_dir = halt_run["paths"]["run_dir"]
    run_state.set_status(halt_run["paths"]["run_dir"], "running")
    run_state.start_attempt(halt_dir, 6163)
    run_state.update_run(
        halt_dir, lambda current: {**current, "stop_requested_at": time.time()}
    )
    halted = run_state.finalize_attempt(
        halt_dir, exit_code=0, reason="completed"
    )
    assert halted["attempts"][-1]["reason"] == "halted"


def test_set_stage_can_clear_stale_detail(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]
    run_state.set_stage(run_dir, "training", "failed", detail="prior boom")
    run_state.set_stage(run_dir, "training", "completed")
    assert _stages_by_name(run_state.load_run(run_dir))["training"]["detail"] == (
        "prior boom"
    )
    cleared = run_state.set_stage(
        run_dir, "training", "completed", detail=None
    )
    assert "detail" not in _stages_by_name(cleared)["training"]


def test_reconcile_classifies_failed_and_interrupted(tmp_path):
    failed_run = _create(tmp_path, run_id=RUN_A)
    failed_dir = failed_run["paths"]["run_dir"]
    run_state.set_stage(failed_dir, "training", "running")
    run_state.set_status(failed_dir, "running")
    run_state.start_attempt(failed_dir, 101)

    failed = run_state.reconcile(failed_dir, pid_alive=False, exit_code=1)
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "training"
    assert _stages_by_name(failed)["training"]["status"] == "failed"
    assert failed["attempts"][-1]["exit_code"] == 1
    assert failed["attempts"][-1]["reason"] == "nonzero_exit"

    interrupted_run = _create(tmp_path, run_id=RUN_B)
    interrupted_dir = interrupted_run["paths"]["run_dir"]
    run_state.set_stage(interrupted_dir, "training", "running")
    run_state.set_status(interrupted_dir, "running")
    run_state.start_attempt(interrupted_dir, 102)

    interrupted = run_state.reconcile(
        interrupted_dir, pid_alive=False, exit_code=None
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["failed_stage"] is None
    assert _stages_by_name(interrupted)["training"]["status"] == "interrupted"
    assert interrupted["attempts"][-1]["exit_code"] is None
    assert interrupted["attempts"][-1]["reason"] == "process_lost"


def test_reconcile_exit_zero_requires_terminal_stages(tmp_path):
    run = _create(tmp_path, run_id=RUN_A)
    run_dir = run["paths"]["run_dir"]
    run_state.set_stage(run_dir, "preflight", "completed")
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_status(run_dir, "running")
    run_state.start_attempt(run_dir, 201)

    failed = run_state.reconcile(run_dir, pid_alive=False, exit_code=0)
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "post_eval"
    assert _stages_by_name(failed)["post_eval"]["status"] == "failed"

    completed_run = _create(tmp_path, run_id=RUN_B)
    completed_dir = completed_run["paths"]["run_dir"]
    run_state.set_stage(completed_dir, "preflight", "completed")
    run_state.set_stage(completed_dir, "training", "completed")
    run_state.set_stage(completed_dir, "post_eval", "skipped")
    run_state.set_status(completed_dir, "running")
    run_state.start_attempt(completed_dir, 202)

    completed = run_state.reconcile(completed_dir, pid_alive=False, exit_code=0)
    assert completed["status"] == "completed"
    assert completed["failed_stage"] is None
    assert completed["current_pid"] is None
    assert completed["attempts"][-1]["reason"] == "completed"


def test_reconcile_leaves_a_fresh_starting_run_alone(tmp_path):
    run = _create(tmp_path)
    run_dir = run["paths"]["run_dir"]

    fresh = run_state.reconcile(run_dir, pid_alive=False)
    assert fresh["status"] == "starting"
    assert fresh["attempts"] == []

    _set_created(run_dir, time.time() - 16)
    stale = run_state.reconcile(run_dir, pid_alive=False)
    assert stale["status"] == "interrupted"


def test_reconcile_ignores_live_and_terminal_runs(tmp_path):
    run = _create(tmp_path, run_id=RUN_A)
    run_dir = run["paths"]["run_dir"]
    run_state.start_attempt(run_dir, 301)
    run_state.set_status(run_dir, "running")

    live = run_state.reconcile(run_dir, pid_alive=True, exit_code=1)
    assert live["status"] == "running"
    assert live["attempts"][-1]["ended_at"] is None

    done_run = _create(tmp_path, run_id=RUN_B)
    done_dir = done_run["paths"]["run_dir"]
    run_state.set_status(done_dir, "completed")
    done = run_state.reconcile(done_dir, pid_alive=False, exit_code=1)
    assert done["status"] == "completed"


def test_reconcile_all_finds_older_live_run(tmp_path):
    root = tmp_path / "runs"

    older = _create(root, RUN_A)
    older_dir = older["paths"]["run_dir"]
    _set_created(older_dir, 100.0)
    run_state.start_attempt(older_dir, 101)
    run_state.set_status(older_dir, "running")

    newer = _create(root, RUN_B)
    newer_dir = newer["paths"]["run_dir"]
    _set_created(newer_dir, 200.0)
    run_state.start_attempt(newer_dir, 202)
    run_state.set_status(newer_dir, "running")

    summary = run_state.reconcile_all(root, is_alive=lambda pid: pid == 101)

    assert summary["active_run"]["run_id"] == RUN_A
    assert summary["latest_run"]["run_id"] == RUN_B
    assert summary["merge_run"] is None
    by_id = {run["run_id"]: run for run in summary["runs"]}
    assert by_id[RUN_A]["status"] == "running"
    assert by_id[RUN_B]["status"] == "interrupted"


def test_reconcile_all_ignores_terminal_run_with_reused_live_pid(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_A)
    run_dir = run["paths"]["run_dir"]
    run_state.start_attempt(run_dir, 1234)
    run_state.set_stage(run_dir, "training", "completed")
    run_state.set_stage(run_dir, "post_eval", "completed")
    run_state.set_status(run_dir, "completed")
    # The PID now belongs to an unrelated live process (PID reuse).
    run_state.update_run(
        run_dir, lambda current: {**current, "current_pid": 1234}
    )

    summary = run_state.reconcile_all(root, is_alive=lambda pid: pid == 1234)

    assert summary["active_run"] is None
    assert summary["latest_run"]["status"] == "completed"


def test_reconcile_all_finds_pid_file_only_cli_run(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_A)
    run_dir = Path(run["paths"]["run_dir"])
    run_state.set_status(run_dir, "running")
    (run_dir / "training.pid").write_text("4321", encoding="utf-8")

    summary = run_state.reconcile_all(root, is_alive=lambda pid: pid == 4321)

    assert summary["active_run"]["run_id"] == RUN_A
    assert summary["runs"][0]["status"] == "running"


def test_reconcile_all_treats_is_alive_errors_as_alive(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_A)
    run_dir = run["paths"]["run_dir"]
    run_state.start_attempt(run_dir, 4040)
    run_state.set_status(run_dir, "running")

    def probe(pid):
        raise OSError("process probe unavailable")

    summary = run_state.reconcile_all(root, is_alive=probe)

    assert summary["active_run"]["run_id"] == RUN_A
    assert summary["latest_run"]["status"] == "running"
    assert summary["runs"][0]["current_pid"] == 4040


def test_reconcile_all_reconciles_scanned_directory(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_A)
    run_dir = Path(run["paths"]["run_dir"])
    # Simulate a manifest whose id no longer matches its directory name. The
    # scanner must reconcile the directory it actually read, not a synthetic
    # runs_root / manifest["run_id"] path.
    run_state.update_run(run_dir, lambda current: {**current, "run_id": RUN_B})
    run_state.start_attempt(run_dir, 5050)
    run_state.set_status(run_dir, "running")

    summary = run_state.reconcile_all(root, is_alive=lambda pid: False)

    assert summary["latest_run"]["run_id"] == RUN_B
    assert run_state.load_run(run_dir)["status"] == "interrupted"
    assert not (root / RUN_B).exists()


def test_reconcile_all_reports_live_merge_run(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_C)
    run_dir = run["paths"]["run_dir"]
    run_state.set_status(run_dir, "completed")
    run_state.update_run(run_dir, lambda current: {**current, "merge_pid": 303})

    summary = run_state.reconcile_all(root, is_alive=lambda pid: pid == 303)
    assert summary["active_run"] is None
    assert summary["merge_run"]["run_id"] == RUN_C

    summary = run_state.reconcile_all(root, is_alive=lambda pid: False)
    assert summary["merge_run"] is None


def test_storage_info_marks_workspace_persistent(tmp_path, monkeypatch):
    workspace = run_state.storage_info(paths.RUNS_DIR_PERSISTENT)
    assert workspace["persistent"] is True
    assert workspace["runs_dir"] == paths.RUNS_DIR_PERSISTENT

    local = run_state.storage_info(tmp_path / "runs")
    assert local["persistent"] is False
    assert local["runs_dir"] == str(tmp_path / "runs")


def test_write_preflight_copy_is_atomic_and_readable(tmp_path):
    run = _create(tmp_path / "runs")
    run_dir = Path(run["paths"]["run_dir"])
    source = tmp_path / "shared" / "preflight.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"clean_wer": 0.1}), encoding="utf-8")

    dest = run_state.write_preflight_copy(run_dir, source)
    assert dest == run_dir / "baselines" / "preflight.json"
    assert json.loads(dest.read_text(encoding="utf-8")) == {"clean_wer": 0.1}
    assert not (dest.parent / "preflight.json.tmp").exists()

    source.write_text(json.dumps({"clean_wer": 0.2}), encoding="utf-8")
    run_state.write_preflight_copy(run_dir, source)
    assert json.loads(dest.read_text(encoding="utf-8")) == {"clean_wer": 0.2}


def test_reconcile_all_reports_duplicate_live_runs(tmp_path):
    root = tmp_path / "runs"

    older = _create(root, RUN_A)
    older_dir = older["paths"]["run_dir"]
    _set_created(older_dir, 100.0)
    run_state.start_attempt(older_dir, 101)
    run_state.set_status(older_dir, "running")

    newer = _create(root, RUN_B)
    newer_dir = newer["paths"]["run_dir"]
    _set_created(newer_dir, 200.0)
    run_state.start_attempt(newer_dir, 202)
    run_state.set_status(newer_dir, "running")

    summary = run_state.reconcile_all(root, is_alive=lambda pid: True)

    assert summary["active_run"]["run_id"] == RUN_B
    assert summary["duplicates"] == [RUN_A]


def test_reconcile_all_duplicates_empty_with_single_live_run(tmp_path):
    root = tmp_path / "runs"
    run = _create(root, RUN_A)
    run_dir = run["paths"]["run_dir"]
    run_state.start_attempt(run_dir, 101)
    run_state.set_status(run_dir, "running")

    summary = run_state.reconcile_all(root, is_alive=lambda pid: pid == 101)
    assert summary["duplicates"] == []
    assert summary["active_run"]["run_id"] == RUN_A
