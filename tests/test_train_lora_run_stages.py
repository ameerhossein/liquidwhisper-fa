"""Offline tests for trainer run-ledger stages and preflight plumbing."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import run_state

RUN_ID = "20261030-194700-1a2b3c"


def _train_lora():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from scripts import train_lora

    return train_lora


def _run(tmp_path, run_id=RUN_ID):
    run = run_state.create_run(
        tmp_path, config={}, trainer_args={}, run_id=run_id
    )
    return Path(tmp_path) / run["run_id"]


def _main_args(tmp_path, run_dir, **overrides):
    """Minimal argparse namespace consumed by main() before/at data prep."""
    values = dict(
        run_dir=str(run_dir),
        output_dir=str(Path(run_dir) / "output"),
        results_dir=str(run_dir),
        fresh_start=False,
        telephony_profiles="customer",
        base_model="BuzzASR/persian",
        dataset_name="test/dataset",
        lora_r=64,
        lora_alpha=128,
        batch_size=8,
        grad_accum=4,
        telephony_prob=0.0,
        telephony_use_ffmpeg=True,
        phone_benchmark=False,
        phone_seed=1234,
        num_proc=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stages(run_dir):
    run = run_state.load_run(run_dir)
    return {stage["name"]: stage for stage in run["stages"]}


def test_parser_accepts_run_dir_and_preflight_cache(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(sys, "argv", [
        "train_lora.py", "--run_dir", str(tmp_path / "run"),
        "--preflight_cache_dir", str(tmp_path / "cache"),
    ])
    args = train_lora.parse_args()
    assert args.run_dir == str(tmp_path / "run")
    assert args.preflight_cache_dir == str(tmp_path / "cache")
    assert args.post_eval_only is False


def test_parser_accepts_post_eval_only_flag(tmp_path, monkeypatch):
    train_lora = _train_lora()
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--post_eval_only"])
    assert train_lora.parse_args().post_eval_only is True
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--no-post_eval_only"])
    assert train_lora.parse_args().post_eval_only is False


def test_copy_preflight_to_run_writes_manifest_copy(tmp_path):
    train_lora = _train_lora()
    cache = tmp_path / "cache"
    (cache / "baselines").mkdir(parents=True)
    src = cache / "baselines" / "preflight.json"
    src.write_text('{"models": {}}')
    run = tmp_path / "run"
    dest = train_lora._copy_preflight_to_run(run, cache)
    assert dest == run / "baselines" / "preflight.json"
    assert dest.read_text() == '{"models": {}}'


def test_record_trainer_args_stores_full_namespace(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    train_lora._record_trainer_args(
        run_dir, {"lora_r": 64, "warmup_steps": 500}
    )
    saved = run_state.load_run(run_dir)
    assert saved["trainer_args"]["lora_r"] == 64
    assert saved["trainer_args"]["warmup_steps"] == 500


def test_record_trainer_args_is_json_safe_for_paths_and_lists(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = SimpleNamespace(
        run_dir=tmp_path / "run",
        telephony_profiles=["customer"],
        lora_r=64,
        warmup_steps=None,
        ratios=[0.5, 1],
    )
    train_lora._record_trainer_args(run_dir, args)
    saved = run_state.load_run(run_dir)
    assert saved["trainer_args"]["run_dir"] == str(tmp_path / "run")
    assert saved["trainer_args"]["telephony_profiles"] == ["customer"]
    assert saved["trainer_args"]["ratios"] == [0.5, 1]
    assert saved["trainer_args"]["warmup_steps"] is None


def test_record_trainer_args_noops_without_run_dir(tmp_path):
    train_lora = _train_lora()
    assert train_lora._record_trainer_args(None, {"lora_r": 64}) is None


def test_mark_run_stage_noops_without_run_dir():
    train_lora = _train_lora()
    assert train_lora._mark_run_stage(None, "preflight", "running") is None


def test_mark_run_stage_updates_stage_detail(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    train_lora._mark_run_stage(
        run_dir, "preflight", "running", detail="preparing datasets"
    )
    stage = _stages(run_dir)["preflight"]
    assert stage["status"] == "running"
    assert stage["detail"] == "preparing datasets"


def test_fresh_start_clears_step_and_event_logs(tmp_path):
    train_lora = _train_lora()
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"
    output_dir.mkdir()
    results_dir.mkdir()
    for name in (
        "loss_history.json",
        "steps.jsonl",
        "events.jsonl",
        "live_telemetry.json",
        "training_live.log",
    ):
        (results_dir / name).write_text("stale", encoding="utf-8")

    train_lora._clear_previous_state(str(output_dir), results_dir=str(results_dir))

    for name in ("loss_history.json", "steps.jsonl", "events.jsonl"):
        assert not (results_dir / name).exists()
    # The server owns the live log/telemetry files; fresh start must not unlink them.
    assert (results_dir / "live_telemetry.json").exists()
    assert (results_dir / "training_live.log").exists()


def test_plain_cli_fresh_start_clears_previous_state(tmp_path, monkeypatch):
    train_lora = _train_lora()
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"
    output_dir.mkdir()
    results_dir.mkdir()
    (output_dir / "checkpoint-1").mkdir()
    (output_dir / "checkpoint-1" / "trainer_state.json").write_text(
        "{}", encoding="utf-8"
    )
    (output_dir / "best_adapter").mkdir()
    (output_dir / "best_adapter" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    for name in (
        "loss_history.json",
        "steps.jsonl",
        "events.jsonl",
        "epoch_benchmarks.json",
        "stop_training.signal",
    ):
        (results_dir / name).write_text("stale", encoding="utf-8")

    args = SimpleNamespace(
        run_dir=None,
        output_dir=str(output_dir),
        results_dir=str(results_dir),
        fresh_start=True,
        post_eval_only=False,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    observed = {}

    def fake_run(parsed, run_dir):
        observed["run_dir"] = run_dir
        observed["checkpoint"] = (output_dir / "checkpoint-1").exists()
        observed["adapter"] = (output_dir / "best_adapter").exists()
        for name in (
            "loss_history.json",
            "steps.jsonl",
            "events.jsonl",
            "epoch_benchmarks.json",
            "stop_training.signal",
        ):
            observed[name] = (results_dir / name).exists()

    monkeypatch.setattr(train_lora, "_run_training", fake_run)

    train_lora.main()

    assert observed["run_dir"] is None
    assert observed["checkpoint"] is False
    assert observed["adapter"] is False
    for name in (
        "loss_history.json",
        "steps.jsonl",
        "events.jsonl",
        "epoch_benchmarks.json",
        "stop_training.signal",
    ):
        assert observed[name] is False, name


def test_main_halts_before_data_prep_when_signal_exists(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")
    args = _main_args(tmp_path, run_dir)
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)

    class _NeverCalled:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("data preparation must not start after a halt")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _NeverCalled)
    train_lora.main()

    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    assert run["current_stage"] == "preflight"
    stages = _stages(run_dir)
    assert stages["preflight"]["status"] == "stopped"
    assert stages["training"]["status"] == "skipped"
    assert stages["post_eval"]["status"] == "skipped"
    assert stages["merge"]["status"] == "skipped"
    assert not (run_dir / "stop_training.signal").exists()
    assert run["current_pid"] is None
    assert run["attempts"][-1]["reason"] == "halted"


def test_main_cli_run_dir_is_self_contained_and_finalizes(tmp_path, monkeypatch):
    train_lora = _train_lora()
    runs_root = tmp_path / "runs"
    run_dir = runs_root / RUN_ID
    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=train_lora._default_output_dir(),
        results_dir=None,
        fresh_start=False,
        post_eval_only=False,
        telephony_profiles="customer",
    )
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)
    observed = {}

    def fake_run(parsed, current_run_dir):
        run = run_state.load_run(current_run_dir)
        observed["results_dir"] = parsed.results_dir
        observed["output_dir"] = parsed.output_dir
        observed["run_dir"] = str(current_run_dir)
        observed["current_pid"] = run["current_pid"]
        observed["attempts"] = list(run["attempts"])
        observed["pid_file"] = (
            Path(current_run_dir) / "training.pid"
        ).read_text(encoding="utf-8")
        # The run-scoped results directory is what the callback writes into.
        run_state.append_step(current_run_dir, {"step": 1, "loss": 0.5})
        run_state.set_status(
            current_run_dir, "completed", failed_stage=None, last_error=None
        )

    monkeypatch.setattr(train_lora, "_run_training", fake_run)

    train_lora.main()

    expected_run = run_dir.resolve()
    assert observed["run_dir"] == str(expected_run)
    assert observed["results_dir"] == str(expected_run)
    assert observed["output_dir"] == str(expected_run / "output")
    assert observed["current_pid"] == os.getpid()
    assert observed["pid_file"] == str(os.getpid())
    assert len(observed["attempts"]) == 1
    assert observed["attempts"][0]["pid"] == os.getpid()

    run = run_state.load_run(expected_run)
    assert run["source"] == "cli"
    assert run["status"] == "completed"
    assert run["paths"]["results_dir"] == str(expected_run)
    assert run["paths"]["output_dir"] == str(expected_run / "output")
    assert run["current_pid"] is None
    assert run["attempts"][-1]["ended_at"] is not None
    assert run["attempts"][-1]["reason"] == "completed"
    assert (expected_run / "steps.jsonl").is_file()
    assert not (expected_run / "training.pid").exists()


def test_main_marks_preflight_failed_when_data_prep_fails(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = _main_args(tmp_path, run_dir)
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)

    class _BrokenProcessor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("processor boom")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _BrokenProcessor)
    with pytest.raises(RuntimeError, match="processor boom"):
        train_lora.main()

    run = run_state.load_run(run_dir)
    assert run["status"] == "failed"
    assert run["failed_stage"] == "preflight"
    assert run["last_error"] == "processor boom"
    assert run["current_pid"] is None
    assert run["attempts"][-1]["ended_at"] is not None
    assert run["attempts"][-1]["exit_code"] == 1
    stage = _stages(run_dir)["preflight"]
    assert stage["status"] == "failed"


def test_run_attempt_number_reads_latest_attempt(tmp_path):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    assert train_lora._run_attempt_number(None) == 1
    assert train_lora._run_attempt_number(run_dir) == 1

    run_state.start_attempt(run_dir, 1234)
    assert train_lora._run_attempt_number(run_dir) == 1
    run_state.finish_attempt(run_dir, 0, "completed")
    run_state.start_attempt(run_dir, 5678, reason="resume")
    assert train_lora._run_attempt_number(run_dir) == 2


def test_post_eval_only_calls_report_without_training(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    output_dir = run_dir / "output"
    adapter_path = output_dir / "best_adapter"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    baseline_models = {
        "buzzasr_base": {"total_samples": 1000, "normalized_wer": 17.2}
    }
    (run_dir / "baselines").mkdir(exist_ok=True)
    (run_dir / "baselines" / "preflight.json").write_text(
        json.dumps({"models": baseline_models}), encoding="utf-8"
    )

    checkpoint = output_dir / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 100, "log_history": [{"step": 100, "loss": 0.4}]}),
        encoding="utf-8",
    )

    calls = {}

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, *a, **k):
            calls["processor_path"] = str(path)
            return SimpleNamespace()

    class _FakeBaseModel:
        @classmethod
        def from_pretrained(cls, path, *a, **k):
            calls["base_model"] = str(path)
            return SimpleNamespace()

    class _FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, *a, **k):
            calls["adapter_base"] = model
            calls["adapter_path"] = str(path)
            return SimpleNamespace(name="peft")

    class _TrainerMustNotRun:
        def __init__(self, *a, **k):
            raise AssertionError("post-eval-only must not construct a Trainer")

    def fake_report(**kwargs):
        calls["report"] = kwargs

    monkeypatch.setattr(train_lora, "WhisperProcessor", _FakeProcessor)
    monkeypatch.setattr(train_lora, "WhisperForConditionalGeneration", _FakeBaseModel)
    monkeypatch.setattr(train_lora, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(train_lora, "Seq2SeqTrainer", _TrainerMustNotRun)
    monkeypatch.setattr(train_lora, "_run_post_training_evaluation", fake_report)

    # Simulate a previous failed post-eval attempt: a successful resume must
    # not leave the old failure visible on a completed run.
    run_state.set_stage(run_dir, "post_eval", "failed", detail="prior boom")
    run_state.set_status(
        run_dir, "failed", failed_stage="post_eval", last_error="prior boom"
    )
    run_state.set_progress(
        run_dir, step=100, step_count=100, last_step_at=123.0
    )
    (run_dir / "live_telemetry.json").write_text(
        json.dumps({"status": "initializing", "step": 0, "max_steps": 0}),
        encoding="utf-8",
    )

    args = _main_args(tmp_path, run_dir, post_eval_only=True)
    train_lora._run_post_eval_only(args)

    assert calls["processor_path"] == str(adapter_path)
    assert calls["base_model"] == "BuzzASR/persian"
    assert calls["adapter_path"] == str(adapter_path)
    report = calls["report"]
    assert report["trainer_state"].global_step == 100
    assert report["baseline_benchmark"] == baseline_models
    assert report["output_dir"] == str(output_dir)
    assert report["results_dir"] == str(run_dir)
    assert report["model"].name == "peft"
    assert callable(report["should_stop"])

    run = run_state.load_run(run_dir)
    assert run["status"] == "completed"
    assert run["failed_stage"] is None
    assert run["last_error"] is None
    post_eval = _stages(run_dir)["post_eval"]
    assert post_eval["status"] == "completed"
    assert "detail" not in post_eval
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "completed"
    assert telemetry["step"] == 100
    # Post-eval-only must never replay or append optimizer steps.
    assert not (run_dir / "steps.jsonl").exists()


def test_post_eval_only_marks_post_eval_failed_when_adapter_missing(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = _main_args(tmp_path, run_dir, post_eval_only=True)

    monkeypatch.setattr(
        train_lora,
        "_run_post_training_evaluation",
        lambda **kwargs: pytest.fail("report must not run without an adapter"),
    )
    with pytest.raises(SystemExit, match="adapter_config.json not found"):
        train_lora._run_post_eval_only(args)

    run = run_state.load_run(run_dir)
    assert run["status"] == "failed"
    assert run["failed_stage"] == "post_eval"
    assert "adapter_config.json not found" in run["last_error"]
    assert _stages(run_dir)["post_eval"]["status"] == "failed"


def test_main_runs_post_eval_only_before_processor_load(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = _main_args(tmp_path, run_dir, post_eval_only=True)
    monkeypatch.setattr(train_lora, "parse_args", lambda: args)

    seen = []
    monkeypatch.setattr(
        train_lora, "_run_post_eval_only", lambda parsed: seen.append(parsed)
    )

    class _NeverCalled:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("post-eval-only must branch before data prep")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _NeverCalled)
    train_lora.main()

    assert seen == [args]


class _FakeEvalDataset:
    """Mapping-like test split: slices return column lists, iteration yields rows."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            rows = self._rows[index]
            return {
                "audio": [row["audio"] for row in rows],
                "text": [row["text"] for row in rows],
            }
        return self._rows[index]


class _FakeFeatures:
    def to(self, device):
        return self


class _FakeEvalProcessor:
    def __call__(self, waveforms, sampling_rate=16000, return_tensors="pt"):
        return SimpleNamespace(input_features=_FakeFeatures())

    def batch_decode(self, pred_ids, skip_special_tokens=True):
        return ["متن"] * len(pred_ids)


class _FakeEvalModel:
    def __init__(self):
        self.generate_calls = 0
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def eval(self):
        return self

    def generate(self, input_features=None, max_new_tokens=225):
        self.generate_calls += 1
        return [[1, 2]]


class _FakeEvalMetrics:
    def evaluate_predictions(self, references, predictions, is_informal_flags):
        return {
            "total_samples": len(references),
            "overall": {
                "normalized_wer": 10.0,
                "normalized_cer": 2.0,
                "diagnostic_wer_zwnj_space": 9.0,
            },
            "formal": {"normalized_wer": 10.0, "count": 1},
            "informal": {"normalized_wer": 10.0, "count": 0},
        }


def _exercise_post_training_eval(
    train_lora, monkeypatch, tmp_path, model, *, should_stop, device=None
):
    rows = [
        {
            "text": "سلام",
            "informal": False,
            "audio": {"array": [0.0] * 160, "sampling_rate": 16000},
        }
    ]

    class _FakePipeline:
        def __init__(self, *a, **k):
            pass

        def load_raw_dataset(self, split):
            assert split == "test"
            return _FakeEvalDataset(rows)

    monkeypatch.setattr(train_lora, "PersianASRDataPipeline", _FakePipeline)
    if device is not None:
        monkeypatch.setattr(train_lora, "_resolve_eval_device", lambda: device)

    train_lora._run_post_training_evaluation(
        model=model,
        processor=_FakeEvalProcessor(),
        metrics_calculator=_FakeEvalMetrics(),
        normalizer=SimpleNamespace(is_colloquial=lambda text: False),
        dataset_name="test/dataset",
        output_dir=str(Path(tmp_path) / "output"),
        results_dir=str(Path(tmp_path) / "results"),
        eval_batch_size=1,
        phone_benchmark=True,
        should_stop=should_stop,
    )


def test_post_training_evaluation_halts_between_passes_on_eval_device(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    model = _FakeEvalModel()
    stop_calls = []

    def should_stop():
        stop_calls.append(1)
        return len(stop_calls) > 1

    with pytest.raises(RuntimeError, match="halt requested"):
        _exercise_post_training_eval(
            train_lora, monkeypatch, tmp_path, model,
            should_stop=should_stop, device="cuda",
        )

    # Model placement is the eval function's responsibility; the post-eval-only
    # loader may hand it a CPU-resident PEFT model.
    assert model.devices == ["cuda"]
    # The signal was consumed between the clean and phone passes.
    assert model.generate_calls == 1
    assert len(stop_calls) == 2


def test_post_eval_halt_after_phone_pass_writes_no_report_and_marks_stopped(
    tmp_path, monkeypatch
):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    signal = run_dir / "stop_training.signal"
    signal.write_text("stop", encoding="utf-8")
    model = _FakeEvalModel()
    stop_calls = []

    def should_stop():
        stop_calls.append(1)
        return len(stop_calls) >= 3

    def evaluate():
        _exercise_post_training_eval(
            train_lora,
            monkeypatch,
            tmp_path,
            model,
            should_stop=should_stop,
        )

    completed = train_lora._run_post_eval_stage(
        run_dir=run_dir,
        results_dir=run_dir,
        evaluate=evaluate,
    )

    assert completed is False
    # Clean + phone passes ran; the third check halted before the report.
    assert model.generate_calls == 2
    assert len(stop_calls) == 3
    assert not (tmp_path / "output" / "final_evaluation_report.json").exists()
    assert not (tmp_path / "output" / "final_evaluation_report.md").exists()
    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    assert _stages(run_dir)["post_eval"]["status"] == "stopped"
    assert not signal.exists()
    telemetry = json.loads(
        (run_dir / "live_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "stopped"


def test_post_eval_only_honors_stop_signal_before_start(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    args = _main_args(tmp_path, run_dir, post_eval_only=True)
    (run_dir / "stop_training.signal").write_text("stop", encoding="utf-8")

    monkeypatch.setattr(
        train_lora,
        "_run_post_training_evaluation",
        lambda **kwargs: pytest.fail("halted post-eval must not start"),
    )
    train_lora._run_post_eval_only(args)

    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    stages = _stages(run_dir)
    assert stages["post_eval"]["status"] == "stopped"
    assert stages["merge"]["status"] == "skipped"
    assert not (run_dir / "stop_training.signal").exists()


def test_post_eval_only_marks_stopped_when_report_halts(tmp_path, monkeypatch):
    train_lora = _train_lora()
    run_dir = _run(tmp_path)
    adapter_path = run_dir / "output" / "best_adapter"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return SimpleNamespace()

    class _FakeBaseModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return SimpleNamespace()

    class _FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, *a, **k):
            return model

    def halt_report(**kwargs):
        raise RuntimeError("halt requested")

    monkeypatch.setattr(train_lora, "WhisperProcessor", _FakeProcessor)
    monkeypatch.setattr(train_lora, "WhisperForConditionalGeneration", _FakeBaseModel)
    monkeypatch.setattr(train_lora, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(train_lora, "_run_post_training_evaluation", halt_report)

    args = _main_args(tmp_path, run_dir, post_eval_only=True)
    train_lora._run_post_eval_only(args)  # a halt is not a failure

    run = run_state.load_run(run_dir)
    assert run["status"] == "stopped"
    assert run["failed_stage"] is None
    stages = _stages(run_dir)
    assert stages["post_eval"]["status"] == "stopped"
    assert stages["merge"]["status"] == "skipped"
    assert not (run_dir / "stop_training.signal").exists()
