from pathlib import Path
from types import SimpleNamespace

from src.trainer_callbacks import HFMirrorCallback


class FakeMirror:
    def __init__(self):
        self.jobs = []

    def enqueue_checkpoint(self, path, step):
        self.jobs.append(("checkpoint", step, Path(path)))

    def enqueue_adapter_snapshot(self, path, step):
        self.jobs.append(("adapter", step, Path(path)))

    def enqueue_ledger(self, status="running"):
        self.jobs.append(("ledger", status, None))


class FakeModel:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        (Path(path) / "adapter_model.safetensors").write_text("")


class FakeProcessor:
    def save_pretrained(self, path):
        (Path(path) / "preprocessor_config.json").write_text("{}")


def test_snapshot_every_250_steps_and_skip_checkpoint_steps(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    cb.on_step_end(None, SimpleNamespace(global_step=250), None)
    assert mirror.jobs == [
        ("adapter", 250, tmp_path / "adapter_snapshots" / "step-000250")
    ]
    assert (
        tmp_path / "adapter_snapshots" / "step-000250" / "adapter_config.json"
    ).is_file()

    cb.on_step_end(None, SimpleNamespace(global_step=1000), None)
    assert [job[0] for job in mirror.jobs] == ["adapter"]


def test_on_save_enqueues_checkpoint(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    ckpt = tmp_path / "checkpoint-1000"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}")
    cb.on_save(None, SimpleNamespace(global_step=1000), None)
    assert mirror.jobs == [("checkpoint", 1000, ckpt)]


def test_train_end_enqueues_ledger(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
    )
    cb.on_train_end(None, SimpleNamespace(global_step=500), None)
    assert mirror.jobs == [("ledger", "training", None)]


class ExplodingModel:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "partial.txt").write_text("x")
        raise RuntimeError("disk full")


def test_snapshot_failure_does_not_stop_training(tmp_path, capsys):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=ExplodingModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
    )
    cb.on_step_end(None, SimpleNamespace(global_step=250), None)
    assert mirror.jobs == []
    assert not (
        tmp_path / "adapter_snapshots" / ".staging-step-000250"
    ).exists()
    assert "Warning" in capsys.readouterr().out


def test_snapshot_disabled_or_off_interval(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
    )
    for step in (0, 249, 251):
        cb.on_step_end(None, SimpleNamespace(global_step=step), None)
    assert mirror.jobs == []

    disabled = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=0,
    )
    disabled.on_step_end(None, SimpleNamespace(global_step=500), None)
    assert mirror.jobs == []


def test_train_begin_ledger_and_missing_checkpoint_is_ignored(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
    )
    cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
    cb.on_save(None, SimpleNamespace(global_step=1000), None)
    assert mirror.jobs == [("ledger", "training", None)]


def test_snapshot_skipped_when_control_will_save(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    cb.on_step_end(
        SimpleNamespace(save_strategy="steps"),
        SimpleNamespace(global_step=250, max_steps=2000),
        SimpleNamespace(should_save=True),
    )
    assert mirror.jobs == []


def test_snapshot_skipped_on_final_implicit_save(tmp_path):
    mirror = FakeMirror()
    cb = HFMirrorCallback(
        mirror,
        model=FakeModel(),
        processor=FakeProcessor(),
        output_dir=tmp_path,
        snapshot_steps=250,
        save_steps=1000,
    )
    # The final step of a steps-strategy run is always saved, even though
    # 1250 is not a save_steps multiple.
    cb.on_step_end(
        SimpleNamespace(save_strategy="steps"),
        SimpleNamespace(global_step=1250, max_steps=1250),
        SimpleNamespace(should_save=False),
    )
    assert mirror.jobs == []

    # A non-final snapshot step with no save due still snapshots.
    cb.on_step_end(
        SimpleNamespace(save_strategy="steps"),
        SimpleNamespace(global_step=250, max_steps=2000),
        SimpleNamespace(should_save=False),
    )
    assert [job[0] for job in mirror.jobs] == ["adapter"]
