"""Adapter publication and run-stage recovery without model downloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_lora
from src import run_state


class _Model:
    generation_config = None
    config = SimpleNamespace(use_cache=True)

    def print_trainable_parameters(self):
        pass

    def save_pretrained(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text('{}')
        (root / "adapter_model.safetensors").write_bytes(b"new weights")


class _Processor:
    def save_pretrained(self, directory):
        (Path(directory) / "preprocessor_config.json").write_text('{}')


@pytest.mark.parametrize("failure", ["processor", "publish", None])
def test_adapter_publication_preserves_previous_save_on_failure(tmp_path, monkeypatch, failure):
    target = tmp_path / "best_adapter"
    _Model().save_pretrained(target)
    (target / "adapter_model.safetensors").write_bytes(b"previous weights")
    processor = _Processor()
    if failure == "processor":
        def fail(directory):
            raise OSError("processor save failed")
        monkeypatch.setattr(processor, "save_pretrained", fail)
    if failure == "publish":
        replace = train_lora.os.replace

        def fail_publish(source, destination):
            if Path(source).name.startswith(".best_adapter.staging-"):
                raise OSError("publish failed")
            return replace(source, destination)

        monkeypatch.setattr(train_lora.os, "replace", fail_publish)
    if failure:
        with pytest.raises(OSError):
            train_lora._save_best_adapter(_Model(), processor, str(tmp_path))
        assert (target / "adapter_model.safetensors").read_bytes() == b"previous weights"
    else:
        assert train_lora._save_best_adapter(_Model(), processor, str(tmp_path)) == target
        assert (target / "adapter_model.safetensors").read_bytes() == b"new weights"
        assert (target / "preprocessor_config.json").is_file()
    assert not list(tmp_path.glob(".best_adapter.*"))


@pytest.mark.parametrize("save_fails", [False, True])
def test_training_completes_only_after_adapter_and_processor_are_published(tmp_path, monkeypatch, save_fails):
    run = run_state.create_run(tmp_path, config={}, trainer_args={})
    run_dir = tmp_path / run["run_id"]
    run_state.set_status(run_dir, "running")
    args = train_lora.build_parser().parse_args([])
    args.output_dir = str(run_dir / "output")
    args.results_dir = str(run_dir)
    args.prep_cache_dir = str(tmp_path / "prep")
    args.base_model = "test/model"
    args.telephony_profiles = ["customer"]
    args.enable_telemetry = args.run_epoch_benchmark = args.run_post_test_eval = False
    args.num_proc = 0
    model = _Model()
    processor = _Processor()
    save_processor = processor.save_pretrained

    def save(directory):
        stages = {s["name"]: s["status"] for s in run_state.load_run(run_dir)["stages"]}
        assert stages["training"] == "running"
        assert not (Path(args.output_dir) / "best_adapter").exists()
        if save_fails:
            raise OSError("processor failed")
        save_processor(directory)

    processor.save_pretrained = save

    class Trainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_lora, "WhisperProcessor", SimpleNamespace(from_pretrained=lambda *a, **k: processor))
    monkeypatch.setattr(train_lora, "WhisperForConditionalGeneration", SimpleNamespace(from_pretrained=lambda *a, **k: model))
    monkeypatch.setattr(train_lora, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(train_lora, "PersianASRDataPipeline", lambda **kwargs: object())
    monkeypatch.setattr(train_lora, "_label_format", lambda model: (None, None))
    monkeypatch.setattr(train_lora.prep_cache, "load_prep_cache", lambda *a: [object()])
    monkeypatch.setattr(train_lora, "_ensure_prep_cache_from_hf", lambda *a: None)
    monkeypatch.setattr(train_lora, "run_preflight", lambda **k: {"models": {}, "cached": True})
    monkeypatch.setattr(train_lora, "_copy_preflight_to_run", lambda *a: None)
    monkeypatch.setattr(train_lora.evaluate, "load", lambda *a: object())
    monkeypatch.setattr(train_lora, "WhisperSeq2SeqTrainer", Trainer)
    monkeypatch.setattr(train_lora.torch.cuda, "is_available", lambda: False)
    if save_fails:
        with pytest.raises(OSError, match="processor failed"):
            train_lora._run_training_stages(args, run_dir, None)
    else:
        train_lora._run_training_stages(args, run_dir, None)
    saved = run_state.load_run(run_dir)
    stages = {s["name"]: s["status"] for s in saved["stages"]}
    if save_fails:
        assert saved["status"] == stages["training"] == "failed"
        assert stages["post_eval"] == "pending"
        assert saved["failed_stage"] == "training"
        assert not (Path(args.output_dir) / "best_adapter").exists()
        assert json.loads((run_dir / "live_telemetry.json").read_text())["status"] == "failed"
    else:
        assert saved["status"] == stages["training"] == "completed"
        assert (Path(args.output_dir) / "best_adapter" / "preprocessor_config.json").exists()
