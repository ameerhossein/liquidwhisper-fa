"""Weights-only forks start fresh and leave their parent artifacts untouched."""

import json
from pathlib import Path

import pytest

from src.training_fork import validate_fork_source


def _source(tmp_path):
    run = tmp_path / "parent"
    source = run / "output" / "checkpoint-9000"
    source.mkdir(parents=True)
    (run / "run.json").write_text('{}')
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    return source


@pytest.mark.parametrize("destination", ["run", "output", "source", "child", "ancestor", "symlink"])
def test_fork_rejects_overlapping_destinations(tmp_path, destination):
    source = _source(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(source.parent.parent, target_is_directory=True)
    destinations = {
        "run": source.parent.parent,
        "output": source.parent,
        "source": source,
        "child": source / "new-output",
        "ancestor": tmp_path,
        "symlink": link / "new-output",
    }
    with pytest.raises(ValueError, match="separate"):
        validate_fork_source(source, [destinations[destination]])


def test_fork_validates_weights_and_can_resume_without_parent(tmp_path):
    source = _source(tmp_path)
    assert validate_fork_source(source, [tmp_path / "fork"]) == str(source)
    (source / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing adapter weights"):
        validate_fork_source(source, [tmp_path / "fork"])
    assert validate_fork_source(source, [tmp_path / "fork"], require_weights=False) == str(source)


def test_fork_loads_actual_weights_with_fresh_optimizer_and_steps(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from scripts import train_lora

    config = transformers.WhisperConfig(
        vocab_size=32, num_mel_bins=4, d_model=8,
        encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=8, max_target_positions=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
    )
    parent = peft.get_peft_model(
        transformers.WhisperForConditionalGeneration(config),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    with torch.no_grad():
        for parameter in parent.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.125)
    source = tmp_path / "parent" / "checkpoint-9000"
    parent.save_pretrained(source)
    # Reading any of these parent state files would fail the test. A fork
    # must import only adapter weights/config, regardless of the old step.
    for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
        (source / name).write_bytes(b"parent state must not be loaded")
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = train_lora.build_parser().parse_args([
        "--init_from_checkpoint", str(source), "--output_dir", str(tmp_path / "fork"),
        "--epochs", "1", "--learning_rate", "1e-5",
        "--lr_scheduler_type", "constant", "--warmup_steps", "0",
    ])
    resume = train_lora._resolve_resume_checkpoint(args)
    assert resume is None
    model = train_lora._configure_lora_model(
        transformers.WhisperForConditionalGeneration(config), args, resume
    )
    assert model.peft_config["default"].r == 2  # saved rank, not CLI default 64
    assert model.peft_config["default"].lora_alpha == 4
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert torch.all(parameter == 0.125)
    trainer = train_lora.WhisperSeq2SeqTrainer(
        model=model,
        args=transformers.Seq2SeqTrainingArguments(
            output_dir=args.output_dir, use_cpu=True, max_steps=1,
            learning_rate=args.learning_rate, lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps, report_to=[], save_strategy="no",
        ),
        train_dataset=[{"input_features": torch.zeros(4, 16), "labels": torch.tensor([1, 3, 2])}],
    )
    assert trainer.state.global_step == 0
    trainer.create_optimizer_and_scheduler(1)
    assert not trainer.optimizer.state
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-5
    trainer.train(resume_from_checkpoint=resume)
    assert trainer.state.global_step == 1
    assert trainer.state.epoch == 1
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}

    # An interrupted fork loads its own adapter even after the parent is gone.
    own = Path(args.output_dir) / "checkpoint-1"
    model.save_pretrained(own)
    (own / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    args.init_from_checkpoint = str(tmp_path / "missing-parent")
    own_resume = train_lora._resolve_resume_checkpoint(args)
    assert own_resume == str(own)
    restored = train_lora._configure_lora_model(
        transformers.WhisperForConditionalGeneration(config), args, own_resume
    )
    assert restored.peft_config["default"].r == 2


@pytest.mark.parametrize("extra", [
    ["--resume_from_checkpoint", "last"], ["--hf_resume", "latest"], ["--fresh_start"],
])
def test_fork_cli_rejects_conflicting_modes(monkeypatch, extra):
    from scripts import train_lora
    import sys

    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--init_from_checkpoint", "/source", *extra])
    with pytest.raises(SystemExit):
        train_lora.parse_args()
