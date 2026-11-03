"""Read-only validation for initializing a new run from a local adapter."""

import json
from pathlib import Path

from src.hf_sync import validate_adapter


def validate_fork_source(source, destinations, *, require_weights=True):
    """Resolve the source and reject writes overlapping its run or output tree.

    Once a fork has its own checkpoint, its parent weights need not remain on
    disk. Path isolation still applies when resuming that fork.
    """
    source = Path(source).expanduser().resolve()
    protected = source
    for parent in source.parents:
        if (parent / "run.json").is_file():
            protected = parent
            break
    for destination in destinations:
        if destination is None:
            continue
        destination = Path(destination).expanduser().resolve()
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError("Fork destinations must be separate from the source adapter/run.")
    if require_weights:
        valid, reason = validate_adapter(source)
        if not valid:
            raise ValueError(f"Invalid init_from_checkpoint: {reason}")
        try:
            config = json.loads((source / "adapter_config.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Invalid init_from_checkpoint adapter_config.json.") from exc
        if not isinstance(config, dict) or config.get("peft_type") != "LORA":
            raise ValueError("init_from_checkpoint requires a LoRA adapter.")
    return str(source)
