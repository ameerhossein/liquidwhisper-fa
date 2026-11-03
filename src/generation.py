"""Persian transcription settings shared by Whisper inference paths."""

from collections.abc import Mapping
from typing import Any


def persian_generation_kwargs(model: Any) -> dict:
    """Use the model's native Persian token mapping, including for PEFT models.

    Tiny/English-only models without a Persian language map retain their own
    generation settings. Never derive BuzzASR's prompt IDs from its tokenizer:
    its tokenizer and model special-token IDs differ.
    """
    config = getattr(model, "generation_config", None)
    languages = getattr(config, "lang_to_id", None)
    if not isinstance(languages, Mapping) or "<|fa|>" not in languages:
        return {}
    config.language = "fa"
    config.task = "transcribe"
    config.forced_decoder_ids = None
    return {"language": "fa", "task": "transcribe"}
