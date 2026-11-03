"""Persian prompting uses model-native IDs for both Whisper and BuzzASR."""

from types import SimpleNamespace

import pytest

from src.generation import persian_generation_kwargs


@pytest.mark.parametrize("fa_id", [50300, 51908])
def test_persian_settings_preserve_model_native_tokens(fa_id):
    config = SimpleNamespace(lang_to_id={"<|fa|>": fa_id}, task_to_id={"transcribe": 51968},
                             eos_token_id=51865, decoder_start_token_id=51866,
                             forced_decoder_ids=[[1, None]])
    assert persian_generation_kwargs(SimpleNamespace(generation_config=config)) == {
        "language": "fa", "task": "transcribe",
    }
    assert config.language == "fa"
    assert config.task == "transcribe"
    assert config.forced_decoder_ids is None
    assert config.lang_to_id["<|fa|>"] == fa_id
    assert (config.eos_token_id, config.decoder_start_token_id) == (51865, 51866)


def test_models_without_persian_mapping_keep_their_configuration():
    config = SimpleNamespace(forced_decoder_ids=[[1, 4]])
    assert persian_generation_kwargs(SimpleNamespace(generation_config=config)) == {}
    assert config.forced_decoder_ids == [[1, 4]]


@pytest.mark.parametrize("peft", [False, True])
def test_real_whisper_generation_uses_persian_without_language_detection(peft, monkeypatch):
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=16, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=16, max_target_positions=16,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        suppress_tokens=[], begin_suppress_tokens=[],
    )
    model = WhisperForConditionalGeneration(config).eval()
    model.generation_config.lang_to_id = {"<|fa|>": 3, "<|en|>": 4}
    model.generation_config.task_to_id = {"transcribe": 5}
    model.generation_config.no_timestamps_token_id = 6
    model.generation_config.is_multilingual = True
    prompts = []

    def capture_prompt(module, args, kwargs):
        if "decoder_input_ids" in kwargs:
            prompts.append(kwargs["decoder_input_ids"].detach().clone())

    model.register_forward_pre_hook(capture_prompt, with_kwargs=True)

    def fail(*a, **k):
        raise AssertionError("Language detection must be skipped")

    monkeypatch.setattr(model, "detect_language", fail)
    if peft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    with torch.inference_mode():
        model.generate(
            input_features=torch.zeros(1, 8, 32), max_new_tokens=1,
            **persian_generation_kwargs(model),
        )
    assert prompts[0][0, :4].tolist() == [1, 3, 5, 6]
