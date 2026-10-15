# Production Notes: Checkpoints, Adapters, Merged Models, Storage

## 1. The 3 Artifacts

| Artifact | How created | Contents | Use for |
|---|---|---|---|
| `checkpoint-*/` | Auto-saved every `--save_steps 500` during training (`scripts/train_lora.py`). Only last 2 kept (`--save_total_limit 2`). | LoRA weights + optimizer + scheduler + `trainer_state.json` | Crash / pod-restart resume only. Not for inference. |
| `best_adapter/` | Auto-saved once at end: `trainer.model.save_pretrained(best_adapter)` (`scripts/train_lora.py`). Selected by `load_best_model_at_end=True, metric_for_best_model="wer"`. | `adapter_model.safetensors` + `adapter_config.json` + processor. No optimizer. | `evaluate_model.py`, `transcribe.py --adapter_path`, `merge_lora.py --adapter_path`. Requires base `BuzzASR/persian` + `peft`. |
| Merged model | **Manual only**: `scripts/merge_lora.py` (`merge_and_unload()`). Nothing auto-runs it. | Full Whisper weights (`W0 + BA`) + processor. No `peft` needed. | Production inference. |

## 2. Production Use (Merged Model)

```bash
# 1. Merge (required, manual, --output_dir has no default)
python scripts/merge_lora.py \
  --base_model BuzzASR/persian \
  --adapter_path outputs/buzzasr-neyshekar-lora/best_adapter \
  --output_dir models/buzzasr-persian-fine-tuned

# 2a. Single file
python scripts/transcribe.py \
  --audio sample.wav \
  --model_path models/buzzasr-persian-fine-tuned \
  --normalize

# 2b. Batch dir -> JSON
python scripts/transcribe.py \
  --audio_dir ./recordings/ \
  --model_path models/buzzasr-persian-fine-tuned \
  --output_file results/my_transcriptions.json \
  --normalize
```

Direct integration:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
processor = WhisperProcessor.from_pretrained("models/buzzasr-persian-fine-tuned")
model = WhisperForConditionalGeneration.from_pretrained(
    "models/buzzasr-persian-fine-tuned", dtype="auto").to("cuda").eval()
```

Inference details (`scripts/transcribe.py`): 16 kHz mono resample, `max_new_tokens=225`, CUDA = BF16/FP16 autocast, CPU = FP32, `--normalize` applies `PersianTextNormalizer` (Hazm/ZWNJ).

## 3. Persistent vs Ephemeral (RunPod)

Only `/workspace` survives pod restarts. Container disk (`~/stt`, `/tmp`, `~/.cache`) does not.

Persistent by default (when `/workspace` exists):
- `ASR_OUTPUT_DIR` default: `/workspace/stt-outputs/buzzasr-neyshekar-lora` (`scripts/train_lora.py`, `run_train.sh`, `configs/l40s_train_config.yaml`, `server/app.py`). Holds `checkpoint-*/`, `best_adapter/`, `final_evaluation_report.json/md`.
- `ASR_RESULTS_DIR` default: `/workspace/stt-outputs/results`. Holds `live_telemetry.json`, `loss_history.json`, `epoch_benchmarks.json`, `training_live.log`, `training.pid`.

Ephemeral (re-downloaded / lost on restart):
- `~/stt/.venv/`, repo-local `outputs/`, `results/`, `models/`
- `~/.cache/huggingface/` (base model + Neyshekar dataset; repo sets no `HF_HOME`)
- Any merge output under `models/...` unless you point it at `/workspace`.

Persistent merge:

```bash
python scripts/merge_lora.py \
  --base_model BuzzASR/persian \
  --adapter_path /workspace/stt-outputs/buzzasr-neyshekar-lora/best_adapter \
  --output_dir /workspace/stt-outputs/buzzasr-persian-merged
```

Check: `echo $ASR_OUTPUT_DIR; ls -d /workspace/stt-outputs/buzzasr-neyshekar-lora/checkpoint-*`

## 4. Continuing Training Later

- Have `checkpoint-*/`? Resume (preserves optimizer/scheduler):
  ```bash
  ./run_train.sh  # auto-resumes latest checkpoint in --output_dir
  ```
- Only have `best_adapter`/merged? Start round 2 with merged as new base (use lower LR to avoid forgetting). Do not stack adapter-on-adapter:
  ```bash
  python scripts/train_lora.py \
    --base_model /workspace/stt-outputs/buzzasr-persian-merged \
    --output_dir outputs/buzzasr-round2-lora \
    --learning_rate 5e-5
  ```
