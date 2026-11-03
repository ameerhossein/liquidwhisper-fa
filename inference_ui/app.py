"""Local voice transcription UI for Hugging Face Whisper artifacts."""
from __future__ import annotations

import io
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.dotenv import load_dotenv
from src.hf_sync import REPO_ID_RE

load_dotenv()
HERE = Path(__file__).resolve().parent
app = FastAPI(title="Voice inference")
app.mount('/static', StaticFiles(directory=HERE / 'static'), name='static')
lock = threading.Lock()
state = {"repo": None, "artifacts": [], "revision": None, "model": None, "processor": None, "selected": None}


class RepoRequest(BaseModel):
    repo: str


class Selection(BaseModel):
    path: str


def discover(files):
    """Find usable adapter and standalone model directories in a Hub file list."""
    folders = {}
    for file in files:
        folder, _, name = file.rpartition('/')
        folders.setdefault(folder, set()).add(name)
    artifacts = []
    for folder, names in folders.items():
        if 'adapter_config.json' in names and names.intersection({'adapter_model.safetensors', 'adapter_model.bin'}):
            kind = 'adapter'
        elif 'config.json' in names and names.intersection({'model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json'}):
            kind = 'merged'
        else:
            continue
        parts = folder.split('/') if folder else []
        label = 'checkpoint' if any(p.startswith('checkpoint-') for p in parts) else ('snapshot' if 'adapter_snapshots' in parts else kind)
        artifacts.append({'path': folder, 'kind': kind, 'label': label})
    return sorted(artifacts, key=lambda a: (a['path'] != '', a['path']), reverse=False)


@app.get('/')
def index():
    return FileResponse(HERE / 'static' / 'index.html')


@app.post('/api/discover')
def find_repo(body: RepoRequest):
    repo = body.repo.strip()
    if not REPO_ID_RE.fullmatch(repo):
        raise HTTPException(400, 'Enter a Hugging Face repository as owner/name.')
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN'))
        info = api.model_info(repo, files_metadata=False)
        artifacts = discover([item.rfilename for item in info.siblings])
    except Exception as exc:
        raise HTTPException(400, f'Could not read repository: {exc}') from exc
    with lock:
        state.update(repo=repo, artifacts=artifacts, revision=info.sha, files=[item.rfilename for item in info.siblings])
    return {'repo': repo, 'artifacts': artifacts}


@app.post('/api/select')
def select(body: Selection):
    with lock:
        artifact = next ( (a for a in state['artifacts'] if a['path'] == body.path), None)
        if artifact is None:
            raise HTTPException(400, 'Choose an artifact from the current repository.')
        repo, revision = state['repo'], state['revision']
        # Release the previous model before loading to avoid doubling GPU memory.
        state.update(model=None, processor=None, selected=None)
        try:
            import gc
            import torch
            from huggingface_hub import snapshot_download
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftConfig, PeftModel
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            prefix = artifact['path']
            # Download only files directly in the artifact folder, excluding Trainer state.
            parent = f'{prefix}/' if prefix else ''
            excluded = ('optimizer', 'scheduler', 'rng_state', 'scaler', 'training_args', 'trainer_state')
            patterns = [f for f in state['files'] if f.startswith(parent) and '/' not in f[len(parent):] and not f[len(parent):].startswith(excluded)]
            root = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns, token=os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')))
            source = str(root / prefix)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else torch.float32
            base = PeftConfig.from_pretrained(source).base_model_name_or_path if artifact['kind'] == 'adapter' else source
            if not base:
                raise ValueError('Adapter configuration does not specify its base model.')
            try:
                processor = WhisperProcessor.from_pretrained(source)
            except (OSError, ValueError):
                processor = WhisperProcessor.from_pretrained(base)
            model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=dtype).to(device)
            if artifact['kind'] == 'adapter':
                model = PeftModel.from_pretrained(model, source).to(device)
            model.eval()
            state.update(model=model, processor=processor, selected=artifact, device=device, dtype=dtype)
        except Exception as exc:
            raise HTTPException(500, f'Could not load model: {exc}') from exc
        return {'artifact': artifact, 'device': device}


@app.post('/api/transcribe')
async def transcribe(request: Request):
    audio = await request.body()
    if not audio or len(audio) > 4_000_000:
        raise HTTPException(400, 'Recording is empty or too large.')
    with lock:
        if state['model'] is None:
            raise HTTPException(400, 'Load a model first.')
        try:
            import librosa
            import numpy as np
            import soundfile as sf
            import torch
            from src.generation import persian_generation_kwargs
            samples, rate = sf.read(io.BytesIO(audio), dtype='float32', always_2d=True)
            if not len(samples) or len(samples) / rate > 30.1:
                raise ValueError('Recording must be between 0 and 30 seconds.')
            waveform = np.mean(samples, axis=1)
            if rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=rate, target_sr=16000)
            features = state['processor'](waveform, sampling_rate=16000, return_tensors='pt').input_features.to(device=state['device'], dtype=state['dtype'])
            with torch.inference_mode():
                ids = state['model'].generate(input_features=features, max_new_tokens=225, **persian_generation_kwargs(state['model']))
            result = state['processor'].batch_decode(ids, skip_special_tokens=True)[0].strip()
            return {'text': result}
        except Exception as exc:
            raise HTTPException(400, f'Transcription failed: {exc}') from exc
