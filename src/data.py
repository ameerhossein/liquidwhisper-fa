"""
Dataset loading and feature preprocessing for Persian ASR.
Supports:
- shekar-ai/Neyshekar and custom Hugging Face / local datasets
- Audio resampling to 16kHz
- Feature extraction using BuzzASR / Whisper feature extractor
- Tokenization using BuzzASR / Whisper tokenizer
- Streaming and subset slicing for quick testing
"""

import os
import random
from typing import Optional, Dict, Any, Sequence
try:
    from datasets import load_dataset, Audio, DatasetDict, Dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    load_dataset = None  # type: ignore
    Audio = None  # type: ignore
    DatasetDict = dict  # type: ignore
    Dataset = object  # type: ignore
    _DATASETS_AVAILABLE = False
from src.normalizer import PersianTextNormalizer

try:
    from src.telephone import apply_random_telephone
    _TELEPHONE_AVAILABLE = True
except ImportError:  # pragma: no cover
    apply_random_telephone = None  # type: ignore
    _TELEPHONE_AVAILABLE = False


def _datasets_version_tuple() -> Optional[tuple]:
    """Return datasets version as (major, minor, ...) or None if unavailable."""
    try:
        import datasets
        ver = getattr(datasets, "__version__", "") or ""
        parts = []
        for tok in ver.split("."):
            num = "".join(c for c in tok if c.isdigit())
            if num == "":
                break
            parts.append(int(num))
        return tuple(parts) if parts else None
    except Exception:
        return None


def _uses_torchcodec_backend() -> bool:
    """True when the installed datasets decodes audio via torchcodec.

    datasets 4.x decodes the ``audio`` column via
    ``torchcodec.decoders.AudioDecoder``; datasets 2.x/3.x uses the
    soundfile/librosa backend and does not need torchcodec at all.
    """
    v = _datasets_version_tuple()
    if v is None:
        return False
    try:
        return v[0] >= 4
    except Exception:
        return False


def _get_audio_array_and_sr(audio: Any) -> tuple:
    """Extract (array, sampling_rate) from dict- or torchcodec-style audio.

    Handles:
    - dicts: ``{"array": np.ndarray, "sampling_rate": int}`` (datasets 2/3).
    - datasets 4.x ``AudioDecoder`` objects supporting ``audio["array"]`` /
      ``audio["sampling_rate"]`` but no ``.get()``.
    """
    if audio is None:
        return None, 16000
    # Dict path (datasets 2/3, and datasets 4 with decode=False).
    if isinstance(audio, dict):
        try:
            arr = audio.get("array", None)
        except Exception:
            arr = None
        try:
            sr = audio.get("sampling_rate", 16000) or 16000
        except Exception:
            sr = 16000
        if arr is None:
            # Maybe torchcodec-style dict access on a non-dict? fall through.
            pass
        else:
            return arr, int(sr)
        # Fall back to __getitem__ for dict-like wrappers.
        try:
            arr = audio["array"]
            sr = audio["sampling_rate"]
            return arr, int(sr or 16000)
        except Exception:
            return None, 16000
    # Object path: datasets 4.x AudioDecoder (has __getitem__, no .get).
    try:
        arr = audio["array"]  # type: ignore[index]
    except Exception:
        arr = None
    try:
        sr = audio["sampling_rate"]  # type: ignore[index]
    except Exception:
        sr = 16000
    # Last resort: raw torchcodec AudioDecoder API (get_all_samples()).
    if arr is None and hasattr(audio, "get_all_samples"):
        try:
            import numpy as _np
            samples = audio.get_all_samples()
            data = samples.data.cpu().numpy() if hasattr(samples, "data") else None
            if data is not None:
                arr = _np.mean(data, axis=tuple(range(data.ndim - 1))) if data.ndim > 1 else data
            sr = getattr(samples, "sample_rate", sr) or sr
        except Exception:
            pass
    try:
        sr = int(sr or 16000)
    except Exception:
        sr = 16000
    return arr, sr


def _check_audio_backend() -> None:
    """Fail fast with an actionable message if audio decoding is unavailable.

    Root cause of the reported crash: datasets 4.x decodes the ``audio``
    column via ``torchcodec.decoders.AudioDecoder``, but the pinned
    ``torchcodec==0.2.1`` (cu124 / torch 2.6) only ships ``VideoDecoder``:

        ImportError: cannot import name 'AudioDecoder' from
        'torchcodec.decoders' (... Did you mean: 'VideoDecoder'?)

    ``AudioDecoder`` was added in later torchcodec releases (0.3+ needs
    torch 2.7+/cu126+), so torchcodec 0.2.1 can never satisfy datasets>=4.
    This repo pins ``datasets<4`` (3.x soundfile backend) to keep the
    cu124/torch-2.6 stack. Only datasets>=4 needs torchcodec.
    """
    if _uses_torchcodec_backend():
        try:
            from torchcodec.decoders import AudioDecoder  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Incompatible audio stack: installed 'datasets>=4' decodes audio "
                "via 'torchcodec.decoders.AudioDecoder', but the installed "
                "torchcodec does not provide it (e.g. torchcodec==0.2.1 only has "
                "VideoDecoder; AudioDecoder needs torchcodec>=0.3 / torch>=2.7). "
                "Pick ONE fix: (1) downgrade datasets to the pinned 3.x backend: "
                "pip install 'datasets>=2.19,<4'  (keeps torch 2.6/cu124), OR "
                "(2) upgrade torch+torchcodec together, e.g. "
                "pip install -U torch torchaudio torchcodec "
                "--index-url https://download.pytorch.org/whl/cu128 "
                "(needs CUDA 12.8 driver + torch>=2.8). Do NOT upgrade torchcodec "
                "alone — it must match torch (else OSError: libnvrtc.so.13)."
            ) from e
        return
    # datasets 2/3 path: soundfile backend; torchcodec is optional.
    try:
        import soundfile  # noqa: F401
        return
    except ImportError as e:
        raise ImportError(
            "Missing audio backend: datasets<4 decodes the 'audio' column via "
            "'soundfile' (librosa). Install it with: pip install soundfile librosa "
            "(or pip install -U 'datasets[audio]'). Then retry training."
        ) from e


class PersianASRDataPipeline:
    def __init__(
        self,
        processor: Any,
        dataset_name: str = "shekar-ai/Neyshekar",
        normalizer: Optional[PersianTextNormalizer] = None,
        max_duration_seconds: float = 30.0,
        min_duration_seconds: float = 0.5,
        num_proc: int = 4,
        telephony_prob: float = 0.0,
        telephony_profiles: Optional[Sequence[str]] = None,
        telephony_seed: Optional[int] = None,
        telephony_use_ffmpeg: bool = True,
    ):
        self.processor = processor
        self.dataset_name = dataset_name
        self.normalizer = normalizer or PersianTextNormalizer()
        self.max_duration_seconds = max_duration_seconds
        self.min_duration_seconds = min_duration_seconds
        self.num_proc = num_proc
        # Telephony augmentation (Iran MCI/Irancell mobile + TCI PSTN).
        # prob=0 keeps legacy clean behavior; train typically uses 0.5.
        self.telephony_prob = float(telephony_prob or 0.0)
        self.telephony_profiles = tuple(telephony_profiles) if telephony_profiles else ("mobile", "pstn")
        self.telephony_seed = telephony_seed
        self.telephony_use_ffmpeg = bool(telephony_use_ffmpeg)
        self._tele_rng = random.Random(telephony_seed) if telephony_seed is not None else random.Random()
        self._tele_warned_missing = False

    def _audio_duration_seconds(self, audio: Any) -> float:
        """Compute duration from waveform length. Returns 0.0 if unknown."""
        try:
            arr, sr = _get_audio_array_and_sr(audio)
            sr = sr or 16000
            if arr is None:
                return 0.0
            return len(arr) / float(sr) if sr else 0.0
        except Exception:
            return 0.0

    def is_valid_duration(self, audio: Any) -> bool:
        """Check min <= duration <= max. Overlong clips are dropped (not truncated)
        to avoid audio/text misalignment: truncating audio while keeping the full
        transcript teaches the model to hallucinate cut-off words."""
        dur = self._audio_duration_seconds(audio)
        return self.min_duration_seconds <= dur <= self.max_duration_seconds

    def load_raw_dataset(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Any:
        """Loads dataset from Hugging Face Hub, resamples audio to 16kHz."""
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets package not installed; cannot load HF dataset.")
        _check_audio_backend()
        print(f"Loading dataset '{self.dataset_name}' (split={split})...")
        try:
            ds = load_dataset(self.dataset_name, split=split)
        except Exception as e:
            print(f"Error loading {self.dataset_name}: {e}. Retrying with trust_remote_code=True...")
            ds = load_dataset(self.dataset_name, split=split, trust_remote_code=True)

        if max_samples and max_samples > 0:
            if isinstance(ds, (DatasetDict, dict)):
                for s in ds:
                    ds[s] = ds[s].select(range(min(len(ds[s]), max_samples)))
            else:
                ds = ds.select(range(min(len(ds), max_samples)))

        # Ensure audio column is cast to 16,000 Hz
        print("Resampling audio to 16,000 Hz...")
        if isinstance(ds, (DatasetDict, dict)):
            for s in ds:
                ds[s] = ds[s].cast_column("audio", Audio(sampling_rate=16000))
        else:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        return ds

    def prepare_sample(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Processes audio waveform into log-mel spectrogram and text into token IDs."""
        audio_raw = batch["audio"]
        array, sr = _get_audio_array_and_sr(audio_raw)
        if array is None:
            raise ValueError("Missing audio array; check dataset 'audio' decoding.")
        try:
            sr = int(sr or 16000)
        except Exception:
            sr = 16000
        raw_text = batch.get("text", "") or ""

        # NOTE: Training keeps punctuation (remove_punctuation=False) to preserve
        # natural transcripts for generation. Evaluation in src/metrics.py strips
        # punctuation (remove_punctuation=True) for standardized WER. This is
        # intentional; do not "fix" by unifying without retraining.
        clean_text = self.normalizer.normalize(
            raw_text,
            remove_punctuation=False,
            zwnj_mode="keep"
        )

        # Defensive guard: overlong clips must have been filtered upstream.
        # Do NOT truncate here — that would misalign audio/text.
        try:
            dur = len(array) / float(sr) if sr else 0.0
        except Exception:
            dur = 0.0
        if dur > self.max_duration_seconds:
            raise ValueError(
                f"Audio {dur:.2f}s exceeds max {self.max_duration_seconds}s; "
                "filter with is_valid_duration / prepare_dataset_split instead of truncating."
            )

        # 0. Optional telephony augmentation (on-the-fly, length-preserving).
        # Degrades 16k -> phone 8k -> back to 16k so Whisper still sees 16k
        # input but with narrowband/codec artifacts (MCI/Irancell/TCI).
        if self.telephony_prob > 0 and self.telephony_profiles:
            if not _TELEPHONE_AVAILABLE:
                if not self._tele_warned_missing:
                    print("Warning: src.telephone not available; skipping phone simulation.")
                    self._tele_warned_missing = True
            else:
                try:
                    degraded, _ = apply_random_telephone(
                        array,
                        sr=sr,
                        prob=self.telephony_prob,
                        profiles=self.telephony_profiles,
                        use_ffmpeg=self.telephony_use_ffmpeg,
                        rng=self._tele_rng,
                    )
                    array = degraded
                except Exception as e:
                    print(f"Warning: telephone augmentation failed ({e}); using clean audio.")

        # 1. Audio feature extraction (128 mel bins for Whisper Large v3 / BuzzASR)
        input_features = self.processor.feature_extractor(
            array,
            sampling_rate=sr
        ).input_features[0]

        # 2. Tokenize target transcript
        labels = self.processor.tokenizer(clean_text).input_ids

        # 3. Detect if sample is informal
        if "informal" in batch:
            is_informal = bool(batch["informal"])
        elif "is_informal" in batch:
            is_informal = bool(batch["is_informal"])
        else:
            is_informal = self.normalizer.is_colloquial(clean_text)

        batch["input_features"] = input_features
        batch["labels"] = labels
        batch["is_informal"] = is_informal
        return batch

    def prepare_dataset_split(
        self,
        dataset: Dataset,
        remove_columns: Optional[list] = None
    ) -> Dataset:
        """Applies mapping over dataset to produce model-ready inputs."""
        _check_audio_backend()
        # Drop out-of-range clips (too short OR over 30s). Overlong are dropped,
        # not truncated, to keep audio/text aligned (Whisper caps at 30s).
        orig_len = len(dataset)
        try:
            dataset = dataset.filter(
                lambda x: self.is_valid_duration(x["audio"]),
                desc="Filtering by duration",
            )
            dropped = orig_len - len(dataset)
            if dropped:
                print(
                    f"Filtered {dropped}/{orig_len} samples outside "
                    f"[{self.min_duration_seconds}s, {self.max_duration_seconds}s]."
                )
        except Exception as e:
            print(f"Duration filtering skipped ({e}). Continuing without filter.")

        cols_to_remove = remove_columns or [c for c in dataset.column_names if c not in ["input_features", "labels", "is_informal"]]

        print("Transforming dataset (feature extraction & tokenization)...")
        # num_proc=0 (or 1) must run single-process; datasets.map with
        # num_proc>1 pickles prepare_sample (bound method) which can fail.
        effective_proc = 1
        try:
            if self.num_proc and self.num_proc > 1:
                effective_proc = min(self.num_proc, os.cpu_count() or 1)
        except Exception:
            effective_proc = 1

        map_kwargs: Dict[str, Any] = {
            "remove_columns": cols_to_remove,
            "desc": "Preparing ASR Features",
        }
        if effective_proc > 1:
            map_kwargs["num_proc"] = effective_proc

        try:
            prepared_ds = dataset.map(self.prepare_sample, **map_kwargs)
        except Exception as e:
            if effective_proc > 1:
                print(f"Multiprocess map failed ({e}). Retrying single-process...")
                map_kwargs.pop("num_proc", None)
                prepared_ds = dataset.map(self.prepare_sample, **map_kwargs)
            else:
                raise
        return prepared_ds
