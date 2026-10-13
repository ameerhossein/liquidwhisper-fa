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
from typing import Optional, Dict, Any
from datasets import load_dataset, Audio, DatasetDict, Dataset
from src.normalizer import PersianTextNormalizer


class PersianASRDataPipeline:
    def __init__(
        self,
        processor: Any,
        dataset_name: str = "shekar-ai/Neyshekar",
        normalizer: Optional[PersianTextNormalizer] = None,
        max_duration_seconds: float = 30.0,
        min_duration_seconds: float = 0.5,
        num_proc: int = 4
    ):
        self.processor = processor
        self.dataset_name = dataset_name
        self.normalizer = normalizer or PersianTextNormalizer()
        self.max_duration_seconds = max_duration_seconds
        self.min_duration_seconds = min_duration_seconds
        self.num_proc = num_proc

    def load_raw_dataset(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Any:
        """Loads dataset from Hugging Face Hub, resamples audio to 16kHz."""
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
        audio = batch["audio"]
        raw_text = batch.get("text", "") or ""

        # Normalize text for target label
        clean_text = self.normalizer.normalize(
            raw_text,
            remove_punctuation=False,
            zwnj_mode="keep"
        )

        # 1. Audio feature extraction (128 mel bins for Whisper Large v3 / BuzzASR)
        input_features = self.processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
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
        cols_to_remove = remove_columns or [c for c in dataset.column_names if c not in ["input_features", "labels", "is_informal"]]

        print("Transforming dataset (feature extraction & tokenization)...")
        prepared_ds = dataset.map(
            self.prepare_sample,
            remove_columns=cols_to_remove,
            num_proc=min(self.num_proc, os.cpu_count() or 1),
            desc="Preparing ASR Features"
        )
        return prepared_ds
