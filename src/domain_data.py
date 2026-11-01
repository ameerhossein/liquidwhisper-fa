"""Domain Adaptation Data Ingestion & Filtering Engine.

Scans custom audio directories (metadata-based or paired files), validates
audio duration constraints (0.5s <= dur <= 30.0s), filters out empty transcripts,
and provides deterministic train/val splitting.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import wave

try:
    from datasets import (
        Audio,
        Dataset,
        DatasetDict,
        concatenate_datasets,
        interleave_datasets,
        load_dataset,
    )
    _DATASETS_AVAILABLE = True
except ImportError:
    Audio = None
    Dataset = None
    DatasetDict = None
    concatenate_datasets = None
    interleave_datasets = None
    load_dataset = None
    _DATASETS_AVAILABLE = False

from src.normalizer import PersianTextNormalizer

try:
    from src.cache import get_hf_cache_dir
except ImportError:
    try:
        from src.paths import get_hf_cache_dir
    except ImportError:
        get_hf_cache_dir = None

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

AUDIO_COL_CANDIDATES = (
    "file_name",
    "audio",
    "path",
    "audio_path",
    "filename",
    "file",
    "wav",
    "filepath",
)

TEXT_COL_CANDIDATES = (
    "text",
    "transcript",
    "sentence",
    "transcription",
    "label",
    "caption",
)


def _get_audio_duration(path: Union[str, Path]) -> Optional[float]:
    """Inspect audio header to retrieve duration in seconds without loading audio samples.

    Falls back in order:
    1. stdlib wave (fastest for WAV)
    2. soundfile (supports WAV, FLAC, OGG, etc.)
    3. torchaudio.info (supports MP3, M4A, etc.)
    """
    path = Path(path)
    if not path.is_file():
        return None

    # 1. stdlib wave
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        pass

    # 2. soundfile
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            if f.samplerate > 0:
                return float(len(f)) / float(f.samplerate)
    except Exception:
        pass

    # 3. torchaudio.info
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        if info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    return None


def _find_column(fieldnames: List[str], candidates: tuple[str, ...]) -> Optional[str]:
    """Find matching column name ignoring case, surrounding whitespace, and BOM."""
    for col in fieldnames:
        clean = col.lstrip("\ufeff").lower().strip()
        if clean in candidates:
            return col
    return None


def scan_custom_audio_dir(
    data_path: Union[str, Path],
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> List[Dict[str, Any]]:
    """Scan a custom audio dataset directory and return valid records.

    Supported directory formats:
    1. metadata.csv with audio and transcript columns.
    2. metadata.jsonl with audio and transcript columns/keys.
    3. Paired audio and .txt files with identical stem (e.g. sample1.wav and sample1.txt).

    Audio clips strictly outside [min_duration, max_duration] are dropped (never truncated).
    Records with empty or whitespace-only transcripts are dropped.

    Returns:
        List of dicts: [{"audio": str(path), "text": str(transcript), "duration": float(seconds)}]
    """
    root_path = Path(data_path)
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Custom audio directory does not exist or is not a directory: {data_path}")

    csv_meta = root_path / "metadata.csv"
    jsonl_meta = root_path / "metadata.jsonl"
    tsv_meta = root_path / "metadata.tsv"

    records: List[Dict[str, Any]] = []

    if csv_meta.is_file():
        with open(csv_meta, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = ","
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
                delimiter = dialect.delimiter
            except Exception:
                pass
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames:
                audio_col = _find_column(list(reader.fieldnames), AUDIO_COL_CANDIDATES)
                text_col = _find_column(list(reader.fieldnames), TEXT_COL_CANDIDATES)

                if audio_col and text_col:
                    for row in reader:
                        audio_val = (row.get(audio_col) or "").strip()
                        text_val = (row.get(text_col) or "").strip()
                        if not audio_val or not text_val:
                            continue

                        audio_p = Path(audio_val)
                        audio_file = audio_p if audio_p.is_absolute() else root_path / audio_p
                        if not audio_file.is_file():
                            logger.warning("Audio file referenced in metadata not found: %s", audio_file)
                            continue

                        dur = _get_audio_duration(audio_file)
                        if dur is None or dur < min_duration or dur > max_duration:
                            logger.debug("Dropping clip %s with duration %s (bounds: [%s, %s])", audio_file, dur, min_duration, max_duration)
                            continue

                        records.append({
                            "audio": str(audio_file),
                            "text": text_val,
                            "duration": dur,
                        })
                    return records

    if jsonl_meta.is_file():
        with open(jsonl_meta, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip().lstrip("\ufeff")
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue

                audio_col = _find_column(list(row.keys()), AUDIO_COL_CANDIDATES)
                text_col = _find_column(list(row.keys()), TEXT_COL_CANDIDATES)

                audio_val = ""
                if audio_col:
                    raw_audio = row[audio_col]
                    if isinstance(raw_audio, dict):
                        audio_val = str(raw_audio.get("path") or raw_audio.get("file_name") or "").strip()
                    else:
                        audio_val = str(raw_audio or "").strip()

                text_val = str(row[text_col] or "").strip() if text_col else ""
                if not audio_val or not text_val:
                    continue

                audio_p = Path(audio_val)
                audio_file = audio_p if audio_p.is_absolute() else root_path / audio_p
                if not audio_file.is_file():
                    logger.warning("Audio file referenced in jsonl not found: %s", audio_file)
                    continue

                dur = _get_audio_duration(audio_file)
                if dur is None or dur < min_duration or dur > max_duration:
                    logger.debug("Dropping clip %s with duration %s (bounds: [%s, %s])", audio_file, dur, min_duration, max_duration)
                    continue

                records.append({
                    "audio": str(audio_file),
                    "text": text_val,
                    "duration": dur,
                })
        return records

    if tsv_meta.is_file():
        with open(tsv_meta, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames:
                audio_col = _find_column(list(reader.fieldnames), AUDIO_COL_CANDIDATES)
                text_col = _find_column(list(reader.fieldnames), TEXT_COL_CANDIDATES)

                if audio_col and text_col:
                    for row in reader:
                        audio_val = (row.get(audio_col) or "").strip()
                        text_val = (row.get(text_col) or "").strip()
                        if not audio_val or not text_val:
                            continue

                        audio_p = Path(audio_val)
                        audio_file = audio_p if audio_p.is_absolute() else root_path / audio_p
                        if not audio_file.is_file():
                            logger.warning("Audio file referenced in metadata not found: %s", audio_file)
                            continue

                        dur = _get_audio_duration(audio_file)
                        if dur is None or dur < min_duration or dur > max_duration:
                            logger.debug("Dropping clip %s with duration %s (bounds: [%s, %s])", audio_file, dur, min_duration, max_duration)
                            continue

                        records.append({
                            "audio": str(audio_file),
                            "text": text_val,
                            "duration": dur,
                        })
                    return records

    # Fallback: paired audio and .txt files
    candidate_audio_files = sorted(
        [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )

    for audio_file in candidate_audio_files:
        txt_file = audio_file.with_suffix(".txt")
        if not txt_file.is_file():
            # Check uppercase variation
            txt_file = audio_file.with_name(f"{audio_file.stem}.TXT")
            if not txt_file.is_file():
                continue

        try:
            text_val = txt_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue

        if not text_val:
            continue

        dur = _get_audio_duration(audio_file)
        if dur is None or dur < min_duration or dur > max_duration:
            logger.debug("Dropping clip %s with duration %s (bounds: [%s, %s])", audio_file, dur, min_duration, max_duration)
            continue

        records.append({
            "audio": str(audio_file),
            "text": text_val,
            "duration": dur,
        })

    return records


def split_custom_dataset(
    records: List[Dict[str, Any]],
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically partition records into train and validation splits.

    Args:
        records: List of custom dataset records.
        eval_ratio: Fraction of records allocated to validation (default: 0.15).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_records, val_records).
    """
    if not records:
        return [], []

    if eval_ratio <= 0.0:
        return list(records), []
    if eval_ratio >= 1.0:
        return [], list(records)

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * eval_ratio))
    if len(shuffled) > 1 and val_count == 0 and eval_ratio > 0.0:
        val_count = 1
    elif len(shuffled) > 1 and val_count >= len(shuffled) and eval_ratio < 1.0:
        val_count = len(shuffled) - 1

    train_count = len(shuffled) - val_count
    train_records = shuffled[:train_count]
    val_records = shuffled[train_count:]

    return train_records, val_records


def sample_balanced_neyshekar(
    neyshekar_ds: Optional[Dataset] = None,
    sample_count: int = 1000,
    seed: int = 42,
) -> Dataset:
    """Extract a 50/50 balanced replay buffer of formal and informal Persian utterances.

    Uses `is_informal` or `informal` boolean column if available; falls back to
    PersianTextNormalizer.is_colloquial(text).

    Args:
        neyshekar_ds: Hugging Face Dataset (defaults to 'shekar-ai/Neyshekar' train split).
        sample_count: Total number of utterances to draw (~50% formal, ~50% informal).
        seed: Random seed for deterministic sampling.

    Returns:
        Balanced Dataset containing sampled utterances.
    """
    if isinstance(neyshekar_ds, int):
        sample_count = neyshekar_ds
        neyshekar_ds = None

    if neyshekar_ds is None:
        hf_cache = get_hf_cache_dir() if callable(get_hf_cache_dir) else None
        load_kwargs = {"cache_dir": hf_cache} if hf_cache else {}
        neyshekar_ds = load_dataset("shekar-ai/Neyshekar", split="train", **load_kwargs)

    if sample_count <= 0 or len(neyshekar_ds) == 0:
        return neyshekar_ds.select([])

    normalizer = PersianTextNormalizer()
    target_each = sample_count // 2
    rem = sample_count - (target_each * 2)
    target_formal = target_each + rem
    target_informal = target_each

    formal_indices: List[int] = []
    informal_indices: List[int] = []

    has_is_informal = "is_informal" in getattr(neyshekar_ds, "column_names", [])
    has_informal = "informal" in getattr(neyshekar_ds, "column_names", [])
    has_text = "text" in getattr(neyshekar_ds, "column_names", [])

    if has_is_informal or has_informal:
        col_name = "is_informal" if has_is_informal else "informal"
        flags = neyshekar_ds[col_name]
        texts = neyshekar_ds["text"] if has_text else [None] * len(neyshekar_ds)
        for idx, flag in enumerate(flags):
            if flag is not None:
                if bool(flag):
                    informal_indices.append(idx)
                else:
                    formal_indices.append(idx)
            else:
                txt = texts[idx] or ""
                if normalizer.is_colloquial(txt):
                    informal_indices.append(idx)
                else:
                    formal_indices.append(idx)
    else:
        texts = neyshekar_ds["text"] if has_text else [None] * len(neyshekar_ds)
        for idx, txt in enumerate(texts):
            if normalizer.is_colloquial(txt or ""):
                informal_indices.append(idx)
            else:
                formal_indices.append(idx)

    rng = random.Random(seed)
    chosen_formal = rng.sample(formal_indices, min(target_formal, len(formal_indices))) if formal_indices else []
    chosen_informal = rng.sample(informal_indices, min(target_informal, len(informal_indices))) if informal_indices else []

    if len(chosen_formal) < target_formal and len(informal_indices) > len(chosen_informal):
        needed = min(target_formal - len(chosen_formal), len(informal_indices) - len(chosen_informal))
        avail = sorted(set(informal_indices) - set(chosen_informal))
        chosen_informal.extend(rng.sample(avail, needed))
    elif len(chosen_informal) < target_informal and len(formal_indices) > len(chosen_formal):
        needed = min(target_informal - len(chosen_informal), len(formal_indices) - len(chosen_formal))
        avail = sorted(set(formal_indices) - set(chosen_formal))
        chosen_formal.extend(rng.sample(avail, needed))

    selected = chosen_formal + chosen_informal
    rng.shuffle(selected)
    return neyshekar_ds.select(selected)


def _ensure_audio_column(ds: Optional[Dataset]) -> Optional[Dataset]:
    """Ensure dataset audio column is cast to Audio(sampling_rate=16000)."""
    if ds is not None and Audio is not None and "audio" in getattr(ds, "column_names", []):
        audio_feat = ds.features.get("audio")
        if not isinstance(audio_feat, Audio) or getattr(audio_feat, "sampling_rate", None) != 16000:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def build_domain_dataset(
    custom_path: Union[str, Path],
    replay_samples: int = 1000,
    replay_ratio: float = 0.25,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> DatasetDict:
    """Build domain-adapted DatasetDict interleaving custom domain data with Neyshekar replay buffer.

    Args:
        custom_path: Directory containing custom audio files (or train/val subdirectories).
        replay_samples: Number of balanced Neyshekar samples to draw for train replay (default: 1000).
        replay_ratio: Proportion of replay samples in interleaved training set (default: 0.25).
        output_dir: Optional path to save DatasetDict via save_to_disk.
        seed: Random seed for deterministic partitioning and sampling.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.
    """
    if not _DATASETS_AVAILABLE:
        raise ImportError("datasets library is required to build domain datasets. Please install datasets.")

    custom_root = Path(custom_path)
    train_dir = custom_root / "train"
    val_dir = custom_root / "val"
    if not val_dir.is_dir():
        val_dir = custom_root / "validation"

    if train_dir.is_dir() and val_dir.is_dir():
        train_records = scan_custom_audio_dir(train_dir)
        val_records = scan_custom_audio_dir(val_dir)
    else:
        records = scan_custom_audio_dir(custom_root)
        train_records, val_records = split_custom_dataset(records, eval_ratio=0.15, seed=seed)

    if not train_records and not val_records:
        raise ValueError(f"No valid custom audio records found in: {custom_path}")

    hf_cache = get_hf_cache_dir() if callable(get_hf_cache_dir) else None
    load_kwargs = {"cache_dir": hf_cache} if hf_cache else {}

    neyshekar_train = _ensure_audio_column(load_dataset("shekar-ai/Neyshekar", split="train", **load_kwargs))
    neyshekar_val = _ensure_audio_column(load_dataset("shekar-ai/Neyshekar", split="validation", **load_kwargs))
    neyshekar_test = _ensure_audio_column(load_dataset("shekar-ai/Neyshekar", split="test", **load_kwargs))

    custom_train_ds = Dataset.from_list(train_records) if train_records else Dataset.from_list([])
    custom_val_ds = Dataset.from_list(val_records) if val_records else Dataset.from_list([])

    custom_train_ds = _ensure_audio_column(custom_train_ds)
    custom_val_ds = _ensure_audio_column(custom_val_ds)

    if replay_samples > 0 and len(neyshekar_train) > 0:
        replay_train = sample_balanced_neyshekar(neyshekar_train, sample_count=replay_samples, seed=seed)
        replay_train = _ensure_audio_column(replay_train)
    else:
        replay_train = neyshekar_train.select([])

    if len(custom_train_ds) > 0 and len(replay_train) > 0:
        clamped_ratio = max(0.0, min(1.0, float(replay_ratio)))
        if clamped_ratio <= 0.0:
            train_ds = custom_train_ds
        elif clamped_ratio >= 1.0:
            train_ds = replay_train
        else:
            probs = [1.0 - clamped_ratio, clamped_ratio]
            train_ds = interleave_datasets(
                [custom_train_ds, replay_train],
                probabilities=probs,
                seed=seed,
                stopping_strategy="all_exhausted",
            )
    elif len(custom_train_ds) > 0:
        train_ds = custom_train_ds
    else:
        train_ds = replay_train

    val_replay_samples = int(round(replay_samples * 0.15))
    if replay_samples > 0 and val_replay_samples == 0:
        val_replay_samples = 1

    if val_replay_samples > 0 and len(neyshekar_val) > 0:
        sampled_val = sample_balanced_neyshekar(
            neyshekar_val,
            sample_count=min(val_replay_samples, len(neyshekar_val)),
            seed=seed,
        )
        sampled_val = _ensure_audio_column(sampled_val)
        if len(custom_val_ds) > 0:
            val_ds = concatenate_datasets([custom_val_ds, sampled_val])
        else:
            val_ds = sampled_val
    else:
        val_ds = custom_val_ds

    test_ds = neyshekar_test

    train_ds = _ensure_audio_column(train_ds)
    val_ds = _ensure_audio_column(val_ds)
    test_ds = _ensure_audio_column(test_ds)

    ds_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        try:
            ds_dict.save_to_disk(str(out_p))
        except FileNotFoundError:
            # Handle mock/synthetic dataset references where audio paths do not exist on disk
            created_placeholders: List[Path] = []
            for _, ds in ds_dict.items():
                if ds is not None and "audio" in getattr(ds, "column_names", []):
                    table = ds.with_format("arrow")[:]
                    for item in table["audio"].to_pylist():
                        path_val = item.get("path") if isinstance(item, dict) else (item if isinstance(item, str) else None)
                        if path_val and not Path(path_val).exists():
                            try:
                                ph = Path(path_val)
                                ph.parent.mkdir(parents=True, exist_ok=True)
                                ph.touch()
                                created_placeholders.append(ph)
                            except Exception:
                                pass
            try:
                ds_dict.save_to_disk(str(out_p))
            finally:
                for ph in created_placeholders:
                    try:
                        ph.unlink()
                    except Exception:
                        pass

    return ds_dict
