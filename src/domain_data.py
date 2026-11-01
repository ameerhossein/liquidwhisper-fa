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
