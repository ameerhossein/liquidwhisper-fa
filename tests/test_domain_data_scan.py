import csv
import json
import wave
import pytest
from pathlib import Path

def _create_mock_wav(path: Path, duration_seconds: float, sample_rate: int = 16000):
    num_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)

def test_scan_csv_metadata_filters_duration(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "audio_csv"
    audio_dir.mkdir()

    # 1 valid (3.0s), 1 too short (0.2s), 1 too long (35.0s)
    _create_mock_wav(audio_dir / "clip1.wav", 3.0)
    _create_mock_wav(audio_dir / "short.wav", 0.2)
    _create_mock_wav(audio_dir / "long.wav", 35.0)

    meta_file = audio_dir / "metadata.csv"
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerow(["clip1.wav", "سلام دنیا"])
        writer.writerow(["short.wav", "کوتاه"])
        writer.writerow(["long.wav", "طولانی"])

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["audio"] == str(audio_dir / "clip1.wav")
    assert records[0]["text"] == "سلام دنیا"
    assert 2.9 <= records[0]["duration"] <= 3.1

def test_scan_tsv_metadata(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "audio_tsv"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "clip1.wav", 3.0)
    _create_mock_wav(audio_dir / "short.wav", 0.2)

    meta_file = audio_dir / "metadata.tsv"
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        f.write("file_name\ttranscript\n")
        f.write("clip1.wav\tمتن تب دار معتبر\n")
        f.write("short.wav\tکوتاه\n")

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["audio"] == str(audio_dir / "clip1.wav")
    assert records[0]["text"] == "متن تب دار معتبر"
    assert 2.9 <= records[0]["duration"] <= 3.1

def test_scan_paired_wav_txt(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "audio_paired"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "sample1.wav", 2.0)
    (audio_dir / "sample1.txt").write_text("متن نمونه اول", encoding="utf-8")

    _create_mock_wav(audio_dir / "sample2.wav", 4.0)
    (audio_dir / "sample2.txt").write_text("متن نمونه دوم", encoding="utf-8")

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 2
    texts = {r["text"] for r in records}
    assert "متن نمونه اول" in texts
    assert "متن نمونه دوم" in texts

def test_split_custom_dataset():
    from src.domain_data import split_custom_dataset

    dummy_records = [{"audio": f"f{i}.wav", "text": f"t{i}"} for i in range(100)]
    train_recs, val_recs = split_custom_dataset(dummy_records, eval_ratio=0.15, seed=42)
    assert len(train_recs) == 85
    assert len(val_recs) == 15

def test_scan_jsonl_metadata(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "audio_jsonl"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "audio1.wav", 2.5)
    _create_mock_wav(audio_dir / "audio2.wav", 0.3)

    jsonl_file = audio_dir / "metadata.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio": "audio1.wav", "transcript": "متن معتبر"}) + "\n")
        f.write(json.dumps({"audio": "audio2.wav", "transcript": "متن خیلی کوتاه"}) + "\n")

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["audio"] == str(audio_dir / "audio1.wav")
    assert records[0]["text"] == "متن معتبر"
    assert 2.4 <= records[0]["duration"] <= 2.6

def test_scan_skips_empty_transcripts(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "audio_empty_txt"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "valid.wav", 2.0)
    (audio_dir / "valid.txt").write_text("دارای متن", encoding="utf-8")

    _create_mock_wav(audio_dir / "empty.wav", 2.0)
    (audio_dir / "empty.txt").write_text("   \n\t  ", encoding="utf-8")

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["audio"] == str(audio_dir / "valid.wav")

def test_scan_missing_dir_raises():
    from src.domain_data import scan_custom_audio_dir

    with pytest.raises(FileNotFoundError):
        scan_custom_audio_dir("/nonexistent/directory/path/here")

def test_split_custom_dataset_deterministic_and_empty():
    from src.domain_data import split_custom_dataset

    # Empty
    tr, va = split_custom_dataset([])
    assert tr == [] and va == []

    # Determinism
    dummy_records = [{"audio": f"f{i}.wav", "text": f"t{i}"} for i in range(50)]
    tr1, va1 = split_custom_dataset(dummy_records, eval_ratio=0.2, seed=123)
    tr2, va2 = split_custom_dataset(dummy_records, eval_ratio=0.2, seed=123)
    assert tr1 == tr2
    assert va1 == va2

    # Boundary eval_ratios
    tr_all, va_none = split_custom_dataset(dummy_records, eval_ratio=0.0)
    assert len(tr_all) == 50 and len(va_none) == 0

    tr_none, va_all = split_custom_dataset(dummy_records, eval_ratio=1.0)
    assert len(tr_none) == 0 and len(va_all) == 50

def test_scan_nested_paired_files(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    sub_dir = tmp_path / "speaker_a" / "session_1"
    sub_dir.mkdir(parents=True)

    _create_mock_wav(sub_dir / "nested.wav", 1.5)
    (sub_dir / "nested.txt").write_text("متن در پوشه تو در تو", encoding="utf-8")

    records = scan_custom_audio_dir(tmp_path)
    assert len(records) == 1
    assert records[0]["text"] == "متن در پوشه تو در تو"
    assert records[0]["audio"] == str(sub_dir / "nested.wav")

def test_scan_custom_duration_bounds(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "custom_bounds"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "clip1.wav", 1.0)
    (audio_dir / "clip1.txt").write_text("یک ثانیه", encoding="utf-8")

    _create_mock_wav(audio_dir / "clip2.wav", 5.0)
    (audio_dir / "clip2.txt").write_text("پنج ثانیه", encoding="utf-8")

    # Filter with min_duration=2.0 and max_duration=6.0
    records = scan_custom_audio_dir(audio_dir, min_duration=2.0, max_duration=6.0)
    assert len(records) == 1
    assert records[0]["text"] == "پنج ثانیه"

def test_scan_missing_referenced_audio(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "missing_ref"
    audio_dir.mkdir()

    meta_file = audio_dir / "metadata.csv"
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerow(["non_existent.wav", "فایل وجود ندارد"])

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 0

def test_get_audio_duration_fallbacks(tmp_path):
    from src.domain_data import _get_audio_duration
    import soundfile as sf
    import numpy as np

    # Non-existent file
    assert _get_audio_duration(tmp_path / "ghost.wav") is None

    # Corrupt file
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a valid audio file")
    assert _get_audio_duration(corrupt) is None

    # Non-WAV file (FLAC) via soundfile fallback
    flac_path = tmp_path / "sample.flac"
    sf.write(str(flac_path), np.zeros((16000,), dtype=np.float32), 16000)
    dur = _get_audio_duration(flac_path)
    assert dur is not None
    assert 0.99 <= dur <= 1.01

def test_scan_csv_with_utf8_bom(tmp_path):
    from src.domain_data import scan_custom_audio_dir

    audio_dir = tmp_path / "bom_dir"
    audio_dir.mkdir()

    _create_mock_wav(audio_dir / "clip_bom.wav", 1.2)
    meta_file = audio_dir / "metadata.csv"
    # Write with utf-8-sig (BOM)
    with open(meta_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerow(["clip_bom.wav", "متن همراه با بی‌او‌ام"])

    records = scan_custom_audio_dir(audio_dir)
    assert len(records) == 1
    assert records[0]["text"] == "متن همراه با بی‌او‌ام"


