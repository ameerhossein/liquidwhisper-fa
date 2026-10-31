import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
from datasets import Dataset

from src import prep_cache


def _mapped_dataset(cache_dir: Path, num_proc=2, n=10):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict({"x": list(range(n))})
    return ds.map(
        _double,
        with_indices=True,
        num_proc=num_proc,
        cache_file_name=str(cache_dir / "data.arrow"),
    )


def _double(row, index):
    return {"y": row["x"] * 2}


def test_publish_and_load_roundtrip(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=2)
    meta = prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert (tmp_path / "train" / "meta.json").is_file()
    assert meta["total_rows"] == 10
    loaded = prep_cache.load_prep_cache(tmp_path / "train", {"mode": "paired"})
    assert loaded is not None
    assert len(loaded) == 10
    assert loaded[0]["y"] == 0


def test_load_missing_meta_is_miss(tmp_path):
    (tmp_path / "train").mkdir()
    assert prep_cache.load_prep_cache(tmp_path / "train", {}) is None


def test_load_wrong_fingerprint_is_miss(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "disjoint"}) is None


def test_load_wrong_shard_size_is_miss(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    shard = next((tmp_path / "train").glob("*.arrow"))
    shard.write_bytes(shard.read_bytes()[:-1])
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "paired"}) is None


def test_publish_removes_unreferenced_files(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    (staging / "stray.tmp").write_text("junk")
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "paired"})
    assert not (tmp_path / "train" / "stray.tmp").exists()


def test_staging_cleanup(tmp_path):
    stale = tmp_path / "train.staging-old"
    stale.mkdir()
    (stale / "junk").write_text("x")
    prep_cache.cleanup_stale_staging(tmp_path / "train")
    assert not stale.exists()


def test_ensure_space_counts_reclaimable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda path: SimpleNamespace(total=100, used=95, free=5),
    )
    prep_cache.ensure_space(tmp_path / "train", required_bytes=12, reclaimable_bytes=10)
    with pytest.raises(prep_cache.PrepCacheSpaceError):
        prep_cache.ensure_space(tmp_path / "train", required_bytes=30, reclaimable_bytes=10)


def test_single_process_layout(tmp_path):
    staging = tmp_path / "train.staging-test"
    ds = _mapped_dataset(staging, num_proc=1)
    assert (staging / "data.arrow").is_file()
    prep_cache.publish_dataset(ds, staging, tmp_path / "train", {"mode": "clean"})
    assert prep_cache.load_prep_cache(tmp_path / "train", {"mode": "clean"}) is not None


def test_disable_caching_honors_explicit_cache_file(tmp_path):
    import datasets
    staging = tmp_path / "stage"
    staging.mkdir()
    datasets.disable_caching()
    try:
        ds = _mapped_dataset(staging, num_proc=2)
    finally:
        datasets.enable_caching()
    assert (staging / "data_00000_of_00002.arrow").is_file()
