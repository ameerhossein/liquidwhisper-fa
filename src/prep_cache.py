"""Direct-shard cache for prepared datasets.

``datasets.map(..., cache_file_name=...)`` writes Arrow shards directly into a
staging directory.  This module publishes those shards as a cache slot:

- ``meta.json`` is written last and atomically;
- the staging directory is renamed into place atomically;
- loading validates the format, fingerprint, shard names/sizes/row counts,
  total row count, and feature schema, treating any mismatch as a cache miss.

No ``Dataset.save_to_disk`` copy is made, so the cache costs one copy of the
prepared features instead of two.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, concatenate_datasets

PREP_CACHE_FORMAT = "shards_v1"
_META_NAME = "meta.json"


class PrepCacheSpaceError(RuntimeError):
    """Raised when the prep cache cannot fit on the target filesystem."""


def staging_dir(cache_dir: Path) -> Path:
    """Return a unique staging sibling for ``cache_dir``."""
    cache_dir = Path(cache_dir)
    return cache_dir.with_name(
        f"{cache_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )


def cleanup_stale_staging(cache_dir: Path) -> None:
    """Remove staging directories left behind by crashed prep runs."""
    cache_dir = Path(cache_dir)
    parent = cache_dir.parent
    if not parent.is_dir():
        return
    for stale in parent.glob(f"{cache_dir.name}.staging-*"):
        shutil.rmtree(stale, ignore_errors=True)


def estimate_feature_bytes(rows: int, copies: int = 1) -> int:
    """Rough feature size for ``rows`` rows: 128 mel bins x 3000 frames x 4 bytes."""
    return int(rows * 128 * 3000 * 4 * copies)


def dir_size(path: Path) -> int:
    """Total size of files under ``path`` in bytes; 0 when it does not exist."""
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _format_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def ensure_space(
    cache_dir: Path,
    required_bytes: int,
    reclaimable_bytes: int = 0,
) -> None:
    """Fail before a large prep when the target filesystem cannot hold it.

    ``reclaimable_bytes`` is space that the caller will free (for example the
    old cache slot) before writing the new one.
    """
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir.parent).free
    available = free + int(reclaimable_bytes)
    if available < int(required_bytes):
        raise PrepCacheSpaceError(
            f"Not enough disk space for prep cache at {cache_dir}: "
            f"need {_format_gb(required_bytes)}, "
            f"have {_format_gb(free)} free + "
            f"{_format_gb(reclaimable_bytes)} reclaimable = "
            f"{_format_gb(available)} available"
        )


def _dataset_shard_paths(dataset: Dataset, staging: Path) -> list:
    """Resolve the dataset's ordered ``cache_files`` and require them in staging."""
    entries = list(getattr(dataset, "cache_files", None) or [])
    if not entries:
        raise ValueError("dataset exposes no cache_files shards to publish")

    staging = staging.resolve()
    shards = []
    for entry in entries:
        filename = entry.get("filename") if isinstance(entry, dict) else entry
        if not filename:
            raise ValueError(f"dataset cache_files entry has no filename: {entry!r}")
        path = Path(filename)
        if not path.is_absolute():
            path = staging / path
        path = path.resolve()
        if not path.is_relative_to(staging):
            raise ValueError(f"dataset shard {path} is outside staging dir {staging}")
        if not path.is_file():
            raise ValueError(f"dataset shard {path} does not exist")
        shards.append(path)
    return shards


def _remove_unreferenced_files(staging: Path, shard_paths: list) -> None:
    """Drop everything in staging except the shards and (later) meta.json."""
    keep = {path.resolve() for path in shard_paths}
    entries = list(staging.rglob("*"))
    for entry in entries:
        if entry.is_file() and entry.resolve() not in keep:
            entry.unlink()
    # Prune now-empty directories so only the shard files remain.
    for entry in sorted(
        (item for item in entries if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            entry.rmdir()
        except OSError:
            pass


def publish_dataset(
    dataset: Dataset,
    staging: Path,
    cache_dir: Path,
    fingerprint: Dict[str, Any],
) -> Dict[str, Any]:
    """Publish staged shards as ``cache_dir`` and return the written metadata."""
    staging = Path(staging).resolve()
    cache_dir = Path(cache_dir).resolve()
    if staging == cache_dir:
        raise ValueError("staging dir and cache dir must differ")
    if not staging.is_dir():
        raise ValueError(f"staging dir {staging} does not exist")

    shard_paths = _dataset_shard_paths(dataset, staging)
    _remove_unreferenced_files(staging, shard_paths)

    shards = []
    total_rows = 0
    for path in shard_paths:
        shard = Dataset.from_file(str(path))
        rows = int(shard.num_rows)
        total_rows += rows
        shards.append(
            {"file": path.name, "rows": rows, "bytes": path.stat().st_size}
        )
    if total_rows != len(dataset):
        raise ValueError(
            f"shard rows ({total_rows}) do not match dataset rows ({len(dataset)})"
        )

    meta = {
        "format": PREP_CACHE_FORMAT,
        "fingerprint": fingerprint,
        "total_rows": len(dataset),
        "features": dataset.features.to_dict(),
        "shards": shards,
    }

    meta_tmp = staging / f"{_META_NAME}.tmp"
    with meta_tmp.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(meta_tmp, staging / _META_NAME)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    remove_cache(cache_dir)
    os.replace(staging, cache_dir)
    return meta


def load_prep_cache(
    cache_dir: Path,
    fingerprint: Dict[str, Any],
) -> Optional[Dataset]:
    """Load a published cache, or return ``None`` on any validation failure."""
    cache_dir = Path(cache_dir)
    try:
        with (cache_dir / _META_NAME).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        if not isinstance(meta, dict):
            raise ValueError("meta.json is not an object")
        if meta.get("format") != PREP_CACHE_FORMAT:
            raise ValueError(f"format {meta.get('format')!r} != {PREP_CACHE_FORMAT!r}")
        if meta.get("fingerprint") != fingerprint:
            raise ValueError("fingerprint mismatch")

        total_rows = int(meta["total_rows"])
        if total_rows <= 0:
            raise ValueError("total_rows must be positive")

        shard_entries = meta["shards"]
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError("shards list is empty")

        meta_features = meta["features"]
        parts = []
        for entry in shard_entries:
            name = entry["file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"invalid shard filename {name!r}")
            path = cache_dir / name
            if not path.is_file():
                raise ValueError(f"missing shard {name}")
            if path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"size mismatch for shard {name}")
            shard = Dataset.from_file(str(path))
            if int(shard.num_rows) != int(entry["rows"]):
                raise ValueError(f"row-count mismatch for shard {name}")
            if shard.features.to_dict() != meta_features:
                raise ValueError(f"feature-schema mismatch for shard {name}")
            parts.append(shard)

        combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if len(combined) != total_rows:
            raise ValueError(
                f"loaded {len(combined)} rows, metadata says {total_rows}"
            )
    except Exception as exc:  # any broken cache is a miss, never a partial load
        print(f"Prep cache MISS at {cache_dir}: {exc}")
        return None

    print(f"Prep cache HIT at {cache_dir}: {total_rows} rows in {len(parts)} shard(s)")
    return combined


def remove_cache(cache_dir: Path) -> None:
    """Remove a cache slot (or a leftover staging directory)."""
    shutil.rmtree(Path(cache_dir), ignore_errors=True)
