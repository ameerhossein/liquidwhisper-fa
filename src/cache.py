"""Hugging Face cache location: prefer the disk with the most free space.

Background: on RunPod /workspace is a small (often ~100G, easily filled)
persistent volume while / (overlay: /root, /tmp) has ~200G free. Pointing
HF raw downloads + datasets.map temp files at /workspace fills it and fails
with OSError 28 during prepare. Checkpoints must stay on /workspace
(persistence); caches are re-downloadable, so they belong on the big disk.

Precedence: explicit env wins (HF_DATASETS_CACHE / HF_HOME /
ASR_HF_CACHE_DIR), else auto-pick the roomiest writable disk. Must run
before any datasets/transformers import (they snapshot config at import).
Stdlib-only so it can run at module top.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _free_bytes(path: str) -> int:
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return 0


def _writable(path: str) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return os.access(str(p), os.W_OK)
    except Exception:
        return False


def _auto_base() -> str:
    """Pick cache base on the roomiest writable disk."""
    home = str(Path.home())
    candidates = [
        os.path.join(home, ".cache", "huggingface"),  # standard, overlay on RunPod
        os.path.join("/tmp", "hf_cache"),  # overlay, ephemeral
    ]
    if Path("/workspace").is_dir():
        candidates.append("/workspace/stt-outputs/hf_cache")  # legacy small volume
    ranked: list[tuple[int, str]] = []
    for c in candidates:
        parent = os.path.dirname(c) or "/"
        if not Path(parent).is_dir():
            continue
        if not _writable(parent):
            continue
        ranked.append((_free_bytes(parent), c))
    if not ranked:
        return os.path.join(home, ".cache", "huggingface")
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[0][1]


def default_hf_cache_dir() -> str:
    """Resolved HF_HOME base without side effects (explicit env wins)."""
    for key in ("ASR_HF_CACHE_DIR", "HF_HOME"):
        v = (os.environ.get(key, "") or "").strip()
        if v:
            return v
    ds = (os.environ.get("HF_DATASETS_CACHE", "") or "").strip()
    if ds:
        # HF_DATASETS_CACHE points at the datasets leaf; base is its parent.
        parent = os.path.dirname(ds.rstrip("/"))
        if os.path.basename(ds.rstrip("/")) == "datasets" and parent:
            return parent
        return parent or ds
    return _auto_base()


def setup_hf_cache(verbose: bool = True) -> str:
    """Point HF caches at the big disk. Returns chosen HF_HOME base."""
    base = default_hf_cache_dir()

    datasets_cache = (os.environ.get("HF_DATASETS_CACHE", "") or "").strip()
    if not datasets_cache:
        datasets_cache = os.path.join(base, "datasets")
        os.environ["HF_DATASETS_CACHE"] = datasets_cache

    if not (os.environ.get("HF_HOME", "") or "").strip():
        os.environ["HF_HOME"] = base

    hub_cache = (
        (os.environ.get("HF_HUB_CACHE", "") or "").strip()
        or (os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
        or os.path.join(base, "hub")
    )
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    # Old transformers (<4.x cache var) shares the hub dir.
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)

    for d in {base, datasets_cache, hub_cache, os.path.join(base, "xet")}:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # datasets snapshots cache paths at import; patch live if already imported.
    try:
        import datasets.config as _cfg  # type: ignore

        for attr, val in (
            ("HF_DATASETS_CACHE", datasets_cache),
            ("HF_HOME", base),
            ("HUB_CACHE", hub_cache),
            ("HF_HUB_CACHE", hub_cache),
        ):
            try:
                if hasattr(_cfg, attr):
                    setattr(_cfg, attr, val)
            except Exception:
                pass
    except Exception:
        pass

    if verbose:
        try:
            free_gb = _free_bytes(datasets_cache if Path(datasets_cache).is_dir() else os.path.dirname(datasets_cache) or "/") / 1e9
            print(f"HF cache: {base} (datasets -> {datasets_cache}, free ~{free_gb:.0f}G)")
        except Exception:
            print(f"HF cache: {base}")
    return base


def log_cache_space(label: str = "Cache") -> None:
    """One-line free-space summary for the chosen cache + CWD volume."""
    try:
        base = default_hf_cache_dir()
        free = _free_bytes(base if Path(base).is_dir() else os.path.dirname(base) or "/") / 1e9
        print(f"{label} disk: {base} (~{free:.0f}G free)")
    except Exception:
        pass
