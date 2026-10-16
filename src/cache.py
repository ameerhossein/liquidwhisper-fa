"""Hugging Face cache location: prefer the disk with the most free space.

Background: on RunPod /workspace is a small (often ~100G, easily filled)
persistent volume while / (overlay: /root, /tmp) has ~200G free. Pointing
HF raw downloads + datasets.map temp files at /workspace fills it and fails
with OSError 28 during prepare. Checkpoints must stay on /workspace
(persistence); caches are re-downloadable, so they belong on the big disk.

Precedence: ASR_HF_CACHE_DIR (our own var, always deliberate) wins, then
explicit HF_DATASETS_CACHE leaf, else auto-pick the roomiest writable disk.
Deliberately does NOT honor HF_HOME: pod images (e.g. RunPod templates)
export HF_HOME=/workspace/.cache/huggingface by default, which silently
pins multi-GB downloads + map temps to the small volume. Override that
unless ASR_HONOR_HF_HOME=1. Must run before any datasets/transformers
import (they snapshot config at import). Stdlib-only for module-top use.
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
    """Resolved HF base without side effects.

    ASR_HF_CACHE_DIR wins, then an explicit HF_DATASETS_CACHE leaf (its
    parent). HF_HOME is ignored here on purpose: image templates export a
    default HF_HOME pointing at the small persistent volume.
    """
    v = (os.environ.get("ASR_HF_CACHE_DIR", "") or "").strip()
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


def default_datasets_cache_dir() -> str:
    """Effective datasets leaf: explicit HF_DATASETS_CACHE or <base>/datasets."""
    ds = (os.environ.get("HF_DATASETS_CACHE", "") or "").strip()
    if ds:
        return ds
    return os.path.join(default_hf_cache_dir(), "datasets")


def setup_hf_cache(verbose: bool = True) -> str:
    """Point HF caches at the big disk. Returns chosen base.

    Overrides a template-default HF_HOME (e.g. /workspace/...) with a loud
    warning; set ASR_HONOR_HF_HOME=1 to keep an HF_HOME you set deliberately.
    """
    base = default_hf_cache_dir()

    datasets_cache = default_datasets_cache_dir()
    os.environ["HF_DATASETS_CACHE"] = datasets_cache

    honor = (os.environ.get("ASR_HONOR_HF_HOME", "") or "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )
    prev_home = (os.environ.get("HF_HOME", "") or "").strip()
    if honor and prev_home:
        base = prev_home
        datasets_cache = default_datasets_cache_dir()
        os.environ["HF_DATASETS_CACHE"] = datasets_cache
    else:
        if prev_home and os.path.normpath(prev_home) != os.path.normpath(base):
            print(f"HF cache: overriding HF_HOME={prev_home} -> {base} (small-volume template default; set ASR_HONOR_HF_HOME=1 to keep).")
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
            ("DEFAULT_DATASETS_CACHE_PATH", datasets_cache),
            ("DATASETS_CACHE_PATH", datasets_cache),
            ("HF_HOME", base),
            ("HUB_CACHE", hub_cache),
            ("HF_HUB_CACHE", hub_cache),
            ("DEFAULT_HUB_CACHE_PATH", hub_cache),
            ("HUB_CACHE_PATH", hub_cache),
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


def default_prep_cache_dir() -> str:
    """Default dir for prepared Arrow datasets (85G+; must avoid /workspace).

    Explicit ASR_PREP_CACHE_DIR wins (use a /workspace/... path there if you
    want persistence across pod restarts at the cost of disk). Otherwise a
    sibling of the HF base, i.e. same roomy disk: <parent>/prep_cache.
    """
    v = (os.environ.get("ASR_PREP_CACHE_DIR", "") or "").strip()
    if v:
        return v
    try:
        parent = os.path.dirname(default_hf_cache_dir().rstrip("/")) or str(Path.home())
    except Exception:
        parent = str(Path.home())
    return os.path.join(parent, "prep_cache")
