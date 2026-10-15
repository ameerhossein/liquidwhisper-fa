"""Centralized managed directories for Persian ASR pipeline.

Single source of truth for persistent-volume-aware defaults:
- output_dir  (checkpoints + best_adapter + final reports)
- results_dir (telemetry / benchmarks / logs)
- merged_dir  (standalone merged model output)

Precedence everywhere: explicit arg/ENV wins, then /workspace if present,
else repo-local fallback. This lets the committed /workspace default work
on RunPod while local runs without /workspace still write locally.

Mirrors the historical behavior of scripts/train_lora.py and server/app.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent

PERSISTENT_ROOTS = ("workspace", "runpod-volume")

OUTPUT_DIR_PERSISTENT = "/workspace/stt-outputs/buzzasr-neyshekar-lora"
OUTPUT_DIR_LOCAL = "outputs/buzzasr-neyshekar-lora"

RESULTS_DIR_PERSISTENT = "/workspace/stt-outputs/results"
RESULTS_DIR_LOCAL = "results"

MERGED_DIR_PERSISTENT = "/workspace/stt-outputs/buzzasr-persian-merged"
MERGED_DIR_LOCAL = "models/buzzasr-persian-fine-tuned"


def _env(name: str) -> str:
    return (os.environ.get(name, "") or "").strip()


def default_output_dir() -> str:
    """Default checkpoints/best_adapter dir."""
    env = _env("ASR_OUTPUT_DIR")
    if env:
        return env
    try:
        if Path("/workspace").is_dir():
            return OUTPUT_DIR_PERSISTENT
    except Exception:
        pass
    return OUTPUT_DIR_LOCAL


def default_results_dir() -> str:
    """Default telemetry/benchmarks dir."""
    env = _env("ASR_RESULTS_DIR")
    if env:
        return env
    try:
        if Path("/workspace").is_dir():
            return RESULTS_DIR_PERSISTENT
    except Exception:
        pass
    return RESULTS_DIR_LOCAL


def default_merged_dir() -> str:
    """Default standalone merged-model dir."""
    env = _env("ASR_MERGED_DIR")
    if env:
        return env
    try:
        if Path("/workspace").is_dir():
            return MERGED_DIR_PERSISTENT
    except Exception:
        pass
    return MERGED_DIR_LOCAL


def _fallback_if_persistent_missing(raw: str, local: str, label: str) -> str:
    """Fall back to repo-local dir when a /workspace|/runpod-volume root is absent."""
    try:
        p = Path(raw)
        if p.is_absolute():
            parts = p.parts
            if len(parts) > 1 and parts[1] in PERSISTENT_ROOTS:
                if not Path("/" + parts[1]).is_dir():
                    print(f"Note: {raw} unavailable (/{parts[1]} missing); using {local}.")
                    return local
    except Exception:
        pass
    return raw


def resolve_output_dir(raw: Optional[str] = None) -> str:
    raw = (raw or "").strip() or default_output_dir()
    return _fallback_if_persistent_missing(raw, OUTPUT_DIR_LOCAL, "output")


def resolve_results_dir(raw: Optional[str] = None) -> str:
    raw = (raw or "").strip() or default_results_dir()
    return _fallback_if_persistent_missing(raw, RESULTS_DIR_LOCAL, "results")


def resolve_merged_dir(raw: Optional[str] = None) -> str:
    raw = (raw or "").strip() or default_merged_dir()
    return _fallback_if_persistent_missing(raw, MERGED_DIR_LOCAL, "merged")


def resolve_adapter_dir(output_dir: Optional[str] = None, adapter_path: Optional[str] = None) -> str:
    """Resolve best_adapter dir: explicit path wins, else <output_dir>/best_adapter."""
    if (adapter_path or "").strip():
        return (adapter_path or "").strip()
    return str(Path(resolve_output_dir(output_dir)) / "best_adapter")


def is_complete_checkpoint(path: Path | str) -> bool:
    """A checkpoint-* dir is complete when trainer_state.json exists."""
    p = Path(path)
    return p.is_dir() and (p / "trainer_state.json").is_file()


def find_complete_checkpoints(output_dir: Optional[str] = None) -> List[Path]:
    """Return complete checkpoint-* dirs sorted by step ascending."""
    base = Path(resolve_output_dir(output_dir))
    if not base.is_dir():
        return []
    complete: list[tuple[int, Path]] = []
    for p in base.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        if is_complete_checkpoint(p):
            complete.append((step, p))
    return [p for _, p in sorted(complete)]


def find_incomplete_checkpoints(output_dir: Optional[str] = None) -> List[Path]:
    """Return checkpoint-* dirs missing trainer_state.json (interrupted saves)."""
    base = Path(resolve_output_dir(output_dir))
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in base.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            int(p.name.split("-")[-1])
        except ValueError:
            continue
        if not is_complete_checkpoint(p):
            out.append(p)
    return sorted(out)


def find_last_checkpoint(output_dir: Optional[str] = None) -> Optional[str]:
    """Return newest complete checkpoint-* dir, or None."""
    complete = find_complete_checkpoints(output_dir)
    for p in find_incomplete_checkpoints(output_dir):
        print(f"Skipping incomplete checkpoint (no trainer_state.json): {p}")
    if not complete:
        return None
    return str(complete[-1])
