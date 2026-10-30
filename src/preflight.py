"""Pure helpers for the automatic preflight baseline.

This module owns the measurable, side-effect-free half of the preflight:

- :func:`build_fingerprint` describes the exact evaluation setup so a cached
  baseline can be reused only for an apples-to-apples run.
- :func:`load_cached_preflight` validates and reads a cached preflight file.
- :func:`parse_evaluation_details` normalizes the JSON written by
  ``scripts/evaluate_model.py`` into one benchmark row.
- :func:`compute_deltas` compares an epoch row against a measured baseline.

Deliberately no ``torch``/``transformers`` imports here: the subprocess runner
is added separately, and these helpers must stay importable without the heavy
training stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PREFLIGHT_VERSION = "1"

DEFAULT_PREFLIGHT_MODELS: Dict[str, str] = {
    "buzzasr_base": "BuzzASR/persian",
    "whisper_v3": "openai/whisper-large-v3",
}


def _round_metric(value: Any) -> Optional[float]:
    """Round a metric to two decimals, preserving missing values as ``None``."""
    if value is None:
        return None
    return round(float(value), 2)


def build_fingerprint(
    *,
    dataset_name: str,
    split: str,
    sample_count: int,
    phone_seed: int,
    telephony_profiles: Sequence[str],
    telephony_use_ffmpeg: bool,
    eval_batch_size: int,
    models: Mapping[str, str] = DEFAULT_PREFLIGHT_MODELS,
) -> Dict[str, Any]:
    """Return the normalized fingerprint that identifies a preflight run.

    Profiles are sorted so ordering differences do not invalidate the cache.
    The model mapping is copied and merged at the top level, matching the
    on-disk preflight schema (``buzzasr_base`` / ``whisper_v3`` keys).
    """
    fingerprint: Dict[str, Any] = {
        "preflight_version": PREFLIGHT_VERSION,
        "dataset_name": str(dataset_name),
        "split": str(split),
        "sample_count": int(sample_count),
        "phone_seed": int(phone_seed),
        "telephony_profiles": sorted(str(profile) for profile in telephony_profiles),
        "telephony_use_ffmpeg": bool(telephony_use_ffmpeg),
        "eval_batch_size": int(eval_batch_size),
    }
    fingerprint.update(dict(models))
    return fingerprint


def load_cached_preflight(
    path: Path,
    fingerprint: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a cached preflight only when its fingerprint and models match.

    Any unreadable, malformed, or mismatched cache is treated as a miss so a
    caller can safely recompute the baseline.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    if not isinstance(data, dict):
        return None
    if data.get("fingerprint") != dict(fingerprint):
        return None

    models = data.get("models")
    if not isinstance(models, dict) or "buzzasr_base" not in models:
        return None
    return data


def parse_evaluation_details(
    payload: Mapping[str, Any],
    *,
    name: str,
    epoch: str,
    source: str = "preflight",
    is_reference: bool = False,
    phone_seed: Optional[int] = None,
    phone_profiles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convert an ``evaluation_details.json`` payload into a benchmark row.

    Clean metrics always come from ``payload["metrics"]``. Phone metrics are
    copied only when ``payload["phone_metrics"]["overall"]`` is present, so a
    clean-only evaluation omits phone keys instead of inventing zeros.
    Metrics are rounded to two decimals (WER/CER are percentages).
    """
    metrics = payload.get("metrics") or {}
    overall = metrics.get("overall") or {}
    formal = metrics.get("formal") or {}
    informal = metrics.get("informal") or {}

    row: Dict[str, Any] = {
        "name": name,
        "epoch": epoch,
        "source": source,
        "is_reference": bool(is_reference),
        "total_samples": int(metrics.get("total_samples") or 0),
        "normalized_wer": _round_metric(overall.get("normalized_wer")),
        "normalized_cer": _round_metric(overall.get("normalized_cer")),
        "diagnostic_wer": _round_metric(overall.get("diagnostic_wer_zwnj_space")),
        "formal_wer": _round_metric(formal.get("normalized_wer")),
        "informal_wer": _round_metric(informal.get("normalized_wer")),
    }

    phone_metrics = payload.get("phone_metrics") or {}
    phone_overall = phone_metrics.get("overall") or {}
    if phone_overall:
        phone_formal = phone_metrics.get("formal") or {}
        phone_informal = phone_metrics.get("informal") or {}

        cached_seed = payload.get("phone_seed")
        cached_profiles = payload.get("phone_profiles")
        effective_profiles = cached_profiles if cached_profiles is not None else phone_profiles

        row.update(
            {
                "phone_normalized_wer": _round_metric(phone_overall.get("normalized_wer")),
                "phone_normalized_cer": _round_metric(phone_overall.get("normalized_cer")),
                "phone_formal_wer": _round_metric(phone_formal.get("normalized_wer")),
                "phone_informal_wer": _round_metric(phone_informal.get("normalized_wer")),
                "phone_seed": cached_seed if cached_seed is not None else phone_seed,
                "phone_profiles": list(effective_profiles) if effective_profiles is not None else None,
            }
        )

    return row


def compute_deltas(
    clean_wer: float,
    phone_wer: Optional[float],
    total_samples: int,
    baseline: Optional[Mapping[str, Any]],
) -> Dict[str, Optional[float]]:
    """Return clean/phone deltas in percentage points against a baseline.

    Deltas are ``None`` unless the baseline was measured on exactly the same
    number of samples. Negative values mean improvement. The phone delta also
    requires both phone WER values to be present.
    """
    deltas: Dict[str, Optional[float]] = {
        "delta_vs_buzzasr": None,
        "delta_phone_vs_buzzasr": None,
    }
    if not baseline or baseline.get("total_samples") != total_samples:
        return deltas

    baseline_clean = baseline.get("normalized_wer")
    if clean_wer is not None and baseline_clean is not None:
        deltas["delta_vs_buzzasr"] = round(float(clean_wer) - float(baseline_clean), 2)

    baseline_phone = baseline.get("phone_normalized_wer")
    if phone_wer is not None and baseline_phone is not None:
        deltas["delta_phone_vs_buzzasr"] = round(float(phone_wer) - float(baseline_phone), 2)

    return deltas
