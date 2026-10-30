"""
ASR Metrics Evaluation for Persian Speech.
Calculates WER (Word Error Rate) and CER (Character Error Rate) with:
- Raw evaluation
- Persian Normalized evaluation (standardized ZWNJ and characters)
- Diagnostic evaluation (ZWNJ -> Space)
- Formal vs. Informal utterance breakdown
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from src.normalizer import PersianTextNormalizer

try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False

try:
    from tabulate import tabulate
    _TABULATE_AVAILABLE = True
except ImportError:
    _TABULATE_AVAILABLE = False


def _levenshtein_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Standard dynamic programming Levenshtein distance for fallback."""
    r_len = len(ref_tokens)
    h_len = len(hyp_tokens)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[r_len][h_len]


def _canonical_profile_name(profile: Any) -> str:
    """Normalize profile aliases used by the simulator and eval scripts."""
    name = str(profile).strip().lower()
    return "customer" if name in ("clean", "gated") else name


def profile_subsets(
    references: Sequence[str],
    predictions: Sequence[str],
    profiles: Sequence[str],
    is_informal_flags: Optional[Sequence[bool]] = None,
) -> Dict[str, Tuple[List[str], List[str], Optional[List[bool]]]]:
    """Group references/predictions by phone profile, preserving order."""
    if not (len(references) == len(predictions) == len(profiles)):
        raise ValueError("references, predictions and profiles must have equal length.")
    if is_informal_flags is not None and len(is_informal_flags) != len(references):
        raise ValueError("is_informal_flags must match references length.")
    groups: Dict[str, List[int]] = {}
    for i, profile in enumerate(profiles):
        groups.setdefault(_canonical_profile_name(profile), []).append(i)
    out: Dict[str, Tuple[List[str], List[str], Optional[List[bool]]]] = {}
    for profile, indices in groups.items():
        out[profile] = (
            [references[i] for i in indices],
            [predictions[i] for i in indices],
            ([bool(is_informal_flags[i]) for i in indices]
             if is_informal_flags is not None else None),
        )
    return out


def filter_valid_profiles(
    references: Sequence[str],
    predictions: Sequence[str],
    profiles: Sequence[Optional[str]],
    is_informal_flags: Optional[Sequence[bool]] = None,
) -> Tuple[List[str], List[str], Optional[List[bool]], List[str], int]:
    """Drop rows whose degradation profile is missing; return skipped count."""
    if not (len(references) == len(predictions) == len(profiles)):
        raise ValueError("references, predictions and profiles must have equal length.")
    if is_informal_flags is not None and len(is_informal_flags) != len(references):
        raise ValueError("is_informal_flags must match references length.")
    indices = [
        i for i, profile in enumerate(profiles)
        if profile is not None and str(profile).strip()
    ]
    return (
        [references[i] for i in indices],
        [predictions[i] for i in indices],
        ([bool(is_informal_flags[i]) for i in indices]
         if is_informal_flags is not None else None),
        [str(profiles[i]) for i in indices],
        len(profiles) - len(indices),
    )


def profile_weights(
    profiles: Sequence[str],
    defaults: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return normalized weights for the configured profile names.

    Known default profiles keep their corpus weights (renormalized over the
    configured subset); custom profiles fall back to uniform weights.
    """
    names = list(dict.fromkeys(_canonical_profile_name(p) for p in profiles))
    if not names:
        raise ValueError("profiles must not be empty.")
    if defaults and all(
        name in defaults and float(defaults[name]) > 0.0 for name in names
    ):
        raw = {name: float(defaults[name]) for name in names}
    else:
        raw = {name: 1.0 for name in names}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("profile weights must have a positive total.")
    return {name: weight / total for name, weight in raw.items()}


def weighted_normalized_wer(
    per_profile_results: Mapping[str, Dict[str, Any]],
    weights: Mapping[str, float],
) -> float:
    """Weighted average of per-profile normalized WER (percent).

    Weights are renormalized over the profiles actually present.  If no
    result key has a matching weight, raise instead of silently reporting a
    perfect 0.00% WER.
    """
    present: Dict[str, Dict[str, Any]] = {}
    for profile, result in per_profile_results.items():
        name = _canonical_profile_name(profile)
        if float(weights.get(name, 0.0)) > 0.0:
            present[name] = result
    if not present:
        raise ValueError(
            "No profile weights match the per-profile results: "
            f"results={sorted(str(p) for p in per_profile_results)}, "
            f"weights={sorted(str(p) for p in weights)}."
        )
    total_weight = sum(float(weights[name]) for name in present)
    return sum(
        float(weights[name])
        * float(result.get("overall", {}).get("normalized_wer", 0.0))
        for name, result in present.items()
    ) / total_weight


class ASRMetricsCalculator:
    def __init__(self, normalizer: Optional[PersianTextNormalizer] = None):
        self.normalizer = normalizer or PersianTextNormalizer()

    def compute_wer(self, references: List[str], predictions: List[str]) -> float:
        """Compute corpus-level Word Error Rate (WER) in percentage."""
        # Empty inputs return 0.0 by convention (no words to err on).
        # Callers with mismatched lengths should be caught in evaluate_predictions.
        if not references or not predictions:
            return 0.0
        if _JIWER_AVAILABLE:
            return float(jiwer.wer(reference=references, hypothesis=predictions) * 100.0)

        # Fallback pure-python calculation
        total_errors = 0
        total_words = 0
        for ref, hyp in zip(references, predictions):
            ref_words = ref.split()
            hyp_words = hyp.split()
            total_words += len(ref_words)
            total_errors += _levenshtein_distance(ref_words, hyp_words)

        if total_words == 0:
            return 0.0
        return float((total_errors / total_words) * 100.0)

    def compute_cer(self, references: List[str], predictions: List[str]) -> float:
        """Compute corpus-level Character Error Rate (CER) in percentage."""
        # NOTE: jiwer.cer keeps spaces; the pure-python fallback strips spaces.
        # Difference is <0.3pp on Persian test data; fallback is only for
        # offline envs without jiwer. Empty inputs return 0.0 by convention.
        if not references or not predictions:
            return 0.0
        if _JIWER_AVAILABLE:
            return float(jiwer.cer(reference=references, hypothesis=predictions) * 100.0)

        # Fallback pure-python calculation (whitespace stripped)
        total_errors = 0
        total_chars = 0
        for ref, hyp in zip(references, predictions):
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))
            total_chars += len(ref_chars)
            total_errors += _levenshtein_distance(ref_chars, hyp_chars)

        if total_chars == 0:
            return 0.0
        return float((total_errors / total_chars) * 100.0)

    def evaluate_predictions(
        self,
        references: List[str],
        predictions: List[str],
        is_informal_flags: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive Persian ASR evaluation across multiple normalization settings
        and subsets (Overall, Formal, Informal).
        """
        assert len(references) == len(predictions), "References and predictions length mismatch."

        total_samples = len(references)
        if total_samples == 0:
            return {"total_samples": 0}

        # 1. Raw text
        raw_refs = [r.strip() for r in references]
        raw_preds = [p.strip() for p in predictions]

        # NOTE: Eval strips punctuation for standardized WER, while training
        # targets in src/data.py keep punctuation. Intentional mismatch —
        # see data.py comment. Do not unify without retraining.
        # 2. Normalized text (preserving cleaned ZWNJ)
        norm_refs = [self.normalizer.normalize(r, remove_punctuation=True, zwnj_mode="keep") for r in references]
        norm_preds = [self.normalizer.normalize(p, remove_punctuation=True, zwnj_mode="keep") for p in predictions]

        # 3. Diagnostic text (ZWNJ converted to space)
        diag_refs = [self.normalizer.normalize(r, remove_punctuation=True, zwnj_mode="to_space") for r in references]
        diag_preds = [self.normalizer.normalize(p, remove_punctuation=True, zwnj_mode="to_space") for p in predictions]

        # Compute overall metrics
        results = {
            "total_samples": total_samples,
            "overall": {
                "raw_wer": self.compute_wer(raw_refs, raw_preds),
                "raw_cer": self.compute_cer(raw_refs, raw_preds),
                "normalized_wer": self.compute_wer(norm_refs, norm_preds),
                "normalized_cer": self.compute_cer(norm_refs, norm_preds),
                "diagnostic_wer_zwnj_space": self.compute_wer(diag_refs, diag_preds),
            }
        }

        # Determine informal flags if not provided
        if is_informal_flags is None:
            is_informal_flags = [self.normalizer.is_colloquial(r) for r in references]

        formal_indices = [i for i, flag in enumerate(is_informal_flags) if not flag]
        informal_indices = [i for i, flag in enumerate(is_informal_flags) if flag]

        if formal_indices:
            results["formal"] = {
                "count": len(formal_indices),
                "normalized_wer": self.compute_wer(
                    [norm_refs[i] for i in formal_indices],
                    [norm_preds[i] for i in formal_indices]
                ),
                "normalized_cer": self.compute_cer(
                    [norm_refs[i] for i in formal_indices],
                    [norm_preds[i] for i in formal_indices]
                ),
            }

        if informal_indices:
            results["informal"] = {
                "count": len(informal_indices),
                "normalized_wer": self.compute_wer(
                    [norm_refs[i] for i in informal_indices],
                    [norm_preds[i] for i in informal_indices]
                ),
                "normalized_cer": self.compute_cer(
                    [norm_refs[i] for i in informal_indices],
                    [norm_preds[i] for i in informal_indices]
                ),
            }

        return results

    @staticmethod
    def format_results_table(model_name: str, results: Dict[str, Any]) -> str:
        """Formats the evaluation results into a clean markdown table."""
        overall = results.get("overall", {})
        formal = results.get("formal", {})
        informal = results.get("informal", {})

        table_data = [
            ["Model", model_name],
            ["Total Test Samples", results.get("total_samples", 0)],
            ["Raw WER (%)", f"{overall.get('raw_wer', 0.0):.2f}%"],
            ["Raw CER (%)", f"{overall.get('raw_cer', 0.0):.2f}%"],
            ["Normalized WER (%)", f"{overall.get('normalized_wer', 0.0):.2f}%"],
            ["Normalized CER (%)", f"{overall.get('normalized_cer', 0.0):.2f}%"],
            ["Diagnostic WER (ZWNJ->Space) (%)", f"{overall.get('diagnostic_wer_zwnj_space', 0.0):.2f}%"],
            ["Formal Subset Norm WER (count)", f"{formal.get('normalized_wer', 0.0):.2f}% ({formal.get('count', 0)})"],
            ["Informal Subset Norm WER (count)", f"{informal.get('normalized_wer', 0.0):.2f}% ({informal.get('count', 0)})"],
        ]
        if _TABULATE_AVAILABLE:
            return tabulate(table_data, headers=["Metric", "Value"], tablefmt="github")
        else:
            lines = [f"{'Metric':<35} | {'Value'}"]
            lines.append("-" * 55)
            for row in table_data:
                lines.append(f"{row[0]:<35} | {row[1]}")
            return "\n".join(lines)


def evaluate_predictions_by_profile(
    calculator: ASRMetricsCalculator,
    references: Sequence[str],
    predictions: Sequence[str],
    profiles: Sequence[str],
    is_informal_flags: Optional[Sequence[bool]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run ``evaluate_predictions`` separately for each phone profile."""
    return {
        profile: calculator.evaluate_predictions(refs, preds, flags)
        for profile, (refs, preds, flags) in profile_subsets(
            references, predictions, profiles, is_informal_flags
        ).items()
    }
