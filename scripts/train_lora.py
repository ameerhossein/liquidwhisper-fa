"""
Fine-tune BuzzASR/persian on Neyshekar using 16-bit LoRA (PEFT).
Optimized for NVIDIA L40S (48 GB VRAM):
- Bfloat16 training
- LoRA on attention & feed-forward projections (r=64, alpha=128)
- Fused AdamW optimizer
- Scaled Dot-Product Attention / FlashAttention
- Native Persian text normalization for validation WER
"""

import os
import sys
import argparse
import json
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace

# Must precede any HF tokenizer import: forked datasets.map workers deadlock
# if the parent created tokenizers with parallelism enabled (see src/data.py).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add project root to sys.path FIRST so src.cache is importable below.
from pathlib import Path as _EarlyPath

sys.path.insert(0, str(_EarlyPath(__file__).resolve().parent.parent))

# HF caches (raw downloads + datasets.map temps) -> biggest disk, not the
# small /workspace volume (OSError 28 root cause). Must precede any
# datasets/transformers import (they snapshot cache paths at import).
# Explicit HF_DATASETS_CACHE / HF_HOME / ASR_HF_CACHE_DIR still wins.
try:
    from src.cache import setup_hf_cache as _setup_hf_cache

    _setup_hf_cache(verbose=True)
except Exception:
    pass
from pathlib import Path
from typing import Dict, Any, Mapping, Optional, Callable
import torch
import evaluate
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, PeftModel

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Project root already on sys.path (see top-of-file early insert for src.cache).
# Load .env (repo root / CWD) before any os.environ reads. Stdlib-only.
try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

from src.normalizer import PersianTextNormalizer
from src.metrics import ASRMetricsCalculator
from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.data import PersianASRDataPipeline, persian_label_ids
from src.trainer_callbacks import (
    WebTelemetryCallback,
    HFMirrorCallback,
    _resolve_infer_dtype,
    _should_autocast,
)
from src.preflight import compute_deltas, preflight_path, run_preflight
from src import paths as _managed_paths
from src import prep_cache
from src import run_state
from src import hf_sync


def _default_output_dir() -> str:
    """Managed default (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.default_output_dir()


def _default_results_dir() -> str:
    """Managed default (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.default_results_dir()


def _resolve_results_dir(raw: Optional[str]) -> str:
    """Managed resolver (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.resolve_results_dir(raw)


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser without applying the YAML overlay."""
    parser = argparse.ArgumentParser(description="Train BuzzASR with LoRA on Persian Speech")
    # Model & Data
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. configs/l40s_train_config.yaml). CLI flags override file.")
    parser.add_argument("--base_model", type=str, default="BuzzASR/persian", help="Base model identifier")
    parser.add_argument("--dataset_name", type=str, default="shekar-ai/Neyshekar", help="Dataset identifier")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir(), help="Output directory for checkpoints (default: /workspace/... on RunPod, else outputs/...)")
    parser.add_argument("--results_dir", type=str, default=None, help="Telemetry/benchmark directory (default: /workspace/stt-outputs/results on RunPod, else results/). Env ASR_RESULTS_DIR also works.")
    parser.add_argument("--run_dir", type=str, default=None, help="Persistent run directory for the cockpit/CLI run ledger (optional).")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (smoke test)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples")

    # Hugging Face run mirror (off unless a repo id is configured).
    parser.add_argument("--hf_run_repo", "--hf-run-repo", type=str, default=None, help="Hugging Face repo id (<owner>/<name>) used to mirror this run. Env ASR_HF_RUN_REPO also works; unset disables mirroring.")
    parser.add_argument("--hf_private", "--hf-private", action=argparse.BooleanOptionalAction, default=True, help="Create the mirror repo private (default).")
    parser.add_argument("--hf_adapter_snapshot_steps", "--hf-adapter-snapshot-steps", type=int, default=250, help="Upload an adapter-only snapshot every N steps; 0 disables snapshots.")
    parser.add_argument("--hf_auto_resume", "--hf-auto-resume", action=argparse.BooleanOptionalAction, default=True, help="Pull a newer remote checkpoint before resuming (ignored with --fresh_start).")
    parser.add_argument("--hf_resume", type=str, default="none", metavar="{none,latest,<run_id>}", help="Restore a mirrored run (latest or a run id) and resume it instead of starting fresh.")

    # LoRA Hyperparameters
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training Hyperparameters (L40S tuned)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch: batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for LoRA")
    parser.add_argument("--epochs", type=int, default=2, help="Total training epochs (paired default; disjoint/clean use 4 for equal compute)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning-rate schedule (e.g. cosine, linear)")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Explicit checkpoint to resume from: path, or 'last' for latest in output_dir. Overrides --auto_resume.")
    parser.add_argument("--auto_resume", "--auto-resume", action=argparse.BooleanOptionalAction, default=True, help="Automatically resume from latest checkpoint in output_dir (for pod restarts). Use --no-auto_resume for a fresh run.")
    parser.add_argument("--fresh_start", "--fresh-start", action=argparse.BooleanOptionalAction, default=False, help="Start fresh: delete prior checkpoint-*, best_adapter, reports and telemetry history in output_dir/results before training.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of preprocessing workers")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device eval + benchmark batch size")
    parser.add_argument("--max_benchmark_samples", type=int, default=1000, help="Test subset for end-of-epoch benchmark, decoded twice (1000 clean + same 1000 phone). 0/None = full set (2149).")
    parser.add_argument("--preflight_samples", type=int, default=None,
                        help="Preflight rows; default follows --max_benchmark_samples (0 = full test).")
    parser.add_argument("--overwrite_preflight", "--overwrite-preflight",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight_cache_dir", type=str, default=None,
                        help="Shared fingerprint-keyed preflight cache directory (default: results_dir).")
    parser.add_argument("--run_post_test_eval", "--run-post-test-eval", action=argparse.BooleanOptionalAction, default=True, help="Run full test evaluation on best model post-training and save report")
    parser.add_argument("--post_eval_only", action=argparse.BooleanOptionalAction, default=False, help="Run only the post-training evaluation from the existing adapter (cockpit resume).")
    parser.add_argument("--enable_telemetry", "--enable-telemetry", action=argparse.BooleanOptionalAction, default=True, help="Enable real-time telemetry for Web UI (--no-enable-telemetry to disable)")
    parser.add_argument("--run_epoch_benchmark", "--run-epoch-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Run test benchmark after each epoch (--no-run-epoch-benchmark to disable)")
    parser.add_argument("--telephony_prob", type=float, default=0.5, help="Share of train samples phone-simulated on-the-fly (0=clean only, 0.5=mixed default). Applies only to --telephony_mode disjoint.")
    parser.add_argument(
        "--telephony_mode",
        choices=("paired", "disjoint", "clean"),
        default="paired",
        help="paired=clean+phone rows (2 epochs), disjoint=prob-weighted coin, clean=no augmentation",
    )
    parser.add_argument("--telephony_profiles", type=str, default="customer", help="Comma-separated phone profiles: customer (default); legacy gated,mobile,pstn also accepted")
    parser.add_argument("--telephony_seed", type=int, default=42, help="RNG seed for training-time phone augmentation")
    parser.add_argument("--telephony_use_ffmpeg", "--telephony-use-ffmpeg", action=argparse.BooleanOptionalAction, default=True, help="Use true ffmpeg AMR-NB codec when available; G.711 A-law storage is always exact")
    parser.add_argument("--phone_benchmark", "--phone-benchmark", action=argparse.BooleanOptionalAction, default=True, help="Dual clean + phone eval each epoch (--no-phone-benchmark for clean only)")
    parser.add_argument("--phone_seed", type=int, default=1234, help="Deterministic seed for phone-simulated eval")
    # Prepared-feature cache (persistent volume): skip the 30-90min
    # "Preparing ASR Features" map on restarts by reusing mel+tokens.
    parser.add_argument("--prep_cache_dir", type=str, default=None, help="Dir for prepared train/eval Arrow cache (default: big disk via src/cache.py; --prep_cache_dir <output_dir>/prep_cache restores persistent-volume caching). Env ASR_PREP_CACHE_DIR also works.")
    parser.add_argument("--reuse_prep", "--reuse-prep", action=argparse.BooleanOptionalAction, default=True, help="Load prepared datasets from prep_cache_dir when fingerprint matches (skips re-prep).")
    parser.add_argument("--overwrite_prep_cache", "--overwrite-prep-cache", action=argparse.BooleanOptionalAction, default=False, help="Force re-prep even if cache hits (use after data/code/telephone-seed changes).")
    parser.add_argument("--gradient_checkpointing", "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False, help="Recompute activations in the backward pass to save VRAM. Off is the fast L40S default; enable it only if a larger micro-batch requires it.")
    return parser


def _explicit_cli_keys(parser: argparse.ArgumentParser, argv) -> set[str]:
    """Return the ``action.dest`` values named explicitly on the command line."""
    dest_by_option: Dict[str, Optional[str]] = {}
    for action in parser._actions:
        for option in getattr(action, "option_strings", ()) or ():
            dest_by_option[option] = getattr(action, "dest", None)
    keys: set[str] = set()
    for token in argv or ():
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        dest = dest_by_option.get(name)
        if dest is None and name.startswith("--no-"):
            base = name[5:]
            for candidate in (
                "--" + base,
                "--" + base.replace("-", "_"),
                "--" + base.replace("_", "-"),
            ):
                if candidate in dest_by_option:
                    dest = dest_by_option[candidate]
                    break
        if dest is not None and dest is not argparse.SUPPRESS:
            keys.add(dest)
    return keys


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    # Restore mode uses the saved trainer_args as its base. Applying the YAML
    # overlay here would silently override them, so --config is ignored in
    # that mode; explicit individual CLI flags remain the override mechanism.
    restore_requested = str(getattr(args, "hf_resume", "none") or "none").strip() != "none"
    if args.config and restore_requested:
        print(
            "Warning: --config is ignored with --hf_resume; the restored "
            "run's trainer_args are used (individual CLI flags still override)."
        )
    elif args.config:
        if not _YAML_AVAILABLE:
            print("Warning: PyYAML not installed; ignoring --config. pip install pyyaml")
        else:
            _apply_yaml_config(parser, args, args.config)
    # The YAML overlay bypasses argparse's choices; reject a typo here so a
    # misspelled telephony.mode cannot silently train a different mode.
    if args.telephony_mode not in ("paired", "disjoint", "clean"):
        parser.error(
            "telephony_mode must be one of paired, disjoint, clean "
            f"(got {args.telephony_mode!r})."
        )
    if bool(getattr(args, "fresh_start", False)) and restore_requested:
        parser.error("--fresh_start cannot be combined with --hf_resume.")
    # load_best_model_at_end needs save_steps to be a multiple of eval_steps; the
    # Trainer only checks this after data prep and model load.
    if args.save_steps % args.eval_steps != 0:
        parser.error(f"save_steps ({args.save_steps}) must be a multiple of eval_steps ({args.eval_steps}).")
    if args.preflight_samples is not None and not (0 <= args.preflight_samples <= 10000):
        parser.error("--preflight_samples must be between 0 and 10000 (0 = full test).")
    if not args.gradient_checkpointing and args.batch_size > 8:
        print("Warning: batch_size > 8 without gradient checkpointing may run out of VRAM on a 48 GB GPU.")
    args._explicit_keys = _explicit_cli_keys(parser, sys.argv[1:])
    return args


def _apply_yaml_config(parser: argparse.ArgumentParser, args: argparse.Namespace, path: str):
    """Overlay YAML file onto parsed args. Only overrides defaults, not explicit CLI flags."""
    import sys as _sys
    cli_flags = {a.split("=")[0] for a in _sys.argv[1:] if a.startswith("--")}
    # Map --no-xxx back to base flag so explicit disables are not clobbered.
    normalized_cli = set()
    for f in cli_flags:
        if f.startswith("--no-"):
            normalized_cli.add("--" + f[5:].replace("-", "_"))
        else:
            normalized_cli.add(f)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to load config {path}: {e}")
        return
    flat: Dict[str, Any] = {}
    for section in ("model", "lora", "training", "benchmark", "telephony"):
        sub = cfg.get(section, {})
        if isinstance(sub, dict):
            flat.update(sub)
    # Flat top-level keys also allowed.
    for k, v in cfg.items():
        if not isinstance(v, dict):
            flat[k] = v
    key_map = {
        "base_model": "base_model", "dataset_name": "dataset_name",
        "output_dir": "output_dir", "r": "lora_r", "lora_r": "lora_r",
        "alpha": "lora_alpha", "lora_alpha": "lora_alpha",
        "dropout": "lora_dropout", "lora_dropout": "lora_dropout",
        "batch_size": "batch_size", "grad_accum": "grad_accum",
        "learning_rate": "learning_rate", "epochs": "epochs",
        "num_train_epochs": "epochs", "warmup_steps": "warmup_steps",
        "lr_scheduler_type": "lr_scheduler_type",
        "eval_steps": "eval_steps", "save_steps": "save_steps",
        "save_total_limit": "save_total_limit", "num_workers": "num_proc",
        "num_proc": "num_proc", "eval_batch_size": "eval_batch_size",
        "max_benchmark_samples": "max_benchmark_samples",
        "preflight_samples": "preflight_samples",
        "overwrite_preflight": "overwrite_preflight",
        "overwrite-preflight": "overwrite_preflight",
        "prob": "telephony_prob", "telephony_prob": "telephony_prob",
        "mode": "telephony_mode", "telephony_mode": "telephony_mode",
        "profiles": "telephony_profiles", "telephony_profiles": "telephony_profiles",
        "seed": "telephony_seed", "telephony_seed": "telephony_seed",
        "use_ffmpeg": "telephony_use_ffmpeg", "telephony_use_ffmpeg": "telephony_use_ffmpeg",
        "phone_benchmark": "phone_benchmark", "phone_seed": "phone_seed",
        "resume_from_checkpoint": "resume_from_checkpoint",
        "auto_resume": "auto_resume", "auto-resume": "auto_resume",
        "fresh_start": "fresh_start", "fresh-start": "fresh_start",
        "results_dir": "results_dir", "results-dir": "results_dir",
        "prep_cache_dir": "prep_cache_dir", "prep-cache-dir": "prep_cache_dir",
        "reuse_prep": "reuse_prep", "reuse-prep": "reuse_prep",
        "overwrite_prep_cache": "overwrite_prep_cache",
        "overwrite-prep-cache": "overwrite_prep_cache",
        "gradient_checkpointing": "gradient_checkpointing",
        "gradient-checkpointing": "gradient_checkpointing",
    }
    _BOOL_KEYS = {"auto_resume", "fresh_start", "phone_benchmark", "reuse_prep", "overwrite_prep_cache", "overwrite_preflight", "gradient_checkpointing"}
    for yk, ak in key_map.items():
        if yk in flat:
            cli_name = "--" + ak
            if cli_name not in normalized_cli and ("--" + ak.replace("_", "-")) not in normalized_cli:
                if ak == "telephony_profiles":
                    # Keep list from YAML as-is; CLI string normalized later.
                    setattr(args, ak, flat[yk])
                    continue
                try:
                    if ak in _BOOL_KEYS:
                        v = flat[yk]
                        if isinstance(v, str):
                            setattr(args, ak, v.strip().lower() in ("1", "true", "yes", "y", "on"))
                        else:
                            setattr(args, ak, bool(v))
                    else:
                        setattr(args, ak, type(getattr(args, ak))(flat[yk]) if getattr(args, ak) is not None else flat[yk])
                except Exception:
                    setattr(args, ak, flat[yk])
    # Target modules are code-fixed; warn if config disagrees.
    if isinstance(cfg.get("lora", {}).get("target_modules"), list):
        print(f"Note: lora.target_modules from {path} ignored (code-fixed).")


def _parse_telephony_profiles(value) -> list:
    if value is None:
        return ["customer"]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _default_prep_cache_dir(output_dir: str) -> str:
    """Prepared-feature cache lives on the big disk (see src/cache.py), NOT
    next to checkpoints: the 85G+ Arrow save does not fit the small
    /workspace volume alongside checkpoints (OSError 28). Explicit
    --prep_cache_dir / ASR_PREP_CACHE_DIR still wins; point it at
    <output_dir>/prep_cache only if that volume has room."""
    try:
        from src.cache import default_prep_cache_dir as _big_prep

        return _big_prep()
    except Exception:
        pass
    try:
        return os.path.join(output_dir, "prep_cache")
    except Exception:
        return "prep_cache"


def _label_format(base_model: str):
    """(prefix_ids, eos_id) for training labels, or (None, None) for tokenizer labels."""
    if base_model != "BuzzASR/persian":
        return None, None
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(base_model, "generation_config.json"), encoding="utf-8") as f:
        return persian_label_ids(json.load(f))


def _effective_train_prob(mode: str, telephony_prob) -> float:
    """Phone-on-the-fly probability the train pipeline actually uses.

    Only ``disjoint`` draws phone augmentation per row. ``paired`` materializes
    both halves up front and ``clean`` trains without augmentation, so both
    (and any unknown mode) use 0.0.
    """
    if mode == "disjoint":
        return float(telephony_prob or 0.0)
    return 0.0


def _prep_fingerprint(args, split: str, max_samples, telephony_mode: str, telephony_prob: float) -> Dict[str, Any]:
    """Fingerprint for a prepared split. Any change forces re-prep."""
    try:
        profiles = list(getattr(args, "telephony_profiles", []) or [])
    except Exception:
        profiles = []
    prefix_ids, eos_id = _label_format(getattr(args, "base_model", ""))
    return {
        "dataset_name": getattr(args, "dataset_name", ""),
        "base_model": getattr(args, "base_model", ""),
        "split": split,
        "max_samples": max_samples,
        "telephony_mode": telephony_mode,
        "telephony_prob": None if telephony_mode == "paired" else float(telephony_prob or 0.0),
        "telephony_profiles": sorted(str(p) for p in profiles),
        "telephony_seed": int(getattr(args, "telephony_seed", 42) or 42),
        "telephony_use_ffmpeg": bool(getattr(args, "telephony_use_ffmpeg", True)),
        # Bump when augmentation draws change; forces re-prep of stale caches.
        "telephony_rng": "per_row_index_v1",
        "label_ids": None if prefix_ids is None else [*prefix_ids, eos_id],
    }


def _resolve_output_dir(raw: str) -> str:
    """Managed resolver (see src/paths.py). Kept as wrapper for backward compat."""
    return _managed_paths.resolve_output_dir(raw)


def _json_safe(value):
    """Return an argparse value that ``json.dump`` can serialize."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _copy_preflight_to_run(run_dir: Path, cache_dir: Path) -> Path:
    """Copy the shared preflight cache into the run for provenance."""
    return run_state.write_preflight_copy(
        Path(run_dir), preflight_path(Path(cache_dir))
    )


_DETAIL_UNSET = object()


def _mark_run_stage(
    run_dir: Optional[str | Path],
    name: str,
    status: str,
    detail: Any = _DETAIL_UNSET,
    preserve_started: bool = False,
) -> Optional[Dict[str, Any]]:
    """Record one stage transition, or no-op for a plain CLI run."""
    if not run_dir:
        return None
    if detail is _DETAIL_UNSET:
        return run_state.set_stage(
            run_dir, name, status, preserve_started=preserve_started
        )
    return run_state.set_stage(
        run_dir, name, status, detail=detail, preserve_started=preserve_started
    )


def _mark_run_status(
    run_dir: Optional[str | Path], status: str, **fields: Any
) -> Optional[Dict[str, Any]]:
    """Record the overall run status, or no-op for a plain CLI run."""
    if not run_dir:
        return None
    return run_state.set_status(run_dir, status, **fields)


def _mark_preflight_failed(run_dir: Optional[str | Path], exc: BaseException) -> str:
    """Record a baseline-measurement failure on the run ledger."""
    message = f"baseline measurement failed: {exc}"
    _mark_run_stage(run_dir, "preflight", "failed", detail=message)
    _mark_run_status(
        run_dir, "failed", failed_stage="preflight", last_error=str(exc)
    )
    return message


def _record_trainer_args(run_dir, args) -> Optional[Dict[str, Any]]:
    """Persist the full effective argparse namespace on ``run.json``."""
    if not run_dir:
        return None
    raw = dict(args) if isinstance(args, Mapping) else vars(args)
    trainer_args = {str(key): _json_safe(value) for key, value in raw.items()}
    return run_state.update_run(
        run_dir, lambda run: {**run, "trainer_args": trainer_args}
    )


def _halt_run(run_dir, results_dir: str, stage: str) -> None:
    """Mark a cooperative halt: the active stage stops and later stages skip."""
    if stage not in run_state.STAGES:
        raise ValueError(f"invalid halt stage: {stage!r}")
    index = run_state.STAGES.index(stage)
    for later in run_state.STAGES[index + 1:]:
        _mark_run_stage(run_dir, later, "skipped")
    # Mark the halted stage last so current_stage points at what was stopped.
    _mark_run_stage(run_dir, stage, "stopped", detail="halt requested")
    _mark_run_status(run_dir, "stopped", failed_stage=None)
    _write_telemetry_status(results_dir, "stopped", run_dir=run_dir)
    try:
        os.remove(os.path.join(results_dir, "stop_training.signal"))
    except OSError:
        pass


def _run_attempt_number(run_dir: Optional[str | Path]) -> int:
    """Return the ledger attempt this trainer process belongs to (1 if unknown)."""
    if not run_dir:
        return 1
    run = run_state.load_run(run_dir)
    attempts = (run or {}).get("attempts") or []
    if not attempts:
        return 1
    try:
        return int(attempts[-1].get("attempt") or 1)
    except (TypeError, ValueError, AttributeError):
        return 1


def _is_cockpit_spawned() -> bool:
    """True when the server already owns the log/PID files for this process."""
    return (os.environ.get("ASR_COCKPIT_SPAWNED", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _TeeStream:
    """Write to the original stream and a run log at the same time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self._streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


@contextmanager
def _tee_run_log(path: Path):
    """Mirror stdout/stderr to *path* while keeping console output."""
    handle = Path(path).open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, handle)
    sys.stderr = _TeeStream(old_err, handle)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        try:
            handle.close()
        except Exception:
            pass


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _write_telemetry_status(
    results_dir: Optional[str | Path],
    status: str,
    *,
    run_dir: Optional[str | Path] = None,
    **fields: Any,
) -> None:
    """Update ``live_telemetry.json`` without discarding existing metrics."""
    if not results_dir:
        return
    path = Path(results_dir) / "live_telemetry.json"
    payload: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            payload = existing
    except (OSError, ValueError):
        payload = {}
    if run_dir:
        run = run_state.load_run(run_dir) or {}
        run_step = int(run.get("last_step") or 0)
        if payload.get("step") in (None, 0) and run_step:
            payload["step"] = run_step
        if payload.get("max_steps") in (None, 0) and run_step:
            payload["max_steps"] = run_step
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = time.time()
    try:
        _write_json_atomic(path, payload)
    except Exception as exc:
        print(f"Warning: failed to write telemetry status {status}: {exc}")


def _cli_run_config(args) -> dict:
    keys = (
        "base_model",
        "dataset_name",
        "epochs",
        "batch_size",
        "grad_accum",
        "learning_rate",
        "eval_batch_size",
        "eval_steps",
        "save_steps",
        "max_benchmark_samples",
        "preflight_samples",
        "num_proc",
        "telephony_mode",
        "telephony_prob",
        "telephony_profiles",
        "telephony_seed",
        "telephony_use_ffmpeg",
        "phone_benchmark",
        "phone_seed",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "warmup_steps",
        "lr_scheduler_type",
        "save_total_limit",
        "resume_from_checkpoint",
        "gradient_checkpointing",
        "fresh_start",
        "auto_resume",
        "output_dir",
        "results_dir",
        "run_post_test_eval",
        "run_epoch_benchmark",
        "enable_telemetry",
        "hf_run_repo",
        "hf_private",
        "hf_adapter_snapshot_steps",
        "hf_auto_resume",
    )
    return {key: _json_safe(getattr(args, key, None)) for key in keys}


def _validate_cli_run_dir(run_dir: Path) -> None:
    """Reject a ``--run_dir`` outside the canonical run-directory contract.

    A CLI ledger run must live directly under the resolved runs root and carry
    a canonical run id, so the cockpit can list, resume, and merge it like any
    other run.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    runs_root = Path(_managed_paths.resolve_runs_dir()).expanduser().resolve()
    if not run_state.run_id_valid(run_dir.name) or run_dir.parent != runs_root:
        print(
            "error: --run_dir must be a canonical run directory directly under "
            f"the resolved runs root ({runs_root}), e.g. "
            "<runs-root>/YYYYMMDD-HHMMSS-xxxxxx. Set ASR_RUNS_DIR to relocate "
            f"the runs root. Got: {run_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _maybe_create_hf_run_dir(args) -> Optional[Path]:
    """Create a canonical run directory when HF mirroring is enabled.

    The remote layout is keyed by run id, so a mirrored run needs a ledger
    directory even when the caller did not pass ``--run_dir``. Allocating the
    run here (before main's run-dir handling) makes output/results defaults
    run-scoped exactly like an explicit CLI run.
    """
    repo = getattr(args, "hf_run_repo", None) or os.environ.get("ASR_HF_RUN_REPO")
    if not str(repo or "").strip():
        return None
    if bool(getattr(args, "post_eval_only", False)):
        return None
    if getattr(args, "run_dir", None):
        return None
    runs_root = Path(_managed_paths.resolve_runs_dir()).expanduser().resolve()
    run_dir = runs_root / run_state.new_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    return run_dir


def _run_hf_repo_id(run: Any) -> str:
    """Return the repo id recorded on a run manifest, if any."""
    if not isinstance(run, Mapping):
        return ""
    hf = run.get("hf")
    if not isinstance(hf, Mapping):
        return ""
    return str(hf.get("repo_id") or "").strip()


def _build_hf_mirror(args, run_dir, run=None):
    """Build and start the run mirror, or return ``None`` when it is off.

    The feature is off unless ``--hf_run_repo`` or ``ASR_HF_RUN_REPO`` is set,
    so no Hugging Face import or network call happens in the default path.
    Mirror failures are logged and never propagate into training.
    """
    if run_dir is None:
        return None
    try:
        config = hf_sync.resolve_config(args)
    except Exception as exc:
        print(f"[hf-sync] disabled: invalid configuration: {exc}")
        return None
    if config is None:
        return None
    try:
        if run is None:
            run = run_state.load_run(run_dir)
        recorded_repo = _run_hf_repo_id(run)
        if recorded_repo:
            if recorded_repo != config.repo_id:
                print(
                    "[hf-sync] using the repo recorded in run.json "
                    f"({recorded_repo}); ignoring {config.repo_id}"
                )
            config = replace(config, repo_id=recorded_repo)
        mirror = hf_sync.HFRunMirror(
            config,
            run_id=Path(run_dir).name,
            run_dir=Path(run_dir),
            output_dir=Path(args.output_dir),
            results_dir=Path(args.results_dir),
            manifest_update=lambda mutate: run_state.update_run(run_dir, mutate),
        )
        mirror.start()
    except Exception as exc:
        print(f"[hf-sync] disabled: {exc}")
        return None
    # A pod restart may have lost the local checkpoint volume while the ledger
    # survived. Pull the newest remote checkpoint before checkpoint selection.
    if bool(getattr(args, "hf_auto_resume", True)) and not bool(
        getattr(args, "fresh_start", False)
    ):
        try:
            mirror.auto_repair()
        except Exception as exc:
            print(f"[hf-sync] auto repair failed: {exc}")
    return mirror


def _local_checkpoint_step(output_dir) -> Optional[int]:
    """Return the highest complete local checkpoint step, if any."""
    best: Optional[int] = None
    try:
        entries = list(Path(output_dir).glob("checkpoint-*"))
    except OSError:
        return None
    for path in entries:
        match = hf_sync.CHECKPOINT_RE.fullmatch(path.name)
        if match is None or not hf_sync.is_complete_checkpoint(path):
            continue
        step = int(match.group(1))
        best = step if best is None else max(best, step)
    return best


def _restore_hf_run(args) -> bool:
    """Handle ``--hf_resume``; return True when main should stop.

    The completed-run case returns True so ``main`` exits 0 before creating an
    attempt, changing status, or entering training. Runs without a complete
    checkpoint are restored (ledger-only) and then rejected with exit code 2.
    """
    requested = str(getattr(args, "hf_resume", "none") or "none").strip()
    if requested == "none":
        return False
    try:
        config = hf_sync.resolve_config(args)
    except Exception as exc:
        print(f"error: invalid HF configuration: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if config is None:
        print(
            "error: --hf_resume requires --hf_run_repo or ASR_HF_RUN_REPO.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        api = hf_sync.make_api()
        run_id = hf_sync.resolve_resume_run(api, config.repo_id, requested)
    except Exception as exc:
        print(
            f"error: could not resolve HF run {requested!r}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    runs_root = Path(_managed_paths.resolve_runs_dir()).expanduser().resolve()
    run_dir = runs_root / run_id
    args.run_dir = str(run_dir)

    local_run = run_state.load_run(run_dir)
    checkpoint_step = (
        _local_checkpoint_step(run_dir / "output") if local_run is not None else None
    )
    if local_run is None or checkpoint_step is None:
        # restore_run stages and validates the whole run before replacing any
        # existing local copy, so force is safe for a partial local ledger.
        try:
            restored = hf_sync.restore_run(
                api,
                config.repo_id,
                run_id,
                runs_root,
                force=local_run is not None,
            )
        except Exception as exc:
            print(
                f"error: could not restore run {run_id}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raw_step = restored.get("checkpoint_step")
        checkpoint_step = int(raw_step) if raw_step is not None else None

    run = run_state.load_run(run_dir)
    if run is None:
        print(
            f"error: restored run ledger not found at {run_dir / 'run.json'}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    hf_sync.apply_restored_args(
        args,
        run,
        explicit=getattr(args, "_explicit_keys", set()),
        run_dir=run_dir,
    )
    args.telephony_profiles = _parse_telephony_profiles(
        getattr(args, "telephony_profiles", "customer")
    )
    if str(run.get("status") or "") == "completed":
        print(f"Restored completed run: {run_dir}")
        return True
    if checkpoint_step is None:
        print(
            f"error: run {run_id} has no complete checkpoint to resume from; "
            f"it was restored to {run_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(
        f"Restored {run_id} at checkpoint-{checkpoint_step}; resuming from {run_dir}"
    )
    return False


def _ensure_cli_run(run_dir: Path, args) -> dict:
    """Create the ledger for a CLI ``--run_dir`` when it does not exist yet."""
    run = run_state.load_run(run_dir)
    if run is not None:
        return run
    output_dir = Path(args.output_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    raw_cache = getattr(args, "preflight_cache_dir", None)
    cache_dir = Path(raw_cache).resolve() if raw_cache else results_dir
    paths = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "merged_dir": str(run_dir / "merged"),
        "preflight_cache": str(cache_dir / "baselines" / "preflight.json"),
    }
    trainer_args = {str(key): _json_safe(value) for key, value in vars(args).items()}
    config = _cli_run_config(args)
    # main() validates the canonical contract first; create_run enforces it.
    return run_state.create_run(
        run_dir.parent,
        run_id=run_dir.name,
        config=config,
        trainer_args=trainer_args,
        paths=paths,
        source="cli",
    )


def _ensure_cli_attempt(run_dir: Path, run: dict) -> None:
    """Record this process as an attempt unless the server already did."""
    pid = os.getpid()
    attempts = (run or {}).get("attempts") or []
    open_attempts = [item for item in attempts if item.get("ended_at") is None]
    if open_attempts:
        try:
            same_pid = int(open_attempts[-1].get("pid") or 0) == pid
        except (TypeError, ValueError):
            same_pid = False
        if same_pid:
            if int((run or {}).get("current_pid") or 0) != pid:
                run_state.update_run(
                    run_dir,
                    lambda current: {**current, "current_pid": pid},
                )
            return
        # A dead earlier attempt must not stay open forever.
        run_state.finalize_attempt(
            run_dir, exit_code=None, reason="process_lost"
        )
    run_state.start_attempt(
        run_dir, pid, reason="initial" if not attempts else "resume"
    )


def _write_cli_pid(run_dir: Path) -> None:
    try:
        (run_dir / "training.pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write {run_dir / 'training.pid'}: {exc}")


def _abort_cli_setup(run_dir: Path) -> None:
    """Close any attempt opened during CLI setup and drop the PID file."""
    try:
        run = run_state.load_run(run_dir)
    except Exception as exc:
        # A corrupt/unreadable ledger must not mask the original setup error.
        print(f"Warning: could not read run ledger: {exc}", file=sys.stderr)
        run = None
    if run is not None:
        try:
            run_state.finalize_attempt(run_dir, exit_code=1, reason="nonzero_exit")
        except Exception as exc:
            print(f"Warning: could not finalize run attempt: {exc}", file=sys.stderr)
    try:
        (Path(run_dir) / "training.pid").unlink(missing_ok=True)
    except OSError:
        pass


def _finalize_cli_run(run_dir: Path, exit_code: int) -> None:
    """Close the CLI attempt and remove its PID file on every exit path."""
    try:
        run = run_state.load_run(run_dir)
        if run is not None:
            if run.get("stop_requested_at") or run.get("status") == "stopped":
                reason = "halted"
            elif exit_code == 0:
                reason = "completed"
            else:
                reason = "nonzero_exit"
            run_state.finalize_attempt(
                run_dir, exit_code=exit_code, reason=reason
            )
    except Exception as exc:
        print(f"Warning: could not finalize run attempt: {exc}", file=sys.stderr)
    try:
        (run_dir / "training.pid").unlink(missing_ok=True)
    except OSError:
        pass


def _clear_previous_state(output_dir: str, results_dir: Optional[str] = None) -> None:
    """Delete prior run state so --fresh_start truly starts clean.

    Removes only known training artifacts inside output_dir
    (checkpoint-*, best_adapter, final reports, runs/) and the live
    telemetry history in results_dir (loss_history, epoch benchmarks are
    rebuilt with baselines by the callback). Never deletes output_dir itself.
    """
    import shutil
    removed = []
    rdir = Path(_resolve_results_dir(results_dir))
    try:
        base = Path(output_dir)
        if base.is_dir():
            for ckpt in sorted(base.glob("checkpoint-*")):
                try:
                    if ckpt.is_dir():
                        shutil.rmtree(ckpt, ignore_errors=True)
                        removed.append(ckpt.name)
                except Exception:
                    continue
            for name in ("best_adapter", "runs"):
                try:
                    t = base / name
                    if t.is_dir():
                        shutil.rmtree(t, ignore_errors=True)
                        removed.append(name)
                except Exception:
                    continue
            for name in ("final_evaluation_report.json", "final_evaluation_report.md", "trainer_state.json"):
                try:
                    t = base / name
                    if t.is_file():
                        t.unlink(missing_ok=True)
                        removed.append(name)
                except Exception:
                    continue
    except Exception as e:
        print(f"Warning: fresh-start output cleanup failed: {e}")
    try:
        for name in ("loss_history.json", "steps.jsonl", "events.jsonl"):
            # steps/events belong to the run ledger; a fresh CLI run must not
            # append to an old run's history. NOTE: do NOT delete
            # live_telemetry.json / training_live.log here. The web server
            # creates both for the *current* run just before spawning this
            # trainer; deleting them orphans the server's open log fd (stdout
            # goes to an unlinked inode) so /api/logs reports "Log buffer
            # empty" while training is actually running.
            try:
                t = rdir / name
                if t.is_file():
                    t.unlink(missing_ok=True)
                    removed.append(f"results/{name}")
            except Exception:
                continue
        # Reset epoch benchmarks to baselines-only (callback rebuilds if missing).
        try:
            bfile = rdir / "epoch_benchmarks.json"
            if bfile.is_file():
                bfile.unlink(missing_ok=True)
                removed.append("results/epoch_benchmarks.json")
        except Exception:
            pass
        # Clear stale stop signal so the fresh run is not halted immediately.
        try:
            sfile = rdir / "stop_training.signal"
            if sfile.is_file():
                sfile.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: fresh-start results cleanup failed: {e}")
    if removed:
        print(f"Fresh start: cleared {', '.join(removed)}")


def _find_last_checkpoint(output_dir: str) -> Optional[str]:
    """Managed lookup (see src/paths.find_last_checkpoint). Wrapper for backward compat."""
    return _managed_paths.find_last_checkpoint(output_dir)


def _resolve_resume_checkpoint(args) -> Optional[bool | str]:
    """Decide what to pass as trainer.train(resume_from_checkpoint=...).

    Returns None (fresh), True (let Trainer pick latest), or a path.
    Explicit --resume_from_checkpoint wins; otherwise --auto_resume picks up
    the latest checkpoint in --output_dir (which should live on the persistent
    volume, e.g. /workspace/stt-outputs/...). --fresh_start disables resume.
    """
    if bool(getattr(args, "fresh_start", False)):
        return None
    explicit = (getattr(args, "resume_from_checkpoint", None) or "").strip() if isinstance(getattr(args, "resume_from_checkpoint", None), str) else getattr(args, "resume_from_checkpoint", None)
    if explicit:
        low = str(explicit).strip().lower()
        if low in ("last", "true", "yes", "1", "auto"):
            last = _find_last_checkpoint(args.output_dir)
            if last:
                print(f"Resuming from latest checkpoint: {last}")
                return last
            print("No checkpoint found in output_dir; starting fresh.")
            return None
        p = Path(str(explicit))
        if p.is_dir():
            print(f"Resuming from explicit checkpoint: {p}")
            return str(p)
        print(f"Warning: --resume_from_checkpoint '{explicit}' not found; starting fresh.")
        return None
    if bool(getattr(args, "auto_resume", True)):
        last = _find_last_checkpoint(args.output_dir)
        if last:
            print(f"Auto-resume: found checkpoint {last}")
            return last
    return None


def _load_post_eval_trainer_state(output_dir: str):
    """Load ``TrainerState`` from the newest complete checkpoint when available."""
    checkpoint = _find_last_checkpoint(output_dir)
    if not checkpoint:
        return None
    state_path = Path(checkpoint) / "trainer_state.json"
    try:
        return TrainerState.load_from_json(str(state_path))
    except Exception as exc:
        print(f"Warning: could not load {state_path}: {exc}")
        return None


def _load_run_baseline(
    run_dir: Optional[Path], results_dir: str
) -> Optional[Dict[str, Any]]:
    """Return the measured preflight model rows used for report deltas."""
    candidates = []
    if run_dir is not None:
        candidates.append(Path(run_dir) / "baselines" / "preflight.json")
    candidates.append(Path(results_dir) / "baselines" / "preflight.json")
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        models = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(models, dict):
            return models
    return None


def _resolve_eval_device() -> str:
    """Device the post-training evaluation passes must run on."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_halt_requested(exc: BaseException) -> bool:
    """True when *exc* is the cooperative-halt signal from evaluation."""
    return isinstance(exc, RuntimeError) and str(exc) == "halt requested"


def _run_post_eval_stage(
    *,
    run_dir: Optional[str | Path],
    results_dir: Optional[str | Path],
    evaluate: Callable[[], None],
    detail: Optional[str] = None,
) -> bool:
    """Run one post-eval attempt, converting a halt into a stopped run.

    Returns ``True`` on success and ``False`` when the user halted. The
    post-eval stage is an active telemetry status from start to finish so the
    cockpit never reports a completed run while evaluation is still running;
    on success the stage also records the terminal ``completed`` snapshot so a
    plain CLI run never leaves the legacy ``training`` status behind.
    """
    _mark_run_stage(run_dir, "post_eval", "running", detail=detail)
    _write_telemetry_status(
        results_dir, "evaluating_benchmark", run_dir=run_dir
    )
    try:
        evaluate()
    except Exception as exc:
        if _is_halt_requested(exc):
            print("Stop signal received during post-eval; halting run.")
            _halt_run(run_dir, results_dir, "post_eval")
            return False
        _mark_run_stage(
            run_dir, "post_eval", "failed", detail=f"post-eval failed: {exc}"
        )
        _mark_run_status(
            run_dir, "failed", failed_stage="post_eval", last_error=str(exc)
        )
        _write_telemetry_status(results_dir, "failed", run_dir=run_dir)
        raise
    _mark_run_stage(run_dir, "post_eval", "completed", detail=None)
    _write_telemetry_status(results_dir, "completed", run_dir=run_dir)
    return True


def _run_post_eval_only(args) -> None:
    """Run only the full-test report from an existing ``best_adapter``.

    Used by cockpit Resume when training completed but post-eval did not. It
    never constructs a Trainer, restores an optimizer, or appends step records.
    """
    output_dir = Path(args.output_dir)
    raw_results = getattr(args, "results_dir", None)
    results_dir = _resolve_results_dir(str(raw_results) if raw_results else None)
    run_dir = Path(args.run_dir).resolve() if getattr(args, "run_dir", None) else None
    stop_signal = os.path.join(results_dir, "stop_training.signal")
    if os.path.exists(stop_signal):
        print("Stop signal found before post-eval; halting run.")
        _halt_run(run_dir, results_dir, "post_eval")
        return

    adapter_path = output_dir / "best_adapter"
    if not (adapter_path / "adapter_config.json").is_file():
        message = (
            "post-eval-only: best_adapter/adapter_config.json not found at "
            f"{adapter_path}"
        )
        _mark_run_stage(run_dir, "post_eval", "failed", detail=message)
        _mark_run_status(
            run_dir, "failed", failed_stage="post_eval", last_error=message
        )
        _write_telemetry_status(results_dir, "failed", run_dir=run_dir)
        raise SystemExit(message)

    _mark_run_status(run_dir, "running")

    def evaluate() -> None:
        print(f"Post-eval-only: loading processor from {adapter_path}...")
        processor = WhisperProcessor.from_pretrained(str(adapter_path))
        normalizer = PersianTextNormalizer()
        metrics_calculator = ASRMetricsCalculator(normalizer=normalizer)

        device = _resolve_eval_device()
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            dtype = torch.float32

        print(f"Post-eval-only: loading base model {args.base_model} in {dtype}...")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))

        trainer_state = _load_post_eval_trainer_state(str(output_dir))
        baseline_benchmark = _load_run_baseline(run_dir, results_dir)

        _run_post_training_evaluation(
            model=model,
            processor=processor,
            metrics_calculator=metrics_calculator,
            normalizer=normalizer,
            dataset_name=args.dataset_name,
            output_dir=str(output_dir),
            results_dir=results_dir,
            eval_batch_size=int(getattr(args, "eval_batch_size", 16) or 16),
            trainer_state=trainer_state,
            baseline_benchmark=baseline_benchmark,
            phone_benchmark=bool(getattr(args, "phone_benchmark", True)),
            phone_seed=int(getattr(args, "phone_seed", 1234) or 1234),
            telephony_profiles=_parse_telephony_profiles(
                getattr(args, "telephony_profiles", "customer")
            ),
            telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
            should_stop=lambda: os.path.exists(stop_signal),
        )

    if not _run_post_eval_stage(
        run_dir=run_dir,
        results_dir=results_dir,
        evaluate=evaluate,
        detail="post-eval only",
    ):
        return
    _mark_run_status(
        run_dir, "completed", failed_stage=None, last_error=None
    )


def main():
    args = parse_args()
    args.telephony_profiles = _parse_telephony_profiles(
        getattr(args, "telephony_profiles", "customer")
    )
    explicit_results = getattr(args, "results_dir", None) not in (None, "")
    restore_requested = str(getattr(args, "hf_resume", "none") or "none").strip() != "none"
    restore_mode = False
    if restore_requested:
        if _restore_hf_run(args):
            return 0
        restore_mode = True
    _maybe_create_hf_run_dir(args)
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if getattr(args, "run_dir", None)
        else None
    )
    if run_dir is not None:
        _validate_cli_run_dir(run_dir)
        # A CLI ledger run is self-contained: telemetry/benchmarks and the
        # default output live under the run unless explicitly overridden.
        if not explicit_results:
            args.results_dir = str(run_dir)
        if str(args.output_dir).strip() == _default_output_dir():
            args.output_dir = str(run_dir / "output")
    args.output_dir = _resolve_output_dir(args.output_dir)
    args.results_dir = _resolve_results_dir(getattr(args, "results_dir", None))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if not bool(getattr(args, "post_eval_only", False)):
        if bool(getattr(args, "fresh_start", False)):
            # Always honor --fresh_start, including a plain CLI run with no
            # ledger directory. A new run directory is already empty, so the
            # same clearing is harmless there.
            _clear_previous_state(args.output_dir, results_dir=args.results_dir)

    if run_dir is not None:
        try:
            run = _ensure_cli_run(run_dir, args)
            # Persist the full effective namespace before any stage work so Resume
            # can recreate the exact LoRA/scheduler/telephony settings. A
            # restored run already carries the saved trainer_args.
            if not restore_mode:
                _record_trainer_args(run_dir, args)
            _mark_run_status(run_dir, "running")
            # The server records the attempt before spawning its child; a standalone
            # CLI invocation owns its own attempt and PID file.
            if not _is_cockpit_spawned():
                _ensure_cli_attempt(run_dir, run)
            _write_cli_pid(run_dir)
        except Exception:
            # A setup failure after the attempt/PID exists must not leave an
            # open attempt behind.
            _abort_cli_setup(run_dir)
            raise

    exit_code = 0
    tee = (
        _tee_run_log(run_dir / "training_live.log")
        if run_dir is not None and not _is_cockpit_spawned()
        else nullcontext()
    )
    try:
        with tee:
            _run_training(args, run_dir)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        if run_dir is not None:
            _finalize_cli_run(run_dir, exit_code)


def _finish_hf_mirror(mirror, run_dir) -> None:
    """Upload the terminal ledger and stop the mirror without failing training."""
    if mirror is None:
        return
    try:
        run = run_state.load_run(run_dir) if run_dir else None
        if run is not None:
            mirror.enqueue_ledger(str(run.get("status") or "running"))
    except Exception as exc:
        print(f"[hf-sync] could not enqueue the final ledger: {exc}")
    try:
        mirror.close()
    except Exception as exc:
        print(f"[hf-sync] could not close the mirror: {exc}")


def _run_training(args, run_dir):
    mirror = _build_hf_mirror(args, run_dir)
    try:
        _run_training_stages(args, run_dir, mirror)
    finally:
        _finish_hf_mirror(mirror, run_dir)


def _run_training_stages(args, run_dir, mirror):
    # Post-eval-only resume: never repeat dataset prep, preflight, or training,
    # and never replay optimizer steps.
    if bool(getattr(args, "post_eval_only", False)):
        if bool(getattr(args, "fresh_start", False)):
            print("Note: --fresh_start is ignored with --post_eval_only.")
        _run_post_eval_only(args)
        if mirror is not None:
            mirror.enqueue_ledger("post_eval", reports=True)
        return

    stop_signal = os.path.join(args.results_dir, "stop_training.signal")
    if os.path.exists(stop_signal):
        print("Stop signal found before data preparation; halting run.")
        _halt_run(run_dir, args.results_dir, "preflight")
        return

    print("=======================================================")
    print(" Persian ASR Fine-Tuning: BuzzASR + Neyshekar LoRA")
    print(f" Base Model:      {args.base_model}")
    print(f" Dataset:         {args.dataset_name}")
    print(f" Output Dir:      {args.output_dir}")
    print(f" Results Dir:     {args.results_dir}")
    print(f" LoRA Rank:       r={args.lora_r}, alpha={args.lora_alpha}")
    print(f" Batch Size:      {args.batch_size} (accum: {args.grad_accum})")
    print(f" Telephony:       prob={getattr(args, 'telephony_prob', 0.0)} profiles={args.telephony_profiles} ffmpeg={getattr(args, 'telephony_use_ffmpeg', True)}")
    print(f" Phone Benchmark: {getattr(args, 'phone_benchmark', True)} (seed={getattr(args, 'phone_seed', 1234)})")
    print(f" Preprocess procs:{args.num_proc}")
    print("=======================================================\n")

    # The preflight stage owns pre-training setup (processor, dataset
    # preparation, cached feature loads). A failure here must be attributed to
    # this stage rather than leaving the run with no responsible stage.
    _mark_run_stage(run_dir, "preflight", "running", detail="preparing datasets")
    try:
        # 1. Load Processor (CPU-only; safe before fork).
        print(f"Loading processor for '{args.base_model}'...")
        processor = WhisperProcessor.from_pretrained(args.base_model)
        normalizer = PersianTextNormalizer()

        # 2. Prepare Data BEFORE any CUDA init / model load.
        # datasets.map(num_proc>1) forks workers. Forking after CUDA init or
        # with the multi-GB model resident deadlocks/bloats workers (observed: 4
        # forked workers stuck at 0% CPU, ~5GB RSS each, bar frozen at 0/58244).
        # Feature extraction + tokenization need only the processor, so prep
        # runs first (train gets phone augmentation, eval stays clean).
        label_prefix_ids, label_eos_id = _label_format(args.base_model)
        mode = getattr(args, "telephony_mode", "paired")
        if mode == "paired":
            print("Note: --telephony_prob is ignored in paired mode; every train row is prepared clean + phone.")
        effective_train_prob = _effective_train_prob(
            mode, getattr(args, "telephony_prob", 0.0)
        )
        train_pipeline = PersianASRDataPipeline(
            processor=processor,
            dataset_name=args.dataset_name,
            normalizer=normalizer,
            num_proc=args.num_proc,
            telephony_prob=effective_train_prob,
            telephony_profiles=args.telephony_profiles,
            telephony_seed=int(getattr(args, "telephony_seed", 42) or 42),
            telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
            label_prefix_ids=label_prefix_ids,
            label_eos_id=label_eos_id,
        )
        eval_pipeline = PersianASRDataPipeline(
            processor=processor,
            dataset_name=args.dataset_name,
            normalizer=normalizer,
            num_proc=args.num_proc,
            telephony_prob=0.0,
            label_prefix_ids=label_prefix_ids,
            label_eos_id=label_eos_id,
        )

        # HF cache was routed to the big disk at import time (see top of file,
        # src/cache.py). Re-log here since output_dir/results are set up by
        # now; checkpoints stay on /workspace (persistent), caches ephemeral.
        try:
            from src.cache import log_cache_space as _log_cache_space

            _log_cache_space("HF cache")
        except Exception:
            pass

        raw_train = None
        raw_eval = None
        raw_test = None
        train_dataset = None
        eval_dataset = None
        prep_cache_dir = (getattr(args, "prep_cache_dir", None) or "").strip() or _default_prep_cache_dir(args.output_dir)
        reuse_prep = bool(getattr(args, "reuse_prep", True))
        overwrite_prep = bool(getattr(args, "overwrite_prep_cache", False))
        print(f" Prep Cache Dir:  {prep_cache_dir} (reuse={reuse_prep}, overwrite={overwrite_prep})")
        try:
            os.makedirs(prep_cache_dir, exist_ok=True)
        except Exception:
            pass
        train_fp = _prep_fingerprint(args, "train", getattr(args, "max_train_samples", None), mode, effective_train_prob)
        eval_fp = _prep_fingerprint(args, "validation", getattr(args, "max_eval_samples", None), "clean", 0.0)
        train_cache = Path(prep_cache_dir) / "train"
        eval_cache = Path(prep_cache_dir) / "validation"
        load_cache = reuse_prep and not overwrite_prep
        if load_cache:
            train_dataset = prep_cache.load_prep_cache(train_cache, train_fp)
            eval_dataset = prep_cache.load_prep_cache(eval_cache, eval_fp)
        need_train = train_dataset is None
        need_eval = eval_dataset is None

        if need_train or need_eval or args.run_epoch_benchmark:
            raw_train = train_pipeline.load_raw_dataset(split="train", max_samples=args.max_train_samples) if need_train else None
            raw_eval = eval_pipeline.load_raw_dataset(split="validation", max_samples=args.max_eval_samples) if need_eval else None
            if args.run_epoch_benchmark:
                print("Loading full test benchmark split for end-of-epoch evaluation...")
                raw_test = eval_pipeline.load_raw_dataset(split="test")

        if need_train:
            assert raw_train is not None
            prep_cache.cleanup_stale_staging(train_cache)
            required = int(
                prep_cache.estimate_feature_bytes(
                    len(raw_train), copies=2 if mode == "paired" else 1
                )
                * 1.05
            )
            reclaimable = prep_cache.dir_size(train_cache)
            # Guard before removing the old cache: the projection counts its space.
            # Pass the cache slot so ensure_space measures the cache root that
            # actually holds the shards (its parent), not the root's parent.
            prep_cache.ensure_space(train_cache, required, reclaimable)
            prep_cache.remove_cache(train_cache)
            staging = prep_cache.staging_dir(train_cache)
            try:
                if mode == "paired":
                    train_dataset = train_pipeline.prepare_paired_dataset(
                        raw_train, cache_dir=str(staging)
                    )
                    print(f"Paired train set: {len(train_dataset) // 2} clean + "
                          f"{len(train_dataset) // 2} phone = {len(train_dataset)} rows")
                else:
                    train_dataset = train_pipeline.prepare_dataset_split(
                        raw_train, cache_dir=str(staging)
                    )
                    print(f"Train set: {len(train_dataset)} rows (mode={mode})")
                prep_cache.publish_dataset(train_dataset, staging, train_cache, train_fp)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            train_dataset = prep_cache.load_prep_cache(train_cache, train_fp)
            if train_dataset is None:
                raise RuntimeError("prep cache publish/load failed")
        if need_eval:
            assert raw_eval is not None
            prep_cache.cleanup_stale_staging(eval_cache)
            required = int(
                prep_cache.estimate_feature_bytes(len(raw_eval), copies=1) * 1.05
            )
            reclaimable = prep_cache.dir_size(eval_cache)
            prep_cache.ensure_space(eval_cache, required, reclaimable)
            prep_cache.remove_cache(eval_cache)
            staging = prep_cache.staging_dir(eval_cache)
            try:
                eval_dataset = eval_pipeline.prepare_dataset_split(
                    raw_eval, cache_dir=str(staging)
                )
                prep_cache.publish_dataset(eval_dataset, staging, eval_cache, eval_fp)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            eval_dataset = prep_cache.load_prep_cache(eval_cache, eval_fp)
            if eval_dataset is None:
                raise RuntimeError("prep cache publish/load failed")
        if train_dataset is not None:
            print(f"Train set ready: mode={mode}, {len(train_dataset)} rows")
        assert train_dataset is not None and eval_dataset is not None
    except Exception as exc:
        print(f"Preflight data preparation failed: {exc}", file=sys.stderr)
        _mark_run_stage(run_dir, "preflight", "failed",
                        detail=f"data preparation failed: {exc}")
        _mark_run_status(run_dir, "failed", failed_stage="preflight",
                         last_error=str(exc))
        raise

    # Mandatory measured baseline: reuse a matching cache or evaluate BuzzASR
    # (and Whisper) on exactly the same rows the epoch benchmark will use.
    # Runs after datasets.map prep and before the model/CUDA load so the
    # subprocess evaluation sees an idle GPU.
    _mark_run_stage(
        run_dir, "preflight", "running",
        detail="measuring baseline", preserve_started=True,
    )
    preflight_count = args.preflight_samples
    if preflight_count is None:
        preflight_count = args.max_benchmark_samples
    cache_dir = (
        Path(args.preflight_cache_dir)
        if getattr(args, "preflight_cache_dir", None)
        else Path(args.results_dir)
    )
    try:
        preflight = run_preflight(
            results_dir=Path(args.results_dir),
            cache_dir=cache_dir,
            dataset_name=args.dataset_name,
            split="test",
            sample_count=int(preflight_count or 0),
            phone_seed=int(args.phone_seed),
            telephony_profiles=args.telephony_profiles,
            telephony_use_ffmpeg=bool(args.telephony_use_ffmpeg),
            eval_batch_size=int(args.eval_batch_size),
            overwrite=bool(args.overwrite_preflight),
            should_stop=lambda: os.path.exists(stop_signal),
        )
        if run_dir is not None:
            _copy_preflight_to_run(run_dir, cache_dir)
    except RuntimeError as exc:
        if str(exc) == "halt requested":
            print("Stop signal received during preflight; halting run.")
            _halt_run(run_dir, args.results_dir, "preflight")
            return
        _mark_preflight_failed(run_dir, exc)
        raise
    except Exception as exc:
        _mark_preflight_failed(run_dir, exc)
        raise

    baseline_models = preflight.get("models") or {}
    measured_samples = int(
        (baseline_models.get("buzzasr_base") or {}).get("total_samples")
        or preflight_count
        or 0
    )
    _mark_run_stage(
        run_dir,
        "preflight",
        "completed",
        detail=("cache hit" if preflight.get("cached") else f"measured, n={measured_samples}"),
    )
    if mirror is not None:
        mirror.enqueue_ledger("preflight")
    baseline_benchmark = baseline_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Release raw Arrow tables before the multi-GB model load.
    del raw_train, raw_eval
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    # 3. CUDA info + base model load (first CUDA touch happens here, after all
    # forking work is done).
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f" BF16 Supported:  {torch.cuda.is_bf16_supported()}")
    # 4. Load Base Model in BF16 (fallback to fp32 on CPU)
    try:
        _bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        _bf16_ok = False
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if _bf16_ok else torch.float16
    else:
        dtype = torch.float32
    print(f"Loading base model in {dtype}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # Disable cache when training with gradient checkpointing
    model.config.use_cache = False

    # 4. Configure LoRA
    print("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. Evaluation Metric Function & Callbacks
    wer_metric = evaluate.load("wer")
    metrics_calculator = ASRMetricsCalculator(normalizer=normalizer)

    callbacks = []
    telemetry_cb = None
    if args.enable_telemetry:
        telemetry_cb = WebTelemetryCallback(
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            run_dir=str(run_dir) if run_dir is not None else None,
            attempt=_run_attempt_number(run_dir),
            test_dataset=raw_test,
            processor=processor,
            metrics_calculator=metrics_calculator,
            normalizer=normalizer,
            eval_batch_size=args.eval_batch_size,
            max_benchmark_samples=args.max_benchmark_samples,
            baseline_benchmark=baseline_benchmark,
            phone_benchmark=bool(getattr(args, "phone_benchmark", True)),
            phone_seed=int(getattr(args, "phone_seed", 1234) or 1234),
            telephony_profiles=args.telephony_profiles,
            telephony_use_ffmpeg=bool(getattr(args, "telephony_use_ffmpeg", True)),
        )
        callbacks.append(telemetry_cb)
    if mirror is not None:
        callbacks.append(
            HFMirrorCallback(
                mirror,
                model=model,
                processor=processor,
                output_dir=args.output_dir,
                snapshot_steps=int(
                    getattr(args, "hf_adapter_snapshot_steps", 250)
                ),
                save_steps=int(getattr(args, "save_steps", 1000)),
            )
        )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize with Persian text normalizer
        norm_preds = [normalizer.normalize(p, remove_punctuation=True, zwnj_mode="keep") for p in pred_str]
        norm_labels = [normalizer.normalize(l, remove_punctuation=True, zwnj_mode="keep") for l in label_str]

        wer = 100 * wer_metric.compute(predictions=norm_preds, references=norm_labels)
        return {"wer": wer}

    # 7. Training Arguments (Configured for L40S)
    try:
        use_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    import inspect as _inspect
    _all_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16 and torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_proc,
        remove_unused_columns=False,
    )
    try:
        _sig_params = _inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        # transformers>=5 removed overwrite_output_dir; v4 still needs it.
        if "overwrite_output_dir" in _sig_params:
            _all_kwargs["overwrite_output_dir"] = True
        training_args = Seq2SeqTrainingArguments(
            **{k: v for k, v in _all_kwargs.items() if k in _sig_params}
        )
    except TypeError:
        # Fallback for very old transformers with evaluation_strategy naming.
        if "eval_strategy" in _all_kwargs:
            _all_kwargs["evaluation_strategy"] = _all_kwargs.pop("eval_strategy")
        training_args = Seq2SeqTrainingArguments(**_all_kwargs)

    # 8. Initialize Trainer (transformers>=4.46 uses processing_class)
    _trainer_common = dict(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    # No processing_class: Trainer.__init__ would align model eos/pad/bos to the
    # tokenizer's <|endoftext|> (id 0), and generation would stop on the wrong token.
    # The processor is saved explicitly after training.
    try:
        trainer = Seq2SeqTrainer(processing_class=None, **_trainer_common)
    except TypeError:
        # Older transformers: no processing_class kwarg.
        trainer = Seq2SeqTrainer(**_trainer_common)

    # 9. Execute Training (auto-resumable from persistent-volume checkpoints)
    resume_ckpt = _resolve_resume_checkpoint(args)
    if resume_ckpt:
        print(f"\nResuming training from: {resume_ckpt}")
    else:
        print("\nStarting training loop (fresh)...")
    _mark_run_stage(run_dir, "training", "running")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except Exception as exc:
        _mark_run_stage(run_dir, "training", "failed",
                        detail=f"training failed: {exc}")
        _mark_run_status(run_dir, "failed", failed_stage="training",
                         last_error=str(exc))
        raise

    halted = bool(telemetry_cb is not None and getattr(telemetry_cb, "halt_requested", False))
    if halted:
        print("\nTraining halted by user request; skipping post-training evaluation.")
        _halt_run(run_dir, args.results_dir, "training")
    else:
        _mark_run_stage(run_dir, "training", "completed", detail=None)

    # 10. Save Best Adapter & Processor
    best_save_path = os.path.join(args.output_dir, "best_adapter")
    print(f"\nTraining completed! Saving best LoRA adapter to: {best_save_path}")
    trainer.model.save_pretrained(best_save_path)
    processor.save_pretrained(best_save_path)
    print("Done saving best adapter.")
    if mirror is not None:
        mirror.enqueue_adapter(best_save_path)

    if halted:
        return

    # 11. Run Full Post-Training Test Benchmark & Generate Final Report
    if args.run_post_test_eval:
        print("\n=======================================================")
        print(" Running Post-Training Full Benchmark on Complete Test Set...")
        print("=======================================================")

        def evaluate() -> None:
            _run_post_training_evaluation(
                model=trainer.model,
                processor=processor,
                metrics_calculator=metrics_calculator,
                normalizer=normalizer,
                dataset_name=args.dataset_name,
                output_dir=args.output_dir,
                results_dir=args.results_dir,
                eval_batch_size=args.eval_batch_size,
                trainer_state=trainer.state,
                baseline_benchmark=baseline_benchmark,
                phone_benchmark=bool(getattr(args, "phone_benchmark", True)),
                phone_seed=int(getattr(args, "phone_seed", 1234) or 1234),
                telephony_profiles=args.telephony_profiles,
                telephony_use_ffmpeg=bool(args.telephony_use_ffmpeg),
                should_stop=lambda: os.path.exists(stop_signal),
            )

        if not _run_post_eval_stage(
            run_dir=run_dir,
            results_dir=args.results_dir,
            evaluate=evaluate,
        ):
            return
        if mirror is not None:
            mirror.enqueue_ledger("post_eval", reports=True)
    else:
        _mark_run_stage(run_dir, "post_eval", "skipped", detail=None)
        _write_telemetry_status(args.results_dir, "completed", run_dir=run_dir)

    # Overall completion requires training completed and post-eval
    # completed/skipped; merge is tracked separately and never blocks this.
    # The post-eval stage owns the terminal telemetry snapshot on the success
    # path; the skipped branch writes its own above.
    _mark_run_status(
        run_dir, "completed", failed_stage=None, last_error=None
    )


def _format_delta(value: Optional[float]) -> str:
    """Format a WER delta in percentage points, with an explicit no-baseline marker."""
    if value is None:
        return "n/a (no matching full-test baseline)"
    return f"{value:+.2f} pp"


def _format_epoch_delta(value: Optional[float]) -> str:
    """Format one epoch-table delta cell (``--`` when no baseline matches)."""
    if value is None:
        return "--"
    return f"{value:+.2f} pp"


def _normalize_epoch_benchmarks(
    epoch_benchmarks: Any,
    baseline: Optional[Mapping[str, Any]],
) -> list:
    """Return report-safe epoch rows with measured, re-gated deltas.

    Stored ``delta_vs_buzzasr`` fields are never trusted: older runs may carry
    deltas against a removed hardcoded baseline. Baseline rows are included
    only when they came from the measured preflight, and every checkpoint row's
    deltas are recomputed against the current baseline, which yields ``None``
    when the sample counts differ.
    """
    if not isinstance(epoch_benchmarks, list):
        return []

    rows = []
    for raw in epoch_benchmarks:
        if not isinstance(raw, dict):
            continue
        if raw.get("is_baseline") and raw.get("source") != "preflight":
            continue
        row = dict(raw)
        if row.get("is_baseline"):
            row["delta_vs_buzzasr"] = None
            row["delta_phone_vs_buzzasr"] = None
        else:
            row.update(compute_deltas(
                row.get("normalized_wer"),
                row.get("phone_normalized_wer"),
                row.get("total_samples"),
                baseline,
            ))
        rows.append(row)
    return rows


def _run_post_training_evaluation(
    model,
    processor,
    metrics_calculator: ASRMetricsCalculator,
    normalizer: PersianTextNormalizer,
    dataset_name: str,
    output_dir: str,
    results_dir: Optional[str] = None,
    eval_batch_size: int = 16,
    trainer_state: Optional[Any] = None,
    phone_benchmark: bool = True,
    phone_seed: int = 1234,
    telephony_profiles: Optional[list] = None,
    telephony_use_ffmpeg: bool = True,
    baseline_benchmark: Optional[Dict[str, Any]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    import time

    phone_profiles = list(telephony_profiles or ["customer"])
    device = _resolve_eval_device()
    dtype = _resolve_infer_dtype(device)
    use_amp = _should_autocast(device)

    def _check_halt() -> None:
        if should_stop is not None and should_stop():
            raise RuntimeError("halt requested")

    # The caller may hand over a CPU-resident model (post-eval-only resume);
    # features below are moved to this device, so the model must match it.
    model.to(device)

    # 1. Load full test dataset
    pipeline = PersianASRDataPipeline(processor=processor, dataset_name=dataset_name)
    test_ds = pipeline.load_raw_dataset(split="test")
    total_samples = len(test_ds)
    print(f"Loaded full test split: {total_samples} samples.")

    references = [s.get("text", "") or "" for s in test_ds]
    is_informal = []
    for s in test_ds:
        if "informal" in s:
            is_informal.append(bool(s["informal"]))
        elif "is_informal" in s:
            is_informal.append(bool(s["is_informal"]))
        else:
            is_informal.append(normalizer.is_colloquial(s.get("text", "")))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _transcribe(label: str, degrade_fn=None):
        preds = []
        audio_seconds = 0.0
        profiles_used = []
        print(f"Transcribing {total_samples} samples ({label}, batch size = {eval_batch_size})...")
        for i in range(0, total_samples, eval_batch_size):
            batch = test_ds[i : i + eval_batch_size]
            audios = batch["audio"] if isinstance(batch["audio"], list) else [batch["audio"]]
            waveforms = []
            for j, a in enumerate(audios):
                arr = a.get("array")
                sr = a.get("sampling_rate", 16000)
                if arr is not None and len(arr) > 0:
                    audio_seconds += len(arr) / float(sr)
                used_profile = None
                if degrade_fn is not None:
                    degraded = degrade_fn(
                        arr, index=i + j, seed_base=phone_seed,
                        profiles=phone_profiles, use_ffmpeg=telephony_use_ffmpeg,
                    )
                    if isinstance(degraded, tuple):
                        arr, used_profile = degraded
                    else:
                        arr = degraded
                profiles_used.append(used_profile)
                waveforms.append(arr)

            feats = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            if use_amp:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                    pred_ids = model.generate(input_features=feats, max_new_tokens=225)
            else:
                with torch.inference_mode():
                    pred_ids = model.generate(input_features=feats, max_new_tokens=225)

            preds.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
        return preds, audio_seconds, profiles_used

    model.eval()
    _check_halt()
    start_eval_time = time.time()
    predictions, total_audio_seconds, _ = _transcribe("clean")
    elapsed_eval_time = max(0.001, time.time() - start_eval_time)
    rtfx = round(total_audio_seconds / elapsed_eval_time, 2) if total_audio_seconds > 0 else 0.0

    # 2. Compute full metrics
    eval_results = metrics_calculator.evaluate_predictions(
        references=references,
        predictions=predictions,
        is_informal_flags=is_informal,
    )

    # Stop between the clean and phone passes instead of starting another full
    # transcription pass after a user halt.
    _check_halt()

    # 2b. Phone-simulated second pass on the same test split (deterministic)
    phone_block = None
    if phone_benchmark:
        try:
            from src.metrics import (
                evaluate_predictions_by_profile,
                filter_valid_profiles,
                profile_weights,
                weighted_normalized_wer,
            )
            from src.telephone import DEFAULT_PROFILE_WEIGHTS, degrade_for_eval_profiled
            phone_predictions, _, phone_sample_profiles = _transcribe(
                "phone-simulated", degrade_fn=degrade_for_eval_profiled
            )
            phone_results = metrics_calculator.evaluate_predictions(
                references=references,
                predictions=phone_predictions,
                is_informal_flags=is_informal,
            )
            print(metrics_calculator.format_results_table("BuzzASR + LoRA (Final Full Test, phone)", phone_results))
            p_overall = phone_results.get("overall", {})
            phone_per_profile = {}
            phone_profile_weights = {}
            phone_weighted_wer = None
            refs_valid, preds_valid, flags_valid, profiles_valid, skipped = (
                filter_valid_profiles(
                    references, phone_predictions, phone_sample_profiles, is_informal
                )
            )
            if profiles_valid:
                phone_per_profile = evaluate_predictions_by_profile(
                    metrics_calculator, refs_valid, preds_valid,
                    profiles_valid, flags_valid,
                )
                phone_profile_weights = profile_weights(
                    phone_profiles, DEFAULT_PROFILE_WEIGHTS
                )
                phone_weighted_wer = weighted_normalized_wer(
                    phone_per_profile, phone_profile_weights
                )
                weights_label = ", ".join(
                    f"{name}:{weight:.2f}"
                    for name, weight in phone_profile_weights.items()
                )
                print("Phone WER by profile:")
                for profile, result in phone_per_profile.items():
                    overall = result.get("overall", {})
                    print(
                        f"  {profile:<8} n={result.get('total_samples', 0):<5} "
                        f"WER={overall.get('normalized_wer', 0.0):5.2f}%  "
                        f"CER={overall.get('normalized_cer', 0.0):5.2f}%"
                    )
                print(
                    f"  weighted ({weights_label}) WER={phone_weighted_wer:5.2f}%  "
                    f"(unweighted overall {p_overall.get('normalized_wer', 0.0):5.2f}%)\n"
                )
                if skipped:
                    print(
                        f"Warning: {skipped} phone samples had no profile; "
                        f"per-profile WER covers {len(profiles_valid)} samples."
                    )
            phone_block = {
                "normalized_wer": round(p_overall.get("normalized_wer", 0.0), 2),
                "normalized_cer": round(p_overall.get("normalized_cer", 0.0), 2),
                "formal_wer": round(phone_results.get("formal", {}).get("normalized_wer", 0.0), 2),
                "informal_wer": round(phone_results.get("informal", {}).get("normalized_wer", 0.0), 2),
                "gap_vs_clean_pp": round(p_overall.get("normalized_wer", 0.0) - eval_results.get("overall", {}).get("normalized_wer", 0.0), 2),
                "seed": phone_seed,
                "profiles": phone_profiles,
                "per_profile": {
                    profile: {
                        "count": result.get("total_samples", 0),
                        "normalized_wer": result.get("overall", {}).get("normalized_wer", 0.0),
                        "normalized_cer": result.get("overall", {}).get("normalized_cer", 0.0),
                    }
                    for profile, result in phone_per_profile.items()
                },
                "weighted_normalized_wer": phone_weighted_wer,
                "profile_weights": phone_profile_weights,
                "profile_skipped_samples": skipped,
                "details": phone_results,
            }
        except Exception as e:
            print(f"[Final Report] Phone pass failed ({e}); report will be clean-only.")

    # A halt can arrive during the phone pass; never write a report after it.
    _check_halt()

    # 3. Read epoch benchmark history & trainer logs
    epoch_benchmarks = []
    benchmarks_path = os.path.join(_resolve_results_dir(results_dir), "epoch_benchmarks.json")
    if os.path.exists(benchmarks_path):
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                epoch_benchmarks = json.load(f)
        except Exception:
            epoch_benchmarks = []

    # Read training loss & step history from trainer_state
    epoch_training_summary = []
    if trainer_state and getattr(trainer_state, "log_history", None):
        # Group log_history by epoch or summarize evaluation/loss points
        eval_logs = [log for log in trainer_state.log_history if "eval_wer" in log or "eval_loss" in log]
        train_logs = [log for log in trainer_state.log_history if "loss" in log]
        epoch_training_summary = {
            "best_metric": trainer_state.best_metric,
            "best_model_checkpoint": trainer_state.best_model_checkpoint,
            "global_step": trainer_state.global_step,
            "train_logs": train_logs,
            "eval_logs": eval_logs,
        }

    # 4. Compare vs the measured preflight baseline. Deltas are only produced
    # when the baseline and this full test split have the same sample count
    # (1,000-row preflight vs 2,149-row full test => n/a).
    buzz = (baseline_benchmark or {}).get("buzzasr_base") or {}
    norm_wer = eval_results.get("overall", {}).get("normalized_wer", 0.0)
    norm_cer = eval_results.get("overall", {}).get("normalized_cer", 0.0)
    diag_wer = eval_results.get("overall", {}).get("diagnostic_wer_zwnj_space", 0.0)
    formal_wer = eval_results.get("formal", {}).get("normalized_wer", 0.0)
    informal_wer = eval_results.get("informal", {}).get("normalized_wer", 0.0)

    deltas = compute_deltas(
        norm_wer,
        phone_block.get("normalized_wer") if phone_block else None,
        total_samples,
        buzz,
    )
    delta_vs_buzzasr = deltas["delta_vs_buzzasr"]
    delta_phone_vs_buzzasr = deltas["delta_phone_vs_buzzasr"]
    # Re-gate epoch rows against the current measured baseline. Never publish
    # stored deltas from an older run, and drop legacy baseline rows from runs
    # before the measured preflight existed.
    report_epoch_benchmarks = _normalize_epoch_benchmarks(epoch_benchmarks, buzz)

    final_report = {
        "dataset_name": dataset_name,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_samples,
        "total_audio_duration_seconds": round(total_audio_seconds, 2),
        "total_inference_time_seconds": round(elapsed_eval_time, 2),
        "rtfx_speedup": rtfx,
        "final_full_test_metrics": {
            "normalized_wer": round(norm_wer, 2),
            "normalized_cer": round(norm_cer, 2),
            "diagnostic_wer": round(diag_wer, 2),
            "formal_wer": round(formal_wer, 2),
            "informal_wer": round(informal_wer, 2),
            "raw_wer": round(eval_results.get("overall", {}).get("raw_wer", 0.0), 2),
            "raw_cer": round(eval_results.get("overall", {}).get("raw_cer", 0.0), 2),
            "delta_vs_buzzasr_base": delta_vs_buzzasr,
            "delta_phone_vs_buzzasr_base": delta_phone_vs_buzzasr,
            "details": eval_results,
            "phone": phone_block,
        },
        "baseline_comparison": {
            "source": buzz.get("source"),
            "sample_count": buzz.get("total_samples"),
            "evaluated_samples": total_samples,
        },
        "all_epoch_benchmarks": report_epoch_benchmarks,
        "trainer_state_summary": epoch_training_summary,
    }

    # Save JSON report next to final model
    json_report_path = os.path.join(output_dir, "final_evaluation_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\n[Final Report] Saved JSON report to: {json_report_path}")

    # Generate Markdown report
    md_lines = [
        "# Final Persian ASR Evaluation Report",
        f"- **Dataset**: `{dataset_name}` (Full Test Split)",
        f"- **Total Test Samples**: {total_samples}",
        f"- **Total Audio Processed**: {round(total_audio_seconds / 60.0, 2)} minutes",
        f"- **Inference Speed**: {rtfx}x Real-Time (RTFX)",
        f"- **Evaluation Timestamp**: {final_report['evaluated_at']}",
        "",
        "## 1. Post-Training Full Test Benchmark (Best Adapter)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Normalized WER** | **{norm_wer:.2f}%** |",
        f"| **Normalized CER** | {norm_cer:.2f}% |",
    ]
    if phone_block:
        md_lines.append(
            f"| **Phone Normalized WER** (customer/legacy sim, seed={phone_block['seed']}) | "
            f"**{phone_block['normalized_wer']:.2f}%** ({phone_block['gap_vs_clean_pp']:+.2f} pp vs clean) |"
        )
    md_lines += [
        f"| **Diagnostic WER (ZWNJ $\\to$ Space)** | {diag_wer:.2f}% |",
        f"| **Formal Subset WER** | {formal_wer:.2f}% ({eval_results.get('formal', {}).get('count', 0)} samples) |",
        f"| **Informal Subset WER** | {informal_wer:.2f}% ({eval_results.get('informal', {}).get('count', 0)} samples) |",
        f"| **$\\Delta$ vs. BuzzASR Base (clean)** | **{_format_delta(delta_vs_buzzasr)}** |",
        f"| **$\\Delta$ vs. BuzzASR Base (phone)** | **{_format_delta(delta_phone_vs_buzzasr)}** |",
        "",
        "## 2. Benchmark History Across All Epochs",
        "",
    ]

    if report_epoch_benchmarks:
        md_lines.append("| Epoch | Samples | Norm WER (%) | Phone WER (%) | Norm CER (%) | Formal WER (%) | Informal WER (%) | $\\Delta$ clean vs base | $\\Delta$ phone vs base |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        for ep in report_epoch_benchmarks:
            phone_wer = ep.get("phone_normalized_wer")
            phone_str = f"{phone_wer:.2f}%" if phone_wer is not None else "--"
            clean_delta_str = _format_epoch_delta(ep.get("delta_vs_buzzasr"))
            phone_delta_str = _format_epoch_delta(ep.get("delta_phone_vs_buzzasr"))
            md_lines.append(
                f"| {ep.get('epoch', 'N/A')} | {ep.get('total_samples', '-')} | "
                f"{ep.get('normalized_wer', 0.0):.2f}% | {phone_str} | {ep.get('normalized_cer', 0.0):.2f}% | "
                f"{ep.get('formal_wer', 0.0):.2f}% | {ep.get('informal_wer', 0.0):.2f}% | "
                f"{clean_delta_str} | {phone_delta_str} |"
            )
        md_lines.append("")
    else:
        md_lines.append("*No per-epoch benchmark telemetry found.*\n")

    md_report_path = os.path.join(output_dir, "final_evaluation_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Final Report] Saved Markdown report to: {md_report_path}")

    # Print summary table to console
    print("\n" + metrics_calculator.format_results_table("BuzzASR + LoRA (Final Full Test)", eval_results))
    print(f"Delta vs. BuzzASR Base (clean): {_format_delta(delta_vs_buzzasr)}")
    print(f"Delta vs. BuzzASR Base (phone): {_format_delta(delta_phone_vs_buzzasr)}")
    if phone_block:
        print(f"Clean WER: {norm_wer:.2f}% | Phone WER: {phone_block['normalized_wer']:.2f}% | Gap: {phone_block['gap_vs_clean_pp']:+.2f}pp\n")


if __name__ == "__main__":
    main()
