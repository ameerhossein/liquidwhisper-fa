"""Single-command CLI orchestrator for Persian STT domain adaptation.

Usage:
    python scripts/train_domain.py \
        --base_model /path/to/merged_standalone \
        --custom_data_path /path/to/custom_data \
        [--output_dir outputs/buzzasr-domain-lora] \
        [--results_dir results/domain-lora] \
        [--replay_samples 1000] \
        [--replay_ratio 0.25] \
        [--epochs 3] \
        [--learning_rate 2e-5] \
        [--lora_r 16] \
        [--lora_alpha 32]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

# Add repo root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

try:
    from src.cache import setup_hf_cache
    setup_hf_cache()
except Exception:
    pass

from src.domain_data import build_domain_dataset

logger = logging.getLogger("train_domain")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def validate_inputs(
    base_model: Union[str, Path],
    custom_data_path: Union[str, Path],
    require_weights: bool = True,
) -> None:
    """Validate that base_model is an existing directory with config.json and custom_data_path exists.

    Args:
        base_model: Path to standalone model directory. Must contain config.json.
        custom_data_path: Path to custom domain audio directory.
        require_weights: If True, raise ValueError if weight files are missing. Default True.
    """
    base_p = Path(base_model)
    if not base_p.is_dir():
        raise ValueError(f"Base model directory does not exist or is not a directory: {base_model}")

    config_file = base_p / "config.json"
    if not config_file.is_file():
        raise ValueError(
            f"Base model directory '{base_model}' is missing config.json. "
            "A valid merged standalone model directory is required."
        )

    has_weights = (
        (base_p / "model.safetensors").is_file()
        or (base_p / "pytorch_model.bin").is_file()
        or any(base_p.glob("model*.safetensors"))
        or any(base_p.glob("pytorch_model*.bin"))
    )
    if not has_weights:
        if require_weights:
            raise ValueError(
                f"Base model directory '{base_model}' does not contain model weights "
                "(model.safetensors or pytorch_model.bin)."
            )
        logger.warning(
            "No model.safetensors or pytorch_model.bin found in %s; ensure model weights are present before training",
            base_model,
        )

    custom_p = Path(custom_data_path)
    if not custom_p.is_dir():
        raise ValueError(f"Custom data directory does not exist or is not a directory: {custom_data_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for domain adaptation."""
    parser = argparse.ArgumentParser(
        description="Single-Command CLI Orchestrator for Persian STT Domain Adaptation"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to merged standalone base model directory containing config.json and weights",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        required=True,
        help="Path to custom audio directory (containing audio files with .txt pairs or metadata.csv/tsv/jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/buzzasr-domain-lora",
        help="Output directory for checkpoints (default: outputs/buzzasr-domain-lora)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/domain-lora",
        help="Results directory for telemetry and metrics (default: results/domain-lora)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional persistent run directory for cockpit/CLI run ledger",
    )
    parser.add_argument(
        "--replay_samples",
        type=int,
        default=1000,
        help="Number of balanced Neyshekar replay samples (default: 1000)",
    )
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.25,
        help="Replay ratio in training set (default: 0.25)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size (default: 8)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--telephony_mode",
        type=str,
        choices=("paired", "disjoint", "clean"),
        default="paired",
        help="Telephony augmentation mode (default: paired)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Evaluate every N steps (default: 50)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=25,
        help="Linear warmup steps (default: 25)",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        default=False,
        help="Delete previous state and start fresh",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data partitioning and sampling (default: 42)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint or 'last' to resume from",
    )
    parser.add_argument(
        "--post_eval_only",
        action="store_true",
        default=False,
        help="Run only post-training evaluation",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        setattr(args, "unknown_args", unknown)
    return args


def _run_training_subprocess(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Execute training script as a subprocess."""
    logger.info("Executing training command: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {res.returncode}")
    return res.returncode


def run_domain_adaptation(
    args: Optional[Union[argparse.Namespace, str, Path]] = None,
    *,
    base_model: Optional[str] = None,
    custom_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    replay_samples: Optional[int] = None,
    replay_ratio: Optional[float] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    telephony_mode: Optional[str] = None,
    eval_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    fresh_start: Optional[bool] = None,
    seed: Optional[int] = None,
    **extra_kwargs: Any,
) -> int:
    """Orchestrate domain adaptation dataset preparation and LoRA training."""
    if isinstance(args, argparse.Namespace):
        if base_model is None:
            base_model = getattr(args, "base_model", None)
        if custom_data_path is None:
            custom_data_path = getattr(args, "custom_data_path", None)
        if output_dir is None:
            output_dir = getattr(args, "output_dir", None)
        if results_dir is None:
            results_dir = getattr(args, "results_dir", None)
        if run_dir is None:
            run_dir = getattr(args, "run_dir", None)
        if replay_samples is None:
            replay_samples = getattr(args, "replay_samples", None)
        if replay_ratio is None:
            replay_ratio = getattr(args, "replay_ratio", None)
        if epochs is None:
            epochs = getattr(args, "epochs", None)
        if learning_rate is None:
            learning_rate = getattr(args, "learning_rate", None)
        if lora_r is None:
            lora_r = getattr(args, "lora_r", None)
        if lora_alpha is None:
            lora_alpha = getattr(args, "lora_alpha", None)
        if batch_size is None:
            batch_size = getattr(args, "batch_size", None)
        if grad_accum is None:
            grad_accum = getattr(args, "grad_accum", None)
        if telephony_mode is None:
            telephony_mode = getattr(args, "telephony_mode", None)
        if eval_steps is None:
            eval_steps = getattr(args, "eval_steps", None)
        if save_steps is None:
            save_steps = getattr(args, "save_steps", None)
        if warmup_steps is None:
            warmup_steps = getattr(args, "warmup_steps", None)
        if fresh_start is None:
            fresh_start = getattr(args, "fresh_start", None)
        if seed is None:
            seed = getattr(args, "seed", None)
    elif isinstance(args, (str, Path)):
        if base_model is None:
            base_model = str(args)

    if base_model is None:
        raise ValueError("base_model must be specified")
    if custom_data_path is None:
        raise ValueError("custom_data_path must be specified")

    output_dir = output_dir or "outputs/buzzasr-domain-lora"
    results_dir = results_dir or "results/domain-lora"
    replay_samples = 1000 if replay_samples is None else replay_samples
    replay_ratio = 0.25 if replay_ratio is None else replay_ratio
    epochs = 3 if epochs is None else epochs
    learning_rate = 2e-5 if learning_rate is None else learning_rate
    lora_r = 16 if lora_r is None else lora_r
    lora_alpha = 32 if lora_alpha is None else lora_alpha
    batch_size = 8 if batch_size is None else batch_size
    grad_accum = 4 if grad_accum is None else grad_accum
    telephony_mode = telephony_mode or "paired"
    eval_steps = 50 if eval_steps is None else eval_steps
    save_steps = 50 if save_steps is None else save_steps
    warmup_steps = 25 if warmup_steps is None else warmup_steps
    fresh_start = bool(fresh_start)
    seed = 42 if seed is None else seed

    require_weights = extra_kwargs.pop("require_weights", True)
    validate_inputs(base_model=base_model, custom_data_path=custom_data_path, require_weights=require_weights)

    if run_dir:
        try:
            from src import run_state
            run_state.set_stage(Path(run_dir), "preflight", "running", detail="preparing domain dataset")
        except Exception as e:
            logger.debug("Could not update run stage: %s", e)

    if run_dir:
        dataset_dir = Path(run_dir) / "dataset"
    else:
        dataset_dir = Path(output_dir) / "prepared_domain_dataset"

    dataset_dict_file = dataset_dir / "dataset_dict.json"
    if not fresh_start and dataset_dict_file.is_file():
        logger.info("Prepared domain dataset already exists at %s; skipping dataset build", dataset_dir)
    else:
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Building domain adaptation dataset in %s", dataset_dir)
        build_domain_dataset(
            custom_path=str(custom_data_path),
            replay_samples=replay_samples,
            replay_ratio=replay_ratio,
            output_dir=str(dataset_dir),
            seed=seed,
        )

    train_lora_script = BASE_DIR / "scripts" / "train_lora.py"
    cmd = [
        sys.executable,
        "-u",
        str(train_lora_script),
        "--base_model", str(base_model),
        "--dataset_name", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--results_dir", str(results_dir),
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--telephony_mode", str(telephony_mode),
        "--eval_steps", str(eval_steps),
        "--save_steps", str(save_steps),
        "--warmup_steps", str(warmup_steps),
    ]
    if run_dir:
        cmd.extend(["--run_dir", str(run_dir)])
    if fresh_start:
        cmd.append("--fresh_start")
    resume_ckpt = extra_kwargs.pop("resume_from_checkpoint", getattr(args, "resume_from_checkpoint", None) if isinstance(args, argparse.Namespace) else None)
    if resume_ckpt:
        cmd.extend(["--resume_from_checkpoint", str(resume_ckpt)])
    post_eval = extra_kwargs.pop("post_eval_only", getattr(args, "post_eval_only", False) if isinstance(args, argparse.Namespace) else False)
    if post_eval:
        cmd.append("--post_eval_only")

    for k, v in extra_kwargs.items():
        flag = f"--{k}"
        if v is True:
            cmd.append(flag)
        elif v is False or v is None:
            continue
        else:
            cmd.extend([flag, str(v)])

    unknown_args = getattr(args, "unknown_args", None) if isinstance(args, argparse.Namespace) else None
    if unknown_args:
        cmd.extend([str(item) for item in unknown_args])

    if run_dir:
        try:
            from src import run_state
            run_state.set_stage(Path(run_dir), "training", "running", detail="running domain adaptation LoRA fine-tuning")
        except Exception as e:
            logger.debug("Could not update run stage: %s", e)

    logger.info("Starting domain adaptation LoRA fine-tuning...")
    return _run_training_subprocess(cmd, cwd=str(BASE_DIR))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        validate_inputs(base_model=args.base_model, custom_data_path=args.custom_data_path)
        return run_domain_adaptation(args)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
