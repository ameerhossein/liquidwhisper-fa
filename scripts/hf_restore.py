#!/usr/bin/env python3
"""Restore a mirrored training run from a Hugging Face repo.

Usage:
  python scripts/hf_restore.py --run <run_id> --repo <owner/name>
  python scripts/hf_restore.py --run latest --repo <owner/name> --include-merged

Exit codes:
  0  restored with a complete checkpoint, ready to resume
  2  restored the ledger only (no complete checkpoint)
  1  restore failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before reading ASR_HF_RUN_REPO / ASR_RUNS_DIR defaults.
try:
    from src.dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except Exception:
    pass

from src.hf_sync import HFSyncError, make_api, resolve_resume_run, restore_run
from src.paths import resolve_runs_dir


def _write_status(
    path: Path,
    *,
    state: str,
    run_id: str,
    message: str,
    checkpoint_step: int | None,
) -> None:
    """Atomically write the restore status JSON consumed by the cockpit."""
    payload = {
        "state": state,
        "run_id": run_id,
        "message": message,
        "updated_at": time.time(),
        "pid": os.getpid(),
        "checkpoint_step": checkpoint_step,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore a mirrored training run from Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("ASR_HF_RUN_REPO", "").strip(),
        help="Hugging Face repo id (default: ASR_HF_RUN_REPO)",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Run id to restore, or 'latest'",
    )
    parser.add_argument(
        "--runs-root",
        default=resolve_runs_dir(),
        help="Local runs root (default: src.paths.resolve_runs_dir())",
    )
    parser.add_argument(
        "--include-merged",
        action="store_true",
        help="Also download the merged model when it exists remotely",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-empty local run directory",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="Optional JSON status file updated at each restore milestone",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    status_path = Path(args.status_file) if args.status_file else None
    progress: dict[str, Any] = {
        "checkpoint_step": None,
        "run_id": args.run,
    }

    def record(state: str, message: str) -> None:
        print(f"[{state}] {message}", flush=True)
        if status_path is None:
            return
        try:
            _write_status(
                status_path,
                state=state,
                run_id=progress["run_id"],
                message=message,
                checkpoint_step=progress["checkpoint_step"],
            )
        except OSError as exc:
            print(f"warning: could not write status file: {exc}", file=sys.stderr)

    repo = (args.repo or "").strip()
    if not repo:
        record(
            "failed",
            "no Hugging Face repo configured; pass --repo or set ASR_HF_RUN_REPO",
        )
        print(
            "error: no Hugging Face repo configured; "
            "pass --repo or set ASR_HF_RUN_REPO",
            file=sys.stderr,
        )
        return 1

    try:
        api = make_api()
        record("resolving", f"Resolving run {args.run} in {repo}")
        run_id = resolve_resume_run(api, repo, args.run)
        progress["run_id"] = run_id
        if run_id != args.run:
            record("resolving", f"Resolved {args.run} to {run_id}")
        result = restore_run(
            api,
            repo,
            run_id,
            args.runs_root,
            include_merged=args.include_merged,
            force=args.force,
            status=record,
        )
    except KeyboardInterrupt:
        record("interrupted", "restore interrupted")
        print("error: restore interrupted", file=sys.stderr)
        return 1
    except HFSyncError as exc:
        record("failed", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        record("failed", str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # keep the CLI failure message actionable
        record("failed", f"{type(exc).__name__}: {exc}")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    run_id = str(result["run_id"])
    run_dir = str(result["run_dir"])
    checkpoint_step = result.get("checkpoint_step")
    progress["checkpoint_step"] = checkpoint_step

    print(f"Restored run directory: {run_dir}")
    print(f"Resume with: python scripts/train_lora.py --hf_resume {run_id}")

    if checkpoint_step is None:
        message = (
            f"restored {run_id} without a complete checkpoint; "
            "no training was started"
        )
        record("failed", message)
        print(f"error: {message}", file=sys.stderr)
        return 2

    record("completed", f"restored {run_id} at checkpoint-{checkpoint_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
