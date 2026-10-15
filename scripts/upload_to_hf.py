"""
Upload training artifacts (best_adapter, merged model, checkpoints) to Hugging Face Hub.

Three artifacts, three repos (HF-standard: from_pretrained expects files at root):

  <prefix>             <- merged standalone model (scripts/merge_lora.py output)
  <prefix>-lora        <- best_adapter/ (LoRA + processor, needs base + peft)
  <prefix>-checkpoints <- checkpoint-*/ under checkpoints/<name>/ (resume-only)

Usage:
  hf auth login  # once, or export HF_TOKEN=hf_...
  python scripts/upload_to_hf.py --artifact all --repo-prefix <user>/<prefix> [--public]
  python scripts/upload_to_hf.py --artifact best_adapter --repo-prefix <user>/<prefix> --dry-run

Paths are managed: defaults come from src/paths.py
(ASR_OUTPUT_DIR / ASR_RESULTS_DIR / ASR_MERGED_DIR, else /workspace/... else local).
Explicit flags always win. API upload bypasses .gitignore.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env (repo root / CWD) before any os.environ reads. Stdlib-only.
try:
    from src.dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

from src.paths import (
    find_complete_checkpoints,
    find_incomplete_checkpoints,
    resolve_adapter_dir,
    resolve_merged_dir,
    resolve_output_dir,
    resolve_results_dir,
)

BASE_MODEL_DEFAULT = "BuzzASR/persian"
DATASET_DEFAULT = "shekar-ai/Neyshekar"


# ---------------------------------------------------------------- validation

def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _list_files(path: Path, limit: int = 20) -> List[str]:
    files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
    files = sorted(files)
    if len(files) > limit:
        return files[:limit] + [f"... (+{len(files) - limit} more)"]
    return files


def _fmt_mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def validate_adapter(path: Path) -> Tuple[bool, str]:
    if not path.is_dir():
        return False, f"not a directory: {path}"
    if not (path / "adapter_config.json").is_file():
        return False, "missing adapter_config.json (wrong dir? expected best_adapter/)"
    weights = list(path.glob("adapter_model.*")) + list(path.glob("adapter_model_*.safetensors"))
    if not weights:
        return False, "missing adapter_model.safetensors/.bin"
    return True, "ok"


def validate_merged(path: Path) -> Tuple[bool, str]:
    if not path.is_dir():
        return False, f"not a directory: {path}"
    if not (path / "config.json").is_file():
        return False, "missing config.json (run scripts/merge_lora.py first?)"
    weights = list(path.glob("model*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights:
        return False, "missing model weights (model*.safetensors / pytorch_model*.bin)"
    return True, "ok"


def validate_checkpoint(path: Path) -> Tuple[bool, str]:
    if not path.is_dir():
        return False, f"not a directory: {path}"
    if not (path / "trainer_state.json").is_file():
        return False, "missing trainer_state.json (incomplete save, skipped)"
    return True, "ok"


def collect_reports(output_dir: Path, results_dir: Path) -> Dict[str, Path]:
    cands = {
        "final_evaluation_report.json": output_dir / "final_evaluation_report.json",
        "final_evaluation_report.md": output_dir / "final_evaluation_report.md",
        "epoch_benchmarks.json": results_dir / "epoch_benchmarks.json",
    }
    return {k: v for k, v in cands.items() if v.is_file()}


def load_metrics_summary(output_dir: Path) -> Dict[str, object]:
    """Best-effort WER summary for the model card; {} when no report yet."""
    try:
        rep = json.loads((output_dir / "final_evaluation_report.json").read_text(encoding="utf-8"))
        m = rep.get("final_full_test_metrics", {}) or {}
        return {
            "normalized_wer": m.get("normalized_wer"),
            "normalized_cer": m.get("normalized_cer"),
            "formal_wer": m.get("formal_wer"),
            "informal_wer": m.get("informal_wer"),
            "delta_vs_whisper_v3": m.get("delta_vs_whisper_v3"),
            "delta_vs_buzzasr_base": m.get("delta_vs_buzzasr_base"),
            "total_test_samples": rep.get("total_test_samples"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------- model card

def build_model_card(
    kind: str,
    repo_id: str,
    base_model: str,
    dataset_name: str,
    metrics: Dict[str, object],
) -> str:
    wer = metrics.get("normalized_wer", "n/a")
    cer = metrics.get("normalized_cer", "n/a")
    lines = [
        "---",
        "language: fa",
        "tags:",
        "- automatic-speech-recognition",
        "- persian",
        "- whisper",
        "- lora" if kind in ("adapter", "checkpoints") else "- fine-tuned",
        "library_name: transformers",
        f"base_model: {base_model}",
        "---",
        "",
        f"# {repo_id} ({kind})",
        "",
        f"Persian ASR fine-tune of `{base_model}` on `{dataset_name}` (LoRA r=64, alpha=128).",
        "",
        "## Artifacts in this org",
        "",
        "- `<prefix>` — merged standalone model (no `peft` needed).",
        "- `<prefix>-lora` — `best_adapter/` (needs base + `peft`).",
        "- `<prefix>-checkpoints` — `checkpoint-*/` under `checkpoints/` (resume-only, not for inference).",
        "",
        "## Metrics (full test split, normalized)",
        "",
        f"- WER: {wer} | CER: {cer}",
        f"- formal WER: {metrics.get('formal_wer', 'n/a')} | informal WER: {metrics.get('informal_wer', 'n/a')}",
        f"- Δ vs Whisper Large v3: {metrics.get('delta_vs_whisper_v3', 'n/a')} | Δ vs BuzzASR base: {metrics.get('delta_vs_buzzasr_base', 'n/a')}",
        f"- test samples: {metrics.get('total_test_samples', 'n/a')}",
        "",
    ]
    if kind == "adapter":
        lines += [
            "## Use (adapter)",
            "",
            "```python",
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor",
            "from peft import PeftModel",
            f"proc = WhisperProcessor.from_pretrained('{repo_id}')",
            f"base = WhisperForConditionalGeneration.from_pretrained('{base_model}')",
            f"model = PeftModel.from_pretrained(base, '{repo_id}')",
            "```",
            "",
        ]
    elif kind == "merged":
        lines += [
            "## Use (merged)",
            "",
            "```python",
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor",
            f"proc = WhisperProcessor.from_pretrained('{repo_id}')",
            f"model = WhisperForConditionalGeneration.from_pretrained('{repo_id}', dtype='auto').to('cuda').eval()",
            "```",
            "",
        ]
    else:
        lines += [
            "## Layout",
            "",
            "- `checkpoints/checkpoint-<step>/` each holds LoRA + optimizer + scheduler + `trainer_state.json`.",
            "- Resume with `scripts/train_lora.py --resume_from_checkpoint <dir>` after downloading.",
            "- Incomplete saves (no `trainer_state.json`) are never uploaded.",
            "",
        ]
    lines += [f"Base: `{base_model}` | Dataset: `{dataset_name}`", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- repo ids

def derive_repo_ids(
    prefix: Optional[str],
    merged_repo: Optional[str],
    adapter_repo: Optional[str],
    checkpoints_repo: Optional[str],
    artifacts: List[str],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if prefix:
        prefix = prefix.strip().strip("/")
        out = {
            "merged": prefix,
            "best_adapter": f"{prefix}-lora",
            "checkpoint": f"{prefix}-checkpoints",
        }
    if merged_repo:
        out["merged"] = merged_repo.strip()
    if adapter_repo:
        out["best_adapter"] = adapter_repo.strip()
    if checkpoints_repo:
        out["checkpoint"] = checkpoints_repo.strip()
    missing = [a for a in artifacts if a not in out or not out[a]]
    if missing:
        raise SystemExit(
            f"Missing repo id for {missing}. Pass --repo-prefix <user>/<prefix> "
            "or explicit --merged-repo/--adapter-repo/--checkpoints-repo."
        )
    for k, v in out.items():
        if "/" not in v:
            raise SystemExit(
                f"Repo id '{v}' for '{k}' needs 'username/repo' form "
                "(pass --repo-prefix <user>/<prefix>)."
            )
    return out


# ---------------------------------------------------------------- upload

def _require_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install -U 'huggingface_hub>=0.23.0' "
            "and: hf auth login  (or export HF_TOKEN)."
        )


def do_upload(
    api,  # HfApi
    repo_id: str,
    local_dir: Path,
    path_in_repo: str,
    private: bool,
    create: bool,
    commit_message: str,
    revision: str = "main",
) -> None:
    if create:
        from huggingface_hub import create_repo

        create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo or ".",
        commit_message=commit_message,
        revision=revision,
    )


def upload_text(api, repo_id: str, text: str, filename: str, commit_message: str, revision: str = "main") -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_text(text, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
            revision=revision,
        )


# ---------------------------------------------------------------- cli

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload best_adapter / merged / checkpoints to HF Hub (3 repos).")
    p.add_argument("--artifact", default="all",
                   choices=["best_adapter", "merged", "checkpoint", "checkpoints", "all"],
                   help="Which artifact to upload (default: all).")
    p.add_argument("--output-dir", default=None, help="Checkpoints root (default: managed ASR_OUTPUT_DIR).")
    p.add_argument("--adapter-path", default=None, help="best_adapter dir (default: <output_dir>/best_adapter).")
    p.add_argument("--merged-path", default=None, help="Merged model dir (default: managed ASR_MERGED_DIR).")
    p.add_argument("--results-dir", default=None, help="Telemetry dir for reports (default: managed ASR_RESULTS_DIR).")
    p.add_argument("--repo-prefix", default=None, help="e.g. <user>/liquidwhisper-fa -> <prefix>, <prefix>-lora, <prefix>-checkpoints.")
    p.add_argument("--merged-repo", default=None, help="Explicit merged repo id (overrides prefix).")
    p.add_argument("--adapter-repo", default=None, help="Explicit adapter repo id (overrides prefix).")
    p.add_argument("--checkpoints-repo", default=None, help="Explicit checkpoints repo id (overrides prefix).")
    p.add_argument("--private", dest="private", action=argparse.BooleanOptionalAction, default=True,
                   help="Private repos by default; --no-private makes them public.")
    p.add_argument("--public", action="store_true", help="Shortcut for --no-private.")
    p.add_argument("--create-repo", dest="create_repo", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="Print plan + file lists, upload nothing.")
    p.add_argument("--include-reports", dest="include_reports",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Also upload final_evaluation_report + epoch_benchmarks to each repo.")
    p.add_argument("--commit-message", default=None, help="Custom commit message (default: auto per artifact).")
    p.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env or cached `hf auth login`).")
    p.add_argument("--max-checkpoints", type=int, default=None, help="Upload only newest N checkpoints.")
    p.add_argument("--checkpoint", action="append", default=None,
                   help="Only this checkpoint name (e.g. checkpoint-500). Repeatable.")
    p.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--dataset-name", default=DATASET_DEFAULT)
    p.add_argument("--revision", default="main")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.public:
        args.private = False

    artifact = "checkpoint" if args.artifact == "checkpoints" else args.artifact
    wanted = ["best_adapter", "merged", "checkpoint"] if artifact == "all" else [artifact]

    output_dir = Path(resolve_output_dir(args.output_dir))
    results_dir = Path(resolve_results_dir(args.results_dir))
    adapter_path = Path(resolve_adapter_dir(str(output_dir), args.adapter_path))
    merged_path = Path(resolve_merged_dir(args.merged_path))
    repo_ids = derive_repo_ids(args.repo_prefix, args.merged_repo, args.adapter_repo,
                               args.checkpoints_repo, wanted)

    # ---- gather + validate (no network yet)
    plan: Dict[str, dict] = {}
    if "best_adapter" in wanted:
        ok, msg = validate_adapter(adapter_path)
        plan["best_adapter"] = {"ok": ok, "msg": msg, "path": adapter_path}
    if "merged" in wanted:
        ok, msg = validate_merged(merged_path)
        plan["merged"] = {"ok": ok, "msg": msg, "path": merged_path}
    checkpoints: List[Path] = []
    if "checkpoint" in wanted:
        all_ckpts = find_complete_checkpoints(str(output_dir))
        if args.checkpoint:
            want = {c.strip().strip("/") for c in args.checkpoint}
            all_ckpts = [c for c in all_ckpts if c.name in want or str(c) in want]
            if not all_ckpts:
                print(f"Warning: --checkpoint filter matched nothing in {output_dir}")
        if args.max_checkpoints is not None:
            all_ckpts = all_ckpts[-max(0, args.max_checkpoints):]
        checkpoints = all_ckpts
        incomplete = find_incomplete_checkpoints(str(output_dir))
        if incomplete and not args.checkpoint:
            print(f"Skipping {len(incomplete)} incomplete checkpoint(s) (no trainer_state.json).")
        plan["checkpoint"] = {"ok": bool(checkpoints), "path": output_dir,
                              "msg": f"{len(checkpoints)} complete" if checkpoints else "none found"}

    reports = collect_reports(output_dir, results_dir) if args.include_reports else {}
    metrics = load_metrics_summary(output_dir)

    # ---- dry-run / pre-flight report
    failures = [f"{k}: {v['msg']} ({v['path']})" for k, v in plan.items()
                if k != "checkpoint" and not v["ok"]]
    if "checkpoint" in plan and not plan["checkpoint"]["ok"]:
        failures.append(f"checkpoint: none found in {output_dir} (train first or pass --output-dir)")
    print("=" * 64)
    print(" HF upload plan (managed dirs, private by default)" if not args.dry_run else " HF upload DRY-RUN")
    print(f" output_dir:  {output_dir}")
    print(f" results_dir: {results_dir}")
    for k in wanted:
        print(f" {k:12s} -> {repo_ids[k]}  [{'private' if args.private else 'public'}]")
    for k, v in plan.items():
        if k == "checkpoint":
            for c in checkpoints:
                print(f"  - {c.name} ({_fmt_mb(_dir_size(c))})")
            if not checkpoints:
                print("  - (no complete checkpoints)")
            continue
        st = "OK " if v["ok"] else "FAIL"
        size = _fmt_mb(_dir_size(v["path"])) if v["ok"] else "-"
        print(f" [{st}] {k}: {v['path']} [{size}] — {v['msg']}")
        if v["ok"]:
            for f in _list_files(v["path"]):
                print(f"       {f}")
    if reports:
        print(f" reports ({len(reports)}): " + ", ".join(sorted(reports)))
    else:
        print(" reports: none (run training/eval first or --no-include-reports to silence)")
    if failures:
        print("-" * 64)
        print("Resolve before uploading:")
        for f in failures:
            print(f"  - {f}")
        if "merged" in [k for k, v in plan.items() if not v["ok"]]:
            print("    Merge first: python scripts/merge_lora.py "
                  f"--adapter_path {adapter_path} --output_dir {merged_path}")
    print("=" * 64)

    if args.dry_run:
        return
    if failures:
        raise SystemExit(1)

    _require_hub()
    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"Authenticated as: {who.get('name')}")
    except Exception:
        print("Warning: no valid HF auth found (export HF_TOKEN or run `hf auth login`). "
              "Trying unauthenticated/cached credentials...")

    base_msg = args.commit_message

    if "best_adapter" in wanted:
        rid = repo_ids["best_adapter"]
        do_upload(api, rid, adapter_path, "", args.private, args.create_repo,
                  base_msg or f"Upload best_adapter (BuzzASR LoRA, {adapter_path.name})",
                  revision=args.revision)
        card = build_model_card("adapter", rid, args.base_model, args.dataset_name, metrics)
        upload_text(api, rid, card, "README.md", base_msg or "Add model card (adapter)",
                    revision=args.revision)
        for name, src in reports.items():
            api.upload_file(path_or_fileobj=str(src), path_in_repo=f"reports/{name}",
                            repo_id=rid, commit_message=base_msg or f"Add report {name}",
                            revision=args.revision)
        print(f"Uploaded adapter -> {rid}")

    if "merged" in wanted:
        rid = repo_ids["merged"]
        do_upload(api, rid, merged_path, "", args.private, args.create_repo,
                  base_msg or "Upload merged standalone model",
                  revision=args.revision)
        card = build_model_card("merged", rid, args.base_model, args.dataset_name, metrics)
        upload_text(api, rid, card, "README.md", base_msg or "Add model card (merged)",
                    revision=args.revision)
        for name, src in reports.items():
            api.upload_file(path_or_fileobj=str(src), path_in_repo=f"reports/{name}",
                            repo_id=rid, commit_message=base_msg or f"Add report {name}",
                            revision=args.revision)
        print(f"Uploaded merged -> {rid}")

    if "checkpoint" in wanted:
        from huggingface_hub import create_repo

        rid = repo_ids["checkpoint"]
        if args.create_repo:
            create_repo(rid, private=args.private, exist_ok=True)
        for ckpt in checkpoints:
            ok, msg = validate_checkpoint(ckpt)
            if not ok:
                print(f"Skipping {ckpt.name}: {msg}")
                continue
            api.upload_folder(repo_id=rid, folder_path=str(ckpt),
                              path_in_repo=f"checkpoints/{ckpt.name}",
                              commit_message=base_msg or f"Upload {ckpt.name}",
                              revision=args.revision)
            print(f"Uploaded {ckpt.name} -> {rid}/checkpoints/{ckpt.name}")
        card = build_model_card("checkpoints", rid, args.base_model, args.dataset_name, metrics)
        upload_text(api, rid, card, "README.md", base_msg or "Add model card (checkpoints)",
                    revision=args.revision)
        for name, src in reports.items():
            api.upload_file(path_or_fileobj=str(src), path_in_repo=f"reports/{name}",
                            repo_id=rid, commit_message=base_msg or f"Add report {name}",
                            revision=args.revision)


if __name__ == "__main__":
    main()
