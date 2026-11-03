"""Prepare the paired clean+phone cache in separate, resumable phases.

Typical GPU-box workflow when the cgroup cannot hold both halves at once:

    python scripts/prepare_paired_cache.py all \
        --prep-cache-dir /root/prep_cache \
        --offload-clean-to-hf <user>/liquidwhisper-neyshekar-prep

The clean phase runs in its own process, is uploaded to the HF dataset repo,
and is deleted locally. The phone phase then runs without the clean page cache
resident. Finally the clean half is downloaded again, both halves are
validated and assembled, and ``prep_cache/train`` is published.

Separate commands are available for manual recovery:

    python scripts/prepare_paired_cache.py clean ...
    python scripts/prepare_paired_cache.py phone ...
    python scripts/prepare_paired_cache.py assemble ...

Existing halves made with variant-dependent HF source hashes can be recovered:

    python scripts/prepare_paired_cache.py assemble \
        --prep-cache-dir /root/prep_cache \
        --allow-source-fingerprint-mismatch --no-clean-up

Use this only for halves from the same source run: labels and informal flags
are checked in row order, but cannot prove identical source audio.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except Exception:
    pass

from datasets import Dataset, concatenate_datasets  # noqa: E402

from src import prep_cache  # noqa: E402

STAGE_FORMAT = "prep_half_v2"
MANIFEST = "manifest.json"
DEFAULT_PREP_DIR = (
    (os.environ.get("ASR_PREP_CACHE_DIR", "") or "").strip()
    or "/workspace/stt-outputs/prep_cache"
)
DEFAULT_BASE_MODEL = "BuzzASR/persian"
DEFAULT_DATASET = "shekar-ai/Neyshekar"
DEFAULT_PREP_REPO_NAME = "liquidwhisper-neyshekar-prep"


# ---------------------------------------------------------------------------
# Half-stage manifest and assembly
# ---------------------------------------------------------------------------


def _cache_paths(dataset: Any) -> List[Path]:
    paths: List[Path] = []
    for entry in getattr(dataset, "cache_files", None) or []:
        filename = entry.get("filename") if isinstance(entry, dict) else entry
        if filename:
            paths.append(Path(str(filename)).resolve())
    return paths


def _preparation_hash() -> str:
    """Version feature extraction, labels, normalization and augmentation."""
    digest = hashlib.sha256()
    for name in ("src/data.py", "src/normalizer.py", "src/telephone.py"):
        digest.update(name.encode())
        digest.update((_REPO_ROOT / name).read_bytes())
    return digest.hexdigest()


def write_stage_manifest(
    stage: Path, dataset: Any, *, variant: str,
    fingerprint: Dict[str, Any], source_fingerprint: str,
) -> Dict[str, Any]:
    """Write a manifest for one prepared half, with per-shard row counts."""
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    shards = []
    for path in _cache_paths(dataset):
        if stage not in path.parents:
            raise ValueError(f"cache shard {path} is outside stage {stage}")
        shard = Dataset.from_file(str(path))
        shards.append(
            {
                "file": path.name,
                "rows": int(shard.num_rows),
                "bytes": int(path.stat().st_size),
            }
        )
    if not shards:
        raise ValueError(f"no Arrow cache files were written under {stage}")
    manifest = {
        "format": STAGE_FORMAT,
        "variant": variant,
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "preparation_hash": _preparation_hash(),
        "rows": int(len(dataset)),
        "features": dataset.features.to_dict(),
        "shards": shards,
    }
    tmp = stage / f"{MANIFEST}.tmp"
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, stage / MANIFEST)
    return manifest


def load_stage_manifest(stage: Path) -> Dict[str, Any]:
    stage = Path(stage).resolve()
    manifest = json.loads((stage / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != STAGE_FORMAT:
        raise ValueError(
            f"unexpected stage format {manifest.get('format')!r}; "
            "regenerate legacy halves to record preparation provenance"
        )
    if manifest.get("variant") not in ("clean", "phone"):
        raise ValueError("stage has no valid clean/phone variant")
    if not isinstance(manifest.get("fingerprint"), dict):
        raise ValueError("stage has no preparation fingerprint; regenerate it")
    if not manifest.get("source_fingerprint"):
        raise ValueError("stage has no source dataset fingerprint; regenerate it")
    if manifest.get("preparation_hash") != _preparation_hash():
        raise ValueError("stage preparation code changed; regenerate it")
    rows = int(manifest.get("rows", 0))
    if rows <= 0:
        raise ValueError("stage has no rows")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("stage has no shards")
    for shard in shards:
        name = shard["file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"invalid stage shard filename {name!r}")
        path = stage / name
        if not path.is_file():
            raise ValueError(f"stage shard missing: {path}")
        if int(path.stat().st_size) != int(shard["bytes"]):
            raise ValueError(f"stage shard size mismatch: {path}")
    return manifest


def load_stage_dataset(stage: Path, manifest: Optional[Dict[str, Any]] = None) -> Dataset:
    stage = Path(stage).resolve()
    manifest = manifest or load_stage_manifest(stage)
    parts = [Dataset.from_file(str(stage / str(s["file"]))) for s in manifest["shards"]]
    dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if len(dataset) != int(manifest["rows"]):
        raise ValueError(
            f"stage row count {len(dataset)} != manifest {manifest['rows']}"
        )
    return dataset


def _stage_shard(src: Path, dst: Path) -> None:
    """Hardlink or move a shard so assembly never needs a third data copy."""
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.replace(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def assemble_stages(
    clean_stage: Path,
    phone_stage: Path,
    train_dir: Path,
    fingerprint: Dict[str, Any],
    *,
    force: bool = False,
    clean_up: bool = True,
    allow_source_fingerprint_mismatch: bool = False,
) -> Dict[str, Any]:
    """Validate both halves and publish them as one paired prep-cache slot."""
    clean_stage = Path(clean_stage).resolve()
    phone_stage = Path(phone_stage).resolve()
    train_dir = Path(train_dir).resolve()

    if train_dir.exists() and not force:
        raise FileExistsError(f"{train_dir} already exists (use --force to replace)")

    clean_manifest = load_stage_manifest(clean_stage)
    phone_manifest = load_stage_manifest(phone_stage)
    for variant, manifest in (("clean", clean_manifest), ("phone", phone_manifest)):
        if manifest["variant"] != variant:
            raise ValueError(f"{variant} stage has variant {manifest['variant']!r}")
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(f"{variant} stage preparation fingerprint mismatch")
    if clean_manifest["source_fingerprint"] != phone_manifest["source_fingerprint"]:
        if not allow_source_fingerprint_mismatch:
            raise ValueError(
                "clean/phone source dataset fingerprints differ; for existing halves "
                "affected by variant-dependent HF hashes, retry assemble with "
                "--allow-source-fingerprint-mismatch (row pairing is still checked)"
            )
        print(
            "[assemble] WARNING: accepting different HF source fingerprints; "
            "checking labels and informal flags in row order. These checks cannot "
            "prove that the source audio is identical.",
            flush=True,
        )
    if int(clean_manifest["rows"]) != int(phone_manifest["rows"]):
        raise ValueError(
            "clean/phone row count mismatch: "
            f"{clean_manifest['rows']} != {phone_manifest['rows']}"
        )
    if clean_manifest["features"] != phone_manifest["features"]:
        raise ValueError("clean/phone feature schema mismatch")

    clean = load_stage_dataset(clean_stage, clean_manifest)
    phone = load_stage_dataset(phone_stage, phone_manifest)
    if clean["labels"] != phone["labels"]:
        raise ValueError("clean/phone labels differ; rows are not paired")
    if clean["is_informal"] != phone["is_informal"]:
        raise ValueError("clean/phone is_informal differs; rows are not paired")

    staging = train_dir.parent / f"{train_dir.name}.staging"
    prep_cache.remove_cache(staging)
    staging.mkdir(parents=True, exist_ok=True)

    moved: List[Path] = []
    for prefix, stage, manifest in (
        ("clean", clean_stage, clean_manifest),
        ("phone", phone_stage, phone_manifest),
    ):
        for index, shard in enumerate(manifest["shards"]):
            src = stage / str(shard["file"])
            dst = staging / f"{prefix}_{index:05d}.arrow"
            _stage_shard(src, dst)
            moved.append(dst)

    parts = [Dataset.from_file(str(path)) for path in moved]
    combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if len(combined) != int(clean_manifest["rows"]) + int(phone_manifest["rows"]):
        raise ValueError("assembled row count does not match the two stage manifests")

    meta = prep_cache.publish_dataset(combined, staging, train_dir, fingerprint)
    if clean_up:
        shutil.rmtree(clean_stage, ignore_errors=True)
        shutil.rmtree(phone_stage, ignore_errors=True)
    return meta


# ---------------------------------------------------------------------------
# Preparation phases
# ---------------------------------------------------------------------------


def build_fingerprint(ns: argparse.Namespace) -> Dict[str, Any]:
    from scripts.train_lora import _prep_fingerprint

    args = SimpleNamespace(
        base_model=ns.base_model,
        dataset_name=ns.dataset_name,
        telephony_profiles=list(ns.telephony_profiles),
        telephony_seed=int(ns.telephony_seed),
        telephony_use_ffmpeg=bool(ns.telephony_use_ffmpeg),
    )
    return _prep_fingerprint(args, "train", ns.max_train_samples, "paired", 0.0)


def prepare_half(ns: argparse.Namespace, variant: str) -> int:
    """Prepare exactly one half (clean or phone) and write its manifest."""
    from transformers import WhisperProcessor

    from scripts.train_lora import _label_format
    from src.data import PersianASRDataPipeline
    from src.normalizer import PersianTextNormalizer

    if variant not in ("clean", "phone"):
        raise ValueError(f"unknown variant {variant!r}")

    stage = Path(ns.stage).resolve()
    if stage.exists():
        if not ns.force:
            raise FileExistsError(f"{stage} already exists (use --force to replace)")
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(ns.base_model)
    normalizer = PersianTextNormalizer()
    prefix_ids, eos_id = _label_format(ns.base_model)
    pipeline = PersianASRDataPipeline(
        processor=processor,
        dataset_name=ns.dataset_name,
        normalizer=normalizer,
        num_proc=ns.num_proc,
        # HF hashes the informal filter's closure, including this pipeline.
        # Keep its state identical while loading either source half.
        telephony_prob=0.0,
        telephony_profiles=list(ns.telephony_profiles),
        telephony_seed=int(ns.telephony_seed),
        telephony_use_ffmpeg=bool(ns.telephony_use_ffmpeg),
        label_prefix_ids=prefix_ids,
        label_eos_id=eos_id,
    )
    print(f"[{variant}] loading {ns.dataset_name} train split", flush=True)
    raw = pipeline.load_raw_dataset(split="train", max_samples=ns.max_train_samples)
    if variant == "phone":
        pipeline.telephony_prob = 1.0
        pipeline.strict_telephony = True
    print(f"[{variant}] raw rows={len(raw)}; preparing to {stage}", flush=True)
    dataset = pipeline.prepare_dataset_split(raw, cache_dir=str(stage))
    manifest = write_stage_manifest(
        stage, dataset, variant=variant, fingerprint=build_fingerprint(ns),
        source_fingerprint=raw._fingerprint,
    )
    print(f"[{variant}] OK rows={manifest['rows']} shards={len(manifest['shards'])}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Hugging Face offload
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=_hf_token())


def _auto_hf_repo() -> Optional[str]:
    env_repo = (os.environ.get("ASR_PREP_HF_REPO", "") or "").strip()
    if env_repo.lower() in ("off", "none", "false", "0"):
        return None
    if env_repo:
        return env_repo
    try:
        user = _hf_api().whoami().get("name")
    except Exception:
        return None
    return f"{user}/{DEFAULT_PREP_REPO_NAME}" if user else None


FREE_PRIVATE_LIMIT_GB = 100.0


def _repo_public(ns: argparse.Namespace) -> bool:
    if getattr(ns, "public_repo", False):
        return True
    value = (os.environ.get("ASR_PREP_HF_PUBLIC", "") or "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _private_limit_bytes() -> float:
    raw = (os.environ.get("ASR_PREP_HF_PRIVATE_LIMIT_GB", "") or "").strip()
    try:
        return (float(raw) if raw else FREE_PRIVATE_LIMIT_GB) * 1e9
    except ValueError:
        return FREE_PRIVATE_LIMIT_GB * 1e9


def _ensure_repo(repo_id: str, public: bool) -> bool:
    """Create the dataset repo when missing; return its actual ``is_private``."""
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _hf_api()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        info = None
    if info is None:
        api.create_repo(
            repo_id, repo_type="dataset", private=not public, exist_ok=True
        )
        info = api.repo_info(repo_id, repo_type="dataset")
        is_private = bool(getattr(info, "private", True))
        print(
            f"[hf] created {'private' if is_private else 'public'} dataset repo {repo_id}",
            flush=True,
        )
        return is_private
    is_private = bool(getattr(info, "private", True))
    if public and is_private:
        print(
            f"[hf] WARNING: {repo_id} is private; visibility is not changed "
            "automatically",
            flush=True,
        )
    elif not public and not is_private:
        print(f"[hf] WARNING: {repo_id} is public; uploading publicly", flush=True)
    return is_private


def _repo_storage_bytes(repo_id: str) -> int:
    """Best-effort sum of files already in the repo (0 when unavailable)."""
    try:
        info = _hf_api().repo_info(
            repo_id, repo_type="dataset", files_metadata=True
        )
    except Exception:
        return 0
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        total += int(getattr(sibling, "size", 0) or 0)
    return total


def _assert_private_fits(repo_id: str, is_private: bool, stage_bytes: int) -> None:
    if not is_private:
        return
    limit = _private_limit_bytes()
    if stage_bytes > limit:
        raise RuntimeError(
            f"private HF repo {repo_id} cannot hold {stage_bytes / 1e9:.0f}G "
            f"(free private limit {limit / 1e9:.0f}G); set "
            "ASR_PREP_HF_PUBLIC=1, upgrade to PRO, or pass --no-upload-final"
        )


def upload_stage(
    repo_id: str, stage: Path, path_in_repo: str, *, public: bool = False
) -> None:
    is_private = _ensure_repo(repo_id, public)
    _assert_private_fits(repo_id, is_private, prep_cache.dir_size(stage))
    print(f"[hf] uploading {stage} -> {repo_id}/{path_in_repo}", flush=True)
    _hf_api().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage),
        path_in_repo=path_in_repo,
        commit_message=f"Add paired prep {path_in_repo}",
    )
    print("[hf] upload complete", flush=True)


def download_stage(repo_id: str, stage: Path, path_in_repo: str) -> None:
    from huggingface_hub import snapshot_download

    stage = Path(stage).resolve()
    stage.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hf] downloading {repo_id}/{path_in_repo} -> {stage}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(stage.parent),
        token=_hf_token(),
    )
    print("[hf] download complete", flush=True)


def _evict_stage_files(stage: Path) -> int:
    """Advise the kernel to drop cached pages for a stage's Arrow files."""
    advised = 0
    for path in sorted(Path(stage).rglob("*.arrow")):
        fd = os.open(path, os.O_RDONLY)
        try:
            advised += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return advised


def delete_stage(repo_id: str, path_in_repo: str) -> None:
    print(f"[hf] deleting {repo_id}/{path_in_repo}", flush=True)
    _hf_api().delete_folder(
        repo_id,
        path_in_repo=path_in_repo,
        repo_type="dataset",
        commit_message=f"Remove paired prep {path_in_repo}",
    )
    print("[hf] delete complete", flush=True)


def _drop_caches() -> None:
    subprocess.run(["sync"], check=False)
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3", encoding="ascii")
    except OSError as exc:
        raise RuntimeError(
            "could not drop page cache; use --clean-strategy offload-hf instead"
        ) from exc


def _cgroup_usage_gb() -> Optional[float]:
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            return int(Path(path).read_text().strip()) / 1e9
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _phase_command(ns: argparse.Namespace, phase: str) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        phase,
        "--prep-cache-dir",
        str(ns.prep_cache_dir),
        "--base-model",
        ns.base_model,
        "--dataset-name",
        ns.dataset_name,
        "--num-proc",
        str(ns.num_proc),
        "--telephony-profiles",
        ",".join(ns.telephony_profiles),
        "--telephony-seed",
        str(ns.telephony_seed),
    ]
    if not ns.telephony_use_ffmpeg:
        cmd.append("--no-telephony-use-ffmpeg")
    if ns.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(ns.max_train_samples)])
    if ns.force:
        cmd.append("--force")
    return cmd


def _run_phase(ns: argparse.Namespace, phase: str) -> None:
    print(f"[all] running {phase} in a fresh process", flush=True)
    subprocess.run(_phase_command(ns, phase), check=True)


_HF_DISABLED = ("off", "none", "false", "0")


def _repo_value(value: Optional[str]) -> Optional[str]:
    """Normalize a repo argument; treat disabled sentinels as unset."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _HF_DISABLED:
        return None
    return text


def _final_hf_repo(ns: argparse.Namespace) -> Optional[str]:
    return (
        _repo_value(ns.hf_train_repo)
        or _repo_value(ns.hf_repo)
        or _repo_value(getattr(ns, "offload_clean_to_hf", None))
        or _auto_hf_repo()
    )


def _clean_hf_repo(ns: argparse.Namespace) -> Optional[str]:
    return (
        _repo_value(getattr(ns, "offload_clean_to_hf", None))
        or _repo_value(ns.hf_repo)
        or _repo_value(ns.hf_train_repo)
        or _auto_hf_repo()
    )


def _guard_projected_train(ns: argparse.Namespace, clean_stage: Path) -> None:
    if getattr(ns, "no_upload_final", False):
        return
    repo = _final_hf_repo(ns)
    if not repo:
        return
    is_private = _ensure_repo(repo, _repo_public(ns))
    if not is_private:
        return
    clean_bytes = prep_cache.dir_size(clean_stage)
    strategy = getattr(ns, "clean_strategy", "auto")
    may_offload = (
        strategy in ("auto", "offload-hf") and _clean_hf_repo(ns) == repo
    )
    projected = (3.1 if may_offload else 2.1) * clean_bytes
    _assert_private_fits(
        repo, is_private, int(projected) + _repo_storage_bytes(repo)
    )


def _offload_clean(ns: argparse.Namespace, clean_stage: Path) -> None:
    repo = _clean_hf_repo(ns)
    if not repo:
        raise RuntimeError(
            "local page-cache eviction is not available and no HF repo is "
            "configured; pass --offload-clean-to-hf or --hf-repo"
        )
    upload_stage(repo, clean_stage, clean_stage.name, public=_repo_public(ns))
    shutil.rmtree(clean_stage)
    print("[all] deleted local clean stage after upload", flush=True)


def _upload_final(ns: argparse.Namespace, train_dir: Path) -> None:
    if ns.no_upload_final:
        print("[hf] final upload disabled; train/ stays local", flush=True)
        return
    repo = _final_hf_repo(ns)
    if not repo:
        print(
            "[hf] no HF repo configured (set ASR_PREP_HF_REPO or --hf-repo); "
            "train/ stays local",
            flush=True,
        )
        return
    upload_stage(repo, train_dir, train_dir.name, public=_repo_public(ns))


def run_all(ns: argparse.Namespace) -> int:
    prep_dir = Path(ns.prep_cache_dir).resolve()
    clean_stage = prep_dir / "clean"
    phone_stage = prep_dir / "phone"
    train_dir = prep_dir / "train"

    _run_phase(ns, "clean")
    usage = _cgroup_usage_gb()
    print(f"[all] clean done; cgroup usage={usage if usage is None else f'{usage:.1f}G'}", flush=True)
    _guard_projected_train(ns, clean_stage)

    offloaded = False
    strategy = ns.clean_strategy
    if strategy == "auto":
        evicted = False
        try:
            _drop_caches()
            evicted = True
            print("[all] dropped local page cache", flush=True)
        except RuntimeError as exc:
            print(f"[all] global page-cache drop unavailable: {exc}", flush=True)
        if not evicted:
            try:
                advised = _evict_stage_files(clean_stage)
                evicted = True
                print(
                    f"[all] evicted {advised / 1e9:.1f}G of clean shard page cache",
                    flush=True,
                )
            except OSError as exc:
                print(f"[all] per-file eviction unavailable: {exc}", flush=True)
        usage = _cgroup_usage_gb()
        too_full = usage is not None and usage > ns.max_pre_phone_gb
        if (not evicted) or too_full:
            print("[all] clean cache still resident; offloading to HF", flush=True)
            _offload_clean(ns, clean_stage)
            offloaded = True
        else:
            print("[all] local clean cache is usable; keeping it", flush=True)
    elif strategy == "offload-hf":
        _offload_clean(ns, clean_stage)
        offloaded = True
    elif strategy == "drop-caches":
        _drop_caches()
        print("[all] dropped page cache", flush=True)
    else:
        print(
            "[all] keeping clean stage; phone may OOM unless num_proc is low",
            flush=True,
        )

    usage = _cgroup_usage_gb()
    print(f"[all] before phone; cgroup usage={usage if usage is None else f'{usage:.1f}G'}", flush=True)
    _run_phase(ns, "phone")

    if offloaded:
        repo = _clean_hf_repo(ns)
        if not repo:
            raise RuntimeError("clean was offloaded but no HF repo is configured")
        download_stage(repo, clean_stage, clean_stage.name)

    print("[all] assembling clean+phone", flush=True)
    meta = assemble_stages(
        clean_stage,
        phone_stage,
        train_dir,
        build_fingerprint(ns),
        force=ns.force,
        clean_up=ns.clean_up,
    )
    print(f"[all] published {train_dir} rows={meta['total_rows']}", flush=True)
    _upload_final(ns, train_dir)
    if offloaded and not ns.keep_hf_clean:
        repo = _clean_hf_repo(ns)
        if repo:
            try:
                delete_stage(repo, clean_stage.name)
            except Exception as exc:
                print(f"[hf] WARNING: could not delete remote clean: {exc}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prep-cache-dir", default=DEFAULT_PREP_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--telephony-profiles", default="customer")
    parser.add_argument("--telephony-seed", type=int, default=42)
    parser.add_argument(
        "--public-repo",
        action="store_true",
        help="Create the HF dataset repo as public (env ASR_PREP_HF_PUBLIC=1)",
    )
    parser.add_argument(
        "--telephony-use-ffmpeg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("ASR_PREP_HF_REPO") or None,
        help=(
            "Hugging Face dataset repo for the final train/ cache "
            "(env ASR_PREP_HF_REPO)"
        ),
    )
    parser.add_argument(
        "--hf-train-repo",
        default=None,
        help="Override the repo used for the final train/ upload",
    )
    parser.add_argument(
        "--no-upload-final",
        action="store_true",
        help="Keep the final train/ cache local only",
    )
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="prepare the clean half only")
    _common(clean)
    clean.add_argument("--stage", default=None)

    phone = sub.add_parser("phone", help="prepare the phone half only")
    _common(phone)
    phone.add_argument("--stage", default=None)

    assemble = sub.add_parser("assemble", help="validate and publish clean+phone")
    _common(assemble)
    assemble.add_argument("--clean-stage", default=None)
    assemble.add_argument("--phone-stage", default=None)
    assemble.add_argument("--train-dir", default=None)
    assemble.add_argument("--no-clean-up", action="store_true")
    assemble.add_argument(
        "--allow-source-fingerprint-mismatch",
        action="store_true",
        help=(
            "Recover existing halves with variant-dependent HF hashes; still "
            "validate settings, schema, row counts, labels and informal flags. "
            "Cannot verify identical source audio."
        ),
    )

    allp = sub.add_parser("all", help="run clean, offload/drop, phone, assemble")
    _common(allp)
    allp.add_argument(
        "--clean-strategy",
        choices=("auto", "offload-hf", "drop-caches", "keep"),
        default="auto",
        help=(
            "auto: try dropping page cache locally, then offload clean to HF "
            "if the cgroup is still too full"
        ),
    )
    allp.add_argument("--offload-clean-to-hf", default=None)
    allp.add_argument(
        "--max-pre-phone-gb",
        type=float,
        default=60.0,
        help="With --clean-strategy auto, offload if usage is still above this",
    )
    allp.add_argument("--no-clean-up", action="store_true")
    allp.add_argument(
        "--keep-hf-clean",
        action="store_true",
        help="Keep the uploaded clean/ half in the HF repo after assembly",
    )
    return parser


def _fill_stage_defaults(ns: argparse.Namespace) -> None:
    prep = Path(ns.prep_cache_dir).resolve()
    if getattr(ns, "stage", None) is None:
        ns.stage = str(prep / ("clean" if ns.command == "clean" else "phone"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    ns.telephony_profiles = [
        p.strip() for p in str(ns.telephony_profiles).split(",") if p.strip()
    ]
    if not ns.telephony_profiles:
        ns.telephony_profiles = ["customer"]
    ns.clean_up = not bool(getattr(ns, "no_clean_up", False))

    if ns.command in ("clean", "phone"):
        _fill_stage_defaults(ns)
        return prepare_half(ns, ns.command)

    prep = Path(ns.prep_cache_dir).resolve()
    if ns.command == "assemble":
        if ns.clean_stage is None:
            ns.clean_stage = str(prep / "clean")
        if ns.phone_stage is None:
            ns.phone_stage = str(prep / "phone")
        if ns.train_dir is None:
            ns.train_dir = str(prep / "train")
        meta = assemble_stages(
            Path(ns.clean_stage),
            Path(ns.phone_stage),
            Path(ns.train_dir),
            build_fingerprint(ns),
            force=ns.force,
            clean_up=ns.clean_up,
            allow_source_fingerprint_mismatch=ns.allow_source_fingerprint_mismatch,
        )
        print(f"published {ns.train_dir} rows={meta['total_rows']}")
        _upload_final(ns, Path(ns.train_dir))
        return 0

    return run_all(ns)


if __name__ == "__main__":
    raise SystemExit(main())
