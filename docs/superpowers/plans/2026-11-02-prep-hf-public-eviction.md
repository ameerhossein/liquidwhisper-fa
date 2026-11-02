# Prep HF Public Repo + Local Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `prepare_paired_cache.py all` publish to a public HF dataset repo, evict the clean half locally before any upload, and remove `clean/` from the repo after assembly.

**Architecture:** All work is inside `scripts/prepare_paired_cache.py`. A visibility helper creates or inspects the repo once and returns the real `private` flag; every upload goes through a size guard tied to that flag. The `auto` clean strategy gains a per-file `posix_fadvise` step between global `drop_caches` and HF offload, and a post-assembly step deletes the remote `clean/` folder.

**Tech Stack:** Python 3.12, `huggingface_hub` 0.36.2, `datasets`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-02-prep-hf-public-eviction-design.md`

## Global Constraints

- Free HF private storage limit constant: `FREE_PRIVATE_LIMIT_GB = 100.0`.
- Assembly projection factor: `2.1 * prep_cache.dir_size(clean_stage)`.
- Default repo visibility stays **private**; public only via `ASR_PREP_HF_PUBLIC` or `--public-repo`.
- An existing repo's visibility is never changed.
- `ASR_PREP_HF_PRIVATE_LIMIT_GB` overrides the limit; unparseable values fall back to 100.0.
- No WebUI/cockpit changes. No trainer changes.
- Every new public function gets a test in `tests/test_prepare_paired_cache.py`.

## Review Focus

- Existing **private** repo plus `ASR_PREP_HF_PUBLIC=1`: expect a warning, no visibility change, and the private quota guard still applied.
- `ASR_PREP_HF_PRIVATE_LIMIT_GB=garbage`: expect the 100 GB default, not a crash.
- HF `repo_info` network/auth error: expect `create_repo` to surface the real error rather than silently publishing.
- Clean stage with zero `.arrow` files: expect `_evict_stage_files` to return `0` and the run to continue.
- Remote `clean/` delete failure: expect a warning and exit `0`, with `train/` still published.

---

### Task 1: Public visibility and private quota guard

**Files:**
- Modify: `scripts/prepare_paired_cache.py` (`upload_stage`, `_offload_clean`, `_upload_final` ~lines 295-440, `_common` ~line 500, `run_all` ~line 445)
- Test: `tests/test_prepare_paired_cache.py`

**Interfaces:**
- Produces:
  - `FREE_PRIVATE_LIMIT_GB: float = 100.0`
  - `_repo_public(ns) -> bool`
  - `_ensure_repo(repo_id: str, public: bool) -> bool` — returns the repo's actual `is_private`
  - `_private_limit_bytes() -> float`
  - `_assert_private_fits(repo_id: str, is_private: bool, stage_bytes: int) -> None`
  - `upload_stage(repo_id: str, stage: Path, path_in_repo: str, *, public: bool = False) -> None`
  - `_guard_projected_train(ns, clean_stage: Path) -> None`
  - CLI flag `--public-repo` on every subcommand

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_paired_cache.py`:

```python
def _all_ns(tmp_path, **overrides):
    from types import SimpleNamespace

    ns = SimpleNamespace(
        prep_cache_dir=str(tmp_path),
        base_model="m", dataset_name="d", num_proc=1,
        telephony_profiles=["customer"], telephony_seed=42,
        telephony_use_ffmpeg=True, max_train_samples=None, force=True,
        clean_strategy="auto", max_pre_phone_gb=60.0, clean_up=True,
        no_upload_final=True, keep_hf_clean=False,
        hf_train_repo=None, hf_repo=None, offload_clean_to_hf=None,
        public_repo=False,
    )
    ns.__dict__.update(overrides)
    return ns


def test_ensure_repo_creates_public_when_requested(monkeypatch):
    calls = {}

    class _Api:
        def repo_info(self, *a, **k):
            raise RuntimeError("not found")

        def create_repo(self, repo_id, **kw):
            calls.update(kw, repo_id=repo_id)

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    assert ppc._ensure_repo("me/prep", public=True) is False
    assert calls["private"] is False
    assert calls["repo_type"] == "dataset"


def test_ensure_repo_never_flips_existing_private_repo(monkeypatch):
    calls = {"created": False}

    class _Api:
        def repo_info(self, *a, **k):
            return type("Info", (), {"private": True})()

        def create_repo(self, *a, **k):
            calls["created"] = True

    monkeypatch.setattr(ppc, "_hf_api", lambda: _Api())
    assert ppc._ensure_repo("me/prep", public=True) is True
    assert calls["created"] is False


def test_private_quota_guard_rejects_oversized_final_cache():
    with pytest.raises(RuntimeError, match="private"):
        ppc._assert_private_fits("me/prep", True, 168 * 10**9)
    ppc._assert_private_fits("me/prep", False, 168 * 10**9)


def test_guard_projected_train_skips_when_no_final_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ppc, "_ensure_repo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("checked")),
    )
    ppc._guard_projected_train(_all_ns(tmp_path, no_upload_final=True), tmp_path)


def test_private_limit_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("ASR_PREP_HF_PRIVATE_LIMIT_GB", "garbage")
    assert ppc._private_limit_bytes() == 100.0 * 1e9

    monkeypatch.setenv("ASR_PREP_HF_PRIVATE_LIMIT_GB", "2048")
    assert ppc._private_limit_bytes() == 2048.0 * 1e9
```

Update the two existing `_upload_final` tests so their `upload_stage` shims accept the new keyword:

```python
    monkeypatch.setattr(
        ppc,
        "upload_stage",
        lambda repo, stage, path, **kw: calls.update(repo=repo, stage=stage, path=path),
    )
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q -k "ensure_repo or quota_guard or projected_train"`
Expected: FAIL, `AttributeError: module ... has no attribute '_ensure_repo'`.

- [ ] **Step 3: Implement the visibility and quota helpers**

In `scripts/prepare_paired_cache.py`, above `upload_stage`:

```python
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
    api = _hf_api()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except Exception:
        info = None
    if info is None:
        api.create_repo(
            repo_id, repo_type="dataset", private=not public, exist_ok=True
        )
        print(
            f"[hf] created {'public' if public else 'private'} dataset repo {repo_id}",
            flush=True,
        )
        return not public
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
```

Change `upload_stage` to take `public` and run the guard:

```python
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
```

Add the projection guard and call it from `run_all` right after the clean phase:

```python
def _guard_projected_train(ns: argparse.Namespace, clean_stage: Path) -> None:
    if getattr(ns, "no_upload_final", False):
        return
    repo = _final_hf_repo(ns)
    if not repo:
        return
    is_private = _ensure_repo(repo, _repo_public(ns))
    if not is_private:
        return
    projected = 2.1 * prep_cache.dir_size(clean_stage)
    _assert_private_fits(repo, is_private, int(projected))
```

Pass visibility at both call sites:

```python
    upload_stage(repo, clean_stage, clean_stage.name, public=_repo_public(ns))
    upload_stage(repo, train_dir, train_dir.name, public=_repo_public(ns))
```

Add `parser.add_argument("--public-repo", action="store_true")` to `_common`.

- [ ] **Step 4: Run the task tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_paired_cache.py tests/test_prepare_paired_cache.py
git commit -m "feat(prep): support public HF prep repo with private quota guard"
```

---

### Task 2: Per-file page-cache eviction before HF offload

**Files:**
- Modify: `scripts/prepare_paired_cache.py` (`_drop_caches` area ~line 338, `run_all` auto branch ~line 455)
- Test: `tests/test_prepare_paired_cache.py`

**Interfaces:**
- Consumes: `_all_ns`, `run_all`, `_offload_clean`.
- Produces: `_evict_stage_files(stage: Path) -> int` — bytes advised, `0` when no `.arrow` files exist.

- [ ] **Step 1: Write the failing tests**

```python
def test_evict_stage_files_advises_every_arrow(tmp_path, monkeypatch):
    (tmp_path / "a.arrow").write_bytes(b"x" * 10)
    (tmp_path / "b.arrow").write_bytes(b"y" * 20)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        ppc.os,
        "posix_fadvise",
        lambda fd, off, length, advice: seen.append(advice),
    )
    assert ppc._evict_stage_files(tmp_path) == 30
    assert seen == [ppc.os.POSIX_FADV_DONTNEED, ppc.os.POSIX_FADV_DONTNEED]
    assert ppc._evict_stage_files(tmp_path / "empty") == 0


def test_auto_keeps_clean_when_fadvise_frees_memory(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(
        ppc, "_drop_caches",
        lambda: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    monkeypatch.setattr(
        ppc, "_evict_stage_files",
        lambda stage: events.append("fadvise") or 84_000_000_000,
    )
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 20.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: None)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)

    ppc.run_all(_all_ns(tmp_path))

    assert "fadvise" in events
    assert "offload" not in events


def test_auto_offloads_when_memory_stays_high_after_fadvise(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(
        ppc, "_drop_caches",
        lambda: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    monkeypatch.setattr(ppc, "_evict_stage_files", lambda stage: 1)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 150.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: None)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)

    ppc.run_all(_all_ns(tmp_path))

    assert "offload" in events
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q -k "evict_stage_files or fadvise"`
Expected: FAIL, `AttributeError: ... '_evict_stage_files'`.

- [ ] **Step 3: Implement eviction and wire it into `auto`**

```python
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
```

Replace the `auto` branch in `run_all`:

```python
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
```

- [ ] **Step 4: Run the task tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_paired_cache.py tests/test_prepare_paired_cache.py
git commit -m "feat(prep): evict clean page cache per file before HF offload"
```

---

### Task 3: Delete remote `clean/` after assembly

**Files:**
- Modify: `scripts/prepare_paired_cache.py` (`download_stage` area ~line 322, `run_all` end, `all` parser ~line 570)
- Test: `tests/test_prepare_paired_cache.py`

**Interfaces:**
- Consumes: `_clean_hf_repo`, `run_all`, `_all_ns`.
- Produces: `delete_stage(repo_id: str, path_in_repo: str) -> None`; CLI flag `--keep-hf-clean`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_all_deletes_remote_clean_after_assembling(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: events.append(phase))
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_drop_caches", lambda: events.append("drop"))
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: events.append("offload"))
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: events.append("download"))
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append(f"delete:{path}"))

    ppc.run_all(_all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep"))

    assert events[-1] == "delete:clean"


def test_run_all_keeps_remote_clean_with_flag(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(ppc, "delete_stage", lambda repo, path: events.append("delete"))

    ppc.run_all(
        _all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep",
                keep_hf_clean=True)
    )

    assert "delete" not in events


def test_delete_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ppc, "_run_phase", lambda ns, phase: None)
    monkeypatch.setattr(ppc, "_guard_projected_train", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_clean_hf_repo", lambda ns: "me/prep")
    monkeypatch.setattr(ppc, "_offload_clean", lambda ns, stage: None)
    monkeypatch.setattr(ppc, "_cgroup_usage_gb", lambda: 10.0)
    monkeypatch.setattr(ppc, "assemble_stages", lambda *a, **k: {"total_rows": 4})
    monkeypatch.setattr(ppc, "build_fingerprint", lambda ns: {})
    monkeypatch.setattr(ppc, "download_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        ppc, "delete_stage",
        lambda repo, path: (_ for _ in ()).throw(RuntimeError("403")),
    )
    assert ppc.run_all(_all_ns(tmp_path, clean_strategy="offload-hf", hf_repo="me/prep")) == 0
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q -k "remote_clean or delete_failure"`
Expected: FAIL, `AttributeError: ... 'delete_stage'`.

- [ ] **Step 3: Implement deletion and wire it into `run_all`**

```python
def delete_stage(repo_id: str, path_in_repo: str) -> None:
    print(f"[hf] deleting {repo_id}/{path_in_repo}", flush=True)
    _hf_api().delete_folder(
        repo_id,
        path_in_repo=path_in_repo,
        repo_type="dataset",
        commit_message=f"Remove paired prep {path_in_repo}",
    )
    print("[hf] delete complete", flush=True)
```

After `_upload_final(ns, train_dir)` at the end of `run_all`:

```python
    if offloaded and not ns.keep_hf_clean:
        repo = _clean_hf_repo(ns)
        if repo:
            try:
                delete_stage(repo, clean_stage.name)
            except Exception as exc:
                print(f"[hf] WARNING: could not delete remote clean: {exc}", flush=True)
```

Add `allp.add_argument("--keep-hf-clean", action="store_true")` to the `all` subparser.

- [ ] **Step 4: Run the task tests**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_paired_cache.py tests/test_prepare_paired_cache.py
git commit -m "feat(prep): remove remote clean half after assembly"
```

---

### Task 4: Document the public-repo workflow and verify

**Files:**
- Modify: `.env.example`, `PRODUCTION.md`

**Interfaces:**
- Consumes: `ASR_PREP_HF_PUBLIC`, `ASR_PREP_HF_PRIVATE_LIMIT_GB`, `--keep-hf-clean`.

- [ ] **Step 1: Add the env keys to `.env.example`**

Under the prepared-cache section:

```dotenv
# Publish the prep cache to a PUBLIC dataset repo (free tier). The repo is
# created public when it does not exist; an existing repo's visibility is
# never changed. Default is private, which the free 100GB limit cannot hold
# (~168GB final cache). Override the assumed limit if needed:
# ASR_PREP_HF_PUBLIC=1
# ASR_PREP_HF_PRIVATE_LIMIT_GB=100
```

- [ ] **Step 2: Update the prep section of `PRODUCTION.md`**

State: public repo via `ASR_PREP_HF_PUBLIC=1`; `auto` evicts global cache, then per-file `posix_fadvise`, then HF; `clean/` is deleted from the repo after assembly unless `--keep-hf-clean`; reruns add LFS objects until squash.

- [ ] **Step 3: Run the full targeted suite**

Run: `.venv/bin/python -m pytest tests/test_prepare_paired_cache.py tests/test_train_lora_run_stages.py tests/test_data_prep.py -q`
Expected: PASS, no failures.

- [ ] **Step 4: Sanity-check the CLI**

Run: `.venv/bin/python scripts/prepare_paired_cache.py all --help | grep -E "public-repo|keep-hf-clean|clean-strategy"`
Expected: all three flags listed.

- [ ] **Step 5: Commit and push**

```bash
git add .env.example PRODUCTION.md docs/superpowers
git commit -m "docs(prep): public repo and remote clean cleanup"
git push origin main
```
