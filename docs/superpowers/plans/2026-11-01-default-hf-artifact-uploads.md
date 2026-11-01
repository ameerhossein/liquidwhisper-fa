# Default Hugging Face Artifact Uploads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable automatic Hugging Face artifact uploads, live run mirroring, and merge uploads defaulting to `<username>/liquidwhisper-neyshekar` whenever an authenticated user is detected and no repository is explicitly specified.

**Architecture:** Centralize default repository resolution in `src/hf_sync.py` via `resolve_default_repo()`, inspect active HF authentication (`get_authenticated_user()`), support explicit opt-out values (`off`, `none`, `0`), and wire this default into `resolve_config()` (used by `scripts/train_lora.py` and `scripts/merge_lora.py`), `scripts/upload_to_hf.py`, and `server/app.py`.

**Tech Stack:** Python 3.10+, FastAPI, Hugging Face Hub (`huggingface_hub`), Pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-default-hf-artifact-uploads-design.md`

## Global Constraints
- Python 3.10+ standard library and existing dependencies only (no new pip packages).
- Keep all unit tests free of real network calls (all Hugging Face API interactions mocked in tests).
- Default base repository name is verbatim: `liquidwhisper-neyshekar`.
- Offline/unauthenticated training runs must remain completely silent and successful locally with zero crashes or forced errors.
- Explicit CLI flags (`--hf_run_repo`, `--repo-prefix`) and valid `ASR_HF_RUN_REPO` env vars always take precedence over the default.

## Review Focus
- Unauthenticated environment: `resolve_config()` and `resolve_default_repo()` must return `None` so plain training proceeds locally without error.
- Explicit opt-out: Setting `ASR_HF_RUN_REPO=off` (or `none`, `false`, `0`) or `--hf_run_repo off` must return `None` and disable mirroring even when authenticated.
- `scripts/upload_to_hf.py` without flags: Must auto-populate all three repositories (`<user>/liquidwhisper-neyshekar`, `<user>/liquidwhisper-neyshekar-lora`, `<user>/liquidwhisper-neyshekar-checkpoints`) when authenticated.
- `scripts/upload_to_hf.py` unauthenticated: Must raise `SystemExit` with clear guidance on authenticating or passing `--repo-prefix`.
- Web UI `/api/hf/config`: Must return the default `<user>/liquidwhisper-neyshekar` in `repo_id` when authenticated and no explicit env var is set.

---

### Task 1: Add Default Repository Resolution and Opt-Out to `src/hf_sync.py`

**Files:**
- Modify: `src/hf_sync.py:150-205`
- Test: `tests/test_hf_sync.py`

**Interfaces:**
- Produces:
  - `DEFAULT_HF_REPO_NAME = "liquidwhisper-neyshekar"`
  - `DISABLE_HF_VALUES = frozenset(("off", "none", "false", "0", "disabled", "no"))`
  - `resolve_default_repo(api: Optional[Any] = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]`
  - Updated `resolve_config(args: Any = None, environ: Mapping[str, str] | None = None) -> HFSyncConfig | None`

- [ ] **Step 1: Write failing tests in `tests/test_hf_sync.py`**

Add tests for:
- `test_resolve_default_repo_returns_authenticated_user_repo`: returns `f"alice/{DEFAULT_HF_REPO_NAME}"` when `get_authenticated_user` returns `"alice"`.
- `test_resolve_default_repo_returns_none_when_unauthenticated`: returns `None` when `get_authenticated_user` returns `None`.
- `test_resolve_default_repo_honors_explicit_env`: returns `"myorg/custom-repo"` when `ASR_HF_RUN_REPO` is set.
- `test_resolve_default_repo_honors_opt_out_env`: returns `None` when `ASR_HF_RUN_REPO="off"`, `"none"`, or `"0"`.
- `test_resolve_config_defaults_to_authenticated_user_repo`: `resolve_config()` returns config with `repo_id="alice/liquidwhisper-neyshekar"` when no repo is passed and user is authenticated.
- `test_resolve_config_is_off_when_unauthenticated_and_no_repo`: returns `None` when unauthenticated and no env/cli repo.
- `test_resolve_config_honors_cli_opt_out`: returns `None` when `args.hf_run_repo="off"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_sync.py -k "test_resolve_default_repo or test_resolve_config_defaults" -v`
Expected: FAIL (missing `resolve_default_repo` / fails to default to authenticated user repo)

- [ ] **Step 3: Implement `resolve_default_repo` and update `resolve_config` in `src/hf_sync.py`**

- Define `DEFAULT_HF_REPO_NAME = "liquidwhisper-neyshekar"`
- Define `DISABLE_HF_VALUES = frozenset(("off", "none", "false", "0", "disabled", "no"))`
- Implement `resolve_default_repo(api: Optional[Any] = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]`:
  1. Inspect `env.get("ASR_HF_RUN_REPO", "")`.
  2. If in `DISABLE_HF_VALUES`: return `None`.
  3. If non-empty: validate and return it.
  4. Query `get_authenticated_user(api)`.
  5. If user found: return `f"{user}/{DEFAULT_HF_REPO_NAME}"`.
  6. Return `None`.
- In `resolve_config()`:
  1. Check CLI `args.hf_run_repo`: if non-empty string in `DISABLE_HF_VALUES`, return `None`.
  2. If explicit CLI `args.hf_run_repo`, validate and use it.
  3. If CLI repo not passed, call `resolve_default_repo(environ=env)`.
  4. If resolved repo is None, return `None`.
  5. Otherwise build and return `HFSyncConfig`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hf_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hf_sync.py tests/test_hf_sync.py
git commit -m "feat(hf_sync): add default repo resolution to liquidwhisper-neyshekar with opt-out"
```

---

### Task 2: Auto-Default Repository in `scripts/upload_to_hf.py`

**Files:**
- Modify: `scripts/upload_to_hf.py:237-271,440-460`
- Test: `tests/test_upload_to_hf.py`

**Interfaces:**
- Consumes: `resolve_default_repo` from `src.hf_sync`
- Produces: Updated `derive_repo_ids(prefix, merged_repo, adapter_repo, checkpoints_repo, artifacts, api=None) -> Dict[str, str]`

- [ ] **Step 1: Write failing tests in `tests/test_upload_to_hf.py`**

Add tests:
- `test_derive_repo_ids_auto_resolves_when_prefix_omitted`: when `prefix=None` and `api` is authenticated as `"bob"`, returns dictionary with:
  - `"merged": "bob/liquidwhisper-neyshekar"`
  - `"best_adapter": "bob/liquidwhisper-neyshekar-lora"`
  - `"checkpoint": "bob/liquidwhisper-neyshekar-checkpoints"`
- `test_derive_repo_ids_raises_clear_error_when_unauthenticated_and_no_prefix`: raises `SystemExit` with text mentioning `"authenticate with 'hf auth login'"` or pass `"--repo-prefix"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_upload_to_hf.py -k "test_derive_repo_ids_auto_resolves" -v`
Expected: FAIL

- [ ] **Step 3: Update `scripts/upload_to_hf.py`**

- Import `resolve_default_repo` from `src.hf_sync`.
- In `derive_repo_ids(prefix, merged_repo, adapter_repo, checkpoints_repo, artifacts, api=None)`:
  - If `not prefix` and any artifact in `wanted` is missing from `out`:
    - Attempt `default_prefix = resolve_default_repo(api=api)`.
    - If `default_prefix`:
      - Set `prefix = default_prefix` and populate `"merged"`, `"best_adapter"`, `"checkpoint"`.
  - If still missing:
    - Raise `SystemExit("Missing repo id. Authenticate with 'hf auth login' (or set HF_TOKEN), or pass --repo-prefix <user>/<prefix>.")`.
- In `main()`:
  - Move `api` initialization before `derive_repo_ids()` call so `api` can be passed to `derive_repo_ids(..., api=api)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_upload_to_hf.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/upload_to_hf.py tests/test_upload_to_hf.py
git commit -m "feat(upload): default artifact upload repos to liquidwhisper-neyshekar when unconfigured"
```

---

### Task 3: Auto-Default Repository in Web UI Cockpit and Merge Workflows

**Files:**
- Modify: `server/app.py:2635-2670,2915-2935,3190-3205`
- Test: `tests/test_server_train.py`, `tests/test_hf_restore_server.py`

**Interfaces:**
- Consumes: `resolve_default_repo` from `src.hf_sync`

- [ ] **Step 1: Write failing tests in `tests/test_server_train.py` and `tests/test_hf_restore_server.py`**

- In `tests/test_hf_restore_server.py`:
  - `test_hf_config_returns_default_repo_when_authenticated`: when `get_authenticated_user` returns `"carol"` and `ASR_HF_RUN_REPO` is unset, `/api/hf/config` returns `repo_id="carol/liquidwhisper-neyshekar"`.
- In `tests/test_server_train.py`:
  - `test_start_run_passes_default_hf_repo_when_authenticated`: when starting a new training run via `/api/train/start` and authenticated as `"carol"`, trainer command includes `--hf_run_repo carol/liquidwhisper-neyshekar`.
  - `test_merge_command_uses_default_hf_repo_when_authenticated`: when merging without a recorded repo or `ASR_HF_RUN_REPO`, merge command passes `--hf_run_repo carol/liquidwhisper-neyshekar` when authenticated.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_restore_server.py -k "test_hf_config_returns_default_repo" tests/test_server_train.py -k "test_start_run_passes_default_hf_repo" -v`
Expected: FAIL

- [ ] **Step 3: Update `server/app.py`**

- In `get_hf_config()`:
  - Default `repo_id` to `(os.environ.get("ASR_HF_RUN_REPO") or "").strip() or (hf_sync.resolve_default_repo() or "")`.
- In `_start_new_run()`:
  - If `trainer_args.get("hf_run_repo")` is not specified:
    - Attempt `default_repo = hf_sync.resolve_default_repo()`.
    - If `default_repo`:
      - Set `trainer_args["hf_run_repo"] = default_repo`.
      - Set `run_config["hf_run_repo"] = default_repo`.
- In `_start_merge()`:
  - If `not hf_repo`:
    - `hf_repo = hf_sync.resolve_default_repo() or ""`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hf_restore_server.py tests/test_server_train.py -k "default_hf_repo" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_server_train.py tests/test_hf_restore_server.py
git commit -m "feat(server): default cockpit runs, merges, and HF config to liquidwhisper-neyshekar"
```

---

### Task 4: Full Suite Verification and Documentation Alignment

**Files:**
- Modify: `.env.example`, `GUIDE.md` (if repo default documentation is mentioned)
- Test: All tests

- [ ] **Step 1: Update `.env.example`**

Clarify in `.env.example` that `ASR_HF_RUN_REPO` defaults to `<authenticated_user>/liquidwhisper-neyshekar`, and setting it to `off` disables mirroring.

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests PASS with 0 failures.

- [ ] **Step 3: Verify working tree clean**

Run: `git status`
Expected: Clean or only expected tracked modifications.

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: document default liquidwhisper-neyshekar HF repo and opt-out values"
```
