# Design: Default Hugging Face Artifact Uploads and Mirroring

**Date:** 2026-11-01  
**Status:** Pending Approval  
**Base Repository Name:** `liquidwhisper-neyshekar`  

---

## 1. Intent and Why

Currently, running Hugging Face artifact uploads (`scripts/upload_to_hf.py`) or enabling live run mirroring (`scripts/train_lora.py`, `server/app.py`, `scripts/merge_lora.py`) requires explicitly passing `--repo-prefix`, `--hf_run_repo`, or setting `ASR_HF_RUN_REPO` in the environment. If omitted, `upload_to_hf.py` exits with an error and training runs operate without HF mirroring.

We want a seamless default experience: when a user is authenticated with Hugging Face (via `hf auth login` or `HF_TOKEN`), all artifact uploads, live run mirrors, and merges automatically default to the authenticated account's namespace under **`liquidwhisper-neyshekar`** without requiring manual configuration.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Auto-default repository namespace**: If no repository is specified, resolve the repository base name to `<username>/liquidwhisper-neyshekar` using the active Hugging Face authentication.
- **Artifact uploads (`scripts/upload_to_hf.py`)**:
  - Automatically derives:
    - Merged model: `<username>/liquidwhisper-neyshekar`
    - LoRA adapter: `<username>/liquidwhisper-neyshekar-lora`
    - Checkpoints: `<username>/liquidwhisper-neyshekar-checkpoints`
  - Uploads proceed without requiring `--repo-prefix`.
- **Training runs (`scripts/train_lora.py` and Web UI start)**:
  - If authenticated and no repo is specified, automatically mirrors steps, adapter snapshots, and run ledgers to `<username>/liquidwhisper-neyshekar`.
- **Merged models (`scripts/merge_lora.py` and Web UI merge)**:
  - If authenticated and no repo is specified, automatically uploads merged models to `<username>/liquidwhisper-neyshekar`.
- **Web UI cockpit (`server/app.py`)**:
  - `/api/hf/config` returns `<username>/liquidwhisper-neyshekar` by default when authenticated.
- **Clean opt-out**:
  - Setting `ASR_HF_RUN_REPO=off` (or `none`, `false`, `0`) or passing `--no-hf-sync` explicitly disables mirroring even when authenticated.
- **Unauthenticated resilience**:
  - If unauthenticated (no token or offline), training and merging run locally without error; `upload_to_hf.py` displays clear remediation prompting the user to log in or supply `--repo-prefix`.

### Non-Goals
- Changing explicit overrides: Any explicit CLI flag (`--hf_run_repo`, `--repo-prefix`, `--merged-repo`) or valid `ASR_HF_RUN_REPO` always takes precedence over the default.
- Changing smoke test isolation: Smoke test continues to use `<username>/stt-smoke-test` to avoid polluting the main model repos.

### Anti-Goals
- Never break local training or crash offline workflows when Hugging Face credentials are absent.
- Never attempt to push to unauthenticated or nonexistent organization namespaces by default.

---

## 3. Approaches Considered

### Approach A: Centralized Resolution in `src/hf_sync.py` (Selected)
Centralize repository resolution in `src/hf_sync.py`:
- Define `DEFAULT_HF_REPO_NAME = "liquidwhisper-neyshekar"`.
- Provide `resolve_default_repo(api=None)` which checks `ASR_HF_RUN_REPO`, inspects `get_authenticated_user()`, and handles opt-outs (`off`, `none`).
- Use this in `resolve_config()` so `train_lora.py`, `merge_lora.py`, `upload_to_hf.py`, and `server/app.py` share consistent logic.
- **Trade-off**: Requires updating `resolve_config()` to query the authenticated user when repo is unset, while keeping offline/unauthenticated execution completely silent and safe.

### Approach B: Script & Web UI Specific Ad-hoc Defaults
Leave `resolve_config()` requiring an explicit repo, and only insert the default into `server/app.py` and `scripts/upload_to_hf.py`.
- **Trade-off**: CLI executions of `train_lora.py` would not mirror unless `ASR_HF_RUN_REPO` was set, causing behavior divergence between the Web UI and command-line training.

**Decision**: Approach A is selected for consistency across all CLI and Web UI interfaces.

---

## 4. Detailed Specification

### 4.1 Constants and Helpers in `src/hf_sync.py`
- `DEFAULT_HF_REPO_NAME = "liquidwhisper-neyshekar"`
- `DISABLE_HF_VALUES = frozenset(("off", "none", "false", "0", "disabled", "no"))`
- `resolve_default_repo(api: Optional[Any] = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]`:
  1. Check `environ.get("ASR_HF_RUN_REPO")` (or `os.environ`):
     - If lowercase value in `DISABLE_HF_VALUES`: return `None` (explicit opt-out).
     - If non-empty valid repo id: return it.
  2. Query `get_authenticated_user(api)`. If returned `<username>`:
     - Return `f"{username}/{DEFAULT_HF_REPO_NAME}"`.
  3. Otherwise: return `None`.

### 4.2 Updating `resolve_config()` in `src/hf_sync.py`
- If `--no-hf-sync` or `getattr(args, "no_hf_sync", False)` is True: return `None`.
- If explicit `args.hf_run_repo`:
  - If in `DISABLE_HF_VALUES`: return `None`.
  - Validate and use it.
- If no explicit `args.hf_run_repo`:
  - Call `resolve_default_repo(environ=env)`.
  - If resolved, return `HFSyncConfig(repo_id=resolved_repo, ...)`.
  - Else return `None`.

### 4.3 Updating `scripts/upload_to_hf.py`
- In `main()` / `derive_repo_ids()`:
  - If `prefix` is not passed and explicit artifact repos are missing:
    - Attempt to resolve default prefix via `resolve_default_repo(api)`.
    - If resolved: use prefix `f"{username}/liquidwhisper-neyshekar"`.
    - If unauthenticated: raise `SystemExit` with clear message:
      `"Missing repo id. Authenticate with 'hf auth login' (or set HF_TOKEN), or pass --repo-prefix <user>/<prefix>."`

### 4.4 Updating `server/app.py`
- In `get_hf_config()`:
  - Return `repo_id`: `hf_sync.resolve_default_repo() or ""` so the Web UI Cockpit display and recovery modal pre-populate with `<username>/liquidwhisper-neyshekar`.

---

## 5. What's Left to the Builder
- Write failing unit tests for `resolve_default_repo()` and updated `resolve_config()` in `tests/test_hf_sync.py`.
- Write failing unit tests for `upload_to_hf.py` default repo resolution in `tests/test_upload_to_hf.py`.
- Write tests for CLI / server default mirroring in `tests/test_train_lora_hf.py` and `tests/test_server_train.py`.
- Verify the full test suite remains green.
