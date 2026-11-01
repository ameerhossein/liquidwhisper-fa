# Smoke Test Mini Training Run with Static HF Mirroring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Cockpit smoke test act like a mini training run by mirroring to a hardcoded static Hugging Face repository, resetting that repository before each smoke run, and isolating smoke runs from production Cockpit ledgers.

**Architecture:** Add repository reset helper in `src/hf_sync.py` that deletes and re-creates (or clears) the target Hugging Face repository. In `server/app.py`, configure the static smoke repo (`shekar-ai/stt-smoke-test`), reset it on every smoke test initiation, clean up scratch run state, pass `--hf_run_repo` and `--hf_adapter_snapshot_steps 5` to the trainer command, and scope runs root via `ASR_RUNS_DIR` in the subprocess environment to prevent polluting the primary Cockpit run list.

**Tech Stack:** Python 3.10+, FastAPI, Hugging Face Hub (`huggingface_hub`), Pytest.

**Spec:** Inline design approved in conversation: hardcoded static repository `shekar-ai/stt-smoke-test` (overridable via `ASR_SMOKE_HF_REPO` or request config), reset on every smoke start, runs like a mini training run exercising HF mirroring end-to-end.

## Global Constraints
- Python 3.10+ standard library and existing dependencies only (no new packages).
- Keep all unit tests free of real network calls (all Hugging Face API interactions mocked in tests).
- Smoke test isolation must be preserved: primary production directories, checkpoints, and `RUNS_DIR` ledger runs must never be touched or polluted by smoke runs.
- Hugging Face repository operations must fail gracefully with informative error messages if credentials or permissions are missing.

## Review Focus
- Repository reset when repo does not exist yet: `missing_ok=True` on `delete_repo` must not raise.
- Repository reset fallback when delete is forbidden: must fall back to deleting repository files.
- Subprocess environment isolation: `ASR_RUNS_DIR` must point to `<smoke_results_dir>/runs` so ledger runs remain strictly isolated.
- Authentication/Network error on smoke start: HTTP 500 with descriptive detail when HF reset fails.
- Existing smoke test and server workflows remain backward-compatible and functional.

---

### Task 1: Add `reset_repo` helper to `src/hf_sync.py` and unit tests

**Files:**
- Modify: `src/hf_sync.py`
- Test: `tests/test_hf_sync.py`

**Interfaces:**
- Produces: `reset_repo(repo_id: str, api: Optional[Any] = None, private: bool = True) -> None`

- [ ] **Step 1: Write failing tests for `reset_repo`**

In `tests/test_hf_sync.py`, test:
- Successful delete and create with mock API.
- Missing repo handling (`missing_ok=True` or `delete_repo` 404).
- Fallback to deleting repo files if `delete_repo` raises permission error.
- Invalid repo ID validation (`validate_repo_id`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hf_sync.py -k test_reset_repo -v`
Expected: FAIL (AttributeError: module 'src.hf_sync' has no attribute 'reset_repo')

- [ ] **Step 3: Implement `reset_repo` in `src/hf_sync.py`**

Implement `reset_repo(repo_id: str, api: Optional[Any] = None, private: bool = True) -> None`:
- Call `validate_repo_id(repo_id)`
- If `api is None`, acquire via `make_api()`
- Call `api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)`
- If `delete_repo` fails, fall back to listing files and calling `api.delete_files`
- Call `api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hf_sync.py -k test_reset_repo -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hf_sync.py tests/test_hf_sync.py
git commit -m "feat(hf_sync): add reset_repo helper with delete and fallback cleanup"
```

---

### Task 2: Wire HF reset and mirroring into smoke test in `server/app.py`

**Files:**
- Modify: `server/app.py`
- Test: `tests/test_server_train.py`

**Interfaces:**
- Consumes: `reset_repo` from `src.hf_sync`
- Produces: `SMOKE_DEFAULT_HF_REPO`, `SMOKE_HF_REPO`, `_reset_smoke_repo(repo_id: str)`

- [ ] **Step 1: Write failing tests for smoke test HF reset and subprocess args**

In `tests/test_server_train.py`, add:
- `test_smoke_start_resets_hf_repo_and_passes_flags`: asserts `_reset_smoke_repo` is called, `--hf_run_repo` and `--hf_adapter_snapshot_steps 5` are present in `cmd`, and `ASR_RUNS_DIR` is set in subprocess env.
- `test_smoke_start_fails_when_hf_reset_fails`: asserts HTTP 500 when `_reset_smoke_repo` raises.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -k "test_smoke_start_resets_hf_repo" -v`
Expected: FAIL

- [ ] **Step 3: Implement smoke test HF reset and mirror wiring in `server/app.py`**

- Define `SMOKE_DEFAULT_HF_REPO = "shekar-ai/stt-smoke-test"`
- Define `SMOKE_HF_REPO = os.environ.get("ASR_SMOKE_HF_REPO", SMOKE_DEFAULT_HF_REPO).strip() or SMOKE_DEFAULT_HF_REPO`
- Add `_reset_smoke_repo(repo_id: str) -> None`: calls `hf_sync.reset_repo(repo_id)`
- In `start_smoke_test(config)`:
  - Extract `target_repo = str(config.get("hf_repo", "")).strip() or SMOKE_HF_REPO`
  - Call `_reset_smoke_repo(target_repo)` and handle exceptions by raising HTTP 500 with clear message
  - Remove previous smoke runs scratch dir (`shutil.rmtree(smoke_results_dir / "runs", ignore_errors=True)`)
  - Append `--hf_run_repo`, target_repo, `--hf_adapter_snapshot_steps`, "5" to `cmd`
  - Pass `env = {**os.environ, "ASR_RUNS_DIR": str(smoke_results_dir / "runs")}` to `subprocess.Popen`
  - Include `"hf_repo": target_repo` in response dictionary
- In `get_smoke_status()`:
  - Include `"hf_repo": SMOKE_HF_REPO` in status payload
- In `client` fixture of `tests/test_server_train.py`:
  - Mock `_reset_smoke_repo` so existing test suite doesn't make real HF network calls

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_train.py -k "smoke" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_server_train.py
git commit -m "feat(server): run smoke test as mini training run with static HF mirror and reset"
```

---

### Task 3: Full test suite verification

**Files:**
- Test: All test suites

- [ ] **Step 1: Run pytest across the whole repository**

Run: `.venv/bin/python -m pytest -q`
Expected: All tests PASS.

- [ ] **Step 2: Verify git status is clean and documentation/types are intact**

Run: `git status`
Expected: Clean working tree.
