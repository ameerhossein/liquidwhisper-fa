# Stateful Cockpit Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every remaining non-blocking finding from the stateful-cockpit final review: server scalability/observability, the CLI run-directory contract, merge/preflight polish, fake-tooling fidelity, and UI state races.

**Architecture:** Keep the existing file-based run ledger and APIs. Add small caches/limits around the 1-second status/WebSocket path, make CLI `--run_dir` strict and self-contained, make fake-run behavior faithful, and extract a few pure UI helpers for testability. No schema changes.

**Tech Stack:** Python 3, FastAPI, stdlib, vanilla JS, Node tests, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-stateful-cockpit-design.md`
**Prior plan:** `docs/superpowers/plans/2026-10-31-stateful-cockpit.md`

## Global Constraints

- CPU-only tests; no CUDA or real training.
- Keep existing API/status/WebSocket fields backward-compatible.
- Do not change the run-state schema or the one-active-run rule.
- All `run.json` writes go through `run_state`.
- Preserve smoke-test isolation.
- Use `.venv/bin/python -m pytest` and `node` for JS tests.
- The `--run_dir` CLI path must be canonical and under the resolved runs root;
  reject anything else with a clear error and document it.

---

## Task 1: Server scalability, exact step progress, and recovery diagnostics

**Files:**
- Modify: `server/app.py`
- Modify: `src/trainer_callbacks.py`
- Modify: `src/run_state.py`
- Test: `tests/test_server_train.py`
- Test: `tests/test_trainer_steps.py`
- Test: `tests/test_run_state.py`

**Fixes:**
1. Add a short TTL cache for the run scan used by `/api/status` and `/ws/live`
   (e.g. `_refresh_runs_cached()`, 1.5s). Lifecycle actions and the monitor use
   a forced refresh. Cache `_merge_running()`'s no-child scan for ~2s.
2. Make the active-run step count exact: maintain a set of seen step numbers in
   `WebTelemetryCallback` (loaded once from `steps.jsonl`), increment only for a
   new step, include `step_count` in every telemetry snapshot, and use it for
   `_steps_summary()` on the active run. Keep the manifest flush every 10 steps.
3. Rate-limit `_log_exception(context)` to once per context per 60s with a
   suppressed-repeat count.
4. Log, instead of silently swallowing, a `_finish_merge` failure to clear
   `merge_pid`.
5. Surface duplicate live trainer PIDs: `reconcile_all` returns the duplicate
   run ids (or the server detects them in the cached scan) and logs a warning.
6. Distinguish "merge artifacts predate the attempt" from "stage start time is
   missing" in `_finish_merge` diagnostics.
7. Honor a run's `paths.results_dir` for telemetry, benchmarks, loss history and
   the stop signal in `_run_paths`; keep the log and PID under the run dir.

**Tests:**
- cached scan is reused within TTL and forced after a lifecycle action;
- a resumed/replayed step does not inflate telemetry `step_count`;
- `_steps_summary` uses the active telemetry count and the manifest for history;
- repeated `_log_exception` calls rate-limit;
- duplicate live PID produces a warning;
- merge freshness detail distinguishes missing `started_at`;
- `_run_paths` uses a custom `paths.results_dir` for telemetry/stop signal.

---

## Task 2: CLI run-directory contract and trainer polish

**Files:**
- Modify: `scripts/train_lora.py`
- Test: `tests/test_train_lora_run_stages.py`
- Test: `tests/test_script_defaults.py`
- Modify: `README.md` (CLI section)

**Fixes:**
1. Reject a `--run_dir` whose name is not a canonical run id or whose parent is
   not the resolved runs root. Print a clear error telling the user to use
   `ASR_RUNS_DIR`/a canonical id. Remove the non-canonical temporary-ledger
   fallback.
2. Move the CLI run/attempt/PID setup inside the finalization guard so a setup
   failure closes any open attempt and clears `current_pid`.
3. Complete `_cli_run_config`: add `eval_batch_size`, `resume_from_checkpoint`,
   `telephony_profiles`, `telephony_seed`, `telephony_use_ffmpeg`,
   `phone_benchmark`, `phone_seed`, `lora_r`, `lora_alpha`, `lora_dropout`,
   `warmup_steps`, `lr_scheduler_type`, `save_total_limit`,
   `run_post_test_eval`, `run_epoch_benchmark`, `enable_telemetry`.
4. Preflight stage duration: preserve `started_at` when updating the stage from
   "preparing datasets" to "measuring baseline". Add a `preserve_started` option
   to `run_state.set_stage` or update the detail through `update_run`.
5. `_halt_run` must raise immediately on an unknown stage instead of falling
   back to index 0.
6. Extract the duplicated preflight failure marking into one helper.
7. Post-eval-only halt before the stage: pass `run_dir` to
   `_write_telemetry_status` so step/max_steps are backfilled.
8. Plain CLI (no ledger) post-eval writes active `evaluating_benchmark`
   telemetry before the passes and a final `completed`/`stopped` snapshot.

**Tests:**
- non-canonical/outside-root `--run_dir` is rejected;
- setup failure finalizes the attempt;
- config contains the full field set;
- preflight `started_at` is preserved;
- unknown `_halt_run` stage raises;
- post-eval-only pre-stage halt telemetry carries the last step;
- plain CLI post-eval telemetry status.

---

## Task 3: Fake trainer fidelity

**Files:**
- Modify: `scripts/dev_fake_trainer.py`
- Test: `tests/test_fake_trainer.py`

**Fixes:**
1. `_fail` merges failure fields into the existing telemetry snapshot so
   `step`, `max_steps`, `epoch`, `loss`, `gpu`, etc. survive.
2. Honor `--fresh_start` by clearing the fake results artifacts and stop signal.
3. Finalize the fake attempt when it owns the run (not cockpit-spawned).
4. Guard `_seed_benchmarks`/`_record_final_benchmark` against non-dict rows.
5. Add a test where `--results_dir` differs from `--run_dir`.

**Tests:**
- failure telemetry preserves step/gpu;
- fresh start clears old fake artifacts;
- direct fake run closes its attempt;
- malformed benchmark rows do not crash;
- separate results dir writes the expected files.

---

## Task 4: UI race, pure helpers, and coverage

**Files:**
- Modify: `server/static/app.js`
- Modify: `server/static/run_view.js`
- Modify: `server/static/style.css` (only if the reduced-motion test needs it)
- Test: `tests/test_run_view.js`
- Test: `tests/test_web_ui_contract.py`

**Fixes:**
1. `refreshRunList()` must not overwrite a user's in-flight picker selection:
   capture a selection generation/current `selectedRunId`, and bail on the
   response if the user changed it.
2. Extract `formValuesFromConfig(config)` and `runViewState(run, adapterExists)`
   (resume/merge visibility) into `run_view.js`; use them in `restoreRunConfig`
   and `renderRun`; add Node tests.
3. Strengthen the reduced-motion contract test so the new stepper/merge
   transitions are covered by the global reduced-motion rule.
4. Add a test that the static `data-stage` values match `STAGE_ORDER` (if not
   already covered).

**Tests:**
- selection race unit test;
- config mapping test (including missing/extra keys);
- run view state test for resume/merge/hidden;
- reduced-motion selector test;
- stage order contract test.

---

## Task 5: Docs and final verification

**Files:**
- Modify: `README.md`
- Modify: `GUIDE.md` (only if CLI behavior is documented there)
- Modify: `PRODUCTION.md` (only if paths changed)
- Test: full suite + Node + fake smoke

**Fixes:**
- Document the canonical `--run_dir` requirement, explicit `--results_dir`
  behavior, exact step count, and fake `--fresh_start`.
- Run `.venv/bin/python -m pytest -q`, both Node tests, and a fake-run smoke.
- Report any warning that remains and why it is third-party.

---

## Final verification

- Full pytest suite passes.
- Both Node suites pass.
- Fake run: start, reload/restart restore, stop, resume, merge.
- No test requires CUDA.
- The working tree is clean and all commits are on `main` (user chose main).
