# Stateful cockpit with a persistent run ledger — design

**Date:** 2026-10-30  
**Status:** Draft for approval  
**Scope:** Make the RunPod training cockpit survive page reloads, server restarts,
and pod restarts; persist every optimizer-step record and every pipeline stage in
one self-contained run directory on `/workspace`.

---

## Intent

Today the cockpit reads a single set of global files under `results/`
(`live_telemetry.json`, `loss_history.json`, `epoch_benchmarks.json`,
`training_live.log`, `training.pid`, `stop_training.signal`). A new run overwrites
or merges into the previous run's state, the form config is never persisted, and
the server's `halt_requested` / `last_exit_code` values exist only in memory.
After a pod or server restart the UI can see a live PID but cannot reconstruct
which run, config, or stage produced it.

This change gives each launch one persistent run directory. A run is the whole
pipeline a user expects from **Start**:

**preflight baseline → training → post-training evaluation**

**Merge** is a later, optional stage of the same run, started from the cockpit
after training. Preflight and post-eval stay inside the existing trainer process;
the cockpit records their stages rather than orchestrating three separate jobs.

Every optimizer step appends a durable metrics record. Model weights are still
checkpointed at `--save_steps`; per-step weight checkpoints are explicitly out of
scope.

## Goals

1. One self-contained run directory per **Start**, under persistent storage.
2. Persist the normalized launch config **before** the trainer is spawned.
3. Append one JSONL record per optimizer step, uncapped (the existing
   `loss_history.json` 500-point cap remains only for the chart/back-compat).
4. Record stage transitions for `preflight`, `training`, `post_eval`, `merge`,
   including attempt/exit history.
5. Rehydrate the latest run on server startup and restore its config, stage,
   progress, and saved-step count in the UI after a reload.
6. Resume an interrupted run with the same paths and checkpoint, and let the UI
   distinguish `completed`, `stopped`, `failed`, and `interrupted`.
7. Run merge as a tracked stage of the same run, with its own log and status.
8. Make it obvious when the runs directory is **not** on RunPod persistent
   storage.
9. Keep `scripts/train_lora.py` usable from the CLI and keep smoke tests isolated
   and outside the run ledger.

## Non-goals

- Multiple simultaneous full runs. One full run at a time, plus the existing
  isolated smoke test.
- A job queue, scheduler, SQLite database, or multi-pod orchestration.
- Per-step model checkpoints (only per-step metrics/telemetry are persisted).
- Changing preflight metrics, training hyperparameters, LoRA settings, or the
  epoch-benchmark algorithm.
- Automatically deleting old runs. Retention is manual.
- A transcription stage or a general artifact browser.
- Making smoke tests appear in run history.

## Anti-goals

- A run that still overwrites another run's telemetry or epoch rows.
- A UI that shows an active run after a restart but cannot say which config it
  used.
- Falling back silently to hardcoded baselines when a run has no preflight.
- Losing the last optimizer step because the browser was closed.
- Treating a failed merge as a failed training run.

## Constraints

- `/workspace` (or `/runpod-volume`) is the only RunPod persistent root. The
  repo checkout, `~/.cache`, and `/tmp` may be ephemeral.
- Existing managed paths live in `src/paths.py`: `ASR_OUTPUT_DIR`,
  `ASR_RESULTS_DIR`, `ASR_MERGED_DIR`, with `/workspace`-aware defaults.
- The cockpit is a single FastAPI process; the trainer is a child process. Both
  must safely share the run manifest.
- Telemetry is already file-based and polled once per second; the new state must
  fit that model without a database.
- The existing WebSocket payload and `/api/status` fields must keep working.
- Smoke tests use isolated `smoke-test/` paths and must continue to do so.
- Preflight is mandatory and runs inside `scripts/train_lora.py`; its cache is
  expensive to rebuild and should be reused across runs.
- Development and CI are GPU-free. The server, ledger, APIs, and UI must run on a
  CPU-only machine; training-stage behavior is exercised with a fake trainer, not
  a real GPU job. Real end-to-end training remains a RunPod acceptance step.

## Approaches considered

1. **Run directory + JSON/JSONL ledger (chosen).** Each run owns a directory with
   `run.json`, `steps.jsonl`, `events.jsonl`, telemetry, benchmarks, logs, the
   preflight copy, checkpoints, and merged output. Server and trainer share a
   small stdlib module. No new dependency; crash-tolerant; inspectable on the
   pod; each run is self-contained.
2. **SQLite ledger + existing JSON artifacts.** Stronger querying and atomic
   transactions, but two sources of truth, a binary database to back up, more
   test surface, and cross-process lock contention for little benefit at one
   active run.
3. **Manifest over today's global files.** Smallest diff, but runs still collide,
   epoch rows still merge, and the run is not self-contained. Rejected.

## Run scope and active-run rules

- One full run may be `starting` or `running` at a time. The ledger keeps
  unlimited completed, stopped, failed, and interrupted runs.
- The **active run** is the run that owns the live trainer PID. It is discovered
  by scanning every run directory at startup and by the background reconciler,
  not by assuming the newest run is active. An older run may own the live
  trainer after a server restart.
- Live telemetry, `/api/status`, `/ws/live`, and Halt always follow the active
  run. Selecting a historical run is a read-only view; its **Resume** or
  **Merge** action is still allowed.
- Resume can target any run with status `interrupted`, `stopped`, or `failed`.
  Merge can target any run whose `training` stage completed and whose adapter
  exists. Only one merge runs at a time globally.
- A new Start is rejected while any full run, smoke test, or merge is active.

## Storage layout

The default runs root is:

- RunPod: `/workspace/stt-outputs/results/runs`
- Local fallback: `results/runs` (repo-local)

`ASR_RUNS_DIR` overrides it. The default is derived from the managed
`ASR_RESULTS_DIR`, so `runs/` inherits the existing persistence behavior.

```
<runs_root>/
  _shared/
    baselines/preflight.json        # reusable fingerprint-keyed preflight cache
  <run_id>/
    run.json                        # atomic manifest: config, stages, attempts
    events.jsonl                    # append-only stage/attempt/merge events
    steps.jsonl                     # append-only per-optimizer-step records
    live_telemetry.json             # atomic latest snapshot (existing schema)
    loss_history.json               # capped 500-point chart cache (back-compat)
    epoch_benchmarks.json           # this run's epoch matrix
    baselines/preflight.json        # copy of the preflight used by this run
    training_live.log               # current attempt's stdout/stderr
    training.pid                    # current trainer PID (crash recovery)
    stop_training.signal            # halt request consumed by the callback
    output/                         # checkpoints, best_adapter, final report
    merged/                          # merge_lora.py output
    merge.log                       # merge stdout/stderr
    merge.pid                       # merge PID (crash recovery)
```

Run ids sort chronologically and are path-safe:

```
YYYYMMDD-HHMMSS-<6 hex chars>      e.g. 20261030-194700-1a2b3c
```

The server validates `^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$` before touching a path.

### Why `output/` is run-scoped

Each run gets its own checkpoints so a new **Start** can never auto-resume a
previous run's checkpoint by accident. The web UI passes
`--output_dir <run_dir>/output`. A CLI user keeps the existing
`ASR_OUTPUT_DIR` default because CLI runs do not create a ledger unless
`--run_dir` is passed.

An explicit `output_dir` in the start payload is still honored for advanced
users; `run.json` records where it actually points. The default is run-scoped.

### Preflight cache

The preflight is expensive and fingerprint-keyed, so it stays shared rather than
being repeated for every run:

- `run_preflight` gains an optional `cache_dir` argument (default: `results_dir`,
  preserving current behavior).
- The cockpit passes a shared cache directory and, after the preflight succeeds,
  copies the matched `preflight.json` into `<run_dir>/baselines/preflight.json`
  for provenance.
- The server reads the run copy first, then the shared cache.
- A cache hit is recorded in the `preflight` stage detail (`cache hit` vs
  `measured BuzzASR + Whisper, n=…`).

## Run manifest (`run.json`)

Atomic writes: write `run.json.tmp`, `os.replace()` over `run.json`. Read-modify-
write updates take an exclusive `fcntl.flock` on `<run_dir>/.run.lock` (no-op
fallback on non-POSIX) so the server and trainer cannot lose each other's fields.
Append-only logs use `O_APPEND` and are only written by one process at a time.

```json
{
  "schema_version": 1,
  "run_id": "20261030-194700-1a2b3c",
  "created_at": 1761846420.0,
  "updated_at": 1761847020.0,
  "status": "running",
  "current_stage": "training",
  "failed_stage": null,
  "config": {
    "base_model": "BuzzASR/persian",
    "dataset_name": "shekar-ai/Neyshekar",
    "epochs": 4,
    "batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 0.0001,
    "eval_batch_size": 16,
    "eval_steps": 1000,
    "save_steps": 1000,
    "max_benchmark_samples": 1000,
    "preflight_samples": 1000,
    "num_proc": 8,
    "telephony_prob": 0.5,
    "gradient_checkpointing": false,
    "auto_resume": true,
    "fresh_start": false,
    "resume_from_checkpoint": null,
    "output_dir": "/workspace/stt-outputs/results/runs/20261030-194700-1a2b3c/output"
  },
  "trainer_args": {
    "base_model": "BuzzASR/persian",
    "epochs": 4,
    "lora_r": 64,
    "warmup_steps": 500,
    "lr_scheduler_type": "cosine",
    "save_total_limit": 2,
    "telephony_profiles": ["customer"],
    "telephony_seed": 42
  },
  "paths": {
    "run_dir": "/workspace/stt-outputs/results/runs/20261030-194700-1a2b3c",
    "output_dir": "/workspace/stt-outputs/results/runs/20261030-194700-1a2b3c/output",
    "results_dir": "/workspace/stt-outputs/results/runs/20261030-194700-1a2b3c",
    "merged_dir": "/workspace/stt-outputs/results/runs/20261030-194700-1a2b3c/merged",
    "preflight_cache": "/workspace/stt-outputs/results/runs/_shared/baselines/preflight.json"
  },
  "stages": [
    {"name": "preflight", "status": "completed", "started_at": 1761846425.0,
     "ended_at": 1761846500.0, "detail": "cache hit"},
    {"name": "training", "status": "running", "started_at": 1761846501.0,
     "ended_at": null, "detail": "attempt 1"},
    {"name": "post_eval", "status": "pending"},
    {"name": "merge", "status": "pending"}
  ],
  "attempts": [
    {"attempt": 1, "pid": 12345, "started_at": 1761846425.0,
     "ended_at": null, "exit_code": null, "reason": "initial"}
  ],
  "step_count": 1234,
  "last_step": 1234,
  "last_step_at": 1761847000.0,
  "last_error": null,
  "source": "cockpit"
}
```

- `status`: `starting | running | completed | stopped | failed | interrupted`.
- Stage `status`: `pending | running | completed | stopped | failed | skipped |
  interrupted`.
- `config` is the UI-safe summary used to repopulate the form. `trainer_args` is
  the **full effective argparse namespace**, not only the six fields the form
  exposes. Cockpit runs and CLI runs created with `--run_dir` both record it, so
  Resume cannot silently change LoRA shape, warmup, scheduler, telephony profile,
  or save limits. The UI reads a whitelisted subset of `config`.
- `step_count` / `last_step*` are refreshed by the trainer every 100 steps and at
  every stage boundary. `steps.jsonl` remains the authority for full history.
- The auth token is never written to `run.json` or `events.jsonl`.

## Step records (`steps.jsonl`)

One line per optimizer step, appended from
`WebTelemetryCallback.on_step_end`:

```json
{"step": 1234, "epoch": 1.23, "loss": 0.4213, "loss_step": 1200,
 "learning_rate": 0.00008, "speed_s_per_step": 0.41,
 "elapsed_seconds": 3600, "eta_seconds": 5400, "attempt": 1,
 "ts": 1761847000.0}
```

- `step` is the global optimizer step.
- `loss` is the latest logged loss carried forward; `loss_step` says which step
  actually produced it. The trainer logs loss every 50 steps today, so a step
  between logging points has the previous value and an honest `loss_step`.
- One line per step is written for **every** step, independent of the 500-point
  `loss_history.json` cap.
- On resume, records continue appending. Readers that build a chart take the
  last record per step, so a re-run from an earlier checkpoint cannot produce
  duplicate points.
- A malformed trailing line (crash mid-write) is ignored by readers.

## Stage lifecycle

| Stage | Who runs it | Starts | Completes when |
|---|---|---|---|
| `preflight` | trainer process | before dataset preparation | cache hit or both baseline evaluations finish |
| `training` | trainer process | before `trainer.train()` | training loop returns or a halt is consumed |
| `post_eval` | trainer process | before `_run_post_training_evaluation` | full-test report is written |
| `merge` | server child process | cockpit **Merge** click | `merge_lora.py` exits 0 and artifacts exist |

- The `preflight` stage owns pre-training setup. Its detail moves from
  `preparing datasets` to `measuring baseline`; a dataset-preparation failure
  fails this stage rather than leaving no stage responsible.
- `post_eval` is `skipped` when `--no-run-post-test-eval` is used.
- Halt applies to the whole run, not only the optimizer loop. Cooperative checks
  run before dataset preparation, before each preflight model, at each training
  step, and between the clean and phone post-eval passes. The existing
  grace-period SIGTERM escalation terminates a long subprocess if cooperation is
  too slow. A halted preflight/training/post-eval stage becomes `stopped`,
  remaining stages become `skipped`, and overall status is `stopped`. This also
  fixes the current behavior where a full post-test evaluation still runs after
  a user halt.
- If `preflight` or `training` fails, the run is `failed` with `failed_stage` set.
  If `post_eval` fails, training artifacts are kept, the run is `failed` with
  `failed_stage: post_eval`, and **Resume** can retry the remaining work.
- A failed merge does not change an otherwise `completed` run; only the
  `merge` stage becomes `failed`.
- Overall status is `completed` when training completed and post-eval is
  `completed` or `skipped`, regardless of merge.

### Resume semantics by stage

| Last state | Resume action |
|---|---|
| preflight `failed`/`stopped` | re-enter the normal trainer; cached dataset prep and preflight are reused |
| training `interrupted`/`stopped`/`failed` | re-enter the normal trainer with `--auto_resume`; continue from the newest complete checkpoint |
| training `completed`, post_eval `failed`/`interrupted`/`stopped` | spawn `train_lora.py --post_eval_only`; load `best_adapter`, skip training, rerun the full-test report |
| post_eval `completed` | no resume; Merge is available |
| merge `failed`/`interrupted` | rerun the merge endpoint; training status is untouched |

`--post_eval_only` is a new trainer mode that loads the processor, base model,
and `best_adapter`, reads `trainer_state.json` from the newest complete
checkpoint when available, and calls the existing
`_run_post_training_evaluation`. It never calls `trainer.train()` and never
appends new step records.

## State module

New stdlib-only module: `src/run_state.py`.

Public surface (names may be refined by the plan, behavior is fixed):

- `new_run_id(now=None) -> str`
- `create_run(runs_dir, *, config, paths) -> dict`
- `load_run(run_dir) -> Optional[dict]`
- `list_runs(runs_dir, limit=50) -> list[dict]` — newest first by `created_at`.
- `update_run(run_dir, mutator) -> dict` — locked read-modify-write.
- `set_status(run_dir, status, **fields) -> dict`
- `set_stage(run_dir, name, status, detail=None, **fields) -> dict`
- `append_event(run_dir, event) -> None`
- `append_step(run_dir, record) -> None`
- `read_steps(run_dir, tail=2000) -> list[dict]`
- `start_attempt(run_dir, pid, reason) -> dict`
- `finish_attempt(run_dir, exit_code, reason) -> dict`
- `reconcile(run_dir, *, pid_alive) -> dict` — terminalize a dead attempt.
- `reconcile_all(runs_dir) -> dict` — scan every run for live trainer/merge PIDs
  and return the active run (if any). Called at startup and by the monitor.

`src/paths.py` gains `default_runs_dir()` / `resolve_runs_dir(raw=None)` with the
same precedence as the other managed dirs:

1. explicit `ASR_RUNS_DIR`
2. `<resolved ASR_RESULTS_DIR>/runs`
3. repo-local `results/runs`

The server exposes a `storage` block:

```json
{"runs_dir": "/workspace/stt-outputs/results/runs", "persistent": true}
```

`persistent` is true when the resolved runs root is under `/workspace` or
`/runpod-volume`. The UI shows a warning banner when `/workspace` exists but the
runs root is elsewhere.

## Server behavior

### Startup / recovery

On import, the server:

1. Resolves and creates the runs root.
2. Scans **every** run directory for live trainer PIDs (`training.pid`) and merge
   PIDs (`merge.pid`). It does not assume the newest run is active.
3. Any run with a live trainer PID becomes the active run. If two live trainer
   PIDs are found (a bug or a manual launch), the newest becomes active and the
   other is reported as a duplicate; no second run is started.
4. Any run whose status is `starting` or `running` but whose PID is dead is
   reconciled using the exit-classification rules below.
5. Any run whose `merge` stage is `running` but whose merge PID is dead is
   reconciled from its artifacts and log.
6. Already-completed runs are never rewritten.
7. Falls back to the legacy global files only when no ledger run exists.

The active run is not a module-level hardcoded path. Helpers resolve
`telemetry_file`, `benchmarks_file`, `loss_history_file`, `log_file`,
`pid_file`, and `stop_signal_file` from the selected run, with the existing
global constants retained for legacy and tests.

### Process monitor and exit classification

A lightweight asyncio task starts with the app and ticks every 2 seconds (it
also runs opportunistically on `/api/status` and WebSocket polls). It polls the
active trainer and merge processes, so an OOM kill while the server stays up
cannot leave a run stuck at `running` until the next restart.

Classification when the trainer process ends:

| Observed | Run result | Attempt |
|---|---|---|
| exit 0 and training/post-eval terminal | `completed` (trainer's own write wins) | `exit_code: 0` |
| exit 0 but post-eval still pending and configured | reconcile from stages; normally `failed` with `failed_stage: post_eval` | `exit_code: 0` |
| non-zero exit, no stop request | `failed`, `failed_stage` = last running stage | `exit_code: N`, `reason: "nonzero_exit"` |
| SIGTERM/SIGKILL after a stop request | `stopped`; remaining stages `skipped` | `exit_code: null` or signal, `reason: "halted"` |
| SIGKILL/OOM with no stop request | `failed`; offer Resume if a checkpoint exists | `exit_code: null`, `reason: "killed"` |
| server restarts, PID dead, no terminal writer | `interrupted`; offer Resume | `exit_code: null`, `reason: "process_lost"` |
| server restarts, PID alive | remains `running`; server attaches | existing attempt continues |

`attempt.exit_code` is `null` whenever the server cannot observe it. The
`reason` distinguishes an interrupt from a definite failure; `interrupted` is
reserved for the no-observable-exit case. A halt requested through the cockpit
always wins over a signal-derived `failed` classification.

### Start

`POST /api/train/start` keeps its validation and response shape, then:

1. Creates `<runs_root>/<run_id>/output/` and writes `run.json` with all stages
   `pending`, status `starting`, and the full effective `trainer_args`.
2. Sets `results_dir = run_dir`, `output_dir = run_dir/output` (unless an explicit
   output dir was supplied), `preflight_cache_dir = <runs_root>/_shared`.
3. Opens `<run_dir>/training_live.log`, initializes telemetry, and stores the
   PID in both `run.json` and `<run_dir>/training.pid`.
4. Passes `--run_dir <run_dir>` and `--preflight_cache_dir <shared>` to the
   trainer.
5. Returns `run_id`, `pid`, `config`, `paths`, and `storage` alongside the
   existing `status` field.

For a cockpit run, an explicit `results_dir` in the start payload is rejected
with `400`: the run must own its results to stay self-contained. Relocate the
whole runs root with `ASR_RUNS_DIR` instead. An explicit `output_dir` is still
honored for advanced use, but `run.json` records it and the run is no longer
portable.

If the spawn fails, the run is marked `failed` with `failed_stage: training` and
the partial log is retained.

### CLI runs

`scripts/train_lora.py --run_dir <path>` is the opt-in for a CLI run to appear
in the cockpit. It writes the same `run.json`, `steps.jsonl`, `events.jsonl`,
and stage transitions. It is listed and viewable, and it is resumable/mergeable
from the cockpit because `trainer_args` captures the full effective namespace.
A CLI run without `--run_dir` keeps its current behavior (global results, no
ledger entry).

### Resume

`POST /api/runs/{run_id}/resume`:

- Rejects if any full run, smoke test, or merge is active.
- Rebuilds the command from `run.json.trainer_args` with `fresh_start=false`,
  `auto_resume=true`, the same `output_dir`/`results_dir`/`preflight_cache_dir`.
- Appends a new attempt with `reason: "resume"` and sets the run `running`.
- Chooses the mode from the stage table above:
  - normal trainer resume for preflight/training failures;
  - `--post_eval_only` when training is already `completed` but post-eval is
    not. This loads `best_adapter` and never replays optimizer steps.
- The trainer reuses the cached preflight and the latest checkpoint.

Resume is offered for `interrupted`, `stopped`, or `failed` runs that still have
their run directory. A `failed` merge is retried through the merge endpoint, not
Resume.

### Stop

`POST /api/train/stop` keeps working. It targets the active run:

- writes `<run_dir>/stop_training.signal`;
- records `stop_requested_at` in `run.json`;
- sets the current stage's detail to `halt requested`;
- schedules the existing grace-period escalation against the live PID (from the
  child handle, or from `run.json` after a server restart);
- after exit, the run becomes `stopped` (not `failed`).

The trainer cooperatively checks the signal during preflight, at each training
step, and between post-eval passes. If a subprocess does not cooperate within
the grace period, the escalation kills the process group and the run is still
classified `stopped` because a halt was requested.

### Merge

`POST /api/runs/{run_id}/merge`:

- Requires no active full run, smoke test, or merge.
- Requires the `training` stage `completed` and
  `<output_dir>/best_adapter/adapter_config.json` to exist.
- Sets `merge` stage `running`, spawns
  `scripts/merge_lora.py --base_model <config> --adapter_path <best_adapter>
  --output_dir <run_dir>/merged`, and writes `<run_dir>/merge.log`.
- On exit 0 with `merged/config.json` and a weights file present → `completed`;
  otherwise `failed`, with the log tail available.
- Records `merge.pid`; startup recovery treats a dead merge PID as `failed` or,
  if the artifacts are complete and newer than the stage start, `completed`.

Merge is available for a completed run even when post-eval failed, because the
adapter is independent of the report.

### Busy rules

One shared busy check covers the full run, smoke test, and merge. The existing
smoke-test isolation is unchanged. A merge of an older run blocks starting a new
training run, because both use the GPU.

## APIs

Existing endpoints stay:

- `GET /api/status`
- `GET /api/benchmarks`
- `GET /api/loss-history`
- `GET /api/logs`
- `POST /api/train/start`
- `POST /api/train/stop`
- `POST /api/train/smoke`
- `GET /api/train/smoke/status`
- `WS /ws/live`

New endpoints:

- `GET /api/runs?limit=20` → `{runs: [summary…], storage: {…}}`
- `GET /api/runs/latest` → `{run, telemetry, benchmarks, loss_history,
  steps: {count, last_step, last_saved_at}, storage, process_running}`
- `GET /api/runs/{run_id}` → same shape for one run
- `GET /api/runs/{run_id}/steps?tail=2000` → `{run_id, total, steps: […]}`;
  `tail` capped (e.g. 10,000)
- `POST /api/runs/{run_id}/resume`
- `POST /api/runs/{run_id}/merge`

`/api/status` adds `run`, `active_run_id`, `steps`, and `storage` while keeping
every current telemetry field at the top level. `WS /ws/live` always streams the
active (or latest, when idle) run and adds the same `run` and `steps` blocks; it
keeps `telemetry`, `benchmarks`, and `loss_history`.

If no ledger run exists, `/api/runs/latest` returns `run: null`, `storage`, and
the legacy telemetry/benchmarks/loss history. The UI then behaves like today.

## Web UI

### Run header and stage stepper

Add above the existing progress area:

- a run selector (latest plus recent runs), run id, and created time;
- a four-step stepper: **Preflight → Training → Post-eval → Merge**, each with a
  dot, label, status detail, and duration;
- a saved-steps line: `N steps saved · last at HH:MM:SS`;
- a storage warning banner when the runs root is not persistent.

The stepper is hidden for `source: "demo"` or when `/api/runs/latest` is
unavailable, so the local zero-dependency demo keeps working.

### State restoration

On first load:

1. Fetch `/api/runs/latest`.
2. If a run exists, populate the form from `run.config` (epochs, batch size,
   grad accum, gradient checkpointing, eval steps, learning rate), render the
   stepper, and load its benchmarks/history/log.
3. If the run is active, the WebSocket keeps the selected run live.
4. If it is interrupted or failed, show **Resume**; if training is completed and
   an adapter exists, show **Merge**.
5. If no run exists, use current defaults and legacy behavior.

Selecting a historical run loads that run's config, benchmarks, steps, and log
read-only; live WebSocket updates pause while a historical run is selected. The
historical view still offers **Resume** for an interrupted/stopped/failed run
and **Merge** for a completed run with an adapter. A **Latest** control returns
to the active/latest run. Starting always creates a new run with the current
form values.

### Controls

- **Start run**: new run (existing behavior plus a new run id).
- **Fresh start**: new run with `fresh_start=true`; confirmation text stays.
- **Halt run**: active run only.
- **Resume**: visible for an interrupted/stop/failed run.
- **Merge**: visible for a run whose training completed and adapter exists.
- **Smoke test**: unchanged, isolated, excluded from the ledger.

During merge, show `merge` stage status and a collapsible log tail. A failed
merge is shown without changing the run's completed training status.

### UI structure

- `server/static/index.html`: run selector, identity, stepper, saved-steps line,
  storage banner, Resume/Merge controls, merge log.
- `server/static/app.js`: load/restore run state, stage rendering, run selection,
  resume/merge calls, step-count display. Keep the existing telemetry, benchmark,
  canary, log, and smoke-test logic.
- `server/static/run_view.js` (new): pure helpers for stage normalization and
  labels, testable with Node like `benchmark_view.js`.
- `server/static/style.css`: stepper, run selector, storage banner, merge states.
  Reuse the existing tokens and visual language.

## Persistence guarantees

- Config is durable before the trainer starts.
- A step record is appended with `flush()` before the next step begins. A stage
  boundary also calls `fsync()`; per-step `fsync()` is intentionally avoided to
  keep the hot path cheap. A container kill may lose only the last partial line;
  a pod power loss may lose the last few buffered steps.
- `run.json` and telemetry snapshots are written atomically, so readers never
  see half a file.
- The run directory is self-contained for archival use: copying it preserves
  its config, history, checkpoints, report, and merged model. Absolute paths in
  `run.json` are not automatically rebased, so a copied run is not loadable or
  resumable from the cockpit without editing those paths. Path rebasing is a
  non-goal for this iteration.
- Old runs are never modified by a new run. A new run gets a new directory.
- The UI's 500-point `loss_history.json` cap remains for rendering speed; the
  uncapped history is in `steps.jsonl` and served by the steps endpoint.

## Local development without a GPU

The cockpit must be fully developable on a CPU-only machine:

- The FastAPI server, `src/run_state.py`, stage transitions, APIs, and UI never
  import CUDA or require a GPU. `get_gpu_info()` already degrades to
  `available: false`.
- Add `scripts/dev_fake_trainer.py`, selected by `ASR_FAKE_TRAINER=1` (or an
  explicit `--fake-trainer` server flag). It accepts the same `--run_dir`,
  `--results_dir`, `--output_dir`, and `--preflight_cache_dir` arguments as the
  real trainer and writes the real ledger files: `run.json`, `events.jsonl`,
  `steps.jsonl`, `live_telemetry.json`, `loss_history.json`,
  `epoch_benchmarks.json`, and a fake `training_live.log`.
- The fake trainer simulates `preflight → training → post_eval` with short
  timings and configurable failures:
  - `ASR_FAKE_STEPS=200` (default 200),
  - `ASR_FAKE_STEP_DELAY=0.05` seconds,
  - `ASR_FAKE_FAIL_STAGE=preflight|training|post_eval`,
  - `ASR_FAKE_HANG=1` to test Halt and the grace period.
- Fake runs get `"source": "fake"` and a visible **Dev simulation** badge in the
  UI. They never touch a real `output/` checkpoint or the shared preflight cache
  unless a test points those paths at its temp directory.
- In fake mode, the merge endpoint spawns `scripts/dev_fake_merge.py`, which
  writes small placeholder files under `<run_dir>/merged/`, so the merge stage
  can be exercised without CUDA.
- `scripts/demo_local.py` remains the zero-dependency UI demo. It does not need
  the ledger; the UI falls back to legacy mode when `/api/runs/latest` is
  unavailable.
- Tests that need torch/transformers import them behind `pytest.importorskip`
  and skip on CPU-only machines. Ledger/API/UI tests must run without those
  packages. A `@pytest.mark.gpu` marker is not required for new tests; real
  trainer coverage stays a RunPod acceptance step.
- Local verification procedure:
  1. `ASR_FAKE_TRAINER=1 ./runpod_web.sh -f` (or uvicorn directly).
  2. Start a run, reload the page, restart the server, and confirm the run,
     config, stages, and saved steps restore.
  3. Exercise Stop, Resume, Merge, a forced stage failure, and a server restart
     during a fake run.
  4. On RunPod, run one short real `--max_train_samples` job to confirm the same
     stages with the real trainer.

## Legacy behavior

- If no ledger run exists, the existing global `results/*` files remain readable
  through the current endpoints and the UI hides the stepper.
- No automatic migration or deletion of legacy files.
- If a legacy trainer is already running during an upgrade, the new server can
  read its global telemetry through the fallback but cannot record it as a run.
  Finish or stop that job before relying on the ledger. (The deployment checklist
  should say this.)

## Tests

- `tests/test_run_state.py`
  - run id format and path-traversal rejection;
  - create/load/list order;
  - locked read-modify-write preserves concurrent fields;
  - stage transitions and attempt lifecycle;
  - append/read steps, corrupt trailing line skipped, last-per-step dedupe;
  - `reconcile` applies the exit-classification table and leaves completed runs
    alone;
  - `reconcile_all` finds an older run's live PID instead of assuming newest.
- `tests/test_server_train.py` (extend)
  - Start creates `<runs>/<id>/output`, `run.json`, and passes `--run_dir`,
    `--preflight_cache_dir`, run-scoped `--output_dir`/`--results_dir`;
  - `/api/status` includes run/steps/storage;
  - `/api/runs/latest`, `/api/runs`, `/api/runs/{id}`, steps tail;
  - explicit `results_dir` is rejected for a cockpit run;
  - Resume rebuilds the same command and appends an attempt; post-eval resume
    uses `--post_eval_only`;
  - a historical run can be resumed/merged while the newest run is selected;
  - Stop writes the run's stop signal;
  - Merge command/output/stage and busy rules;
  - startup recovery for live, dead, completed, and merge PIDs, including an
    older run owning the live PID;
  - the 2-second monitor terminalizes an exit while the server stays up;
  - storage `persistent` flag;
  - legacy fallback when no run exists.
- `tests/test_run_state_cli.py` or `tests/test_train_lora_run_stages.py`
  - `--run_dir`, `--preflight_cache_dir`, and `--post_eval_only` parsing;
  - full `trainer_args` are recorded;
  - preflight copy into the run directory;
  - halt skips post-eval;
  - stage helpers set the expected statuses on success/failure.
- `tests/test_trainer_steps.py`
  - `WebTelemetryCallback.on_step_end` appends one JSONL line per step;
  - the 500-point loss cache still works;
  - a resumed callback continues the file.
- `tests/test_web_ui_contract.py` (extend)
  - stepper, run selector, saved-steps line, storage banner, Resume/Merge IDs;
  - app.js references `/api/runs/latest`, `resume`, and `merge`.
- `tests/test_run_view.js` (Node, like `test_benchmark_view.js`)
  - stage normalization and labels.
- `tests/test_fake_trainer.py` (CPU-only)
  - the fake trainer writes a valid run, stages, steps, and terminal status;
  - forced stage failures and `ASR_FAKE_HANG` produce the expected statuses;
  - fake merge writes its placeholder artifacts.
- Existing preflight, smoke-test, benchmark, and final-report tests must keep
  passing.

## Docs and config

- `README.md`, `GUIDE.md`, `PRODUCTION.md`: run layout, resume, merge, and the
  `/workspace/stt-outputs/results/runs/` path.
- `.env.example`: `ASR_RUNS_DIR` (optional; defaults under `ASR_RESULTS_DIR`).
- `PRODUCTION.md` persistent-artifacts list: add `results/runs/<run_id>/`.
- Deployment note: finish any legacy run before upgrading, or it will remain
  visible only as legacy telemetry.

## Calls made in the design

- Run-scoped output directories are the default for cockpit runs; CLI defaults
  stay unchanged.
- Preflight cache stays shared, with a per-run provenance copy.
- The `preflight` stage owns dataset preparation plus baseline measurement.
- Merge is a stage of the run but can fail without failing training.
- Halt applies to the whole run and skips post-eval.
- Resume after training is complete uses a new `--post_eval_only` mode.
- One active full run, one smoke test, and one merge at a time, but the ledger is
  multi-run-aware: an older run can own the live PID, and an older completed run
  can be merged.
- CLI runs with `--run_dir` are first-class ledger runs; `trainer_args` stores
  the full effective argparse namespace.
- A copied run directory is archival only; paths are not rebased.
- A cockpit start payload cannot override `results_dir`.
- `interrupted` means no observable exit; a definite nonzero/killed exit is
  `failed`, and a requested halt is `stopped`.
- No database, no automatic retention, no per-step weights.

## What is left to the builder

- Implement `src/run_state.py` and the `src/paths.py` additions.
- Add `--run_dir`, `--preflight_cache_dir`, and `--post_eval_only` to
  `scripts/train_lora.py`; record the full effective `trainer_args`; wire stage
  updates around preflight, training, and post-eval; copy the preflight into the
  run; make halt cooperative and skip post-eval.
- Extend `WebTelemetryCallback` to append `steps.jsonl`, update run progress, and
  write stage-aware telemetry.
- Refactor `server/app.py` to resolve per-run files, add the run APIs, startup
  scan, 2-second monitor, exit classification, resume, merge, storage reporting,
  and keep the legacy fallbacks.
- Add the UI run header, stepper, restoration, run selection, Resume/Merge, and
  storage banner.
- Add `scripts/dev_fake_trainer.py` and `scripts/dev_fake_merge.py`, plus the
  server env switch and the Dev simulation badge.
- Add the tests and docs above, including the CPU-only verification procedure.
