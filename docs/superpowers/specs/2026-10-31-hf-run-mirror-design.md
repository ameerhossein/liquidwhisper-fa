# Durable HF run mirror — design

**Date:** 2026-10-31  
**Status:** Draft for approval  
**Scope:** Mirror each training run's durable artifacts to a private Hugging Face repository as it progresses, and restore the newest run on a fresh RunPod pod.

---

## Intent

RunPod `/workspace` is persistent only for the life of the pod. A pod loss destroys
checkpoints, the best adapter, reports, and the run ledger. The existing
`scripts/upload_to_hf.py` can publish artifacts, but only manually and only after
training; `best_adapter/` does not exist until the end, and local auto-resume has
no remote fallback.

Make HF the durable store for the run while keeping local disk as the fast working
copy. Upload at checkpoint boundaries, stage boundaries, and optional adapter
snapshots. A fresh pod can discover the run, download the ledger and newest
complete checkpoint, and resume.

## Goals

1. Enable with one repo ID; private by default; create the repo if missing.
2. Upload every completed Trainer checkpoint in the background; never block or
   fail training.
3. Upload an adapter-only snapshot every 250 steps (configurable, `0` disables)
   so a usable model exists between full checkpoints.
4. Upload stage artifacts: preflight baseline, final adapter, post-eval reports,
   merged model; keep the run ledger synced.
5. On a fresh pod, discover the newest run (or an explicit run ID), restore its
   ledger and newest complete checkpoint, and resume; fail loudly rather than
   silently starting fresh.
6. Record sync status in the run directory, expose clear logs, and retry failures.
7. Keep the existing manual publish script and stable model repos.

## Non-goals

- Per-optimizer-step weight uploads.
- Replacing local storage; training remains local-first.
- Multi-pod concurrent training or distributed coordination.
- Automatic remote retention or deletion of old runs.
- An HF authentication UI.
- A new cockpit upload screen; the restore script plus the existing Resume button
  are enough for v1.

## Anti-goals

- HF outage stops or fails training.
- An incomplete checkpoint appears complete remotely.
- A fresh pod silently starts a new run when the user asked to resume.
- Remote restore overwrites a newer local checkpoint.
- Tokens or secrets appear in uploaded files.

## Constraints

- `save_steps=1000`, `save_total_limit=2`, `eval_steps=1000`;
  `load_best_model_at_end` ties save and eval intervals.
- `run_state` stages: `preflight`, `training`, `post_eval`, `merge`.
- `huggingface_hub` is optional; import it lazily.
- HF commits are atomic; `upload_folder` is one commit per call.
- `run_id` is path-safe and timestamp-sortable.
- Existing managed paths and `ASR_*` environment conventions must keep working.

## Approaches considered

**A. One private repo per run.** Isolation and easy deletion, but a fresh pod no
longer knows the run ID, the account accumulates many repos, and finding "the
latest run" needs an external index.

**B. One private repo per experiment, `runs/<run_id>/` folders plus
`latest.json` (chosen).** A stable repo ID survives pod recreation; all runs live
in one place; `latest` resolves without knowing the old run ID; no repo sprawl.
The trade-off is shared history: deleting one run removes its folder, not its
past commits, and the repo grows with every run.

**C. One repo with one branch per run.** Isolation plus a single repo, but
branches are less discoverable in the HF UI, and restore/download code has to
carry a revision everywhere. A stable pointer branch would still be needed.

Chosen B. The mirror repo is `user/stt-<experiment>` (for example
`user/stt-buzzasr-neyshekar`) and persists across pod lifetimes. Stable publish
repos remain `<prefix>`, `<prefix>-lora`, and `<prefix>-checkpoints` from the
existing script.

**D. External watcher process (rejected).** Decouples from the Trainer but adds a
second lifecycle, races with checkpoint rotation, and is harder for the cockpit
to manage.

**E. `huggingface_hub.CommitScheduler` (rejected).** Periodic folder sync can
capture half-written checkpoints and cannot distinguish checkpoint, adapter, and
stage boundaries.

## How it works

### Configuration

- `--hf_run_repo <user>/<repo>` or `ASR_HF_RUN_REPO`. With no repo, the feature
  is off and no network call is made.
- `--hf_private/--no-hf_private` (default private).
- `--hf_adapter_snapshot_steps N` (default `250`; `0` disables).
- `--hf_resume none|latest|<run_id>` (default `none`).
- `--hf_auto_resume/--no-hf_auto_resume` (default on). This only repairs the same
  run when the local checkpoint is missing or older; it never selects a different
  run.
- `HF_TOKEN` or a cached `hf auth login`. No token is accepted on the command
  line, so it cannot leak through process listings.
- Cockpit: the server passes its environment to the trainer, so
  `ASR_HF_RUN_REPO` is enough. The trainer records the effective mirror config in
  `run.json` under `hf`.

**Config precedence.** The mirror repo recorded in `run.json` is authoritative for
an existing run. A different `--hf_run_repo` or `ASR_HF_RUN_REPO` on resume is
ignored with a warning; it only selects the repo for a new run. When mirroring is
first enabled on an existing run, the newest complete local checkpoint and
`best_adapter/` (when present) are backfilled once, so the checkpoint being
resumed from reaches HF even if it predates the feature.

**Plain CLI runs.** `run_train.sh` does not pass `--run_dir`. When
`ASR_HF_RUN_REPO` is set and no `--run_dir` is given, `train_lora.py` creates a
canonical run directory (`<runs_root>/<new_run_id>/`) automatically, with the same
run-scoped defaults as an explicit `--run_dir`; explicit `--output_dir` and
`--results_dir` still win. Without a mirror repo, plain CLI behavior is unchanged.
This keeps the production shell path protected without requiring a new flag.

### Repo layout

```
README.md
latest.json
runs/<run_id>/
  run.json
  steps.jsonl
  events.jsonl
  live_telemetry.json
  loss_history.json
  epoch_benchmarks.json
  baselines/preflight.json
  hf_state.json
  checkpoint-1000/
  checkpoint-2000/
  adapter_snapshots/step-0250/
  best_adapter/
  reports/final_evaluation_report.json
  reports/final_evaluation_report.md
  merged/
  training_live.log          # stage boundaries only; optional
```

`latest.json` is the discovery pointer to the most recently active run:

```json
{"run_id": "20261031-120000-abcdef", "status": "running",
 "last_step": 2000, "updated_at": 1790000000.0, "layout": 1}
```

`runs/<run_id>/hf_state.json` is the per-run sync high-water mark, used for
ledger conflict checks:

```json
{"last_step": 2000, "updated_at": 1790000000.0,
 "checkpoints": [1000, 2000], "status": "running"}
```

`latest.json` is only a pointer and display summary; the per-run state file is
authoritative for an older run that is resumed after another run became latest.

Checkpoints sit at the run-folder root so `snapshot_download` can place
`checkpoint-N/` directly into the local output directory for the Trainer's
resume path. `adapter_snapshots/` and `merged/` are not inference roots; load
them with `subfolder=` when needed.

### What is uploaded when

| Trigger | Upload |
|---|---|
| run start | `run.json`, `latest.json`, initial ledger |
| preflight complete | ledger plus `baselines/preflight.json` |
| every Trainer checkpoint (`on_save`, 1000 steps) | full `checkpoint-<step>/` plus ledger and `latest.json` |
| every 250 steps, when not a checkpoint step | `adapter_snapshots/step-<n>/` plus `latest.json` |
| training complete | `best_adapter/` plus final ledger |
| post-eval complete | `reports/` plus ledger |
| merge complete | `merged/` plus ledger |
| terminal status | final ledger plus `latest.json`; drain with a timeout |

Per-optimizer-step uploads are explicitly not done. `steps.jsonl` is synced at
checkpoints and stage boundaries, not on every optimizer step.

### Upload worker

- One background daemon thread, one upload at a time.
- Job kinds: `checkpoint`, `adapter`, `ledger`, `merged`, `latest`.
- FIFO for checkpoints; coalesce pending adapter snapshots to the newest; coalesce
  `latest.json` updates.
- Validate before upload. A checkpoint requires `trainer_state.json` and adapter
  weights; anything incomplete is dropped with a log line.
- Use `upload_folder` for directories, `create_commit` with
  `CommitOperationAdd` for multi-file ledger snapshots (including
  `hf_state.json`), and `upload_file` for `latest.json`.
- Retry three times with exponential backoff (2 s, 8 s, 30 s). After the last
  failure, record the error in `hf_sync.json` and keep training; the next
  checkpoint retries. Pending jobs are persisted to `hf_sync_pending.json` so a
  graceful restart can retry them; a job whose local path vanished is dropped.
- Write adapter snapshots to a staging directory, then `os.replace` to
  `step-<n>` before enqueueing, so the uploader never sees a partial directory.
- Local retention: the Trainer's `save_total_limit` is unchanged. Adapter
  snapshots keep the last two locally; older ones are deleted only after their
  upload succeeded, or when a newer snapshot has already been uploaded.
- `close(timeout)` drains on normal exit. A killed pod simply leaves the last
  committed HF state; no local cleanup runs.

### Failure and status

- Every upload exception is caught and logged. Training never fails because of HF.
- `<run_dir>/hf_sync.json` is local-only and excluded from uploads:

```json
{
  "enabled": true,
  "repo_id": "user/stt-buzzasr",
  "status": "idle",
  "last_success_at": 1790000000.0,
  "last_success_step": 2000,
  "last_error": null,
  "pending": 0,
  "bytes_uploaded": 123456789
}
```

- Failures append `{"type": "hf_sync", "status": "error", ...}` to `events.jsonl`.
  Successful per-checkpoint uploads only update `hf_sync.json` and the training
  log, avoiding ledger bloat.
- Console lines: `[hf-sync] uploaded checkpoint-1000 in 42s (1.1 GB)` and
  `[hf-sync] upload failed (...), retrying`.
- No token is written to any run file.

### Resume and restore

Two entry points share `src/hf_sync.py`.

0. **Recover by run ID (the primary fresh-pod path).**
   - `python scripts/hf_restore.py --run <run_id> [--repo <user>/<repo>]` restores
     the run's state and artifacts only; it never starts training.
   - `python scripts/train_lora.py --hf_resume <run_id>` restores the same state,
     then resumes training when the run is not terminal. If the run is
     `completed`, it restores and exits with a message instead of re-training.
   - The repo is resolved in this order: explicit `--repo`/`--hf_run_repo`, the
     `hf.repo_id` recorded in a local `run.json` for that run ID, then
     `ASR_HF_RUN_REPO`. If none is available, the command fails before any
     download and prints the repo options. With `ASR_HF_RUN_REPO` set once in
     `.env`, the run ID is the only argument needed on a fresh pod.
   - The run ID maps directly to `runs/<run_id>/` in the repo. A missing run ID
     fails with a short list of available run IDs.
   - This pulls exactly what is needed to rebuild the local run:
     `run.json`, `steps.jsonl`, `events.jsonl`, `live_telemetry.json`,
     `loss_history.json`, `epoch_benchmarks.json`, `baselines/preflight.json`,
     `hf_state.json`, `best_adapter/`, `reports/`, and the newest complete
     `checkpoint-<step>/`. `merged/` is pulled only with `--include-merged`
     because it is large and can be regenerated.
   - If `best_adapter/` is absent but a checkpoint exists, the checkpoint is
     enough to resume training. If no checkpoint exists, the command restores
     the ledger and adapter snapshot, prints what is available, and exits
     non-zero rather than starting fresh.
   - Restoring into an existing non-empty run directory is refused unless
     `--force` is passed, and a local ledger newer than the remote high-water
     mark is never overwritten.

1. **Automatic repair while resuming an existing run.** When `--run_dir` exists
   and the local output has no complete checkpoint, or the remote newest is newer,
   and `--hf_auto_resume` is on, download the newest complete checkpoint for that
   run into `output/`. This covers a pod that kept the run ledger but lost its
   output, and a pod that restored an older volume snapshot.

2. **Latest-run restore (`--run latest` / `--hf_resume latest`).**
   - `--hf_resume latest` resolves `latest.json` (the most recently active run);
     if it is missing, list `runs/*/run.json` and pick the newest `run_id` (the ID
     is timestamp-sortable, so no clock comparison is needed). If that run is
     `completed`, the command fails with a message naming it and suggesting an
     explicit `--hf_resume <run_id>` or `--fresh_start`; it never silently skips
     to an older run. `latest.json` is only the pointer; the downloaded
     `runs/<run_id>/run.json` status is authoritative.
   - `latest` or `<run_id>` creates `<runs_root>/<run_id>/`, downloads the ledger,
     `best_adapter/`, reports, and the newest complete checkpoint.
   - The restored `run.json` paths are rewritten to the new local run directory
     (`output/`, results, merged, preflight cache); `trainer_args` paths are
     rewritten too. Original metrics, hyperparameters, and stage history are
     preserved.
   - The restore script prints `python scripts/train_lora.py --hf_resume <run_id>`
     plus the resolved local run directory. `train_lora --hf_resume` applies the
     saved `trainer_args` as the base configuration; explicit CLI flags override
     them. The following keys are always reset or derived, never taken from the
     saved values: `run_dir`, `output_dir`, `results_dir`, `preflight_cache_dir`,
     `resume_from_checkpoint`, `fresh_start`, `auto_resume`, `hf_resume`,
     `hf_run_repo`, and `post_eval_only`.
   - `post_eval_only` is derived from stage state, not saved: when the `training`
     stage is `completed` and `post_eval` is not, restore resumes in post-eval-only
     mode so the optimizer is not replayed.
   - `--fresh_start` and `--hf_resume` are mutually exclusive. `--fresh_start`
     also suppresses automatic repair, so a cleared output tree is never
     repopulated from HF behind the user's back. An explicit
     `--resume_from_checkpoint <path>` suppresses automatic download unless the
     value is `last`.
   - After restore, the cockpit's existing run list and Resume button work because
     the run directory is self-contained again.

Checkpoint selection:

- `list_repo_files` for `runs/<id>/checkpoint-<step>/trainer_state.json`; pick the
  maximum step.
- Compare with local `find_last_checkpoint`; never download an older remote
  checkpoint over a newer local one.
- Download into a temp directory next to `output/`, verify `trainer_state.json`,
  then atomically move `checkpoint-<step>` into `output/`.
- If the newest remote checkpoint fails validation, try the next older one and log
  the fallback.

**When the selected run has no complete checkpoint.** Fail loudly; do not start a
fresh run under the same ID. The restore still downloads the ledger,
`best_adapter/`, reports, and the newest adapter snapshot when present, then
prints what it found and exits non-zero. Two explicit escape hatches:

- `--post_eval_only` when the training stage is complete and only post-eval needs
  to run (the restored `best_adapter/` is enough).
- `--fresh_start` to start a new run, or a manual resume from the newest adapter
  snapshot with a fresh optimizer (printed as a suggestion, never automatic).

**Ledger monotonicity.** Remote `steps.jsonl`/`events.jsonl` are treated as
append-only. The mirror records a per-run high-water mark (`last_step`) in
`runs/<run_id>/hf_state.json` and refuses to upload a ledger whose `last_step` is
lower, so a pod restored from an older volume snapshot cannot overwrite newer
remote history. Auto-repair compares local and remote high-water marks: remote
wins when strictly newer, local wins when strictly newer, and local wins on a tie.
When remote wins, the local ledger is replaced before any upload starts.
`latest.json` is not used for this comparison because it points at whichever run
was active most recently.

**Lifecycle normalization after restore.** Rewrite paths and clear
`current_pid`, `merge_pid`, `training.pid`, and open attempts. A run that was
`starting` or `running` becomes `interrupted` with `last_error` describing the
restore, so the cockpit shows Resume instead of treating a dead PID as live.
Terminal statuses (`completed`, `stopped`, `failed`) and completed/skipped stage
states are preserved. `running` stages from the dead pod become `interrupted`.

### Integration points

- `src/hf_sync.py`: config, repo layout, uploader, validation, restore. Stdlib
  only at import; `huggingface_hub` is imported lazily.
- `src/trainer_callbacks.py`: `HFMirrorCallback` with `on_train_begin`,
  `on_step_end`, `on_save`, and `on_train_end`. Callback order is telemetry first,
  mirror second, so epoch benchmark files exist before sync.
- `scripts/train_lora.py`: parse flags, build the mirror after `run_dir` is known,
  pass the callback, upload stage artifacts, close in `finally`, and restore
  before the existing resume decision.
- `scripts/merge_lora.py`: optional `--run_dir` and `--hf_run_repo`; upload the
  merged output after saving. The cockpit passes them; environment variables are
  the fallback.
- `server/app.py`: no new upload UI. The merge child performs the upload, and the
  run manifest `hf` section gives the server enough information to log status.
- `scripts/hf_restore.py`: restore-only CLI.
- `src/paths.py`: local behavior unchanged; add a helper only if the restore code
  needs one.
- `scripts/upload_to_hf.py`: unchanged as the stable-publish path; it may reuse
  validators from `src/hf_sync.py`.

### Run manifest

`run.json` gains:

```json
"hf": {
  "enabled": true,
  "repo_id": "user/stt-buzzasr",
  "run_path": "runs/20261031-120000-abcdef",
  "private": true,
  "adapter_snapshot_steps": 250,
  "auto_resume": true
}
```

`hf_sync.json` stays separate and local-only.

## Failure modes

| Failure | Behavior |
|---|---|
| HF down or 5xx | Log, retry, keep training; the next checkpoint retries |
| Bad token or no write access | Disable the mirror for this process, log remediation; training continues |
| Pod dies mid-upload | HF commit is atomic; the last committed artifact is intact |
| Pod dies before the first upload | Same as today: local-only loss; the first upload happens at run start, so the window is small |
| Local checkpoint rotated before upload | The uploader drops the vanished job; the newest checkpoint is uploaded |
| Remote newest checkpoint corrupt | Validate, then fall back to the next older complete checkpoint |
| Remote newer than local | Download only when `hf_auto_resume` is on; log the choice |
| Disk full from adapter snapshots | Keep the last two locally; the Trainer's `save_total_limit` is unchanged |
| Repo missing | Create a private repo; if creation fails, log and continue locally |

## Testing

Unit tests with a fake `HfApi`/uploader, no network:

- config parsing and default-off behavior;
- repo path/layout construction;
- checkpoint completeness validation;
- adapter snapshot scheduling at 250 steps and skipping checkpoint steps;
- queue coalescing and retry;
- `latest.json` selection and fallback, including the completed-run failure;
- restore applies saved `trainer_args`, lets explicit flags override, and resets
  paths and resume controls;
- restore picks the newest complete checkpoint, rewrites paths, and never
  overwrites a newer local checkpoint;
- restore with no complete checkpoint fails non-zero and does not start fresh;
- per-run `hf_state.json` high-water mark prevents a ledger downgrade, and
  auto-repair chooses the newer ledger;
- `--fresh_start` plus `--hf_resume` is rejected, and `--fresh_start` suppresses
  auto-repair;
- a plain CLI run with a mirror repo but no `--run_dir` gets a canonical run
  directory;
- run manifest `hf` section and `hf_sync.json` writing.

Integration: a fake trainer run with a stub mirror verifies stage uploads for
preflight/training/post-eval/merge without network access.

Manual smoke: a private test repo, a short fake run, kill the process after a
checkpoint, restore into a clean directory, and confirm the resume step.

Tests must not require `HF_TOKEN`; network tests are skipped unless an explicit
environment variable is set.

## Rollout

1. Add the module and tests with the feature off by default.
2. Enable on a short fake or one-epoch run with a private repo; verify layout and
   restore.
3. Enable on a real L40S run; verify checkpoint upload time and resume after a
   forced pod restart.
4. Document `ASR_HF_RUN_REPO`, the restore commands, and retention in
   `README.md`, `PRODUCTION.md`, and `.env.example`.

## Decisions left to the builder

- Exact pending-queue persistence (`hf_sync_pending.json` versus re-scanning on
  startup). Either is acceptable if a vanished path is dropped safely.
- Whether `latest.json` updates on every adapter snapshot or only at checkpoints
  and stages; choose the lower-commit option if it keeps `last_step` useful.
- Whether `training_live.log` is uploaded at stage boundaries. Skip it if large;
  `steps.jsonl` and the telemetry files are the durable history.
- Whether the cockpit shows an HF sync badge in v1 or later; logs plus
  `hf_sync.json` are sufficient for v1.
