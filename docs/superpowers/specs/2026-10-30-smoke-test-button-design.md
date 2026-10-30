# Web UI smoke-test button — design

**Date:** 2026-10-30  
**Status:** Approved (2026-10-30)  
**Scope:** Add a one-click **Smoke test** button to the training cockpit. It runs a
short, isolated training job and stops when it finishes. It never touches the
production output directory, checkpoints, logs, or telemetry.

---

## Intent

A full training run is expensive. Today the dashboard’s **Start run** button
launches the real job immediately, so a broken environment, bad batch size, or
data-pipeline failure is only discovered after a long wait.

The smoke test gives the operator a fast confidence check using the real trainer,
dataset, model, LoRA setup, and evaluation path on a tiny subset. If it passes,
the full run is unlikely to fail for setup reasons.

## Goals

1. Add a **Smoke test** button next to the existing run controls.
2. Run the real `scripts/train_lora.py` on a small subset, with no duplicated
   training implementation.
3. Use isolated output, results, log, and PID paths under a dedicated
   `smoke-test` location.
4. Show `running`, `passed`, or `failed` in the dashboard, including the exit
   code and a short log tail on failure.
5. Never resume from, overwrite, or delete production checkpoints or telemetry.
6. Keep the existing full-run Start/Fresh/Halt behavior unchanged.

## Non-goals

- Automatically starting the full run after a successful smoke test. The button
  only checks; the operator presses **Start run** when ready.
- Testing every epoch, the full test split, or the final post-training evaluation.
- Adding a general-purpose “training profile” system.
- Replacing a real preflight benchmark of model quality. This is an
  infrastructure/data-pipeline check, not a WER measurement.
- Stopping a smoke run from the dashboard. Smoke runs are short; the normal
  Halt control remains scoped to the full run.

## Approaches considered

1. **Isolated smoke job (chosen).** A dedicated server endpoint launches the
   trainer with smoke-specific paths and flags. The dashboard gets a small
   status indicator. Production artifacts are untouched.
2. **Preset on the existing run.** The button fills smoke values into the normal
   start payload. Less code, but it uses the production output/results paths, can
   auto-resume a real checkpoint, and overwrites live telemetry and logs. That
   can damage the very run the check is meant to protect.
3. **Environment-only preflight.** Check imports, GPU, and dataset access without
   training. Faster, but it would miss LoRA injection, data preparation,
   tokenization, checkpointing, and the benchmark path.

## Smoke run behavior

### Parameters

The smoke job reuses the form’s memory-relevant settings so it validates the
intended full-run configuration:

- `batch_size`, `grad_accum`, `learning_rate`, and `gradient_checkpointing` come
  from the dashboard form.
- `dataset_name`, `base_model`, telephony settings, and preprocessing worker
  count use the same server defaults as a normal run.

Everything else is fixed for speed:

| Setting | Value |
|---|---|
| `max_train_samples` | 200 |
| `max_eval_samples` | 50 |
| `epochs` | 1 |
| `eval_steps` | 5 |
| `save_steps` | 5 |
| `warmup_steps` | 2 |
| `max_benchmark_samples` | 25 |
| post-training full test eval | off (`--no-run-post-test-eval`) |
| resume | off (`--fresh_start`) |

With the default effective batch of 32, 200 samples produce roughly 6 optimizer
steps, so the step-5 evaluation and checkpoint actually fire.

The per-epoch benchmark remains enabled with 25 samples (clean plus phone) to
exercise the benchmark path without the 2,149-row final evaluation.

### Isolation

The smoke job uses dedicated paths:

- output: sibling `smoke-test/` next to the managed default output directory,
  resolved through the existing `_sanitize_output_dir` allow-list
- results: `<RESULTS_DIR>/smoke-test/`
- log: `<smoke results>/training_live.log`
- PID: `<smoke results>/training.pid`

The production `output_dir`, `results_dir`, `live_telemetry.json`,
`loss_history.json`, `epoch_benchmarks.json`, `training_live.log`, and
`training.pid` are not read or written by the smoke job.

Each click starts clean inside the smoke paths. Reusing the same smoke directory
keeps disk usage bounded; `--fresh_start` clears old smoke checkpoints and smoke
telemetry before the new attempt.

### Pass/fail

A run **passes** when the trainer exits `0` and
`smoke-test/best_adapter/adapter_config.json` exists.

A run **fails** when the trainer exits non-zero, is killed, or exits `0` but the
adapter artifact is missing. The status response includes the exit code and the
last lines of the smoke log so the dashboard can show the actual error.

If the server restarts while a smoke job is running, the PID file is used to
report `running`. If the job finished while the server was down, the adapter
artifact is used to infer `passed`; otherwise the status is `failed` with an
“ended without a result” message.

## Server API

### `POST /api/train/smoke`

- Accepts the same optional auth payload as the normal start endpoint.
- Rejects with `400` if any training job (full or smoke) is already running.
- Validates `batch_size`, `grad_accum`, `learning_rate`, and
  `gradient_checkpointing` with the same rules as `/api/train/start`.
- Spawns the trainer with the fixed smoke flags and isolated paths.
- Returns `{ "status": "started", "mode": "smoke", "pid": ..., "output_dir": ...,
  "results_dir": ... }`.

### `GET /api/train/smoke/status`

Returns:

```json
{
  "status": "idle | running | passed | failed",
  "running": false,
  "exit_code": 0,
  "output_dir": ".../smoke-test",
  "adapter_exists": true,
  "log_tail": ["..."]
}
```

- `log_tail` is only included when not running, and is capped to the last ~30
  lines.
- `idle` means no smoke run has been started since the server process began.

### Concurrency

A single shared busy check prevents a full run and a smoke run from executing at
the same time. The smoke job uses its own process handle and PID file, while the
existing full-run process management remains in place.

`/api/train/stop` continues to control only the full run. A smoke run is not
exposed to the Halt button.

## Web UI

- Add a **Smoke test** button in the configuration action row.
- Add a status line below the actions, hidden until the first smoke run.
- On click:
  1. disable Start, Fresh, and Smoke;
  2. `POST /api/train/smoke` with the current batch/accum/LR/checkpointing
     values;
  3. poll `GET /api/train/smoke/status` every two seconds.
- While running: “Smoke test running — 200 train / 50 eval samples.”
- On pass: “Smoke test passed. Ready to start the full run.”
- On fail: “Smoke test failed (exit N).” plus a collapsible log tail.
- Re-enable the controls when the run finishes.
- On page load, query the status once so a running or recently finished smoke
  test is not hidden by a refresh.
- The button includes a tooltip explaining that it uses an isolated directory
  and does not affect the real run.

## Files to change

- `server/app.py` — smoke paths and process state, start/status endpoints,
  shared busy check, command construction.
- `server/static/index.html` — Smoke test button, status line, log detail.
- `server/static/app.js` — element references, start/poll logic, button state.
- `server/static/style.css` — minimal passed/failed status styling if needed.
- `tests/test_server_train.py` — smoke command, isolation, concurrency, and
  status tests.

No changes to `scripts/train_lora.py`, `run_train.sh`, or the YAML config are
needed: the smoke job uses existing CLI flags.

## Tests

- Smoke start builds a command containing the fixed sample/epoch/step flags,
  `--no-run-post-test-eval`, and `--fresh_start`.
- Smoke start uses `smoke-test` output and results paths, never the production
  paths.
- Smoke start forwards the form’s batch size, accumulation, learning rate, and
  gradient-checkpointing flag.
- Smoke start rejects an invalid batch/accumulation/learning-rate combination.
- Smoke start is rejected while a full run is active, and full start is rejected
  while a smoke run is active.
- Status reports `passed` for exit `0` with an adapter artifact.
- Status reports `failed` for non-zero exit, missing adapter, or an ended run
  with no result.
- Status is `idle` before any smoke run.

## What is left to the builder

- Implement the smoke process state and the two endpoints.
- Add the UI controls and polling.
- Add the tests above.
