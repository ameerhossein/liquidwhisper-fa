# Automatic preflight baseline — design

**Date:** 2026-10-30  
**Status:** Approved (2026-10-30)  
**Scope:** Run a mandatory, cached base-model benchmark before training, delete
the hardcoded baseline values, and show measured WER deltas in the epoch matrix
and web cockpit.

---

## Intent

Training currently compares 300-sample epoch WERs against unverified,
full-test-only constants in `configs/baselines.json`. The 28.5% Whisper and
17.2% BuzzASR values were introduced as hardcoded placeholders in the initial
commit; no benchmark artifact in the repo produced them.

This change removes those numbers entirely and replaces them with a measured
preflight:

- **BuzzASR base** is the mandatory baseline for fine-tuning deltas.
- **Whisper Large v3** is measured clean and phone as a **reference only**.
- Both are evaluated on the **same 1,000 test recordings** used by the epoch
  benchmark.
- Deltas are shown in **percentage points (pp)**; negative is improvement.

## Goals

1. Delete `configs/baselines.json` and every hardcoded copy of its values from
   the production path.
2. Run the preflight automatically at the start of every training run.
3. Cache it so later runs do not repeat the evaluation.
4. Use the same 1,000-row subset, phone profile, phone seed, normalization,
   decoding, and metric code as the epoch benchmark.
5. Feed measured baseline rows into `epoch_benchmarks.json`.
6. Show clean and phone deltas vs. the BuzzASR base in the web UI.
7. Keep Whisper as a measured reference row, without fine-tuned deltas.
8. Make missing baselines visibly absent, never silently substituted with
   unverified values.

## Non-goals

- Running a full 2,149-row preflight by default.
- Replacing the final post-training full-test evaluation.
- Real telephone-call evaluation; the preflight uses the existing
  `customer` simulator.
- Changing the model, dataset, LoRA settings, or training loop.
- Removing the optional manual `run_baseline.sh`; it simply does not feed the
  production baseline path.

## Apples-to-apples invariants

The preflight and epoch benchmark must use:

- the same `test` split and the **first 1,000 rows** (`select(range(1000))`),
- clean audio from the same dataset version,
- `telephony_profiles=["customer"]`,
- the same `phone_seed` (default `1234`),
- the same `telephony_use_ffmpeg` setting,
- the same `ASRMetricsCalculator` normalization,
- greedy decoding with `max_new_tokens=225`,
- the same eval batch size and dtype policy.

The BuzzASR baseline is the base model **without** the LoRA adapter.

## Approaches considered

1. **Cached preflight inside the training entrypoint (chosen).** The trainer
   checks a fingerprint; on a miss it invokes the existing
   `scripts/evaluate_model.py` for BuzzASR and Whisper, reads their
   `evaluation_details.json`, and writes one unified preflight JSON. Works for
   both CLI and web-UI launches.
2. **Shell-only preflight in `run_train.sh`.** Simple, but the web-UI start
   path would need a duplicate implementation and separate process handling.
3. **Callback `on_train_begin`.** Avoids a subprocess, but the model already
   has LoRA injected; temporarily disabling adapters is fragile.

## Chosen flow

1. `scripts/train_lora.py` resolves:
   - `--preflight_samples` (default: use `max_benchmark_samples`, which is
     `1000`; `0` means full 2,149),
   - `--overwrite_preflight` (default off).
   There is no disable flag: the preflight is mandatory.
2. Data preparation runs first, preserving the existing fork-before-CUDA rule.
3. After data prep, compute a fingerprint over:
   base models, dataset, split, sample count, phone profiles/seed/ffmpeg,
   eval batch size, and metric/code version.
4. If `results/baselines/preflight.json` matches, reuse it.
5. Otherwise, invoke `scripts/evaluate_model.py` for:
   - `BuzzASR/persian` (mandatory),
   - `openai/whisper-large-v3` (reference),
   each with `--max_samples 1000 --phone_eval --phone_seed 1234
   --telephony_profiles customer`.
   Per-model outputs go under `results/baselines/preflight/<slug>/`.
6. Write the unified preflight JSON atomically.
7. Pass its `models` mapping to `WebTelemetryCallback` as
   `baseline_benchmark`; the callback writes baseline rows and computes deltas.
8. Free CUDA memory before loading the training model.

### Failure behavior

- **BuzzASR preflight fails:** retry once, then abort training with a clear
  error. There is no fallback baseline.
- **Whisper preflight fails:** warn and continue; Whisper is reference-only.
  Its row is omitted rather than filled with a placeholder.

## Preflight file schema

`results/baselines/preflight.json`:

```json
{
  "fingerprint": {
    "dataset_name": "shekar-ai/Neyshekar",
    "split": "test",
    "sample_count": 1000,
    "phone_seed": 1234,
    "telephony_profiles": ["customer"],
    "telephony_use_ffmpeg": true,
    "eval_batch_size": 16,
    "buzzasr_base": "BuzzASR/persian",
    "whisper_v3": "openai/whisper-large-v3"
  },
  "created_at": "...",
  "models": {
    "buzzasr_base": {
      "name": "BuzzASR Base (preflight)",
      "epoch": "BuzzASR Base (preflight, 1,000)",
      "source": "preflight",
      "is_reference": false,
      "total_samples": 1000,
      "normalized_wer": 0.0,
      "normalized_cer": 0.0,
      "diagnostic_wer": 0.0,
      "formal_wer": 0.0,
      "informal_wer": 0.0,
      "phone_normalized_wer": 0.0,
      "phone_normalized_cer": 0.0,
      "phone_formal_wer": 0.0,
      "phone_informal_wer": 0.0,
      "phone_seed": 1234,
      "phone_profiles": ["customer"]
    },
    "whisper_v3": {
      "name": "Whisper Large v3 (preflight reference)",
      "source": "preflight",
      "is_reference": true,
      "total_samples": 1000
    }
  }
}
```

`configs/baselines.json` is deleted. No production module may carry fallback
baseline metrics. The local demo may keep its own **simulated demo baseline**
constant, clearly labeled as fake and never read by training or the server.

## Benchmark records and deltas

`WebTelemetryCallback` initializes baseline rows from the preflight mapping.

For each epoch record:

- `delta_vs_buzzasr` = epoch clean WER − BuzzASR preflight clean WER, only if
  `total_samples` matches.
- `delta_phone_vs_buzzasr` = epoch phone WER − BuzzASR preflight phone WER,
  only if sample counts and phone metrics match.
- `delta_vs_whisper` is removed from the UI and is not computed.
- Mismatched or missing baselines produce `null`, rendered as `—`.

## Web UI

- Subtitle becomes dynamic: “Held-out test split, 1,000-recording benchmark
  subset” (or full when `max_benchmark_samples=0`).
- Table columns:
  1. Checkpoint
  2. Clean WER
  3. Δ clean vs BuzzASR base
  4. Phone WER
  5. Δ phone vs BuzzASR base
  6. CER
  7. Formal
  8. Informal
  9. ZWNJ → space
- Remove the “vs Whisper v3” delta column.
- Whisper row is tagged **Reference**, with measured clean and phone WER.
- BuzzASR row is tagged **Baseline (preflight)**.
- Add a summary line: best phone checkpoint, base phone WER, and improvement in
  pp. Example: `Phone WER 24.10% → 17.30% (−6.80 pp)`.
- If no preflight exists, show “No measured baseline yet. Start a run to
  generate the preflight.” Do not render fake baseline rows.

## Server

- `server/app.py` serves preflight baseline rows from
  `results/baselines/preflight.json`.
- Remove `_HARDCODED_BASELINES` and the `configs/baselines.json` loader.
- `/api/benchmarks` returns `[]` when neither preflight nor epoch benchmarks
  exist; the UI renders the empty-state message.
- Training-start defaults: `max_benchmark_samples=1000`; preflight always on.

## Config and docs

- `configs/l40s_train_config.yaml`: `max_benchmark_samples: 1000`;
  add `preflight_samples: 1000`.
- `run_train.sh`: pass `--max_benchmark_samples 1000` and preflight defaults.
- README/GUIDE: document automatic mandatory preflight, the 1,000-row subset,
  BuzzASR as authoritative, Whisper as reference, and the removal of the
  unverified baseline file.
- PRODUCTION: list `results/baselines/preflight.json` as a reusable artifact.
- Delete `configs/baselines.json` and its tests.

## Final full-test report

The existing post-training full evaluation still runs on all 2,149 test rows.
It reports full-test fine-tuned metrics and **no baseline deltas** unless a
full preflight exists. The authoritative measured improvement is the
1,000-row epoch-matrix delta. A later `--preflight_samples 0` run can measure
the full baseline.

## Tests

- Fingerprint/cache reuse and invalidation.
- Unified JSON parsing from two `evaluation_details.json` files.
- BuzzASR failure aborts; Whisper failure warns and continues.
- Callback baseline rows carry phone metrics and source tags.
- Delta calculation only when sample counts match; phone delta unit test.
- `parse_args` defaults: preflight samples follow `max_benchmark_samples`
  (1000), benchmark 1000.
- Server prefers preflight JSON and returns `[]` without it.
- No test or module reads `configs/baselines.json`.
- Config/launcher consistency tests updated to 1000.

## What is left to the builder

- Implement `src/preflight.py` (fingerprint, cache, subprocess orchestration,
  parsing, atomic write).
- Wire it into `scripts/train_lora.py` after data prep.
- Extend callback baseline rows and delta fields.
- Remove hardcoded baselines from trainer, server, UI, demo, and final report.
- Update server/UI and docs.
- Add the tests above.
