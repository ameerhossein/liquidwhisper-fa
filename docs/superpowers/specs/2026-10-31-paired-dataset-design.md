# Paired Clean + Phone Training — Design

- Date: 2026-10-31
- Status: Draft for approval
- Scope: training data pipeline, CLI defaults, cockpit, cache, tests, docs

## 1. Intent

Fine-tune `BuzzASR/persian` on Neyshekar v6 for telephone use while retaining
clean/general ability. Today the prepared training set assigns each utterance
either clean or phone-simulated audio at prep time (`telephony_prob=0.5`), so
over 4 epochs each utterance is seen in only one condition.

Paired training puts every utterance in the training set twice — once clean,
once through the configured phone channel — so 2 epochs over 116,488 rows gives
every utterance both conditions at roughly the same compute as 4 disjoint
epochs (7,282 vs 7,284 optimizer steps at batch 8 × grad-accum 4).

Paired becomes the default training mode. Disjoint 50/50 and clean-only remain
explicit modes for ablation and backward compatibility.

## 2. Goals

- Every training utterance has exactly one clean row and one phone row with
  identical transcript, tokenized labels, and `is_informal`.
- The phone row is deterministic per utterance and seed, reproducible across
  runs and workers.
- `--telephony_mode paired` is the default; the default epoch count becomes 2.
- The prepared-dataset schema stays exactly `input_features`, `labels`,
  `is_informal`, so the Trainer, collator, and callbacks are unchanged.
- The prep-cache fingerprint includes the mode, so a mode switch cannot reuse a
  stale cache.
- The validation split stays clean and keeps its cache across train-mode
  switches.
- Preflight, per-epoch benchmark, and final full-test evaluation are unchanged:
  clean + phone on the test split.
- The cockpit can start paired, disjoint, or clean runs; paired is preselected
  and the epoch field follows the mode.
- Tests cover pairing correctness, determinism, counts, cache fingerprints,
  defaults, server commands, and the UI contract.

## 3. Non-goals

- On-the-fly phone augmentation during training.
- New phone profiles or changes to the telephone simulator.
- Paired validation or test splits.
- Changing preflight/benchmark row counts, deltas, or benchmark UI.
- Reworking the collator, Trainer, or epoch-benchmark callback.
- Keeping paired and disjoint prepared caches simultaneously by default.
- A new dataset schema, metadata column, or per-row channel column.

## 4. Anti-goals

- Training where a phone row is paired with a different utterance's labels.
- Silently reusing a disjoint cache in paired mode.
- A cockpit default of 2 epochs that still prepares disjoint 50/50 data.
- Pairing only one half of the data or dropping rows from one half.
- Changing the test rows used by preflight or evaluation.
- Silently training on clean-only audio when paired mode was requested.

## 5. Constraints

- L40S 48 GB; batch 8 × grad-accum 4; gradient checkpointing off.
- Data preparation runs before model/CUDA load; `datasets.map` with
  `num_proc > 1` forks safely only in that order.
- `datasets < 4`; `concatenate_datasets` is available.
- The overlay has ~250 GB+ free (user-confirmed). The paired shard cache is
  roughly 170 GB; the direct-shard strategy writes each `datasets.map` shard
  once, so peak and steady state are both ~170 GB. Raw audio/model weights add
  ~25 GB on the same disk, leaving ~55 GB headroom.
- Shard-cache publication is atomic: map into a staging directory, validate row
  counts and features, write the metadata last, then rename into place.
- Adding `telephony_mode` to the fingerprint invalidates all existing prepared
  caches; the first paired run re-preps.
- A checkpoint trained on disjoint data must not be resumed as paired. Cockpit
  Resume uses the run's stored arguments; CLI users must pass the same mode or
  start fresh.
- The preflight cache is independent of the training mode and must not be
  invalidated by this change.

## 6. Approaches considered

### A. Two prepared passes + concatenate (chosen)

Filter the raw train split once. Prepare it once with `telephony_prob=0.0`
(clean) and once with `telephony_prob=1.0` (phone), then concatenate the two
prepared splits.

- Reuses the tested `prepare_sample`; the cache format becomes direct shards.
- New public API is one method, `prepare_paired_dataset`.
- Deterministic phone row via the existing per-index RNG.
- Costs 2× prep time and ~170 GB of cache; the direct-shard cache avoids the
  duplicate `save_to_disk` copy, so peak ≈ steady ≈ 170 GB.
- Easy to test: count, identical labels, different features, determinism.

### B. One batched map emitting both rows per input

Decode each clip once, produce clean and phone features in the same
`datasets.map(batched=True)` pass, and emit 2N rows.

- Saves the duplicate audio decode and one intermediate dataset; perhaps
  10–20% faster prep.
- Requires a batched refactor of `prepare_sample`/`_prepare_split`, handling
  variable-length audio and indices in a new code path.
- Still ~170 GB of cache and a larger test surface. The saving does not justify
  the complexity.

### C. On-the-fly phone augmentation per batch

Keep the clean prepared set and degrade/generate phone features during
training.

- Keeps the cache at ~85 GB and varies the channel each epoch.
- Moves ffmpeg/STFT into the training loop, risks making the L40S data-bound,
  and no longer matches the cached paired-dataset epoch math. Rejected.

**Chosen: A.** It matches the current pipeline, keeps the disjoint path intact,
and the doubled prep is a one-time cost.

## 7. Design

### 7.1 Data pipeline

Add to `src/data.py`:

```python
def prepare_paired_dataset(
    self,
    dataset: Dataset,
    remove_columns: Optional[list] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    ...
```

`cache_dir` is a staging directory. When set, each half is mapped with an
explicit `cache_file_name` under it, so `datasets.map` writes its shard files
there instead of the HF datasets cache. The caller publishes the directory.

Behavior:

1. Resolve effective `num_proc` exactly like `prepare_dataset_split`, including
   the CUDA-fork guard and the torch-thread cap.
2. Filter durations once on the input dataset.
3. Prepare the clean half with `telephony_prob=0.0`.
4. Prepare the phone half with `telephony_prob=1.0` using a shallow copy of the
   pipeline that keeps the same processor, normalizer, profiles, seed, and
   ffmpeg setting.
5. Concatenate `[clean, phone]` with `datasets.concatenate_datasets`.
6. Return the combined dataset, `2 × filtered` rows, with the same columns.

Details:

- Both halves map the same filtered dataset with `with_indices=True`, so row
  order and the per-index RNG match; each phone row is the degraded twin of the
  clean row at the same position.
- No prep-time shuffle. The Trainer's random sampler shuffles every epoch.
- `_prepare_split` gains internal `filter_durations` and `cache_file_name`
  parameters (or an equivalent private helper) so the second pass does not redo
  duration filtering and each pass writes its shards directly to the staging
  directory. `prepare_dataset_split` gains an optional `cache_dir`; when it is
  omitted, existing behavior is unchanged.
- After both maps, assert the clean and phone feature schemas are identical
  before concatenating. A schema mismatch aborts preparation.
- Paired mode requires the telephone module. If `src.telephone` is unavailable,
  abort preparation with a clear error instead of silently producing clean-only
  rows. A per-row simulator exception also aborts paired preparation: a paired
  cache must not contain a "phone" row that is actually clean. (Disjoint mode
  keeps the existing warn-and-fallback behavior.)
- An empty filtered input returns an empty dataset with the same features.

### 7.2 Modes, flags, and defaults

- `scripts/train_lora.py`:
  - `--telephony_mode {paired,disjoint,clean}`, default `paired`.
    - `paired`: 2N rows, every utterance clean + phone; `--telephony_prob` is
      ignored with a one-line note if explicitly set.
    - `disjoint`: existing behavior, `--telephony_prob` coin (default 0.5).
    - `clean`: no augmentation.
  - `--epochs` default changes from 4 to 2.
  - `_cli_run_config` includes `telephony_mode`.
  - `_prep_fingerprint` signature gains `telephony_mode`; the dict gains
    `"telephony_mode"`. For train, pass the active mode and the effective prob
    (`None` for paired, the float for disjoint, `0.0` for clean). For
    validation, always pass `"clean"` and `0.0`, so switching the train mode
    does not re-prep validation.
  - Train selection (each call receives the staging `cache_dir`):
    - `paired` → `prepare_paired_dataset(raw_train, cache_dir=staging)`.
    - `disjoint` → `prepare_dataset_split(raw_train, cache_dir=staging)` with
      the configured prob.
    - `clean` → `prepare_dataset_split(raw_train, cache_dir=staging)` with
      prob 0.0.
  - Log the mode and sizes, e.g. `Paired train set: 58,244 clean + 58,244 phone
    = 116,488 rows`.
  - The cache writer maps into a staging directory, validates the shard layout,
    writes `meta.json` last, then atomically renames the staging directory into
    `prep_cache_dir/train`. On a crash or validation failure, the staging
    directory is removed; the loader treats an absent/invalid metadata file as
    a miss.
  - Update the reused-cache note to say the cached draw (both halves in paired
    mode) is frozen; `--overwrite_prep_cache` forces a fresh draw.
  - Resume rule: cockpit resume reuses the stored mode; CLI resume of an old
    disjoint run must pass `--telephony_mode disjoint` (and `--epochs 4`) or
    start fresh. Document this in the help and docs.
- `configs/l40s_train_config.yaml`:
  - `training.epochs: 2`.
  - `telephony.mode: paired`.
  - Keep `telephony.prob: 0.5` with a comment that it applies to `disjoint`.
- `_apply_yaml_config`: map `mode`/`telephony_mode` to `telephony_mode`.
- `run_train.sh`: `--epochs 2`, `--telephony_mode paired`.
- `server/app.py`:
  - `_normalize_start_config` reads and validates `telephony_mode` (default
    `paired`), returns it, and the default epochs becomes 2.
  - `_TRAINER_VALUE_ARGS` includes `telephony_mode`.
  - `trainer_args`, `run_config`, and `response_config` include the mode.
  - `_start_resume_run` (and any other resume path) must treat a stored run
    without `telephony_mode` as `disjoint`, because every pre-change run was
    disjoint. Do not let an old checkpoint silently resume as paired.
  - Smoke endpoint accepts and validates `telephony_mode` (default `paired`)
    and passes it. The smoke set stays 200 raw rows, 1 epoch; paired doubles it
    to 400 prepared rows.
- Cockpit UI:
  - `server/static/index.html`: add a `Training data` select
    (`id="inputTelephonyMode"`) with paired (selected), disjoint, clean. Change
    `inputEpochs` default to `2`.
  - `server/static/app.js`: `DEFAULT_EPOCHS = 2`; add the select to `el`; send
    `telephony_mode` in `/api/train/start`; on mode change set epochs to 2 for
    paired and 4 for disjoint/clean. Keep a small pure helper
    (`epochsForTelephonyMode`) so the Node test can cover the mapping.
  - `server/static/run_view.js`: `formValuesFromConfig` maps
    `telephony_mode`; `restoreRunConfig` restores the select and the matching
    epoch value for the selected historical run.
  - Optional run header text showing the mode is a nice-to-have, not required;
    the run config stores it either way.

### 7.3 Cache and disk behavior

**Chosen strategy: direct shard cache.** `datasets.map(..., cache_file_name=...)`
writes its Arrow shards straight into the prep cache. In datasets 3.6.0 with
`num_proc > 1`, the map writes `data_00000_of_00004.arrow`, … and returns a
concatenation of those shards; it does not write a final combined `data.arrow`
and does not keep a separate HF map-cache copy. With `num_proc=1`, the exact
`cache_file_name` is written. The cache is the shard files plus a metadata file;
loading uses `Dataset.from_file` per shard and `concatenate_datasets`.

- The train cache remains a single `prep_cache_dir/train` slot keyed by the
  fingerprint. A mode switch misses and re-preps.
- On a fingerprint miss, remove the stale `train` cache before mapping (after
  the free-space check) so the old disjoint ~85 GB does not coexist with the
  new paired ~170 GB.
- Free-space guard: require at least ~180 GB projected free (current free plus
  the stale train cache that will be removed) before the first paired map;
  abort with a clear message if not. The overlay has ~250 GB+ free.
- Peak and steady state are both ~170 GB (plus ~25 GB raw/model on the same
  disk). No `save_to_disk` duplicate and no `cleanup_cache_files()` step.
- Metadata (`meta.json`) contains a format version, the full fingerprint, the
  ordered shard list with row counts and byte sizes, the total row count, and a
  feature-schema signature. It is written last. On load, validate all of it;
  any missing shard, wrong count, or schema mismatch is a miss, not a partial
  load. The dataset returned by the pipeline exposes its ordered shard paths
  through `cache_files`; the cache writer uses that list for `meta.json`.
- Publication: map into `prep_cache_dir/train.staging-<pid>`; validate; write
  `meta.json`; remove any old `train`; atomically rename the staging directory
  to `train`. A crash leaves only the staging directory, which is cleaned on the
  next attempt.
- Support both layouts: multi-process shard files and the single-process
  `data.arrow` fallback (the existing multiprocess-to-single-process retry can
  change the layout). The metadata, not a glob, is authoritative.
- Validation cache is mode-independent, uses the same shard format, and
  survives train-mode changes.
- To keep both paired and disjoint caches, use a different `--prep_cache_dir`
  (or `ASR_PREP_CACHE_DIR`) per mode. Document this; do not create two slots
  automatically.
- Existing `save_to_disk` caches are invalidated by the new fingerprint field
  and the new metadata format. No legacy loader; the first run replaces them.

### 7.4 Evaluation behavior

- Validation: clean, `telephony_prob=0.0`, unchanged.
- Preflight: unchanged; it evaluates the test split, not training data.
- Per-epoch benchmark: unchanged; clean + deterministic phone pass on the
  benchmark subset.
- Final full-test report: unchanged; clean + phone, sample-count-gated deltas.
- The phone row in training is a fixed precomputed realization per utterance,
  the same across epochs. That matches the cache model and keeps prep
  one-time. Per-epoch variation is a separate future feature.

### 7.5 Documentation

- `README.md` key flags: paired default, 2 epochs, 116,488 rows, ~170 GB shard
  cache (peak ≈ steady; ~250 GB overlay), disjoint command with
  `--telephony_prob 0.5 --epochs 4`, clean command with `--epochs 4`.
- `GUIDE.md` training table: replace the 50/50 row with paired 1:1; update the
  epoch/duration and cache notes; keep the same total optimizer steps.
- `PRODUCTION.md`: shard-cache format/validation, free-space requirement, mode
  fingerprint, resume rule, current checkpoint expectations.
- `configs/l40s_train_config.yaml` comments.

## 8. Testing

- `tests/test_data_prep.py`:
  - paired output has exactly `2 × filtered` rows;
  - corresponding clean/phone rows have identical labels and `is_informal`;
  - phone features differ from clean for the sampled rows;
  - the phone half is deterministic across runs and independent of num_proc;
  - duration filtering removes the same utterances from both halves;
  - empty input remains empty;
  - unavailable telephone module aborts paired prep;
  - a per-row phone degradation failure aborts paired prep.
- `tests/test_prep_cache.py` (new):
  - multi-process map publishes only shard files plus `meta.json`, with no
    final `data.arrow` and no HF map-cache copy;
  - single-process fallback publishes one `data.arrow` and loads;
  - fingerprint mismatch, missing shard, wrong row count, or schema mismatch is
    a cache miss;
  - a crash leaves no valid `train` cache (staging only);
  - the free-space guard rejects an undersized volume;
  - `datasets.disable_caching()` still honors an explicit `cache_file_name`.
- `tests/test_train_lora_run_stages.py`: update the old `_save_prep_cache`
  monkeypatch to the new staging/publish writer.
- `tests/test_script_defaults.py`:
  - parser default mode is `paired`, epochs 2;
  - config has mode paired and epochs 2;
  - launcher has `--epochs 2` and `--telephony_mode paired`;
  - `_prep_fingerprint` differs between paired/disjoint/clean and is stable for
    validation across train modes.
- `tests/test_server_train.py`:
  - default normalized config has mode paired and epochs 2;
  - built command includes `--telephony_mode paired`;
  - explicit disjoint/clean survives;
  - resume of a legacy run without the field uses `disjoint`;
  - smoke command includes the mode.
- `tests/test_run_view.js` / `tests/test_web_ui_contract.py`:
  - `formValuesFromConfig` maps the mode;
  - index has the select with paired selected and epochs 2;
  - app.js sends `telephony_mode`;
  - `epochsForTelephonyMode` returns 2/4/4.
- Full suite, `node --check` on changed JS, `bash -n run_train.sh`,
  `git diff --check`.

## 9. Risks and mitigations

- **Disk headroom:** ~170 GB shard cache + ~25 GB raw/model on a ~250 GB+
  overlay; log free space and abort early if below the guard. No duplicate
  map/save copy.
- **Prep time (~2×):** one-time; the prep cache is reused afterward.
- **Pairing bug:** tests assert identical labels, deterministic phone rows, and
  exact 2× counts.
- **Silent clean-only training:** paired prep aborts if the telephone module is
  unavailable or any row fails phone degradation.
- **Partial cache:** atomic staging publish + metadata-last validation; a
  missing or invalid shard is a miss, never a partial load.
- **Stale cache:** mode is in the fingerprint; validation fingerprint stays
  clean.
- **Resume mismatch:** cockpit stores/injects the mode; CLI is documented; old
  runs default to disjoint on resume.
- **UI/trainer mismatch:** parser, config, launcher, server, and UI all move to
  2 epochs together; tests assert each.
- **Preflight invalidation:** none; the preflight fingerprint is untouched.

## 10. Decisions made in this design

- Approach A (two passes + concatenate).
- Paired is the default; disjoint and clean remain explicit modes.
- Default epochs 2; disjoint/clean use 4 in the cockpit and docs.
- The cockpit gets a mode selector (paired preselected, epoch auto-fill).
- Paired ignores `--telephony_prob` with a note, not an error.
- Direct shard cache: map writes shards into the prep cache; no `save_to_disk`
  duplicate. Peak ≈ steady ≈ 170 GB.
- Atomic staging publish with metadata last; validate fingerprint, shard list,
  row counts, and feature schema on load.
- Free-space guard (~180 GB) and stale-cache removal before the first paired
  map.
- Paired prep fails closed on any phone-augmentation failure.
- The train cache is a single slot; separate `--prep_cache_dir` values keep
  both modes cached.
- Cockpit resume of a pre-change run treats it as disjoint.
- Smoke test uses paired by default (400 prepared rows).
- Training phone rows are precomputed and fixed across epochs.
