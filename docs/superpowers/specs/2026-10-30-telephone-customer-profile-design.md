# Telephone customer profile — design

**Date:** 2026-10-30  
**Status:** Draft for approval  
**Scope:** Replace the three-profile telephony default with one `customer` profile, renamed/recalibrated from `gated`.

---

## Intent

The telephone simulator should reproduce the channel that the actual customers are recorded on, not a mixture of customer speech, ringback music, and internal extensions.

The 2026-10-22 `RealRecordedPhoneCalls` corpus and the larger `~/TerasVoice/Outside` corpus were re-checked with an AudioSet content classifier. A large share of the recordings are **ringback/hold music before pickup**, not customer speech. That contamination means the current `gated` profile was fitted to a speech+music mixture and is too dark; the current `mobile` profile was fitted to noisy internal extensions and is too quiet/noisy. Neither is the right customer profile.

The chosen outcome is **one default profile, `customer`**, built from the `gated` chain (level, gating, A-law) and recalibrated to **answered customer speech only**. `mobile` and `pstn` remain callable explicitly as legacy profiles but are removed from all defaults.

---

## What we learned

### Content classification (AudioSet AST)

The AST model (`MIT/ast-finetuned-audioset-10-10-0.4593`) was run on 10-second windows. Per-file label = majority over the first/middle/last windows.

**Outside 213-file sample:**

| Channel class | n | Answered speech | Music→speech | Music only |
|---|---:|---:|---:|---:|
| `gated` (sample) | 50 | 31 | 0 | 19 |
| `quiet_noisy` | 111 | 97 | 9 | 4 |
| `noisy` | 52 | 9 | 1 | 42 |

**Original `RealRecordedPhoneCalls` (107 files):**

| Content class | n |
|---|---:|
| speech only | 62 |
| music only | 38 |
| music then speech | 4 |
| mixed | 1 |
| empty | 2 |

Of the 77 `gated` files, **42 are answered speech and 35 are music only**. The current `gated` profile was therefore calibrated against an approximately 45%-music mixture. Music-only gated files are very dark (HF 1–2 kHz ≈ −39 dB) while answered gated speech is much brighter (≈ −15 dB), which explains the profile's 12–13 dB high-band deficit.

### Answered customer target

Calibration set: **67 answered gated mobile-customer files**, all `speech_only`, all `is_gated=True`:

- 36 files from the in-repo 2026-10-22 corpus with mobile caller numbers (`speech_only`, not the 39 that include 3 `music_then_speech`).
- 31 files from the `Outside` gated answered sample (`2026-10-13`–`2026-10-22`); all 31 are mobile callers.

Landline customers are **out of scope** for this profile: they are 2.6% of the Outside customer files and are mostly gated, but the one-profile target is the mobile-customer population. They can use the legacy `pstn` profile if needed. This keeps the calibration set mobile-only and matches the target medians exactly.

| Metric | Median | IQR |
|---|---:|---:|
| Active speech RMS | **−25.6 dBFS** | −28.5 … −19.5 |
| Noise floor | **−71.3 dBFS** | −72.0 … −69.0 |
| HF 1–2 kHz rel. 0.3–1 kHz | **−15.1 dB** | −24.3 … −9.2 |
| HF 2–3.4 kHz rel. 0.3–1 kHz | **−27.5 dB** | −33.8 … −20.8 |
| Speech centroid | **459 Hz** | 450 … 545 |
| 95% rolloff | **688 Hz** | 563 … 1,094 |
| Speech fraction | **0.70** | 0.70 … 0.70 |

### Current profiles vs target

| Profile | Speech | Noise | HF 1–2k | HF 2–3.4k | Centroid | Roll95 |
|---|---:|---:|---:|---:|---:|---:|
| **Target** | −25.6 | −71.3 | −15.1 | −27.5 | 459 | 688 |
| Current `gated` | −21.6 | −72.8 | −27.7 | −39.8 | 493 | 688 |
| Current `mobile` | −31.9 | −52.0 | −11.5 | −24.6 | 445 | 688 |

Read:

- `gated` has the right noise floor, gating, A-law storage, and rolloff; it is **4 dB too loud** and **12–13 dB too dark** in the 1–3.4 kHz region.
- `mobile` has a high-band tilt close to the target, but its level is **6 dB too quiet** and its noise floor **19 dB too noisy**. It models a noisy extension, not a customer.
- The correct profile is therefore **gated structure + a substantially brighter 1–3.4 kHz response**, not a tuned `mobile` and not the current `gated` unchanged.

Centroid and rolloff are **not fit targets**. The current gated centroid (493 Hz) and roll95 (688 Hz) are already inside the answered-speech IQR (450–545 Hz, 563–1,094 Hz). A brighter high band may raise the centroid, but the EQ fit is driven by the 1/3-octave bands; if the two pull in different directions, the 1/3-octave HF fit wins.

---

## Goals

1. Make `customer` the only default telephony profile.
2. Recalibrate the customer channel to the answered-speech target above.
3. Keep `gated`/`clean` as aliases for `customer` so existing call sites keep working.
4. Keep `mobile` and `pstn` callable explicitly as legacy profiles, but remove them from defaults, scripts, reports, and docs.
5. Publish the answered/music labels and the calibration target so the result is reproducible.
6. Add regression tests for the new default, the alias behavior, and the customer chain.

## Non-goals

- Detecting music/ringback at training or inference time. Content classification is an offline analysis step only.
- Modeling internal extensions (out of scope by decision).
- Modeling the cellular/AMR network leg. The corpus is already post-network A-law; AMR is not directly observable and is not added to the customer chain.
- Changing the ASR model, training loop, or dataset.
- Modeling multi-turn conversation structure; this change is about the acoustic channel only.

## Anti-goals

- A profile that matches the **music** population (too dark, too loud) rather than answered speech.
- A profile that matches the **noisy extension** population (too quiet, too noisy) and calls it “mobile.”
- Silently changing `profile="mobile"` semantics while scripts still expect the noisy-line behavior.
- Removing explicit legacy profile support without a migration path.

---

## Approach chosen

Three naming options were considered:

1. Replace `mobile` with the new customer chain.
2. Add `customer` as the new default; keep `mobile`/`pstn` as explicit legacy.
3. Rename/recalibrate `gated` to `customer`; drop `mobile`/`pstn` from defaults.

**Chosen: option 3.** It reuses the profile that already has the right structure (level, gating, A-law), gives it an accurate name, and removes the out-of-scope profiles from every default. `gated` and `clean` become aliases, so existing explicit calls continue to work.

---

## Profile specification

### Chain

```
16 kHz input
  → resample to 8 kHz
  → 300–3400 Hz band-pass (existing order-5 Butterworth)
  → handset/channel EQ fitted to the answered-speech target
  → active-RMS calibration to the customer range
  → optional small gain jitter
  → silence gate (−35 dB, 150 ms hangover, 5 ms fade)
  → G.711 A-law round-trip
  → optional 16 kHz upsample
```

No line-noise bed, no dropouts, no AMR in the default customer chain. The answered gated population has a −71.3 dBFS noise floor; the current A-law/gate chain produces about −72.8 dBFS, within 2 dB of the target. No noise source is added for v1.

### Constants

Replace the gated constants with customer equivalents:

- `CUSTOMER_PROFILES = ("customer",)`
- `DEFAULT_PROFILES = ("customer",)`
- `DEFAULT_PROFILE_WEIGHTS = {"customer": 1.0}`
- `CUSTOMER_TARGET_RMS_DB_RANGE = (-28.5, -21.5)` — target median −25.6 dBFS; keep the width for training diversity. Builder may tighten to (−28.0, −23.0) if validation shows too much spread.
- Keep `GATED_*` names only as deprecated aliases if needed; prefer renaming all internal references.
- Keep `MOBILE_*` and `PSTN_*` constants for explicit legacy use; do not use them in defaults.
- Keep `LEGACY_PROFILE_WEIGHTS = {"gated": 0.70, "mobile": 0.15, "pstn": 0.15}` for analysis scripts. Explicit `gated,mobile,pstn` remains **uniform 1/3** unless weights are supplied; the old 70/15/15 is not the default. Historical comparisons can pass the legacy weights explicitly. No new CLI flag is required.

### EQ fitting

Do **not** hard-code a +13 dB shelf from the table. Fit the filter empirically:

1. Recompute the answered-only loud-frame 1/3-octave PSD relative to 1 kHz for the 67-file calibration set.
2. Compare to the current `simulate_customer` output on the same clean-voice set.
3. Start from the current gated chain, remove or reduce the 4 dB low shelf at 500 Hz, and add a high shelf / 1/3-octave correction in the 1–3.4 kHz region.
4. Iterate until the synth median is within **±2 dB per 1/3-octave band from 1–3.4 kHz** and the broadband metrics are within the target IQR.
5. Validate on held-out voices; reject filters that overfit one speaker or sound unnaturally bright.

The existing `RecordingTesting/analysis/gated_spectrum_check.py` is the starting point; it must learn a `--content-manifest` filter so music-only files are excluded.

### Gating / speech fraction

Keep the current gate settings. Validate the customer profile's speech fraction in **0.60–0.75**; the 0.70 target is content-dependent and the current gated chain already measures 0.62 [0.53–0.68] on the 46-voice set. Do not retune the gate in this change; revisit only if gate clicks or a clear speech-fraction failure appear.

---

## Reference data and manifests

Add committed analysis manifests (no runtime dependency on AST):

- `RecordingTesting/analysis/answered_customer_manifest.csv`
  - Columns: `file`, `source` (`in_repo` / `outside`), `caller_class`, `is_gated`, `content_class`, `duration_s`, `speech_rms_db`, `noise_rms_db`, `hf_1_2k_rel_db`, `hf_2_3k_rel_db`, `spec_centroid`, `spec_roll95`, `calibration`.
  - Rows: the 107 `RealRecordedPhoneCalls` files with AST content labels, plus the 213-file `Outside` sample.
  - The `Outside` sample is exactly: all 163 readable `noisy` + `quiet_noisy` customer calls, plus 50 random `gated` customer calls sampled with `random.seed(42)` from the 1,842 readable gated customer calls. The authoritative list is `content_sample_out/music_files.csv`; copy it into the manifest.
  - The 67-file calibration subset is the rows with `content_class=speech_only`, `is_gated=true`, `caller_class=mobile`, and `calibration=true`.
- `RecordingTesting/analysis/answered_customer_bands.csv`
  - One row per calibration file, one column per 1/3-octave band (dB relative to 1 kHz, loud-frame median). This lets the EQ fit be reproduced without the external `Outside` audio.
  - The in-repo 36 audio files remain available; the 31 Outside files are only represented by their committed band rows and metadata.
- A short `RecordingTesting/analysis/answered_customer_manifest.md` explaining the AST model, the windowing, the label families, the 213-file sampling, and the limitation that labels are model-derived and should be spot-checked.

Do not add `transformers`/AST to `requirements.txt`; this is offline analysis only.

**Privacy note:** the manifest contains real call filenames, which include customer phone numbers, consistent with the existing `RealRecordedPhoneCalls` files in the repo. If the repo's data policy requires it, store a stable hash as the `file` value and keep the hash→path mapping local.

---

## Code changes

### `src/telephone.py`

- Module docstring: replace the 70/15/15 default description with the single `customer` default; document the music-contamination correction.
- Rename `GATED_PROFILES`/`GATED_TARGET_RMS_DB_RANGE` to `CUSTOMER_*`.
- Rename `simulate_gated` to `simulate_customer`; keep `simulate_gated` as a deprecated wrapper.
- `DEFAULT_PROFILES = ("customer",)`; `DEFAULT_PROFILE_WEIGHTS = {"customer": 1.0}`.
- `simulate_telephone`: map `customer`, `gated`, `clean` to `simulate_customer`; keep `mobile`/`pstn` branches.
- `degrade_for_eval_profiled`: map `clean`/`gated` to `customer`; default profiles one.
- `apply_random_telephone` docstring: remove the 70/15/15 wording.
- `choose_telephone_profile`: with one default profile, still support explicit multi-profile lists and weights.

### `src/metrics.py`

- `_canonical_profile_name`: map `clean` and `gated` to `customer`; leave `mobile`/`pstn`.
- `profile_weights`: default weights now `{"customer": 1.0}`; explicit subsets of legacy profiles remain uniform unless weights are supplied.

### `src/data.py`

- Default `telephony_profiles` comes from `DEFAULT_PROFILES`, now `("customer",)`.
- Update the comment that says 70% gated / 15% mobile / 15% PSTN.
- `telephony_prob` semantics unchanged.

### `src/trainer_callbacks.py`

- Default `self.telephony_profiles` becomes `["customer"]`.
- Keep `DEFAULT_PROFILE_WEIGHTS` usage; reports now show one profile.
- `phone_profiles` / `phone_profile_weights` outputs remain, with one entry.

### `scripts/train_lora.py`

- Default `--telephony_profiles` becomes `"customer"`.
- `_parse_telephony_profiles` default becomes `["customer"]`.
- Post-training evaluation default becomes `["customer"]`.
- Update help text and comments.

### `scripts/evaluate_model.py`

- Default `--telephony_profiles` becomes `"customer"`.
- `_parse_profiles` default becomes `["customer"]`.
- Per-profile WER now reports one profile; weighted average equals that profile.

### Tests

- `tests/test_telephone.py`
  - Replace `test_default_profiles_include_gated` with a customer-default test: `DEFAULT_PROFILES == ("customer",)`, weights sum to 1.0.
  - Add alias test: `simulate_telephone(..., profile="gated")` equals `simulate_telephone(..., profile="customer")` for the same seed.
  - Add `simulate_customer` determinism / length / A-law-grid test.
  - Adjust `test_choose_telephone_profile_is_weighted` to test an explicit `["customer", "mobile"]` set, not the default.
  - Keep explicit `mobile`/`pstn` tests.
- `tests/test_normalizer_metrics.py`
  - Update alias expectations: `clean`/`gated` → `customer`.
- `tests/test_data_prep.py`
  - Explicit `["mobile", "pstn"]` test remains valid.
- Add a regression test that the customer profile's active RMS is within ±1 dB of −25.6 and its noise floor is between −75 and −69 dBFS on a fixed reference clip (the current chain's −72.8 is acceptable).

### Analysis scripts

- `RecordingTesting/analysis/gated_spectrum_check.py`
  - Add `--content-manifest`; default to the answered customer manifest when present.
  - Rename output labels to `customer`; keep `gated` aliases.
- `RecordingTesting/analysis/multi_voice_check.py`
  - Default profile `customer`; compare only to answered customer files.
- `RecordingTesting/analysis/compare_telephone.py`
  - Update group labels and the default real-corpus filtering; keep the historical report reproducible with an explicit `--legacy` flag if needed.
- `RecordingTesting/analysis/caller_class_check.py`
  - Use `customer` in the synth comparison; optionally include content class.
- `RecordingTesting/analysis/smoke_wer.py`, `make_listening_kit.py`
  - Default `--profiles` becomes `["customer"]`; legacy profiles remain explicit choices.
- `RecordingTesting/analysis/report.md`
  - Add a **2026-10-30 correction** section documenting the music contamination, the 67-file answered target, and the profile recalibration.

### Docs

- `src/telephone.py` docstring and `src/data.py` comments.
- `README.md` / `GUIDE.md` / `PRODUCTION.md` only if they mention telephony profiles; currently they do not.
- Add the design doc path to the report's “Reproduce” section.

---

## Validation plan

1. **Target fit:** Run `multi_voice_check.py` on the 46 clean voices with `customer`. Compare medians to the 67-file target:
   - Active RMS within ±1 dB of −25.6.
   - Noise floor within ±2 dB of −71.3 **or inside the target IQR (−72.0 … −69.0)**. The current gated chain measures −72.8, which is 1.5 dB below the median and 0.8 dB below the IQR; that is acceptable for v1 and no noise source is added.
   - HF 1–2 kHz within ±2 dB of −15.1.
   - HF 2–3.4 kHz within ±2 dB of −27.5.
   - Centroid/roll95 inside the target IQR (450–545 Hz, 563–1,094 Hz). Do not chase the medians.
2. **Spectral fit:** Re-run the 1/3-octave loud-frame comparison on answered-only files; every band 1–3.4 kHz within ±2 dB. This is the authoritative fit.
3. **Format:** A-law grid fraction 1.00, 8 kHz native levels ≤256, 16 kHz output length preserved.
4. **Gating:** Speech fraction **0.60–0.75** on the reference set. The 0.70 target is content-dependent (the real files' IQR is 0.70–0.70, not a typo); the current gated chain is 0.62 [0.53–0.68] on 46 voices, and retuning the gate is **not** part of this change. No gate clicks (boundary discontinuity no worse than the real answered files).
5. **Determinism:** Same seed → identical output; different seeds → expected variation.
6. **Backward compatibility:** `profile="gated"` and `profile="clean"` produce the customer chain; `profile="mobile"`/`"pstn"` still work explicitly.
7. **Integration:** `PersianASRDataPipeline.prepare_sample` with `telephony_prob=0.5` draws only `customer` by default; length preserved.
8. **Tests:** Full `pytest` suite passes.
9. **Listening:** A/B three answered customer calls against the customer synth; confirm it is not too dark or too bright. The user should do at least one Persian listening pass.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 67-file calibration set is small / one call centre | Use the in-repo 36 mobile + Outside 31 mobile; validate on held-out days/voices; target medians and IQRs, not exact curves. |
| AST labels are model-derived and may mislabel some speech/music | Spot-check 20 files by listening; keep the manifest editable; do not use labels at runtime. |
| A +13 dB high-band correction sounds unnatural | Fit 1/3-octave gains, cap individual shelf gains, validate with multi-voice and listening. |
| Level range too wide/narrow | Start with the target IQR; tighten if the generated level distribution is too spread. |
| Existing reports/tests assume 70/15/15 | Update in the same change; keep legacy explicit profiles for old analyses. |
| `gated` alias hides the semantic change | Document that `gated` now means the recalibrated customer chain; update docs and report. |

---

## What is left to the builder

- Measure the current gated chain on the 67-file answered set and fit the EQ.
- Choose the exact filter topology (shelves vs 1/3-octave FIR/IIR) after measurement.
- Decide the final `CUSTOMER_TARGET_RMS_DB_RANGE` within the target IQR.
- Decide whether to keep a deprecated `simulate_gated` wrapper or only a name alias.
- Add the manifest CSVs and the report correction.
- Run the validation plan and report the before/after table.

## Open questions

None blocking. The 9% answered quiet-noisy calls are intentionally left out of the single profile for v1; if they matter later, add a low-level noise variant rather than changing the customer default.
