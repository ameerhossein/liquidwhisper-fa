# Customer High-Band Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native) or superpowers:subagent-driven-development. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Correct the customer 1/3-octave target normalization and re-fit the EQ so the simulator no longer over-cuts 1600–3150 Hz relative to the Outside recordings.

**Architecture:** Add a per-file normalized 1/3-octave shape helper, use it for both the real target and synth measurement, re-fit from `CUSTOMER_EQ_INITIAL_GAINS_DB`, then regenerate the validation artifacts and docs. The runtime profile, aliases, level range, and A-law chain stay unchanged.

**Tech Stack:** Python 3.12, NumPy, SciPy, SoundFile, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md`

## Global Constraints

- One `customer` profile; `gated`/`clean` remain aliases; `mobile`/`pstn` remain explicit legacy profiles.
- Keep the 300–3400 Hz band-pass, low shelf, RMS calibration, silence gate, and A-law order unchanged.
- Do not change `configs/l40s_train_config.yaml`; it already uses `profiles: ["customer"]`.
- Work on `main`; do not push.
- Use the same DeepSeek V4.1 Flash model for any subagent review.
- The fit must be reproducible from `CUSTOMER_EQ_INITIAL_GAINS_DB` with documented iterations/damping.

## Review Focus

- Per-file target: `read_bands` must take the median of `(band − that file's 1 kHz)` values, not `median(band) − median(1 kHz)`.
- Synth measurement: `synth_levels` must use the same per-file normalization.
- High-band gains must move materially toward the target; 3150 Hz must no longer sit near −19.8 dB.
- Low/mid Outside match should stay near 2 dB. Accepted tradeoff: Outside
  500 Hz may remain ~+3 dB because the all-67 target is 2.3 dB brighter there;
  the all-67 residual must stay ≤1 dB.
- Report/spec target tables and JSON artifacts must match the new method and constants.

---

### Task 1: Add failing analysis tests for per-file normalization

**Files:**
- Create: `tests/test_customer_eq_analysis.py`
- Test: `tests/test_customer_eq_analysis.py`

**Interfaces:**
- Consumes: `RecordingTesting/analysis/fit_customer_eq.py` (`read_bands`), `RecordingTesting/analysis/gated_spectrum_check.py` (`octave_shape`, to be added).
- Produces: regression tests for the corrected statistic.

- [x] **Step 1: Write the failing tests**

```python
def test_read_bands_uses_median_per_file_shape(tmp_path):
    # rows chosen so median(per-file shape) == -30, but
    # median(band) - median(1k) == -10
    ...

def test_octave_shape_is_median_of_per_file_shapes():
    # two PSDs with per-file 3150/1k shapes -20 and -10 dB
    # expected median -15 dB, pooled median is about -10 dB
    ...
```

- [x] **Step 2: Run tests and verify they fail for the expected reason**

Run: `.venv/bin/python -m pytest tests/test_customer_eq_analysis.py -q`
Expected: `read_bands` returns −10 instead of −30; `octave_shape` is missing.

- [x] **Step 3: Commit the failing tests**

```bash
git add tests/test_customer_eq_analysis.py
git commit -m "test: pin per-file customer EQ target normalization"
```

### Task 2: Implement per-file normalized analysis

**Files:**
- Modify: `RecordingTesting/analysis/gated_spectrum_check.py`
- Modify: `RecordingTesting/analysis/fit_customer_eq.py`
- Test: `tests/test_customer_eq_analysis.py`

**Interfaces:**
- Produces: `octave_shape(psds, freqs, centers=CENTERS, ref_hz=1000.0) -> np.ndarray`
- Produces: `read_bands(path) -> dict[float, float]` using per-file normalization.

- [x] **Step 1: Add `octave_shape`**

Compute each PSD's 1/3-octave levels, subtract its own 1 kHz level, then take the median across PSDs.

- [x] **Step 2: Use it in `fit_customer_eq.py`**

`read_bands` normalizes each CSV row by its own `hz_1000` before medians.
`synth_levels` calls `gs.octave_shape(psds, freqs)`.

- [x] **Step 3: Use it in `gated_spectrum_check.py` main**

Replace `octave_levels(real/clean/synth, freqs)` with `octave_shape(...)` so the report table uses the corrected statistic.

- [x] **Step 4: Run the analysis tests**

Run: `.venv/bin/python -m pytest tests/test_customer_eq_analysis.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add RecordingTesting/analysis/gated_spectrum_check.py RecordingTesting/analysis/fit_customer_eq.py
git commit -m "fix: use per-file normalized 1/3-octave targets"
```

### Task 3: Re-fit EQ constants with TDD

**Files:**
- Modify: `tests/test_telephone.py`
- Modify: `src/telephone.py`
- Modify: `RecordingTesting/analysis/customer_eq_gains.json`
- Test: `tests/test_telephone.py`

**Interfaces:**
- Consumes: corrected `fit_customer_eq.py`.
- Produces: new `CUSTOMER_EQ_GAINS_DB`.

- [x] **Step 1: Write the failing gain guard**

Replace the old `CUSTOMER_EQ_GAINS_DB[3150] < -15.0` assertion with a regression guard that the high band is not over-cut:

```python
def test_customer_eq_high_band_not_overcut():
    from src import telephone
    gains = telephone.CUSTOMER_EQ_GAINS_DB
    assert gains[1600.0] > 4.0
    assert gains[2000.0] > 1.0
    assert gains[2500.0] > 4.0
    assert gains[3150.0] > -10.0
```

- [x] **Step 2: Run it and verify RED**

Run: `.venv/bin/python -m pytest tests/test_telephone.py::test_customer_eq_high_band_not_overcut -q`
Expected: FAIL on the current constants.

- [x] **Step 3: Re-fit from the documented start**

Run:

```bash
.venv/bin/python RecordingTesting/analysis/fit_customer_eq.py \
  --initial-gains initial --iterations 8 --damping 0.5 \
  --out-dir /tmp/opencode/customer_eq_perfile_final
```

Expected: residuals ≤ ~1.5 dB at all primary bands; 3150 Hz gain near −7 dB, not −20 dB.

- [x] **Step 4: Paste the fitted gains into `src/telephone.py`**

Update `CUSTOMER_EQ_GAINS_DB` and the comment. Keep `CUSTOMER_EQ_Q_BY_BAND = {3150.0: 3.0}` unless the fit shows a narrower/wider Q is required.

- [x] **Step 5: Run the targeted tests**

Run:
`.venv/bin/python -m pytest tests/test_telephone.py -k "customer_eq or high_band" -q`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/telephone.py tests/test_telephone.py RecordingTesting/analysis/customer_eq_gains.json
git commit -m "fix: re-fit customer high-band EQ on per-file targets"
```

### Task 4: Regenerate validation artifacts and Outside comparison

**Files:**
- Modify: `RecordingTesting/analysis/customer_octave_levels.csv`
- Modify: `RecordingTesting/analysis/customer_octave_levels.png`
- Modify: `RecordingTesting/analysis/multi_voice_files.csv`
- Modify: `RecordingTesting/analysis/multi_voice_summary.json`
- Modify: `RecordingTesting/analysis/customer_eq_holdout.json`
- Modify: `RecordingTesting/analysis/caller_class_medians.csv`
- Modify: `RecordingTesting/analysis/caller_class_medians.json`
- Modify: `RecordingTesting/analysis/report.md`
- Modify: `docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md`

- [x] **Step 1: Run the 67-file 1/3-octave validation**

```bash
.venv/bin/python RecordingTesting/analysis/gated_spectrum_check.py \
  --content-manifest RecordingTesting/analysis/answered_customer_manifest.csv \
  --n-sources 46 --seeds 1 2 3 4 5 \
  --out-dir /tmp/opencode/customer_spectrum_refit
```

- [x] **Step 2: Run the multi-voice validation**

```bash
.venv/bin/python RecordingTesting/analysis/multi_voice_check.py \
  --profiles customer --voice-dir /home/ameer/Hobbies/voiceprint/fa_clips \
  --seeds 1 2 3 4 5 --out-dir /tmp/opencode/customer_multi_voice_refit
```

- [x] **Step 3: Re-run the Outside-only per-file comparison**

Use the comparison script/artifacts from `/tmp/opencode/outside_customer_compare/`; assert the 1600–3150 Hz per-file residuals are now within about ±2 dB.

- [x] **Step 4: Run the held-out speaker fit**

```bash
.venv/bin/python RecordingTesting/analysis/fit_customer_eq.py \
  --initial-gains initial --iterations 8 --damping 0.5 \
  --holdout-speakers spk_91858 \
  --out-dir /tmp/opencode/customer_eq_holdout_refit
```

- [x] **Step 5: Copy artifacts and update report/spec**

Update the target table, final medians, the “within ±2 dB” qualifier, and add an Outside-only validation section. Remove the “secondary to the 1/3-octave fit” framing.

- [x] **Step 6: Commit**

```bash
git add RecordingTesting/analysis docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md
git commit -m "docs: record corrected customer high-band calibration"
```

### Task 5: Full verification and review

- [x] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [x] **Step 2: Confirm the Outside high-band result**

Report the per-file 1600/2000/2500/3150 residuals and the speech RMS/noise floor.

- [x] **Step 3: Run a read-only review with DeepSeek V4.1 Flash**

Review the diff against this plan; fix critical/important findings before calling it done.

- [x] **Step 4: Final commit if review fixes are needed**
