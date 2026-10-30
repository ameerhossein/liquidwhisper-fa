# Telephone Customer Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three-profile telephony default with one `customer` profile, renamed/recalibrated from `gated` and fitted to 67 answered gated mobile-customer recordings.

**Architecture:** Rename the existing gated chain to `customer`, add a measured 1/3-octave peaking EQ, canonicalize `gated`/`clean` aliases, and remove `mobile`/`pstn` from every default. The legacy chains stay callable explicitly. Content labels and target bands are committed analysis artifacts; AST is not a runtime dependency.

**Tech Stack:** Python 3.12, numpy, scipy, soundfile, pytest. Existing repo `.venv` is the test environment.

**Spec:** `docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md`

## Global Constraints

- Default profiles: `DEFAULT_PROFILES = ("customer",)`, `DEFAULT_PROFILE_WEIGHTS = {"customer": 1.0}`.
- `gated` and `clean` are aliases for `customer`. `mobile` and `pstn` are explicit legacy profiles only.
- The customer chain is gated structure: 16 kHz → 8 kHz → 300–3400 band-pass → 1/3-octave EQ → active-RMS calibration → silence gate → A-law. No line-noise bed, no dropouts, no AMR.
- Calibration target (67 answered gated mobile-customer files): speech −25.6 dBFS, noise −71.3 dBFS, HF 1–2k −15.1 dB, HF 2–3.4k −27.5 dB, centroid 459 Hz (IQR 450–545), roll95 688 Hz (IQR 563–1,094), speech fraction 0.70.
- Primary 1/3-octave fit: **315–3150 Hz within ±2 dB**. 125–250 Hz and 4000 Hz are secondary.
- Legacy weights: `LEGACY_PROFILE_WEIGHTS = {"gated": 0.70, "mobile": 0.15, "pstn": 0.15}` for analysis only; never a default.
- No new runtime dependencies. Do not import `transformers` in `src/` or `tests/` for this work.
- Deterministic per seed; output length preserved; A-law grid at 8 kHz output.
- Existing repo style: scipy-only DSP in `src/telephone.py`, no torch import.

## Review Focus

These inputs/failure modes are easy to miss and get a test in the owning task:

1. **Empty waveform** — `simulate_customer` returns an empty array, not a crash.
2. **Non-16 kHz input** — raises `ValueError` with the existing message pattern.
3. **Alias equivalence** — `profile="gated"`/`"clean"` produces the same audio as `"customer"` for the same seed, and eval labels canonicalize to `"customer"`.
4. **Default draws** — `apply_random_telephone(prob=1.0)` never returns `mobile`/`pstn` with default profiles.
5. **EQ order** — EQ is applied after the band-pass and before level calibration; length and A-law grid survive.
6. **Stereo input** — `_as_float32_mono` averages channels; test through `simulate_customer`.
7. **Missing scipy** — the customer path raises the existing clear `ImportError` (skip when scipy is present).

---

### Task 1: Commit answered-customer manifests and 1/3-octave bands

**Files:**
- Create: `RecordingTesting/analysis/ast_labels_in_repo.csv` (copy of `/tmp/opencode/real_corpus_music/music_files.csv`)
- Create: `RecordingTesting/analysis/ast_labels_outside.csv` (copy of `/tmp/opencode/content_sample_out/music_files.csv`)
- Create: `RecordingTesting/analysis/build_answered_manifest.py`
- Create: `RecordingTesting/analysis/build_answered_bands.py`
- Create: `RecordingTesting/analysis/answered_customer_manifest.csv`
- Create: `RecordingTesting/analysis/answered_customer_bands.csv`
- Create: `RecordingTesting/analysis/answered_customer_manifest.md`
- Test: `tests/test_analysis_manifests.py`

**Interfaces:**
- Produces: `answered_customer_manifest.csv` with columns `path,file,source,caller_class,is_gated,content_class,duration_s,speech_rms_db,noise_rms_db,hf_1_2k_rel_db,hf_2_3k_rel_db,spec_centroid,spec_roll95,calibration`.
- Produces: `answered_customer_bands.csv` with `file,source` plus `hz_125,hz_160,...,hz_4000` (16 columns), one row per calibration file.
- Produces: 67 `calibration=true` rows (`speech_only`, `is_gated=true`, `caller_class=mobile`).

- [ ] **Step 1: Copy the AST label tables into the repo**

Run:

```bash
cp /tmp/opencode/real_corpus_music/music_files.csv RecordingTesting/analysis/ast_labels_in_repo.csv
cp /tmp/opencode/content_sample_out/music_files.csv RecordingTesting/analysis/ast_labels_outside.csv
```

Expected: both files exist; the in-repo file has 107 data rows, the outside file has 213.

- [ ] **Step 2: Write the failing manifest test**

Create `tests/test_analysis_manifests.py`:

```python
import csv
from pathlib import Path
from statistics import median

ANALYSIS = Path(__file__).resolve().parent.parent / "RecordingTesting" / "analysis"
MANIFEST = ANALYSIS / "answered_customer_manifest.csv"
BANDS = ANALYSIS / "answered_customer_bands.csv"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_manifest_counts_and_calibration_subset():
    rows = _rows(MANIFEST)
    assert len(rows) == 320
    calib = [r for r in rows if r["calibration"].lower() == "true"]
    assert len(calib) == 67
    assert {r["source"] for r in calib} == {"in_repo", "outside"}
    assert {r["caller_class"] for r in calib} == {"mobile"}
    assert all(r["is_gated"].lower() == "true" for r in calib)
    assert all(r["content_class"] == "speech_only" for r in calib)
    for key, expected in {
        "speech_rms_db": -25.6,
        "noise_rms_db": -71.3,
        "hf_1_2k_rel_db": -15.1,
        "hf_2_3k_rel_db": -27.5,
        "spec_centroid": 459.2,
        "spec_roll95": 687.5,
    }.items():
        assert abs(median(float(r[key]) for r in calib) - expected) < 0.2


def test_answered_bands_have_67_rows_and_16_bands():
    rows = _rows(BANDS)
    assert len(rows) == 67
    band_cols = [c for c in rows[0] if c.startswith("hz_")]
    assert len(band_cols) == 16
    assert band_cols[0] == "hz_125"
    assert band_cols[-1] == "hz_4000"
    assert all(abs(float(r["hz_1000"])) < 0.05 for r in rows)
    for col, expected in {
        "hz_315": 4.52,
        "hz_400": 12.99,
        "hz_500": 12.95,
        "hz_3150": -19.77,
    }.items():
        assert abs(median(float(r[col]) for r in rows) - expected) < 0.5
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analysis_manifests.py -v`  
Expected: FAIL because the manifest and bands files do not exist.

- [ ] **Step 4: Implement `build_answered_manifest.py`**

The script must:
- Read `ast_labels_in_repo.csv`, derive `file = Path(path).name`, merge `caller_class` from `caller_class_check.classify_caller(file)`, and merge metrics from `RecordingTesting/analysis/features_files.csv` by `file` (`group == "real"`).
- Read `ast_labels_outside.csv` directly; use its `caller_class` and metrics; set `source="outside"`.
- Set `source="in_repo"` for the first group.
- Set `calibration = content_class == "speech_only" and is_gated is True and caller_class == "mobile"`.
- Write exactly the manifest columns in the Interfaces block, 320 rows total, 67 calibration rows.

Run: `.venv/bin/python RecordingTesting/analysis/build_answered_manifest.py`  
Expected: writes `answered_customer_manifest.csv` with 320 data rows.

- [ ] **Step 5: Implement `build_answered_bands.py`**

The script must:
- Read the manifest; keep `calibration == true`.
- Resolve audio: `in_repo` paths under `RecordingTesting/telephone_clips/RealRecordedPhoneCalls/`; `outside` paths under `--outside-dir` (default `/home/ameer/TerasVoice/Outside`).
- Use `gated_spectrum_check.loud_psd_for_audio` + `ct.load_audio` + `octave_levels` (the same method that produced the target table).
- Write one row per file: `file,source,hz_125,...,hz_4000`.
- Print the medians for 315, 400, 500, 3150 so they can be compared to the target.

Run: `.venv/bin/python RecordingTesting/analysis/build_answered_bands.py`  
Expected: 67 rows; the printed medians are within 0.5 dB of the target table.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analysis_manifests.py -v`  
Expected: PASS.

- [ ] **Step 7: Write `answered_customer_manifest.md`**

Document in 10–15 lines: the AST model, 10 s windows with first/middle/last sampling, the music/speech label families, the 213-file Outside sample definition, the 67-file calibration filter, and the warning that labels are model-derived and should be spot-checked.

- [ ] **Step 8: Commit**

```bash
git add RecordingTesting/analysis/ast_labels_in_repo.csv \
        RecordingTesting/analysis/ast_labels_outside.csv \
        RecordingTesting/analysis/build_answered_manifest.py \
        RecordingTesting/analysis/build_answered_bands.py \
        RecordingTesting/analysis/answered_customer_manifest.csv \
        RecordingTesting/analysis/answered_customer_bands.csv \
        RecordingTesting/analysis/answered_customer_manifest.md \
        tests/test_analysis_manifests.py
git commit -m "analysis: add answered-customer manifest and 1/3-octave bands"
```

---

### Task 2: Add the `customer` profile, defaults, and aliases

**Files:**
- Modify: `src/telephone.py` (constants, `simulate_customer`, `simulate_gated` wrapper, `simulate_telephone`, `degrade_for_eval_profiled`)
- Modify: `src/metrics.py` (`_canonical_profile_name`)
- Modify: `tests/test_telephone.py`
- Modify: `tests/test_normalizer_metrics.py`

**Interfaces:**
- Produces: `simulate_customer(waveform, sr=TARGET_SR, seed=None, use_ffmpeg=True, gain_db_range=(-1.0, 1.0), band_low_hz=300.0, band_high_hz=3400.0, output_sr=TARGET_SR, normalize_level=True, target_rms_db=None, gate_silence=True, gate_relative_db=-35.0, gate_hangover_ms=150.0, gate_fade_ms=5.0, low_shelf_db=4.0, low_shelf_hz=500.0) -> np.ndarray`. Task 3 adds `eq_gains_db`.
- Produces: `simulate_gated(...) -> np.ndarray` as a thin alias to `simulate_customer`.
- Produces: `DEFAULT_PROFILES == ("customer",)`, `DEFAULT_PROFILE_WEIGHTS == {"customer": 1.0}`, `LEGACY_PROFILE_WEIGHTS == {"gated": 0.70, "mobile": 0.15, "pstn": 0.15}`.
- Produces: `_canonical_profile_name("gated" | "clean") == "customer"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_telephone.py`:

```python
def test_default_profiles_is_customer_only():
    from src import telephone
    assert telephone.DEFAULT_PROFILES == ("customer",)
    assert telephone.DEFAULT_PROFILE_WEIGHTS == {"customer": 1.0}


def test_simulate_telephone_defaults_to_customer():
    import inspect
    from src import telephone
    params = inspect.signature(telephone.simulate_telephone).parameters
    assert params["profile"].default == "customer"


@requires_scipy
def test_gated_and_clean_alias_to_customer():
    from src.telephone import simulate_telephone
    x = _sine_mix(dur=0.5)
    y_customer = simulate_telephone(x, profile="customer", seed=7, use_ffmpeg=False)
    for alias in ("gated", "clean"):
        y = simulate_telephone(x, profile=alias, seed=7, use_ffmpeg=False)
        np.testing.assert_allclose(y, y_customer, atol=1e-7)


@requires_scipy
def test_simulate_customer_preserves_length_and_alaw():
    from src.telephone import simulate_customer
    x = _sine_mix(dur=0.5)
    y = simulate_customer(x, seed=1, output_sr=8000, use_ffmpeg=False)
    assert len(y) == 4000
    xi = np.round(y * 32768.0).astype(np.int32)
    assert np.mean(xi % 8 == 0) > 0.99


@requires_scipy
def test_simulate_customer_handles_empty_and_stereo():
    from src.telephone import simulate_customer
    assert len(simulate_customer(np.zeros(0, dtype=np.float32), use_ffmpeg=False)) == 0
    x = _sine_mix(dur=0.5)
    stereo = np.stack([x, x], axis=1)
    mono = simulate_customer(x, seed=3, use_ffmpeg=False)
    st = simulate_customer(stereo, seed=3, use_ffmpeg=False)
    np.testing.assert_allclose(mono, st, atol=1e-7)


@requires_scipy
def test_simulate_customer_rejects_non_16k():
    from src.telephone import simulate_customer
    with pytest.raises(ValueError):
        simulate_customer(np.zeros(1600, dtype=np.float32), sr=8000, use_ffmpeg=False)


def test_simulate_customer_requires_scipy(monkeypatch):
    from src import telephone
    monkeypatch.setattr(telephone, "_SCIPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        telephone.simulate_customer(_sine_mix(dur=0.1), use_ffmpeg=False)


@requires_scipy
def test_degrade_for_eval_profiled_canonicalizes_customer_aliases():
    from src import telephone
    x = _sine_mix(dur=0.5)
    for alias in ("gated", "clean"):
        _, prof = telephone.degrade_for_eval_profiled(
            x, index=0, seed_base=1, profiles=[alias], use_ffmpeg=False
        )
        assert prof == "customer"


def test_apply_random_telephone_defaults_never_mobile_or_pstn():
    from src import telephone
    import random
    rng = random.Random(5)
    seen = {
        telephone.choose_telephone_profile(
            telephone.DEFAULT_PROFILES, rng, telephone.DEFAULT_PROFILE_WEIGHTS
        )
        for _ in range(100)
    }
    assert seen == {"customer"}
```

Add to `tests/test_normalizer_metrics.py`:

```python
def test_profile_subsets_maps_gated_and_clean_to_customer():
    from src.metrics import profile_subsets
    subsets = profile_subsets(["a", "b", "c"], ["a", "b", "c"],
                              ["clean", "gated", "mobile"])
    assert set(subsets) == {"customer", "mobile"}
    assert subsets["customer"][0] == ["a", "b"]
```

Update `test_profile_weights_for_subsets_and_aliases` to use `{"customer": 0.70, "mobile": 0.15, "pstn": 0.15}` and assert `weights["customer"]`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telephone.py tests/test_normalizer_metrics.py -v`  
Expected: FAIL on the new customer/alias assertions.

- [ ] **Step 3: Implement the constants and rename**

In `src/telephone.py`:

```python
CUSTOMER_PROFILES = ("customer",)
DEFAULT_PROFILES = ("customer",)
DEFAULT_PROFILE_WEIGHTS = {"customer": 1.0}
LEGACY_PROFILE_WEIGHTS = {"gated": 0.70, "mobile": 0.15, "pstn": 0.15}
CUSTOMER_TARGET_RMS_DB_RANGE = (-28.5, -21.5)
```

- Rename `simulate_gated` to `simulate_customer`; keep `band_low_hz`, `band_high_hz`, `low_shelf_db`, and `low_shelf_hz` parameters so the existing analysis scripts and legacy `simulate_gated` wrapper keep working.
- `simulate_telephone` default `profile="customer"`; map `"customer"`, `"gated"`, `"clean"` to `simulate_customer`.
- `degrade_for_eval_profiled`: canonicalize `"clean"`/`"gated"` to `"customer"` before dispatch and return `"customer"`.
- `src/metrics.py`: `_canonical_profile_name` maps `"clean"` and `"gated"` to `"customer"`.
- Update the module docstring and `apply_random_telephone` docstring: remove the 70/15/15 default wording and describe the customer default.

Do **not** add the EQ yet; `simulate_customer` uses the current gated chain.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telephone.py tests/test_normalizer_metrics.py -v`  
Expected: PASS (the target-fit analysis tests are not part of this task).

- [ ] **Step 5: Commit**

```bash
git add src/telephone.py src/metrics.py tests/test_telephone.py tests/test_normalizer_metrics.py
git commit -m "feat: add customer telephony profile and canonical aliases"
```

---

### Task 3: Fit and apply the customer 1/3-octave EQ

**Files:**
- Modify: `src/telephone.py` (`apply_customer_eq`, `CUSTOMER_EQ_GAINS_DB`, `simulate_customer`)
- Create: `RecordingTesting/analysis/fit_customer_eq.py`
- Create: `RecordingTesting/analysis/customer_eq_gains.json`
- Modify: `tests/test_telephone.py`

**Interfaces:**
- Produces: `apply_customer_eq(wav8k, gains_db=CUSTOMER_EQ_GAINS_DB, sr=PHONE_SR) -> np.ndarray`.
- Produces: `CUSTOMER_EQ_GAINS_DB: Dict[float, float]` with keys `315.0, 400.0, 500.0, 630.0, 800.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0` (and optionally `1000.0: 0.0`).
- `simulate_customer(..., low_shelf_db=4.0, low_shelf_hz=500.0, eq_gains_db: Optional[Mapping[float, float]] = None)`: `None` uses `CUSTOMER_EQ_GAINS_DB`; `{}` disables the EQ for the fit baseline.

- [ ] **Step 1: Write the failing EQ tests**

Add to `tests/test_telephone.py`:

```python
def _tone(freq, sr=8000, dur=2.0):
    t = np.arange(int(sr * dur)) / sr
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _tone_db(x, freq, sr=8000, bw=60.0):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= freq - bw) & (freqs <= freq + bw)
    return float(20.0 * np.log10(np.max(spec[mask]) + 1e-20))


@requires_scipy
def test_apply_customer_eq_boosts_requested_band():
    from src.telephone import apply_customer_eq
    x = _tone(1000) + _tone(2000)
    y = apply_customer_eq(x, {2000.0: 10.0})
    before = _tone_db(x, 2000) - _tone_db(x, 1000)
    after = _tone_db(y, 2000) - _tone_db(y, 1000)
    assert 9.5 <= after - before <= 10.5


@requires_scipy
def test_apply_customer_eq_cuts_requested_band():
    from src.telephone import apply_customer_eq
    x = _tone(1000) + _tone(3150)
    y = apply_customer_eq(x, {3150.0: -8.6})
    before = _tone_db(x, 3150) - _tone_db(x, 1000)
    after = _tone_db(y, 3150) - _tone_db(y, 1000)
    assert -9.1 <= after - before <= -8.1


@requires_scipy
def test_apply_customer_eq_preserves_length():
    from src.telephone import apply_customer_eq
    x = _tone(1000, dur=1.0)
    y = apply_customer_eq(x, {400.0: 4.0, 3150.0: -8.6})
    assert len(y) == len(x)


@requires_scipy
def test_simulate_customer_applies_eq():
    from src.telephone import simulate_customer
    x = _tone(1000, dur=3.0) + _tone(3150, dur=3.0)
    base = simulate_customer(x, seed=1, output_sr=8000, use_ffmpeg=False,
                             eq_gains_db={})
    cut = simulate_customer(x, seed=1, output_sr=8000, use_ffmpeg=False,
                            eq_gains_db={3150.0: -8.6})
    base_ratio = _tone_db(base, 3150) - _tone_db(base, 1000)
    cut_ratio = _tone_db(cut, 3150) - _tone_db(cut, 1000)
    assert cut_ratio - base_ratio < -5.0


def test_customer_eq_gains_cover_primary_bands():
    from src import telephone
    assert {315.0, 400.0, 500.0, 630.0, 800.0, 1250.0,
            1600.0, 2000.0, 2500.0, 3150.0}.issubset(
        set(telephone.CUSTOMER_EQ_GAINS_DB)
    )
    assert all(-15.0 <= v <= 15.0 for v in telephone.CUSTOMER_EQ_GAINS_DB.values())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telephone.py -k "customer_eq" -v`  
Expected: FAIL because `apply_customer_eq` and `CUSTOMER_EQ_GAINS_DB` do not exist.

- [ ] **Step 3: Implement `apply_customer_eq`**

Use an RBJ peaking EQ per band with `Q = 1 / (2**(1/6) - 2**(-1/6))` (≈4.318):

```python
A = 10 ** (gain_db / 40.0)
w0 = 2 * np.pi * f0 / sr
alpha = np.sin(w0) / (2 * Q)
b0 = 1 + alpha * A
b1 = -2 * np.cos(w0)
b2 = 1 - alpha * A
a0 = 1 + alpha / A
a1 = -2 * np.cos(w0)
a2 = 1 - alpha / A
sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
y = scipy.signal.sosfilt(sos, y)
```

Apply bands in ascending frequency, after `bandpass_narrowband`, before `calibrate_active_rms`. Skip zero gains.

Set the initial constants from the measured delta:

```python
CUSTOMER_EQ_GAINS_DB = {
    315.0: -4.8,
    400.0: 4.3,
    500.0: 4.0,
    630.0: 1.9,
    800.0: 0.6,
    1250.0: 1.9,
    1600.0: 1.3,
    2000.0: -1.6,
    2500.0: -1.8,
    3150.0: -8.6,
}
```

- [ ] **Step 4: Run the EQ unit tests**

Run: `.venv/bin/python -m pytest tests/test_telephone.py -k "customer_eq" -v`  
Expected: PASS.

- [ ] **Step 5: Implement `fit_customer_eq.py`**

The script must:
- Read `answered_customer_bands.csv`; compute the median target level per band.
- Generate the baseline customer output on the 46 clean voices with `eq_gains_db={}` and the same seeds as the analysis (start with seeds 1–3 for speed, full 1–5 for the final run).
- Compute the residual `target - synth` at 315–3150 Hz; add it to the current gains with damping 0.7, clip to ±15 dB.
- Regenerate with the updated gains; repeat 3 iterations.
- Write `customer_eq_gains.json` with the final gains and a before/after table; print the 315–3150 residuals.

Run:

```bash
.venv/bin/python RecordingTesting/analysis/fit_customer_eq.py \
  --voice-dir /home/ameer/Hobbies/voiceprint/fa_clips \
  --out-dir /tmp/opencode/customer_eq_fit
```

Expected: every 315–3150 residual within ±2 dB; if not, adjust the low-frequency/315/3150 gains and rerun.

- [ ] **Step 6: Paste the fitted gains into `CUSTOMER_EQ_GAINS_DB` and rerun the unit tests**

Run: `.venv/bin/python -m pytest tests/test_telephone.py -k "customer_eq" -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/telephone.py RecordingTesting/analysis/fit_customer_eq.py \
        RecordingTesting/analysis/customer_eq_gains.json tests/test_telephone.py
git commit -m "feat: fit customer 1/3-octave EQ"
```

---

### Task 4: Update data, trainer, and script defaults

**Files:**
- Modify: `src/data.py` (comments; default comes from `DEFAULT_PROFILES`)
- Modify: `src/trainer_callbacks.py` (add `DEFAULT_TELEPHONY_PROFILES = ["customer"]`; use it)
- Modify: `scripts/train_lora.py` (defaults)
- Modify: `scripts/evaluate_model.py` (defaults)
- Test: `tests/test_telephone.py`, `tests/test_script_defaults.py`

**Interfaces:**
- Produces: `src.trainer_callbacks.DEFAULT_TELEPHONY_PROFILES == ["customer"]`.
- Produces: `scripts.train_lora._parse_telephony_profiles(None) == ["customer"]`.
- Produces: `scripts.evaluate_model._parse_profiles("") == ["customer"]`.

- [ ] **Step 1: Write the failing default tests**

Add to `tests/test_telephone.py`:

```python
def test_data_pipeline_default_profiles_include_customer():
    from src.data import PersianASRDataPipeline
    pipe = PersianASRDataPipeline(processor=None)
    assert tuple(pipe.telephony_profiles) == ("customer",)
```

Create `tests/test_script_defaults.py`:

```python
import pytest


def test_trainer_default_profiles_customer():
    from src.trainer_callbacks import DEFAULT_TELEPHONY_PROFILES
    assert DEFAULT_TELEPHONY_PROFILES == ["customer"]


def test_train_lora_parser_defaults_to_customer():
    pytest.importorskip("torch")
    from scripts.train_lora import _parse_telephony_profiles
    assert _parse_telephony_profiles(None) == ["customer"]


def test_evaluate_model_parser_defaults_to_customer():
    from scripts.evaluate_model import _parse_profiles
    assert _parse_profiles("") == ["customer"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telephone.py tests/test_script_defaults.py -v`  
Expected: FAIL on the new default assertions.

- [ ] **Step 3: Implement the defaults**

- `src/trainer_callbacks.py`: add `DEFAULT_TELEPHONY_PROFILES = ["customer"]`; use it in the callback `__init__`.
- `scripts/train_lora.py`: `--telephony_profiles` default `"customer"`; `_parse_telephony_profiles(None)` returns `["customer"]`; post-training evaluation default `["customer"]`.
- `scripts/evaluate_model.py`: `--telephony_profiles` default `"customer"`; `_parse_profiles("")` returns `["customer"]`.
- `src/data.py`: update the comment that says 70% gated / 15% mobile / 15% PSTN.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telephone.py tests/test_script_defaults.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py src/trainer_callbacks.py scripts/train_lora.py \
        scripts/evaluate_model.py tests/test_telephone.py tests/test_script_defaults.py
git commit -m "feat: default to the customer telephony profile everywhere"
```

---

### Task 5: Update analysis scripts and the report

**Files:**
- Modify: `RecordingTesting/analysis/gated_spectrum_check.py`
- Modify: `RecordingTesting/analysis/multi_voice_check.py`
- Modify: `RecordingTesting/analysis/caller_class_check.py`
- Modify: `RecordingTesting/analysis/smoke_wer.py`
- Modify: `RecordingTesting/analysis/make_listening_kit.py`
- Modify: `RecordingTesting/analysis/report.md`

**Interfaces:**
- `gated_spectrum_check.py` accepts `--content-manifest` (default `answered_customer_manifest.csv`) and filters real files to `calibration=true`.
- `multi_voice_check.py` defaults to `--profiles customer`.
- `caller_class_check.py` compares synth `customer` instead of `gated`.
- `smoke_wer.py` and `make_listening_kit.py` default to `customer`; legacy choices remain.

- [ ] **Step 1: Update `gated_spectrum_check.py`**

Add `--content-manifest` and filter `real_gated_psds` to calibration rows. Rename `synth_gated_psds` to `synth_customer_psds` and call `tel.simulate_customer`. Keep a `--legacy-gated` flag that reproduces the old all-gated, no-EQ behavior. Add `--eq/--no-eq` (default `--eq`) so the fit script can measure the baseline.

- [ ] **Step 2: Update the other scripts**

- `multi_voice_check.py`: add a `--profiles` argument (default `["customer"]`, choices `customer,gated,mobile,pstn`); add `--content-manifest`; filter the real comparison to calibration rows; update labels from “gated” to “customer”. The existing run loop must iterate `args.profiles` instead of the hard-coded list.
- `caller_class_check.py`: replace `summarize_synth(..., "gated")` with `"customer"`; add a `content_class` column if the manifest is present.
- `smoke_wer.py`: `--profiles` default `["customer"]`; keep `choices=["customer","gated","mobile","pstn"]`.
- `make_listening_kit.py`: current synth profile default `customer`; keep the legacy mobile/pstn pair for the legacy comparison.
- `report.md`: add a **2026-10-30 correction** section with the music-contamination counts, the 67-file target, the 1/3-octave table, and the before/after customer metrics.
- `compare_telephone.py`: mark this as the historical legacy report in its module docstring; it uses `FakePhoneCalls` and the old `mobile`/`pstn` labels. Do not repurpose it for the customer validation. If it is run, keep its existing labels and add a note that `gated` is now the legacy name for `customer`.

- [ ] **Step 3: Smoke-run the updated scripts**

Run:

```bash
.venv/bin/python RecordingTesting/analysis/gated_spectrum_check.py \
  --content-manifest RecordingTesting/analysis/answered_customer_manifest.csv \
  --n-sources 12 --seeds 1 2 \
  --out-dir /tmp/opencode/customer_spectrum_smoke
```

Expected: prints the 1/3-octave table and writes `gated_octave_levels.csv` without errors.

```bash
.venv/bin/python RecordingTesting/analysis/smoke_wer.py --list-only --profiles customer
```

Expected: lists pairs for the `customer` profile without a traceback.

- [ ] **Step 4: Commit**

```bash
git add RecordingTesting/analysis/gated_spectrum_check.py \
        RecordingTesting/analysis/multi_voice_check.py \
        RecordingTesting/analysis/caller_class_check.py \
        RecordingTesting/analysis/smoke_wer.py \
        RecordingTesting/analysis/make_listening_kit.py \
        RecordingTesting/analysis/report.md
git commit -m "analysis: report and validate the customer profile"
```

---

### Task 6: Full validation and final report

**Files:**
- Modify: `RecordingTesting/analysis/report.md` (final before/after table)
- Modify: `docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md` (add the final values)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`  
Expected: all tests pass (skips allowed for optional ffmpeg/torch paths).

- [ ] **Step 2: Run the full 1/3-octave validation**

Run:

```bash
.venv/bin/python RecordingTesting/analysis/gated_spectrum_check.py \
  --content-manifest RecordingTesting/analysis/answered_customer_manifest.csv \
  --n-sources 46 --seeds 1 2 3 4 5 \
  --out-dir /tmp/opencode/customer_spectrum_final
```

Expected: every band 315–3150 Hz within ±2 dB of the target; record the table in `report.md`.

- [ ] **Step 3: Run the multi-voice validation**

Run:

```bash
.venv/bin/python RecordingTesting/analysis/multi_voice_check.py \
  --profiles customer \
  --voice-dir /home/ameer/Hobbies/voiceprint/fa_clips \
  --seeds 1 2 3 4 5 \
  --out-dir /tmp/opencode/customer_multi_voice_final
```

Expected: active RMS −25.6 ±1 dB; noise floor −71.3 ±2 dB; centroid/roll95 inside the target IQR. Record the medians.

- [ ] **Step 4: Run the integration check**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py tests/test_telephone.py -v`  
Expected: PASS. The default-profile draw is covered by `test_apply_random_telephone_defaults_never_mobile_or_pstn` and `test_data_pipeline_default_profiles_include_customer`; the pipeline length/determinism tests run with explicit profiles and remain valid.

- [ ] **Step 5: Listening pass**

Build the listening kit and A/B three answered customer calls against the customer synth. Record the subjective result in `report.md`; flag if the 3150 Hz cut sounds dull or the 400–500 Hz boost sounds boomy.

- [ ] **Step 6: Update the design's final values**

Add the final fitted gains and the before/after table to the design doc. If the final gains differ materially from the initial delta, note the reason. Check `README.md`, `GUIDE.md`, and `PRODUCTION.md` for telephony-profile wording; update only if they mention the old defaults.

- [ ] **Step 7: Commit**

```bash
git add RecordingTesting/analysis/report.md \
        docs/superpowers/specs/2026-10-30-telephone-customer-profile-design.md
git commit -m "docs: record customer profile calibration results"
```
