"""
Offline unit tests for Iranian telephony simulation (src/telephone.py).
No network, no torch, no ffmpeg required (ffmpeg paths skip gracefully).
Scipy-dependent tests skip gracefully when scipy is absent; full run needs:
  uv run --with numpy --with scipy --with soundfile --with pytest -- python -m pytest tests/ -q
Run: uv run --with numpy --with scipy --with soundfile --with pytest -- python -m pytest tests/test_telephone.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import scipy  # noqa: F401
    _SCIPY_MISSING = False
except ImportError:
    _SCIPY_MISSING = True

requires_scipy = pytest.mark.skipif(
    _SCIPY_MISSING,
    reason="scipy not installed; telephony DSP requires scipy>=1.12.0 (pip install scipy>=1.12.0)",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.telephone import (
    alaw_quantize,
    apply_random_telephone,
    bandpass_narrowband,
    degrade_for_eval,
    is_ffmpeg_available,
    resample_16k_to_8k,
    resample_8k_to_16k,
    simulate_mobile,
    simulate_pstn,
)


def _sine_mix(sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / float(sr)
    # 440 Hz in-band + 6000 Hz out-of-band (phone should kill 6 kHz).
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)
    return x.astype(np.float32)


@requires_scipy
def test_resample_roundtrip_length():
    x = _sine_mix()
    x8 = resample_16k_to_8k(x)
    assert len(x8) == len(x) // 2
    x16 = resample_8k_to_16k(x8)
    assert abs(len(x16) - len(x)) <= 2


@requires_scipy
def test_low_shelf_boosts_low_frequencies():
    from src import telephone

    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (
        0.1 * np.sin(2 * np.pi * 300.0 * t)
        + 0.1 * np.sin(2 * np.pi * 2000.0 * t)
    ).astype(np.float32)
    y = telephone.apply_low_shelf(x, sr=sr, gain_db=4.0, corner_hz=500.0)

    def tone_amp(wav, f0, bw=20.0):
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
        m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        return float(np.max(spec[m]) + 1e-20)

    before = 20.0 * np.log10(tone_amp(x, 300.0) / tone_amp(x, 2000.0))
    after = 20.0 * np.log10(tone_amp(y, 300.0) / tone_amp(y, 2000.0))
    assert after - before > 2.5

    x50 = (0.1 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    y50 = telephone.apply_low_shelf(x50, sr=sr, gain_db=4.0, corner_hz=500.0)
    gain50 = 20.0 * np.log10(tone_amp(y50, 50.0) / tone_amp(x50, 50.0))
    assert 3.0 < gain50 < 5.0

    gain2k = 20.0 * np.log10(tone_amp(y, 2000.0) / tone_amp(x, 2000.0))
    assert abs(gain2k) < 0.5


@requires_scipy
def test_bandpass_suppresses_high_freq():
    sr = 16000
    dur = 1.5
    t = np.arange(int(sr * dur)) / float(sr)
    # 440 Hz in-band vs 3800 Hz above phone cutoff (both below 8k Nyquist).
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 3800 * t)).astype(np.float32)
    x8 = resample_16k_to_8k(x)
    y8 = bandpass_narrowband(x8)
    # Skip filter startup transient (first 0.3s).
    skip = int(8000 * 0.3)
    x8s, y8s = x8[skip:], y8[skip:]

    def band_energy(wav, sr, f0, bw=80):
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
        m = (freqs > f0 - bw) & (freqs < f0 + bw)
        return float(np.mean(spec[m]) + 1e-12)

    in_before = band_energy(x8s, 8000, 440)
    in_after = band_energy(y8s, 8000, 440)
    hi_before = band_energy(x8s, 8000, 3800)
    hi_after = band_energy(y8s, 8000, 3800)
    # In-band preserved (within 6 dB), out-of-band attenuated by >6 dB.
    assert in_after > in_before * 0.25
    assert hi_after < hi_before * 0.25


@requires_scipy
def test_lite_deterministic_and_length():
    x = _sine_mix(dur=0.8)
    a = simulate_mobile(x, seed=7, use_ffmpeg=False)
    b = simulate_mobile(x, seed=7, use_ffmpeg=False)
    c = simulate_mobile(x, seed=8, use_ffmpeg=False)
    assert len(a) == len(x)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not np.allclose(a, c)
    # Must actually change the signal.
    assert not np.allclose(a, x)


@requires_scipy
def test_pstn_lite_runs_and_differs_from_mobile():
    x = _sine_mix(dur=0.8)
    m = simulate_mobile(x, seed=11, use_ffmpeg=False)
    p = simulate_pstn(x, seed=11, use_ffmpeg=False)
    assert len(m) == len(x) == len(p)
    assert not np.allclose(m, p)


def test_as_float32_mono_handles_empty():
    from src import telephone
    y = telephone._as_float32_mono(np.array([], dtype=np.float32))
    assert y.size == 0
    assert y.dtype == np.float32


@requires_scipy
def test_public_simulators_handle_empty_input():
    from src import telephone
    empty = np.zeros(0, dtype=np.float32)
    for fn in (telephone.simulate_gated, telephone.simulate_mobile, telephone.simulate_pstn):
        y = fn(empty, seed=0, use_ffmpeg=False)
        assert y.size == 0, fn.__name__
    assert telephone.resample_16k_to_8k(empty).size == 0
    assert telephone.resample_8k_to_16k(empty).size == 0
    assert telephone.bandpass_narrowband(empty).size == 0
    assert telephone.active_speech_rms_db(empty) == -200.0
    assert telephone.active_band_rms_db(empty) == -200.0
    assert telephone.add_colored_noise_at_snr(
        empty, 10.0, np.random.default_rng(0)
    ).size == 0
    assert telephone.add_noise_bed(
        empty, 10.0, np.random.default_rng(0)
    ).size == 0


@requires_scipy
def test_simulate_gated_old_positional_signature_still_works():
    from src import telephone
    x = _sine_mix(dur=0.2)
    y = telephone.simulate_gated(
        x, 16000, 0, False, (-1.0, 1.0), 300.0, 3400.0, 8000
    )
    assert len(y) == len(x) // 2


@requires_scipy
def test_simulate_mobile_old_positional_signature_still_works():
    from src import telephone
    x = _sine_mix(dur=0.2)
    y = telephone.simulate_mobile(
        x, 16000, 0, False, (18.0, 33.0), 0.0015, (-1.0, 1.0), "pink", 8000
    )
    assert len(y) == len(x) // 2


@requires_scipy
def test_simulate_telephone_old_positional_signature_still_works():
    from src import telephone
    x = _sine_mix(dur=0.2)
    y = telephone.simulate_telephone(x, 16000, "mobile", 0, False, 8000)
    assert len(y) == len(x) // 2


@requires_scipy
def test_add_colored_noise_band_limit_without_active_is_inband():
    from src import telephone
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (
        0.1 * np.sin(2 * np.pi * 1000.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3800.0 * t)
    ).astype(np.float32)
    y = telephone.add_colored_noise_at_snr(
        x, 10.0, np.random.default_rng(0), color="pink",
        active=False, band_limit=True,
    )
    noise = y.astype(np.float64) - x.astype(np.float64)
    from scipy import signal as ss
    sos = ss.butter(5, [300.0, 3400.0], btype="band", fs=sr, output="sos")
    xb = ss.sosfilt(sos, x.astype(np.float64))
    nb = ss.sosfilt(sos, noise)
    snr = 10.0 * np.log10(np.mean(xb ** 2) / (np.mean(nb ** 2) + 1e-20))
    assert abs(snr - 10.0) < 1.5


@requires_scipy
def test_simulate_profiles_keep_stereo_input_length():
    from src import telephone
    t = np.arange(1600, dtype=np.float64) / 16000.0
    mono = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([mono, 0.5 * mono], axis=1)
    for fn in (telephone.simulate_gated, telephone.simulate_mobile, telephone.simulate_pstn):
        y = fn(stereo, seed=0, use_ffmpeg=False)
        y_mono = fn(mono, seed=0, use_ffmpeg=False)
        assert len(y) == len(mono), fn.__name__
        np.testing.assert_allclose(y, y_mono, atol=1e-6)


@requires_scipy
def test_amr_fallback_warns_once_and_counts(monkeypatch):
    import warnings
    from src import telephone

    def boom(*args, **kwargs):
        raise RuntimeError("libopencore_amrnb missing")

    monkeypatch.setattr(telephone, "_ffmpeg_roundtrip_8k", boom)
    monkeypatch.setattr(telephone, "_AMR_FALLBACK_WARNED", False)
    monkeypatch.setattr(telephone, "_AMR_FALLBACK_COUNT", 0)
    x = _sine_mix(dur=0.2)
    with pytest.warns(RuntimeWarning, match="AMR-NB"):
        telephone.simulate_mobile(x, seed=0, use_ffmpeg=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        telephone.simulate_mobile(x, seed=1, use_ffmpeg=True)
    assert not caught
    assert telephone._AMR_FALLBACK_COUNT == 2


@requires_scipy
def test_amr_fallback_handles_unexpected_codec_error(monkeypatch):
    from src import telephone

    def boom(*args, **kwargs):
        raise ValueError("unexpected codec path")

    monkeypatch.setattr(telephone, "_ffmpeg_roundtrip_8k", boom)
    monkeypatch.setattr(telephone, "_AMR_FALLBACK_WARNED", False)
    monkeypatch.setattr(telephone, "_AMR_FALLBACK_COUNT", 0)
    with pytest.warns(RuntimeWarning, match="AMR-NB"):
        telephone.simulate_mobile(_sine_mix(dur=0.2), seed=0, use_ffmpeg=True)
    assert telephone._AMR_FALLBACK_COUNT == 1


@requires_scipy
def test_simulate_mobile_rejects_invalid_amr_bitrate():
    from src import telephone
    with pytest.raises(ValueError, match="amr_bitrate"):
        telephone.simulate_mobile(
            _sine_mix(dur=0.2), seed=0, use_ffmpeg=False, amr_bitrate="9.9k"
        )


@requires_scipy
def test_simulate_mobile_passes_amr_bitrate(monkeypatch):
    from src import telephone
    seen = []

    def fake(wav, mode, bitrate="7.4k"):
        seen.append((mode, bitrate))
        return wav

    monkeypatch.setattr(telephone, "_ffmpeg_roundtrip_8k", fake)
    x = _sine_mix(dur=0.2)
    telephone.simulate_mobile(
        x, seed=0, use_ffmpeg=True, amr_bitrate="12.2k"
    )
    assert seen
    assert seen[0] == ("amr_nb", "12.2k")


def test_alaw_quantize_bounds():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.3).astype(np.float32)
    y = alaw_quantize(x)
    assert y.dtype == np.float32
    assert float(np.max(np.abs(y))) <= 1.0


@requires_scipy
def test_apply_random_prob_edges():
    x = _sine_mix(dur=0.5)
    clean, prof = apply_random_telephone(x, prob=0.0, seed=1, use_ffmpeg=False)
    np.testing.assert_allclose(clean, x, atol=1e-7)
    assert prof is None
    deg, prof = apply_random_telephone(x, prob=1.0, profiles=["mobile"], seed=2, use_ffmpeg=False)
    assert prof == "mobile"
    assert not np.allclose(deg, x)


@requires_scipy
def test_degrade_for_eval_profiled_returns_profiles_and_matches_audio():
    from src import telephone
    x = _sine_mix(dur=0.5)
    profiles = ["customer", "mobile", "pstn"]
    for idx, expected in enumerate(profiles):
        y, prof = telephone.degrade_for_eval_profiled(
            x, index=idx, seed_base=1234, profiles=profiles, use_ffmpeg=False
        )
        z = telephone.degrade_for_eval(
            x, index=idx, seed_base=1234, profiles=profiles, use_ffmpeg=False
        )
        assert prof == expected
        np.testing.assert_allclose(y, z, atol=1e-6)
    for alias in ("gated", "clean"):
        _, prof = telephone.degrade_for_eval_profiled(
            x, index=0, seed_base=1234, profiles=[alias], use_ffmpeg=False
        )
        assert prof == "customer"


@requires_scipy
def test_degrade_for_eval_deterministic_profiles():
    x = _sine_mix(dur=0.5)
    a = degrade_for_eval(x, index=0, seed_base=1234, profiles=["mobile", "pstn"], use_ffmpeg=False)
    b = degrade_for_eval(x, index=0, seed_base=1234, profiles=["mobile", "pstn"], use_ffmpeg=False)
    c = degrade_for_eval(x, index=1, seed_base=1234, profiles=["mobile", "pstn"], use_ffmpeg=False)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not np.allclose(a, c)  # idx 0=mobile vs idx 1=pstn


def test_data_pipeline_stores_telephony_params():
    from src.data import PersianASRDataPipeline
    pipe = PersianASRDataPipeline(
        processor=None, telephony_prob=0.5,
        telephony_profiles=["mobile", "pstn"],
        telephony_seed=42, telephony_use_ffmpeg=False,
    )
    assert pipe.telephony_prob == 0.5
    assert tuple(pipe.telephony_profiles) == ("mobile", "pstn")


def test_ffmpeg_encoder_probe_runs_once(monkeypatch):
    from src import telephone
    calls = []

    class _Out:
        stdout = " V..... libopencore_amrnb\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Out()

    telephone.ffmpeg_has_encoder.cache_clear()
    monkeypatch.setattr(telephone, "is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(telephone.subprocess, "run", fake_run)
    try:
        assert telephone.ffmpeg_has_encoder("libopencore_amrnb")
        assert telephone.ffmpeg_has_encoder("libopencore_amrnb")
        assert len(calls) == 1
    finally:
        telephone.ffmpeg_has_encoder.cache_clear()


@requires_scipy
def test_ffmpeg_roundtrip_if_available():
    if not is_ffmpeg_available():
        print("SKIP ffmpeg roundtrip (no binary).")
        return
    from src.telephone import ffmpeg_has_encoder
    if not ffmpeg_has_encoder("libopencore_amrnb"):
        print("SKIP AMR-NB (encoder missing).")
        return
    x = _sine_mix(dur=0.5)
    y = simulate_mobile(x, seed=3, use_ffmpeg=True)
    assert len(y) == len(x)
    assert not np.allclose(y, x)


# ---------------------------------------------------------------------------
# Match-report P0 regressions: A-law storage, gated profile, noise/dropouts.
# ---------------------------------------------------------------------------

def test_legacy_profile_weights_sum_to_one():
    from src import telephone
    assert abs(sum(telephone.LEGACY_PROFILE_WEIGHTS.values()) - 1.0) < 1e-9
    assert telephone.LEGACY_PROFILE_WEIGHTS["gated"] >= 0.6


def test_choose_telephone_profile_is_weighted_for_explicit_profiles():
    import random
    from src import telephone
    rng = random.Random(0)
    profiles = ["customer", "mobile"]
    weights = {"customer": 0.75, "mobile": 0.25}
    counts = {p: 0 for p in profiles}
    for _ in range(2000):
        counts[telephone.choose_telephone_profile(profiles, rng, weights)] += 1
    assert counts["customer"] > counts["mobile"]
    assert 0.70 < counts["customer"] / 2000.0 < 0.80


def test_apply_dropouts_leaves_residual_not_zero():
    from src import telephone
    x = np.ones(8000, dtype=np.float32)
    rng = np.random.default_rng(0)
    y = telephone.apply_dropouts(x, 8000, p_frame=1.0, rng=rng)
    assert float(np.min(y)) >= 0.019
    assert bool(np.any(y < 0.5))


def test_mobile_dropout_default_is_rare():
    import inspect
    from src import telephone
    params = inspect.signature(telephone.simulate_mobile).parameters
    assert params["dropout_p"].default <= 0.003


def test_pstn_hum_default_is_light():
    import inspect
    from src import telephone
    params = inspect.signature(telephone.simulate_pstn).parameters
    assert params["hum_db_range"].default[1] <= -45.0
    assert params["hum_prob"].default <= 0.2


def test_alaw_roundtrip_grid_and_idempotent():
    from src import telephone
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8000) * 0.3).astype(np.float32)
    y = telephone.alaw_roundtrip_8k(x)
    yi = np.rint(y * 32768).astype(np.int32)
    assert np.all(yi % 8 == 0)
    assert len(np.unique(yi)) <= 256
    assert int(np.min(np.abs(yi[yi != 0]))) >= 8
    y2 = telephone.alaw_roundtrip_8k(y)
    np.testing.assert_array_equal(yi, np.rint(y2 * 32768).astype(np.int32))


def test_all_profiles_final_8k_is_alaw():
    from src import telephone
    x = _sine_mix(dur=0.5)
    for profile in ("mobile", "pstn", "gated"):
        y = telephone.simulate_telephone(
            x, profile=profile, seed=3, use_ffmpeg=False, output_sr=8000
        )
        yi = np.rint(y * 32768).astype(np.int32)
        assert len(y) == len(x) // 2
        assert np.all(yi % 8 == 0), profile
        assert len(np.unique(yi)) <= 256, profile


def test_gated_silence_has_no_added_noise():
    from src import telephone
    x = np.zeros(16000, dtype=np.float32)
    y = telephone.simulate_gated(x, seed=1, use_ffmpeg=False, output_sr=8000)
    yi = np.rint(y * 32768).astype(np.int32)
    assert int(np.max(np.abs(yi))) <= 8
    assert len(np.unique(yi)) <= 2


@requires_scipy
def test_gated_profile_keeps_2_5k_energy():
    from src import telephone
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.1 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)
    x8 = telephone.resample_16k_to_8k(x)
    y = telephone.simulate_gated(
        x, seed=1, use_ffmpeg=False, output_sr=8000,
        normalize_level=False, gate_silence=False, gain_db_range=(0.0, 0.0),
    )

    def tone_amp(wav, f0, bw=30.0):
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / 8000)
        m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        return float(np.max(spec[m]) + 1e-20)

    assert tone_amp(y, 2500.0) / tone_amp(x8, 2500.0) > 0.5


def test_colored_noise_is_low_frequency_heavy():
    from src import telephone
    n = 8000
    rng = np.random.default_rng(0)
    y = telephone.colored_noise(n, 8000, rng, color="pink")
    assert len(y) == n
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / 8000)
    lo = np.mean(spec[(freqs >= 100) & (freqs < 500)])
    hi = np.mean(spec[(freqs >= 2000) & (freqs < 3500)])
    assert lo > 2.0 * hi
    y2 = telephone.colored_noise(n, 8000, np.random.default_rng(0), color="pink")
    np.testing.assert_allclose(y, y2, atol=1e-6)


def test_add_colored_noise_hits_target_snr():
    from src import telephone
    x = _sine_mix(dur=0.5)
    y = telephone.add_colored_noise_at_snr(x, 10.0, np.random.default_rng(0), color="pink")
    noise = y.astype(np.float64) - x.astype(np.float64)
    snr = 10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10.0) < 0.2


@requires_scipy
def test_active_band_rms_rejects_out_of_band():
    from src import telephone
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (
        0.1 * np.sin(2 * np.pi * 1000.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3800.0 * t)
    ).astype(np.float32)
    full = telephone.active_speech_rms_db(x)
    band = telephone.active_band_rms_db(x)
    assert band < full - 1.5


@requires_scipy
def test_add_colored_noise_uses_active_inband_snr():
    from src import telephone
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    speech = (0.1 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    def measure(x, y):
        noise = y.astype(np.float64) - x.astype(np.float64)
        sp = 10.0 ** (telephone.active_band_rms_db(x) / 10.0)
        npow = 10.0 ** (telephone.active_band_rms_db(noise) / 10.0)
        return 10.0 * np.log10(sp / (npow + 1e-20))

    x1 = np.concatenate([speech, np.zeros(sr, dtype=np.float32)])
    x2 = np.concatenate([speech, np.zeros(3 * sr, dtype=np.float32)])
    y1 = telephone.add_colored_noise_at_snr(
        x1, 20.0, np.random.default_rng(0), color="pink",
        active=True, band_limit=True,
    )
    y2 = telephone.add_colored_noise_at_snr(
        x2, 20.0, np.random.default_rng(0), color="pink",
        active=True, band_limit=True,
    )
    snr1 = measure(x1, y1)
    snr2 = measure(x2, y2)
    assert abs(snr1 - 20.0) < 1.5
    assert abs(snr1 - snr2) < 1.0


@requires_scipy
def test_add_colored_noise_has_slow_level_variation():
    from src import telephone
    sr = 8000
    t = np.arange(4 * sr, dtype=np.float64) / sr
    x = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    y = telephone.add_colored_noise_at_snr(
        x, 10.0, np.random.default_rng(1), color="pink",
        active=True, band_limit=True, modulation_db=3.0,
    )
    noise = y.astype(np.float64) - x.astype(np.float64)
    n = len(noise) // 800
    frames = noise[:n * 800].reshape(n, 800)
    levels = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    variation = float(np.percentile(levels, 90) - np.percentile(levels, 10))
    assert variation > 2.5


@requires_scipy
def test_add_noise_bed_line_shape_matches_real():
    from src import telephone
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    x = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    y = telephone.add_noise_bed(
        x, 10.0, np.random.default_rng(0), color="line",
        rumble_mix=0.55, white_mix=0.0, active=True,
    )
    noise = y.astype(np.float64) - x.astype(np.float64)
    spec = np.abs(np.fft.rfft(noise)) ** 2
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sr)
    lo = np.sum(spec[freqs < 500.0])
    hi = np.sum(spec[freqs >= 500.0])
    ratio = lo / (hi + 1e-20)
    m = (freqs >= 100.0) & (freqs <= 3500.0)
    slope = np.polyfit(
        np.log10(freqs[m] + 1.0), 10.0 * np.log10(spec[m] + 1e-20), 1
    )[0]
    assert 3.0 < ratio < 9.0
    assert -12.0 < slope < -6.0


@requires_scipy
def test_add_noise_bed_has_slow_level_variation():
    from src import telephone
    sr = 8000
    t = np.arange(4 * sr, dtype=np.float64) / sr
    x = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    y = telephone.add_noise_bed(
        x, 10.0, np.random.default_rng(1), color="line",
        rumble_mix=0.55, white_mix=0.0, active=True, modulation_db=3.0,
    )
    noise = y.astype(np.float64) - x.astype(np.float64)
    n = len(noise) // 800
    frames = noise[:n * 800].reshape(n, 800)
    levels = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    variation = float(np.percentile(levels, 90) - np.percentile(levels, 10))
    assert variation > 2.5


def test_data_pipeline_default_profiles_include_customer():
    from src.data import PersianASRDataPipeline
    pipe = PersianASRDataPipeline(processor=None)
    assert tuple(pipe.telephony_profiles) == ("customer",)


def test_active_speech_rms_and_calibration():
    from src import telephone
    t = np.arange(16000) / 16000.0
    x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    cur = telephone.active_speech_rms_db(x)
    assert abs(cur - (-23.0)) < 1.0
    y = telephone.calibrate_active_rms(
        x, (-20.0, -20.0), np.random.default_rng(0), enabled=True
    )
    assert abs(telephone.active_speech_rms_db(y) - (-20.0)) < 1.0


def test_add_noise_bed_is_low_frequency_and_hits_snr():
    from src import telephone
    t = np.arange(16000) / 16000.0
    x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = telephone.add_noise_bed(
        x, 10.0, np.random.default_rng(0), color="brown", white_mix=0.12
    )
    noise = y.astype(np.float64) - x.astype(np.float64)
    snr = 10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10.0) < 0.3
    spec = np.abs(np.fft.rfft(noise)) ** 2
    freqs = np.fft.rfftfreq(len(noise), 1.0 / 8000)
    lo = np.mean(spec[(freqs >= 100) & (freqs < 500)])
    hi = np.mean(spec[(freqs >= 2000) & (freqs < 3500)])
    assert lo > 2.0 * hi


def test_line_bed_darkens_mobile_profile():
    from src import telephone
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(16000) * 0.1).astype(np.float32)

    def centroid(y8):
        spec = np.abs(np.fft.rfft(y8 * np.hanning(len(y8)))) ** 2
        freqs = np.fft.rfftfreq(len(y8), 1.0 / 8000)
        return float(np.sum(spec * freqs) / (np.sum(spec) + 1e-20))

    common = dict(seed=1, use_ffmpeg=False, output_sr=8000,
                  normalize_level=False, dropout_p=0.0)
    dry = telephone.simulate_mobile(x, add_line_bed=False, **common)
    wet = telephone.simulate_mobile(x, add_line_bed=True, **common)
    assert centroid(wet) < centroid(dry)


def test_gated_level_calibration():
    from src import telephone
    t = np.arange(16000) / 16000.0
    x = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = telephone.simulate_gated(
        x, seed=0, normalize_level=True, target_rms_db=-21.2, output_sr=8000
    )
    assert abs(telephone.active_speech_rms_db(y) - (-21.2)) < 1.0


@requires_scipy
def test_customer_profile_preserves_speech_band_highs():
    """The customer profile must keep the 1-3 kHz plateau, not a 350-900 cliff."""
    import inspect
    from src import telephone

    params = inspect.signature(telephone.simulate_customer).parameters
    assert params["band_low_hz"].default <= 300.0
    assert params["band_high_hz"].default >= 3400.0

    t = np.arange(16000, dtype=np.float64) / 16000.0
    x = (
        0.1 * np.sin(2 * np.pi * 500.0 * t)
        + 0.1 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.1 * np.sin(2 * np.pi * 2500.0 * t)
    ).astype(np.float32)
    y = telephone.simulate_customer(
        x,
        seed=0,
        use_ffmpeg=False,
        output_sr=8000,
        normalize_level=False,
        gate_silence=False,
        gain_db_range=(0.0, 0.0),
    )

    def tone_amp(wav, sr, f0, bw=30.0):
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
        freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
        m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        return float(np.max(spec[m]) + 1e-20)

    ref = tone_amp(y, 8000, 500.0)
    assert tone_amp(y, 8000, 1500.0) / ref > 0.4
    assert tone_amp(y, 8000, 2500.0) / ref > 0.4


def test_gate_silence_removes_low_level_pauses_without_hangover():
    from src import telephone
    t = np.arange(4000) / 8000.0
    speech = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.0005 * np.random.default_rng(0).standard_normal(4000)).astype(np.float32)
    x = np.concatenate([speech, noise])
    y = telephone.gate_silence_8k(x, hangover_ms=0.0, fade_ms=0.0)
    assert float(np.max(np.abs(y[4000:]))) < 1e-4
    assert float(np.sqrt(np.mean(y[:4000] ** 2))) > 0.05


def test_gate_silence_gates_trailing_partial_frame():
    from src import telephone
    sr = 8000
    t = np.arange(4000, dtype=np.float64) / sr
    speech = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    tail = np.full(100, 0.003, dtype=np.float32)  # < one 20 ms frame
    y = telephone.gate_silence_8k(
        np.concatenate([speech, tail]),
        relative_db=-25.0, hangover_ms=0.0, fade_ms=0.0,
    )
    assert float(np.max(np.abs(y[4000:]))) < 1e-4


def test_gate_default_matches_simulate_gated():
    import inspect
    from src import telephone
    helper = inspect.signature(telephone.gate_silence_8k).parameters["relative_db"].default
    profile = inspect.signature(telephone.simulate_gated).parameters["gate_relative_db"].default
    assert helper == profile


def test_gate_silence_hangover_preserves_tails_with_fade():
    from src import telephone
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    speech = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    # -50 dBFS tail, below a -25 dB gate threshold, but still audible as a
    # word ending.  It must survive the 150 ms hangover and then fade out.
    tail = np.full(sr // 2, 0.003, dtype=np.float32)
    y = telephone.gate_silence_8k(
        np.concatenate([speech, tail]),
        relative_db=-25.0,
        hangover_ms=150.0,
        fade_ms=5.0,
    )
    assert float(np.max(np.abs(y[sr:sr + 1100]))) > 1e-4
    assert float(np.max(np.abs(y[sr + 1300:]))) < 1e-4
    # The old hard gate jumped from 0.003 to 0 in one sample.  A 5 ms fade
    # keeps the largest sample-to-sample step well below that.
    assert float(np.max(np.abs(np.diff(y[sr + 1100:sr + 1300])))) < 1e-3


def test_gated_profile_gates_pauses():
    from src import telephone
    import inspect

    params = inspect.signature(telephone.simulate_gated).parameters
    assert params["gate_relative_db"].default >= -38.0
    assert params["gate_hangover_ms"].default >= 100.0
    assert params["gate_fade_ms"].default > 0.0

    t = np.arange(8000) / 16000.0
    speech = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.0005 * np.random.default_rng(0).standard_normal(8000)).astype(np.float32)
    x = np.concatenate([speech, noise])
    y = telephone.simulate_gated(x, seed=0, output_sr=8000)
    # Speech ends at 4000 samples (8 kHz).  The 150 ms hangover rounds to
    # 8 frames (160 ms), and the 5 ms fade follows it; require silence after
    # that boundary.
    assert float(np.max(np.abs(y[5440:]))) <= 8.0 / 32768.0 + 1e-9


# ---------------------------------------------------------------------------
# Customer profile: one default, aliases, and edge inputs.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Customer 1/3-octave EQ.
# ---------------------------------------------------------------------------

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
    x = _tone(1000, sr=16000, dur=3.0) + _tone(3150, sr=16000, dur=3.0)
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
    assert all(-20.0 <= v <= 15.0 for v in telephone.CUSTOMER_EQ_GAINS_DB.values())


@requires_scipy
def test_customer_profile_active_rms_and_noise_floor():
    from src.telephone import simulate_customer, active_speech_rms_db
    speech = _tone(440, sr=16000, dur=1.0)
    x = np.concatenate([speech, np.zeros(16000, dtype=np.float32)])
    y = simulate_customer(x, seed=11, output_sr=8000, use_ffmpeg=False,
                          target_rms_db=-25.6)
    assert abs(active_speech_rms_db(y) - (-25.6)) <= 1.0
    frames = y[: len(y) // 160 * 160].reshape(-1, 160).astype(np.float64)
    energy = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    noise_floor = float(np.median(np.sort(energy)[: max(1, len(energy) // 10)]))
    assert -75.0 <= noise_floor <= -69.0


@requires_scipy
def test_simulate_customer_averages_stereo_channels():
    from src.telephone import active_speech_rms_db, simulate_customer
    x = _sine_mix(dur=0.5)
    # Right-channel-only: taking the first channel would produce silence.
    stereo = np.stack([np.zeros_like(x), x], axis=1)
    mono = simulate_customer(x, seed=3, use_ffmpeg=False)
    st = simulate_customer(stereo, seed=3, use_ffmpeg=False)
    # Averaging halves the level, so the absolute level may differ slightly;
    # the regression is that dropping the right channel would produce silence.
    assert float(np.max(np.abs(st))) > 0.01
    assert active_speech_rms_db(st) > -35.0


def test_customer_eq_initial_and_final_gains_differ():
    from src import telephone
    assert telephone.CUSTOMER_EQ_INITIAL_GAINS_DB[3150.0] == -8.6
    # The per-file refit makes the 3150 Hz cut shallower than the measured
    # delta, not deeper; the old -19.8 dB cut overcorrected the high band.
    assert telephone.CUSTOMER_EQ_GAINS_DB[3150.0] > -8.6


def test_customer_eq_high_band_not_overcut():
    from src import telephone
    gains = telephone.CUSTOMER_EQ_GAINS_DB
    assert gains[1600.0] > 4.0
    assert gains[2000.0] > 1.0
    assert gains[2500.0] > 4.0
    assert gains[3150.0] > -10.0


@requires_scipy
def test_apply_random_telephone_default_draws_are_customer():
    from src.telephone import apply_random_telephone
    x = _sine_mix(dur=0.05)
    seen = {
        apply_random_telephone(x, prob=1.0, seed=seed, use_ffmpeg=False)[1]
        for seed in range(20)
    }
    assert seen == {"customer"}
