"""
Offline unit tests for Iranian telephony simulation (src/telephone.py).
No network, no torch, no ffmpeg required (ffmpeg paths skip gracefully).
Run: uv run --with numpy --with scipy --with soundfile -- pytest tests/test_telephone.py -q
"""

import sys
from pathlib import Path

import numpy as np

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


def test_resample_roundtrip_length():
    x = _sine_mix()
    x8 = resample_16k_to_8k(x)
    assert len(x8) == len(x) // 2
    x16 = resample_8k_to_16k(x8)
    assert abs(len(x16) - len(x)) <= 2


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


def test_pstn_lite_runs_and_differs_from_mobile():
    x = _sine_mix(dur=0.8)
    m = simulate_mobile(x, seed=11, use_ffmpeg=False)
    p = simulate_pstn(x, seed=11, use_ffmpeg=False)
    assert len(m) == len(x) == len(p)
    assert not np.allclose(m, p)


def test_alaw_quantize_bounds():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.3).astype(np.float32)
    y = alaw_quantize(x)
    assert y.dtype == np.float32
    assert float(np.max(np.abs(y))) <= 1.0


def test_apply_random_prob_edges():
    x = _sine_mix(dur=0.5)
    clean, prof = apply_random_telephone(x, prob=0.0, seed=1, use_ffmpeg=False)
    np.testing.assert_allclose(clean, x, atol=1e-7)
    assert prof is None
    deg, prof = apply_random_telephone(x, prob=1.0, profiles=["mobile"], seed=2, use_ffmpeg=False)
    assert prof == "mobile"
    assert not np.allclose(deg, x)


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
