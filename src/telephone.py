"""
Iranian telephony simulation for Persian ASR training/eval.

Converts clean 16 kHz Neyshekar audio to phone-call quality matching:
- gated clean landline (dominant real-call population: no line noise/hum,
  dark handset response, A-law digital silence)
- MCI / Irancell cellular (AMR-NB, coloured noise, fading/packet loss)
- TCI landline / PSTN (G.711 A-law, coloured line noise, rare low hum)

Chain (order follows a real call):
  mobile: 16k -> 8k -> bandpass -> level calibration -> noise/gain -> AMR-NB
          -> packet loss -> line-noise bed -> A-law storage -> 8k -> 16k
  PSTN:   16k -> 8k -> bandpass -> level calibration -> noise/hum/gain
          -> line-noise bed -> A-law storage -> 8k -> 16k
  gated:  16k -> 8k -> handset EQ -> level calibration/gain -> silence gate
          -> A-law storage -> 8k -> 16k

The default profile mixture is 70 % gated / 15 % mobile / 15 % PSTN,
calibrated against the 2026-10-22 real-call corpus.  Every profile is
A-law-quantized at 8 kHz before the final upsample, and can return the 8 kHz
A-law stream directly with ``output_sr=8000``.

Two codec tiers:
- full (default for mobile): true AMR-NB via system ffmpeg
  (libopencore_amrnb), falling back to the lite approximation if missing.
- lite: scipy-only bandpass + A-law approximation, no subprocess.
  A-law storage itself is always exact (pure numpy), even in lite mode.

Only depends on numpy/scipy/soundfile (already in requirements.txt).
No torch / datasets import so it stays usable inside data pipelines.
"""

import functools
import random
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy import signal as _ssignal
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ssignal = None  # type: ignore
    _SCIPY_AVAILABLE = False

try:
    import soundfile as _sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _sf = None  # type: ignore
    _SOUNDFILE_AVAILABLE = False


TARGET_SR = 16000
PHONE_SR = 8000
BAND_LOW = 300.0
BAND_HIGH = 3400.0
FRAME_LEN = 160  # 20 ms at 8 kHz

MOBILE_PROFILES = ("mobile",)
PSTN_PROFILES = ("pstn",)
GATED_PROFILES = ("gated",)
DEFAULT_PROFILES = ("gated", "mobile", "pstn")
DEFAULT_PROFILE_WEIGHTS = {"gated": 0.70, "mobile": 0.15, "pstn": 0.15}

# Calibrated against the 2026-10-22 call corpus (see RecordingTesting/analysis).
MOBILE_SNR_DB_RANGE = (18.0, 33.0)
PSTN_SNR_DB_RANGE = (19.5, 34.5)
PSTN_HUM_DB_RANGE = (-60.0, -45.0)
PSTN_HUM_PROB = 0.10
MOBILE_DROPOUT_P = 0.0015
PSTN_DROPOUT_P = 0.0
DROPOUT_FLOOR = 0.02
AMR_NB_BITRATES = ("4.75k", "5.9k", "7.4k", "12.2k")
_AMR_FALLBACK_WARNED = False
_AMR_FALLBACK_COUNT = 0

# Level/noise calibration fitted to the real noisy-line vs gated populations.
MOBILE_TARGET_RMS_DB_RANGE = (-34.8, -23.5)
PSTN_TARGET_RMS_DB_RANGE = (-34.8, -23.5)
GATED_TARGET_RMS_DB_RANGE = (-25.9, -17.8)
# Bed ranges are deliberately wider than the fitted median for training
# diversity; the medians remain the corpus-calibrated 21 / 19 dB.
MOBILE_BED_SNR_DB_RANGE = (17.0, 25.0)
PSTN_BED_SNR_DB_RANGE = (15.0, 23.0)
NOISE_BED_WHITE_MIX = 0.0
NOISE_BED_RUMBLE_MIX = 0.55
MAX_NORMALIZE_GAIN_DB = 18.0


# ---------------------------------------------------------------------------
# ffmpeg probing
# ---------------------------------------------------------------------------

def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _warn_amr_fallback(exc: BaseException) -> None:
    """Count an AMR fallback and warn once per process."""
    global _AMR_FALLBACK_WARNED, _AMR_FALLBACK_COUNT
    _AMR_FALLBACK_COUNT += 1
    if not _AMR_FALLBACK_WARNED:
        _AMR_FALLBACK_WARNED = True
        warnings.warn(
            f"AMR-NB unavailable ({exc}); falling back to the A-law lite path. "
            "Mobile training samples will lack AMR artifacts in this process.",
            RuntimeWarning,
            stacklevel=3,
        )


@functools.lru_cache(maxsize=None)
def ffmpeg_has_encoder(encoder: str) -> bool:
    """Check `ffmpeg -encoders` output for a given encoder name (cached per process)."""
    if not is_ffmpeg_available():
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return encoder in (out.stdout or "")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DSP helpers (numpy/scipy only)
# ---------------------------------------------------------------------------

def _require_scipy() -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for telephony simulation (pip install scipy>=1.12.0).")


def _as_float32_mono(wav: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float64).squeeze()
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if arr.ndim != 1:
        # Take first channel if accidentally stereo.
        arr = np.asarray(wav, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, 0] if arr.shape[1] <= arr.shape[0] else arr[0]
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    # Guard against int16 input.
    if np.max(np.abs(arr)) > 2.0:
        arr = arr / 32768.0
    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def _fix_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    if len(wav) == target_len:
        return wav
    if len(wav) > target_len:
        return wav[:target_len]
    pad = np.zeros(target_len - len(wav), dtype=wav.dtype)
    return np.concatenate([wav, pad])


def resample_16k_to_8k(wav16k: np.ndarray) -> np.ndarray:
    _require_scipy()
    x = np.asarray(wav16k, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    return _ssignal.resample_poly(x, up=1, down=2).astype(np.float32)


def resample_8k_to_16k(wav8k: np.ndarray) -> np.ndarray:
    _require_scipy()
    x = np.asarray(wav8k, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    return _ssignal.resample_poly(x, up=2, down=1).astype(np.float32)


def bandpass_narrowband(wav8k: np.ndarray, sr: int = PHONE_SR,
                        low: float = BAND_LOW, high: float = BAND_HIGH,
                        order: int = 5) -> np.ndarray:
    """Causal Butterworth bandpass mimicking the IRS/G.712 phone filter."""
    _require_scipy()
    x = np.asarray(wav8k, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    sos = _ssignal.butter(order, [low, high], btype="band", fs=sr, output="sos")
    return _ssignal.sosfilt(sos, x).astype(np.float32)


def apply_low_shelf(wav8k: np.ndarray, sr: int = PHONE_SR,
                    gain_db: float = 4.0, corner_hz: float = 500.0,
                    q: float = 0.70710678) -> np.ndarray:
    """Apply a gentle RBJ low-shelf to restore handset low-frequency weight.

    This is a static equaliser, not an AGC.  A small shelf keeps the gated
    profile from sounding thin after the 300-3400 Hz band-pass without
    reproducing the corpus's +20 dB 400-500 Hz resonance.
    """
    _require_scipy()
    x = np.asarray(wav8k, dtype=np.float32)
    if len(x) == 0 or abs(float(gain_db)) < 1e-9:
        return x.copy()
    a = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * np.pi * float(np.clip(corner_hz, 1.0, sr * 0.45)) / float(sr)
    cos_w0 = float(np.cos(w0))
    sin_w0 = float(np.sin(w0))
    alpha = sin_w0 / (2.0 * float(q))
    sqrt_a = float(np.sqrt(a))
    b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
    a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
                   dtype=np.float64)
    return _ssignal.sosfilt(sos, x).astype(np.float32)


# --- G.711 A-law lite approximation ----------------------------------------

_A = 87.6
_LN_A_PLUS_1 = float(np.log(1.0 + _A))
_INV_A = 1.0 / _A


def _alaw_compress(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x).astype(np.float64)
    out = np.empty_like(ax)
    small = ax < _INV_A
    out[small] = _A * ax[small] / (1.0 + np.log(_A))
    out[~small] = (1.0 + np.log(_A * np.clip(ax[~small], 1e-12, 1.0))) / _LN_A_PLUS_1
    return (np.sign(x) * out).astype(np.float64)


def _alaw_expand(y: np.ndarray) -> np.ndarray:
    ay = np.abs(y).astype(np.float64)
    out = np.empty_like(ay)
    thresh = 1.0 / _LN_A_PLUS_1
    small = ay < thresh
    out[small] = ay[small] * (1.0 + np.log(_A)) / _A
    out[~small] = np.exp(ay[~small] * _LN_A_PLUS_1 - 1.0) / _A
    return (np.sign(y) * out).astype(np.float64)


def alaw_quantize(wav: np.ndarray) -> np.ndarray:
    """8-bit A-law-style quantization (lite fallback when ffmpeg is absent)."""
    x = np.clip(np.asarray(wav, dtype=np.float64), -1.0, 1.0)
    c = _alaw_compress(x)
    q = np.round((c * 0.5 + 0.5) * 255.0) / 255.0 * 2.0 - 1.0
    return np.clip(_alaw_expand(q), -1.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _alaw_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Return (decoded_levels[256], encode_lut[65536]) for G.711 A-law.

    The real call recordings are bit-exact A-law decodes, so the final
    augmentation stage must land on the same 256-level codebook.  The encode
    LUT maps every int16 value to the nearest A-law reconstruction level.
    """
    levels = np.empty(256, dtype=np.int32)
    for code in range(256):
        a = code ^ 0x55
        t = (a & 0x0F) << 4
        seg = (a & 0x70) >> 4
        if seg == 0:
            t += 8
        elif seg == 1:
            t += 0x108
        else:
            t = (t + 0x108) << (seg - 1)
        levels[code] = t if (a & 0x80) else -t

    order = np.argsort(levels)
    sorted_levels = levels[order]
    sorted_codes = order.astype(np.uint8)

    values = np.arange(-32768, 32768, dtype=np.int32)
    idx = np.searchsorted(sorted_levels, values)
    idx = np.clip(idx, 1, len(sorted_levels) - 1)
    left = sorted_levels[idx - 1]
    right = sorted_levels[idx]
    # Ties go to the higher reconstruction level (0 -> +8, as in audioop).
    choose_left = (values - left) < (right - values)
    chosen = np.where(choose_left, idx - 1, idx)
    encode = sorted_codes[chosen]
    return levels, encode


def alaw_roundtrip_8k(wav8k: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Quantize 8 kHz float audio through the G.711 A-law codebook.

    Unlike the lite ``alaw_quantize`` approximation, this produces exactly the
    256 reconstruction levels found in real A-law recordings (all multiples of
    8, minimum non-zero magnitude 8).
    """
    x = _as_float32_mono(wav8k)
    levels, encode = _alaw_tables()
    xi = np.rint(x.astype(np.float64) * 32768.0)
    xi = np.clip(xi, -32768, 32767).astype(np.int32)
    codes = encode[xi + 32768]
    return (levels[codes].astype(np.float32) / 32768.0).astype(np.float32)


def add_gaussian_noise_at_snr(wav: np.ndarray, snr_db: float,
                              rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float64)
    sig_power = float(np.mean(x ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def colored_noise(n: int, sr: int = PHONE_SR,
                  rng: Optional[np.random.Generator] = None,
                  color: str = "pink") -> np.ndarray:
    """Unit-RMS coloured noise, shaped like real line/room noise.

    Real call noise is low-frequency dominated (roughly -8 dB/decade in the
    2026-10 corpus); white Gaussian noise was one of the largest mismatches.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    if color == "white":
        shape = np.ones_like(freqs)
    elif color == "pink":
        shape = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    elif color in ("brown", "red"):
        shape = 1.0 / np.maximum(freqs, 1.0)
    elif color == "line":
        # Measured real line noise is about -8.6 dB/decade in power, i.e.
        # amplitude ~ f**-0.43.  Drop the sub-50 Hz rumble here; the bed adds
        # it back as a separate low-frequency component.
        shape = (np.maximum(freqs, 1.0) / 100.0) ** -0.43
        shape[freqs < 50.0] = 0.0
    else:
        raise ValueError(f"Unknown noise color '{color}' (expected white/pink/brown/line).")
    shape[0] = 0.0  # remove DC
    noise = np.fft.irfft(spec * shape, n=n)
    rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-12
    return (noise / rms).astype(np.float32)


def add_colored_noise_at_snr(wav: np.ndarray, snr_db: float,
                             rng: np.random.Generator,
                             color: str = "pink",
                             active: bool = False,
                             band_limit: bool = False,
                             band: Tuple[float, float] = (BAND_LOW, BAND_HIGH),
                             modulation_db: float = 0.0,
                             modulation_hz_range: Tuple[float, float] = (0.2, 1.5)) -> np.ndarray:
    """Add coloured noise at a target SNR to 8 kHz audio.

    ``active=True`` measures speech power on active frames instead of the
    whole clip, so leading/trailing silence no longer changes the effective
    SNR.  ``band_limit=True`` filters the noise and, when ``active=False``,
    the whole-clip reference power to the telephone band, making ``snr_db``
    an in-band SNR in both modes.
    """
    x = np.asarray(wav, dtype=np.float64)
    if not np.isfinite(snr_db):
        return x.astype(np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    if band_limit:
        if active:
            sig_power = 10.0 ** (active_band_rms_db(x, *band) / 10.0)
        else:
            band_x = bandpass_narrowband(
                x.astype(np.float32), PHONE_SR, low=band[0], high=band[1]
            ).astype(np.float64)
            sig_power = float(np.mean(band_x ** 2)) + 1e-12
    elif active:
        sig_power = 10.0 ** (active_speech_rms_db(x) / 10.0)
    else:
        sig_power = float(np.mean(x ** 2)) + 1e-12
    noise = colored_noise(len(x), PHONE_SR, rng, color=color).astype(np.float64)
    if band_limit:
        noise = bandpass_narrowband(
            noise.astype(np.float32), PHONE_SR, low=band[0], high=band[1]
        ).astype(np.float64)
    if modulation_db > 0 and len(x) > PHONE_SR:
        mod_hz = float(rng.uniform(modulation_hz_range[0], modulation_hz_range[1]))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(len(x), dtype=np.float64) / float(PHONE_SR)
        envelope_db = float(modulation_db) * np.sin(2.0 * np.pi * mod_hz * t + phase)
        noise *= 10.0 ** (envelope_db / 20.0)
        noise /= np.sqrt(np.mean(noise ** 2)) + 1e-12
    target_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    noise *= np.sqrt(target_power / noise_power)
    return (x + noise).astype(np.float32)


def add_noise_bed(wav: np.ndarray, snr_db: float,
                  rng: np.random.Generator,
                  color: str = "line",
                  white_mix: float = NOISE_BED_WHITE_MIX,
                  rumble_mix: float = NOISE_BED_RUMBLE_MIX,
                  active: bool = True,
                  modulation_db: float = 3.0,
                  modulation_hz_range: Tuple[float, float] = (0.2, 1.5)) -> np.ndarray:
    """Add a post-codec line-noise bed with real-corpus spectral shape.

    The bed is a low-frequency rumble layer, not the primary in-band noise
    source.  ``snr_db`` is measured against active speech power when
    ``active=True``; the pre-codec noise stage is what sets the in-band SNR.
    ``rumble_mix`` blends a 30-120 Hz brown component back in so the bed
    keeps the real corpus's low-frequency weight without the old 1/f**2
    cliff.
    """
    x = np.asarray(wav, dtype=np.float64)
    if not np.isfinite(snr_db):
        return x.astype(np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    noise = colored_noise(len(x), PHONE_SR, rng, color=color).astype(np.float64)
    rumble_mix = float(np.clip(rumble_mix, 0.0, 1.0))
    if rumble_mix > 0:
        rumble = colored_noise(len(x), PHONE_SR, rng, color="brown").astype(np.float64)
        rumble = bandpass_narrowband(
            rumble.astype(np.float32), PHONE_SR, low=30.0, high=120.0
        ).astype(np.float64)
        rumble /= np.sqrt(np.mean(rumble ** 2)) + 1e-12
        noise = (1.0 - rumble_mix) * noise + rumble_mix * rumble
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    white_mix = float(np.clip(white_mix, 0.0, 1.0))
    if white_mix > 0:
        white = rng.standard_normal(len(x))
        white /= np.sqrt(np.mean(white ** 2)) + 1e-12
        noise = (1.0 - white_mix) * noise + white_mix * white
        noise /= np.sqrt(np.mean(noise ** 2)) + 1e-12
    if modulation_db > 0 and len(x) > PHONE_SR:
        mod_hz = float(rng.uniform(modulation_hz_range[0], modulation_hz_range[1]))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        t = np.arange(len(x), dtype=np.float64) / float(PHONE_SR)
        envelope_db = float(modulation_db) * np.sin(2.0 * np.pi * mod_hz * t + phase)
        noise *= 10.0 ** (envelope_db / 20.0)
        noise /= np.sqrt(np.mean(noise ** 2)) + 1e-12
    if active:
        sig_power = 10.0 ** (active_speech_rms_db(x) / 10.0)
    else:
        sig_power = float(np.mean(x ** 2)) + 1e-12
    target_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    noise *= np.sqrt(target_power / noise_power)
    return (x + noise).astype(np.float32)


def _active_frame_energy_db(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return 20 ms frame energies and the relative-VAD active mask."""
    n = len(x) // FRAME_LEN
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
    frames = x[:n * FRAME_LEN].reshape(n, FRAME_LEN)
    energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    noise_floor = float(np.percentile(energy, 10))
    speech_level = float(np.percentile(energy, 95))
    threshold = max(noise_floor + 7.0, speech_level - 30.0)
    return energy, energy > threshold


def active_speech_rms_db(wav8k: Union[np.ndarray, Sequence[float]]) -> float:
    """Median active-frame RMS (dBFS) using the same relative VAD as the report."""
    x = np.asarray(wav8k, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -200.0
    if len(x) < FRAME_LEN:
        return 10.0 * np.log10(float(np.mean(x ** 2)) + 1e-20)
    energy, active = _active_frame_energy_db(x)
    if not np.any(active):
        return 10.0 * np.log10(float(np.mean(x ** 2)) + 1e-20)
    return float(np.median(energy[active]))


def active_band_rms_db(wav8k: Union[np.ndarray, Sequence[float]],
                       low: float = BAND_LOW,
                       high: float = BAND_HIGH) -> float:
    """Median active-frame RMS in the telephone speech band (dBFS).

    The VAD mask comes from the full-band signal, so a loud out-of-band
    component does not make the in-band level look higher than it is.
    """
    x = np.asarray(wav8k, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -200.0
    _require_scipy()
    band = bandpass_narrowband(
        x.astype(np.float32), PHONE_SR, low=low, high=high
    ).astype(np.float64)
    if len(x) < FRAME_LEN:
        return 10.0 * np.log10(float(np.mean(band ** 2)) + 1e-20)
    _, active = _active_frame_energy_db(x)
    n = len(band) // FRAME_LEN
    frames = band[:n * FRAME_LEN].reshape(n, FRAME_LEN)
    energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    if not np.any(active):
        return 10.0 * np.log10(float(np.mean(band ** 2)) + 1e-20)
    return float(np.median(energy[active[:n]]))


def calibrate_active_rms(wav8k: Union[np.ndarray, Sequence[float]],
                         target_db_range: Tuple[float, float],
                         rng: np.random.Generator,
                         enabled: bool = True,
                         target_db: Optional[float] = None,
                         max_gain_db: float = MAX_NORMALIZE_GAIN_DB) -> np.ndarray:
    """Scale 8 kHz audio so active speech hits a sampled real-corpus target."""
    x = np.asarray(wav8k, dtype=np.float32).reshape(-1)
    if not enabled:
        return x
    target = (float(target_db) if target_db is not None
              else float(rng.uniform(target_db_range[0], target_db_range[1])))
    current = active_speech_rms_db(x)
    gain_db = float(np.clip(target - current, -max_gain_db, max_gain_db))
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def gate_silence_8k(wav8k: Union[np.ndarray, Sequence[float]],
                    relative_db: float = -38.0,
                    hangover_ms: float = 150.0,
                    fade_ms: float = 5.0) -> np.ndarray:
    """Zero 20 ms frames that sit far below the active speech level.

    Real gated recordings replace inter-utterance noise with the A-law zero
    code.  A hard frame gate also cuts weak word endings and clicks, so the
    gate keeps ``hangover_ms`` of signal after the last speech frame and
    applies a short raised-cosine fade at each zero boundary.
    """
    x = _as_float32_mono(wav8k)
    if len(x) < FRAME_LEN:
        return x
    n = len(x) // FRAME_LEN
    frames = x[:n * FRAME_LEN].reshape(n, FRAME_LEN)
    energy = 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    speech = active_speech_rms_db(x)
    gate = energy < (speech + float(relative_db))

    hangover_frames = max(0, int(round(float(hangover_ms) / 20.0)))
    if hangover_frames > 0 and np.any(gate):
        last_open = -10 ** 9
        for i in range(n):
            if not gate[i]:
                last_open = i
            elif i - last_open <= hangover_frames:
                gate[i] = False

    out = x.copy()
    if not np.any(gate):
        return out
    for i in np.flatnonzero(gate):
        out[i * FRAME_LEN:(i + 1) * FRAME_LEN] = 0.0

    fade_samples = max(0, int(round(float(fade_ms) * PHONE_SR / 1000.0)))
    if fade_samples > 0:
        zero = np.zeros(len(out), dtype=bool)
        for i in np.flatnonzero(gate):
            zero[i * FRAME_LEN:(i + 1) * FRAME_LEN] = True
        edges = np.diff(zero.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        gain = np.ones(len(out), dtype=np.float32)
        for a, b in zip(starts, ends):
            if a > 0:
                lo = max(0, a - fade_samples)
                ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, a - lo)))
                gain[lo:a] = np.minimum(gain[lo:a], ramp.astype(np.float32))
            if b < len(out):
                hi = min(len(out), b + fade_samples)
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, hi - b)))
                gain[b:hi] = np.minimum(gain[b:hi], ramp.astype(np.float32))
        out = out * gain
    return out.astype(np.float32)


def add_mains_hum(wav: np.ndarray, sr: int, hum_db: float,
                  rng: np.random.Generator,
                  freqs: Sequence[float] = (50.0, 100.0, 150.0)) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float64)
    sig_rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    hum_gain = sig_rms * (10.0 ** (hum_db / 20.0))
    t = np.arange(len(x)) / float(sr)
    hum = np.zeros_like(x)
    for i, f in enumerate(freqs):
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = hum_gain / (i + 1)  # harmonics decay
        hum += amp * np.sin(2 * np.pi * f * t + phase)
    return (x + hum).astype(np.float32)


def apply_dropouts(wav: np.ndarray, sr: int, p_frame: float,
                   rng: np.random.Generator,
                   frame_ms: float = 20.0,
                   max_dropout_ms: float = 60.0,
                   floor: float = DROPOUT_FLOOR,
                   fade_ms: float = 2.0) -> np.ndarray:
    """Attenuate random 20-60 ms spans to mimic packet loss / fading.

    ``floor`` keeps a low-level residual instead of writing exact digital
    zero: real AMR networks conceal lost frames, and the A-law recorder never
    stores zero spans inside speech.  Edges are raised-cosine fades to avoid
    clicks.
    """
    x = np.asarray(wav, dtype=np.float32).copy()
    if p_frame <= 0:
        return x
    floor = float(np.clip(floor, 0.0, 1.0))
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    gain_env = np.ones(len(x), dtype=np.float32)
    for start in range(0, len(x), frame_len):
        if float(rng.uniform()) < p_frame:
            dur_ms = float(rng.uniform(frame_ms, max_dropout_ms))
            drop_len = min(int(sr * dur_ms / 1000.0), len(x) - start)
            if drop_len <= 0:
                continue
            env = np.full(drop_len, floor, dtype=np.float64)
            fade = min(int(sr * fade_ms / 1000.0), max(1, drop_len // 2))
            if fade > 0:
                ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, fade)))
                env[:fade] = 1.0 - (1.0 - floor) * ramp
                env[-fade:] = floor + (1.0 - floor) * ramp
            gain_env[start:start + drop_len] = np.minimum(
                gain_env[start:start + drop_len], env.astype(np.float32)
            )
    return (x * gain_env).astype(np.float32)


def apply_gain_and_clip(wav: np.ndarray, gain_db: float,
                        clip: float = 0.99) -> np.ndarray:
    """Apply static gain plus soft saturation (not adaptive AGC)."""
    x = np.asarray(wav, dtype=np.float64) * (10.0 ** (gain_db / 20.0))
    x = np.tanh(x)
    return np.clip(x, -clip, clip).astype(np.float32)


# ---------------------------------------------------------------------------
# ffmpeg true-codec round-trip
# ---------------------------------------------------------------------------

def _ffmpeg_roundtrip_8k(wav8k: np.ndarray, mode: str,
                         bitrate: str = "7.4k") -> np.ndarray:
    """Encode+decode 8 kHz audio through a real telephony codec.

    mode="amr_nb" -> libopencore_amrnb at ``bitrate`` (MCI/Irancell).
    A-law storage is handled exactly in numpy, so it has no ffmpeg path.
    """
    if not _SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile is required for ffmpeg codec round-trip.")
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg binary not found.")
    orig_len = len(wav8k)
    with tempfile.TemporaryDirectory(prefix="phone_codec_") as td:
        td = Path(td)
        in_wav = td / "in_8k.wav"
        _sf.write(str(in_wav), np.asarray(wav8k, dtype=np.float32), PHONE_SR, subtype="PCM_16")

        if mode == "amr_nb":
            if not ffmpeg_has_encoder("libopencore_amrnb"):
                raise RuntimeError("ffmpeg lacks libopencore_amrnb encoder.")
            coded = td / "coded.amr"
            cmd_enc = ["ffmpeg", "-y", "-v", "error",
                       "-i", str(in_wav), "-ar", "8000", "-ac", "1",
                       "-c:a", "libopencore_amrnb", "-b:a", str(bitrate),
                       str(coded)]
        else:
            raise ValueError(f"Unknown codec mode: {mode}")

        r = subprocess.run(cmd_enc, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not coded.exists():
            raise RuntimeError(f"ffmpeg encode failed ({mode}): {(r.stderr or '')[-500:]}")

        decoded = td / "decoded.wav"
        cmd_dec = ["ffmpeg", "-y", "-v", "error",
                   "-i", str(coded), "-ar", "8000", "-ac", "1",
                   "-c:a", "pcm_s16le", str(decoded)]
        r = subprocess.run(cmd_dec, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not decoded.exists():
            raise RuntimeError(f"ffmpeg decode failed ({mode}): {(r.stderr or '')[-500:]}")

        out, sr = _sf.read(str(decoded), dtype="float32", always_2d=False)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return _fix_length(out, orig_len)


def _format_phone_output(x8: np.ndarray, target_len: int,
                         output_sr: int) -> np.ndarray:
    """Return A-law-grid audio at 8 kHz or upsampled 16 kHz input rate."""
    x8 = np.asarray(x8, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if output_sr == PHONE_SR:
        return _fix_length(x8, target_len // 2)
    if output_sr != TARGET_SR:
        raise ValueError(
            f"output_sr must be {PHONE_SR} or {TARGET_SR}, got {output_sr}."
        )
    x16 = resample_8k_to_16k(x8)
    return _fix_length(x16.astype(np.float32), target_len)


def choose_telephone_profile(
    profiles: Sequence[str] = DEFAULT_PROFILES,
    rng: Optional[random.Random] = None,
    profile_weights: Optional[Mapping[str, float]] = None,
) -> str:
    """Pick a profile, weighted by the real-corpus mixture by default.

    Explicitly passing ``["mobile", "pstn"]`` (the old default) is still
    uniform, because both weights are equal.  Unknown profile names fall back
    to uniform choice unless ``profile_weights`` is supplied.
    """
    profs = [str(p) for p in profiles]
    if not profs:
        raise ValueError("profiles must not be empty.")
    if rng is None:
        rng = random.Random()

    weights: Optional[list[float]] = None
    if profile_weights is not None:
        weights = [float(profile_weights.get(p, 0.0)) for p in profs]
    elif all(p in DEFAULT_PROFILE_WEIGHTS for p in profs):
        weights = [DEFAULT_PROFILE_WEIGHTS[p] for p in profs]

    if weights is not None:
        if sum(weights) <= 0:
            raise ValueError("profile_weights must have a positive total.")
        return rng.choices(profs, weights=weights, k=1)[0]
    return rng.choice(profs)


# ---------------------------------------------------------------------------
# Public simulation entry points
# ---------------------------------------------------------------------------

def simulate_mobile(waveform: Union[np.ndarray, Sequence[float]],
                    sr: int = TARGET_SR,
                    seed: Optional[int] = None,
                    use_ffmpeg: bool = True,
                    snr_db_range: Tuple[float, float] = MOBILE_SNR_DB_RANGE,
                    dropout_p: float = MOBILE_DROPOUT_P,
                    gain_db_range: Tuple[float, float] = (-1.0, 1.0),
                    noise_color: str = "pink",
                    output_sr: int = TARGET_SR,
                    normalize_level: bool = True,
                    target_rms_db: Optional[float] = None,
                    add_line_bed: bool = True,
                    bed_snr_db_range: Tuple[float, float] = MOBILE_BED_SNR_DB_RANGE,
                    bed_white_mix: float = NOISE_BED_WHITE_MIX,
                    amr_bitrate: str = "7.4k") -> np.ndarray:
    """Simulate MCI/Irancell cellular: AMR-NB + coloured noise + fading.

    Active speech is normalised to the real noisy-line level distribution,
    acoustic noise and gain are applied *before* the codec (as in a real
    handset), dropouts after it (network loss), a post-codec line-noise bed
    restores the low-frequency rumble, and a final G.711 A-law round-trip
    matches the recorder's 8-bit storage format.  The real corpus is stored
    as A-law, so its AMR mode is not directly observable; ``amr_bitrate``
    defaults to the conservative 7.4k and can be varied for diversity.
    """
    if sr != TARGET_SR:
        raise ValueError(f"simulate_mobile expects {TARGET_SR} Hz input, got {sr}.")
    rng = np.random.default_rng(seed)
    x = _as_float32_mono(waveform)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = len(x)

    x8 = resample_16k_to_8k(x)
    x8 = bandpass_narrowband(x8)
    x8 = calibrate_active_rms(
        x8, MOBILE_TARGET_RMS_DB_RANGE, rng,
        enabled=normalize_level, target_db=target_rms_db,
    )

    snr = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    x8 = add_colored_noise_at_snr(
        x8, snr, rng, color=noise_color, active=True, band_limit=True,
        modulation_db=5.0,
    )
    gain = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    x8 = apply_gain_and_clip(x8, gain)

    if use_ffmpeg:
        try:
            x8 = _ffmpeg_roundtrip_8k(x8, "amr_nb", bitrate=amr_bitrate)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            _warn_amr_fallback(exc)
            x8 = alaw_quantize(x8)
    else:
        x8 = alaw_quantize(x8)

    x8 = apply_dropouts(x8, PHONE_SR, dropout_p, rng)
    if add_line_bed:
        bed_snr = float(rng.uniform(bed_snr_db_range[0], bed_snr_db_range[1]))
        x8 = add_noise_bed(
            x8, bed_snr, rng, color="line",
            white_mix=bed_white_mix, rumble_mix=NOISE_BED_RUMBLE_MIX,
        )
    x8 = alaw_roundtrip_8k(x8)
    return _format_phone_output(x8, target_len, output_sr)


def simulate_pstn(waveform: Union[np.ndarray, Sequence[float]],
                  sr: int = TARGET_SR,
                  seed: Optional[int] = None,
                  use_ffmpeg: bool = True,
                  snr_db_range: Tuple[float, float] = PSTN_SNR_DB_RANGE,
                  hum_db_range: Tuple[float, float] = PSTN_HUM_DB_RANGE,
                  hum_prob: float = PSTN_HUM_PROB,
                  dropout_p: float = PSTN_DROPOUT_P,
                  gain_db_range: Tuple[float, float] = (-1.0, 1.0),
                  noise_color: str = "pink",
                  output_sr: int = TARGET_SR,
                  normalize_level: bool = True,
                  target_rms_db: Optional[float] = None,
                  add_line_bed: bool = True,
                  bed_snr_db_range: Tuple[float, float] = PSTN_BED_SNR_DB_RANGE,
                  bed_white_mix: float = NOISE_BED_WHITE_MIX) -> np.ndarray:
    """Simulate TCI landline: coloured line noise + optional hum + A-law.

    Noise/hum/gain happen before the A-law codec; the final round-trip is the
    recorder's 8-bit storage format.  Hum is now a rare, low-level condition
    (real 50 Hz prominence is ~0.2 dB, not the ~10 dB in the old default).
    ``use_ffmpeg`` is accepted for API symmetry; A-law storage is exact in
    numpy and does not need an external codec.
    """
    if sr != TARGET_SR:
        raise ValueError(f"simulate_pstn expects {TARGET_SR} Hz input, got {sr}.")
    rng = np.random.default_rng(seed)
    x = _as_float32_mono(waveform)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = len(x)

    x8 = resample_16k_to_8k(x)
    x8 = bandpass_narrowband(x8)
    x8 = calibrate_active_rms(
        x8, PSTN_TARGET_RMS_DB_RANGE, rng,
        enabled=normalize_level, target_db=target_rms_db,
    )

    snr = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    x8 = add_colored_noise_at_snr(
        x8, snr, rng, color=noise_color, active=True, band_limit=True,
        modulation_db=5.0,
    )
    if hum_prob > 0 and float(rng.uniform()) < hum_prob:
        hum_db = float(rng.uniform(hum_db_range[0], hum_db_range[1]))
        x8 = add_mains_hum(x8, PHONE_SR, hum_db, rng)
    gain = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    x8 = apply_gain_and_clip(x8, gain)

    x8 = apply_dropouts(x8, PHONE_SR, dropout_p, rng)
    if add_line_bed:
        bed_snr = float(rng.uniform(bed_snr_db_range[0], bed_snr_db_range[1]))
        x8 = add_noise_bed(
            x8, bed_snr, rng, color="line",
            white_mix=bed_white_mix, rumble_mix=NOISE_BED_RUMBLE_MIX,
        )
    x8 = alaw_roundtrip_8k(x8)
    return _format_phone_output(x8, target_len, output_sr)


def simulate_gated(waveform: Union[np.ndarray, Sequence[float]],
                   sr: int = TARGET_SR,
                   seed: Optional[int] = None,
                   use_ffmpeg: bool = True,
                   gain_db_range: Tuple[float, float] = (-1.0, 1.0),
                   band_low_hz: float = 300.0,
                   band_high_hz: float = 3400.0,
                   output_sr: int = TARGET_SR,
                   normalize_level: bool = True,
                   target_rms_db: Optional[float] = None,
                   gate_silence: bool = True,
                   gate_relative_db: float = -35.0,
                   gate_hangover_ms: float = 150.0,
                   gate_fade_ms: float = 5.0,
                   low_shelf_db: float = 4.0,
                   low_shelf_hz: float = 500.0) -> np.ndarray:
    """Simulate a clean/gated landline recording.

    This is the dominant real-call population: no added line noise or hum,
    a standard 300-3400 Hz telephone band with a gentle low-frequency shelf,
    A-law digital silence between utterances, and active speech normalised
    to the real gated level range.  The shelf is deliberately modest: the
    real corpus has a strong 400-500 Hz peak, but that peak is partly
    source/F0 dependent, so matching it blindly hurts intelligibility and
    overfits the corpus.  ``use_ffmpeg`` is accepted for API symmetry;
    A-law storage is exact and does not need an external codec.
    """
    if sr != TARGET_SR:
        raise ValueError(f"simulate_gated expects {TARGET_SR} Hz input, got {sr}.")
    rng = np.random.default_rng(seed)
    x = _as_float32_mono(waveform)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = len(x)

    x8 = resample_16k_to_8k(x)
    x8 = bandpass_narrowband(
        x8, PHONE_SR, low=band_low_hz, high=band_high_hz, order=5
    )
    x8 = apply_low_shelf(
        x8, PHONE_SR, gain_db=low_shelf_db, corner_hz=low_shelf_hz
    )
    x8 = calibrate_active_rms(
        x8, GATED_TARGET_RMS_DB_RANGE, rng,
        enabled=normalize_level, target_db=target_rms_db,
    )
    gain = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    x8 = apply_gain_and_clip(x8, gain)
    if gate_silence:
        x8 = gate_silence_8k(
            x8, relative_db=gate_relative_db,
            hangover_ms=gate_hangover_ms, fade_ms=gate_fade_ms,
        )
    x8 = alaw_roundtrip_8k(x8)
    return _format_phone_output(x8, target_len, output_sr)


def simulate_telephone(waveform: Union[np.ndarray, Sequence[float]],
                       sr: int = TARGET_SR,
                       profile: str = "mobile",
                       seed: Optional[int] = None,
                       use_ffmpeg: bool = True,
                       output_sr: int = TARGET_SR,
                       normalize_level: bool = True,
                       target_rms_db: Optional[float] = None,
                       amr_bitrate: str = "7.4k") -> np.ndarray:
    if profile == "mobile":
        return simulate_mobile(waveform, sr=sr, seed=seed, use_ffmpeg=use_ffmpeg,
                               amr_bitrate=amr_bitrate,
                               output_sr=output_sr, normalize_level=normalize_level,
                               target_rms_db=target_rms_db)
    if profile == "pstn":
        return simulate_pstn(waveform, sr=sr, seed=seed, use_ffmpeg=use_ffmpeg,
                             output_sr=output_sr, normalize_level=normalize_level,
                             target_rms_db=target_rms_db)
    if profile in ("gated", "clean"):
        return simulate_gated(waveform, sr=sr, seed=seed, use_ffmpeg=use_ffmpeg,
                              output_sr=output_sr, normalize_level=normalize_level,
                              target_rms_db=target_rms_db)
    raise ValueError(
        f"Unknown telephone profile '{profile}' "
        "(expected 'gated', 'mobile' or 'pstn')."
    )


def apply_random_telephone(
    waveform: Union[np.ndarray, Sequence[float]],
    sr: int = TARGET_SR,
    prob: float = 0.5,
    profiles: Sequence[str] = DEFAULT_PROFILES,
    seed: Optional[int] = None,
    use_ffmpeg: bool = True,
    rng: Optional[random.Random] = None,
    profile_weights: Optional[Mapping[str, float]] = None,
    output_sr: int = TARGET_SR,
) -> Tuple[np.ndarray, Optional[str]]:
    """Apply phone simulation with probability `prob`.

    Returns (waveform_out, applied_profile_or_None).
    Uses stdlib `random` (picklable, safe inside datasets.map workers).
    Profile choice follows the real-corpus mixture when the default profile
    names are used (70 % gated / 15 % mobile / 15 % PSTN).
    """
    x = _as_float32_mono(waveform)
    if prob <= 0 or not profiles:
        return x, None
    local_rng = rng if rng is not None else random.Random(seed)
    if local_rng.random() >= prob:
        return x, None
    profile = choose_telephone_profile(profiles, local_rng, profile_weights)
    child_seed = local_rng.randint(0, 2 ** 31 - 1)
    degraded = simulate_telephone(x, sr=sr, profile=str(profile),
                                  seed=child_seed, use_ffmpeg=use_ffmpeg,
                                  output_sr=output_sr)
    return degraded, str(profile)


def degrade_for_eval_profiled(waveform: Union[np.ndarray, Sequence[float]],
                              index: int,
                              seed_base: int = 1234,
                              profiles: Sequence[str] = DEFAULT_PROFILES,
                              use_ffmpeg: bool = True) -> Tuple[np.ndarray, str]:
    """Deterministic per-sample degradation returning the applied profile."""
    profs = list(profiles) or list(DEFAULT_PROFILES)
    profile = str(profs[index % len(profs)])
    if profile.lower() == "clean":
        profile = "gated"
    degraded = simulate_telephone(waveform, sr=TARGET_SR, profile=profile,
                                  seed=int(seed_base) + int(index),
                                  use_ffmpeg=use_ffmpeg)
    return degraded, profile


def degrade_for_eval(waveform: Union[np.ndarray, Sequence[float]],
                     index: int,
                     seed_base: int = 1234,
                     profiles: Sequence[str] = DEFAULT_PROFILES,
                     use_ffmpeg: bool = True) -> np.ndarray:
    """Deterministic per-sample degradation for eval (round-robin profiles)."""
    return degrade_for_eval_profiled(
        waveform, index=index, seed_base=seed_base,
        profiles=profiles, use_ffmpeg=use_ffmpeg,
    )[0]
