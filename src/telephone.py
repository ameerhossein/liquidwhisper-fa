"""
Iranian telephony simulation for Persian ASR training/eval.

Converts clean 16 kHz Neyshekar audio to phone-call quality matching:
- MCI / Irancell cellular (AMR-NB, fading, packet loss, street noise)
- TCI landline / PSTN (G.711 A-law, 50 Hz hum, line hiss)

Chain (order matters):
  16k -> 8k downsample -> 300-3400 Hz bandpass -> codec round-trip
  -> noise/hum -> dropouts -> gain/clip -> 8k -> 16k upsample

Two tiers:
- full (default): true codec via system ffmpeg
  (libopencore_amrnb for mobile, pcm_alaw for PSTN).
  Falls back to lite approximation if ffmpeg/encoder is missing.
- lite: scipy-only bandpass + A-law quantization approx, no subprocess.

Only depends on numpy/scipy/soundfile (already in requirements.txt).
No torch / datasets import so it stays usable inside data pipelines.
"""

import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

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

MOBILE_PROFILES = ("mobile",)
PSTN_PROFILES = ("pstn",)
DEFAULT_PROFILES = ("mobile", "pstn")


# ---------------------------------------------------------------------------
# ffmpeg probing
# ---------------------------------------------------------------------------

def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_has_encoder(encoder: str) -> bool:
    """Check `ffmpeg -encoders` output for a given encoder name."""
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
    return _ssignal.resample_poly(np.asarray(wav16k, dtype=np.float32), up=1, down=2).astype(np.float32)


def resample_8k_to_16k(wav8k: np.ndarray) -> np.ndarray:
    _require_scipy()
    return _ssignal.resample_poly(np.asarray(wav8k, dtype=np.float32), up=2, down=1).astype(np.float32)


def bandpass_narrowband(wav8k: np.ndarray, sr: int = PHONE_SR,
                        low: float = BAND_LOW, high: float = BAND_HIGH,
                        order: int = 5) -> np.ndarray:
    """Causal Butterworth bandpass mimicking the IRS/G.712 phone filter."""
    _require_scipy()
    sos = _ssignal.butter(order, [low, high], btype="band", fs=sr, output="sos")
    return _ssignal.sosfilt(sos, np.asarray(wav8k, dtype=np.float32)).astype(np.float32)


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


def add_gaussian_noise_at_snr(wav: np.ndarray, snr_db: float,
                              rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float64)
    sig_power = float(np.mean(x ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


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
                   max_dropout_ms: float = 60.0) -> np.ndarray:
    """Zero random 20-60 ms spans to mimic packet loss / fading."""
    x = np.asarray(wav, dtype=np.float32).copy()
    if p_frame <= 0:
        return x
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    for start in range(0, len(x), frame_len):
        if float(rng.uniform()) < p_frame:
            dur_ms = float(rng.uniform(frame_ms, max_dropout_ms))
            drop_len = min(int(sr * dur_ms / 1000.0), len(x) - start)
            # Short raised-cosine edges to avoid clicks.
            x[start:start + drop_len] = 0.0
    return x


def apply_gain_and_clip(wav: np.ndarray, gain_db: float,
                        clip: float = 0.99) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float64) * (10.0 ** (gain_db / 20.0))
    x = np.tanh(x)  # soft saturation like phone AGC
    return np.clip(x, -clip, clip).astype(np.float32)


# ---------------------------------------------------------------------------
# ffmpeg true-codec round-trip
# ---------------------------------------------------------------------------

def _ffmpeg_roundtrip_8k(wav8k: np.ndarray, mode: str) -> np.ndarray:
    """Encode+decode 8 kHz audio through a real telephony codec.

    mode="amr_nb" -> libopencore_amrnb @ 7.4k (MCI/Irancell cellular)
    mode="alaw"   -> pcm_alaw (TCI PSTN / E1 A-law)
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
                       "-c:a", "libopencore_amrnb", "-b:a", "7.4k", str(coded)]
        elif mode == "alaw":
            coded = td / "coded_alaw.wav"
            cmd_enc = ["ffmpeg", "-y", "-v", "error",
                       "-i", str(in_wav), "-ar", "8000", "-ac", "1",
                       "-c:a", "pcm_alaw", str(coded)]
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


# ---------------------------------------------------------------------------
# Public simulation entry points
# ---------------------------------------------------------------------------

def simulate_mobile(waveform: Union[np.ndarray, Sequence[float]],
                    sr: int = TARGET_SR,
                    seed: Optional[int] = None,
                    use_ffmpeg: bool = True,
                    snr_db_range: Tuple[float, float] = (5.0, 15.0),
                    dropout_p: float = 0.02,
                    gain_db_range: Tuple[float, float] = (-6.0, 6.0)) -> np.ndarray:
    """Simulate MCI/Irancell cellular: AMR-NB + noise + fading dropouts."""
    if sr != TARGET_SR:
        raise ValueError(f"simulate_mobile expects {TARGET_SR} Hz input, got {sr}.")
    rng = np.random.default_rng(seed)
    target_len = len(np.asarray(waveform).reshape(-1))
    x = _as_float32_mono(waveform)

    x8 = resample_16k_to_8k(x)
    x8 = bandpass_narrowband(x8)

    if use_ffmpeg:
        try:
            x8 = _ffmpeg_roundtrip_8k(x8, "amr_nb")
        except Exception:
            x8 = alaw_quantize(x8)
    else:
        x8 = alaw_quantize(x8)

    snr = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    x8 = add_gaussian_noise_at_snr(x8, snr, rng)
    x8 = apply_dropouts(x8, PHONE_SR, dropout_p, rng)
    gain = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    x8 = apply_gain_and_clip(x8, gain)

    x16 = resample_8k_to_16k(x8)
    return _fix_length(x16.astype(np.float32), target_len)


def simulate_pstn(waveform: Union[np.ndarray, Sequence[float]],
                  sr: int = TARGET_SR,
                  seed: Optional[int] = None,
                  use_ffmpeg: bool = True,
                  snr_db_range: Tuple[float, float] = (15.0, 25.0),
                  hum_db_range: Tuple[float, float] = (-40.0, -30.0),
                  dropout_p: float = 0.005,
                  gain_db_range: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """Simulate TCI landline: G.711 A-law + 50 Hz hum + light hiss."""
    if sr != TARGET_SR:
        raise ValueError(f"simulate_pstn expects {TARGET_SR} Hz input, got {sr}.")
    rng = np.random.default_rng(seed)
    target_len = len(np.asarray(waveform).reshape(-1))
    x = _as_float32_mono(waveform)

    x8 = resample_16k_to_8k(x)
    x8 = bandpass_narrowband(x8)

    if use_ffmpeg:
        try:
            x8 = _ffmpeg_roundtrip_8k(x8, "alaw")
        except Exception:
            x8 = alaw_quantize(x8)
    else:
        x8 = alaw_quantize(x8)

    snr = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    x8 = add_gaussian_noise_at_snr(x8, snr, rng)
    hum_db = float(rng.uniform(hum_db_range[0], hum_db_range[1]))
    x8 = add_mains_hum(x8, PHONE_SR, hum_db, rng)
    x8 = apply_dropouts(x8, PHONE_SR, dropout_p, rng)
    gain = float(rng.uniform(gain_db_range[0], gain_db_range[1]))
    x8 = apply_gain_and_clip(x8, gain)

    x16 = resample_8k_to_16k(x8)
    return _fix_length(x16.astype(np.float32), target_len)


def simulate_telephone(waveform: Union[np.ndarray, Sequence[float]],
                       sr: int = TARGET_SR,
                       profile: str = "mobile",
                       seed: Optional[int] = None,
                       use_ffmpeg: bool = True) -> np.ndarray:
    if profile == "mobile":
        return simulate_mobile(waveform, sr=sr, seed=seed, use_ffmpeg=use_ffmpeg)
    if profile == "pstn":
        return simulate_pstn(waveform, sr=sr, seed=seed, use_ffmpeg=use_ffmpeg)
    raise ValueError(f"Unknown telephone profile '{profile}' (expected 'mobile' or 'pstn').")


def apply_random_telephone(
    waveform: Union[np.ndarray, Sequence[float]],
    sr: int = TARGET_SR,
    prob: float = 0.5,
    profiles: Sequence[str] = DEFAULT_PROFILES,
    seed: Optional[int] = None,
    use_ffmpeg: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Apply phone simulation with probability `prob`.

    Returns (waveform_out, applied_profile_or_None).
    Uses stdlib `random` (picklable, safe inside datasets.map workers).
    """
    x = _as_float32_mono(waveform)
    if prob <= 0 or not profiles:
        return x, None
    local_rng = rng if rng is not None else random.Random(seed)
    if local_rng.random() >= prob:
        return x, None
    profile = local_rng.choice(list(profiles))
    child_seed = local_rng.randint(0, 2 ** 31 - 1)
    degraded = simulate_telephone(x, sr=sr, profile=str(profile),
                                  seed=child_seed, use_ffmpeg=use_ffmpeg)
    return degraded, str(profile)


def degrade_for_eval(waveform: Union[np.ndarray, Sequence[float]],
                     index: int,
                     seed_base: int = 1234,
                     profiles: Sequence[str] = DEFAULT_PROFILES,
                     use_ffmpeg: bool = True) -> np.ndarray:
    """Deterministic per-sample degradation for eval (50/50 mobile/pstn)."""
    profs = list(profiles) or list(DEFAULT_PROFILES)
    profile = profs[index % len(profs)]
    return simulate_telephone(waveform, sr=TARGET_SR, profile=profile,
                              seed=int(seed_base) + int(index),
                              use_ffmpeg=use_ffmpeg)
