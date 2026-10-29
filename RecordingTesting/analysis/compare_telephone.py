#!/usr/bin/env python3
"""
Compare synthetic telephone-simulated speech against real recorded phone calls.

Produces:
  features_files.csv      one row per recording
  features_segments.csv   one row per VAD speech segment
  summary.json            aggregate medians / IQRs
  report.md               human-readable findings
  plots/*.png             comparison figures

Usage:
  /tmp/opencode/audioenv/bin/python RecordingTesting/analysis/compare_telephone.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as ss
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAVE_PRAAT = True
except Exception:  # pragma: no cover
    HAVE_PRAAT = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "telephone_clips"
REAL_DIR = ROOT / "RealRecordedPhoneCalls"
FAKE_DIR = ROOT / "FakePhoneCalls"
OUT = HERE

SR = 8000
FRAME = 160          # 20 ms at 8 kHz
HOP = 160            # non-overlapping 20 ms frames for VAD
SPEC_N = 256         # 32 ms PSD frames
SPEC_HOP = 128

BANDS = [
    ("0-100", 0, 100),
    ("100-300", 100, 300),
    ("300-1000", 300, 1000),
    ("1-2k", 1000, 2000),
    ("2-3k", 2000, 3000),
    ("3-3.4k", 3000, 3400),
    ("3.4-3.9k", 3400, 3900),
    ("3.9-4k", 3900, 4000),
]

GROUPS = ["real_gated", "real_noisy", "fake_mobile", "fake_pstn"]
COLORS = {"real": "#1f77b4", "real_gated": "#1f77b4", "real_noisy": "#ff7f0e",
          "fake": "#7f7f7f", "fake_mobile": "#d62728", "fake_pstn": "#2ca02c"}
LABELS = {"real": "Real calls", "real_gated": "Real: gated/clean",
          "real_noisy": "Real: noisy line", "fake": "Synth",
          "fake_mobile": "Synth mobile", "fake_pstn": "Synth PSTN"}


# ---------------------------------------------------------------------------
# I/O and basic DSP
# ---------------------------------------------------------------------------

def db(x):
    return 20.0 * np.log10(np.maximum(float(x), 1e-12))


def load_audio(path: Path):
    x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr != SR:
        # scipy's polyphase resampler low-passes, so no aliasing images.
        x8 = ss.resample_poly(x, SR, sr)
    else:
        x8 = x.copy()
    return x, sr, x8


def frame_energies(x, frame=FRAME, hop=HOP):
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0:
        return np.array([]), np.empty((0, frame))
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    e = 10.0 * np.log10(np.mean(fr ** 2, axis=1) + 1e-20)
    return e, fr


def close_gaps(mask, max_gap):
    """Fill False runs of length <= max_gap."""
    m = mask.copy()
    i, n = 0, len(m)
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def remove_short(mask, min_len):
    m = mask.copy()
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) < min_len:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def energy_vad(x, frame=FRAME, hop=HOP):
    """Relative-threshold VAD tuned for both clean synth clips and noisy calls."""
    e, fr = frame_energies(x, frame, hop)
    if len(e) == 0:
        return np.array([], dtype=bool), e, {}
    nf = float(np.percentile(e, 10))
    sl = float(np.percentile(e, 95))
    thr = max(nf + 7.0, sl - 30.0)
    # No usable dynamic range -> treat as non-speech.
    if sl - nf < 4.0:
        sp = np.zeros(len(e), dtype=bool)
    else:
        sp = e > thr
    sp = median_filter(sp.astype(np.uint8), size=3, mode="nearest").astype(bool)
    sp = close_gaps(sp, 3)       # fill <= 60 ms
    sp = remove_short(sp, 3)     # drop <= 60 ms
    info = dict(noise_floor_20ms=nf, speech_level_20ms=sl, vad_threshold_20ms=thr)
    return sp, e, info


def mask_to_samples(mask, frame=FRAME, hop=HOP, n=None):
    if n is None:
        n = len(mask) * hop
    out = np.zeros(n, dtype=bool)
    for i, v in enumerate(mask):
        if v:
            out[i * hop:min(n, i * hop + frame)] = True
    return out


def segments_from_mask(mask, frame=FRAME, hop=HOP, min_frames=8, merge_gap_frames=0):
    """Return (start_sample, end_sample) speech segments."""
    if len(mask) == 0:
        return []
    m = mask.astype(bool)
    if merge_gap_frames:
        m = close_gaps(m, merge_gap_frames)
    segs = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) >= min_frames:
                segs.append((i * hop, min(j * hop, n * hop)))
            i = j
        else:
            i += 1
    return segs


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def frame_psd(x, sr=SR, nfft=SPEC_N, hop=SPEC_HOP):
    win = np.hanning(nfft)
    n = 1 + max(0, (len(x) - nfft) // hop)
    if n <= 0:
        return np.empty((0, nfft // 2 + 1)), np.fft.rfftfreq(nfft, 1 / sr), np.array([])
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx] * win
    S = np.abs(np.fft.rfft(fr, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    return S, freqs, idx[:, 0]


def spec_shape_metrics(S, freqs):
    total = np.sum(S, axis=1) + 1e-20
    norm = S / total[:, None]
    centroid = np.sum(norm * freqs[None, :], axis=1)
    # rolloff 85 / 95
    cum = np.cumsum(norm, axis=1)
    roll85 = np.array([freqs[min(np.searchsorted(c, 0.85), len(freqs) - 1)] for c in cum])
    roll95 = np.array([freqs[min(np.searchsorted(c, 0.95), len(freqs) - 1)] for c in cum])
    # spectral flatness (power)
    flat = np.exp(np.mean(np.log(S[:, 1:] + 1e-20), axis=1)) / (np.mean(S[:, 1:], axis=1) + 1e-20)
    return dict(centroid=centroid, roll85=roll85, roll95=roll95, flatness=flat)


def band_fractions_from_psd(psd, freqs):
    out = {}
    total = float(np.sum(psd)) + 1e-20
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.sum(psd[m]) / total)
    return out


def spectral_summary(x, speech_mask_sample=None, exclude_silence=False):
    S, freqs, starts = frame_psd(x)
    if len(S) == 0:
        return {}
    if speech_mask_sample is not None:
        sf_frac = np.array([np.mean(speech_mask_sample[s:s + SPEC_N]) if s + SPEC_N <= len(speech_mask_sample)
                            else np.mean(speech_mask_sample[s:]) for s in starts])
    else:
        sf_frac = np.ones(len(S))
    sp_sel = sf_frac > 0.5
    no_sel = sf_frac < 0.1
    if exclude_silence:
        # Skip A-law digital silence; it is a recorder gate, not line noise.
        e = 10 * np.log10(np.mean(S, axis=1) + 1e-20)
        no_sel &= e > -65.0
    out = {}
    if np.any(sp_sel):
        Ssp = S[sp_sel]
        m = spec_shape_metrics(Ssp, freqs)
        out.update({
            "spec_centroid": float(np.median(m["centroid"])),
            "spec_roll85": float(np.median(m["roll85"])),
            "spec_roll95": float(np.median(m["roll95"])),
            "spec_flatness": float(np.median(m["flatness"])),
        })
        med = np.median(Ssp, axis=0)
        out["speech_psd"] = med
        out["speech_psd_norm"] = med / (np.sum(med) + 1e-20)
        out["speech_band"] = band_fractions_from_psd(med, freqs)
        # spectral flux between consecutive speech frames
        if np.sum(sp_sel) > 2:
            nsp = Ssp / (np.sum(Ssp, axis=1, keepdims=True) + 1e-20)
            flux = np.mean(np.abs(np.diff(10 * np.log10(nsp + 1e-12), axis=0)))
            out["spec_flux_db"] = float(flux)
        # loud-frame spectrum: isolates voice timbre from the noise floor
        e = 10 * np.log10(np.mean(Ssp, axis=1) + 1e-20)
        loud = e >= np.percentile(e, 90)
        if np.sum(loud) >= 3:
            Pl = np.mean(Ssp[loud], axis=0)
            Pl = Pl / (np.sum(Pl) + 1e-20)
            out["loud_psd_norm"] = Pl
            out["loud_band"] = band_fractions_from_psd(Pl, freqs)
            msk = (freqs >= 300) & (freqs <= 3400)
            lf = np.log2(freqs[msk] + 1)
            lp = 10 * np.log10(Pl[msk] + 1e-20)
            out["spec_tilt_db_oct"] = float(np.polyfit(lf, lp, 1)[0])
            ref = np.sum(Pl[(freqs >= 300) & (freqs < 1000)]) + 1e-20
            out["hf_1_2k_rel_db"] = float(10 * np.log10(
                (np.sum(Pl[(freqs >= 1000) & (freqs < 2000)]) + 1e-20) / ref))
            out["hf_2_3k_rel_db"] = float(10 * np.log10(
                (np.sum(Pl[(freqs >= 2000) & (freqs < 3400)]) + 1e-20) / ref))
            out["hf_3_4k_rel_db"] = float(10 * np.log10(
                (np.sum(Pl[(freqs >= 3400) & (freqs < 4000)]) + 1e-20) / ref))
    if np.any(no_sel):
        Sno = S[no_sel]
        medn = np.median(Sno, axis=0)
        out["noise_psd"] = medn
        out["noise_psd_norm"] = medn / (np.sum(medn) + 1e-20)
        out["noise_band"] = band_fractions_from_psd(medn, freqs)
        m = spec_shape_metrics(Sno, freqs)
        out["noise_centroid"] = float(np.median(m["centroid"]))
        out["noise_flatness"] = float(np.median(m["flatness"]))
        out["noise_roll95"] = float(np.median(m["roll95"]))
        # slope in log-log space, 100-3500 Hz
        msk = (freqs >= 100) & (freqs <= 3500)
        lf = np.log10(freqs[msk] + 1)
        lp = 10 * np.log10(medn[msk] + 1e-20)
        out["noise_slope_db_per_dec"] = float(np.polyfit(lf, lp, 1)[0])
        lo = np.sum(medn[(freqs < 500)])
        hi = np.sum(medn[(freqs >= 500)])
        out["noise_low_ratio"] = float(lo / (hi + 1e-20))
    return out, freqs


def hum_features(x, speech_mask_sample):
    """Prominence of mains-hum harmonics measured on noise-only 128 ms frames."""
    nfft = 1024
    hop = 512
    win = np.hanning(nfft)
    n = 1 + max(0, (len(x) - nfft) // hop)
    if n <= 0:
        return {}
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    if speech_mask_sample is not None:
        noise_frac = 1.0 - np.array([
            np.mean(speech_mask_sample[s:s + nfft]) if s + nfft <= len(speech_mask_sample)
            else np.mean(speech_mask_sample[s:]) for s in idx[:, 0]
        ])
        sel = noise_frac >= 0.9
        if np.sum(sel) < 3:
            # fall back to the quietest 10% of windows
            e = np.mean(fr ** 2, axis=1)
            sel = e <= np.percentile(e, 10)
    else:
        sel = np.ones(n, dtype=bool)
    if not np.any(sel):
        return {}
    P = np.mean(np.abs(np.fft.rfft(fr[sel] * win, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    out = {}
    for f in (50, 100, 150, 200, 250, 60, 120, 180):
        band = (freqs >= f - 6) & (freqs <= f + 6)
        neigh = (freqs >= f - 25) & (freqs <= f + 25) & ~band
        if not np.any(band) or not np.any(neigh):
            out[f"hum{f}"] = np.nan
            continue
        out[f"hum{f}"] = float(10 * np.log10((np.mean(P[band]) + 1e-20) /
                                             (np.median(P[neigh]) + 1e-20)))
    return out


# ---------------------------------------------------------------------------
# Voice / pitch
# ---------------------------------------------------------------------------

def pitch_stats(x, sr=SR):
    if not HAVE_PRAAT or len(x) < sr * 0.15:
        return {}
    try:
        snd = parselmouth.Sound(x, sr)
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60,
                                pitch_ceiling=400, voicing_threshold=0.45)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) < 5:
            return {}
        p10, p50, p90 = np.percentile(f0, [10, 50, 90])
        out = {
            "f0_median": float(p50),
            "f0_p10": float(p10),
            "f0_p90": float(p90),
            "f0_range_st": float(12 * np.log2(max(p90, 1e-6) / max(p10, 1e-6))),
        }
        try:
            harm = praat_call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
            h = np.asarray(harm.values[0], dtype=float)
            h = h[np.isfinite(h) & (h > -50)]
            if len(h):
                out["hnr_median"] = float(np.median(h))
        except Exception:
            pass
        try:
            pp = praat_call(snd, "To PointProcess (periodic, cc)", 60, 400)
            out["jitter_local"] = float(praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            out["shimmer_local"] = float(praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            pass
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Format / codec signature
# ---------------------------------------------------------------------------

def codec_signature(x8):
    xi = np.clip(np.round(x8 * 32768.0), -32768, 32767).astype(np.int32)
    if len(xi) == 0:
        return {}
    absx = np.abs(xi)
    nz = absx[absx > 0]
    uniq = np.unique(xi)
    out = {
        "n_unique_levels": int(len(uniq)),
        "unique_per_sample": float(len(uniq) / len(xi)),
        "mod8_ratio": float(np.mean(xi % 8 == 0)),
        "mod4_ratio": float(np.mean(xi % 4 == 0)),
        "odd_ratio": float(np.mean(xi % 2 != 0)),
        "min_abs_nonzero": int(nz.min()) if len(nz) else 0,
    }
    out["alaw_like"] = bool(out["mod8_ratio"] > 0.99 and out["n_unique_levels"] <= 256
                            and out["min_abs_nonzero"] >= 8)
    return out


def zero_run_features(x8, speech_mask_sample):
    """Short near-silent dips inside speech (dropouts/fades) + digital-silence runs."""
    if len(x8) == 0:
        return {}
    z = np.abs(x8) < 1e-4
    # run-length encode
    d = np.diff(np.r_[0, z.astype(np.int8), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        return {"short_dip_rate_per_min": 0.0, "zero_frac": 0.0, "long_zero_frac": 0.0}
    durations = (ends - starts) / SR
    short = (durations >= 0.005) & (durations <= 0.100)
    long = durations > 0.100
    # only count short dips that occur inside detected speech
    if speech_mask_sample is not None and np.any(speech_mask_sample):
        inside = np.array([np.mean(speech_mask_sample[s:e]) > 0.5 if e <= len(speech_mask_sample)
                           else np.mean(speech_mask_sample[s:]) > 0.5
                           for s, e in zip(starts, ends)])
    else:
        inside = np.ones(len(starts), dtype=bool)
    speech_min = max(np.sum(speech_mask_sample) / SR / 60.0, 1e-6) if speech_mask_sample is not None else len(x8) / SR / 60.0
    return {
        "short_dip_rate_per_min": float(np.sum(short & inside) / speech_min),
        "short_dip_mean_ms": float(np.mean(durations[short & inside]) * 1000) if np.any(short & inside) else 0.0,
        "zero_frac": float(np.mean(z)),
        "long_zero_frac": float(np.sum(durations[long]) / (len(x8) / SR)),
    }


# ---------------------------------------------------------------------------
# File-level analysis
# ---------------------------------------------------------------------------

def analyze_file(path: Path, group: str):
    x, sr_native, x8 = load_audio(path)
    duration = len(x8) / SR
    if duration < 0.02:
        return None
    sp, e, vad_info = energy_vad(x8)
    sp_sample = mask_to_samples(sp, n=len(x8))
    segs = segments_from_mask(sp, min_frames=8, merge_gap_frames=0)
    # merge segments separated by <= 100 ms for pause statistics
    segs_merged = segments_from_mask(sp, min_frames=8, merge_gap_frames=5)

    row = {
        "group": group,
        "file": path.name,
        "duration_s": duration,
        "sr_native": sr_native,
        "peak_db": db(np.max(np.abs(x8))) if len(x8) else np.nan,
        "rms_db": db(np.sqrt(np.mean(x8 ** 2))) if len(x8) else np.nan,
        "dc": float(np.mean(x8)),
        "clip_frac": float(np.mean(np.abs(x8) >= 0.999)),
        "clip_frac_099": float(np.mean(np.abs(x8) >= 0.99)),
    }
    row.update(vad_info)
    row["speech_frac"] = float(np.mean(sp)) if len(sp) else 0.0
    if len(e) and np.any(sp):
        row["speech_rms_db"] = float(np.median(e[sp]))
        row["speech_iqr_db"] = float(np.percentile(e[sp], 75) - np.percentile(e[sp], 25))
        row["peak_speech_db"] = db(np.max(np.abs(x8[sp_sample]))) if np.any(sp_sample) else np.nan
    if len(e):
        noise_sel = e < (row.get("vad_threshold_20ms", np.inf) - 4.0)
        if np.sum(noise_sel) < 5:
            noise_sel = e <= np.percentile(e, 10)
        row["noise_rms_db"] = float(np.median(e[noise_sel])) if np.any(noise_sel) else np.nan
        row["noise_iqr_db"] = float(np.percentile(e[noise_sel], 90) - np.percentile(e[noise_sel], 10)) if np.any(noise_sel) else np.nan
        # A-law digital silence (code +/-8) is a recorder gate, not line noise.
        row["is_gated"] = bool(np.isfinite(row.get("noise_rms_db", np.nan)) and row["noise_rms_db"] < -65.0)
        line_sel = noise_sel & (e > -65.0)
        row["line_noise_db"] = float(np.median(e[line_sel])) if np.any(line_sel) else np.nan
        if "speech_rms_db" in row and np.isfinite(row.get("noise_rms_db", np.nan)):
            row["snr_db"] = row["speech_rms_db"] - row["noise_rms_db"]
        if "speech_rms_db" in row and np.isfinite(row.get("line_noise_db", np.nan)):
            row["line_snr_db"] = row["speech_rms_db"] - row["line_noise_db"]
    # pause / segment statistics
    if segs_merged:
        seg_durs = np.array([(b - a) / SR for a, b in segs_merged])
        gaps = []
        for (a1, b1), (a2, b2) in zip(segs_merged[:-1], segs_merged[1:]):
            gaps.append((a2 - b1) / SR)
        gaps = np.array(gaps) if gaps else np.array([0.0])
        row.update({
            "n_speech_segments": int(len(segs_merged)),
            "seg_dur_median": float(np.median(seg_durs)),
            "seg_dur_p90": float(np.percentile(seg_durs, 90)),
            "seg_dur_max": float(np.max(seg_durs)),
            "pause_median": float(np.median(gaps)),
            "pause_p90": float(np.percentile(gaps, 90)),
            "pause_max": float(np.max(gaps)),
            "speech_segments_per_min": float(len(segs_merged) / max(duration / 60, 1e-6)),
            "lead_silence_s": float(segs_merged[0][0] / SR),
            "tail_silence_s": float((len(x8) - segs_merged[-1][1]) / SR),
        })
    row.update(codec_signature(x8))
    row.update(zero_run_features(x8, sp_sample))

    spec, freqs = spectral_summary(x8, sp_sample, exclude_silence=True)
    row["_freqs"] = freqs
    row["_speech_psd"] = spec.pop("speech_psd", None)
    row["_speech_psd_norm"] = spec.pop("speech_psd_norm", None)
    row["_loud_psd_norm"] = spec.pop("loud_psd_norm", None)
    row["_noise_psd"] = spec.pop("noise_psd", None)
    row["_noise_psd_norm"] = spec.pop("noise_psd_norm", None)
    for k, v in spec.items():
        if isinstance(v, dict):
            prefix = k.replace("_band", "")
            for kk, vv in v.items():
                row[f"{prefix}_band_{kk}"] = vv
        else:
            row[k] = v
    row.update(hum_features(x8, sp_sample) if not row.get("is_gated") else {})
    row.update(pitch_stats(x8))

    # 1-second level variation (rough AGC / level consistency)
    if duration >= 1.0:
        n1 = len(x8) // SR
        lev = np.array([db(np.sqrt(np.mean(x8[i * SR:(i + 1) * SR] ** 2))) for i in range(n1)])
        lev = lev[np.isfinite(lev)]
        if len(lev) >= 2:
            row["level_1s_p90_p10_db"] = float(np.percentile(lev, 90) - np.percentile(lev, 10))
    return row


# ---------------------------------------------------------------------------
# Segment-level analysis
# ---------------------------------------------------------------------------

MAX_SEG_S = 20.0


def analyze_segments(path: Path, group: str):
    x, sr_native, x8 = load_audio(path)
    sp, e, _ = energy_vad(x8)
    segs = segments_from_mask(sp, min_frames=8, merge_gap_frames=5)
    out = []
    for a, b in segs:
        # split very long continuous speech into <=20 s chunks
        pieces = []
        if (b - a) / SR > MAX_SEG_S:
            step = int(MAX_SEG_S * SR)
            for s in range(a, b, step):
                pieces.append((s, min(b, s + step)))
        else:
            pieces.append((a, b))
        for s, e2 in pieces:
            seg = x8[s:e2]
            if len(seg) < SR * 0.3:
                continue
            mask = np.ones(len(seg), dtype=bool)
            spec, _ = spectral_summary(seg, mask)
            r = {
                "group": group,
                "file": path.name,
                "seg_start_s": s / SR,
                "seg_dur_s": len(seg) / SR,
                "seg_rms_db": db(np.sqrt(np.mean(seg ** 2))),
                "seg_peak_db": db(np.max(np.abs(seg))),
                "seg_crest_db": db(np.max(np.abs(seg))) - db(np.sqrt(np.mean(seg ** 2))),
            }
            for k, v in spec.items():
                if isinstance(v, dict):
                    continue
                r[k] = v
            if "speech_band" in spec:
                for k, v in spec["speech_band"].items():
                    r[f"band_{k}"] = v
            r.update(pitch_stats(seg))
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def med_iqr(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    return [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), int(len(v))]


def summarise(rows, keys):
    groups = {}
    for g in ("real", "real_gated", "real_noisy", "fake", "fake_mobile", "fake_pstn"):
        if g == "real":
            sub = [r for r in rows if r["group"] == "real"]
        elif g == "real_gated":
            sub = [r for r in rows if r["group"] == "real" and r.get("is_gated")]
        elif g == "real_noisy":
            sub = [r for r in rows if r["group"] == "real" and not r.get("is_gated")]
        elif g == "fake":
            sub = [r for r in rows if str(r["group"]).startswith("fake")]
        else:
            sub = [r for r in rows if r["group"] == g]
        groups[g] = {"n": len(sub), "metrics": {k: med_iqr([r.get(k) for r in sub]) for k in keys}}
    return groups


def plot_psd(rows, key, title, fname, freqs, rel=False):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for g in GROUPS:
        arr = [r.get(key) for r in rows if r.get("_group4") == g and r.get(key) is not None]
        if not arr:
            continue
        A = np.vstack(arr)
        A = 10 * np.log10(A + 1e-12)
        med = np.median(A, axis=0)
        lo = np.percentile(A, 25, axis=0)
        hi = np.percentile(A, 75, axis=0)
        ax.plot(freqs, med, color=COLORS[g], lw=1.6, label=f"{LABELS[g]} (n={len(arr)})")
        ax.fill_between(freqs, lo, hi, color=COLORS[g], alpha=0.10, lw=0)
    ax.set_xlim(0, 4000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised PSD (dB)" if rel else "PSD (dB, arbitrary)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / fname, dpi=150)
    plt.close(fig)


def plot_boxes(rows, keys, fname, ncols=3):
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax, (key, label) in zip(axes, keys):
        data = []
        for g in GROUPS:
            v = [r.get(key) for r in rows if r.get("_group4") == g
                 and r.get(key) is not None and np.isfinite(r.get(key))]
            data.append(v)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["real\ngated", "real\nnoisy", "synth\nmobile", "synth\nPSTN"], fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / fname, dpi=150)
    plt.close(fig)


def plot_examples():
    real_sel = [
        REAL_DIR / "out-09122594551-105-20261022-123448-1792659886.19905.wav",
        REAL_DIR / "out-206-104-20261022-100640-1792651000.19648.wav",
    ]
    fake_sel = [
        FAKE_DIR / "spk_52762_clip0_id7_mobile-MCI-Irancell.wav",
        FAKE_DIR / "spk_52762_clip0_id7_pstn-TCI.wav",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    for i, f in enumerate(real_sel + fake_sel):
        x, _, x8 = load_audio(f)
        full_dur = len(x8) / SR
        if i < 2 and len(x8) > 10 * SR:
            # Pick the most speech-dense 10 s window for a readable example.
            win = 10 * SR
            starts = np.arange(0, len(x8) - win + 1, win)
            energy = [np.mean(x8[s:s + win] ** 2) for s in starts]
            s0 = int(starts[int(np.argmax(energy))])
            x8 = x8[s0:s0 + win]
        t = np.arange(len(x8)) / SR
        ax = axes[i, 0]
        ax.plot(t, x8, lw=0.35, color="#1f4e79")
        kind = "REAL" if i < 2 else "SYNTH"
        extra = f"  ({full_dur:.1f}s total)" if full_dur > len(x8) / SR + 0.1 else f"  ({full_dur:.1f}s)"
        ax.set_title(f"{kind}: {f.name[:55]}{extra}", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("s")
        ax.grid(alpha=0.2)
        ax = axes[i, 1]
        fq, tt, S = ss.spectrogram(x8, SR, nperseg=256, noverlap=192, mode="magnitude")
        S = 20 * np.log10(S + 1e-10)
        ax.pcolormesh(tt, fq / 1000, S, vmin=-105, vmax=-30, shading="auto", cmap="magma")
        ax.set_ylim(0, 4)
        ax.set_xlabel("s")
        ax.set_ylabel("kHz")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "examples.png", dpi=150)
    plt.close(fig)


def plot_pca(rows, keys):
    groups = GROUPS
    markers = {"real_gated": "o", "real_noisy": "o", "fake_mobile": "^", "fake_pstn": "s"}
    X, y = [], []
    for r in rows:
        v = [r.get(k) for k in keys]
        X.append([np.nan if x is None else float(x) for x in v])
        y.append(r.get("_group4"))
    if len(X) < 4:
        return
    X = np.asarray(X, float)
    med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Z = X @ V[:, :2]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for g in groups:
        m = np.array([yy == g for yy in y])
        if np.any(m):
            ax.scatter(Z[m, 0], Z[m, 1], c=COLORS[g], marker=markers[g],
                       s=38 if g.startswith("real") else 110,
                       alpha=0.65, edgecolors="white" if not g.startswith("real") else "none",
                       label=f"{LABELS[g]} (n={int(m.sum())})", zorder=3)
    ax.set_xlabel(f"PC1 ({100*w[0]/w.sum():.0f}% var)")
    ax.set_ylabel(f"PC2 ({100*w[1]/w.sum():.0f}% var)")
    ax.set_title("File-level channel/format fingerprint (PCA)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "pca_channels.png", dpi=150)
    plt.close(fig)


def plot_quantization(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = ["real", "fake_mobile", "fake_pstn"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, key, title in [
        (axes[0], "n_unique_levels", "Distinct PCM sample levels per file"),
        (axes[1], "mod8_ratio", "Fraction of samples on the A-law 8-bit grid"),
    ]:
        data = [[r.get(key) for r in rows if r["group"] == g and r.get(key) is not None] for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["real", "synth\nmobile", "synth\nPSTN"], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "quantization.png", dpi=150)
    plt.close(fig)


def plot_segment_durations(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in [
        (axes[0], "seg_dur_median", "Median speech segment duration (s)"),
        (axes[1], "speech_frac", "Speech fraction of file"),
        (axes[2], "pause_median", "Median inter-segment pause (s)"),
    ]:
        data = [[r.get(key) for r in rows if r.get("_group4") == g and r.get(key) is not None] for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["real\ngated", "real\nnoisy", "synth\nmobile", "synth\nPSTN"], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "segments.png", dpi=150)
    plt.close(fig)


def plot_f0(seg_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "f0_median", "Median F0 per speech segment (Hz)"),
        (axes[1], "f0_range_st", "Within-segment F0 range (semitones)"),
    ]:
        data = [[r.get(key) for r in seg_rows if r.get("_group4") == g and r.get(key) is not None] for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["real\ngated", "real\nnoisy", "synth\nmobile", "synth\nPSTN"], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "f0.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(REAL_DIR.glob("*.wav")):
        files.append((p, "real"))
    for p in sorted(FAKE_DIR.glob("*.wav")):
        name = p.name.lower()
        files.append((p, "fake_mobile" if "mobile" in name else "fake_pstn"))

    rows = []
    seg_rows = []
    empty = []
    for i, (p, g) in enumerate(files, 1):
        try:
            r = analyze_file(p, g)
            if r is None:
                empty.append(p.name)
                continue
            rows.append(r)
            seg_rows.extend(analyze_segments(p, g))
        except Exception as ex:
            print(f"FAILED {p.name}: {type(ex).__name__}: {ex}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    # Attach the four-way comparison label used by plots.
    by_file = {}
    for r in rows:
        if r["group"] == "real":
            r["_group4"] = "real_gated" if r.get("is_gated") else "real_noisy"
        else:
            r["_group4"] = r["group"]
        by_file[r["file"]] = r["_group4"]
    for r in seg_rows:
        r["_group4"] = by_file.get(r["file"], r["group"])

    # Remove helper arrays before CSV
    helper_keys = {"_freqs", "_speech_psd", "_speech_psd_norm", "_loud_psd_norm",
                   "_noise_psd", "_noise_psd_norm", "_group4"}
    csv_rows = [{k: v for k, v in r.items() if k not in helper_keys} for r in rows]

    import csv
    key_order = ["group", "file", "duration_s", "sr_native", "rms_db", "peak_db", "speech_rms_db",
                 "noise_rms_db", "line_noise_db", "snr_db", "line_snr_db", "is_gated", "speech_frac",
                 "n_speech_segments", "seg_dur_median", "pause_median", "pause_max",
                 "n_unique_levels", "mod8_ratio", "min_abs_nonzero", "alaw_like",
                 "short_dip_rate_per_min", "clip_frac", "hum50", "hum100", "hum150",
                 "spec_centroid", "spec_roll95", "spec_tilt_db_oct",
                 "hf_1_2k_rel_db", "hf_2_3k_rel_db", "hf_3_4k_rel_db",
                 "spec_flatness", "noise_flatness", "f0_median", "f0_range_st", "hnr_median"]
    with open(OUT / "features_files.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        cols = key_order + sorted(set().union(*(r.keys() for r in csv_rows)) - set(key_order))
        w.writerow(cols)
        for r in csv_rows:
            w.writerow([r.get(c, "") for c in cols])
    with open(OUT / "features_segments.csv", "w", newline="") as fh:
        if seg_rows:
            cols = sorted(set().union(*(r.keys() for r in seg_rows)))
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(seg_rows)

    keys = ["duration_s", "rms_db", "peak_db", "speech_rms_db", "noise_rms_db", "line_noise_db",
            "snr_db", "line_snr_db", "noise_iqr_db", "speech_frac", "n_speech_segments", "seg_dur_median",
            "pause_median", "pause_max", "n_unique_levels", "mod8_ratio", "min_abs_nonzero",
            "short_dip_rate_per_min", "clip_frac", "hum50", "hum100", "hum150",
            "spec_centroid", "spec_roll95", "spec_tilt_db_oct", "hf_1_2k_rel_db",
            "hf_2_3k_rel_db", "hf_3_4k_rel_db", "spec_flatness", "noise_flatness",
            "noise_low_ratio", "noise_slope_db_per_dec", "f0_median", "f0_range_st",
            "hnr_median", "level_1s_p90_p10_db"]
    summary = summarise(rows, keys)
    summary["empty_files"] = empty
    summary["speech_segments"] = {
        g: {"n": len([r for r in seg_rows if r.get("_group4") == g]),
            "seg_dur_s": med_iqr([r.get("seg_dur_s") for r in seg_rows if r.get("_group4") == g]),
            "f0_median": med_iqr([r.get("f0_median") for r in seg_rows if r.get("_group4") == g]),
            "f0_range_st": med_iqr([r.get("f0_range_st") for r in seg_rows if r.get("_group4") == g]),
            "hnr_median": med_iqr([r.get("hnr_median") for r in seg_rows if r.get("_group4") == g]),
            "spec_centroid": med_iqr([r.get("spec_centroid") for r in seg_rows if r.get("_group4") == g]),
            "spec_roll95": med_iqr([r.get("spec_roll95") for r in seg_rows if r.get("_group4") == g]),
            }
        for g in GROUPS}
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plots
    freqs = np.fft.rfftfreq(SPEC_N, 1 / SR)
    plot_psd(rows, "_speech_psd_norm", "Long-term speech spectrum (median, IQR band)", "psd_speech.png", freqs, rel=True)
    plot_psd(rows, "_loud_psd_norm", "Loud-frame speech spectrum (voice timbre, median)", "psd_loud.png", freqs, rel=True)
    plot_psd(rows, "_noise_psd_norm", "Long-term line-noise spectrum (median, IQR band)", "psd_noise.png", freqs, rel=True)
    plot_boxes(rows, [
        ("speech_rms_db", "Active speech level (dBFS)"),
        ("noise_rms_db", "Noise floor / silence (dBFS)"),
        ("line_noise_db", "Line noise during pauses (dBFS)"),
        ("line_snr_db", "Line SNR (dB)"),
        ("speech_frac", "Speech fraction"),
        ("seg_dur_median", "Median speech segment (s)"),
        ("short_dip_rate_per_min", "Short dropouts per speech-minute"),
        ("noise_iqr_db", "Noise-floor variation p90-p10 (dB)"),
        ("hf_1_2k_rel_db", "1-2 kHz energy rel. 0.3-1 kHz (dB)"),
        ("hf_2_3k_rel_db", "2-3.4 kHz energy rel. 0.3-1 kHz (dB)"),
        ("hum50", "50 Hz hum prominence (dB)"),
        ("n_unique_levels", "Distinct PCM levels"),
        ("mod8_ratio", "A-law 8-bit grid fraction"),
    ], "metric_boxes.png", ncols=3)
    plot_examples()
    plot_pca(rows, ["rms_db", "peak_db", "speech_rms_db", "speech_frac", "seg_dur_median",
                    "pause_median", "spec_centroid", "spec_roll95", "spec_tilt_db_oct",
                    "hf_1_2k_rel_db", "hf_2_3k_rel_db", "hf_3_4k_rel_db", "spec_flatness",
                    "f0_median", "f0_range_st", "hnr_median", "short_dip_rate_per_min",
                    "level_1s_p90_p10_db"])
    plot_quantization(rows)
    plot_segment_durations(rows)
    plot_f0(seg_rows)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
