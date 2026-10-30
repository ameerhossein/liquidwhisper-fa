#!/usr/bin/env python3
"""Build a listening kit for the telephone-channel simulator.

Outputs a self-contained folder with an ``index.html`` player and WAV files:

1. **Paired, same sentence** - clean source, legacy synth mobile/PSTN,
   simulated mobile/PSTN, simulated gated.  These six `FakePhoneCalls` files
   are **not** real carrier recordings; they are older synth outputs with
   known source clips.  This section compares the current simulator to that
   legacy baseline.  Levels are matched on active speech so timbre, bandlimit
   and noise are what you hear, not loudness.
2. **Gated, different content** - real gated call-center excerpts vs simulated
   gated clips.  There are no transcripts and no clean counterparts for the
   real gated calls, so this is a perceptual check, not a paired WER test.

Usage:
  python RecordingTesting/analysis/make_listening_kit.py
  python RecordingTesting/analysis/make_listening_kit.py --out-dir /tmp/listen
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.telephone import simulate_telephone  # noqa: E402

FAKE_RE = re.compile(
    r"^spk_(\d+)_clip(\d+)_id(\d+)_(mobile|pstn)-", re.IGNORECASE
)
TARGET_ACTIVE_DB = -26.0
MAX_LEVEL_GAIN_DB = 15.0


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    from math import gcd
    from scipy.signal import resample_poly

    factor = gcd(int(sr_in), int(sr_out))
    return resample_poly(
        audio, sr_out // factor, sr_in // factor
    ).astype(np.float32)


def frame_energy_db(audio: np.ndarray, sr: int, frame_s: float = 0.02):
    frame = max(1, int(round(frame_s * sr)))
    n = len(audio) // frame
    if n == 0:
        return np.array([10.0 * np.log10(float(np.mean(audio ** 2)) + 1e-20)])
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)


def active_rms_db(audio: np.ndarray, sr: int) -> float:
    energy = frame_energy_db(audio, sr)
    if energy.size == 0:
        return -200.0
    low, high = np.percentile(energy, [10.0, 95.0])
    threshold = max(low + 7.0, high - 30.0)
    active = energy > threshold
    if not np.any(active):
        return float(np.median(energy))
    return float(np.median(energy[active]))


def level_match(audio: np.ndarray, sr: int,
                target_db: float = TARGET_ACTIVE_DB) -> Tuple[np.ndarray, float]:
    current = active_rms_db(audio, sr)
    gain_db = float(np.clip(target_db - current, -MAX_LEVEL_GAIN_DB,
                            MAX_LEVEL_GAIN_DB))
    out = (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.98:
        scale = 0.98 / peak
        out = (out * scale).astype(np.float32)
        gain_db += 20.0 * np.log10(scale)
    return out, float(gain_db)


def densest_window(audio: np.ndarray, sr: int,
                   seconds: float = 10.0) -> np.ndarray:
    energy = frame_energy_db(audio, sr)
    frame = max(1, int(round(0.02 * sr)))
    n = len(energy)
    win = max(1, int(round(seconds / 0.02)))
    if n <= win:
        return audio
    low, high = np.percentile(energy, [10.0, 95.0])
    active = (energy > max(low + 7.0, high - 30.0)).astype(np.float64)
    counts = np.convolve(active, np.ones(win), mode="valid")
    start_frame = int(np.argmax(counts))
    start = max(0, start_frame * frame - int(0.1 * sr))
    end = min(len(audio), start + int(seconds * sr))
    return audio[start:end]


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def parse_fake(path: Path) -> Optional[Dict[str, object]]:
    match = FAKE_RE.match(path.name)
    if not match:
        return None
    speaker, clip, dataset_id, profile = match.groups()
    return {
        "speaker": speaker,
        "clip": clip,
        "id": int(dataset_id),
        "profile": profile.lower(),
        "base": f"spk_{speaker}_clip{clip}_id{dataset_id}",
        "transcript_key": f"{speaker}_{dataset_id}",
    }


def load_gated_rows(features_csv: Path, real_dir: Path, limit: int,
                    min_speech_frac: float = 0.45) -> List[Path]:
    if not features_csv.is_file():
        return []
    with open(features_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    selected = []
    for row in rows:
        if row.get("group") != "real":
            continue
        if str(row.get("is_gated", "")).lower() != "true":
            continue
        try:
            if float(row.get("speech_frac", 0.0)) < min_speech_frac:
                continue
            if float(row.get("duration_s", 0.0)) < 4.0:
                continue
            # A very high crest factor cannot be level-matched without
            # clipping, so it would be quieter than the rest of the kit.
            if (float(row.get("peak_db", 0.0))
                    - float(row.get("speech_rms_db", -200.0)) > 24.0):
                continue
        except ValueError:
            continue
        path = real_dir / row["file"]
        if path.is_file():
            selected.append(path)
    return selected[:limit]


def audio_row(label: str, relpath: str, note: str = "") -> str:
    return (
        '<div class="row">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<audio controls preload="none" src="{html.escape(relpath)}"></audio>'
        f'<div class="note">{html.escape(note)}</div>'
        '</div>'
    )


def render_html(cards: Sequence[Dict[str, object]],
                gated_real: Sequence[Dict[str, object]],
                gated_sim: Sequence[Dict[str, object]],
                out_dir: Path) -> str:
    def rel(path: Path) -> str:
        return path.relative_to(out_dir).as_posix()

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Telephone simulator listening kit</title>",
        "<style>",
        ":root{color-scheme:dark}body{font:15px/1.45 system-ui,sans-serif;",
        "margin:0;padding:24px;background:#14171c;color:#e7e9ee}",
        "main{max-width:1000px;margin:auto}h1{font-size:1.6rem}",
        "h2{margin-top:2rem;border-bottom:1px solid #333a45;padding-bottom:.3rem}",
        ".card{background:#1d222a;border:1px solid #2d3541;border-radius:10px;",
        "padding:14px 16px;margin:14px 0}.row{display:grid;",
        "grid-template-columns:150px minmax(220px,1fr) 220px;gap:12px;",
        "align-items:center;padding:6px 0;border-top:1px solid #262d37}",
        ".row:first-of-type{border-top:0}.label{font-weight:600}",
        ".note{color:#9aa4b2;font-size:.85rem}audio{width:100%}",
        "details{margin-top:10px;color:#b9c1cc}summary{cursor:pointer}",
        ".rtl{direction:rtl;text-align:right;font-size:1.05rem}",
        ".warn{background:#2a2214;border-color:#5a4a24;color:#f0d9a6}",
        "code{background:#11151a;padding:1px 4px;border-radius:4px}",
        "@media(max-width:700px){.row{grid-template-columns:1fr}}",
        "</style></head><body><main>",
        "<h1>Telephone simulator listening kit</h1>",
        '<p class="card warn">All clips are level-matched on active speech to '
        f"{TARGET_ACTIVE_DB:.0f} dBFS (max gain {MAX_LEVEL_GAIN_DB:.0f} dB), "
        "so differences are timbre, bandlimit, noise, dropouts and gating, "
        "not playback loudness. Listen on the same device and volume. "
        "The six paired files are <strong>legacy synth</strong>, not real "
        "carrier recordings.</p>",
        "<h2>1. Paired same sentence: legacy synth vs current synth</h2>",
        "<p>These six files are <strong>legacy synth outputs</strong>, not real "
        "carrier recordings. Compare the legacy and current simulator outputs "
        "<em>within</em> each card; this is not a real-network validation.</p>",
    ]
    for card in cards:
        transcript = html.escape(str(card.get("transcript", "")))
        parts.append('<section class="card">')
        parts.append(f"<h3>{html.escape(str(card['base']))}</h3>")
        for entry in card["entries"]:
            parts.append(audio_row(
                str(entry["label"]), rel(Path(entry["path"])),
                str(entry.get("note", "")),
            ))
        if transcript:
            parts.append(
                "<details><summary>Gold transcript</summary>"
                f'<p class="rtl" lang="fa">{transcript}</p></details>'
            )
        parts.append("</section>")

    parts += [
        "<h2>2. Gated channel: real call-center excerpts vs simulated</h2>",
        '<p class="card warn">These are <strong>not paired</strong>: the real '
        "gated calls have no transcripts and no clean counterparts. Judge the "
        "noise floor, bandlimit and gating character only, not word accuracy "
        "on identical content.</p>",
        '<section class="card"><h3>Real gated calls (call-center recordings, mostly mobile-originated)</h3>',
    ]
    for entry in gated_real:
        parts.append(audio_row(
            str(entry["label"]), rel(Path(entry["path"])),
            str(entry.get("note", "")),
        ))
    parts += [
        "</section>",
        '<section class="card"><h3>Simulated gated profile</h3>',
    ]
    for entry in gated_sim:
        parts.append(audio_row(
            str(entry["label"]), rel(Path(entry["path"])),
            str(entry.get("note", "")),
        ))
    parts += [
        "</section>",
        "<h2>What to listen for</h2>",
        "<ul>",
        "<li><strong>Intelligibility:</strong> are consonants and final "
        "syllables preserved?</li>",
        "<li><strong>Bandlimit:</strong> does the simulated channel sound too "
        "dark, too bright, or about right?</li>",
        "<li><strong>Noise:</strong> hiss, line rumble, hum, and whether noise "
        "sits under speech or pumps between words.</li>",
        "<li><strong>Gating:</strong> abrupt cuts, lost word tails, or audible "
        "fades at pauses.</li>",
        "<li><strong>Codec:</strong> mobile warble/robotics from AMR-NB; PSTN "
        "should be cleaner and narrower.</li>",
        "</ul>",
        "<p>Regenerate with <code>python RecordingTesting/analysis/"
        "make_listening_kit.py</code>.</p>",
        "</main></body></html>",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-dir", type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "FakePhoneCalls",
        help="Legacy synth outputs, not real carrier recordings.",
    )
    parser.add_argument(
        "--call-dir", type=Path,
        default=ROOT / "RecordingTesting" / "telephone_clips" / "RealRecordedPhoneCalls",
    )
    parser.add_argument(
        "--features-csv", type=Path,
        default=HERE / "features_files.csv",
    )
    parser.add_argument(
        "--clips-dir", type=Path,
        default=Path("/home/ameer/Hobbies/voiceprint/fa_clips"),
    )
    parser.add_argument(
        "--transcripts", type=Path,
        default=HERE / "neyshekar_transcripts.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gated", type=int, default=6)
    parser.add_argument("--gated-seconds", type=float, default=10.0)
    parser.add_argument(
        "--use-ffmpeg", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "telephone_listening_kit",
    )
    args = parser.parse_args()

    transcripts = {}
    if args.transcripts.is_file():
        transcripts = json.loads(args.transcripts.read_text(encoding="utf-8"))

    out_dir = args.out_dir
    manifest: Dict[str, object] = {
        "target_active_dbfs": TARGET_ACTIVE_DB,
        "max_level_gain_db": MAX_LEVEL_GAIN_DB,
        "paired": [],
        "gated_real": [],
        "gated_sim": [],
    }
    cards = []
    paired_bases = set()
    pairs: Dict[str, Dict[str, object]] = {}

    for legacy_path in sorted(args.legacy_dir.glob("*.wav")):
        info = parse_fake(legacy_path)
        if info is None:
            continue
        clean_path = (
            args.clips_dir / f"spk_{info['speaker']}"
            / f"clip{info['clip']}_id{info['id']}.wav"
        )
        if not clean_path.is_file():
            print(f"SKIP {legacy_path.name}: clean source not found")
            continue
        base = str(info["base"])
        paired_bases.add(base)
        pair = pairs.setdefault(base, {
            "base": base,
            "clean_path": clean_path,
            "transcript": transcripts.get(str(info["transcript_key"]), ""),
            "legacy": {},
        })
        pair["legacy"][str(info["profile"])] = legacy_path

    for base in sorted(pairs):
        pair = pairs[base]
        clean_path = Path(pair["clean_path"])
        clean, clean_sr = read_audio(clean_path)
        transcript = str(pair["transcript"])
        entries = []
        variants = [
            ("Clean source (16 kHz)", clean, clean_sr, "clean",
             "fullband source, not telephone", str(clean_path)),
        ]
        legacy_paths = pair["legacy"]
        for profile in ("mobile", "pstn"):
            if profile not in legacy_paths:
                continue
            legacy_path = Path(legacy_paths[profile])
            legacy, legacy_sr = read_audio(legacy_path)
            variants.append((
                f"Legacy synth {profile.upper()}", legacy, legacy_sr,
                f"legacy_{profile}", "older synth output", str(legacy_path),
            ))
        for profile in ("mobile", "pstn", "gated"):
            sim = simulate_telephone(
                clean, sr=16000, profile=profile, seed=args.seed,
                use_ffmpeg=args.use_ffmpeg, output_sr=8000,
            )
            sim = resample(sim, 8000, 16000)
            variants.append((
                f"Simulated {profile}", sim, 16000, f"sim_{profile}",
                f"seed {args.seed}", "simulator",
            ))

        for label, audio, sr, condition, note, source in variants:
            audio, gain_db = level_match(audio, sr)
            path = out_dir / "paired" / f"{base}__{condition}.wav"
            write_wav(path, audio, sr)
            entry = {
                "label": label,
                "condition": condition,
                "path": str(path),
                "sr": sr,
                "duration_s": round(len(audio) / sr, 3),
                "gain_db": round(gain_db, 2),
                "note": note,
                "source": source,
            }
            entries.append(entry)
            manifest["paired"].append({**entry, "base": base,
                                       "transcript": transcript})
        cards.append({
            "base": base,
            "transcript": transcript,
            "entries": entries,
        })

    gated_real = []
    for index, path in enumerate(load_gated_rows(
        args.features_csv, args.call_dir, args.n_gated
    ), 1):
        audio, sr = read_audio(path)
        audio = densest_window(audio, sr, args.gated_seconds)
        audio, gain_db = level_match(audio, sr)
        out = out_dir / "gated_real" / f"real_gated_{index:02d}.wav"
        write_wav(out, audio, sr)
        entry = {
            "label": f"Real gated {index:02d}",
            "path": str(out),
            "sr": sr,
            "duration_s": round(len(audio) / sr, 3),
            "gain_db": round(gain_db, 2),
            "source": str(path),
            "note": "real gated call; mostly mobile-originated; no transcript",
        }
        gated_real.append(entry)
        manifest["gated_real"].append(entry)

    other_clips = [
        path for path in sorted(args.clips_dir.glob("spk_*/*.wav"))
        if f"{path.parent.name}_{path.stem}" not in paired_bases
    ]
    by_speaker: Dict[str, List[Path]] = {}
    for path in other_clips:
        by_speaker.setdefault(path.parent.name, []).append(path)
    selected = [paths[0] for _, paths in sorted(by_speaker.items())]
    selected += [path for path in other_clips if path not in selected]
    gated_sim = []
    for index, path in enumerate(selected[: args.n_gated], 1):
        audio, sr = read_audio(path)
        sim = simulate_telephone(
            audio, sr=sr, profile="gated", seed=args.seed,
            use_ffmpeg=args.use_ffmpeg, output_sr=8000,
        )
        sim, gain_db = level_match(sim, 8000)
        out = out_dir / "gated_sim" / f"sim_gated_{index:02d}.wav"
        write_wav(out, sim, 8000)
        entry = {
            "label": f"Simulated gated {index:02d}",
            "path": str(out),
            "sr": 8000,
            "duration_s": round(len(sim) / 8000, 3),
            "gain_db": round(gain_db, 2),
            "source": str(path),
            "note": "different speaker/sentence from the real gated set",
        }
        gated_sim.append(entry)
        manifest["gated_sim"].append(entry)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "index.html").write_text(
        render_html(cards, gated_real, gated_sim, out_dir), encoding="utf-8"
    )
    readme = [
        "# Telephone simulator listening kit",
        "",
        "Open `index.html` in a browser. All clips are level-matched on",
        f"active speech to {TARGET_ACTIVE_DB:.0f} dBFS.",
        "",
        "- `paired/`: same-sentence clean, legacy synth mobile/PSTN, current",
        "  synth mobile/PSTN/gated. This compares current vs legacy synth, not",
        "  real networks.",
        "- `gated_real/`: real gated call-center excerpts, mostly mobile-originated\n  (no transcripts).",
        "- `gated_sim/`: simulated gated clips with different content.",
        "",
        "The gated section is perceptual only; it is not a WER comparison.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Wrote {out_dir / 'index.html'}")
    print(f"  paired cards: {len(cards)}")
    print(f"  real gated excerpts: {len(gated_real)}")
    print(f"  simulated gated clips: {len(gated_sim)}")
    print(f"  manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
