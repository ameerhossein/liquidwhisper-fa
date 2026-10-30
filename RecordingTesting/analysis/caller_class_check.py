#!/usr/bin/env python3
"""Compare real caller-number classes with the synthesized telephone profiles.

The first field of a real filename is the caller number:

    out-09134510792-102-20261022-112728-....wav

Iranian mobile prefixes (09xx) identify customer mobile calls; 3-digit values
are internal extensions; the rest are landline/international numbers.  This is
a proxy, not a carrier label, but it exposes a confound in the corpus: most
mobile-originated calls are in the gated/clean acoustic subset, while the
"noisy line" subset is mostly internal extensions.

Medians are read from the committed feature tables; no audio is decoded.

Usage:
  python RecordingTesting/analysis/caller_class_check.py
  python RecordingTesting/analysis/caller_class_check.py --out-dir /tmp/opencode/caller_check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent

# Iranian mobile national significant numbers after dropping 0 / +98.
MOBILE_PREFIXES = {
    "901", "902", "903", "904", "905",
    "910", "911", "912", "913", "914", "915", "916", "917", "918", "919",
    "920", "921", "922",
    "930", "931", "932", "933", "934", "935", "936", "937", "938", "939",
    "990", "991", "992", "993", "994", "995",
}

REAL_METRICS = [
    "speech_rms_db", "noise_rms_db", "speech_frac",
    "hf_1_2k_rel_db", "hf_2_3k_rel_db", "spec_centroid", "spec_roll95",
]
SYNTH_METRICS = [f"out_{key}" for key in REAL_METRICS]
OUTPUT_METRICS = REAL_METRICS


def classify_caller(filename: str) -> str:
    match = re.match(r"out-(.+?)-(\d{3})-", filename)
    if not match:
        return "other"
    number = match.group(1)
    if number.startswith("00"):
        number = number[2:]
    elif number.startswith("0"):
        number = number[1:]
    if number.startswith("98"):
        number = number[2:]
    if len(number) >= 10 and number[:3] in MOBILE_PREFIXES:
        return "mobile"
    if len(number) <= 4:
        return "extension"
    if len(number) >= 10:
        return "landline"
    return "other"


def group_rows(rows: Iterable[Dict[str, str]],
               selector) -> List[Dict[str, str]]:
    return [row for row in rows if selector(row)]


def medians(rows: Sequence[Dict[str, str]], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        values = [
            float(row[key]) for row in rows
            if row.get(key) not in (None, "", "nan", "NaN")
        ]
        out[OUTPUT_METRICS[keys.index(key)]] = (
            round(median(values), 2) if values else float("nan")
        )
    return out


def summarize_real(rows: Sequence[Dict[str, str]], caller_class: str):
    selected = [row for row in rows if classify_caller(row["file"]) == caller_class]
    gated = sum(str(row.get("is_gated", "")).lower() == "true" for row in selected)
    result = {
        "group": f"real_{caller_class}",
        "n": len(selected),
        "gated": gated,
        "noisy": len(selected) - gated,
    }
    result.update(medians(selected, REAL_METRICS))
    return result


def summarize_synth(rows: Sequence[Dict[str, str]], profile: str):
    selected = [row for row in rows if row["profile"] == profile]
    result = {
        "group": f"current_{profile}",
        "n": len(selected),
        "gated": "",
        "noisy": "",
    }
    result.update(medians(selected, SYNTH_METRICS))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=HERE / "features_files.csv"
    )
    parser.add_argument(
        "--multi-voice", type=Path, default=HERE / "multi_voice_files.csv"
    )
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    with open(args.features, newline="", encoding="utf-8") as fh:
        features = [
            row for row in csv.DictReader(fh) if row.get("group") == "real"
        ]
    with open(args.multi_voice, newline="", encoding="utf-8") as fh:
        multi_voice = list(csv.DictReader(fh))

    results = [
        summarize_real(features, "mobile"),
        summarize_real(features, "extension"),
        summarize_real(features, "landline"),
        summarize_real(features, "other"),
        summarize_synth(multi_voice, "gated"),
        summarize_synth(multi_voice, "mobile"),
        summarize_synth(multi_voice, "pstn"),
    ]
    results = [row for row in results if row["n"]]

    columns = [
        "group", "n", "gated", "noisy", *OUTPUT_METRICS,
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "caller_class_medians.csv", "w",
              newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    (args.out_dir / "caller_class_medians.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = (
        f"{'group':<20} {'n':>4} {'gated':>6} {'noisy':>6} "
        f"{'speech':>8} {'noise':>8} {'hf1-2k':>8} {'hf2-3k':>8} "
        f"{'centroid':>9} {'roll95':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['group']:<20} {row['n']:>4} {row['gated']!s:>6} "
            f"{row['noisy']!s:>6} {row['speech_rms_db']:>8.1f} "
            f"{row['noise_rms_db']:>8.1f} {row['hf_1_2k_rel_db']:>8.1f} "
            f"{row['hf_2_3k_rel_db']:>8.1f} {row['spec_centroid']:>9.1f} "
            f"{row['spec_roll95']:>8.1f}"
        )
    print(f"\nWrote {args.out_dir / 'caller_class_medians.csv'}")


if __name__ == "__main__":
    main()
