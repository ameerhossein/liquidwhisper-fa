#!/usr/bin/env python3
"""Compare the final customer shape with the Outside gated customer subset.

The 31 Outside files are all in the 67-file calibration set, so this is an
in-sample check. It reads the committed band table and the final synth band
table; no audio is decoded.

Usage:
  python RecordingTesting/analysis/outside_customer_check.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "answered_customer_manifest.csv"
BANDS = HERE / "answered_customer_bands.csv"
SYNTH_BANDS = HERE / "customer_octave_levels.csv"
MULTI = HERE / "multi_voice_files.csv"
OUT = HERE / "outside_customer_check.csv"
OUT_JSON = HERE / "outside_customer_check.json"

BAND_COLUMNS = [
    "hz_315", "hz_400", "hz_500", "hz_630", "hz_800",
    "hz_1250", "hz_1600", "hz_2000", "hz_2500", "hz_3150",
]
METRICS = [
    "speech_rms_db", "noise_rms_db", "hf_1_2k_rel_db",
    "hf_2_3k_rel_db", "spec_centroid", "spec_roll95", "speech_frac",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    manifest = [
        row for row in read_csv(MANIFEST)
        if row["source"] == "outside"
        and row["caller_class"] == "mobile"
        and row["content_class"] == "speech_only"
        and str(row["is_gated"]).lower() == "true"
    ]
    outside_bands = [
        row for row in read_csv(BANDS) if row["source"] == "outside"
    ]
    synth_bands = {
        row["hz"]: float(row["synth_db_re_1k"])
        for row in read_csv(SYNTH_BANDS)
    }
    synth_rows = [
        row for row in read_csv(MULTI) if row["profile"] == "customer"
    ]
    outside_features = {
        row["path"]: row for row in read_csv(HERE / "outside_spectral_features.csv")
    }

    real_shape = {
        col: median(
            float(row[col]) - float(row["hz_1000"]) for row in outside_bands
        )
        for col in BAND_COLUMNS
    }

    rows = []
    for col in BAND_COLUMNS:
        hz = str(int(col.removeprefix("hz_")))
        real = real_shape[col]
        synth = synth_bands[hz]
        rows.append({
            "hz": hz,
            "outside_db_re_1k": round(real, 3),
            "customer_db_re_1k": round(synth, 3),
            "diff_db": round(synth - real, 3),
        })

    broadband = {}
    for metric in METRICS:
        real_vals = []
        for row in manifest:
            value = row.get(metric)
            if value in (None, "", "nan", "NaN"):
                value = outside_features.get(row["path"], {}).get(metric)
            if value not in (None, "", "nan", "NaN"):
                real_vals.append(float(value))
        synth_vals = [
            float(row[f"out_{metric}"]) for row in synth_rows
            if row.get(f"out_{metric}") not in (None, "", "nan", "NaN")
        ]
        broadband[metric] = {
            "outside_median": round(median(real_vals), 3),
            "customer_median": round(median(synth_vals), 3),
            "diff_db" if metric.endswith("_db") else "diff": round(
                median(synth_vals) - median(real_vals), 3
            ),
        }

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "n_outside": len(manifest),
        "note": "all 31 are in the 67-file calibration set; in-sample check",
        "bands": rows,
        "broadband": broadband,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Outside gated mobile speech-only: {len(manifest)} (in-sample)")
    print("Hz   outside  customer  diff")
    for row in rows:
        print(
            f"{row['hz']:>4} {row['outside_db_re_1k']:8.2f} "
            f"{row['customer_db_re_1k']:9.2f} {row['diff_db']:+7.2f}"
        )
    print("\nBroadband medians:")
    for metric, values in broadband.items():
        key = "diff_db" if metric.endswith("_db") else "diff"
        print(
            f"  {metric:20} outside={values['outside_median']:8.2f} "
            f"customer={values['customer_median']:8.2f} "
            f"diff={values[key]:+7.2f}"
        )
    print(f"\nWrote {OUT}\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
