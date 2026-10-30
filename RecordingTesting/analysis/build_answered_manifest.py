#!/usr/bin/env python3
"""Build the answered-customer manifest from committed AST label tables.

The output is used only for offline calibration/validation; it is not read by
the runtime telephone simulator.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from caller_class_check import classify_caller  # noqa: E402

COLUMNS = [
    "path", "file", "source", "caller_class", "is_gated", "content_class",
    "duration_s", "speech_rms_db", "noise_rms_db", "hf_1_2k_rel_db",
    "hf_2_3k_rel_db", "spec_centroid", "spec_roll95", "calibration",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def metric(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return "" if value is None else str(value)


def main() -> None:
    in_repo_labels = read_csv(HERE / "ast_labels_in_repo.csv")
    outside_labels = read_csv(HERE / "ast_labels_outside.csv")

    features = {
        row["file"]: row
        for row in read_csv(HERE / "features_files.csv")
        if row.get("group") == "real"
    }
    outside_features = {
        row["path"]: row
        for row in read_csv(HERE / "outside_spectral_features.csv")
    }

    rows: list[dict[str, str]] = []
    for label in in_repo_labels:
        name = Path(label["path"]).name
        feature = features.get(name, {})
        caller_class = classify_caller(name)
        is_gated = as_bool(feature.get("is_gated", ""))
        content_class = label.get("content_class", "")
        calibration = (
            content_class == "speech_only"
            and is_gated
            and caller_class == "mobile"
        )
        rows.append({
            "path": name,
            "file": name,
            "source": "in_repo",
            "caller_class": caller_class,
            "is_gated": str(is_gated),
            "content_class": content_class,
            "duration_s": metric(feature, "duration_s"),
            "speech_rms_db": metric(feature, "speech_rms_db"),
            "noise_rms_db": metric(feature, "noise_rms_db"),
            "hf_1_2k_rel_db": metric(feature, "hf_1_2k_rel_db"),
            "hf_2_3k_rel_db": metric(feature, "hf_2_3k_rel_db"),
            "spec_centroid": metric(feature, "spec_centroid"),
            "spec_roll95": metric(feature, "spec_roll95"),
            "calibration": str(calibration),
        })

    for label in outside_labels:
        name = Path(label["path"]).name
        feature = outside_features.get(label["path"], label)
        caller_class = label.get("caller_class", "")
        is_gated = as_bool(label.get("is_gated", ""))
        content_class = label.get("content_class", "")
        calibration = (
            content_class == "speech_only"
            and is_gated
            and caller_class == "mobile"
        )
        rows.append({
            "path": label["path"],
            "file": name,
            "source": "outside",
            "caller_class": caller_class,
            "is_gated": str(is_gated),
            "content_class": content_class,
            "duration_s": metric(feature, "duration_s"),
            "speech_rms_db": metric(feature, "speech_rms_db"),
            "noise_rms_db": metric(feature, "noise_rms_db"),
            "hf_1_2k_rel_db": metric(feature, "hf_1_2k_rel_db"),
            "hf_2_3k_rel_db": metric(feature, "hf_2_3k_rel_db"),
            "spec_centroid": metric(feature, "spec_centroid"),
            "spec_roll95": metric(feature, "spec_roll95"),
            "calibration": str(calibration),
        })

    out = HERE / "answered_customer_manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_calib = sum(r["calibration"] == "True" for r in rows)
    print(f"Wrote {out} ({len(rows)} rows, {n_calib} calibration)")


if __name__ == "__main__":
    main()
