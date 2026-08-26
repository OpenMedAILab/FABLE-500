#!/usr/bin/env python3
"""Export FABLE-500 benchmark input JSONL files from the public metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = [
    "Cataract",
    "Vitreous hemorrhage",
    "High myopia",
    "Refractive error",
    "Retinal detachment",
]


def split_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [p.strip() for p in str(value).split(";") if p.strip()]


def row_to_record(row: pd.Series) -> dict:
    fundus_paths = split_paths(row["fundus_image_paths"])
    bscan_paths = split_paths(row["bscan_image_paths"])
    diagnosis = row.get("reference_diagnosis", row.get("diagnosis_en", ""))
    diagnosis_cn = row.get("diagnosis", row.get("reference_diagnosis_cn", ""))
    sex = row.get("sex", "")
    sex_en = row.get("sex_en", sex)
    age = row.get("age_years", row.get("age", None))
    same_day = row.get("same_day_fundus_bscan_pair", row.get("same_day_fundus_bscan", True))
    finding = row.get("bscan_finding", "")
    impression = row.get("bscan_impression", "")
    finding_en = row.get("bscan_finding_en", finding)
    impression_en = row.get("bscan_impression_en", impression)
    return {
        "case_id": str(row["case_id"]),
        "patient_id": str(row["patient_id"]),
        "split": str(row["split"]),
        "reference_diagnosis": str(diagnosis),
        "reference_diagnosis_en": str(diagnosis),
        "reference_diagnosis_cn": "" if pd.isna(diagnosis_cn) else str(diagnosis_cn),
        "candidate_labels": LABELS,
        "age": int(age) if pd.notna(age) else None,
        "sex": str(sex),
        "sex_en": "" if pd.isna(sex_en) else str(sex_en),
        "same_day_fundus_bscan": bool(same_day),
        "fundus_image_paths": fundus_paths,
        "bscan_image_paths": bscan_paths,
        "n_fundus_images": int(row["n_fundus_images"]),
        "n_bscan_images": int(row["n_bscan_images"]),
        "bscan_finding": "" if pd.isna(finding) else str(finding),
        "bscan_finding_en": "" if pd.isna(finding_en) else str(finding_en),
        "bscan_impression": "" if pd.isna(impression) else str(impression),
        "bscan_impression_en": "" if pd.isna(impression_en) else str(impression_en),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_records(records: list[dict], release_root: Path) -> dict:
    missing_paths: list[str] = []
    invalid_labels: list[str] = []
    count_mismatch: list[str] = []

    for rec in records:
        if rec["reference_diagnosis"] not in LABELS:
            invalid_labels.append(rec["case_id"])
        if len(rec["fundus_image_paths"]) != rec["n_fundus_images"]:
            count_mismatch.append(f"{rec['case_id']}:fundus")
        if len(rec["bscan_image_paths"]) != rec["n_bscan_images"]:
            count_mismatch.append(f"{rec['case_id']}:bscan")
        for rel in rec["fundus_image_paths"] + rec["bscan_image_paths"]:
            if not (release_root / rel).exists():
                missing_paths.append(f"{rec['case_id']}::{rel}")

    return {
        "n_records": len(records),
        "missing_paths": missing_paths,
        "invalid_labels": invalid_labels,
        "count_mismatch": count_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=REPO_ROOT / "data" / "FABLE-500",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Default: <release-root>/FABLE-500_metadata.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark" / "inputs",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    metadata = args.metadata or (args.release_root / "FABLE-500_metadata.xlsx")
    df = pd.read_excel(metadata, sheet_name="cases")

    test = df[df["split"] == "test"].copy()
    if len(test) != 100:
        raise ValueError(f"Expected 100 test cases, found {len(test)}")

    diagnosis_col = "reference_diagnosis" if "reference_diagnosis" in test.columns else "diagnosis_en"
    per_class = test[diagnosis_col].value_counts().to_dict()
    expected = {label: 20 for label in LABELS}
    if per_class != expected:
        raise ValueError(f"Expected 20 test cases per class, found {per_class}")

    smoke_parts = [
        test[test["diagnosis_en"] == label].sample(n=2, random_state=args.seed)
        if "diagnosis_en" in test.columns
        else test[test["reference_diagnosis"] == label].sample(n=2, random_state=args.seed)
        for label in LABELS
    ]
    smoke = pd.concat(smoke_parts, axis=0).sort_values([diagnosis_col, "case_id"]).reset_index(drop=True)
    test = test.sort_values([diagnosis_col, "case_id"]).reset_index(drop=True)

    smoke_records = [row_to_record(row) for _, row in smoke.iterrows()]
    test_records = [row_to_record(row) for _, row in test.iterrows()]

    smoke_path = args.output_dir / "smoke10_cases.jsonl"
    test_path = args.output_dir / "test100_cases.jsonl"
    write_jsonl(smoke_records, smoke_path)
    write_jsonl(test_records, test_path)

    summary = {
        "dataset_version": "FABLE-500 v1.0",
        "metadata": str(metadata),
        "release_root": str(args.release_root),
        "seed": args.seed,
        "candidate_labels": LABELS,
        "smoke10": validate_records(smoke_records, args.release_root),
        "test100": validate_records(test_records, args.release_root),
        "test_class_counts": per_class,
        "outputs": {
            "smoke10_cases": str(smoke_path),
            "test100_cases": str(test_path),
        },
    }
    summary_path = args.output_dir / "benchmark_input_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    errors = (
        summary["smoke10"]["missing_paths"]
        + summary["smoke10"]["invalid_labels"]
        + summary["smoke10"]["count_mismatch"]
        + summary["test100"]["missing_paths"]
        + summary["test100"]["invalid_labels"]
        + summary["test100"]["count_mismatch"]
    )
    if errors:
        raise RuntimeError(f"Benchmark input validation failed; see {summary_path}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
