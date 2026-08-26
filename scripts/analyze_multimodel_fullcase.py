#!/usr/bin/env python3
"""Analyze FABLE-500 full-case benchmark outputs across multiple models."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
SCRIPTS_DIR = DATASET_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_api_benchmark import (  # noqa: E402
    LABELS,
    accuracy,
    bootstrap_ci,
    confusion_rows,
    macro_f1,
    per_class_rows,
)


DEFAULT_OUTPUT_DIR = DATASET_DIR / "benchmark" / "analysis" / "model_family_fullcase_test100"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def latest_by_case(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") in {"success", "parse_error", "error"}:
            latest[str(row.get("case_id", ""))] = row
    return list(latest.values())


def inferred_provider_model(row: dict[str, Any]) -> tuple[str, str]:
    provider = row.get("provider")
    model = row.get("model")
    if (not provider or not model) and isinstance(row.get("provider_errors"), list) and row["provider_errors"]:
        first_error = row["provider_errors"][0] or {}
        provider = provider or first_error.get("provider")
        model = model or first_error.get("model")
    return str(provider or "unknown_provider"), str(model or "unknown_model")


def model_key(row: dict[str, Any]) -> str:
    provider, model = inferred_provider_model(row)
    return f"{provider}::{model}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for path in args.predictions:
        if not path.exists():
            raise SystemExit(f"Missing prediction file: {path}")
        rows = read_jsonl(path)
        latest = latest_by_case(rows)
        all_rows.extend(latest)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_model.setdefault(model_key(row), []).append(row)

    summary_rows = []
    per_class_all = []
    confusion_all = []
    case_rows = []

    for key, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda r: str(r.get("case_id", "")))
        valid = [
            r
            for r in rows
            if r.get("parse_status") in {"valid", "label_from_text", "label_from_truncated_json"}
            and r.get("predicted_diagnosis")
        ]
        if not rows:
            continue
        provider, model = inferred_provider_model(valid[0] if valid else rows[0])
        display = f"{provider} / {model}"
        n_records = len(rows)
        n_valid = len(valid)
        if valid:
            y_true = [r["reference_diagnosis"] for r in valid]
            y_pred = [r["predicted_diagnosis"] for r in valid]
            acc = accuracy(y_true, y_pred)
            mf1 = macro_f1(y_true, y_pred)
            acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, "accuracy", args.bootstrap, args.seed)
            f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, "macro_f1", args.bootstrap, args.seed + 17)
            per_class_all.extend(per_class_rows(y_true, y_pred, key, display))
            confusion_all.extend(confusion_rows(y_true, y_pred, key, display))
        else:
            acc = mf1 = acc_lo = acc_hi = f1_lo = f1_hi = float("nan")

        summary_rows.append(
            {
                "model_key": key,
                "provider": provider,
                "model": model,
                "n_records": n_records,
                "n_valid": n_valid,
                "parse_rate": n_valid / n_records if n_records else float("nan"),
                "accuracy": acc,
                "accuracy_ci95_low": acc_lo,
                "accuracy_ci95_high": acc_hi,
                "macro_f1": mf1,
                "macro_f1_ci95_low": f1_lo,
                "macro_f1_ci95_high": f1_hi,
                "predicted_label_counts": json.dumps(
                    Counter(r.get("predicted_diagnosis") for r in valid),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        for r in rows:
            case_rows.append(
                {
                    "model_key": key,
                    "provider": inferred_provider_model(r)[0],
                    "model": inferred_provider_model(r)[1],
                    "case_id": r.get("case_id"),
                    "reference_diagnosis": r.get("reference_diagnosis"),
                    "predicted_diagnosis": r.get("predicted_diagnosis"),
                    "correct": r.get("reference_diagnosis") == r.get("predicted_diagnosis"),
                    "status": r.get("status"),
                    "parse_status": r.get("parse_status"),
                    "confidence": r.get("confidence"),
                    "source_file": r.get("_source_file"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_all)
    confusion_df = pd.DataFrame(confusion_all)
    cases_df = pd.DataFrame(case_rows)

    xlsx = args.output_dir / "current_model_fullcase_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_class_df.to_excel(writer, sheet_name="per_class", index=False)
        confusion_df.to_excel(writer, sheet_name="confusion_matrix_long", index=False)
        cases_df.to_excel(writer, sheet_name="case_predictions", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    per_class_df.to_csv(args.output_dir / "per_class.csv", index=False)
    confusion_df.to_csv(args.output_dir / "confusion_matrix_long.csv", index=False)
    cases_df.to_csv(args.output_dir / "case_predictions.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nWrote {xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
