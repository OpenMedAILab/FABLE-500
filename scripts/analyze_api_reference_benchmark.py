#!/usr/bin/env python3
"""Analyze FABLE-500 API-based reference benchmark predictions.

This script combines the five single-call API input settings with the optional
four-stage workflow-based full-case comparator output and computes the same metrics used
in the manuscript tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    mcnemar_exact_pvalue,
    paired_bootstrap_delta_ci,
    per_class_rows,
)


DEFAULT_SINGLE_CALL = DATASET_DIR / "benchmark" / "outputs" / "test100_ablation_openrouter_gpt56sol.jsonl"
DEFAULT_WORKFLOW = DATASET_DIR / "benchmark" / "outputs" / "api_workflow_openrouter_gpt56sol.jsonl"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "benchmark" / "analysis" / "api_reference_test100"

SYSTEM_ORDER = [
    "report_text_only",
    "fundus_only",
    "bscan_only",
    "fundus_bscan_image_only",
    "full_case_multimodal",
    "api_agentic_full_case_workflow",
]

DISPLAY_NAMES = {
    "report_text_only": "Report-text only",
    "fundus_only": "Fundus only",
    "bscan_only": "B-scan only",
    "fundus_bscan_image_only": "Fundus+B-scan image only",
    "full_case_multimodal": "Full case multimodal",
    "api_agentic_full_case_workflow": "Four-stage workflow-based full-case comparator",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_valid_predictions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id", "")), str(row.get("system_id", "")))
        if row.get("status") in {"success", "parse_error"}:
            latest[key] = row
    return list(latest.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-call-predictions", type=Path, default=DEFAULT_SINGLE_CALL)
    parser.add_argument("--workflow-predictions", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--reference-system", default="full_case_multimodal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_predictions = args.workflow_predictions
    rows = latest_valid_predictions(read_jsonl(args.single_call_predictions) + read_jsonl(workflow_predictions))
    valid_rows = [
        r
        for r in rows
        if r.get("parse_status") in {"valid", "label_from_text", "label_from_truncated_json"}
        and r.get("predicted_diagnosis")
    ]
    by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_system[row["system_id"]].append(row)

    summary_rows = []
    class_rows = []
    confusion = []
    predictions_flat = []
    workflow_rows = []

    for system_id in SYSTEM_ORDER:
        system_rows = sorted(by_system.get(system_id, []), key=lambda r: r["case_id"])
        if not system_rows:
            continue
        y_true = [r["reference_diagnosis"] for r in system_rows]
        y_pred = [r["predicted_diagnosis"] for r in system_rows]
        system_name = DISPLAY_NAMES.get(system_id, system_rows[0].get("system_name", system_id))
        acc = accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, "accuracy", args.bootstrap, args.seed)
        f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, "macro_f1", args.bootstrap, args.seed + 17)
        parse_total = sum(1 for r in rows if r.get("system_id") == system_id)
        parse_valid = len(system_rows)
        summary_rows.append(
            {
                "system_id": system_id,
                "system_name": system_name,
                "n_valid": len(system_rows),
                "n_records": parse_total,
                "parse_rate": parse_valid / parse_total if parse_total else float("nan"),
                "accuracy": acc,
                "accuracy_ci95_low": acc_lo,
                "accuracy_ci95_high": acc_hi,
                "macro_f1": mf1,
                "macro_f1_ci95_low": f1_lo,
                "macro_f1_ci95_high": f1_hi,
                "providers": "; ".join(
                    f"{k}:{v}" for k, v in sorted(Counter(r.get("provider", "") for r in system_rows).items())
                ),
                "models": "; ".join(
                    f"{k}:{v}" for k, v in sorted(Counter(r.get("model", "") for r in system_rows).items())
                ),
                "mean_api_calls_per_case": sum(float(r.get("n_api_calls", 1) or 1) for r in system_rows)
                / len(system_rows),
            }
        )
        class_rows.extend(per_class_rows(y_true, y_pred, system_id, system_name))
        confusion.extend(confusion_rows(y_true, y_pred, system_id, system_name))
        for r in system_rows:
            predictions_flat.append(
                {
                    "case_id": r["case_id"],
                    "system_id": system_id,
                    "system_name": system_name,
                    "reference_diagnosis": r["reference_diagnosis"],
                    "predicted_diagnosis": r["predicted_diagnosis"],
                    "correct": r["reference_diagnosis"] == r["predicted_diagnosis"],
                    "confidence": r.get("confidence"),
                    "provider": r.get("provider"),
                    "model": r.get("model"),
                    "evidence_summary": r.get("evidence_summary"),
                    "uncertainty_note": r.get("uncertainty_note"),
                    "cross_modal_consistency": r.get("cross_modal_consistency"),
                    "main_deciding_evidence": r.get("main_deciding_evidence"),
                    "n_api_calls": r.get("n_api_calls", 1),
                }
            )
            if system_id == "api_agentic_full_case_workflow":
                workflow_rows.append(
                    {
                        "case_id": r["case_id"],
                        "reference_diagnosis": r["reference_diagnosis"],
                        "predicted_diagnosis": r["predicted_diagnosis"],
                        "correct": r["reference_diagnosis"] == r["predicted_diagnosis"],
                        "cross_modal_consistency": r.get("cross_modal_consistency"),
                        "main_deciding_evidence": r.get("main_deciding_evidence"),
                        "fundus_candidate": (r.get("workflow_steps") or {}).get("fundus", {}).get("candidate_diagnosis"),
                        "bscan_image_candidate": (r.get("workflow_steps") or {})
                        .get("bscan_image", {})
                        .get("candidate_diagnosis"),
                        "bscan_report_candidate": (r.get("workflow_steps") or {})
                        .get("bscan_report", {})
                        .get("candidate_diagnosis"),
                    }
                )

    paired_rows = []
    ref_rows = sorted(by_system.get(args.reference_system, []), key=lambda r: r["case_id"])
    ref_by_case = {r["case_id"]: r for r in ref_rows}
    for system_id in SYSTEM_ORDER:
        if system_id == args.reference_system:
            continue
        comp_rows = sorted(by_system.get(system_id, []), key=lambda r: r["case_id"])
        paired_cases = sorted(set(ref_by_case) & {r["case_id"] for r in comp_rows})
        if not paired_cases:
            continue
        comp_by_case = {r["case_id"]: r for r in comp_rows}
        y_true = [ref_by_case[c]["reference_diagnosis"] for c in paired_cases]
        y_ref = [ref_by_case[c]["predicted_diagnosis"] for c in paired_cases]
        y_comp = [comp_by_case[c]["predicted_diagnosis"] for c in paired_cases]
        acc_delta, acc_lo, acc_hi = paired_bootstrap_delta_ci(
            y_true, y_ref, y_comp, "accuracy", args.bootstrap, args.seed + 31
        )
        f1_delta, f1_lo, f1_hi = paired_bootstrap_delta_ci(
            y_true, y_ref, y_comp, "macro_f1", args.bootstrap, args.seed + 47
        )
        paired_rows.append(
            {
                "reference_system": args.reference_system,
                "comparison_system": system_id,
                "n_paired": len(paired_cases),
                "accuracy_delta_reference_minus_comparison": acc_delta,
                "accuracy_delta_ci95_low": acc_lo,
                "accuracy_delta_ci95_high": acc_hi,
                "accuracy_mcnemar_exact_p": mcnemar_exact_pvalue(y_true, y_ref, y_comp),
                "macro_f1_delta_reference_minus_comparison": f1_delta,
                "macro_f1_delta_ci95_low": f1_lo,
                "macro_f1_delta_ci95_high": f1_hi,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(class_rows)
    confusion_df = pd.DataFrame(confusion)
    paired_df = pd.DataFrame(paired_rows)
    predictions_df = pd.DataFrame(predictions_flat)
    workflow_df = pd.DataFrame(workflow_rows)

    xlsx = args.output_dir / "api_reference_benchmark_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        paired_df.to_excel(writer, sheet_name="paired_comparisons", index=False)
        class_df.to_excel(writer, sheet_name="per_class", index=False)
        confusion_df.to_excel(writer, sheet_name="confusion_matrix_long", index=False)
        predictions_df.to_excel(writer, sheet_name="case_predictions", index=False)
        workflow_df.to_excel(writer, sheet_name="workflow_details", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    paired_df.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    class_df.to_csv(args.output_dir / "per_class.csv", index=False)
    confusion_df.to_csv(args.output_dir / "confusion_matrix_long.csv", index=False)
    predictions_df.to_csv(args.output_dir / "case_predictions.csv", index=False)
    workflow_df.to_csv(args.output_dir / "workflow_details.csv", index=False)

    print(summary_df.to_string(index=False))
    if not paired_df.empty:
        print("\nPaired comparisons:")
        print(paired_df.to_string(index=False))
    print(f"\nWrote {xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
