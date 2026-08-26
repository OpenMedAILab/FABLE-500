#!/usr/bin/env python3
"""Analyze FABLE-500 API-based diagnosis benchmark predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
DEFAULT_PREDICTIONS = DATASET_DIR / "benchmark" / "outputs" / "api_benchmark_predictions.jsonl"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "benchmark" / "analysis" / "api_benchmark"

LABELS = [
    "Cataract",
    "Vitreous hemorrhage",
    "High myopia",
    "Refractive error",
    "Retinal detachment",
]

SYSTEM_ORDER = [
    "report_text_only",
    "fundus_only",
    "bscan_only",
    "fundus_bscan_image_only",
    "full_case_multimodal",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_valid_predictions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row.get("case_id", ""), row.get("system_id", ""))
        if row.get("status") in {"success", "parse_error"}:
            latest[key] = row
    return list(latest.values())


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    values = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        values.append(f1_from_counts(tp, fp, fn))
    return sum(values) / len(values)


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    if not y_true:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        vals.append(accuracy(yt, yp) if metric == "accuracy" else macro_f1(yt, yp))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return lo, hi


def paired_bootstrap_delta_ci(
    y_true: Sequence[str],
    y_a: Sequence[str],
    y_b: Sequence[str],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(y_true)
    if metric == "accuracy":
        obs = accuracy(y_true, y_a) - accuracy(y_true, y_b)
    else:
        obs = macro_f1(y_true, y_a) - macro_f1(y_true, y_b)
    vals = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idxs]
        ya = [y_a[i] for i in idxs]
        yb = [y_b[i] for i in idxs]
        val = accuracy(yt, ya) - accuracy(yt, yb) if metric == "accuracy" else macro_f1(yt, ya) - macro_f1(yt, yb)
        vals.append(val)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return obs, lo, hi


def mcnemar_exact_pvalue(y_true: Sequence[str], y_a: Sequence[str], y_b: Sequence[str]) -> float:
    b = sum((a == t) and (bb != t) for t, a, bb in zip(y_true, y_a, y_b))
    c = sum((a != t) and (bb == t) for t, a, bb in zip(y_true, y_a, y_b))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def per_class_rows(y_true: Sequence[str], y_pred: Sequence[str], system_id: str, system_name: str) -> List[Dict[str, Any]]:
    rows = []
    n = len(y_true)
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        tn = n - tp - fp - fn
        support = sum(t == label for t in y_true)
        rows.append(
            {
                "system_id": system_id,
                "system_name": system_name,
                "class": label,
                "support": support,
                "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "f1": f1_from_counts(tp, fp, fn),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def confusion_rows(y_true: Sequence[str], y_pred: Sequence[str], system_id: str, system_name: str) -> List[Dict[str, Any]]:
    rows = []
    for true_label in LABELS:
        for pred_label in LABELS:
            rows.append(
                {
                    "system_id": system_id,
                    "system_name": system_name,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "n": sum(t == true_label and p == pred_label for t, p in zip(y_true, y_pred)),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--reference-system", default="full_case_multimodal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = latest_valid_predictions(read_jsonl(args.predictions))
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
    for system_id in SYSTEM_ORDER:
        system_rows = sorted(by_system.get(system_id, []), key=lambda r: r["case_id"])
        if not system_rows:
            continue
        y_true = [r["reference_diagnosis"] for r in system_rows]
        y_pred = [r["predicted_diagnosis"] for r in system_rows]
        system_name = system_rows[0].get("system_name", system_id)
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
                "providers": "; ".join(f"{k}:{v}" for k, v in sorted(Counter(r.get("provider", "") for r in system_rows).items())),
                "models": "; ".join(f"{k}:{v}" for k, v in sorted(Counter(r.get("model", "") for r in system_rows).items())),
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

    xlsx = args.output_dir / "api_benchmark_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        paired_df.to_excel(writer, sheet_name="paired_comparisons", index=False)
        class_df.to_excel(writer, sheet_name="per_class", index=False)
        confusion_df.to_excel(writer, sheet_name="confusion_matrix_long", index=False)
        predictions_df.to_excel(writer, sheet_name="case_predictions", index=False)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    paired_df.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    class_df.to_csv(args.output_dir / "per_class.csv", index=False)
    confusion_df.to_csv(args.output_dir / "confusion_matrix_long.csv", index=False)
    predictions_df.to_csv(args.output_dir / "case_predictions.csv", index=False)
    print(summary_df.to_string(index=False))
    if not paired_df.empty:
        print("\nPaired comparisons:")
        print(paired_df.to_string(index=False))
    print(f"\nWrote {xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
