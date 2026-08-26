#!/usr/bin/env python3
"""Create FABLE-500 benchmark inputs with diagnosis terms masked in report text.

The masked inputs preserve image paths, public identifiers, splits, and
reference labels. Only direct disease-label terms in the released B-scan finding
and impression fields are replaced with a neutral placeholder.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATASET_DIR / "benchmark" / "inputs" / "test100_cases.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "benchmark" / "inputs" / "test100_cases_masked_report.jsonl"
DEFAULT_SUMMARY = DATASET_DIR / "benchmark" / "inputs" / "masked_report_export_summary.json"

REPORT_FIELDS = [
    "bscan_finding",
    "bscan_finding_en",
    "bscan_impression",
    "bscan_impression_en",
]

DIAGNOSIS_PATTERNS = [
    r"\bcataracts?\b",
    r"\bvitreous\s+h[ae]morrhage\b",
    r"\bhigh\s+myopia\b",
    r"\bpatholog(?:ic|ical)\s+myopia\b",
    r"\brefractive\s+error\b",
    r"\bametropia\b",
    r"\bretinal\s+detachment\b",
    "白内障",
    "玻璃体积血",
    "玻璃体出血",
    "高度近视",
    "病理性近视",
    "屈光不正",
    "视网膜脱离",
]

PLACEHOLDER = "[MASKED_DIAGNOSIS_TERM]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mask_text(text: str) -> tuple[str, int]:
    masked = text
    n = 0
    for pattern in DIAGNOSIS_PATTERNS:
        masked, count = re.subn(pattern, PLACEHOLDER, masked, flags=re.IGNORECASE)
        n += count
    return masked, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out_rows: list[dict[str, Any]] = []
    field_counts = {field: 0 for field in REPORT_FIELDS}
    case_counts: dict[str, int] = {}

    for row in rows:
        out = dict(row)
        case_total = 0
        for field in REPORT_FIELDS:
            value = "" if out.get(field) is None else str(out.get(field))
            masked, count = mask_text(value)
            out[field] = masked
            field_counts[field] += count
            case_total += count
        if case_total:
            case_counts[str(out.get("case_id", ""))] = case_total
        out_rows.append(out)

    write_jsonl(out_rows, args.output)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "n_cases": len(rows),
        "n_cases_with_masked_terms": len(case_counts),
        "n_masked_terms_total": sum(field_counts.values()),
        "masked_terms_by_field": field_counts,
        "placeholder": PLACEHOLDER,
        "case_mask_counts": case_counts,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
