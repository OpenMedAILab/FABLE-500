#!/usr/bin/env python3
"""Run an API-based multimodal workflow benchmark for FABLE-500.

This runner evaluates a workflow-based baseline without retrieval, web search,
external guideline evidence, or fine-tuned specialist tools. It uses the same
released full-case inputs as the single-call full-case VLM baseline but
decomposes each case into:

1. fundus-image evidence extraction;
2. B-scan-image evidence extraction;
3. B-scan-report evidence extraction;
4. cross-modal adjudication from structured intermediate outputs only.

The final record is intentionally shaped like the single-call benchmark output
so that downstream metric scripts can consume it with minimal changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
SCRIPTS_DIR = DATASET_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_api_benchmark import (  # noqa: E402
    LABELS,
    RELEASE_ROOT,
    Provider,
    add_images,
    call_provider,
    default_providers,
    extract_json_object,
    extract_message_text,
    initialize_runtime_environment,
    normalize_label,
    parse_prediction,
    read_jsonl,
)


DEFAULT_INPUT = DATASET_DIR / "benchmark" / "inputs" / "test100_cases.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "benchmark" / "outputs" / "api_workflow_openrouter_gpt56sol.jsonl"

SYSTEM_ID = "api_agentic_full_case_workflow"
SYSTEM_NAME = "Four-stage workflow-based full-case comparator"

LABEL_TEXT = "\n".join(f"{i}. {label}" for i, label in enumerate(LABELS, 1))

FUNDUS_PROMPT = f"""You are an ophthalmic vision-language model module in a research benchmark.

Module task: extract evidence from de-identified ultra-widefield fundus images only.

Candidate diagnosis labels:
{LABEL_TEXT}

Use only the provided fundus images. Do not use external information, patient identifiers, filenames, folder names, B-scan images, report text, or diagnosis labels.

Return valid JSON using this schema:
{{
  "fundus_findings": "Concise fundus-image findings relevant to the candidate labels.",
  "candidate_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment | Insufficient evidence",
  "confidence": 0.0,
  "uncertainty_note": "Brief note about image quality, ambiguity, or missing evidence."
}}"""

BSCAN_IMAGE_PROMPT = f"""You are an ophthalmic vision-language model module in a research benchmark.

Module task: extract evidence from de-identified ophthalmic B-scan ultrasonography images only.

Candidate diagnosis labels:
{LABEL_TEXT}

Use only the provided B-scan images. Do not use external information, patient identifiers, filenames, folder names, fundus images, report text, or diagnosis labels.

Return valid JSON using this schema:
{{
  "bscan_image_findings": "Concise B-scan image findings relevant to the candidate labels.",
  "candidate_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment | Insufficient evidence",
  "confidence": 0.0,
  "uncertainty_note": "Brief note about image quality, ambiguity, or missing evidence."
}}"""

BSCAN_REPORT_PROMPT = f"""You are an ophthalmic report-interpretation module in a research benchmark.

Module task: extract diagnosis-relevant evidence from the released B-scan finding and impression text only.

Candidate diagnosis labels:
{LABEL_TEXT}

Use only the provided B-scan finding and B-scan impression. Do not use external information, patient identifiers, filenames, folder names, images, or diagnosis labels.

B-scan finding:
__BSCAN_FINDING__

B-scan impression:
__BSCAN_IMPRESSION__

Return valid JSON using this schema:
{{
  "bscan_report_evidence": "Concise report-derived evidence relevant to the candidate labels.",
  "candidate_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment | Insufficient evidence",
  "confidence": 0.0,
  "uncertainty_note": "Brief note about ambiguity or missing evidence."
}}"""

ADJUDICATION_PROMPT = f"""You are the final cross-modal adjudication module in a research benchmark.

Task: classify the de-identified ophthalmic case into exactly one of the five candidate labels.

Candidate diagnosis labels:
{LABEL_TEXT}

You are given only structured intermediate outputs from three modules:
1. fundus-image evidence extraction;
2. B-scan-image evidence extraction;
3. B-scan-report evidence extraction.

Do not use external information, patient identifiers, filenames, folder names, original images, original report text, or diagnosis labels. Choose exactly one final label even when evidence is incomplete or conflicting.

Fundus module output:
__FUNDUS_OUTPUT__

B-scan image module output:
__BSCAN_IMAGE_OUTPUT__

B-scan report module output:
__BSCAN_REPORT_OUTPUT__

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation integrating only the structured intermediate outputs.",
  "cross_modal_consistency": "consistent | partially_consistent | conflicting | insufficient_evidence",
  "main_deciding_evidence": "fundus | bscan_image | bscan_report | combined | uncertain",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}"""


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


def existing_success_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("system_id") == SYSTEM_ID
                and row.get("status") == "success"
                and row.get("parse_status") in {"valid", "label_from_text", "label_from_truncated_json"}
                and row.get("predicted_diagnosis")
            ):
                done.add(str(row.get("case_id", "")))
    return done


def clamp_confidence(value: Any) -> float | None:
    try:
        conf = float(value)
        if conf > 1.0:
            conf /= 100.0
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return None


def parse_module_output(text: str, evidence_keys: Sequence[str]) -> Dict[str, Any]:
    parsed = extract_json_object(text)
    if parsed is None:
        return {
            "parse_status": "invalid_json",
            "candidate_diagnosis": None,
            "confidence": None,
            "evidence": "",
            "uncertainty_note": "",
            "parsed_json": None,
        }
    candidate = parsed.get("candidate_diagnosis")
    label = normalize_label(candidate, text) if str(candidate).strip().lower() != "insufficient evidence" else None
    evidence = ""
    for key in evidence_keys:
        if parsed.get(key):
            evidence = str(parsed.get(key))
            break
    return {
        "parse_status": "valid",
        "candidate_diagnosis": label or ("Insufficient evidence" if str(candidate).strip().lower() == "insufficient evidence" else None),
        "confidence": clamp_confidence(parsed.get("confidence")),
        "evidence": evidence,
        "uncertainty_note": str(parsed.get("uncertainty_note") or ""),
        "parsed_json": parsed,
    }


def call_with_providers(
    providers: Sequence[Provider],
    messages: List[Dict[str, Any]],
    uses_vision: bool,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> Tuple[Provider | None, Dict[str, Any] | None, str, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    for provider in providers:
        if uses_vision and not provider.supports_vision:
            continue
        try:
            response = call_provider(provider, messages, max_tokens, temperature, timeout, retries)
            return provider, response, extract_message_text(response), errors
        except Exception as exc:  # noqa: BLE001
            errors.append({"provider": provider.name, "model": provider.model, "error": str(exc)})
    return None, None, "", errors


def text_message(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def image_message(prompt: str, paths: Sequence[str], label: str, release_root: Path, max_images: int) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    add_images(content, paths, label, release_root, max_images)
    return [{"role": "user", "content": content}]


def run_one(
    case: Dict[str, Any],
    providers: Sequence[Provider],
    release_root: Path,
    max_images_per_modality: int,
    max_tokens_module: int,
    max_tokens_final: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    started = time.time()
    step_records: Dict[str, Any] = {}
    provider_errors: List[Dict[str, Any]] = []
    total_usage: Dict[str, int] = {}
    providers_used: List[str] = []
    models_used: List[str] = []

    def add_usage(response: Dict[str, Any] | None) -> None:
        if not response:
            return
        usage = response.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total_usage[key] = int(total_usage.get(key, 0) + value)

    steps = [
        (
            "fundus",
            image_message(
                FUNDUS_PROMPT,
                case.get("fundus_image_paths") or [],
                "Fundus",
                release_root,
                max_images_per_modality,
            ),
            True,
            ["fundus_findings"],
        ),
        (
            "bscan_image",
            image_message(
                BSCAN_IMAGE_PROMPT,
                case.get("bscan_image_paths") or [],
                "B-scan",
                release_root,
                max_images_per_modality,
            ),
            True,
            ["bscan_image_findings"],
        ),
        (
            "bscan_report",
            text_message(
                BSCAN_REPORT_PROMPT.replace(
                    "__BSCAN_FINDING__",
                    case.get("bscan_finding_en") or case.get("bscan_finding") or "",
                ).replace(
                    "__BSCAN_IMPRESSION__",
                    case.get("bscan_impression_en") or case.get("bscan_impression") or "",
                )
            ),
            False,
            ["bscan_report_evidence"],
        ),
    ]

    for step_id, messages, uses_vision, evidence_keys in steps:
        provider, response, raw_text, errors = call_with_providers(
            providers,
            messages,
            uses_vision,
            max_tokens_module,
            temperature,
            timeout,
            retries,
        )
        provider_errors.extend({"step": step_id, **err} for err in errors)
        if provider is None or response is None:
            return {
                "status": "error",
                "case_id": case["case_id"],
                "patient_id": case.get("patient_id"),
                "system_id": SYSTEM_ID,
                "system_name": SYSTEM_NAME,
                "reference_diagnosis": case.get("reference_diagnosis"),
                "runtime_seconds": round(time.time() - started, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "parse_status": f"{step_id}_provider_error",
                "predicted_diagnosis": None,
                "confidence": None,
                "provider_errors": provider_errors,
                "workflow_steps": step_records,
            }
        providers_used.append(provider.name)
        models_used.append(provider.model)
        add_usage(response)
        parsed = parse_module_output(raw_text, evidence_keys)
        step_records[step_id] = {
            "provider": provider.name,
            "model": provider.model,
            "parse_status": parsed["parse_status"],
            "candidate_diagnosis": parsed["candidate_diagnosis"],
            "confidence": parsed["confidence"],
            "evidence": parsed["evidence"],
            "uncertainty_note": parsed["uncertainty_note"],
            "raw_text": raw_text,
            "parsed_json": parsed["parsed_json"],
        }

    adjudication_prompt = (
        ADJUDICATION_PROMPT.replace(
            "__FUNDUS_OUTPUT__",
            json.dumps(step_records["fundus"]["parsed_json"] or step_records["fundus"], ensure_ascii=False),
        )
        .replace(
            "__BSCAN_IMAGE_OUTPUT__",
            json.dumps(
                step_records["bscan_image"]["parsed_json"] or step_records["bscan_image"],
                ensure_ascii=False,
            ),
        )
        .replace(
            "__BSCAN_REPORT_OUTPUT__",
            json.dumps(
                step_records["bscan_report"]["parsed_json"] or step_records["bscan_report"],
                ensure_ascii=False,
            ),
        )
    )
    provider, response, raw_text, errors = call_with_providers(
        providers,
        text_message(adjudication_prompt),
        False,
        max_tokens_final,
        temperature,
        timeout,
        retries,
    )
    provider_errors.extend({"step": "adjudication", **err} for err in errors)
    if provider is None or response is None:
        return {
            "status": "error",
            "case_id": case["case_id"],
            "patient_id": case.get("patient_id"),
            "system_id": SYSTEM_ID,
            "system_name": SYSTEM_NAME,
            "reference_diagnosis": case.get("reference_diagnosis"),
            "runtime_seconds": round(time.time() - started, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "parse_status": "adjudication_provider_error",
            "predicted_diagnosis": None,
            "confidence": None,
            "provider_errors": provider_errors,
            "workflow_steps": step_records,
        }
    providers_used.append(provider.name)
    models_used.append(provider.model)
    add_usage(response)
    parsed_final = parse_prediction(raw_text)
    final_json = parsed_final.get("parsed_json") or {}
    step_records["adjudication"] = {
        "provider": provider.name,
        "model": provider.model,
        "parse_status": parsed_final["parse_status"],
        "raw_text": raw_text,
        "parsed_json": parsed_final["parsed_json"],
    }

    return {
        "status": "success" if parsed_final["predicted_diagnosis"] else "parse_error",
        "case_id": case["case_id"],
        "patient_id": case.get("patient_id"),
        "system_id": SYSTEM_ID,
        "system_name": SYSTEM_NAME,
        "reference_diagnosis": case.get("reference_diagnosis"),
        "provider": "+".join(dict.fromkeys(providers_used)),
        "model": "+".join(dict.fromkeys(models_used)),
        "runtime_seconds": round(time.time() - started, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parse_status": parsed_final["parse_status"],
        "predicted_diagnosis": parsed_final["predicted_diagnosis"],
        "confidence": parsed_final["confidence"],
        "evidence_summary": parsed_final["evidence_summary"],
        "uncertainty_note": parsed_final["uncertainty_note"],
        "cross_modal_consistency": str(final_json.get("cross_modal_consistency") or ""),
        "main_deciding_evidence": str(final_json.get("main_deciding_evidence") or ""),
        "raw_text": raw_text,
        "parsed_json": parsed_final["parsed_json"],
        "workflow_steps": step_records,
        "usage": total_usage,
        "provider_errors_before_success": provider_errors,
        "n_api_calls": 4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument(
        "--provider-chain",
        default="openrouter_gpt56_sol",
        help=(
            "Provider ID for the reported run. Multiple comma-separated providers are rejected "
            "unless --allow-provider-fallback is set."
        ),
    )
    parser.add_argument(
        "--allow-provider-fallback",
        action="store_true",
        help="Allow a comma-separated provider fallback chain for exploratory runs only.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-images-per-modality", type=int, default=0, help="0 means use all released images.")
    parser.add_argument("--max-tokens-module", type=int, default=1600)
    parser.add_argument("--max-tokens-final", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Explicit HTTP(S) proxy; otherwise use FABLE500_PROXY_URL from .env or the process environment.",
    )
    parser.add_argument(
        "--preserve-process-env",
        action="store_true",
        help="Keep already-exported variables instead of letting the project .env override them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proxy_display = initialize_runtime_environment(
        proxy_url=args.proxy_url,
        preserve_process_env=args.preserve_process_env,
    )
    provider_map = default_providers()
    provider_ids = [p.strip() for p in args.provider_chain.split(",") if p.strip()]
    if len(provider_ids) > 1 and not args.allow_provider_fallback:
        raise SystemExit(
            "Multiple providers were requested. Reported benchmark runs should use one provider/model "
            "per output file. Use --allow-provider-fallback only for exploratory debugging."
        )
    providers = []
    for provider_id in provider_ids:
        if provider_id not in provider_map:
            raise SystemExit(f"Unknown provider: {provider_id}")
        provider = provider_map[provider_id]
        if os.environ.get(provider.api_key_env):
            providers.append(provider)
        else:
            print(f"[WARN] skipping {provider_id}: {provider.api_key_env} is not set", file=sys.stderr)
    if not providers:
        raise SystemExit("No usable providers in provider chain.")

    cases = read_jsonl(args.input)
    if args.limit > 0:
        cases = cases[: args.limit]
    done = set() if args.force else existing_success_case_ids(args.output)
    tasks = [case for case in cases if case["case_id"] not in done]
    print(
        f"Loaded {len(cases)} cases; {len(done)} already complete; running {len(tasks)} cases "
        f"with providers {[p.name for p in providers]}; proxy={proxy_display}"
    )
    lock = threading.Lock()

    def submit(case: Dict[str, Any]) -> Dict[str, Any]:
        return run_one(
            case,
            providers,
            args.release_root,
            args.max_images_per_modality,
            args.max_tokens_module,
            args.max_tokens_final,
            args.temperature,
            args.timeout,
            args.retries,
        )

    if args.workers <= 1:
        for idx, case in enumerate(tasks, 1):
            result = submit(case)
            append_jsonl(args.output, [result], lock)
            print(
                f"[{idx}/{len(tasks)}] {case['case_id']} {result.get('status')} "
                f"{result.get('predicted_diagnosis')} ref={case.get('reference_diagnosis')}"
            )
    else:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(submit, case): case for case in tasks}
            for idx, future in enumerate(futures.as_completed(future_map), 1):
                case = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "error",
                        "case_id": case["case_id"],
                        "patient_id": case.get("patient_id"),
                        "system_id": SYSTEM_ID,
                        "system_name": SYSTEM_NAME,
                        "reference_diagnosis": case.get("reference_diagnosis"),
                        "runtime_seconds": None,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "parse_status": "unhandled_exception",
                        "predicted_diagnosis": None,
                        "confidence": None,
                        "provider_errors": [{"error": repr(exc)}],
                    }
                append_jsonl(args.output, [result], lock)
                print(
                    f"[{idx}/{len(tasks)}] {case['case_id']} {result.get('status')} "
                    f"{result.get('predicted_diagnosis')} ref={case.get('reference_diagnosis')}"
                )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
