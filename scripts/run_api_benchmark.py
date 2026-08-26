#!/usr/bin/env python3
"""Run API-based multimodal diagnosis benchmarks for FABLE-500.

The runner is intentionally conservative: it saves one JSONL record per
case/system call, keeps the raw provider response, and can resume without
repeating successful calls.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT
RELEASE_ROOT = Path(os.environ.get("FABLE500_RELEASE_ROOT", REPO_ROOT / "data" / "FABLE-500"))
DEFAULT_INPUT = DATASET_DIR / "benchmark" / "inputs" / "test100_cases.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "benchmark" / "outputs" / "api_benchmark_predictions.jsonl"


LABELS = [
    "Cataract",
    "Vitreous hemorrhage",
    "High myopia",
    "Refractive error",
    "Retinal detachment",
]

LABEL_ALIASES = {
    "cataract": "Cataract",
    "白内障": "Cataract",
    "vitreous hemorrhage": "Vitreous hemorrhage",
    "vitreous haemorrhage": "Vitreous hemorrhage",
    "玻璃体积血": "Vitreous hemorrhage",
    "玻璃体出血": "Vitreous hemorrhage",
    "high myopia": "High myopia",
    "pathologic myopia": "High myopia",
    "pathological myopia": "High myopia",
    "高度近视": "High myopia",
    "病理性近视": "High myopia",
    "refractive error": "Refractive error",
    "ametropia": "Refractive error",
    "屈光不正": "Refractive error",
    "retinal detachment": "Retinal detachment",
    "视网膜脱离": "Retinal detachment",
}


SYSTEMS = {
    "report_text_only": {
        "display_name": "Report-text only",
        "uses_fundus": False,
        "uses_bscan": False,
        "uses_report": True,
        "prompt": """You are evaluating a de-identified ophthalmic case for a research benchmark.

Task: classify the case into exactly one of the following five diagnostic categories:
1. Cataract
2. Vitreous hemorrhage
3. High myopia
4. Refractive error
5. Retinal detachment

Use only the provided B-scan finding and B-scan impression text. Do not use any external information, patient identifier, filename, or folder name.

B-scan finding:
{bscan_finding}

B-scan impression:
{bscan_impression}

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation based only on the provided B-scan report fields.",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}""",
    },
    "fundus_only": {
        "display_name": "Fundus only",
        "uses_fundus": True,
        "uses_bscan": False,
        "uses_report": False,
        "prompt": """You are evaluating de-identified ultra-widefield fundus images for a research benchmark.

Task: classify the case into exactly one of the following five diagnostic categories:
1. Cataract
2. Vitreous hemorrhage
3. High myopia
4. Refractive error
5. Retinal detachment

Use only the provided fundus images. Do not use any external information, patient identifier, filename, or folder name.

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation based only on the fundus images.",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}""",
    },
    "bscan_only": {
        "display_name": "B-scan only",
        "uses_fundus": False,
        "uses_bscan": True,
        "uses_report": False,
        "prompt": """You are evaluating de-identified ophthalmic B-scan ultrasonography images for a research benchmark.

Task: classify the case into exactly one of the following five diagnostic categories:
1. Cataract
2. Vitreous hemorrhage
3. High myopia
4. Refractive error
5. Retinal detachment

Use only the provided B-scan images. Do not use any external information, patient identifier, filename, or folder name.

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation based only on the B-scan images.",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}""",
    },
    "fundus_bscan_image_only": {
        "display_name": "Fundus+B-scan image only",
        "uses_fundus": True,
        "uses_bscan": True,
        "uses_report": False,
        "prompt": """You are evaluating a de-identified multimodal ophthalmic imaging case for a research benchmark.

Task: classify the case into exactly one of the following five diagnostic categories:
1. Cataract
2. Vitreous hemorrhage
3. High myopia
4. Refractive error
5. Retinal detachment

Use only the provided fundus and B-scan images. Do not use any external information, patient identifier, filename, folder name, or report text.

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation integrating only the provided fundus and B-scan images.",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}""",
    },
    "full_case_multimodal": {
        "display_name": "Full case multimodal",
        "uses_fundus": True,
        "uses_bscan": True,
        "uses_report": True,
        "prompt": """You are evaluating a de-identified multimodal ophthalmic case for a research benchmark.

Task: classify the case into exactly one of the following five diagnostic categories:
1. Cataract
2. Vitreous hemorrhage
3. High myopia
4. Refractive error
5. Retinal detachment

Use only the provided fundus images, B-scan images, B-scan finding text, and B-scan impression text. Do not use any external information, patient identifier, filename, or folder name.

B-scan finding:
{bscan_finding}

B-scan impression:
{bscan_impression}

Return valid JSON using this schema:
{{
  "predicted_diagnosis": "one of: Cataract | Vitreous hemorrhage | High myopia | Refractive error | Retinal detachment",
  "confidence": 0.0,
  "evidence_summary": "Brief explanation integrating the provided images and B-scan report fields.",
  "uncertainty_note": "Brief note if relevant; do not use this field to avoid choosing a label."
}}""",
    },
}


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    model: str
    supports_vision: bool = True
    response_format: bool = True

    @property
    def chat_url(self) -> str:
        url = self.base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        return key


def initialize_runtime_environment(
    proxy_url: Optional[str] = None,
    preserve_process_env: bool = False,
) -> str:
    """Load `.env` if present and configure provider networking."""

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if preserve_process_env and key in os.environ:
                continue
            os.environ[key] = value

    selected_proxy = (
        proxy_url
        or os.environ.get("FABLE500_PROXY_URL")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
        or ""
    )
    if selected_proxy:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ[key] = selected_proxy
    return effective_proxy_display()


def effective_proxy_display() -> str:
    return (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
        or ""
    )


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def default_providers() -> Dict[str, Provider]:
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    bailian_base = os.environ.get("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return {
        "openai_gpt56_sol": Provider(
            name="openai_gpt56_sol",
            base_url=openai_base,
            api_key_env="OPENAI_API_KEY",
            model=os.environ.get("FABLE500_OPENAI_MODEL", "gpt-5.6-sol"),
            supports_vision=True,
        ),
        "openrouter_gpt56_sol": Provider(
            name="openrouter_gpt56_sol",
            base_url=openrouter_base,
            api_key_env="OPENROUTER_API_KEY",
            model=os.environ.get("FABLE500_OPENROUTER_GPT56_SOL_MODEL", "openai/gpt-5.6-sol"),
            supports_vision=True,
        ),
        "openrouter_gemini37_flash": Provider(
            name="openrouter_gemini37_flash",
            base_url=openrouter_base,
            api_key_env="OPENROUTER_API_KEY",
            model=os.environ.get("FABLE500_OPENROUTER_GEMINI37_MODEL", "google/gemini-3.7-flash"),
            supports_vision=True,
        ),
        "openrouter_claude_sonnet5": Provider(
            name="openrouter_claude_sonnet5",
            base_url=openrouter_base,
            api_key_env="OPENROUTER_API_KEY",
            model=os.environ.get("FABLE500_OPENROUTER_CLAUDE_MODEL", "anthropic/claude-sonnet-5"),
            supports_vision=True,
        ),
        "bailian_qwen_vl": Provider(
            name="bailian_qwen_vl",
            base_url=bailian_base,
            api_key_env="BAILIAN_API_KEY",
            model=os.environ.get("FABLE500_BAILIAN_VISION_MODEL", "qwen3.7-plus"),
            supports_vision=True,
        ),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()


def existing_success_keys(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success" and row.get("parse_status") in {
                "valid",
                "label_from_text",
                "label_from_truncated_json",
            }:
                keys.add((row.get("case_id", ""), row.get("system_id", "")))
    return keys


def make_text_prompt(system_id: str, case: Dict[str, Any]) -> str:
    template = SYSTEMS[system_id]["prompt"]
    return template.format(
        bscan_finding=case.get("bscan_finding_en") or case.get("bscan_finding") or "",
        bscan_impression=case.get("bscan_impression_en") or case.get("bscan_impression") or "",
    )


def resolve_release_path(relative_path: str, release_root: Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return release_root / path


def add_images(
    content: List[Dict[str, Any]],
    paths: Sequence[str],
    label: str,
    release_root: Path,
    max_images: int,
) -> None:
    selected = list(paths)
    if max_images > 0:
        selected = selected[:max_images]
    for idx, relative in enumerate(selected, 1):
        path = resolve_release_path(relative, release_root)
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} image: {relative}")
        content.append({"type": "text", "text": f"{label} image {idx}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path), "detail": "auto"},
            }
        )


def build_messages(
    system_id: str,
    case: Dict[str, Any],
    release_root: Path,
    max_images_per_modality: int,
) -> List[Dict[str, Any]]:
    spec = SYSTEMS[system_id]
    prompt = make_text_prompt(system_id, case)
    if not spec["uses_fundus"] and not spec["uses_bscan"]:
        return [{"role": "user", "content": prompt}]
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if spec["uses_fundus"]:
        add_images(
            content,
            case.get("fundus_image_paths") or [],
            "Fundus",
            release_root,
            max_images_per_modality,
        )
    if spec["uses_bscan"]:
        add_images(
            content,
            case.get("bscan_image_paths") or [],
            "B-scan",
            release_root,
            max_images_per_modality,
        )
    return [{"role": "user", "content": content}]


def call_provider(
    provider: Provider,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider.response_format:
        payload["response_format"] = {"type": "json_object"}
    if "dashscope" in provider.base_url.lower() or "aliyuncs" in provider.base_url.lower():
        payload["enable_thinking"] = False

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter" in provider.base_url.lower():
        headers["HTTP-Referer"] = "https://github.com/OpenMedAILab/FABLE-500"
        headers["X-Title"] = "FABLE-500 benchmark"

    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(provider.chat_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail[:2000]}"
            if exc.code == 400 and payload.get("response_format"):
                payload.pop("response_format", None)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            retry_after = 0
            try:
                retry_after = int(exc.headers.get("Retry-After") or 0)
            except (TypeError, ValueError):
                retry_after = 0
            retryable_budget_402 = exc.code == 402 and "in_flight_budget_exhausted" in detail
            if attempt < retries and (exc.code == 429 or retryable_budget_402 or 500 <= exc.code < 600):
                time.sleep(max(3 * (attempt + 1), min(180, retry_after)))
                continue
        except urllib.error.URLError as exc:
            last_error = f"{exc!r}; effective_proxy={effective_proxy_display()}"
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
                continue
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
                continue
    raise RuntimeError(f"{provider.name} failed: {last_error}")


def extract_message_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict))
    return str(content)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(stripped)
    brace = re.search(r"\{.*\}", stripped, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def extract_label_field_from_text(text: str) -> Optional[str]:
    """Recover a label from truncated JSON when the diagnosis field is intact."""
    match = re.search(
        r'"(?:predicted_diagnosis|candidate_diagnosis)"\s*:\s*"([^"]+)"',
        text,
        flags=re.I,
    )
    if not match:
        return None
    return normalize_label(match.group(1), "")


def extract_confidence_field_from_text(text: str) -> Optional[float]:
    match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.I)
    if not match:
        return None
    try:
        confidence = float(match.group(1))
        if confidence > 1.0:
            confidence = confidence / 100.0
        return max(0.0, min(1.0, confidence))
    except ValueError:
        return None


def normalize_label(value: Any, raw_text: str = "") -> Optional[str]:
    if value is not None:
        text = str(value).strip()
        if text in LABELS:
            return text
        alias = LABEL_ALIASES.get(text.lower()) or LABEL_ALIASES.get(text)
        if alias:
            return alias
    lowered = raw_text.lower()
    hits = []
    for label in LABELS:
        if label.lower() in lowered:
            hits.append(label)
    for alias, label in LABEL_ALIASES.items():
        if alias.lower() in lowered:
            hits.append(label)
    unique = list(dict.fromkeys(hits))
    return unique[0] if len(unique) == 1 else None


def parse_prediction(text: str) -> Dict[str, Any]:
    parsed = extract_json_object(text)
    if parsed is None:
        label = extract_label_field_from_text(text) or normalize_label(None, text)
        return {
            "parse_status": "label_from_truncated_json" if label else "invalid_json",
            "predicted_diagnosis": label,
            "confidence": extract_confidence_field_from_text(text),
            "evidence_summary": "",
            "uncertainty_note": "",
            "parsed_json": None,
        }
    label = normalize_label(parsed.get("predicted_diagnosis"), text)
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = None
    return {
        "parse_status": "valid" if label else "invalid_label",
        "predicted_diagnosis": label,
        "confidence": confidence,
        "evidence_summary": str(parsed.get("evidence_summary") or ""),
        "uncertainty_note": str(parsed.get("uncertainty_note") or ""),
        "parsed_json": parsed,
    }


def run_one(
    case: Dict[str, Any],
    system_id: str,
    providers: Sequence[Provider],
    release_root: Path,
    max_images_per_modality: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> Dict[str, Any]:
    started = time.time()
    messages = build_messages(system_id, case, release_root, max_images_per_modality)
    uses_vision = SYSTEMS[system_id]["uses_fundus"] or SYSTEMS[system_id]["uses_bscan"]
    provider_errors = []
    for provider in providers:
        if uses_vision and not provider.supports_vision:
            continue
        try:
            response = call_provider(provider, messages, max_tokens, temperature, timeout, retries)
            raw_text = extract_message_text(response)
            parsed = parse_prediction(raw_text)
            return {
                "status": "success" if parsed["predicted_diagnosis"] else "parse_error",
                "case_id": case["case_id"],
                "patient_id": case.get("patient_id"),
                "system_id": system_id,
                "system_name": SYSTEMS[system_id]["display_name"],
                "reference_diagnosis": case.get("reference_diagnosis"),
                "provider": provider.name,
                "model": provider.model,
                "runtime_seconds": round(time.time() - started, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "parse_status": parsed["parse_status"],
                "predicted_diagnosis": parsed["predicted_diagnosis"],
                "confidence": parsed["confidence"],
                "evidence_summary": parsed["evidence_summary"],
                "uncertainty_note": parsed["uncertainty_note"],
                "raw_text": raw_text,
                "parsed_json": parsed["parsed_json"],
                "usage": response.get("usage") or {},
                "provider_errors_before_success": provider_errors,
            }
        except Exception as exc:  # noqa: BLE001
            provider_errors.append({"provider": provider.name, "model": provider.model, "error": str(exc)})
    return {
        "status": "error",
        "case_id": case["case_id"],
        "patient_id": case.get("patient_id"),
        "system_id": system_id,
        "system_name": SYSTEMS[system_id]["display_name"],
        "reference_diagnosis": case.get("reference_diagnosis"),
        "runtime_seconds": round(time.time() - started, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parse_status": "provider_error",
        "predicted_diagnosis": None,
        "confidence": None,
        "provider_errors": provider_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated system IDs or 'all'.")
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
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--proxy-url",
        default=None,
        help=(
            "Explicit HTTP(S) proxy for provider calls. If omitted, uses "
            "FABLE500_PROXY_URL and then the standard HTTP(S)_PROXY environment variables."
        ),
    )
    parser.add_argument(
        "--preserve-process-env",
        action="store_true",
        help="Keep already-exported variables instead of letting the project .env override them.",
    )
    parser.add_argument("--force", action="store_true", help="Do not skip existing successful case/system records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Experiment runs should be reproducible from the checked project configuration.
    # In particular, do not let a stale exported API key silently shadow a newer .env key.
    proxy_display = initialize_runtime_environment(
        proxy_url=args.proxy_url,
        preserve_process_env=args.preserve_process_env,
    )
    provider_map = default_providers()
    if args.systems.strip().lower() == "all":
        system_ids = list(SYSTEMS)
    else:
        system_ids = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown_systems = [s for s in system_ids if s not in SYSTEMS]
    if unknown_systems:
        raise SystemExit(f"Unknown systems: {unknown_systems}")

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
    done = set() if args.force else existing_success_keys(args.output)
    jobs = []
    for case in cases:
        for system_id in system_ids:
            key = (case["case_id"], system_id)
            if key not in done:
                jobs.append((case, system_id))

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "n_cases": len(cases),
                "systems": system_ids,
                "providers": [{"name": p.name, "model": p.model, "supports_vision": p.supports_vision} for p in providers],
                "effective_proxy": proxy_display,
                "dotenv_overrides_process_env": not args.preserve_process_env,
                "jobs_to_run": len(jobs),
                "already_done": len(done),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not jobs:
        return 0

    lock = threading.Lock()
    completed = 0
    started = time.time()

    def task(job: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
        case, system_id = job
        return run_one(
            case=case,
            system_id=system_id,
            providers=providers,
            release_root=args.release_root,
            max_images_per_modality=args.max_images_per_modality,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )

    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_job = {executor.submit(task, job): job for job in jobs}
        for future in futures.as_completed(future_to_job):
            result = future.result()
            append_jsonl(args.output, [result], lock)
            completed += 1
            print(
                f"[{completed}/{len(jobs)}] {result['case_id']} {result['system_id']} "
                f"{result['status']} {result.get('predicted_diagnosis')} "
                f"{result.get('provider','')} {result.get('runtime_seconds')}s",
                flush=True,
            )
    print(f"Completed {completed} jobs in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
