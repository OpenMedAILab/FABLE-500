#!/usr/bin/env python3
"""Diagnose API keys, proxy reachability, and provider endpoints for FABLE-500.

This script is intended to be run from a normal host terminal before launching
benchmark jobs. It loads the project .env with override by default, so stale
shell-exported keys do not silently shadow the checked configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
SCRIPTS_DIR = DATASET_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_api_benchmark import (  # noqa: E402
    default_providers,
    initialize_runtime_environment,
)


def redact_key(value: str) -> str:
    if not value:
        return "MISSING"
    if len(value) <= 12:
        return f"SET(len={len(value)})"
    return f"{value[:8]}...{value[-4:]}(len={len(value)})"


def endpoint_for_models(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/models"
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def parse_proxy_url(raw: str) -> urllib.parse.SplitResult | None:
    if not raw:
        return None
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return parsed


def check_tcp_proxy(proxy_url: str, timeout: float = 3.0) -> dict[str, Any]:
    parsed = parse_proxy_url(proxy_url)
    if parsed is None:
        return {"ok": False, "error": "No valid HTTP(S) proxy URL configured."}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return {"ok": True, "host": parsed.hostname, "port": port}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "host": parsed.hostname, "port": port, "error": repr(exc)}


def check_models_endpoint(provider_id: str, timeout: int) -> dict[str, Any]:
    provider = default_providers()[provider_id]
    url = endpoint_for_models(provider.base_url)
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "User-Agent": "FABLE-500-api-environment-check",
    }
    if "openrouter" in provider.base_url.lower():
        headers["HTTP-Referer"] = "https://github.com/OpenMedAILab/FABLE-500"
        headers["X-Title"] = "FABLE-500 environment check"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(2000).decode("utf-8", errors="replace")
        return {"ok": True, "status": getattr(response, "status", None), "url": url, "body_preview": body[:500]}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "url": url, "error": detail[:1000]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": repr(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="openrouter_gpt56_sol,openrouter_gemini37_flash,openrouter_claude_sonnet5,bailian_qwen_vl",
        help="Comma-separated provider IDs to inspect.",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Explicit HTTP(S) proxy, e.g. http://127.0.0.1:7890.",
    )
    parser.add_argument(
        "--preserve-process-env",
        action="store_true",
        help="Do not let .env override already-exported environment variables.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proxy_display = initialize_runtime_environment(
        proxy_url=args.proxy_url,
        preserve_process_env=args.preserve_process_env,
    )
    provider_map = default_providers()
    provider_ids = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in provider_ids if p not in provider_map]
    if unknown:
        raise SystemExit(f"Unknown provider IDs: {unknown}")

    proxy_url = args.proxy_url or os.environ.get("FABLE500_PROXY_URL") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    report: dict[str, Any] = {
        "dotenv_overrides_process_env": not args.preserve_process_env,
        "effective_proxy": proxy_display,
        "proxy_tcp_check": check_tcp_proxy(proxy_url) if proxy_url else {"ok": False, "error": "No proxy URL configured."},
        "providers": [],
    }
    for provider_id in provider_ids:
        provider = provider_map[provider_id]
        key = os.environ.get(provider.api_key_env, "")
        item: dict[str, Any] = {
            "provider_id": provider_id,
            "model": provider.model,
            "base_url": provider.base_url,
            "api_key_env": provider.api_key_env,
            "api_key": redact_key(key),
        }
        if key:
            item["models_endpoint"] = check_models_endpoint(provider_id, args.timeout)
        else:
            item["models_endpoint"] = {"ok": False, "error": f"{provider.api_key_env} is not set"}
        report["providers"].append(item)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = [
        p["provider_id"]
        for p in report["providers"]
        if not p.get("models_endpoint", {}).get("ok")
    ]
    if failed:
        print(f"[WARN] Provider endpoint checks failed for: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
