"""Reusable account-backed provider endpoints and usage snapshots."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import httpx

from .credentials import current_credential_store
from .errors import AuthenticationError
from .oauth_providers import (
    CursorTokens,
    request_chatgpt_headers,
    request_cursor_headers,
    valid_chatgpt_tokens,
    valid_cursor_tokens,
)

__all__ = [
    "APPEND_PATH",
    "AGENT_OPEN_URL",
    "AGENT_PRIVACY_URL",
    "AVAILABLE_MODELS_URL",
    "CLIENT_TYPE",
    "CLIENT_VERSION",
    "GET_ME_URL",
    "MODELS_URL",
    "ORIGINATOR",
    "RESPONSES_URL",
    "RUN_HOSTS",
    "RUN_PATH",
    "STATUS_RESOURCE_EXHAUSTED",
    "STATUS_UNAUTHENTICATED",
    "UNKNOWN_CONTEXT_WINDOW",
    "cached_chatgpt_models",
    "cached_cursor_models",
    "capture_usage_headers",
    "clear_chatgpt_models_cache",
    "clear_cursor_models_cache",
    "clear_usage_snapshot",
    "display_cursor_account",
    "fetch_chatgpt_models",
    "fetch_cursor_models",
    "get_usage_snapshot",
    "machine_time_zone",
    "observed_context_window",
    "record_context_window",
    "request_chatgpt_headers",
    "request_cursor_headers",
    "set_usage_snapshot",
    "USABLE_MODELS_URL",
]

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
CLIENT_VERSION = "0.152.1"
ORIGINATOR = "codex_cli_rs"

RUN_PATH = "/agent.v1.AgentService/RunSSE"
APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"
AGENT_PRIVACY_URL = "https://agent.api5.cursor.sh"
AGENT_OPEN_URL = "https://agentn.api5.cursor.sh"
RUN_HOSTS = ("https://api2.cursor.sh", AGENT_PRIVACY_URL, AGENT_OPEN_URL)
USABLE_MODELS_URL = "https://api2.cursor.sh/agent.v1.AgentService/GetUsableModels"
AVAILABLE_MODELS_URL = "https://api2.cursor.sh/aiserver.v1.AiService/AvailableModels"
GET_ME_URL = "https://api2.cursor.sh/aiserver.v1.DashboardService/GetMe"
STATUS_RESOURCE_EXHAUSTED = 8
STATUS_UNAUTHENTICATED = 16
CLIENT_TYPE = "cli"
UNKNOWN_CONTEXT_WINDOW = 200_000

_chatgpt_models: dict[str, dict[str, Any]] = {}
_cursor_models: dict[str, dict[str, Any]] = {}
_observed_windows: dict[str, int] = {}
_usage_snapshot: dict[str, Any] | None = None
logger = logging.getLogger(__name__)


def _response_models(response: httpx.Response) -> list[Mapping[str, Any]]:
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("provider model response is not an object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("provider model response has no model list")
    return [entry for entry in models if isinstance(entry, Mapping)]


async def fetch_chatgpt_models() -> dict[str, dict[str, Any]]:
    if _chatgpt_models:
        return deepcopy(_chatgpt_models)
    try:
        tokens = await valid_chatgpt_tokens()
        headers = {
            key: value for key, value in request_chatgpt_headers(tokens).items() if key != "Accept"
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                MODELS_URL, params={"client_version": CLIENT_VERSION}, headers=headers
            )
            response.raise_for_status()
            for entry in _response_models(response):
                if entry.get("slug"):
                    _chatgpt_models[str(entry["slug"])] = {
                        "name": entry.get("display_name") or entry["slug"],
                        "context": int(entry.get("context_window") or 0),
                    }
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return deepcopy(_chatgpt_models)


async def fetch_cursor_models() -> dict[str, dict[str, Any]]:
    if _cursor_models:
        return deepcopy(_cursor_models)
    try:
        tokens = await valid_cursor_tokens()
        headers = {
            **request_cursor_headers(tokens, str(uuid.uuid4())),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "connect-protocol-version": "1",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            usable_response = await client.post(
                USABLE_MODELS_URL, headers=headers, json={"customModelIds": []}
            )
            usable_response.raise_for_status()
            usable_entries = _response_models(usable_response)
            variants_response = await client.post(
                AVAILABLE_MODELS_URL,
                headers=headers,
                json={
                    "isNightly": False,
                    "excludeMaxNamedModels": True,
                    "additionalModelNames": [],
                    "useModelParameters": True,
                    "useReactModelPicker": True,
                },
            )
            variants_response.raise_for_status()
            variants: dict[str, dict[str, Any]] = {}
            for entry in _response_models(variants_response):
                if not entry.get("name"):
                    continue
                raw_variants = entry.get("variants", [])
                if not isinstance(raw_variants, list):
                    continue
                for variant in raw_variants:
                    if not isinstance(variant, Mapping):
                        continue
                    raw_parameters = variant.get("parameterValues", [])
                    if not isinstance(raw_parameters, list):
                        continue
                    parameters = {
                        str(item.get("id")): str(item.get("value"))
                        for item in raw_parameters
                        if isinstance(item, Mapping) and item.get("id") is not None
                    }
                    context_match = re.fullmatch(
                        r"(\d+(?:\.\d+)?)([km])?", parameters.get("context", "").lower()
                    )
                    context = (
                        round(
                            float(context_match.group(1))
                            * {"k": 1_000, "m": 1_000_000}.get(context_match.group(2) or "", 1)
                        )
                        if context_match
                        else 0
                    )
                    variants[str(entry["name"])] = {
                        "server_model": str(entry.get("serverModelName") or entry["name"]),
                        "maximum_mode": variant.get("isMaxMode") is True,
                        "parameters": tuple(sorted(parameters.items())),
                        "context": context,
                    }
            if not usable_entries:
                usable_entries = [
                    {"modelId": model_name, "displayName": model_name} for model_name in variants
                ]
            for entry in usable_entries:
                model_identifier = entry.get("modelId") or entry.get("displayModelId")
                if not model_identifier:
                    continue
                model_name = str(model_identifier)
                variant = variants.get(model_name)
                if variant is None:
                    names = [name for name in variants if model_name.startswith(name)]
                    variant = variants[max(names, key=len)] if names else None
                _cursor_models[model_name] = {
                    "name": entry.get("displayName") or model_name,
                    "context": variant.get("context", 0) if variant else 0,
                    "variant": variant,
                }
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return deepcopy(_cursor_models)


def cached_chatgpt_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_chatgpt_models)


def cached_cursor_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_cursor_models)


def clear_chatgpt_models_cache() -> None:
    _chatgpt_models.clear()


def clear_cursor_models_cache() -> None:
    _cursor_models.clear()


def machine_time_zone() -> str:
    configured = os.environ.get("TZ", "").strip()
    if configured:
        return configured
    try:
        target = os.readlink("/etc/localtime")
        if "zoneinfo/" in target:
            return target.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return time.tzname[0] if time.tzname else "UTC"


def record_context_window(model_identifier: str, maximum_tokens: int) -> None:
    if maximum_tokens > _observed_windows.get(model_identifier, 0):
        _observed_windows[model_identifier] = maximum_tokens


def observed_context_window(model_identifier: str) -> int:
    return _observed_windows.get(model_identifier, 0)


async def display_cursor_account(tokens: CursorTokens) -> str:
    if tokens.account:
        return tokens.account
    try:
        headers = {
            **request_cursor_headers(tokens, str(uuid.uuid4())),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "connect-protocol-version": "1",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(GET_ME_URL, headers=headers, json={})
            response.raise_for_status()
            payload = response.json()
        for key in ("email", "userEmail", "cachedEmail"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and value.strip():
                updated = CursorTokens(
                    tokens.access_token, tokens.refresh_token, value.strip(), tokens.expires_at
                )
                current_credential_store().save("cursor", updated)
                return value.strip()
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    return ""


def _header_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _header_int(value: Any) -> int | None:
    parsed = _header_float(value)
    return int(parsed) if parsed is not None else None


def _header_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def capture_usage_headers(headers: Mapping[str, str]) -> None:
    global _usage_snapshot
    if "x-codex-plan-type" not in headers and "x-codex-primary-window-minutes" not in headers:
        return
    windows: list[dict[str, Any]] = []
    for window_name in ("primary", "secondary"):
        duration = _header_int(headers.get(f"x-codex-{window_name}-window-minutes")) or 0
        if duration:
            resets_at = _header_int(headers.get(f"x-codex-{window_name}-reset-at"))
            if resets_at is None:
                reset_after = _header_int(headers.get(f"x-codex-{window_name}-reset-after-seconds"))
                resets_at = int(time.time()) + reset_after if reset_after is not None else None
            windows.append(
                {
                    "key": window_name,
                    "used_percent": _header_float(
                        headers.get(f"x-codex-{window_name}-used-percent")
                    )
                    or 0.0,
                    "window_minutes": duration,
                    "resets_at": resets_at,
                }
            )
    _usage_snapshot = {
        "plan_type": headers.get("x-codex-plan-type", ""),
        "active_limit": headers.get("x-codex-active-limit", ""),
        "captured_at": int(time.time()),
        "credits": {
            "has_credits": _header_bool(headers.get("x-codex-credits-has-credits")),
            "balance": _header_float(headers.get("x-codex-credits-balance")),
            "unlimited": _header_bool(headers.get("x-codex-credits-unlimited")),
        },
        "windows": windows,
    }


def get_usage_snapshot() -> dict[str, Any] | None:
    return deepcopy(_usage_snapshot) if _usage_snapshot else None


def set_usage_snapshot(usage: dict[str, Any] | None) -> None:
    global _usage_snapshot
    _usage_snapshot = deepcopy(usage) if usage else None


def clear_usage_snapshot() -> None:
    global _usage_snapshot
    _usage_snapshot = None
