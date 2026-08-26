"""Reusable account-backed provider endpoints and usage snapshots."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import httpx

from .auth import (
    AuthenticationError,
    ChatGPTTokens,
    CursorTokens,
    current_credential_store,
    request_chatgpt_headers,
    request_cursor_headers,
    valid_chatgpt_tokens,
    valid_cursor_tokens,
)

__all__ = [
    "APPEND_PATH", "AVAILABLE_MODELS_URL", "CLIENT_TYPE", "CLIENT_VERSION", "MODELS_URL",
    "ORIGINATOR", "RESPONSES_URL", "RUN_HOSTS", "RUN_PATH", "STATUS_RESOURCE_EXHAUSTED",
    "STATUS_UNAUTHENTICATED", "UNKNOWN_CONTEXT_WINDOW", "cached_chatgpt_models",
    "cached_cursor_models", "capture_usage_headers", "clear_chatgpt_models_cache",
    "clear_cursor_models_cache", "clear_usage_snapshot", "display_cursor_account",
    "fetch_chatgpt_models", "fetch_cursor_models", "get_usage_snapshot", "machine_time_zone",
    "observed_context_window", "record_context_window", "request_chatgpt_headers",
    "request_cursor_headers", "set_usage_snapshot",
]

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
CLIENT_VERSION = "0.144.4"
ORIGINATOR = "codex_cli_rs"

RUN_PATH = "/agent.v1.AgentService/RunSSE"
APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"
RUN_HOSTS = ("https://api2.cursor.sh", "https://agent.api5.cursor.sh", "https://agentn.api5.cursor.sh")
AVAILABLE_MODELS_URL = "https://api2.cursor.sh/aiserver.v1.AiService/AvailableModels"
STATUS_RESOURCE_EXHAUSTED = 8
STATUS_UNAUTHENTICATED = 16
CLIENT_TYPE = "cli"
UNKNOWN_CONTEXT_WINDOW = 200_000

_chatgpt_models: dict[str, dict[str, Any]] = {}
_cursor_models: dict[str, dict[str, Any]] = {}
_observed_windows: dict[str, int] = {}
_usage_snapshot: dict[str, Any] | None = None
_model_lock = asyncio.Lock()
logger = logging.getLogger(__name__)


async def fetch_chatgpt_models() -> dict[str, dict[str, Any]]:
    if _chatgpt_models:
        return dict(_chatgpt_models)
    try:
        tokens = await valid_chatgpt_tokens()
        headers = {key: value for key, value in request_chatgpt_headers(tokens).items() if key != "Accept"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(MODELS_URL, params={"client_version": CLIENT_VERSION}, headers=headers)
            response.raise_for_status()
            for entry in response.json().get("models", []):
                if isinstance(entry, dict) and entry.get("slug"):
                    _chatgpt_models[str(entry["slug"])] = {"name": entry.get("display_name") or entry["slug"], "context": int(entry.get("context_window") or 0)}
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return dict(_chatgpt_models)


async def fetch_cursor_models() -> dict[str, dict[str, Any]]:
    if _cursor_models:
        return dict(_cursor_models)
    try:
        tokens = await valid_cursor_tokens()
        headers = {**request_cursor_headers(tokens, str(uuid.uuid4())), "Content-Type": "application/json", "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(AVAILABLE_MODELS_URL, headers=headers, json={"isNightly": False, "excludeMaxNamedModels": True})
            response.raise_for_status()
            for entry in response.json().get("models", []):
                if isinstance(entry, dict) and entry.get("name"):
                    model_identifier = str(entry["name"])
                    _cursor_models[model_identifier] = {"name": model_identifier, "context": 0}
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return dict(_cursor_models)


def cached_chatgpt_models() -> dict[str, dict[str, Any]]:
    return dict(_chatgpt_models)


def cached_cursor_models() -> dict[str, dict[str, Any]]:
    return dict(_cursor_models)


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
    return ""


def _header_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _header_int(value: Any) -> int | None:
    parsed = _header_float(value)
    return int(parsed) if parsed is not None else None


def capture_usage_headers(headers: Mapping[str, str]) -> None:
    global _usage_snapshot
    if "x-codex-plan-type" not in headers and "x-codex-primary-window-minutes" not in headers:
        return
    windows: list[dict[str, Any]] = []
    for window_name in ("primary", "secondary"):
        duration = _header_int(headers.get(f"x-codex-{window_name}-window-minutes")) or 0
        if duration:
            windows.append({"key": window_name, "used_percent": _header_float(headers.get(f"x-codex-{window_name}-used-percent")) or 0.0, "window_minutes": duration, "resets_at": _header_int(headers.get(f"x-codex-{window_name}-reset-at"))})
    _usage_snapshot = {"plan_type": headers.get("x-codex-plan-type", ""), "active_limit": headers.get("x-codex-active-limit", ""), "captured_at": int(time.time()), "windows": windows}


def get_usage_snapshot() -> dict[str, Any] | None:
    return dict(_usage_snapshot) if _usage_snapshot else None


def set_usage_snapshot(usage: dict[str, Any] | None) -> None:
    global _usage_snapshot
    _usage_snapshot = dict(usage) if usage else None


def clear_usage_snapshot() -> None:
    global _usage_snapshot
    _usage_snapshot = None
