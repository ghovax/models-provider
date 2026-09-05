"""OpenCode Zen request metadata and OpenAI-compatible wire helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict


# The official OpenCode source currently checked out alongside this repository is
# version 1.18.29. Keep the protocol identity in one place so consumers do not
# accidentally advertise their own application name to the OpenCode gateway.
OPENCODE_VERSION = "1.18.29"
OPENCODE_USER_AGENT = f"opencode/{OPENCODE_VERSION}"
OPENCODE_CLIENT = "cli"
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/v1"
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_MAX_OUTPUT_TOKENS = 32_000


class OpenCodeRequestContext(BaseModel):
    """Consumer-owned identifiers attached to one OpenCode model request."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    request_id: str = ""
    project_id: str = ""
    parent_session_id: str = ""


def opencode_default_headers() -> dict[str, str]:
    """Return the static OpenCode headers shared by provider profiles."""
    return {
        "User-Agent": OPENCODE_USER_AGENT,
        "x-opencode-client": OPENCODE_CLIENT,
    }


def opencode_headers(
    context: OpenCodeRequestContext,
    *,
    request_id: str = "",
    api_key: str = "",
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the official OpenCode request headers.

    ``request_id`` is the current user-message identifier. The public sentinel
    is intentionally sent as ``Bearer public``: that is what the official
    OpenAI-compatible provider emits for anonymous Zen/Go requests.
    """
    session_id = context.session_id.strip()
    request_id = request_id.strip() or context.request_id.strip()
    if not session_id:
        raise ValueError("OpenCode request context requires a session_id")
    if not request_id:
        raise ValueError("OpenCode request requires a request_id")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": OPENCODE_USER_AGENT,
        "x-opencode-session": session_id,
        "x-opencode-request": request_id,
        "x-opencode-client": OPENCODE_CLIENT,
    }
    if context.project_id.strip():
        headers["x-opencode-project"] = context.project_id.strip()
    if context.parent_session_id.strip():
        headers["x-parent-session-id"] = context.parent_session_id.strip()
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    if overrides:
        headers.update({str(key): str(value) for key, value in overrides.items()})
    return headers


def opencode_payload(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    stop: Sequence[str] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any = None,
    store: bool | None = None,
) -> dict[str, Any]:
    """Build the OpenAI Chat body used by OpenCode's compatible provider."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    optional = {
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "stop": list(stop) if stop is not None else None,
        "tools": list(tools) if tools else None,
        "tool_choice": tool_choice,
        "store": store,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def opencode_top_p(model: str) -> float | None:
    """Return the model-specific sampling default used by OpenCode today."""
    return 0.95 if "deepseek-v4-flash" in model.lower() else None


def opencode_max_output_tokens(output_limit: int = 0) -> int:
    """Match OpenCode's 32k output cap while respecting a smaller model limit."""
    if output_limit <= 0:
        return OPENCODE_MAX_OUTPUT_TOKENS
    return min(output_limit, OPENCODE_MAX_OUTPUT_TOKENS)


__all__ = [
    "OPENCODE_CLIENT",
    "OPENCODE_GO_BASE_URL",
    "OPENCODE_MAX_OUTPUT_TOKENS",
    "OPENCODE_USER_AGENT",
    "OPENCODE_VERSION",
    "OPENCODE_ZEN_BASE_URL",
    "OpenCodeRequestContext",
    "opencode_default_headers",
    "opencode_headers",
    "opencode_max_output_tokens",
    "opencode_payload",
    "opencode_top_p",
]
