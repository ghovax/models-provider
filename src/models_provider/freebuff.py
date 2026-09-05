"""Strict client transport for the official Freebuff agent protocol.

Freebuff's chat endpoint uses an OpenAI-compatible payload, but a valid free
request is also tied to an account session and an agent run.  This module owns
that lifecycle and keeps the provider-specific fields out of the generic model
provider contracts.

The public Freebuff repository is a client mirror; its backend is private.  The
constants below therefore describe the current official public client revision,
not a promise that an undocumented endpoint is stable forever.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, cast

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from .credentials import CredentialStore, FreebuffCredential, current_credential_store
from .core import ModelRecord
from .errors import AuthenticationError
from .usage import ModelUsage

FREEBUFF_BASE_URL = "https://www.codebuff.com"
FREEBUFF_SESSION_PATH = "/api/v1/freebuff/session"
FREEBUFF_RUN_PATH = "/api/v1/agent-runs"
FREEBUFF_CHAT_PATH = "/api/v1/chat/completions"
FREEBUFF_SOURCE_REVISION = "2161ae71cbd2e98716891ea695201b17ad11ca95"
FREEBUFF_RUNTIME_USER_AGENT = "Bun/1.3.14"
FREEBUFF_CHAT_USER_AGENT = (
    "ai-sdk/openai-compatible/0.10.7/codebuff ai-sdk/provider-utils/3.0.25 runtime/browser"
)
FREEBUFF_SESSION_POLL_SECONDS = 5.0
FREEBUFF_SESSION_LEEWAY_SECONDS = 5.0

# Current official CLI mapping from CodebuffAI/freebuff's base3 harness.
FREEBUFF_CLI_AGENT_BY_MODEL: Mapping[str, str] = {
    "deepseek/deepseek-v4-pro": "base3-free-deepseek",
    "deepseek/deepseek-v4-flash": "base3-free-deepseek-flash",
    "mimo/mimo-v2.5": "base3-free-mimo",
    "minimax/minimax-m3": "base3-free-minimax-m3",
    "openai/gpt-5.6-luna": "base3-free-luna",
    "z-ai/glm-5.2": "base3-free-glm",
    "z-ai/glm-5.3-flash": "base3-free-glm-5-3-flash",
    "anthropic/claude-fable-5": "base3-free-fable",
    "stealth/ox-alpha": "base3-free-ox-alpha",
    "google/gemini-3.8-flash": "base3-free-gemini-3-8-flash",
    "crof/kimi-k3-eco": "base3-free-kimi-k3-eco",
    "openai/gpt-5.6-luna-es": "base3-free-luna-es",
    "meta/muse-spark-1.3-contributor": "base3-free-muse-spark-1-3",
    "upstage/solar-pro-4": "base3-free-solar-pro4",
}

# These are the official root prompt openings exported by the public client.
# We validate rather than inject them; issue #14 tracks whether construction of
# the complete official prompt is authorized for an external provider.
FREEBUFF_ROOT_PROMPT_OPENINGS = (
    "You are Buffy, the coding agent behind Codebuff.",
    "You are Buffy, the strategic coding assistant.",
    "You are Buffy, the Freebuff Cloud project planner.",
    "You are Buffy, the auto-run agent behind Freebuff Desktop.",
)


@dataclass(frozen=True, slots=True)
class FreebuffSessionState:
    model: str
    instance_id: str
    expires_at: float

    def is_valid(self) -> bool:
        return (
            not self.expires_at or time.time() < self.expires_at - FREEBUFF_SESSION_LEEWAY_SECONDS
        )


class FreebuffProtocolError(RuntimeError):
    """The official endpoint rejected an invalid or incomplete protocol request."""


def _credential(store: CredentialStore | None) -> FreebuffCredential:
    selected = store or current_credential_store()
    value = selected.load("freebuff")
    if not isinstance(value, FreebuffCredential) or not value.account_token.strip():
        raise AuthenticationError("Not signed in to Freebuff.")
    return value


def _text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if not isinstance(message.content, Sequence) or isinstance(message.content, (bytes, bytearray)):
        return ""
    return "".join(
        part if isinstance(part, str) else str(part.get("text", ""))
        for part in message.content
        if isinstance(part, str)
        or (isinstance(part, Mapping) and isinstance(part.get("text"), str))
    )


def _content(message: BaseMessage) -> str | list[dict[str, Any]]:
    """Preserve OpenAI-compatible text and image content parts verbatim."""
    if isinstance(message.content, str):
        return message.content
    if not isinstance(message.content, Sequence) or isinstance(message.content, (bytes, bytearray)):
        return _text(message)
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, Mapping):
            parts.append(dict(part))
    return parts


def _official_prompt_present(messages: Sequence[BaseMessage]) -> bool:
    return any(
        isinstance(message, SystemMessage)
        and any(
            _text(message).lstrip().startswith(opening) for opening in FREEBUFF_ROOT_PROMPT_OPENINGS
        )
        for message in messages
    )


def _client_id() -> str:
    """Match the official SDK's Math.random().toString(36).substring(2, 15)."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    value = int.from_bytes(uuid.uuid4().bytes[:8], "big")
    result = ""
    while value:
        value, remainder = divmod(value, 36)
        result = alphabet[remainder] + result
    return (result or "0")[-13:]


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "user"

    payload: dict[str, Any] = {"role": role, "content": _content(message)}
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.get("id") or str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": (
                        call.get("args")
                        if isinstance(call.get("args"), str)
                        else json.dumps(call.get("args") or {}, separators=(",", ":"))
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: Any) -> dict[str, Any]:
    converted = convert_to_openai_tool(tool)
    return cast(dict[str, Any], converted)


class FreebuffTransport:
    """Account/session/run transport matching the official client lifecycle."""

    def __init__(self, credential: FreebuffCredential) -> None:
        self.credential = credential
        self._user_id = credential.user_id.strip()
        self._session: FreebuffSessionState | None = None
        self._session_lock = asyncio.Lock()

    def _headers(self, *, user_agent: str, include_user_id: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.credential.account_token}",
            "User-Agent": user_agent,
        }
        if include_user_id and self._user_id:
            headers["x-freebuff-acting-user-id"] = self._user_id
        return headers

    async def authenticated_user_id(self, client: httpx.AsyncClient) -> str:
        """Resolve the account id the official SDK reads before starting a run."""
        response = await client.get(
            f"{FREEBUFF_BASE_URL}/api/v1/me?fields=id",
            headers=self._headers(user_agent=FREEBUFF_RUNTIME_USER_AGENT),
        )
        if response.status_code in (401, 403):
            raise AuthenticationError("Freebuff rejected the account token.")
        if not response.is_success:
            raise FreebuffProtocolError(f"Freebuff account lookup failed: {response.status_code}")
        payload = response.json()
        user_id = str(payload.get("id") or "") if isinstance(payload, Mapping) else ""
        if not user_id:
            raise FreebuffProtocolError("Freebuff account lookup returned no user id.")
        self._user_id = user_id
        return user_id

    async def ensure_session(self, model: str, client: httpx.AsyncClient) -> FreebuffSessionState:
        async with self._session_lock:
            if self._session and self._session.model == model and self._session.is_valid():
                return self._session

            response = await client.post(
                f"{FREEBUFF_BASE_URL}{FREEBUFF_SESSION_PATH}",
                headers={
                    **self._headers(user_agent=FREEBUFF_RUNTIME_USER_AGENT),
                    "x-freebuff-model": model,
                },
            )
            state = await self._session_response(response)
            while state.get("status") == "queued":
                instance_id = str(state.get("instanceId") or "")
                if not instance_id:
                    raise FreebuffProtocolError("Freebuff queued session has no instanceId.")
                await asyncio.sleep(FREEBUFF_SESSION_POLL_SECONDS)
                response = await client.get(
                    f"{FREEBUFF_BASE_URL}{FREEBUFF_SESSION_PATH}",
                    headers={
                        **self._headers(user_agent=FREEBUFF_RUNTIME_USER_AGENT),
                        "x-freebuff-instance-id": instance_id,
                        "x-freebuff-compact-session": "1",
                    },
                )
                state = await self._session_response(response)
            if state.get("status") != "active":
                raise FreebuffProtocolError(
                    f"Freebuff session unavailable: {state.get('status', 'unknown')}"
                )
            instance_id = str(state.get("instanceId") or "")
            if not instance_id:
                raise FreebuffProtocolError("Freebuff active session has no instanceId.")
            expires_at = _timestamp(state.get("expiresAt"))
            self._session = FreebuffSessionState(model, instance_id, expires_at)
            return self._session

    async def start_run(self, agent_id: str, client: httpx.AsyncClient) -> str:
        response = await client.post(
            f"{FREEBUFF_BASE_URL}{FREEBUFF_RUN_PATH}",
            headers=self._headers(user_agent=FREEBUFF_RUNTIME_USER_AGENT, include_user_id=True),
            json={"action": "START", "agentId": agent_id, "ancestorRunIds": []},
        )
        if not response.is_success:
            raise FreebuffProtocolError(f"Freebuff agent run start failed: {response.status_code}")
        payload = response.json()
        run_id = str(payload.get("runId") or "") if isinstance(payload, Mapping) else ""
        if not run_id:
            raise FreebuffProtocolError("Freebuff agent run response has no runId.")
        return run_id

    async def finish_run(self, run_id: str, client: httpx.AsyncClient) -> None:
        try:
            await client.post(
                f"{FREEBUFF_BASE_URL}{FREEBUFF_RUN_PATH}",
                headers=self._headers(user_agent=FREEBUFF_RUNTIME_USER_AGENT, include_user_id=True),
                json={
                    "action": "FINISH",
                    "runId": run_id,
                    "status": "completed",
                    "totalSteps": 1,
                    "directCredits": 0,
                    "totalCredits": 0,
                    "steps": [],
                },
            )
        except httpx.HTTPError:
            return

    @staticmethod
    async def _session_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 404:
            raise FreebuffProtocolError("Freebuff free sessions are unavailable.")
        if not response.is_success:
            body = (await response.aread()).decode("utf-8", "replace")[:500]
            raise FreebuffProtocolError(
                f"Freebuff session request failed: {response.status_code} {body}"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise FreebuffProtocolError("Freebuff session response is not an object.")
        return dict(payload)


class FreebuffChatModel(BaseChatModel):
    """A strict Freebuff chat model using the current official client protocol."""

    model: str
    temperature: float = 0.0
    reasoning_effort: str | None = "high"
    context_length: int = 0
    timeout: float | None = 300.0
    credential_store: CredentialStore | None = Field(default=None, exclude=True)
    _transport: FreebuffTransport | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "freebuff-chat-completions"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "reasoning_effort": self.reasoning_effort}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        bound: dict[str, Any] = {"tools": [_tool_payload(tool) for tool in tools]}
        if tool_choice is not None:
            bound["tool_choice"] = tool_choice
        return self.bind(**bound, **kwargs)

    def _get_transport(self) -> FreebuffTransport:
        if self._transport is None:
            self._transport = FreebuffTransport(_credential(self.credential_store))
        return self._transport

    def _payload(
        self,
        messages: Sequence[BaseMessage],
        *,
        stream: bool,
        run_id: str,
        session: FreebuffSessionState,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not _official_prompt_present(messages):
            raise FreebuffProtocolError(
                "Freebuff requires the actual official root-agent system prompt; "
                "marker construction is intentionally unresolved in issue #14."
            )
        metadata: dict[str, str] = {
            "run_id": run_id,
            "client_id": _client_id(),
            "cost_mode": "free",
            "trace_session_id": str(uuid.uuid4()),
            "freebuff_instance_id": session.instance_id,
        }
        if self.reasoning_effort:
            metadata["freebuff_reasoning_effort"] = self.reasoning_effort
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_payload(message) for message in messages],
            "stream": stream,
            "codebuff_metadata": metadata,
            "provider": {"data_collection": "deny"},
        }
        if kwargs.get("tools"):
            payload["tools"] = [_tool_payload(tool) for tool in kwargs["tools"]]
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]
        return payload

    async def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        transport = self._get_transport()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await transport.authenticated_user_id(client)
            session = await transport.ensure_session(self.model, client)
            agent_id = FREEBUFF_CLI_AGENT_BY_MODEL.get(self.model)
            if not agent_id:
                raise FreebuffProtocolError(
                    f"No current official Freebuff agent for {self.model!r}."
                )
            run_id = await transport.start_run(agent_id, client)
            try:
                payload = self._payload(
                    messages,
                    stream=True,
                    run_id=run_id,
                    session=session,
                    stop=stop,
                    **kwargs,
                )
                headers = transport._headers(
                    user_agent=FREEBUFF_CHAT_USER_AGENT, include_user_id=True
                )
                headers["Content-Type"] = "application/json"
                async with client.stream(
                    "POST",
                    f"{FREEBUFF_BASE_URL}{FREEBUFF_CHAT_PATH}",
                    headers=headers,
                    json=payload,
                ) as response:
                    if not response.is_success:
                        raise _http_error(
                            response.status_code,
                            (await response.aread()).decode("utf-8", "replace"),
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(parsed, Mapping):
                            continue
                        chunk = _chunk_from_payload(parsed)
                        if chunk is not None:
                            yield chunk
            finally:
                await transport.finish_run(run_id, client)

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks: list[AIMessageChunk] = []
        async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            chunks.append(cast(AIMessageChunk, chunk.message))
        if not chunks:
            return ChatResult(generations=[])
        message = chunks[0]
        for chunk in chunks[1:]:
            message = message + chunk
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))


def _timestamp(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _http_error(status: int, body: str) -> Exception:
    if status in (401, 403):
        return AuthenticationError(f"Freebuff rejected the account token: {body[:800]}")
    if status in (400, 409, 410, 428):
        return FreebuffProtocolError(f"Freebuff rejected the request ({status}): {body[:800]}")
    return RuntimeError(f"Freebuff chat endpoint returned {status}: {body[:800]}")


def _chunk_from_payload(payload: Mapping[str, Any]) -> ChatGenerationChunk | None:
    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        return _usage_chunk(payload)
    choice = raw_choices[0]
    if not isinstance(choice, Mapping):
        return _usage_chunk(payload)
    delta = choice.get("delta") or {}
    if not isinstance(delta, Mapping):
        return None
    content = str(delta.get("content") or "")
    reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
    tool_calls = delta.get("tool_calls")
    chunks: dict[str, Any] = {}
    if content:
        chunks["content"] = content
    if reasoning:
        chunks["additional_kwargs"] = {"reasoning_content": reasoning}
    if isinstance(tool_calls, list):
        chunks["tool_call_chunks"] = [_tool_call_chunk(call) for call in tool_calls]
    if not chunks:
        return _usage_chunk(payload)
    return ChatGenerationChunk(message=AIMessageChunk(**chunks))


def _usage_chunk(payload: Mapping[str, Any]) -> ChatGenerationChunk | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    normalized = ModelUsage.from_mapping(cast(Mapping[str, Any], usage))
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            usage_metadata={
                "input_tokens": normalized.input_tokens,
                "output_tokens": normalized.output_tokens,
                "total_tokens": normalized.total_tokens,
            },
        )
    )


def _tool_call_chunk(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"id": None, "name": None, "args": "", "index": 0}
    function = value.get("function")
    function = function if isinstance(function, Mapping) else {}
    raw_index = value.get("index", 0)
    index = int(raw_index) if isinstance(raw_index, (int, float, str)) else 0
    return {
        "id": str(value.get("id")) if value.get("id") else None,
        "name": str(function.get("name")) if function.get("name") else None,
        "args": str(function.get("arguments") or ""),
        "index": index,
    }


def freebuff_model_records() -> tuple[ModelRecord, ...]:
    """Return conservative records for the current official CLI model set."""
    return tuple(
        ModelRecord(
            identifier=f"freebuff/{model}",
            provider="freebuff",
            model=model,
            name=model,
            reasoning=True,
            tool_call=True,
            extra={"official_agent_id": agent_id, "source_revision": FREEBUFF_SOURCE_REVISION},
        )
        for model, agent_id in FREEBUFF_CLI_AGENT_BY_MODEL.items()
    )


__all__ = [
    "FREEBUFF_BASE_URL",
    "FREEBUFF_CLI_AGENT_BY_MODEL",
    "FREEBUFF_SOURCE_REVISION",
    "FreebuffChatModel",
    "FreebuffCredential",
    "FreebuffProtocolError",
    "freebuff_model_records",
]
