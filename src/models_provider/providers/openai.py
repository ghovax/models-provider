"""OpenAI provider implementation, account access, and model discovery."""

from __future__ import annotations

import asyncio
import base64
import binascii
import html
import json
import os
import platform
import time
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, AsyncIterator, Callable, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import add_ai_message_chunks
from langchain_core.messages.content import ContentBlock, ReasoningContentBlock, TextContentBlock
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from websockets.asyncio.client import connect

from ..auth import ProviderAuthentication
from ..auth import (
    OAuthAdapter,
    OAuthAuthorizationRequest,
    OAuthConfiguration,
    OAuthProvider,
    OAuthTokens,
)
from ..catalogue import ModelRecord, ProviderRecord
from ..errors import AuthenticationError, ContextWindowError
from .litellm import LiteLLMChatModel, _SDK_PREFIXES


CONTEXT_OVERFLOW_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_length_error",
        "input_too_large",
        "string_above_max_length",
        "request_too_large",
    }
)

RESPONSES_WEBSOCKET_BETA = "responses_websockets=2026-02-06"


class _ResponsesWebSocketUnavailable(RuntimeError):
    """The websocket handshake failed before a request was sent."""


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray, str)):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _reasoning_items(message: BaseMessage, model: str) -> list[dict[str, Any]]:
    additional = getattr(message, "additional_kwargs", {}) or {}
    if additional.get("reasoning_model") != model:
        return []
    items = additional.get("reasoning_items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


class OpenAIAccountResponsesModel(BaseChatModel):
    """A model backed by the OpenAI account subscription Codex Responses endpoint."""

    model: str
    context_length: int = 0
    session_id: str = ""
    timeout: float | None = 300.0
    credential_values: dict[str, Any] = Field(default_factory=dict, exclude=True)
    request_parameters: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "openai-account-responses"

    def context_window(self) -> int:
        live = cached_openai_models().get(self.model)
        live_context = int(live.get("context") or 0) if isinstance(live, Mapping) else 0
        return max(live_context, max(0, int(self.context_length or 0)))

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, **self.request_parameters}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        bound: dict[str, Any] = {"tools": formatted_tools}
        if tool_choice is not None:
            bound["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            bound["parallel_tool_calls"] = parallel_tool_calls
        return self.bind(**bound, **kwargs)

    def build_payload(
        self, messages: Sequence[BaseMessage], *, stream: bool, **kwargs: Any
    ) -> dict[str, Any]:
        instructions, input_items = self._to_responses_input(messages)
        parameters = {**self.request_parameters, **kwargs}
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
            "stream": stream,
            "parallel_tool_calls": parameters.get("parallel_tool_calls", True),
            "tool_choice": parameters.get("tool_choice") or "auto",
            "reasoning": {
                "effort": parameters.get("reasoning_effort"),
                "summary": "auto",
            },
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            payload["instructions"] = instructions
        tools = parameters.get("tools")
        if tools:
            payload["tools"] = [self.responses_tool(tool) for tool in tools]
        for key in (
            "temperature",
            "top_p",
            "max_output_tokens",
            "top_logprobs",
            "metadata",
            "prompt_cache_key",
            "service_tier",
            "text",
            "truncation",
        ):
            value = parameters.get(key)
            if value is not None:
                payload[key] = value
        if self.session_id:
            payload["client_metadata"] = {
                "session_id": self.session_id,
                "thread_id": self.session_id,
                "x-codex-window-id": f"{self.session_id}:0",
            }
            payload["prompt_cache_key"] = self.session_id
        return payload

    def _to_responses_input(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[str, list[dict[str, Any]]]:
        instructions = ""
        items: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                text = _text(message)
                if not instructions:
                    instructions = text
                elif text:
                    items.append(
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": text}],
                        }
                    )
                continue
            if isinstance(message, ToolMessage):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": _text(message),
                    }
                )
                continue
            if isinstance(message, AIMessage):
                items.extend(_reasoning_items(message, self.model))
                text = _text(message)
                if text:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                for call in message.tool_calls or []:
                    arguments = call.get("args")
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id"),
                            "name": call.get("name"),
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments, separators=(",", ":")),
                        }
                    )
                continue
            role = "developer" if message.additional_kwargs.get("reminder") else "user"
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": _text(message)}],
                }
            )
        return instructions, items

    @staticmethod
    def responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
        function = tool.get("function", tool)
        return {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
            "strict": False,
        }

    async def _headers(self) -> dict[str, str]:
        return request_openai_account_headers(
            await valid_openai_account_tokens(self.credential_values), self.session_id
        )

    @staticmethod
    def _websocket_url() -> str:
        parsed = urlsplit(RESPONSES_URL)
        scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
        return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))

    @staticmethod
    def _websocket_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {"type": "response.create", **payload}

    @staticmethod
    def _http_error(status: int, body: str) -> Exception:
        if status in (401, 403):
            return AuthenticationError(f"OpenAI rejected the account token: {body[:800]}")
        try:
            data = json.loads(body)
        except (TypeError, ValueError):
            data = {}
        detail = data.get("error") if isinstance(data, dict) else None
        code = detail.get("code") if isinstance(detail, dict) else ""
        if status == 400 and code in CONTEXT_OVERFLOW_CODES:
            return ContextWindowError(
                "The request exceeded this model's context window.",
                model="",
            )
        return RuntimeError(f"OpenAI account endpoint returned {status}: {body[:800]}")

    @classmethod
    def _translate_event(
        cls, data: dict[str, Any], state: dict[str, Any]
    ) -> ChatGenerationChunk | None:
        event_type = data.get("type", "")
        if event_type == "response.output_text.delta":
            return cls._chunk(content_block=cls._text_content_block(data))
        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            return cls._chunk(content_block=cls._reasoning_content_block(data))
        if event_type == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                return cls._chunk(
                    reasoning_item={
                        "type": "reasoning",
                        "id": item.get("id"),
                        "summary": item.get("summary") or [],
                        "encrypted_content": item["encrypted_content"],
                    },
                    model=str(state.get("model") or ""),
                )
            return None
        if event_type == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                state["saw_tool_call"] = True
                return cls._chunk(
                    tool_call_chunk={
                        "index": int(data.get("output_index", 0) or 0),
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "args": item.get("arguments") or "",
                        "type": "tool_call_chunk",
                    }
                )
            return None
        if event_type == "response.function_call_arguments.delta":
            return cls._chunk(
                tool_call_chunk={
                    "index": int(data.get("output_index", 0) or 0),
                    "id": None,
                    "name": None,
                    "args": str(data.get("delta", "")),
                    "type": "tool_call_chunk",
                }
            )
        if event_type == "response.completed":
            response = data.get("response") or {}
            usage = cls._usage(response.get("usage"))
            return ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=usage),
                generation_info={
                    "finish_reason": "tool_calls" if state.get("saw_tool_call") else "stop"
                },
            )
        if event_type in ("response.failed", "response.error", "error"):
            response = data.get("response") or {}
            detail = response.get("error") or data.get("error") or {}
            structured = detail if isinstance(detail, dict) else {}
            code = str(structured.get("code") or "")
            message = str(structured.get("message") or detail or "unknown error")
            if code in CONTEXT_OVERFLOW_CODES:
                raise ContextWindowError(
                    message,
                    model=str(state.get("model") or ""),
                    context_window=int(state.get("context_window") or 0),
                )
            raise RuntimeError(f"OpenAI account stream failed: {message}")
        return None

    @staticmethod
    def _content_block_index(data: dict[str, Any], block_type: str) -> int:
        output_index = int(data.get("output_index", 0) or 0)
        content_index = int(
            data.get("summary_index", data.get("content_index", 0))
            if block_type == "reasoning"
            else data.get("content_index", 0)
        )
        total = output_index + content_index
        return total * (total + 1) // 2 + content_index

    @staticmethod
    def _content_block_identifier(data: dict[str, Any]) -> str:
        return str(
            data.get("item_id") or f"response-output-{int(data.get('output_index', 0) or 0)}"
        )

    @classmethod
    def _text_content_block(cls, data: dict[str, Any]) -> TextContentBlock:
        return TextContentBlock(
            type="text",
            text=str(data.get("delta", "")),
            id=cls._content_block_identifier(data),
            index=cls._content_block_index(data, "text"),
        )

    @classmethod
    def _reasoning_content_block(cls, data: dict[str, Any]) -> ReasoningContentBlock:
        return ReasoningContentBlock(
            type="reasoning",
            reasoning=str(data.get("delta", "")),
            id=cls._content_block_identifier(data),
            index=cls._content_block_index(data, "reasoning"),
        )

    @staticmethod
    def _chunk(
        content_block: ContentBlock | None = None,
        tool_call_chunk: ToolCallChunk | None = None,
        reasoning_item: dict[str, Any] | None = None,
        model: str = "",
    ) -> ChatGenerationChunk:
        blocks = [content_block] if content_block is not None else []
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=cast(Any, blocks),
                tool_call_chunks=[tool_call_chunk] if tool_call_chunk else [],
                additional_kwargs=(
                    {"reasoning_items": [reasoning_item], "reasoning_model": model}
                    if reasoning_item
                    else {}
                ),
            )
        )

    @staticmethod
    def _usage(usage: Any) -> Any:
        if not usage:
            return None
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        if not (input_tokens or output_tokens or total_tokens):
            return None
        metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        details = usage.get("input_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        written = int(details.get("cache_write_tokens") or 0)
        if cached or written:
            metadata["input_token_details"] = {
                "cache_read": cached,
                "cache_creation": written,
            }
        reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        if reasoning:
            metadata["output_token_details"] = {"reasoning": reasoning}
        return metadata

    async def _astream_websocket(
        self, payload: dict[str, Any], headers: dict[str, str], state: dict[str, Any]
    ) -> AsyncIterator[ChatGenerationChunk]:
        websocket_headers = dict(headers)
        websocket_headers.pop("Content-Type", None)
        websocket_headers.pop("Accept", None)
        websocket_headers["OpenAI-Beta"] = RESPONSES_WEBSOCKET_BETA
        websocket = connect(
            self._websocket_url(),
            additional_headers=websocket_headers,
            user_agent_header=websocket_headers.get("User-Agent"),
            open_timeout=self.timeout,
            close_timeout=10,
            max_size=None,
        )
        try:
            connection = await websocket.__aenter__()
        except Exception as error:  # noqa: BLE001 — the caller owns the HTTP fallback
            raise _ResponsesWebSocketUnavailable(str(error)) from error
        try:
            await connection.send(
                json.dumps(self._websocket_payload(payload), separators=(",", ":"))
            )
            async for message in connection:
                if isinstance(message, bytes):
                    message = message.decode("utf-8", "replace")
                if not isinstance(message, str):
                    continue
                try:
                    data = json.loads(message)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                chunk = self._translate_event(data, state)
                if chunk is not None:
                    yield chunk
                if data.get("type") == "response.completed":
                    return
            raise RuntimeError("OpenAI account websocket closed before response.completed")
        finally:
            await websocket.__aexit__(None, None, None)

    async def _astream_http(
        self, payload: dict[str, Any], headers: dict[str, str], state: dict[str, Any]
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", RESPONSES_URL, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise self._http_error(response.status_code, body)
                capture_usage_headers(response.headers)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(data, dict):
                        chunk = self._translate_event(data, state)
                        if chunk is not None:
                            yield chunk

    async def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self.build_payload(messages, stream=True, **kwargs)
        headers = await self._headers()
        state: dict[str, Any] = {
            "saw_tool_call": False,
            "model": self.model,
            "context_window": self.context_window(),
        }
        try:
            async for chunk in self._astream_websocket(payload, headers, state):
                yield chunk
        except _ResponsesWebSocketUnavailable:
            async for chunk in self._astream_http(payload, headers, state):
                yield chunk

    async def stream_generations(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream provider generations for an embedding model wrapper."""
        async for chunk in self._astream(messages, stop=stop, **kwargs):
            yield chunk

    def generate_result(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate one result for an embedding model wrapper."""
        return self._generate(messages, stop=stop, **kwargs)

    @classmethod
    def _chunks_to_result(cls, chunks: list[AIMessageChunk]) -> ChatResult:
        aggregate = add_ai_message_chunks(chunks[0], *chunks[1:]) if chunks else None
        if aggregate is None:
            return ChatResult(generations=[])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=aggregate.content,
                        tool_calls=list(aggregate.tool_calls or []),
                        additional_kwargs=aggregate.additional_kwargs,
                        usage_metadata=aggregate.usage_metadata,
                    )
                )
            ]
        )

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
        return self._chunks_to_result(chunks)

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tokens = openai_account_tokens(self.credential_values)
        if not isinstance(tokens, OpenAIAccountTokens) or tokens.is_expired():
            raise AuthenticationError("Not signed in to OpenAI (or the session expired).")
        payload = self.build_payload(messages, stream=True, **kwargs)
        headers = request_openai_account_headers(tokens, self.session_id)
        chunks: list[AIMessageChunk] = []
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", RESPONSES_URL, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    raise self._http_error(
                        response.status_code, response.read().decode("utf-8", "replace")
                    )
                state: dict[str, Any] = {
                    "saw_tool_call": False,
                    "model": self.model,
                    "context_window": self.context_window(),
                }
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(data, dict):
                        chunk = self._translate_event(data, state)
                        if chunk is not None:
                            chunks.append(cast(AIMessageChunk, chunk.message))
        return self._chunks_to_result(chunks)


OPENAI_AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"

OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"

OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

OPENAI_SCOPES = ("openid", "profile", "email", "offline_access")

OPENAI_LOOPBACK_REDIRECT_URI = "http://localhost:1455/auth/callback"

OPENAI_CLIENT_VERSION = "0.152.1"

OPENAI_ORIGINATOR = "codex_cli_rs"

OPENAI_OAUTH_CONFIGURATION = OAuthConfiguration(
    authorization_url=OPENAI_AUTHORIZATION_URL,
    token_url=OPENAI_TOKEN_URL,
    client_id=OPENAI_CLIENT_ID,
    scopes=OPENAI_SCOPES,
    redirect_uri=OPENAI_LOOPBACK_REDIRECT_URI,
)


_openai_models: dict[str, dict[str, Any]] = {}
_usage_snapshot: dict[str, Any] | None = None


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        _, payload, _ = token.split(".")
        decoded = json.loads(_b64url_decode(payload))
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
        return {}


@dataclass(frozen=True, slots=True, init=False)
class OpenAIAccountTokens(OAuthTokens):
    """OpenAI account subscription credentials."""

    id_token: str = ""
    account_id: str = ""
    email: str = ""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        id_token: str = "",
        account_id: str = "",
        email: str = "",
        expires_at: float = 0.0,
    ) -> None:
        object.__setattr__(self, "access_token", access_token)
        object.__setattr__(self, "refresh_token", refresh_token)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "id_token", id_token)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "email", email)

    @property
    def account(self) -> str:
        return self.email or self.account_id


_openai_account_refresh_lock = asyncio.Lock()


def _openai_account_from_payload(
    payload: Mapping[str, Any], previous: OpenAIAccountTokens | None = None
) -> OpenAIAccountTokens:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("OpenAI returned an invalid account token response.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("OpenAI returned no account access token.")
    id_token = str(payload.get("id_token") or (previous.id_token if previous else ""))
    claims = _jwt_claims(id_token)
    auth_claim = claims.get("https://api.openai.com/auth")
    account_id = auth_claim.get("chatgpt_account_id", "") if isinstance(auth_claim, dict) else ""
    return OpenAIAccountTokens(
        access_token=access_token,
        refresh_token=str(
            payload.get("refresh_token") or (previous.refresh_token if previous else "")
        ),
        id_token=id_token,
        account_id=str(account_id or (previous.account_id if previous else "")),
        email=str(claims.get("email") or (previous.email if previous else "")),
        expires_at=time.time() + float(payload.get("expires_in") or 3600),
    )


def _save(provider: str, credentials: OAuthTokens, values: dict[str, Any]) -> None:
    values[provider] = credentials


def openai_account_tokens_to_mapping(tokens: OpenAIAccountTokens) -> dict[str, Any]:
    """Return the provider-owned persisted representation of an OpenAI account session."""
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "id_token": tokens.id_token,
        "account_id": tokens.account_id,
        "email": tokens.email,
        "expires_at": tokens.expires_at,
    }


def openai_account_tokens_from_mapping(payload: Mapping[str, Any]) -> OpenAIAccountTokens:
    """Rebuild an OpenAI account session from a persisted provider representation."""
    if not isinstance(payload, Mapping):
        raise AuthenticationError("Stored OpenAI account credentials are invalid.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("Stored OpenAI account credentials contain no access token.")
    try:
        expires_at = float(payload.get("expires_at") or 0.0)
    except (TypeError, ValueError) as error:
        raise AuthenticationError(
            "Stored OpenAI account credentials have an invalid expiry."
        ) from error
    return OpenAIAccountTokens(
        access_token=access_token,
        refresh_token=str(payload.get("refresh_token") or ""),
        id_token=str(payload.get("id_token") or ""),
        account_id=str(payload.get("account_id") or ""),
        email=str(payload.get("email") or ""),
        expires_at=expires_at,
    )


def openai_account_tokens(values: dict[str, Any]) -> OpenAIAccountTokens | None:
    value = values.get("openai")
    if isinstance(value, OpenAIAccountTokens):
        return value
    if isinstance(value, Mapping):
        try:
            return openai_account_tokens_from_mapping(value)
        except AuthenticationError:
            return None
    return None


async def valid_openai_account_tokens(values: dict[str, Any]) -> OpenAIAccountTokens:
    tokens = openai_account_tokens(values)
    if tokens is None:
        raise AuthenticationError("Not signed in to OpenAI.")
    if not tokens.is_expired():
        return tokens
    async with _openai_account_refresh_lock:
        current = openai_account_tokens(values) or tokens
        if not current.is_expired():
            return current
        if not current.refresh_token:
            raise AuthenticationError("OpenAI session expired; sign in again.")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    OPENAI_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": current.refresh_token,
                        "client_id": OPENAI_CLIENT_ID,
                        "scope": " ".join(OPENAI_SCOPES),
                    },
                )
                response.raise_for_status()
                refreshed = _openai_account_from_payload(response.json(), current)
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not refresh the OpenAI session: {error}") from error
        _save("openai", refreshed, values)
        return refreshed


class OpenAIAccountLoginFlow:
    """PKCE loopback login. The host opens ``authorize_url`` and owns the browser policy."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self._authorization = OAuthAuthorizationRequest(
            "openai",
            OPENAI_OAUTH_CONFIGURATION,
            token_parser=_openai_account_from_payload,
        )
        self._server: HTTPServer | None = None
        self._captured: dict[str, str] = {}

    @property
    def authorize_url(self) -> str:
        return self._authorization.authorize_url

    async def start(self) -> None:
        flow = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format_string: str, *arguments: object) -> None:
                return

            def do_GET(self) -> None:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if urllib.parse.urlparse(self.path).path != "/auth/callback":
                    flow._captured["error"] = "Invalid callback path."
                elif query.get("state", [""])[0] != flow._authorization.state:
                    flow._captured["error"] = "Authorization state mismatch."
                elif query.get("code", [""])[0]:
                    flow._captured["code"] = query["code"][0]
                else:
                    flow._captured["error"] = query.get("error", ["Authorization failed."])[0]
                body = f"<html><body>{html.escape(flow._captured.get('error', 'Signed in.'))}</body></html>".encode()
                self.send_response(200 if "code" in flow._captured else 400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
        self._server.timeout = 0.5

    async def wait(self, timeout: float = 300.0) -> OpenAIAccountTokens:  # noqa: ASYNC109
        if self._server is None:
            raise AuthenticationError("start() must be called before wait().")
        deadline = time.monotonic() + timeout
        try:
            while not self._captured:
                if time.monotonic() >= deadline:
                    raise AuthenticationError("OpenAI sign-in timed out.")
                await asyncio.to_thread(self._server.handle_request)
            if "code" not in self._captured:
                raise AuthenticationError(self._captured.get("error", "OpenAI sign-in failed."))
            tokens = await self._authorization.exchange(self._captured["code"])
            _save("openai", tokens, self._values)
            return tokens
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not complete OpenAI sign-in: {error}") from error
        finally:
            await self.close()

    async def close(self) -> None:
        if self._server is not None:
            await asyncio.to_thread(self._server.server_close)
            self._server = None


def _terminal_user_agent() -> str:
    program = os.environ.get("TERM_PROGRAM", "").strip()
    version = os.environ.get("TERM_PROGRAM_VERSION", "").strip()
    if program:
        normalized = "".join(char.lower() for char in program if char not in " -_.")
        names = {
            "appleterminal": "Apple_Terminal",
            "ghostty": "Ghostty",
            "iterm": "iTerm.app",
            "iterm2": "iTerm.app",
            "itermapp": "iTerm.app",
            "warp": "WarpTerminal",
            "warpterminal": "WarpTerminal",
            "vscode": "vscode",
            "wezterm": "WezTerm",
            "kitty": "kitty",
            "alacritty": "Alacritty",
            "konsole": "Konsole",
            "gnometerminal": "gnome-terminal",
            "vte": "VTE",
            "windowsterminal": "WindowsTerminal",
        }
        name = names.get(normalized, program)
        return f"{name}/{version}" if version else name
    return os.environ.get("TERM", "").strip() or "unknown"


def _openai_account_user_agent() -> str:
    if platform.system() == "Darwin":
        operating_system = "Mac OS"
        operating_system_version = platform.mac_ver()[0] or platform.release()
    else:
        operating_system = platform.system() or "unknown"
        operating_system_version = platform.release() or "unknown"
    architecture = platform.machine() or "unknown"
    originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", OPENAI_ORIGINATOR)
    return (
        f"{originator}/{OPENAI_CLIENT_VERSION} "
        f"({operating_system} {operating_system_version}; {architecture}) "
        f"{_terminal_user_agent()}"
    )


def request_openai_account_headers(
    tokens: OpenAIAccountTokens, session_identifier: str = ""
) -> dict[str, str]:
    """Headers required by the OpenAI account Responses endpoint."""
    session_id = session_identifier or str(uuid.uuid4())
    originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", OPENAI_ORIGINATOR)
    return {
        "Authorization": f"Bearer {tokens.access_token}",
        "ChatGPT-Account-ID": tokens.account_id,
        "originator": originator,
        "User-Agent": _openai_account_user_agent(),
        "session-id": session_id,
        "thread-id": session_id,
        "x-client-request-id": session_id,
        "x-codex-window-id": f"{session_id}:0",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

MODELS_URL = "https://chatgpt.com/backend-api/codex/models"

CLIENT_VERSION = "0.152.1"

ORIGINATOR = "codex_cli_rs"


def _response_models(response: httpx.Response) -> list[Mapping[str, Any]]:
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("provider model response is not an object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("provider model response has no model list")
    return [entry for entry in models if isinstance(entry, Mapping)]


async def fetch_openai_models(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if _openai_models:
        return deepcopy(_openai_models)
    try:
        tokens = await valid_openai_account_tokens(values)
        headers = {
            key: value
            for key, value in request_openai_account_headers(tokens).items()
            if key != "Accept"
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                MODELS_URL, params={"client_version": CLIENT_VERSION}, headers=headers
            )
            response.raise_for_status()
            for entry in _response_models(response):
                if entry.get("slug"):
                    _openai_models[str(entry["slug"])] = {
                        "name": entry.get("display_name") or entry["slug"],
                        "context": int(entry.get("context_window") or 0),
                    }
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return deepcopy(_openai_models)


def cached_openai_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_openai_models)


def clear_openai_models_cache() -> None:
    _openai_models.clear()


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


def openai_oauth_adapter() -> OAuthProvider:
    return OAuthAdapter(
        "openai",
        OPENAI_OAUTH_CONFIGURATION,
        flow_factory=OpenAIAccountLoginFlow,
        token_parser=lambda payload, previous: _openai_account_from_payload(
            payload,
            previous if isinstance(previous, OpenAIAccountTokens) else None,
        ),
        header_builder=lambda token, _request, session: request_openai_account_headers(
            token, session
        ),
        authorization_factory=lambda redirect_uri, **kwargs: OAuthAuthorizationRequest(
            "openai",
            OPENAI_OAUTH_CONFIGURATION,
            token_parser=_openai_account_from_payload,
            redirect_uri=redirect_uri,
            **kwargs,
        ),
        token_serializer=openai_account_tokens_to_mapping,
        token_deserializer=openai_account_tokens_from_mapping,
    )


class OpenAI:
    """Concrete OpenAI provider implementation."""

    def supports(self, record: ModelRecord) -> bool:
        return record.provider == "openai"

    def chat(
        self,
        record: ModelRecord,
        provider: ProviderRecord,
        *,
        values: dict[str, Any],
        authentication: ProviderAuthentication,
        timeout_seconds: float | None,
        request_parameters: Mapping[str, Any],
    ) -> BaseChatModel:
        value = values.get("openai")
        account_access = hasattr(value, "access_token") or (
            isinstance(value, Mapping) and (not value or "access_token" in value)
        )
        if account_access:
            authentication.token("openai")
            return OpenAIAccountResponsesModel(
                model=record.model,
                timeout=timeout_seconds,
                context_length=record.context_length,
                credential_values=values,
                request_parameters=dict(request_parameters),
            )
        resolution = authentication.resolve(
            provider.identifier,
            environment_variables=provider.environment_variables,
        )
        if not resolution.available:
            raise AuthenticationError(f"No configured access is available for {record.provider!r}.")
        model = LiteLLMChatModel(
            model=f"{_SDK_PREFIXES.get(provider.npm, 'openai')}/{record.model}",
            api_base=provider.api_base or None,
            timeout=timeout_seconds,
            context_length=record.context_length,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
            request_parameters=dict(request_parameters),
        )
        model._authentication = authentication
        return model


__all__ = [
    "OpenAI",
    "OpenAIAccountResponsesModel",
    "OpenAIAccountTokens",
    "openai_account_tokens",
    "openai_account_tokens_from_mapping",
    "openai_account_tokens_to_mapping",
    "openai_oauth_adapter",
    "request_openai_account_headers",
    "valid_openai_account_tokens",
    "fetch_openai_models",
    "cached_openai_models",
    "clear_openai_models_cache",
    "capture_usage_headers",
    "get_usage_snapshot",
    "set_usage_snapshot",
    "clear_usage_snapshot",
]
