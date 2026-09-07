"""OpenAI provider implementation, account access, and model discovery."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

from ..auth import (
    OAuthAdapter,
    OAuthAuthorizationRequest,
    OAuthConfiguration,
    OAuthProvider,
    OAuthTokens,
    ProviderAuthentication,
)
from ..catalogue import ModelRecord, ProviderRecord
from ..errors import AuthenticationError, ContextWindowError
from .litellm import LiteLLM


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


class OpenAIAccountResponsesModel(BaseChatModel):
    """A model backed by the OpenAI account subscription Codex Responses endpoint."""

    model: str
    context_length: int = 0
    session_id: str = ""
    timeout: float | None = 300.0
    credential_values: dict[str, Any] = Field(default_factory=dict, exclude=True)
    request_parameters: dict[str, Any] = Field(default_factory=dict, exclude=True)
    authentication: ProviderAuthentication | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "openai-account-responses"

    def context_window(self) -> int:
        return max(0, int(self.context_length or 0))

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
    ) -> Runnable[Any, Any]:
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
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.get("function", tool).get("name"),
                    "description": tool.get("function", tool).get("description", ""),
                    "parameters": tool.get("function", tool).get("parameters", {}),
                    "strict": False,
                }
                for tool in tools
            ]
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
                additional = getattr(message, "additional_kwargs", {}) or {}
                if additional.get("reasoning_model") == self.model:
                    reasoning_items = additional.get("reasoning_items")
                    if isinstance(reasoning_items, list):
                        items.extend(item for item in reasoning_items if isinstance(item, dict))
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

    async def _headers(self) -> dict[str, str]:
        if self.authentication is None:
            raise AuthenticationError("OpenAI account authentication is not configured.")
        return await self.authentication.request_headers(
            "openai", session_identifier=self.session_id
        )

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
            output_index = int(data.get("output_index", 0) or 0)
            content_index = int(data.get("content_index", 0) or 0)
            index = (output_index + content_index) * (output_index + content_index + 1) // 2
            index += content_index
            return cls._chunk(
                content_block=TextContentBlock(
                    type="text",
                    text=str(data.get("delta", "")),
                    id=str(
                        data.get("item_id")
                        or f"response-output-{int(data.get('output_index', 0) or 0)}"
                    ),
                    index=index,
                )
            )
        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            output_index = int(data.get("output_index", 0) or 0)
            content_index = int(data.get("summary_index", data.get("content_index", 0)))
            index = (output_index + content_index) * (output_index + content_index + 1) // 2
            index += content_index
            return cls._chunk(
                content_block=ReasoningContentBlock(
                    type="reasoning",
                    reasoning=str(data.get("delta", "")),
                    id=str(
                        data.get("item_id")
                        or f"response-output-{int(data.get('output_index', 0) or 0)}"
                    ),
                    index=index,
                )
            )
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
            raw_usage = response.get("usage")
            usage: Any = None
            if raw_usage:
                input_tokens = int(raw_usage.get("input_tokens") or 0)
                output_tokens = int(raw_usage.get("output_tokens") or 0)
                total_tokens = int(raw_usage.get("total_tokens") or (input_tokens + output_tokens))
                if input_tokens or output_tokens or total_tokens:
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                    details = raw_usage.get("input_tokens_details") or {}
                    cached = int(details.get("cached_tokens") or 0)
                    written = int(details.get("cache_write_tokens") or 0)
                    if cached or written:
                        usage["input_token_details"] = {
                            "cache_read": cached,
                            "cache_creation": written,
                        }
                    reasoning = int(
                        (raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
                    )
                    if reasoning:
                        usage["output_token_details"] = {"reasoning": reasoning}
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

    async def _astream_websocket(
        self, payload: dict[str, Any], headers: dict[str, str], state: dict[str, Any]
    ) -> AsyncIterator[ChatGenerationChunk]:
        websocket_headers = dict(headers)
        websocket_headers.pop("Content-Type", None)
        websocket_headers.pop("Accept", None)
        websocket_headers["OpenAI-Beta"] = RESPONSES_WEBSOCKET_BETA
        parsed = urlsplit(RESPONSES_URL)
        scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
        websocket = connect(
            urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)),
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
                json.dumps({"type": "response.create", **payload}, separators=(",", ":"))
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

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        raise RuntimeError(
            "OpenAIAccountResponsesModel cannot run synchronously inside an active event loop; use ainvoke."
        )


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
        super().__init__(access_token, refresh_token, expires_at)
        object.__setattr__(self, "id_token", id_token)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "email", email)

    @property
    def account(self) -> str:
        return self.email or self.account_id

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], previous: OAuthTokens | None = None
    ) -> OpenAIAccountTokens:
        if not isinstance(payload, Mapping):
            raise AuthenticationError("OpenAI returned an invalid account token response.")
        access_token = str(payload.get("access_token") or "")
        if not access_token:
            raise AuthenticationError("OpenAI returned no account access token.")
        previous_account = previous if isinstance(previous, cls) else None
        id_token = str(
            payload.get("id_token") or (previous_account.id_token if previous_account else "")
        )
        try:
            _, payload_part, _ = id_token.split(".")
            claims = json.loads(
                base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
            )
            claims = claims if isinstance(claims, dict) else {}
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
            claims = {}
        auth_claim = claims.get("https://api.openai.com/auth")
        account_id = (
            auth_claim.get("chatgpt_account_id", "") if isinstance(auth_claim, dict) else ""
        )
        return cls(
            access_token=access_token,
            refresh_token=str(
                payload.get("refresh_token") or (previous.refresh_token if previous else "")
            ),
            id_token=id_token,
            account_id=str(account_id or (previous_account.account_id if previous_account else "")),
            email=str(claims.get("email") or (previous_account.email if previous_account else "")),
            expires_at=time.time() + float(payload.get("expires_in") or 3600),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> OpenAIAccountTokens:
        """Rebuild an OpenAI account session from its persisted representation."""
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
        return cls(
            access_token=access_token,
            refresh_token=str(payload.get("refresh_token") or ""),
            id_token=str(payload.get("id_token") or ""),
            account_id=str(payload.get("account_id") or ""),
            email=str(payload.get("email") or ""),
            expires_at=expires_at,
        )

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> OpenAIAccountTokens | None:
        value = values.get("openai")
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls.from_mapping(value)
            except AuthenticationError:
                return None
        return None

    @staticmethod
    def to_mapping(tokens: OAuthTokens) -> Mapping[str, Any]:
        if not isinstance(tokens, OpenAIAccountTokens):
            raise AuthenticationError("OpenAI OAuth returned an invalid token type.")
        return {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
            "account_id": tokens.account_id,
            "email": tokens.email,
            "expires_at": tokens.expires_at,
        }

    @staticmethod
    def request_headers(
        tokens: OpenAIAccountTokens, session_identifier: str = ""
    ) -> dict[str, str]:
        """Return headers required by the OpenAI account Responses endpoint."""
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
            terminal = names.get(normalized, program)
            terminal_user_agent = f"{terminal}/{version}" if version else terminal
        else:
            terminal_user_agent = os.environ.get("TERM", "").strip() or "unknown"
        if platform.system() == "Darwin":
            operating_system = "Mac OS"
            operating_system_version = platform.mac_ver()[0] or platform.release()
        else:
            operating_system = platform.system() or "unknown"
            operating_system_version = platform.release() or "unknown"
        architecture = platform.machine() or "unknown"
        originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", OPENAI_ORIGINATOR)
        user_agent = (
            f"{originator}/{OPENAI_CLIENT_VERSION} "
            f"({operating_system} {operating_system_version}; {architecture}) "
            f"{terminal_user_agent}"
        )
        session_id = session_identifier or str(uuid.uuid4())
        return {
            "Authorization": f"Bearer {tokens.access_token}",
            "ChatGPT-Account-ID": tokens.account_id,
            "originator": originator,
            "User-Agent": user_agent,
            "session-id": session_id,
            "thread-id": session_id,
            "x-client-request-id": session_id,
            "x-codex-window-id": f"{session_id}:0",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }


RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


class OpenAI:
    """Concrete OpenAI provider implementation."""

    identifier = "openai"
    oauth: OAuthProvider = OAuthAdapter(
        "openai",
        OPENAI_OAUTH_CONFIGURATION,
        token_parser=OpenAIAccountTokens.from_payload,
        header_builder=lambda token, _request, session: OpenAIAccountTokens.request_headers(
            cast(OpenAIAccountTokens, token), session
        ),
        authorization_factory=lambda redirect_uri, **kwargs: OAuthAuthorizationRequest(
            "openai",
            OPENAI_OAUTH_CONFIGURATION,
            token_parser=OpenAIAccountTokens.from_payload,
            redirect_uri=redirect_uri,
            **kwargs,
        ),
        token_serializer=OpenAIAccountTokens.to_mapping,
        token_deserializer=OpenAIAccountTokens.from_mapping,
    )

    def supports(self, record: ModelRecord) -> bool:
        return record.provider == self.identifier

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
                authentication=authentication,
            )
        return LiteLLM().chat(
            record,
            provider,
            values=values,
            authentication=authentication,
            timeout_seconds=timeout_seconds,
            request_parameters=request_parameters,
        )


__all__ = [
    "OpenAI",
    "OpenAIAccountResponsesModel",
    "OpenAIAccountTokens",
]
