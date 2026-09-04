"""ChatGPT subscription model transport for the Codex Responses backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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

from .errors import AuthenticationError, ContextWindowError
from .oauth_providers import ChatGPTTokens, request_chatgpt_headers, valid_chatgpt_tokens
from .subscriptions import RESPONSES_URL, cached_chatgpt_models, capture_usage_headers


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


class ChatGPTResponsesModel(BaseChatModel):
    """A LangChain model backed by the ChatGPT subscription Codex Responses endpoint."""

    model: str
    reasoning_effort: str | None = None
    temperature: float = 0.0
    context_length: int = 0
    session_id: str = ""
    timeout: float | None = 300.0
    credential_store: Any = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "chatgpt-responses"

    def context_window(self) -> int:
        live = cached_chatgpt_models().get(self.model)
        live_context = int(live.get("context") or 0) if isinstance(live, Mapping) else 0
        return max(live_context, max(0, int(self.context_length or 0)))

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "reasoning_effort": self.reasoning_effort}

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
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
            "stream": stream,
            "parallel_tool_calls": kwargs.get("parallel_tool_calls", True),
            "tool_choice": kwargs.get("tool_choice") or "auto",
            "reasoning": {"effort": self.reasoning_effort or None, "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            payload["instructions"] = instructions
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = [self._to_responses_tool(tool) for tool in tools]
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
        return request_chatgpt_headers(
            await valid_chatgpt_tokens(self.credential_store), self.session_id
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
            return AuthenticationError(f"ChatGPT rejected the subscription token: {body[:800]}")
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
        return RuntimeError(f"ChatGPT Codex endpoint returned {status}: {body[:800]}")

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
            raise RuntimeError(f"ChatGPT Codex stream failed: {message}")
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
            raise RuntimeError("ChatGPT Codex websocket closed before response.completed")
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
        tokens = self.credential_store.load("chatgpt") if self.credential_store else None
        if not isinstance(tokens, ChatGPTTokens) or tokens.is_expired():
            raise AuthenticationError("Not signed in to ChatGPT (or the session expired).")
        payload = self.build_payload(messages, stream=True, **kwargs)
        headers = request_chatgpt_headers(tokens, self.session_id)
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


__all__ = ["ChatGPTResponsesModel"]
