"""A small, independent LiteLLM-backed LangChain model implementation."""

from __future__ import annotations

import json
from inspect import isawaitable
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import litellm
from litellm.exceptions import (
    APIConnectionError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import invalid_tool_call, tool_call, tool_call_chunk
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr

from ..auth import OAuthProvider, ProviderAuthentication
from ..catalogue import ModelRecord, ProviderRecord
from ..errors import AuthenticationError, ContextWindowError, TransientProviderError
from ..usage import ModelUsage


_SDK_PREFIXES = {
    "@ai-sdk/anthropic": "anthropic",
    "@ai-sdk/amazon-bedrock": "bedrock",
    "@ai-sdk/azure": "azure",
    "@ai-sdk/cerebras": "cerebras",
    "@ai-sdk/cohere": "cohere",
    "@ai-sdk/deepinfra": "deepinfra",
    "@ai-sdk/google": "gemini",
    "@ai-sdk/groq": "groq",
    "@ai-sdk/mistral": "mistral",
    "@ai-sdk/openai": "openai",
    "@ai-sdk/openai-compatible": "openai",
    "@ai-sdk/perplexity": "perplexity",
    "@ai-sdk/togetherai": "together_ai",
    "@ai-sdk/xai": "xai",
    "@openrouter/ai-sdk-provider": "openrouter",
}

_LITELLM_ENVIRONMENT_PARAMETERS = {
    "GOOGLE_VERTEX_PROJECT": "vertex_project",
    "GOOGLE_VERTEX_LOCATION": "vertex_location",
    "GOOGLE_APPLICATION_CREDENTIALS": "vertex_credentials",
    "VERTEXAI_PROJECT": "vertex_project",
    "VERTEXAI_LOCATION": "vertex_location",
    "VERTEXAI_CREDENTIALS": "vertex_credentials",
    "AWS_ACCESS_KEY_ID": "aws_access_key_id",
    "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    "AWS_SESSION_TOKEN": "aws_session_token",
    "AWS_REGION": "aws_region_name",
    "AWS_DEFAULT_REGION": "aws_region_name",
    "AWS_BEARER_TOKEN_BEDROCK": "aws_bearer_token",
}

_OPENCODE_USER_AGENT = "opencode/1.18.29"
_OPENCODE_CLIENT = "cli"
_OPENCODE_MAXIMUM_OUTPUT_TOKENS = 32_000


@dataclass(frozen=True, slots=True)
class OpenCodeRequestContext:
    """Stable session identity and per-request identity for OpenCode."""

    session_id: str
    project_id: str = ""
    parent_session_id: str = ""


class LiteLLMChatModel(BaseChatModel):
    """A provider-qualified model usable by any LangChain-compatible application."""

    model: str
    api_key: SecretStr | None = None
    api_base: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    maximum_tokens: int | None = None
    supports_temperature: bool = True
    timeout: float | None = 300.0
    reasoning_effort: str | None = None
    context_length: int = 0
    input_modalities: tuple[str, ...] = ()
    default_headers: dict[str, str] = Field(default_factory=dict)
    request_parameters: dict[str, Any] = Field(default_factory=dict, exclude=True)
    request_context: OpenCodeRequestContext | None = None

    provider_identifier: str = ""
    provider_environment_variables: tuple[str, ...] = ()
    _authentication: ProviderAuthentication | None = None

    @property
    def _llm_type(self) -> str:
        return "litellm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "api_base": self.api_base, **self.request_parameters}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        parameters: dict[str, Any] = {"tools": [convert_to_openai_tool(tool) for tool in tools]}
        if isinstance(tool_choice, bool):
            tool_choice = "required" if tool_choice else "none"
        elif tool_choice == "any":
            tool_choice = "required"
        if tool_choice is not None:
            parameters["tool_choice"] = (
                tool_choice
                if tool_choice in ("auto", "none", "required") or isinstance(tool_choice, dict)
                else {"type": "function", "function": {"name": tool_choice}}
            )
        if parallel_tool_calls is not None:
            parameters["parallel_tool_calls"] = parallel_tool_calls
        return self.bind(**{**parameters, **kwargs})

    def context_window(self) -> int:
        return self.context_length

    def _image_part(self, block: Mapping[str, Any]) -> dict[str, Any]:
        if self.input_modalities and "image" not in self.input_modalities:
            raise ValueError(f"Model {self.model!r} does not support image input")
        url = block.get("url")
        if isinstance(url, str) and url:
            return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
        mime_type, base64_data = block.get("mime_type"), block.get("base64")
        if not isinstance(mime_type, str) or not isinstance(base64_data, str):
            raise ValueError("Image block requires MIME type and base64 data")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_data}", "detail": "auto"},
        }

    def _message(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "user"
        content: Any = message.content
        if isinstance(message, (HumanMessage, ToolMessage)):
            blocks: list[dict[str, Any]] = [dict(block) for block in message.content_blocks]
            if any(block.get("type") == "image" for block in blocks):
                parts: list[dict[str, Any]] = []
                for block in blocks:
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        if isinstance(message, HumanMessage):
                            parts.append(self._image_part(block))
                    else:
                        raise ValueError(f"Unsupported content block: {block.get('type')!r}")
                content = parts or ""
        item: dict[str, Any] = {"role": role, "content": content}
        if isinstance(message, AIMessage):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str):
                item["reasoning_content"] = reasoning
        if isinstance(message, ToolMessage):
            item["tool_call_id"] = message.tool_call_id
        elif isinstance(message, AIMessage) and message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    },
                }
                for call in message.tool_calls
            ]
        return item

    def _messages(self, messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        """Keep text-only tool replies together, then provide their images as vision input."""
        items: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        for message in messages:
            if images and not isinstance(message, ToolMessage):
                items.append({"role": "user", "content": images})
                images = []
            items.append(self._message(message))
            if isinstance(message, ToolMessage):
                images.extend(
                    self._image_part(dict(block))
                    for block in message.content_blocks
                    if block.get("type") == "image"
                )
        if images:
            items.append({"role": "user", "content": images})
        return items

    def _parameters(self, **kwargs: Any) -> dict[str, Any]:
        request_context = kwargs.pop("opencode_request_context", None)
        params: dict[str, Any] = {"model": self.model}
        if self.supports_temperature and self.temperature is not None:
            params["temperature"] = self.temperature
        resolved = None
        if self._authentication is not None and self.provider_identifier:
            resolved = self._authentication.resolve(
                self.provider_identifier,
                environment_variables=self.provider_environment_variables,
            )
        if resolved is not None:
            if resolved.api_key:
                params["api_key"] = resolved.api_key
            elif resolved.method == "environment":
                environment_parameters = {
                    _LITELLM_ENVIRONMENT_PARAMETERS[name]: value
                    for name, value in resolved.environment.items()
                    if name in _LITELLM_ENVIRONMENT_PARAMETERS
                }
                if not environment_parameters:
                    raise AuthenticationError(
                        f"No credentials are available for {self.provider_identifier!r}."
                    )
                params.update(environment_parameters)
            elif self.api_key is not None and self.api_key.get_secret_value():
                params["api_key"] = self.api_key.get_secret_value()
            else:
                raise AuthenticationError(
                    f"No credentials are available for {self.provider_identifier!r}."
                )
        elif self.api_key is not None and self.api_key.get_secret_value():
            params["api_key"] = self.api_key.get_secret_value()
        else:
            raise AuthenticationError("No explicit credentials are available for this model.")
        if resolved is not None and resolved.api_base:
            params["api_base"] = resolved.api_base
        elif self.api_base:
            params["api_base"] = self.api_base
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.maximum_tokens is not None:
            params["max_tokens"] = self.maximum_tokens
        headers = dict(self.default_headers)
        if resolved is not None:
            headers = {**resolved.headers, **headers}
        is_opencode = self.provider_identifier.lower() in {"opencode", "opencode-go"}
        if is_opencode:
            context = request_context or self.request_context
            if context is None or not context.session_id.strip():
                raise AuthenticationError("OpenCode models require a request context.")
            reserved = {
                "user-agent",
                "x-opencode-client",
                "x-opencode-session",
                "x-opencode-request",
                "authorization",
            }
            if context.project_id.strip():
                reserved.add("x-opencode-project")
            if context.parent_session_id.strip():
                reserved.add("x-parent-session-id")
            headers = {
                name: value for name, value in headers.items() if name.lower() not in reserved
            }
            request_id = uuid4().hex
            headers.update(
                {
                    "User-Agent": _OPENCODE_USER_AGENT,
                    "x-opencode-client": _OPENCODE_CLIENT,
                    "x-opencode-session": context.session_id.strip(),
                    "x-opencode-request": request_id,
                }
            )
            if context.project_id.strip():
                headers["x-opencode-project"] = context.project_id.strip()
            if context.parent_session_id.strip():
                headers["x-parent-session-id"] = context.parent_session_id.strip()
            api_key = str(params.get("api_key") or "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            model_name = self.model.rsplit("/", 1)[-1].lower()
            if self.provider_identifier.lower() == "opencode" and model_name in {
                "kimi-k2-thinking",
                "glm-4.6",
            }:
                params["extra_body"] = {
                    **(params.get("extra_body") or {}),
                    "chat_template_args": {"enable_thinking": True},
                }
        params.update(
            {
                key: value
                for key, value in {**self.request_parameters, **kwargs}.items()
                if value is not None
            }
        )
        custom_headers = params.get("extra_headers") or {}
        if not isinstance(custom_headers, Mapping):
            raise ValueError("extra_headers must be a mapping")
        if is_opencode:
            reserved = {name.lower() for name in headers}
            custom_headers = {
                name: value
                for name, value in custom_headers.items()
                if name.lower() not in reserved
            }
            params["extra_headers"] = {**custom_headers, **headers}
        elif headers or custom_headers:
            params["extra_headers"] = {**headers, **custom_headers}
        return params

    @staticmethod
    def _payload(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            result = dump(exclude_none=True)
            return dict(result) if isinstance(result, Mapping) else {}
        return vars(value) if hasattr(value, "__dict__") else {}

    @staticmethod
    def _usage(payload: Mapping[str, Any]) -> UsageMetadata:
        usage = ModelUsage.from_mapping(payload)
        return {
            "input_tokens": usage.tokens.input_tokens,
            "output_tokens": usage.tokens.output_tokens,
            "total_tokens": usage.tokens.total_tokens,
            "input_token_details": {
                "cache_read": usage.cache.cache_read_tokens,
                "cache_creation": usage.cache.cache_write_tokens,
            },
            "output_token_details": {"reasoning": usage.tokens.reasoning_tokens},
        }

    def _response(self, response: Any) -> ChatResult:
        payload = self._payload(response)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("The provider returned no completion choices.")
        choice = self._payload(choices[0])
        source = self._payload(choice.get("message"))
        calls, invalid = [], []
        for value in source.get("tool_calls") or []:
            call = self._payload(value)
            function = self._payload(call.get("function"))
            raw = function.get("arguments", "")
            try:
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if not isinstance(function.get("name"), str) or not function["name"]:
                    raise ValueError("The provider returned a tool call without a function name.")
                calls.append(
                    tool_call(name=function.get("name") or "", args=arguments, id=call.get("id"))
                )
            except (TypeError, ValueError) as error:
                invalid.append(
                    invalid_tool_call(
                        name=function.get("name"),
                        args=raw if isinstance(raw, str) else None,
                        id=call.get("id"),
                        error=str(error),
                    )
                )
        raw_usage = payload.get("usage")
        metadata: dict[str, Any] = {"finish_reason": choice.get("finish_reason")}
        if raw_usage is not None:
            metadata["models_provider_usage"] = asdict(
                ModelUsage.from_mapping(self._payload(raw_usage))
            )
        reasoning = source.get("reasoning_content")
        message = AIMessage(
            content=source.get("content") or "",
            tool_calls=calls,
            invalid_tool_calls=invalid,
            additional_kwargs={"reasoning_content": reasoning}
            if isinstance(reasoning, str)
            else {},
            usage_metadata=self._usage(self._payload(raw_usage)) if raw_usage is not None else None,
            response_metadata=metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream_chunk(self, response: Any, state: dict[str, Any]) -> ChatGenerationChunk | None:
        payload = self._payload(response)
        if payload.get("usage") is not None:
            state["usage"] = self._payload(payload["usage"])
        choices = payload.get("choices") or []
        if not choices:
            return None
        choice = self._payload(choices[0])
        delta = self._payload(choice.get("delta"))
        calls = []
        for position, value in enumerate(delta.get("tool_calls") or []):
            call = self._payload(value)
            function = self._payload(call.get("function"))
            index = call.get("index", position)
            index = position if index is None else index
            arguments = function.get("arguments") or ""
            state["arguments"][index] = state["arguments"].get(index, "") + arguments
            ids = state.setdefault("ids", {})
            names = state.setdefault("names", {})
            call_id, name = call.get("id"), function.get("name")
            if call_id is not None and ids.get(index) == call_id:
                call_id = None
            elif call_id is not None:
                ids[index] = ids.get(index, "") + call_id
            if name is not None and names.get(index) == name:
                name = None
            elif name is not None:
                names[index] = names.get(index, "") + name
            calls.append(tool_call_chunk(index=index, id=call_id, name=name, args=arguments))
        finish = choice.get("finish_reason")
        if finish:
            state["finished"] = True
            for raw in state["arguments"].values():
                try:
                    if not isinstance(json.loads(raw), dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "The provider returned incomplete or malformed tool arguments."
                    ) from error
        reasoning = delta.get("reasoning_content")
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=delta.get("content") or "",
                tool_call_chunks=calls,
                additional_kwargs={"reasoning_content": reasoning}
                if isinstance(reasoning, str)
                else {},
                response_metadata={"finish_reason": finish} if finish else {},
            )
        )

    def _stream_end(self, state: dict[str, Any]) -> ChatGenerationChunk | None:
        if not state["finished"]:
            raise TransientProviderError("The provider stream ended before its completion marker.")
        if "usage" not in state:
            return None
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=self._usage(state["usage"]),
                response_metadata={
                    "models_provider_usage": asdict(ModelUsage.from_mapping(state["usage"]))
                },
            )
        )

    def _stream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del run_manager
        parameters = self._parameters(stop=stop, **kwargs)
        parameters.update(
            stream=True,
            stream_options={**(parameters.get("stream_options") or {}), "include_usage": True},
        )
        try:
            response = litellm.completion(messages=self._messages(messages), **parameters)
        except Exception as error:
            self._request_error(error)
            raise
        state: dict[str, Any] = {"finished": False, "arguments": {}}
        try:
            for part in response:
                chunk = self._stream_chunk(part, state)
                if chunk is not None:
                    yield chunk
            final = self._stream_end(state)
            if final is not None:
                yield final
        except Exception as error:
            self._request_error(error)
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    async def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del run_manager
        if self._authentication is not None and self.provider_identifier:
            await self._authentication.ensure_valid(self.provider_identifier)
        parameters = self._parameters(stop=stop, **kwargs)
        parameters.update(
            stream=True,
            stream_options={**(parameters.get("stream_options") or {}), "include_usage": True},
        )
        try:
            response = await litellm.acompletion(messages=self._messages(messages), **parameters)
        except Exception as error:
            self._request_error(error)
            raise
        if not isinstance(response, AsyncIterable):
            raise RuntimeError("The provider did not return an asynchronous stream.")
        state: dict[str, Any] = {"finished": False, "arguments": {}}
        try:
            async for part in response:
                chunk = self._stream_chunk(part, state)
                if chunk is not None:
                    yield chunk
            final = self._stream_end(state)
            if final is not None:
                yield final
        except Exception as error:
            self._request_error(error)
            raise
        finally:
            close = getattr(response, "aclose", None)
            if callable(close):
                result = close()
                if isawaitable(result):
                    await result

    def _request_error(self, error: Exception) -> None:
        message = str(error).strip() or "The provider request failed."
        if isinstance(error, ContextWindowExceededError):
            raise ContextWindowError(
                message, model=self.model, context_window=self.context_length
            ) from error
        if isinstance(
            error,
            (
                APIConnectionError,
                Timeout,
                RateLimitError,
                ServiceUnavailableError,
                InternalServerError,
            ),
        ):
            headers = getattr(getattr(error, "response", None), "headers", {})
            value = headers.get("retry-after") if isinstance(headers, Mapping) else None
            try:
                retry_after = max(0, float(value)) if value is not None else None
            except (TypeError, ValueError):
                retry_after = None
            raise TransientProviderError(message, retry_after=retry_after) from error

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        parameters = self._parameters(stop=stop, **kwargs)
        try:
            response = litellm.completion(messages=self._messages(messages), **parameters)
        except Exception as error:
            self._request_error(error)
            raise
        return self._response(response)

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        if self._authentication is not None and self.provider_identifier:
            await self._authentication.ensure_valid(self.provider_identifier)
        parameters = self._parameters(stop=stop, **kwargs)
        try:
            response = await litellm.acompletion(messages=self._messages(messages), **parameters)
        except Exception as error:
            self._request_error(error)
            raise
        return self._response(response)


class LiteLLM:
    """Concrete generic provider implementation backed by LiteLLM."""

    identifier = "litellm"
    oauth: OAuthProvider | None = None

    def supports(self, record: ModelRecord) -> bool:
        return True

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
        provider_identifier = provider.identifier.lower()
        is_opencode = provider_identifier in {"opencode", "opencode-go"}
        model_name = record.model.lower()
        model_identifier = (
            f"openai/responses/{record.model}"
            if is_opencode and provider.npm == "@ai-sdk/openai"
            else f"{_SDK_PREFIXES.get(provider.npm, 'openai')}/{record.model}"
        )
        temperature: float | None = None
        top_p: float | None = None
        maximum_tokens: int | None = None
        if is_opencode:
            if "claude" not in model_name:
                if any(
                    marker in model_name
                    for marker in (
                        "north-mini-code",
                        "gemini-2.5",
                        "gemini-3-",
                        "glm-4.6",
                        "glm-4.7",
                        "minimax-m2",
                    )
                ):
                    temperature = 1.0
                elif "kimi-k2" in model_name:
                    temperature = (
                        1.0
                        if any(
                            marker in model_name for marker in ("thinking", "k2.", "k2p", "k2-5")
                        )
                        else 0.6
                    )
            if any(
                marker in model_name
                for marker in (
                    "gemini-2.5",
                    "gemini-3-",
                    "minimax-m2",
                    "kimi-k2.5",
                    "kimi-k2p5",
                    "kimi-k2-5",
                    "deepseek-v4-flash",
                )
            ):
                top_p = 0.95
            maximum_tokens = (
                min(record.output_limit, _OPENCODE_MAXIMUM_OUTPUT_TOKENS)
                or _OPENCODE_MAXIMUM_OUTPUT_TOKENS
            )

        resolution = authentication.resolve(
            provider.identifier,
            environment_variables=provider.environment_variables,
        )
        if not resolution.available:
            raise AuthenticationError(f"No configured access is available for {record.provider!r}.")
        model = LiteLLMChatModel(
            model=model_identifier,
            api_base=provider.api_base or None,
            temperature=temperature,
            top_p=top_p,
            maximum_tokens=maximum_tokens,
            supports_temperature=record.temperature if is_opencode else True,
            timeout=timeout_seconds,
            context_length=record.context_length,
            input_modalities=record.input_modalities,
            request_context=OpenCodeRequestContext(session_id=uuid4().hex) if is_opencode else None,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
            request_parameters=dict(request_parameters),
        )
        model._authentication = authentication
        return model


__all__ = ["LiteLLM", "LiteLLMChatModel"]
