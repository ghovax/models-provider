"""A small, independent LiteLLM-backed LangChain model implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import litellm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from ..auth import OAuthProvider, ProviderAuthentication
from ..catalogue import ModelRecord, ProviderRecord
from ..errors import AuthenticationError
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

    def context_window(self) -> int:
        return self.context_length

    @staticmethod
    def _message(message: BaseMessage) -> dict[str, Any]:
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
        item: dict[str, Any] = {"role": role, "content": message.content}
        if isinstance(message, ToolMessage):
            item["tool_call_id"] = message.tool_call_id
        return item

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
        if headers:
            params["extra_headers"] = headers
        params.update(
            {
                key: value
                for key, value in {**self.request_parameters, **kwargs}.items()
                if value is not None
            }
        )
        return params

    def _response(self, response: Any) -> ChatResult:
        choices = getattr(response, "choices", ()) or ()
        if not choices:
            return ChatResult(generations=[])
        source = getattr(choices[0], "message", None)
        content = getattr(source, "content", "") or ""
        tool_calls: list[dict[str, Any]] = []
        for call in getattr(source, "tool_calls", ()) or ():
            function = getattr(call, "function", None)
            raw = getattr(function, "arguments", "{}") if function else "{}"
            try:
                arguments = json.loads(raw)
            except (TypeError, ValueError):
                arguments = raw
            tool_calls.append(
                {
                    "name": getattr(function, "name", ""),
                    "args": arguments,
                    "id": getattr(call, "id", ""),
                }
            )
        raw_usage = getattr(response, "usage", None)
        usage_payload = (
            raw_usage if isinstance(raw_usage, Mapping) else getattr(raw_usage, "__dict__", {})
        )
        usage = ModelUsage.from_mapping(usage_payload)
        usage_metadata = {
            "input_tokens": usage.tokens.input_tokens,
            "output_tokens": usage.tokens.output_tokens,
            "total_tokens": usage.tokens.total_tokens,
            "input_token_details": {
                "cache_read": usage.cache.cache_read_tokens,
                "cache_creation": usage.cache.cache_write_tokens,
            },
            "output_token_details": {"reasoning": usage.tokens.reasoning_tokens},
        }
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
            response_metadata={"models_provider_usage": asdict(usage)},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        parameters = self._parameters(stop=stop, **kwargs)
        response = litellm.completion(
            messages=[self._message(message) for message in messages], **parameters
        )
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
        response = await litellm.acompletion(
            messages=[self._message(message) for message in messages], **parameters
        )
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
            request_context=OpenCodeRequestContext(session_id=uuid4().hex) if is_opencode else None,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
            request_parameters=dict(request_parameters),
        )
        model._authentication = authentication
        return model


__all__ = ["LiteLLM", "LiteLLMChatModel"]
