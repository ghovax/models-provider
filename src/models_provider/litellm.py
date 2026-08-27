"""A small, independent LiteLLM-backed LangChain model implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import litellm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr

from .auth import ProviderAuthentication
from .usage import ModelUsage


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


class LiteLLMChatModel(BaseChatModel):
    """A provider-qualified model usable by any LangChain-compatible application."""

    model: str
    api_key: SecretStr | None = None
    api_base: str | None = None
    temperature: float = 0.0
    timeout: float | None = 300.0
    reasoning_effort: str | None = None
    context_length: int = 0
    default_headers: dict[str, str] = Field(default_factory=dict)

    provider_identifier: str = ""
    provider_environment_variables: tuple[str, ...] = ()
    _authentication: ProviderAuthentication | None = None

    @property
    def _llm_type(self) -> str:
        return "models-provider-litellm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "api_base": self.api_base, "temperature": self.temperature}

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
        params: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        resolved = None
        if self._authentication is not None and self.provider_identifier:
            resolved = self._authentication.resolve(
                self.provider_identifier,
                environment_variables=self.provider_environment_variables,
            )
        if resolved is not None and resolved.api_key:
            params["api_key"] = resolved.api_key
        elif self.api_key is not None:
            params["api_key"] = self.api_key.get_secret_value()
        if resolved is not None and resolved.api_base:
            params["api_base"] = resolved.api_base
        elif self.api_base:
            params["api_base"] = self.api_base
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        headers = dict(self.default_headers)
        if resolved is not None:
            headers = {**resolved.headers, **headers}
        if headers:
            params["extra_headers"] = headers
        params.update({key: value for key, value in kwargs.items() if value is not None})
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
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "input_token_details": {
                "cache_read": usage.cache_read_tokens,
                "cache_creation": usage.cache_write_tokens,
            },
            "output_token_details": {"reasoning": usage.reasoning_tokens},
        }
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
            response_metadata={"models_provider_usage": asdict(usage)},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        parameters = self._parameters(stop=stop, **kwargs)
        response = litellm.completion(
            messages=[self._message(message) for message in messages], **parameters
        )
        return self._response(response)

    async def _agenerate(
        self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        if self._authentication is not None and self.provider_identifier:
            await self._authentication.ensure_valid(self.provider_identifier)
        parameters = self._parameters(stop=stop, **kwargs)
        response = await litellm.acompletion(
            messages=[self._message(message) for message in messages], **parameters
        )
        return self._response(response)


__all__ = ["LiteLLMChatModel"]
