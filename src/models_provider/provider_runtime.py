"""Private provider-selection runtime used by the public model facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from .core import ModelCatalogue, ModelRecord, ProviderRecord
from .errors import AuthenticationError
from .litellm import LiteLLMChatModel, _SDK_PREFIXES
from .oauth import LoginFlow
from .openai_account import OpenAIAccountResponsesModel
from .provider_auth import ProviderAuthentication


class _ProviderImplementation(Protocol):
    def supports(self, record: ModelRecord) -> bool: ...

    def chat(
        self,
        record: ModelRecord,
        provider: ProviderRecord,
        *,
        values: dict[str, Any],
        authentication: ProviderAuthentication,
        timeout_seconds: float | None,
        request_parameters: Mapping[str, Any],
    ) -> BaseChatModel: ...


class _LiteLLMImplementation:
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


class _OpenAIImplementation:
    def __init__(self) -> None:
        self._generic = _LiteLLMImplementation()

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
        if not account_access:
            return self._generic.chat(
                record,
                provider,
                values=values,
                authentication=authentication,
                timeout_seconds=timeout_seconds,
                request_parameters=request_parameters,
            )
        authentication.token("openai")
        return OpenAIAccountResponsesModel(
            model=record.model,
            timeout=timeout_seconds,
            context_length=record.context_length,
            credential_values=values,
            request_parameters=dict(request_parameters),
        )


class ProviderRuntime:
    """Resolve authorization and provider implementations behind ``Models``."""

    def __init__(self, catalogue: ModelCatalogue) -> None:
        self._catalogue = catalogue
        self._implementations: tuple[_ProviderImplementation, ...] = (
            _OpenAIImplementation(),
            _LiteLLMImplementation(),
        )

    def authentication(self, values: dict[str, Any]) -> ProviderAuthentication:
        return ProviderAuthentication(values, catalogue=self._catalogue)

    def sign_in(self, provider: str, values: dict[str, Any]) -> LoginFlow:
        return self.authentication(values).flow(provider)

    def chat(
        self,
        record: ModelRecord,
        *,
        values: dict[str, Any],
        timeout_seconds: float | None,
        request_parameters: Mapping[str, Any],
    ) -> BaseChatModel:
        authentication = self.authentication(values)
        provider = self._catalogue.provider(record.provider)
        if provider is None:
            raise ValueError(f"Provider {record.provider!r} is missing from the catalogue.")
        for implementation in self._implementations:
            if implementation.supports(record):
                return implementation.chat(
                    record,
                    provider,
                    values=values,
                    authentication=authentication,
                    timeout_seconds=timeout_seconds,
                    request_parameters=request_parameters,
                )
        raise ValueError(f"No provider implementation supports {record.provider!r}.")
