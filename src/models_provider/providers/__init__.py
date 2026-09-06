"""Concrete provider implementations and their internal dispatch contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from ..auth import LoginFlow, OAuthProvider, ProviderAuthentication
from ..catalogue import ModelCatalogue, ModelRecord, ProviderRecord
from .cursor import Cursor
from .litellm import LiteLLM
from .openai import OpenAI


class Provider(Protocol):
    identifier: str

    @property
    def oauth(self) -> OAuthProvider | None: ...

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


class ProviderRegistry:
    """Resolve concrete providers behind the public ``Models`` client."""

    _catalogue: ModelCatalogue
    _providers: tuple[Provider, ...]

    def __init__(self, catalogue: ModelCatalogue) -> None:
        self._catalogue = catalogue
        self._providers: tuple[Provider, ...] = (OpenAI(), Cursor(), LiteLLM())

    def authentication(self, values: dict[str, Any]) -> ProviderAuthentication:
        oauth_adapters = {
            provider.identifier: provider.oauth
            for provider in self._providers
            if provider.oauth is not None
        }
        return ProviderAuthentication(
            values,
            catalogue=self._catalogue,
            oauth_adapters=oauth_adapters,
        )

    def sign_in(self, provider: str, values: dict[str, Any]) -> LoginFlow:
        provider = provider.strip().lower()
        implementation = next(
            (candidate for candidate in self._providers if candidate.identifier == provider),
            None,
        )
        if implementation is None or implementation.oauth is None:
            raise ValueError(f"Provider {provider!r} does not support OAuth.")
        return implementation.oauth.flow(values)

    def chat(
        self,
        record: ModelRecord,
        *,
        values: dict[str, Any],
        timeout_seconds: float | None,
        request_parameters: Mapping[str, Any],
    ) -> BaseChatModel:
        provider = self._catalogue.provider(record.provider)
        if provider is None:
            raise ValueError(f"Provider {record.provider!r} is missing from the catalogue.")
        authentication = self.authentication(values)
        for implementation in self._providers:
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


__all__ = ["Provider", "ProviderRegistry"]
