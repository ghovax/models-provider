"""Concrete provider implementations and their internal dispatch contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from ..auth import ProviderAuthentication
from ..catalogue import ModelCatalogue, ModelRecord, ProviderRecord
from ..auth import LoginFlow, OAuthProvider
from .cursor import Cursor, cursor_oauth_adapter
from .litellm import LiteLLM
from .openai import OpenAI, openai_oauth_adapter


class Provider(Protocol):
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

    def __init__(self, catalogue: ModelCatalogue) -> None:
        self._catalogue = catalogue
        self._providers: tuple[Provider, ...] = (OpenAI(), Cursor(), LiteLLM())

    def authentication(self, values: dict[str, Any]) -> ProviderAuthentication:
        return ProviderAuthentication(
            values,
            catalogue=self._catalogue,
            oauth_adapters=default_oauth_adapters(),
        )

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


def default_oauth_adapters() -> dict[str, OAuthProvider]:
    return {
        "openai": openai_oauth_adapter(),
        "cursor": cursor_oauth_adapter(),
    }


__all__ = ["Provider", "ProviderRegistry", "default_oauth_adapters"]
