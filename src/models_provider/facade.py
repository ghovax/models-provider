"""The concise application-facing model provider facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from .auth import (
    CredentialStore,
    OAuthAuthorization,
    ProviderAuthentication,
)
from .core import ModelCatalogue, ModelRecord


_MODELS_DEV_URL = "https://models.dev/api.json"


def _fetch_models(*, url: str, timeout_seconds: float, client: Any | None) -> ModelCatalogue:
    if timeout_seconds <= 0:
        raise ValueError("catalogue_timeout_seconds must be positive")
    import httpx

    client_was_created = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http_client.get(url)
        response.raise_for_status()
        payload: Any = response.json()
    finally:
        if client_was_created:
            http_client.close()
    if not isinstance(payload, Mapping):
        raise ValueError("models.dev returned a non-object catalogue")
    return ModelCatalogue.from_payload(payload)


class Models:
    """Discover, authenticate, and create models without exposing catalogue machinery."""

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        catalogue: ModelCatalogue | None = None,
        catalogue_url: str = _MODELS_DEV_URL,
        catalogue_timeout_seconds: float = 10.0,
        catalogue_client: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._catalogue = catalogue
        self._catalogue_url = catalogue_url
        self._catalogue_timeout_seconds = catalogue_timeout_seconds
        self._catalogue_client = catalogue_client
        self._authentication: ProviderAuthentication | None = None

    def _catalogue_snapshot(self) -> ModelCatalogue:
        if self._catalogue is None:
            self._catalogue = _fetch_models(
                url=self._catalogue_url,
                timeout_seconds=self._catalogue_timeout_seconds,
                client=self._catalogue_client,
            )
        return self._catalogue

    def _authentication_service(self) -> ProviderAuthentication:
        if self._authentication is None:
            self._authentication = ProviderAuthentication(
                catalogue=self._catalogue_snapshot(),
                store=self._credentials,
            )
        return self._authentication

    def list(self, provider: str | None = None) -> tuple[ModelRecord, ...]:
        """Return the available catalogue records, optionally filtered by provider."""
        return self._catalogue_snapshot().models(provider)

    def find(self, model_identifier: str) -> ModelRecord | None:
        """Find one provider-qualified model identifier."""
        return self._catalogue_snapshot().find(model_identifier)

    def chat(
        self,
        model_identifier: str,
        *,
        temperature: float = 0.0,
        reasoning_effort: str | None = "high",
        timeout_seconds: float | None = 300.0,
    ) -> BaseChatModel:
        """Create a ready-to-use chat model from ``provider/model``."""
        catalogue = self._catalogue_snapshot()
        if "/" not in model_identifier:
            raise ValueError("model_identifier must have the form 'provider/model'")
        provider_identifier, _model_suffix = model_identifier.split("/", 1)
        record = catalogue.find(model_identifier)
        provider = catalogue.provider(provider_identifier)
        if record is None or provider is None:
            raise ValueError(f"model {model_identifier!r} is not in the models.dev catalogue")
        from .litellm import LiteLLMChatModel, _SDK_PREFIXES

        model = LiteLLMChatModel(
            model=f"{_SDK_PREFIXES.get(provider.npm, 'openai')}/{record.model}",
            api_base=provider.api_base or None,
            temperature=temperature,
            timeout=timeout_seconds,
            reasoning_effort=reasoning_effort,
            context_length=record.context_length,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
        )
        model._authentication = self._authentication_service()
        return model

    async def sign_in(self, provider: str) -> OAuthAuthorization:
        """Prepare OAuth and return its URL; the host decides whether to display it."""
        flow = self._authentication_service().flow(provider)
        await flow.start()
        return OAuthAuthorization(flow)

    def authentication(self) -> ProviderAuthentication:
        """Return advanced authentication controls for hosts that need them."""
        return self._authentication_service()


__all__ = ["Models"]
