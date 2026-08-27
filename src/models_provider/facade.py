"""The concise application-facing model provider facade."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from .auth import (
    CredentialStore,
    OAuthAuthorization,
    ProviderAuthentication,
)
from .catalogue import MODELS_DEV_API_URL, _fetch_catalogue
from .core import ModelCatalogue, ModelRecord


class Models:
    """Discover, authenticate, and create models without exposing catalogue machinery."""

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        catalogue: ModelCatalogue | None = None,
        catalogue_url: str = MODELS_DEV_API_URL,
        catalogue_timeout_seconds: float = 10.0,
        catalogue_client: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._catalogue = catalogue
        self._catalogue_url = catalogue_url
        self._catalogue_timeout_seconds = catalogue_timeout_seconds
        self._catalogue_client = catalogue_client
        self._authentication: ProviderAuthentication | None = None
        self._provider: Any | None = None

    def _catalogue_snapshot(self) -> ModelCatalogue:
        if self._catalogue is None:
            self._catalogue = _fetch_catalogue(
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

    def _model_provider(self) -> Any:
        if self._provider is None:
            from .litellm import LiteLLMProvider

            self._provider = LiteLLMProvider(
                self._catalogue_snapshot(),
                authentication=self._authentication_service(),
            )
        return self._provider

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
        return self._model_provider().chat(
            model_identifier,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )

    async def sign_in(self, provider: str) -> OAuthAuthorization:
        """Prepare OAuth and return its URL; the host decides whether to display it."""
        flow = self._authentication_service().flow(provider)
        await flow.start()
        return OAuthAuthorization(flow)

    def authentication(self) -> ProviderAuthentication:
        """Return advanced authentication controls for hosts that need them."""
        return self._authentication_service()


__all__ = ["Models"]
