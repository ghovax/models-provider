"""The concise application-facing model provider facade."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from .credentials import CredentialStore
from .core import ModelCatalogue, ModelRecord
from .oauth import OAuthAuthorization
from .opencode import OpenCodeRequestContext, opencode_max_output_tokens, opencode_top_p
from .provider_auth import ProviderAuthentication


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
        environment: Mapping[str, str] | None = None,
        catalogue: ModelCatalogue | None = None,
        catalogue_url: str = _MODELS_DEV_URL,
        catalogue_timeout_seconds: float = 10.0,
        catalogue_client: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._environment = dict(environment or {})
        self._catalogue = catalogue
        self._catalogue_url = catalogue_url
        self._catalogue_timeout_seconds = catalogue_timeout_seconds
        self._catalogue_client = catalogue_client
        self._authentication: ProviderAuthentication | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        environment: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "Models":
        """Build a model facade from an explicit snapshot of process variables."""
        return cls(environment=dict(os.environ if environment is None else environment), **kwargs)

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
                environment=self._environment,
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
        if provider_identifier == "chatgpt":
            from .chatgpt import ChatGPTResponsesModel

            return ChatGPTResponsesModel(
                model=record.model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout_seconds,
                context_length=record.context_length,
                credential_store=self._credentials,
            )
        from .litellm import LiteLLMChatModel, _SDK_PREFIXES

        is_opencode = provider_identifier.strip().lower().startswith("opencode")
        litellm_prefix = _SDK_PREFIXES.get(provider.npm, "openai")
        if is_opencode and provider.npm == "@ai-sdk/openai":
            litellm_prefix = "openai/responses"

        model = LiteLLMChatModel(
            model=f"{litellm_prefix}/{record.model}",
            api_base=provider.api_base or None,
            temperature=temperature,
            top_p=opencode_top_p(record.model) if is_opencode else None,
            maximum_tokens=opencode_max_output_tokens(record.output_limit) if is_opencode else None,
            supports_temperature=record.temperature if is_opencode else True,
            timeout=timeout_seconds,
            reasoning_effort=reasoning_effort,
            context_length=record.context_length,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
            request_context=OpenCodeRequestContext(session_id=uuid4().hex) if is_opencode else None,
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
