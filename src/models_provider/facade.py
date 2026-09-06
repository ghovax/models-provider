"""The application-facing model facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from .core import ModelCatalogue, ModelRecord
from .errors import AuthenticationError
from .oauth import OAuthAuthorization
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
    """Select models using caller-provided provider values and hidden access resolution."""

    def __init__(
        self,
        provider_values: Mapping[str, Any] | None = None,
        *,
        catalogue_url: str = _MODELS_DEV_URL,
        catalogue_timeout_seconds: float = 10.0,
        catalogue_client: Any | None = None,
    ) -> None:
        self._provider_values = (
            provider_values if isinstance(provider_values, dict) else dict(provider_values or {})
        )
        self._catalogue: ModelCatalogue | None = None
        self._catalogue_url = catalogue_url
        self._catalogue_timeout_seconds = catalogue_timeout_seconds
        self._catalogue_client = catalogue_client

    def _catalogue_snapshot(self) -> ModelCatalogue:
        if self._catalogue is None:
            self._catalogue = _fetch_models(
                url=self._catalogue_url,
                timeout_seconds=self._catalogue_timeout_seconds,
                client=self._catalogue_client,
            )
        return self._catalogue

    def _authentication_service(self, values: dict[str, Any]) -> ProviderAuthentication:
        return ProviderAuthentication(values, catalogue=self._catalogue_snapshot())

    @staticmethod
    def _uses_openai_account_access(provider_identifier: str, values: Mapping[str, Any]) -> bool:
        """Select account access when the OpenAI value contains session data."""
        if provider_identifier != "openai":
            return False
        value = values.get("openai")
        if hasattr(value, "access_token"):
            return True
        if isinstance(value, Mapping):
            return not value or "access_token" in value
        return False

    def list(self, provider: str | None = None) -> tuple[ModelRecord, ...]:
        """Return catalogue records; the catalogue itself remains private."""
        return self._catalogue_snapshot().models(provider)

    def find(self, model_identifier: str) -> ModelRecord | None:
        """Find one provider-qualified model identifier."""
        return self._catalogue_snapshot().find(model_identifier)

    def chat(
        self,
        model_identifier: str,
        *,
        authorization: OAuthAuthorization | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a ready-to-use model from one provider-qualified identifier."""
        catalogue = self._catalogue_snapshot()
        if "/" not in model_identifier:
            raise ValueError("model_identifier must have the form 'provider/model'")
        provider_identifier, _model_suffix = model_identifier.split("/", 1)
        provider_identifier = provider_identifier.strip().lower()
        record = catalogue.require(model_identifier)
        provider = catalogue.provider(provider_identifier)
        if provider is None:
            raise ValueError(f"provider {provider_identifier!r} is not in the models.dev catalogue")

        values = (
            authorization.values
            if isinstance(authorization, OAuthAuthorization)
            else self._provider_values
            if authorization is None
            else dict(authorization)
        )
        parameters = dict(kwargs)
        reasoning_effort = parameters.get("reasoning_effort")
        if reasoning_effort is not None:
            parameters["reasoning_effort"] = record.validate_reasoning_effort(reasoning_effort)
        timeout_seconds = parameters.pop("timeout_seconds", 300.0)
        authentication = self._authentication_service(values)

        if self._uses_openai_account_access(provider_identifier, values):
            from .openai_account import OpenAIAccountResponsesModel

            authentication.token("openai")
            return OpenAIAccountResponsesModel(
                model=record.model,
                timeout=timeout_seconds,
                context_length=record.context_length,
                credential_values=values,
                request_parameters=parameters,
            )

        from .litellm import LiteLLMChatModel, _SDK_PREFIXES

        resolution = authentication.resolve(
            provider.identifier,
            environment_variables=provider.environment_variables,
        )
        if not resolution.available:
            raise AuthenticationError(
                f"No configured access is available for {provider_identifier!r}."
            )
        model = LiteLLMChatModel(
            model=f"{_SDK_PREFIXES.get(provider.npm, 'openai')}/{record.model}",
            api_base=provider.api_base or None,
            timeout=timeout_seconds,
            context_length=record.context_length,
            provider_identifier=provider.identifier,
            provider_environment_variables=provider.environment_variables,
            request_parameters=parameters,
        )
        model._authentication = authentication
        return model

    async def sign_in(self, provider: str) -> OAuthAuthorization:
        """Prepare OAuth and return its URL; the host decides how to display it."""
        values: dict[str, Any] = {}
        flow = self._authentication_service(values).flow(provider)
        await flow.start()
        return OAuthAuthorization(flow, values)


__all__ = ["Models"]
