"""The application-facing model facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from .catalogue import ModelCatalogue, ModelRecord
from .auth import OAuthAuthorization
from .providers import ProviderRegistry


_MODELS_DEV_URL = "https://models.dev/api.json"


def _fetch_models(*, timeout_seconds: float) -> ModelCatalogue:
    """Fetch and parse the fixed models.dev catalogue endpoint."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    import httpx

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(_MODELS_DEV_URL)
        response.raise_for_status()
        payload: Any = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("models.dev returned a non-object catalogue")
    return ModelCatalogue.from_payload(payload)


class Models:
    """Select models using caller-provided provider values and hidden access resolution."""

    _provider_values: dict[str, Any]
    _catalogue: ModelCatalogue
    _runtime: ProviderRegistry

    def __init__(
        self,
        provider_values: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._provider_values = (
            provider_values if isinstance(provider_values, dict) else dict(provider_values or {})
        )
        self._catalogue = _fetch_models(timeout_seconds=timeout_seconds)
        self._runtime = ProviderRegistry(self._catalogue)

    def list(self, provider: str | None = None) -> tuple[ModelRecord, ...]:
        """Return catalogue records; the catalogue itself remains private."""
        return self._catalogue.models(provider)

    def find(self, model_identifier: str) -> ModelRecord | None:
        """Find one provider-qualified model identifier."""
        return self._catalogue.find(model_identifier)

    def chat(
        self,
        model_identifier: str,
        *,
        authorization: OAuthAuthorization | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a ready-to-use model from one provider-qualified identifier."""
        if "/" not in model_identifier:
            raise ValueError("model_identifier must have the form 'provider/model'")
        record = self._catalogue.require(model_identifier)

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
        return self._runtime.chat(
            record,
            values=values,
            timeout_seconds=timeout_seconds,
            request_parameters=parameters,
        )

    async def sign_in(self, provider: str) -> OAuthAuthorization:
        """Prepare OAuth and return its URL; the host decides how to display it."""
        values: dict[str, Any] = {}
        flow = self._runtime.sign_in(provider, values)
        await flow.start()
        return OAuthAuthorization(flow, values)


__all__ = ["Models"]
