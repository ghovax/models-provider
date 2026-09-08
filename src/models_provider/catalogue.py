"""Public model catalogue and provider-construction contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .auth import OAuthAuthorization

__all__ = [
    "ModelProvider",
    "ModelRecord",
    "ProviderRecord",
]


class _ModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str = ""
    description: str = ""
    family: str = ""
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False
    structured_output: bool = False
    temperature: bool = False
    open_weights: bool = False
    modalities: Mapping[str, Sequence[str]] = Field(default_factory=dict)
    limit: Mapping[str, int] = Field(default_factory=dict)
    cost: Mapping[str, object] = Field(default_factory=dict)
    release_date: str = ""
    last_updated: str = ""
    knowledge_cutoff: str = Field(default="", validation_alias="knowledge")
    status: str = ""
    reasoning_options: tuple[Mapping[str, object], ...] = ()


class _ProviderPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str = ""
    npm: str = ""
    env: tuple[str, ...] = ()
    doc: str = ""
    api: str = ""
    models: Mapping[str, object] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    """Provider metadata published by models.dev."""

    identifier: str
    name: str
    npm: str = ""
    environment_variables: tuple[str, ...] = ()
    documentation_url: str = ""
    api_base: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One provider-served model and its models.dev capabilities."""

    identifier: str
    provider: str
    model: str
    name: str
    description: str = ""
    family: str = ""
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False
    structured_output: bool = False
    temperature: bool = False
    open_weights: bool = False
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    context_length: int = 0
    input_limit: int = 0
    output_limit: int = 0
    cost: Mapping[str, float | None] = field(default_factory=dict)
    release_date: str = ""
    last_updated: str = ""
    knowledge_cutoff: str = ""
    status: str = ""
    reasoning_options: tuple[Mapping[str, object], ...] = ()
    extra: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls, provider: str, model: str, payload: Mapping[str, object]
    ) -> "ModelRecord":
        try:
            definition = _ModelPayload.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"invalid model metadata for {provider}/{model}") from error
        model_id = definition.id.strip() or model
        modalities = definition.modalities
        limits = definition.limit
        normalized_cost: dict[str, float] = {}
        for key, value in definition.cost.items():
            if isinstance(value, (int, float)):
                normalized_cost[key] = float(value)
        return cls(
            identifier=f"{provider}/{model_id}",
            provider=provider,
            model=model_id,
            name=definition.name.strip() or model_id,
            description=definition.description.strip(),
            family=definition.family.strip(),
            reasoning=definition.reasoning,
            tool_call=definition.tool_call,
            attachment=definition.attachment,
            structured_output=definition.structured_output,
            temperature=definition.temperature,
            open_weights=definition.open_weights,
            input_modalities=tuple(
                item.strip() for item in modalities.get("input", ()) if item.strip()
            ),
            output_modalities=tuple(
                item.strip() for item in modalities.get("output", ()) if item.strip()
            ),
            context_length=max(0, limits.get("context", 0)),
            input_limit=max(0, limits.get("input", 0)),
            output_limit=max(0, limits.get("output", 0)),
            cost=normalized_cost,
            release_date=definition.release_date.strip(),
            last_updated=definition.last_updated.strip(),
            knowledge_cutoff=definition.knowledge_cutoff.strip(),
            status=definition.status.strip(),
            reasoning_options=definition.reasoning_options,
            extra=dict(payload),
        )

    def validate_reasoning_effort(self, reasoning_effort: str | None) -> str | None:
        """Validate a requested reasoning effort against catalogue metadata."""
        if reasoning_effort is None:
            return None
        normalized = reasoning_effort.strip()
        if not normalized:
            raise ValueError("reasoning_effort must be a non-empty string or None")
        supported_values: list[str] = []
        for option in self.reasoning_options:
            if str(option.get("type") or "").strip() != "effort":
                continue
            raw_values = option.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(
                raw_values, (str, bytes, bytearray)
            ):
                continue
            supported_values.extend(
                str(value).strip() for value in raw_values if str(value).strip()
            )
        supported = tuple(dict.fromkeys(supported_values))
        if supported and normalized not in supported:
            choices = ", ".join(supported)
            raise ValueError(
                f"reasoning_effort {reasoning_effort!r} is not supported for "
                f"{self.identifier}; choose one of: {choices}"
            )
        return normalized


class ModelCatalogue:
    """An immutable, searchable snapshot of models.dev provider/model data."""

    def __init__(
        self,
        providers: Sequence[ProviderRecord] = (),
        models: Sequence[ModelRecord] = (),
    ) -> None:
        self._providers = {provider.identifier: provider for provider in providers}
        self._models = {model.identifier: model for model in models}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ModelCatalogue":
        """Parse the object returned by ``https://models.dev/api.json``."""
        providers: list[ProviderRecord] = []
        models: list[ModelRecord] = []
        for raw_identifier, raw_provider in payload.items():
            if not isinstance(raw_provider, Mapping):
                continue
            try:
                provider_data = _ProviderPayload.model_validate(raw_provider)
            except ValidationError:
                continue
            identifier = provider_data.id.strip() or str(raw_identifier).strip()
            if not identifier:
                continue
            provider = ProviderRecord(
                identifier=identifier,
                name=provider_data.name.strip() or identifier,
                npm=provider_data.npm.strip(),
                environment_variables=tuple(
                    item.strip() for item in provider_data.env if item.strip()
                ),
                documentation_url=provider_data.doc.strip(),
                api_base=provider_data.api.strip(),
                extra=dict(raw_provider),
            )
            providers.append(provider)
            for raw_model, definition in provider_data.models.items():
                if not isinstance(definition, Mapping):
                    continue
                try:
                    models.append(ModelRecord.from_payload(identifier, raw_model, definition))
                except ValueError:
                    continue
        return cls(providers, models)

    def providers(self) -> tuple[ProviderRecord, ...]:
        return tuple(self._providers[key] for key in sorted(self._providers))

    def models(self, provider: str | None = None) -> tuple[ModelRecord, ...]:
        values = self._models.values()
        if provider is not None:
            values = (model for model in values if model.provider == provider.strip().lower())
        return tuple(sorted(values, key=lambda model: model.identifier))

    def find(self, identifier: str) -> ModelRecord | None:
        return self._models.get(identifier.strip())

    def require(self, identifier: str) -> ModelRecord:
        """Return one model or raise a clear selection error."""
        record = self.find(identifier)
        if record is None:
            raise ValueError(f"model {identifier!r} is not in the models.dev catalogue")
        return record

    def provider(self, identifier: str) -> ProviderRecord | None:
        return self._providers.get(identifier.strip().lower())

    def __len__(self) -> int:
        return len(self._models)


@runtime_checkable
class ModelProvider(Protocol):
    """Creates chat models from provider-qualified identifiers and keyword options."""

    def chat(
        self,
        model_identifier: str,
        *,
        authorization: OAuthAuthorization | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a model using the selected user's authorization and keyword options."""
        ...
