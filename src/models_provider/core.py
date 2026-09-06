"""Public model catalogue and provider-construction contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from .usage import ModelUsage

__all__ = [
    "ModelUsage",
    "ModelProvider",
    "ModelRecord",
    "ProviderRecord",
]


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    """Provider metadata published by models.dev."""

    identifier: str
    name: str
    npm: str = ""
    environment_variables: tuple[str, ...] = ()
    documentation_url: str = ""
    api_base: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


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
    reasoning_options: tuple[Mapping[str, Any], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, provider: str, model: str, payload: Mapping[str, Any]) -> "ModelRecord":
        direct_fields = {
            item.name
            for item in fields(cls)
            if item.name
            not in {
                "identifier",
                "provider",
                "model",
                "name",
                "input_modalities",
                "output_modalities",
                "context_length",
                "input_limit",
                "output_limit",
                "cost",
                "reasoning_options",
                "extra",
            }
        }
        values = {name: payload[name] for name in direct_fields if name in payload}
        modalities = payload.get("modalities")
        limits = payload.get("limit")
        costs = payload.get("cost")
        modalities = modalities if isinstance(modalities, Mapping) else {}
        limits = limits if isinstance(limits, Mapping) else {}
        costs = costs if isinstance(costs, Mapping) else {}
        input_modalities = tuple(_text(item) for item in modalities.get("input", ()) if _text(item))
        output_modalities = tuple(
            _text(item) for item in modalities.get("output", ()) if _text(item)
        )
        model_id = _text(payload.get("id")) or model
        normalized_cost: dict[str, float] = {}
        for name, value in costs.items():
            parsed = _number(value)
            if parsed is not None:
                normalized_cost[str(name)] = parsed
        values.update(
            {
                "identifier": f"{provider}/{model_id}",
                "provider": provider,
                "model": model_id,
                "name": _text(payload.get("name")) or model_id,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "context_length": _positive_int(limits.get("context")),
                "input_limit": _positive_int(limits.get("input")),
                "output_limit": _positive_int(limits.get("output")),
                "cost": normalized_cost,
                "reasoning_options": tuple(
                    item
                    for item in payload.get("reasoning_options", ())
                    if isinstance(item, Mapping)
                ),
                "extra": dict(payload),
            }
        )
        for name in (
            "description",
            "family",
            "release_date",
            "last_updated",
            "knowledge_cutoff",
            "status",
        ):
            if name in values:
                values[name] = _text(values[name])
        for name in (
            "reasoning",
            "tool_call",
            "attachment",
            "structured_output",
            "temperature",
            "open_weights",
        ):
            if name in values:
                values[name] = bool(values[name])
        return cls(**values)

    def validate_reasoning_effort(self, reasoning_effort: str | None) -> str | None:
        """Validate a requested reasoning effort against catalogue metadata."""
        if reasoning_effort is None:
            return None
        normalized = reasoning_effort.strip()
        if not normalized:
            raise ValueError("reasoning_effort must be a non-empty string or None")
        supported_values: list[str] = []
        for option in self.reasoning_options:
            if _text(option.get("type")) != "effort":
                continue
            raw_values = option.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(
                raw_values, (str, bytes, bytearray)
            ):
                continue
            supported_values.extend(
                value for value in (_text(item) for item in raw_values) if value
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
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelCatalogue":
        """Parse the object returned by ``https://models.dev/api.json``."""
        providers: list[ProviderRecord] = []
        models: list[ModelRecord] = []
        for raw_identifier, raw_provider in payload.items():
            if not isinstance(raw_provider, Mapping):
                continue
            identifier = _text(raw_provider.get("id")) or _text(raw_identifier)
            if not identifier:
                continue
            provider = ProviderRecord(
                identifier=identifier,
                name=_text(raw_provider.get("name")) or identifier,
                npm=_text(raw_provider.get("npm")),
                environment_variables=tuple(
                    _text(item) for item in raw_provider.get("env", ()) if _text(item)
                ),
                documentation_url=_text(raw_provider.get("doc")),
                api_base=_text(raw_provider.get("api")),
                extra=dict(raw_provider),
            )
            providers.append(provider)
            raw_models = raw_provider.get("models") or {}
            if not isinstance(raw_models, Mapping):
                continue
            for raw_model, raw_definition in raw_models.items():
                if isinstance(raw_definition, Mapping):
                    models.append(
                        ModelRecord.from_payload(identifier, _text(raw_model), raw_definition)
                    )
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
        authorization: Any | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a model using the selected user's authorization and keyword options."""
        ...
