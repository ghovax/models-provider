"""Public model catalogue and provider-construction contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from .usage import ModelUsage

__all__ = [
    "ModelConfiguration",
    "ModelUsage",
    "ModelProvider",
    "ModelRecord",
    "ModelCatalogue",
    "ProviderRecord",
    "ProviderRegistry",
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
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    reasoning_cost_per_million: float | None = None
    cache_read_cost_per_million: float | None = None
    cache_write_cost_per_million: float | None = None
    release_date: str = ""
    last_updated: str = ""
    knowledge_cutoff: str = ""
    status: str = ""
    reasoning_options: tuple[Mapping[str, Any], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, provider: str, model: str, payload: Mapping[str, Any]) -> "ModelRecord":
        modalities = payload.get("modalities") or {}
        limits = payload.get("limit") or {}
        costs = payload.get("cost") or {}
        input_modalities = tuple(_text(item) for item in modalities.get("input", ()) if _text(item))
        output_modalities = tuple(_text(item) for item in modalities.get("output", ()) if _text(item))
        model_id = _text(payload.get("id")) or model
        return cls(
            identifier=f"{provider}/{model_id}",
            provider=provider,
            model=model_id,
            name=_text(payload.get("name")) or model_id,
            description=_text(payload.get("description")),
            family=_text(payload.get("family")),
            reasoning=bool(payload.get("reasoning")),
            tool_call=bool(payload.get("tool_call")),
            attachment=bool(payload.get("attachment")),
            structured_output=bool(payload.get("structured_output")),
            temperature=bool(payload.get("temperature")),
            open_weights=bool(payload.get("open_weights")),
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            context_length=_positive_int(limits.get("context")),
            input_limit=_positive_int(limits.get("input")),
            output_limit=_positive_int(limits.get("output")),
            input_cost_per_million=_number(costs.get("input")),
            output_cost_per_million=_number(costs.get("output")),
            reasoning_cost_per_million=_number(costs.get("reasoning")),
            cache_read_cost_per_million=_number(costs.get("cache_read")),
            cache_write_cost_per_million=_number(costs.get("cache_write")),
            release_date=_text(payload.get("release_date")),
            last_updated=_text(payload.get("last_updated")),
            knowledge_cutoff=_text(payload.get("knowledge")),
            status=_text(payload.get("status")),
            reasoning_options=tuple(
                item for item in payload.get("reasoning_options", ()) if isinstance(item, Mapping)
            ),
            extra=dict(payload),
        )


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
                    models.append(ModelRecord.from_payload(identifier, _text(raw_model), raw_definition))
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

    def provider(self, identifier: str) -> ProviderRecord | None:
        return self._providers.get(identifier.strip().lower())

    def __len__(self) -> int:
        return len(self._models)


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """The model choice and request defaults an application gives a provider."""

    provider: str
    model: str
    reasoning_effort: str | None = "high"
    temperature: float = 0.0
    timeout_seconds: float | None = 300.0
    context_length: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model = self.model.strip()
        if not provider or not model:
            raise ValueError("provider and model cannot be empty")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when set")
        if self.context_length < 0:
            raise ValueError("context_length cannot be negative")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)

    @property
    def identifier(self) -> str:
        return f"{self.provider}/{self.model}"


@runtime_checkable
class ModelProvider(Protocol):
    """Builds a chat model from a provider-qualified model configuration."""

    def create(self, configuration: ModelConfiguration) -> BaseChatModel:
        """Create a model; credentials and transport state belong to the provider instance."""
        ...

    def scope(self) -> Any:
        """Return a context manager for temporary provider-local state."""
        ...


ProviderFactory = Callable[[ModelConfiguration, ModelRecord | None], BaseChatModel]


class ProviderRegistry:
    """Dispatches model configurations to concrete provider implementations."""

    def __init__(self, catalogue: ModelCatalogue | None = None) -> None:
        self.catalogue = catalogue or ModelCatalogue()
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider: str, factory: ProviderFactory) -> None:
        name = provider.strip().lower()
        if not name:
            raise ValueError("provider cannot be empty")
        self._factories[name] = factory

    def create(self, configuration: ModelConfiguration) -> BaseChatModel:
        try:
            factory = self._factories[configuration.provider]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"no model implementation registered for {configuration.provider!r}; available: {available}"
            ) from error
        return factory(configuration, self.catalogue.find(configuration.identifier))

    def scope(self) -> Any:
        return nullcontext()

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
