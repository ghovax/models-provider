"""Provider contracts and the registry used to compose them."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

__all__ = ["ModelConfiguration", "ModelProvider", "ProviderRegistry"]


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """The model choice and request defaults an application gives a provider."""

    provider: str
    model: str
    reasoning_effort: str | None = "high"
    temperature: float = 0.0
    timeout_seconds: float | None = 300.0
    context_length: int = 0
    session_id: str = ""
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
        object.__setattr__(self, "session_id", self.session_id.strip())

    @property
    def identifier(self) -> str:
        """The stable provider/model identifier used in logs and cache keys."""
        return f"{self.provider}/{self.model}"


@runtime_checkable
class ModelProvider(Protocol):
    """Builds a LangChain chat model without prescribing credential or transport storage."""

    def create(
        self,
        configuration: ModelConfiguration,
        *,
        working_directory: Path,
        session_id: str = "",
    ) -> BaseChatModel:
        """Create a model for one request context."""
        ...

    def scope(self) -> Any:
        """Return a context manager that activates provider-local request state."""
        ...


ProviderFactory = Callable[[ModelConfiguration, Path, str], BaseChatModel]


class ProviderRegistry:
    """Maps provider names to application-owned model factories."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider: str, factory: ProviderFactory) -> None:
        """Register or replace one provider factory."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("provider cannot be empty")
        self._factories[name] = factory

    def create(
        self,
        configuration: ModelConfiguration,
        *,
        working_directory: Path,
        session_id: str = "",
    ) -> BaseChatModel:
        """Build a model using the factory registered for its provider."""
        try:
            factory = self._factories[configuration.provider]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"no model provider registered for {configuration.provider!r}; available: {available}"
            ) from error
        return factory(configuration, working_directory, session_id or configuration.session_id)

    def scope(self) -> Any:
        """A no-op scope so registries satisfy the same application contract as adapters."""
        return nullcontext()

    def providers(self) -> tuple[str, ...]:
        """Return registered provider names in stable order."""
        return tuple(sorted(self._factories))
