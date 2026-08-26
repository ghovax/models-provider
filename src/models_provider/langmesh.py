"""Adapter for LangMesh's configured model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel

from models_provider.core import ModelSpec

__all__ = ["LangMeshProvider"]


class LangMeshProvider:
    """Expose LangMesh models through the storage-neutral provider contract."""

    def __init__(
        self,
        *,
        configuration: Any | None = None,
        providers: Mapping[str, str | Mapping[str, str]] | None = None,
        credential_store: Any | None = None,
    ) -> None:
        from langmesh import Configuration

        self._configuration = configuration or Configuration()
        self._providers = dict(providers or {})
        self._credential_store = credential_store

    def create(
        self,
        spec: ModelSpec,
        *,
        working_directory: Path,
        session_id: str = "",
    ) -> BaseChatModel:
        from langmesh import AgentConfiguration
        from langmesh.base.configuration import ProviderCredential
        from langmesh.runtime.runtime import build_chat_model

        configuration = self._configuration.model_copy(deep=True)
        for provider, value in self._providers.items():
            credential = configuration.providers.get(provider) or ProviderCredential()
            update = {"api_key": value} if isinstance(value, str) else dict(value)
            configuration.providers[provider] = credential.model_copy(update=update)
        agent = AgentConfiguration(
            provider=spec.provider,
            model=spec.model,
            reasoning_effort=spec.reasoning_effort or "high",
        )
        model = build_chat_model(
            spec.identifier,
            configuration,
            agent,
            str(working_directory.resolve()),
            session_id=session_id or spec.session_id,
        )
        updates = {
            key: value
            for key, value in {
                "temperature": spec.temperature,
                "timeout": spec.timeout_seconds,
                "context_length": spec.context_length,
                **dict(spec.extra),
            }.items()
            if key in model.model_fields
        }
        return model.model_copy(update=updates) if updates else model

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Bind LangMesh's caller-owned credential store for native providers."""
        if self._credential_store is None:
            yield
            return
        from langmesh.base.identity.credential_store import (
            bind_credential_store,
            reset_credential_store,
        )

        token = bind_credential_store(self._credential_store)
        try:
            yield
        finally:
            reset_credential_store(token)
