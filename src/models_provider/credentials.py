"""Credential storage and task-local credential selection."""

from __future__ import annotations

import contextvars
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any


class CredentialStore(ABC):
    """Abstract storage contract supplied by the embedding application."""

    @classmethod
    def from_mapping(cls, credentials: Mapping[str, Any]) -> "InMemoryCredentialStore":
        """Build an in-memory store from provider credentials."""
        return InMemoryCredentialStore(credentials)

    @abstractmethod
    def load(self, provider_identifier: str) -> Any:
        """Load credentials for one provider, or return ``None``."""
        raise NotImplementedError

    @abstractmethod
    def save(self, provider_identifier: str, credentials: Any) -> None:
        """Persist credentials for one provider."""
        raise NotImplementedError

    @abstractmethod
    def clear(self, provider_identifier: str) -> None:
        """Remove credentials for one provider."""
        raise NotImplementedError


class InMemoryCredentialStore(CredentialStore):
    """Small concrete store for embedded applications and isolated mock runs."""

    def __init__(self, credentials: Mapping[str, Any] | None = None) -> None:
        self._credentials: dict[str, Any] = {}
        for provider_identifier, credential in (credentials or {}).items():
            self.save(provider_identifier, credential)

    def load(self, provider_identifier: str) -> Any:
        value = self._credentials.get(provider_identifier)
        if value is None:
            return None
        return replace(value) if hasattr(value, "__dataclass_fields__") else value

    def save(self, provider_identifier: str, credentials: Any) -> None:
        self._credentials[provider_identifier] = credentials

    def clear(self, provider_identifier: str) -> None:
        self._credentials.pop(provider_identifier, None)


@dataclass(frozen=True, slots=True)
class ApiKeyCredential:
    """An API key held by the application's credential store."""

    api_key: str


@dataclass(frozen=True, slots=True)
class EnvironmentCredential:
    """Named provider environment values held by an application's credential store."""

    values: Mapping[str, str]


_default_store = InMemoryCredentialStore()

_store_context: contextvars.ContextVar[CredentialStore] = contextvars.ContextVar(
    "models_provider_credential_store", default=_default_store
)


def current_credential_store() -> CredentialStore:
    """Return the store bound to the current application/task."""
    return _store_context.get()


def bind_credential_store(store: CredentialStore) -> contextvars.Token[CredentialStore]:
    """Bind caller-owned credential storage for the current task."""
    return _store_context.set(store)


def reset_credential_store(token: contextvars.Token[CredentialStore]) -> None:
    """Restore the previous credential-store binding."""
    _store_context.reset(token)
