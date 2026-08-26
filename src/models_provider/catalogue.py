"""Loading and querying the public models.dev catalogue."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import ModelCatalogue

MODELS_DEV_API_URL = "https://models.dev/api.json"


def load_catalogue(
    *,
    url: str = MODELS_DEV_API_URL,
    timeout_seconds: float = 10.0,
    client: Any | None = None,
) -> ModelCatalogue:
    """Fetch and parse a models.dev snapshot.

    Network access is explicit. Model construction never performs an implicit catalogue fetch;
    applications can load once at startup, cache the result, or provide a fixture payload.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    import httpx

    own_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http_client.get(url)
        response.raise_for_status()
        payload: Any = response.json()
    finally:
        if own_client:
            http_client.close()
    if not isinstance(payload, Mapping):
        raise ValueError("models.dev returned a non-object catalogue")
    return ModelCatalogue.from_payload(payload)


__all__ = ["MODELS_DEV_API_URL", "load_catalogue"]
