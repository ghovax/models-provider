"""Independent models.dev catalogue and interchangeable model implementations."""

from .catalogue import MODELS_DEV_API_URL, load_catalogue
from .core import (
    ModelCatalogue,
    ModelConfiguration,
    ModelProvider,
    ModelRecord,
    ProviderRecord,
    ProviderRegistry,
)
__all__ = [
    "MODELS_DEV_API_URL",
    "ModelCatalogue",
    "ModelConfiguration",
    "ModelProvider",
    "ModelRecord",
    "ProviderRecord",
    "ProviderRegistry",
    "load_catalogue",
    "LiteLLMChatModel",
    "LiteLLMProvider",
    "provider_registry",
]


def __getattr__(name: str):
    """Load the optional heavy transport implementation only when requested."""
    if name in {"LiteLLMChatModel", "LiteLLMProvider", "provider_registry"}:
        from .litellm import LiteLLMChatModel, LiteLLMProvider, provider_registry

        return {
            "LiteLLMChatModel": LiteLLMChatModel,
            "LiteLLMProvider": LiteLLMProvider,
            "provider_registry": provider_registry,
        }[name]
    raise AttributeError(name)
