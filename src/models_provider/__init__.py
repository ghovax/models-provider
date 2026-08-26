"""A small boundary between applications and chat-model implementations."""

from models_provider.core import ModelConfiguration, ModelProvider, ProviderRegistry

__all__ = ["ModelConfiguration", "ModelProvider", "ProviderRegistry"]
