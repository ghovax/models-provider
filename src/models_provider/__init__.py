"""Public model-selection and authorization interface."""

from .auth import (
    DeviceLoginFlow,
    HostedAuthorization,
    LoginFlow,
    OAuthAdapter,
    OAuthAuthorization,
    OAuthAuthorizationRequest,
    OAuthConfiguration,
    OAuthLoginFlow,
    OAuthProvider,
    OAuthTokens,
)
from .catalogue import ModelProvider, ModelRecord, ProviderRecord
from .client import Models
from .providers.litellm import OpenCodeRequestContext
from .errors import AuthenticationError, ContextWindowError, TransientProviderError
from .usage import (
    AudioUsage,
    CacheUsage,
    CostUsage,
    ModelUsage,
    TokenUsage,
    UsageLedger,
    UsageSnapshot,
    UsageWindow,
)


__all__ = [
    "AuthenticationError",
    "AudioUsage",
    "CacheUsage",
    "ContextWindowError",
    "CostUsage",
    "DeviceLoginFlow",
    "HostedAuthorization",
    "LoginFlow",
    "ModelProvider",
    "ModelRecord",
    "ModelUsage",
    "Models",
    "OAuthAdapter",
    "OAuthAuthorization",
    "OAuthAuthorizationRequest",
    "OAuthConfiguration",
    "OAuthLoginFlow",
    "OAuthProvider",
    "OAuthTokens",
    "ProviderRecord",
    "OpenCodeRequestContext",
    "TokenUsage",
    "TransientProviderError",
    "UsageLedger",
    "UsageSnapshot",
    "UsageWindow",
]
