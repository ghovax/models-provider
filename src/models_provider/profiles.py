"""Provider authentication profiles and key-resolution values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .opencode import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_ZEN_BASE_URL,
    opencode_default_headers,
)


@dataclass(frozen=True, slots=True)
class AuthenticationStatus:
    """Safe account state; it intentionally contains no keys or token values."""

    provider: str
    method: str
    signed_in: bool = False
    expired: bool = False
    account: str = ""
    source: str = "none"


@dataclass(frozen=True, slots=True)
class ApiKeyResolution:
    """The non-secret request settings resolved for one provider."""

    provider: str
    api_key: str = ""
    api_base: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    method: str = "api_key"
    source: str = "none"

    @property
    def available(self) -> bool:
        """Whether the resolved credentials can authorize or configure a provider call."""
        return (
            bool(self.api_key)
            if self.method == "api_key"
            else bool(self.api_key or self.environment)
        )


@dataclass(frozen=True, slots=True)
class ProviderAuthProfile:
    """Authentication metadata for a provider, independent of any model transport."""

    identifier: str
    environment_variables: tuple[str, ...] = ()
    credential_environment_variables: tuple[str, ...] = ()
    default_base_url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    method: str = "api_key"
    anonymous_api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    credential_identifier: str = ""


_AUTH_PROFILE_OVERRIDES: dict[str, ProviderAuthProfile] = {
    "anthropic": ProviderAuthProfile(
        "anthropic",
        environment_variables=("ANTHROPIC_API_KEY",),
        api_key_header="x-api-key",
        api_key_prefix="",
    ),
    "azure": ProviderAuthProfile(
        "azure",
        environment_variables=("AZURE_API_KEY",),
        credential_environment_variables=("AZURE_RESOURCE_NAME",),
        api_key_header="api-key",
        api_key_prefix="",
    ),
    "chatgpt": ProviderAuthProfile("chatgpt", method="oauth"),
    "commandcode": ProviderAuthProfile(
        "commandcode",
        environment_variables=("COMMAND_CODE_API_KEY",),
        default_base_url="https://api.commandcode.ai/provider/v1",
    ),
    "cursor": ProviderAuthProfile("cursor", method="oauth"),
    "custom": ProviderAuthProfile("custom"),
    "google": ProviderAuthProfile(
        "google",
        environment_variables=(
            "GOOGLE_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
            "GEMINI_API_KEY",
        ),
        api_key_header="x-goog-api-key",
        api_key_prefix="",
    ),
    "google-vertex": ProviderAuthProfile(
        "google-vertex",
        method="environment",
        credential_environment_variables=(
            "GOOGLE_VERTEX_PROJECT",
            "GOOGLE_VERTEX_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
    ),
    "google-vertex-anthropic": ProviderAuthProfile(
        "google-vertex-anthropic",
        method="environment",
        credential_environment_variables=(
            "GOOGLE_VERTEX_PROJECT",
            "GOOGLE_VERTEX_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
    ),
    "amazon-bedrock": ProviderAuthProfile(
        "amazon-bedrock",
        method="environment",
        credential_environment_variables=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_BEARER_TOKEN_BEDROCK",
        ),
    ),
    "bedrock": ProviderAuthProfile(
        "bedrock",
        method="environment",
        credential_environment_variables=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_BEARER_TOKEN_BEDROCK",
        ),
    ),
    "github-copilot": ProviderAuthProfile(
        "github-copilot",
        environment_variables=("GITHUB_TOKEN",),
        default_base_url="https://api.githubcopilot.com",
    ),
    "azure-cognitive-services": ProviderAuthProfile(
        "azure-cognitive-services",
        environment_variables=("AZURE_COGNITIVE_SERVICES_API_KEY",),
        credential_environment_variables=("AZURE_COGNITIVE_SERVICES_RESOURCE_NAME",),
        api_key_header="api-key",
        api_key_prefix="",
    ),
    "opencode": ProviderAuthProfile(
        "opencode",
        environment_variables=("OPENCODE_API_KEY",),
        default_base_url=OPENCODE_ZEN_BASE_URL,
        headers=opencode_default_headers(),
        anonymous_api_key="public",
    ),
    "opencode-go": ProviderAuthProfile(
        "opencode-go",
        environment_variables=("OPENCODE_API_KEY",),
        default_base_url=OPENCODE_GO_BASE_URL,
        headers=opencode_default_headers(),
        anonymous_api_key="public",
        credential_identifier="opencode",
    ),
}


def provider_auth_profile(
    provider_identifier: str,
    *,
    environment_variables: tuple[str, ...] = (),
    credential_environment_variables: tuple[str, ...] = (),
    default_base_url: str = "",
    headers: Mapping[str, str] | None = None,
    anonymous_api_key: str = "",
    credential_identifier: str = "",
) -> ProviderAuthProfile:
    """Build the standard authentication profile for a provider identifier."""
    provider = provider_identifier.strip().lower()
    if not provider:
        raise ValueError("provider identifier cannot be empty")
    override = _AUTH_PROFILE_OVERRIDES.get(provider)
    if override is None:
        return ProviderAuthProfile(
            identifier=provider,
            environment_variables=environment_variables,
            credential_environment_variables=credential_environment_variables,
            default_base_url=default_base_url,
            headers=dict(headers or {}),
            anonymous_api_key=anonymous_api_key,
            credential_identifier=credential_identifier,
            method="environment" if credential_environment_variables else "api_key",
        )
    return replace(
        override,
        environment_variables=(
            override.environment_variables
            if override.method == "environment"
            else override.environment_variables or environment_variables
        ),
        default_base_url=override.default_base_url or default_base_url,
        headers=dict(headers or override.headers),
        anonymous_api_key=override.anonymous_api_key or anonymous_api_key,
        credential_identifier=override.credential_identifier or credential_identifier,
    )
