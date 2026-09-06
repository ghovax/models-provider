"""Provider authentication and access-value resolution."""

from __future__ import annotations

from collections.abc import Mapping
import os
import re
from dataclasses import replace
from typing import Any, Callable

from .errors import AuthenticationError
from .oauth import (
    HostedAuthorization,
    LoginFlow,
    OAuthAdapter,
    OAuthConfiguration,
    OAuthProvider,
    OAuthTokens,
)
from .oauth_providers import _default_oauth_adapters
from .profiles import (
    _AUTH_PROFILE_OVERRIDES,
    ApiKeyResolution,
    AuthenticationStatus,
    ProviderAuthProfile,
    provider_auth_profile,
)


_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _resolve_value(value: Any) -> Any:
    """Resolve environment references inside caller-provided provider values."""
    if isinstance(value, str):
        name = value.strip()
        if _ENVIRONMENT_NAME.fullmatch(name):
            if name not in os.environ:
                raise AuthenticationError(f"Environment variable {name!r} is not set")
            return os.environ[name]
        return value
    if isinstance(value, Mapping):
        return {str(key): _resolve_value(item) for key, item in value.items()}
    return value


class ProviderAuthentication:
    """Resolve caller-provided values without a credential-store abstraction."""

    def __init__(
        self,
        values: dict[str, Any],
        profiles: Mapping[str, ProviderAuthProfile] | None = None,
        *,
        catalogue: Any | None = None,
        api_keys: Mapping[str, str] | None = None,
        api_bases: Mapping[str, str] | None = None,
    ) -> None:
        self._values = values
        self._profiles = {key.lower(): value for key, value in (profiles or {}).items()}
        self._catalogue = catalogue
        self._api_keys = dict(api_keys or {})
        self._api_bases = dict(api_bases or {})
        self._oauth_adapters: dict[str, OAuthProvider] = _default_oauth_adapters()

    def profile(
        self, provider_identifier: str, *, environment_variables: tuple[str, ...] = ()
    ) -> ProviderAuthProfile:
        provider = provider_identifier.strip().lower()
        existing = self._profiles.get(provider)
        if existing is not None:
            return existing
        if self._catalogue is not None:
            record = self._catalogue.provider(provider)
            if record is not None:
                override = _AUTH_PROFILE_OVERRIDES.get(provider)
                if override is not None:
                    return replace(
                        override,
                        environment_variables=(
                            override.environment_variables
                            if override.method == "environment"
                            else override.environment_variables or record.environment_variables
                        ),
                        default_base_url=override.default_base_url or record.api_base,
                    )
                return provider_auth_profile(
                    record.identifier,
                    environment_variables=record.environment_variables,
                    default_base_url=record.api_base,
                )
        return provider_auth_profile(provider, environment_variables=environment_variables)

    def _configured(self, provider_identifier: str, profile: ProviderAuthProfile) -> Any:
        credential_identifier = profile.credential_identifier or profile.identifier
        value = self._values.get(credential_identifier)
        if value is None:
            value = self._values.get(provider_identifier)
        return _resolve_value(value)

    def resolve(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
    ) -> ApiKeyResolution:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        provider = profile.identifier
        configured = self._configured(provider_identifier, profile)
        environment: dict[str, str] = {}
        key = (
            self._api_keys.get(profile.credential_identifier, "")
            or self._api_keys.get(provider, "")
            or self._api_keys.get(provider_identifier, "")
        )
        source = "configured" if key else "none"
        if not key and isinstance(configured, str):
            key = configured.strip()
            source = "configured" if key else "none"
        elif not key and isinstance(configured, Mapping):
            raw_key = configured.get("api_key") or configured.get("apiKey")
            if raw_key:
                key = str(raw_key).strip()
                source = "configured" if key else "none"
            environment = {
                str(name): str(value).strip()
                for name, value in configured.items()
                if str(value).strip()
            }
            if environment and source == "none":
                source = "configured"
        base = self._api_bases.get(provider, "") or self._api_bases.get(provider_identifier, "")
        if not key and not environment and profile.anonymous_api_key:
            key = profile.anonymous_api_key
            source = "anonymous"
        return ApiKeyResolution(
            provider=provider,
            api_key=key,
            api_base=base or profile.default_base_url,
            headers=dict(profile.headers),
            environment=environment,
            method=profile.method,
            source=source,
        )

    def _stored_oauth(self, provider_identifier: str) -> OAuthTokens | None:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is None:
            return None
        profile = self.profile(provider)
        value = self._values.get(profile.credential_identifier or provider)
        if isinstance(value, OAuthTokens):
            return value
        if isinstance(value, Mapping):
            try:
                return adapter.deserialize_tokens(value)
            except (AuthenticationError, TypeError, ValueError):
                return None
        return None

    def status(
        self, provider_identifier: str, *, environment_variables: tuple[str, ...] = ()
    ) -> AuthenticationStatus:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        credentials = self._stored_oauth(profile.identifier)
        if credentials is not None:
            return AuthenticationStatus(
                profile.identifier,
                profile.method,
                signed_in=True,
                expired=credentials.is_expired(),
                account=getattr(credentials, "account", ""),
                source="oauth",
            )
        resolution = self.resolve(
            profile.identifier,
            environment_variables=profile.environment_variables,
        )
        return AuthenticationStatus(
            profile.identifier,
            profile.method,
            signed_in=resolution.available and resolution.source != "anonymous",
            source=resolution.source,
        )

    def token(self, provider_identifier: str) -> OAuthTokens:
        token = self._stored_oauth(provider_identifier)
        if token is None:
            raise AuthenticationError(f"Not signed in to {provider_identifier}.")
        return token

    def flow(self, provider_identifier: str) -> LoginFlow:
        provider = provider_identifier.strip().lower()
        try:
            return self._oauth_adapters[provider].flow(self._values)
        except KeyError as error:
            raise AuthenticationError(
                f"{provider_identifier!r} has no registered OAuth flow."
            ) from error

    def redirect_uri(self, provider_identifier: str) -> str:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is None:
            raise AuthenticationError(f"{provider_identifier!r} has no registered OAuth flow.")
        return adapter.redirect_uri()

    def authorization_request(
        self,
        provider_identifier: str,
        redirect_uri: str,
        *,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> HostedAuthorization:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is None:
            raise AuthenticationError(
                f"{provider_identifier!r} has no registered hosted OAuth authorization."
            )
        return adapter.authorization_request(
            redirect_uri,
            client_id=client_id,
            state=state,
            code_verifier=code_verifier,
        )

    def serialize_token(self, provider_identifier: str, tokens: OAuthTokens) -> Mapping[str, Any]:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is None:
            raise AuthenticationError(f"{provider_identifier!r} has no registered OAuth adapter.")
        return adapter.serialize_tokens(tokens)

    def deserialize_token(
        self, provider_identifier: str, payload: Mapping[str, Any]
    ) -> OAuthTokens:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is None:
            raise AuthenticationError(f"{provider_identifier!r} has no registered OAuth adapter.")
        return adapter.deserialize_tokens(payload)

    def register_oauth(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        *,
        flow_factory: Callable[[dict[str, Any]], LoginFlow] | None = None,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
        header_builder: Callable[[OAuthTokens, str, str], Mapping[str, str]] | None = None,
        authorization_factory: Callable[..., HostedAuthorization] | None = None,
        token_serializer: Callable[[OAuthTokens], Mapping[str, Any]] | None = None,
        token_deserializer: Callable[[Mapping[str, Any]], OAuthTokens] | None = None,
    ) -> None:
        provider = provider_identifier.strip().lower()
        if not provider:
            raise ValueError("provider identifier cannot be empty")
        self._oauth_adapters[provider] = OAuthAdapter(
            provider,
            configuration,
            flow_factory=flow_factory,
            token_parser=token_parser,
            header_builder=header_builder,
            authorization_factory=authorization_factory,
            token_serializer=token_serializer,
            token_deserializer=token_deserializer,
        )

    def sign_out(self, provider_identifier: str) -> None:
        profile = self.profile(provider_identifier)
        self._values.pop(profile.credential_identifier or profile.identifier, None)

    def save_api_key(self, provider_identifier: str, api_key: str) -> None:
        profile = self.profile(provider_identifier)
        credential_identifier = profile.credential_identifier or profile.identifier
        if api_key.strip():
            self._values[credential_identifier] = api_key.strip()
        else:
            self._values.pop(credential_identifier, None)

    async def valid_token(self, provider_identifier: str) -> OAuthTokens:
        provider = provider_identifier.strip().lower()
        try:
            return await self._oauth_adapters[provider].valid_token(self._values)
        except KeyError as error:
            raise AuthenticationError(
                f"{provider_identifier!r} has no registered OAuth refresh adapter."
            ) from error

    async def ensure_valid(self, provider_identifier: str) -> None:
        provider = provider_identifier.strip().lower()
        if provider in self._oauth_adapters:
            if self._stored_oauth(provider) is not None:
                await self.valid_token(provider)

    async def request_headers(
        self,
        provider_identifier: str,
        *,
        request_identifier: str = "",
        session_identifier: str = "",
    ) -> dict[str, str]:
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is not None and self._stored_oauth(provider) is not None:
            token = await adapter.valid_token(self._values)
            return dict(adapter.request_headers(token, request_identifier, session_identifier))
        profile = self.profile(provider)
        resolution = self.resolve(provider, environment_variables=profile.environment_variables)
        if not resolution.api_key:
            if resolution.method == "environment" and resolution.environment:
                return dict(resolution.headers)
            raise AuthenticationError(f"No credentials are available for {provider_identifier!r}.")
        header_value = resolution.api_key
        if profile.api_key_prefix:
            header_value = f"{profile.api_key_prefix} {header_value}"
        return {**resolution.headers, profile.api_key_header: header_value}
