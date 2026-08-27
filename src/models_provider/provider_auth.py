"""Authentication orchestration for API-key and OAuth providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable


from .credentials import (
    ApiKeyCredential,
    CredentialStore,
    EnvironmentCredential,
    current_credential_store,
)
from .errors import AuthenticationError
from .oauth import (
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


class ProviderAuthentication:
    """Resolve API keys and OAuth credentials without changing process-global state.

    Explicit keys win over environment variables.  OAuth profiles are used only when a
    provider adapter is registered; models.dev metadata is sufficient for key providers but
    cannot describe OAuth endpoints safely.
    """

    def __init__(
        self,
        profiles: Mapping[str, ProviderAuthProfile] | None = None,
        *,
        catalogue: Any | None = None,
        api_keys: Mapping[str, str] | None = None,
        api_bases: Mapping[str, str] | None = None,
        store: CredentialStore | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._profiles = {key.lower(): value for key, value in (profiles or {}).items()}
        self._catalogue = catalogue
        self._api_keys = dict(api_keys or {})
        self._api_bases = dict(api_bases or {})
        self._store = store
        self._environment = dict(environment or {})
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
        profile = provider_auth_profile(provider, environment_variables=environment_variables)
        if provider in self._oauth_adapters and profile.method == "api_key":
            return replace(profile, method="oauth")
        return profile

    def _store_for(self, store: CredentialStore | None) -> CredentialStore:
        return store or self._store or current_credential_store()

    def resolve(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
        store: CredentialStore | None = None,
    ) -> ApiKeyResolution:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        provider = profile.identifier
        credential_identifier = profile.credential_identifier or provider
        environment: dict[str, str] = {}
        key = (
            self._api_keys.get(credential_identifier, "")
            or self._api_keys.get(provider, "")
            or self._api_keys.get(provider_identifier, "")
        )
        source = "configured" if key else "none"
        if not key:
            stored = self._store_for(store).load(credential_identifier)
            if isinstance(stored, ApiKeyCredential):
                key = stored.api_key.strip()
            elif isinstance(stored, EnvironmentCredential):
                environment = {
                    name: str(value).strip()
                    for name, value in stored.values.items()
                    if str(value).strip()
                }
            elif isinstance(stored, OAuthTokens) and not stored.is_expired():
                key = stored.access_token
                source = "oauth"
            elif isinstance(stored, str):
                key = stored.strip()
            if key:
                source = source if source == "oauth" else "stored"
            elif environment:
                source = "stored"
        if not environment:
            for environment_name in profile.credential_environment_variables:
                value = self._environment.get(environment_name, "").strip()
                if value:
                    environment[environment_name] = value
            if environment and source == "none":
                source = "environment"
        if not key:
            for environment_name in profile.environment_variables or environment_variables:
                key = self._environment.get(environment_name, "").strip()
                if key:
                    source = "environment"
                    break
        if not key and profile.anonymous_api_key:
            key = profile.anonymous_api_key
            source = "anonymous"
        base = self._api_bases.get(provider, "") or self._api_bases.get(provider_identifier, "")
        return ApiKeyResolution(
            provider=provider,
            api_key=key,
            api_base=base or profile.default_base_url,
            headers=dict(profile.headers),
            environment=environment,
            method=profile.method,
            source=source,
        )

    def resolve_key(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
        store: CredentialStore | None = None,
    ) -> ApiKeyResolution:
        """Backward-compatible API-key-focused name for :meth:`resolve`."""
        return self.resolve(
            provider_identifier, environment_variables=environment_variables, store=store
        )

    def status(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
        store: CredentialStore | None = None,
    ) -> AuthenticationStatus:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        credential_identifier = profile.credential_identifier or profile.identifier
        credentials = self._store_for(store).load(credential_identifier)
        if not isinstance(credentials, OAuthTokens):
            resolution = self.resolve(
                profile.identifier,
                environment_variables=profile.environment_variables,
                store=store,
            )
            return AuthenticationStatus(
                profile.identifier,
                profile.method,
                signed_in=resolution.available and resolution.source != "anonymous",
                source=resolution.source,
            )
        return AuthenticationStatus(
            profile.identifier,
            profile.method,
            signed_in=True,
            expired=credentials.is_expired(),
            account=getattr(credentials, "account", ""),
            source="oauth",
        )

    def token(
        self, provider_identifier: str, *, store: CredentialStore | None = None
    ) -> OAuthTokens:
        provider = provider_identifier.strip().lower()
        profile = self.profile(provider)
        credentials = self._store_for(store).load(
            profile.credential_identifier or profile.identifier
        )
        if not isinstance(credentials, OAuthTokens):
            raise AuthenticationError(f"Not signed in to {provider_identifier}.")
        return credentials

    def flow(self, provider_identifier: str, *, store: CredentialStore | None = None) -> LoginFlow:
        """Create the provider's login flow; the host decides how to open its URL."""
        provider = provider_identifier.strip().lower()
        selected_store = store or self._store or current_credential_store()
        try:
            return self._oauth_adapters[provider].flow(selected_store)
        except KeyError as error:
            raise AuthenticationError(
                f"{provider_identifier!r} has no registered OAuth flow."
            ) from error

    def register_oauth(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        *,
        flow_factory: Callable[[CredentialStore], LoginFlow] | None = None,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
        header_builder: Callable[[OAuthTokens, str, str], Mapping[str, str]] | None = None,
    ) -> None:
        """Register a provider's standard OAuth endpoints without coupling to its model transport."""
        provider = provider_identifier.strip().lower()
        if not provider:
            raise ValueError("provider identifier cannot be empty")
        self._oauth_adapters[provider] = OAuthAdapter(
            provider,
            configuration,
            flow_factory=flow_factory,
            token_parser=token_parser,
            header_builder=header_builder,
        )

    def sign_out(self, provider_identifier: str, *, store: CredentialStore | None = None) -> None:
        """Remove account credentials from the caller-owned store."""
        profile = self.profile(provider_identifier)
        self._store_for(store).clear(profile.credential_identifier or profile.identifier)

    def save_api_key(
        self,
        provider_identifier: str,
        api_key: str,
        *,
        store: CredentialStore | None = None,
    ) -> None:
        """Persist an API key through the application's credential store."""
        profile = self.profile(provider_identifier)
        credential_identifier = profile.credential_identifier or profile.identifier
        selected_store = self._store_for(store)
        if api_key.strip():
            selected_store.save(credential_identifier, ApiKeyCredential(api_key.strip()))
        else:
            selected_store.clear(credential_identifier)

    async def valid_token(
        self, provider_identifier: str, *, store: CredentialStore | None = None
    ) -> OAuthTokens:
        """Return a live OAuth token and refresh it once when the provider supports refresh."""
        provider = provider_identifier.strip().lower()
        try:
            return await self._oauth_adapters[provider].valid_token(
                store or self._store or current_credential_store()
            )
        except KeyError as error:
            raise AuthenticationError(
                f"{provider_identifier!r} has no registered OAuth refresh adapter."
            ) from error

    async def ensure_valid(
        self, provider_identifier: str, *, store: CredentialStore | None = None
    ) -> None:
        """Refresh a registered OAuth credential before an asynchronous model call."""
        provider = provider_identifier.strip().lower()
        if provider in self._oauth_adapters:
            await self.valid_token(provider, store=store)

    async def request_headers(
        self,
        provider_identifier: str,
        *,
        request_identifier: str = "",
        session_identifier: str = "",
        store: CredentialStore | None = None,
    ) -> dict[str, str]:
        """Build authenticated headers for an account-backed provider without exposing its token."""
        provider = provider_identifier.strip().lower()
        adapter = self._oauth_adapters.get(provider)
        if adapter is not None:
            token = await adapter.valid_token(store or self._store or current_credential_store())
            return dict(adapter.request_headers(token, request_identifier, session_identifier))
        profile = self.profile(provider)
        resolution = self.resolve(
            provider, environment_variables=profile.environment_variables, store=store
        )
        if not resolution.api_key:
            if resolution.method == "environment" and resolution.environment:
                return dict(resolution.headers)
            raise AuthenticationError(f"No credentials are available for {provider_identifier!r}.")
        header_value = resolution.api_key
        if profile.api_key_prefix:
            header_value = f"{profile.api_key_prefix} {header_value}"
        return {**resolution.headers, profile.api_key_header: header_value}
