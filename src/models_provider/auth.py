"""Provider authentication primitives, including API keys and subscription sign-in.

This module deliberately knows nothing about LangMesh, Teacher, or a daemon.  Applications
provide a credential store and decide how a login URL is shown to a person; the provider
package owns token shape, refresh, expiry, PKCE, and request material.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
import hashlib
import html
import json
import os
import secrets
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

__all__ = [
    "ApiKeyResolution",
    "AuthenticationError",
    "AuthenticationStatus",
    "ApiKeyCredential",
    "EnvironmentCredential",
    "ChatGPTLoginFlow",
    "ChatGPTTokens",
    "CredentialStore",
    "CursorLoginFlow",
    "CursorTokens",
    "DeviceLoginFlow",
    "LoginFlow",
    "MemoryCredentialStore",
    "OAuthAdapter",
    "OAuthConfiguration",
    "OAuthLoginFlow",
    "OAuthProvider",
    "OAuthTokens",
    "ProviderAuthentication",
    "ProviderAuthProfile",
    "provider_auth_profile",
    "current_credential_store",
    "bind_credential_store",
    "reset_credential_store",
    "request_chatgpt_headers",
    "request_cursor_headers",
    "chatgpt_tokens",
    "cursor_tokens",
    "valid_chatgpt_tokens",
    "valid_cursor_tokens",
]


class AuthenticationError(RuntimeError):
    """Raised when a provider cannot authenticate a request or complete sign-in."""


@runtime_checkable
class CredentialStore(Protocol):
    """Storage supplied by the embedding application for provider credentials."""

    def load(self, provider_identifier: str) -> Any: ...

    def save(self, provider_identifier: str, credentials: Any) -> None: ...

    def clear(self, provider_identifier: str) -> None: ...


class MemoryCredentialStore:
    """Small storage implementation for applications, workers, and isolated mock runs."""

    def __init__(self) -> None:
        self._credentials: dict[str, Any] = {}

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


_default_store = MemoryCredentialStore()
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
        default_base_url="https://opencode.ai/zen/v1",
        headers={"User-Agent": "opencode/0.0.0", "x-opencode-client": "models-provider"},
        anonymous_api_key="public",
    ),
    "opencode-go": ProviderAuthProfile(
        "opencode-go",
        environment_variables=("OPENCODE_API_KEY",),
        default_base_url="https://opencode.ai/zen/v1",
        headers={"User-Agent": "opencode/0.0.0", "x-opencode-client": "models-provider"},
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
    ) -> None:
        self._profiles = {key.lower(): value for key, value in (profiles or {}).items()}
        self._catalogue = catalogue
        self._api_keys = dict(api_keys or {})
        self._api_bases = dict(api_bases or {})
        self._store = store
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
                value = os.environ.get(environment_name, "").strip()
                if value:
                    environment[environment_name] = value
            if environment and source == "none":
                source = "environment"
        if not key:
            for environment_name in profile.environment_variables or environment_variables:
                key = os.environ.get(environment_name, "").strip()
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


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    """Common token fields shared by account-backed providers."""

    access_token: str
    refresh_token: str
    expires_at: float

    def is_expired(self, leeway_seconds: float = 60.0) -> bool:
        return time.time() >= self.expires_at - leeway_seconds


@dataclass(frozen=True, slots=True)
class OAuthConfiguration:
    """Standard OAuth endpoints and request policy for one model provider."""

    authorization_url: str = ""
    token_url: str = ""
    client_id: str = ""
    scopes: tuple[str, ...] = ()
    redirect_uri: str = "http://127.0.0.1:8765/callback"
    device_authorization_url: str = ""
    client_secret: str = field(default="", repr=False)
    grant_type: str = "authorization_code"
    token_endpoint_auth_method: str = "none"
    access_header: str = "Authorization"
    access_prefix: str = "Bearer"
    authorization_parameters: Mapping[str, str] = field(default_factory=dict)
    token_parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_url or not self.client_id:
            raise ValueError("OAuth token_url and client_id are required")
        if (
            not self.authorization_url
            and not self.device_authorization_url
            and self.grant_type != "client_credentials"
        ):
            raise ValueError("OAuth authorization_url or device_authorization_url is required")
        if self.grant_type not in {
            "authorization_code",
            "device_code",
            "client_credentials",
        }:
            raise ValueError("unsupported OAuth grant type")
        if self.token_endpoint_auth_method not in {
            "none",
            "client_secret_post",
            "client_secret_basic",
        }:
            raise ValueError("unsupported OAuth token endpoint authentication method")
        if self.authorization_url and not self.redirect_uri:
            raise ValueError("redirect_uri is required for authorization-code OAuth")


@runtime_checkable
class LoginFlow(Protocol):
    """The small host-facing contract for a browser or device sign-in flow."""

    @property
    def authorize_url(self) -> str: ...

    async def start(self) -> None: ...

    async def wait(self, timeout: float = 300.0) -> OAuthTokens: ...  # noqa: ASYNC109

    async def close(self) -> None: ...


@runtime_checkable
class OAuthProvider(Protocol):
    """An OAuth adapter used by :class:`ProviderAuthentication`."""

    def flow(self, store: CredentialStore) -> LoginFlow: ...

    async def valid_token(self, store: CredentialStore) -> OAuthTokens: ...

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]: ...


def _oauth_tokens_from_payload(
    payload: Mapping[str, Any], previous: OAuthTokens | None = None
) -> OAuthTokens:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("OAuth returned an invalid token response.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("OAuth returned no access token.")
    try:
        expires_in = float(payload.get("expires_in") or 3600.0)
    except (TypeError, ValueError):
        expires_in = 3600.0
    return OAuthTokens(
        access_token=access_token,
        refresh_token=str(
            payload.get("refresh_token") or (previous.refresh_token if previous else "")
        ),
        expires_at=time.time() + max(1.0, expires_in),
    )


class OAuthLoginFlow:
    """Reusable authorization-code + PKCE flow for providers with a loopback callback."""

    def __init__(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        store: CredentialStore,
        *,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
    ) -> None:
        if not configuration.authorization_url:
            raise ValueError("authorization_url is required for a browser OAuth flow")
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._store = store
        self._token_parser = token_parser or _oauth_tokens_from_payload
        self._verifier = _pkce_verifier()
        self._state = secrets.token_urlsafe(24)
        self._server: HTTPServer | None = None
        self._captured: dict[str, str] = {}

    @property
    def authorize_url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        parameters = {
            **self.configuration.authorization_parameters,
            "response_type": "code",
            "client_id": self.configuration.client_id,
            "redirect_uri": self.configuration.redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": self._state,
        }
        if self.configuration.scopes:
            parameters["scope"] = " ".join(self.configuration.scopes)
        return f"{self.configuration.authorization_url}?{urllib.parse.urlencode(parameters)}"

    async def start(self) -> None:
        redirect = urllib.parse.urlparse(self.configuration.redirect_uri)
        if redirect.scheme != "http" or redirect.hostname not in {"127.0.0.1", "localhost"}:
            raise AuthenticationError("OAuth loopback redirect_uri must use localhost")
        if redirect.port is None or redirect.port == 0:
            raise AuthenticationError("OAuth loopback redirect_uri must specify a fixed port")
        flow = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format_string: str, *arguments: object) -> None:
                return

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                expected_path = urllib.parse.urlparse(flow.configuration.redirect_uri).path
                if parsed.path != expected_path:
                    flow._captured["error"] = "Invalid OAuth callback path."
                elif query.get("state", [""])[0] != flow._state:
                    flow._captured["error"] = "OAuth authorization state mismatch."
                elif query.get("code", [""])[0]:
                    flow._captured["code"] = query["code"][0]
                else:
                    flow._captured["error"] = query.get("error", ["OAuth authorization failed."])[0]
                message = flow._captured.get("error", "Signed in.")
                body = f"<html><body>{html.escape(message)}</body></html>".encode()
                self.send_response(200 if "code" in flow._captured else 400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer((redirect.hostname, redirect.port), CallbackHandler)
        self._server.timeout = 0.5

    async def wait(self, timeout: float = 300.0) -> OAuthTokens:  # noqa: ASYNC109
        if self._server is None:
            raise AuthenticationError("start() must be called before wait().")
        deadline = time.monotonic() + timeout
        try:
            while not self._captured:
                if time.monotonic() >= deadline:
                    raise AuthenticationError("OAuth sign-in timed out.")
                await asyncio.to_thread(self._server.handle_request)
            if "code" not in self._captured:
                raise AuthenticationError(self._captured.get("error", "OAuth sign-in failed."))
            data = {
                **self.configuration.token_parameters,
                "grant_type": "authorization_code",
                "code": self._captured["code"],
                "redirect_uri": self.configuration.redirect_uri,
                "client_id": self.configuration.client_id,
                "code_verifier": self._verifier,
            }
            response = await self._token_request(data)
            tokens = self._token_parser(response, None)
            self._store.save(self.provider_identifier, tokens)
            return tokens
        finally:
            await self.close()

    async def _token_request(self, data: Mapping[str, str]) -> Mapping[str, Any]:
        request_data = dict(data)
        auth = None
        if self.configuration.token_endpoint_auth_method == "client_secret_post":
            request_data["client_secret"] = self.configuration.client_secret
        elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
            auth = (self.configuration.client_id, self.configuration.client_secret)
            request_data.pop("client_id", None)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.configuration.token_url, data=request_data, auth=auth)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise AuthenticationError("OAuth returned an invalid token response.")
        return payload

    async def close(self) -> None:
        if self._server is not None:
            await asyncio.to_thread(self._server.server_close)
            self._server = None


class DeviceLoginFlow:
    """Reusable OAuth device-authorization flow for CLIs and hosts without callbacks."""

    def __init__(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        store: CredentialStore,
        *,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
    ) -> None:
        if not configuration.device_authorization_url:
            raise ValueError("device_authorization_url is required for a device flow")
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._store = store
        self._token_parser = token_parser or _oauth_tokens_from_payload
        self._device_code = ""
        self._verification_url = ""
        self._interval = 5.0
        self._expires_at = 0.0
        self._closed = False

    @property
    def authorize_url(self) -> str:
        return self._verification_url

    async def start(self) -> None:
        data = {
            **self.configuration.token_parameters,
            "client_id": self.configuration.client_id,
        }
        if self.configuration.scopes:
            data["scope"] = " ".join(self.configuration.scopes)
        auth = None
        if self.configuration.token_endpoint_auth_method == "client_secret_post":
            data["client_secret"] = self.configuration.client_secret
        elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
            auth = (self.configuration.client_id, self.configuration.client_secret)
            data.pop("client_id", None)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.configuration.device_authorization_url, data=data, auth=auth
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping) or not payload.get("device_code"):
            raise AuthenticationError("OAuth returned an invalid device authorization response.")
        self._device_code = str(payload["device_code"])
        self._verification_url = str(
            payload.get("verification_uri_complete")
            or payload.get("verification_uri")
            or payload.get("verification_url")
            or ""
        )
        self._interval = max(1.0, float(payload.get("interval") or 5.0))
        self._expires_at = time.monotonic() + max(1.0, float(payload.get("expires_in") or 600.0))

    async def wait(self, timeout: float = 600.0) -> OAuthTokens:  # noqa: ASYNC109
        if not self._device_code:
            raise AuthenticationError("start() must be called before wait().")
        deadline = min(time.monotonic() + timeout, self._expires_at)
        interval = self._interval
        async with httpx.AsyncClient(timeout=30) as client:
            while not self._closed and time.monotonic() < deadline:
                data = {
                    **self.configuration.token_parameters,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": self._device_code,
                    "client_id": self.configuration.client_id,
                }
                auth = None
                if self.configuration.token_endpoint_auth_method == "client_secret_post":
                    data["client_secret"] = self.configuration.client_secret
                elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
                    auth = (self.configuration.client_id, self.configuration.client_secret)
                    data.pop("client_id", None)
                response = await client.post(self.configuration.token_url, data=data, auth=auth)
                if response.is_success:
                    payload = response.json()
                    if not isinstance(payload, Mapping):
                        raise AuthenticationError(
                            "OAuth returned an invalid device token response."
                        )
                    tokens = self._token_parser(payload, None)
                    self._store.save(self.provider_identifier, tokens)
                    return tokens
                try:
                    error_payload = response.json()
                    error = (
                        error_payload.get("error", "") if isinstance(error_payload, Mapping) else ""
                    )
                except (TypeError, ValueError):
                    error = ""
                if error == "authorization_pending":
                    await asyncio.sleep(interval)
                    continue
                if error == "slow_down":
                    interval += 5.0
                    await asyncio.sleep(interval)
                    continue
                if error in {"expired_token", "access_denied"}:
                    raise AuthenticationError(f"OAuth device sign-in {error.replace('_', ' ')}.")
                response.raise_for_status()
            raise AuthenticationError("OAuth device sign-in timed out.")

    async def close(self) -> None:
        self._closed = True


class OAuthAdapter:
    """Default OAuth adapter with refresh, PKCE/device login, and bearer-header support."""

    def __init__(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        *,
        flow_factory: Callable[[CredentialStore], LoginFlow] | None = None,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
        header_builder: Callable[[OAuthTokens, str, str], Mapping[str, str]] | None = None,
    ) -> None:
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._flow_factory = flow_factory
        self._token_parser = token_parser or _oauth_tokens_from_payload
        self._header_builder = header_builder
        self._refresh_lock = asyncio.Lock()

    def flow(self, store: CredentialStore) -> LoginFlow:
        if (
            not self.configuration.authorization_url
            and not self.configuration.device_authorization_url
        ):
            raise AuthenticationError(
                f"{self.provider_identifier!r} has no interactive OAuth flow."
            )
        if self._flow_factory is not None:
            return self._flow_factory(store)
        flow_type = (
            DeviceLoginFlow if self.configuration.device_authorization_url else OAuthLoginFlow
        )
        return flow_type(
            self.provider_identifier,
            self.configuration,
            store,
            token_parser=self._token_parser,
        )

    async def valid_token(self, store: CredentialStore) -> OAuthTokens:
        tokens = store.load(self.provider_identifier)
        if self.configuration.grant_type == "client_credentials":
            if isinstance(tokens, OAuthTokens) and not tokens.is_expired():
                return tokens
            async with self._refresh_lock:
                current = store.load(self.provider_identifier)
                if isinstance(current, OAuthTokens) and not current.is_expired():
                    return current
                try:
                    payload = await self._request_token(
                        {
                            **self.configuration.token_parameters,
                            "grant_type": "client_credentials",
                            "client_id": self.configuration.client_id,
                        }
                    )
                except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
                    raise AuthenticationError(
                        f"Could not obtain the {self.provider_identifier} access token: {error}"
                    ) from error
                refreshed = self._token_parser(
                    payload, current if isinstance(current, OAuthTokens) else None
                )
                store.save(self.provider_identifier, refreshed)
                return refreshed
        if not isinstance(tokens, OAuthTokens):
            raise AuthenticationError(f"Not signed in to {self.provider_identifier}.")
        if not tokens.is_expired():
            return tokens
        async with self._refresh_lock:
            current = store.load(self.provider_identifier)
            if isinstance(current, OAuthTokens) and not current.is_expired():
                return current
            current = current if isinstance(current, OAuthTokens) else tokens
            if not current.refresh_token:
                raise AuthenticationError(
                    f"{self.provider_identifier} session expired; sign in again."
                )
            data = {
                **self.configuration.token_parameters,
                "grant_type": "refresh_token",
                "refresh_token": current.refresh_token,
                "client_id": self.configuration.client_id,
            }
            auth = None
            if self.configuration.token_endpoint_auth_method == "client_secret_post":
                data["client_secret"] = self.configuration.client_secret
            elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
                auth = (self.configuration.client_id, self.configuration.client_secret)
                data.pop("client_id", None)
            try:
                payload = await self._request_token(data, auth=auth)
                refreshed = self._token_parser(payload, current)
            except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
                raise AuthenticationError(
                    f"Could not refresh the {self.provider_identifier} session: {error}"
                ) from error
            store.save(self.provider_identifier, refreshed)
            return refreshed

    async def _request_token(
        self, data: Mapping[str, str], *, auth: tuple[str, str] | None = None
    ) -> Mapping[str, Any]:
        request_data = dict(data)
        request_auth = auth
        if self.configuration.token_endpoint_auth_method == "client_secret_post":
            request_data["client_secret"] = self.configuration.client_secret
        elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
            request_auth = (
                self.configuration.client_id,
                self.configuration.client_secret,
            )
            request_data.pop("client_id", None)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self.configuration.token_url, data=request_data, auth=request_auth
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise AuthenticationError("OAuth returned an invalid token response.")
        return payload

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]:
        if self._header_builder is not None:
            return self._header_builder(token, request_identifier, session_identifier)
        value = token.access_token
        if self.configuration.access_prefix:
            value = f"{self.configuration.access_prefix} {value}"
        return {self.configuration.access_header: value}


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        _, payload, _ = token.split(".")
        decoded = json.loads(_b64url_decode(payload))
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
        return {}


@dataclass(frozen=True, slots=True, init=False)
class ChatGPTTokens(OAuthTokens):
    """ChatGPT subscription credentials."""

    id_token: str = ""
    account_id: str = ""
    email: str = ""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        id_token: str = "",
        account_id: str = "",
        email: str = "",
        expires_at: float = 0.0,
    ) -> None:
        object.__setattr__(self, "access_token", access_token)
        object.__setattr__(self, "refresh_token", refresh_token)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "id_token", id_token)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "email", email)

    @property
    def account(self) -> str:
        return self.email or self.account_id


@dataclass(frozen=True, slots=True, init=False)
class CursorTokens(OAuthTokens):
    """Cursor subscription credentials."""

    account: str = ""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        account: str,
        expires_at: float,
    ) -> None:
        object.__setattr__(self, "access_token", access_token)
        object.__setattr__(self, "refresh_token", refresh_token)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "account", account)


_chatgpt_refresh_lock = asyncio.Lock()
_cursor_refresh_lock = asyncio.Lock()


def _chatgpt_from_payload(
    payload: Mapping[str, Any], previous: ChatGPTTokens | None = None
) -> ChatGPTTokens:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("ChatGPT returned an invalid token response.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("ChatGPT returned no access token.")
    id_token = str(payload.get("id_token") or (previous.id_token if previous else ""))
    claims = _jwt_claims(id_token)
    auth_claim = claims.get("https://api.openai.com/auth")
    account_id = auth_claim.get("chatgpt_account_id", "") if isinstance(auth_claim, dict) else ""
    return ChatGPTTokens(
        access_token=access_token,
        refresh_token=str(
            payload.get("refresh_token") or (previous.refresh_token if previous else "")
        ),
        id_token=id_token,
        account_id=str(account_id or (previous.account_id if previous else "")),
        email=str(claims.get("email") or (previous.email if previous else "")),
        expires_at=time.time() + float(payload.get("expires_in") or 3600),
    )


def _cursor_from_payload(
    payload: Mapping[str, Any], previous: CursorTokens | None = None
) -> CursorTokens:
    access_token = str(payload.get("accessToken") or "")
    if not access_token:
        raise AuthenticationError("Cursor returned no access token.")
    claims = _jwt_claims(access_token)
    expiry = claims.get("exp")
    return CursorTokens(
        access_token=access_token,
        refresh_token=str(
            payload.get("refreshToken") or (previous.refresh_token if previous else "")
        ),
        expires_at=float(expiry) if isinstance(expiry, (int, float)) else time.time() + 3600,
        account=str(
            claims.get("email") or claims.get("name") or (previous.account if previous else "")
        ),
    )


def _save(provider: str, credentials: OAuthTokens, store: CredentialStore | None) -> None:
    (store or current_credential_store()).save(provider, credentials)


def chatgpt_tokens(store: CredentialStore | None = None) -> ChatGPTTokens | None:
    value = (store or current_credential_store()).load("chatgpt")
    return value if isinstance(value, ChatGPTTokens) else None


def cursor_tokens(store: CredentialStore | None = None) -> CursorTokens | None:
    value = (store or current_credential_store()).load("cursor")
    return value if isinstance(value, CursorTokens) else None


async def valid_chatgpt_tokens(store: CredentialStore | None = None) -> ChatGPTTokens:
    selected_store = store or current_credential_store()
    tokens = chatgpt_tokens(selected_store)
    if tokens is None:
        raise AuthenticationError("Not signed in to ChatGPT.")
    if not tokens.is_expired():
        return tokens
    async with _chatgpt_refresh_lock:
        current = chatgpt_tokens(selected_store) or tokens
        if not current.is_expired():
            return current
        if not current.refresh_token:
            raise AuthenticationError("ChatGPT session expired; sign in again.")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://auth.openai.com/oauth/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": current.refresh_token,
                        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                        "scope": "openid profile email offline_access",
                    },
                )
                response.raise_for_status()
                refreshed = _chatgpt_from_payload(response.json(), current)
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not refresh the ChatGPT session: {error}") from error
        _save("chatgpt", refreshed, selected_store)
        return refreshed


async def valid_cursor_tokens(store: CredentialStore | None = None) -> CursorTokens:
    selected_store = store or current_credential_store()
    tokens = cursor_tokens(selected_store)
    if tokens is None:
        raise AuthenticationError("Not signed in to Cursor.")
    if not tokens.is_expired():
        return tokens
    async with _cursor_refresh_lock:
        current = cursor_tokens(selected_store) or tokens
        if not current.is_expired():
            return current
        if not current.refresh_token:
            raise AuthenticationError("Cursor session expired; sign in again.")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api2.cursor.sh/auth/exchange_user_api_key",
                    headers={"Authorization": f"Bearer {current.refresh_token}"},
                    content="{}",
                )
                response.raise_for_status()
                refreshed = _cursor_from_payload(response.json(), current)
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not refresh the Cursor session: {error}") from error
        _save("cursor", refreshed, selected_store)
        return refreshed


def _pkce_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()


class ChatGPTLoginFlow:
    """PKCE loopback login. The host opens ``authorize_url`` and owns the browser policy."""

    def __init__(self, store: CredentialStore | None = None) -> None:
        self._store = store or current_credential_store()
        self._verifier = _pkce_verifier()
        self._state = secrets.token_urlsafe(24)
        self._server: HTTPServer | None = None
        self._captured: dict[str, str] = {}

    @property
    def authorize_url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        parameters = {
            "response_type": "code",
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "redirect_uri": "http://localhost:1455/auth/callback",
            "scope": "openid profile email offline_access",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": self._state,
        }
        return "https://auth.openai.com/oauth/authorize?" + urllib.parse.urlencode(parameters)

    async def start(self) -> None:
        flow = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format_string: str, *arguments: object) -> None:
                return

            def do_GET(self) -> None:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if urllib.parse.urlparse(self.path).path != "/auth/callback":
                    flow._captured["error"] = "Invalid callback path."
                elif query.get("state", [""])[0] != flow._state:
                    flow._captured["error"] = "Authorization state mismatch."
                elif query.get("code", [""])[0]:
                    flow._captured["code"] = query["code"][0]
                else:
                    flow._captured["error"] = query.get("error", ["Authorization failed."])[0]
                body = f"<html><body>{html.escape(flow._captured.get('error', 'Signed in.'))}</body></html>".encode()
                self.send_response(200 if "code" in flow._captured else 400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
        self._server.timeout = 0.5

    async def wait(self, timeout: float = 300.0) -> ChatGPTTokens:  # noqa: ASYNC109
        if self._server is None:
            raise AuthenticationError("start() must be called before wait().")
        deadline = time.monotonic() + timeout
        try:
            while not self._captured:
                if time.monotonic() >= deadline:
                    raise AuthenticationError("ChatGPT sign-in timed out.")
                await asyncio.to_thread(self._server.handle_request)
            if "code" not in self._captured:
                raise AuthenticationError(self._captured.get("error", "ChatGPT sign-in failed."))
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://auth.openai.com/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": self._captured["code"],
                        "redirect_uri": "http://localhost:1455/auth/callback",
                        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                        "code_verifier": self._verifier,
                    },
                )
                response.raise_for_status()
                tokens = _chatgpt_from_payload(response.json())
            _save("chatgpt", tokens, self._store)
            return tokens
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not complete ChatGPT sign-in: {error}") from error
        finally:
            await self.close()

    async def close(self) -> None:
        if self._server is not None:
            await asyncio.to_thread(self._server.server_close)
            self._server = None


class CursorLoginFlow:
    """Cursor's browser login and polling flow, with no daemon dependency."""

    def __init__(self, store: CredentialStore | None = None) -> None:
        self._store = store or current_credential_store()
        self._verifier = _pkce_verifier()
        self._identifier = str(uuid.uuid4())
        self._cancelled = False

    @property
    def authorize_url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        return "https://cursor.com/loginDeepControl?" + urllib.parse.urlencode(
            {
                "challenge": challenge,
                "uuid": self._identifier,
                "mode": "login",
                "redirectTarget": "cli",
            }
        )

    async def start(self) -> None:
        return

    async def wait(self, timeout: float = 300.0) -> CursorTokens:  # noqa: ASYNC109
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancelled:
                raise AuthenticationError("Cursor sign-in was cancelled.")
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        "https://api2.cursor.sh/auth/poll",
                        params={"uuid": self._identifier, "verifier": self._verifier},
                    )
                if response.status_code == 404:
                    await asyncio.sleep(1)
                    continue
                response.raise_for_status()
                tokens = _cursor_from_payload(response.json())
                _save("cursor", tokens, self._store)
                return tokens
            except httpx.HTTPError as error:
                raise AuthenticationError(f"Cursor sign-in failed: {error}") from error
        raise AuthenticationError("Cursor sign-in timed out.")

    async def close(self) -> None:
        self._cancelled = True


def request_chatgpt_headers(tokens: ChatGPTTokens, session_identifier: str = "") -> dict[str, str]:
    """Headers required by the ChatGPT subscription Responses endpoint."""
    return {
        "Authorization": f"Bearer {tokens.access_token}",
        "ChatGPT-Account-Id": tokens.account_id,
        "originator": "codex_cli_rs",
        "session-id": session_identifier or str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def request_cursor_headers(tokens: CursorTokens, request_identifier: str) -> dict[str, str]:
    """Headers required by Cursor's agent protocol."""
    slot = int(time.time() // 1800) * 1800
    stamp = (slot * 1000) // 1_000_000
    obfuscated = bytearray(stamp.to_bytes(6, "big"))
    previous_byte = 165
    for index in range(len(obfuscated)):
        obfuscated[index] = ((obfuscated[index] ^ previous_byte) + index) & 0xFF
        previous_byte = obfuscated[index]
    checksum = base64.urlsafe_b64encode(bytes(obfuscated)).rstrip(b"=").decode()
    token_segments = tokens.access_token.split(".")
    payload_digest = (
        hashlib.sha256(token_segments[1].encode()).hexdigest()[:8]
        if len(token_segments) > 1
        else "00000000"
    )
    token_digest = hashlib.sha256(tokens.access_token.encode()).hexdigest()[:8]
    checksum = f"{checksum}{payload_digest}/{token_digest}"
    return {
        "Authorization": f"Bearer {tokens.access_token}",
        "Content-Type": "application/grpc-web+proto",
        "x-cursor-checksum": checksum,
        "x-cursor-client-version": "cli-2026.02.13-41ac335",
        "x-cursor-client-type": "cli",
        "x-cursor-timezone": _machine_time_zone(),
        "x-ghost-mode": "true",
        "x-cursor-streaming": "true",
        "x-request-id": request_identifier,
    }


def _machine_time_zone() -> str:
    configured = os.environ.get("TZ", "").strip()
    if configured:
        return configured
    try:
        target = os.readlink("/etc/localtime")
        if "zoneinfo/" in target:
            return target.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return time.tzname[0] if time.tzname else "UTC"


class _BuiltInOAuthAdapter:
    """Adapt a provider-specific login implementation to the common authentication contract."""

    def __init__(
        self,
        flow_factory: Callable[[CredentialStore], LoginFlow],
        valid_token: Callable[[CredentialStore], Any],
        header_builder: Callable[[OAuthTokens, str, str], Mapping[str, str]],
    ) -> None:
        self._flow_factory = flow_factory
        self._valid_token = valid_token
        self._header_builder = header_builder

    def flow(self, store: CredentialStore) -> LoginFlow:
        return self._flow_factory(store)

    async def valid_token(self, store: CredentialStore) -> OAuthTokens:
        return await self._valid_token(store)

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]:
        return self._header_builder(token, request_identifier, session_identifier)


def _default_oauth_adapters() -> dict[str, OAuthProvider]:
    return {
        "chatgpt": _BuiltInOAuthAdapter(
            ChatGPTLoginFlow,
            valid_chatgpt_tokens,
            lambda token, _request, session: request_chatgpt_headers(token, session),
        ),
        "cursor": _BuiltInOAuthAdapter(
            CursorLoginFlow,
            valid_cursor_tokens,
            lambda token, request, _session: request_cursor_headers(
                token, request or str(uuid.uuid4())
            ),
        ),
    }
