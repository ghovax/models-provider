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
from typing import Any, Protocol, runtime_checkable

import httpx

__all__ = [
    "ApiKeyResolution",
    "AuthenticationError",
    "AuthenticationStatus",
    "ChatGPTLoginFlow",
    "ChatGPTTokens",
    "CredentialStore",
    "CursorLoginFlow",
    "CursorTokens",
    "MemoryCredentialStore",
    "OAuthTokens",
    "ProviderAuthentication",
    "ProviderAuthProfile",
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


@dataclass(frozen=True, slots=True)
class ApiKeyResolution:
    """The non-secret request settings resolved for one provider."""

    provider: str
    api_key: str = ""
    api_base: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderAuthProfile:
    """Authentication metadata for a provider, independent of any model transport."""

    identifier: str
    environment_variables: tuple[str, ...] = ()
    default_base_url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    method: str = "api_key"
    anonymous_api_key: str = ""


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
        api_keys: Mapping[str, str] | None = None,
        api_bases: Mapping[str, str] | None = None,
        store: CredentialStore | None = None,
    ) -> None:
        self._profiles = {key.lower(): value for key, value in (profiles or {}).items()}
        self._api_keys = dict(api_keys or {})
        self._api_bases = dict(api_bases or {})
        self._store = store

    def profile(self, provider_identifier: str, *, environment_variables: tuple[str, ...] = ()) -> ProviderAuthProfile:
        provider = provider_identifier.strip().lower()
        existing = self._profiles.get(provider)
        if existing is not None:
            return existing
        method = "oauth" if provider in {"chatgpt", "cursor"} else "api_key"
        return ProviderAuthProfile(provider, environment_variables=environment_variables, method=method)

    def _store_for(self, store: CredentialStore | None) -> CredentialStore:
        return store or self._store or current_credential_store()

    def resolve_key(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
    ) -> ApiKeyResolution:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        provider = profile.identifier
        key = self._api_keys.get(provider, "") or self._api_keys.get(provider_identifier, "")
        if not key:
            for environment_name in profile.environment_variables or environment_variables:
                key = os.environ.get(environment_name, "").strip()
                if key:
                    break
        key = key or profile.anonymous_api_key
        base = self._api_bases.get(provider, "") or self._api_bases.get(provider_identifier, "")
        return ApiKeyResolution(
            provider=provider,
            api_key=key,
            api_base=base or profile.default_base_url,
            headers=dict(profile.headers),
        )

    def status(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
        store: CredentialStore | None = None,
    ) -> AuthenticationStatus:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        credentials = self._store_for(store).load(profile.identifier)
        if not isinstance(credentials, OAuthTokens):
            resolution = self.resolve_key(profile.identifier)
            return AuthenticationStatus(
                profile.identifier,
                profile.method,
                signed_in=bool(resolution.api_key),
            )
        return AuthenticationStatus(
            profile.identifier,
            profile.method,
            signed_in=True,
            expired=credentials.is_expired(),
            account=getattr(credentials, "account", ""),
        )

    def token(self, provider_identifier: str, *, store: CredentialStore | None = None) -> OAuthTokens:
        provider = provider_identifier.strip().lower()
        credentials = self._store_for(store).load(provider)
        if not isinstance(credentials, OAuthTokens):
            raise AuthenticationError(f"Not signed in to {provider_identifier}.")
        return credentials

    def flow(self, provider_identifier: str, *, store: CredentialStore | None = None) -> ChatGPTLoginFlow | CursorLoginFlow:
        """Create the provider's login flow; the host decides how to open its URL."""
        provider = provider_identifier.strip().lower()
        selected_store = store or self._store or current_credential_store()
        if provider == "chatgpt":
            return ChatGPTLoginFlow(selected_store)
        if provider == "cursor":
            return CursorLoginFlow(selected_store)
        raise AuthenticationError(f"{provider_identifier!r} does not expose a built-in OAuth flow.")

    def sign_out(self, provider_identifier: str, *, store: CredentialStore | None = None) -> None:
        """Remove account credentials from the caller-owned store."""
        self._store_for(store).clear(provider_identifier.strip().lower())

    async def valid_token(self, provider_identifier: str, *, store: CredentialStore | None = None) -> OAuthTokens:
        """Return a live OAuth token and refresh it once when the provider supports refresh."""
        provider = provider_identifier.strip().lower()
        if provider == "chatgpt":
            return await valid_chatgpt_tokens(store or self._store)
        if provider == "cursor":
            return await valid_cursor_tokens(store or self._store)
        raise AuthenticationError(f"{provider_identifier!r} has no built-in OAuth refresh adapter.")

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
        token = await self.valid_token(provider, store=store)
        if provider == "chatgpt" and isinstance(token, ChatGPTTokens):
            return request_chatgpt_headers(token, session_identifier)
        if provider == "cursor" and isinstance(token, CursorTokens):
            return request_cursor_headers(token, request_identifier or str(uuid.uuid4()))
        raise AuthenticationError(f"No request-header adapter exists for {provider_identifier!r}.")


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    """Common token fields shared by account-backed providers."""

    access_token: str
    refresh_token: str
    expires_at: float

    def is_expired(self, leeway_seconds: float = 60.0) -> bool:
        return time.time() >= self.expires_at - leeway_seconds


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


def _chatgpt_from_payload(payload: Mapping[str, Any], previous: ChatGPTTokens | None = None) -> ChatGPTTokens:
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
        refresh_token=str(payload.get("refresh_token") or (previous.refresh_token if previous else "")),
        id_token=id_token,
        account_id=str(account_id or (previous.account_id if previous else "")),
        email=str(claims.get("email") or (previous.email if previous else "")),
        expires_at=time.time() + float(payload.get("expires_in") or 3600),
    )


def _cursor_from_payload(payload: Mapping[str, Any], previous: CursorTokens | None = None) -> CursorTokens:
    access_token = str(payload.get("accessToken") or "")
    if not access_token:
        raise AuthenticationError("Cursor returned no access token.")
    claims = _jwt_claims(access_token)
    expiry = claims.get("exp")
    return CursorTokens(
        access_token=access_token,
        refresh_token=str(payload.get("refreshToken") or (previous.refresh_token if previous else "")),
        expires_at=float(expiry) if isinstance(expiry, (int, float)) else time.time() + 3600,
        account=str(claims.get("email") or claims.get("name") or (previous.account if previous else "")),
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
                    data={"grant_type": "refresh_token", "refresh_token": current.refresh_token, "client_id": "app_EMoamEEZ73f0CkXaXp7hrann", "scope": "openid profile email offline_access"},
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
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest()).rstrip(b"=").decode()
        parameters = {"response_type": "code", "client_id": "app_EMoamEEZ73f0CkXaXp7hrann", "redirect_uri": "http://localhost:1455/auth/callback", "scope": "openid profile email offline_access", "code_challenge": challenge, "code_challenge_method": "S256", "state": self._state}
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
                response = await client.post("https://auth.openai.com/oauth/token", data={"grant_type": "authorization_code", "code": self._captured["code"], "redirect_uri": "http://localhost:1455/auth/callback", "client_id": "app_EMoamEEZ73f0CkXaXp7hrann", "code_verifier": self._verifier})
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
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self._verifier.encode()).digest()).rstrip(b"=").decode()
        return "https://cursor.com/loginDeepControl?" + urllib.parse.urlencode({"challenge": challenge, "uuid": self._identifier, "mode": "login", "redirectTarget": "cli"})

    async def start(self) -> None:
        return

    async def wait(self, timeout: float = 300.0) -> CursorTokens:  # noqa: ASYNC109
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancelled:
                raise AuthenticationError("Cursor sign-in was cancelled.")
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get("https://api2.cursor.sh/auth/poll", params={"uuid": self._identifier, "verifier": self._verifier})
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
    return {"Authorization": f"Bearer {tokens.access_token}", "ChatGPT-Account-Id": tokens.account_id, "originator": "codex_cli_rs", "session-id": session_identifier or str(uuid.uuid4()), "Content-Type": "application/json", "Accept": "text/event-stream"}


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
    payload_digest = hashlib.sha256(token_segments[1].encode()).hexdigest()[:8] if len(token_segments) > 1 else "00000000"
    token_digest = hashlib.sha256(tokens.access_token.encode()).hexdigest()[:8]
    checksum = f"{checksum}{payload_digest}/{token_digest}"
    return {"Authorization": f"Bearer {tokens.access_token}", "Content-Type": "application/grpc-web+proto", "x-cursor-checksum": checksum, "x-cursor-client-version": "cli-2026.02.13-41ac335", "x-cursor-client-type": "cli", "x-cursor-timezone": _machine_time_zone(), "x-ghost-mode": "true", "x-cursor-streaming": "true", "x-request-id": request_identifier}


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
