"""Provider-independent OAuth contracts and flows."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import secrets
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

from .credentials import CredentialStore
from .errors import AuthenticationError


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


class OAuthAuthorization:
    """Host-facing OAuth authorization with a URL and explicit completion."""

    def __init__(self, flow: LoginFlow) -> None:
        self._flow = flow

    @property
    def url(self) -> str:
        """Authorization URL for the host to display or open."""
        return self._flow.authorize_url

    async def complete(self, timeout: float = 300.0) -> OAuthTokens:  # noqa: ASYNC109
        """Wait for the user-controlled authorization and persist its token."""
        return await self._flow.wait(timeout)

    async def close(self) -> None:
        """Stop the callback listener without completing authorization."""
        await self._flow.close()


@runtime_checkable
class HostedAuthorization(Protocol):
    """Authorization contract for a host-owned browser sign-in."""

    @property
    def authorize_url(self) -> str: ...

    @property
    def state(self) -> str: ...

    @property
    def code_verifier(self) -> str: ...

    async def exchange(self, code: str = "") -> OAuthTokens: ...


@runtime_checkable
class OAuthProvider(Protocol):
    """An OAuth adapter used by :class:`ProviderAuthentication`."""

    def flow(self, store: CredentialStore) -> LoginFlow: ...

    def authorization_request(
        self,
        redirect_uri: str,
        *,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> HostedAuthorization: ...

    def serialize_tokens(self, tokens: OAuthTokens) -> Mapping[str, Any]: ...

    def deserialize_tokens(self, payload: Mapping[str, Any]) -> OAuthTokens: ...

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


def _oauth_tokens_to_mapping(tokens: OAuthTokens) -> dict[str, Any]:
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": tokens.expires_at,
    }


def _oauth_tokens_from_mapping(payload: Mapping[str, Any]) -> OAuthTokens:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("Stored OAuth credentials are invalid.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("Stored OAuth credentials contain no access token.")
    try:
        expires_at = float(payload.get("expires_at") or 0.0)
    except (TypeError, ValueError) as error:
        raise AuthenticationError("Stored OAuth credentials have an invalid expiry.") from error
    return OAuthTokens(
        access_token=access_token,
        refresh_token=str(payload.get("refresh_token") or ""),
        expires_at=expires_at,
    )


class OAuthAuthorizationRequest:
    """Provider-neutral authorization-code request for a host-owned callback."""

    def __init__(
        self,
        provider_identifier: str,
        configuration: OAuthConfiguration,
        *,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
        redirect_uri: str | None = None,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> None:
        if not configuration.authorization_url:
            raise ValueError("authorization_url is required for a browser OAuth request")
        selected_redirect_uri = redirect_uri or configuration.redirect_uri
        parsed = urllib.parse.urlparse(selected_redirect_uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("redirect_uri must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("redirect_uri must use HTTPS outside localhost")
        selected_client_id = client_id.strip() or configuration.client_id
        if not selected_client_id:
            raise ValueError("client_id cannot be empty")
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = replace(
            configuration,
            redirect_uri=selected_redirect_uri,
            client_id=selected_client_id,
        )
        self._token_parser = token_parser or _oauth_tokens_from_payload
        self._state = state or secrets.token_urlsafe(24)
        self._code_verifier = code_verifier or _pkce_verifier()

    @property
    def state(self) -> str:
        return self._state

    @property
    def code_verifier(self) -> str:
        return self._code_verifier

    @property
    def authorize_url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self._code_verifier.encode()).digest())
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

    async def exchange(self, code: str = "") -> OAuthTokens:
        if not code.strip():
            raise AuthenticationError("OAuth authorization returned no code.")
        data = {
            **self.configuration.token_parameters,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.configuration.redirect_uri,
            "client_id": self.configuration.client_id,
            "code_verifier": self._code_verifier,
        }
        auth = None
        if self.configuration.token_endpoint_auth_method == "client_secret_post":
            data["client_secret"] = self.configuration.client_secret
        elif self.configuration.token_endpoint_auth_method == "client_secret_basic":
            auth = (self.configuration.client_id, self.configuration.client_secret)
            data.pop("client_id", None)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(self.configuration.token_url, data=data, auth=auth)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, Mapping):
                raise AuthenticationError("OAuth returned an invalid token response.")
            return self._token_parser(payload, None)
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(
                f"Could not complete {self.provider_identifier} sign-in: {error}"
            ) from error


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
        if redirect.scheme != "http" or redirect.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
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
        authorization_factory: Callable[..., HostedAuthorization] | None = None,
        token_serializer: Callable[[OAuthTokens], Mapping[str, Any]] | None = None,
        token_deserializer: Callable[[Mapping[str, Any]], OAuthTokens] | None = None,
    ) -> None:
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._flow_factory = flow_factory
        self._token_parser = token_parser or _oauth_tokens_from_payload
        self._header_builder = header_builder
        self._authorization_factory = authorization_factory
        self._token_serializer = token_serializer or _oauth_tokens_to_mapping
        self._token_deserializer = token_deserializer or _oauth_tokens_from_mapping
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

    def authorization_request(
        self,
        redirect_uri: str,
        *,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> HostedAuthorization:
        if self._authorization_factory is not None:
            return self._authorization_factory(
                redirect_uri,
                client_id=client_id,
                state=state,
                code_verifier=code_verifier,
            )
        if not self.configuration.authorization_url:
            raise AuthenticationError(
                f"{self.provider_identifier!r} has no callback-based OAuth flow."
            )
        return OAuthAuthorizationRequest(
            self.provider_identifier,
            self.configuration,
            token_parser=self._token_parser,
            redirect_uri=redirect_uri,
            client_id=client_id,
            state=state,
            code_verifier=code_verifier,
        )

    def serialize_tokens(self, tokens: OAuthTokens) -> Mapping[str, Any]:
        return self._token_serializer(tokens)

    def deserialize_tokens(self, payload: Mapping[str, Any]) -> OAuthTokens:
        return self._token_deserializer(payload)

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


def _pkce_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
