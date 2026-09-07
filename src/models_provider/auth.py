"""Authentication values, profiles, and OAuth mechanics."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import os
import re
import secrets
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

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

    def __init__(self, flow: LoginFlow, values: dict[str, Any]) -> None:
        self._flow = flow
        self._values = values

    @property
    def values(self) -> dict[str, Any]:
        """Return this authorization's provider values for model construction."""
        return self._values

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

    def flow(self, values: dict[str, Any]) -> LoginFlow: ...

    def redirect_uri(self) -> str: ...

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

    async def valid_token(self, values: dict[str, Any]) -> OAuthTokens: ...

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
        values: dict[str, Any],
        *,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
    ) -> None:
        if not configuration.authorization_url:
            raise ValueError("authorization_url is required for a browser OAuth flow")
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._values = values
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
            self._values[self.provider_identifier] = tokens
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
        values: dict[str, Any],
        *,
        token_parser: Callable[[Mapping[str, Any], OAuthTokens | None], OAuthTokens] | None = None,
    ) -> None:
        if not configuration.device_authorization_url:
            raise ValueError("device_authorization_url is required for a device flow")
        self.provider_identifier = provider_identifier.strip().lower()
        self.configuration = configuration
        self._values = values
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
                    self._values[self.provider_identifier] = tokens
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
        flow_factory: Callable[[dict[str, Any]], LoginFlow] | None = None,
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
        if token_serializer is None:
            self._token_serializer = lambda tokens: {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "expires_at": tokens.expires_at,
            }
        else:
            self._token_serializer = token_serializer
        if token_deserializer is None:

            def deserialize(payload: Mapping[str, Any]) -> OAuthTokens:
                if not isinstance(payload, Mapping):
                    raise AuthenticationError("Stored OAuth credentials are invalid.")
                access_token = str(payload.get("access_token") or "")
                if not access_token:
                    raise AuthenticationError("Stored OAuth credentials contain no access token.")
                try:
                    expires_at = float(payload.get("expires_at") or 0.0)
                except (TypeError, ValueError) as error:
                    raise AuthenticationError(
                        "Stored OAuth credentials have an invalid expiry."
                    ) from error
                return OAuthTokens(
                    access_token=access_token,
                    refresh_token=str(payload.get("refresh_token") or ""),
                    expires_at=expires_at,
                )

            self._token_deserializer = deserialize
        else:
            self._token_deserializer = token_deserializer
        self._refresh_lock = asyncio.Lock()

    def flow(self, values: dict[str, Any]) -> LoginFlow:
        if (
            not self.configuration.authorization_url
            and not self.configuration.device_authorization_url
        ):
            raise AuthenticationError(
                f"{self.provider_identifier!r} has no interactive OAuth flow."
            )
        if self._flow_factory is not None:
            return self._flow_factory(values)
        flow_type = (
            DeviceLoginFlow if self.configuration.device_authorization_url else OAuthLoginFlow
        )
        return flow_type(
            self.provider_identifier,
            self.configuration,
            values,
            token_parser=self._token_parser,
        )

    def redirect_uri(self) -> str:
        """Return the redirect URI registered for this provider's OAuth client."""
        return self.configuration.redirect_uri

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

    async def valid_token(self, values: dict[str, Any]) -> OAuthTokens:
        tokens = self._stored_tokens(values.get(self.provider_identifier))
        if self.configuration.grant_type == "client_credentials":
            if tokens is not None and not tokens.is_expired():
                return tokens
            async with self._refresh_lock:
                current = self._stored_tokens(values.get(self.provider_identifier))
                if current is not None and not current.is_expired():
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
                refreshed = self._token_parser(payload, current)
                values[self.provider_identifier] = dict(self._token_serializer(refreshed))
                return refreshed
        if tokens is None:
            raise AuthenticationError(f"Not signed in to {self.provider_identifier}.")
        if not tokens.is_expired():
            return tokens
        async with self._refresh_lock:
            current = self._stored_tokens(values.get(self.provider_identifier))
            if current is not None and not current.is_expired():
                return current
            current = current or tokens
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
            values[self.provider_identifier] = dict(self._token_serializer(refreshed))
            return refreshed

    def _stored_tokens(self, value: Any) -> OAuthTokens | None:
        if isinstance(value, OAuthTokens):
            return value
        if isinstance(value, Mapping):
            try:
                return self._token_deserializer(value)
            except (AuthenticationError, TypeError, ValueError):
                return None
        return None

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


_VERTEX_ENVIRONMENT_VARIABLES = (
    "GOOGLE_VERTEX_PROJECT",
    "GOOGLE_VERTEX_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
_AWS_ENVIRONMENT_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
)

_AUTH_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "environment_variables": ("ANTHROPIC_API_KEY",),
        "api_key_header": "x-api-key",
        "api_key_prefix": "",
    },
    "azure": {
        "environment_variables": ("AZURE_API_KEY",),
        "credential_environment_variables": ("AZURE_RESOURCE_NAME",),
        "api_key_header": "api-key",
        "api_key_prefix": "",
    },
    "commandcode": {
        "environment_variables": ("COMMAND_CODE_API_KEY",),
        "default_base_url": "https://api.commandcode.ai/provider/v1",
    },
    "cursor": {"method": "oauth"},
    "google": {
        "environment_variables": (
            "GOOGLE_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
            "GEMINI_API_KEY",
        ),
        "api_key_header": "x-goog-api-key",
        "api_key_prefix": "",
    },
    "google-vertex": {
        "method": "environment",
        "credential_environment_variables": _VERTEX_ENVIRONMENT_VARIABLES,
    },
    "google-vertex-anthropic": {
        "method": "environment",
        "credential_environment_variables": _VERTEX_ENVIRONMENT_VARIABLES,
    },
    "amazon-bedrock": {
        "method": "environment",
        "credential_environment_variables": _AWS_ENVIRONMENT_VARIABLES,
    },
    "bedrock": {
        "method": "environment",
        "credential_environment_variables": _AWS_ENVIRONMENT_VARIABLES,
    },
    "github-copilot": {
        "environment_variables": ("GITHUB_TOKEN",),
        "default_base_url": "https://api.githubcopilot.com",
    },
    "azure-cognitive-services": {
        "environment_variables": ("AZURE_COGNITIVE_SERVICES_API_KEY",),
        "credential_environment_variables": ("AZURE_COGNITIVE_SERVICES_RESOURCE_NAME",),
        "api_key_header": "api-key",
        "api_key_prefix": "",
    },
    "opencode": {
        "environment_variables": ("OPENCODE_API_KEY",),
        "default_base_url": "https://opencode.ai/zen/v1",
        "headers": {"User-Agent": "opencode/0.0.0", "x-opencode-client": "models-provider"},
        "anonymous_api_key": "public",
    },
    "opencode-go": {
        "environment_variables": ("OPENCODE_API_KEY",),
        "default_base_url": "https://opencode.ai/zen/v1",
        "headers": {"User-Agent": "opencode/0.0.0", "x-opencode-client": "models-provider"},
        "anonymous_api_key": "public",
        "credential_identifier": "opencode",
    },
}


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
        *,
        catalogue: Any,
        oauth_adapters: Mapping[str, OAuthProvider] | None = None,
    ) -> None:
        self._values = values
        self._catalogue = catalogue
        self._oauth_adapters: dict[str, OAuthProvider] = dict(oauth_adapters or {})

    def profile(
        self, provider_identifier: str, *, environment_variables: tuple[str, ...] = ()
    ) -> ProviderAuthProfile:
        provider = provider_identifier.strip().lower()
        if not provider:
            raise ValueError("provider identifier cannot be empty")
        catalogue_record = (
            self._catalogue.provider(provider) if self._catalogue is not None else None
        )
        profile = ProviderAuthProfile(
            identifier=provider,
            environment_variables=(
                catalogue_record.environment_variables
                if catalogue_record is not None
                else environment_variables
            ),
            default_base_url=catalogue_record.api_base if catalogue_record is not None else "",
            method="api_key",
        )
        override = _AUTH_PROFILE_OVERRIDES.get(provider)
        if override is not None:
            profile = replace(profile, **override)
        return profile

    def resolve(
        self,
        provider_identifier: str,
        *,
        environment_variables: tuple[str, ...] = (),
    ) -> ApiKeyResolution:
        profile = self.profile(provider_identifier, environment_variables=environment_variables)
        provider = profile.identifier
        configured = self._values.get(profile.credential_identifier or provider)
        if configured is None:
            configured = self._values.get(provider_identifier)
        configured = _resolve_value(configured)
        environment: dict[str, str] = {}
        key = ""
        source = "none"
        if isinstance(configured, str):
            key = configured.strip()
            source = "configured" if key else "none"
        elif isinstance(configured, Mapping):
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
        if not key and not environment and profile.anonymous_api_key:
            key = profile.anonymous_api_key
            source = "anonymous"
        return ApiKeyResolution(
            provider=provider,
            api_key=key,
            api_base=profile.default_base_url,
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
