"""Provider-specific OAuth tokens, flows, and request headers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import html
import json
import os
import platform
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

import httpx

from .errors import AuthenticationError
from .oauth import (
    HostedAuthorization,
    LoginFlow,
    OAuthAuthorizationRequest,
    OAuthConfiguration,
    OAuthProvider,
    OAuthTokens,
    _pkce_verifier,
)


OPENAI_AUTHORIZATION_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_SCOPES = ("openid", "profile", "email", "offline_access")
OPENAI_LOOPBACK_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CLIENT_VERSION = "0.152.1"
OPENAI_ORIGINATOR = "codex_cli_rs"
OPENAI_OAUTH_CONFIGURATION = OAuthConfiguration(
    authorization_url=OPENAI_AUTHORIZATION_URL,
    token_url=OPENAI_TOKEN_URL,
    client_id=OPENAI_CLIENT_ID,
    scopes=OPENAI_SCOPES,
    redirect_uri=OPENAI_LOOPBACK_REDIRECT_URI,
)


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
class OpenAIAccountTokens(OAuthTokens):
    """OpenAI account subscription credentials."""

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


_openai_account_refresh_lock = asyncio.Lock()

_cursor_refresh_lock = asyncio.Lock()


def _openai_account_from_payload(
    payload: Mapping[str, Any], previous: OpenAIAccountTokens | None = None
) -> OpenAIAccountTokens:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("OpenAI returned an invalid account token response.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("OpenAI returned no account access token.")
    id_token = str(payload.get("id_token") or (previous.id_token if previous else ""))
    claims = _jwt_claims(id_token)
    auth_claim = claims.get("https://api.openai.com/auth")
    account_id = auth_claim.get("chatgpt_account_id", "") if isinstance(auth_claim, dict) else ""
    return OpenAIAccountTokens(
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


def _save(provider: str, credentials: OAuthTokens, values: dict[str, Any]) -> None:
    values[provider] = credentials


def openai_account_tokens_to_mapping(tokens: OpenAIAccountTokens) -> dict[str, Any]:
    """Return the provider-owned persisted representation of an OpenAI account session."""
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "id_token": tokens.id_token,
        "account_id": tokens.account_id,
        "email": tokens.email,
        "expires_at": tokens.expires_at,
    }


def openai_account_tokens_from_mapping(payload: Mapping[str, Any]) -> OpenAIAccountTokens:
    """Rebuild an OpenAI account session from a persisted provider representation."""
    if not isinstance(payload, Mapping):
        raise AuthenticationError("Stored OpenAI account credentials are invalid.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("Stored OpenAI account credentials contain no access token.")
    try:
        expires_at = float(payload.get("expires_at") or 0.0)
    except (TypeError, ValueError) as error:
        raise AuthenticationError(
            "Stored OpenAI account credentials have an invalid expiry."
        ) from error
    return OpenAIAccountTokens(
        access_token=access_token,
        refresh_token=str(payload.get("refresh_token") or ""),
        id_token=str(payload.get("id_token") or ""),
        account_id=str(payload.get("account_id") or ""),
        email=str(payload.get("email") or ""),
        expires_at=expires_at,
    )


def cursor_tokens_to_mapping(tokens: CursorTokens) -> dict[str, Any]:
    """Return the provider-owned persisted representation of a Cursor session."""
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "account": tokens.account,
        "expires_at": tokens.expires_at,
    }


def cursor_tokens_from_mapping(payload: Mapping[str, Any]) -> CursorTokens:
    """Rebuild a Cursor session from a previously persisted provider representation."""
    if not isinstance(payload, Mapping):
        raise AuthenticationError("Stored Cursor credentials are invalid.")
    access_token = str(payload.get("access_token") or "")
    if not access_token:
        raise AuthenticationError("Stored Cursor credentials contain no access token.")
    try:
        expires_at = float(payload.get("expires_at") or 0.0)
    except (TypeError, ValueError) as error:
        raise AuthenticationError("Stored Cursor credentials have an invalid expiry.") from error
    return CursorTokens(
        access_token=access_token,
        refresh_token=str(payload.get("refresh_token") or ""),
        account=str(payload.get("account") or ""),
        expires_at=expires_at,
    )


def openai_account_tokens(values: dict[str, Any]) -> OpenAIAccountTokens | None:
    value = values.get("openai")
    if isinstance(value, OpenAIAccountTokens):
        return value
    if isinstance(value, Mapping):
        try:
            return openai_account_tokens_from_mapping(value)
        except AuthenticationError:
            return None
    return None


def cursor_tokens(values: dict[str, Any]) -> CursorTokens | None:
    value = values.get("cursor")
    if isinstance(value, CursorTokens):
        return value
    if isinstance(value, Mapping):
        try:
            return cursor_tokens_from_mapping(value)
        except AuthenticationError:
            return None
    return None


async def valid_openai_account_tokens(values: dict[str, Any]) -> OpenAIAccountTokens:
    tokens = openai_account_tokens(values)
    if tokens is None:
        raise AuthenticationError("Not signed in to OpenAI.")
    if not tokens.is_expired():
        return tokens
    async with _openai_account_refresh_lock:
        current = openai_account_tokens(values) or tokens
        if not current.is_expired():
            return current
        if not current.refresh_token:
            raise AuthenticationError("OpenAI session expired; sign in again.")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    OPENAI_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": current.refresh_token,
                        "client_id": OPENAI_CLIENT_ID,
                        "scope": " ".join(OPENAI_SCOPES),
                    },
                )
                response.raise_for_status()
                refreshed = _openai_account_from_payload(response.json(), current)
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not refresh the OpenAI session: {error}") from error
        _save("openai", refreshed, values)
        return refreshed


async def valid_cursor_tokens(values: dict[str, Any]) -> CursorTokens:
    tokens = cursor_tokens(values)
    if tokens is None:
        raise AuthenticationError("Not signed in to Cursor.")
    if not tokens.is_expired():
        return tokens
    async with _cursor_refresh_lock:
        current = cursor_tokens(values) or tokens
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
        _save("cursor", refreshed, values)
        return refreshed


class OpenAIAccountLoginFlow:
    """PKCE loopback login. The host opens ``authorize_url`` and owns the browser policy."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self._authorization = OAuthAuthorizationRequest(
            "openai",
            OPENAI_OAUTH_CONFIGURATION,
            token_parser=_openai_account_from_payload,
        )
        self._server: HTTPServer | None = None
        self._captured: dict[str, str] = {}

    @property
    def authorize_url(self) -> str:
        return self._authorization.authorize_url

    async def start(self) -> None:
        flow = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format_string: str, *arguments: object) -> None:
                return

            def do_GET(self) -> None:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if urllib.parse.urlparse(self.path).path != "/auth/callback":
                    flow._captured["error"] = "Invalid callback path."
                elif query.get("state", [""])[0] != flow._authorization.state:
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

    async def wait(self, timeout: float = 300.0) -> OpenAIAccountTokens:  # noqa: ASYNC109
        if self._server is None:
            raise AuthenticationError("start() must be called before wait().")
        deadline = time.monotonic() + timeout
        try:
            while not self._captured:
                if time.monotonic() >= deadline:
                    raise AuthenticationError("OpenAI sign-in timed out.")
                await asyncio.to_thread(self._server.handle_request)
            if "code" not in self._captured:
                raise AuthenticationError(self._captured.get("error", "OpenAI sign-in failed."))
            tokens = await self._authorization.exchange(self._captured["code"])
            _save("openai", tokens, self._values)
            return tokens
        except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
            raise AuthenticationError(f"Could not complete OpenAI sign-in: {error}") from error
        finally:
            await self.close()

    async def close(self) -> None:
        if self._server is not None:
            await asyncio.to_thread(self._server.server_close)
            self._server = None


class CursorLoginFlow:
    """Cursor's browser login and polling flow, with no daemon dependency."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
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
                _save("cursor", tokens, self._values)
                return tokens
            except httpx.HTTPError as error:
                raise AuthenticationError(f"Cursor sign-in failed: {error}") from error
        raise AuthenticationError("Cursor sign-in timed out.")

    async def close(self) -> None:
        self._cancelled = True


class CursorAuthorizationRequest:
    """Host-owned Cursor authorization using Cursor's browser and polling flow."""

    def __init__(
        self,
        _redirect_uri: str,
        *,
        state: str | None = None,
        code_verifier: str | None = None,
        **_: Any,
    ) -> None:
        self._state = state or str(uuid.uuid4())
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
        return "https://cursor.com/loginDeepControl?" + urllib.parse.urlencode(
            {
                "challenge": challenge,
                "uuid": self._state,
                "mode": "login",
                "redirectTarget": "cli",
            }
        )

    async def exchange(self, _code: str = "") -> CursorTokens:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        "https://api2.cursor.sh/auth/poll",
                        params={"uuid": self._state, "verifier": self._code_verifier},
                    )
                if response.status_code == 404:
                    await asyncio.sleep(1)
                    continue
                response.raise_for_status()
                return _cursor_from_payload(response.json())
            except httpx.HTTPError as error:
                raise AuthenticationError(f"Cursor sign-in failed: {error}") from error
        raise AuthenticationError("Cursor sign-in is still pending.")


def _terminal_user_agent() -> str:
    program = os.environ.get("TERM_PROGRAM", "").strip()
    version = os.environ.get("TERM_PROGRAM_VERSION", "").strip()
    if program:
        normalized = "".join(char.lower() for char in program if char not in " -_.")
        names = {
            "appleterminal": "Apple_Terminal",
            "ghostty": "Ghostty",
            "iterm": "iTerm.app",
            "iterm2": "iTerm.app",
            "itermapp": "iTerm.app",
            "warp": "WarpTerminal",
            "warpterminal": "WarpTerminal",
            "vscode": "vscode",
            "wezterm": "WezTerm",
            "kitty": "kitty",
            "alacritty": "Alacritty",
            "konsole": "Konsole",
            "gnometerminal": "gnome-terminal",
            "vte": "VTE",
            "windowsterminal": "WindowsTerminal",
        }
        name = names.get(normalized, program)
        return f"{name}/{version}" if version else name
    return os.environ.get("TERM", "").strip() or "unknown"


def _openai_account_user_agent() -> str:
    if platform.system() == "Darwin":
        operating_system = "Mac OS"
        operating_system_version = platform.mac_ver()[0] or platform.release()
    else:
        operating_system = platform.system() or "unknown"
        operating_system_version = platform.release() or "unknown"
    architecture = platform.machine() or "unknown"
    originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", OPENAI_ORIGINATOR)
    return (
        f"{originator}/{OPENAI_CLIENT_VERSION} "
        f"({operating_system} {operating_system_version}; {architecture}) "
        f"{_terminal_user_agent()}"
    )


def request_openai_account_headers(
    tokens: OpenAIAccountTokens, session_identifier: str = ""
) -> dict[str, str]:
    """Headers required by the OpenAI account Responses endpoint."""
    session_id = session_identifier or str(uuid.uuid4())
    originator = os.environ.get("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", OPENAI_ORIGINATOR)
    return {
        "Authorization": f"Bearer {tokens.access_token}",
        "ChatGPT-Account-ID": tokens.account_id,
        "originator": originator,
        "User-Agent": _openai_account_user_agent(),
        "session-id": session_id,
        "thread-id": session_id,
        "x-client-request-id": session_id,
        "x-codex-window-id": f"{session_id}:0",
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
        flow_factory: Callable[[dict[str, Any]], LoginFlow],
        valid_token: Callable[[dict[str, Any]], Any],
        header_builder: Callable[[OAuthTokens, str, str], Mapping[str, str]],
        authorization_factory: Callable[..., HostedAuthorization],
        token_serializer: Callable[[OAuthTokens], Mapping[str, Any]],
        token_deserializer: Callable[[Mapping[str, Any]], OAuthTokens],
        registered_redirect_uri: str = "",
    ) -> None:
        self._flow_factory = flow_factory
        self._valid_token = valid_token
        self._header_builder = header_builder
        self._authorization_factory = authorization_factory
        self._token_serializer = token_serializer
        self._token_deserializer = token_deserializer
        self._registered_redirect_uri = registered_redirect_uri

    def flow(self, values: dict[str, Any]) -> LoginFlow:
        return self._flow_factory(values)

    async def valid_token(self, values: dict[str, Any]) -> OAuthTokens:
        return await self._valid_token(values)

    def redirect_uri(self) -> str:
        """Return the redirect URI accepted by the built-in OAuth client."""
        return self._registered_redirect_uri

    def authorization_request(
        self,
        redirect_uri: str,
        *,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> HostedAuthorization:
        return self._authorization_factory(
            redirect_uri,
            client_id=client_id,
            state=state,
            code_verifier=code_verifier,
        )

    def serialize_tokens(self, tokens: OAuthTokens) -> Mapping[str, Any]:
        return self._token_serializer(tokens)

    def deserialize_tokens(self, payload: Mapping[str, Any]) -> OAuthTokens:
        return self._token_deserializer(payload)

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]:
        return self._header_builder(token, request_identifier, session_identifier)


def _default_oauth_adapters() -> dict[str, OAuthProvider]:
    return {
        "openai": _BuiltInOAuthAdapter(
            OpenAIAccountLoginFlow,
            valid_openai_account_tokens,
            lambda token, _request, session: request_openai_account_headers(token, session),
            lambda redirect_uri, **kwargs: OAuthAuthorizationRequest(
                "openai",
                OPENAI_OAUTH_CONFIGURATION,
                token_parser=_openai_account_from_payload,
                redirect_uri=redirect_uri,
                **kwargs,
            ),
            openai_account_tokens_to_mapping,
            openai_account_tokens_from_mapping,
            OPENAI_LOOPBACK_REDIRECT_URI,
        ),
        "cursor": _BuiltInOAuthAdapter(
            CursorLoginFlow,
            valid_cursor_tokens,
            lambda token, request, _session: request_cursor_headers(
                token, request or str(uuid.uuid4())
            ),
            CursorAuthorizationRequest,
            cursor_tokens_to_mapping,
            cursor_tokens_from_mapping,
        ),
    }
