"""Provider-specific OAuth tokens, flows, and request headers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import html
import json
import os
import secrets
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

import httpx

from .credentials import CredentialStore, current_credential_store
from .errors import AuthenticationError
from .oauth import LoginFlow, OAuthProvider, OAuthTokens, _pkce_verifier


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
