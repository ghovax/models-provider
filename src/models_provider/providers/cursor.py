"""Cursor provider implementation, account access, and model discovery."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel

from ..auth import (
    HostedAuthorization,
    LoginFlow,
    OAuthProvider,
    OAuthTokens,
    ProviderAuthentication,
    _pkce_verifier,
)
from ..catalogue import ModelRecord, ProviderRecord
from ..errors import AuthenticationError


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


_cursor_refresh_lock = asyncio.Lock()
_cursor_models: dict[str, dict[str, Any]] = {}


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        _, payload, _ = token.split(".")
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _save(provider: str, credentials: OAuthTokens, values: dict[str, Any]) -> None:
    values[provider] = credentials


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


RUN_PATH = "/agent.v1.AgentService/RunSSE"

APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"

AGENT_PRIVACY_URL = "https://agent.api5.cursor.sh"

AGENT_OPEN_URL = "https://agentn.api5.cursor.sh"

RUN_HOSTS = ("https://api2.cursor.sh", AGENT_PRIVACY_URL, AGENT_OPEN_URL)

USABLE_MODELS_URL = "https://api2.cursor.sh/agent.v1.AgentService/GetUsableModels"

AVAILABLE_MODELS_URL = "https://api2.cursor.sh/aiserver.v1.AiService/AvailableModels"

GET_ME_URL = "https://api2.cursor.sh/aiserver.v1.DashboardService/GetMe"

STATUS_RESOURCE_EXHAUSTED = 8

STATUS_UNAUTHENTICATED = 16

CLIENT_TYPE = "cli"

UNKNOWN_CONTEXT_WINDOW = 200_000


def _response_models(response: httpx.Response) -> list[Mapping[str, Any]]:
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("provider model response is not an object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("provider model response has no model list")
    return [entry for entry in models if isinstance(entry, Mapping)]


async def fetch_cursor_models(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if _cursor_models:
        return deepcopy(_cursor_models)
    try:
        tokens = await valid_cursor_tokens(values)
        headers = {
            **request_cursor_headers(tokens, str(uuid.uuid4())),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "connect-protocol-version": "1",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            usable_response = await client.post(
                USABLE_MODELS_URL, headers=headers, json={"customModelIds": []}
            )
            usable_response.raise_for_status()
            usable_entries = _response_models(usable_response)
            variants_response = await client.post(
                AVAILABLE_MODELS_URL,
                headers=headers,
                json={
                    "isNightly": False,
                    "excludeMaxNamedModels": True,
                    "additionalModelNames": [],
                    "useModelParameters": True,
                    "useReactModelPicker": True,
                },
            )
            variants_response.raise_for_status()
            variants: dict[str, dict[str, Any]] = {}
            for entry in _response_models(variants_response):
                if not entry.get("name"):
                    continue
                raw_variants = entry.get("variants", [])
                if not isinstance(raw_variants, list):
                    continue
                for variant in raw_variants:
                    if not isinstance(variant, Mapping):
                        continue
                    raw_parameters = variant.get("parameterValues", [])
                    if not isinstance(raw_parameters, list):
                        continue
                    parameters = {
                        str(item.get("id")): str(item.get("value"))
                        for item in raw_parameters
                        if isinstance(item, Mapping) and item.get("id") is not None
                    }
                    context_match = re.fullmatch(
                        r"(\d+(?:\.\d+)?)([km])?", parameters.get("context", "").lower()
                    )
                    context = (
                        round(
                            float(context_match.group(1))
                            * {"k": 1_000, "m": 1_000_000}.get(context_match.group(2) or "", 1)
                        )
                        if context_match
                        else 0
                    )
                    variants[str(entry["name"])] = {
                        "server_model": str(entry.get("serverModelName") or entry["name"]),
                        "maximum_mode": variant.get("isMaxMode") is True,
                        "parameters": tuple(sorted(parameters.items())),
                        "context": context,
                    }
            if not usable_entries:
                usable_entries = [
                    {"modelId": model_name, "displayName": model_name} for model_name in variants
                ]
            for entry in usable_entries:
                model_identifier = entry.get("modelId") or entry.get("displayModelId")
                if not model_identifier:
                    continue
                model_name = str(model_identifier)
                variant = variants.get(model_name)
                if variant is None:
                    names = [name for name in variants if model_name.startswith(name)]
                    variant = variants[max(names, key=len)] if names else None
                _cursor_models[model_name] = {
                    "name": entry.get("displayName") or model_name,
                    "context": variant.get("context", 0) if variant else 0,
                    "variant": variant,
                }
    except (AuthenticationError, httpx.HTTPError, ValueError, TypeError):
        return {}
    return deepcopy(_cursor_models)


def cached_cursor_models() -> dict[str, dict[str, Any]]:
    return deepcopy(_cursor_models)


def clear_cursor_models_cache() -> None:
    _cursor_models.clear()


async def display_cursor_account(tokens: CursorTokens, values: dict[str, Any]) -> str:
    if tokens.account:
        return tokens.account
    try:
        headers = {
            **request_cursor_headers(tokens, str(uuid.uuid4())),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "connect-protocol-version": "1",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(GET_ME_URL, headers=headers, json={})
            response.raise_for_status()
            payload = response.json()
        for key in ("email", "userEmail", "cachedEmail"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and value.strip():
                updated = CursorTokens(
                    tokens.access_token, tokens.refresh_token, value.strip(), tokens.expires_at
                )
                values["cursor"] = updated
                return value.strip()
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    return ""


class CursorOAuth:
    """Cursor-specific OAuth adapter."""

    def flow(self, values: dict[str, Any]) -> LoginFlow:
        return CursorLoginFlow(values)

    async def valid_token(self, values: dict[str, Any]) -> CursorTokens:
        return await valid_cursor_tokens(values)

    def redirect_uri(self) -> str:
        return ""

    def authorization_request(
        self,
        redirect_uri: str,
        *,
        client_id: str = "",
        state: str | None = None,
        code_verifier: str | None = None,
    ) -> HostedAuthorization:
        return CursorAuthorizationRequest(
            redirect_uri,
            state=state,
            code_verifier=code_verifier,
            client_id=client_id,
        )

    def serialize_tokens(self, tokens: OAuthTokens) -> Mapping[str, Any]:
        if not isinstance(tokens, CursorTokens):
            raise AuthenticationError("Cursor OAuth returned an invalid token type.")
        return cursor_tokens_to_mapping(tokens)

    def deserialize_tokens(self, payload: Mapping[str, Any]) -> OAuthTokens:
        return cursor_tokens_from_mapping(payload)

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]:
        if not isinstance(token, CursorTokens):
            raise AuthenticationError("Cursor OAuth returned an invalid token type.")
        return request_cursor_headers(token, request_identifier or str(uuid.uuid4()))


class Cursor:
    """Concrete Cursor provider implementation."""

    identifier = "cursor"
    oauth: OAuthProvider = CursorOAuth()

    def supports(self, record: ModelRecord) -> bool:
        return record.provider == self.identifier

    def chat(
        self,
        record: ModelRecord,
        provider: ProviderRecord,
        *,
        values: dict[str, Any],
        authentication: ProviderAuthentication,
        timeout_seconds: float | None,
        request_parameters: Mapping[str, Any],
    ) -> BaseChatModel:
        from .litellm import LiteLLM

        return LiteLLM().chat(
            record,
            provider,
            values=values,
            authentication=authentication,
            timeout_seconds=timeout_seconds,
            request_parameters=request_parameters,
        )


__all__ = [
    "Cursor",
    "CursorAuthorizationRequest",
    "CursorLoginFlow",
    "CursorOAuth",
    "CursorTokens",
    "cursor_tokens",
    "cursor_tokens_from_mapping",
    "cursor_tokens_to_mapping",
    "request_cursor_headers",
    "valid_cursor_tokens",
    "fetch_cursor_models",
    "cached_cursor_models",
    "clear_cursor_models_cache",
    "display_cursor_account",
]
