"""Cursor provider, authentication, and native agent transport."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import platform
import struct
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import add_ai_message_chunks
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

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


# Cursor's agent endpoint is Connect/gRPC-Web, but the provider deliberately keeps
# the small codec here so its transport and wire contract stay in one file.
_VARINT = 0
_FIXED64 = 1
_LENGTH_DELIMITED = 2
TRAILER_FLAG = 0x80


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _tag(number: int, wire_type: int) -> bytes:
    return _varint((number << 3) | wire_type)


def scalar(number: int, value: int) -> bytes:
    return b"" if value == 0 else _tag(number, _VARINT) + _varint(value)


def boolean(number: int, value: bool) -> bytes:
    return b"" if not value else _tag(number, _VARINT) + b"\x01"


def blob(number: int, value: bytes) -> bytes:
    return _tag(number, _LENGTH_DELIMITED) + _varint(len(value)) + value


def text(number: int, value: str) -> bytes:
    return b"" if not value else blob(number, value.encode())


@dataclass(frozen=True, slots=True)
class WireField:
    number: int
    wire_type: int
    data: bytes = b""
    number_value: int = 0

    @property
    def is_message(self) -> bool:
        return self.wire_type == _LENGTH_DELIMITED

    def as_text(self) -> str:
        return self.data.decode("utf-8", "replace")


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    return value, offset


def parse(data: bytes) -> list[WireField]:
    fields: list[WireField] = []
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire_type = tag >> 3, tag & 0x7
        if wire_type == _LENGTH_DELIMITED:
            length, offset = _read_varint(data, offset)
            fields.append(WireField(number, wire_type, data[offset : offset + length]))
            offset += length
        elif wire_type == _VARINT:
            value, offset = _read_varint(data, offset)
            fields.append(WireField(number, wire_type, number_value=value))
        elif wire_type == _FIXED64:
            fields.append(WireField(number, wire_type, data=data[offset : offset + 8]))
            offset += 8
        else:
            break
    return fields


def first(fields: list[WireField], number: int) -> WireField | None:
    return next((entry for entry in fields if entry.number == number), None)


def message_at(fields: list[WireField], number: int) -> list[WireField] | None:
    entry = first(fields, number)
    return parse(entry.data) if entry is not None and entry.is_message else None


def string_at(fields: list[WireField], number: int) -> str:
    entry = first(fields, number)
    return entry.as_text() if entry is not None and entry.is_message else ""


def encode_value(value: Any) -> bytes:
    if value is None:
        return _tag(1, _VARINT) + b"\x00"
    if isinstance(value, bool):
        return boolean(4, value) or (_tag(4, _VARINT) + b"\x00")
    if isinstance(value, (int, float)):
        return _tag(2, _FIXED64) + struct.pack("<d", float(value))
    if isinstance(value, str):
        return blob(3, value.encode())
    if isinstance(value, (list, tuple)):
        return blob(6, b"".join(blob(1, encode_value(item)) for item in value))
    if isinstance(value, dict):
        return blob(
            5,
            b"".join(
                blob(1, text(1, str(key)) + blob(2, encode_value(item)))
                for key, item in value.items()
            ),
        )
    return text(3, str(value))


def decode_value(data: bytes) -> Any:
    for entry in parse(data):
        if entry.number == 1:
            return None
        if entry.number == 2 and entry.wire_type == _FIXED64:
            number = struct.unpack("<d", entry.data)[0]
            return int(number) if number.is_integer() else number
        if entry.number == 3 and entry.is_message:
            return entry.as_text()
        if entry.number == 4:
            return bool(entry.number_value)
        if entry.number == 5 and entry.is_message:
            result: dict[str, Any] = {}
            for pair_field in parse(entry.data):
                pair = parse(pair_field.data) if pair_field.is_message else []
                key = string_at(pair, 1)
                value = first(pair, 2)
                if key:
                    result[key] = decode_value(value.data) if value is not None else None
            return result
        if entry.number == 6 and entry.is_message:
            return [
                decode_value(item.data)
                for item in parse(entry.data)
                if item.number == 1 and item.is_message
            ]
    return None


def frame(payload: bytes, flags: int = 0) -> bytes:
    return bytes([flags]) + struct.pack(">I", len(payload)) + payload


@dataclass(slots=True)
class Deframer:
    _buffer: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> Iterator[tuple[int, bytes]]:
        self._buffer.extend(chunk)
        while len(self._buffer) >= 5:
            length = struct.unpack(">I", self._buffer[1:5])[0]
            if len(self._buffer) < length + 5:
                return
            flags = self._buffer[0]
            payload = bytes(self._buffer[5 : length + 5])
            del self._buffer[: length + 5]
            yield flags, payload


def parse_trailer(payload: bytes) -> tuple[int, str]:
    status = 0
    message = ""
    for line in payload.decode("utf-8", "replace").splitlines():
        name, separator, value = line.partition(":")
        if not separator:
            continue
        if name.strip().lower() == "grpc-status":
            try:
                status = int(value.strip())
            except ValueError:
                status = -1
        elif name.strip().lower() == "grpc-message":
            from urllib.parse import unquote

            message = unquote(value.strip())
    return status, message


def mcp_tool_definition(name: str, description: str, schema: dict[str, Any]) -> bytes:
    return (
        text(1, name)
        + text(2, description)
        + blob(3, encode_value(schema))
        + text(4, "models-provider")
        + text(5, name)
    )


def request_context_env(workspace: str, shell: str, os_version: str, time_zone: str) -> bytes:
    return (
        text(1, os_version)
        + text(2, workspace)
        + text(3, shell)
        + text(10, time_zone)
        + text(11, workspace)
    )


def request_context(env: bytes, tools: list[bytes]) -> bytes:
    return blob(4, env) + b"".join(blob(7, tool) for tool in tools)


def user_message(body: str, message_id: str) -> bytes:
    return text(1, body) + text(2, message_id) + scalar(4, 1)


def conversation_state(root_prompt_blob_ids: list[bytes]) -> bytes:
    return b"".join(blob(1, blob_id) for blob_id in root_prompt_blob_ids)


def model_details(model_id: str) -> bytes:
    return text(1, model_id) + text(3, model_id) + text(4, model_id)


def agent_run_request(
    *, state: bytes, action: bytes, model: bytes, tools: list[bytes], conversation_id: str
) -> bytes:
    parts = [blob(1, state), blob(2, action), blob(3, model)]
    if tools:
        parts.append(blob(4, b"".join(blob(1, tool) for tool in tools)))
    parts.append(text(5, conversation_id))
    return b"".join(parts)


def bidi_request_id(request_id: str) -> bytes:
    return text(1, request_id)


def bidi_append_request(request_id: str, sequence: int, payload: bytes) -> bytes:
    return (
        text(1, payload.hex())
        + blob(2, bidi_request_id(request_id))
        + _tag(3, _VARINT)
        + _varint(sequence)
    )


def client_message_run(run_request: bytes) -> bytes:
    return blob(1, run_request)


def client_message_kv(kv_id: int, result_field: int, result: bytes) -> bytes:
    return blob(3, scalar(1, kv_id) + blob(result_field, result))


@dataclass(slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ServerMessage:
    text_delta: str = ""
    thinking_delta: str = ""
    tool_call: ToolCall | None = None
    blob_request: BlobRequest | None = None
    turn_ended: bool = False
    output_token_delta: int = 0


@dataclass(slots=True)
class BlobRequest:
    kv_id: int
    blob_id: bytes
    blob_data: bytes | None = None

    @property
    def is_read(self) -> bool:
        return self.blob_data is None


def parse_server_message(payload: bytes) -> ServerMessage:
    message = ServerMessage()
    for entry in parse(payload):
        if not entry.is_message:
            continue
        if entry.number == 1:
            for update in parse(entry.data):
                if update.number == 1 and update.is_message:
                    message.text_delta = string_at(parse(update.data), 1)
                elif update.number == 2 and update.is_message:
                    tool_update = first(parse(update.data), 2)
                    if tool_update is None or not tool_update.is_message:
                        continue
                    fields = parse(tool_update.data)
                    mcp = message_at(fields, 15)
                    args = first(mcp, 1) if mcp is not None else None
                    if args is None or not args.is_message:
                        continue
                    call_fields = parse(args.data)
                    arguments: dict[str, Any] = {}
                    for item in call_fields:
                        if item.number != 2 or not item.is_message:
                            continue
                        pair = parse(item.data)
                        key = string_at(pair, 1)
                        value = first(pair, 2)
                        if key:
                            arguments[key] = decode_value(value.data) if value is not None else None
                    message.tool_call = ToolCall(
                        call_id=string_at(call_fields, 3),
                        tool_name=string_at(call_fields, 5) or string_at(call_fields, 1),
                        arguments=arguments,
                    )
                elif update.number == 4 and update.is_message:
                    message.thinking_delta = string_at(parse(update.data), 1)
                elif update.number == 8 and update.is_message:
                    counter = first(parse(update.data), 1)
                    message.output_token_delta = counter.number_value if counter else 0
                elif update.number == 14:
                    message.turn_ended = True
        elif entry.number == 4:
            fields = parse(entry.data)
            identifier = first(fields, 1)
            if (get_args := message_at(fields, 2)) is not None:
                target = first(get_args, 1)
                message.blob_request = BlobRequest(
                    kv_id=identifier.number_value if identifier is not None else 0,
                    blob_id=target.data if target is not None else b"",
                )
            elif (set_args := message_at(fields, 3)) is not None:
                target = first(set_args, 1)
                body = first(set_args, 2)
                message.blob_request = BlobRequest(
                    kv_id=identifier.number_value if identifier is not None else 0,
                    blob_id=target.data if target is not None else b"",
                    blob_data=body.data if body is not None else b"",
                )
    return message


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
        super().__init__(access_token, refresh_token, expires_at)
        object.__setattr__(self, "account", account)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], previous: CursorTokens | None = None
    ) -> CursorTokens:
        access_token = str(payload.get("accessToken") or "")
        if not access_token:
            raise AuthenticationError("Cursor returned no access token.")
        try:
            _, payload_part, _ = access_token.split(".")
            claims = json.loads(
                base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
            )
            claims = claims if isinstance(claims, dict) else {}
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            claims = {}
        expiry = claims.get("exp")
        return cls(
            access_token=access_token,
            refresh_token=str(
                payload.get("refreshToken") or (previous.refresh_token if previous else "")
            ),
            expires_at=float(expiry) if isinstance(expiry, (int, float)) else time.time() + 3600,
            account=str(
                claims.get("email") or claims.get("name") or (previous.account if previous else "")
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CursorTokens:
        """Rebuild a Cursor session from its persisted provider representation."""
        if not isinstance(payload, Mapping):
            raise AuthenticationError("Stored Cursor credentials are invalid.")
        access_token = str(payload.get("access_token") or "")
        if not access_token:
            raise AuthenticationError("Stored Cursor credentials contain no access token.")
        try:
            expires_at = float(payload.get("expires_at") or 0.0)
        except (TypeError, ValueError) as error:
            raise AuthenticationError(
                "Stored Cursor credentials have an invalid expiry."
            ) from error
        return cls(
            access_token=access_token,
            refresh_token=str(payload.get("refresh_token") or ""),
            account=str(payload.get("account") or ""),
            expires_at=expires_at,
        )

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> CursorTokens | None:
        value = values.get("cursor")
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls.from_mapping(value)
            except AuthenticationError:
                return None
        return None


_cursor_refresh_lock = asyncio.Lock()
_CURSOR_RUN_PATH = "/agent.v1.AgentService/RunSSE"
_CURSOR_APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"
_CURSOR_RUN_HOST = "https://api2.cursor.sh"


class CursorLoginFlow:
    """Cursor's browser login and polling flow, with no daemon dependency."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self._authorization = CursorAuthorizationRequest(
            "", state=str(uuid.uuid4()), code_verifier=_pkce_verifier()
        )
        self._cancelled = False

    @property
    def authorize_url(self) -> str:
        return self._authorization.authorize_url

    async def start(self) -> None:
        return

    async def wait(self, timeout: float = 300.0) -> CursorTokens:  # noqa: ASYNC109
        if self._cancelled:
            raise AuthenticationError("Cursor sign-in was cancelled.")
        tokens = await self._authorization.exchange(timeout=timeout)
        self._values["cursor"] = tokens
        return tokens

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

    async def exchange(self, code: str = "", *, timeout: float = 30.0) -> CursorTokens:  # noqa: ASYNC109
        del code
        deadline = time.monotonic() + timeout
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
                return CursorTokens.from_payload(response.json())
            except httpx.HTTPError as error:
                raise AuthenticationError(f"Cursor sign-in failed: {error}") from error
        raise AuthenticationError("Cursor sign-in is still pending.")


class CursorOAuth:
    """Cursor-specific OAuth adapter."""

    def flow(self, values: dict[str, Any]) -> LoginFlow:
        return CursorLoginFlow(values)

    async def valid_token(self, values: dict[str, Any]) -> CursorTokens:
        tokens = CursorTokens.from_values(values)
        if tokens is None:
            raise AuthenticationError("Not signed in to Cursor.")
        if not tokens.is_expired():
            return tokens
        async with _cursor_refresh_lock:
            current = CursorTokens.from_values(values) or tokens
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
                    refreshed = CursorTokens.from_payload(response.json(), current)
            except (httpx.HTTPError, AuthenticationError, TypeError, ValueError) as error:
                raise AuthenticationError(
                    f"Could not refresh the Cursor session: {error}"
                ) from error
            values["cursor"] = refreshed
            return refreshed

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
        return {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "account": tokens.account,
            "expires_at": tokens.expires_at,
        }

    def deserialize_tokens(self, payload: Mapping[str, Any]) -> OAuthTokens:
        return CursorTokens.from_mapping(payload)

    def request_headers(
        self, token: OAuthTokens, request_identifier: str, session_identifier: str
    ) -> Mapping[str, str]:
        if not isinstance(token, CursorTokens):
            raise AuthenticationError("Cursor OAuth returned an invalid token type.")
        del session_identifier
        request_identifier = request_identifier or str(uuid.uuid4())
        slot = int(time.time() // 1800) * 1800
        stamp = (slot * 1000) // 1_000_000
        obfuscated = bytearray(stamp.to_bytes(6, "big"))
        previous_byte = 165
        for index in range(len(obfuscated)):
            obfuscated[index] = ((obfuscated[index] ^ previous_byte) + index) & 0xFF
            previous_byte = obfuscated[index]
        checksum = base64.urlsafe_b64encode(bytes(obfuscated)).rstrip(b"=").decode()
        token_segments = token.access_token.split(".")
        payload_digest = (
            hashlib.sha256(token_segments[1].encode()).hexdigest()[:8]
            if len(token_segments) > 1
            else "00000000"
        )
        token_digest = hashlib.sha256(token.access_token.encode()).hexdigest()[:8]
        timezone = os.environ.get("TZ", "").strip()
        if not timezone:
            try:
                target = os.readlink("/etc/localtime")
                timezone = target.split("zoneinfo/", 1)[1] if "zoneinfo/" in target else ""
            except OSError:
                timezone = ""
        return {
            "Authorization": f"Bearer {token.access_token}",
            "Content-Type": "application/grpc-web+proto",
            "x-cursor-checksum": f"{checksum}{payload_digest}/{token_digest}",
            "x-cursor-client-version": "cli-2026.02.13-41ac335",
            "x-cursor-client-type": "cli",
            "x-cursor-timezone": timezone or (time.tzname[0] if time.tzname else "UTC"),
            "x-ghost-mode": "true",
            "x-cursor-streaming": "true",
            "x-request-id": request_identifier,
        }


class CursorChatModel(BaseChatModel):
    """LangChain model backed by Cursor's native agent stream."""

    model: str
    context_length: int = 0
    timeout: float | None = 300.0
    authentication: ProviderAuthentication = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "cursor"

    def context_window(self) -> int:
        return max(0, int(self.context_length or 0))

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        del tool_choice, parallel_tool_calls
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        if not isinstance(message.content, Sequence):
            return str(message.content)
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in message.content
            if isinstance(item, (str, Mapping))
        )

    def _transcript(self, messages: Sequence[BaseMessage]) -> str:
        blocks: list[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            elif isinstance(message, ToolMessage):
                heading = f"Tool result ({message.name or 'tool'})"
            elif isinstance(message, AIMessage):
                heading = "Assistant"
            else:
                heading = "User"
            blocks.append(f"## {heading}\n\n{self._message_text(message).strip()}")
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    arguments = call.get("args", {})
                    blocks.append(
                        f"## Assistant tool call ({call.get('name', 'tool')})\n\n"
                        f"{json.dumps(arguments, separators=(',', ':')) if not isinstance(arguments, str) else arguments}"
                    )
        return "\n\n".join(block for block in blocks if block.rsplit("\n\n", 1)[-1])

    def _turn(
        self, messages: Sequence[BaseMessage], tools: list[dict[str, Any]]
    ) -> tuple[bytes, dict[bytes, bytes]]:
        workspace = os.getcwd()
        blobs: dict[bytes, bytes] = {}
        system_prompt = "\n\n".join(
            self._message_text(message).strip()
            for message in messages
            if isinstance(message, SystemMessage) and self._message_text(message).strip()
        )
        root_prompt_blob_ids: list[bytes] = []
        if system_prompt:
            system_blob = json.dumps(
                {"role": "system", "content": system_prompt}, separators=(",", ":")
            ).encode()
            system_blob_id = hashlib.sha256(system_blob).digest()
            blobs[system_blob_id] = system_blob
            root_prompt_blob_ids.append(system_blob_id)
        tool_definitions = [
            mcp_tool_definition(
                name=(function := tool.get("function", tool)).get("name", ""),
                description=function.get("description", "") or "",
                schema=function.get("parameters") or {"type": "object", "properties": {}},
            )
            for tool in tools
        ]
        context = request_context(
            request_context_env(
                workspace=workspace,
                shell=os.environ.get("SHELL", "/bin/sh"),
                os_version=f"{platform.system().lower()} {platform.release()}",
                time_zone=os.environ.get("TZ", "") or (time.tzname[0] if time.tzname else "UTC"),
            ),
            tool_definitions,
        )
        message = user_message(self._transcript(messages), str(uuid.uuid4()))
        action = blob(1, blob(1, message) + blob(2, context))
        request = agent_run_request(
            state=conversation_state(root_prompt_blob_ids),
            action=action,
            model=model_details(self.model),
            tools=tool_definitions,
            conversation_id=str(uuid.uuid4()),
        )
        return client_message_run(request), blobs

    async def _astream(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del stop, run_manager
        request_id = str(uuid.uuid4())
        headers = dict(
            await self.authentication.request_headers("cursor", request_identifier=request_id)
        )
        turn, blobs = self._turn(messages, kwargs.get("tools") or [])
        append_headers = dict(headers)
        append_headers["Content-Type"] = "application/grpc-web+proto"
        async with httpx.AsyncClient(timeout=self.timeout, http2=False) as client:
            opening = asyncio.create_task(
                client.send(
                    client.build_request(
                        "POST",
                        f"{_CURSOR_RUN_HOST}{_CURSOR_RUN_PATH}",
                        content=frame(bidi_request_id(request_id)),
                        headers=headers,
                    ),
                    stream=True,
                )
            )
            append_sequence = 0

            async def append(payload: bytes) -> None:
                nonlocal append_sequence
                append_response = await client.post(
                    f"{_CURSOR_RUN_HOST}{_CURSOR_APPEND_PATH}",
                    content=frame(bidi_append_request(request_id, append_sequence, payload)),
                    headers=append_headers,
                )
                append_sequence += 1
                try:
                    append_response.raise_for_status()
                finally:
                    await append_response.aclose()

            try:
                await append(turn)
            except BaseException:
                opening.cancel()
                with contextlib.suppress(BaseException):
                    await opening
                raise
            response = await opening
            try:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise AuthenticationError(
                        f"Cursor agent service returned {response.status_code}: {detail[:800]}"
                    )
                deframer = Deframer()
                input_tokens = 0
                output_tokens = 0
                async for chunk in response.aiter_bytes():
                    for flags, payload in deframer.feed(chunk):
                        if flags & TRAILER_FLAG:
                            status, detail = parse_trailer(payload)
                            if status == 16:
                                raise AuthenticationError(
                                    "Cursor rejected the subscription token. Sign in again."
                                )
                            if status == 8:
                                raise RuntimeError(
                                    "Cursor reports this subscription's usage limit is reached."
                                )
                            if status != 0:
                                raise RuntimeError(
                                    f"Cursor agent stream failed ({status}): {detail or 'no detail'}"
                                )
                            continue
                        server_message = parse_server_message(payload)
                        output_tokens += server_message.output_token_delta
                        if server_message.blob_request is not None:
                            request = server_message.blob_request
                            if request.is_read:
                                body = blobs.get(request.blob_id, b"")
                                await append(client_message_kv(request.kv_id, 2, blob(1, body)))
                            else:
                                blobs[request.blob_id] = request.blob_data or b""
                                await append(client_message_kv(request.kv_id, 3, b""))
                        if server_message.text_delta:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(content=server_message.text_delta)
                            )
                        if server_message.thinking_delta:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={
                                        "reasoning_content": server_message.thinking_delta
                                    },
                                )
                            )
                        if server_message.tool_call is not None:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    tool_call_chunks=[
                                        {
                                            "index": 0,
                                            "id": server_message.tool_call.call_id
                                            or str(uuid.uuid4()),
                                            "name": server_message.tool_call.tool_name,
                                            "args": json.dumps(
                                                server_message.tool_call.arguments,
                                                separators=(",", ":"),
                                            ),
                                            "type": "tool_call_chunk",
                                        }
                                    ],
                                ),
                                generation_info={"finish_reason": "tool_calls"},
                            )
                            return
                        if server_message.turn_ended:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    usage_metadata={
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "total_tokens": input_tokens + output_tokens,
                                    },
                                ),
                                generation_info={"finish_reason": "stop"},
                            )
                            return
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    ),
                    generation_info={"finish_reason": "stop"},
                )
            finally:
                await response.aclose()

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks: list[AIMessageChunk] = []
        async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            chunks.append(
                chunk.message
                if isinstance(chunk.message, AIMessageChunk)
                else AIMessageChunk(content=chunk.message.content)
            )
        aggregate = add_ai_message_chunks(chunks[0], *chunks[1:]) if chunks else None
        if aggregate is None:
            return ChatResult(generations=[])
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=aggregate.content,
                        tool_calls=list(aggregate.tool_calls or []),
                        additional_kwargs=aggregate.additional_kwargs,
                        usage_metadata=aggregate.usage_metadata,
                    )
                )
            ]
        )

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        raise RuntimeError(
            "CursorChatModel cannot run synchronously inside an active event loop; use ainvoke."
        )


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
        del provider, values, request_parameters
        authentication.token("cursor")
        return CursorChatModel(
            model=record.model,
            context_length=record.context_length,
            timeout=timeout_seconds,
            authentication=authentication,
        )


__all__ = [
    "Cursor",
    "CursorAuthorizationRequest",
    "CursorChatModel",
    "CursorLoginFlow",
    "CursorOAuth",
    "CursorTokens",
]
