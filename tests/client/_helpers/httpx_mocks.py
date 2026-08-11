"""httpx.MockTransport handlers — no assertions."""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Callable

import httpx

_EVENTSTREAM_HEADER_TYPE_STRING = 7


def json_response(payload: dict, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def sse_response(
    chunks: list[bytes],
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    body = b"".join(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    return handler


def ndjson_response(
    frames: list[dict],
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Ollama's streaming reply: one JSON object per line."""
    body = "".join(json.dumps(frame) + "\n" for frame in frames).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": "application/x-ndjson"},
        )

    return handler


def eventstream_frame(event_type: str, payload: dict, *, message_type: str = "event") -> bytes:
    """One `vnd.amazon.eventstream` frame, built byte by byte.

    Prelude, its CRC32, the header block, the JSON body, then the message
    CRC32 — the real framing, so the decoder's buffering and checksum paths
    are exercised rather than mocked away."""
    headers = {
        ":message-type": message_type,
        ":event-type": event_type,
        ":content-type": "application/json",
    }
    header_bytes = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        header_bytes += (
            bytes([len(nb)]) + nb + bytes([_EVENTSTREAM_HEADER_TYPE_STRING]) + struct.pack(">H", len(vb)) + vb
        )
    body = json.dumps(payload).encode()
    total = 12 + len(header_bytes) + len(body) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + header_bytes + body
    return message + struct.pack(">I", zlib.crc32(message))


def eventstream_response(
    frames: list[bytes],
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Bedrock's streaming reply. The SSE sibling, for Amazon's binary
    framing."""
    body = b"".join(frames)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": "application/vnd.amazon.eventstream"},
        )

    return handler


def error_response(
    status_code: int,
    body: dict | None = None,
    headers: dict | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=body or {}, headers=headers or {})

    return handler


def make_sync_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_async_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
