"""httpx.MockTransport handlers — no assertions."""

from __future__ import annotations

from collections.abc import Callable

import httpx


def json_response(payload: dict, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def sse_response(
    chunks: list[bytes], status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    body = b"".join(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": "text/event-stream"},
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
