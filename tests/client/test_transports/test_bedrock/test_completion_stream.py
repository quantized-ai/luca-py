"""BedrockTransport.completion_stream() — end to end over a mock client,
sync and async. Usage (`metadata`) arrives after `messageStop`, so the
finish lands at wire end."""

import json
import struct
import zlib

import httpx

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_async_client, make_sync_client

_HEADER_TYPE_STRING = 7

REQUEST = ChatCompletionRequest(
    model="us.amazon.nova-pro-v1:0",
    provider="bedrock",
    messages=[UserMessage(content="hi")],
)

USAGE = Usage(input_tokens=2, output_tokens=3, total_tokens=5)

EXPECTED_EVENTS = [
    StartEvent(),
    TextStartEvent(index=0),
    TextDeltaEvent(index=0, delta="Hel"),
    TextDeltaEvent(index=0, delta="lo"),
    TextEndEvent(index=0, content="Hello"),
    UsageEvent(usage=USAGE),
    FinishEvent(
        message=AssistantMessage(
            content=[TextBlock(text="Hello")],
            finish_reason="stop",
            provider_finish_reason="end_turn",
            provider="bedrock",
            model="us.amazon.nova-pro-v1:0",
            usage=USAGE,
        ),
        finish_reason="stop",
        provider_finish_reason="end_turn",
        usage=USAGE,
        tool_calls=[],
    ),
]


def _frame(event_type, payload):
    headers = {
        ":message-type": "event",
        ":event-type": event_type,
        ":content-type": "application/json",
    }
    header_bytes = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        header_bytes += bytes([len(nb)]) + nb + bytes([_HEADER_TYPE_STRING]) + struct.pack(">H", len(vb)) + vb
    body = json.dumps(payload).encode()
    total = 12 + len(header_bytes) + len(body) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + header_bytes + body
    return message + struct.pack(">I", zlib.crc32(message))


def _body() -> bytes:
    return b"".join(
        [
            _frame("messageStart", {"role": "assistant"}),
            _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
            _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "lo"}}),
            _frame("contentBlockStop", {"contentBlockIndex": 0}),
            _frame("messageStop", {"stopReason": "end_turn"}),
            _frame("metadata", {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}),
        ]
    )


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        content=_body(),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )


def test_bedrock_transport_completion_stream(bedrock_transport_factory):
    transport = bedrock_transport_factory(http_client=make_sync_client(_handler))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events == EXPECTED_EVENTS


async def test_bedrock_transport_acompletion_stream(bedrock_transport_factory):
    transport = bedrock_transport_factory(async_http_client=make_async_client(_handler))

    async with transport.acompletion_stream(REQUEST) as s:
        events = [event async for event in s]

    assert events == EXPECTED_EVENTS
