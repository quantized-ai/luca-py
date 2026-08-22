"""AnthropicTransport.acompletion_stream() — async streaming and the
streaming error mapping (a rejected request raises the typed exception at
`async with` / `with`)."""

import pytest

from luca.client.exceptions import AuthenticationError, RateLimitError
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
from tests.client._helpers.httpx_mocks import (
    error_response,
    make_async_client,
    make_sync_client,
    sse_response,
)

REQUEST = ChatCompletionRequest(
    model="claude-test",
    provider="anthropic",
    messages=[UserMessage(content="hi")],
)


def _sse(event_type: str, data: str) -> bytes:
    return f"event: {event_type}\ndata: {data}\n\n".encode()


CHUNKS = [
    _sse(
        "message_start",
        '{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5,"output_tokens":0}}}',
    ),
    _sse("content_block_start", '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'),
    _sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}'),
    _sse("content_block_stop", '{"type":"content_block_stop","index":0}'),
    _sse("message_delta", '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}'),
    _sse("message_stop", '{"type":"message_stop"}'),
]

USAGE = Usage(input_tokens=5, output_tokens=2, total_tokens=7)


async def test_anthropic_transport_acompletion_stream(anthropic_transport_factory):
    transport = anthropic_transport_factory(async_http_client=make_async_client(sse_response(CHUNKS)))

    async with transport.acompletion_stream(REQUEST) as s:
        events = [event async for event in s]

    assert events == [
        StartEvent(),
        TextStartEvent(index=0),
        TextDeltaEvent(index=0, delta="Hi"),
        TextEndEvent(index=0, content="Hi"),
        UsageEvent(usage=USAGE),
        FinishEvent(
            message=AssistantMessage(
                content=[TextBlock(text="Hi")],
                finish_reason="stop",
                provider_finish_reason="end_turn",
                provider="anthropic",
                model="claude-test",
                usage=USAGE,
            ),
            finish_reason="stop",
            provider_finish_reason="end_turn",
            usage=USAGE,
            tool_calls=[],
        ),
    ]


def test_a_rejected_stream_request_raises_the_mapped_error(anthropic_transport_factory):
    transport = anthropic_transport_factory(
        http_client=make_sync_client(
            error_response(401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
        ),
    )

    with (
        pytest.raises(AuthenticationError, match="invalid x-api-key") as excinfo,
        transport.completion_stream(REQUEST),
    ):
        pass

    assert excinfo.value.provider == "anthropic"


async def test_a_rejected_async_stream_request_raises_the_mapped_error(anthropic_transport_factory):
    transport = anthropic_transport_factory(
        async_http_client=make_async_client(
            error_response(
                429,
                {"error": {"type": "rate_limit_error", "message": "slow down"}},
                headers={"retry-after": "7"},
            ),
        ),
    )

    with pytest.raises(RateLimitError, match="slow down") as excinfo:
        async with transport.acompletion_stream(REQUEST):
            pass

    assert excinfo.value.provider == "anthropic"
    assert excinfo.value.retry_after == 7.0
