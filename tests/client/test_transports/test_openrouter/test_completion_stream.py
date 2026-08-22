"""OpenRouterTransport streaming — the chat-completions wire under the
openrouter label: reasoning deltas become a thinking block, and the host
name is stamped on the message."""

from luca.client.transports import OpenRouterTransport
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_async_client, make_sync_client, sse_response

REQUEST = ChatCompletionRequest(
    model="deepseek/deepseek-r1",
    provider="openrouter",
    messages=[UserMessage(content="hi")],
)

CHUNKS = [
    b'data: {"choices":[{"index":0,"delta":{"reasoning":"Let me think."}}]}\n\n',
    b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4}}\n\n',
    b"data: [DONE]\n\n",
]

USAGE = Usage(input_tokens=1, output_tokens=3, total_tokens=4)

EXPECTED_EVENTS = [
    StartEvent(),
    ThinkingStartEvent(index=0),
    ThinkingDeltaEvent(index=0, delta="Let me think."),
    TextStartEvent(index=1),
    TextDeltaEvent(index=1, delta="Hi"),
    ThinkingEndEvent(index=0, content="Let me think."),
    TextEndEvent(index=1, content="Hi"),
    UsageEvent(usage=USAGE),
    FinishEvent(
        message=AssistantMessage(
            content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
            finish_reason="stop",
            provider_finish_reason="stop",
            provider="openrouter",
            model="deepseek/deepseek-r1",
            usage=USAGE,
        ),
        finish_reason="stop",
        provider_finish_reason="stop",
        usage=USAGE,
        tool_calls=[],
    ),
]


def _transport(**kwargs) -> OpenRouterTransport:
    return OpenRouterTransport(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        **kwargs,
    )


def test_openrouter_transport_completion_stream():
    transport = _transport(http_client=make_sync_client(sse_response(CHUNKS)))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events == EXPECTED_EVENTS


async def test_openrouter_transport_acompletion_stream():
    transport = _transport(async_http_client=make_async_client(sse_response(CHUNKS)))

    async with transport.acompletion_stream(REQUEST) as s:
        events = [event async for event in s]

    assert events == EXPECTED_EVENTS
