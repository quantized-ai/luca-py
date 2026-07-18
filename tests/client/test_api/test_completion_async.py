"""Async path forwarding."""

import pytest

from luca.client import acompletion
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    UserMessage,
)


RESP = ChatCompletionResponse(
    message=AssistantMessage(
        content=[TextBlock(text="ok")],
        finish_reason="stop", provider_finish_reason="stop",
        provider="stub", model="m",
    ),
)


async def test_acompletion_forwards_to_provider(stub_provider):
    stub_provider.configure(responses=[RESP])
    result = await acompletion(model="stub:m", messages=[UserMessage(content="hi")])
    assert result is RESP
    assert stub_provider.instances[0].calls[0].method == "acompletion"


async def test_acompletion_stream_returns_synchronously(stub_provider):
    sentinel = object()
    stub_provider.configure(responses=[sentinel])
    from luca.client import acompletion_stream

    # NO await — function must return synchronously.
    result = acompletion_stream(model="stub:m", messages=[UserMessage(content="hi")])
    assert result is sentinel
