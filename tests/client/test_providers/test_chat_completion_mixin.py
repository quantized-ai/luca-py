"""ChatCompletionMixin forwards every method to the transport."""

import pytest

from luca.client.providers import OpenAIProvider
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextBlock,
    UserMessage,
)
from tests.client._helpers.stub_transports import StubTransport, TransportCall


REQUEST = ChatCompletionRequest(
    model="gpt-4o", provider="openai",
    messages=[UserMessage(content="hi")],
)
RESPONSE = ChatCompletionResponse(
    message=AssistantMessage(
        content=[TextBlock(text="ok")],
        finish_reason="stop", provider_finish_reason="stop",
        provider="openai", model="gpt-4o",
    ),
)


def test_completion_forwards_to_transport():
    stub = StubTransport(responses=[RESPONSE])
    provider = OpenAIProvider(transport=stub)
    assert provider.completion(REQUEST) is RESPONSE
    assert stub.calls == [TransportCall("completion", REQUEST)]


async def test_acompletion_forwards_to_transport():
    stub = StubTransport(responses=[RESPONSE])
    provider = OpenAIProvider(transport=stub)
    assert (await provider.acompletion(REQUEST)) is RESPONSE
    assert stub.calls == [TransportCall("acompletion", REQUEST)]


def test_completion_stream_forwards_to_transport():
    sentinel = object()
    stub = StubTransport(responses=[sentinel])
    provider = OpenAIProvider(transport=stub)
    assert provider.completion_stream(REQUEST) is sentinel
    assert stub.calls == [TransportCall("completion_stream", REQUEST)]


def test_acompletion_stream_forwards_to_transport():
    sentinel = object()
    stub = StubTransport(responses=[sentinel])
    provider = OpenAIProvider(transport=stub)
    assert provider.acompletion_stream(REQUEST) is sentinel
    assert stub.calls == [TransportCall("acompletion_stream", REQUEST)]
