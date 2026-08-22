"""Builders for streamer-machinery tests: faux streams for the shared
mixin behavior, a chat-completions transport over a mock client for the
paths the faux cannot produce (multi-event chunks, HTTP opens)."""

import pytest

from luca.client.transports import OpenAITransport
from luca.client.transports.faux import FauxTransport
from luca.client.types import ChatCompletionRequest, UserMessage


def faux_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[UserMessage(content="hi")])


def faux_stream(message, timeout=None):
    faux = FauxTransport()
    faux.set_responses([message])
    return faux.completion_stream(faux_request(), timeout=timeout)


def faux_astream(message, timeout=None):
    faux = FauxTransport()
    faux.set_responses([message])
    return faux.acompletion_stream(faux_request(), timeout=timeout)


async def drain(stream) -> list:
    return [event async for event in stream]


@pytest.fixture
def openai_transport_factory():
    def make(*, http_client=None, async_http_client=None):
        return OpenAITransport(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
