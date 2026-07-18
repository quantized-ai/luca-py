"""Async streaming + async cancellation through the faux transport."""

import asyncio

import pytest

from luca.client.transports.faux import (
    FauxTransport,
    faux_assistant_message,
    faux_text,
)
from luca.client.types import ChatCompletionRequest, UserMessage


def _req():
    return ChatCompletionRequest(
        model="test-model", messages=[UserMessage(content="hi")],
    )


async def test_async_stream_returns_synchronously():
    """acompletion_stream is NOT async — it must return the stream object directly."""
    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text("hello")], finish_reason="stop")])
    s = faux.acompletion_stream(_req())
    assert hasattr(s, "__aiter__")
    assert hasattr(s, "__aenter__")


async def test_async_stream_iterates_and_finishes():
    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text("Hello world")], finish_reason="stop")])

    events = []
    async with faux.acompletion_stream(_req()) as s:
        async for ev in s:
            events.append(ev)
    assert events[0].type == "start"
    assert events[-1].type == "finish"
    assert events[-1].finish_reason == "stop"
