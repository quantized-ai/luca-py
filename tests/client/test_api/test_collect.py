"""collect() at the helper layer: drain the stream and return the same
ChatCompletionResponse the non-streaming call would produce; re-raise on an
ErrorEvent."""

import pytest

from luca.client import acompletion_stream, completion_stream
from luca.client.exceptions import StreamError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
)
from luca.client.types import AssistantMessage, TextBlock, Usage, UserMessage

USAGE = Usage(input_tokens=3, output_tokens=5, total_tokens=8)

EXPECTED_MESSAGE = AssistantMessage(
    content=[TextBlock(text="Hi")],
    finish_reason="stop",
    provider_finish_reason="stop",
    provider="faux",
    model="test-model",
    usage=USAGE,
)


def _faux(scripted) -> FauxProvider:
    faux = FauxProvider()
    faux.set_responses([scripted])
    return faux


def test_collect_returns_the_full_response():
    faux = _faux(faux_assistant_message([faux_text("Hi")], finish_reason="stop", usage=USAGE))
    stream = completion_stream("faux:test-model", [UserMessage(content="hi")], provider=faux)

    response = stream.collect()

    assert response.messages == [EXPECTED_MESSAGE]


async def test_async_collect_returns_the_full_response():
    faux = _faux(faux_assistant_message([faux_text("Hi")], finish_reason="stop", usage=USAGE))
    stream = acompletion_stream("faux:test-model", [UserMessage(content="hi")], provider=faux)

    response = await stream.collect()

    assert response.messages == [EXPECTED_MESSAGE]


def test_collect_reraises_the_stream_error():
    faux = _faux(faux_assistant_message([faux_text("partial")], error=faux_error("boom")))
    stream = completion_stream("faux:test-model", [UserMessage(content="hi")], provider=faux)

    with pytest.raises(StreamError, match="boom"):
        stream.collect()


async def test_async_collect_reraises_the_stream_error():
    faux = _faux(faux_assistant_message([faux_text("partial")], error=faux_error("boom")))
    stream = acompletion_stream("faux:test-model", [UserMessage(content="hi")], provider=faux)

    with pytest.raises(StreamError, match="boom"):
        await stream.collect()


def test_collect_on_a_consumed_stream_raises():
    faux = _faux(faux_assistant_message([faux_text("Hi")]))
    stream = completion_stream("faux:test-model", [UserMessage(content="hi")], provider=faux)

    with stream as s:
        list(s)

    with pytest.raises(StreamError, match="already been consumed"):
        stream.collect()
