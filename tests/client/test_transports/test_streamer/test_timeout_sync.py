"""The sync deadline — cooperative: a monotonic clock checked before every
read, best-effort by design (a read blocked on the socket runs to
completion first). Expiry mid-stream is one terminal ErrorEvent carrying the
SDK TimeoutError; expiry during the open raises it."""

import time

import httpx
import pytest

from luca.client.exceptions import TimeoutError as SDKTimeoutError
from luca.client.transports.faux import faux_assistant_message, faux_text
from luca.client.types import ChatCompletionRequest, StartEvent, UserMessage

from .conftest import faux_stream


def test_an_already_expired_deadline_yields_one_terminal_timeout_error():
    with faux_stream(faux_assistant_message([faux_text("never")]), timeout=0.0) as s:
        events = list(s)

    # Even an immediate failure narrates start → terminal.
    assert [e.type for e in events] == ["start", "error"]
    error = events[-1].error
    assert isinstance(error, SDKTimeoutError)
    assert str(error) == "stream exceeded timeout=0.0s"
    assert error.provider == "faux"


def test_a_generous_deadline_is_inert():
    with faux_stream(faux_assistant_message([faux_text("Hi")]), timeout=60.0) as s:
        events = list(s)

    assert events[-1].type == "finish"
    assert not any(e.type == "error" for e in events)


def test_expiry_mid_stream_stops_at_the_next_read():
    # The clock is checked before every read: rewinding the deadline after
    # the first event makes the next pull the terminal timeout error.
    with faux_stream(faux_assistant_message([faux_text("Hello")]), timeout=60.0) as s:
        it = iter(s)
        first = next(it)
        s.deadline = time.monotonic() - 1
        rest = list(it)

    assert first == StartEvent()
    assert [e.type for e in rest] == ["error"]
    assert isinstance(rest[0].error, SDKTimeoutError)
    assert str(rest[0].error) == "stream exceeded timeout=60.0s"


def test_expiry_during_the_open_raises(openai_transport_factory):
    def slow(request: httpx.Request) -> httpx.Response:
        time.sleep(0.05)
        return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})

    transport = openai_transport_factory(http_client=httpx.Client(transport=httpx.MockTransport(slow)))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with (
        pytest.raises(SDKTimeoutError, match=r"request did not open within timeout=0\.01s"),
        transport.completion_stream(request, timeout=0.01),
    ):
        pass
