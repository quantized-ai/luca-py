"""Stream protocol-violation paths.

True user-initiated cancellation requires a real socket — with
`httpx.MockTransport` the body is pre-buffered, so `stream.cancel()` cannot
interrupt mid-iteration. That path is exercised in integration testing rather
than here. This file covers the structural invariants we *can* exercise:
premature termination, malformed JSON.
"""

from luca.client.types import ChatCompletionRequest, UserMessage
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response


REQUEST = ChatCompletionRequest(
    model="gpt-4o", provider="openai",
    messages=[UserMessage(content="hi")],
)


def _data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


def test_premature_finish_emits_error_event(openai_transport_factory):
    """A stream that ends without RawFinish emits a terminal ErrorEvent."""
    chunks = [
        _data('{"choices":[{"index":0,"delta":{"content":"oops"}}]}'),
        _data("[DONE]"),
    ]
    client = make_sync_client(sse_response(chunks))
    transport = openai_transport_factory(http_client=client)

    events = []
    with transport.completion_stream(REQUEST) as s:
        for ev in s:
            events.append(ev)

    assert events[-1].type == "error"
    assert "RawFinish" in str(events[-1].error)
    # Partial content should still be present.
    assert events[-1].partial_message.content[0].text == "oops"


def test_cancellation_path_via_accumulator():
    """The accumulator builds a cancelled FinishEvent when asked."""
    from luca.client.types.streaming import (
        RawBlockStart,
        RawBlockStop,
        RawTextDelta,
        _ChatCompletionAccumulator,
    )

    class _Req:
        model = "m"
        response_format = None

    acc = _ChatCompletionAccumulator(request=_Req(), provider="openai")
    # Emit some content, but no RawFinish.
    list(acc.handle_raw(RawBlockStart(index=0, block_type="text")))
    list(acc.handle_raw(RawTextDelta(index=0, text="Hi")))
    list(acc.handle_raw(RawBlockStop(index=0)))

    def _classify(provider_value, message):
        return ("stop", None) if provider_value == "stop" else (provider_value, None)

    finish = acc.build_terminal_finish(classify_finish=_classify, cancelled=True)
    assert finish.cancelled is True
    # No terminal arrived → finish_reason / provider_finish_reason are None.
    assert finish.finish_reason is None
    assert finish.provider_finish_reason is None
    # The partial message is still on the event.
    assert finish.message.content[0].text == "Hi"
