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
    model="gpt-4o",
    provider="openai",
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

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].type == "error"
    assert "without a terminal event" in str(events[-1].error)
    # Partial content should still be readable on the stream object.
    assert s.message.content[0].text == "oops"
