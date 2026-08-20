"""The SSE decoder. A response stream can be split at any byte, so the same
stream fed whole, one byte at a time, or cut at every offset must decode
identically."""

from __future__ import annotations

import pytest

from luca.agent.contrib.mcp.sse import SseDecoder, SseEvent

STREAM = (
    b": keep-alive\r\n"
    b"\r\n"
    b"event: message\r\n"
    b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\r\n'
    b"\r\n"
    b"id: 7\n"
    b'data: {"jsonrpc":"2.0","id":1,"result":{}}\n'
    b"\n"
)

EXPECTED = [
    SseEvent(data='{"jsonrpc":"2.0","method":"notifications/progress"}', name="message", id=None, retry=None),
    SseEvent(data='{"jsonrpc":"2.0","id":1,"result":{}}', name="message", id="7", retry=None),
]


def decode(chunks: list[bytes]) -> list[SseEvent]:
    """Every event a decoder produces from a chunking of one stream. A
    module-level factory rather than logic in a test body: the point of the
    parametrized cases is that all of them go through exactly this path."""
    decoder = SseDecoder()
    events: list[SseEvent] = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.flush())
    return events


WHOLE = [STREAM]
BYTE_BY_BYTE = [STREAM[index : index + 1] for index in range(len(STREAM))]
SPLIT_ACROSS_CRLF = [STREAM[:14], STREAM[14:]]  # cuts between the CR and the LF of line one
CHUNKED_ODDLY = [STREAM[:1], STREAM[1:37], STREAM[37:38], STREAM[38:]]


@pytest.mark.parametrize(
    "chunks",
    [WHOLE, BYTE_BY_BYTE, SPLIT_ACROSS_CRLF, CHUNKED_ODDLY],
    ids=["whole", "byte-by-byte", "split-across-crlf", "chunked-oddly"],
)
def test_every_chunking_of_one_stream_decodes_identically(chunks):
    assert decode(chunks) == EXPECTED


def test_a_stream_of_comments_produces_no_events():
    assert decode([b": keep-alive\n\n: still here\n\n"]) == []


def test_data_lines_are_joined_with_newlines():
    assert decode([b"data: one\ndata: two\n\n"]) == [SseEvent(data="one\ntwo")]


def test_a_bare_lf_terminates_a_line():
    assert decode([b"data: hello\n\n"]) == [SseEvent(data="hello")]


def test_a_bare_cr_terminates_a_line():
    assert decode([b"data: hello\r\r"]) == [SseEvent(data="hello")]


def test_a_field_with_no_space_after_the_colon_keeps_its_value():
    assert decode([b"data:hello\n\n"]) == [SseEvent(data="hello")]


def test_only_one_leading_space_is_stripped():
    assert decode([b"data:  hello\n\n"]) == [SseEvent(data=" hello")]


def test_a_named_event_carries_its_name():
    assert decode([b"event: ping\ndata: {}\n\n"]) == [SseEvent(data="{}", name="ping")]


def test_a_retry_field_is_carried():
    assert decode([b"retry: 5000\ndata: {}\n\n"]) == [SseEvent(data="{}", retry=5000)]


def test_an_unterminated_final_frame_is_flushed():
    # A server that closes right after the final response is common, and
    # discarding it would turn a completed call into a transport failure.
    assert decode([b"data: last\n"]) == [SseEvent(data="last")]


def test_an_event_name_does_not_leak_into_the_next_event():
    assert decode([b"event: first\ndata: a\n\ndata: b\n\n"]) == [SseEvent(data="a", name="first"), SseEvent(data="b")]
