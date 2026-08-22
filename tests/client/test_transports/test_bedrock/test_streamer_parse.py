"""BedrockStreamer.parse() — `vnd.amazon.eventstream` frame decoding at the
one seam the mixins call. Frames are built byte by byte so the buffering and
CRC paths are exercised for real."""

import json
import struct
import zlib

import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.bedrock.streamer import SyncBedrockStreamer
from luca.client.types import ChatCompletionRequest, UserMessage

_HEADER_TYPE_STRING = 7

REQUEST = ChatCompletionRequest(
    model="us.amazon.nova-pro-v1:0",
    provider="bedrock",
    messages=[UserMessage(content="hi")],
)


def _frame(event_type, payload, *, message_type="event", extra_headers=None):
    headers = {
        ":message-type": message_type,
        ":event-type": event_type,
        ":content-type": "application/json",
        **(extra_headers or {}),
    }
    header_bytes = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        header_bytes += bytes([len(nb)]) + nb + bytes([_HEADER_TYPE_STRING]) + struct.pack(">H", len(vb)) + vb
    body = json.dumps(payload).encode()
    total = 12 + len(header_bytes) + len(body) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + header_bytes + body
    return message + struct.pack(">I", zlib.crc32(message))


def _streamer() -> SyncBedrockStreamer:
    return SyncBedrockStreamer(REQUEST)


def test_one_frame_decodes_with_its_event_type_stamped_in():
    frame = _frame("messageStop", {"stopReason": "end_turn"})

    assert _streamer().parse(frame) == [{"type": "messageStop", "stopReason": "end_turn"}]


def test_two_frames_in_one_read_both_decode():
    chunk = _frame("messageStart", {"role": "assistant"}) + _frame("messageStop", {"stopReason": "end_turn"})

    assert _streamer().parse(chunk) == [
        {"type": "messageStart", "role": "assistant"},
        {"type": "messageStop", "stopReason": "end_turn"},
    ]


def test_a_frame_straddling_two_reads_completes_on_the_second():
    frame = _frame("messageStop", {"stopReason": "end_turn"})
    s = _streamer()

    assert s.parse(frame[:10]) == []
    assert s.parse(frame[10:]) == [{"type": "messageStop", "stopReason": "end_turn"}]


def test_byte_at_a_time_delivery_completes_on_the_last_byte():
    frame = _frame("messageStop", {"stopReason": "end_turn"})
    s = _streamer()

    events = [event for byte in frame for event in s.parse(bytes([byte]))]

    assert events == [{"type": "messageStop", "stopReason": "end_turn"}]


def test_a_corrupted_message_crc_is_a_stream_error():
    frame = bytearray(_frame("messageStop", {"stopReason": "end_turn"}))
    frame[-1] ^= 0xFF

    with pytest.raises(StreamError, match="CRC mismatch"):
        _streamer().parse(bytes(frame))


def test_an_exception_frame_carries_the_server_message():
    frame = _frame(
        "modelStreamErrorException",
        {"message": "The model is overloaded"},
        message_type="exception",
        extra_headers={":exception-type": "modelStreamErrorException"},
    )

    with pytest.raises(StreamError, match=r"\[modelStreamErrorException\]: The model is overloaded"):
        _streamer().parse(frame)


def test_a_partial_trailing_frame_is_dropped_silently():
    frame = _frame("messageStop", {"stopReason": "end_turn"})

    assert _streamer().parse(frame[: len(frame) // 2]) == []
