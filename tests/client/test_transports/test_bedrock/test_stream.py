"""The `vnd.amazon.eventstream` decoder → RawStreamEvent list.

Frames are built byte by byte here so the buffering and CRC paths are exercised
for real, not mocked away. The parser is driven directly with a fake response
whose `iter_bytes` we control, which lets a frame be split across reads or two
frames arrive in one.
"""

import json
import struct
import zlib

import pytest

from luca.client.exceptions import StreamError
from luca.client.types.completion import Usage
from luca.client.types.streaming import (
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
)
from luca.client.transports.bedrock.stream import BedrockChatCompletionStream

_HEADER_TYPE_STRING = 7


def _frame(event_type, payload, *, message_type="event"):
    headers = {
        ":message-type": message_type,
        ":event-type": event_type,
        ":content-type": "application/json",
    }
    header_bytes = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        header_bytes += (
            bytes([len(nb)]) + nb + bytes([_HEADER_TYPE_STRING])
            + struct.pack(">H", len(vb)) + vb
        )
    body = json.dumps(payload).encode()
    total = 12 + len(header_bytes) + len(body) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + header_bytes + body
    return message + struct.pack(">I", zlib.crc32(message))


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_bytes(self):
        yield from self._chunks


def _events(chunks):
    stream = BedrockChatCompletionStream.__new__(BedrockChatCompletionStream)
    stream._http_response = _FakeResponse(chunks)
    return list(stream.parse_chunks())


_TEXT_FRAMES = [
    _frame("messageStart", {"role": "assistant"}),
    _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hi"}}),
    _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": " there"}}),
    _frame("contentBlockStop", {"contentBlockIndex": 0}),
    _frame("messageStop", {"stopReason": "end_turn"}),
    _frame("metadata", {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}),
]

_TEXT_EVENTS = [
    RawBlockStart(index=0, block_type="text"),
    RawTextDelta(index=0, text="Hi"),
    RawTextDelta(index=0, text=" there"),
    RawBlockStop(index=0),
    RawFinish(reason="end_turn"),
    RawUsage(usage=Usage(input_tokens=2, output_tokens=3, total_tokens=5)),
]


def test_a_text_completion_decodes_to_the_expected_event_list():
    assert _events([b"".join(_TEXT_FRAMES)]) == _TEXT_EVENTS


def test_a_block_start_is_synthesised_because_converse_omits_it_for_text():
    # The first event is a RawBlockStart even though no contentBlockStart frame
    # was sent — the accumulator downstream needs one before any delta.
    assert _events([b"".join(_TEXT_FRAMES)])[0] == RawBlockStart(index=0, block_type="text")


def test_a_frame_split_across_two_reads_still_decodes():
    joined = b"".join(_TEXT_FRAMES)
    midpoint = len(joined) // 2
    assert _events([joined[:midpoint], joined[midpoint:]]) == _TEXT_EVENTS


def test_two_frames_arriving_in_one_read_both_decode():
    one_read = _TEXT_FRAMES[0] + _TEXT_FRAMES[1]
    rest = b"".join(_TEXT_FRAMES[2:])
    assert _events([one_read, rest]) == _TEXT_EVENTS


def test_byte_at_a_time_delivery_still_decodes():
    joined = b"".join(_TEXT_FRAMES)
    assert _events([joined[i:i + 1] for i in range(len(joined))]) == _TEXT_EVENTS


def test_a_tool_use_block_carries_its_id_name_and_argument_deltas():
    frames = [
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockStart", {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "t1", "name": "multiply"}},
        }),
        _frame("contentBlockDelta", {
            "contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"a":8,'}},
        }),
        _frame("contentBlockDelta", {
            "contentBlockIndex": 0, "delta": {"toolUse": {"input": '"b":9}'}},
        }),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "tool_use"}),
    ]
    assert _events([b"".join(frames)]) == [
        RawBlockStart(index=0, block_type="tool_call", tool_id="t1", tool_name="multiply"),
        RawToolArgumentsDelta(index=0, arguments_delta='{"a":8,'),
        RawToolArgumentsDelta(index=0, arguments_delta='"b":9}'),
        RawBlockStop(index=0),
        RawFinish(reason="tool_use"),
    ]


def test_a_reasoning_block_maps_text_then_signature():
    frames = [
        _frame("contentBlockDelta", {
            "contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}},
        }),
        _frame("contentBlockDelta", {
            "contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}},
        }),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
    ]
    assert _events([b"".join(frames)]) == [
        RawBlockStart(index=0, block_type="thinking"),
        RawThinkingDelta(index=0, text="hmm"),
        RawThinkingDelta(index=0, text="", signature="sig"),
        RawBlockStop(index=0),
    ]


def test_a_corrupt_message_crc_raises():
    good = _TEXT_FRAMES[1]
    corrupt = good[:-1] + bytes([good[-1] ^ 0xFF])
    with pytest.raises(StreamError, match="CRC"):
        _events([corrupt])


def test_a_truncated_final_frame_yields_only_the_complete_frames():
    # messageStart + a full text delta + a half-written stop frame: the
    # incomplete frame is held in the buffer and never emitted.
    joined = _TEXT_FRAMES[0] + _TEXT_FRAMES[1] + _TEXT_FRAMES[3][:6]
    assert _events([joined]) == [
        RawBlockStart(index=0, block_type="text"),
        RawTextDelta(index=0, text="Hi"),
    ]


def test_an_exception_frame_raises_stream_error():
    frame = _frame(
        "internalServerException", {"message": "kaboom"}, message_type="exception",
    )
    with pytest.raises(StreamError, match="kaboom"):
        _events([frame])


def test_a_protocol_error_frame_also_raises_stream_error():
    frame = _frame("", {"message": "bad frame"}, message_type="error")
    with pytest.raises(StreamError, match="bad frame"):
        _events([frame])


def test_streamed_usage_carries_cache_tokens():
    frames = [
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", {"usage": {
            "inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
            "cacheReadInputTokens": 8, "cacheWriteInputTokens": 2,
        }}),
    ]
    assert _events([b"".join(frames)]) == [
        RawFinish(reason="end_turn"),
        RawUsage(usage=Usage(
            input_tokens=10, output_tokens=5, total_tokens=15,
            cached_input_tokens=8, cache_write_tokens=2,
        )),
    ]
