"""AnthropicTransport.completion_stream() — sync."""

import pytest

from luca.client.types import (
    ChatCompletionRequest,
    Usage,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response
from tests.client._helpers.stream_iteration import collect_events_with_snapshots


def _sse(event_type: str, data: str) -> bytes:
    return f"event: {event_type}\ndata: {data}\n\n".encode()


def test_anthropic_streaming_text_block(anthropic_transport_factory):
    chunks = [
        _sse("message_start", '{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}'),
        _sse("content_block_start", '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'),
        _sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}'),
        _sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}'),
        _sse("content_block_stop", '{"type":"content_block_stop","index":0}'),
        _sse("message_delta", '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}'),
        _sse("message_stop", '{"type":"message_stop"}'),
    ]
    client = make_sync_client(sse_response(chunks))
    transport = anthropic_transport_factory(http_client=client)

    req = ChatCompletionRequest(
        model="claude-test", provider="anthropic",
        messages=[UserMessage(content="hi")],
    )
    with transport.completion_stream(req) as s:
        events = collect_events_with_snapshots(s)

    types = [e.type for e in events]
    assert types[0] == "start"
    assert "text_start" in types
    assert types.count("text_delta") == 2
    assert "text_end" in types
    assert "usage" in types
    assert types[-1] == "finish"
    assert events[-1].finish_reason == "stop"
    assert events[-1].provider_finish_reason == "end_turn"
    assert events[-1].usage == Usage(input_tokens=5, output_tokens=2, total_tokens=7)


def test_anthropic_streaming_tool_use(anthropic_transport_factory):
    chunks = [
        _sse("message_start", '{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}'),
        _sse("content_block_start", '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}'),
        _sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\""}}'),
        _sse("content_block_delta", '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\\"NYC\\"}"}}'),
        _sse("content_block_stop", '{"type":"content_block_stop","index":0}'),
        _sse("message_delta", '{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}'),
        _sse("message_stop", '{"type":"message_stop"}'),
    ]
    client = make_sync_client(sse_response(chunks))
    transport = anthropic_transport_factory(http_client=client)

    req = ChatCompletionRequest(
        model="claude-test", provider="anthropic",
        messages=[UserMessage(content="weather?")],
    )
    with transport.completion_stream(req) as s:
        events = collect_events_with_snapshots(s)

    finish = events[-1]
    assert finish.type == "finish"
    assert finish.finish_reason == "tool_use"
    assert finish.provider_finish_reason == "tool_use"
    assert len(finish.tool_calls) == 1
    assert finish.tool_calls[0].name == "get_weather"
    assert finish.tool_calls[0].arguments == {"city": "NYC"}
