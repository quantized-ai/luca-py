"""Stream paths where the provider does not play by the happy-path rules.

The counterpart to `test_openai/test_stream_cancellation.py`. True
user-initiated cancellation needs a real socket (`httpx.MockTransport`
pre-buffers the body), so what is exercised here is the structural side: a
terminal arriving mid-block, premature end, malformed frames.
"""

import pytest

from luca.client.types import ChatCompletionRequest, TextBlock, ThinkingBlock, UserMessage
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

REQUEST = ChatCompletionRequest(
    model="gpt-5.4",
    provider="openai",
    messages=[UserMessage(content="hi")],
)


def _data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


OPEN_TEXT = [
    _data(
        '{"type":"response.output_item.added","output_index":0,'
        '"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}'
    ),
    _data(
        '{"type":"response.content_part.added","output_index":0,"content_index":0,'
        '"part":{"type":"output_text","text":""}}'
    ),
    _data('{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Once"}'),
]

OPEN_TOOL_CALL = [
    _data(
        '{"type":"response.output_item.added","output_index":0,'
        '"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"rm","arguments":""}}'
    ),
    _data('{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\"path\\":\\"/t"}'),
]

OPEN_REASONING = [
    _data(
        '{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}'
    ),
    _data('{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"planning"}'),
]

TRUNCATED = _data(
    '{"type":"response.incomplete","response":{"id":"r","status":"incomplete",'
    '"incomplete_details":{"reason":"max_output_tokens"},'
    '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
)


def test_a_text_block_open_at_the_terminal_still_closes(responses_transport_factory):
    transport = responses_transport_factory(http_client=make_sync_client(sse_response([*OPEN_TEXT, TRUNCATED])))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert [e.type for e in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "usage",
        "finish",
    ]
    assert events[-1].message.content == [TextBlock(text="Once")]


def test_a_reasoning_block_open_at_the_terminal_still_closes(responses_transport_factory):
    # Its `encrypted_content` never arrived, so the block stays unsigned and
    # the next turn drops it from the wire, which is correct.
    transport = responses_transport_factory(http_client=make_sync_client(sse_response([*OPEN_REASONING, TRUNCATED])))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert [e.type for e in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "usage",
        "finish",
    ]
    assert events[-1].message.content == [ThinkingBlock(text="planning", id="rs_1")]


def test_a_tool_call_truncated_mid_arguments_is_an_error_not_a_finish(responses_transport_factory):
    # The whole point: an unclosed tool call used to reach the caller as
    # `arguments={}, complete=False` beside a `tool_use` finish, and the agent
    # dispatches every tool call without checking `complete`.
    transport = responses_transport_factory(http_client=make_sync_client(sse_response([*OPEN_TOOL_CALL, TRUNCATED])))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].type == "error"
    assert "malformed JSON" in str(events[-1].error)


def test_a_stream_that_ends_without_a_terminal_emits_an_error_event(responses_transport_factory):
    transport = responses_transport_factory(
        http_client=make_sync_client(sse_response([*OPEN_TEXT, _data("[DONE]")])),
    )

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].type == "error"
    assert "RawFinish" in str(events[-1].error)
    assert events[-1].partial_message.content == [TextBlock(text="Once")]


def test_malformed_json_is_a_stream_error(responses_transport_factory):
    transport = responses_transport_factory(
        http_client=make_sync_client(sse_response([*OPEN_TEXT, _data("{not json")])),
    )

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].type == "error"
    assert "non-JSON data" in str(events[-1].error)


def test_an_error_frame_becomes_a_terminal_error_event(responses_transport_factory):
    transport = responses_transport_factory(
        http_client=make_sync_client(
            sse_response([*OPEN_TEXT, _data('{"type":"error","message":"boom"}')]),
        ),
    )

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].type == "error"
    assert "boom" in str(events[-1].error)


def test_a_terminal_missing_its_status_falls_back_to_the_event_name(responses_transport_factory):
    # Never "completed": reporting a failed response as a successful stop is
    # the one answer that must not happen here.
    transport = responses_transport_factory(
        http_client=make_sync_client(
            sse_response([*OPEN_TEXT, _data('{"type":"response.failed","response":{"id":"r"}}')]),
        ),
    )

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert events[-1].provider_finish_reason == "failed"
    assert events[-1].finish_reason == "error"


def test_a_duplicate_terminal_is_ignored(responses_transport_factory):
    completed = _data(
        '{"type":"response.completed","response":{"id":"r","status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
    )
    transport = responses_transport_factory(
        http_client=make_sync_client(sse_response([*OPEN_TEXT, completed, completed])),
    )

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    assert [e.type for e in events].count("finish") == 1


@pytest.mark.parametrize(
    "chunks",
    [OPEN_TEXT, OPEN_TOOL_CALL, OPEN_REASONING],
    ids=["text", "tool_call", "reasoning"],
)
def test_no_resource_warning_when_a_stream_is_abandoned_open(chunks, responses_transport_factory):
    # The suite runs with -W error::ResourceWarning, so a leaked response here
    # is a build break rather than a nuisance.
    transport = responses_transport_factory(http_client=make_sync_client(sse_response(list(chunks))))

    with transport.completion_stream(REQUEST) as s:
        s.cancel()

    assert s.cancelled is True
