"""OpenAIResponsesStreamer handlers — one raw wire event in, exactly these
public events out, plus the message state. The streamer is never entered."""

import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.openai_responses.native_tools import ApplyPatchToolCall
from luca.client.transports.openai_responses.streamer import SyncOpenAIResponsesStreamer
from luca.client.types import (
    ChatCompletionRequest,
    StartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserMessage,
)

REQUEST = ChatCompletionRequest(
    model="gpt-5.1",
    provider="openai",
    messages=[UserMessage(content="hi")],
)

_NATIVE_ADDED = {
    "type": "response.output_item.added",
    "output_index": 0,
    "item": {
        "id": "apc_1",
        "type": "apply_patch_call",
        "status": "in_progress",
        "call_id": "call_1",
        "operation": {"type": "create_file", "diff": "", "path": "x.txt"},
    },
}

_NATIVE_DONE = {
    "type": "response.output_item.done",
    "output_index": 0,
    "item": {
        "id": "apc_1",
        "type": "apply_patch_call",
        "status": "completed",
        "call_id": "call_1",
        "operation": {"type": "create_file", "diff": "+hi\n", "path": "x.txt"},
    },
}


def _streamer() -> SyncOpenAIResponsesStreamer:
    return SyncOpenAIResponsesStreamer(REQUEST)


def test_a_native_item_opens_with_the_typed_in_progress_skeleton():
    s = _streamer()

    events = s.handle(_NATIVE_ADDED)

    assert events == [StartEvent(), ToolCallStartEvent(index=0, id="call_1", name="apply_patch")]
    assert s.message.content == [
        ApplyPatchToolCall(
            id="call_1",
            name="apply_patch",
            arguments={"type": "create_file", "diff": "", "path": "x.txt"},
            item_id="apc_1",
            status="in_progress",
            complete=False,
        )
    ]


def test_the_complete_native_item_replaces_the_skeleton_whole():
    s = _streamer()
    s.handle(_NATIVE_ADDED)

    events = s.handle(_NATIVE_DONE)

    done = ApplyPatchToolCall(
        id="call_1",
        name="apply_patch",
        arguments={"type": "create_file", "diff": "+hi\n", "path": "x.txt"},
        item_id="apc_1",
        status="completed",
        complete=True,
    )
    assert events == [ToolCallEndEvent(index=0, tool_call=done)]
    assert s.message.content == [done]


def test_a_duplicate_item_done_is_ignored():
    s = _streamer()
    s.handle(_NATIVE_ADDED)
    s.handle(_NATIVE_DONE)

    assert s.handle(_NATIVE_DONE) == []


def test_a_terminal_mid_native_call_closes_it_empty_and_complete():
    s = _streamer()
    s.handle(_NATIVE_ADDED)

    events = s.handle(
        {
            "type": "response.incomplete",
            "response": {
                "id": "r",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }
    )

    assert [e.type for e in events] == ["tool_call_end", "usage", "finish"]
    end = events[0]
    assert type(end.tool_call) is ApplyPatchToolCall
    assert end.tool_call.arguments == {}
    assert end.tool_call.complete is True
    assert events[-1].finish_reason == "length"
    assert events[-1].provider_finish_reason == "incomplete:max_output_tokens"


def test_reasoning_summary_parts_join_and_the_payload_lands_at_done():
    s = _streamer()

    opened = s.handle(
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "rs_1", "type": "reasoning"}}
    )
    first_part = s.handle({"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0})
    first_delta = s.handle(
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "First."}
    )
    second_part = s.handle({"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 1})
    second_delta = s.handle(
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 1, "delta": "Second."}
    )
    done = s.handle(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "enc-1"},
        }
    )

    assert opened == [StartEvent(), ThinkingStartEvent(index=0)]
    assert first_part == []
    assert first_delta == [ThinkingDeltaEvent(index=0, delta="First.")]
    assert second_part == [ThinkingDeltaEvent(index=0, delta="\n\n")]
    assert second_delta == [ThinkingDeltaEvent(index=0, delta="Second.")]
    assert done == [
        ThinkingDeltaEvent(index=0, delta=""),
        ThinkingEndEvent(index=0, content="First.\n\nSecond."),
    ]
    assert s.message.content == [ThinkingBlock(text="First.\n\nSecond.", id="rs_1", signature="enc-1")]


def test_unknown_event_types_are_ignored():
    s = _streamer()

    assert s.handle({"type": "response.web_search_call.in_progress", "output_index": 0}) == [StartEvent()]
    assert s.handle({"type": "response.some_future_event"}) == []


def test_a_function_call_item_without_a_call_id_is_a_stream_error():
    s = _streamer()

    with pytest.raises(StreamError, match="without a call_id or name"):
        s.handle(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "name": "add"},
            }
        )


def test_an_error_event_raises_with_the_wire_message():
    s = _streamer()

    with pytest.raises(StreamError, match="boom"):
        s.handle({"type": "error", "message": "boom"})


def test_a_terminal_missing_its_status_falls_back_to_the_event_name():
    s = _streamer()

    events = s.handle({"type": "response.failed", "response": {"id": "r"}})

    assert [e.type for e in events] == ["start", "finish"]
    assert events[-1].provider_finish_reason == "failed"
    assert events[-1].finish_reason == "error"
