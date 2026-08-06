"""Accumulator handling of transport-built native tool calls: prebuilt on
RawBlockStart, replacement on RawBlockStop, truncation close, and subclass
fields surviving event dumps."""

from typing import Literal

from luca.client.types import ChatCompletionRequest, TextBlock, ToolCall
from luca.client.types.streaming import (
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawTextDelta,
    RawToolArgumentsDelta,
    ToolCallEndEvent,
    ToolCallStartEvent,
    _ChatCompletionAccumulator,
)


class StreamNativeToolCall(ToolCall):
    type: Literal["stream_native_call"] = "stream_native_call"
    item_id: str | None = None
    status: str = "completed"


def _drive(raws):
    acc = _ChatCompletionAccumulator(
        request=ChatCompletionRequest(model="m"),
        provider="openai",
    )
    events = []
    for raw in raws:
        events.extend(acc.handle_raw(raw))
    return acc, events


def _skeleton(id: str = "call_1") -> StreamNativeToolCall:  # noqa: A002
    return StreamNativeToolCall(
        id=id,
        name="native",
        arguments={"path": "hello.txt", "diff": ""},
        item_id="item_1",
        status="in_progress",
        complete=False,
    )


def _final(id: str = "call_1") -> StreamNativeToolCall:  # noqa: A002
    return StreamNativeToolCall(
        id=id,
        name="native",
        arguments={"path": "hello.txt", "diff": "+hi\n"},
        item_id="item_1",
        status="completed",
        complete=True,
    )


def test_prebuilt_start_appends_typed_call_and_emits_start():
    acc, events = _drive(
        [RawBlockStart(index=0, block_type="tool_call", tool_id="call_1", tool_name="native", prebuilt=_skeleton())],
    )
    assert [type(e) for e in events] == [ToolCallStartEvent]
    assert (events[0].id, events[0].name) == ("call_1", "native")
    assert acc.message.content == [_skeleton()]
    assert type(acc.message.content[0]) is StreamNativeToolCall


def test_replacement_stop_swaps_in_final_call_without_json_parse():
    acc, events = _drive(
        [
            RawBlockStart(index=0, block_type="text"),
            RawTextDelta(index=0, text="Creating."),
            RawBlockStop(index=0),
            RawBlockStart(index=1, block_type="tool_call", tool_id="call_1", tool_name="native", prebuilt=_skeleton()),
            RawBlockStop(index=1, replacement=_final()),
            RawFinish(reason="completed"),
        ],
    )
    end = events[-1]
    assert type(end) is ToolCallEndEvent
    assert end.tool_call == _final()
    assert acc.message.content == [TextBlock(text="Creating."), _final()]
    assert acc.tool_calls == [_final()]


def test_stop_without_replacement_closes_prebuilt_with_empty_arguments():
    # A terminal that truncates the stream closes open blocks with plain
    # RawBlockStops — the degenerate empty-arguments close, class preserved.
    acc, events = _drive(
        [
            RawBlockStart(index=0, block_type="tool_call", tool_id="call_1", tool_name="native", prebuilt=_skeleton()),
            RawBlockStop(index=0),
            RawFinish(reason="incomplete:max_output_tokens"),
        ],
    )
    end = events[-1]
    assert type(end) is ToolCallEndEvent
    assert end.tool_call.arguments == {}
    assert end.tool_call.complete
    assert type(acc.message.content[0]) is StreamNativeToolCall


def test_standard_and_native_calls_coexist():
    acc, _ = _drive(
        [
            RawBlockStart(index=0, block_type="tool_call", tool_id="call_a", tool_name="add"),
            RawToolArgumentsDelta(index=0, arguments_delta='{"a": 1, '),
            RawToolArgumentsDelta(index=0, arguments_delta='"b": 2}'),
            RawBlockStop(index=0),
            RawBlockStart(index=1, block_type="tool_call", tool_id="call_1", tool_name="native", prebuilt=_skeleton()),
            RawBlockStop(index=1, replacement=_final()),
            RawFinish(reason="completed"),
        ],
    )
    assert acc.message.content == [
        ToolCall(id="call_a", name="add", arguments={"a": 1, "b": 2}),
        _final(),
    ]
    assert type(acc.message.content[0]) is ToolCall
    assert type(acc.message.content[1]) is StreamNativeToolCall


def test_event_and_finish_dumps_keep_subclass_fields():
    acc, events = _drive(
        [
            RawBlockStart(index=0, block_type="tool_call", tool_id="call_1", tool_name="native", prebuilt=_skeleton()),
            RawBlockStop(index=0, replacement=_final()),
            RawFinish(reason="completed"),
        ],
    )
    end_dump = events[-1].model_dump()
    assert end_dump["tool_call"]["item_id"] == "item_1"
    assert end_dump["partial"]["content"][0]["status"] == "completed"

    finish = acc.build_terminal_finish(classify_finish=lambda terminal, message: ("tool_use", None))
    assert finish.model_dump()["tool_calls"][0]["item_id"] == "item_1"
