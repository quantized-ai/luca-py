"""Native calls over the Responses stream: start + end with no public
argument deltas, driven by the LIVE SSE captures from 2026-08-06 (see
captures/), plus a synthetic truncation case."""

from pathlib import Path

from luca.client.transports.openai_responses.native_tools import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellToolCall,
)
from luca.client.types import ChatCompletionRequest, UserMessage
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

_CAPTURES = Path(__file__).parent / "captures"


def _capture_client(name: str):
    return make_sync_client(sse_response([(_CAPTURES / name).read_bytes()]))


def _request(tool) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-5.1",
        provider="openai",
        messages=[UserMessage(content="hi")],
        tools=[tool],
    )


def test_apply_patch_stream_is_start_then_end_with_full_operation(responses_transport_factory):
    transport = responses_transport_factory(
        http_client=_capture_client("openai_apply_patch_stream.sse"),
    )
    with transport.completion_stream(_request(ApplyPatchTool())) as stream:
        events = list(stream)

    assert [e.type for e in events] == ["start", "tool_call_start", "tool_call_end", "usage", "finish"]
    start = events[1]
    assert (start.id, start.name) == ("call_bNSZXJtrFzni0yQhLu4BOkFQ", "apply_patch")
    # The in-progress skeleton is already typed and carries the partial
    # operation (type + path known, diff empty).
    assert start.partial.content[0] == ApplyPatchToolCall(
        id="call_bNSZXJtrFzni0yQhLu4BOkFQ",
        name="apply_patch",
        arguments={"type": "create_file", "diff": "", "path": "hello.txt"},
        item_id="apc_078e3489cb16e5e4016a74c4c4041c819da54557a9ff83b34f",
        status="in_progress",
        complete=False,
    )
    end = events[2]
    assert end.tool_call == ApplyPatchToolCall(
        id="call_bNSZXJtrFzni0yQhLu4BOkFQ",
        name="apply_patch",
        arguments={"type": "create_file", "diff": "+hi luca\n", "path": "hello.txt"},
        item_id="apc_078e3489cb16e5e4016a74c4c4041c819da54557a9ff83b34f",
        status="completed",
    )
    finish = events[-1]
    assert finish.finish_reason == "tool_use"
    assert finish.tool_calls == [end.tool_call]


def test_shell_stream_ignores_command_deltas_and_ends_complete(responses_transport_factory):
    # The capture contains response.shell_call_command.added/.delta/.done
    # raw-text events — they must fall through unhandled.
    transport = responses_transport_factory(http_client=_capture_client("openai_shell_stream.sse"))
    with transport.completion_stream(_request(LocalShellTool())) as stream:
        events = list(stream)

    assert [e.type for e in events] == ["start", "tool_call_start", "tool_call_end", "usage", "finish"]
    end = events[2]
    assert end.tool_call == ShellToolCall(
        id="call_LMJPpE1eAP7UemzDoNA9sY0E",
        name="shell",
        arguments={"commands": ["printf 'luca' | wc -c"], "max_output_length": 10240, "timeout_ms": 8972},
        item_id="sh_062ebb77e3a2f837016a74cbf4f7b4819eb2a7ced350a1dbde",
        status="completed",
    )
    assert events[-1].finish_reason == "tool_use"


_TRUNCATED_NATIVE = [
    (
        b'data: {"type":"response.output_item.added","output_index":0,'
        b'"item":{"id":"apc_1","type":"apply_patch_call","status":"in_progress",'
        b'"call_id":"call_1","operation":{"type":"create_file","diff":"","path":"x"}}}\n\n'
    ),
    (
        b'data: {"type":"response.incomplete","response":{"id":"r","status":"incomplete",'
        b'"incomplete_details":{"reason":"max_output_tokens"},'
        b'"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    ),
]


def test_stream_truncated_mid_native_call_closes_with_empty_arguments(responses_transport_factory):
    transport = responses_transport_factory(http_client=make_sync_client(sse_response(_TRUNCATED_NATIVE)))
    with transport.completion_stream(_request(ApplyPatchTool())) as stream:
        events = list(stream)

    end = next(e for e in events if e.type == "tool_call_end")
    assert end.tool_call.arguments == {}
    assert end.tool_call.complete
    assert type(end.tool_call) is ApplyPatchToolCall
    assert events[-1].finish_reason == "length"
