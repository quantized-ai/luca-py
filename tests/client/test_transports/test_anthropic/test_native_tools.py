"""Anthropic native tools: only the declaration differs. Calls round-trip as
base ToolCalls, results as plain tool_result blocks; foreign (OpenAI-family)
native calls are dropped at projection; native streams ride today's parser
unchanged (driven by the live captures from 2026-08-06)."""

from pathlib import Path

from luca.client.transports.anthropic.native_tools import BashTool, TextEditorTool
from luca.client.transports.openai_responses.native_tools import ShellToolCall, ShellToolMessage
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ThinkingBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

_CAPTURES = Path(__file__).parent / "captures"

ADD = Tool(name="add", description="Add.", parameters={"type": "object", "properties": {}})


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet-4-5",
        provider="anthropic",
        messages=[UserMessage(content="hi")],
        **kwargs,
    )


# --- declarations ----------------------------------------------------------


def test_native_declarations_ride_alongside_standard_tools(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD, TextEditorTool(max_characters=10_000), BashTool()]),
    )
    assert payload["tools"] == [
        {
            "name": "add",
            "description": "Add.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10_000},
        {"type": "bash_20250124", "name": "bash"},
    ]


def test_text_editor_without_max_characters_omits_the_key(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    payload = transport._build_chat_completion_payload(_request(tools=[TextEditorTool()]))
    assert payload["tools"] == [{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}]


# --- calls + results are the standard shapes -------------------------------


def test_native_call_round_trips_as_base_toolcall(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    data = {
        "id": "msg_1",
        "model": "claude-sonnet-4-5",
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "echo hi luca"}},
        ],
    }
    message = transport._parse_assistant_message(data, _request())
    assert message.content == [ToolCall(id="toolu_1", name="bash", arguments={"command": "echo hi luca"})]
    assert type(message.content[0]) is ToolCall

    projected = transport._project_messages(
        [message, ToolMessage(tool_call_id="toolu_1", content="hi luca\n")],
        _request(tools=[BashTool()]),
    )
    assert projected == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "echo hi luca"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "hi luca\n", "is_error": False},
            ],
        },
    ]


# --- foreign-drop policy ---------------------------------------------------

_FOREIGN_SHELL_CALL = ShellToolCall(
    id="call_2",
    name="shell",
    arguments={"commands": ["ls"], "max_output_length": 10240},
    item_id="sh_1",
)


def test_openai_native_call_and_result_are_dropped(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[TextBlock(text="running"), _FOREIGN_SHELL_CALL]),
            ShellToolMessage(tool_call_id="call_2", results=[]),
        ],
        _request(),
    )
    assert projected == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "running"}]},
    ]


def test_fully_dropped_assistant_turn_is_omitted_entirely(anthropic_transport_factory):
    # The typical post-provider-switch OpenAI turn: [reasoning, shell_call].
    # Both blocks are foreign here; an empty content array is a 400 on this
    # wire, so the whole message is omitted, along with the call's result.
    transport = anthropic_transport_factory()
    foreign_thinking = ThinkingBlock(text="planning", id="rs_1", signature="enc_1")
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(
                content=[foreign_thinking, _FOREIGN_SHELL_CALL],
                provider="openai",
                model="gpt-5.1",
            ),
            ShellToolMessage(tool_call_id="call_2", results=[]),
            UserMessage(content="continue"),
        ],
        _request(),
    )
    assert projected == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]


# --- tool_choice -----------------------------------------------------------


def test_tool_choice_forces_native_tools_by_name(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD, BashTool()], tool_choice={"name": "bash"}),
    )
    assert payload["tool_choice"] == {"type": "tool", "name": "bash"}


def test_tool_choice_for_an_undeclared_name_keeps_the_blind_shape(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD], tool_choice={"name": "ghost"}),
    )
    assert payload["tool_choice"] == {"type": "tool", "name": "ghost"}


# --- streaming: today's parser, unchanged ----------------------------------


def _capture_client(name: str):
    return make_sync_client(sse_response([(_CAPTURES / name).read_bytes()]))


def test_bash_stream_rides_the_existing_tool_call_events(anthropic_transport_factory):
    transport = anthropic_transport_factory(http_client=_capture_client("anthropic_bash_stream.sse"))
    with transport.completion_stream(_request(tools=[BashTool()])) as stream:
        events = list(stream)

    types_seen = [e.type for e in events]
    assert types_seen.count("tool_call_delta") > 0  # full JSON argument deltas
    start = next(e for e in events if e.type == "tool_call_start")
    end = next(e for e in events if e.type == "tool_call_end")
    assert start.name == "bash"
    assert type(end.tool_call) is ToolCall
    assert end.tool_call.arguments == {"command": "echo hi luca"}
    assert events[-1].finish_reason == "tool_use"


def test_text_editor_stream_parses_the_create_command(anthropic_transport_factory):
    transport = anthropic_transport_factory(http_client=_capture_client("anthropic_text_editor_stream.sse"))
    with transport.completion_stream(_request(tools=[TextEditorTool()])) as stream:
        events = list(stream)

    end = next(e for e in events if e.type == "tool_call_end")
    assert end.tool_call.name == "str_replace_based_edit_tool"
    assert end.tool_call.arguments == {
        "command": "create",
        "path": "/repo/hello.txt",
        "file_text": "hi luca",
    }
