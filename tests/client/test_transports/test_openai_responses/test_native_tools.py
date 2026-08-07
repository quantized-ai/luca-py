"""Native tools on the Responses wire: declarations, parse, replay, results,
tool_choice forcing, and the foreign-drop policy. Wire shapes mirror the live
captures from 2026-08-06.

Every projection test runs twice — once with the typed native class, once with
the generic `extras` form of the same call/result — against ONE expected
payload. The two forms are interchangeable or these tests fail."""

from typing import ClassVar, Literal

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports.openai_responses.native_tools import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellCommandResult,
    ShellExitOutcome,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import (
    AssistantMessage,
    BaseTool,
    ChatCompletionRequest,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProjector,
    UserMessage,
)

ADD = Tool(name="add", description="Add.", parameters={"type": "object", "properties": {}})

APPLY_PATCH_ITEM = {
    "id": "apc_1",
    "type": "apply_patch_call",
    "status": "completed",
    "call_id": "call_1",
    "operation": {"type": "create_file", "diff": "+hi luca\n", "path": "hello.txt"},
}

SHELL_ITEM = {
    "id": "sh_1",
    "type": "shell_call",
    "status": "completed",
    "call_id": "call_2",
    "action": {"commands": ["wc -c hello.txt"], "timeout_ms": 10000, "max_output_length": 10240},
    "environment": None,
}

APPLY_CALL = ApplyPatchToolCall(
    id="call_1",
    name="apply_patch",
    arguments={"type": "create_file", "diff": "+hi luca\n", "path": "hello.txt"},
    item_id="apc_1",
    status="completed",
)

SHELL_CALL = ShellToolCall(
    id="call_2",
    name="shell",
    arguments={"commands": ["wc -c hello.txt"], "timeout_ms": 10000, "max_output_length": 10240},
    item_id="sh_1",
    status="completed",
)

# The generic twins: the same calls/results expressed with canonical types
# only. Written out literally rather than derived with as_generic(), so the
# shape a caller has to produce is what these tests actually exercise.
APPLY_CALL_GENERIC = ToolCall(
    id="call_1",
    name="apply_patch",
    arguments={"type": "create_file", "diff": "+hi luca\n", "path": "hello.txt"},
    extras={"custom_type": "apply_patch_call", "item_id": "apc_1", "status": "completed"},
)

SHELL_CALL_GENERIC = ToolCall(
    id="call_2",
    name="shell",
    arguments={"commands": ["wc -c hello.txt"], "timeout_ms": 10000, "max_output_length": 10240},
    extras={"custom_type": "shell_call", "item_id": "sh_1", "status": "completed"},
)

SHELL_RESULT = ShellToolMessage(
    tool_call_id="call_2",
    results=[ShellCommandResult(stdout="5\n", stderr="", outcome=ShellExitOutcome(exit_code=0))],
)

SHELL_RESULT_GENERIC = ToolMessage(
    tool_call_id="call_2",
    content="",
    extras={
        "custom_type": "shell_call_output",
        "results": [{"stdout": "5\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
    },
)

APPLY_CALLS = pytest.mark.parametrize("apply_call", [APPLY_CALL, APPLY_CALL_GENERIC], ids=["typed", "generic"])
SHELL_CALLS = pytest.mark.parametrize("shell_call", [SHELL_CALL, SHELL_CALL_GENERIC], ids=["typed", "generic"])
SHELL_RESULTS = pytest.mark.parametrize(
    "shell_result",
    [SHELL_RESULT, SHELL_RESULT_GENERIC],
    ids=["typed", "generic"],
)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-5.1",
        provider="openai",
        messages=[UserMessage(content="hi")],
        **kwargs,
    )


# --- declarations ----------------------------------------------------------


def test_native_declarations_ride_alongside_standard_tools(responses_transport_factory):
    transport = responses_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD, ApplyPatchTool(), LocalShellTool()]),
    )
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "add",
            "description": "Add.",
            "parameters": {"type": "object", "properties": {}},
        },
        {"type": "apply_patch"},
        {"type": "shell", "environment": {"type": "local"}},
    ]


# --- parse -----------------------------------------------------------------


def test_parse_builds_typed_native_calls(responses_transport_factory):
    transport = responses_transport_factory()
    data = {"id": "resp_1", "model": "gpt-5.1", "status": "completed", "output": [APPLY_PATCH_ITEM, SHELL_ITEM]}
    message = transport._parse_assistant_message(data, _request())
    assert message.content == [APPLY_CALL, SHELL_CALL]
    assert type(message.content[0]) is ApplyPatchToolCall
    assert type(message.content[1]) is ShellToolCall
    assert message.content[0].operation == APPLY_PATCH_ITEM["operation"]
    assert message.content[1].action == SHELL_ITEM["action"]


def test_unknown_item_type_is_still_ignored(responses_transport_factory):
    transport = responses_transport_factory()
    data = {"id": "resp_1", "output": [{"type": "web_search_call", "id": "ws_1"}]}
    message = transport._parse_assistant_message(data, _request())
    assert message.content == []


# --- replay + results ------------------------------------------------------


@APPLY_CALLS
def test_apply_patch_replays_verbatim_and_result_carries_status(responses_transport_factory, apply_call):
    transport = responses_transport_factory()
    items = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[apply_call]),
            ToolMessage(tool_call_id="call_1", content="Created hello.txt"),
        ],
        _request(),
    )
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {
            "type": "apply_patch_call",
            "id": "apc_1",
            "call_id": "call_1",
            "status": "completed",
            "operation": {"type": "create_file", "diff": "+hi luca\n", "path": "hello.txt"},
        },
        {
            "type": "apply_patch_call_output",
            "call_id": "call_1",
            "status": "completed",
            "output": "Created hello.txt",
        },
    ]


@APPLY_CALLS
def test_apply_patch_error_result_projects_failed(responses_transport_factory, apply_call):
    transport = responses_transport_factory()
    items = transport._project_messages(
        [
            AssistantMessage(content=[apply_call]),
            ToolMessage(tool_call_id="call_1", content="no such file", is_error=True),
        ],
        _request(),
    )
    assert items[-1] == {
        "type": "apply_patch_call_output",
        "call_id": "call_1",
        "status": "failed",
        "output": "no such file",
    }


@SHELL_CALLS
@SHELL_RESULTS
def test_shell_result_is_structured_and_echoes_max_output_length(
    responses_transport_factory,
    shell_call,
    shell_result,
):
    transport = responses_transport_factory()
    items = transport._project_messages(
        [AssistantMessage(content=[shell_call]), shell_result],
        _request(),
    )
    assert items == [
        {
            "type": "shell_call",
            "id": "sh_1",
            "call_id": "call_2",
            "status": "completed",
            "action": {"commands": ["wc -c hello.txt"], "timeout_ms": 10000, "max_output_length": 10240},
        },
        {
            "type": "shell_call_output",
            "call_id": "call_2",
            "output": [{"stdout": "5\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
            "max_output_length": 10240,
        },
    ]


@SHELL_CALLS
@SHELL_RESULTS
def test_shell_result_without_call_limit_omits_the_echo(
    responses_transport_factory,
    shell_call,
    shell_result,
):
    transport = responses_transport_factory()
    call = shell_call.model_copy(update={"arguments": {"commands": ["ls"]}})
    items = transport._project_messages([AssistantMessage(content=[call]), shell_result], _request())
    assert "max_output_length" not in items[-1]


@SHELL_CALLS
def test_plain_tool_message_for_a_shell_call_is_refused(responses_transport_factory, shell_call):
    transport = responses_transport_factory()
    with pytest.raises(BadRequestError, match="ShellToolMessage"):
        transport._project_messages(
            [
                AssistantMessage(content=[shell_call]),
                ToolMessage(tool_call_id="call_2", content="5"),
            ],
            _request(),
        )


def test_result_without_lineage_falls_back_to_standard_shape(responses_transport_factory):
    transport = responses_transport_factory()
    items = transport._project_messages(
        [ToolMessage(tool_call_id="call_9", content="ok")],
        _request(),
    )
    assert items == [{"type": "function_call_output", "call_id": "call_9", "output": "ok"}]


# --- the foreign-drop policy -----------------------------------------------


class _ForeignProjector(ToolProjector):
    pass


class _ForeignToolCall(ToolCall):
    type: Literal["foreign_test_call"] = "foreign_test_call"


_ForeignToolCall.projector_class = _ForeignProjector


class _ForeignTool(BaseTool):
    name: ClassVar[str] = "foreign"

    def get_projector(self) -> ToolProjector:
        return _ForeignProjector()


@pytest.mark.parametrize(
    "foreign_call",
    [
        _ForeignToolCall(id="call_f", name="x"),
        ToolCall(id="call_f", name="x", extras={"custom_type": "foreign_test_call"}),
    ],
    ids=["typed", "generic"],
)
def test_foreign_native_call_and_its_result_are_dropped(responses_transport_factory, foreign_call):
    transport = responses_transport_factory()
    items = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[TextBlock(text="doing it"), foreign_call]),
            ToolMessage(tool_call_id="call_f", content="result"),
        ],
        _request(),
    )
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "doing it"}],
        },
    ]


def test_incompatible_tool_declaration_fails_before_any_http(responses_transport_factory):
    transport = responses_transport_factory()
    with pytest.raises(BadRequestError, match="targets another transport"):
        transport._project_tools([_ForeignTool()])


def test_foreign_registry_hit_is_ignored_at_parse(responses_transport_factory):
    transport = responses_transport_factory()
    data = {"id": "resp_1", "output": [{"type": "foreign_test_call", "id": "f_1", "call_id": "call_f"}]}
    message = transport._parse_assistant_message(data, _request())
    assert message.content == []


# --- tool_choice forcing ---------------------------------------------------


def test_tool_choice_forces_native_tools_by_type(responses_transport_factory):
    transport = responses_transport_factory()
    tools = [ADD, ApplyPatchTool(), LocalShellTool()]
    payload = transport._build_chat_completion_payload(_request(tools=tools, tool_choice={"name": "apply_patch"}))
    assert payload["tool_choice"] == {"type": "apply_patch"}
    payload = transport._build_chat_completion_payload(_request(tools=tools, tool_choice={"name": "shell"}))
    assert payload["tool_choice"] == {"type": "shell"}


def test_tool_choice_for_a_standard_tool_is_unchanged(responses_transport_factory):
    transport = responses_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD, ApplyPatchTool()], tool_choice={"name": "add"}),
    )
    assert payload["tool_choice"] == {"type": "function", "name": "add"}


def test_tool_choice_for_an_undeclared_name_keeps_the_blind_shape(responses_transport_factory):
    transport = responses_transport_factory()
    payload = transport._build_chat_completion_payload(
        _request(tools=[ADD], tool_choice={"name": "ghost"}),
    )
    assert payload["tool_choice"] == {"type": "function", "name": "ghost"}


# --- the two forms ---------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "generic"),
    [
        (APPLY_CALL, APPLY_CALL_GENERIC),
        (SHELL_CALL, SHELL_CALL_GENERIC),
        (SHELL_RESULT, SHELL_RESULT_GENERIC),
    ],
    ids=["apply_patch_call", "shell_call", "shell_call_output"],
)
def test_each_native_converts_into_its_generic_twin_and_back(typed, generic):
    assert typed.as_generic() == generic
    assert generic.as_native() == typed


def test_an_unregistered_custom_type_is_refused_at_projection(responses_transport_factory):
    transport = responses_transport_factory()
    call = ToolCall(id="call_1", name="ghost", extras={"custom_type": "gpt_9_telepathy_call"})
    with pytest.raises(BadRequestError, match="Unknown native tool-call type 'gpt_9_telepathy_call'"):
        transport._project_messages([AssistantMessage(content=[call])], _request())


# --- serialization ---------------------------------------------------------


def test_native_conversation_survives_a_json_round_trip():
    shell_result = ShellToolMessage(
        tool_call_id="call_2",
        results=[ShellCommandResult(stdout="5\n", stderr="", outcome=ShellExitOutcome(exit_code=0))],
    )
    assistant = AssistantMessage(content=[APPLY_CALL, SHELL_CALL])

    restored_assistant = AssistantMessage.model_validate(assistant.model_dump())
    restored_result = ToolMessage.model_validate(shell_result.model_dump())

    assert restored_assistant == assistant
    assert type(restored_assistant.content[0]) is ApplyPatchToolCall
    assert type(restored_assistant.content[1]) is ShellToolCall
    assert restored_result == shell_result
    assert type(restored_result) is ShellToolMessage
