"""Chat completions ships no native tools, but gets the full mechanism: the
compatibility gate (both OTHER families' tools are rejected before HTTP) and
the foreign-drop policy for native calls in history."""

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports.anthropic.native_tools import BashTool
from luca.client.transports.openai_responses.native_tools import (
    ApplyPatchTool,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_FOREIGN_SHELL_CALL = ShellToolCall(
    id="call_2",
    name="shell",
    arguments={"commands": ["ls"]},
    item_id="sh_1",
)


def test_other_families_native_tools_are_rejected_before_http(openai_transport_factory):
    transport = openai_transport_factory()
    with pytest.raises(BadRequestError, match="targets another transport"):
        transport._project_tools([BashTool()])
    with pytest.raises(BadRequestError, match="targets another transport"):
        # apply_patch exists only on the RESPONSES wire — the two OpenAI
        # protocols are siblings, not a family.
        transport._project_tools([ApplyPatchTool()])


def test_foreign_native_call_and_result_are_dropped(openai_transport_factory):
    transport = openai_transport_factory()
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(
                content=[
                    TextBlock(text="running"),
                    ToolCall(id="call_1", name="add", arguments={"a": 1}),
                    _FOREIGN_SHELL_CALL,
                ],
            ),
            ToolMessage(tool_call_id="call_1", content="2"),
            ShellToolMessage(tool_call_id="call_2", results=[]),
        ],
    )
    assert projected == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "running",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    ]


def test_fully_dropped_assistant_turn_is_omitted_entirely(openai_transport_factory):
    transport = openai_transport_factory()
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[_FOREIGN_SHELL_CALL]),
            ShellToolMessage(tool_call_id="call_2", results=[]),
            UserMessage(content="continue"),
        ],
    )
    assert projected == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]
