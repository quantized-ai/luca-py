"""Bedrock ships no native tools but gets the full mechanism: compatibility
gate, foreign-drop, and empty-turn omission (repaired by role coalescing)."""

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports.openai_responses.native_tools import (
    ApplyPatchTool,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_FOREIGN_SHELL_CALL = ShellToolCall(id="call_2", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1")

_FOREIGN_SHELL_RESULT = ShellToolMessage(tool_call_id="call_2", results=[])

# The same foreign exchange in the generic form — dropped identically.
_FOREIGN_SHELL_CALL_GENERIC = ToolCall(
    id="call_2",
    name="shell",
    arguments={"commands": ["ls"]},
    extras={"custom_type": "shell_call", "item_id": "sh_1", "status": "completed"},
)

_FOREIGN_SHELL_RESULT_GENERIC = ToolMessage(
    tool_call_id="call_2",
    content="",
    extras={"custom_type": "shell_call_output", "results": []},
)

FOREIGN_EXCHANGES = pytest.mark.parametrize(
    ("foreign_call", "foreign_result"),
    [
        (_FOREIGN_SHELL_CALL, _FOREIGN_SHELL_RESULT),
        (_FOREIGN_SHELL_CALL_GENERIC, _FOREIGN_SHELL_RESULT_GENERIC),
    ],
    ids=["typed", "generic"],
)

FOREIGN_CALLS = pytest.mark.parametrize(
    "foreign_call",
    [_FOREIGN_SHELL_CALL, _FOREIGN_SHELL_CALL_GENERIC],
    ids=["typed", "generic"],
)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="us.amazon.nova-pro-v1:0",
        provider="bedrock",
        messages=[UserMessage(content="hi")],
        **kwargs,
    )


def test_native_tool_from_another_family_is_rejected_before_http(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    with pytest.raises(BadRequestError, match="targets another transport"):
        transport._project_tools([ApplyPatchTool()])


@FOREIGN_EXCHANGES
def test_foreign_native_call_and_result_are_dropped_and_users_coalesce(
    bedrock_transport_factory,
    foreign_call,
    foreign_result,
):
    transport = bedrock_transport_factory()
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(content=[foreign_call]),
            foreign_result,
            UserMessage(content="continue"),
        ],
        _request(),
    )
    # The empty assistant turn is omitted and its result dropped; the two
    # surviving user messages merge into one (Converse alternation).
    assert projected == [
        {"role": "user", "content": [{"text": "hi"}, {"text": "continue"}]},
    ]


@FOREIGN_CALLS
def test_surviving_blocks_keep_the_assistant_turn(bedrock_transport_factory, foreign_call):
    transport = bedrock_transport_factory()
    projected = transport._project_messages(
        [AssistantMessage(content=[TextBlock(text="running"), foreign_call])],
        _request(),
    )
    assert projected == [{"role": "assistant", "content": [{"text": "running"}]}]
