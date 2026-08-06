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
    UserMessage,
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


def test_foreign_native_call_and_result_are_dropped_and_users_coalesce(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    projected = transport._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(
                content=[ShellToolCall(id="call_2", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1")],
            ),
            ShellToolMessage(tool_call_id="call_2", results=[]),
            UserMessage(content="continue"),
        ],
        _request(),
    )
    # The empty assistant turn is omitted and its result dropped; the two
    # surviving user messages merge into one (Converse alternation).
    assert projected == [
        {"role": "user", "content": [{"text": "hi"}, {"text": "continue"}]},
    ]


def test_surviving_blocks_keep_the_assistant_turn(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    projected = transport._project_messages(
        [
            AssistantMessage(
                content=[
                    TextBlock(text="running"),
                    ShellToolCall(id="call_2", name="shell", arguments={}, item_id="sh_1"),
                ],
            ),
        ],
        _request(),
    )
    assert projected == [{"role": "assistant", "content": [{"text": "running"}]}]
