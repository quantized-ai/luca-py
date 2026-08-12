"""ChatCompletionResponse holds a list of AssistantMessages and forwards nothing."""

import pytest
from pydantic import ValidationError

from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    ToolCall,
    Usage,
)


def test_messages_are_held_in_order():
    first = AssistantMessage(content=[TextBlock(text="one")])
    second = AssistantMessage(content=[TextBlock(text="two")], finish_reason="stop")
    response = ChatCompletionResponse(messages=[first, second])
    assert response == ChatCompletionResponse(messages=[first, second])
    assert response.messages == [first, second]
    assert response.messages[0] is first
    assert response.messages[1] is second


def test_an_empty_message_list_is_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionResponse(messages=[])


def test_terminal_state_is_read_off_the_last_message():
    msg = AssistantMessage(
        content=[TextBlock(text="hi")],
        finish_reason="stop",
        provider_finish_reason="end_turn",
        provider="anthropic",
        model="claude",
        usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
    )
    response = ChatCompletionResponse(messages=[msg])
    assert response.messages[-1] == msg


def test_tool_calls_are_the_same_instances_as_in_the_message():
    tc = ToolCall(id="c", name="t", arguments={"a": 1})
    msg = AssistantMessage(content=[tc])
    response = ChatCompletionResponse(messages=[msg])
    assert response.messages[-1].tool_calls == [tc]
    assert response.messages[-1].tool_calls[0] is tc


def test_message_attributes_are_not_forwarded_onto_the_response():
    # Deliberate: there is no __getattr__ delegation. Callers index `messages`.
    response = ChatCompletionResponse(messages=[AssistantMessage(content=[], finish_reason="stop")])
    with pytest.raises(AttributeError):
        _ = response.finish_reason


def test_raw_is_excluded_from_the_dump():
    msg = AssistantMessage(content=[TextBlock(text="hi")])
    response = ChatCompletionResponse(messages=[msg], raw={"k": "v"})
    assert response.raw == {"k": "v"}
    assert response.model_dump() == {"messages": [msg.model_dump()]}
