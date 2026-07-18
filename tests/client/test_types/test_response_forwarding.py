"""ChatCompletionResponse.__getattr__ forwards to self.message."""

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    ToolCall,
    Usage,
)


def test_forwards_finish_reason_to_message():
    msg = AssistantMessage(
        content=[TextBlock(text="hi")],
        finish_reason="stop",
        provider_finish_reason="end_turn",
        provider="anthropic",
        model="claude",
    )
    response = ChatCompletionResponse(message=msg)
    assert response.finish_reason == "stop"
    assert response.provider_finish_reason == "end_turn"
    assert response.provider == "anthropic"


def test_forwards_tool_calls_same_instances():
    tc = ToolCall(id="c", name="t", arguments={"a": 1})
    msg = AssistantMessage(content=[tc])
    response = ChatCompletionResponse(message=msg)
    assert response.tool_calls == [tc]
    assert response.tool_calls[0] is tc


def test_declared_fields_are_not_forwarded():
    msg = AssistantMessage(content=[], provider="x", model="y")
    response = ChatCompletionResponse(message=msg, raw={"k": "v"})
    assert response.message is msg
    assert response.raw == {"k": "v"}


def test_unknown_attribute_raises():
    msg = AssistantMessage(content=[], provider="x", model="y")
    response = ChatCompletionResponse(message=msg)
    with pytest.raises(AttributeError):
        _ = response.nonexistent_attribute


def test_usage_forwarded():
    msg = AssistantMessage(
        content=[], provider="x", model="y",
        usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
    )
    response = ChatCompletionResponse(message=msg)
    assert response.usage.total_tokens == 7
