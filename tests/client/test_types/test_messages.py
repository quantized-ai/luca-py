"""Discriminated union, tool_calls @property, extra='forbid'."""

import pytest
from pydantic import TypeAdapter, ValidationError

from luca.client.types import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)


message_adapter = TypeAdapter(Message)


def test_user_message_dict_coerces():
    m = message_adapter.validate_python({"role": "user", "content": "hi"})
    assert isinstance(m, UserMessage)
    assert m.content == "hi"


def test_assistant_message_dict_coerces():
    m = message_adapter.validate_python(
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    )
    assert isinstance(m, AssistantMessage)
    assert m.content[0].text == "hi"


def test_tool_message_dict_coerces():
    m = message_adapter.validate_python(
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    )
    assert isinstance(m, ToolMessage)
    assert m.tool_call_id == "c1"


def test_system_role_dict_rejected_by_discriminator():
    with pytest.raises(ValidationError):
        message_adapter.validate_python({"role": "system", "content": "be brief"})


def test_assistant_tool_calls_filters_content():
    msg = AssistantMessage(
        content=[
            TextBlock(text="hello"),
            ToolCall(id="c1", name="t1", arguments={"a": 1}),
            ToolCall(id="c2", name="t2", arguments={"b": 2}),
        ],
    )
    assert [tc.id for tc in msg.tool_calls] == ["c1", "c2"]


def test_assistant_tool_calls_same_instances_as_content():
    tc = ToolCall(id="c1", name="t1", arguments={})
    msg = AssistantMessage(content=[tc])
    msg.tool_calls[0].arguments["mutated"] = True
    # Same instance — mutation via tool_calls view shows up in content.
    assert msg.content[0].arguments == {"mutated": True}


def test_extra_forbid_on_user_message():
    with pytest.raises(ValidationError):
        UserMessage(content="hi", unknown_field=1)
