"""Native tool-call / tool-message registries: registration at subclass
definition, wrap-validator dispatch on the base classes, and subclass fields
surviving serialization (SerializeAsAny)."""

from typing import Literal

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from luca.client.types import (
    AssistantMessage,
    BaseTool,
    Message,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProjector,
)
from luca.client.types.content import NATIVE_TOOL_CALL_TYPES
from luca.client.types.messages import NATIVE_TOOL_MESSAGE_TYPES

message_adapter = TypeAdapter(Message)


class FakeNativeToolCall(ToolCall):
    type: Literal["fake_native_call"] = "fake_native_call"
    item_id: str | None = None
    status: str = "completed"


class FakeNativeToolMessage(ToolMessage):
    type: Literal["fake_native_output"] = "fake_native_output"
    results: list[dict] = Field(default_factory=list)
    content: str | list = ""


# --- registration ----------------------------------------------------------


def test_native_call_subclass_registers_under_type_literal():
    assert NATIVE_TOOL_CALL_TYPES["fake_native_call"] is FakeNativeToolCall


def test_native_message_subclass_registers_under_type_literal():
    assert NATIVE_TOOL_MESSAGE_TYPES["fake_native_output"] is FakeNativeToolMessage


def test_duplicate_call_type_from_different_class_raises():
    with pytest.raises(TypeError, match="fake_native_call"):

        class Duplicate(ToolCall):
            type: Literal["fake_native_call"] = "fake_native_call"


def test_duplicate_message_type_from_different_class_raises():
    with pytest.raises(TypeError, match="fake_native_output"):

        class Duplicate(ToolMessage):
            type: Literal["fake_native_output"] = "fake_native_output"
            content: str | list = ""


def test_plain_toolcall_subclass_does_not_register():
    class NotNative(ToolCall):
        pass

    assert NATIVE_TOOL_CALL_TYPES.get("tool_call") is None


def test_toolmessage_subclass_without_type_does_not_register():
    before = dict(NATIVE_TOOL_MESSAGE_TYPES)

    class NotNative(ToolMessage):
        pass

    assert before == NATIVE_TOOL_MESSAGE_TYPES


# --- dispatch --------------------------------------------------------------


def test_base_validation_dispatches_native_call():
    tc = ToolCall.model_validate(
        {
            "type": "fake_native_call",
            "id": "call_1",
            "name": "fake",
            "arguments": {"path": "x"},
            "item_id": "item_1",
        },
    )
    assert tc == FakeNativeToolCall(
        id="call_1",
        name="fake",
        arguments={"path": "x"},
        item_id="item_1",
        status="completed",
    )
    assert type(tc) is FakeNativeToolCall


def test_plain_payload_validates_as_base():
    tc = ToolCall.model_validate({"type": "tool_call", "id": "c1", "name": "add"})
    assert type(tc) is ToolCall


def test_unregistered_call_type_fails_validation():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"type": "never_registered", "id": "c1", "name": "x"})


def test_base_validation_dispatches_native_message():
    msg = ToolMessage.model_validate(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "type": "fake_native_output",
            "results": [{"stdout": "hi"}],
        },
    )
    assert msg == FakeNativeToolMessage(tool_call_id="call_1", results=[{"stdout": "hi"}])
    assert type(msg) is FakeNativeToolMessage


def test_tool_message_without_type_stays_base():
    msg = ToolMessage.model_validate({"role": "tool", "tool_call_id": "c1", "content": "ok"})
    assert type(msg) is ToolMessage


def test_tool_message_with_unknown_type_fails_extra_forbid():
    with pytest.raises(ValidationError):
        ToolMessage.model_validate(
            {"role": "tool", "tool_call_id": "c1", "content": "ok", "type": "never_registered"},
        )


def test_message_union_dispatches_native_tool_message():
    msg = message_adapter.validate_python(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "type": "fake_native_output",
            "results": [],
        },
    )
    assert type(msg) is FakeNativeToolMessage


# --- serialization ---------------------------------------------------------


def test_assistant_content_round_trip_keeps_native_call_fields():
    msg = AssistantMessage(
        content=[
            TextBlock(text="hi"),
            FakeNativeToolCall(id="call_1", name="fake", arguments={"a": 1}, item_id="item_1"),
        ],
    )
    restored = AssistantMessage.model_validate(msg.model_dump())
    assert restored == msg
    assert type(restored.content[1]) is FakeNativeToolCall
    assert restored.content[1].item_id == "item_1"


def test_base_toolcall_dump_shape_unchanged():
    msg = AssistantMessage(content=[ToolCall(id="c1", name="add", arguments={"a": 1})])
    assert msg.model_dump()["content"] == [
        {
            "type": "tool_call",
            "id": "c1",
            "name": "add",
            "arguments": {"a": 1},
            "partial_arguments": "",
            "complete": True,
            "thought_signature": None,
        },
    ]


def test_native_tool_message_round_trip():
    msg = FakeNativeToolMessage(tool_call_id="call_1", results=[{"stdout": "hi"}])
    restored = ToolMessage.model_validate(msg.model_dump())
    assert restored == msg
    assert type(restored) is FakeNativeToolMessage


# --- BaseTool / ToolProjector ---------------------------------------------


def test_tool_is_a_basetool_and_default_projector_is_none():
    t = Tool(name="add", description="Add.", parameters={"type": "object"})
    assert isinstance(t, BaseTool)
    assert t.get_projector() is None


def test_projector_hooks_raise_not_implemented():
    p = ToolProjector()
    t = Tool(name="add", description="Add.", parameters={})
    tc = ToolCall(id="c1", name="add")
    msg = ToolMessage(tool_call_id="c1", content="ok")
    with pytest.raises(NotImplementedError):
        p.project_tool_to_llm(t)
    with pytest.raises(NotImplementedError):
        p.build_tool_call({})
    with pytest.raises(NotImplementedError):
        p.project_tool_call_to_llm(tc)
    with pytest.raises(NotImplementedError):
        p.project_tool_message_to_llm(msg, None)
    with pytest.raises(NotImplementedError):
        p.project_tool_choice_to_llm(t)
