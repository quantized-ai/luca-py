"""The generic form of a native call/result: `extras` + `custom_type`.

`as_native()` / `as_generic()` are pure type-layer conversions — no transport,
no provider. The invariant they owe: `x.as_generic().as_native() == x`.
"""

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from luca.client.exceptions import BadRequestError
from luca.client.types import AssistantMessage, TextBlock, ToolCall, ToolMessage


class FakeResult(BaseModel):
    stdout: str

    model_config = ConfigDict(extra="forbid")


class FakeNativeCall(ToolCall):
    type: Literal["fake_extras_call"] = "fake_extras_call"
    item_id: str | None = None
    status: str = "completed"


class FakeNativeMessage(ToolMessage):
    type: Literal["fake_extras_call_output"] = "fake_extras_call_output"
    results: list[FakeResult]
    content: str = ""


NATIVE_CALL = FakeNativeCall(
    id="call_1",
    name="fake",
    arguments={"a": 1},
    item_id="fk_1",
    status="completed",
)

GENERIC_CALL = ToolCall(
    id="call_1",
    name="fake",
    arguments={"a": 1},
    extras={"custom_type": "fake_extras_call", "item_id": "fk_1", "status": "completed"},
)

NATIVE_MESSAGE = FakeNativeMessage(tool_call_id="call_1", results=[FakeResult(stdout="hi")])

GENERIC_MESSAGE = ToolMessage(
    tool_call_id="call_1",
    content="",
    extras={"custom_type": "fake_extras_call_output", "results": [{"stdout": "hi"}]},
)


# --- the conversion pair ---------------------------------------------------


def test_generic_call_becomes_the_registered_native_class():
    assert GENERIC_CALL.as_native() == NATIVE_CALL


def test_native_call_becomes_the_generic_form():
    assert NATIVE_CALL.as_generic() == GENERIC_CALL


def test_generic_message_becomes_the_registered_native_class():
    assert GENERIC_MESSAGE.as_native() == NATIVE_MESSAGE


def test_native_message_becomes_the_generic_form():
    assert NATIVE_MESSAGE.as_generic() == GENERIC_MESSAGE


@pytest.mark.parametrize("entry", [NATIVE_CALL, NATIVE_MESSAGE], ids=["call", "message"])
def test_the_round_trip_is_lossless(entry):
    assert entry.as_generic().as_native() == entry


# --- nothing to do ---------------------------------------------------------


def test_a_plain_call_is_returned_unchanged_by_both():
    call = ToolCall(id="call_1", name="add", arguments={"a": 1})
    assert call.as_native() is call
    assert call.as_generic() is call


def test_extras_without_a_custom_type_are_carried_but_inert():
    call = ToolCall(id="call_1", name="add", extras={"trace_id": "t-9"})
    assert call.as_native() is call


def test_a_native_instance_is_already_native():
    assert NATIVE_CALL.as_native() is NATIVE_CALL


def test_extras_on_a_native_instance_are_inert_and_survive_the_round_trip():
    call = FakeNativeCall(id="call_1", name="fake", item_id="fk_1", extras={"trace_id": "t-9"})
    assert call.as_native() is call
    assert call.as_generic() == ToolCall(
        id="call_1",
        name="fake",
        extras={
            "trace_id": "t-9",
            "custom_type": "fake_extras_call",
            "item_id": "fk_1",
            "status": "completed",
        },
    )


# --- errors ----------------------------------------------------------------


def test_an_unregistered_custom_type_is_refused():
    call = ToolCall(id="call_1", name="ghost", extras={"custom_type": "never_registered_call"})
    with pytest.raises(BadRequestError, match="Unknown native tool-call type 'never_registered_call'"):
        call.as_native()


def test_an_unregistered_custom_type_on_a_result_is_refused():
    msg = ToolMessage(tool_call_id="call_1", content="", extras={"custom_type": "never_registered_output"})
    with pytest.raises(BadRequestError, match="Unknown native tool-message type 'never_registered_output'"):
        msg.as_native()


def test_extras_the_native_class_rejects_are_refused():
    msg = ToolMessage(tool_call_id="call_1", content="", extras={"custom_type": "fake_extras_call_output"})
    with pytest.raises(BadRequestError, match="extras do not describe a valid FakeNativeMessage"):
        msg.as_native()


def test_an_unknown_extras_key_is_refused():
    call = ToolCall(id="call_1", name="fake", extras={"custom_type": "fake_extras_call", "nope": 1})
    with pytest.raises(BadRequestError, match="extras do not describe a valid FakeNativeCall"):
        call.as_native()


# --- persistence -----------------------------------------------------------


def test_the_generic_form_round_trips_through_json_as_the_base_class():
    assistant = AssistantMessage(content=[TextBlock(text="hi"), GENERIC_CALL])

    restored_assistant = AssistantMessage.model_validate(assistant.model_dump())
    restored_message = ToolMessage.model_validate(GENERIC_MESSAGE.model_dump())

    assert restored_assistant == assistant
    assert type(restored_assistant.content[1]) is ToolCall
    assert restored_message == GENERIC_MESSAGE
    assert type(restored_message) is ToolMessage
