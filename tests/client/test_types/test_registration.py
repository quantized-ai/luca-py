"""Importing `luca.client` alone registers every first-party native type —
this pins the eager import chain (client -> providers -> transports ->
native_tools). Subset asserts: other test modules legitimately register
throwaway types of their own."""

import luca.client  # noqa: F401  — the import IS the act under test
from luca.client.transports.openai_responses.native_tools import (
    ApplyPatchProjector,
    ApplyPatchToolCall,
    ShellProjector,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types.content import NATIVE_TOOL_CALL_TYPES
from luca.client.types.messages import NATIVE_TOOL_MESSAGE_TYPES


def test_first_party_call_types_register_at_import():
    assert NATIVE_TOOL_CALL_TYPES["apply_patch_call"] is ApplyPatchToolCall
    assert NATIVE_TOOL_CALL_TYPES["shell_call"] is ShellToolCall


def test_first_party_message_types_register_at_import():
    assert NATIVE_TOOL_MESSAGE_TYPES["shell_call_output"] is ShellToolMessage


def test_call_classes_bind_their_projectors():
    assert ApplyPatchToolCall.projector_class is ApplyPatchProjector
    assert ShellToolCall.projector_class is ShellProjector
