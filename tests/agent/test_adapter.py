"""Smoke tests for the adapter's two remaining translations: a KNOWN client
assistant message renders into KNOWN agent message parts (the inbound
direction), and a KNOWN agent Tool projects to a KNOWN client Tool definition.
Declarative — hardcoded invariant in, full expected out. No logic, no helpers.
(Conversation → LLM-message projection lives in `test_projection.py`.)
"""

from pydantic import BaseModel, ConfigDict

from luca.agent.core.adapter import message_to_parts, tool_to_luca_tool
from luca.agent.core.models import TextContent, ThinkingContent, ToolCall
from luca.agent.core.tools import Tool
from luca.client.types import TextBlock, ThinkingBlock
from luca.client.types import AssistantMessage as LucaAssistantMessage
from luca.client.types import Tool as LucaTool
from luca.client.types import ToolCall as LucaToolCall


class BinaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    b: int


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    Args = BinaryArgs


def test_message_to_parts_preserves_block_order():
    message = LucaAssistantMessage(
        content=[
            ThinkingBlock(text="Let me add."),
            TextBlock(text="Sure —"),
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ],
    )

    assert message_to_parts(message) == [
        ThinkingContent(thinking="Let me add."),
        TextContent(text="Sure —"),
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    ]


def test_tool_projects_to_client_tool():
    assert tool_to_luca_tool(AddTool()) == LucaTool(
        name="add", description="Add two numbers.", parameters=BinaryArgs,
    )
