"""OpenRouter inherits OpenAITransport's projector acceptance set — the
class-based compatibility check composes with transport subclassing."""

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports.anthropic.native_tools import BashTool
from luca.client.transports.openai.transport import OpenAIToolProjector
from luca.client.transports.openai_responses.native_tools import ShellToolCall, ShellToolMessage
from luca.client.transports.openrouter.transport import OpenRouterTransport
from luca.client.types import AssistantMessage, TextBlock, UserMessage


def _transport() -> OpenRouterTransport:
    return OpenRouterTransport(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
    )


def test_projector_base_is_inherited_not_redefined():
    assert OpenRouterTransport.TOOL_PROJECTOR_BASE is OpenAIToolProjector


def test_native_tool_from_another_family_is_rejected_before_http():
    with pytest.raises(BadRequestError, match="targets another transport"):
        _transport()._project_tools([BashTool()])


def test_foreign_native_call_and_result_are_dropped():
    projected = _transport()._project_messages(
        [
            UserMessage(content="hi"),
            AssistantMessage(
                content=[
                    TextBlock(text="running"),
                    ShellToolCall(id="call_2", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1"),
                ],
            ),
            ShellToolMessage(tool_call_id="call_2", results=[]),
        ],
    )
    assert projected == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "running"},
    ]
