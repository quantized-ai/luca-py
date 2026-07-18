"""AnthropicTransport.completion() — non-streaming sync."""

from dataclasses import dataclass

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextBlock,
    ToolCall,
    Usage,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import json_response, make_sync_client


@dataclass(frozen=True)
class Case:
    name: str
    request: ChatCompletionRequest
    mock_response_json: dict
    expected: ChatCompletionResponse


CASES = [
    Case(
        name="simple_text",
        request=ChatCompletionRequest(
            model="claude-3-5-sonnet-latest", provider="anthropic",
            messages=[UserMessage(content="Hello")],
        ),
        mock_response_json={
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[TextBlock(text="Hi!")],
                finish_reason="stop", provider_finish_reason="end_turn",
                provider="anthropic", model="claude-3-5-sonnet-20241022",
                response_id="msg_01",
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            ),
        ),
    ),

    Case(
        name="tool_use_canonicalizes",
        request=ChatCompletionRequest(
            model="claude-3-5-sonnet-latest", provider="anthropic",
            messages=[UserMessage(content="Weather?")],
        ),
        mock_response_json={
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 8},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[ToolCall(id="toolu_01", name="get_weather",
                                  arguments={"city": "NYC"}, complete=True)],
                finish_reason="tool_use", provider_finish_reason="tool_use",
                provider="anthropic", model="claude-3-5-sonnet-20241022",
                response_id="msg_02",
                usage=Usage(input_tokens=12, output_tokens=8, total_tokens=20),
            ),
        ),
    ),

    Case(
        name="max_tokens_terminal_maps_to_length",
        request=ChatCompletionRequest(
            model="claude-3-5-sonnet-latest", provider="anthropic",
            messages=[UserMessage(content="...")],
        ),
        mock_response_json={
            "id": "msg_03",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "partial..."}],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 100},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[TextBlock(text="partial...")],
                finish_reason="length", provider_finish_reason="max_tokens",
                provider="anthropic", model="claude-3-5-sonnet-20241022",
                response_id="msg_03",
                usage=Usage(input_tokens=5, output_tokens=100, total_tokens=105),
            ),
        ),
    ),

    Case(
        name="refusal_terminal_maps_to_error",
        request=ChatCompletionRequest(
            model="claude-3-5-sonnet-latest", provider="anthropic",
            messages=[UserMessage(content="...")],
        ),
        mock_response_json={
            "id": "msg_04",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 0},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[TextBlock(text="")],
                finish_reason="error", provider_finish_reason="refusal",
                error_message="Anthropic refusal stop reason",
                provider="anthropic", model="claude-3-5-sonnet-20241022",
                response_id="msg_04",
                usage=Usage(input_tokens=5, output_tokens=0, total_tokens=5),
            ),
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_anthropic_transport_completion(case, anthropic_transport_factory):
    client = make_sync_client(json_response(case.mock_response_json))
    transport = anthropic_transport_factory(http_client=client)
    actual = transport.completion(case.request)
    expected = case.expected.model_copy(update={"raw": case.mock_response_json})
    assert actual == expected
