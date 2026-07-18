"""OpenAITransport.completion() — non-streaming sync.

Each case is a triple (request, mock_response_json, expected). The classifier
behavior is observable through the public response, so it lives here as
ordinary rows in CASES."""

from dataclasses import dataclass

import pytest

from luca.client.transports import OpenAITransport
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
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
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Hello")],
        ),
        mock_response_json={
            "id": "chatcmpl-abc",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[TextBlock(text="Hi!")],
                finish_reason="stop", provider_finish_reason="stop",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-abc",
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        ),
    ),

    Case(
        name="tool_call_canonicalizes_finish_reason",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Weather?")],
        ),
        mock_response_json={
            "id": "chatcmpl-xyz",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call_abc", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[ToolCall(
                    id="call_abc", name="get_weather",
                    arguments={"city": "NYC"}, complete=True,
                )],
                finish_reason="tool_use",
                provider_finish_reason="tool_calls",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-xyz",
                usage=Usage(input_tokens=12, output_tokens=8, total_tokens=20),
            ),
        ),
    ),

    Case(
        name="content_filter_terminal_maps_to_error",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="...")],
        ),
        mock_response_json={
            "id": "chatcmpl-cf",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "content_filter",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[],
                finish_reason="error",
                provider_finish_reason="content_filter",
                error_message="Provider safety filter (content_filter)",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-cf",
                usage=Usage(input_tokens=10, output_tokens=0, total_tokens=10),
            ),
        ),
    ),

    Case(
        name="strict_mode_refusal_lifts_to_error",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="...")],
        ),
        mock_response_json={
            "id": "chatcmpl-ref",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "refusal": "I can't help with that.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[RefusalBlock(text="I can't help with that.")],
                finish_reason="error",
                provider_finish_reason="stop",
                error_message="OpenAI refusal: I can't help with that.",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-ref",
                usage=Usage(input_tokens=5, output_tokens=6, total_tokens=11),
            ),
        ),
    ),

    Case(
        name="reasoning_becomes_thinking_block",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Think then answer.")],
        ),
        mock_response_json={
            "id": "chatcmpl-think",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": "Let me work through this.",
                    "content": "42",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[
                    ThinkingBlock(text="Let me work through this."),
                    TextBlock(text="42"),
                ],
                finish_reason="stop", provider_finish_reason="stop",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-think",
                usage=Usage(
                    input_tokens=4, output_tokens=9, total_tokens=13,
                    reasoning_tokens=7,
                ),
            ),
        ),
    ),

    Case(
        name="length_terminal",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="...")],
        ),
        mock_response_json={
            "id": "chatcmpl-len",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "partial..."},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105},
        },
        expected=ChatCompletionResponse(
            message=AssistantMessage(
                content=[TextBlock(text="partial...")],
                finish_reason="length",
                provider_finish_reason="length",
                provider="openai", model="gpt-4o",
                response_id="chatcmpl-len",
                usage=Usage(input_tokens=5, output_tokens=100, total_tokens=105),
            ),
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_completion(case, openai_transport_factory):
    client = make_sync_client(json_response(case.mock_response_json))
    transport = openai_transport_factory(http_client=client)
    actual = transport.completion(case.request)
    expected = case.expected.model_copy(update={"raw": case.mock_response_json})
    assert actual == expected
