"""OpenAIResponsesTransport.completion() — non-streaming sync.

Each case is a triple (request, mock_response_json, expected). The Responses
API has no `finish_reason`, so terminal classification is derived from `status`
plus the assembled content; that behavior is observable through the public
response and lives here as ordinary rows in CASES."""

from dataclasses import dataclass

import pytest

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


REQUEST = ChatCompletionRequest(
    model="gpt-5.4",
    provider="openai",
    messages=[UserMessage(content="Hello")],
)


CASES = [
    Case(
        name="simple_text",
        request=REQUEST,
        mock_response_json={
            "id": "resp_1",
            "model": "gpt-5.4-2026-01-01",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi!"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="Hi!")],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4-2026-01-01",
                    response_id="resp_1",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                )
            ],
        ),
    ),
    Case(
        name="a_reasoning_item_keeps_its_id_and_encrypted_payload",
        request=REQUEST,
        mock_response_json={
            "id": "resp_2",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": "enc-1",
                    "summary": [
                        {"type": "summary_text", "text": "**Reading the puzzle**"},
                        {"type": "summary_text", "text": "Then answering it."},
                    ],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "42"}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 16},
            },
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[
                        ThinkingBlock(
                            text="**Reading the puzzle**\n\nThen answering it.",
                            id="rs_1",
                            signature="enc-1",
                        ),
                        TextBlock(text="42"),
                    ],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_2",
                    usage=Usage(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        cached_input_tokens=4,
                        reasoning_tokens=16,
                    ),
                )
            ],
        ),
    ),
    Case(
        name="a_function_call_makes_a_completed_status_mean_tool_use",
        request=REQUEST,
        mock_response_json={
            "id": "resp_3",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[
                        ToolCall(
                            id="call_1",
                            name="get_weather",
                            arguments={"city": "NYC"},
                            complete=True,
                        )
                    ],
                    finish_reason="tool_use",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_3",
                    usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
                )
            ],
        ),
    ),
    Case(
        name="a_refusal_part_classifies_as_error",
        request=REQUEST,
        mock_response_json={
            "id": "resp_4",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 3, "total_tokens": 6},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[RefusalBlock(text="I can't help with that.")],
                    finish_reason="error",
                    provider_finish_reason="completed",
                    error_message="OpenAI refusal: I can't help with that.",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_4",
                    usage=Usage(input_tokens=3, output_tokens=3, total_tokens=6),
                )
            ],
        ),
    ),
    Case(
        name="max_output_tokens_classifies_as_length",
        request=REQUEST,
        mock_response_json={
            "id": "resp_5",
            "model": "gpt-5.4",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Once upon a"}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="Once upon a")],
                    finish_reason="length",
                    provider_finish_reason="incomplete:max_output_tokens",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_5",
                    usage=Usage(input_tokens=2, output_tokens=4, total_tokens=6),
                )
            ],
        ),
    ),
    Case(
        name="a_content_filter_is_a_response_not_an_exception",
        request=REQUEST,
        mock_response_json={
            "id": "resp_6",
            "model": "gpt-5.4",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [],
            "usage": {"input_tokens": 2, "output_tokens": 0, "total_tokens": 2},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[],
                    finish_reason="error",
                    provider_finish_reason="incomplete:content_filter",
                    error_message="Provider safety filter (content_filter)",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_6",
                    usage=Usage(input_tokens=2, output_tokens=0, total_tokens=2),
                )
            ],
        ),
    ),
    Case(
        name="a_failed_response_carries_the_providers_own_error_message",
        request=REQUEST,
        mock_response_json={
            "id": "resp_7",
            "model": "gpt-5.4",
            "status": "failed",
            "error": {"code": "server_error", "message": "upstream exploded"},
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[],
                    finish_reason="error",
                    provider_finish_reason="failed",
                    error_message="upstream exploded",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_7",
                    usage=Usage(input_tokens=1, output_tokens=0, total_tokens=1),
                )
            ],
        ),
    ),
    Case(
        name="a_hosted_tool_item_is_ignored_not_guessed_at",
        request=REQUEST,
        mock_response_json={
            "id": "resp_8",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {"id": "fs_1", "type": "file_search_call", "status": "completed"},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Found it."}],
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
        expected=ChatCompletionResponse(
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="Found it.")],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    response_model="gpt-5.4",
                    response_id="resp_8",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                )
            ],
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_responses_transport_completion(case, responses_transport_factory):
    client = make_sync_client(json_response(case.mock_response_json))
    transport = responses_transport_factory(http_client=client)

    actual = transport.completion(case.request)

    expected = case.expected.model_copy(update={"raw": case.mock_response_json})
    assert actual == expected


def test_valid_but_non_object_tool_arguments_do_not_crash(responses_transport_factory):
    # `ToolCall.arguments` is a dict, so "null" would escape as a raw pydantic
    # ValidationError instead of a ClientError.
    transport = responses_transport_factory(
        http_client=make_sync_client(
            json_response(
                {
                    "id": "resp_9",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [
                        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "null"}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            )
        )
    )

    response = transport.completion(REQUEST)

    assert response.messages[-1].tool_calls == [ToolCall(id="call_1", name="f", arguments={}, complete=True)]


def test_cache_write_tokens_are_read_from_the_wire(responses_transport_factory):
    transport = responses_transport_factory(
        http_client=make_sync_client(
            json_response(
                {
                    "id": "resp_10",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "total_tokens": 11,
                        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 6},
                    },
                }
            )
        )
    )

    response = transport.completion(REQUEST)

    assert response.messages[-1].usage == Usage(
        input_tokens=10,
        output_tokens=1,
        total_tokens=11,
        cached_input_tokens=4,
        cache_write_tokens=6,
    )


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled"], ids=["queued", "in_progress", "cancelled"])
def test_a_non_terminal_status_does_not_leak_into_the_canonical_finish_reason(status, responses_transport_factory):
    # Reachable through provider_options={"openai": {"background": True}}.
    # `finish_reason` is pinned to stop | length | tool_use | error | None.
    transport = responses_transport_factory(
        http_client=make_sync_client(
            json_response({"id": "r", "model": "gpt-5.4", "status": status, "output": [], "usage": {}})
        )
    )

    response = transport.completion(REQUEST)

    assert (response.messages[-1].finish_reason, response.messages[-1].provider_finish_reason) == (None, status)
