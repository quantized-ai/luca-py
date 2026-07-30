"""What OpenAIResponsesTransport sends on the wire."""

import json as _json
from dataclasses import dataclass

import httpx
import pytest

from luca.client.exceptions import BadRequestError, UnsupportedParameterError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ImageBlock,
    MediaBase64,
    MediaFileId,
    MediaURL,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_EMPTY_RESPONSE = {
    "id": "resp_x",
    "model": "gpt-5.4",
    "status": "completed",
    "output": [],
    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
}


@dataclass(frozen=True)
class PayloadCase:
    name: str
    request: ChatCompletionRequest
    expected_url: str
    expected_body: dict
    expected_auth: str


CASES = [
    PayloadCase(
        name="a_plain_turn_is_stateless_with_one_input_item",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hello")],
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="the_system_prompt_becomes_instructions",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hello")],
            system_message="You are concise.",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "store": False,
            "instructions": "You are concise.",
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="sampling_kwargs_use_the_responses_names",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            parallel_tool_calls=False,
            user="u-1",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 100,
            "parallel_tool_calls": False,
            "user": "u-1",
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="reasoning_asks_for_all_turns_and_the_encrypted_payload",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            reasoning="high",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
            "reasoning": {"effort": "high", "context": "all_turns", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="provider_default_reasoning_sends_no_reasoning_key",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            reasoning="provider-default",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="tools_are_flat_with_no_function_envelope",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Weather?")],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_choice={"name": "get_weather"},
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]}],
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="one_assistant_turn_expands_to_reasoning_message_and_function_call_items",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[
                UserMessage(content="Weather?"),
                AssistantMessage(
                    content=[
                        ThinkingBlock(text="Look it up.", id="rs_1", signature="enc-1"),
                        TextBlock(text="Checking."),
                        ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"}),
                    ],
                    provider="openai",
                    model="gpt-5.4",
                ),
                ToolMessage(tool_call_id="call_1", content="sunny"),
            ],
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc-1",
                    "summary": [{"type": "summary_text", "text": "Look it up."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="provider_options_override_anything_derived",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            provider_options={"openai": {"store": True, "mine": 1}, "anthropic": {"theirs": 2}},
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": True,
            "mine": 1,
        },
        expected_auth="Bearer sk-test",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_responses_transport_outbound_payload(case, responses_transport_factory):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_EMPTY_RESPONSE)

    transport = responses_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transport.completion(case.request)

    assert captured == {
        "url": case.expected_url,
        "body": case.expected_body,
        "auth": case.expected_auth,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop", ["END"]),
        ("seed", 42),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logprobs", True),
    ],
)
def test_a_chat_completions_only_parameter_is_refused(field, value, responses_transport_factory):
    # The Responses API has no equivalent. Dropping it would change the output
    # with nothing to notice.
    transport = responses_transport_factory()

    with pytest.raises(UnsupportedParameterError, match=field):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(content="hi")],
                **{field: value},
            ),
        )


def test_every_offending_parameter_is_named_at_once(responses_transport_factory):
    transport = responses_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="stop, seed"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(content="hi")],
                stop=["END"],
                seed=1,
            ),
        )


def test_image_sources_project_to_input_image(responses_transport_factory):
    transport = responses_transport_factory()

    assert transport._project_user_block(
        ImageBlock(source=MediaURL(url="https://example.com/a.png")),
    ) == {"type": "input_image", "image_url": "https://example.com/a.png"}
    assert transport._project_user_block(
        ImageBlock(source=MediaBase64(data="aGk=", media_type="image/png")),
    ) == {"type": "input_image", "image_url": "data:image/png;base64,aGk="}
    # Unlike chat completions, this endpoint takes an uploaded file by id.
    assert transport._project_user_block(
        ImageBlock(source=MediaFileId(file_id="file-abc123")),
    ) == {"type": "input_image", "file_id": "file-abc123"}


def test_an_image_in_a_tool_result_is_refused(responses_transport_factory):
    # Dropping it would tell the model the call succeeded with nothing in it.
    transport = responses_transport_factory()

    with pytest.raises(BadRequestError, match="only text in a function call output"):
        transport._project_tool_message(
            ToolMessage(
                tool_call_id="call_1",
                content=[
                    TextBlock(text="here"),
                    ImageBlock(source=MediaURL(url="https://example.com/a.png")),
                ],
            ),
        )
