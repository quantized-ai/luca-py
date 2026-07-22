"""What OpenAITransport sends on the wire."""

from dataclasses import dataclass

import httpx
import json as _json

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ImageBlock,
    MediaBase64,
    MediaFileId,
    MediaURL,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)


@dataclass(frozen=True)
class PayloadCase:
    name: str
    request: ChatCompletionRequest
    expected_url: str
    expected_body: dict
    expected_auth: str


CASES = [
    PayloadCase(
        name="system_message_prepended_as_wire_system_message",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Hello")],
            system_message="You are concise.",
        ),
        expected_url="https://api.openai.com/v1/chat/completions",
        expected_body={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hello"},
            ],
        },
        expected_auth="Bearer sk-test",
    ),

    PayloadCase(
        name="sampling_kwargs_forwarded",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Hi")],
            temperature=0.5, top_p=0.9, max_tokens=100,
            stop=["END"], seed=42,
        ),
        expected_url="https://api.openai.com/v1/chat/completions",
        expected_body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5, "top_p": 0.9,
            "max_tokens": 100, "stop": ["END"], "seed": 42,
        },
        expected_auth="Bearer sk-test",
    ),

    PayloadCase(
        name="provider_options_merge_into_payload",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[UserMessage(content="Hi")],
            provider_options={"openai": {"custom_flag": True}},
        ),
        expected_url="https://api.openai.com/v1/chat/completions",
        expected_body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "custom_flag": True,
        },
        expected_auth="Bearer sk-test",
    ),

    PayloadCase(
        name="tool_call_blocks_project_to_wire_tool_calls",
        request=ChatCompletionRequest(
            model="gpt-4o", provider="openai",
            messages=[
                UserMessage(content="Weather?"),
                AssistantMessage(content=[
                    ToolCall(id="call_abc", name="get_weather",
                             arguments={"city": "NYC"}, complete=True),
                ]),
                ToolMessage(tool_call_id="call_abc", content=[TextBlock(text="18C")]),
            ],
        ),
        expected_url="https://api.openai.com/v1/chat/completions",
        expected_body={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call_abc", "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "18C"},
            ],
        },
        expected_auth="Bearer sk-test",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_outbound_payload(case, openai_transport_factory):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "id": "x", "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    transport = openai_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transport.completion(case.request)

    assert captured == {
        "url": case.expected_url,
        "body": case.expected_body,
        "auth": case.expected_auth,
    }


def test_image_sources_project_to_image_url(openai_transport_factory):
    transport = openai_transport_factory()

    assert transport._project_user_block(
        ImageBlock(source=MediaURL(url="https://example.com/a.png")),
    ) == {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    assert transport._project_user_block(
        ImageBlock(source=MediaBase64(data="aGk=", media_type="image/png")),
    ) == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }


def test_an_image_file_id_is_refused_with_a_useful_message(openai_transport_factory):
    # chat-completions has no file-id shape for images (that is Responses API);
    # sending one would return an opaque 400 from the provider
    transport = openai_transport_factory()

    with pytest.raises(BadRequestError, match="cannot take an image by file id"):
        transport._project_user_block(
            ImageBlock(source=MediaFileId(file_id="file-abc123")),
        )


def test_an_image_in_a_tool_result_is_refused(openai_transport_factory):
    # chat-completions allows only text in a `role: tool` message; dropping
    # the image would tell the model the call succeeded with nothing in it
    transport = openai_transport_factory()
    message = ToolMessage(
        tool_call_id="call_abc",
        content=[
            ImageBlock(source=MediaBase64(data="aGk=", media_type="image/png")),
            TextBlock(text="shot.png"),
        ],
    )

    with pytest.raises(BadRequestError, match="only text in a tool result"):
        transport._project_tool_message(message)


# ── reasoning ──────────────────────────────────────────────────────────────────


def test_the_openai_wire_key_stays_reasoning_effort(openai_transport_factory):
    # `reasoning` is OUR field name; `reasoning_effort` is OpenAI's wire key
    # and every OpenAI-compatible host expects it. Renaming one must not
    # rename the other.
    transport = openai_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-4o", messages=[UserMessage(content="hi")],
            reasoning="high",
        ),
    )

    assert payload["reasoning_effort"] == "high"
    assert "reasoning" not in payload


def test_provider_default_sends_no_reasoning_key(openai_transport_factory):
    transport = openai_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-4o", messages=[UserMessage(content="hi")],
            reasoning="provider-default",
        ),
    )

    assert "reasoning_effort" not in payload


def test_only_this_providers_options_are_merged(openai_transport_factory):
    transport = openai_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-4o", messages=[UserMessage(content="hi")],
            provider_options={"openai": {"mine": 1}, "anthropic": {"theirs": 2}},
        ),
    )

    assert payload["mine"] == 1
    assert "theirs" not in payload
