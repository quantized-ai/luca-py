"""OpenRouterTransport — subclass of OpenAITransport."""

from luca.client.transports import OpenAITransport, OpenRouterTransport
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ThinkingBlock,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import json_response, make_sync_client


def test_openrouter_is_subclass_of_openai():
    assert issubclass(OpenRouterTransport, OpenAITransport)


def test_openrouter_transport_completion():
    payload = {
        "id": "chatcmpl-or",
        "model": "openai/gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi from OpenRouter."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    client = make_sync_client(json_response(payload))
    transport = OpenRouterTransport(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="or-test",
        http_client=client,
    )
    resp = transport.completion(ChatCompletionRequest(
        model="openai/gpt-4o", provider="openrouter",
        messages=[UserMessage(content="Hi")],
    ))
    assert resp.provider == "openrouter"  # stamped via __init__, not the class
    assert resp.message.content == [TextBlock(text="Hi from OpenRouter.")]
    assert resp.finish_reason == "stop"


def test_openrouter_inherits_reasoning_parsing():
    """The subclass inherits the base's data-driven reasoning → ThinkingBlock."""
    payload = {
        "id": "chatcmpl-or-think",
        "model": "openai/gpt-5.4-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning": "Reasoning from OpenRouter.",
                "content": "Done.",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    client = make_sync_client(json_response(payload))
    transport = OpenRouterTransport(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="or-test",
        http_client=client,
    )
    resp = transport.completion(ChatCompletionRequest(
        model="openai/gpt-5.4-mini", provider="openrouter",
        messages=[UserMessage(content="Hi")],
    ))
    assert resp.message.content == [
        ThinkingBlock(text="Reasoning from OpenRouter."),
        TextBlock(text="Done."),
    ]
    assert resp.finish_reason == "stop"
