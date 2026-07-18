"""End-to-end smoke tests — helper -> provider -> transport, only httpx mocked.

Catches wire-up regressions where every layer passes its own tests but the
helper isn't actually instantiating the provider it should, etc.
"""

import json

import httpx
import pytest

from luca.client import completion, completion_stream
from luca.client.exceptions import RateLimitError
from luca.client.providers import AnthropicProvider, OpenAIProvider
from luca.client.types import TextBlock, Usage, UserMessage


def _smoke_provider(monkeypatch, real_provider):
    """Force the helper to use a specific pre-built provider regardless of cache."""
    monkeypatch.setattr(
        "luca.client._client._get_cached_provider",
        lambda *a, **kw: real_provider,
    )


def test_openai_completion_smoke(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })

    real_provider = OpenAIProvider(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    response = completion(
        model="openai:gpt-4o",
        messages=[UserMessage(content="Hi")],
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert response.finish_reason == "stop"
    assert response.provider == "openai"
    assert response.message.content == [TextBlock(text="Hello!")]
    # Token counts must match exactly; `cost` is auto-populated from the catalog.
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (5, 3, 8)
    assert response.usage.cost is not None and response.usage.cost.total > 0


def test_openai_streaming_smoke(monkeypatch):
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    body = b"".join(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body,
            headers={"content-type": "text/event-stream"},
        )

    real_provider = OpenAIProvider(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    deltas = []
    with completion_stream(
        model="openai:gpt-4o",
        messages=[UserMessage(content="Hi")],
    ) as s:
        for event in s:
            if event.type == "text_delta":
                deltas.append(event.delta)
            elif event.type == "finish":
                final = event
    assert "".join(deltas) == "Hi"
    assert final.finish_reason == "stop"


def test_anthropic_completion_smoke(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "sk-ant-test"
        return httpx.Response(200, json={
            "id": "msg_01", "type": "message", "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        })

    real_provider = AnthropicProvider(
        api_key="sk-ant-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    response = completion(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=[UserMessage(content="Hi")],
    )
    assert response.provider == "anthropic"
    assert response.finish_reason == "stop"
    assert response.provider_finish_reason == "end_turn"


def test_openai_429_maps_to_rate_limit_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "rate_limit_exceeded", "message": "slow down"}},
            headers={"retry-after": "5"},
        )

    real_provider = OpenAIProvider(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    with pytest.raises(RateLimitError) as exc_info:
        completion(
            model="openai:gpt-4o",
            messages=[UserMessage(content="Hi")],
        )
    assert exc_info.value.retry_after == 5.0


def test_structured_output_parse_smoke(monkeypatch):
    from pydantic import BaseModel

    class Movie(BaseModel):
        title: str
        year: int

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "x", "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": '{"title":"Hi","year":2024}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })

    real_provider = OpenAIProvider(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    response = completion(
        model="openai:gpt-4o",
        messages=[UserMessage(content="Give me a movie.")],
        response_format=Movie,
    )
    movie = response.parse()
    assert isinstance(movie, Movie)
    assert movie.title == "Hi"
    assert movie.year == 2024


def test_tool_call_round_trip_smoke(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "x", "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    real_provider = OpenAIProvider(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _smoke_provider(monkeypatch, real_provider)

    response = completion(
        model="openai:gpt-4o",
        messages=[UserMessage(content="Weather?")],
        tools=[{
            "name": "get_weather",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
    )
    assert response.finish_reason == "tool_use"
    assert response.provider_finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "NYC"}
