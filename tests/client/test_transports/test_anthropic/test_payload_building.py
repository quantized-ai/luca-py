"""What AnthropicTransport sends on the wire."""

import json as _json

import httpx
import pytest

from luca.client.types import (
    ChatCompletionRequest,
    UserMessage,
)


def _ok_response():
    return {
        "id": "x", "type": "message", "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": ""}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_system_message_projected_to_top_level_system(anthropic_transport_factory):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        captured["x_api_key"] = request.headers.get("x-api-key")
        captured["anthropic_version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json=_ok_response())

    transport = anthropic_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transport.completion(ChatCompletionRequest(
        model="claude-test", provider="anthropic",
        messages=[UserMessage(content="Hi")],
        system_message="Be brief.",
    ))

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["body"]["system"] == "Be brief."
    # The wire `messages` should NOT contain a system entry.
    assert all(m["role"] != "system" for m in captured["body"]["messages"])
    assert captured["x_api_key"] == "sk-ant-test"
    assert captured["anthropic_version"] is not None


def test_max_tokens_required_default_used(anthropic_transport_factory):
    captured = {}

    def handler(request):
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=_ok_response())

    transport = anthropic_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transport.completion(ChatCompletionRequest(
        model="claude-test", provider="anthropic",
        messages=[UserMessage(content="Hi")],
    ))
    # max_tokens must always be present on the Anthropic wire.
    assert "max_tokens" in captured["body"]
    assert captured["body"]["max_tokens"] > 0
