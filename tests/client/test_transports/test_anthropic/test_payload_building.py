"""What AnthropicTransport sends on the wire."""

import json as _json

import httpx
import pytest

from luca.client.exceptions import UnsupportedParameterError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ThinkingBlock,
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


# ── extended thinking (wiring only; the mapping lives in test_capabilities) ────


def test_no_reasoning_leaves_thinking_off(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5", messages=[UserMessage(content="hi")],
        ),
    )

    assert "thinking" not in payload
    assert "output_config" not in payload


def test_the_resolved_thinking_reaches_the_payload(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5", messages=[UserMessage(content="hi")],
            reasoning="high",
        ),
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["max_tokens"] == 128_000


def test_max_tokens_comes_from_the_models_own_ceiling(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-haiku-4-5-20251001", messages=[UserMessage(content="hi")],
        ),
    )

    assert payload["max_tokens"] == 64_000


def test_display_is_transport_policy_not_a_model_fact(anthropic_transport_factory):
    class Quiet(type(anthropic_transport_factory())):
        THINKING_DISPLAY = None

    transport = Quiet(provider="anthropic", base_url="https://x", api_key="k")
    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5", messages=[UserMessage(content="hi")],
            reasoning="high",
        ),
    )

    assert payload["thinking"] == {"type": "adaptive"}


def test_sampling_is_refused_on_a_model_that_rejects_it(anthropic_transport_factory):
    # claude-sonnet-5 refuses temperature outright, thinking or not
    transport = anthropic_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="does not accept"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="claude-sonnet-5", messages=[UserMessage(content="hi")],
                temperature=0.2,
            ),
        )


def test_sampling_survives_where_the_model_allows_it(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-haiku-4-5-20251001", messages=[UserMessage(content="hi")],
            temperature=0.2,
        ),
    )

    assert payload["temperature"] == 0.2


# ── provider options ───────────────────────────────────────────────────────────


def test_only_this_providers_options_are_merged(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5", messages=[UserMessage(content="hi")],
            provider_options={"anthropic": {"mine": 1}, "openai": {"theirs": 2}},
        ),
    )

    assert payload["mine"] == 1
    assert "theirs" not in payload


def test_raw_thinking_options_replace_resolution_rather_than_merging(
    anthropic_transport_factory,
):
    # a caller who spelled out `thinking` gets exactly that; the two are
    # never merged
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5", messages=[UserMessage(content="hi")],
            reasoning="high",
            provider_options={"anthropic": {"thinking": {"type": "disabled"}}},
        ),
    )

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


# ── thinking round-trip ────────────────────────────────────────────────────────


def test_a_signed_thinking_block_is_replayed_with_its_signature(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()

    wire = transport._project_assistant_message(
        AssistantMessage(content=[
            ThinkingBlock(text="let me think", signature="sig-abc"),
            TextBlock(text="the answer"),
        ]),
    )

    assert wire["content"] == [
        {"type": "thinking", "thinking": "let me think", "signature": "sig-abc"},
        {"type": "text", "text": "the answer"},
    ]


def test_an_unsigned_thinking_block_is_dropped_not_sent(anthropic_transport_factory):
    # Anthropic 400s on a thinking block with no signature but accepts the
    # turn without it. Unsigned blocks are reachable: a truncated response
    # never gets its signature_delta, and reasoning from an OpenAI-compatible
    # host was never signed at all.
    transport = anthropic_transport_factory()

    wire = transport._project_assistant_message(
        AssistantMessage(content=[
            ThinkingBlock(text="unsigned reasoning"),
            TextBlock(text="the answer"),
        ]),
    )

    assert wire["content"] == [{"type": "text", "text": "the answer"}]


def test_a_redacted_block_is_replayed_in_its_own_wire_shape(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()

    wire = transport._project_assistant_message(
        AssistantMessage(content=[
            ThinkingBlock(text="", signature="encrypted-payload", redacted=True),
        ]),
    )

    assert wire["content"] == [
        {"type": "redacted_thinking", "data": "encrypted-payload"},
    ]


def test_a_redacted_block_survives_a_full_receive_then_send(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()
    request = ChatCompletionRequest(
        model="claude-sonnet-5", messages=[UserMessage(content="hi")],
    )
    message = transport._parse_chat_completion_response(
        httpx.Response(200, json={
            "id": "x", "type": "message", "role": "assistant",
            "model": "claude-test",
            "content": [
                {"type": "redacted_thinking", "data": "encrypted-payload"},
                {"type": "text", "text": "done"},
            ],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }),
        request,
    ).message

    assert transport._project_assistant_message(message)["content"] == [
        {"type": "redacted_thinking", "data": "encrypted-payload"},
        {"type": "text", "text": "done"},
    ]
