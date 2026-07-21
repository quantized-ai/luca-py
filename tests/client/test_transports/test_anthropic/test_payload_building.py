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


# ── extended thinking ──────────────────────────────────────────────────────────


def test_no_reasoning_effort_leaves_thinking_off(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
        ),
    )

    assert "thinking" not in payload
    assert payload["max_tokens"] == 4096


def test_adaptive_model_gets_adaptive_thinking_and_effort(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            reasoning_effort="high",
        ),
    )

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in payload["thinking"]


def test_manual_model_gets_a_token_budget(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[UserMessage(content="hi")],
            reasoning_effort="high",
        ),
    )

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert payload["max_tokens"] == 8192 + 4096  # budget + COMPLETION_HEADROOM
    assert "output_config" not in payload


def test_a_model_without_thinking_support_is_sent_disabled(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-3-5-haiku-latest",
            messages=[UserMessage(content="hi")],
            reasoning_effort="high",
        ),
    )

    assert payload["thinking"] == {"type": "disabled"}


def test_effort_none_disables_thinking(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            reasoning_effort="none",
            temperature=0.2,
        ),
    )

    assert payload["thinking"] == {"type": "disabled"}
    # disabled is not active, so the sampling controls stay legal
    assert payload["temperature"] == 0.2


def test_effort_auto_sends_adaptive_without_an_effort_key(anthropic_transport_factory):
    # "auto" means let the model decide, which is the absence of the key
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            reasoning_effort="auto",
        ),
    )

    assert payload["thinking"]["type"] == "adaptive"
    assert "output_config" not in payload


def test_effort_minimal_clamps_to_the_nearest_accepted_value(anthropic_transport_factory):
    # Anthropic accepts only low/medium/high/xhigh
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            reasoning_effort="minimal",
        ),
    )

    assert payload["output_config"] == {"effort": "low"}


def test_a_caller_max_tokens_shrinks_the_budget_rather_than_being_raised(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[UserMessage(content="hi")],
            reasoning_effort="xhigh",  # wants 16384
            max_tokens=4000,
        ),
    )

    assert payload["max_tokens"] == 4000  # the cap is a billing contract
    assert payload["thinking"]["budget_tokens"] == 4000 - 1024


def test_a_max_tokens_too_small_for_any_budget_raises(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="leaves no room"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="claude-haiku-4-5-20251001",
                messages=[UserMessage(content="hi")],
                reasoning_effort="high",
                max_tokens=1500,  # 1500 - 1024 = 476, below the 1024 minimum
            ),
        )


def test_sampling_controls_are_refused_while_thinking(anthropic_transport_factory):
    # Anthropic rejects them when thinking is active. Refusing beats
    # stripping: a silently dropped temperature changes the output with
    # nothing for the caller to notice.
    transport = anthropic_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="temperature, top_p, top_k"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="claude-sonnet-5",
                messages=[UserMessage(content="hi")],
                reasoning_effort="high",
                temperature=0.2, top_p=0.9, top_k=40,
            ),
        )


def test_a_model_without_thinking_support_keeps_its_sampling_controls(
    anthropic_transport_factory,
):
    # thinking is off for these, so nothing about sampling changes
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-3-5-haiku-latest",
            messages=[UserMessage(content="hi")],
            reasoning_effort="high",
            temperature=0.2,
        ),
    )

    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.2


def test_sampling_controls_survive_when_thinking_is_off(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            temperature=0.2,
        ),
    )

    assert payload["temperature"] == 0.2


def test_thinking_mode_survives_aliases_suffixes_and_gateway_prefixes(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()

    assert transport._thinking_mode("claude-haiku-4-5") == "manual"
    assert transport._thinking_mode("claude-haiku-4-5-20251001") == "manual"
    assert transport._thinking_mode("us.anthropic.claude-haiku-4-5") == "manual"
    assert transport._thinking_mode("claude-3-5-haiku-latest") == "none"
    assert transport._thinking_mode("claude-sonnet-5") == "adaptive"
    assert transport._thinking_mode("claude-opus-4-8") == "adaptive"


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


def test_thinking_mode_normalizes_dotted_versions(anthropic_transport_factory):
    # a dotted version is the same model as the dashed one; getting this
    # wrong silently sends the other mode and the request 400s
    transport = anthropic_transport_factory()

    assert transport._thinking_mode("claude-sonnet-4.5") == "manual"
    assert transport._thinking_mode("anthropic/claude-opus-4.5") == "manual"
    assert transport._thinking_mode("claude-3.5-sonnet") == "none"
    assert transport._thinking_mode(
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    ) == "none"
    assert transport._thinking_mode("some-other-model") == "adaptive"
