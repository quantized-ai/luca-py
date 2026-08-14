"""What AnthropicTransport sends on the wire."""

import json as _json

import httpx
import pytest

from luca.client.exceptions import BadRequestError, UnsupportedParameterError
from luca.client.transports.anthropic.transport import _project_file_block
from luca.client.types import (
    AssistantMessage,
    AudioBlock,
    ChatCompletionRequest,
    FileBlock,
    MediaBase64,
    MediaFileId,
    MediaURL,
    TextBlock,
    ThinkingBlock,
    UserMessage,
)


def _ok_response():
    return {
        "id": "x",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": ""}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
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
    transport.completion(
        ChatCompletionRequest(
            model="claude-test",
            provider="anthropic",
            messages=[UserMessage(content="Hi")],
            system_message="Be brief.",
        )
    )

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
    transport.completion(
        ChatCompletionRequest(
            model="claude-test",
            provider="anthropic",
            messages=[UserMessage(content="Hi")],
        )
    )
    # max_tokens must always be present on the Anthropic wire.
    assert "max_tokens" in captured["body"]
    assert captured["body"]["max_tokens"] > 0


# ── extended thinking (wiring only; the mapping lives in test_capabilities) ────


def test_no_reasoning_leaves_thinking_off(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
        ),
    )

    assert "thinking" not in payload
    assert "output_config" not in payload


def test_the_resolved_thinking_reaches_the_payload(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
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
            model="claude-haiku-4-5-20251001",
            messages=[UserMessage(content="hi")],
        ),
    )

    assert payload["max_tokens"] == 64_000


def test_display_is_transport_policy_not_a_model_fact(anthropic_transport_factory):
    class Quiet(type(anthropic_transport_factory())):
        THINKING_DISPLAY = None

    transport = Quiet(provider="anthropic", base_url="https://x", api_key="k")
    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
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
                model="claude-sonnet-5",
                messages=[UserMessage(content="hi")],
                temperature=0.2,
            ),
        )


def test_sampling_survives_where_the_model_allows_it(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-haiku-4-5-20251001",
            messages=[UserMessage(content="hi")],
            temperature=0.2,
        ),
    )

    assert payload["temperature"] == 0.2


# ── provider options ───────────────────────────────────────────────────────────


def test_only_this_providers_options_are_merged(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
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
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
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
        AssistantMessage(
            content=[
                ThinkingBlock(text="let me think", signature="sig-abc"),
                TextBlock(text="the answer"),
            ]
        ),
        ChatCompletionRequest(model="claude-sonnet-5", messages=[]),
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
        AssistantMessage(
            content=[
                ThinkingBlock(text="unsigned reasoning"),
                TextBlock(text="the answer"),
            ]
        ),
        ChatCompletionRequest(model="claude-sonnet-5", messages=[]),
    )

    assert wire["content"] == [{"type": "text", "text": "the answer"}]


def test_a_redacted_block_is_replayed_in_its_own_wire_shape(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()

    wire = transport._project_assistant_message(
        AssistantMessage(
            content=[
                ThinkingBlock(text="", signature="encrypted-payload", redacted=True),
            ]
        ),
        ChatCompletionRequest(model="claude-sonnet-5", messages=[]),
    )

    assert wire["content"] == [
        {"type": "redacted_thinking", "data": "encrypted-payload"},
    ]


def test_a_redacted_block_survives_a_full_receive_then_send(
    anthropic_transport_factory,
):
    transport = anthropic_transport_factory()
    request = ChatCompletionRequest(
        model="claude-sonnet-5",
        messages=[UserMessage(content="hi")],
    )
    message = transport._parse_chat_completion_response(
        httpx.Response(
            200,
            json={
                "id": "x",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted-payload"},
                    {"type": "text", "text": "done"},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        request,
    ).messages[-1]

    assert transport._project_assistant_message(message, request)["content"] == [
        {"type": "redacted_thinking", "data": "encrypted-payload"},
        {"type": "text", "text": "done"},
    ]


# ── structured output ──────────────────────────────────────────────────────────


def test_response_format_projects_to_output_config_format(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
        ),
    )

    assert payload["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        }
    }


def test_the_format_merges_with_an_adaptive_thinking_effort(anthropic_transport_factory):
    # resolve_reasoning already owns `output_config` on adaptive models;
    # assigning over it would silently drop the reasoning level.
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            reasoning="high",
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
        ),
    )

    assert payload["output_config"] == {
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }


def test_a_raw_output_config_does_not_swallow_the_response_format(anthropic_transport_factory):
    # `payload.update(options)` replaces the whole key, so applying the format
    # before the merge lost it silently and the caller got prose back.
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
            provider_options={"anthropic": {"output_config": {"mine": True}}},
        ),
    )

    assert payload["output_config"] == {
        "mine": True,
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }


def test_a_hand_written_format_still_wins(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
            provider_options={"anthropic": {"output_config": {"format": {"type": "text"}}}},
        ),
    )

    assert payload["output_config"] == {"format": {"type": "text"}}


def test_response_format_is_refused_on_a_model_known_to_predate_it(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="predates Anthropic structured outputs"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="claude-3-5-sonnet-latest",
                messages=[UserMessage(content="hi")],
                response_format={"type": "object"},
            ),
        )


def test_an_unknown_model_is_sent_through_rather_than_refused(anthropic_transport_factory):
    # The conservative all-false capability record exists to stop us sending
    # the wrong THINKING shape; refusing here would block every model released
    # after the table was written.
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-something-new",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object"},
        ),
    )

    assert payload["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "required": [], "additionalProperties": False},
        }
    }


def test_a_signature_minted_by_another_pair_is_dropped(anthropic_transport_factory):
    # Anthropic 400s on a foreign attestation exactly as it does on a missing
    # one, and `/model` in the TUI makes the switch a keystroke away.
    transport = anthropic_transport_factory()

    wire = transport._project_assistant_message(
        AssistantMessage(
            content=[
                ThinkingBlock(text="reasoning", signature="sig-from-openai"),
                TextBlock(text="the answer"),
            ],
            provider="openai",
            model="gpt-5.4",
        ),
        ChatCompletionRequest(model="claude-sonnet-5", messages=[]),
    )

    assert wire["content"] == [{"type": "text", "text": "the answer"}]


def test_constraints_anthropics_grammar_rejects_move_into_the_description(
    anthropic_transport_factory,
):
    # Field(ge=…)/Field(min_length=…) are everyday Pydantic and 400 here while
    # working on OpenAI. Anthropic's own SDKs strip and describe them.
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        age: int = Field(ge=0, le=120)
        name: str = Field(min_length=2, description="the name")

    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-sonnet-5",
            messages=[UserMessage(content="hi")],
            response_format=Person,
        ),
    )

    assert payload["output_config"]["format"]["schema"]["properties"] == {
        "age": {
            "title": "Age",
            "type": "integer",
            "description": "Constraints: maximum: 120, minimum: 0",
        },
        "name": {
            "title": "Name",
            "type": "string",
            "description": "the name (minLength: 2)",
        },
    }


def test_structured_output_is_accepted_on_a_4_1_model(anthropic_transport_factory):
    # Anthropic documents 4.5+, but the API accepts opus-4-1 and returns
    # conforming JSON. Trust the wire.
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="claude-opus-4-1",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
        ),
    )

    assert payload["output_config"]["format"]["type"] == "json_schema"


def test_a_file_projects_to_the_document_block_from_every_source():
    # https://platform.claude.com/docs/en/build-with-claude/pdf-support
    base64_block = FileBlock(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"), name="report.pdf")

    assert (
        _project_file_block(base64_block),
        _project_file_block(FileBlock(source=MediaURL(url="https://example.com/a.pdf"))),
        _project_file_block(FileBlock(source=MediaFileId(file_id="file_abc"))),
    ) == (
        # `name` is deliberately absent — a document block has no filename field
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
        },
        {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}},
        {"type": "document", "source": {"type": "file", "file_id": "file_abc"}},
    )


def test_an_audio_block_is_refused_rather_than_stringified(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    message = UserMessage(content=[AudioBlock(source=MediaBase64(data="aGk=", media_type="audio/mpeg"))])

    with pytest.raises(BadRequestError, match="no shape for a AudioBlock"):
        transport._project_user_message(message)
