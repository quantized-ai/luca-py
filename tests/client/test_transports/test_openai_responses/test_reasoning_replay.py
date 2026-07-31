"""Which reasoning items go back on the wire, and which are dropped.

A reasoning item is only replayable with BOTH halves of its provider identity
(the `rs_…` id and the encrypted payload) and only to the (provider, model) pair
that minted it. Everything else is a 400 that would make the whole conversation
permanently unusable — reachable in one keystroke, since the agent TUI's
`/model` rewrites the session model between turns."""

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ThinkingBlock,
)

CALL = ChatCompletionRequest(model="gpt-5.4", provider="openai", messages=[])


def _items(transport, message):
    return transport._project_assistant_message(message, CALL)


def test_a_reasoning_item_from_the_same_pair_is_replayed_whole(responses_transport_factory):
    transport = responses_transport_factory()

    assert _items(
        transport,
        AssistantMessage(
            content=[
                ThinkingBlock(text="Think.", id="rs_1", signature="enc-1"),
                TextBlock(text="Answer."),
            ],
            provider="openai",
            model="gpt-5.4",
        ),
    ) == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-1",
            "summary": [{"type": "summary_text", "text": "Think."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer."}],
        },
    ]


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-5"),  # provider switched
        ("openai", "gpt-5.4-mini"),  # same provider, different model
        ("openrouter", "openai/gpt-5.4"),  # same model, different host
    ],
    ids=["provider_switched", "model_switched", "host_switched"],
)
def test_an_item_minted_by_another_pair_is_dropped(provider, model, responses_transport_factory):
    transport = responses_transport_factory()

    assert _items(
        transport,
        AssistantMessage(
            content=[
                ThinkingBlock(text="Think.", id="rs_1", signature="enc-1"),
                TextBlock(text="Answer."),
            ],
            provider=provider,
            model=model,
        ),
    ) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer."}],
        },
    ]


@pytest.mark.parametrize(
    "block",
    [
        ThinkingBlock(text="Think.", signature="enc-1"),  # no id
        ThinkingBlock(text="Think.", id="rs_1"),  # no encrypted payload
        ThinkingBlock(text="Think."),  # neither — e.g. reasoning from a chat-completions host
    ],
    ids=["no_id", "no_encrypted_content", "neither"],
)
def test_a_half_identified_block_is_dropped(block, responses_transport_factory):
    transport = responses_transport_factory()

    assert (
        _items(
            transport,
            AssistantMessage(content=[block], provider="openai", model="gpt-5.4"),
        )
        == []
    )


def test_a_message_with_no_provenance_is_trusted(responses_transport_factory):
    # A hand-built message means the caller is driving; stripping data they
    # deliberately supplied would be the wrong default.
    transport = responses_transport_factory()

    assert _items(
        transport,
        AssistantMessage(content=[ThinkingBlock(text="Think.", id="rs_1", signature="enc-1")]),
    ) == [
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-1",
            "summary": [{"type": "summary_text", "text": "Think."}],
        },
    ]


def test_an_encrypted_only_item_replays_with_an_empty_summary(responses_transport_factory):
    # `summary: "auto"` is a request, not a guarantee: an item can come back
    # with the payload and no renderable text, and it is still replayable.
    transport = responses_transport_factory()

    assert _items(
        transport,
        AssistantMessage(
            content=[ThinkingBlock(text="", id="rs_1", signature="enc-1")],
            provider="openai",
            model="gpt-5.4",
        ),
    ) == [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1", "summary": []},
    ]
