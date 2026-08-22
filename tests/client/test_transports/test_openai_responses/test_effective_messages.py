"""effective_messages on the Responses wire: the payload invariant
`payload(messages) == payload(effective(messages))`, idempotence, and the
block rules — own/foreign privates (incl. the in_progress stub) and thinking
identity halves. No foreign-family native call exists for this wire today,
so lineage drops are covered on the other transports. Wire fixtures are
shared with test_web_tools.py."""

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    PrivateProviderBlock,
    TextBlock,
    ThinkingBlock,
    UserMessage,
    WebSearchBlock,
)
from tests.client.test_transports.test_openai_responses.test_web_tools import (
    HISTORY,
    SEARCH_ITEM,
)


def _request(messages) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-5.1",
        provider="openai",
        messages=messages,
    )


IN_PROGRESS_STUB = AssistantMessage(
    content=[
        PrivateProviderBlock(
            format="openai.responses",
            data={"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
        ),
        TextBlock(text="partial"),
    ],
    provider="openai",
    model="gpt-5.1",
)


@pytest.mark.parametrize(
    "messages",
    [
        [UserMessage(content="hi"), HISTORY],
        [UserMessage(content="hi"), IN_PROGRESS_STUB],
    ],
    ids=["history", "in-progress-stub"],
)
def test_effective_messages_equivalence(responses_transport_factory, messages):
    transport = responses_transport_factory()
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport._build_chat_completion_payload(_request(effective)) == transport._build_chat_completion_payload(
        request
    )


def test_the_history_keeps_own_privates_and_text_only(responses_transport_factory):
    transport = responses_transport_factory()
    messages = [UserMessage(content="hi"), HISTORY]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                PrivateProviderBlock(format="openai.responses", data=SEARCH_ITEM),
                TextBlock(text="Apple's latest quarterly report says..."),
            ],
            provider="openai",
            model="gpt-5.1",
        ),
    ]


def test_the_in_progress_stub_is_dropped(responses_transport_factory):
    transport = responses_transport_factory()
    messages = [UserMessage(content="hi"), IN_PROGRESS_STUB]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(content=[TextBlock(text="partial")], provider="openai", model="gpt-5.1"),
    ]


def test_a_fully_filtered_message_returns_empty_not_omitted(responses_transport_factory):
    transport = responses_transport_factory()
    foreign_only = AssistantMessage(
        content=[
            PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s1"}),
            WebSearchBlock(queries=["apple"]),
        ],
        provider="anthropic",
        model="claude-sonnet-5",
    )
    messages = [UserMessage(content="hi"), foreign_only]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(content=[], provider="anthropic", model="claude-sonnet-5"),
    ]


def test_own_thinking_survives_and_foreign_thinking_drops(responses_transport_factory):
    transport = responses_transport_factory()
    own = AssistantMessage(
        content=[ThinkingBlock(text="own", id="rs_1", signature="enc-1"), TextBlock(text="a")],
        provider="openai",
        model="gpt-5.1",
    )
    foreign = AssistantMessage(
        content=[ThinkingBlock(text="foreign", signature="sig-2"), TextBlock(text="b")],
        provider="anthropic",
        model="claude-sonnet-5",
    )
    messages = [UserMessage(content="hi"), own, foreign]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        own,
        AssistantMessage(content=[TextBlock(text="b")], provider="anthropic", model="claude-sonnet-5"),
    ]


def test_effective_is_idempotent(responses_transport_factory):
    transport = responses_transport_factory()
    messages = [UserMessage(content="hi"), HISTORY, IN_PROGRESS_STUB]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport.effective_messages(effective, _request(effective)) == effective
