"""effective_messages on the Converse wire — the first private/web coverage
this wire gets: every private block is foreign (the wire mints none), the
portable web blocks are never sent, thinking survives by provenance, and a
foreign native call takes its toolResult down with it. Plus the payload
invariant and idempotence."""

import pytest

from luca.client.transports.openai_responses.native_tools import ShellToolCall, ShellToolMessage
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    PrivateProviderBlock,
    TextBlock,
    ThinkingBlock,
    UserMessage,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
)


def _request(messages) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="us.amazon.nova-pro-v1:0",
        provider="bedrock",
        messages=messages,
    )


WEB_TURN = AssistantMessage(
    content=[
        PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s1"}),
        WebSearchBlock(queries=["apple"]),
        WebFetchBlock(web_page=WebPagePart(url="https://apple.com")),
        TextBlock(text="Apple is up."),
    ],
    provider="anthropic",
    model="claude-sonnet-5",
)

FOREIGN_NATIVE_EXCHANGE = [
    AssistantMessage(
        content=[
            TextBlock(text="running"),
            ShellToolCall(id="call_2", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1"),
        ],
        provider="openai",
        model="gpt-5.1",
    ),
    ShellToolMessage(tool_call_id="call_2", results=[]),
]


@pytest.mark.parametrize(
    "messages",
    [
        [UserMessage(content="hi"), WEB_TURN],
        [UserMessage(content="hi"), *FOREIGN_NATIVE_EXCHANGE],
    ],
    ids=["web-turn", "foreign-native-exchange"],
)
def test_effective_messages_equivalence(bedrock_transport_factory, messages):
    transport = bedrock_transport_factory()
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport._build_chat_completion_payload(_request(effective)) == transport._build_chat_completion_payload(
        request
    )


def test_foreign_privates_and_web_blocks_drop(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    messages = [UserMessage(content="hi"), WEB_TURN]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[TextBlock(text="Apple is up.")],
            provider="anthropic",
            model="claude-sonnet-5",
        ),
    ]


def test_a_foreign_native_call_drops_with_its_tool_message(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    messages = [UserMessage(content="hi"), *FOREIGN_NATIVE_EXCHANGE]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[TextBlock(text="running")],
            provider="openai",
            model="gpt-5.1",
        ),
    ]


def test_a_fully_filtered_message_returns_empty_not_omitted(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    foreign_only = AssistantMessage(
        content=[
            ThinkingBlock(text="hm", signature="sig-1"),
            PrivateProviderBlock(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
        ],
        provider="openai",
        model="gpt-5.1",
    )
    messages = [UserMessage(content="hi"), foreign_only]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(content=[], provider="openai", model="gpt-5.1"),
    ]


def test_own_thinking_survives_and_foreign_thinking_drops(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    own = AssistantMessage(
        content=[ThinkingBlock(text="own", signature="sig-1"), TextBlock(text="a")],
        provider="bedrock",
        model="us.amazon.nova-pro-v1:0",
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


def test_effective_is_idempotent(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    messages = [UserMessage(content="hi"), WEB_TURN, *FOREIGN_NATIVE_EXCHANGE]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport.effective_messages(effective, _request(effective)) == effective
