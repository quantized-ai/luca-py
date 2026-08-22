"""effective_messages on the Anthropic wire: the payload invariant
`payload(messages) == payload(effective(messages))`, idempotence, and the
block rules — own/foreign privates, the merged-cited replay, thinking
provenance, foreign native lineage. Wire fixtures are shared with
test_web_tools.py."""

import pytest

from luca.client.transports.openai_responses.native_tools import ShellToolCall, ShellToolMessage
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    PrivateProviderBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    URLCitationAnnotation,
    UserMessage,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
)
from tests.client.test_transports.test_anthropic.test_web_tools import (
    CITED_RUN,
    FETCH_CALL,
    FETCH_RESULT,
    SEARCH_CALL,
    SEARCH_RESULT,
)


def _request(messages) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet-5",
        provider="anthropic",
        messages=messages,
    )


WEB_RUN = AssistantMessage(
    content=[
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_RESULT),
        WebSearchBlock(queries=["Apple latest quarterly results"], results=[]),
        TextBlock(text="Apple's latest quarterly report says..."),
    ],
    provider="anthropic",
    model="claude-sonnet-5",
)

FETCH_RUN = AssistantMessage(
    content=[
        PrivateProviderBlock(format="anthropic.messages", data=FETCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=FETCH_RESULT),
        WebFetchBlock(web_page=WebPagePart(url="https://apple.com")),
    ],
    provider="anthropic",
    model="claude-sonnet-5",
)

CITED_ANSWER = AssistantMessage(
    content=[
        TextBlock(
            text="Market update: Apple rose 2.8% today. NVIDIA was flat.",
            annotations=[
                URLCitationAnnotation(
                    url="https://example.com/apple",
                    title="Apple shares rise",
                    start_index=15,
                    end_index=37,
                )
            ],
        ),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[0]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[1]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[2]),
    ],
    provider="anthropic",
    model="claude-sonnet-5",
)

FOREIGN_TURN = AssistantMessage(
    content=[
        ThinkingBlock(text="hm", signature="rs-sig", id="rs_1"),
        PrivateProviderBlock(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
        WebSearchBlock(queries=["apple"]),
        TextBlock(text="Found it."),
    ],
    provider="openai",
    model="gpt-5.1",
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
        [UserMessage(content="hi"), WEB_RUN],
        [UserMessage(content="hi"), FETCH_RUN],
        [UserMessage(content="hi"), CITED_ANSWER],
        [UserMessage(content="hi"), FOREIGN_TURN],
        [UserMessage(content="hi"), *FOREIGN_NATIVE_EXCHANGE],
    ],
    ids=["web-run", "fetch-run", "cited-answer", "foreign-turn", "foreign-native-exchange"],
)
def test_effective_messages_equivalence(anthropic_transport_factory, messages):
    transport = anthropic_transport_factory()
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport._build_chat_completion_payload(_request(effective)) == transport._build_chat_completion_payload(
        request
    )


def test_the_web_run_keeps_privates_and_sheds_the_portable_block(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    messages = [UserMessage(content="hi"), WEB_RUN]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
                PrivateProviderBlock(format="anthropic.messages", data=SEARCH_RESULT),
                TextBlock(text="Apple's latest quarterly report says..."),
            ],
            provider="anthropic",
            model="claude-sonnet-5",
        ),
    ]


def test_the_fetch_run_keeps_privates_and_sheds_the_portable_block(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    messages = [UserMessage(content="hi"), FETCH_RUN]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                PrivateProviderBlock(format="anthropic.messages", data=FETCH_CALL),
                PrivateProviderBlock(format="anthropic.messages", data=FETCH_RESULT),
            ],
            provider="anthropic",
            model="claude-sonnet-5",
        ),
    ]


def test_the_merged_cited_text_drops_when_its_privates_replay(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    messages = [UserMessage(content="hi"), CITED_ANSWER]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[0]),
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[1]),
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[2]),
            ],
            provider="anthropic",
            model="claude-sonnet-5",
        ),
    ]


def test_a_foreign_native_call_drops_with_its_tool_message(anthropic_transport_factory):
    transport = anthropic_transport_factory()
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


def test_a_fully_filtered_message_returns_empty_not_omitted(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    foreign_only = AssistantMessage(
        content=[
            PrivateProviderBlock(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
            WebSearchBlock(queries=["apple"]),
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


def test_own_thinking_survives_and_foreign_thinking_drops(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    own = AssistantMessage(
        content=[ThinkingBlock(text="own", signature="sig-1"), TextBlock(text="a")],
        provider="anthropic",
        model="claude-sonnet-5",
    )
    foreign = AssistantMessage(
        content=[ThinkingBlock(text="foreign", signature="sig-2"), TextBlock(text="b")],
        provider="openai",
        model="gpt-5.1",
    )
    messages = [UserMessage(content="hi"), own, foreign]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        own,
        AssistantMessage(content=[TextBlock(text="b")], provider="openai", model="gpt-5.1"),
    ]


def test_effective_is_idempotent(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    messages = [UserMessage(content="hi"), WEB_RUN, FETCH_RUN, CITED_ANSWER, FOREIGN_TURN]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport.effective_messages(effective, _request(effective)) == effective


# A dropped foreign thinking block sits between plain text and a cited run's
# privates: selection creates a text/private adjacency the original list never
# had, and a bare-adjacency merged-cited rule would misclassify the plain text
# on the next pass. The rule keys on text equality with the private run, so
# the plain text survives and the invariant holds.
ADJACENCY_TRAP = AssistantMessage(
    content=[
        TextBlock(text="Let me check the market."),
        ThinkingBlock(text="hm", signature="foreign-sig"),
        TextBlock(
            text="Market update: Apple rose 2.8% today. NVIDIA was flat.",
            annotations=[
                URLCitationAnnotation(
                    url="https://example.com/apple",
                    title="Apple shares rise",
                    start_index=15,
                    end_index=37,
                )
            ],
        ),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[0]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[1]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[2]),
    ],
    provider="anthropic",
    model="claude-opus-4",  # a DIFFERENT Anthropic model: the thinking drops
)


def test_selection_created_adjacency_does_not_eat_plain_text(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    messages = [UserMessage(content="hi"), ADJACENCY_TRAP]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                TextBlock(text="Let me check the market."),
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[0]),
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[1]),
                PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[2]),
            ],
            provider="anthropic",
            model="claude-opus-4",
        ),
    ]
    assert transport.effective_messages(effective, _request(effective)) == effective
    assert transport._build_chat_completion_payload(_request(effective)) == transport._build_chat_completion_payload(
        request
    )


def test_duplicate_call_ids_do_not_clobber_a_kept_call(anthropic_transport_factory):
    # A kept-family call and a foreign native call sharing an id: the kept
    # call must serialize (per-block resolution) while the foreign one drops.
    transport = anthropic_transport_factory()
    message = AssistantMessage(
        content=[
            ToolCall(id="x", name="grep", arguments={"pattern": "bug"}),
            ShellToolCall(id="x", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1"),
        ],
        provider="anthropic",
        model="claude-sonnet-5",
    )

    projected = transport._project_assistant_message(message, _request([message]))

    assert projected == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "x", "name": "grep", "input": {"pattern": "bug"}}],
    }
