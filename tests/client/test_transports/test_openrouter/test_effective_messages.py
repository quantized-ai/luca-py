"""effective_messages through OpenRouter — inherited wholesale from
OpenAITransport: "openai.responses" privates are foreign on this wire even
for an OpenAI model (the D1 scenario the wire-format key exists for), web
blocks are never sent, and the payload invariant holds."""

from luca.client.transports.openrouter.transport import OpenRouterTransport
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    PrivateProviderBlock,
    TextBlock,
    UserMessage,
    WebSearchBlock,
)


def _transport() -> OpenRouterTransport:
    return OpenRouterTransport(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
    )


def _request(messages) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="openai/gpt-5.1",
        provider="openrouter",
        messages=messages,
    )


WEB_TURN = AssistantMessage(
    content=[
        PrivateProviderBlock(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
        WebSearchBlock(queries=["apple"]),
        TextBlock(text="Apple is up."),
    ],
    provider="openai",
    model="gpt-5.1",
)


def test_effective_messages_equivalence():
    transport = _transport()
    messages = [UserMessage(content="hi"), WEB_TURN]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport._build_chat_completion_payload(_request(effective)) == transport._build_chat_completion_payload(
        request
    )


def test_responses_format_privates_are_foreign_on_this_wire():
    # An OpenAI-minted session routed through OpenRouter goes over chat
    # completions, so its "openai.responses" privates must drop here.
    transport = _transport()
    messages = [UserMessage(content="hi"), WEB_TURN]

    effective = transport.effective_messages(messages, _request(messages))

    assert effective == [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[TextBlock(text="Apple is up.")],
            provider="openai",
            model="gpt-5.1",
        ),
    ]


def test_effective_is_idempotent():
    transport = _transport()
    messages = [UserMessage(content="hi"), WEB_TURN]
    request = _request(messages)

    effective = transport.effective_messages(messages, request)

    assert transport.effective_messages(effective, _request(effective)) == effective
