"""effective_messages resolution: provider=/transport= handling mirrors
acompletion(), the resolved provider is cached and reused, and the call is
entirely offline — no HTTP is ever touched."""

import httpx
import pytest

from luca.client import effective_messages, get_provider
from luca.client._client import _provider_cache
from luca.client.exceptions import ProviderNotFoundError
from luca.client.testing import FauxProvider, FauxTransport
from luca.client.types import (
    AssistantMessage,
    PrivateProviderBlock,
    TextBlock,
    UserMessage,
    WebSearchBlock,
)

MESSAGES = [
    UserMessage(content="find it"),
    AssistantMessage(
        content=[
            TextBlock(text="Looking."),
            PrivateProviderBlock(
                format="openai.responses",
                data={"id": "ws_1", "type": "web_search_call", "status": "completed"},
            ),
            WebSearchBlock(queries=["apple"]),
        ],
        provider="openai",
        model="gpt-5.1",
    ),
]


def test_a_prebuilt_transport_is_used_directly():
    # The faux transport's selection is identity, so playback through the
    # transport= path returns the list unchanged.
    assert effective_messages("faux:m", MESSAGES, transport=FauxTransport()) == MESSAGES


def test_a_prebuilt_provider_is_used_directly():
    assert effective_messages("faux:m", MESSAGES, provider=FauxProvider()) == MESSAGES


def test_an_unregistered_provider_without_a_transport_raises():
    with pytest.raises(ProviderNotFoundError):
        effective_messages("faux:m", MESSAGES)


def test_the_anthropic_view_drops_the_foreign_private_and_web_blocks():
    result = effective_messages("anthropic:claude-sonnet-5", MESSAGES)

    assert result == [
        UserMessage(content="find it"),
        AssistantMessage(
            content=[TextBlock(text="Looking.")],
            provider="openai",
            model="gpt-5.1",
        ),
    ]


def test_dict_shaped_messages_are_coerced_like_a_completion_call():
    result = effective_messages(
        "anthropic:claude-sonnet-5",
        [{"role": "user", "content": "hi"}],
    )

    assert result == [UserMessage(content="hi")]


def test_the_resolved_provider_is_cached_and_reused(monkeypatch):
    # Prime the cache entry effective_messages resolves to, then make any
    # fresh resolution fail: only the cached instance can serve the calls.
    get_provider("anthropic:claude-sonnet-5")
    assert len(_provider_cache) > 0

    def boom(*args, **kwargs):
        raise AssertionError("a cached target must not be re-resolved")

    monkeypatch.setattr("luca.client._client.resolve_provider", boom)

    result = effective_messages("anthropic:claude-sonnet-5", [UserMessage(content="hi")])

    assert result == [UserMessage(content="hi")]


def test_no_http_is_ever_touched(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("effective_messages must never touch HTTP")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)

    result = effective_messages("anthropic:claude-sonnet-5", MESSAGES)

    assert result == [
        UserMessage(content="find it"),
        AssistantMessage(
            content=[TextBlock(text="Looking.")],
            provider="openai",
            model="gpt-5.1",
        ),
    ]
