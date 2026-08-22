"""The D7 config surface: the three accepted input shapes, no shared mutable
state between instances, and `extra="forbid"` down to the client tool classes
(the ClassVar `name` is not a field, so a stray `name` key fails loudly)."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.websearch import AnthropicWebConfig, OpenAIWebConfig, WebSearchConfig, WebSearchPlugin
from luca.client.providers.anthropic import (
    WebFetchTool as AnthropicWebFetchTool,
    WebSearchTool as AnthropicWebSearchTool,
)
from luca.client.providers.openai import WebSearchTool as OpenAIWebSearchTool


def test_none_dict_and_typed_config_all_validate():
    from_none = WebSearchPlugin(None)
    from_dict = WebSearchPlugin(
        {
            "openai": {"options": {"search_context_size": "high"}},
            "anthropic": {"search": {"max_uses": 5}, "fetch": {"max_content_tokens": 20_000}},
        }
    )
    typed = WebSearchConfig(
        enabled=True,
        openai=OpenAIWebConfig(options=OpenAIWebSearchTool(search_context_size="high")),
        anthropic=AnthropicWebConfig(
            search=AnthropicWebSearchTool(max_uses=5),
            fetch=AnthropicWebFetchTool(max_content_tokens=20_000),
        ),
    )
    from_typed = WebSearchPlugin(typed)

    assert from_none.config == WebSearchConfig()
    assert from_dict.config == typed
    assert from_typed.config == typed


def test_two_configs_share_no_mutable_state():
    # instance defaults on the config fields: pydantic deep-copies them per
    # model, so one plugin's mutation can never leak into another's
    first = WebSearchPlugin(None)
    second = WebSearchPlugin(None)

    first.config.anthropic.search.max_uses = 99

    assert second.config.anthropic.search.max_uses is None
    assert first.config.anthropic.search is not second.config.anthropic.search


def test_a_stray_name_key_is_rejected():
    # `name` is a ClassVar on the client tool classes, not a field —
    # `extra="forbid"` refuses it instead of silently renaming a hosted tool
    with pytest.raises(ValidationError):
        WebSearchPlugin({"anthropic": {"search": {"name": "my_search"}}})
    with pytest.raises(ValidationError):
        WebSearchPlugin({"openai": {"options": {"name": "my_search"}}})
