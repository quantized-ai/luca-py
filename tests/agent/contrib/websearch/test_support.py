"""The hosted-web-tool support table: every listed model, the documented
exceptions, the providers whose transports carry no hosted web items, and id
normalization. Snapshot researched 2026-08-22 from the vendor model pages —
see `luca/agent/contrib/websearch/support.py`."""

import pytest

from luca.agent.contrib.websearch import supported_web_tools
from luca.agent.core import LLMConfig

SEARCH_ONLY = frozenset({"web_search"})
SEARCH_AND_FETCH = frozenset({"web_search", "web_fetch"})
NOTHING: frozenset[str] = frozenset()


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-5.6", SEARCH_ONLY),
        ("openai", "gpt-5.6-sol", SEARCH_ONLY),
        ("openai", "gpt-5.6-terra", SEARCH_ONLY),
        ("openai", "gpt-5.6-luna", SEARCH_ONLY),
        ("openai", "gpt-5.5", SEARCH_ONLY),
        ("openai", "gpt-5.5-pro", SEARCH_ONLY),
        ("openai", "gpt-5.4", SEARCH_ONLY),
        ("openai", "gpt-5.4-mini", SEARCH_ONLY),
        ("openai", "gpt-5.4-nano", SEARCH_ONLY),
        ("openai", "gpt-5.2", SEARCH_ONLY),
        ("openai", "gpt-5.1", SEARCH_ONLY),
        ("openai", "gpt-5", SEARCH_ONLY),
        ("openai", "gpt-5-mini", SEARCH_ONLY),
        ("openai", "gpt-5-pro", SEARCH_ONLY),
        ("openai", "o3", SEARCH_ONLY),
        ("openai", "o3-pro", SEARCH_ONLY),
        ("openai", "o4-mini", SEARCH_ONLY),
        ("openai", "gpt-4.1", SEARCH_ONLY),
        ("openai", "gpt-4.1-mini", SEARCH_ONLY),
        ("openai", "gpt-4o", SEARCH_ONLY),
        ("openai", "gpt-4o-mini", SEARCH_ONLY),
        ("anthropic", "claude-fable-5", SEARCH_AND_FETCH),
        ("anthropic", "claude-mythos-5", SEARCH_AND_FETCH),
        ("anthropic", "claude-sonnet-5", SEARCH_AND_FETCH),
        ("anthropic", "claude-opus-4-8", SEARCH_AND_FETCH),
        ("anthropic", "claude-opus-4-7", SEARCH_AND_FETCH),
        ("anthropic", "claude-opus-4-6", SEARCH_AND_FETCH),
        ("anthropic", "claude-sonnet-4-6", SEARCH_AND_FETCH),
        # the web-fetch page's printed enumeration omits opus-5 (docs lag or
        # not — the printed list wins until verified live)
        ("anthropic", "claude-opus-5", SEARCH_ONLY),
    ],
)
def test_every_listed_model_supports_its_tools(provider, model, expected):
    assert supported_web_tools(LLMConfig(model=model, provider=provider)) == expected


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # documented as unsupported / ambiguous on the vendor's own pages
        ("openai", "gpt-4.1-nano"),
        ("openai", "gpt-5-nano"),
        # below "Claude 4.6 and later": the current tool versions 400 without
        # the direct-only escape hatch this table cannot assume
        ("anthropic", "claude-haiku-4-5"),
        ("anthropic", "claude-sonnet-4-5"),
        ("anthropic", "claude-opus-4-5-20251101"),
        # a model the snapshot has never heard of
        ("openai", "gpt-7"),
        ("anthropic", "claude-8"),
    ],
)
def test_an_unknown_model_supports_nothing(provider, model):
    assert supported_web_tools(LLMConfig(model=model, provider=provider)) == NOTHING


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # the hosted wire items only exist on /v1/responses and Anthropic's
        # own wire; the same models over other transports get nothing
        ("openrouter", "openai/gpt-5.1"),
        ("openrouter", "anthropic/claude-sonnet-5"),
        ("bedrock", "us.anthropic.claude-sonnet-4-6-v1:0"),
        ("faux", "gpt-5.1"),
    ],
)
def test_a_provider_whose_transport_has_no_hosted_web_items_gets_nothing(provider, model):
    assert supported_web_tools(LLMConfig(model=model, provider=provider)) == NOTHING


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-5", SEARCH_AND_FETCH),  # vendor prefix drops
        ("claude-sonnet-4.6", SEARCH_AND_FETCH),  # dotted Claude id rewrites dashed
        ("claude-sonnet-4-6-20260125", SEARCH_AND_FETCH),  # snapshot suffix strips
        ("CLAUDE-SONNET-5", SEARCH_AND_FETCH),  # case-insensitive
    ],
)
def test_model_id_normalization(model, expected):
    assert supported_web_tools(LLMConfig(model=model, provider="anthropic")) == expected


def test_openai_ids_keep_their_dots():
    # `gpt-5.4` is the id and `gpt-5-4` is nothing: the dash rewrite is
    # anchored on `claude` and must never touch an OpenAI id
    assert supported_web_tools(LLMConfig(model="gpt-5-4", provider="openai")) == NOTHING
