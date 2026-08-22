"""Which (provider, model) pairs support each provider-hosted web tool.

A hardcoded snapshot, taken 2026-08-22 from the vendors' own documentation:
the per-model pages under https://developers.openai.com/api/docs/models/<id>
(each lists the hosted tools it supports) plus the web-search guide, and
https://platform.claude.com/docs/en/agents-and-tools/tool-use/ (the
web-search / web-fetch tool pages, the tool reference, and the model
deprecations page). It answers CAPABILITY only — whether the hosted tool may
be advertised at all — never whether the application WANTS web search
(`WebSearchConfig` is the caller's).

The same two facts that shape the shell-native tables shape these:

1. **Support is per MODEL, and the exceptions are real.** `gpt-4.1` has the
   hosted web_search and `gpt-4.1-nano` does not; `gpt-5-nano`'s own page is
   ambiguous (web_search under "Supported tools" but missing from
   "Features", and the guide historically named it unsupported) — excluded
   until verified live. So membership is exact ids, never a prefix match.
2. **The hosted wire items only exist on two transports.** OpenAI's
   `web_search_call` is a `/v1/responses` item and Anthropic's
   `server_tool_use` belongs to `AnthropicTransport`; OpenRouter (chat
   completions) and Bedrock (Converse — where Anthropic's docs say web
   search is NOT available at all) reach the same models over wires with no
   such items. Hence a table keyed on the two built-in PROVIDER NAMES —
   fail-safe for a custom host: the tools are lost, never wrongly advertised.

Anthropic's docs stopped publishing a flat per-model list; the binding
wording (retrieved 2026-08-22) is "Dynamic filtering is available with
Claude 4.6 and later models", and the CURRENT tool versions our client
declares (`web_search_20260318` / `web_fetch_20260318`) return a 400 on
earlier models unless `allowed_callers: ["direct"]` is set — a knob this
table cannot assume. So the sets hold exactly the "4.6 and later" family and
the below-4.6 actives (haiku-4-5, sonnet-4-5, opus-4-5) are excluded
fail-safe. The web-fetch page's dynamic-filtering enumeration omits
`claude-opus-5` while the web-search page's own examples use it — read as
docs lag, but the printed enumeration wins until verified live, so opus-5 is
in the SEARCH set and not the FETCH set.

An id the tables have never heard of resolves to no web tools — the agent
simply has no web capability there, matching the client's rule that model
metadata is never a gate. **When a new model ships**, add its id (dateless,
lowercase, Claude ids dashed — see `_base_model_id`) after checking the
vendor's page, and a case to
`tests/agent/contrib/websearch/test_support.py` in the same commit.
"""

from __future__ import annotations

import re

from luca.agent.core import LLMConfig

# Dateless, lowercase model ids. `gpt-5.6` and `gpt-5.6-sol` are the same
# model under two ids; both are listed because either can be configured.
OPENAI_WEB_SEARCH_MODELS = frozenset(
    {
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        # gpt-5-nano is NOT here: its page is self-contradictory (see module
        # docstring) and the guide historically named it unsupported.
        "gpt-5-pro",
        "o3",
        "o3-pro",
        "o4-mini",
        "gpt-4.1",
        # gpt-4.1-nano is NOT here: its model page omits web_search.
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    }
)

# "Claude 4.6 and later" (the docs' wording for the current tool versions),
# resolved against the active-model list on the deprecations page.
ANTHROPIC_WEB_SEARCH_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)

# The web-fetch page's own enumeration — search's family minus
# `claude-opus-5` (absent from the printed list; see module docstring).
ANTHROPIC_WEB_FETCH_MODELS = ANTHROPIC_WEB_SEARCH_MODELS - {"claude-opus-5"}

# provider → hosted tool name → the models that support it. A provider absent
# from this table has no hosted web tools at all.
WEB_TOOL_MODELS: dict[str, dict[str, frozenset[str]]] = {
    "openai": {
        "web_search": OPENAI_WEB_SEARCH_MODELS,
    },
    "anthropic": {
        "web_search": ANTHROPIC_WEB_SEARCH_MODELS,
        "web_fetch": ANTHROPIC_WEB_FETCH_MODELS,
    },
}

# The snapshot suffixes a vendor pins a release with: Anthropic's `-20250929`,
# OpenAI's `-2025-11-13`, and the `-latest` alias. Support is a property of the
# model, not of the snapshot, so they are stripped before lookup.
_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|latest)$")


def _base_model_id(model: str) -> str:
    """The form the tables are keyed on: lowercased, vendor prefix dropped
    (`anthropic/claude-opus-5` → `claude-opus-5`), snapshot suffix stripped
    (`claude-sonnet-4-6-20260125` → `claude-sonnet-4-6`), and — for a Claude
    id only — the dotted version rewritten dashed (`claude-sonnet-4.6` →
    `claude-sonnet-4-6`).

    Duplicated from `contrib/shell/native/support.py` (~20 lines): the
    shell's copy is module-private and a cross-contrib underscore import
    smells worse than the drift risk. The dot rewrite is the client's own
    rule (`transports/anthropic/capabilities.py:normalize_model_id`),
    anchored on `claude` so it never touches an OpenAI id, where the dot is
    part of the name — `gpt-5.4` is the id and `gpt-5-4` is nothing."""
    base = model.rsplit("/", 1)[-1].lower()
    anchor = base.find("claude")
    if anchor >= 0:
        base = base[anchor:].replace(".", "-")
    return _SNAPSHOT_SUFFIX.sub("", base)


def supported_web_tools(llm_config: LLMConfig) -> frozenset[str]:
    """The hosted web tools this (provider, model) pair actually supports —
    a subset of `{"web_search", "web_fetch"}`. Empty when the provider has no
    hosted web tools, or the model supports none."""
    by_tool = WEB_TOOL_MODELS.get(llm_config.provider)
    if by_tool is None:
        return frozenset()
    model = _base_model_id(llm_config.model)
    return frozenset(name for name, models in by_tool.items() if model in models)
