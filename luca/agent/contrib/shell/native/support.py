"""Which (provider, model) pairs actually support each provider-native tool.

A hardcoded snapshot, taken 2026-08-07 from
`specs/0010-native-tools-in-agent/model_notes.md` and the OpenAI / Anthropic
model pages it cites. It answers CAPABILITY only — whether the native tool may
be advertised at all — never whether the application WANTS natives
(`AgentSession.use_native_tools` is the caller's to read).

Two facts shape the tables:

1. **Support is per MODEL, not per provider, and the exceptions are real.**
   `gpt-5.5-pro` has no `apply_patch` while `gpt-5.5` does; `gpt-5.4-pro` has
   no `shell` while `gpt-5.4` does. So the tables are sets of exact model ids
   and membership is exact — a prefix match would get both wrong.
2. **The native wire ITEMS only exist on two transports.** OpenAI's
   `apply_patch` / `shell` items are a `/v1/responses` feature
   (`OpenAIResponsesTransport`) and the Anthropic items belong to
   `AnthropicTransport`. Every other host reaches the same models over a
   different protocol whose projector rejects or drops those items:
   `openrouter` runs the chat-completions transport (so an OpenAI model via
   OpenRouter gets NO natives) and `bedrock` runs the Converse transport (so a
   Claude model on Bedrock gets none either). Hence a provider table, not a
   model-family one.

   The table is keyed on the PROVIDER NAME, and the two names below are the
   built-in ones that run those two transports. That is a proxy, not an
   identity: `luca/client/transports/__init__.py` publishes `openai-responses`
   in `TRANSPORTS`, and a `luca.json` can point a custom provider at it
   (`tui/config.py:register_config_providers`, necessarily under a name of its
   own — the first-class names are reserved). Such a host speaks the native
   wire and still gets nothing from this table. The failure is fail-safe: the
   natives are lost, never wrongly advertised, and the session downgrades to
   the generic tools. Registering the custom name here is the fix if it ever
   matters.

An id the tables have never heard of — a model released after this snapshot, a
self-hosted or `ollama` id — resolves to no natives. That is a downgrade to the
generic `read`/`write`/`edit`/`bash` tools, never a refusal to run, matching the
client's rule that the model catalog is metadata and never a gate.

**When a new model ships**, add its id to the sets below (dateless, lowercase,
and for a Claude id dashed rather than dotted — see `_base_model_id`) after
checking the vendor's model page for the specific tool; add a case to
`tests/agent/contrib/shell/native/test_support.py` in the same commit. Only the
CURRENT tool versions are here: the retired `text_editor_20250124` /
`bash_20241022` target models that no longer exist on the API, and those models
fall back to the generic tools like anything else.
This lives in the plugin because that is where the provider knowledge belongs;
if the client ever grows per-model capability metadata, move it there.
"""

from __future__ import annotations

import re

from luca.agent.core import LLMConfig

# Dateless, lowercase model ids. `gpt-5.6` and `gpt-5.6-sol` are the same
# model under two ids; both are listed because either can be configured.
OPENAI_APPLY_PATCH_MODELS = frozenset(
    {
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        # gpt-5.5-pro is NOT here: its model page says apply_patch is unsupported.
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.1",
    }
)

OPENAI_SHELL_MODELS = frozenset(
    {
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        # gpt-5.4-pro is NOT here: its model page says shell is unsupported.
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.1",
    }
)

# One set for both Anthropic tools. `text_editor_20250728` is documented for
# Claude 4 and later; `bash_20250124` for Claude Sonnet 3.7 and later — and
# every Claude model still active on the API is 4 or later, so the two
# statements pick out the same list. `claude-mythos-5` is gated to customers
# with access; an id nobody can call resolves to nothing anyway.
#
# Cross-checked 2026-08-08 against the vendored catalog
# (`luca/client/catalog/_data/models.json`), not only `model_notes.md`: every
# `provider == "anthropic"` record it carries is listed here, `claude-opus-4-1`
# (which the enumeration missed) included. The catalog holds no other live
# Claude model, and no Claude 3.x at all.
ANTHROPIC_TOOL_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    }
)

# provider → internal tool name → the models that support it. A provider absent
# from this table has no native wire items at all.
NATIVE_TOOL_MODELS: dict[str, dict[str, frozenset[str]]] = {
    "openai": {
        "openai_apply_patch": OPENAI_APPLY_PATCH_MODELS,
        "openai_shell": OPENAI_SHELL_MODELS,
    },
    "anthropic": {
        "anthropic_text_editor_20250728": ANTHROPIC_TOOL_MODELS,
        "anthropic_bash_20250124": ANTHROPIC_TOOL_MODELS,
    },
}

# The snapshot suffixes a vendor pins a release with: Anthropic's `-20250929`,
# OpenAI's `-2025-11-13`, and the `-latest` alias. Support is a property of the
# model, not of the snapshot, so they are stripped before lookup.
_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|latest)$")


def _base_model_id(model: str) -> str:
    """The form the tables are keyed on: lowercased, vendor prefix dropped
    (`anthropic/claude-opus-5` → `claude-opus-5`), snapshot suffix stripped
    (`claude-sonnet-4-5-20250929` → `claude-sonnet-4-5`), and — for a Claude id
    only — the dotted version rewritten dashed (`claude-sonnet-4.5` →
    `claude-sonnet-4-5`).

    The dot rewrite is the client's own rule, applied where the client applies
    it: `transports/anthropic/capabilities.py:normalize_model_id` rewrites
    every dot because it only ever sees Anthropic ids, and it recognizes one by
    the `claude` anchor (which also drops a gateway prefix like
    `us.anthropic.claude-…`). The anchor is what keeps the rewrite off OpenAI's
    ids, where a dot is part of the name — `gpt-5.4-mini` is the id and
    `gpt-5-4-mini` is nothing."""
    base = model.rsplit("/", 1)[-1].lower()
    anchor = base.find("claude")
    if anchor >= 0:
        base = base[anchor:].replace(".", "-")
    return _SNAPSHOT_SUFFIX.sub("", base)


def supported_native_tools(llm_config: LLMConfig) -> frozenset[str]:
    """The internal names of the provider-native tools this (provider, model)
    pair actually supports. Empty when the provider has no natives, or the
    model does not support any."""
    by_tool = NATIVE_TOOL_MODELS.get(llm_config.provider)
    if by_tool is None:
        return frozenset()
    model = _base_model_id(llm_config.model)
    return frozenset(name for name, models in by_tool.items() if model in models)
