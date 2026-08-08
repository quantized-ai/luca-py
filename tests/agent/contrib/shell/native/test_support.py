"""The native-tool support table: every model id in
`specs/0010-native-tools-in-agent/model_notes.md`, the two `-pro` exceptions,
the transports that carry no native items, and id normalization."""

import pytest

from luca.agent.contrib.shell.native import supported_native_tools
from luca.agent.core import LLMConfig

APPLY_PATCH = "openai_apply_patch"
SHELL = "openai_shell"
TEXT_EDITOR = "anthropic_text_editor_20250728"
BASH = "anthropic_bash_20250124"

BOTH_OPENAI = frozenset({APPLY_PATCH, SHELL})
BOTH_ANTHROPIC = frozenset({TEXT_EDITOR, BASH})
NOTHING: frozenset[str] = frozenset()


# ── the OpenAI table ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6", BOTH_OPENAI),
        ("gpt-5.6-sol", BOTH_OPENAI),
        ("gpt-5.6-terra", BOTH_OPENAI),
        ("gpt-5.6-luna", BOTH_OPENAI),
        ("gpt-5.5", BOTH_OPENAI),
        ("gpt-5.5-pro", frozenset({SHELL})),
        ("gpt-5.4", BOTH_OPENAI),
        ("gpt-5.4-pro", frozenset({APPLY_PATCH})),
        ("gpt-5.4-mini", BOTH_OPENAI),
        ("gpt-5.4-nano", BOTH_OPENAI),
        ("gpt-5.2", BOTH_OPENAI),
        ("gpt-5.1", BOTH_OPENAI),
    ],
)
def test_openai_models_get_the_natives_their_model_page_documents(model, expected):
    assert supported_native_tools(LLMConfig(model=model, provider="openai")) == expected


# ── the Anthropic table ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1",
        "claude-opus-4-1-20250805",
    ],
)
def test_every_current_claude_model_gets_both_anthropic_natives(model):
    assert supported_native_tools(LLMConfig(model=model, provider="anthropic")) == BOTH_ANTHROPIC


# ── the two exceptions ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.5", BOTH_OPENAI),
        ("gpt-5.5-pro", frozenset({SHELL})),
        ("gpt-5.4", BOTH_OPENAI),
        ("gpt-5.4-pro", frozenset({APPLY_PATCH})),
    ],
    ids=["5.5-has-apply-patch", "5.5-pro-has-not", "5.4-has-shell", "5.4-pro-has-not"],
)
def test_a_pro_variant_does_not_inherit_its_base_models_support(model, expected):
    assert supported_native_tools(LLMConfig(model=model, provider="openai")) == expected


# ── the transports that carry no native items ────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openrouter", "openai/gpt-5.4-mini"),
        ("openrouter", "openai/gpt-5.6"),
        ("openrouter", "anthropic/claude-opus-5"),
        ("bedrock", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
        ("bedrock", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("groq", "moonshotai/kimi-k2-instruct"),
        ("deepseek", "deepseek-chat"),
        ("ollama", "llama3.3"),
        ("faux", "gpt-5.6"),
    ],
    ids=[
        "openrouter-gpt-mini",
        "openrouter-gpt",
        "openrouter-claude",
        "bedrock-claude-inference-profile",
        "bedrock-claude",
        "groq",
        "deepseek",
        "ollama",
        "faux",
    ],
)
def test_a_provider_whose_transport_has_no_native_items_gets_nothing(provider, model):
    # openrouter/groq/deepseek/ollama run chat completions, bedrock runs
    # Converse; the native items only exist on /v1/responses and Anthropic's
    # own wire, so the same model reached this way is generic-tools-only.
    assert supported_native_tools(LLMConfig(model=model, provider=provider)) == NOTHING


# ── unknown models fall back to the generic tools ────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-7"),
        ("openai", "o3-pro"),
        ("openai", "codex-mini-latest"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("anthropic", "claude-2.1"),
        ("openai", "my-finetune-of-gpt-5.4"),
    ],
)
def test_a_model_the_table_never_heard_of_gets_nothing(provider, model):
    assert supported_native_tools(LLMConfig(model=model, provider=provider)) == NOTHING


def test_an_unregistered_provider_gets_nothing():
    assert supported_native_tools(LLMConfig(model="gpt-5.6", provider="acme")) == NOTHING


# ── id normalization ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-sonnet-4-5", BOTH_ANTHROPIC),
        ("anthropic", "claude-sonnet-4-5-20250929", BOTH_ANTHROPIC),
        ("anthropic", "claude-haiku-4-5-latest", BOTH_ANTHROPIC),
        ("anthropic", "anthropic/claude-opus-5", BOTH_ANTHROPIC),
        ("anthropic", "Claude-Opus-5", BOTH_ANTHROPIC),
        ("openai", "gpt-5.1-2025-11-13", BOTH_OPENAI),
        ("openai", "GPT-5.4-Mini", BOTH_OPENAI),
        ("openai", "openai/gpt-5.4-pro", frozenset({APPLY_PATCH})),
    ],
    ids=[
        "alias-without-date",
        "dated-snapshot",
        "latest-alias",
        "vendor-prefix",
        "mixed-case",
        "openai-dated-snapshot",
        "openai-mixed-case",
        "vendor-prefix-keeps-the-exception",
    ],
)
def test_aliases_snapshots_prefixes_and_case_resolve_to_the_same_model(provider, model, expected):
    assert supported_native_tools(LLMConfig(model=model, provider=provider)) == expected


# ── the dotted version form: normalized for Claude ids, left alone elsewhere ──


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4.5",
        "claude-opus-4.8",
        "claude-opus-4.1",
        "claude-haiku-4.5-20251001",
        "anthropic/claude-sonnet-4.5",
        "Claude-Sonnet-4.5",
    ],
)
def test_a_dotted_claude_version_is_the_same_model_as_the_dashed_one(model):
    # The same rule the client's own resolver applies
    # (`transports/anthropic/capabilities.py:normalize_model_id`).
    assert supported_native_tools(LLMConfig(model=model, provider="anthropic")) == BOTH_ANTHROPIC


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4-mini", BOTH_OPENAI),
        ("gpt-5-4-mini", NOTHING),
        ("gpt-5.6", BOTH_OPENAI),
        ("gpt-5-6", NOTHING),
    ],
    ids=["dotted-is-the-id", "dashed-is-not", "dotted-is-the-id-2", "dashed-is-not-2"],
)
def test_an_openai_id_keeps_its_dots(model, expected):
    # OpenAI ids carry dots natively (`gpt-5.4-mini`), so the Claude rewrite
    # must not reach them — and the dashed spelling is simply a different id.
    assert supported_native_tools(LLMConfig(model=model, provider="openai")) == expected
