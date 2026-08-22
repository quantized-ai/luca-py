"""`WebSearchMiddleware`'s advertisement half: the unit declaration rules
(full-list equality over hand-built lists) and the end-to-end advertisement
battery over the mocked LLM boundary (kwargs capture — what `acompletion`
actually received as `tools=`)."""

import pytest

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.websearch import WebSearchMiddleware, WebSearchPlugin
from luca.agent.core import AgentSession, AgentSessionRunner, LLMConfig
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    TextBlock as ClientTextBlock,
    Tool as ClientTool,
)


def session(model: str, provider: str) -> AgentSession:
    return AgentSessionRunner.new_session(
        LLMConfig(model=model, provider=provider),
        session_id="s_web_mw",
        conversation_id="c1",
    )


READ_TOOL = ClientTool(name="read", description="Read a file.", parameters={"type": "object", "properties": {}})


# ── the unit declaration rules ───────────────────────────────────────────────


def test_an_openai_model_gets_the_single_openai_item():
    plugin = WebSearchPlugin({"openai": {"options": {"search_context_size": "high"}}})
    middleware = WebSearchMiddleware(session("gpt-5.1", "openai"), plugin.config)

    adapted = middleware.adapt_tool_declarations(middleware.session, "c1", [READ_TOOL])

    assert adapted == [READ_TOOL, plugin.config.openai.options]


def test_an_anthropic_model_gets_search_and_opt_in_fetch():
    without_fetch = WebSearchPlugin({"anthropic": {"search": {"max_uses": 5}}})
    with_fetch = WebSearchPlugin({"anthropic": {"search": {"max_uses": 5}, "fetch": {"max_content_tokens": 20_000}}})
    anthropic = session("claude-sonnet-5", "anthropic")

    search_only = WebSearchMiddleware(anthropic, without_fetch.config).adapt_tool_declarations(anthropic, "c1", [])
    both = WebSearchMiddleware(anthropic, with_fetch.config).adapt_tool_declarations(anthropic, "c1", [])

    assert search_only == [without_fetch.config.anthropic.search]
    assert both == [with_fetch.config.anthropic.search, with_fetch.config.anthropic.fetch]


def test_a_configured_fetch_is_withheld_where_only_search_is_supported():
    # claude-opus-5: in the search table, absent from the fetch table — the
    # per-tool intersection is exact, or the request 400s before HTTP
    plugin = WebSearchPlugin({"anthropic": {"fetch": {"max_content_tokens": 20_000}}})
    opus = session("claude-opus-5", "anthropic")

    adapted = WebSearchMiddleware(opus, plugin.config).adapt_tool_declarations(opus, "c1", [])

    assert adapted == [plugin.config.anthropic.search]


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("kimi-2.7", "openrouter"),
        ("openai/gpt-5.1", "openrouter"),
        ("gpt-4.1-nano", "openai"),
        ("claude-haiku-4-5", "anthropic"),
        ("fake-model", "faux"),
    ],
)
def test_an_unsupported_provider_or_model_gets_nothing(model, provider):
    plugin = WebSearchPlugin(None)
    unsupported = session(model, provider)

    adapted = WebSearchMiddleware(unsupported, plugin.config).adapt_tool_declarations(unsupported, "c1", [READ_TOOL])

    assert adapted == [READ_TOOL]


def test_a_disabled_config_appends_nothing():
    disabled_all = WebSearchPlugin({"enabled": False})
    disabled_provider = WebSearchPlugin({"openai": {"enabled": False}})
    openai = session("gpt-5.1", "openai")

    assert WebSearchMiddleware(openai, disabled_all.config).adapt_tool_declarations(openai, "c1", []) == []
    assert WebSearchMiddleware(openai, disabled_provider.config).adapt_tool_declarations(openai, "c1", []) == []


def test_the_gate_ignores_use_native_tools():
    # hosted web tools are orthogonal to the native shell/editor swap: the
    # flag being off must not silence the declarations
    plugin = WebSearchPlugin(None)
    openai = session("gpt-5.1", "openai")
    openai.use_native_tools = False
    openai.session_config.use_native_tools = False

    adapted = WebSearchMiddleware(openai, plugin.config).adapt_tool_declarations(openai, "c1", [])

    assert adapted == [plugin.config.openai.options]


# ── the advertisement battery (mocked LLM boundary, kwargs capture) ──────────


def _text(text: str) -> ClientAssistantMessage:
    return ClientAssistantMessage(content=[ClientTextBlock(text=text)], finish_reason="stop")


async def test_a_supported_model_advertises_the_hosted_tool(llm):
    plugin = WebSearchPlugin(None)
    runner = PluginAgentSessionRunner(session("gpt-5.1", "openai"), plugins=[plugin])
    runner.post_message("find apple results")
    llm.queue(_text("done"))

    await runner.run()

    assert llm.calls[0]["tools"] == [plugin.config.openai.options]


async def test_an_unsupported_model_advertises_nothing(llm):
    runner = PluginAgentSessionRunner(session("gpt-4.1-nano", "openai"), plugins=[WebSearchPlugin(None)])
    runner.post_message("find apple results")
    llm.queue(_text("done"))

    await runner.run()

    assert llm.calls[0]["tools"] is None


async def test_a_mid_session_model_flip_rederives_the_declarations(llm):
    # the tool set is re-derived before every call from the ACTIVE config,
    # which the drive re-stamps from the CONFIGURED one each iteration
    plugin = WebSearchPlugin({"anthropic": {"fetch": {"max_content_tokens": 20_000}}})
    runner = PluginAgentSessionRunner(session("gpt-5.1", "openai"), plugins=[plugin])
    runner.post_message("first")
    llm.queue(_text("one"))
    await runner.run()

    runner.session.session_config.llm_config = LLMConfig(model="claude-sonnet-5", provider="anthropic")
    runner.post_message("second")
    llm.queue(_text("two"))
    await runner.run()

    assert llm.calls[0]["tools"] == [plugin.config.openai.options]
    assert llm.calls[1]["tools"] == [plugin.config.anthropic.search, plugin.config.anthropic.fetch]
