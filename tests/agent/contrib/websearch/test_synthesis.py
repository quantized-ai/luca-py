"""`WebSearchMiddleware`'s synthesis half: the D9 templates as full-object
asserts (unit), the D9 wording table, and the runner integration over
faux-scripted web blocks — path order, event silence, spec normalization."""

import pytest

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.websearch import WEB_FETCH_SPEC, WEB_SEARCH_SPEC, WebSearchMiddleware, WebSearchPlugin
from luca.agent.core import (
    AgentSessionRunner,
    AssistantMessage,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    URLCitation,
    WebFetchContent,
    WebPageContent,
    WebSearchContent,
)
from luca.agent.core.events import FinishReason, TextBlock as TextBlockEvent, WebSearchBlock as WebSearchBlockEvent
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.types import (
    PrivateProviderBlock as ClientPrivateProviderBlock,
    WebPagePart as ClientWebPagePart,
    WebSearchBlock as ClientWebSearchBlock,
)

MODEL = LLMConfig(model="fake-model", provider="faux")


def entry_with(parts: list) -> AssistantMessage:
    return AssistantMessage(
        id="am1",
        created_at=1000,
        parts=parts,
        llm_config=MODEL,
        stop_reason="stop",
    )


def middleware() -> WebSearchMiddleware:
    plugin = WebSearchPlugin(None)
    session = AgentSessionRunner.new_session(MODEL, session_id="s_syn", conversation_id="c1")
    return WebSearchMiddleware(session, plugin.config)


class DeterministicPluginRunner(PluginAgentSessionRunner):
    """The plugin runner with the id/clock hooks scripted, like
    `DeterministicRunner`."""

    def __init__(self, *args, ids=(), now=1000, **kwargs):
        self._ids = iter(ids)
        self._now = now
        super().__init__(*args, **kwargs)

    def generate_id(self) -> str:
        return next(self._ids)

    def now_ms(self) -> int:
        return self._now


# ── the D9 templates, unit ───────────────────────────────────────────────────


def test_a_completed_search_mints_the_d9_execution():
    part = WebSearchContent(
        queries=["Apple stock price today"],
        results=[
            WebPageContent(url="https://cnn.com/markets", title="Markets"),
            WebPageContent(url="https://www.tradingview.com/apple", title="Apple"),
        ],
        extras={"id": "srvtoolu_01"},
    )

    templates = middleware().synthesize_executions(None, "c1", entry_with([part]))

    assert templates == [
        ToolExecution(
            tool_call_id="srvtoolu_01",
            raw_tool_call=ToolCall(
                id="srvtoolu_01",
                name="web_search",
                arguments={"queries": ["Apple stock price today"]},
            ),
            tool_spec=WEB_SEARCH_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[
                    TextContent(text='Web search: "Apple stock price today" — 2 results (cnn.com, tradingview.com)')
                ],
                structured_content=part.model_dump(),
            ),
            extras={"websearch": {"message_entry_id": "am1"}},
        )
    ]


def test_a_completed_fetch_mints_its_twin():
    part = WebFetchContent(
        web_page=WebPageContent(url="https://apple.com/report", title="Apple report", content="x" * 18_203),
        extras={"id": "srvtoolu_02"},
    )

    templates = middleware().synthesize_executions(None, "c1", entry_with([part]))

    assert templates == [
        ToolExecution(
            tool_call_id="srvtoolu_02",
            raw_tool_call=ToolCall(
                id="srvtoolu_02",
                name="web_fetch",
                arguments={"url": "https://apple.com/report"},
            ),
            tool_spec=WEB_FETCH_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text='Web fetch: https://apple.com/report ("Apple report", 18,203 chars)')],
                structured_content=part.model_dump(),
            ),
            extras={"websearch": {"message_entry_id": "am1"}},
        )
    ]


def test_a_failed_fetch_maps_to_failed():
    part = WebFetchContent(
        web_page=WebPageContent(url="https://apple.com/x"),
        extras={
            "id": "srvtoolu_03",
            "error": {"type": "web_fetch_tool_result_error", "error_code": "url_not_in_prior_context"},
        },
    )

    templates = middleware().synthesize_executions(None, "c1", entry_with([part]))

    assert templates == [
        ToolExecution(
            tool_call_id="srvtoolu_03",
            raw_tool_call=ToolCall(id="srvtoolu_03", name="web_fetch", arguments={"url": "https://apple.com/x"}),
            tool_spec=WEB_FETCH_SPEC,
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="web_operation_error",
                error_message="url_not_in_prior_context",
                details={
                    "provider_error": {
                        "type": "web_fetch_tool_result_error",
                        "error_code": "url_not_in_prior_context",
                    }
                },
            ),
            extras={"websearch": {"message_entry_id": "am1"}},
        )
    ]


def test_a_missing_op_id_leaves_the_correlation_fields_empty():
    # no client-stamped id: the template carries the EMPTY correlation fields
    # and the runner's minting loop backfills one generated id onto both
    # (pinned in test_runner_middleware.py)
    part = WebSearchContent(queries=["apple"], results=None)

    [template] = middleware().synthesize_executions(None, "c1", entry_with([part]))

    assert (template.tool_call_id, template.raw_tool_call.id) == ("", "")


def test_a_provider_error_maps_to_failed():
    # both providers' stamped shapes: Anthropic's wire error dict carries a
    # code; OpenAI's failed item carries only the status string
    anthropic_part = WebSearchContent(
        queries=["apple"],
        results=None,
        extras={
            "id": "srvtoolu_01",
            "error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
        },
    )
    openai_part = WebSearchContent(
        queries=["apple"],
        results=None,
        extras={"id": "ws_1", "error": {"status": "failed"}},
    )

    templates = middleware().synthesize_executions(None, "c1", entry_with([anthropic_part, openai_part]))

    assert templates == [
        ToolExecution(
            tool_call_id="srvtoolu_01",
            raw_tool_call=ToolCall(id="srvtoolu_01", name="web_search", arguments={"queries": ["apple"]}),
            tool_spec=WEB_SEARCH_SPEC,
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="web_operation_error",
                error_message="max_uses_exceeded",
                details={"provider_error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
            ),
            extras={"websearch": {"message_entry_id": "am1"}},
        ),
        ToolExecution(
            tool_call_id="ws_1",
            raw_tool_call=ToolCall(id="ws_1", name="web_search", arguments={"queries": ["apple"]}),
            tool_spec=WEB_SEARCH_SPEC,
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="web_operation_error",
                error_message="failed",
                details={"provider_error": {"status": "failed"}},
            ),
            extras={"websearch": {"message_entry_id": "am1"}},
        ),
    ]


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        (
            WebSearchContent(
                queries=["Apple stock price today"],
                results=[
                    WebPageContent(url="https://cnn.com/a"),
                    WebPageContent(url="https://www.tradingview.com/b"),
                    WebPageContent(url="https://coinbase.com/c"),
                    WebPageContent(url="https://cnbc.com/d"),
                    WebPageContent(url="https://finance.yahoo.com/e"),
                    WebPageContent(url="https://f.com"),
                    WebPageContent(url="https://g.com"),
                    WebPageContent(url="https://h.com"),
                    WebPageContent(url="https://i.com"),
                ],
            ),
            'Web search: "Apple stock price today" — 9 results '
            "(cnn.com, tradingview.com, coinbase.com, cnbc.com, finance.yahoo.com, +4 more)",
        ),
        (
            WebSearchContent(queries=["Apple stock price today"], results=None),
            'Web search: "Apple stock price today" (result metadata not returned)',
        ),
        (
            WebSearchContent(queries=["Apple stock price today"], results=[]),
            'Web search: "Apple stock price today" — no results',
        ),
        (
            WebSearchContent(
                queries=["apple"],
                extras={"error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
            ),
            'Web search: "apple" — failed: max_uses_exceeded',
        ),
        (
            WebFetchContent(
                web_page=WebPageContent(url="https://apple.com/report", title="Apple report", content="x" * 18_203),
            ),
            'Web fetch: https://apple.com/report ("Apple report", 18,203 chars)',
        ),
    ],
    ids=["search-results", "search-no-metadata", "search-empty", "search-failed", "fetch"],
)
def test_summaries_match_the_d9_wording_table(part, expected):
    assert middleware()._summary(part) == expected


# ── runner integration (faux-scripted web blocks) ────────────────────────────

SEARCH_BLOCK = ClientWebSearchBlock(
    queries=["apple"],
    results=[ClientWebPagePart(url="https://apple.com", title="Apple")],
    extras={"id": "srv_1"},
)
PRIVATE_BLOCK = ClientPrivateProviderBlock(
    format="anthropic.messages",
    data={"type": "server_tool_use", "id": "srv_1"},
)


def web_session():
    return AgentSessionRunner.new_session(MODEL, session_id="s_web", conversation_id="c1")


async def test_path_order_is_assistant_then_synthetics_then_births():
    from luca.client.testing import faux_tool_call
    from tests.agent.scenarios import AddTool, FakeToolRegistry

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [PRIVATE_BLOCK, SEARCH_BLOCK, faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Apple is up.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicPluginRunner(
        web_session(),
        plugins=[WebSearchPlugin(None)],
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "s1", "te1", "a2", "tf"],
        now=1000,
    )
    runner.post_message("find apple")

    await runner.run()

    # assistant → its synthetic → the RECEIVED birth of its tool call
    assert runner.session.conversations["c1"].nodes == ["u1", "ts", "a1", "s1", "te1", "a2", "tf"]
    assert runner.session.entries["te1"].raw_tool_call.name == "add"
    assert runner.session.entries["s1"] == ToolExecution(
        id="s1",
        parent_id="a1",
        created_at=1000,
        conversation_id="c1",
        tool_call_id="srv_1",
        raw_tool_call=ToolCall(id="srv_1", name="web_search", arguments={"queries": ["apple"]}),
        tool_spec=runner.session.tool_specs[WEB_SEARCH_SPEC.spec_id()],
        tool_spec_id=WEB_SEARCH_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text='Web search: "apple" — 1 result (apple.com)')],
            structured_content=WebSearchContent(
                queries=["apple"],
                results=[WebPageContent(url="https://apple.com", title="Apple")],
                extras={"id": "srv_1"},
            ).model_dump(),
        ),
        extras={"websearch": {"message_entry_id": "a1"}},
        finished_at=1000,
        context_tokens=10,  # len('Web search: "apple" — 1 result (apple.com)') // 4
    )


async def test_synthetics_fire_no_lifecycle_events_end_to_end():
    # BOTH always-tier block events fire from the recorded parts, no tool
    # lifecycle events fire for the synthetics, and the provider's hosted-tool
    # usage counters land in the session's usage record
    from luca.agent.core.events import WebFetchBlock as WebFetchBlockEvent
    from luca.agent.core.models import Usage
    from luca.client.types import Usage as ClientUsage, WebFetchBlock as ClientWebFetchBlock

    fetch_block = ClientWebFetchBlock(
        web_page=ClientWebPagePart(url="https://apple.com/report", title="Report"),
        extras={"id": "srv_2"},
    )
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [PRIVATE_BLOCK, SEARCH_BLOCK, fetch_block, faux_text("Apple is up.")],
                finish_reason="stop",
                usage=ClientUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    tool_requests={"web_search": 1, "web_fetch": 1},
                ),
            ),
        ]
    )
    runner = DeterministicPluginRunner(
        web_session(),
        plugins=[WebSearchPlugin(None)],
        provider=faux,
        ids=["u1", "ts", "a1", "s1", "s2", "tf"],
        now=1000,
    )
    runner.post_message("find apple")

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        WebSearchBlockEvent(
            conversation_id="c1",
            id="srv_1",
            queries=["apple"],
            results=[WebPageContent(url="https://apple.com", title="Apple")],
        ),
        WebFetchBlockEvent(
            conversation_id="c1",
            id="srv_2",
            web_page=WebPageContent(url="https://apple.com/report", title="Report"),
        ),
        TextBlockEvent(conversation_id="c1", text="Apple is up."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    # the hosted-tool counters rode the runner's usage recording end to end
    assert runner.session.usages["c1"]["a1"] == Usage(
        conversation_id="c1",
        entry_id="a1",
        input=10,
        output=5,
        total_tokens=15,
        tool_requests={"web_search": 1, "web_fetch": 1},
    )


async def test_the_spec_row_is_filed_once_across_turns():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([SEARCH_BLOCK, faux_text("one")], finish_reason="stop"),
            faux_assistant_message([SEARCH_BLOCK, faux_text("two")], finish_reason="stop"),
        ]
    )
    runner = DeterministicPluginRunner(
        web_session(),
        plugins=[WebSearchPlugin(None)],
        provider=faux,
        ids=["u1", "ts1", "a1", "s1", "tf1", "u2", "ts2", "a2", "s2", "tf2"],
        now=1000,
    )
    runner.post_message("first")
    await runner.run()
    runner.post_message("second")
    await runner.run()

    assert runner.session.tool_specs == {WEB_SEARCH_SPEC.spec_id(): WEB_SEARCH_SPEC}
    assert runner.session.entries["s1"].tool_spec is runner.session.entries["s2"].tool_spec


async def test_a_citation_annotation_survives_to_the_session():
    from luca.client.types import URLCitationAnnotation

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_text(
                        "Apple is up.",
                        annotations=[
                            URLCitationAnnotation(url="https://apple.com", title="Apple", start_index=0, end_index=12)
                        ],
                    )
                ],
                finish_reason="stop",
            ),
        ]
    )
    runner = DeterministicPluginRunner(
        web_session(),
        plugins=[WebSearchPlugin(None)],
        provider=faux,
        ids=["u1", "ts", "a1", "tf"],
        now=1000,
    )
    runner.post_message("find apple")

    await runner.run()

    assert runner.session.entries["a1"].parts == [
        TextContent(
            text="Apple is up.",
            annotations=[URLCitation(url="https://apple.com", title="Apple", start_index=0, end_index=12)],
        )
    ]


async def test_a_web_bearing_pause_still_resumes():
    # THE FLAGSHIP D8 scenario: the paused response carries web content, so
    # its synthetics land on the path AFTER the paused entry — the Resumed
    # detection's synthetic tolerance is what keeps the pair firing (a
    # "paused entry is the last entry" predicate would never match this)
    from luca.agent.core.events import ResponsePaused, ResponseResumed

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [PRIVATE_BLOCK, SEARCH_BLOCK, faux_text("Searching...")],
                finish_reason="pause",
            ),
            faux_assistant_message([faux_text("Apple is up.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicPluginRunner(
        web_session(),
        plugins=[WebSearchPlugin(None)],
        provider=faux,
        ids=["u1", "ts", "a1", "s1", "a2", "tf"],
        now=1000,
    )
    runner.post_message("find apple")

    async with runner.run() as run:
        events = [event async for event in run]

    assert ResponsePaused(conversation_id="c1", finish_reason="pause", provider_finish_reason="pause") in events
    assert ResponseResumed(conversation_id="c1") in events
    # the synthetic sits between the paused entry and the continuation, and
    # the replayed request still ends with the paused assistant content
    assert runner.session.conversations["c1"].nodes == ["u1", "ts", "a1", "s1", "a2", "tf"]
    assert len(faux.requests) == 2
    assert faux.requests[1].messages[-1].role == "assistant"
    assert faux.requests[1].messages[-1].content[-1].text == "Searching..."
