"""`usage.py` — token totals, estimated cost, and the 1k cost screen state.

Two session literals: `COSTED_SESSION` uses a model this module REGISTERS in
the catalog at $5/$25/$0.5/$6.25 per Mtok — every price binary-exact, so
fractions assert exactly — and `UNPRICED_SESSION` a model nothing prices. The
consumers list is asserted through the public `cost_state` only.

The price is registered rather than read from the shipped catalog on purpose:
these tests check the arithmetic, and pinning them to a real model would make
them fail the day models.dev repriced it. Pricing is keyed on
`(provider, model)`, so the registration names both.

The catalog knows no other `faux` models, so the context window is the 200k
default throughout.
"""

import pytest

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.usage import UsageTotals, cost_state, estimated_cost, status_counter, usage_totals
from luca.agent.core.models import (
    AssistantMessage,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnStart,
    Usage,
    UserMessage,
)
from tests.agent.scenarios import conversation, make_session, spec

PRICED_MODEL = LLMConfig(model="anthropic/claude-fable-5", provider="faux")
UNPRICED_MODEL = LLMConfig(model="mystery-9", provider="faux")


@pytest.fixture(autouse=True)
def _priced_model():
    from luca.client import catalog
    from luca.client.catalog import _store
    from luca.client.types import ModelCost, ModelInfo

    catalog.register(
        provider=PRICED_MODEL.provider,
        model=PRICED_MODEL.model,
        info=ModelInfo(
            cost=ModelCost(
                input_per_million_tokens=5.0,
                output_per_million_tokens=25.0,
                cached_input_per_million_tokens=0.5,
                cache_write_per_million_tokens=6.25,
            ),
        ),
    )
    yield
    _store._clear_for_tests()


# One answered tool turn plus an archived conversation `c0` that still holds a
# usage record — totals must sum across conversations AND entries.
COSTED_SESSION = make_session(
    id="s_costed",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=60_000,
            parts=[TextContent(text="audit the repo")],
            context_tokens=1_500,
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=61_000),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=90_000,
            parts=[
                TextContent(text="Let me look."),
                ToolCall(id="tc1", name="bash", arguments={"command": "ls -la"}),
            ],
            llm_config=PRICED_MODEL,
            stop_reason="tool_use",
            context_tokens=500,
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=90_000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="bash", arguments={"command": "ls -la"}),
            tool_spec=spec("bash"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="listing")]),
            context_tokens=800,
        ),
        "a2": AssistantMessage(
            id="a2",
            parent_id="te1",
            created_at=120_000,
            parts=[TextContent(text="All clean.")],
            llm_config=PRICED_MODEL,
            stop_reason="stop",
            context_tokens=100,
        ),
        "tf": TurnFinish(id="tf", parent_id="a2", created_at=180_000),
    },
    usages={
        "c0": {
            "a0": Usage(
                conversation_id="c0",
                entry_id="a0",
                input=100_000,
                output=10_000,
                cache_read=500_000,
                cache_write=50_000,
            ),
        },
        "c1": {
            "a1": Usage(
                conversation_id="c1",
                entry_id="a1",
                input=600_000,
                output=60_000,
                cache_read=3_000_000,
                cache_write=150_000,
            ),
            "a2": Usage(
                conversation_id="c1",
                entry_id="a2",
                input=300_000,
                output=30_000,
                cache_read=1_500_000,
                cache_write=200_000,
            ),
        },
    },
    conversations={
        "c0": conversation("c0"),
        "c1": conversation("c1", ["u1", "ts", "a1", "te1", "a2", "tf"], previous_conversation_id="c0"),
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=PRICED_MODEL),
)

UNPRICED_SESSION = make_session(
    id="s_unpriced",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=60_000,
            parts=[TextContent(text="hello")],
            context_tokens=1_000,
        ),
        "a1": AssistantMessage(
            id="a1",
            parent_id="u1",
            created_at=60_000,
            parts=[TextContent(text="Hi.")],
            llm_config=UNPRICED_MODEL,
            stop_reason="stop",
        ),
    },
    usages={
        "c1": {
            "a1": Usage(
                conversation_id="c1", entry_id="a1", input=4_000, output=1_000, cache_read=8_000, cache_write=2_000
            ),
        },
    },
    conversations={"c1": conversation("c1", ["u1", "a1"])},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=UNPRICED_MODEL),
)

EMPTY_SESSION = make_session(
    id="s_empty",
    conversations={"c1": conversation("c1")},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=UNPRICED_MODEL),
)


# ── totals and estimates ──────────────────────────────────────────────────────


def test_usage_totals_sums_every_conversation_and_entry():
    assert usage_totals(COSTED_SESSION) == UsageTotals(
        input=1_000_000,
        output=100_000,
        cache_read=5_000_000,
        cache_write=400_000,
    )


def test_usage_totals_total_spans_all_four_categories():
    assert usage_totals(COSTED_SESSION).total == 6_500_000


def test_estimated_cost_prices_a_known_model_through_its_route():
    # $5 + $2.50 + $2.50 (cache read) + $2.50 (cache write)
    assert estimated_cost(COSTED_SESSION) == 12.5


def test_estimated_cost_is_none_for_an_unlisted_model():
    assert estimated_cost(UNPRICED_SESSION) is None


# ── the status-bar counter ────────────────────────────────────────────────────


def test_status_counter_shows_context_tokens_and_estimated_cost():
    assert status_counter(COSTED_SESSION) == ("2.9k", "$12.50")


def test_status_counter_omits_cost_for_an_unlisted_model():
    assert status_counter(UNPRICED_SESSION) == ("1.0k", None)


def test_status_counter_is_empty_for_a_fresh_session():
    assert status_counter(EMPTY_SESSION) == (None, None)


# ── the cost screen (1k) ──────────────────────────────────────────────────────


def test_cost_state_for_a_priced_model_meters_by_cost():
    assert cost_state(COSTED_SESSION) == vm.CostState(
        headline="$12.50",
        subline="1 turn · 2m · claude-fable-5",
        items=[
            vm.CostItem(label="input", tokens="1000.0k", cost="$5.000", fraction=1.0, color="accent"),
            vm.CostItem(label="output", tokens="100.0k", cost="$2.500", fraction=0.5, color="foreground"),
            vm.CostItem(label="cache write", tokens="400.0k", cost="$2.500", fraction=0.5, color="faint"),
            vm.CostItem(label="cache read", tokens="5000.0k", cost="$2.500", fraction=0.5, color="rule"),
        ],
        context=vm.ContextWindowState(
            used="2.9k / 200.0k",
            percent="1%",
            context_fraction=0.0145,
            reply_fraction=0.0,
            legend=["[accent]▪[/] context 2.9k", "free 197.1k"],
        ),
        consumers=[
            vm.ConsumerRow(label="message · user", tokens="1.5k"),
            vm.ConsumerRow(label="output · bash ls -la", tokens="800"),
            vm.ConsumerRow(label="message · assistant", tokens="500"),
        ],
    )


def test_cost_state_for_an_unlisted_model_meters_by_tokens():
    assert cost_state(UNPRICED_SESSION) == vm.CostState(
        headline="15.0k tokens",
        subline="0 turns · 0s · mystery-9",
        items=[
            vm.CostItem(label="input", tokens="4.0k", cost="—", fraction=0.5, color="accent"),
            vm.CostItem(label="output", tokens="1.0k", cost="—", fraction=0.125, color="foreground"),
            vm.CostItem(label="cache write", tokens="2.0k", cost="—", fraction=0.25, color="faint"),
            vm.CostItem(label="cache read", tokens="8.0k", cost="—", fraction=1.0, color="rule"),
        ],
        context=vm.ContextWindowState(
            used="1.0k / 200.0k",
            percent="0%",
            context_fraction=0.005,
            reply_fraction=0.0,
            legend=["[accent]▪[/] context 1.0k", "free 199.0k"],
        ),
        consumers=[vm.ConsumerRow(label="message · user", tokens="1.0k")],
    )
