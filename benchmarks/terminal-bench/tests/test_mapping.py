"""`mapping` — Harbor's model strings and trajectories in luca's terms.

No Docker, no harbor import, no benchmark run. If these pass, the two places
the adapter could silently get the wrong answer are covered.
"""

import pytest

from luca.client import catalog
from luca.client.types import ModelCost, ModelInfo
from luca_tb.mapping import UsageTotals, context_from_session, estimate_cost, parse_model

PRICED_PROVIDER = "faux"
PRICED_MODEL = "priced-9"


@pytest.fixture
def priced_model():
    """A model priced at $5/$25/$0.5/$6.25 per Mtok — every rate binary-exact,
    so the assertions below are exact rather than approximate.

    Registered rather than borrowed from the shipped catalog: this tests the
    arithmetic, and pinning it to a real model would break the day models.dev
    repriced it."""
    from luca.client.catalog import _store

    catalog.register(
        provider=PRICED_PROVIDER,
        model=PRICED_MODEL,
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


def session_json(usages: dict, provider: str = PRICED_PROVIDER, model: str = PRICED_MODEL) -> dict:
    return {
        "usages": usages,
        "session_config": {"llm_config": {"provider": provider, "model": model}},
    }


# ── parse_model ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # the head is a registered host, so it is the provider
        ("openrouter/openai/gpt-5.4-mini", ("openrouter", "openai/gpt-5.4-mini")),
        ("anthropic/claude-opus-4-5", ("anthropic", "claude-opus-4-5")),
        ("bedrock/anthropic.claude-v2", ("bedrock", "anthropic.claude-v2")),
        # `openai` IS a registered host, so it wins over the default
        ("openai/gpt-5.4-mini", ("openai", "gpt-5.4-mini")),
        # no slash at all, and an unknown head, both fall to the default
        ("some-local-model", ("openrouter", "some-local-model")),
        ("moonshotai/kimi-k2.7-code", ("openrouter", "moonshotai/kimi-k2.7-code")),
    ],
)
def test_the_leading_segment_is_a_provider_only_when_luca_knows_that_host(spec, expected):
    assert parse_model(spec) == expected


def test_the_default_provider_is_overridable():
    assert parse_model("mistral-large", default_provider="ollama") == ("ollama", "mistral-large")


def test_an_empty_model_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        parse_model("")


def test_a_bare_provider_with_no_model_is_rejected():
    with pytest.raises(ValueError, match="no model"):
        parse_model("anthropic/")


# ── context_from_session ─────────────────────────────────────────────────────


def test_usage_sums_across_conversations_and_entries(priced_model):
    session = session_json(
        {
            # an archived conversation still holds usage; it has to count
            "c0": {"a0": {"input": 100_000, "output": 10_000, "cache_read": 500_000, "cache_write": 50_000}},
            "c1": {
                "a1": {"input": 600_000, "output": 60_000, "cache_read": 3_000_000, "cache_write": 150_000},
                "a2": {"input": 300_000, "output": 30_000, "cache_read": 1_500_000, "cache_write": 200_000},
            },
        }
    )

    assert context_from_session(session) == UsageTotals(
        # 1M raw input + 5M cache read: harbor's n_input_tokens is inclusive
        n_input_tokens=6_000_000,
        n_output_tokens=100_000,
        n_cache_tokens=5_000_000,
        # $5 (input) + $2.50 (output) + $2.50 (cache read) + $2.50 (cache write)
        cost_usd=12.5,
    )


def test_an_unpriced_model_reports_tokens_but_no_cost():
    session = session_json({"c1": {"a1": {"input": 4_000, "output": 1_000}}}, model="nobody-prices-this")

    assert context_from_session(session) == UsageTotals(
        n_input_tokens=4_000,
        n_output_tokens=1_000,
        n_cache_tokens=0,
        cost_usd=None,
    )


def test_a_session_with_no_usage_records_is_all_zeroes(priced_model):
    assert context_from_session(session_json({})) == UsageTotals()


def test_a_trajectory_missing_the_fields_entirely_does_not_raise():
    # a session written by a different luca version must still report what it can
    assert context_from_session({}) == UsageTotals()


def test_partial_usage_records_treat_missing_categories_as_zero(priced_model):
    session = session_json({"c1": {"a1": {"input": 1_000_000}}})

    assert context_from_session(session) == UsageTotals(
        n_input_tokens=1_000_000,
        n_output_tokens=0,
        n_cache_tokens=0,
        cost_usd=5.0,
    )


# ── estimate_cost ────────────────────────────────────────────────────────────


def test_an_unknown_model_costs_none_rather_than_zero(priced_model):
    cost = estimate_cost(
        provider=PRICED_PROVIDER,
        model="mystery-9",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )

    assert cost is None


def test_a_priced_model_that_consumed_nothing_costs_none(priced_model):
    cost = estimate_cost(
        provider=PRICED_PROVIDER,
        model=PRICED_MODEL,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )

    assert cost is None


def test_a_missing_provider_or_model_costs_none():
    assert (
        estimate_cost(provider=None, model=PRICED_MODEL, input_tokens=1, output_tokens=0, cache_read=0, cache_write=0)
        is None
    )
    assert (
        estimate_cost(
            provider=PRICED_PROVIDER, model=None, input_tokens=1, output_tokens=0, cache_read=0, cache_write=0
        )
        is None
    )
