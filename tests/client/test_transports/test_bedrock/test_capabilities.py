"""Per-model Bedrock facts and the reasoning resolver built on them.

Pure lookup and derivation — no transport, no HTTP. The Bedrock-specific part
is `normalize_model_id`: a region prefix and a version suffix that Anthropic's
own ids do not carry.
"""

import pytest

from luca.client.exceptions import UnsupportedParameterError
from luca.client.transports.bedrock.capabilities import (
    ModelCapabilities,
    check_sampling,
    get_model_capabilities,
    normalize_model_id,
    resolve_reasoning,
)

NOVA = get_model_capabilities("us.amazon.nova-lite-v1:0")
LLAMA = get_model_capabilities("us.meta.llama3-3-70b-instruct-v1:0")
ANTHROPIC_MANUAL = get_model_capabilities("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
UNKNOWN = get_model_capabilities("us.cohere.command-r-v1:0")


# ── id normalisation ─────────────────────────────────────────────────────────


def test_region_prefix_is_stripped_before_matching():
    assert normalize_model_id("us.amazon.nova-lite-v1:0") == "amazon.nova-lite-v1:0"


def test_every_region_prefix_and_a_bare_id_reach_the_same_model():
    assert normalize_model_id("eu.anthropic.claude-3-5-sonnet-20240620-v1:0") == (
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    assert normalize_model_id("apac.anthropic.claude-3-5-sonnet-20240620-v1:0") == (
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    assert normalize_model_id("anthropic.claude-3-5-sonnet-20240620-v1:0") == (
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )


def test_a_gateway_namespace_is_stripped_before_the_region_prefix():
    assert normalize_model_id("bedrock/us.meta.llama3-3-70b-instruct-v1:0") == (
        "meta.llama3-3-70b-instruct-v1:0"
    )


# ── the table ────────────────────────────────────────────────────────────────


def test_nova_is_known_with_no_reasoning_and_a_ten_thousand_ceiling():
    assert NOVA == ModelCapabilities(
        max_output_tokens=10_000, is_known_model=True,
    )


def test_llama_is_known_with_no_reasoning_and_an_eight_thousand_ceiling():
    assert LLAMA == ModelCapabilities(
        max_output_tokens=8_192, is_known_model=True,
    )


def test_a_dated_anthropic_profile_resolves_to_the_thinking_tier():
    assert ANTHROPIC_MANUAL == ModelCapabilities(
        max_output_tokens=64_000, supports_thinking=True, is_known_model=True,
    )


def test_an_unknown_model_is_conservative_and_flagged():
    # The default record is all-false with the smallest ceiling and
    # is_known_model=False — the whole object pins that.
    assert UNKNOWN == ModelCapabilities()


# ── the resolver ─────────────────────────────────────────────────────────────


def test_no_reasoning_requested_sends_nothing_and_keeps_the_known_ceiling():
    assert resolve_reasoning(None, NOVA, None) == ({}, 10_000)
    assert resolve_reasoning("provider-default", NOVA, None) == ({}, 10_000)


def test_a_non_thinking_model_never_emits_a_thinking_field():
    assert resolve_reasoning("high", NOVA, None) == ({}, 10_000)


def test_explicit_none_on_a_thinking_model_emits_no_field():
    assert resolve_reasoning("none", ANTHROPIC_MANUAL, None) == ({}, 64_000)


def test_a_manual_thinking_model_gets_a_percentage_budget():
    fields, max_tokens = resolve_reasoning("medium", ANTHROPIC_MANUAL, None)
    assert fields == {"thinking": {"type": "enabled", "budget_tokens": 19_200}}
    assert max_tokens == 20_224


def test_a_caller_max_tokens_shrinks_the_budget_to_fit():
    fields, max_tokens = resolve_reasoning("high", ANTHROPIC_MANUAL, 5_000)
    assert fields == {"thinking": {"type": "enabled", "budget_tokens": 3_976}}
    assert max_tokens == 5_000


def test_a_max_tokens_with_no_room_for_thinking_raises():
    with pytest.raises(UnsupportedParameterError):
        resolve_reasoning("high", ANTHROPIC_MANUAL, 1_500)


# ── sampling ─────────────────────────────────────────────────────────────────


def test_a_plain_model_accepts_sampling_with_thinking_off():
    assert check_sampling(NOVA, {}, temperature=0.2, top_p=None, top_k=None) is None


def test_sampling_is_refused_while_thinking_is_active():
    with pytest.raises(UnsupportedParameterError):
        check_sampling(
            ANTHROPIC_MANUAL,
            {"thinking": {"type": "enabled", "budget_tokens": 2000}},
            temperature=0.2, top_p=None, top_k=None,
        )
