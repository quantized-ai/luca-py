"""Per-model Anthropic facts and the reasoning resolver built on them.

Pure lookup and derivation — no transport, no HTTP. A KNOWN model id in, a
KNOWN capability record out; a level plus a record in, a KNOWN thinking
payload out.
"""

import pytest

from luca.client.exceptions import UnsupportedParameterError
from luca.client.transports.anthropic.capabilities import (
    ModelCapabilities,
    check_sampling,
    get_model_capabilities,
    normalize_model_id,
    resolve_reasoning,
)

ADAPTIVE = get_model_capabilities("claude-sonnet-5")
ADAPTIVE_NO_XHIGH = get_model_capabilities("claude-sonnet-4-6")
MANUAL = get_model_capabilities("claude-haiku-4-5")
NO_THINKING = get_model_capabilities("claude-3-5-haiku-latest")
UNKNOWN = get_model_capabilities("some-future-model")


# ── the table ──────────────────────────────────────────────────────────────────


def test_the_newest_models_are_adaptive_and_refuse_sampling():
    assert ADAPTIVE == ModelCapabilities(
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_adaptive_thinking=True,
        supports_xhigh_effort=True,
        rejects_sampling_parameters=True,
        is_known_model=True,
    )


def test_the_4_6_generation_is_adaptive_without_xhigh_or_sampling_refusal():
    assert ADAPTIVE_NO_XHIGH == ModelCapabilities(
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_adaptive_thinking=True,
        is_known_model=True,
    )


def test_the_4_5_generation_thinks_but_only_manually():
    assert MANUAL == ModelCapabilities(
        max_output_tokens=64_000, supports_thinking=True, is_known_model=True,
    )


def test_the_3_x_models_are_known_but_cannot_think():
    assert NO_THINKING == ModelCapabilities(
        max_output_tokens=4_096, is_known_model=True,
    )


def test_an_unknown_model_gets_the_conservative_record():
    # a model we have never seen is as likely to be old as new, and sending
    # the wrong thinking shape is a hard 400
    assert UNKNOWN == ModelCapabilities()


def test_a_specific_id_is_not_swallowed_by_a_broader_one():
    # ordering in the table is load-bearing
    assert get_model_capabilities("claude-sonnet-4-5").max_output_tokens == 64_000
    assert get_model_capabilities("claude-sonnet-4-6").max_output_tokens == 128_000


# ── id normalization ───────────────────────────────────────────────────────────


def test_normalization_handles_every_id_shape_we_have_seen():
    assert normalize_model_id("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"
    assert normalize_model_id("claude-haiku-4-5-latest") == "claude-haiku-4-5-latest"
    assert normalize_model_id("us.anthropic.claude-haiku-4-5") == "claude-haiku-4-5"
    assert normalize_model_id("anthropic/claude-sonnet-4.5") == "claude-sonnet-4-5"
    assert normalize_model_id("claude-3.5-sonnet") == "claude-3-5-sonnet"


def test_every_id_shape_resolves_to_the_same_capabilities():
    for model in (
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5-latest",
        "us.anthropic.claude-haiku-4-5",
        "anthropic/claude-haiku-4.5",
    ):
        assert get_model_capabilities(model) == MANUAL, model


# ── resolver ───────────────────────────────────────────────────────────────────


def test_provider_default_and_none_requested_send_no_thinking_key():
    # "provider-default" means let the provider decide, which is the absence
    # of the key — distinct from "none", an explicit request for no thinking
    assert resolve_reasoning(None, ADAPTIVE, None) == ({}, 128_000)
    assert resolve_reasoning("provider-default", ADAPTIVE, None) == ({}, 128_000)


def test_none_disables_thinking_explicitly():
    payload, _ = resolve_reasoning("none", ADAPTIVE, None)

    assert payload == {"thinking": {"type": "disabled"}}


def test_a_model_that_cannot_think_is_sent_disabled_whatever_was_asked():
    payload, _ = resolve_reasoning("high", NO_THINKING, None)

    assert payload == {"thinking": {"type": "disabled"}}


def test_an_unknown_model_is_never_sent_thinking():
    payload, max_tokens = resolve_reasoning("high", UNKNOWN, None)

    assert payload == {"thinking": {"type": "disabled"}}
    assert max_tokens == 4_096


def test_adaptive_models_get_an_effort_word():
    payload, _ = resolve_reasoning("high", ADAPTIVE, None, display="summarized")

    assert payload == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    }


def test_minimal_clamps_to_the_nearest_accepted_effort():
    # Anthropic accepts only low/medium/high/xhigh
    payload, _ = resolve_reasoning("minimal", ADAPTIVE, None)

    assert payload["output_config"] == {"effort": "low"}


def test_xhigh_degrades_to_max_where_xhigh_is_unsupported():
    assert resolve_reasoning("xhigh", ADAPTIVE, None)[0]["output_config"] == {
        "effort": "xhigh",
    }
    assert resolve_reasoning("xhigh", ADAPTIVE_NO_XHIGH, None)[0]["output_config"] == {
        "effort": "max",
    }


def test_manual_models_get_a_budget_scaled_to_their_output_ceiling():
    # 0.6 of haiku-4.5's 64000 ceiling
    payload, max_tokens = resolve_reasoning("high", MANUAL, None)

    assert payload == {"thinking": {"type": "enabled", "budget_tokens": 38_400}}
    assert max_tokens == 39_424  # budget + the minimum room for an answer


def test_the_smallest_level_still_clears_anthropics_floor():
    # 0.02 of 64000 is 1280, but the floor is what matters on smaller models
    payload, _ = resolve_reasoning("minimal", MANUAL, None)

    assert payload["thinking"]["budget_tokens"] >= 1024


def test_a_caller_max_tokens_shrinks_the_budget_rather_than_being_raised():
    payload, max_tokens = resolve_reasoning("xhigh", MANUAL, 4_000)

    assert max_tokens == 4_000  # the cap is a billing contract
    assert payload["thinking"]["budget_tokens"] == 4_000 - 1_024


def test_a_max_tokens_too_small_for_any_budget_raises():
    with pytest.raises(UnsupportedParameterError, match="leaves no room"):
        resolve_reasoning("high", MANUAL, 1_500)


# ── sampling ───────────────────────────────────────────────────────────────────


def test_the_newest_models_refuse_sampling_even_with_thinking_off():
    # this is a model property, not a thinking property
    with pytest.raises(UnsupportedParameterError, match="does not accept"):
        check_sampling(ADAPTIVE, {}, temperature=0.2, top_p=None, top_k=None)


def test_any_model_refuses_sampling_while_thinking_is_active():
    thinking = {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    with pytest.raises(UnsupportedParameterError, match="thinking is active"):
        check_sampling(MANUAL, thinking, temperature=0.2, top_p=None, top_k=None)


def test_sampling_survives_on_a_model_that_accepts_it_with_thinking_off():
    check_sampling(MANUAL, {}, temperature=0.2, top_p=0.9, top_k=40)
    check_sampling(
        MANUAL, {"thinking": {"type": "disabled"}},
        temperature=0.2, top_p=None, top_k=None,
    )


def test_no_sampling_values_is_always_fine():
    check_sampling(ADAPTIVE, {}, temperature=None, top_p=None, top_k=None)
