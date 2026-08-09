"""`LLMConfig` → the client call.

Two layers, asserted separately. `completion_options` is the whole translation
and is a pure function, so its result is asserted as ONE dict — including which
keys are absent, because "a knob nobody set must not appear on the wire" is the
contract. Then the runner end-to-end, where what matters is the OUTBOUND
request the provider actually saw, identically streaming and not.

`base_url` / `transport_class` / `api_key` are provider-CONSTRUCTION arguments,
not request fields, so they are only visible in the translation: a test driving
the `FauxProvider` passes an instance, which short-circuits provider resolution
entirely.
"""

import json

import pytest

from luca.agent.core.models import (
    LLMConfig,
    SessionConfig,
    TextContent,
    UserMessage,
)
from luca.agent.core.runner import completion_options
from luca.client.providers import resolve_provider
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.transports import OpenAITransport
from tests.agent.scenarios import (
    MODEL,
    DeterministicRunner,
    conversation,
    make_session,
)

CONFIGURED = LLMConfig(
    model="test-model",
    provider="faux",
    model_options={"max_tokens": 6000, "temperature": 0.2},
    provider_options={"provider": {"order": ["baseten"]}, "transforms": ["middle-out"]},
)

CUSTOM_HOST = LLMConfig(
    model="test-model",
    provider="my_host",
    model_options={"max_tokens": 100},
    provider_options={
        "base_url": "https://custom.api.example/v1",
        "transport": "luca.client.transports.OpenAITransport",
        "mycustom_param": 1,
    },
)


def _session(session_id: str, llm_config: LLMConfig):
    return make_session(
        id=session_id,
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=llm_config),
    )


# ── the translation ───────────────────────────────────────────────────────────


def test_model_options_pass_through_and_raw_options_are_keyed_by_provider():
    assert completion_options(CONFIGURED) == {
        "max_tokens": 6000,
        "temperature": 0.2,
        "provider_options": {"faux": {"provider": {"order": ["baseten"]}, "transforms": ["middle-out"]}},
    }


def test_base_url_and_transport_are_lifted_out_of_provider_options():
    # The two keys the client takes as named parameters; the rest of the block
    # stays raw and keyed by the provider it is meant for.
    assert completion_options(CUSTOM_HOST) == {
        "max_tokens": 100,
        "base_url": "https://custom.api.example/v1",
        "transport_class": OpenAITransport,
        "provider_options": {"my_host": {"mycustom_param": 1}},
    }


def test_an_unconfigured_config_sets_nothing():
    assert completion_options(MODEL) == {}


def test_runtime_overrides_win_per_key_over_the_stored_config():
    assert completion_options(
        CONFIGURED,
        model_options={"max_tokens": 500},
        provider_options={"transforms": []},
    ) == {
        "max_tokens": 500,  # overridden
        "temperature": 0.2,  # untouched by the override dict
        "provider_options": {"faux": {"provider": {"order": ["baseten"]}, "transforms": []}},
    }


def test_the_api_key_is_absent_unless_one_is_given():
    # Absent, not None: the client falls back to the provider's own environment
    # variable, and an explicit None would be indistinguishable from "no key".
    assert "api_key" not in completion_options(MODEL)
    assert completion_options(MODEL, api_key="sk-test") == {"api_key": "sk-test"}


def test_what_the_translation_produces_actually_builds_a_provider():
    # The cross-package seam, asserted for real rather than by shape: a host
    # `luca.client` has never heard of resolves per call, with no registration
    # anywhere, because base_url + transport_class are what its escape hatch
    # needs. This is what replaced registering hosts globally at boot.
    kwargs = completion_options(CUSTOM_HOST)
    provider = resolve_provider(
        "my_host",
        base_url=kwargs["base_url"],
        transport_class=kwargs["transport_class"],
        api_key="sk-test",
    )
    try:
        assert (type(provider).__name__, type(provider.transport)) == ("GenericProvider", OpenAITransport)
    finally:
        provider.close()


def test_a_transport_path_that_does_not_resolve_names_the_path():
    config = MODEL.model_copy(update={"provider_options": {"transport": "luca.client.transports.NoSuchTransport"}})

    with pytest.raises(ValueError, match="NoSuchTransport"):
        completion_options(config)


def test_a_transport_that_is_not_a_dotted_path_says_so():
    config = MODEL.model_copy(update={"provider_options": {"transport": "OpenAITransport"}})

    with pytest.raises(ValueError, match="dotted path"):
        completion_options(config)


# ── the runner ────────────────────────────────────────────────────────────────


async def test_configured_options_reach_the_request():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    runner = DeterministicRunner(
        _session("s_options", CONFIGURED),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    request = faux.requests[0]
    assert (request.max_tokens, request.temperature, request.top_p, request.provider_options) == (
        6000,
        0.2,
        None,
        {"faux": {"provider": {"order": ["baseten"]}, "transforms": ["middle-out"]}},
    )


async def test_streaming_sends_the_same_options():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    runner = DeterministicRunner(
        _session("s_options_stream", CONFIGURED),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run(streaming=True) as run:
        _ = [event async for event in run]

    request = faux.requests[0]
    assert (request.max_tokens, request.temperature, request.top_p, request.provider_options) == (
        6000,
        0.2,
        None,
        {"faux": {"provider": {"order": ["baseten"]}, "transforms": ["middle-out"]}},
    )


async def test_a_session_with_no_options_sends_none_of_them():
    # The unconfigured session is the one that must not change: every knob
    # absent, so the provider's own defaults stand.
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    runner = DeterministicRunner(
        _session("s_no_options", MODEL),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    request = faux.requests[0]
    assert (request.max_tokens, request.temperature, request.top_p, request.provider_options) == (
        None,
        None,
        None,
        None,
    )


async def test_a_runner_override_bounds_the_call_without_touching_the_session():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    session = _session("s_override", CONFIGURED)
    runner = DeterministicRunner(
        session,
        provider=faux,
        model_options={"max_tokens": 500},
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert (faux.requests[0].max_tokens, session.session_config.llm_config.model_options) == (
        500,
        {"max_tokens": 6000, "temperature": 0.2},
    )


async def test_options_are_stamped_on_the_assistant_entry_as_provenance():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    session = _session("s_options_provenance", CONFIGURED)
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    async with runner.run() as run:
        _ = [event async for event in run]

    assert session.entries["a1"].llm_config == CONFIGURED


async def test_the_api_key_is_nowhere_in_the_serialized_session():
    # The reason the key is a runner attribute and not an LLMConfig field: the
    # config is persisted AND copied onto every assistant entry, so a key
    # stored there would be written to disk once per message.
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    session = _session("s_secret", CONFIGURED)
    runner = DeterministicRunner(
        session,
        provider=faux,
        api_key="sk-do-not-persist",
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert "sk-do-not-persist" not in json.dumps(session.model_dump(mode="json"))
