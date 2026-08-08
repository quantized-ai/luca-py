"""`LLMConfig.options` → the client request.

The runner is the only thing that turns a session's configured invocation
settings into completion kwargs, and it must do it identically streaming and
not. What is asserted here is the OUTBOUND request the provider saw, because
that is the whole contract: a knob nobody set must not appear on the wire at
all, and a raw `provider_options` block must arrive keyed by provider name,
untouched.
"""

from luca.agent.core.models import (
    LLMConfig,
    ModelOptions,
    SessionConfig,
    TextContent,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import (
    MODEL,
    DeterministicRunner,
    conversation,
    make_session,
)

CONFIGURED = LLMConfig(
    model="test-model",
    provider="faux",
    options=ModelOptions(
        max_tokens=6000,
        temperature=0.2,
        provider_options={"faux": {"provider": {"order": ["baseten"]}, "transforms": ["middle-out"]}},
    ),
)


def _session(session_id: str, llm_config: LLMConfig):
    return make_session(
        id=session_id,
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=llm_config),
    )


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


async def test_options_are_stamped_on_the_assistant_entry_as_provenance():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    session = _session("s_options_provenance", CONFIGURED)
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    async with runner.run() as run:
        _ = [event async for event in run]

    assert session.entries["a1"].llm_config == CONFIGURED
