"""Regression: a host that kills the stream mid-flight must FAIL the turn.

Captured 2026-08-08 from a real run of `main.py` against
`openrouter:openai/gpt-5.4-mini` (session `a52daccd`). OpenRouter serves that
model off OpenAI's Responses API; the upstream stream died before emitting a
terminal event, so OpenRouter closed the SSE with three chunks — an `error`
object carrying a 502 `provider_unavailable`, then `finish_reason: "error"`,
then usage. `captures/openrouter_midstream_502.sse` is that body, verbatim.

WHAT LUCA DOES WITH IT TODAY — the bug these tests pin:

1. `_process_chunk` (`transports/openai/stream.py`) reads `choices`, `delta`,
   `finish_reason` and `usage`. The error chunk carries `"choices": []`, so
   every branch is skipped and the 502 is dropped on the floor.
2. `_classify_finish` (`transports/openai/transport.py`) has no case for
   `"error"` and falls through to `return (provider_value, None)`, so the
   stream ends in a `FinishEvent` — a SUCCESS — with `finish_reason="error"`
   and no error message.
3. The runner keys the round off `message.tool_calls`, never off
   `finish_reason` (deliberately, so a misclassifying provider cannot wedge or
   loop the conversation). No calls means "final answer", so the turn closes
   `COMPLETED` with `error=None` and a truncated thought recorded against it.

The application therefore cannot tell a dead provider from a real answer. In
the TUI the symptom is a thought that stops mid-sentence and a prompt that
comes back with nothing said.

WHAT THEY ASSERT: the contract the framework ALREADY has for a mid-stream LLM
failure — `test_runner_failures.py::test_streaming_llm_error_closes_the_turn_after_the_deltas`
— the turn closes `ERRORED` carrying the provider's message, the partial
assistant message is dropped, and the failure reaches the caller. Nothing new
is invented here; the only claim is that a 502 off the wire must land in the
same place a `StreamError` already does. WHERE the fix goes (teach the parser
the `error` chunk, teach `_classify_finish` the `"error"` reason, or teach the
runner about non-answer finish reasons) is open — these tests only say it must
not be nowhere.

Nothing is faked at the framework boundary: the real `OpenRouterProvider`, the
real transport, the real SSE parser and the real demo composition
(`build_runner`) all run. Only the socket is replaced.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from luca.agent.contrib.tui.wiring import build_runner
from luca.agent.core import AgentSessionRunner, LLMConfig, TurnOutcome
from luca.client.exceptions import StreamError
from luca.client.providers.openrouter import OpenRouterProvider

CAPTURES = Path(__file__).parent / "captures"

# The exact (provider, model) pair of the session this was captured from. It
# matters: `openrouter` is absent from the native-tools capability table, so
# `ShellNativeMiddleware` is inert for the whole run and no native item is on
# the wire. This failure has nothing to do with provider-native tools.
MODEL = LLMConfig(model="openai/gpt-5.4-mini", provider="openrouter", reasoning="medium")

DEAD_STREAM = (CAPTURES / "openrouter_midstream_502.sse").read_bytes()

# Two hand-authored healthy streams, used only to put a completed turn behind
# the failure. Ordinary OpenAI chat-completions vocabulary, which is what
# OpenRouter speaks.
TOOL_CALL_STREAM = b"""data: {"id":"gen-ok-1","object":"chat.completion.chunk","created":1786173700,"model":"openai/gpt-5.4-mini","provider":"OpenAI","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_add_1","type":"function","function":{"name":"add","arguments":""}}]},"finish_reason":null}]}

data: {"id":"gen-ok-1","object":"chat.completion.chunk","created":1786173700,"model":"openai/gpt-5.4-mini","provider":"OpenAI","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\": 2, \\"b\\": 3}"}}]},"finish_reason":null}]}

data: {"id":"gen-ok-1","object":"chat.completion.chunk","created":1786173700,"model":"openai/gpt-5.4-mini","provider":"OpenAI","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","native_finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}

data: [DONE]

"""

FINAL_ANSWER_STREAM = b"""data: {"id":"gen-ok-2","object":"chat.completion.chunk","created":1786173710,"model":"openai/gpt-5.4-mini","provider":"OpenAI","choices":[{"index":0,"delta":{"role":"assistant","content":"2 + 3 = 5."},"finish_reason":null}]}

data: {"id":"gen-ok-2","object":"chat.completion.chunk","created":1786173710,"model":"openai/gpt-5.4-mini","provider":"OpenAI","choices":[{"index":0,"delta":{},"finish_reason":"stop","native_finish_reason":"completed"}],"usage":{"prompt_tokens":120,"completion_tokens":8,"total_tokens":128}}

data: [DONE]

"""


@pytest.fixture
async def openrouter():
    """Builds the REAL `OpenRouterProvider` over an `httpx.MockTransport` that
    serves one canned SSE body per request, in order. Everything above the
    socket — transport, parser, `_classify_finish` — is production code."""
    clients: list[httpx.AsyncClient] = []

    def build(*bodies: bytes) -> OpenRouterProvider:
        remaining = list(bodies)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=remaining.pop(0),
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return OpenRouterProvider(api_key="test-key", async_http_client=client)

    yield build
    for client in clients:
        await client.aclose()


def demo_runner(session, provider, workspace):
    """`main.py`'s composition, minus the parts that read the developer's
    machine (skills discovery, instruction files) and minus subagents."""
    runner, _strategy, _questions = build_runner(
        session,
        workspace=workspace,
        provider=provider,
        mode="yolo",
        skills=False,
        instructions=False,
        subagents=False,
    )
    return runner


def node_types(session) -> list[str]:
    conversation = session.conversations[session.main_conversation_id]
    return [session.entries[node].type for node in conversation.nodes]


def last_entry(session):
    conversation = session.conversations[session.main_conversation_id]
    return session.entries[conversation.nodes[-1]]


async def test_a_dead_stream_on_the_very_first_call_fails_the_turn(openrouter, tmp_path):
    """Session `a52daccd` exactly: a fresh session, one user message, and the
    first LLM call dies mid-reasoning."""
    session = AgentSessionRunner.new_session(MODEL)
    runner = demo_runner(session, openrouter(DEAD_STREAM), tmp_path)
    runner.post_message("Write a simple console based tic tac toe game in game.py")
    run = runner.run(streaming=True)

    with pytest.raises(StreamError, match="Stream ended before a terminal response event"):  # noqa: PT012
        async with run:
            async for _event in run:
                pass

    # The partial assistant message is dropped, exactly as it is for every
    # other mid-stream failure — no truncated thought frozen into the history.
    assert node_types(session) == ["user", "turn_start", "turn_finish"]
    # Substring, not equality: WHICH wrapper the fix puts around the provider's
    # message is the fix's to choose. That the message survives is not.
    assert last_entry(session).outcome == TurnOutcome.ERRORED
    assert "Stream ended before a terminal response event" in (last_entry(session).error or "")
    assert runner.idle()  # a closed bracket is IDLE whatever its outcome


async def test_a_dead_stream_after_a_completed_turn_fails_only_the_second_turn(openrouter, tmp_path):
    """The same failure, but on a session that already has history: one turn
    that ran a tool and answered, then a second user message whose call dies.
    The finished work must survive; the new turn must not close COMPLETED."""
    session = AgentSessionRunner.new_session(MODEL)
    provider = openrouter(TOOL_CALL_STREAM, FINAL_ANSWER_STREAM, DEAD_STREAM)
    runner = demo_runner(session, provider, tmp_path)
    runner.post_message("What is 2 + 3?")
    async with runner.run(streaming=True) as first:
        async for _event in first:
            pass
    runner.post_message("Now multiply them instead.")
    run = runner.run(streaming=True)

    with pytest.raises(StreamError, match="Stream ended before a terminal response event"):  # noqa: PT012
        async with run:
            async for _event in run:
                pass

    assert node_types(session) == [
        # turn 1 — untouched by the failure that follows it
        "user",
        "turn_start",
        "assistant",
        "tool_execution",
        "assistant",
        "turn_finish",
        # turn 2 — opened, died, closed; no assistant entry
        "user",
        "turn_start",
        "turn_finish",
    ]
    assert last_entry(session).outcome == TurnOutcome.ERRORED
    assert "Stream ended before a terminal response event" in (last_entry(session).error or "")
    assert runner.idle()
