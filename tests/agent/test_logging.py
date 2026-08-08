"""What the runner writes to the `logging` module.

The runner turns exceptions into durable state — a raise becomes a
`ToolExecutionError` or a `TurnFinish(ERRORED)`, and only `str(exc)` survives
into the session. These tests pin the other half of that: the ERROR record
carrying the traceback, which is the only place the stack still exists.

The library configures nothing, so every assertion here is about records
reaching a handler the TEST installed (`caplog`), never about formatting or
destinations.
"""

import logging

import pytest

from luca.agent.core.exceptions import ToolNotFound
from luca.agent.core.models import (
    SessionConfig,
    TextContent,
    UserMessage,
)
from luca.client.exceptions import ProviderAPIError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_tool_call,
)
from tests.agent.scenarios import (
    MODEL,
    DeterministicRunner,
    FakeToolRegistry,
    conversation,
    make_session,
)
from tests.agent.test_runner_failures import LookupTool


def one_call_session(session_id: str):
    return make_session(
        id=session_id,
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


def test_the_luca_logger_carries_a_null_handler():
    # Without one, logging.lastResort writes WARNING+ to stderr — which paints
    # over a running TUI. This is the guard against that.
    assert [type(handler) for handler in logging.getLogger("luca").handlers] == [logging.NullHandler]


async def test_a_raising_tool_body_logs_one_error_with_the_traceback(caplog):
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("lookup", {"a": 4, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It failed.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicRunner(
        one_call_session("s_log_tool"),
        tool_registry=FakeToolRegistry([LookupTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    with caplog.at_level(logging.ERROR, logger="luca"):
        async with runner.run() as run:
            [event async for event in run]

    records = [record for record in caplog.records if record.name == "luca.agent.core.runner"]
    assert [(record.levelname, record.getMessage(), record.exc_info[0]) for record in records] == [
        ("ERROR", "conv=c1 tool=lookup raised", ToolNotFound),
    ]


async def test_a_failing_llm_call_logs_one_error_with_the_traceback(caplog):
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [],
                error=faux_error("provider 500", error_class=ProviderAPIError),
            ),
        ]
    )
    runner = DeterministicRunner(
        one_call_session("s_log_llm"),
        provider=faux,
        ids=["ts", "tf"],
        now=1000,
    )

    with caplog.at_level(logging.ERROR, logger="luca"), pytest.raises(ProviderAPIError):
        async with runner.run() as run:
            [event async for event in run]

    records = [record for record in caplog.records if record.name == "luca.agent.core.runner"]
    assert [(record.levelname, record.getMessage(), record.exc_info[0]) for record in records] == [
        ("ERROR", f"conv=c1 LLM call failed (model={MODEL.model})", ProviderAPIError),
    ]


async def test_a_successful_turn_logs_nothing_at_warning_or_above(caplog):
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    runner = DeterministicRunner(
        one_call_session("s_log_quiet"),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    with caplog.at_level(logging.WARNING, logger="luca"):
        async with runner.run() as run:
            [event async for event in run]

    assert [record.getMessage() for record in caplog.records if record.name.startswith("luca")] == []
