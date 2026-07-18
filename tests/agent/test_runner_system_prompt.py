"""System-prompt scenarios: how the runner resolves its `system_prompt_parts`
and drives the assembler around every LLM call.

Same declarative shape as `test_runner.py` — a KNOWN session + scripted faux
responses + a recording assembler double → ONE action (drain `run()`), then
assert the exact parts the assembler received (every static form and every
callable-returned form — str, dict, `SystemPromptPart` — coerces to a
`SystemPromptPart`) and the `system_message` each `FauxProvider` request
actually carried.
"""

import pytest

from luca.agent.core.models import (
    AgentSession,
    Conversation,
    ConversationStatus,
    SessionConfig,
    SessionRuntimeStatus,
    SystemPromptPart,
    TextContent,
    UserMessage,
)
from luca.agent.core.system_prompt import SystemPromptAssembler
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
)


# ── test doubles ──────────────────────────────────────────────────────────────


class ScriptedAssembler(SystemPromptAssembler):
    """Returns a scripted prompt; records every parts list it received."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.calls: list[list[SystemPromptPart]] = []

    def assemble_system_prompt(self, parts: list[SystemPromptPart]) -> str:
        self.calls.append(list(parts))
        return self.prompt


class PartCallable:
    """Callable part double: returns a scripted value (str, dict, or
    `SystemPromptPart`); records every (session_config, runtime_status) pair
    it was invoked with."""

    def __init__(self, value) -> None:
        self.value = value
        self.calls: list[tuple[SessionConfig, SessionRuntimeStatus]] = []

    def __call__(
        self,
        session_config: SessionConfig,
        runtime_status: SessionRuntimeStatus,
    ):
        self.calls.append((session_config, runtime_status))
        return self.value


# ── scenarios ─────────────────────────────────────────────────────────────────


async def test_no_system_prompt_parts_sends_no_system_message():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert [r.system_message for r in faux.requests] == [None]


async def test_string_part_reaches_the_assembler_as_a_part():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    session = AgentSession(
        id="s2",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=["You are helpful."],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert assembler.calls == [[SystemPromptPart(text="You are helpful.")]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_dict_part_reaches_the_assembler_as_a_part():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    session = AgentSession(
        id="s3",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[
            {"text": "You are helpful.", "priority": 5, "source": "env"},
        ],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert assembler.calls == [[
        SystemPromptPart(text="You are helpful.", source="env", priority=5),
    ]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_system_prompt_part_reaches_the_assembler_unchanged():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    session = AgentSession(
        id="s4",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[
            SystemPromptPart(text="You are helpful.", source="model", priority=2),
        ],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert assembler.calls == [[
        SystemPromptPart(text="You are helpful.", source="model", priority=2),
    ]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_callable_part_returning_string_is_invoked_and_coerced():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    part = PartCallable("You are helpful.")
    session = AgentSession(
        id="s5",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[part],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert part.calls == [(
        runner.session.session_config,
        SessionRuntimeStatus(
            status=ConversationStatus.RUNNING, turn_count=1, step_count=0,
        ),
    )]
    assert assembler.calls == [[SystemPromptPart(text="You are helpful.")]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_callable_part_returning_dict_is_invoked_and_coerced():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    part = PartCallable({"text": "You are helpful.", "priority": 5, "source": "env"})
    session = AgentSession(
        id="s6",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[part],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert part.calls == [(
        runner.session.session_config,
        SessionRuntimeStatus(
            status=ConversationStatus.RUNNING, turn_count=1, step_count=0,
        ),
    )]
    assert assembler.calls == [[
        SystemPromptPart(text="You are helpful.", source="env", priority=5),
    ]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_callable_part_returning_part_is_invoked_and_passed_through():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    part = PartCallable(
        SystemPromptPart(text="You are helpful.", source="model", priority=2),
    )
    session = AgentSession(
        id="s7",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[part],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert part.calls == [(
        runner.session.session_config,
        SessionRuntimeStatus(
            status=ConversationStatus.RUNNING, turn_count=1, step_count=0,
        ),
    )]
    assert assembler.calls == [[
        SystemPromptPart(text="You are helpful.", source="model", priority=2),
    ]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_assembler_receives_priority_sorted_parts_across_all_forms():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    session = AgentSession(
        id="s8",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=[
            {"text": "middle", "priority": 5, "source": "env"},
            "unranked",
            PartCallable({"text": "last", "priority": 9}),
            SystemPromptPart(text="first", source="model", priority=0),
        ],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert assembler.calls == [[
        SystemPromptPart(text="unranked"),
        SystemPromptPart(text="first", source="model", priority=0),
        SystemPromptPart(text="middle", source="env", priority=5),
        SystemPromptPart(text="last", priority=9),
    ]]
    assert [r.system_message for r in faux.requests] == ["ASSEMBLED PROMPT"]


async def test_blank_assembled_prompt_sends_no_system_message():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("   \n  ")
    session = AgentSession(
        id="s9",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        system_prompt_parts=["x"],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert assembler.calls == [[SystemPromptPart(text="x")]]
    assert [r.system_message for r in faux.requests] == [None]


async def test_parts_resolved_and_assembled_before_every_llm_call():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    assembler = ScriptedAssembler("ASSEMBLED PROMPT")
    part = PartCallable("Use the tools.")
    session = AgentSession(
        id="s10",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]),
        system_prompt_parts=[part],
        system_prompt_assembler=assembler,
        provider=faux, ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert part.calls == [
        (
            runner.session.session_config,
            SessionRuntimeStatus(
                status=ConversationStatus.RUNNING, turn_count=1, step_count=0,
            ),
        ),
        (
            runner.session.session_config,
            SessionRuntimeStatus(
                status=ConversationStatus.RUNNING, turn_count=1, step_count=1,
            ),
        ),
    ]
    assert assembler.calls == [
        [SystemPromptPart(text="Use the tools.")],
        [SystemPromptPart(text="Use the tools.")],
    ]
    assert [r.system_message for r in faux.requests] == [
        "ASSEMBLED PROMPT",
        "ASSEMBLED PROMPT",
    ]


def test_invalid_static_part_raises_type_error_at_construction():
    session = AgentSession(
        id="s11",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )

    with pytest.raises(TypeError, match="SystemPromptPart, str, or dict"):
        DeterministicRunner(session, system_prompt_parts=[42], now=1000)
