"""Shared scaffolding for the subagent scenarios.

Everything here is core-only plus `contrib.subagents` — the package under test.
The doubles are built on `scenarios.FakeTool` / `FakeToolRegistry`, so a
subagent scenario reads exactly like every other runner scenario: a known
session, one action, a full-object postcondition.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.subagents import (
    RESULT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    CreateConversationResult,
    SpawnSubagent,
)
from luca.agent.core import (
    AgentSession,
    RuntimeConfig,
    SessionConfig,
    ToolSpec,
    declares_spawn,
)
from luca.agent.core.tool_registry import ToolRegistry
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import (
    MODEL,
    DeterministicRunner,
    FakeTool,
    FakeToolRegistry,
    conversation,
    make_session,
)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


__all__ = [
    "MODEL",
    "IncompleteSpawnTool",
    "UndeclaredSpawnTool",
    "RESULT_TOOL_NAME",
    "SPAWN_TOOL_NAME",
    "DeterministicRunner",
    "SubagentRegistry",
    "conversation",
    "faux_assistant_message",
    "faux_text",
    "faux_tool_call",
    "make_session",
    "spawn_call",
    "subagent_session",
]


def subagent_session(
    session_id: str = "s_sub",
    *,
    enabled: bool = True,
    max_depth: int = 1,
    **runtime,
) -> AgentSession:
    """An empty session with subagents turned on. `subagents_enabled` defaults
    to False in production, so every scenario has to ask for it."""
    return make_session(
        id=session_id,
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(
                subagents_enabled=enabled,
                subagents_max_depth=max_depth,
                **runtime,
            ),
        ),
    )


def spawn_call(prompt: str, description: str, *, call_id: str, task_id: str = "t1"):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


class SubagentRegistry(ToolRegistry):
    """The real subagent tools beside whatever else a scenario needs, with the
    real depth gate — composed by hand rather than through `ProxyToolRegistry`
    so a core-level scenario stays one object deep.

    `get_tools` withholds spawn-declaring specs exactly as
    `SubagentToolRegistry` does; everything else routes by name to whichever
    half owns it."""

    def __init__(self, tools=(), result_tool_name: str = RESULT_TOOL_NAME) -> None:
        from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

        self.subagent_tools = SimpleToolRegistry(
            tools=[SpawnSubagent(result_tool_name=result_tool_name), CreateConversationResult()],
            permission_policy=YoloPermissionPolicy(),
        )
        self.other = FakeToolRegistry(tools)
        self._subagent_names = {SPAWN_TOOL_NAME, result_tool_name}

    def _for(self, name: str):
        return self.subagent_tools if name in self._subagent_names else self.other

    async def get_tools(self, session, conversation_id) -> list[ToolSpec]:
        from luca.agent.contrib.subagents import spawn_gate_open

        specs = await self.subagent_tools.get_tools(session, conversation_id)
        if not spawn_gate_open(session, conversation_id):
            specs = [spec for spec in specs if not declares_spawn(spec)]
        return [*specs, *await self.other.get_tools(session, conversation_id)]

    async def create_execution(self, session, conversation_id, call):
        return await self._for(call.name).create_execution(session, conversation_id, call)

    async def decide(self, session, conversation_id, tool_execution):
        return await self._for(tool_execution.raw_tool_call.name).decide(session, conversation_id, tool_execution)

    async def prepare(self, session, conversation_id, tool_execution):
        return await self._for(tool_execution.raw_tool_call.name).prepare(session, conversation_id, tool_execution)


class UndeclaredSpawnTool(FakeTool):
    """Returns a spawn payload from a spec that never declared one — the shape
    that goes AROUND the gate, and that the runner refuses."""

    name = "undeclared_spawn"
    description = "Claims to spawn without declaring it."
    Args = _NoArgs
    # deliberately no `output_schema`

    async def execute(self, args, session, conversation_id, *, cancellation_token):
        from luca.agent.core import ExecutionResult, TextContent

        return ExecutionResult(
            content=[TextContent(text="spawned")],
            structured_content={
                "is_subagent_spawn": True,
                "task_id": "t1",
                "prompt": "go",
                "description": "d",
                "process_subagent_result_tool_name": RESULT_TOOL_NAME,
            },
        )


class IncompleteSpawnTool(FakeTool):
    """Declares the schema but returns a payload missing `prompt` — a contract
    violation, not a subagent with empty fields. Nothing else validates a
    payload against its own declaration, so the runner checks the fields it is
    about to ACT on, at the moment it acts on them."""

    name = "incomplete_spawn"
    description = "Declares a spawn schema and returns a broken payload."
    Args = _NoArgs
    output_schema: ClassVar[dict] = {"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}}

    async def execute(self, args, session, conversation_id, *, cancellation_token):
        from luca.agent.core import ExecutionResult, TextContent

        return ExecutionResult(
            content=[TextContent(text="spawned")],
            structured_content={
                "is_subagent_spawn": True,
                "task_id": "t1",
                "description": "d",
                "process_subagent_result_tool_name": RESULT_TOOL_NAME,
            },
        )


@pytest.fixture
def faux() -> FauxProvider:
    return FauxProvider()
