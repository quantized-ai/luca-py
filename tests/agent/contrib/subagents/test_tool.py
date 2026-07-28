"""The `task` tool enqueues a background sub-agent and returns at once."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from luca.agent.contrib.subagents import SpawnSubAgentTool
from luca.agent.core import (
    AgentSessionRunner,
    CancellationToken,
    ExecutionResult,
    TextContent,
)
from luca.client.testing import faux_assistant_message, faux_text

from .support import FAUX_MODEL, factory_over, scripted


async def test_task_tool_spawns_and_returns_started_text(make_manager):
    manager = make_manager(
        runner_factory=factory_over([scripted(faux_assistant_message([faux_text("done")]))]),
        child_model=FAUX_MODEL,
        generate_id=iter(["t1"]).__next__,
    )
    tool = SpawnSubAgentTool(manager)
    session = AgentSessionRunner.new_session(FAUX_MODEL)

    result = await tool.execute(
        {"agent_type": "explore", "prompt": "map it", "title": None},
        session,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(
        content=[
            TextContent(
                text=(
                    "Started explore sub-agent (task t1). It runs in the background; "
                    "its findings will be delivered when it finishes."
                )
            )
        ]
    )
    assert manager.get("t1").agent_type == "explore"


def test_task_tool_rejects_an_unknown_agent_type():
    with pytest.raises(ValidationError):
        SpawnSubAgentTool.Args(agent_type="hacker", prompt="x")
