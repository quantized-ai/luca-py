"""End-to-end: a background sub-agent's result is injected into the parent
conversation and re-drives, without cancelling the live turn that spawned it."""

from __future__ import annotations

from luca.agent.contrib.subagents import TaskStatus
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import AssistantCell, SubAgentCell, ToolCallCell
from luca.agent.core.models import ExecutionStatus, TextContent, UserMessage
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.contrib.subagents.support import factory_over, scripted

from .helpers import fresh_session, submit, wait_until


async def test_background_subagent_result_is_injected_and_redrives(tmp_path):
    parent = scripted(
        faux_assistant_message(
            [faux_tool_call("task", {"agent_type": "explore", "prompt": "find X", "title": "findX"})],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Spawned the explorer.")]),
        faux_assistant_message([faux_text("The explorer found the answer.")]),
    )
    app = AgentApp(
        fresh_session(),
        provider=parent,
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",
        subagent_runner_factory=factory_over([scripted(faux_assistant_message([faux_text("X lives in runner.py")]))]),
    )

    async with app.run_test() as pilot:
        await submit(pilot, "go")
        await wait_until(
            pilot,
            lambda: any("found the answer" in cell.text for cell in app.query(AssistantCell)),
        )

        task = app.subagents.tasks()[0]
        assert task.status is TaskStatus.DONE
        assert task.result == "X lives in runner.py"
        assert [cell.status for cell in app.query(SubAgentCell)] == [TaskStatus.DONE]
        # the parent's `task` call completed — its live turn was never cancelled
        assert [cell.status for cell in app.query(ToolCallCell)] == [ExecutionStatus.COMPLETED]
        # the result reached the model as an injected user message
        injected = [
            entry
            for entry in app.runner.session.entries.values()
            if isinstance(entry, UserMessage)
            and any(isinstance(part, TextContent) and "X lives in runner.py" in part.text for part in entry.parts)
        ]
        assert len(injected) == 1
