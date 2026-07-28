"""`SpawnSubAgentTool` — the `task` tool the model calls to delegate.

Calling it enqueues a background sub-agent and returns immediately with a task
id; the sub-agent's findings are delivered back into the conversation when it
finishes. The body never awaits the child, so the parent turn is not blocked.
The tool is deliberately absent from a sub-agent's own toolset, so a sub-agent
cannot spawn another one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.tools import Tool
from luca.agent.core import AgentSession, CancellationToken, ToolKind

from .manager import SubAgentManager


class SpawnSubAgentTool(Tool):
    name = "task"
    description = (
        "Delegate read-only research to a background sub-agent. The sub-agent "
        "explores with read/glob/grep in its own context and cannot modify "
        "anything. Returns immediately with a task id; its findings are "
        "delivered to you when it finishes. Spawn several for independent "
        "questions to run them in parallel."
    )
    tool_kind = ToolKind.OTHER

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        agent_type: Literal["explore", "general"] = Field(
            description=(
                "Which sub-agent to run: 'explore' to locate and map code, "
                "'general' to answer a focused research question."
            ),
        )
        prompt: str = Field(
            min_length=1,
            description="The task or question for the sub-agent, self-contained.",
        )
        title: str | None = Field(
            default=None,
            description="Optional short label shown in the UI.",
        )

    def __init__(self, manager: SubAgentManager) -> None:
        self.manager = manager

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        task_id = self.manager.spawn(args["agent_type"], args["prompt"], args.get("title"))
        return (
            f"Started {args['agent_type']} sub-agent (task {task_id}). It runs in "
            "the background; its findings will be delivered when it finishes."
        )
