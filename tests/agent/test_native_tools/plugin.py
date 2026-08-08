"""The battery's plugin: eight fixture tools in a plain `SimpleToolRegistry`,
the REAL `ShellNativeMiddleware`, and a controllable permission policy — wired
through `PluginAgentSessionRunner` exactly like an application wires
`ShellAccessPlugin`.

The middleware is the shipped one, tables included; only the TOOLS are
fixtures, because this battery is about what reaches `acompletion()`, not
about patching files. They carry the shipped tools' exact internal names, so
the tables address them unchanged."""

from __future__ import annotations

from luca.agent.contrib.shell import ShellNativeMiddleware
from luca.agent.contrib.simple_tool_registry import PermissionPolicy, SimpleToolRegistry
from luca.agent.core import AgentSession, ApprovalDecision, ApprovalOption, ToolExecution

from .tools import ALL_TOOL_CLASSES


class ControllablePermissionPolicy(PermissionPolicy):
    """Three modes, flippable between runs: `allow_all` resolves ALLOW,
    `deny_all` resolves DENY, `require_approval` answers PENDING until the
    test flips the mode (the drive re-asks per `run()`)."""

    def __init__(self, mode: str = "allow_all") -> None:
        self.mode = mode

    async def decide(
        self,
        session: AgentSession,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        if self.mode == "allow_all":
            return ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
        if self.mode == "deny_all":
            return ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)
        return ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)


class NativeToolsPlugin:
    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=[cls() for cls in ALL_TOOL_CLASSES],
            permission_policy=self.policy,
        )

    def get_middleware(self, agent_session: AgentSession) -> list:
        return [ShellNativeMiddleware(agent_session)]
