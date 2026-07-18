"""ToolRegistry — the runner's single contract with the tool world.

The core framework does not resolve tools, validate arguments, or decide
approvals. The `AgentSessionRunner` is constructed with one `ToolRegistry`
(or none, for a toolless agent) and touches tools through exactly four
methods — everything else (permission policies, registries with behavior,
plugins) is application space; `luca.agent.contrib.simple_tool_registry` is
the batteries-included implementation.

Contract semantics:

- `get_tools` is DYNAMIC: the runner calls it fresh per LLM call, and the
  result may vary with session state. It is a query — never a lifecycle hook.
- `create_execution` returns a birth DRAFT. The registry owns the call-scoped
  facts: `tool_call_id`, `raw_tool_call`, `tool_spec` (including
  `timeout_in_ms`; `None` if unresolved), `status` (PENDING, or a
  terminal-at-birth NOT_FOUND / INVALID / FAILED), `error` (a
  `ToolExecutionError` for terminal births — the registry authors it), and
  `extras`. It uses placeholder identity (`id=""`, `created_at=0`); the
  RUNNER stamps `id`, `parent_id`, `created_at`, `ended_at` (when terminal at
  birth), `context_tokens`, and `is_doom_loop_flagged`, so ids and timestamps
  keep flowing through `generate_id()` / `now_ms()`.
- `decide` exceptions propagate and abort the run; the executions stay
  unresolved and the next `run()` asks again. Implementations must be
  idempotent queries of their own state — record answers out-of-band, return
  them when asked. A PENDING decision parks only that execution.
- `execute` raises map to statuses at the runner: `ToolNotFound` →
  NOT_FOUND; `InvalidToolArguments` or a pydantic `ValidationError` →
  INVALID; any other exception → FAILED. A returned `ExecutionResult` →
  COMPLETED (after `ContextManager.process_tool_output`).
"""

from __future__ import annotations

from .context import CancellationToken, ToolContext
from .models import (
    AgentSession,
    ApprovalDecision,
    ExecutionResult,
    ToolCall,
    ToolExecution,
)
from .tools import Tool


class ToolRegistry:
    """The four-method tool-lifecycle contract. A duck-typed concrete base
    (no ABC), matching the house strategy style — subclass and override."""

    def get_tools(self, agent_session: AgentSession) -> list[Tool]:
        raise NotImplementedError

    async def create_execution(
        self, call: ToolCall, context: ToolContext,
    ) -> ToolExecution:
        raise NotImplementedError

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        raise NotImplementedError

    async def execute(
        self,
        tool_execution: ToolExecution,
        context: ToolContext,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        raise NotImplementedError
