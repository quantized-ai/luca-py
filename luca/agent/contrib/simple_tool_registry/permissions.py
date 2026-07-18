"""Tool-approval strategy — `SimpleToolRegistry`'s one opinion hook.

The core framework knows nothing about permission policies: the runner talks
to a `ToolRegistry`, and each registry answers `decide()` for its own tools.
`SimpleToolRegistry` delegates that answer to the `PermissionPolicy` it was
constructed with. Modes, rules, resource globs, interactive prompts, remote
approval services — all of that is application logic implemented in a
strategy subclass (see `luca.agent.contrib.resource_permissions` for a
full-featured example used by `main.py`).

The contract:

- `decide()` receives the live `ToolExecution` — treat it as **read-only**
  (the runner owns every session write). The registry-supplied
  `extras["approval_context"]` dict on it is the strategy's input
  vocabulary; the core stores `extras` verbatim and never interprets it.
- Return ALLOW or DENY to resolve the call (it executes / is REJECTED), or
  PENDING to punt: the run pauses (status AWAITING_APPROVAL, the generator
  ends) and the application resolves out-of-band — asks its user, records the
  answer on the strategy — then calls `run()` again.
- `decide()` is re-invoked, once per `run()` entry, for a call that stays
  unresolved — implementations must be idempotent queries of their own state,
  not one-shot notifications. Sibling calls in one batch are decided
  concurrently (`asyncio.gather`).
- A resolved call is never re-decided: each execution carries at most one
  ALLOW/DENY, ever; only PENDING can repeat.
- Exceptions propagate and abort the run; the session stays consistent (the
  executions remain unresolved) and the next `run()` asks again.
"""

from __future__ import annotations

from luca.agent.core import ApprovalDecision, ApprovalOption, ToolExecution


class PermissionPolicy:
    """Strategy base: one hook, `decide()`. A duck-typed concrete base (no
    ABC), matching the system-prompt strategy — subclass and override."""

    async def decide(self, tool_execution: ToolExecution) -> ApprovalDecision:
        raise NotImplementedError


class YoloPermissionPolicy(PermissionPolicy):
    """Allow everything — the simplest strategy. Handy for demos, tests, and
    fully-trusted tool sets."""

    async def decide(self, tool_execution: ToolExecution) -> ApprovalDecision:
        return ApprovalDecision(decision=ApprovalOption.ALLOW)
