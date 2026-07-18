# Permissions & the tool registry

The framework has **no** opinion about whether a tool may run — it doesn't even
resolve tools. The runner is constructed with one **`ToolRegistry`** and drives
the whole tool lifecycle through its four methods; approval is just one of
them. Modes, rules, resource globs, human prompts, remote approval services —
all of that lives inside *your* registry (or the batteries-included one in
contrib). The core only knows the contract.

```python
from luca.agent.core import ToolRegistry, ApprovalDecision, ApprovalOption
```

## 1. The contract

```python
class ToolRegistry:
    def get_tools(self, agent_session) -> list[Tool]: ...
    async def create_execution(self, call, context) -> ToolExecution: ...
    async def decide(self, tool_execution, context) -> ApprovalDecision: ...
    async def execute(self, tool_execution, context, *, cancellation_token) -> ExecutionResult: ...
```

| Method | Owns | Notes |
|---|---|---|
| `get_tools` | the tool list for the next LLM call | queried fresh per call — may vary with session state |
| `create_execution` | the birth draft | placeholder identity (`id=""`, `created_at=0`); the runner stamps ids/timestamps. PENDING, or terminal-at-birth `NOT_FOUND`/`INVALID`/`FAILED` with a registry-authored `error` |
| `decide` | approval | ALLOW / DENY / PENDING; exceptions abort the run and the next `run()` asks again |
| `execute` | the body | raises map to statuses: `ToolNotFound` → `NOT_FOUND`, `InvalidToolArguments`/`ValidationError` → `INVALID`, anything else → `FAILED`; a return → `COMPLETED` |

Each `decide()` response is applied twice on the execution: `approval_status`
(the current state — `allowed` / `rejected` / `pending`) and a new entry in the
append-only `approval_decisions` audit log. Read state from `approval_status`,
history from the log. Treat the passed `tool_execution` as **read-only** — the
runner owns every session write.

> ⚠️ **No global gate.** Each registry answers `decide()` for its own tools —
> there is no cross-registry approval hook anywhere. Cross-cutting policy
> ("ASK for everything") is composition: share one strategy instance across
> your registries.

## 2. The batteries-included registry

`contrib/simple_tool_registry` covers the common case: a static tool list gated
by one **`PermissionPolicy`** — a strategy with a single async hook:

```python
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry, PermissionPolicy, YoloPermissionPolicy,
)

registry = SimpleToolRegistry(tools=TOOLS, permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

```python
class PermissionPolicy:
    async def decide(self, tool_execution: ToolExecution) -> ApprovalDecision: ...
```

```python
ApprovalOption.ALLOW    # run the tool
ApprovalOption.DENY     # never run it → terminal REJECTED on the spot
ApprovalOption.PENDING  # I can't decide yet → defer this call, ask again later
```

`YoloPermissionPolicy` allows everything. A custom policy is just code — an
allowlist by tool name, everything else denied:

```python
class AllowlistPolicy(PermissionPolicy):
    def __init__(self, allowed: set[str]):
        self.allowed = allowed
    async def decide(self, tool_execution):
        ok = tool_execution.raw_tool_call.name in self.allowed
        return ApprovalDecision(decision=ApprovalOption.ALLOW if ok else ApprovalOption.DENY)
```

`ApprovalDecision.metadata` is free-form provenance the core stores but never
reads (e.g. `{"via": "allowlist"}`); `created_at` self-stamps.

## 3. The gate — how `PENDING` pauses and resumes

Returning `PENDING` is how you ask a human (or any out-of-band system). The
sequence:

1. Your `decide()` returns `PENDING`.
2. The runner pauses: status → `AWAITING_APPROVAL`, emits `ApprovalRequired`, ends
   the run.
3. You read the awaiting calls, get an answer, and **record it on your policy**.
4. You call `run()` again. The runner re-asks `decide()`, which now returns
   `ALLOW`/`DENY` from the recorded state.

```python
while True:
    if runner.awaiting_approval():
        for execution in runner.pending_approvals():
            answer = ask_user(execution)                 # your UI
            policy.record(execution.id, answer)          # store it ON the policy
    async with runner.run() as run:                      # re-asks decide(); now resolves
        async for event in run:
            render(event)
```

**The runner is not a mailbox** — you never post the answer back through it. The
session stays `AWAITING_APPROVAL` until the next `run()` asks your registry again.

## 4. Idempotency — the one rule that matters

Because the runner **re-invokes `decide()` on every `run()`** for any still-
unresolved call, `decide()` must be an *idempotent query of your own state*, not
a one-shot notification. Record answers somewhere on the policy; return them when
asked:

```python
class HumanGatePolicy(PermissionPolicy):
    def __init__(self):
        self._answers: dict[str, ApprovalOption] = {}   # execution id → verdict
    def record(self, execution_id: str, verdict: ApprovalOption):
        self._answers[execution_id] = verdict
    async def decide(self, tool_execution):
        verdict = self._answers.get(tool_execution.id, ApprovalOption.PENDING)
        return ApprovalDecision(decision=verdict)
```

A resolved call is never re-asked (at most one `ALLOW`/`DENY` per call, ever);
only `PENDING` repeats. Sibling calls in one batch are decided concurrently, and
every call keeps an **independent** outcome: an `ALLOW`ed sibling proceeds to
execute even while another call sits deferred — the run parks at the gate only
after all currently runnable work has advanced, and the model is never called
again until every call in the batch is terminal.

## 5. `extras["approval_context"]` — the tool ↔ policy vocabulary

`decide()` only sees a `ToolExecution`. Its richest input is the approval
context `SimpleToolRegistry` stored under `extras["approval_context"]` — the
free-form dict the tool supplied via its duck-typed `get_approval_context`
([`03-tools.md`](03-tools.md) §3). The core never interprets `extras`; the
vocabulary is a contract you own on both ends. A common convention:

```python
# tool side:
async def get_approval_context(self, args, context):
    return {"requests": [{
        "resources": [{"permission": "read", "resource": args["path"]}],
        "answer_options": [
            {"resource_permissions": [{"permission": "read", "resource": "/repo/*"}],
             "metadata": {"preview": "Allow all reads in /repo"}},
        ],
        "metadata": {"preview": f"Read {args['path']}"},
    }]}

# policy side:
async def decide(self, tool_execution):
    ctx = tool_execution.extras.get("approval_context", {})
    for request in ctx.get("requests", []):
        for pair in request.get("resources", []):
            if self.matches_a_deny_rule(pair["permission"], pair["resource"]):
                return ApprovalDecision(decision=ApprovalOption.DENY)
    ...
```

This is exactly how
[`contrib/resource_permissions`](contrib/resource_permissions/README.md) builds
modes, path-glob rules, and "always allow" grants — a complete, rule-based
strategy (plus a typed tool mixin for this vocabulary) shipped outside the
core. The framework never sees any of it.

## 6. Composing registries

`ProxyToolRegistry` concatenates children and routes each call to the child
that owns the tool — each child keeps its own approval policy:

```python
from luca.agent.contrib.simple_tool_registry import ProxyToolRegistry

app_tools = SimpleToolRegistry(tools=TOOLS, permission_policy=ask_strategy)
trusted   = SimpleToolRegistry(tools=[ClockTool()], permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=ProxyToolRegistry(app_tools, trusted))
```

Duplicate tool names across children raise; nesting proxies works transparently.
See [`contrib/simple_tool_registry`](contrib/simple_tool_registry/README.md)
for the routing/miss semantics. Next:
[`06-system-prompts.md`](06-system-prompts.md).
