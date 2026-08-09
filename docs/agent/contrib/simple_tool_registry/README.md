# Simple tool registry

The batteries-included `ToolRegistry`. The core only knows the four-method
contract ([`05-permissions.md`](../../05-permissions.md)); this package
supplies the implementations that cover most applications:
**`SimpleToolRegistry`** (a static tool list gated by one `PermissionPolicy`)
and **`ProxyToolRegistry`** (composition + routing over child registries),
plus the `PermissionPolicy` strategy contract and `YoloPermissionPolicy`.

```python
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry, ProxyToolRegistry,            # the registries
    PermissionPolicy, YoloPermissionPolicy,           # the approval strategy
)
from luca.agent.contrib.tools import Tool             # the tools it holds
```

## 1. `SimpleToolRegistry` in 30 seconds

Tools + one policy → a registry the runner drives:

```python
registry = SimpleToolRegistry(tools=TOOLS, permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

What it does per contract method — all four `async`, session first, the
conversation second:

| Method | Behavior |
|---|---|
| `get_tools(session, conversation_id)` | one `ToolSpec` per tool, from `Tool.get_tool_spec()` |
| `create_execution(session, conversation_id, call)` | the classic preflight: resolve the name (miss → `NOT_FOUND` birth), validate `Args` (failure → `INVALID` birth), collect the duck-typed `get_approval_context(args, session, conversation_id)` (a raise → `FAILED` birth with `details={"phase": "approval_context"}`; success → stored under `extras["approval_context"]`), else a `PENDING` birth carrying the tool's `ToolSpec` (incl. `timeout_in_ms`) |
| `decide(session, conversation_id, execution)` | delegates to `permission_policy.decide(session, execution)` — the policy needs no id, the execution carries one |
| `prepare(session, conversation_id, execution)` | resolves by `raw_tool_call.name` (the middleware-effective call), validates `Args`, and returns a closure binding the validated arguments, the session, the conversation **and the call's identity** |

`prepare()` does everything fallible; the callable it returns runs the body:

```python
prepared = await registry.prepare(session, conversation_id, execution)   # ToolNotFound / InvalidToolArguments
result = await prepared(cancellation_token=token)                        # tool.execute(args, session, conversation_id, …)
```

**The closure is where the call identity comes from.** `PreparedTool` still
takes only the cancellation token — that is what keeps the core free of any
Python tool class — so `tool_name` (the name this call resolved under) and
`tool_call_id` are bound here, at prepare time, and nothing is re-derived at
invocation ([`tools/README.md`](../tools/README.md) §2).

> ⚠️ **`prepare()` runs once per dispatch *attempt*, not once per call.** It was
> always contractually re-callable and idempotent; a deferring tool
> ([`tools/README.md`](../tools/README.md) §10) makes that load-bearing — a call
> parked at `AWAITING_RESULT` is re-dispatched from scratch, straight back
> through `prepare()`, on every drive until it resolves. Resolve and validate;
> never take a lock, a lease or a slot here, and never do I/O.

The registry holds no per-call state, which is what makes it safe for several
conversations at once ([`13-subagents.md`](../../13-subagents.md)). A subclass
that adds some keys it by `conversation_id`, and a `get_tools` override is where
a per-conversation tool list belongs:

```python
class GatedRegistry(SimpleToolRegistry):
    async def get_tools(self, session, conversation_id):
        specs = await super().get_tools(session, conversation_id)
        if session.conversations[conversation_id].depth:      # a subagent
            return [s for s in specs if s.name not in self.main_only]
        return specs
```

A raise means the body never ran — `ToolNotFound` records `NOT_FOUND`,
`InvalidToolArguments` records `INVALID`, both with **no `ExecutionAttempt`
appended** (`execution.dispatched` stays False). That is the invariant, and it
holds on a re-dispatch too, where the call demonstrably ran before.

> ⚠️ **`timeout_in_ms` bounds the body only.** The spec's deadline applies to
> the prepared callable; resolution, validation and approval sit outside it, so
> a tool with `timeout_in_ms=5000` is not bounded end to end.

## 2. The policy seam

`PermissionPolicy` is one async hook over the live session and the execution —
both read-only:

```python
class ReadOnlyPolicy(PermissionPolicy):
    async def decide(self, session, tool_execution) -> ApprovalDecision:
        allowed = tool_execution.raw_tool_call.name.startswith("read")
        return ApprovalDecision(
            decision=ApprovalOption.ALLOW if allowed else ApprovalOption.DENY,
        )
```

See [`05-permissions.md`](../../05-permissions.md) for the contract, the PENDING
gate, and idempotency; see
[`resource_permissions`](../resource_permissions/README.md) for the
full-featured rule-based implementation:

```python
strategy = PermissionStrategy(mode=PermissionMode.ASK)      # from resource_permissions
registry = SimpleToolRegistry(tools=TOOLS, permission_policy=strategy)
```

## 3. `ProxyToolRegistry` — composition

Concatenate registries; each child keeps its own tools *and* its own approval
policy. `add_registry()` appends a child after construction:

```python
proxy = ProxyToolRegistry(app_registry, plugin_registry)
proxy.add_registry(another)
runner = AgentSessionRunner(session, tool_registry=proxy)
```

Routing, per method:

| Method | Routing |
|---|---|
| `get_tools` | concatenates the children's specs in child order — duplicate tool names raise `ValueError` — and rebuilds the internal `{name → child}` cache |
| `create_execution` | reads that cache as-is; a name no child claimed gets a `NOT_FOUND` birth (a tool call only ever arrives after an LLM call, which warmed the cache on its way out) |
| `decide` / `prepare` | resolve *independently* of the cache: on a miss they warm it once from the children and try again |

So a call left pending approval by a previous process resolves in a fresh one
with no LLM call first, is gated by its owning child's policy, and dispatches.
Nesting proxies needs nothing special.

> ⚠️ **An unresolvable name is allowed, then not found.** For a name still
> unclaimed after that resolution, `decide` returns ALLOW and `prepare` raises
> `ToolNotFound`, so the call records the honest `NOT_FOUND` instead of a false
> `REJECTED`. Anything that resolves is always gated.

Next: [`questions/README.md`](../questions/README.md).
