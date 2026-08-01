# Permissions & the tool registry

The framework has **no** opinion about whether a tool may run — it doesn't even
resolve tools. The runner is constructed with one **`ToolRegistry`** and drives
the whole tool lifecycle through its four methods; approval is just one of
them. Modes, rules, resource globs, human prompts, remote approval services —
all of that lives inside *your* registry (or the batteries-included one in
contrib). The core only knows the contract.

```python
from luca.agent.core import (
    ToolRegistry, PreparedTool, ApprovalDecision, ApprovalOption,
)
```

## 1. The contract

```python
class ToolRegistry:
    async def get_tools(self, session: AgentSession, conversation_id: str) -> list[ToolSpec]: ...
    async def create_execution(
        self, session: AgentSession, conversation_id: str, call: ToolCall
    ) -> ToolExecution: ...
    async def decide(
        self, session: AgentSession, conversation_id: str, tool_execution: ToolExecution
    ) -> ApprovalDecision: ...
    async def prepare(
        self, session: AgentSession, conversation_id: str, tool_execution: ToolExecution
    ) -> PreparedTool: ...
```

All four are async and take the live `AgentSession` first and the conversation
they are answering for second; none receives the cancellation token — the runner
races each call against it instead. Treat the session and the passed
`tool_execution` as **read-only**: the runner owns every session write.

A session can hold several conversations advancing at once
([13](13-subagents.md)), and one registry instance serves all of them:

| Rule | Why |
|---|---|
| No per-call state on `self` | two conversations call the same method concurrently; a field written in `get_tools` and read in `prepare` is a race, silently |
| State keyed by `conversation_id` needs no lock | dispatch within one conversation is sequential |
| Deliberately shared state needs an `asyncio.Lock` around the **mutation**, never around the I/O | locking the I/O serializes the parallelism you asked for |
| `asyncio.to_thread` is real parallelism | two bodies genuinely run at once; process-global resources (a chdir, an env var) are scoped per conversation, not mutexed |

| Method | Owns | Notes |
|---|---|---|
| `get_tools` | the tool list for the next LLM call | returns `ToolSpec`s — plain JSON-serializable data, not Python classes, so a registry fronting a remote tool server hands back the JSON Schema it already has. Queried fresh per call — may vary with session state |
| `create_execution` | the birth draft | carries no identity (`id` / `created_at` stay `None`); the runner stamps those and the ledger files the spec and stamps `tool_spec_id`. PENDING, or terminal-at-birth `NOT_FOUND`/`INVALID`/`FAILED` with a registry-authored `error` |
| `decide` | approval | ALLOW / DENY / PENDING; exceptions abort the run and the next `run()` asks again |
| `prepare` | resolution + argument validation | returns a **callable** that runs the body; called once per dispatch attempt, and only for an already-approved call — a deferred or denied one is never prepared |

Each `decide()` response is applied twice on the execution: `approval_status`
(the current state — `allowed` / `rejected` / `pending`) and a new entry in the
append-only `approval_decisions` audit log. Read state from `approval_status`,
history from the log.

> ⚠️ **No global gate.** Each registry answers `decide()` for its own tools —
> there is no cross-registry approval hook anywhere. Cross-cutting policy
> ("ASK for everything") is composition: share one strategy instance across
> your registries.

### The prepared callable

`prepare()` does everything fallible up front — resolve the tool, validate the
arguments — then hands back the thing that runs the body:

```python
PreparedTool = Callable[..., Awaitable[ExecutionResult]]

async def run(*, cancellation_token: CancellationToken) -> ExecutionResult: ...
```

It takes the run's cancellation token and nothing else, so capture whatever
else the body needs during `prepare()`. It must be a *callable*, not a
coroutine object: the runner may never invoke it (a cancellation landing
between the return and the call), and a bare coroutine would warn. The
`AgentSession` is the live object, but the `tool_execution` is a detached
snapshot — by the time the callable runs the runner has persisted RUNNING, so
read current durable state through `session.entries[execution.id]`.

Raising means the body never runs and the execution is never marked RUNNING —
`started_at=None`, `dispatched=False`, no `ToolExecutionStarted` event, and
`details["phase"] == "prepare"` on the recorded error:

| Raised from `prepare()` | Status |
|---|---|
| `ToolNotFound` | `NOT_FOUND` |
| `InvalidToolArguments` / pydantic `ValidationError` | `INVALID` |
| anything else — or a return that isn't callable | `FAILED` |

Once the callable is invoked the mapping ends: every raise from there on is
`FAILED`, with `started_at` set and `dispatched=True`.

> ⚠️ **`timeout_in_ms` bounds the body only.** `get_tools`,
> `create_execution`, `decide` and `prepare` have no deadline. A tool with
> `timeout_in_ms=5000` is not bounded end to end — the 5s bounds its prepared
> callable.

### Rules for registry authors

| Rule | Why |
|---|---|
| `prepare()` must be safe to call more than once | a crash during preparation leaves the call PENDING and the next drive prepares it again. No call-scoped side effects — don't consume a token, advance a cursor, or write a record keyed to this call |
| `prepare()` must not return holding a lock, lease, slot or connection | the runner may never invoke the callable, and nothing runs a cleanup path for it. `async with` *inside* `prepare()` is fine; otherwise acquire inside the callable, under `try/finally` |
| `prepare()` must not block | no deadline applies to it. Resolve from local state; a registry fronting a remote tool server keeps a cached tool list refreshed out of band and does its network work inside the callable |
| The returned callable is where wrapping belongs | retries, rate limiting, metrics, tracing, exception translation, result post-processing. Returning a tool's bound method directly is valid but gives all of that up |

> ⚠️ **The core never validates arguments.** It knows every tool's
> `input_schema` and never checks a call against it — a registry may delegate
> to a remote server that validates on its own side, and double validation
> would break it. Validation is yours, in `prepare()`.

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
    async def decide(self, session: AgentSession, tool_execution: ToolExecution) -> ApprovalDecision: ...
```

It takes no `conversation_id`: the execution already carries one
(`tool_execution.conversation_id`), which is precisely why that field is on the
one entry type a consumer receives detached from a path. A policy that answers
differently inside a subagent reads it there.

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
    async def decide(self, session, tool_execution):
        ok = tool_execution.raw_tool_call.name in self.allowed
        return ApprovalDecision(decision=ApprovalOption.ALLOW if ok else ApprovalOption.DENY)
```

`ApprovalDecision.metadata` is free-form provenance the core stores but never
reads (e.g. `{"via": "allowlist"}`); `created_at` self-stamps.

## 3. The gate — how `PENDING` pauses and resumes

Returning `PENDING` is how you ask a human (or any out-of-band system). The
sequence:

1. Your `decide()` returns `PENDING`.
2. The call parks and `ApprovalRequired` is emitted. Once nothing else in the
   conversation's subtree can advance, the run ends and the status derives
   `BLOCKED`.
3. You read the awaiting calls, get an answer, and **record it on your policy**.
4. You cause `decide()` to be re-asked.

**The runner is not a mailbox** — no answer ever travels through it. There are
two ways to trigger the re-ask, and they differ only in *when*:

| | Use it when | What it does |
|---|---|---|
| the next `runner.run()` | the run has ended (the ordinary case) | every undecided call is re-asked at the top of the drive |
| `run.notify(execution)` | the run is still going — a subagent gated while its siblings work | marks that execution's conversation for a re-check immediately, and restarts its drive if it had already parked |

```python
while not runner.idle():
    async with runner.run() as run:                      # re-asks decide(); now resolves
        async for event in run:
            render(event)
    if runner.blocked():
        for execution in runner.pending_approvals():     # subtree-scoped: subagents included
            policy.record(execution.id, ask_user(execution))
```

`pending_approvals(conversation_id=None)` returns every gated execution in that
conversation's **subtree**, and each one names its own conversation
(`execution.conversation_id`) — so an interactive app can say "subagent B is
asking" with no wrapper type. Answering from inside a live run instead:

```python
async for execution in run.approvals:      # gates as they are raised, at-least-once
    policy.record(execution.id, ask_user(execution))
    run.notify(execution)                  # look again NOW
```

## 4. Idempotency — the one rule that matters

Because the runner **re-invokes `decide()` on every `run()`** (and on every
`notify()`) for any still-unresolved call, `decide()` must be an *idempotent
query of your own state*, not a one-shot notification. Record answers somewhere on the policy; return them when
asked:

```python
class HumanGatePolicy(PermissionPolicy):
    def __init__(self):
        self._answers: dict[str, ApprovalOption] = {}   # execution id → verdict
    def record(self, execution_id: str, verdict: ApprovalOption):
        self._answers[execution_id] = verdict
    async def decide(self, session, tool_execution):
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

`decide()` sees the session and a `ToolExecution`. Its richest input is the
approval context `SimpleToolRegistry` stored under `extras["approval_context"]`
— the free-form dict the tool supplied via its duck-typed
`get_approval_context` ([`contrib/tools/`](contrib/tools/README.md) §3). The core never
interprets `extras`; the vocabulary is a contract you own on both ends. A
common convention:

```python
# tool side:
async def get_approval_context(self, args, session):
    return {"requests": [{
        "resources": [{"permission": "read", "resource": args["path"]}],
        "answer_options": [
            {"resource_permissions": [{"permission": "read", "resource": "/repo/*"}],
             "metadata": {"preview": "Allow all reads in /repo"}},
        ],
        "metadata": {"preview": f"Read {args['path']}"},
    }]}

# policy side:
async def decide(self, session, tool_execution):
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

Duplicate tool names across children raise; nesting proxies works
transparently. `get_tools` rebuilds the `{name → child}` routing cache, but
`decide` and `prepare` resolve **independently of it** — on a miss they warm it
once from the children and try again — so a call left pending approval by a
previous process is gated by its owning child on a cold resume and then
dispatches, with no LLM call in between.

> ⚠️ **A cache miss is not an ALLOW.** The proxy allows only a name that is
> still unresolvable after that cache-independent lookup; `prepare()` then
> raises `ToolNotFound` and the call records `NOT_FOUND`, which is the honest
> outcome — a DENY there would record `REJECTED` for a tool that never existed.

See [`contrib/simple_tool_registry`](contrib/simple_tool_registry/README.md)
for the routing/miss semantics. Next:
[`06-system-prompts.md`](06-system-prompts.md).
