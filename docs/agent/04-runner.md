# The runner

`AgentSessionRunner` is the async engine that drives a session forward: project
the conversation to model messages, call the model, record the turn, run tool
calls, loop — bracketed by `TurnStart` / `TurnFinish`. It's resumable: each
`run()` advances as far as it can, then stops at the next point that needs you.

```python
from luca.agent.core import AgentSessionRunner
runner = AgentSessionRunner(
    session,
    tool_registry=registry,         # None = toolless agent — see 05
    system_prompt_parts=None,       # optional — see 06
    system_prompt_assembler=None,   # optional — see 06
    middleware=None,                # optional — see 07
    conversation_projector=None,    # optional — see 10
    provider=None,                  # optional — a prebuilt luca.client provider instance
)
```

(Plugins install through `PluginAgentSessionRunner` in contrib — see
[09-plugins.md](09-plugins.md).) Start a fresh session with the classmethod,
then arm it with a message:

```python
session = AgentSessionRunner.new_session(LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"))
runner = AgentSessionRunner(session, tool_registry=registry)
runner.post_message("Summarize the repo.")     # appends a UserMessage, status → PENDING
```

`post_message` takes a string, or an ordered list of parts to mix text and
images ([02](02-data-model.md)):

```python
runner.post_message([
    ImageContent(source=ImageBase64(data=b64_bytes, media_type="image/png")),
    TextContent(text="What is in this screenshot?"),
])
```

## 1. Drive it: `run()`

`run()` returns an **`AgentRun`** handle. It is **lazy** — nothing happens until
you await or iterate it. Two consumption forms:

```python
# (a) await → RunResult (drives to the next stopping point, discards events)
result = await runner.run()

# (b) iterate → render events as they happen (iteration REQUIRES 'async with')
async with runner.run() as run:
    async for event in run:
        render(event)
```

Iterating a lazy run *is* the engine — pulling the next event is what advances
the agent. Stop iterating and the agent stops.

## 2. `RunResult` — where it stopped

```python
result.status              # IDLE (turn finished) | AWAITING_APPROVAL (paused at a gate)
result.outcome             # TurnOutcome if the turn closed this run, else None
result.pending_approvals   # list[ToolExecution] — non-empty iff AWAITING_APPROVAL
```

No usage on the result — provider usage is recorded per assistant entry in
`session.usages` ([11](11-context-and-usage.md)).

A timeout or model failure does **not** produce a result: the turn closes
`TIMED_OUT` / `ERRORED` and the exception re-raises through `await`/iteration
(status becomes `PENDING`, retry-ready). A pending cancel is the exception — it
consumes the failure and returns normally.

## 3. Eager: `start()`

`start()` is the eager twin: it begins immediately in a background task and runs
to its stopping point **whether or not you observe it**. Await to join;
`.cancel()` to stop. A late consumer still sees every event from the beginning.

```python
run = runner.start()                 # already running
result = await run                    # join
```

Use `run()` when the consumer paces the agent (a UI reading events); use
`start()` for fire-and-forget work you'll join later.

## 4. Events

Both forms deliver the same `AgentEvent` union (`luca.agent.core.events`). Every
text-bearing event exposes `.text`, so one `match` serves every case.

**Block events** — fire in *both* modes, once each block is complete:

| Event | Carries |
|---|---|
| `ReasoningBlock` | `.text` — a completed reasoning block |
| `TextBlock` | `.text` — a completed assistant text block |
| `ToolCallReceived` | `.tool_call_id`, `.execution` — the newborn execution (PENDING, or terminal at birth) |
| `ToolExecutionStarted` | `.tool_call_id`, `.execution` — RUNNING, emitted iff the tool body dispatches |
| `ToolExecuted` | `.tool_call_id`, `.execution` (terminal), `.result_text`, `.is_error` — what the model is told |
| `FinishReason` | `.finish_reason` |
| `ApprovalRequired` | `.executions` — emitted as the **last** event before an approval gate |

The three tool events carry a **deep snapshot** of the durable `ToolExecution`
at that moment (name/arguments live on `execution.raw_tool_call`); later
transitions never mutate an event you already received. `ToolExecuted`'s
`result_text` / `is_error` come from the same projection that builds the next
LLM request ([`10-projection.md`](10-projection.md)), so what you render is
what the model sees.

```python
from luca.agent.core.events import TextBlock, ToolCallReceived, ToolExecuted, ApprovalRequired

async with runner.run() as run:
    async for event in run:
        match event:
            case ToolCallReceived(execution=ex): print(f"→ {ex.raw_tool_call.name}({ex.raw_tool_call.arguments})")
            case ToolExecuted(execution=ex, result_text=text): print(f"← {ex.raw_tool_call.name}: {text}")
            case TextBlock(text=text): print(text)
            case ApprovalRequired(executions=execs): print(f"{len(execs)} call(s) need approval")
```

Per execution the order is strict: `ToolCallReceived` → (`ToolExecutionStarted`
iff it dispatches) → `ToolExecuted`. There is no `TurnFinished` event —
`RunResult` is the completion signal (a cancel flush may emit zero events).

## 5. Streaming

Pass `streaming=True` to *add* token-level delta events as they arrive. This
changes only the **event vocabulary** — the session updates are byte-for-byte
identical either way.

**Delta events** — fire *only* under `streaming=True`:

| Event | Carries |
|---|---|
| `ReasoningStart` / `ReasoningDelta` | `.text` (delta) |
| `TextStart` / `TextDelta` | `.text` (delta) |
| `ToolCallStart` | `.tool_call_id`, `.name` |

```python
from luca.agent.core.events import TextStart, TextDelta

async with runner.run(streaming=True) as run:
    async for event in run:
        match event:
            case TextStart(): print("assistant: ", end="", flush=True)
            case TextDelta(text=text): print(text, end="", flush=True)
```

Under streaming you still receive the block events too — a streaming renderer
typically prints the deltas live and treats `TextBlock` / `ReasoningBlock` as
no-ops (else text prints twice).

## 6. `on_event` — a callback either way

`on_event` (sync or async) fires inline for every event, *even when you only
await* the run:

```python
def log(event): audit.write(event.model_dump_json())
await runner.run(on_event=log)        # no iteration, still see every event
```

## 7. Async only — driving from sync code

The agent loop is **async-only** (unlike `luca.client`, which offers sync
helpers). There is no synchronous runner. From a sync entry point, wrap it:

```python
import asyncio
asyncio.run(drive(runner))            # one async fn that owns the run loop
```

## 8. Cancellation

`cancel()` is a pure, synchronous signal — callable in any state, from any handle
(`runner.cancel()` and `run.cancel()` are equivalent; cancellation is
turn-scoped, not handle-scoped):

```python
run.cancel()                          # or: run.cancel(TurnOutcome.CANCELLED, error="user hit stop")
```

It appends a durable `CancelRequested`, trips the live cancellation token, sets
status `CANCELLING`, and returns immediately. The wind-down happens at the
engine's next step boundary: unrun calls are stamped `cancel_signalled_at` and
become `CANCELLED`; an in-flight call is persisted with `cancel_signalled_at`
first, then gets a grace window — a within-grace return is `COMPLETED` with its
real result, expiry is `INTERRUPTED`; the turn closes with the requested
outcome. A parked cancel **survives save/reload** — the next `run()`/`start()`
is the flush (instant, no model call). A second `cancel()` while one is pending
raises `AlreadyCancellingError`.

## 9. The status machine

Poll these predicates to decide what to do next:

| Status | Predicate | Meaning → your move |
|---|---|---|
| `IDLE` | `runner.idle()` | Nothing queued → `post_message()` |
| `PENDING` | `runner.pending()` | Work queued (message / resolved approval / retry) → `run()` |
| `AWAITING_APPROVAL` | `runner.awaiting_approval()` | Paused at a gate → resolve, then `run()` ([05](05-permissions.md)) |
| `CANCELLING` | `runner.cancelling()` | Unconsumed cancel → `run()` flushes it |
| `RUNNING` | `runner.running()` | A run is actively driving (internal; self-heals on load) |

The runner **re-derives status from the entries** when it takes ownership of a
session, so a reloaded or crashed session lands in the right state on its own.

**Suspend vs. advance.** Exiting a lazy run's `async with` block *suspends* — it
closes the engine where it is, re-derives status, and finalizes that handle
without writing anything. The open turn resumes on a later `run()`. A finalized
handle is spent; create a fresh `runner.run()` to continue. Next:
[`05-permissions.md`](05-permissions.md).
