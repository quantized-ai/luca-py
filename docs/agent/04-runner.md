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
    context_manager=None,           # optional — see 11 (context) and 12 (compaction)
    provider=None,                  # optional — a prebuilt luca.client provider instance
    api_key=None,                   # optional — the credential for this session's provider
    model_options=None,             # optional — client kwargs, over the session's
    provider_options=None,          # optional — base_url / transport / raw, over the session's
)
```

The last four are RUNTIME state: they are never serialized, and a reloaded
session does not carry them. `api_key` is here rather than on `LLMConfig`
precisely because the config is persisted and copied onto every assistant
message; passing none leaves the argument off the client call entirely, so it
falls back to the provider's own environment variable. The two option dicts
win per key over `LLMConfig`'s ([02](02-data-model.md)), which lets one process
cap or reroute its calls without rewriting what the session records.

`runner.recalculate_context_tokens()` re-derives every entry's
`context_tokens` through that context manager; nothing in the framework calls
it ([11](11-context-and-usage.md)).

(Plugins install through `PluginAgentSessionRunner` in contrib — see
[09-plugins.md](09-plugins.md).) Start a fresh session with the classmethod,
then arm it with a message:

```python
session = AgentSessionRunner.new_session(LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"))
runner = AgentSessionRunner(session, tool_registry=registry)
runner.post_message("Summarize the repo.")     # appends a UserMessage, status → BUSY
```

`post_message` takes a string, or an ordered list of parts to mix text and
images ([02](02-data-model.md)). Parts are validated against the `ContentPart`
union, and empty input (`""`, `[]`) raises `AgentError`:

```python
runner.post_message([
    ImageContent(source=ImageBase64(data=b64_bytes, media_type="image/png")),
    TextContent(text="What is in this screenshot?"),
])
```

Posting is **not idle-only** — a conversation accepts a message whenever
something will eventually answer it, including **mid-turn** while the agent
works:

```python
run = runner.start()
runner.post_message("Prefer the standard library, please.")  # lands INSIDE the open turn
await run   # the model sees it on its next call, and the turn cannot close
            # COMPLETED until it has been answered
```

| Target state | `post_message` |
|---|---|
| `IDLE` main conversation | accepts — the next `run()` opens a turn |
| trailing message queued (`BUSY`, no open turn) | accepts — one turn answers all of them |
| open turn (`BUSY` / `BLOCKED`) | accepts — the mid-turn append; a gated turn answers the post past the gate: the conversation derives `BUSY`, the next drive projects the gated call as an awaiting-approval placeholder ([10](10-projection.md) §2) and runs ONE model round, then re-parks at the same gate. The gate is untouched — a post is not an approval |
| open turn with unresolved subagents | accepts — the children never see it, but the PARENT does: a parked drive is woken and the model can steer, including stopping a task ([13](13-subagents.md) §6) |
| `CANCELLING` | raises `ConversationCancellingError` — retry after the flush |
| compaction scheduled / in flight | raises `AgentError` ([12](12-compaction.md)) |
| finished subagent, archived conversation, unknown id | raises `AgentError` — nothing will ever drive them |

`conversation_id=` targets any conversation — a live subagent included;
`None` (the default) resolves to the main one at call time.

> ⚠️ **A turn never closes `COMPLETED` over an unseen message.** A message
> that lands during what turns out to be the final LLM call is answered by
> one extra round in the same turn — the premature answer stays recorded. A
> turn that closes `CANCELLED` / `ERRORED` / `TIMED_OUT` instead **buries**
> the message: unanswered in that turn, but projected into the next request —
> late, never lost. Catch the dedicated `ConversationCancellingError` to keep
> the user's draft and resubmit after the flush.

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
result.status              # the DERIVED status where the run stopped
result.outcome             # how the last bracket this run closed ended, else None
result.pending_approvals   # list[ToolExecution] — non-empty iff BLOCKED
```

`status` is derived, not assumed: a turn that finished is `IDLE`, a gate nothing
can get past is `BLOCKED`. `outcome` is carried from the close rather than read
off the path, because after a compaction the closing marker is on the archived
conversation ([12](12-compaction.md)).

No usage on the result — provider usage is recorded per assistant entry in
`session.usages` ([11](11-context-and-usage.md)).

A timeout or model failure does **not** produce a result: the turn closes
`TIMED_OUT` / `ERRORED` and the exception re-raises through `await`/iteration.
The conversation is then `IDLE` — a failed turn is a closed turn, so recovering
means posting a new message, not re-driving the same request. A pending cancel
is the exception — it consumes the failure and returns normally.

A model that **stops without answering** counts as a failure too. A round with
no tool calls and a finish reason that is not an answer — `length` (the token
cap cut it off) or `error` (a refusal, safety filter or guardrail, which every
transport canonicalizes to that value) — closes `ERRORED` and raises
`IncompleteResponseError`. Unlike a transport failure it **keeps** the partial
assistant message, because the model really did produce those tokens and on a
truncation they are the useful half; the natural recovery is another
`post_message()` ("continue"), not re-running an identical request that would
truncate in the identical place. A truncated round that still produced complete
tool calls is *not* a failure — the calls parsed, so they run and the turn goes
on.

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
text-bearing event exposes `.text`, so one `match` serves every case, and
**every event carries `.conversation_id`** — the stream is the whole subtree's,
so with subagents running two conversations can be mid-block at once and a
renderer keys its live state by that field.

**Block events** — fire in *both* modes, once each block is complete:

| Event | Carries |
|---|---|
| `ReasoningBlock` | `.text` — a completed reasoning block |
| `TextBlock` | `.text` — a completed assistant text block |
| `ToolCallReceived` | `.tool_call_id`, `.execution` — the newborn execution (PENDING, or terminal at birth) |
| `ToolExecutionStarted` | `.tool_call_id`, `.execution` — RUNNING, emitted iff the tool body dispatches |
| `ToolExecuted` | `.tool_call_id`, `.execution` (terminal), `.result_text`, `.is_error` — what the model is told |
| `FinishReason` | `.finish_reason` |
| `ApprovalRequired` | `.executions` — emitted when a call parks at a gate (not necessarily last: a sibling subagent may keep going) |
| `SubagentsSpawned` | `.conversation_ids` — one batch of children, announced before any of them starts ([13](13-subagents.md)) |
| `SubagentStarted` | the subagent (`.conversation_id`) began — or resumed — doing work; a spawned child with no start yet is queued behind `subagents_max_workers` ([13](13-subagents.md)) |
| `SubagentPaused` | its drive ended with the turn still open (an approval gate, a suspension); it will run again |
| `SubagentFinished` | its turn closed; `.outcome` is the closing `TurnFinish`'s |
| `CompactionScheduled` / `CompactionStarted` / `CompactionFinished` | `.entry` (a deep snapshot) — one compaction's lifecycle ([12](12-compaction.md)) |

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
iff the body is dispatched, §8) → `ToolExecuted`. There is no `TurnFinished` event —
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

## 8. Tool dispatch — prepare, then run

The runner touches tools through exactly four `ToolRegistry` methods
([05](05-permissions.md)): all async, all taking the live session **and the
conversation being answered for**, none receiving the cancellation token.

| Call | When | What the runner does with it |
|---|---|---|
| `get_tools(session, conversation_id)` | per LLM call, through the runner's `async resolve_tool_specs()` | drops private specs (§[03](03-tools.md)), runs the (synchronous) `build_tool_list` middleware over the remaining `ToolSpec`s, then converts what it returns to the wire list ([07](07-middleware.md)) |
| `create_execution(session, conversation_id, call)` | once per tool call in the assistant response | stamps identity (including `conversation_id`), appends, emits `ToolCallReceived` |
| `decide(session, conversation_id, execution)` | for every undecided execution | applies the decision; a `DENY` is `REJECTED` on the spot |
| `prepare(session, conversation_id, execution)` | once per dispatch attempt of an ALLOWED call | invokes the callable it returns |

Dispatch is split in two. `prepare()` resolves the tool and validates the
arguments and hands back a callable; only once it has returned does the runner
persist `RUNNING` + `started_at` and emit `ToolExecutionStarted`; then it
invokes that callable. What that buys you:

- `started_at` / `execution.dispatched` mean "the body was dispatched", for
  every outcome — and `ToolExecutionStarted` fires iff it was.
- `NOT_FOUND` / `INVALID` mean resolution and validation failed. A *body* that
  raises `ToolNotFound` looking up a sub-resource is `FAILED`, like any other
  raise after dispatch.
- `error.details["phase"]` is a fact the runner knows, not an inference from
  `started_at`.

| What happened | status | `started_at` | `dispatched` | `details["phase"]` |
|---|---|---|---|---|
| `create_execution` raised | `FAILED` | `None` | `False` | `create_execution` |
| toolless runner, at birth | `NOT_FOUND` | `None` | `False` | `create_execution` |
| the registry authored a terminal draft | `NOT_FOUND`/`INVALID`/`FAILED` | `None` | `False` | the registry's own |
| `decide` returned `DENY` | `REJECTED` | `None` | `False` | — |
| a framework runtime limit refused the call at birth (the spawn budget, [13](13-subagents.md)) | `REFUSED` | `None` | `False` | — |
| `prepare` raised `ToolNotFound` | `NOT_FOUND` | `None` | `False` | `prepare` |
| `prepare` raised `InvalidToolArguments` / `ValidationError` | `INVALID` | `None` | `False` | `prepare` |
| `prepare` raised anything else | `FAILED` | `None` | `False` | `prepare` |
| `prepare` returned a non-callable | `FAILED` | `None` | `False` | `prepare` |
| the callable raised (any type) | `FAILED` | set | `True` | `execution` |
| the callable returned a non-awaitable | `FAILED` | set | `True` | `execution` |
| the callable returned | `COMPLETED` | set | `True` | — |
| the deadline expired on the callable | `TIMED_OUT` | set | `True` | — |
| the cancel grace expired on the callable | `INTERRUPTED` | set | `True` | — |
| cancelled up to and including `prepare()` settling | `CANCELLED` | `None` | `False` | — |
| crash after `RUNNING` was persisted | `INTERRUPTED` | set | `True` | — |

`to_tool_execution_error(execution, exception, *, phase)` builds the durable
`ToolExecutionError` from the live exception (which is never persisted).
Override it to redact secrets, keep domain codes, or add a traceback — the
`phase` is handed to you.

> ⚠️ **The deadline bounds the body, not the call.** `ToolSpec.timeout_in_ms`
> (beating `RuntimeConfig.tool_execution_timeout_in_ms`) applies to the prepared
> callable only — the four registry calls have no deadline at all, so a tool
> configured with `timeout_in_ms=5000` is not bounded end to end
> ([08](08-runtime-config.md)).

## 9. Cancellation

`cancel()` is a pure, synchronous signal — callable in any state, from any handle
(`runner.cancel()` and `run.cancel()` are equivalent; cancellation is
turn-scoped, not handle-scoped):

```python
run.cancel()                          # or: run.cancel(TurnOutcome.CANCELLED, error="user hit stop")
```

It appends a durable `CancelRequested`, trips the live cancellation token
(which makes the conversation derive `CANCELLING`), and returns immediately.
Cancellation **cascades**: every live subagent beneath the cancelled
conversation gets its own `CancelRequested` and winds down too, and each
unresolved link is resolved with a result that says so — a closed turn with an
unresolved child would be unprojectable. Cancelling one subagent leaves its
siblings and its parent running.

The wind-down happens at the
engine's next step boundary: unrun calls are stamped `cancel_signalled_at` and
become `CANCELLED`; an in-flight call is persisted with `cancel_signalled_at`
first, then gets a grace window — a within-grace return is `COMPLETED` with its
real result, expiry is `INTERRUPTED`; the turn closes with the requested
outcome. A parked cancel **survives save/reload** — the next `run()`/`start()`
is the flush (instant, no model call). A second `cancel()` while one is pending
raises `AlreadyCancellingError`.

No tool-owned code can make `cancel()` a no-op: all four registry calls **and**
the prepared callable are raced against the token. The four registry calls race
with zero grace, and the runner waits for the killed task to unwind, so a
`finally` / `async with` inside them still completes; only the body keeps a
grace window (`tool_cancellation_grace_period`, [08](08-runtime-config.md)).
The registry never learns a token exists — there is no partial answer worth
having from listing tools, minting a record, deciding an approval, or preparing
a dispatch.

| Cancelled during | Durable outcome |
|---|---|
| `get_tools` | no LLM call, nothing recorded; the turn winds down |
| `create_execution` | a PENDING draft is synthesized — the wind-down records `CANCELLED` |
| `decide` | no decision recorded, approval state untouched — the wind-down records `CANCELLED` |
| before a ready call's dispatch | left PENDING, no middleware fired — the wind-down records `CANCELLED` |
| `prepare`, up to and including its return | no `RUNNING` row, no `ToolExecutionStarted`; the dispatch path records `CANCELLED` in place |
| the prepared callable | the grace window decides: `COMPLETED` with its real result, or `INTERRUPTED` |

> ⚠️ **A cancelled birth is `CANCELLED`, never `FAILED`.** N tool calls always
> yield N tool executions — a cancellation landing mid-batch never drops one.

## 10. The status machine

Four statuses, **derived from the entries on every read** — nothing is stored,
so a reloaded or crashed session lands in the right state on its own. The
runner's predicates answer for the main conversation; ask the session for any
other ([02](02-data-model.md) §10).

| Status | Predicate | Meaning → your move |
|---|---|---|
| `IDLE` | `runner.idle()` | Nothing queued → `post_message()` |
| `BUSY` | `runner.busy()` | Something can advance (a queued message, a running tool, a working subagent, a subagent result or mid-turn post the model has not seen — a post into a gated turn included) → `run()` |
| `BLOCKED` | `runner.blocked()` | Nothing can advance until you act — a gate → resolve, then `run()` ([05](05-permissions.md)). A post flips it to `BUSY` for exactly one round (the acceptance table above), then it re-derives `BLOCKED` |
| `CANCELLING` | `runner.cancelling()` | Unconsumed cancel → `run()` flushes it |

```python
while not runner.idle():
    async with runner.run() as run:
        async for event in run:
            render(event)
    if runner.blocked():
        resolve(runner.pending_approvals())      # then loop: the next run re-asks
```

That loop is the whole protocol, subagents included: `pending_approvals()` is
**subtree-scoped**, so a subagent's gate surfaces on the main runner and each
execution names the conversation it came from.

Two consequences of the derivation, both deliberate: a **failed** turn is a
closed turn and therefore `IDLE` (retry by posting, not by re-driving), and a
trailing user message is `BUSY` — more can still be posted behind it. The
status says what the next `run()` will do, never whether input is accepted;
the acceptance table at the top of this page owns that.

A **closed compaction bracket is transparent** to that derivation: it is
skipped, and the leaf before it decides. That is what keeps a failed compaction
from looking retry-ready (a spin) and a completed one from burying a queued
question. An *open* compaction bracket derives `BUSY` like any open turn —
which is how a scheduled compaction survives a reload.

**Suspend vs. advance.** Exiting a lazy run's `async with` block *suspends* — it
closes the engine where it is, re-derives status, and finalizes that handle
without writing anything. The open turn resumes on a later `run()`. A finalized
handle is spent; create a fresh `runner.run()` to continue.

## 11. Compaction

The `context_manager`'s compaction pair runs as a step at the top of a drive —
*before* the conversational bracket opens:

```python
runner = AgentSessionRunner(session, context_manager=MyContextManager())

runner.schedule_compaction()   # optional: arm it explicitly (idempotent, durable)
await runner.run()             # compacts if due, then drives the turn
```

The drive order is: flush a parked cancel → resume, skip, or ask the manager →
run the compaction → then the ordinary turn. At most one compaction per drive,
and never while a conversational turn is open — an approval pause or a
crashed-mid-turn bracket is resumed first. `start()` decides at call time, so an
eager run opens a compaction bracket instead of a `TurnStart` when one is due.

Everything else — the compaction contract, the plan, the events, the guarantees —
is [`12-compaction.md`](12-compaction.md).

## 12. The run is a tree

When a turn spawns subagents ([13](13-subagents.md)), the handle you are holding
is the root of a tree of runs — one per live conversation:

```python
run.children                 # dict[str, AgentRun] — by conversation id, grows as spawns land
run.child(conversation_id)   # one handle anywhere in this run's subtree, or None
run.notify(execution)        # "something changed out of band for that execution's
                             #  conversation — look again NOW"
async for execution in run.approvals:     # gates raised during this run, as they are raised
    ...
```

`autostart_subagents=` (default `True`) decides who drives them:

| | `True` — the framework drives | `False` — you drive |
|---|---|---|
| when a child starts | as soon as `subagents_max_workers` admits it (immediately by default) — announced by `SubagentStarted` | when you iterate the handle |
| where its events arrive | on this run's stream, tagged with its conversation | on the child handle only (no double delivery) |
| your obligation | none | drive or cancel every spawn, or the parent's turn never ends |
| `subagents_max_workers` | applies ([13](13-subagents.md)) | refused — `run()` raises when a cap is set |

```python
from luca.agent.core.events import SubagentsSpawned

async with runner.run(autostart_subagents=False) as run:
    async for event in run:
        if isinstance(event, SubagentsSpawned):
            for cid in event.conversation_ids:      # handles exist by announcement time
                async with run.child(cid) as child:
                    async for child_event in child:
                        render(child_event)
```

Two rules explain the rest of the behavior:

- **A drive returns only when nothing in its SUBTREE can advance.** So a gated
  child returns (its drive is gone; answering restarts it) while a gated parent
  whose children are still working waits — which is why `ApprovalRequired` is
  not terminal on a parent's stream.
- **A run handle is single-use.** Resuming a subagent parked at a gate means a
  fresh handle: under `autostart_subagents=True` the framework makes one for you
  (on `notify()`, and on the next `run()`); under `False`, `run.child(cid)`
  hands you a new one.

Next: [`05-permissions.md`](05-permissions.md).
