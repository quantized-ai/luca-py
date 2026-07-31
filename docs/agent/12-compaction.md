# Compaction

A long conversation eventually fills the model's context window. **Compaction**
replaces the older span with a summary of it, so the agent keeps working with
the recent exchanges intact and everything older represented densely.

Nothing is destroyed. Compaction opens a **new conversation inside the same
session**: both paths stay in the catalog, every compacted entry stays in
`entries`, and the summary entry records exactly which ids it replaced.

```
entries:               u1 ts1 a1 tf1 u2 ts2 a2 tf2 cmp    ← one addition
conversations:         c1 → [u1, ts1, a1, tf1, …]         ← the predecessor, intact
                       c2 → [cmp, u2, ts2, a2, tf2]       ← the new view
main_conversation_id:  "c2"
c2.previous_conversation_id == "c1"
```

The session id does not change. The session file does not change. The next
request projects only the active path, so the summarized entries stop costing
context while remaining inspectable.

## 1. Configure a manager

Compaction has exactly one extension point, and it is the `ContextManager` you
already know from [11](11-context-and-usage.md) — the collaborator that counts
context also decides when there is too much of it. The runner triggers, stamps
and archives; it never decides *when* a compaction is worth doing, *what* the
summary says, *which model* writes it, or *which nodes* survive.

```python
from luca.agent.core import AgentSessionRunner

runner = AgentSessionRunner(session, context_manager=MyContextManager())
```

A ready-made one ships in contrib: `SummarizingContextManager` (a context
gauge plus an LLM summary, with a keep_turns knob) —
[contrib/simple_context_manager/](contrib/simple_context_manager/README.md).
Write your own by subclassing `ContextManager` (section 4).

The shipped default never compacts: `should_compact` returns `False` and
`compact` raises `NotImplementedError`. Nothing consults it unless you call
`schedule_compaction()` by hand — which then fails on the drive, since a
manager always exists and there is no "unconfigured" state to reject at
schedule time.

## 2. Trigger it

**Automatically** — the manager is asked at the top of every drive:

```python
result = await runner.run()          # may compact first, then drive the turn
```

**On demand** — the caller schedules it; the next drive does the work:

```python
runner.schedule_compaction()         # writes the bracket + the entry, returns its id
await runner.run()                   # compacts, then drives if work is queued
```

`schedule_compaction()` is idempotent and **durable**: the intent survives a
crash, and the conversation derives `BUSY` until it has been driven.

> ⚠️ **`post_message` raises while a compaction is scheduled or in flight.**
> It requires a closed bracket, and a compaction has one of its own — durably,
> across a reload. Schedule immediately before driving.

## 3. Observe it

Three events, mapping one-to-one onto the tool events:

```python
from luca.agent.core.events import (
    CompactionScheduled, CompactionStarted, CompactionFinished,
)
from luca.agent.core.models import CompactionSource, TurnOutcome

async with runner.run() as run:
    async for event in run:
        match event:
            case CompactionScheduled(entry=e) if e.source is CompactionSource.POLICY:
                print("context is full — compacting")
            case CompactionStarted():
                print("compacting…")
            case CompactionFinished(entry=e, outcome=TurnOutcome.COMPLETED) if e.parts:
                print(f"replaced {len(e.compacted_nodes)} entries with {e.llm_config.model}")
            case CompactionFinished(outcome=TurnOutcome.COMPLETED):
                print("nothing to compact")
            case CompactionFinished(outcome=o, error=err):
                print(f"compaction {o.value}: {err}")
```

| Event | Fires | Snapshot |
|---|---|---|
| `CompactionScheduled` | the bracket + entry are on the path (opened or resumed) | `started_at` is None on a fresh open |
| `CompactionStarted` | `started_at` stamped, immediately before the LLM call | `llm_config` is still None — the manager picks the model |
| `CompactionFinished` | after the bracket closes or the transition commits, **whatever the outcome** | terminal; `parts` set → summarized, None → nothing done |

`CompactionFinished` fires on failure too. For a policy-initiated failure —
which degrades silently so the user's turn survives — it is the *only* signal.

## 4. Write a manager

```python
from luca.agent.core import (
    CompactionPlan, ContextManager, LLMConfig, TextContent, UsageCounters,
)

CHEAP = LLMConfig(model="openai/gpt-4o-mini", provider="openrouter")

class KeepLastTurn(ContextManager):
    def should_compact(self, session, conversation_id) -> bool:
        nodes = session.conversations[conversation_id].nodes
        total = sum(session.entries[n].context_tokens for n in nodes)
        return total > 100_000                       # your gauge, your threshold

    async def compact(self, session, conversation_id, nodes, entry):
        keep = list(nodes[-5:-1])                    # the tail you carry over
        summary, usage = await self._summarize(session, nodes)
        return CompactionPlan(
            entry=entry.model_copy(update={
                "parts": [TextContent(text=summary)],
                "llm_config": CHEAP,                 # what actually produced it
                "metadata": {"strategy": "keep-last-turn"},
            }),
            nodes=[entry.id, *keep],                 # the NEW path
            usage=UsageCounters(input=usage.input_tokens, output=usage.output_tokens),
        )
```

**`should_compact`** is sync (so `start()` can consult it at call time) and gets
the whole session plus the conversation being asked about. The threshold, the
sum and the window size are all yours — core has no context-total API, and
`luca.client.catalog` carries `context_window` per model. Core remembers nothing
about past failures either: a manager that should stop trying returns `False`.

> ⚠️ **Only the MAIN conversation is ever compaction-checked (V0).** A subagent
> is bounded by its own step ceiling instead ([08](08-runtime-config.md)); the
> pair still takes a `conversation_id` because the decision is about one
> conversation and always was.

**`compact`** is async — it makes an LLM call, which you own end to end. Return
`None` for "nothing to do".

Subclassing `ContextManager` means you inherit its per-entry accounting
(`calculate_context`, `prune_entry`, `process_tool_output`) unchanged — override
those too if your compaction strategy needs a different measure of size.

### `nodes` — the path you may rewrite

`nodes` is the active conversation's path **minus this compaction's own
`TurnStart`**, ending with your entry. You may carry any of those ids, in any
order, with new entries interleaved — and nothing else. An id outside the tuple
is a plan rejection, so you never have to recognize or route around framework
markers, and `plan.nodes = list(nodes)` is always a legal full carry.

### `plan.nodes` — the new conversation, before ids exist

| Element | Means |
|---|---|
| a `str` | carry this existing node over at this position |
| an entry object | create it here — uncommitted, no `id`/`created_at` yet |

The runner stamps `id`, `parent_id` and `created_at` on everything you create:
one shared timestamp for the whole transition, parents threaded left to right,
and `None` for a plan that opens with a created entry. A created `ToolExecution`
also has its `tool_spec` filed in `session.tool_specs` and its `tool_spec_id`
stamped, exactly like one an ordinary turn wrote.

```python
[entry.id]                                              # fold everything
[entry.id, "u3"]                                        # fold, keep the question
[UserMessage(parts=[…]), entry.id, "u2", "ts2", "a2"]   # a framing message first
```

## 5. What the framework guarantees

| # | Invariant |
|---|---|
| 1 | Carried entries are never copied, renumbered, reordered or mutated. |
| 2 | The pre-compaction conversation stays in the catalog with its exact path plus the closing marker; the successor names it in `previous_conversation_id`. |
| 3 | Every entry you did not carry stays in `entries` and is listed, in path order, in `compacted_nodes`. Nothing is ever deleted. |
| 4 | Entries you create get their identity stamped by the framework (above). |
| 5 | If you return `None`, raise, time out or are cancelled, the conversation is identical to before plus one closed bracket. No partial state exists, ever. |
| 6 | Your `parts` never reach the projector unless the transition committed — a failed compaction cannot tell the model "here is a summary". |
| 7 | Your `usage` is recorded against the pre-compaction conversation even if the plan is then rejected or cancelled. Not recorded if you return `None` or raise. |
| 8 | You are handed a **deep copy** of the entry and the **live** session. The framework applies exactly `parts`, `llm_config` and `metadata`, and discards the rest. |
| 9 | A crash before the transition leaves an open bracket the next drive resumes **with the same entry**; a closed bracket is never retried; an entry that already has `parts` is never re-run; a second bracket never piles up. |
| 10 | At most one compaction per drive. `should_compact` is not consulted while a conversational turn is open, nor when the session is `IDLE`. |
| 11 | `before_llm_call`, `after_llm_response`, `build_model_string` and `build_tool_list` never fire for your LLM call. `before_entry_written` fires for everything compaction writes. |
| 12 | The entry's `context_tokens` is recalculated when your `parts` land, before entry middleware. |
| 13 | A structurally valid plan is **always** committed. Core never judges whether the compaction was worth doing — that is `should_compact`'s question, asked before the LLM call rather than after it. |

## 6. What gets refused

The runner validates **structure, never meaning**. A rejected plan closes the
bracket `ERRORED` and leaves the conversation untouched:

| Rejection | Message |
|---|---|
| an id that does not exist | `plan references unknown entry 'x'` |
| an id you were not offered (an archived one, the bracket's own `TurnStart`) | `plan references entry 'x', which is not on conversation 'c1'` |
| the same id twice | `plan references entry 'x' twice` |
| an empty plan | `an empty plan is not a compaction` |
| the plan omits the compaction entry | `plan omits the compaction entry 'cmp'` |
| the conversation moved under the plan | `the … conversation changed under the plan` |
| no content (None, empty, or whitespace-only text) | `plan carries no content` |

All raise `CompactionPlanError`. An image-only summary is legitimate content
and is accepted.

> ⚠️ **"Trim without summarizing" is not expressible.** The no-content rule
> means a manager that wants to drop history with no summary must emit a marker
> part (`"[earlier messages dropped]"`) or return `None`.

## 7. Hazards you own

Committed as given, never checked — checking them would mean the runner knowing
what your entries *mean*:

- **splitting a tool call from its result**, in either direction → a
  provider-side 400 on the very next request;
- **carrying a nonterminal `ToolExecution`** → `ProjectionError` on the next
  request;
- **carrying a turn marker without its pair** → a carried `TurnStart` with no
  finish is a *phantom open turn* and the next drive resumes a turn that never
  happened;
- **carrying an unresolved `ChildConversation`** → `ProjectionError` on the next
  request, exactly like a nonterminal execution;
- **reordering carried ids** — you chose the path;
- **summarizing away a trailing unanswered `UserMessage`** → **the one silent
  failure**: the question disappears with no error anywhere.

That last one is not exotic. A "summarize everything, keep nothing" manager
produces it by default, because compaction runs at the *top* of a drive, when a
just-posted message is on the path and unanswered. The framework does not
prevent it — a manager may legitimately fold that message into the summary text
— so carrying it is your decision to make deliberately.

Cutting on turn brackets avoids every hazard on this list.

## 8. How a failure behaves

A failed compaction never damages the conversation and never blocks the user's
turn. The bracket closes on the pre-compaction path and status derivation skips
it, so the leaf before it decides what happens next.

| Ending | Bracket | Raises? | Next drive |
|---|---|---|---|
| the manager returned `None` | closed `COMPLETED` | no | ordinary |
| success | closed `COMPLETED`, archived | no | ordinary |
| plan rejected / `compact` raised / provider error | closed `ERRORED` | **`source=USER` only** | not retried |
| deadline expired | closed `TIMED_OUT` | **`source=USER` only** | not retried |
| cancelled | closed `CANCELLED` | no | not retried — the drive stops |
| crash before or during the summary | **open** | — | resumes in place, same entry |

**A user-scheduled failure raises** — you asked for it, so you are told. **A
policy-initiated failure degrades**: the bracket closes, `CompactionFinished`
carries the outcome, and the drive goes on to the conversational turn.
Compaction is an optimization and must not cost the user their turn; the price
is that a manager bug surfaces only on the event stream.

**A cancel always stops the drive**, whatever outcome it carried — the cancel is
against the drive, not against the compaction alone.

## 9. Recovering a compacted conversation

After a compaction nothing has been destroyed:

```python
main = session.conversations[session.main_conversation_id]
archived = session.conversations[main.previous_conversation_id]   # the pre-compaction path
summary = session.entries[main.nodes[0]]
summary.compacted_nodes                              # precisely which ids it replaced
summary.llm_config                                   # and what wrote it
session.usages[archived.id][summary.id]              # and what that cost
```

`previous_conversation_id` chains one hop at a time, so several stacked
compactions walk back to the original conversation.

Every compacted entry is still in `session.entries`, so a bad summary is a
recoverable mistake. (No "undo" command ships — the data supports one.)

Next: [`13-subagents.md`](13-subagents.md).
