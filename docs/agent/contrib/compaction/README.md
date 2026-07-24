# Compaction

Long conversations eventually overflow the model's context window. Compaction
summarizes the conversation and continues in a **new** session seeded with that
summary. The original session is never touched, so a compaction is atomic (the
swap is all-or-nothing) and reversible (reopen the old `<id>.json`).

`luca.agent.contrib.compaction` is a contrib package: the core knows nothing
about it. It has two pieces.

## `CompactionStrategy` — what to keep verbatim

A concrete base class, one hook, the same house style as `PermissionPolicy` and
`ContextManager`. Its only job is the split: which trailing nodes survive
verbatim; everything before them is summarized.

| Strategy | Keeps |
|---|---|
| `CompactionStrategy()` (base) | nothing — summarize the whole conversation (the Claude-Code `/compact` shape) |
| `RecentTurnsStrategy(keep_turns=N)` | the last N exchanges verbatim (the user message plus its turn), summarize the rest |

The cut always lands on an exchange boundary (a `TurnStart`, extended back to
include the `UserMessage` that prompted it), never mid-turn — a mid-turn cut
would strand a tool call from its result.

## `Compactor` — the gauge and the operation

```python
from luca.agent.contrib.compaction import Compactor, RecentTurnsStrategy

compactor = Compactor(RecentTurnsStrategy(keep_turns=2), threshold=0.8)
if compactor.should_compact(session):
    new_session = await compactor.compact(session, provider=provider)
```

- `context_used(session)` sums `Entry.context_tokens` over the active path.
- `context_window(session)` reads `luca.client.catalog` for the session's model,
  falling back to `default_window` when the model is not catalogued.
- `utilization` / `should_compact` drive the auto-trigger and the context bar.
- `compact()` summarizes the older span with one `acompletion` call (the
  session's own model) and returns a fresh `AgentSession` whose first node is a
  `CompactionEntry`, followed by verbatim copies of the kept tail. It returns
  the input unchanged when there is nothing older than the tail.

`build_compacted_session(...)` is the pure transform underneath — ids and clocks
are injectable, the source is never mutated, and it rebuilds the
`tool_executions` index for any copied executions.

## In the TUI

Session hand-off is an app concern (a strategy object cannot swap the runner's
session), so the TUI wires the `Compactor`:

- Auto-compact fires at the turn boundary once utilization crosses the
  threshold; the turn that crossed it finishes first, the next starts compacted.
- `/compact` runs it on demand.
- The context bar under the transcript shows utilization, colored toward red as
  it approaches the threshold.
- Flags: `--no-autocompact`, `--compact-threshold`, `--compact-keep-turns N`.
