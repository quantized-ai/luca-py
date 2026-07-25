# Compaction — product spec

Status: design agreed, not implemented. Supersedes the implementation currently
on `feat/compaction`.

Scope of this document: the **core** framework contract — the data model, the
policy contract, the runner integration, the events, the public surface, and
every safety invariant. No implementation code. The default policy (how a
summary is produced, which nodes are kept, how the gauge works) is contrib and
is planned separately.

---

## 1. What compaction is

A long conversation eventually fills the model's context window. Compaction
replaces the older span of the conversation with a summary of it, so the agent
can keep working without losing the thread.

**Outcomes:**

- The agent continues past the context window with the recent exchanges intact
  and everything older represented by a dense summary.
- Nothing is destroyed. The pre-compaction conversation stays in the session,
  complete and inspectable.
- Compaction is atomic: it either fully happened or it never started. There is
  no partial state, ever.
- It happens automatically when the window fills, and on demand when the user
  asks.
- Applications choose the summarization strategy, or supply their own. The
  framework provides a default.
- A compaction that fails never damages the conversation and never blocks the
  user's turn.

**Non-goals:**

- Summarizing anything other than the active conversation.
- Cross-session or cross-conversation compaction.
- Automatic re-expansion of a summary ("uncompaction") — the data supports it,
  but no command ships.
- A context-window field or a context gauge in core. Core has no
  context-total API today (deliberately), and `should_compact` is the policy's
  judgment: the threshold, the sum, and the window size all live in contrib.
  `luca.client.catalog` already carries `context_window` per model, so contrib
  needs nothing new from core to look it up.
- Plugin composition of a policy. `PluginAgentSessionRunner` composes
  registries, prompt parts and middleware; a fourth collaborator waits for a
  real second case.

## 2. The central design decision

**Compaction opens a new conversation inside the same session. It does not
create a new session.**

`AgentSession.entries` is an append-only store keyed by id. A `Conversation` is
an ordered list of those ids — a *view* over the store, not a container. So
compaction is expressible as: add one entry, archive the current view, open a
new one.

Before — a session with two exchanges on conversation `c1`:

```
entries:              u1 ts1 a1 tf1 u2 ts2 a2 tf2
active_conversation:  c1 → [u1, ts1, a1, tf1, u2, ts2, a2, tf2]
conversation_history: []
```

After summarizing the first exchange and keeping the second:

```
entries:              u1 ts1 a1 tf1 u2 ts2 a2 tf2 cmp     ← one addition
active_conversation:  c2 → [cmp, u2, ts2, a2, tf2]        ← new view
conversation_history: [c1 → [u1, ts1, a1, tf1, …]]        ← archived intact
```

The session id does not change. The session file does not change. The model's
next request projects only the active path. The active path's context total
sums only the entries it visits, so the summarized entries stop counting while
remaining in the store.

**Everything follows from this:** kept entries never move, so nothing is
copied; `PrunedEntry` referents never leave the store, so nothing needs to be
chased; the `tool_executions` index is session-wide, so it stays correct;
`usages` is keyed by conversation id, so old records survive.

One consequence to state plainly, since it is easy to assume otherwise:
`parent_id` remains **resolvable** (every referent is still in `entries`) but
stops describing the new path. `cmp.parent_id` points at the `TurnStart` of the
compaction bracket, which is not on the new conversation, and every carried
node keeps the parent it had. This is harmless — `parent_id` is documented as a
recovery backstop and is never traversed — but the design relies on that being
true, so it is written down here rather than discovered later.

**Why this matters most:** the previous design built a new session and copied a
subset of entries into it. Every complication in it — deep copies, recursive
referent chasing, index rebuilding, a new session id, a discarded history,
dangling parent references — was undo-work for a copy that should not have been
happening.

## 3. Ownership

| Layer | Owns |
|---|---|
| **core** | The `CompactionPolicy` contract, the `CompactionPlan` result, the `CompactionEntry` lifecycle, the conversation transition, the ledger doors, the status-derivation rule, the events, and every safety invariant in §11–§13. |
| **contrib** | The default policy: how a summary is generated, which prompt is used, which nodes are kept, how the context gauge and threshold work, where the window size comes from. |
| **application** | Persisting the session, including writing it atomically (§11, G1). |

Compaction adds no middleware surface — the policy *is* the extension point
(§10).

Core never learns that "which nodes to keep" is a pluggable decision. At the
transition, the kept set is just a list of ids.

The precedent is `ToolRegistry`: the contract lives in core, every concrete
implementation lives in contrib or in application code.

Note that `CompactionEntry` and `conversation_history` are *already* core
models. Core owning the transition that produces them is the correction of an
existing inconsistency, not new surface area.

## 4. The compaction entry

**A compaction is a durable, mutable lifecycle record, exactly like a
`ToolExecution`.** It is written the moment a compaction is intended, mutated
in place as it progresses, and left in its terminal state whether it succeeded
or not. It is the source of truth about that compaction's whole lifecycle.

```python
class CompactionSource(str, Enum):
    USER = "user"       # schedule_compaction()
    POLICY = "policy"   # should_compact() said so


class CompactionEntry(Entry):
    type: Literal["compaction"] = "compaction"

    source: CompactionSource                  # who asked

    parts: list[ContentPart] | None = None    # None → nothing was produced
    compacted_nodes: list[str] | None = None  # None → nothing was replaced
    llm_config: LLMConfig | None = None       # what produced the content

    started_at: int | None = None             # None → scheduled, not yet started
    ended_at: int | None = None

    metadata: dict = {}                       # free-form; opaque to core
```

Changes from today: `summary: str` → `parts: list[ContentPart] | None`,
`summarized` → `compacted_nodes`, `details` → `metadata`, plus the new
`source` / `llm_config` / `started_at` / `ended_at` fields.

**There is no `status` field.** Every state a compaction can be in is already
readable from the turn bracket plus these fields, and a denormalized status
would be one more thing free to drift out of agreement with the log. `source`
is the only fact here that nothing else records, which is why it is the only
one that is stored rather than derived.

**Why `parts` rather than `summary`.** A summary is content, and content in
this data model is a list of `ContentPart` — the same shape `UserMessage.parts`
uses. Today that list will almost always be a single text part, which is
exactly the old `summary` string. But a future policy might want to carry an
image, a structured block, or several parts, and a `str` field forecloses that
for no benefit. It also removes a special case: the projector no longer builds
a `TextBlock` from a string, it maps parts the same way it does for a user
message, and the context calculation follows the same path.

`parts is None` and `compacted_nodes is None` mean the compaction has not
produced anything yet. `None` and an empty list are different facts, and the
type should say which one it is.

### Field ownership

Who writes what, and when. This table is the contract — nothing outside it
writes to the entry.

| Field | Written by | When |
|---|---|---|
| `id`, `parent_id`, `created_at` | the ledger's append door | when the bracket opens |
| `source` | the runner | same append (`USER` or `POLICY`) |
| `started_at` | the runner | immediately before the LLM call |
| `parts`, `llm_config`, `metadata` | the **policy**, via `plan.entry` | applied at the commit point |
| `compacted_nodes` | the runner, derived from `plan.nodes` | the commit point |
| `ended_at` | the runner | the commit point / the bracket close |
| `context_tokens` | `ContextManager` then `before_entry_written` | recalculated when `parts` land, before the commit point |

`llm_config` is **not** stamped by the runner. §10's whole point is that a
policy may summarize with a cheaper model than the session's, and the runner
cannot know which model that is until `compact()` returns. The consequence is
in §9: `CompactionStarted` carries `llm_config is None`.

`context_tokens` must be **recalculated when `parts` land**. At append time
`parts is None`, so the entry counts 0; if that stood, the summary would
contribute nothing to the active path's context total and the policy's own gauge
would immediately conclude the window is still full — compacting again on the
next drive, indefinitely. With G5 gone (§11) core no longer second-guesses that,
so this recalculation is the *only* thing that keeps a correct policy's gauge
honest. The recalculation follows the existing
`ToolExecution` precedent exactly (`_finalize_outcome` recalculates on the
terminal transition, before middleware, never after).

`ended_at` is kept even though `TurnFinish.created_at` records the same instant
within microseconds. The reason is the asymmetry below: the entry carries into
the new conversation while its bracket stays behind, so on the new path
`started_at` with no `ended_at` would read as a compaction that started and
never finished. `started_at` lives on the entry for the same reason — the entry
has to be self-describing wherever it is read.

### Lifecycle

The compaction runs **inside a `TurnStart`/`TurnFinish` bracket**, like any
other LLM exchange. The bracket owns how the attempt ended; the entry owns what
it produced. Between them, every state is readable:

| Log state | Means |
|---|---|
| Open bracket, `started_at is None` | scheduled, not yet started |
| Open bracket, `started_at` set | running — or crashed mid-run, which is the same thing to a resume |
| Closed `COMPLETED`, `parts` set | succeeded |
| Closed `COMPLETED`, `parts is None` | nothing to compact |
| Closed `ERRORED` / `TIMED_OUT` / `CANCELLED` | failed; do not retry this entry |

`TurnFinish` owns the outcome vocabulary and the `error` string. The entry
carries neither, so no detail is written down twice and nothing can disagree.

The distinction that matters for recovery is **open bracket versus closed
bracket**, not a field: an open one is an interrupted attempt and resumes, a
closed one is finished whatever its outcome. This is why `ToolExecution`'s
`status` field is not a precedent to copy — an execution has no bracket of its
own, several run concurrently inside one turn, and its approval trail is
append-only, so its state genuinely cannot be reconstructed from surrounding
structure. A compaction is one-to-one with a bracket.

This is what makes the intent **durable**. A process that dies before starting
leaves a session that still knows a compaction was asked for. A process that
dies mid-summary leaves an open bracket, which the runner already knows how to
resume.

### Mutability, and what it costs

`CompactionEntry` becomes the **second mutable entry type** alongside
`ToolExecution`. Three things follow, and all three are core changes:

- `SessionLedger.put_execution` is the only in-place update door and is
  `ToolExecution`-shaped. It generalizes to an entry-shaped door (§12).
- `AGENTS.agent.md` design principle 4 states "`ToolExecution` is the **only**
  mutable entry type". That sentence becomes false and must change, along with
  the model docstrings that repeat it.
- `before_entry_written`'s contract — "every update" — now covers compaction
  entries (§10).

### What the bracket buys

Wrapping the compaction in a turn is not decoration — it is what lets
compaction reuse machinery that already exists instead of growing a parallel
copy of it:

| Concern | Handled by |
|---|---|
| How the attempt ended | `TurnFinish.outcome` — `COMPLETED`, `CANCELLED`, `TIMED_OUT`, `ERRORED` |
| Cancelling mid-summary | `CancelRequested` attaches to the open bracket, exactly like a turn |
| Timeouts | the runner's existing turn timeout |
| Status while in flight | an open bracket already derives `PENDING` / `CANCELLING` |
| Crash recovery | the runner already resumes an open turn on load |

The summary is a real LLM call with the same failure modes as any other, and
the bracket is where this codebase already records how an LLM call ended.

**The bracket stays behind; only the entry carries over.** The pre-compaction
conversation ends `[…, ts, cmp, tf]`; the new one begins `[cmp, …]`. The
markers belong to the conversation where the work happened. The entry belongs
to both — it is the record of the work in one and the summary of it in the
other. This is the one asymmetry in the design, and it is deliberate.

### Projection

One rule, and it is **positional**:

> **A compaction bracket projects as nothing — the entire span
> `ts_c cmp [cr] tf_c`, whatever the outcome. A `CompactionEntry` that is not
> inside a bracket projects as a synthetic user message carrying its `parts`.**

A summary only means something on the path where the history it replaces is
gone. That is the new conversation, where the entry sits bare at the top level.
Inside its bracket the entry is the *record of the operation*, next to the
markers that say how the operation ended — bookkeeping, on a path that still
holds the originals.

Both halves fall out of one walk. `project_compaction` keeps a falsy-`parts`
guard and its return type widens to `ClientUserMessage | None` — a policy may
create a bare `CompactionEntry` in `plan.nodes`, and an empty user message must
not reach the wire — but that is a detail inside the method, not the governing
rule. `project_entry` already admits `None` (`project_turn_finish` and
`project_pruned` already return it on some inputs), so nothing structurally new
is introduced.

**Skipping the span is what makes it correct, not just tidy.**
`project_turn_finish` emits the `CANCELLED_TURN_MARKER` —
`"[Request interrupted by user]"` — for **any** `TurnFinish(CANCELLED)`. A
cancelled compaction never transitions, so its bracket stays on the active
path, and without the rule every later request carries this:

```
c1.nodes = [… u4, ts_c, cmp, cr, tf_c(CANCELLED)]
projects   [… "what is X?", "[Request interrupted by user]"]
```

The user cancelled a *compaction*; the model is told the question was
interrupted, and it was never shown that question in the first place. Silent,
durable, and reachable through the ordinary cancel path (§13). `TurnStart` and
`CancelRequested` already project nothing, so `tf_c` is the one node that
needed the rule — but stating it per-marker would be two rules where one does.

Nothing is lost by discarding the whole span: only markers can ever be inside a
compaction bracket. `post_message` raises while one is open, so no user,
assistant or tool entry can land there.

The rule lives on `ConversationProjector.project()`, which already walks
`conversation.nodes` and is already documented as the seat of path-level
policy. Deciding whether a node is inside a compaction bracket needs the path,
which the per-entry methods deliberately do not receive, so every per-entry
signature stays as it is.

### Usage and configuration

The summary is a real LLM call with real cost. Today it is recorded nowhere —
the response's usage is discarded and the model that produced it is not written
down. Both are fixed by the entry being durable and on the path:

- **`llm_config`** on the entry records what actually produced the content,
  mirroring `AssistantMessage.llm_config`. Note that `CompactionEntry` does
  **not** subclass `AssistantMessage`: they share this one field and disagree
  on discriminator, projection role, and content shape.
- **Usage** travels back on the plan (§5) and is written through
  `SessionLedger.record_usage()`, landing in `usages[conversation_id][entry_id]`
  like any other consumption.

The usage record keys to the conversation the call was made in — the
pre-compaction one, which is about to be archived. That is correct rather than
unfortunate: usage describes a request, and the input tokens on that request
were the old conversation's context.

**Usage is recorded for the attempt, not for the success.** It is written
before the commit point and is therefore kept even when the compaction then
fails or is rejected. The tokens were spent; a failed compaction that cost
money and left no trace of it would be the worse outcome. This also means cost
visibility does not depend on the transition happening.

## 5. The policy contract

```python
class CompactionPolicy:
    def should_compact(self, session: AgentSession) -> bool:
        """Has the active conversation crossed the point where compaction is
        worth doing? Consulted by the runner at the top of every drive."""
        return False

    async def compact(
        self,
        session: AgentSession,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        """Fill in `entry` and describe the resulting conversation.
        `nodes` is the path you may rewrite. `None` means nothing to do."""
        raise NotImplementedError()
```

```python
class CompactionPlan(BaseModel):
    entry: CompactionEntry              # the lifecycle record, filled in
    nodes: list[Union[AnyEntry, str]]   # the new path
    # an entry  → create it at this position (uncommitted, `id is None`)
    # a str     → carry this existing node over at this position
    usage: UsageCounters = UsageCounters()   # counters for the summarization call
```

The runner has already opened the bracket, committed the entry, and stamped its
`started_at`. It then hands the policy a **deep copy** of that entry. The
policy fills in `parts`, `llm_config` and `metadata` on the copy and returns it
alongside the path it produced. The runner applies those three fields to the
live entry at the commit point.

### `nodes` — the path the policy may rewrite

**The runner hands over an explicit view of what is compactable, and the plan
is validated against that view rather than against the raw path.** By the time
`compact()` is called the active path ends `[…, ts_c, cmp]` — the compaction's
own bracket. `ts_c` is framework bookkeeping for the operation currently
running; it is not conversation content and must never appear in the plan. So
the runner removes it and passes what remains:

```
active path   […, u3, ts_c, cmp]
nodes         […, u3,       cmp]      ← handed to the policy
```

Three consequences, and the second is the point:

- **A policy never has to know the bracket exists.** `nodes` is the ordinary
  conversation plus the entry it is filling in. No trailing markers to
  recognize, no framework trivia to route around.
- **Carrying `ts_c` becomes a plan rejection, not a hazard.** §13's existing
  rule — "references an id that is not on the current path" — now means *not
  among the nodes the runner handed over*. The runner is not judging that the
  plan is unwise; it is saying **you may only carry back nodes I offered you**,
  which is as structural as a rule gets. The most likely accidental form of the
  phantom-open-turn hazard (a policy slicing the tail of the path and sweeping
  up `ts_c` with it) stops being reachable.
- **The identity transform is legal.** `plan.nodes = list(nodes)` is a valid
  full-carry plan that commits with `compacted_nodes == []`. This is why `cmp`
  is **kept** in the view while `ts_c` is removed: stripping the entry too
  would trade one piece of invisible trivia ("never carry `ts_c`") for another
  ("always add an id I did not give you"), since §13 requires every plan to
  include the compaction entry.

The removal is exact, not a filter: `[…, ts_c, cmp]` is the tail by
construction — the two appends are consecutive, `post_message` raises while the
bracket is open, and a parked cancel flushes before the policy is ever called.

`nodes` is a **tuple** — a copy the policy cannot use to mutate the session,
and the same shape the runner's own G2 snapshot carries.

This is ergonomics plus one validation rule; it is **not** enforcement.
`session.active_conversation.nodes` still holds the markers, and a policy can
still carry an earlier turn's `TurnStart` or invent one. G6 is what keeps those
from doing damage.

**The copy is load-bearing, not politeness.** Entries are mutable Pydantic
models, so a policy handed the live entry could write `parts` onto it and then
fail. The bracket would close `ERRORED`, the path would be unchanged — and the
entry would now project as a synthetic user message. The very next request
would tell the model "here is a summary of the conversation so far" with
nothing actually summarized. Handing out a copy makes that unreachable: the
only writer to the session is the runner, at one instant.

**`nodes` is the new conversation, expressed before ids exist.** It cannot be a
`Conversation` because `Conversation.nodes` is a list of ids and the entries a
policy invents do not have one yet. The union is what distinguishes them: an
entry object is something to create, a string is something to carry over —
including the compaction entry itself, which is already committed and therefore
appears by id.

**`usage`** carries the summarization call's counters back. The policy owns the
LLM call, so it is the only thing that sees `response.usage`; the runner passes
the counters to `SessionLedger.record_usage(entry_id, **counters)`, which is the
only door onto the usage store and which builds the id-carrying `Usage` record
itself. The shape is exactly what the runner's existing `_to_usage_counters()`
produces, so there is one vocabulary rather than two.

**`UsageCounters` is a typed model, not a `dict[str, int]`.** The five counter
names are a closed set and `Usage` is `extra="forbid"`, so with an untyped dict
the only thing standing between a policy passing a provider's own vocabulary
(`prompt_tokens`, `completion_tokens`) and a raise was a runtime check inside
`record_usage` — a runner-internal write, on the one line of the step that
cannot produce a clean plan rejection. Typing the field moves that failure to
plan construction, inside the policy, where the author sees it: a bad counter
name never reaches the runner, so no rejection rule is needed for it.

- `should_compact` takes the whole session: it needs `entries` and the active
  path for its own accounting. It is **sync**, which is what lets `start()`
  consult it at call time (§7).
- `compact` is **async** — it makes an LLM call.
- The policy never touches the session. The runner performs the transition.
- `None` is the single "nothing to do" signal. The entry keeps `parts is None`
  and the bracket closes `COMPLETED` — nothing went wrong, there was simply
  nothing to do.
- The summarized span is not declared separately. It is everything on the
  current path `nodes` does not carry over, and the runner records it in
  `compacted_nodes`.
- The policy owns its own backoff. Nothing in core remembers that a previous
  compaction failed; a policy that should stop trying returns `False` from
  `should_compact`.

**Why a flat list rather than "a summary entry plus a kept suffix":** the shape
`head + keep` bakes in a policy decision — one summary entry, at the front,
followed by a contiguous tail. A policy might reasonably want to keep an
unanswered user message after the summary, fold that message's text into the
summary instead, put a framing message *before* the summary, or emit several
new entries. All of those are the policy's judgment, and none of them are things
the runner should have an opinion about:

```python
[entry.id]                                        # fold everything
[entry.id, "u3"]                                  # fold, keep the pending question
[UserMessage(parts=…), entry.id, "u2","ts2","a2","tf2"]   # a framing message first
```

### Uncommitted entries

An entry object in `nodes` is a **template**: it has not been committed, so it
has no identity yet. `Entry.id` becomes `str | None` and `Entry.created_at`
becomes `int | None`, both defaulting to `None`, and `None` means "not yet
committed". The runner assigns both when it commits the entry.

The same applies to `parent_id`. A policy fills in content only; the runner
stamps identity. Specifically:

- Every entry a plan creates receives the transition's timestamp — one
  `created_at` for the whole compaction.
- `parent_id` is threaded left to right across the plan, so a plan that creates
  several entries produces a correct chain rather than a set of orphans. The
  **first** created entry's parent is the node that precedes it in `nodes` when
  there is one, and `None` when the plan opens with it — a conversation's first
  node has no parent, which is already true of every session's first entry.
- An `id` or `created_at` a policy sets on a plan entry is ignored and
  overwritten.

**This migration is repo-wide, on purpose.** Three existing sites express
"uncommitted" as `id=""` / `created_at=0`: `ContextManager.prune_entry`,
`ToolRegistry.create_execution`'s documented birth draft (plus
`SimpleToolRegistry` and the runner's own synthesized drafts), and the test
doubles. All of them migrate in the same change. Half-migrating would leave two
conventions for one state, which is worse than the single dishonest one we have
now. The cost is small and measured: 12 `entry.id` reads in core, one
`created_at` read (`utils.py:453`), four construction sites, six test
assertions — and there is no type checker configured, so nothing breaks
statically. The gain is that constructing a template is now the natural thing
in **application-facing** API: a policy writes `CompactionEntry(parts=[…])` and
a registry writes `ToolExecution(tool_call_id=…, raw_tool_call=…)`, with no
placeholder noise. `ApprovalDecision.created_at` is a different model — a
self-stamping value object, not an `Entry` — and does not change.

### Construction

The runner is constructed with one policy:

```python
runner = AgentSessionRunner(session, compaction_policy=Compactor(...))
```

Omitted, compaction never happens — `should_compact` is never consulted and
`schedule_compaction()` raises. `AgentSessionRunner.__eq__` compares
configuration equivalence, so the policy joins the collaborators it compares
via `_equivalent` (otherwise two differently-configured runners compare equal,
which `tests/agent/contrib/test_plugins.py` asserts against).

## 6. API ergonomics

Compaction is a step inside a drive, not a separate operation. This is what
makes the events, the status, and the concurrency guarantees fall out.

**Automatic** — the policy decides, at the top of each drive:

```python
result = await runner.run()          # may compact first, then drive the turn
```

**Explicit** — the caller schedules it; it happens on the next drive:

```python
runner.schedule_compaction()
result = await runner.run()          # compacts, then drives if work is queued
```

**With the streaming handle** — the same drive, plus one call-time rule (§7):

```python
run = runner.start()
result = await run
```

**Observing it:**

```python
async with runner.run() as run:
    async for event in run:
        match event:
            case CompactionScheduled(): ...
            case CompactionStarted(): ...
            case CompactionFinished(entry=e, outcome=o): ...
```

### `schedule_compaction()`

```python
def schedule_compaction(self) -> str:
    """Open a compaction bracket and write a CompactionEntry(source=USER).
    Returns the entry id. Idempotent. Requires a CLOSED bracket."""
```

- It **opens a bracket and appends the entry**, then returns. It does not
  compact.
- **Precondition: a closed bracket**, the same guard `post_message` uses. An
  open conversational turn — a resumable bracket, `AWAITING_APPROVAL`,
  `CANCELLING` — raises `AgentError`. Without this check the appended
  `TurnStart` would nest inside the open turn, and `open_turn_index()` walks
  back to the *nearest* `TurnStart`, so the bracket model would silently
  corrupt: the open turn's eventual `TurnFinish` would close the wrong bracket.
- **Idempotent**: called again while a compaction bracket is already open, it
  returns the existing entry's id and writes nothing.
- With no `compaction_policy` configured it raises — the runner would open a
  bracket nothing can close.

The intent is therefore durable, and the conversation's status follows from it
without new machinery: an open bracket already derives `PENDING` — the existing
"work is queued, call `run()`" meaning, now genuinely true of the entry log
rather than asserted by a flag.

### The `post_message` consequence

`post_message` requires a closed bracket, so **while a compaction is scheduled
or in flight, posting a message raises** — and because the bracket is durable,
that survives a reload. An application schedules immediately before driving:

```python
runner.schedule_compaction()
await runner.run()                   # now post_message is legal again
```

This is accepted rather than worked around. It is exactly the treatment every
open turn already gets: you cannot queue input into a session that has an
unfinished bracket. A session reloaded with a scheduled compaction must be
driven before it takes new input, the same as a session that crashed mid-turn.
The alternative — teaching `post_message` to append behind an unstarted
compaction — buys type-ahead in a window measured in milliseconds and costs a
special case in the guard plus a plan that has to carry the trailing message.

## 7. Runner integration

Three integration points. The first two are the ones a naive reading gets
wrong.

### The compaction step runs before the conversational bracket

At the top of `_drive`, **before** `_ensure_open_turn()` and before the status
is set to `RUNNING`:

1. **Flush first.** An unconsumed `CancelRequested` inside an open compaction
   bracket ends the compaction now: close the bracket with the requested
   outcome, emit `CompactionFinished`, and **return** — the drive does not go
   on to the turn (§13).
2. **Resume, skip, or decide.**
   - An open compaction bracket **whose entry has `parts is None`** → that
     compaction resumes, reusing the same entry (G6).
   - Otherwise, **an open conversational turn → no compaction this drive.**
     Resume the turn instead. An open bracket whose entry already has `parts`
     falls here too: it is a committed compaction wearing a bracket's shape,
     not an interrupted one (G6), so the drive treats it as the phantom turn
     it is.
   - Otherwise, `should_compact()` → open a bracket and append an entry with
     `source=POLICY`.
   - An already-scheduled compaction wins over the policy — one compaction, and
     `entry.source` records who asked.
   - `CompactionScheduled` fires whenever a bracket is opened or resumed.
3. **Run it.** `started_at` is stamped, `CompactionStarted` is emitted, the
   path is snapshotted and the `nodes` view built (§5), the policy is
   consulted, and the result either commits the transition or closes the
   bracket. The bracket never stays open past the end of a drive that did not
   crash.
4. **Then drive normally** — `_ensure_open_turn()`, orphan recovery, the loop.

**The "no open turn" precondition is a real check, not a natural consequence.**
An earlier draft claimed the separate brackets made it automatic. They do not:
at drive top an open conversational turn legitimately exists after an approval
pause, after a crash mid-turn, and after a suspended lazy run. Skipping
compaction in that case is also what keeps the path projectable — a turn with a
nonterminal `ToolExecution` on it must never be handed to a policy to carve up.

Orphan recovery (`_recover_orphans`) stays where it is, after the compaction
step. Orphans only exist inside an open turn, and an open turn means no
compaction ran, so the two can never interleave.

### `start()` must decide at call time

`AgentRun.__init__` calls `_ensure_open_turn()` **synchronously at `start()`
time**, so by the time `_drive` executes, an eager run already has a
conversational turn open — and the drive-top compaction step would find one and
skip, forever. Policy-driven compaction would never fire for eager runs.

So `start()` consults the compaction decision at call time, before it opens
anything, and opens a **compaction** bracket instead of a `TurnStart` when one
is due. `should_compact` is sync precisely so this is possible. `_drive` then
finds the open compaction bracket and runs it, opening the conversational turn
afterwards. A compaction that was already scheduled needs nothing special: the
bracket is open, so `_ensure_open_turn()` is already a no-op.

### `RunResult` reports the bracket the run closed

```python
result = await runner.run()      # RunResult(status, outcome, pending_approvals)
```

`_build_run_result` currently reads `outcome` off `nodes[-1]` and hardcodes
`status=IDLE`. Both assumptions break here:

- After a successful transition the closing `TurnFinish` is on the **archived**
  conversation, so the active path's leaf is not it.
- After a compaction-only drive the leaf may be the `CompactionEntry` itself
  (fold-everything) or a carried `AssistantMessage` (a keep-last-N-messages
  policy) — neither has an `.outcome`, so today's code raises `AttributeError`.

Two changes: `outcome` means **how the last bracket this run closed ended**,
carried by the drive rather than re-read from the path; and `status` comes from
derivation instead of being hardcoded. No new field — a caller that needs to
distinguish "the agent answered" from "a compaction ran" reads
`CompactionFinished`, and `RunResult` already cannot distinguish an empty turn
from an answered one.

A useful narrowing falls out of §13: for a compaction-only drive, `outcome` is
only ever `COMPLETED` or `CANCELLED`. Every failure path either propagates
(so there is no `RunResult`) or degrades and goes on to drive a real turn.

## 8. Status

Status is per-conversation; `AgentSession.status` reads
`active_conversation.status`. **The `ConversationStatus` values are unchanged**
— `IDLE`, `PENDING`, `RUNNING`, `AWAITING_APPROVAL`, `CANCELLING` — and no
`COMPACTING` status is added.

But one derivation rule **is** added, and it has to be:

> **A closed compaction bracket is transparent to status derivation.** When the
> trailing nodes form a closed compaction bracket (`ts_c cmp tf_c`), skip it —
> repeatedly, if several stack — and derive from the leaf before it. If nothing
> remains after skipping, `IDLE`.

A bracket is a *compaction* bracket when the node immediately after its
`TurnStart` is the `CompactionEntry` — the adjacent pair the runner always
writes. A trailing `TurnFinish` with **no** `TurnStart` before it (a policy
carried one without its opener) is not a compaction bracket: stop skipping and
apply the ordinary closed-turn rules, so a carried `TurnFinish(ERRORED)` still
derives retry-ready `PENDING` rather than swallowing the whole path into
`IDLE`.

Open compaction brackets are unaffected: they derive `PENDING`, or `CANCELLING`
with an unconsumed cancel, exactly like any open turn. That is what resumes
them and what tells an application to call `run()`.

### Why the rule is required

Without it the existing rules produce two wrong answers, both silent.

**A failed compaction would be retry-ready.** A failure closes the bracket on
the pre-compaction path, so the leaf becomes `tf_c(ERRORED)` — which hits the
existing retry-ready rule and derives `PENDING`. `should_compact` is still true
(nothing was compacted), so a loop of the shape
`while not runner.idle(): await runner.run()` opens a *fresh* bracket every
drive and fails again. G6 does not prevent this: it forbids resuming the same
entry, not opening a new one. A failed compaction must derive `IDLE`.

**A closed bracket would bury a queued user message.** This one is caused by
core, not by a policy:

```
[u1 ts1 a1 tf1 u3]                          → leaf is u3    → PENDING ✓
[u1 ts1 a1 tf1 u3 ts_c cmp tf_c(COMPLETED)] → leaf is tf_c  → IDLE ✗
```

`u3` stops being the leaf, so it stops deriving `PENDING`, and the question is
silently dropped. Reachable whenever a compaction closes without transitioning
while a message is queued: the no-op case, a cancel (where the user cancelled
the *compaction*, not their question), and every failure. In-process the drive
continues to the turn regardless of status, so it only bites across a suspend,
a crash, or an application that polls `runner.idle()` between runs — and then
the input is lost with no error anywhere.

### The rule in action

```
[… u3 ts_c cmp tf_c(COMPLETED)]           → skip → u3    → PENDING   the question still drives
[… u3 ts_c cmp tf_c(CANCELLED)]           → skip → u3    → PENDING   cancelling ≠ dropping input
[… tf2(COMPLETED) ts_c cmp tf_c(ERRORED)] → skip → tf2   → IDLE      no spin
[… tf2(ERRORED) ts_c cmp tf_c(COMPLETED)] → skip → tf2   → PENDING   the real turn is still retry-ready
[ts_c cmp tf_c(COMPLETED)]                → skip → (none)→ IDLE      compaction on a fresh session
```

### The derived table

| State of the log | Derives to | Why |
|---|---|---|
| Open bracket around a compaction, started or not | `PENDING` | an open turn already derives `PENDING` |
| …with an unconsumed `CancelRequested` | `CANCELLING` | the existing cancel rule |
| Closed compaction bracket | *the leaf before it* | the new rule |
| After a successful transition | the new path's leaf | ordinary derivation on `c2` |

`RUNNING` is never durable for a compaction: `_drive` sets it transiently, and a
session that dies mid-summary re-derives on load (open bracket → `PENDING`),
which is the existing self-heal. `AWAITING_APPROVAL` is unreachable inside a
compaction bracket — there are no tool approvals in one.

### One concept, three consequences

A compaction bracket **is not a conversational turn**. That single idea drives
three separate changes: this derivation rule, the turn-count fix (§14), and
at-most-one-compaction-per-drive (G4).

### Post-transition statuses

The new conversation's status is ordinary derivation over the path the policy
chose, which means a policy can produce any of them:

| New path ends with | Derives to | Note |
|---|---|---|
| A carried `TurnFinish(COMPLETED)` | `IDLE` | the common case |
| The `CompactionEntry` itself | `IDLE` | fold-everything |
| A carried `AssistantMessage` | `IDLE` | keep-last-N-messages |
| A carried unanswered `UserMessage` | `PENDING` | what preserves a queued question |
| A carried `TurnFinish(ERRORED/TIMED_OUT)` | `PENDING` | retry-ready; the drive goes on to retry that turn |
| A carried `TurnStart` with no `TurnFinish` | `PENDING` | **a phantom open turn** — see §13 |
| …plus a carried `PENDING`-approval execution | `AWAITING_APPROVAL` | ditto |

The last two are policy hazards, not framework states, and are listed in §13.
The archived conversation is frozen at `IDLE` — set explicitly at the
transition, because `_drive` set it to `RUNNING` on the way in and archived
conversations are never re-derived again.

## 9. Events

Three, on the existing run stream, mapping one-to-one onto the tool events —
same lifecycle shape, same payload discipline. Purely observational, following
the persistence of the state they describe, each carrying a deep snapshot of
the entry at that moment.

```python
class CompactionScheduled(BaseModel):
    type: Literal["compaction_scheduled"] = "compaction_scheduled"
    entry: CompactionEntry           # started_at is None; source says who asked


class CompactionStarted(BaseModel):
    type: Literal["compaction_started"] = "compaction_started"
    entry: CompactionEntry           # started_at set; llm_config still None


class CompactionFinished(BaseModel):
    type: Literal["compaction_finished"] = "compaction_finished"
    entry: CompactionEntry           # parts set → summarized; parts None → nothing done
    outcome: TurnOutcome             # how the bracket closed
    error: str | None = None         # detail for TIMED_OUT / ERRORED
    created: list[AnyEntry] = []     # other entries the plan created
    conversation_id: str | None = None   # the new conversation; None if nothing changed
```

All three join the `AgentEvent` union as lifecycle events, alongside
`ApprovalRequired` — they fire in both streaming modes.

| Event | Emitted | Analogous to |
|---|---|---|
| `CompactionScheduled` | At the top of the drive, once the bracket and entry are on the path | `ToolCallReceived` |
| `CompactionStarted` | Once `started_at` is stamped, immediately before the LLM call | `ToolExecutionStarted` |
| `CompactionFinished` | After the bracket closes, whatever the outcome | `ToolExecuted` |

```python
async with runner.run() as run:
    async for event in run:
        match event:
            case CompactionScheduled(entry=CompactionEntry(source=CompactionSource.POLICY)):
                print("context is full — compacting")
            case CompactionStarted():
                print("compacting…")
            case CompactionFinished(entry=e, outcome=TurnOutcome.COMPLETED) if e.parts:
                print(f"compacted {len(e.compacted_nodes)} entries with {e.llm_config.model}")
            case CompactionFinished(outcome=TurnOutcome.COMPLETED):
                print("nothing to compact")
            case CompactionFinished(outcome=o, error=err):
                print(f"compaction {o.value}: {err}")
```

**`CompactionStarted` cannot carry `llm_config`.** The policy chooses the
model, and it has not been called yet (§4). The model that produced a summary
is known at `CompactionFinished`.

**`CompactionScheduled` fires for both sources**, always at the top of the
drive — not at the moment `schedule_compaction()` is called, which happens
outside any run and therefore has no stream to emit on. A user-scheduled
compaction announces itself on the next drive, and `entry.source` distinguishes
the two. This keeps the stream a complete, ordered record of the lifecycle
regardless of who initiated it.

The gap between `Scheduled` and `Started` is usually microseconds, but it is a
real window: a consumer may cancel in it, and the framework should let them.
Collapsing the two events would take that away.

**`CompactionFinished` fires on failure too.** A failed tool execution still
emits `ToolExecuted`; this is the same rule. Without it an observer that is not
the caller — a renderer, a logger — sees `CompactionStarted` and then silence
forever, with no way to know the compaction ended. For a degraded POLICY
failure (§13) it is the *only* signal, since nothing is raised. `outcome` and
`error` come from the closing `TurnFinish`, so no new vocabulary is introduced,
and `conversation_id` is `None` whenever nothing was actually compacted.

**There is no separate `CompactionFailed`.** The outcome vocabulary already
covers it.

## 10. Middleware

**No new middleware methods.** Compaction already has a first-class extension
point — `CompactionPolicy` — and hooks alongside it would create two ways to do
the same thing. Every candidate lands on the policy or on a hook that already
exists:

| Want | Served by |
|---|---|
| Change the summarization prompt | the policy |
| Summarize with a cheaper model | the policy — it sets `llm_config` |
| Decide *when* to compact | `should_compact` |
| Enforce an application rule on the plan | wrap the policy; it is a plain class |
| Redact or enrich the content before it persists | `before_entry_written` |
| Stamp external ids or metadata on the entry | `before_entry_written` |
| Observe the lifecycle | the three events (§9) |

Two rules follow, and both must be stated explicitly — left unsaid they get
decided by accident during implementation.

### `before_entry_written` fires for everything compaction writes

The bracket markers, the compaction entry on append **and on the mutation that
lands its content**, and every entry a plan creates. This is not a new hook; it
closes a gap that exists today, where the compaction entry bypasses entry
middleware entirely because nothing goes through the runner. An application
doing redaction or per-entry persistence currently never sees summaries.

Since `CompactionEntry` becomes the second mutable entry type, the hook's
docstring — and `docs/agent/07-middleware.md`, which embeds the mixin's source
verbatim — must say so.

### The turn hooks do NOT fire for the summarization call

`before_llm_call`, `after_llm_response`, `build_model_string` and
`build_tool_list` must not run for the compaction's own LLM request.

Those hooks are written in terms of the conversational turn —
`after_llm_response` says it "fires on every round, both tool-call rounds and
final answers". A middleware that appends a trailing reminder for the agent, or
routes the model by turn count, would silently start corrupting summarization
requests, and it has no argument telling it which call it is in. The policy
owns that request end to end: its prompt, its model, its messages.

If a genuine need to intercept the summarization request appears, that is when
`before_compaction_llm_call` gets added — with a real case behind it, per the
project's own rule about speculative hooks. Adding a "purpose" argument to the
existing hooks so middleware can discriminate is the alternative, and it is only
worth it if applications routinely want both behaviours; four changed signatures
for a hypothetical is the wrong trade today.

---

## 11. Corruption safety

**This is the part of the spec that matters most.** Compaction is the only
operation in the framework that rewrites what the conversation *is*. Getting it
wrong does not produce an error — it produces a session that loads fine and has
silently lost its history.

### The governing rule

> Everything before the transition may fail freely. The transition itself must
> contain no operation that can fail.

Compaction's expensive and failure-prone work — the LLM call — sits entirely
before that line, which is what makes the operation safe by construction rather
than by recovery logic.

Note that the lifecycle entry means there are several durable writes, but only
one of them is dangerous. Opening a bracket and appending an entry are plain
appends — exactly as safe as recording a user message, because that is all they
are. Only the transition rewrites the shape of the conversation.

Steps 1–3 are also where G6's two refusals live: a **closed** bracket is never
reopened, and an **open** bracket around an entry that already has `parts` is
never resumed. Both stop before step 4, so neither spends an LLM call.

```
1. precondition checks (closed conversational bracket)   ← may fail, nothing written
2. open the bracket, append the entry                    ← plain appends
3. stamp started_at, snapshot the path, build the `nodes` view (§5)
                                                         ← plain mutation + pure reads
4. policy produces the plan (LLM call)         (seconds) ← may fail, bracket stays open
5. G2 only: the active conversation has not moved        ← may fail, bracket closes ERRORED
6. record usage for the attempt                          ← may fail, bracket closes ERRORED
7. the rest of plan validation (§13)                     ← may fail, bracket closes ERRORED
8. preparation — all of it fallible, none of it stored:
     · recalculate the entry's context_tokens
     · build the updated compaction entry (before_entry_written)
     · build every plan-created entry (context + before_entry_written)
     · build the closing TurnFinish (context + before_entry_written)
     · mint the new conversation id + timestamps
────────────────────────── THE TRANSITION ──────────────────────────
9. plain assignments only, in any order:
     · store the updated compaction entry
     · store every created entry
     · store the TurnFinish and extend the outgoing path
     · set the outgoing conversation's status to IDLE
     · append the outgoing conversation to conversation_history
     · install the new Conversation as active_conversation
────────────────────────────── PERSIST ─────────────────────────────
10. the application writes the session                   ← may fail; memory ahead of disk
```

**Steps 5–7 are ordered, and the order is load-bearing.** Usage is recorded for
the *attempt*, so it must survive a rejection — which puts it before the rest of
validation. But `record_usage` verifies the entry is still on the active
conversation's path, so a policy that replaced `active_conversation` makes it
raise its own unrelated error and pre-empt the `CompactionPlanError` G2 exists
to produce. Hence G2 first, then the write, then everything else. All three are
inside the step's failure handling: an escape between them would leave the
bracket **open**, which is the resume state, so the next drive would replay the
same failing call — with `should_compact` never consulted, since a bracket is
already open, so the policy could not back off.

### Steps 1–8 — free to fail

The conversation's shape is untouched. A failure here closes the bracket with
`ERRORED`, `TIMED_OUT` or `CANCELLED`, leaving the path otherwise as it was —
the log gains an audit record instead of losing anything, and
`CompactionFinished` carries the outcome to every observer. This covers
provider errors, timeouts, cancellation, a policy raising, middleware raising,
and an unusable response.

A crash rather than a failure leaves the bracket **open**, which is the
recoverable state: the session derives `PENDING` on load and the next drive
resumes it. Open versus closed is the entire distinction, which is why nothing
needs to be written on the entry to record it.

Step 8 is where the closing `TurnFinish` gets *built* but not stored. That
matters: `_close_turn` runs `before_entry_written`, so closing the bracket is a
fallible operation, and a bracket that closed `COMPLETED` before a transition
that then failed would leave a summary projecting onto the unchanged path — the
same corruption a mutating policy could cause (§5). Prepare the marker, assign
it inside the transition.

### Step 9 — the only corruption window

The transition changes several fields. If it fails partway, the result is a
genuinely broken session — worst case, the same conversation is both active and
archived.

**Requirement: every fallible operation happens before the first mutation,
after which the remaining steps are plain assignments that cannot fail.** No
exception handling, no rollback, no try/finally. If it can fail it does not
belong after the commit point.

### Step 10 — memory and disk may diverge

This is the one place the design is weaker than the one it replaces. Previously
compaction wrote a *new* file, so the old one stayed valid by construction. Now
it is the same file. If the write fails, disk holds the pre-compaction session
— still valid, just stale — and a paid-for summary is lost. Recoverable by
reloading; not corrupting.

### Required guards

**G1 — Atomic session writes.** A crash mid-write leaves truncated JSON: an
unloadable session. This is the only true corruption risk in the system today
and it exists independently of compaction. **Core owns no persistence**, so
this guard lands in two places: `luca/agent/contrib/tui/sessions.py`'s
`save_session` must write to a temporary file and `os.replace` it into place
(it currently truncates in place via `path.write_text`), and the docs must
state atomic writing as an application responsibility.

**G2 — The plan must be validated against the path it was computed from.** The
policy's `compact()` awaits an LLM call. Running compaction inside the drive,
before the turn opens, means nothing can append during that window — but the
validation must not depend on that being true. The runner snapshots the active
conversation's **id and node list** before calling the policy and refuses to
commit if either changed.

**G3 — Reject empty content.** `parts` that is `None`, empty, or carries no
content at all produces a structurally valid session whose history has been
replaced by nothing. "No content" means: no text part with a non-whitespace
character **and** no non-text part. An image-only summary is legitimate content
and must not be rejected.

**G4 — At most one compaction per drive.** If `should_compact` is still true
after a compaction, the next drive handles it. Re-checking within the same
drive loops: summarizing a summary, then summarizing that. This is structural —
the compaction step sits outside the drive's loop — and with G5 gone it is the
**only** bound core places on repeated compaction. Everything past one attempt
per drive is the policy's to decline.

**G5 — removed. Whether a compaction is *worth doing* is the policy's call,
never core's.** An earlier draft had core refuse a plan whose replaced span was
only a `CompactionEntry` ("refuse to summarize a summary"). It is gone, because
it contradicted the rule the rest of this design is built on — §13's *the runner
validates structure, never meaning* — and because it was the only check that
discarded a **structurally valid plan the policy had already paid an LLM call to
produce**, reporting the result as indistinguishable from "nothing to do". The
adjacent row of §13's table already assigns the identical judgment ("the span is
too small to be worth summarizing") to the policy; there is no principled line
between the two. It was also unreliable: a span of `[ts, cmp, tf]` — the likely
real shape of "just a summary" — never triggered it, because bracket markers are
entries too.

What still bounds the degradation spiral it was aimed at: **G4** keeps it to one
compaction per drive (structural — the step sits outside the loop), a drive only
exists when there is real work, and `should_compact` belongs to the policy, which
already owns its own backoff (§5). The accepted cost is that a policy whose gauge
never comes down burns one LLM call per drive and grows `entries` and
`conversation_history` a little each time. That is waste, not damage — nothing is
deleted and every archived conversation stays intact — and the fix is
policy-shaped. **The default contrib policy must carry the floor**, which §13
already required of it.

**G6 — An interrupted compaction resumes in place; a finished one does not.**
An **open** bracket found on load means the process died mid-attempt: the next
drive retries it, reusing the same entry, with the previous attempt still
visible in `started_at`. A **closed** bracket is finished whatever its outcome
and must not be retried, or a permanently failing compaction would retry
forever. What must never happen either way is a second bracket piling up behind
the first. §8's derivation rule is the other half of this guard: without it, a
closed failed bracket derives `PENDING` and a polling loop opens a fresh
bracket every drive.

**"Interrupted" means `parts is None`, not "the bracket looks open."** An open
bracket is the *signal*; the entry's own state is the *test*. `parts` are
written at the commit point and nowhere else — every non-transition ending
leaves them `None` precisely so a failed compaction cannot project — so an
entry that has them describes a compaction that already committed. The runner
resumes only when `parts is None`; an open bracket around an entry that already
has content is not resumed, and no `compact()` call is made for it.
`schedule_compaction()` reads the same test, so on such a path it raises the
ordinary open-turn error rather than reporting itself idempotent over a
compaction that already happened.

Without this test the guard keys on bracket *shape*, which a plan can
counterfeit: a path that opens `[TurnStart, cmp, …]` reads as an interrupted
compaction, so the next drive calls `compact()` on an entry whose work is done
and overwrites its `parts`, `llm_config` and `compacted_nodes` — destroying the
record of what that summary replaced. That is the only hazard in this design
that damages an existing audit trail rather than the current path. §5's `nodes`
view removes the likeliest way to produce the shape; this removes the damage
from every remaining way, including a policy carrying an unrelated `TurnStart`
or inventing one, without core inspecting the plan's meaning.

The degradation is the ordinary phantom-open-turn hazard (§13): the bracket is
left alone, so the drive treats it as a conversational turn. Loud, and the
policy's own mistake. What survives is the compaction record.

### Recovery

After a successful compaction nothing has been destroyed:

- `conversation_history[-1]` is the exact pre-compaction path.
- Every compacted entry is still in `entries`.
- `CompactionEntry.compacted_nodes` lists precisely which ids were replaced,
  and `llm_config` plus the usage record say what it cost to produce.

A bad summary is therefore a recoverable mistake, and "undo compaction" is a
real operation should we ever want to ship it. Under the design this replaces,
the compacted entries survived only because the *old file* was still on disk
under a different name — lose track of that file and they were gone.

---

## 12. Ledger doors

The spec's transition cannot be written with the ledger as it stands.
`SessionLedger` is the single append/read door onto the entry log, and every
compaction write has to go through it or the bookkeeping it centralizes (path,
`parent_id`, `updated_at`, the execution index) starts drifting between call
sites. Three changes:

- **The update door generalizes.** `put_execution(execution)` is
  `ToolExecution`-shaped and is the only in-place mutation path. A mutable
  `CompactionEntry` needs the same door, entry-shaped.
- **Plan-created entries need their own creation path.** `append()` always
  appends to the *active* conversation's path, which is precisely what these
  entries must not do — they belong to a conversation that does not exist yet.
- **The transition is a door.** It is the only place `conversation_history` is
  ever written and the only place `active_conversation` is ever replaced, and
  it is the atomic region of §11 step 9. It must be one function with no
  fallible call inside it.

Also on the ledger: `record_usage()` needs no change. It already verifies that
the entry is on the active path, which is true of the compaction entry right up
to the transition — one more reason usage is recorded before the commit point.

`derive_status()` gains the skip rule from §8.

One read is added: **"is there a compaction to resume?"** — the entry inside an
open compaction bracket, or nothing. It answers "nothing" for three inputs that
mean the same thing to every caller: no open bracket, an open *conversational*
turn, and an open compaction-shaped bracket whose entry already has `parts`
(G6). Both callers — the drive-top step and `schedule_compaction()` — ask this
one question, which is what keeps the two from disagreeing about whether a
compaction is in flight.

## 13. Runner behavior and edge cases

### Failure taxonomy

Every way the compaction step can end. A failed compaction never transitions,
so its bracket closes on the **pre-compaction** path and `tf_c` becomes the
leaf — which is why §8's skip rule is what makes this table read correctly.

| Ending | Bracket | Raises? | `RunResult` | Derived status | Next drive |
|---|---|---|---|---|---|
| Policy returned `None` | closed `COMPLETED` | no | yes | skip → the leaf before | ordinary |
| Success, transition committed | closed `COMPLETED`, archived | no | yes | the new path's leaf | ordinary |
| Plan rejected (§13 validation, G3) | closed `ERRORED` | **`source=USER`** | — / degraded | skip → before | no retry of this entry |
| Policy raised | closed `ERRORED` | **`source=USER`** | — / degraded | skip → before | ” |
| Provider error / middleware raised | closed `ERRORED` | **`source=USER`** | — / degraded | skip → before | ” |
| LLM timeout | closed `TIMED_OUT` | **`source=USER`** | — / degraded | skip → before | ” |
| Cancelled (Scheduled→Started window, or mid-summary) | closed `CANCELLED` | no | yes | skip → before | ” |
| Crash before `started_at` | **open** | — | — | `PENDING` | resumes in place (G6) |
| Crash mid-summary | **open** | — | — | `PENDING` | resumes in place (G6) |
| Crash after the transition, before the app saved | — | — | — | disk holds the pre-compaction session | the summary is lost (§11 step 10) |

### Failures are handled by source

**A `source=USER` failure raises.** `ERRORED` and `TIMED_OUT` close the bracket
and the exception propagates through `await` / iteration, and the drive does
not continue. This is the treatment a failed turn already gets, and it is the
honest one: the user asked for a compaction, so they are told it failed.

**A `source=POLICY` failure degrades.** The bracket closes `ERRORED` /
`TIMED_OUT`, `CompactionFinished` carries the outcome and error, and the drive
**continues to the conversational turn**. Compaction is an optimization, and
failing to optimize must not destroy the user's turn.

Three consequences of degrading, all accepted:

- The next LLM call runs on the un-compacted path. If the context really was
  full, that request fails on the provider side and closes the turn `ERRORED`,
  which does raise. The user still learns something went wrong — later, and
  less precisely.
- A policy bug (a `TypeError`, say) is not propagated. It is recorded in
  `TurnFinish.error` and delivered on `CompactionFinished`, and that is the
  only place it appears. Observers that log the event stream see it; a caller
  that only awaits `run()` does not.
- Repeated failures burn one attempt per drive, indefinitely. Core deliberately
  remembers nothing about past failures; backoff belongs to the policy, which
  owns `should_compact`.

**A cancel always stops the drive**, whatever the source. The wind-down closes
the compaction bracket with the requested outcome and returns. Consuming the
`CancelRequested` and then going on to answer the queued turn would defy the
user's instruction — the cancel is against the drive, not against the
compaction alone.

### Plan validation

**The runner validates structure, never meaning.** It does not know what a turn
is, what a tool call needs, or which message must survive. Those are the
policy's judgment, and a framework that second-guesses them cannot be extended.

The runner rejects a plan that:

- **references an id that does not exist**, or one that is not among the
  `nodes` handed to `compact()` — which covers ids from an already-archived
  conversation, ids from nowhere, and the compaction bracket's own `TurnStart`
  (§5);
- **references the same id twice** — the entry would project twice;
- **is empty** — a conversation with no nodes is not a compaction;
- **omits the compaction entry itself** — the lifecycle record must be on the
  path it produced;
- **fails G2** — the active conversation is not the one the plan was computed
  against;
- **fails G3** — the content is empty.

There is no usage-key rejection: `plan.usage` is typed (§5), so a bad counter
name fails inside the policy and never reaches the runner.

Note the first rule is checked against **the offered `nodes`**, not against
`active_conversation.nodes`. The two differ by exactly one element, and that
element is the marker a plan must never carry. Reusing the rule this way is
what turns "don't carry `ts_c`" from undocumented policy trivia into an error
the author sees, with no new rule and no new judgment — the runner is checking
its own offer, not the plan's wisdom.

Everything else it commits as given. In particular, the ORDER of carried ids is
the policy's to choose: a plan may reorder or interleave them. A policy that
scrambles the path produces a nonsensical conversation, and that is a policy
bug, not something the runner prevents.

### Hazards the policy owns

These produce broken conversations. None of them are checked, because checking
them means the runner knowing what the entries mean:

- **Splitting a tool call from its result**, in either direction — a carried
  `ToolExecution` whose requesting `AssistantMessage` was summarized, or a
  carried `AssistantMessage` whose executions were. Both produce a provider-side
  400 on the very next request, so the failure is at least loud.
- **Carrying a `TurnFinish` without its `TurnStart`** — not merely malformed.
  Status derivation walks back from the leaf looking for the opener, and the
  compaction-bracket skip rule (§8) has to decide what an unanchored
  `TurnFinish` means. It means "not a compaction bracket, stop skipping", so a
  carried `TurnFinish(ERRORED)` still derives `PENDING` and is still retried —
  but only because the rule says so explicitly. The projection is fine; the
  derivation is what is at risk.
- **Carrying a `TurnStart` without its `TurnFinish`** — a *phantom open turn*.
  The new conversation derives `PENDING` (or `AWAITING_APPROVAL`, if a
  `PENDING`-approval execution came with it) and the very next thing the drive
  does is "resume" a turn that never existed, calling the model. Carrying a
  nonterminal `ToolExecution` is the same class of mistake and raises
  `ProjectionError` on the next request instead.

  The variant that used to be worse — the summary entry landing immediately
  after that carried `TurnStart`, so the pair reads as a *compaction* bracket
  and the next drive re-runs an already-committed compaction over its own
  record — is **contained from both ends now**. The likeliest way to produce it
  is gone: the compaction's own `ts_c` is not among the `nodes` the policy is
  offered, so carrying it is a plan rejection (§5). And an unrelated
  `TurnStart` that still produces the shape no longer does damage: G6 resumes
  only an entry with `parts is None`, so a committed compaction is never
  re-run. What remains is the ordinary phantom open turn above — loud, and the
  policy's.
- **Summarizing away an unanswered trailing `UserMessage`** — see below. This
  one is silent.

The default policy avoids all of these by cutting on turn brackets.

### Edge cases

**Nothing to compact — the policy returns `None`; `parts` stays `None` and the
bracket closes `COMPLETED`:**

| Case | Behavior |
|---|---|
| Empty conversation | No-op. `schedule_compaction()` on a fresh session is legal and compacts nothing. |
| The policy would carry everything over | No-op. One exchange with "keep the last two turns" leaves nothing to summarize. |
| The span to summarize is a single `CompactionEntry` | **Committed.** Core does not refuse it (G5 removed, §11); re-summarizing a summary is the policy's judgment and its floor to enforce. |
| The span is too small to be worth summarizing | Policy's judgment. One exchange summarized into a longer summary is structurally valid and strictly a pessimization; a default policy needs a floor. |

**Malformed plans — rejected by the runner; bracket closes `ERRORED`:**

| Case | Behavior |
|---|---|
| An id that does not exist, or is not among the offered `nodes` | Refused. The conversation is unchanged. |
| The compaction bracket's own `ts_c` | Refused — it is not among the offered `nodes` (§5). Same rule, no special case. |
| The same id referenced twice | Refused — it would project twice. |
| An empty plan | Refused. |
| The plan omits the compaction entry itself | Refused. |
| The active path changed under the plan | Refused (G2). |

**Hazards the policy owns — committed as given:**

| Case | Behavior |
|---|---|
| Carried tool execution orphaned from its request | Committed. Provider 400 on the next request. |
| Carried assistant tool call orphaned from its result | Committed. Provider 400 on the next request. |
| Carried `TurnFinish` without its `TurnStart` | Committed. Projects fine; the risk is status derivation — an unanchored `TurnFinish` must stop the skip rule, not swallow the path. |
| Carried `TurnStart` without its `TurnFinish` | Committed. A phantom open turn; the next drive resumes a turn that never happened. If the summary entry sits right after it the pair reads as a compaction bracket — but G6 will not re-run an entry that already has `parts`, so the record survives. |
| Carried nonterminal `ToolExecution` | Committed. `ProjectionError` on the next request — loud. |
| Carried ids out of their original order | Committed. The policy chose the path. |
| **A trailing unanswered `UserMessage` summarized away** | **Committed, and it loses user input.** |

The last one is the only silent failure in the table, and it is worth stating
in full. If the user submits a message and compaction runs at the top of the
next drive, that message is on the path and unanswered. A plan that does not
carry it over folds it into the summary: the new path ends with the summary
entry, status derives to `IDLE` rather than `PENDING`, and the drive stops
without answering. The user's question disappears with no error anywhere.

This is not exotic. A "summarize everything, keep nothing" policy — the
full-summary shape — produces it by default. It is also a hazard this design
introduces: compaction previously ran after a turn completed, when no message
was ever pending.

The framework does not prevent it, deliberately. A policy might legitimately
want to fold that message into the summary text rather than carry it. The
default policy must carry it, and this must be tested.

Note the distinct, core-owned version of the same loss that §8's skip rule
fixes: a queued message *buried behind a closed compaction bracket*. That one
is not the policy's fault and is not allowed to happen.

**The content itself:**

| Case | Behavior |
|---|---|
| The plan's `parts` are empty or contentless | Rejected (G3). Bracket closes `ERRORED`. |
| Larger than the span it replaced | Committed, but `should_compact` stays true. G4 bounds it to one attempt per drive; stopping is the policy's own backoff (§11). |
| LLM call fails or returns nothing usable | Bracket closes `ERRORED`; raises or degrades per source. |
| LLM call times out | Bracket closes `TIMED_OUT` — the runner's existing turn timeout, unchanged. |

**Lifecycle:**

| Case | Behavior |
|---|---|
| `should_compact` still true after committing | Next drive handles it (G4). |
| `schedule_compaction()` called twice | Idempotent — the second call returns the same entry id and writes nothing. |
| `schedule_compaction()` with an open conversational turn | Raises `AgentError` (§6). Nesting the bracket would corrupt `open_turn_index()`. |
| `schedule_compaction()` with no policy configured | Raises `AgentError`. |
| Scheduled *and* `should_compact` true | One compaction, `source=USER`. |
| `should_compact` true but a conversational turn is open | No compaction this drive; the turn resumes first (§7). |
| `should_compact` true but the session is `IDLE` | No compaction — `run()` on an `IDLE` session already raises "Nothing to run". Automatic compaction rides on a drive that has work; an explicit one arms the session itself. |
| `start()` with a compaction due | Decided at call time; the compaction bracket opens instead of the `TurnStart` (§7). |
| `post_message()` while a compaction is scheduled or running | Raises `AgentError` (§6). |
| Cancelled between `Scheduled` and `Started` | A real window, deliberately preserved. `CancelRequested` attaches to the open bracket, which closes `CANCELLED`; the drive returns. |
| Cancelled mid-summary | Same — identical to cancelling a turn. |
| The policy raises | Bracket closes `ERRORED`; raises or degrades per source. |
| Process dies before the drive | The entry and its open bracket survive. The next drive picks it up (G6). |
| Process dies mid-summary | The open bracket survives, derives `PENDING`, and the next drive retries in place — no second bracket (G6). |
| A previous compaction failed | Not retried: its bracket is closed (G6) and derives `IDLE` (§8). A fresh attempt happens only if `should_compact` says so on a later drive. |
| An open bracket around an entry that already has `parts` | Not resumed, no `compact()` call (G6); `schedule_compaction()` raises. Only a plan carrying turn markers can produce this; the committed record is preserved and the path is left as the phantom open turn it is. |

**After the transition:**

| Case | Behavior |
|---|---|
| Status derivation | Ordinary derivation over the new path (§8). |
| The archived conversation's status | Set to `IDLE` at the transition. `_drive` set it to `RUNNING`; archived conversations are never re-derived, so leaving it would freeze a lie. |
| The new conversation's id and timestamps | `generate_id()` and `now_ms()` — determinism principle 8. `DeterministicRunner`'s `ids` script consumes an extra draw per compaction. |
| Usage records | The compaction's own usage lands under the pre-compaction conversation id, where the request was made. The new conversation starts with no usage records for its carried entries, so aggregating a session's cost means walking `conversation_history`. |
| Turn counting | Scoped to the active conversation AND excluding compaction brackets (§14). |
| Repeated failures | Each leaves a closed bracket and an inert entry that project to nothing. Accepted. |

---

## 14. Collateral fixes

**`Entry.id` becomes `str | None` and `Entry.created_at` becomes `int | None`,**
both defaulting to `None` — see §5. All three existing placeholder sites
migrate in the same change: `ContextManager.prune_entry`,
`ToolRegistry.create_execution`'s birth-draft contract (with
`SimpleToolRegistry`, the runner's synthesized drafts, and the test doubles),
and the docstrings that document `id=""` / `created_at=0`.
`ApprovalDecision.created_at` is untouched — different model, different job.

**Turn counting** needs two fixes, both caused by this design:

- It currently counts turn markers across the entire entry store rather than
  the active conversation. The old design masked this by discarding entries;
  once entries outlive their conversation, the count silently includes every
  archived turn.
- It must exclude compaction brackets. "How many turns has this conversation
  had" means model exchanges with the user, and a compaction is not one. The
  rule is the implementation proposal's §4.3 predicate — count every bracket
  that is not a compaction bracket. (An earlier draft proposed "count only
  brackets containing an `AssistantMessage`"; that reaches the same answer for
  compaction but silently drops the open turn until its first assistant message
  lands, and drops failed turns forever, both visible to system-prompt
  callables today.)

Both change what `SessionRuntimeStatus.turn_count` reports to system-prompt
callables — after a compaction it reflects the turns visible on the new path.
That is the correct reading and a behavior change worth noting.

**`CompactionEntry` changes:** `summary: str` → `parts: list[ContentPart] | None`,
`summarized` → `compacted_nodes`, `details` → `metadata`, plus `source`,
`llm_config`, `started_at`, `ended_at`. `metadata` loses `source_session_id` —
it is the same session now — and loses `source_conversation_id` too: `metadata`
is written by the **policy** (§4's field-ownership table), and the runner never
stamps anything into it, so nothing would populate that key. A policy that
wants it is free to write it; core does not promise it. The archived
conversation is reachable through `conversation_history` regardless.

Consequences of the rename, beyond the projector:

- **`ContextManager._model_facing_text`** maps `parts` the way it maps a user
  message's, instead of reading a `str`. That is the simplification the shape
  change buys.
- **`ContextManager._media_parts` gains a `CompactionEntry` branch.** `parts`
  can carry an image, and images are counted separately from text. This is an
  addition, not a simplification.
- **`luca/agent/core/utils.py:255`** (`pretty_print`) reads `entry.summary` and
  `entry.summarized`; `tests/agent/test_utils.py` asserts the rendered output
  through its `COMPACTED_SESSION` fixture. Both break on the rename. (That file
  postdates the first draft of this spec, which is why it was missed.)
- **Saved sessions holding the old shape stop loading** — `extra="forbid"`
  turns the removed fields into a `ValidationError`. Acceptable under the
  project's "V1, not released, no compat shims" rule; stated so it is a choice
  rather than a surprise.

**`AgentSessionRunner.__eq__`** must compare `compaction_policy` via
`_equivalent`, like every other collaborator.

**`AGENTS.agent.md`** needs: design principle 4's "only mutable entry type"
claim, the `core/` file layout, the event tiers, the status-derivation rules
(principle 11), and the test-file table.

**Cleanup.** The baseline suite is **green** — `910 passed, 14 skipped`, no
ignores. An earlier draft of this document said otherwise, because
`tests/agent/contrib/compaction/` then held seven untracked files importing a
package that does not exist; that directory has since been removed. What is
left is an empty `luca/agent/contrib/compaction/` (a stale `__pycache__` and
nothing else) — delete the directory.

**Two runner subclasses must forward `compaction_policy=`**, or nothing can use
the feature: `PluginAgentSessionRunner` (`contrib/plugins/runner.py`), which is
how the TUI builds its runner (`contrib/tui/wiring.py:146`), and the test
double `DeterministicRunner` (`tests/agent/scenarios.py`). Both declare
explicit keyword lists, so a new base-class argument is invisible to them. This
is pass-through only — plugin *composition* of a policy stays out of scope.

**Contrib follow-ups** (not core, but the demo stops exercising compaction
without them):

- A `/compact` slash command in the TUI, calling `schedule_compaction()` then
  driving. The TUI's drive loop already has the right shape — it runs until
  `runner.idle()`.
- A transcript cell for `CompactionEntry`. The TUI rebuild
  (`contrib/tui/app.py:318`) is an `if/elif` chain over entry types with no
  `else`, so a resumed compacted session silently renders no summary today.
- `save_session` atomicity (G1).

## 15. Public surface

New exports from `luca.agent.core`:

| Name | Kind |
|---|---|
| `CompactionPolicy` | the contract (concrete base, override two methods) |
| `CompactionPlan` | the result value object |
| `CompactionSource` | `USER` / `POLICY` |
| `CompactionEntry` | already exported; changed shape |

New in `luca.agent.core.events`: `CompactionScheduled`, `CompactionStarted`,
`CompactionFinished`, all three added to the `AgentEvent` union.

New on `AgentSessionRunner`: the `compaction_policy=` constructor argument and
`schedule_compaction() -> str`.

Changed: `Entry.id`, `Entry.created_at`, `ToolRegistry.create_execution`'s
placeholder convention, `ContextManager.prune_entry`'s template,
`ConversationProjector.project_compaction`'s return type and
`ConversationProjector.project`'s bracket rule,
`SessionLedger`'s update / creation / transition doors and `derive_status`,
`RunResult.outcome`'s meaning, `AgentSessionRunner.__eq__`.

`CompactionPolicy.compact` takes `(session, nodes, entry)` — `nodes` is the
compactable path, the active path minus the compaction bracket's `TurnStart`
(§5). It is the contract's only positional argument that is not a session
object, and it is what plans are validated against.

## 16. Test plan

Core behavior needs core tests, and **core tests must not import contrib** —
so the compaction analogue of `FakeToolRegistry` belongs in
`tests/agent/scenarios.py`: a `FakeCompactionPolicy` with a scripted
`should_compact` and a scripted plan, plus mid-state session constants for a
scheduled compaction, an interrupted one, and a completed one.

**Core — new file `tests/agent/test_runner_compaction.py`:**

- The transition: `conversation_history[-1]` is the exact pre-compaction path,
  `entries` keeps every compacted entry, `compacted_nodes` lists exactly the
  ids that were replaced, the archived conversation's status is `IDLE`, the new
  conversation's id and timestamps come from the hooks.
- The bracket asymmetry: the archived conversation ends `[…, ts, cmp, tf]` and
  the new one begins `[cmp, …]`, with no markers carried across.
- Plan shapes committed as given: several created entries, an entry after a
  carried id, a reordered path — and every malformed plan refused.
- Every row of the failure taxonomy (§13), including the source split:
  a `USER` failure raises, a `POLICY` failure degrades and the queued turn
  still runs, a cancel stops the drive in both cases.
- `start()` compacts (the regression that motivates §7).
- No compaction while a conversational turn is open; `post_message` and
  `schedule_compaction` preconditions; idempotent scheduling.
- Crash recovery, and the three cases that must not be confused: an **open**
  bracket around an entry with `parts is None` resumes on the next drive in
  place, without opening a second; a **closed** bracket is left alone whatever
  its outcome; an **open** bracket around an entry that already has `parts` is
  not resumed and the policy is never called (G6).
- The `nodes` view: `compact()` is handed the path without the bracket's
  `TurnStart`, `plan.nodes = list(nodes)` is a legal full carry, and a plan
  carrying `ts_c` is rejected by the ordinary not-on-the-path rule.
- Usage and `llm_config` recorded, including usage recorded for a *failed*
  attempt.
- `context_tokens` recalculated when `parts` land.
- `RunResult` after a compaction-only drive, including the paths that used to
  raise `AttributeError` (leaf is the `CompactionEntry`; leaf is a carried
  `AssistantMessage`).

**Core — existing files:**

- `test_ledger.py`: the new doors, and the derivation matrix for §8 — every row
  of the skip-rule table, including stacked closed brackets and the buried
  queued message.
- `test_projection.py`: a compaction with `parts` projects a synthetic user
  message; with `parts is None` it projects nothing; and a compaction
  bracket's own `TurnFinish(CANCELLED)` projects nothing, while a
  *conversational* `TurnFinish(CANCELLED)` still projects the interrupted
  marker (§4).
- `test_context_manager.py`: `parts`-based counting, and an image-carrying
  summary counted through `_media_parts`.
- `test_runner_middleware.py`: two tests that assert the *absence* of behaviour
  as much as its presence — `before_entry_written` sees every entry a
  compaction writes including the content mutation, and `before_llm_call` /
  `after_llm_response` / `build_model_string` are **not** invoked for the
  summarization request (§10).
- `test_models.py` / `test_utils.py`: the new entry shape, and `pretty_print`
  over it.
- Because there is no `status` field, the log-state table in §4 is asserted by
  construction: a compaction that is scheduled, running, succeeded, skipped and
  failed, each distinguished only by the bracket and `started_at` / `parts`.

**Contrib** (planned with the default policy, not here): the strategy, the
gauge, the summarization call, and the trailing-unanswered-user-message case —
which the runner deliberately does not prevent, so the default policy's test is
the only thing that does.

## 17. Docs plan

The repo's rules require it, and none of it was in the first draft.

| File | Change |
|---|---|
| `docs/agent/12-compaction.md` | New. The policy contract (`compact(session, nodes, entry)` and what `nodes` is), the plan, the events, the lifecycle, the guarantees, the hazards a policy owns. |
| `docs/agent/02-data-model.md` | The new `CompactionEntry`, the second mutable entry type, `Entry.id` / `created_at` optionality, `conversation_history`. |
| `docs/agent/04-runner.md` | `compaction_policy=`, `schedule_compaction()`, the drive sequence, the `start()` rule, `RunResult.outcome`'s meaning, the status-derivation rule. |
| `docs/agent/07-middleware.md` | Embeds the mixin source verbatim — update `before_entry_written`'s docstring there too. |
| `docs/agent/10-projection.md` | `project_compaction` returning `None`, and `project()` suppressing a compaction bracket's `TurnFinish`. |
| `docs/agent/11-context-and-usage.md` | Where a compaction's usage lands and why; aggregating across `conversation_history`; atomic session writes as an application responsibility. |
| `docs/agent/README.md` | Index entry. |
| `docs/agent/contrib/compaction/README.md` | With the contrib policy, later. |

## 18. What this removes

- Building a new session, and with it the deep copies, the referent chasing,
  the index rebuild, and the four injectable id/clock parameters that existed
  only to make a non-runner-driven transform testable.
- The whole-session deep copy. "Never mutate the source" was load-bearing only
  because the old design destroyed history; with history preserved there is
  nothing to protect, and the copy grows more expensive on every compaction.
- Compaction as an operation the application performs on a session. It becomes
  an operation the runner performs during a drive.
- Any new conversation status. `ConversationStatus` is untouched — one
  derivation rule is added, no value is.
- Any parallel vocabulary for how an attempt ended, how it is cancelled, or how
  it times out. `TurnOutcome` and `CancelRequested` already cover all three.
- Any new middleware hooks. `before_entry_written` covers mutation, the policy
  covers behaviour, the events cover observation (§10).

## 19. Decisions

Recorded so they are not re-litigated. Each was an open question in the first
draft.

1. **Durable scheduling, and `post_message` pays for it.**
   `schedule_compaction()` writes a bracket and an entry, so posting a message
   is illegal until the compaction is driven — durably, across a reload.
   Accepted as consistent with every other open bracket; applications schedule
   immediately before driving (§6).
2. **`Entry.id: str | None` *and* `Entry.created_at: int | None`, migrated
   everywhere.** One convention for "uncommitted", including the public
   registry contract. Measured cost, and it removes placeholder noise from
   application-facing API (§5).
3. **The plan carries a filled copy of the entry**, plus the usage counters.
   The policy never writes to the session, which makes the
   summary-injected-without-a-compaction corruption unreachable (§5).
4. **`RunResult.outcome` is "how the last bracket this run closed ended"**, and
   `status` is derived rather than hardcoded. No new field (§7).
5. **`derive_status()` skips a closed compaction bracket.** The one derivation
   rule this design adds, and it is required: without it a failed compaction is
   retry-ready (a spin) and a closed bracket buries a queued user message
   (silent input loss) (§8).
6. **Runner-side plan rejections raise**, like any failed turn — subject to (7).
7. **Failures split by source:** a `USER` compaction raises, a `POLICY`
   compaction degrades and lets the turn proceed, a cancel always stops the
   drive — **whatever outcome the cancel carried.** `cancel()` takes the
   outcome as an argument and only `COMPLETED` is forbidden, so
   `cancel(ERRORED)` and `cancel(TIMED_OUT)` must end the drive exactly as
   `cancel(CANCELLED)` does; the runner keys that on having consumed the
   cancellation, never on the outcome's value. Compaction is an optimization
   and must not cost the user their turn; the price is that a policy bug
   surfaces only on the event stream (§13).
8. **`ended_at` stays on the entry**, despite duplicating
   `TurnFinish.created_at` within microseconds. The entry carries into the new
   conversation without its bracket, and it has to be self-describing there —
   the same reason `started_at` is a field (§4).
9. **`plan.usage` is a typed `UsageCounters`** rather than a `dict[str, int]`,
   and the usage write is ordered after G2 and inside the step's failure
   handling (§5, §11). This was an open question in the previous draft; it is
   closed because the untyped dict put its only validation inside a
   runner-internal write that cannot produce a clean plan rejection.
10. **Projection is positional: a compaction bracket projects as nothing, a
    bare `CompactionEntry` projects its `parts`** (§4). One rule on
    `ConversationProjector.project()`, not a content test per marker. It is
    required, not tidiness — without it a cancelled compaction tells the model
    "[Request interrupted by user]" about a question the model was never
    shown. It also stops an archived conversation from projecting both the
    original history and a summary of it. The per-entry methods have no path
    and cannot classify the bracket they sit in; `project()` already owns
    path-level policy, so no override signature changes.
11. **`compact()` is handed the path it may rewrite** — `(session, nodes,
    entry)`, where `nodes` is the active path minus the compaction bracket's
    `TurnStart` (§5). A policy author never learns the bracket exists, and
    plans are validated against that view, so carrying `ts_c` is caught by the
    existing "not on the path" rule instead of being undocumented trivia. The
    compaction entry stays *in* the view: stripping it too would make
    `plan.nodes = list(nodes)` illegal, trading one hidden requirement for
    another. A `tuple`, not a `Conversation` — a filtered `Conversation` would
    be an identity-bearing object claiming to be the live one while not being
    it, and G2 compares plans against the real one.
12. **G6 resumes on `parts is None`, not on bracket shape** (§11). A plan can
    counterfeit an open compaction bracket; it cannot counterfeit a committed
    entry's content. This is what stops an already-committed compaction from
    being re-run and its `compacted_nodes` overwritten — the one hazard that
    damages an existing audit trail rather than the current path. Decision 11
    removes the likeliest way to produce the shape; this removes the damage
    from every remaining way, with core still never inspecting a plan's
    meaning.

Still open, deliberately:

- **Undo.** The data supports it. Whether a command ships is a separate call.
- **Plugin composition of a compaction policy** — a fourth
  `PluginAgentSessionRunner` collaborator, when a real case turns up. (Note
  this is composition only; the plain `compaction_policy=` pass-through is
  required and is in §14.)

Resolved during review, recorded because it was briefly open:

- **The full-carry plan.** A plan that carries every node over has an empty
  compacted span, and G5 as drafted was vacuously true for one — so it would
  have been discarded as "nothing to compact", contradicting decision 10 and its
  test. Removing G5 (§11) dissolves the question: with no worth-doing check in
  core, a full-carry plan commits with `compacted_nodes == []`, exactly as
  decision 10 says, and `[]` vs `None` keeps the meaning the field's type exists
  to record.
