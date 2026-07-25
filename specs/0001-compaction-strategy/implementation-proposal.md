# Compaction — implementation proposal

Companion to `prd.md`. The PRD says *what* the contract is; this says *where
every line goes, who is allowed to know what, and exactly which tests pin it*.

Two rules govern the whole document:

1. **Core triggers, stamps, and archives. Nothing else.** The runner opens a
   bracket, calls one policy method, validates the shape of what comes back,
   and swaps one conversation for another. It never decides what to summarize,
   how, with which model, when it is worth it, or whether to try again. The
   ledger's transition door does not even learn that compaction exists — it
   archives one conversation and installs another, and could serve a fork or a
   branch unchanged. What the ledger *does* know is the entry type and the
   bracket shape, because its reads answer questions about them (§4.3); what it
   never knows is what a policy is or how a summary was produced.
2. **A strategy author's unit is the strategy.** Everything the framework does
   around `compact()` is pinned by core tests listed in §11 and published as a
   numbered invariant list (§9). A policy author implements `compact()`, tests
   `compact()`, and treats the session/conversation handling as given.

---

## 1. Responsibility map

| Module | Owns | Must never know |
|---|---|---|
| `core/models.py` | `CompactionEntry` (durable shape), `CompactionSource`, `Entry` identity optionality | anything runtime |
| `core/compaction.py` **(new)** | `CompactionPolicy` contract, `CompactionPlan` value object, `validate_plan()` (pure), the G3 content predicate | the session's mutation, the ledger, asyncio, events |
| `core/ledger.py` | `put_entry` (generalized update door), `transition_conversation` (the atomic swap), `open_compaction_entry` (read, incl. G6's `parts` test), the `derive_status` skip rule | what a policy is; how a summary is produced; why a conversation is being swapped |
| `core/runner.py` | bracket lifecycle, `started_at`, building the offered `nodes` view, the `compact()` call under cancel/timeout, plan validation call, preparation of every fallible artifact, the commit call, failure-by-source, the three events, `schedule_compaction()` | prompts, models, node selection, thresholds, backoff |
| `core/events.py` | three observational events | — |
| `core/projection.py` | `parts` → synthetic user message, `None` → nothing; `project()` suppresses a compaction bracket's `TurnFinish` | why a bracket is a compaction bracket (it consumes the shared predicate) |
| `core/context_manager.py` | `parts`-based counting incl. images | — |
| `contrib/<policy>` | thresholds, gauge, window size, prompt, model, which nodes to keep, backoff | the transition, the ledger, the bracket |

**The enforcement mechanism is narrow reads.** Core reads exactly five things
off a returned plan: `entry.parts`, `entry.llm_config`, `entry.metadata`,
`nodes`, `usage`. Every other field on the returned entry copy is discarded.
That is the whole coupling surface, and it is the reason no compaction policy
can ever need a core change.

**And a narrow write.** In the other direction core hands the policy exactly
three things: the live session, the `nodes` it may rewrite, and a deep copy of
the entry. `nodes` is the surface that keeps the framework's own bookkeeping
out of a policy author's head — it is the active path minus this compaction's
`TurnStart`, and it doubles as the set carried ids are validated against
(PRD §5).

**New/changed file inventory**

| File | Change | Rough size |
|---|---|---|
| `core/compaction.py` | new | ~140 lines |
| `core/models.py` | `CompactionEntry` reshape, `CompactionSource`, `Entry.id`/`created_at` optional | ~40 lines net |
| `core/ledger.py` | 2 doors, 1 read, skip rule | ~110 lines |
| `core/runner.py` | compaction step + prepare/commit split + `schedule_compaction` + `_build_run_result` + `_closed_outcome` (§5.1a) | ~245 lines |
| `core/events.py` | 3 events + union | ~55 lines |
| `core/projection.py` | `project_compaction` + `project()`'s bracket rule (§7) | ~30 lines |
| `core/context_manager.py`, `core/utils.py`, `core/exceptions.py`, `core/__init__.py` | small | ~30 lines total |
| `tests/agent/scenarios.py` | `FakeCompactionPolicy` + 7 literals | ~250 lines |
| `tests/agent/test_compaction.py` | new (unit: contract + validator) | ~350 lines |
| `tests/agent/test_runner_compaction.py` | new (integration) | ~1400 lines |
| existing tests | ledger/projection/context/models/utils/middleware additions | ~450 lines |

---

## 2. Data model (`core/models.py`)

```python
class CompactionSource(str, Enum):
    USER = "user"
    POLICY = "policy"


class CompactionEntry(Entry):
    """The durable, MUTABLE lifecycle record of one compaction attempt — the
    second mutable entry type alongside `ToolExecution`. Written when the
    intent exists, mutated in place as it progresses, left in its terminal
    state whether it succeeded or not. There is deliberately no `status`
    field: the turn bracket owns how the attempt ended (`TurnFinish.outcome`
    + `error`) and these fields own what it produced, so nothing can
    disagree.

    The entry carries into the new conversation while its bracket stays
    behind, so it must be self-describing wherever it is read — which is why
    `started_at` / `ended_at` live here even though the bracket records the
    same instants."""

    type: Literal["compaction"] = "compaction"
    source: CompactionSource
    parts: list[ContentPart] | None = None       # None → nothing produced yet
    compacted_nodes: list[str] | None = None     # None → nothing replaced
    llm_config: LLMConfig | None = None          # what produced the content
    started_at: int | None = None
    ended_at: int | None = None
    metadata: dict = Field(default_factory=dict)
```

`Entry.id: str | None = None` and `Entry.created_at: int | None = None`, both
meaning "not yet committed".

### The identity migration is its own commit, landed first

Sites, all in one change, suite green at the end:

| Site | Now | After |
|---|---|---|
| `context_manager.prune_entry` | `PrunedEntry(id="", created_at=0, …)` | omit both |
| `runner._birth_draft` (×2 synthesized drafts) | `id="", created_at=0` | omit both |
| `tool_registry.py` docstring | "placeholder identity (`id=""`, `created_at=0`)" | "no identity — `id`/`created_at` stay `None`" |
| `contrib/simple_tool_registry/registry.py:69` | `id="", created_at=0` | omit both |
| `tests/agent/scenarios.py` `FakeToolRegistry.draft` | `id="", created_at=0` | omit both |
| `tests/.../test_simple_tool_registry.py` (×3), `test_runner_tool_output.py:155`, `test_context_manager.py:174` | assert `id=""` | assert `id is None` |
| `utils.py:453` `_timestamp` | `entry.created_at / 1000` | guard `None` → `"—"` |

`ApprovalDecision.created_at` is untouched (a self-stamping value object, not
an `Entry`). `ledger.put_entry` rejects `id is None` — that is the one place
where an uncommitted template could silently become durable.

### `SessionRuntimeStatus.turn_count` — deviation from the PRD

The PRD proposes "count only brackets containing an `AssistantMessage`". Use
**"count the brackets on the active conversation that are not *compaction
brackets* under §4.3's adjacency predicate"** instead — the node right after
the `TurnStart`, never "contains a `CompactionEntry` anywhere in the span",
for the reasons §4.3 gives. A bare open `TurnStart` is conversational and
still counts. Same result for compaction (a compaction bracket never counts),
but it does not change the number for any existing session: the
PRD's rule silently drops the open turn from the count until the first
assistant message lands, and drops failed turns forever, and both are visible
to system-prompt callables today. Scoping to the active conversation is
unchanged from the PRD and is required (entries now outlive their
conversation).

---

## 3. The contract module (`core/compaction.py`)

Why its own module: `models.py` is declarative-only, and `CompactionPlan` is
transient (like `RunResult`, which lives in `runner.py`). The precedent is
`tool_registry.py` — the contract next to nothing else.

```python
class CompactionPolicy:
    """The single extension point for compaction. A concrete base: subclass
    and override the two methods. Core never learns that "which nodes to
    keep" is a decision — at the transition the kept set is a list of ids."""

    def should_compact(self, session: AgentSession) -> bool:
        """SYNC so `start()` can consult it at call time. The threshold, the
        context sum and the window size are all yours; core has no
        context-total API. Return False to back off — core remembers nothing
        about previous failures."""
        return False

    async def compact(
        self,
        session: AgentSession,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        """Fill in a DEEP COPY of `entry` and describe the resulting
        conversation.

        `nodes` is THE PATH YOU MAY REWRITE: the active conversation's path
        with this compaction's own `TurnStart` removed, ending with `entry`.
        `plan.nodes` may carry any of these ids, in any order, with new
        entries interleaved — and nothing else; an id outside this tuple is a
        plan rejection. `plan.nodes = list(nodes)` is a legal full carry.

        `None` means nothing to do. You own the LLM call end to end; the turn
        middleware hooks do not fire for it. Never mutate `session` — the
        runner refuses to commit if the active path moved under you (G2)."""
        raise NotImplementedError()


class CompactionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: CompactionEntry                # the copy, filled in
    nodes: list[Union[AnyEntry, str]]     # the new path: entry → create here,
                                          #   str → carry this node here
    usage: UsageCounters = Field(default_factory=UsageCounters)
```

`usage` is a **typed model, not a `dict[str, int]`.** The counter names are a
closed set, and `Usage` is `extra="forbid"`, so a policy passing a provider's
own vocabulary (`prompt_tokens`, `completion_tokens`) would otherwise raise
from inside `record_usage` — a runner-internal write, on the one line of the
step that cannot be a clean plan rejection (§5.4). A typed field moves that
failure into the policy's own code, at plan construction, where the author can
see it:

```python
class UsageCounters(BaseModel):
    """The counters a policy reports for its summarization call. Field names
    and semantics are `Usage`'s, minus the ids the ledger owns — one
    vocabulary, and the same shape the runner's `_to_usage_counters()`
    produces. `extra="forbid"` is the point: a provider's own counter names
    fail here, in the policy, not later inside the ledger."""

    model_config = ConfigDict(extra="forbid")

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
```

The runner passes it through as `record_usage(entry.id, **plan.usage.model_dump())`.

Plus, in the same module:

```python
def has_content(parts: list[ContentPart] | None) -> bool:
    """G3: a text part with a non-whitespace character, or ANY non-text part.
    An image-only summary is legitimate content."""

def validate_plan(
    plan: CompactionPlan,
    *,
    entry_id: str,
    session: AgentSession,
    snapshot: ConversationSnapshot,
) -> None:
    """Structure only — never meaning. Raises `CompactionPlanError`."""
```

`ConversationSnapshot` is a frozen value the runner takes before calling the
policy, and it carries **both** paths:

```python
class ConversationSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    nodes: tuple[str, ...]        # the FULL active path, incl. ts_c and cmp — G2's input
    offered: tuple[str, ...]      # nodes minus the bracket's TurnStart — what the
                                  #   policy was handed, and what carried ids are
                                  #   checked against (PRD §5)
```

Two fields rather than one because the two questions are different. G2 asks
*did the session move under the plan?* — that is about the live path, markers
included. Rule 5 asks *is this id something I offered?* — that is about the
view. Deriving one from the other at validation time would work, but storing
both keeps the runner from re-deriving the strip in a second place, and the
strip is where a bug would be silent.

`offered` is what the runner passes to `compact()` as its `nodes` argument, so
there is exactly one construction of it.

**The rejection list, in evaluation order** (each a distinct message, each its
own unit test):

| # | Rule | Message shape |
|---|---|---|
| 1 | G2: `snapshot.id != session.active_conversation.id` | `the active conversation changed under the plan (…→…)` |
| 2 | G2: `snapshot.nodes != tuple(active.nodes)` | `the active conversation's path changed under the plan` |
| 3 | `nodes` is empty | `an empty plan is not a compaction` |
| 4 | a carried id is not in `session.entries` | `plan references unknown entry 'x'` |
| 5 | a carried id is not in `snapshot.offered` | `plan references entry 'x', which is not on conversation 'c1'` (covers ids from an archived conversation, ids from nowhere, and the bracket's own `ts_c` — PRD §5) |
| 6 | a carried id appears twice | `plan references entry 'x' twice` |
| 7 | `entry_id not in` carried ids | `plan omits the compaction entry 'cmp'` |
| 8 | G3: `not has_content(plan.entry.parts)` | `plan carries no content` |

There is **no usage-key rule.** An earlier draft had one; typing `plan.usage`
(above) makes it unreachable — a bad counter name cannot survive plan
construction, so the runner never sees one to reject.

Not checked, deliberately (the runner validates structure, never meaning):
node order, tool-call/result pairing, orphaned brackets, nonterminal executions
carried over, a summary larger than the span, whether the trailing user message
survives. Those are §10's hazards.

`CompactionPlanError(AgentError)` goes in `exceptions.py`. It is what the
`source=USER` path raises to the caller and what `TurnFinish.error` records.

**Consequence to accept, stated once:** G3 makes "trim without summarizing"
impossible through a plan — a policy that wants to drop history with no
summary must emit a marker part (`"[earlier messages dropped]"`) or return
`None`. That is the price of the guard, and it is the right side of the trade.

---

## 4. Ledger (`core/ledger.py`)

Two doors and one read. The PRD lists three doors; plan-created entries do not
need one, because storing them inside the transition keeps every write in the
atomic region instead of half of them outside it.

### 4.1 `put_entry` — the generalized update door

```python
def put_entry(self, entry: AnyEntry) -> AnyEntry:
    """Store a fully formed replacement for an entry that is ALREADY in the
    store, and touch `Conversation.updated_at`. The only in-place mutation
    door; the caller owns building the copy and threading
    `before_entry_written`. Refuses an uncommitted entry (`id is None`) and
    an unknown id — an update door must never create."""
```

`put_execution` is deleted; `runner._persist_execution` becomes a thin wrapper
that adds the `ToolExecution`-only `updated_at` stamp (see §5.2). `updated_at`
is **not** promoted to `Entry`: it is tool-execution bookkeeping, and
`CompactionEntry` is written twice at most.

### 4.2 `transition_conversation` — the atomic swap

Deliberately compaction-agnostic. The ledger's job is "archive the active
conversation, install a new one, store these prepared entries" — it never
mentions compaction, summaries, or policies.

```python
def transition_conversation(
    self,
    *,
    updates: list[AnyEntry],     # already-stored entries, replaced in place
    created: list[AnyEntry],     # brand-new entries, identity already stamped
    closing: AnyEntry | None,    # appended to the OUTGOING path (a TurnFinish)
    nodes: list[str],            # the NEW conversation's path
    ts: int,
) -> Conversation:
    """The only writer of `conversation_history` and the only replacer of
    `active_conversation`. §11's atomic region: every fallible operation
    happens in the precondition block BEFORE the first mutation, after which
    the remaining steps are plain assignments. No try/except, no rollback —
    if it can fail it does not belong after the commit point."""
```

Body, in order:

```
PRECONDITIONS (fallible, nothing written yet)
  - every `updates` entry has an id present in session.entries
  - every `created` entry has id is not None and NOT in session.entries
  - `closing`, if given, has an id not in session.entries
  - conversation_id = self.gen_id()
─────────────────────── COMMIT POINT ───────────────────────
  - session.entries[e.id] = e            for e in updates
  - session.entries[e.id] = e            for e in created
  - index each created ToolExecution into session.tool_executions
  - if closing: session.entries[closing.id] = closing
               outgoing.nodes.append(closing.id)
  - outgoing.updated_at = ts
  - outgoing.status = ConversationStatus.IDLE
  - session.conversation_history.append(outgoing)
  - session.active_conversation = Conversation(
        id=conversation_id, nodes=list(nodes),
        created_at=ts, updated_at=ts, status=IDLE)
  - session.active_conversation.status = self.derive_status()
  - return session.active_conversation
```

Three notes. Indexing a created `ToolExecution` is there so the door cannot
drift from `append`, even though no sane policy creates one. `outgoing` is read
once at the top into a local, so the append-then-replace order cannot alias.
`derive_status()` is a pure read over ids that plan validation already proved
resolvable, so it cannot fail — and installing a conversation with a status
nobody derived would freeze a lie, exactly the bug the explicit `IDLE` on the
outgoing side fixes.

### 4.3 Reads

```python
def open_compaction_entry(self) -> CompactionEntry | None:
    """The RESUMABLE `CompactionEntry` inside the open bracket, or None.

    `None` means there is no compaction to resume, for any of three reasons:
    no open bracket; an open CONVERSATIONAL turn; or an open compaction-shaped
    bracket whose entry already has `parts` — a compaction that committed and
    whose markers a plan carried, not an interrupted attempt (G6). Callers
    treat all three identically, which is why they are one return value."""
```

**G6's content test lives here, not in the runner.** An earlier draft put it in
the compaction step as an extra arm. Here it is one condition serving both
callers, and it fixes `schedule_compaction()` as a side effect: with the test
in the step, `schedule_compaction()` on a counterfeited bracket would find the
committed entry, report itself idempotent, return that id — and then the drive
would refuse to run it, so the call silently did nothing. With the test here,
`open_compaction_entry()` returns `None`, the closed-bracket guard sees the
open bracket, and the call raises `AgentError` like any other open turn. Honest
in both places, one condition.

Note what this deliberately does **not** touch: `derive_status`'s skip rule and
`turn_count` still use the raw §4.3 predicate. A counterfeited bracket is open,
so the skip rule (which only skips *closed* brackets) never sees it, and it
derives `PENDING` — correct, because the drive is going to treat it as the
phantom conversational turn it is.

**One definition, used in four places:** a bracket is a *compaction bracket*
iff the node **immediately after** its `TurnStart` is a `CompactionEntry`;
every other bracket is conversational. A bare `TurnStart` is conversational,
and so is a bracket with no preceding `TurnStart` to anchor it (a carried
`TurnFinish` orphaned from its opener — §10's hazard). This single predicate
drives the resume check, the `derive_status` skip rule, `turn_count` (§2), and
§7's projection rule.

**It lives in `models.py`**, as a plain module-level function over
`(nodes, entries, turn_start_index)`, not as a `SessionLedger` method. Its four
consumers are the ledger, `SessionRuntimeStatus.turn_count` (already in
`models.py`), the projector and the runner — and `projection.py` does not
import `ledger.py` today, so putting it on the ledger would either add that
edge or fork the definition, which is the one thing this section exists to
prevent. `models.py` is upstream of all four.

**Adjacency, not "contains".** `_open_compaction_bracket` (§5.3) is two
synchronous appends with nothing between them, so `[ts_c, cmp]` is adjacent by
construction and the test is exact. "Contains a `CompactionEntry` anywhere in
the span" would also be true today, but it couples a framework-internal
classification to policy behavior: the compaction entry **outlives its
bracket** and lands wherever `plan.nodes` puts it (§5 of the PRD advertises
that freedom). A policy that carries a turn marker without its pair — already
listed as a hazard — can therefore drop a summary inside an ordinary bracket,
and "contains" would misread that bracket as a compaction: `derive_status`
would skip it, `turn_count` would omit it, and an open one would make the next
drive "resume" a compaction that already committed, re-calling the policy and
overwriting the committed entry's `compacted_nodes`. Adjacency costs nothing
and removes the coupling. It does not make the hazard harmless — see §9.

### 4.4 `derive_status` — the skip rule

Insert **before** the existing closed-turn rules, after the open-bracket rules:

```
while the trailing nodes form a CLOSED compaction bracket:
    drop that whole span from consideration
if nothing remains: IDLE
then apply the existing closed-turn rules to the remaining leaf
```

Open compaction brackets are untouched: they hit the existing open-turn rules
(`PENDING`, or `CANCELLING` with an unconsumed cancel). `AWAITING_APPROVAL`
is unreachable inside a compaction bracket — there are no approvals in one.

Implementation: walk back from the leaf; if it is a `TurnFinish`, find the
nearest preceding `TurnStart`; if the node right after that `TurnStart` is a
`CompactionEntry` (§4.3), shorten the working slice to before the `TurnStart`
and repeat. **No preceding `TurnStart` → not a compaction bracket → stop
skipping** and fall through to the existing closed-turn rules; an orphaned
`TurnFinish` is a policy hazard, not a licence to swallow the whole path.
Operates on a slice index, never on `conversation.nodes` itself.

---

## 5. Runner (`core/runner.py`)

### 5.1 The drive sequence

```python
async def _drive(self, streaming, context, token):
    self._set_status(ConversationStatus.RUNNING)      # moved above the step

    async for event in self._compaction_step(token):  # (1)
        yield event
    if self._compaction_consumed_cancel:              # (2) a cancel ends the drive
        return
    if (                                              # (3) compaction was the
        self._compaction_ran                          #     whole drive — ONLY
        and self.ledger.derive_status() == ConversationStatus.IDLE
    ):
        return

    self._set_status(ConversationStatus.RUNNING)      # again: a transition
                                                      #   installed a new one
    self._ensure_open_turn()                          # (4) then drive normally
    for event in self._recover_orphans():
        yield event
    while True:
        ...                                           # unchanged
```

Four things about this ordering, three of which the PRD leaves implicit:

- **(2) is how a cancel ends the drive**, and it is a flag set by the step, not
  an inference from the outcome. `_compaction_consumed_cancel` (reset in
  `_begin_run` alongside `_closed_outcome`; see §5.1a) is set at each of the two
  places the step consumes a `CancelRequested`. Testing
  `_closed_outcome == CANCELLED` instead would be wrong: `cancel()` takes the
  outcome as an argument and `CancelRequested` forbids only `COMPLETED`
  (`models.py:451`), so `cancel(TurnOutcome.ERRORED)` and
  `cancel(TurnOutcome.TIMED_OUT)` are legal — they would consume the
  cancellation, close the bracket, and then fall through into the
  conversational turn and call the model, which is the one thing a cancel must
  never allow. Consuming the cancel and then answering the queued turn defies
  the instruction — the cancel is against the drive, not the compaction.
- **(3) is required and is missing from the PRD.** A `source=USER` compaction
  on an otherwise-finished session is `PENDING` only because its bracket is
  open. Once the bracket closes, the derived status is `IDLE` and there is
  nothing to drive — without this check `_ensure_open_turn` would open a
  bracket and call the model with no user input. It is the mid-drive form of
  the existing "Nothing to run" rule. For a `source=POLICY` compaction the
  check never fires: the drive only exists because there was other work
  (a queued message or a retry-ready failed turn), and both survive the skip
  rule.
  **It is gated on the step having closed or committed a bracket**, not run
  unconditionally: `_compaction_step` returns without doing anything for the
  three common cases (no policy configured, an open conversational turn,
  `should_compact` False), and on those drives the check must not be able to
  end the drive at all. `_begin_run` gates on the *cached* status while (3)
  reads the *derived* one, so an unconditional check turns any drift between
  the two into a drive that silently returns having done nothing. Gating costs
  a bool and makes every non-compaction drive provably unchanged.
- **`RUNNING` is set before the step**, which is exactly why the transition
  must set the outgoing conversation's status to `IDLE` explicitly — archived
  conversations are never re-derived. **Re-set it after (4)**: a committed
  transition installs a new conversation whose status came from
  `derive_status()`, and nothing puts it back to `RUNNING`, so the turn that
  follows would drive a conversation reporting `PENDING`. Harmless durably —
  `RUNNING` is never trusted across a reload — but an application polling
  status mid-drive (the TUI's `while not runner.idle()` loop) would see the
  wrong one.
- **G4 needs no flag.** The step sits outside `while True`, so at most one
  compaction per drive is structural.

### 5.1a Three pieces of per-run runner state, one of which does not exist yet

```python
self._closed_outcome: TurnOutcome | None = None   # NEW — see below
self._compaction_consumed_cancel: bool = False
self._compaction_ran: bool = False
```

All three are per-run and are reset in `_begin_run`, next to the existing
token/context setup. The two compaction flags are new by construction. The
first is the trap:

**`_closed_outcome` does not exist today.** `grep` finds it nowhere in `luca/`
or `tests/`; `_close_turn` (`runner.py:1586`) only appends the marker and calls
`_refresh_status()`. So §5.7's `RunResult` change is not "read a field instead
of the path" — the field has to be built first, and it is **not** compaction
surface: `_build_run_result` stops reading `nodes[-1].outcome` for *every* run,
so `_closed_outcome` must be set at every close or ordinary turns lose their
outcome. `_close_turn` is the single writer for all of them — the normal
COMPLETED close, the wind-down, the hard-max-steps close, and the LLM-failure
close all funnel through it — so one assignment there covers the existing
paths, plus the explicit one in `_commit` for the transition, where
`_close_turn` never runs (§5.5).

Reset-per-run matters as much as the write: a handle that stops at an approval
pause and is later re-driven, or a suspended lazy run resumed with a fresh
`run()`, must not report the previous bracket's outcome. `None` is the correct
value for a run that closed nothing, and `RunResult.outcome` is already
`TurnOutcome | None`.

Budget: ~15 lines in `runner.py` beyond the compaction work, touching
`__init__`, `_begin_run`, `_close_turn` and `_build_run_result`. It lands with
step 5 of §12 because `_build_run_result` cannot change without it.

### 5.2 The prepare/commit split — the seam that makes §11 work

`_append` today does four things in one: build, calculate context, run
`before_entry_written`, commit. Compaction needs the first three without the
fourth. Split it:

```python
def _complete_entry(self, entry: AnyEntry) -> AnyEntry:
    """Context calculation (always BEFORE middleware, never after) then
    `before_entry_written`. The fallible half of preparing any entry."""
    entry.context_tokens = self.context_manager.calculate_context(entry)
    return self._run_middlewares("before_entry_written", entry)

def _append(self, build_fn) -> AnyEntry:                   # unchanged behavior
    return self.ledger.append(
        lambda i, p, t: self._complete_entry(build_fn(i, p, t)))

def _prepare(self, build_fn, parent_id: str | None, ts: int) -> AnyEntry:
    """A complete, fully-middlewared entry that is NOT committed. Identity
    comes from the hooks; the caller supplies the parent because the new
    conversation's leaf is not the active path's leaf."""
    return self._complete_entry(build_fn(self.generate_id(), parent_id, ts))

def _persist_entry(
    self, entry: AnyEntry, *, recalculate: bool = True, **changes,
) -> AnyEntry:
    """In-place update: copy, complete, store. The one update path.
    `recalculate=False` skips the context calculation and runs middleware
    only — see the note below."""
    updated = entry.model_copy(update=changes)
    updated = (
        self._complete_entry(updated)
        if recalculate
        else self._run_middlewares("before_entry_written", updated)
    )
    return self.ledger.put_entry(updated)

def _persist_execution(self, execution, **changes):        # thin wrapper
    return self._persist_entry(execution, **changes, updated_at=self.now_ms())
```

One behavior change worth naming: `_persist_entry` recalculates
`context_tokens` on every update, where `_persist_execution` previously left it
to `_finalize_outcome`. `_finalize_outcome` already recalculates before
`after_tool_execution`, and recalculating again on the final persist is
idempotent for the same content — but to keep "middleware has the final say"
literally true, `_persist_entry` takes a `recalculate: bool = True` keyword and
`_finalize_outcome`'s final persist passes `recalculate=False`. That is the
only way the existing middleware-final-say tests stay honest.

### 5.3 `schedule_compaction()`

```python
def schedule_compaction(self) -> str:
    """Open a compaction bracket and write a `CompactionEntry(source=USER)`.
    Returns the entry id. Idempotent. Requires a CLOSED bracket."""
    if self.compaction_policy is None:
        raise AgentError("no compaction_policy is configured")   # nothing written
    existing = self.ledger.open_compaction_entry()
    if existing is not None:
        return existing.id                                       # idempotent
    if self.ledger.open_turn_index() is not None:
        raise AgentError(
            "schedule_compaction requires a closed turn "
            f"(status={self.status.value})."
        )
    entry = self._open_compaction_bracket(CompactionSource.USER)
    self._set_status(ConversationStatus.PENDING)
    return entry.id
```

The closed-bracket guard is not cosmetic: `open_turn_index()` walks back to the
*nearest* `TurnStart`, so a nested bracket would make the conversational turn's
eventual `TurnFinish` close the wrong one.

```python
def _open_compaction_bracket(self, source) -> CompactionEntry:
    """TurnStart then CompactionEntry — two plain appends, exactly as safe as
    recording a user message. Emits nothing; the drive emits
    `CompactionScheduled`."""
    self._append(lambda i, p, t: TurnStart(id=i, parent_id=p, created_at=t))
    return self._append(
        lambda i, p, t: CompactionEntry(
            id=i, parent_id=p, created_at=t, source=source))
```

`post_message` needs no change; the durable open bracket makes it raise while a
compaction is scheduled or in flight, across a reload, which is the treatment
every open bracket already gets.

### 5.4 The compaction step

```python
async def _compaction_step(self, token) -> AsyncIterator[AgentEvent]:
    if self.compaction_policy is None:
        return

    entry = self.ledger.open_compaction_entry()

    # 1) FLUSH FIRST. A parked cancel inside an open compaction bracket ends
    #    it now — no Scheduled, no Started, no policy call.
    if entry is not None:
        cancel = self.ledger.open_turn_cancel_requested()
        if cancel is not None:
            self._compaction_consumed_cancel = True      # → the drive returns
            yield self._close_compaction(entry, cancel.outcome, cancel.error)
            return

    # 2) RESUME, SKIP, or DECIDE.
    if entry is None:
        if self.ledger.open_turn_index() is not None:
            return                      # open CONVERSATIONAL turn → not this drive
        if not self.compaction_policy.should_compact(self.session):
            return
        entry = self._open_compaction_bracket(CompactionSource.POLICY)
    self._compaction_ran = True      # past every early return → (3) may fire
    yield CompactionScheduled(entry=entry.model_copy(deep=True))

    # 3) RUN IT.
    if entry.started_at is None:                     # first attempt only
        entry = self._persist_entry(entry, started_at=self.now_ms())
    yield CompactionStarted(entry=entry.model_copy(deep=True))

    snapshot = self._snapshot_conversation()         # full path + offered view
    try:
        plan = await self._invoke_policy(entry, snapshot.offered, token)
    except _CompactionCancelled as stop:
        yield self._close_compaction(entry, stop.outcome, stop.error)
        return
    except Exception as exc:
        yield self._close_compaction(entry, _outcome_for(exc), str(exc))
        if entry.source == CompactionSource.USER:
            raise
        return                                                  # POLICY degrades

    if plan is not None:
        try:
            check_snapshot(session=self.session, snapshot=snapshot)  # G2 ONLY
            self.ledger.record_usage(                      # for the ATTEMPT
                entry.id, **plan.usage.model_dump(),
            )
        except Exception as exc:
            yield self._close_compaction(entry, TurnOutcome.ERRORED, str(exc))
            if entry.source == CompactionSource.USER:
                raise
            return
    cancel = self.ledger.open_turn_cancel_requested()
    if cancel is not None:                       # arrived within the grace window
        self._compaction_consumed_cancel = True
        yield self._close_compaction(entry, cancel.outcome, cancel.error)
        return
    if plan is None:                             # the ONE "nothing to do" signal
        yield self._close_compaction(entry, TurnOutcome.COMPLETED)
        return
    try:
        conversation, final, created = self._commit(entry, plan, snapshot)
    except Exception as exc:
        yield self._close_compaction(entry, TurnOutcome.ERRORED, str(exc))
        if entry.source == CompactionSource.USER:
            raise
        return
    self._closed_outcome = TurnOutcome.COMPLETED
    yield CompactionFinished(
        entry=final.model_copy(deep=True), outcome=TurnOutcome.COMPLETED,
        created=[e.model_copy(deep=True) for e in created],
        conversation_id=conversation.id,
    )
```

The mechanics this pins down, in the order the step meets them:

**The offered view is built once, in `_snapshot_conversation`.**

```python
def _snapshot_conversation(self) -> ConversationSnapshot:
    """The full active path (G2's input) plus the view handed to the policy:
    the same path with THIS compaction's `TurnStart` removed. Exactly one
    element — the bracket's tail is `[…, ts_c, cmp]` by construction, so this
    is a positional removal, not a filter over types."""
    conversation = self.session.active_conversation
    nodes = tuple(conversation.nodes)
    bracket = self.ledger.open_turn_index()          # the ts_c index
    return ConversationSnapshot(
        id=conversation.id,
        nodes=nodes,
        offered=nodes[:bracket] + nodes[bracket + 1:],
    )
```

`open_turn_index()` is the same call `_commit` uses to compute
`compacted_nodes`, so the two agree by construction. The tail is exactly
`[…, ts_c, cmp]` at this point: `_open_compaction_bracket` appends both with
nothing between them, `post_message` raises while the bracket is open, and a
parked cancel flushes at (1) before the policy is reached. **`cmp` stays in the
view** — see PRD §5; stripping it would make `plan.nodes = list(nodes)` fail
rule 7 and trade one hidden requirement for another.

**G6 tests the entry, not the bracket — and the step gets it for free.**
`open_compaction_entry()` (§4.3) already excludes an entry that has `parts`, so
the step needs no arm of its own: `entry is None` → the existing
open-conversational-turn branch → `return`. `parts` land only at the commit
point, and every non-transition ending leaves them `None` (`_close_compaction`
is explicit about it), so an open bracket around an entry with content cannot
be an interrupted attempt — it is a path a plan shaped to look like one.
Returning leaves the bracket alone, so the drive falls through and treats it as
the phantom open turn it is: loud, the policy's, and with the committed
`parts`/`llm_config`/`compacted_nodes` intact. Keying on bracket shape instead
would re-call `compact()` on finished work and overwrite that record.

It costs the real resume path nothing: a genuinely interrupted compaction
always has `parts is None`.

**`started_at` is stamped once.** A resumed attempt keeps the original stamp,
so "open bracket + `started_at` set" keeps reading as "running, or crashed
mid-run" and the previous attempt stays visible. `ended_at` records the final
close, so the pair spans every attempt.

**How `compact()` is cancelled and timed out.** The PRD says "the runner's
existing turn timeout", but that timeout is a client kwarg the runner passes to
`acompletion` — it cannot reach a call the policy makes itself. So
`_invoke_policy` wraps the coroutine in a task and reuses the machinery the LLM
step already uses:

```python
async def _invoke_policy(self, entry, offered, token):
    config = self.session.session_config.runtime_config
    task = asyncio.ensure_future(
        self.compaction_policy.compact(
            self.session, offered, entry.model_copy(deep=True)))
    grace = config.llm_completion_cancellation_grace_period
    # the wall-clock tier, converted EXACTLY like the LLM step (runner.py:1116):
    deadline = _ms_to_seconds(config.client_completion_timeout_in_ms)
    # deadline is None  → no asyncio.timeout at all, just _race_cancellation
    # deadline is not None → asyncio.timeout(deadline) around
    #     _race_cancellation(task, token, grace, None, detach=False)
    # → expiry raises _CompactionCancelled(TIMED_OUT, "compaction exceeded …")
    # → token fired + grace expired raises _CompactionCancelled(CANCELLED, …)
```

**The conversion is not optional, and neither is the `None` branch.**
`client_completion_timeout_in_ms` is int **milliseconds** while
`asyncio.timeout` takes **float seconds**, and its default is the `Inf`
sentinel — the integer `-1` (`models.py:510`, `:535`), meaning "no limit".
Passing the raw field would make `asyncio.timeout(-1)` fire immediately, so
every compaction on a default configuration would close `TIMED_OUT` before the
policy's call went anywhere: a `source=USER` one raising, a `source=POLICY` one
silently giving up on every drive. `_ms_to_seconds` (`runner.py:1790`) already
returns `None` for `Inf`, which is exactly the "skip the timeout" branch the
LLM step takes. Consequence to accept, stated once: with the default config a
compaction has **no deadline**, and a hung policy hangs the drive until
cancelled — identical to the conversational LLM call's default, which is why
it stays consistent rather than growing its own knob.

`detach=False` (like the LLM step) so the policy's wire is closed before
control returns. `builtin_client_completion_timeout_in_ms` is inert here — it
is a per-phase httpx knob only the policy can pass. A dedicated
`compaction_timeout_in_ms` waits for a real need.

**The policy gets a deep copy of the entry and the LIVE session.** The copy is
load-bearing (§5 of the PRD): a policy that writes `parts` and then fails would
otherwise leave a summary projecting onto an unchanged path. The session is not
copied — that whole-session deep copy is what this design removed — and G2
doubles as the detector for a policy that appends to it.

**Usage is recorded for the attempt, inside the failure handler.** Before the
commit point, so a rejected or cancelled compaction still accounts for
the tokens it spent — the compaction call's input is the whole conversation
being compacted, at the moment the window is nearly full, so it is the most
expensive request in the session and losing it from the accounting is the worse
outcome (PRD §4).

Two things make this write safe, and both are load-bearing:

- **It is guarded.** An earlier draft placed `record_usage` between the two
  `try` blocks, in neither. `record_usage` can raise — it verifies the entry is
  still on the active conversation's path (`ledger.py:118`) — and an escape
  there would leave the bracket **open**, which is the "resume me" state: the
  next drive would replay the same policy call and raise again, with
  `should_compact` never consulted (a bracket is already open), so the policy
  could not back off. A stuck session instead of a clean rejection.
- **G2 is checked first.** `check_snapshot` is the G2 half of `validate_plan`,
  factored out so it can run before the write. A policy that replaced
  `session.active_conversation` takes the compaction entry off the active path,
  so `record_usage` would raise its own unrelated `AgentError` and pre-empt the
  `CompactionPlanError` G2 is supposed to produce. `_commit` still runs the
  full `validate_plan`, G2 rules included — they re-run harmlessly, nothing
  moves in between.

The counter *names* need no runtime check: `plan.usage` is a typed
`UsageCounters` (§3), so a provider's own vocabulary fails at plan
construction, in the policy.

**A cancel discards a plan that arrived within grace.** This differs from the
LLM path, where a within-grace answer *is* recorded before the cancel outcome
wins. The asymmetry is deliberate: recording an assistant message adds a node,
while committing a plan rewrites what the conversation is. `parts` therefore
never land on a cancelled compaction.

**`_close_compaction` is the single non-transition ending.**

```python
def _close_compaction(self, entry, outcome, error=None) -> CompactionFinished:
    """Stamp `ended_at`, close the bracket on the PRE-COMPACTION path, return
    the event. `parts` and `compacted_nodes` stay None — nothing was
    committed, so nothing may project."""
    final = self._persist_entry(entry, ended_at=self.now_ms())
    self._close_turn(outcome, error)          # the one TurnFinish writer;
                                              #   sets _closed_outcome (§5.1a)
    return CompactionFinished(
        entry=final.model_copy(deep=True), outcome=outcome, error=error,
        created=[], conversation_id=None)
```

Both writes here run `before_entry_written` and are therefore fallible. A
middleware raise leaves the bracket open — the recoverable state — and
propagates. That is correct and is tested.

**`None` is the only "nothing to do", and there is no `_is_noop`.** An earlier
draft had the runner discard a valid plan whose replaced span was only a
`CompactionEntry` (G5, "refuse to summarize a summary"). That guard is gone —
see the PRD's §11 for the reasoning; the short version is that it was the single
place the runner judged *meaning* rather than structure, and it threw away a
structurally valid plan the policy had already paid an LLM call to produce.

Three things follow, all of them simplifications:

- **A plan that re-summarizes a summary commits.** So does one whose summary is
  larger than the span it replaced. Both are the policy's judgment.
- **A full-carry plan commits with `compacted_nodes == []`** (§13's decision 10),
  with no special case. This was briefly an open contradiction in this document:
  `all()` over the empty span of a full-carry plan is vacuously `True`, so
  `_is_noop` would have discarded a plan decision 10 and
  `test_a_full_carry_plan_commits_with_an_empty_compacted_span` both require to
  commit. Deleting the guard dissolves it.
- **Bounding a runaway is G4 plus the policy.** The step sits outside
  `while True`, so one compaction per drive is structural; anything beyond that
  is `should_compact`'s to decline. The **default contrib policy owns the
  floor** — a minimum span worth summarizing, and a gauge that actually falls
  after a commit (which is why the `context_tokens` recalculation of §5.5 is now
  load-bearing rather than merely tidy).

### 5.5 Preparation and commit

```python
def _commit(self, entry, plan, snapshot):
    """Everything fallible, then one infallible door."""
    validate_plan(plan, entry_id=entry.id, session=self.session, snapshot=snapshot)

    ts = self.now_ms()                       # ONE timestamp for the whole transition
    bracket = self.ledger.open_turn_index()  # ts_c; still open — the closing
                                             #   TurnFinish is only BUILT below.
                                             #   G2 has just proven this index is
                                             #   the same one _snapshot_conversation
                                             #   stripped at.
    carried = {n for n in plan.nodes if isinstance(n, str)}
    compacted = [n for n in snapshot.nodes[:bracket] if n not in carried]

    final = self._complete_entry(entry.model_copy(update={          # fallible
        "parts": plan.entry.parts,
        "llm_config": plan.entry.llm_config,
        "metadata": plan.entry.metadata,
        "compacted_nodes": compacted,
        "ended_at": ts,
    }))
    nodes, created = [], []
    parent = None
    for node in plan.nodes:                                        # fallible
        if isinstance(node, str):
            parent = node
        else:
            built = self._prepare(lambda i, p, t, n=node: n.model_copy(
                update={"id": i, "parent_id": p, "created_at": t}), parent, ts)
            created.append(built)
            parent = built.id
        nodes.append(parent)
    closing = self._prepare(                                       # fallible
        lambda i, p, t: TurnFinish(id=i, parent_id=p, created_at=t,
                                   outcome=TurnOutcome.COMPLETED),
        self.session.active_conversation.nodes[-1], ts)
    # ─────────────────────────── THE TRANSITION ───────────────────────────
    conversation = self.ledger.transition_conversation(
        updates=[final], created=created, closing=closing, nodes=nodes, ts=ts)
    return conversation, final, created
```

Four details the PRD leaves to be discovered:

- **`compacted_nodes` is computed over the path BEFORE the compaction
  bracket**, not the whole pre-transition path. Otherwise `ts_c` — the
  compaction's own marker — lands in the list of "ids this entry replaced".
  `bracket` comes from the same `open_turn_index()` call
  `_snapshot_conversation` used to build `offered` (§5.4), and G2 has just
  proven the path did not move between them, so the strip and the span can
  never disagree about where the bracket is.
- **`parent_id` threads left to right** across `plan.nodes`; the first created
  entry's parent is the node preceding it, or `None` when the plan opens with
  it (a conversation's first node has no parent, already true of every
  session's first entry). Everything a policy set on `id`/`created_at` is
  overwritten.
- **The closing `TurnFinish` is built but not stored.** `_close_turn` cannot be
  used on the success path: it appends to the *active* conversation, and the
  marker belongs to the outgoing one. Building it here is what stops a bracket
  from closing `COMPLETED` ahead of a transition that then fails.
- **`_closed_outcome`** is set to `COMPLETED` by the step (not by
  `_close_turn`, which never runs on this path) so `RunResult` reports the
  bracket this run actually closed. This is the one write of it that is
  compaction-specific; the rest live on `_close_turn` (§5.1a).

### 5.6 `start()` decides at call time

`AgentRun.__init__` currently calls `_ensure_open_turn()` synchronously, so an
eager run always presents the drive with an open conversational turn and the
compaction step would skip forever. Replace that one call:

```python
self._runner._begin_run(self)              # raises on IDLE / concurrent run
try:
    self._runner._open_bracket_for_start() # ← was _ensure_open_turn()
except BaseException:
    self._runner._end_run(self)            # release the guard, then propagate
    raise
self._task = loop.create_task(self._consume())
```

**The try/except is mandatory, not defensive coding.** `_begin_run` takes the
one-run-at-a-time guard by assigning `self._active_run = run`
(`runner.py:725`), and both of its own rejections raise *before* that line — so
a rejected `start()` leaves the runner clean today. `_open_bracket_for_start`
runs *after* the guard is taken and now calls application code
(`should_compact`), and the three sites that release the guard — `_pump`'s
error path, `_consume`'s `finally`, `__aexit__` — are all downstream of
`loop.create_task`, which never runs. Without the release, one `TypeError` in
a policy's threshold arithmetic makes every later `run()` and `start()` on that
runner raise "another run is already active", with no public way to clear it:
the handle holding the guard has `_task is None`, so even `await run` fails.
The session data is untouched; the runner object is dead.

The lazy `run()` path needs no change — `_pump` calls `_begin_run` and then
merely *constructs* the `_drive` async generator, which executes nothing, and
the first `__anext__` raise is already caught and released by `_pump`.

This hole exists today via `before_entry_written` raising inside
`_ensure_open_turn`, so the fix is worth making on its own; the decision below
is what turns it from a middleware-author edge case into reachable behavior of
the compaction API.

```python
def _open_bracket_for_start(self) -> None:
    """A compaction bracket instead of a TurnStart when one is due —
    `should_compact` is sync precisely so this is possible. An
    already-scheduled compaction needs nothing: the bracket is open, so
    `_ensure_open_turn` is already a no-op."""
    if (
        self.compaction_policy is not None
        and self.ledger.open_turn_index() is None
        and self.compaction_policy.should_compact(self.session)
    ):
        self._open_compaction_bracket(CompactionSource.POLICY)
        return
    self._ensure_open_turn()
```

The `start()` docstring promise — "the bracket opens durably at call time so an
immediate `cancel()` has something to attach to" — is preserved: a compaction
bracket is an open bracket, and the first drive is then the flush.

**A `should_compact` that raises propagates.** The PRD does not say; the
alternative (swallow → no compaction) makes a broken policy indistinguishable
from one that declines, and nothing durable exists yet to record the failure
on. From `start()` that is a synchronous raise at call time — and, because of
the release above, one that leaves the runner reusable rather than wedged.

### 5.7 `RunResult`

```python
def _build_run_result(self) -> RunResult:
    if self.status == ConversationStatus.AWAITING_APPROVAL:
        return RunResult(status=..., outcome=None, pending_approvals=...)
    return RunResult(
        status=self.status,                 # derived, not hardcoded IDLE
        outcome=self._closed_outcome,       # carried, not re-read from nodes[-1]
        pending_approvals=[])
```

Both old assumptions break here: after a successful transition the closing
`TurnFinish` is on the archived conversation, and after a compaction-only drive
the active leaf may be the `CompactionEntry` or a carried `AssistantMessage` —
neither has `.outcome`, which is today's `AttributeError`.

**Collateral, outside compaction:** deriving `status` changes the hard-max-steps
result from `IDLE` to `PENDING`, which is what the session already says
(retry-ready). `tests/agent/test_runner_limits.py:127` asserts the old value
and must be updated; `RunResult`'s docstring changes from "IDLE |
AWAITING_APPROVAL" to "the derived status where the run stopped".

### 5.8 Construction and equality

```python
runner = AgentSessionRunner(session, compaction_policy=Compactor(...))
```

Keyword-only, defaults to `None` (compaction never happens; `should_compact` is
never consulted and `schedule_compaction()` raises). `__eq__` gains
`_equivalent(self.compaction_policy, other.compaction_policy)` — without it two
differently-configured runners compare equal, which
`tests/agent/contrib/test_plugins.py` asserts against.

**Two subclasses must forward the new argument, or nothing can use it.** Both
declare explicit keyword lists rather than `**kwargs`, so a new constructor
argument on the base is invisible to them:

- `PluginAgentSessionRunner` (`contrib/plugins/runner.py:23-58`). The TUI builds
  its runner through this class (`contrib/tui/wiring.py:146`), so without the
  pass-through §8's `/compact` command cannot be wired at all. This is
  forwarding, **not** the plugin *composition* the PRD defers — no
  `get_compaction_policy()` hook, no fourth collaborator; a fourth collaborator
  still waits for a real second case.
- `DeterministicRunner` (`tests/agent/scenarios.py:200-223`). Every integration
  test in §11.6 constructs one, so this lands with the test doubles rather than
  with the production code.

---

## 6. Events (`core/events.py`)

Three lifecycle events mapping one-to-one onto the tool events, each carrying a
deep snapshot, each following the persistence of the state it shows, all three
in the `AgentEvent` union and firing in both streaming modes.

| Event | Fires | Snapshot state |
|---|---|---|
| `CompactionScheduled` | bracket + entry on the path (opened or resumed) | `started_at is None` on a fresh open; the previous stamp on a resume |
| `CompactionStarted` | `started_at` stamped, immediately before the policy call | `llm_config is None` — the policy picks the model and has not been called |
| `CompactionFinished` | after the bracket closes or the transition commits, **whatever the outcome** | terminal; `parts` set → summarized, `None` → nothing done |

`CompactionFinished` carries `outcome`, `error`, `created`, and
`conversation_id` (`None` whenever nothing was compacted). It fires on failure
because for a degraded `POLICY` failure it is the *only* signal — an observer
that is not the caller would otherwise see `Started` and then silence forever.
There is no `CompactionFailed`: `TurnOutcome` already covers it.

`CompactionScheduled` fires at the top of the drive for both sources, never
inside `schedule_compaction()` — that call happens outside any run and has no
stream. The Scheduled→Started window is usually microseconds and is preserved
on purpose: a consumer may cancel in it, which means the events must be yielded
from the generator as they happen, not buffered and flushed after `compact()`
returns.

---

## 7. Projection, context, presentation

**`projection.py`** — `project_compaction` returns `ClientUserMessage | None`:

```python
if not entry.parts:
    return None            # scheduled, running, skipped or failed → bookkeeping
return ClientUserMessage(content=[self._content_block(p) for p in entry.parts])
```

`project_entry`'s signature already admits `None`, and `_content_block` gives
images for free — that is the simplification the `parts` shape buys.

The `parts` guard above is a detail, not the rule. **The rule is positional and
lives in `project()`** (PRD §4, decision 10):

```
while walking conversation.nodes:
    a TurnStart whose NEXT node is a CompactionEntry opens a compaction bracket
    → skip every node up to and including its TurnFinish (or to the end, if open)
    everything else dispatches through project_entry as it does today
```

A `CompactionEntry` reached outside any bracket dispatches normally and
projects its `parts`. So: the bracket contributes nothing, ever; the entry
contributes its summary exactly where the history it replaces is gone.

**Skipping the span is what makes it correct, not just symmetric.**
`project_turn_finish` (`projection.py:288`) returns the
`CANCELLED_TURN_MARKER` for **any** `TurnFinish(CANCELLED)`, and a cancelled
compaction never transitions, so `tf_c` stays on the *active* path and every
later request carries `"[Request interrupted by user]"` about a question the
model was never shown:

```
c1.nodes = [… u4, ts_c, cmp, cr, tf_c(CANCELLED)]
projects   [… "what is X?", "[Request interrupted by user]"]
```

`ts_c` and `cr` already project `None`, so `tf_c` is the one node that needed
it — but a per-marker content test would be two rules where one does, and it is
reachable on the plainest cancel path in §11.6 group E.

Discarding the whole span loses nothing: only markers can ever be inside a
compaction bracket, because `post_message` raises while one is open. And it
gives the archived conversation the right behavior for free — `c1` still holds
every original entry, so projecting it must **not** also emit a summary of
them. The positional rule drops the summary there and keeps it on the new path,
where the originals are gone; a content test would have emitted both.

`project()` already walks `conversation.nodes` and is already documented as the
seat of path-level policy, and the bracket test is §4.3's shared predicate.
Every per-entry method keeps its signature — classifying a bracket needs the
path, which those methods deliberately do not receive, so pushing the rule down
would mean a public signature change for a fact the caller already holds. A
subclass overriding `project()` wholesale owns its own path policy, as today.

**`context_manager.py`** — `_model_facing_text` maps `entry.parts or []`
through the same `_text_of` a user message uses; `_media_parts` gains a
`CompactionEntry` branch returning `entry.parts or []`, so an image-carrying
summary is counted. `parts is None` counts 0, which is why the recalculation
when `parts` land is mandatory: otherwise the summary contributes nothing to
the active path's total and the policy's own gauge immediately concludes the
window is still full.

**`utils.py:255`** — `pretty_print` reads `len(entry.compacted_nodes or [])`
and `_content_text(entry.parts or [])`, and `_timestamp` guards a `None`
`created_at`.

**`core/__init__.py`** — export `CompactionPolicy`, `CompactionPlan`,
`CompactionSource`, `CompactionPlanError` (`CompactionEntry` is already
exported).

---

## 8. Contrib follow-ups (separately planned, listed so nothing is lost)

- **G1, atomic session writes.** `contrib/tui/sessions.py:33` truncates in
  place via `path.write_text`; it must write a temp file and `os.replace`.
  Core owns no persistence, so the docs must also state atomic writing as an
  application responsibility. This is the only true corruption risk in the
  system today and it exists independently of compaction.
- `/compact` slash command (schedule then drive — the TUI's loop already runs
  until `runner.idle()`). Needs the `PluginAgentSessionRunner` pass-through of
  §5.8 first.
- A transcript cell for `CompactionEntry`: `contrib/tui/app.py:318` is an
  `if/elif` chain with no `else`, so a resumed compacted session silently
  renders no summary today.
- **The default policy owns the floor that core no longer provides.** With G5
  deleted (PRD §11), nothing in core declines a compaction it judges pointless,
  so the contrib policy's own spec must pin: a minimum span worth summarizing
  (one exchange folded into a longer summary is a pessimization), a refusal to
  re-summarize when the span is just a previous summary and its markers, and a
  gauge that actually falls after a commit — otherwise `should_compact` stays
  true and the policy compacts once per drive forever. Its tests are the only
  thing that pins any of this.

---

## 9. What a strategy author may treat as invariant

Published in `docs/agent/12-compaction.md`. Each line is pinned by named tests
in §11, which is the point: a policy author implements and tests `compact()`
and takes the rest as given.

| # | Invariant |
|---|---|
| INV-1 | Carried ids are never copied, renumbered, reordered, or mutated. Kept entries stay exactly as they were. |
| INV-2 | The pre-compaction conversation is archived in `conversation_history` with its exact path plus the closing marker, frozen at `IDLE`. |
| INV-3 | Every entry you did not carry stays in `entries` and is listed, in path order, in `compacted_nodes`. Nothing is ever deleted. |
| INV-4 | Entries you create get `id`/`parent_id`/`created_at` stamped by the framework: one shared timestamp, parents threaded left to right, `None` for a plan that opens with a created entry. |
| INV-5 | If you return `None`, raise, time out, or are cancelled, the conversation is byte-identical to before plus one closed bracket. No partial state exists, ever. |
| INV-6 | Your `parts` never reach the projector unless the transition committed. A failed compaction cannot tell the model "here is a summary". |
| INV-7 | Your `usage` is recorded against the pre-compaction conversation even if the plan is then rejected or cancelled. It is **not** recorded if you return `None` (there are no counters to record) or if you raise (the framework never sees the response) — so catch your own post-processing errors and return a plan if you want the spend to appear. |
| INV-8 | You are handed a deep copy of the entry and the live session. The framework applies exactly `parts`, `llm_config`, `metadata` and discards the rest. |
| INV-9 | A crash before the transition leaves an open bracket that the next drive resumes **with the same entry**; a closed bracket is never retried; an entry that already has `parts` is never re-run; a second bracket never piles up. |
| INV-10 | At most one compaction per drive. `should_compact` is not consulted while a conversational turn is open, nor when the session is `IDLE`. |
| INV-11 | `before_llm_call`, `after_llm_response`, `build_model_string` and `build_tool_list` never fire for your LLM call. `before_entry_written` fires for everything compaction writes. |
| INV-12 | The compaction entry's `context_tokens` is recalculated when your `parts` land, before entry middleware. |
| INV-13 | A structurally valid plan is **always** committed. Core never judges whether the compaction was worth doing — not a summary of a summary, not one larger than the span it replaced, not a full carry with nothing replaced. Deciding that is `should_compact`'s job, and it is asked before the LLM call, not after. |
| INV-14 | `nodes` is the complete set of ids you may carry — the active path minus this compaction's own `TurnStart`, ending with your entry. Anything outside it is a plan rejection, so you never have to recognize or route around framework markers. `plan.nodes = list(nodes)` is always legal. |

### Hazards you own (committed as given, never checked)

Splitting a tool call from its result in either direction (provider 400 on the
next request); carrying a nonterminal `ToolExecution` (`ProjectionError` on the
next request); reordering carried ids; **carrying a turn marker without its
pair** (below); and the one silent failure — **summarizing away a trailing
unanswered `UserMessage`**, which drops the user's question with no error
anywhere. The framework does not prevent that last one, because a policy may
legitimately want to fold the message into the summary text. The default policy
must carry it, and its own tests are the only thing that pins that.

**Carrying a turn marker without its pair is worse than "malformed".** An
earlier draft described these two as cosmetic; they are not, because
`derive_status` and the resume check both read brackets:

- **A `TurnFinish` without its `TurnStart`.** Status derivation walks back from
  the leaf looking for the opener. §4.4 stops skipping when there is none, so
  the existing closed-turn rules apply and a carried `TurnFinish(ERRORED)`
  still derives `PENDING` — but only because §4.4 says so explicitly. Get that
  branch wrong and a retry-ready turn silently derives `IDLE` and is never
  retried.
- **A `TurnStart` without its `TurnFinish`** — a phantom open turn. The new
  conversation derives `PENDING` (or `AWAITING_APPROVAL` if a
  `PENDING`-approval execution came with it) and the next drive "resumes" a
  turn that never existed, calling the model. If your summary entry lands
  immediately after that carried `TurnStart`, §4.3's adjacency test also reads
  the pair as a **compaction** bracket — but the next drive still will not
  re-run the compaction: G6 resumes only an entry with `parts is None`, and a
  committed one has content (§5.4). So the committed
  `parts`/`llm_config`/`compacted_nodes` survive, and what is left is the
  phantom open turn above.

  This used to be the one hazard on the list that could damage an existing
  audit trail. It is now contained from both ends: the compaction's own `ts_c`
  is not among the `nodes` you are offered, so carrying *it* is a plan
  rejection rather than a hazard (PRD §5), and any other `TurnStart` that
  produces the same shape is caught by G6's content test.

The default policy avoids all of these by cutting on turn brackets and placing
the summary entry at the front of the new path.

---

## 10. Determinism budget

`DeterministicRunner`'s `ids` script is positional, so the draw order is part
of the contract. Documented here and asserted by the tests.

| Operation | ids drawn, in order |
|---|---|
| `schedule_compaction()` | `TurnStart`, `CompactionEntry` |
| drive-top policy-initiated open | `TurnStart`, `CompactionEntry` |
| `start()` with a compaction due | `TurnStart`, `CompactionEntry` (no bare `TurnStart` first) |
| any non-transition ending | closing `TurnFinish` |
| the transition | each created entry in `nodes` order, then the closing `TurnFinish`, then the new conversation id |

Every timestamp in one transition is a single `now_ms()` read: the compaction
entry's `ended_at`, every created entry's `created_at`, the closing
`TurnFinish`, and the new conversation's `created_at`/`updated_at`.

---

## 11. Test plan

The mandate: **each responsibility tested individually, as a unit**, and one
integration file that proves the session/conversation handling is coherent for
every ending — so a future strategy is a unit that plugs into an already-proven
frame.

### 11.1 Scope table — what each file may and may not touch

| File | Tests | Must NOT |
|---|---|---|
| `test_compaction.py` | `validate_plan` (every rule), `has_content` (G3 truth table), the `CompactionPolicy` base contract, `CompactionPlan` shape/`extra="forbid"` | construct a runner, a ledger, or a provider; be async |
| `test_ledger.py` | `put_entry`, `transition_conversation`, `open_compaction_entry`, `derive_status` skip matrix | know a policy exists; run an engine |
| `test_projection.py` | `parts` → synthetic user message; `None` → `None`; image parts | assert token counts |
| `test_context_manager.py` | `parts` counting, image counting, `None` → 0 | project |
| `test_models.py` | the new entry shape, JSON round-trip, `turn_count` rules | anything runtime |
| `test_utils.py` | `pretty_print` over the new shape | — |
| `test_runner_middleware.py` | `before_entry_written` sees every compaction write; the four turn hooks do **not** fire | assert transition mechanics |
| `test_runner_compaction.py` | the drive: brackets, events, transitions, failures, cancels, resumes, statuses, `RunResult` | assert token arithmetic or projection wording |

### 11.2 Test doubles and literals (`tests/agent/scenarios.py`)

Core tests must not import contrib, so the compaction analogue of
`FakeToolRegistry` lives here.

```python
class FakeCompactionPolicy(CompactionPolicy):
    """Scripted core-only policy double.

    `should` scripts `should_compact` (a bool, or a list popped per call —
    exhaustion returns the last value); `plan` is a `CompactionPlan`, or a
    callable `(session, nodes, entry) -> plan | None` for plans that need the live
    entry id; `raises` raises instead of returning; `hang` awaits
    `release` forever (cancel/timeout scenarios, cooperatively released by
    the test — never a second timer); `mutate` writes `parts` onto the entry
    it was handed before failing, to prove the deep copy is load-bearing.
    `seen` records `(session, nodes, entry)` per call and `should_calls`
    counts the sync consultations."""
```

Literals, all `created_at=500` so entries written by the frozen test clock
(`now=1000`) are visually distinct:

| Literal | Shape | Derived status |
|---|---|---|
| `RICH_SESSION` | the canonical feature-rich conversation (below), trailing unanswered `u4` | `PENDING` |
| `RICH_IDLE_SESSION` | `RICH_SESSION` minus `u4` | `IDLE` |
| `COMPACTION_SCHEDULED_SESSION` | `RICH_SESSION` + `[ts_c, cmp(source=USER, started_at=None)]` | `PENDING` |
| `COMPACTION_INTERRUPTED_SESSION` | the same with `started_at=600`, persisted status a stale `RUNNING` | `PENDING` (self-heals) |
| `COMPACTION_CANCEL_PARKED_SESSION` | scheduled + `CancelRequested` | `CANCELLING` |
| `COMPACTION_FAILED_SESSION` | `RICH_IDLE_SESSION` + `[ts_c, cmp, tf_c(ERRORED)]` | `IDLE` — the no-spin case |
| `COMPACTION_BURIED_SESSION` | `RICH_SESSION` + `[ts_c, cmp, tf_c(COMPLETED)]` (so `u4` is buried) | `PENDING` — the no-input-loss case |
| `POST_COMPACTION_SESSION` | a committed transition: `conversation_history=[c0, c1]`, active `c2 = [cmp, u4]` | `PENDING` |

**`RICH_SESSION` — the one precondition most scenarios share.** Conversation
`c1`, with `conversation_history=[c0]` already present (this session has been
compacted once before, which is itself a case worth carrying):

| id | entry | why it is here |
|---|---|---|
| `cmp0` | `CompactionEntry(source=POLICY, parts=[…], compacted_nodes=["u0","a0"], llm_config=CHEAP, started_at, ended_at)` | a prior summary on the path: "compacting a compacted session", history depth |
| `u1` | `UserMessage(parts=[TextContent, ImageContent])` | text + image content (media counting) |
| `ts1` | `TurnStart` | |
| `a1` | `AssistantMessage([Thinking(signature), Text, ToolCall add(tc1), ToolCall read_file(tc2)])` | reasoning durability, parallel calls |
| `te1` | `ToolExecution(COMPLETED, result="3")` | **successful** tool execution |
| `te2` | `ToolExecution(FAILED, error=…)` | **failed** tool execution |
| `pr1` | `PrunedEntry(pruned_entry_id="te0", pruned_entry_type="tool_execution")` | a pruned node whose referent is in `entries` but on no path |
| `a2` | `AssistantMessage([Text])` | |
| `tf1` | `TurnFinish(COMPLETED)` | **successful** turn |
| `u2` | `UserMessage` | |
| `ts2` | `TurnStart` | |
| `tf2` | `TurnFinish(ERRORED, error="…")` | **failed** turn (retry-ready leaf if carried) |
| `u3` | `UserMessage` | |
| `ts3` | `TurnStart` | |
| `a3` | `AssistantMessage([ToolCall multiply(tc3)])` | |
| `te3` | `ToolExecution(REJECTED)` | **rejected** execution |
| `cr1` | `CancelRequested` (consumed) | cancellation audit inside a closed bracket |
| `tf3` | `TurnFinish(CANCELLED)` | **cancelled** turn (projects the interrupted marker) |
| `u4` | `UserMessage` | the **trailing unanswered** question |

Plus `entries["te0"]` (the pruned referent, on no path), `entries["u0"]`/`a0`
(compacted by `cmp0`, on `c0` only), `tool_executions` for `tc1`/`tc2`/`tc3`,
and `usages["c0"]` + `usages["c1"]` records. `c0` is a complete archived
conversation so tests can assert it is never touched again.

### 11.3 Unit: `tests/agent/test_compaction.py`

Pure and synchronous. One session literal, one snapshot, one plan per test.

- `has_content`: `None` → False; `[]` → False; `[Text("")]` → False;
  `[Text("  \n\t")]` → False; `[Text("x")]` → True; `[Image(...)]` → **True**;
  `[Text("  "), Image(...)]` → True.
- One test per rejection rule in §3's table (8 rules, ~11 tests counting the
  G2 id-vs-path split and the archived-id variant of rule 5), each asserting
  `CompactionPlanError` and the message.
- `check_snapshot` (the G2 half, called early by the step — §5.4) raises the
  same two messages on its own, and is a no-op when nothing moved.
- `validate_plan` accepts: a fold-everything plan; a plan with created entries
  interleaved between carried ids; a plan whose carried ids are reordered; a
  plan whose `usage` is default-constructed; a plan whose `usage` sets all five
  counters.
- `validate_plan` ignores every non-content field of `plan.entry` (a plan whose
  entry has a foreign `id`, a `source` mismatch, or a stale
  `compacted_nodes` still validates).
- `validate_plan` checks carried ids against `snapshot.offered`, not
  `snapshot.nodes`: a plan carrying the bracket's `TurnStart` is rejected by
  rule 5, and the two G2 rules still compare against the full `nodes`. One
  test per side, so a single-field snapshot cannot pass both.
- `CompactionPolicy` base: `should_compact` returns False; `compact` raises
  `NotImplementedError` and accepts `(session, nodes, entry)`.
- `CompactionPlan` rejects an unknown field (`extra="forbid"`) and accepts a
  mixed `nodes` list of ids and entry objects without copying the objects.
- `UsageCounters` rejects a provider's own counter names
  (`prompt_tokens=`, `completion_tokens=`) with a `ValidationError` at
  construction, defaults every counter to 0, and `model_dump()` produces
  exactly the kwargs `record_usage` takes — the same five
  `_to_usage_counters()` produces. This is what makes a usage-key rejection
  rule unnecessary.

### 11.4 Unit: `tests/agent/test_ledger.py` additions

**`put_entry`** — stores the replacement and touches
`Conversation.updated_at`; raises on `id is None`; raises on an id absent from
`entries`; updates a `CompactionEntry` as readily as a `ToolExecution`; does
**not** touch `nodes`.

**`transition_conversation`** — one full-object postcondition test plus the
edges:
- the outgoing conversation is archived with `nodes + [closing.id]`,
  `status=IDLE`, `updated_at=ts`; the new conversation is active with the given
  `nodes`, a fresh id from `gen_id`, and `created_at == updated_at == ts`;
- `conversation_history[-1] is not session.active_conversation` (the same
  conversation is never both);
- created entries land in `entries`; a created `ToolExecution` is indexed into
  `tool_executions`;
- `closing=None` archives without extending the outgoing path;
- the new conversation's status is derived (four cases: carried
  `TurnFinish(COMPLETED)` → IDLE, carried `UserMessage` → PENDING, carried
  `TurnFinish(ERRORED)` → PENDING, leaf is the `CompactionEntry` → IDLE);
- preconditions raise **before any mutation** (`updates` id unknown, `created`
  id already present, `created` id `None`) — each asserting the session is
  byte-identical to before via a full-object compare.

**`open_compaction_entry`** — None on an empty path; None with an open
conversational turn (bare `TurnStart`; `TurnStart`+`AssistantMessage`); the
entry with an open compaction bracket; None with a closed compaction bracket;
the entry when a `CancelRequested` sits inside the open compaction bracket;
**None when the open bracket's entry already has `parts`** (G6 — the
counterfeit case, and the one that separates "interrupted" from "committed").

**`derive_status` skip matrix** — every row of the PRD's table plus the ones it
implies:

| Path tail | Expected |
|---|---|
| `… u3 ts_c cmp tf_c(COMPLETED)` | `PENDING` (the question still drives) |
| `… u3 ts_c cmp tf_c(CANCELLED)` | `PENDING` (cancelling ≠ dropping input) |
| `… tf2(COMPLETED) ts_c cmp tf_c(ERRORED)` | `IDLE` (no spin) |
| `… tf2(COMPLETED) ts_c cmp tf_c(TIMED_OUT)` | `IDLE` |
| `… tf2(ERRORED) ts_c cmp tf_c(COMPLETED)` | `PENDING` (the real turn is still retry-ready) |
| `ts_c cmp tf_c(COMPLETED)` alone | `IDLE` (compaction on a fresh session) |
| two stacked closed compaction brackets over `u3` | `PENDING` |
| `… ts_c cmp` (open, no `started_at`) | `PENDING` |
| `… ts_c cmp(started_at)` (open) | `PENDING` |
| `… ts_c cmp cr` (open + unconsumed cancel) | `CANCELLING` |
| `… ts_c cmp tf_c(COMPLETED) u5` | `PENDING` (a node after the bracket → no skip) |
| a closed **conversational** bracket | unchanged from today (regression guard) |

### 11.5 Unit: the other single-responsibility files

- **`test_projection.py`** — `parts=[Text]` → `ClientUserMessage([TextBlock])`;
  `parts=[Text, Image]` → both blocks in order; `parts=None` → `None`;
  `parts=[]` → `None`; a conversation containing a `parts=None` compaction
  projects the rest of the path unchanged; a subclass overriding
  `project_compaction` is honored. Plus the positional rule (§7): a path
  ending `[… u4, ts_c, cmp, cr, tf_c(CANCELLED)]` projects `u4` and
  **nothing** after it; the same path with `tf_c(COMPLETED)` likewise; an
  **archived** conversation `[… ts_c, cmp(parts set), tf_c]` projects its
  original entries and **no** summary; the new conversation `[cmp, u4]`
  projects the summary as a user message; two stacked closed compaction
  brackets are both skipped; an *open* compaction bracket at the tail is
  skipped to the end; and — the regression guard — a *conversational*
  `TurnFinish(CANCELLED)` still projects the interrupted marker, one predicate
  separating the two.
- **`test_context_manager.py`** — `parts=[Text(35 chars)]` → 8;
  `parts=None` → 0; `parts=[]` → 0; `parts=[Text, Image]` → text + 1000 (the
  `_media_parts` branch); a subclass overriding `CHARS_PER_TOKEN` scales it.
- **`test_models.py`** — the new `CompactionEntry` shape; `extra="forbid"`
  rejects `summary=`/`summarized=`/`details=`; a full session containing a
  compaction round-trips through `model_dump_json`/`model_validate_json`;
  `Entry(id=None, created_at=None)` is valid and `AnyEntry` still
  discriminates; `turn_count` excludes compaction brackets, counts a failed
  and an open conversational bracket, and is scoped to the active conversation
  (an archived conversation's turns are not counted).
- **`test_utils.py`** — `pretty_print` renders a committed compaction
  (`replaced N entries` + the text) and a `parts=None` one; the local
  `COMPACTED_SESSION` fixture in that file migrates to the new shape.

### 11.6 Integration: `tests/agent/test_runner_compaction.py`

House style throughout: precondition (a literal, `model_copy(deep=True)`, or a
cold reload) → one action → full-object postcondition. `DeterministicRunner`
with scripted `ids` and `now=1000`; `FakeCompactionPolicy`; `FauxProvider` for
the turn that follows.

Every scenario asserts the same three things unless noted, which is exactly
what the user asked for: **the new conversation is what it should be, the
pre-existing entries are unmutated, and the old conversation is in history.**

**A. The successful transition**

| Test | Postcondition |
|---|---|
| `test_transition_archives_the_old_path_and_installs_the_new_one` | full `runner.session ==` literal: `conversation_history == [c0, c1]`, `c1.nodes == pre + [tf_c]`, `c1.status == IDLE`, active `= Conversation(id="c2", nodes=[cmp, u4], created_at=1000, updated_at=1000, status=PENDING)` |
| `test_every_carried_and_compacted_entry_is_unmutated` | `{k: v for k, v in session.entries.items() if k != "cmp"} ==` the pre-drive mapping; keys grew by exactly `{ts_c, cmp, tf_c}` |
| `test_compacted_nodes_lists_exactly_the_replaced_ids` | `cmp.compacted_nodes == ["cmp0","u1","ts1","a1","te1","te2","pr1","a2","tf1", …]` in path order, **without** `ts_c` |
| `test_the_bracket_stays_behind_and_only_the_entry_carries_over` | `c1.nodes[-3:] == [ts_c, cmp, tf_c]`; `c2.nodes[0] == cmp`; no marker id appears in `c2` |
| `test_the_pruned_referent_and_archived_entries_stay_reachable` | `entries["te0"]`, `entries["u0"]`, `c0` byte-identical |
| `test_tool_executions_index_and_old_usages_survive` | `tool_executions` unchanged; `usages["c0"]`/`usages["c1"]` intact plus the compaction's record; no `usages["c2"]` |
| `test_the_entry_is_self_describing_on_the_new_path` | `source`, `started_at=1000`, `ended_at=1000`, `llm_config` (the policy's, not the session's), `metadata` from the plan |
| `test_a_plan_with_several_created_entries_commits_them_in_order` | ids in `nodes` order, `parent_id` threaded left→right, one shared `created_at` |
| `test_a_plan_that_opens_with_a_created_entry_gives_it_no_parent` | `created[0].parent_id is None` |
| `test_a_plan_may_interleave_created_entries_between_carried_ids` | committed as given; `c2.nodes` matches the plan exactly |
| `test_a_plan_may_reorder_carried_ids` | committed as given (the policy chose the path) |
| `test_a_fold_everything_plan_leaves_the_compaction_entry_as_the_leaf` | `c2.nodes == [cmp]`, status `IDLE`, `RunResult(IDLE, COMPLETED, [])` — the old `AttributeError` path |
| `test_a_keep_last_assistant_plan_leaves_an_assistant_leaf` | status `IDLE`, `RunResult` builds — the other `AttributeError` path |
| `test_a_full_carry_plan_commits_with_an_empty_compacted_span` | `compacted_nodes == []` (not `None`) — core does not second-guess the policy |
| `test_the_next_request_projects_only_the_new_path` | `faux.requests[0].messages ==` the summary as a synthetic user message + the carried nodes; nothing from the archived span |
| `test_the_compacted_session_round_trips_through_json` | reload → `AgentSession ==` |
| `test_events_are_scheduled_started_finished` | exact list, exact payloads (`Scheduled.entry.started_at is None`, `Started.entry.llm_config is None`, `Finished.conversation_id == "c2"`, `Finished.created == [...]`) |
| `test_the_policy_is_handed_a_deep_copy_and_the_live_session` | `seen[0][2] is not session.entries["cmp"]`; `seen[0][0] is session` |
| `test_the_policy_is_offered_the_path_without_the_bracket_turn_start` | `seen[0][1] == tuple(c1.nodes without "ts_c")`, ends with `"cmp"`; it is a `tuple`, and `"ts_c" not in seen[0][1]` |
| `test_carrying_the_offered_nodes_verbatim_is_a_legal_full_carry` | `plan.nodes = list(nodes)` → commits, `compacted_nodes == []` |
| `test_a_policy_that_writes_to_its_copy_and_fails_cannot_inject_a_summary` | `mutate=True, raises=…` → `entries["cmp"].parts is None`, `project()` emits no synthetic summary |

**B. Nothing to compact**

- `test_policy_returning_none_closes_completed_without_transitioning` — path
  `pre + [tf_c(COMPLETED)]`, `parts is None`, `conversation_history == [c0]`,
  `CompactionFinished(conversation_id=None, created=[])`.
- `test_a_span_of_only_a_compaction_entry_is_committed` — the inverse of the
  deleted G5 test: a plan replacing `RICH_SESSION`'s `cmp0` and nothing else
  **transitions normally**, `compacted_nodes == ["cmp0"]`. Core does not judge
  whether re-summarizing a summary was worthwhile.
- `test_scheduling_on_an_empty_session_compacts_nothing` — fresh session,
  `schedule_compaction()`, policy returns `None` → path
  `[ts_c, cmp, tf_c]`, status `IDLE`. `None` is now the only route into this
  group.
- `test_a_noop_does_not_bury_a_queued_message` — `RICH_SESSION`: after the
  no-op the drive goes on and answers `u4` in the same run.

**C. Rejected plans** (each: `CompactionPlanError` raised for `source=USER`,
bracket `ERRORED` with the message in `TurnFinish.error`, path unchanged apart
from `tf_c`, `conversation_history == [c0]`, `parts is None`)

One test per rule: unknown id; id on the archived `c0`; duplicate id; empty
plan; omits the compaction entry; G2 by conversation id; G2 by path (a policy
that appends to `session.active_conversation.nodes`); G3 `parts=None`; G3
`parts=[]`; G3 whitespace-only. Plus:
- `test_a_plan_carrying_the_bracket_turn_start_is_rejected` — the policy
  reaches around `nodes` into `session.active_conversation.nodes` and carries
  `ts_c`. Rejected by rule 5 with the ordinary not-on-the-path message, no
  special case. This is the test that pins PRD §5's whole point: the hazard is
  now an error the author sees.
- `test_an_image_only_summary_is_accepted` — G3's non-text branch commits.
- `test_usage_is_recorded_for_a_rejected_plan` — the tokens were spent. Pair it
  with the G2 variants, which are the ones that could not record at all if the
  write ran before `check_snapshot` (§5.4): a policy that **replaces**
  `session.active_conversation` must still close `ERRORED` with G2's message,
  not with an `AgentError` from the ledger.
- `test_a_rejected_plan_leaves_the_entry_projecting_nothing`.
- `test_a_policy_returning_provider_counter_names_fails_in_its_own_code` — the
  `UsageCounters` construction raises inside `compact()`, so this arrives as an
  ordinary policy raise (bracket `ERRORED`, USER raises / POLICY degrades) and
  never as a mid-step crash with the bracket left open.

**D. Failures and the source split**

| Test | Behavior |
|---|---|
| `test_a_user_policy_raise_closes_errored_and_propagates_through_await` | `TurnFinish(ERRORED, error="kaboom")`; the exception reaches `await runner.run()` |
| `test_an_iterated_user_failure_yields_finished_before_raising` | events `[Scheduled, Started, Finished(ERRORED)]`, then the raise |
| `test_a_policy_source_failure_degrades_and_the_queued_turn_still_runs` | no raise; `Finished(ERRORED, error=…)`; the drive answers `u4`; final `c1` holds the failed compaction bracket **and** the new turn |
| `test_a_degraded_failure_is_only_visible_on_the_event_stream` | `await runner.run()` returns a normal `RunResult(IDLE, COMPLETED)` |
| `test_a_client_timeout_error_from_the_policy_closes_timed_out` | `ClientTimeoutError` → `TIMED_OUT`, not `ERRORED` |
| `test_a_provider_error_from_the_policy_closes_errored` | `ProviderAPIError` → `ERRORED` |
| `test_the_policy_deadline_closes_timed_out` | `hang=True` + `client_completion_timeout_in_ms=50` → `TIMED_OUT`; USER raises, POLICY degrades (two tests) |
| `test_middleware_raising_during_preparation_leaves_the_path_unchanged` | `before_entry_written` raises on the content mutation → `ERRORED`, no transition, `conversation_history == [c0]` |
| `test_middleware_raising_while_closing_a_failed_bracket_leaves_it_open` | the bracket stays open → resumable, status `PENDING` |
| `test_repeated_policy_failures_burn_one_attempt_per_drive` | two drives → two closed brackets, one entry each, no stacking |

**E. Cancellation** (never two timers: the policy hangs on an event the test
releases, and `on_event` is the cancel trigger)

| Test | Behavior |
|---|---|
| `test_cancelling_between_scheduled_and_started_closes_cancelled_without_calling_the_policy` | `policy.seen == []`; `tf_c(CANCELLED)`; path otherwise unchanged |
| `test_cancelling_mid_summary_closes_cancelled_and_tears_the_policy_down` | the policy task is cancelled; no transition |
| `test_a_plan_arriving_within_the_grace_window_is_discarded_by_the_cancel` | `parts is None`, no transition, **usage recorded** (the tokens were spent) |
| `test_a_cancel_stops_the_drive_and_the_queued_message_is_not_answered` | `faux.requests == []`; status derives `PENDING` via the skip rule → `u4` still drives on the next run, and **that run's request carries no interrupted marker** (§7's bracket rule, asserted on the wire) |
| `test_a_parked_cancel_flushes_on_the_next_drive_without_calling_the_policy` | `COMPACTION_CANCEL_PARKED_SESSION` cold → events `[Finished(CANCELLED)]` only (no `Scheduled`, no `Started`), `policy.seen == []` |
| `test_an_immediate_cancel_on_start_parks_the_compaction_flush` | `start()` opened a compaction bracket; `run.cancel()` before the first tick → `CANCELLED`, no policy call |
| `test_a_second_cancel_inside_a_compaction_bracket_raises_already_cancelling` | existing semantics hold |
| `test_run_result_after_a_cancelled_compaction` | `RunResult(status=PENDING, outcome=CANCELLED, pending_approvals=[])` |
| `test_a_cancel_with_a_non_cancelled_outcome_still_stops_the_drive` | `cancel(TurnOutcome.ERRORED)` **and** `cancel(TurnOutcome.TIMED_OUT)` inside a compaction bracket → `tf_c` carries that outcome, `faux.requests == []`, the queued `u4` is not answered. The regression guard for §5.1's flag: keying the drive-stops check off `_closed_outcome == CANCELLED` passes every other test in this group and fails only these two |

**F. Crash, resume, suspend (G6)**

| Test | Behavior |
|---|---|
| `test_a_scheduled_compaction_survives_a_reload_and_resumes_in_place` | `COMPACTION_SCHEDULED_SESSION` cold → `PENDING`; the drive reuses `cmp` (no new id), one bracket only |
| `test_an_interrupted_compaction_resumes_with_the_same_entry_and_keeps_started_at` | `started_at` stays `600`; `ended_at=1000`; stale `RUNNING` self-healed at construction |
| `test_a_closed_failed_bracket_is_never_retried` | `COMPACTION_FAILED_SESSION` cold → `IDLE`; `run()` raises "Nothing to run"; `should_compact` never consulted |
| `test_a_closed_completed_bracket_does_not_bury_a_queued_message` | `COMPACTION_BURIED_SESSION` cold → `PENDING`; the drive answers `u4` and opens **no** second compaction bracket |
| `test_no_second_bracket_ever_piles_up_behind_the_first` | after a resume, `[ts_c, cmp, tf_c]` occurs exactly once in `nodes` |
| `test_a_committed_compaction_inside_a_counterfeit_bracket_is_not_re_run` | cold session whose active path is `[ts3, cmp(parts set, compacted_nodes=[…]), u4]` → `should_calls == 0`, `policy.seen == []`, `cmp` byte-identical after the drive. G6's content test; keying on bracket shape re-runs `compact()` and overwrites `compacted_nodes` |
| `test_schedule_compaction_raises_on_a_counterfeit_bracket` | same session → `AgentError` (the open-turn guard), nothing written. Without the test in `open_compaction_entry`, this returns `cmp.id` and silently does nothing |
| `test_an_interrupted_entry_still_resumes_because_parts_is_none` | the companion guard — `COMPACTION_INTERRUPTED_SESSION` (`parts is None`) resumes normally, so the content test costs the real resume path nothing |
| `test_suspending_a_lazy_run_mid_compaction_leaves_the_bracket_open` | `break` out of the iteration → bracket open, status `PENDING`, no `TurnFinish`; a later `run()` resumes the same entry |
| `test_an_on_event_raise_during_started_leaves_the_bracket_open` | crash semantics, resumable |
| `test_a_reloaded_compacted_session_drives_normally` | `POST_COMPACTION_SESSION` cold → the next turn projects only `c2` |

**G. The trailing entry (explicitly requested)**

| Test | Behavior |
|---|---|
| `test_a_carried_trailing_user_message_keeps_driving` | plan carries `u4` → `c2` leaf is `u4` → `PENDING` → the same drive answers it; `RunResult.outcome` is the **turn's** close |
| `test_a_folded_trailing_user_message_is_committed_and_the_question_is_lost` | plan drops `u4` → committed, `c2` leaf is `cmp`, status `IDLE`, `faux.requests == []`, `u4` still in `entries` and in `compacted_nodes`. The documented silent failure, asserted so it cannot change by accident |
| `test_a_trailing_turn_finish_gives_a_compaction_only_drive` | `RICH_IDLE_SESSION` + `schedule_compaction()` → `RunResult(IDLE, COMPLETED, [])`, no LLM call |
| `test_a_carried_failed_turn_finish_is_retried_in_the_same_drive` | plan carries `tf2(ERRORED)` as the leaf → `PENDING` → the drive retries that turn |
| `test_a_carried_cancelled_turn_finish_projects_the_interrupted_marker` | wire assertion on the next request |
| `test_a_carried_phantom_open_turn_is_committed_as_given` | plan carries `ts3` without `tf3` → status `PENDING`, the next drive calls the model; the hazard is the policy's |
| `test_a_phantom_bracket_around_the_summary_does_not_re_run_the_compaction` | plan carries `ts3` with `cmp` immediately after it → `c2 = [ts3, cmp, u4]`, which §4.3 reads as an open *compaction* bracket. The next drive must leave `cmp` byte-identical (`compacted_nodes` intact) and drive it as a phantom turn instead. Pairs with group F's counterfeit test — this one gets there through a committed plan rather than a cold literal |
| `test_a_carried_pending_approval_execution_derives_awaiting_approval` | the phantom-turn + gate variant |

**H. Preconditions and API**

`schedule_compaction` returns the id and writes exactly `[ts_c, cmp]` with
`source=USER`, `started_at=None`, `parts=None`, status `PENDING`; a second call
is idempotent (`session ==` the post-first-call literal, `ids` script
unconsumed); an open conversational turn raises `AgentError` with nothing
written; `AWAITING_APPROVAL` (`GATED_SESSION`) raises; `CANCELLING`
(`CANCEL_PARKED_SESSION`) raises; no policy configured raises; `post_message`
raises while scheduled and again after a cold reload, and is legal once the
drive has run; `start()` with `should_compact=True` opens a compaction bracket
instead of a `TurnStart` (assert `nodes` has no bare `TurnStart` before `ts_c`)
and then drives the turn; `start()` with an already-scheduled compaction opens
nothing extra; a scheduled compaction wins over a `True` `should_compact` (one
compaction, `source=USER`); `should_compact` is not consulted when a
conversational turn is open (`CLEARED_SESSION`, `should_calls == 0`), when the
session is `IDLE` (`run()` raises first), or when no policy is configured;
`should_compact` still `True` after a commit compacts only once in that drive
and again on the next (G4); a `should_compact` that raises propagates from both
`run()` and `start()`; two runners with different policies are not equal and
with equivalent policies are.

**I. Usage, `llm_config`, context**

- usage lands in `usages["c1"]["cmp"]` — the pre-compaction conversation, where
  the request was made — with the plan's counters;
- no usage record when the policy returns `None`, raises, or is cancelled
  before producing a plan;
- `llm_config` is the plan's (a cheaper model than the session's), proving the
  runner does not stamp it;
- `context_tokens` is 0 on the `Scheduled` snapshot and recalculated when
  `parts` land (assert the number on the stored entry);
- `before_entry_written` has the final say on the compaction entry's
  `context_tokens` (nothing recalculated after it).

**J. Middleware (`test_runner_middleware.py`)**

- `test_before_entry_written_sees_every_entry_a_compaction_writes` — the exact
  ordered list of `(type, id)` observed: `turn_start ts_c`, `compaction cmp`
  (append), `compaction cmp` (the `started_at` stamp), `compaction cmp` (the
  content mutation), each created entry, `turn_finish tf_c`.
- `test_the_turn_hooks_are_not_invoked_for_the_summarization_call` — asserts
  the **absence**: `before_llm_call`, `after_llm_response`,
  `build_model_string`, `build_tool_list` record zero calls across a
  compaction-only drive, and exactly one each across the turn that follows.
- `test_before_entry_written_may_redact_the_summary_before_it_persists` — the
  rewritten content is what lands and what projects.

**K. `RunResult`**

Compaction-only → `(IDLE, COMPLETED, [])`; compaction + turn → the turn's
outcome; degraded failure + turn → `COMPLETED` (the turn's); cancel →
`(PENDING|IDLE, CANCELLED, [])`; leaf-is-`CompactionEntry` and
leaf-is-`AssistantMessage` build without raising; the hard-max-steps regression
now reports `PENDING`.

### 11.7 Suite hygiene

**The baseline is green.** `uv run py.test tests/` reports `910 passed, 14
skipped` with no ignores. An earlier draft of this document said the suite was
red because `tests/agent/contrib/compaction/` held seven untracked files
importing a package that does not exist; that directory has since been removed,
so there is nothing to delete and no collection error to work around.

One leftover remains: `luca/agent/contrib/compaction/` exists but is empty
(a stale `__pycache__` and nothing else). Remove the directory.

---

## 12. Work sequence

Each step ends with `uv run py.test tests/` green.

| # | Step | Gate |
|---|---|---|
| 0 | Remove the empty `luca/agent/contrib/compaction/` (§11.7) | baseline unchanged: 910 passed, 14 skipped |
| 1 | `Entry.id` / `created_at` optional + all seven placeholder sites + docstrings | existing suite green with updated assertions |
| 2 | `CompactionEntry` reshape, `CompactionSource`, the shared is-a-compaction-bracket predicate (§4.3), projection incl. §7's bracket rule / context / utils / `test_models` / `test_projection` / `test_context_manager` / `test_utils` | unit tests for the new shape |
| 3 | `core/compaction.py` (`CompactionPolicy` with the 3-arg `compact`, `CompactionPlan`, `UsageCounters`, `ConversationSnapshot` with `nodes` **and** `offered`, `validate_plan`, `check_snapshot`, `has_content`) + `CompactionPlanError` + `test_compaction.py` | the validator is fully covered with no runner in sight; rule 5 checks `offered`, G2 checks `nodes` |
| 4 | Ledger: `put_entry`, `transition_conversation`, `open_compaction_entry` (incl. G6's `parts is None` test, §4.3), skip rule + `test_ledger.py` additions | the transition and the derivation matrix pinned without an engine |
| 5 | Runner: `_closed_outcome` + the two per-run flags (§5.1a), `_complete_entry`/`_prepare`/`_persist_entry` split, `compaction_policy=` (plus the `PluginAgentSessionRunner` / `DeterministicRunner` pass-throughs, §5.8), `schedule_compaction`, the step incl. `_snapshot_conversation`, `_open_bracket_for_start` + its guard release, `_build_run_result`, `__eq__` | `test_runner_compaction.py` groups A–K; `test_runner_limits.py:127` updated |
| 6 | Events + `core/__init__.py` exports | event payload tests |
| 7 | `turn_count` fix | `test_models.py` |
| 8 | Docs: new `12-compaction.md` + the six files in the PRD's §17 table, and `AGENTS.agent.md` (design principle 4's "only mutable entry type", the `core/` layout, event tiers, principle 11's derivation rules, the test-file table) | — |
| 9 | Contrib: `save_session` atomicity (G1), `/compact`, the transcript cell | `tests/agent/contrib/tui/` |

Steps 1–4 are independently landable and independently valuable. Step 5 is the
only one that cannot be split.

---

## 13. Decisions this proposal makes beyond the PRD

Recorded so they are not re-litigated, and flagged because each is a judgment
call rather than a transcription.

1. **Two ledger doors, not three.** Plan-created entries are stored inside
   `transition_conversation`, so no compaction write happens outside the atomic
   region.
2. **The transition door is compaction-agnostic** (`transition_conversation`),
   so it could serve a fork or a branch unchanged. This is the concrete form of
   "the ledger just stamps the past and present conversations". The ledger's
   *reads* do know the entry type and bracket shape — `open_compaction_entry`
   and the `derive_status` skip rule cannot answer their questions otherwise —
   but no ledger method knows what a policy is or how a summary was made.
3. **`turn_count` counts brackets that are not compaction brackets**, not
   brackets with an `AssistantMessage` — same compaction result, no behavior
   change for any existing session (§2), and it reuses §4.3's one predicate.
4. **The runner races `compact()` itself** — `asyncio.timeout` from
   `client_completion_timeout_in_ms` around `_race_cancellation` with
   `llm_completion_cancellation_grace_period`. The PRD's "the runner's existing
   turn timeout" is a client kwarg that cannot reach a policy's own call (§5.4).
   The value goes through `_ms_to_seconds`, and the default (`Inf`) means **no
   deadline at all**, matching the conversational LLM call.
5. **The drive returns after the compaction step when the re-derived status is
   `IDLE`** — missing from the PRD and required, or a `schedule_compaction()`
   on a finished session would call the model with no user input (§5.1).
6. **`compacted_nodes` is computed over the path before the compaction
   bracket**, so `ts_c` never appears in "ids this entry replaced" (§5.5).
7. **A cancel discards a plan that arrived within the grace window** — no
   transition, `parts` stays `None`. Deliberately unlike the LLM path, which
   records a within-grace answer: adding a node is not rewriting a
   conversation (§5.4).
8. **`started_at` is stamped once**, on the first attempt, and survives a
   resume (§5.4).
9. **A `should_compact` that raises propagates** — a swallowed exception is
   indistinguishable from a policy that declines (§5.6).
10. **A full-carry plan is committed with `compacted_nodes == []`**, not turned
    into a no-op: an empty span is the policy's judgment, and `[]` vs `None` is
    exactly the distinction the field's type exists to record. Free once
    decision 12 removed the guard that would have discarded it.
11. **G3 forecloses "trim without summarizing"** through a plan; such a policy
    must emit a marker part or return `None` (§3).
12. **G5 is deleted; there is no `_is_noop`, and `None` is the only "nothing to
    do".** Core never judges whether a compaction was worth doing — that is
    `should_compact`'s question, asked before the LLM call rather than after it.
    The guard was the one place the runner validated meaning instead of
    structure, it discarded a valid plan the policy had already paid for, and it
    missed the `[ts, cmp, tf]` span anyway. G4 keeps a runaway to one attempt per
    drive; the default contrib policy owns the floor and the gauge (PRD §11,
    §5.4).
13. **`RunResult.status` becomes derived for every run**, which changes the
    hard-max-steps result from `IDLE` to `PENDING` and updates one existing
    assertion (§5.7). Its sibling, **`_closed_outcome`, has to be built** —
    it does not exist in the runner today, and once `_build_run_result` stops
    reading `nodes[-1].outcome` it is what every ordinary turn's outcome
    depends on, not just compaction's (§5.1a).
14. **`_persist_entry` takes `recalculate=False`** for the tool-execution final
    persist, so "middleware has the final say on context" stays literally true
    after the update door is generalized (§5.2).
15. **`plan.usage` is a typed `UsageCounters`, not `dict[str, int]`** (§3). The
    PRD left this open ("a nicety for the day a second producer exists"); the
    day is now, because the untyped dict put its only validation inside
    `record_usage` — a runner-internal write that cannot produce a clean plan
    rejection. Typing moves the failure into the policy, and the usage-key
    rejection rule disappears.
16. **The usage write is guarded and preceded by `check_snapshot`** (§5.4).
    Unguarded, a `record_usage` raise escapes the step with the bracket open —
    the "resume me" state — so the next drive replays the same failing policy
    call with `should_compact` never consulted. G2 runs first so a policy that
    replaced the active conversation still gets G2's error rather than the
    ledger's.
17. **The drive-stops-after-cancel check is a flag set by the step**
    (`_compaction_consumed_cancel`), not `_closed_outcome == CANCELLED` (§5.1).
    `cancel()` takes its outcome as an argument and only `COMPLETED` is
    forbidden, so the value test would let `cancel(ERRORED)` fall through into
    the conversational turn.
18. **`start()` releases the one-run guard if `_open_bracket_for_start` raises**
    (§5.6). Otherwise decision 9 leaves the runner permanently unusable after a
    single policy exception.
19. **A compaction bracket is identified by adjacency** — the node right after
    the `TurnStart` — not by "contains a `CompactionEntry` anywhere" (§4.3).
    The entry outlives its bracket and lands where the policy puts it, so the
    weaker test couples a framework classification to policy behavior. The
    same predicate serves `derive_status`'s skip rule, the resume check,
    `turn_count` (§2) and §7's projection rule — four call sites, one
    definition.
20. **Two runner subclasses forward `compaction_policy=`** (§5.8):
    `PluginAgentSessionRunner`, without which the TUI cannot use compaction at
    all, and `DeterministicRunner`, without which none of §11.6 can be written.
21. **Projection is positional, and the rule lives on `project()`** (§7, PRD
    §4/decision 10): a compaction bracket projects as nothing, a bare
    `CompactionEntry` projects its `parts`. One rule, not a content test per
    marker. Without it a cancelled compaction puts
    `"[Request interrupted by user]"` on the wire, durably, about a question
    the model never saw — and an archived conversation projects both the
    original history and a summary of it. Classifying a bracket needs the
    path, so pushing it down to `project_turn_finish` would mean changing a
    public override signature for a fact the caller already holds.
22. **The drive's post-step `IDLE` return is gated on the step having run**
    (§5.1). Unconditional, it lets any cached-vs-derived status drift end a
    drive that has nothing to do with compaction.
23. **`ConversationSnapshot` carries two paths, `nodes` and `offered`** (§3).
    G2 asks whether the live path moved, which needs the markers; rule 5 asks
    whether an id was offered, which must not include them. Storing both keeps
    the strip in one place — `_snapshot_conversation` (§5.4) — and the strip is
    exactly where a bug would be silent. `offered` is also the `nodes`
    argument passed to `compact()`, so there is one construction, not two.
24. **G6's resume test is `entry.parts is None`, not the bracket's shape, and
    it lives in `open_compaction_entry()`** (§4.3, PRD §11). A plan can
    counterfeit an open compaction bracket; it cannot counterfeit a committed
    entry's content. Putting it in the ledger read rather than in the step
    serves both callers with one condition and keeps `schedule_compaction()`
    honest — otherwise it reports itself idempotent over a committed entry and
    silently does nothing. Without the test,
    `test_a_committed_compaction_inside_a_counterfeit_bracket_is_not_re_run`
    fails by overwriting `compacted_nodes` — the only way this design can
    damage an existing audit record.
