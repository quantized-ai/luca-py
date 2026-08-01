# The subagent worker pool — `subagents_max_workers`

**Status: design agreed. This document is the specification.** Every decision in
it is settled; nothing is left open.

It is written to be read cold. §0 is the background the rest depends on; skip it
only if you already know how a subagent conversation is created, driven, and
observed. §1–§4 are the design. §5 is a set of worked timelines. §6 audits every
existing behavior the change touches. §7 records the alternatives that were
analyzed and rejected, with the reasons. §8–§10 are configuration, tests and
work. §11 specifies the observability surface — the lifecycle events an
application uses to show which subagents are running and which are waiting.

Every line reference is to the tree at commit `91f6be1`.

---

## 0. Background

### 0.1 A session holds many conversations

An `AgentSession` is not one conversation. `session.conversations` is a dict
keyed by id; `session.main_conversation_id` names the one the user talks to.
Every other row in that dict is either an archived predecessor (from a
compaction) or a **subagent**.

Each conversation carries a `depth`: the main conversation is `0`, a subagent
spawned from it is `1`, a subagent spawned by *that* one is `2`.
`RuntimeConfig.subagents_max_depth` caps it (default `1`).

A subagent is linked into its parent's path by a `ChildConversation` entry —
appended unresolved when the child is created, and given an `execution_result`
when the child's turn closes.

### 0.2 How a subagent is created

Spawning is a handshake between an ordinary tool and the runner. A tool whose
`ToolSpec.output_schema` declares a property named `is_subagent_spawn` is a
spawn tool; when such a tool completes with
`structured_content["is_subagent_spawn"] is True`, the runner creates the child.

Creation (`_spawn_one`, `runner.py:2604`) does exactly three things:

1. mints a `Conversation` row with `depth = parent.depth + 1`;
2. appends **one** `UserMessage` to it — the spawn prompt, and the only user
   message that conversation will ever receive;
3. appends the `ChildConversation` link to the parent's path.

**It does not start anything.** A freshly spawned child conversation holds
exactly one node: its seed `UserMessage`.

### 0.3 How a conversation gets driven

`runner.run()` and `runner.start()` return an `AgentRun` handle. A handle drives
**one** conversation — the guard is per conversation, not per session, because
that is what parallel subagents are: several conversations advancing at once,
each with exactly one engine behind it.

Two pieces of runtime state matter throughout this document:

| | |
|---|---|
| `runner._runs: dict[str, AgentRun]` | the conversations that currently have a **live drive**. `_begin_run` (`runner.py:1268`) inserts; `_end_run` (`runner.py:1344`) removes. |
| `AgentRun._children: dict[str, AgentRun]` | the handles a run has adopted for the subagents it spawned, in insertion order. |

`_begin_run` / `_end_run` are airtight: *every* path through the handle calls
them — the eager constructor (`runner.py:318`), the background task's `finally`
(`runner.py:721`), the lazy iterator on exhaustion or error
(`runner.py:581`/`587`/`598`), and a suspending `__aexit__` (`runner.py:526`).
Nothing drives a conversation without being in `_runs` for the duration.

Today, `_start_children` (`runner.py:2677`) constructs one **eager** `AgentRun`
per spawn, and the eager branch of `AgentRun.__init__` (`runner.py:309-329`)
immediately calls `_begin_run`, opens the turn bracket, and
`loop.create_task(self._consume())`. So N spawns become N concurrent tasks the
moment the handshake lands.

### 0.4 Who owns a child's lifecycle — `autostart_subagents`

`run(autostart_subagents=True)` — the default, and implied by `start()` — hands
the **framework** the children's lifecycles: each child begins immediately,
advances on its own, and forwards its events onto the parent's stream.

`run(autostart_subagents=False)` hands the **application** a lazy, unstarted
handle per child (`run.child(cid)`), and the obligation to drive or cancel every
one.

### 0.5 The parent waits — the two parking points

A parent's turn cannot end until every child it spawned has resolved: an
unresolved `ChildConversation` on the path is unprojectable, so calling the model
would raise. The drive loop (`_drive_loop`, `runner.py:2241`) therefore parks,
and there is exactly **one** function it parks in — `_await_subtree`
(`runner.py:2853`) — reached from two call sites:

- `runner.py:2367` — this conversation has an execution awaiting approval;
- `runner.py:2382` — this conversation has unresolved children.

`_await_subtree` returns `True` ("loop again") or `False` ("the drive should
return"), the latter when nothing in the subtree can advance. That single rule is
why a gated subagent's drive ends outright while a gated parent whose children
still work keeps waiting.

### 0.6 Cancellation is a signal that a drive consumes

`cancel()` (`runner.py:1095`) appends a durable `CancelRequested` entry, trips
the live run's cancellation token, **cascades to every unresolved descendant**
(`_cascade_cancel`, `runner.py:1154`), and returns. It performs no bookkeeping
itself. The next drive step consumes the request: `_drive_loop`'s step 0 sees it
and calls `_wind_down_async` (`runner.py:2915`), which waits for the
already-cancelled children's drives to settle, resolves every unresolved link
with a cancellation result, terminalizes the still-`PENDING` executions, and
closes the turn.

Wind-down **never calls the model and never dispatches a tool body**. It
terminalizes pending executions without dispatching them, and it resolves child
links *without running the result tool*. It is pure bookkeeping. That fact is
load-bearing in §4.9.

### 0.7 Status is derived, and a never-started child is `BUSY`

`session.get_conversation_status(id)` recomputes status from the entries on every
call (`models.py:1274`). A conversation whose last node is a `UserMessage`
derives **`BUSY`** (`models.py:1350`). A spawned-but-never-driven child is exactly
that shape, so it is `BUSY` — "there is work here, and a `run()` would do it".

### 0.8 How an event reaches the application

There are exactly **two channels**, and §11 uses both.

**The engine yields.** `_drive` (`runner.py:2178`) is an async generator. Every
event it `yield`s is pulled by the `AgentRun` driving that conversation — either
directly (a lazy run: iteration *is* the engine) or by the background task (an
eager run: `_consume`, `runner.py:690`, drains it into a buffer).

**The run tree's inbox.** `AgentRun._inbox` is an `asyncio.Queue`. A child run
pushes each of its events onto its parent's inbox (`_forward`, `runner.py:731`),
and `_next_own_or_forwarded` (`runner.py:603`) races the parent's own engine step
against that queue, so a parent's stream is the stream of its whole subtree.
`_consume` re-forwards *everything* it pulls, inbox items included, so an event
placed on any run's inbox bubbles all the way to the root handle and reaches
`on_event` like any other.

The second channel is what makes it possible to emit an event from code that is
**not** inside the drive generator — `asyncio.Queue.put_nowait` is synchronous
and never blocks. `_publish_approval` (`runner.py:492`, `2906`) already does
exactly this for the `run.approvals` stream.

---

## 1. What we are building

One new persisted knob:

```python
class RuntimeConfig(BaseConfigModel):
    ...
    # How many subagent conversations may be DOING WORK at the same time,
    # across the whole session. `Inf` (the default) means no limit.
    subagents_max_workers: int = Inf
```

**We cap the number of subagents that are working at any instant, session-wide,
and we schedule the rest.** A spawn that would exceed the cap creates its child
conversation exactly as it does today; the child simply waits for a slot before
its drive begins.

Alongside it, **three lifecycle events** tell an application which subagents are
running, which are waiting, and which are done (§11).

### 1.1 The rule

> **A slot is held by a conversation only while it is doing its own productive
> work — calling the model, running a tool body, deciding an approval, or
> resolving a finished child. No conversation holds a slot while it waits for
> another conversation, waits for a human, or winds a cancelled turn down.**

Everything else in this specification is a consequence of that one sentence.

### 1.2 What holds a slot, stated exhaustively

| conversation state | holds a slot? | why |
|---|---|---|
| main conversation (`depth == 0`), any state | **never** | it is not a subagent. The cap bounds subagent work; the main conversation always has the floor |
| subagent, drive live, calling the model / running tools / deciding | **yes** | it is working |
| subagent, drive live, running the result tool for a finished child | **yes** | it is working |
| subagent, drive live, parked in `_await_subtree` waiting on its own children | **no** | it is waiting, not working — it releases on entry and re-acquires on wake |
| subagent, drive live, winding a cancelled turn down | **no** | wind-down is bookkeeping, and it must never wait for a slot |
| subagent, drive ended because it gated on an approval | **no** | its drive ended; `_end_run` released the slot |
| subagent, drive ended because its turn finished | **no** | same |
| subagent, drive ended because the parent suspended (`__aexit__`) | **no** | same |
| subagent spawned but never started (**queued**) | **no** | it has not begun |
| subagent with an unconsumed `CancelRequested` | **no**, and it is never granted one | its parent's drive will flush it; starting a drive purely to wind it down is waste (§4.3) |

### 1.3 One-paragraph summary

Spawning always succeeds and always creates the child conversation. The framework
starts as many children as the cap allows and leaves the rest queued in spawn
order, holding one durable node each (their seed message). When a working
subagent finishes, gates, parks to wait on its own children, or begins winding
down, its slot returns to the pool and the next waiter is started. The main
conversation never competes for a slot. Nothing about the pool is persisted; it
is rebuilt from the session's own state on the next `run()`. Every start, pause
and finish is announced as an event, so an application can render the pool
without inspecting the session.

---

## 2. Why this is the right model

### 2.1 The main conversation is exempt, and that is what makes it safe

The main conversation spends most of a subagent turn waiting for its children.
Because it holds no slot, that wait costs nothing. If it counted,
`subagents_max_workers=1` would mean *no subagent ever runs*.

### 2.2 A subagent that spawns is both a worker and a waiter

Once `subagents_max_depth > 1`, a subagent can spawn its own subagents. That
conversation is then in the pool (it is a subagent) **and** waiting on the pool
(its children need slots). This is the classic bounded-pool hazard: a worker that
submits work to its own pool and blocks on the result.

§1.1's rule removes it: the parent steps out of the pool the moment it starts
waiting. §7.1 shows exactly what happens if it does not.

### 2.3 The pool is never larger than the tree's *width*

A parent never calls the model while one of its children is unresolved: the drive
loop parks at `runner.py:2379-2384` and only reaches the model call at step 4
below it. Its only work while children run is resolving a finished child's link —
a short, private tool call.

So along any root-to-leaf chain, **at most one conversation is working at a
time**. Depth adds *sequence*, not *concurrency*. `subagents_max_workers` should
be sized to the fan-out you want (how many siblings work at once), never derived
from `subagents_max_depth`.

### 2.4 The pool cannot deadlock — the argument in full

**Invariant.** Every slot-holder is doing productive work that completes without
waiting on another conversation.

Three states could threaten that invariant, and each is excluded by construction:

| state | why it cannot hold a slot |
|---|---|
| waiting on its own children | releases on entry to `_await_subtree` (§4.7) |
| waiting on a human at an approval gate | its drive ends outright, so `_end_run` releases (§4.8) |
| waiting for its cancelled children to settle during wind-down | releases on entry to `_wind_down_async` (§4.9) |

**Proof of progress.** Suppose the pool is full and nothing is progressing. Every
slot-holder is, by the invariant, executing its own work — a model call, a tool
body, a decide, or a result-tool dispatch. Each of those terminates on its own
(bounded by the client and tool timeouts already in `RuntimeConfig`), at which
point its drive continues, ends, parks, or winds down; the last three all release
the slot. So a slot always becomes available in finite time and the waiter queue
drains. No cycle among slot-holders can form, because holding a slot and waiting
on another conversation are mutually exclusive states. ∎

**Corollary.** `subagents_max_workers = 1` is legal and correct at any depth. It
serializes subagents completely — useful for deterministic debugging — and never
wedges.

---

## 3. The lifecycle of a slot

Five transitions, and nothing else.

| # | transition | trigger | where |
|---|---|---|---|
| 1 | **acquire at admission** | a queued child is granted a slot and its drive starts | the pump → `AgentRun._start_eager()` (§4.5) |
| 2 | **release on park** | the drive begins waiting on its subtree or on an approval | `_await_subtree` entry, `runner.py:2853` (§4.7) |
| 3 | **re-acquire on wake** | the drive is woken and is about to do work again | `_await_subtree` exit (§4.7) |
| 4 | **release at wind-down** | the drive consumes a `CancelRequested` | `_wind_down_async` entry, `runner.py:2915` (§4.9) |
| 5 | **release at drive end** | the drive returns — finished, gated, suspended, cancelled, or failed | `_end_run`, `runner.py:1344` (§4.8) |

Transitions 2, 4 and 5 feed the pump. Transitions 1 and 3 take from it.

Two properties make this safe:

- **The grant is synchronous.** Starting a child calls `_begin_run`
  synchronously, which registers it in `_runs` before any `await`. There is no
  window in which a slot is "in flight" and unaccounted for.
- **The re-acquire is raced against the cancellation token.** Every collaborator
  await in this runner already is; a drive that could sit blocked on a slot
  without watching for cancellation would make `cancel()` silently stop working.

---

## 4. Mechanics

Pseudocode throughout — illustrative shape, not final text.

### 4.1 Configuration

```python
# models.py — RuntimeConfig, in the subagents block

# How many subagent conversations may be doing work at the same time, across
# the whole session. `Inf` (the default) means no limit — the unconfigured
# behavior is unchanged. The main conversation never counts against it: it is
# not a subagent, and it must always be able to advance.
#
# Sized by FAN-OUT (how many siblings should work at once), never by
# `subagents_max_depth` — a conversation waiting on its own children holds no
# slot, so depth costs sequence, not concurrency. 20-30 is the useful range for
# a real workload; see §8.
subagents_max_workers: int = Inf

@field_validator("subagents_max_workers")
@classmethod
def _inf_or_positive(cls, value: int) -> int:
    """`Inf` (-1) or at least 1. `0` is not "disabled" — it would mean no
    subagent may ever run, which is `subagents_enabled=False` spelled
    incorrectly."""
    if value != Inf and value < 1:
        raise ValueError(f"must be >= 1 or {Inf} ({Inf} = no limit)")
    return value
```

The name mirrors `subagents_max_depth` and `subagents_enabled`: the subject is
what it bounds.

**The TUI exposes it as `--subagents-max-workers N`**, alongside the existing
`--no-subagents`. Like that flag, it writes onto the runtime config of the
session being launched — including a *resumed* one (`cli.py:152-154` already does
this for `subagents_enabled`), so the flag always describes the run you are
starting now. Its default is `Inf`, matching the library default.

### 4.2 The pool and the queue — runner state

Runtime-only, never serialized — the same class of state as `_runs` and `_wakes`:

```python
# runner.py — AgentSessionRunner.__init__

# Conversations currently holding a slot. A strict subset of `_runs` keys,
# minus the main conversation, minus any drive parked on its subtree, and
# minus any drive winding a cancelled turn down. A set rather than a
# derivation over `_runs`, so a slot granted to a child that has not called
# `_begin_run` yet is already reserved.
self._working: set[str] = set()

# Everything that wants a slot, in FIFO order: queued children waiting to
# start, parked drives waiting to resume, and gated children waiting to be
# re-driven after an answered approval. One queue, so ordering is a single,
# testable policy (§4.4).
self._waiters: list[_SlotWaiter] = []
```

```python
@dataclass
class _SlotWaiter:
    conversation_id: str
    grant: Callable[[], None]        # what to do when the slot is handed over
    announce: AgentRun | None = None # publish SubagentStarted through this handle (§11.5)
```

`grant` is `handle._start_eager` for a never-started child, `handle._redrive` for
a gated child being re-driven, and `event.set` for a parked drive. All three are
synchronous.

### 4.3 The pool operations

```python
def _slot_limit(self) -> int:
    return self.session.session_config.runtime_config.subagents_max_workers

def _needs_slot(self, conversation_id: str) -> bool:
    """The main conversation never competes, and an unlimited pool never
    accounts."""
    return (
        self._slot_limit() != Inf
        and self.session.conversations[conversation_id].depth > 0
    )

def _request_slot(self, conversation_id, grant, *, announce=None) -> None:
    """Ask for a slot. Granted synchronously if one is free."""
    waiter = _SlotWaiter(conversation_id, grant, announce)
    if not self._needs_slot(conversation_id):
        self._grant(waiter)
        return
    self._waiters.append(waiter)
    self._pump()

def _grant(self, waiter: _SlotWaiter) -> None:
    """The ONE place a slot is handed over, so accounting and announcement
    cannot drift apart."""
    if self._needs_slot(waiter.conversation_id):
        self._working.add(waiter.conversation_id)
    waiter.grant()
    if waiter.announce is not None:
        self._publish_subagent_event(
            waiter.announce,
            SubagentStarted(conversation_id=waiter.conversation_id),
        )

def _release_slot(self, conversation_id: str) -> None:
    """Give a slot back (a no-op for a conversation that holds none) and hand
    it on."""
    self._working.discard(conversation_id)
    self._pump()

def _pump(self) -> None:
    while self._waiters and (
        self._slot_limit() == Inf or len(self._working) < self._slot_limit()
    ):
        waiter = self._waiters.pop(0)
        if not self._still_wants_slot(waiter):
            continue                      # stale — drop it and take the next
        self._grant(waiter)

def _still_wants_slot(self, waiter: _SlotWaiter) -> bool:
    """A waiter can go stale between enqueue and grant."""
    conversation_id = waiter.conversation_id
    if conversation_id not in self.session.conversations:
        return False
    if self.session.get_conversation_status(conversation_id).status is ConversationStatus.IDLE:
        return False                      # already finished or already flushed
    # A CANCELLED CONVERSATION IS NEVER GRANTED A SLOT. Its parent's drive
    # flushes it (`_flush_cancelled_children`, runner.py:2953), which is the
    # path that already exists for a child with no drive. Starting one purely
    # so it can wind itself down would burn a slot on bookkeeping AND race the
    # parent for the same flush.
    if self.ledger.open_turn_cancel_requested(conversation_id) is not None:
        return False
    return True
```

With `subagents_max_workers = Inf`, `_needs_slot` is False for every
conversation, every `_request_slot` grants inline, `_working` and `_waiters` stay
empty, and the pool is invisible — including in the order in which ids and
`TurnStart` entries are consumed.

### 4.4 Admission order is FIFO

One queue, first-come first-served, across the whole tree. A parked parent
enqueues when it parks, which is *after* the children it is waiting for enqueued,
so it re-acquires once they have run rather than once per child completion.
Results therefore resolve in a batch.

This is a decision, not an accident: FIFO is one policy, it is testable, and its
worst-case behavior is a throughput win. §7.7 records the alternative that was
analyzed and rejected.

### 4.5 Deferred start on the handle

`AgentRun.__init__`'s eager branch becomes a method, so it can be called later:

```python
class AgentRun:
    def __init__(self, ..., eager: bool, defer_start: bool = False, ...):
        ...
        if eager and not defer_start:
            self._start_eager()

    def _start_eager(self) -> None:
        """Begin this handle's drive. Exactly the eager branch of __init__ as
        it stands today (runner.py:309-329): resolve the loop first, take the
        one-run guard, open the bracket durably, then spawn the task."""
        loop = asyncio.get_running_loop()
        self._runner._begin_run(self)
        try:
            self._runner._open_bracket_for_start(self.conversation_id)
        except BaseException:
            self._runner._end_run(self)
            raise
        self._task = loop.create_task(self._consume())
        self._wake.set()      # a consumer blocked on an unstarted handle re-checks
```

`runner.start()` is unchanged — it constructs with `defer_start=False` and starts
synchronously at call time, exactly as documented.

### 4.6 Spawning: adopt every child, admit what fits

`_start_children` now **returns the conversation ids it started**, so the drive
loop can announce them in order (§11.5):

```python
def _start_children(self, conversation_id, spawned) -> list[str]:
    """Give every fresh child a handle on the run that spawned it, started or
    queued according to who owns them and what the pool allows. Returns the
    ids that started, in spawn order."""
    run = self._runs.get(conversation_id)
    if run is None:
        return []
    started: list[str] = []
    for entry in spawned:
        child = AgentRun(
            self,
            conversation_id=entry.conversation_id,
            streaming=run._streaming,
            on_event=None,                        # the parent's callback sees it via the fan-in
            eager=run.autostart_subagents,
            defer_start=True,                     # ← the pool decides when
            autostart_subagents=run.autostart_subagents,
            parent=run,
        )
        run._adopt(child)
        if run.autostart_subagents:
            # announce=None: this batch is announced by the engine, right after
            # `SubagentsSpawned`, so the two arrive in the right order (§11.5).
            self._request_slot(child.conversation_id, child._start_eager)
            if child._task is not None:
                started.append(child.conversation_id)
    return started
```

```python
# _drive_loop, step 2b (runner.py:2311-2326)
spawned = list(self._spawn_children(conversation_id))
if spawned:
    started = self._start_children(conversation_id, spawned)
    yield SubagentsSpawned(
        conversation_id=conversation_id,
        conversation_ids=[entry.conversation_id for entry in spawned],
    )
    for child_id in started:
        yield SubagentStarted(conversation_id=child_id)
```

Two properties preserved from today, both load-bearing:

- **Every handle exists before `SubagentsSpawned` is yielded**
  (`runner.py:2315-2326`). The event announces the whole batch, queued children
  included, so `run.child(cid)` resolves for all of them inside that event's
  branch.
- **Handles are adopted in spawn order**, and `_children` is insertion-ordered,
  so admission order is the model's request order.

### 4.7 Release on park, re-acquire on wake

```python
async def _await_subtree(self, conversation_id, token, wake) -> bool:
    if conversation_id in self._recheck:
        return True
    if not self._can_subtree_advance(conversation_id):
        return False                       # the drive returns; _end_run releases
    if token.cancelled:
        return True

    # PARKED IS NOT WORKING. Releasing here is what lets this conversation's
    # own children be admitted, and it is the whole reason a nested tree
    # cannot deadlock.
    self._release_slot(conversation_id)
    try:
        waiter = asyncio.ensure_future(wake.wait())
        cancelled = asyncio.ensure_future(token.wait_cancelled())
        try:
            await asyncio.wait({waiter, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel_quietly(waiter)
            await _cancel_quietly(cancelled)
    finally:
        # Raced against the token: a drive blocked on a slot must still be
        # cancellable. When the token wins this returns WITHOUT a slot, and the
        # loop top winds the turn down — which needs no slot (§4.9).
        await self._acquire_slot(conversation_id, token)
    return True
```

```python
async def _acquire_slot(self, conversation_id: str, token: CancellationToken) -> bool:
    if not self._needs_slot(conversation_id):
        return True
    granted = asyncio.Event()
    # announce=None: resuming a parked drive is not a start (§11.9).
    self._request_slot(conversation_id, granted.set)
    if granted.is_set():
        return True                        # granted inline
    waiter = asyncio.ensure_future(granted.wait())
    cancelled = asyncio.ensure_future(token.wait_cancelled())
    try:
        await asyncio.wait({waiter, cancelled}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await _cancel_quietly(waiter)
        await _cancel_quietly(cancelled)
    if granted.is_set():
        return True
    self._drop_waiter(conversation_id)     # cancelled while queued
    return False
```

`_release_slot` pumps *before* the park, so a parent's own queued children are
admitted in the same synchronous stretch in which it decides to wait for them.

### 4.8 Release at drive end

```python
def _end_run(self, run: AgentRun) -> None:
    if self._runs.get(run.conversation_id) is not run:
        return
    del self._runs[run.conversation_id]
    # ANNOUNCE BEFORE RELEASING. Releasing pumps, and the pump may start a
    # queued sibling that announces its own `SubagentStarted`. Publishing this
    # conversation's terminal event first is what keeps "S1 paused, so S4
    # started" in that order on the stream.
    self._publish_run_ended(run)
    self._release_slot(run.conversation_id)
```

This single method covers every way a drive can end: a finished turn, a gate, a
suspension, a cancellation, a failure. There is no second release path.
`_publish_run_ended` is specified in §11.5.

### 4.9 Release at wind-down — a cancelled turn never waits for a slot

A conversation consuming a `CancelRequested` releases its slot at the top of
`_wind_down_async` and never re-acquires:

```python
async def _wind_down_async(self, conversation_id, cancel_entry):
    # WIND-DOWN IS BOOKKEEPING, NOT WORK: no model call, no tool body — pending
    # executions are terminalized without dispatch and child links resolve
    # without running the result tool. So the slot is not needed back, and
    # holding one while waiting for this conversation's cancelled children to
    # settle would be a slot-holder waiting on other conversations — the one
    # thing §1.1 forbids.
    self._release_slot(conversation_id)
    for child_id in self._unresolved_child_ids(conversation_id):
        ...                                # unchanged from here down
```

Why this is the shape it is:

- **Correctness.** Without it, the invariant in §2.4 would have an exception,
  because `_wind_down_async` waits for its children's drives to settle
  (`runner.py:2932-2936`) while its own drive is still registered. That
  particular wait provably terminates — `cancel()` cascades first, so every child
  it waits on is already winding down and never asks for a slot — but a proof
  with an exception in it is one the next person to touch cancellation cannot
  rely on.
- **Latency.** A cancelled tool body gets `tool_cancellation_grace_period` to
  settle. Releasing at the top of the wind-down instead of at its end hands that
  capacity to an unrelated queued subagent immediately rather than a grace window
  later.
- **It must never block.** Wind-down is how a cancellation becomes durable. Any
  design in which it waits for a slot could make `cancel()` slow or, worse,
  dependent on other conversations finishing.

The guard in `_still_wants_slot` (§4.3) is the other half: because a slot can now
be granted while a subtree is winding down, the pump must not hand one to a
conversation that is itself cancelled.

### 4.10 The two restart paths

Both must go through the pool, or answering an approval and resuming a session
would each silently exceed the cap. Both announce, because both are starts a
consumer has to see (§11.5).

```python
def _restart_unresolved_children(self, run: AgentRun) -> None:
    """Re-drive the subtree a previous run left parked. Unchanged except that
    handles are adopted unstarted and admitted by the pool."""
    for child_id in self._unresolved_child_ids(run.conversation_id):
        if child_id not in self.session.conversations or self._runs.get(child_id) is not None:
            continue
        if self.session.get_conversation_status(child_id).status is ConversationStatus.IDLE:
            continue
        child = AgentRun(self, conversation_id=child_id, ..., eager=True, defer_start=True, parent=run)
        run._adopt(child)
        self._request_slot(child_id, child._start_eager, announce=child)
```

```python
def _ensure_driven(self, conversation_id: str) -> None:
    """Something changed out of band — make sure a drive looks again."""
    wake = self._wakes.get(conversation_id)
    if wake is not None:
        wake.set()                          # a live drive: signal it, no slot involved
        return
    run = self._runs.get(conversation_id) or self._parked_handle(conversation_id)
    if run is not None and run._framework_owned:
        self._request_slot(conversation_id, run._redrive, announce=run)
```

An answered gate therefore restarts its subagent **when a slot is free**, and
queues behind the currently-working ones otherwise. The user's answer is never
lost — it is recorded on the registry, and the re-drive re-asks `decide()`
whenever it lands.

### 4.11 A queued handle is still a handle

`run.child(cid)` hands back a queued handle, so the consumption paths must
tolerate `_task is None`:

- `_next_buffered` (`runner.py:741`): guard the `self._task.done()` checks and
  wait on `self._wake`, which `_start_eager` sets when the task appears.
- `_join` / `_finalize_eager` (`runner.py:759`/`771`): wait for admission before
  awaiting the task. `await child` on a queued handle therefore blocks until it
  is admitted, which is the honest answer to "join this child".

### 4.12 `autostart_subagents=False` — fail fast

`subagents_max_workers` is a **framework-owned** bound. It works by withholding a
start, and under `autostart_subagents=False` the framework does not own the
starts: the application does, via `run.child(cid)`.

We therefore refuse the combination, loudly, at the earliest point the runner can
see it. That is `run()` — `autostart_subagents` is an argument to it, not a
runner constructor parameter, and the cap lives on the session's `RuntimeConfig`,
so no earlier check is possible:

```python
def run(self, *, streaming=False, on_event=None, autostart_subagents=True) -> AgentRun:
    if not autostart_subagents and self.session.session_config.runtime_config.subagents_max_workers != Inf:
        raise AgentError(
            "subagents_max_workers is framework-owned: it works by withholding a "
            "subagent's start, which is only the framework's to withhold under "
            "autostart_subagents=True. Drive the children yourself and pace them "
            "yourself, or let the framework drive them and cap them."
        )
    ...
```

`start()` implies `autostart_subagents=True` and is never affected.

The raise is synchronous at the call site, before any drive begins. A session
saved with a cap and later resumed by an application that wants to drive children
itself fails on its first `run()` with a message that names both knobs — rather
than silently ignoring one of them.

### 4.13 Changing the cap on a live session

**Allowed, with no special handling.** `RuntimeConfig` is persisted on the
session and read live on every use, and the TUI already rewrites one of these
fields on a *resumed* session (`cli.py:154`, for `subagents_enabled`), so
`--subagents-max-workers` on a resume is an ordinary path.

The behavior falls out of `_pump`'s loop condition with no extra code:

- **Lowering it below the number of subagents already working** does not kill
  anything. The pool simply grants nothing until the working count falls below
  the new limit, and it converges as those subagents finish.
- **Raising it** admits queued children on the next release. Adding a `_pump()`
  call wherever the config is written would make it immediate; we deliberately do
  not, because a release always follows shortly and a config write is not a
  runner event.

The rule to document: *the pool never interrupts work in progress; it only
decides what starts next.*

### 4.14 A start that fails after the first batch

**Keep the explicit error channel.** A grant can only fail where `_start_eager`
can fail, which is `_open_bracket_for_start` → the `TurnStart` append → a
`before_entry_written` middleware raising. Today that exception propagates
synchronously out of `_start_children` and aborts the parent's drive, which is
the correct fail-loud behavior.

A grant issued later — from another child's `finally` — has no such caller. We
preserve fail-loud explicitly: the pump records the exception against the child's
conversation and wakes the parent, whose drive re-raises it at its loop top.

```python
# in _grant
try:
    waiter.grant()
except BaseException as exc:
    self._working.discard(waiter.conversation_id)
    self._admission_errors[waiter.conversation_id] = exc
    self._ensure_driven(<the parent of waiter.conversation_id>)
    return
```

Letting the exception simply escape the sibling's task is not an option: it would
surface on *that* sibling's handle while the parent still parks forever on the
unstarted child, whose seed message is still its last node and which therefore
still derives `BUSY`. The parent has to be told, so the channel is explicit.

### 4.15 The model is not told about the cap

The subagents plugin contributes a prompt part (`spawning_prompt_part`,
`plugin.py:65`) telling the model it can spawn subagents, and the depth gate
withholds both that part and the tool together. The worker cap adds nothing to
either.

The cap is a scheduling detail with no bearing on what the model should do: the
spawn tool succeeds immediately whether or not a slot is free, results simply
arrive later, and nothing in the transcript reveals the difference. A number in a
prompt would also drift from the config the moment either is changed.

Consequence worth noting: **this change touches `core` only.** No contrib
package, no plugin, no prompt part. (The TUI gains a flag and may render the new
events, but nothing there is required.)

---

## 5. Worked examples

### 5.1 Flat fan-out — the default shape

`subagents_max_workers = 2`, `subagents_max_depth = 1`. One assistant message
spawns four subagents `S1…S4`.

| step | what happens | `_working` | `_waiters` | events emitted |
|---|---|---|---|---|
| 1 | `_spawn_children` creates four child conversations, each with a seed `UserMessage` | `{}` | — | — |
| 2 | `_start_children` adopts four handles and requests four slots; two are granted inline | `{S1, S2}` | `S3, S4` | — |
| 3 | the drive announces the batch, then the two that started | `{S1, S2}` | `S3, S4` | `SubagentsSpawned([S1..S4])`, `SubagentStarted(S1)`, `SubagentStarted(S2)` |
| 4 | the parent reaches "blocked on children" and parks. It is the main conversation, so it holds no slot and releases nothing | `{S1, S2}` | `S3, S4` | — |
| 5 | `S2`'s turn closes. `_end_run(S2)` announces, then releases; the pump grants `S3` **synchronously**, so `S3` is in `_runs` before the parent is woken | `{S1, S3}` | `S4` | `SubagentFinished(S2, COMPLETED)`, `SubagentStarted(S3)` |
| 6 | the parent wakes, resolves `S2`'s link by running the result tool | `{S1, S3}` | `S4` | the ordinary tool events |
| 7 | `S1` closes → `S4` admitted; `S3` closes; `S4` closes | `{}` | — | `SubagentFinished(S1)`, `SubagentStarted(S4)`, `SubagentFinished(S3)`, `SubagentFinished(S4)` |
| 8 | every link resolved, the parent calls the model with four results | `{}` | — | — |

`S3` and `S4` derive `BUSY` throughout steps 2–5, so `_can_subtree_advance`
(`runner.py:2882`) counts them and the parent waits rather than concluding its
subtree is dead. `_resolve_children` (`runner.py:2710`) skips them, because it
resolves only `IDLE` children — a queued child is never mistaken for a finished
one and never gets an empty result.

### 5.2 Nesting — the case the rule exists for

`subagents_max_workers = 1`, `subagents_max_depth = 2`. Main spawns `A`; `A`
spawns `A1`.

| step | what happens | `_working` | `_waiters` |
|---|---|---|---|
| 1 | main spawns `A`; a slot is free | `{A}` | — |
| 2 | `A` calls the model and spawns `A1`; the pool is full | `{A}` | `A1` |
| 3 | `A` reaches "blocked on children" → **releases its slot** → the pump grants `A1` inline → `A` parks | `{A1}` | `A` (re-acquiring) |
| 4 | `A1` works, answers, its turn closes → `_end_run` releases → the pump grants `A` | `{A}` | — |
| 5 | `A` wakes with its slot, resolves `A1`'s link, calls the model, answers; its drive ends → releases | `{}` | — |
| 6 | main resolves `A`'s link and answers | `{}` | — |

Step 3 is the whole design. Without the release, `A` would hold the only slot
while doing nothing, `A1` would never be admitted, and the tree would hang
forever with no exception, no timeout and no cancel (§7.1).

Events: `A` emits `SubagentStarted` at step 1 and `SubagentFinished` at step 5.
`A1` emits `SubagentStarted` at step 3 and `SubagentFinished` at step 4. `A`
parking at step 3 emits **nothing** — see §11.9.

### 5.3 An approval gate mid-flight

`subagents_max_workers = 2`, three subagents `S1, S2, S3`.

| step | what happens | `_working` | `_waiters` | events |
|---|---|---|---|---|
| 1 | `S1`, `S2` admitted; `S3` queued | `{S1, S2}` | `S3` | `SubagentsSpawned`, `SubagentStarted(S1)`, `SubagentStarted(S2)` |
| 2 | `S1`'s tool call is deferred by the policy. Nothing in `S1`'s subtree can advance, so its drive **ends** — today's behavior, unchanged | `{S2}` | `S3` | `ApprovalRequired`, `SubagentPaused(S1)` |
| 3 | `_end_run(S1)` released; the pump admits `S3` | `{S2, S3}` | — | `SubagentStarted(S3)` |
| 4 | `run.approvals` surfaces the gate; the application answers on the registry and calls `run.notify(execution)` | `{S2, S3}` | — | — |
| 5 | `_ensure_driven(S1)` finds no live drive and no wake, so it requests a slot for `S1._redrive` | `{S2, S3}` | `S1` | — |
| 6 | `S2` finishes → `S1` is re-driven, re-asks `decide()`, and proceeds | `{S3, S1}` | — | `SubagentFinished(S2)`, `SubagentStarted(S1)` |

A gated subagent releasing its slot is deliberate: it is doing no work, and the
alternative would let one unanswered gate hold capacity hostage. The visible
consequence is that the cap does not bound *how many subagents sit at a gate* —
with a cap of 2 and ten spawns, they can all end up waiting on the human. That is
the correct division of labor: `subagents_max_per_turn` bounds how many subagents
**exist**, `subagents_max_workers` bounds how many **work**.

### 5.4 Cancelling a queued subagent

`run.child(S4).cancel()` while `S4` is queued. `S4` has no turn bracket, so
`cancel()` opens one for it (`runner.py:1124-1134`) and appends the
`CancelRequested`; `S4` derives `CANCELLING`.

`S4` is never granted a slot: `_still_wants_slot` drops any waiter whose
conversation has an unconsumed `CancelRequested` (§4.3). Its parent's drive
reaches `_flush_cancelled_children` (`runner.py:2953`) and winds it down without a
drive — the path that already exists today for a never-driven child — then
resolves its link with the cancellation result. No slot is ever spent on it, and
it emits no lifecycle event, because it never had a drive (§11.9).

### 5.5 Cancelling a working subtree

`subagents_max_workers = 2`. `A` (depth 1) is working and holds a slot; it has two
children `A1`, `A2`, of which `A1` holds the other slot and `A2` is queued. The
application calls `run.child(A).cancel()`.

| step | what happens | `_working` | `_waiters` |
|---|---|---|---|
| 1 | `cancel()` writes `CancelRequested` for `A`, cascades to `A1` and `A2`, and trips `A`'s and `A1`'s tokens | `{A, A1}` | `A2` |
| 2 | `A`'s drive reaches step 0, enters `_wind_down_async` → **releases immediately** | `{A1}` | `A2` |
| 3 | the pump considers `A2` and drops it: it carries an unconsumed cancel | `{A1}` | — |
| 4 | `A1`'s drive sees its own cancel, winds down, ends → releases. Both slots are now free for unrelated subagents while `A` is still unwinding | `{}` | — |
| 5 | `A`'s wind-down finishes waiting on `A1`, flushes `A2`, resolves both links, closes its turn | `{}` | — |

The capacity freed at step 2 is the practical payoff of §4.9: an unrelated queued
subagent starts immediately rather than after `A`'s entire subtree has unwound.

Events: `SubagentFinished(A1, CANCELLED)` at step 4 and `SubagentFinished(A,
CANCELLED)` at step 5. `A2` emits nothing — it never started.

### 5.6 Reload mid-tree

Nothing about the pool is persisted, and nothing needs to be. On the next
`run()`, `_begin_run` → `_restart_unresolved_children` adopts one handle per
unresolved, non-`IDLE` child and requests a slot for each; the cap re-applies from
scratch in link order, and each admitted child emits `SubagentStarted`.

A child that had already been admitted before the crash is, on disk, just a child
conversation with an open turn bracket. A child that was still queued is one with
a single seed message. Both are `BUSY`, both are restartable, and the runner does
not need to tell them apart — admission drives whichever it grants forward from
wherever it is.

---

## 6. Interaction audit

Every existing behavior the change is adjacent to, and what happens to it.

| # | behavior | where | outcome |
|---|---|---|---|
| I1 | a queued child derives `BUSY` | `models.py:1350` | required and free — the parent's own status stays `BUSY` while children are queued, so `runner.busy()` and `post_message`'s IDLE guard behave correctly |
| I2 | `_resolve_children` resolves only `IDLE` children | `runner.py:2736` | a queued child is `BUSY`, so it is skipped — it can never be mistaken for a finished child or given an empty result |
| I3 | `_can_subtree_advance` counts `BUSY` children | `runner.py:2882` | a queued child counts, so the parent parks and waits instead of returning. **This is the mechanism that makes queuing legal at all** |
| I4 | `cancel()` on a child with no bracket opens one | `runner.py:1124-1134` | cancelling a queued child already works; §5.4 |
| I5 | `_flush_cancelled_children` winds down a child with no drive | `runner.py:2953` | already handles the never-driven child, and is now the *only* path that closes a cancelled queued child, since the pool never grants one a slot |
| I6 | `_wind_down_async` awaits live child tasks | `runner.py:2932` | the awaiting drive releases its slot first (§4.9), so no slot-holder ever waits on another conversation. Queued children are correctly not awaited — they are in no `_runs` — and the loop below resolves their links |
| I7 | `cancel()` cascades before anything winds down | `runner.py:1154` | what guarantees the children a wind-down waits on are themselves already cancelled, and therefore never request a slot |
| I8 | approvals are subtree-scoped | `pending_approvals`, `run.approvals` | unchanged — a queued child raises no gates, and an admitted one raises them exactly as before |
| I9 | the spawn depth gate | `plugin.py:46`, `runner.py:1739` | untouched. The cap schedules; it never refuses a spawn |
| I10 | `subagents_max_per_turn` (spawn budget) | separate | orthogonal: that bounds how many children **exist**, this bounds how many **work**. Neither reads the other |
| I11 | the spawn prompt part | `plugin.py:65` | untouched (§4.15). This change is core-only |
| I12 | compaction | `_compaction_step` | main-conversation only; the main conversation never holds a slot and never emits a subagent lifecycle event. `_rebind_run` moves a run's id on a transition — depth 0, so nothing to move in `_working` |
| I13 | step limits (`subagent_*_max_steps`) | `runner.py:3541` | per conversation, unaffected by when the conversation starts |
| I14 | event forwarding / fan-in | `_forward`, `_next_own_or_forwarded` | unchanged in mechanism, and reused: the lifecycle events ride the same inbox (§0.8, §11.5) |
| I15 | forwarding follows ownership | `_forward`, `runner.py:731` | the lifecycle events obey the same rule — they are published only for framework-driven children (§11.7), so under `autostart_subagents=False` a child's id never appears on the parent's stream |
| I16 | suspension cascade (`__aexit__` → `child.cancel_drive()`) | `runner.py:530` | a queued child has no task, so `cancel_drive` is a no-op. A started child's drive ends with its turn open → `SubagentPaused`. Handles are discarded with the parent's; the next `run()` re-adopts through `_restart_unresolved_children` |
| I17 | `run.child(cid)` minting under `autostart_subagents=False` | `runner.py:346` | unreachable in a capped session — §4.12 refuses the combination |
| I18 | `--no-subagents` rewriting a resumed session's config | `cli.py:152-154` | the same mechanism carries `--subagents-max-workers`; mid-session changes are supported (§4.13) |
| I19 | the TUI keys live cells by `event.conversation_id` | `app.py:148` | the lifecycle events are attributed to the subagent (§11.6), so they route to that subagent's cell with no special casing |
| I20 | `Inf` (the default) | everywhere | `_needs_slot` is False for every conversation, every request grants inline, and the id/entry order is identical to today. The lifecycle events still fire (§11.8) |

---

## 7. Alternatives analyzed and rejected

### 7.1 Counting live drives, without the release-on-park rule

The obvious implementation: a slot is held by any subagent with a live drive.
Correct at `subagents_max_depth = 1`, because a depth-1 child can never spawn and
therefore never waits on anything. **It hangs the moment nesting is turned on.**

`subagents_max_workers = 3`, `subagents_max_depth = 2`:

1. Main spawns `A`, `B`, `C` → 3/3, the pool is full.
2. `A` spawns `A1`, `B` spawns `B1`, `C` spawns `C1` → all three queued.
3. `A`, `B`, `C` now wait for their children. They do no work at all, but their
   drives are alive, so they still hold all three slots.
4. `A1`, `B1`, `C1` wait for a slot. `A`, `B`, `C` wait for `A1`, `B1`, `C1`.

`runner.run()` never returns. No exception, no timeout, no cancellation — the turn
is simply never finished. At `subagents_max_workers = 1` this fires on the very
first nesting, always.

Rejected: it makes a documented configuration combination (`subagents_max_workers`
+ `subagents_max_depth > 1`) an unconditional hang.

### 7.2 Park-aware counting, with no waiting

Same rule as §1.1, but instead of a queue that parked drives re-enter, the pool
simply does not *count* parked drives, and a woken parent resumes immediately
without asking. This can never hang either, needs no waiter queue, and is exactly
equivalent to the chosen design at `subagents_max_depth = 1`.

Its cost is transient overshoot: a parent that wakes while the pool is already
full puts the count at N+1 until it parks or finishes. At depth ≥ 2 the excess is
bounded by the number of simultaneously waking parents.

Rejected in favor of the strict queue because the knob should mean what it says:
with an explicit `subagents_max_workers = 3`, three is a bound, not an average.
The queue costs one FIFO and one token-aware await, and it removes an "except
sometimes" from the contract.

### 7.3 A per-parent cap instead of a session-wide one

Give every conversation its own budget of N children. Deadlock-free by
construction (a parent's slot and its children's slots come from different pools),
and the scheduler state lives on the `AgentRun` rather than the runner — the
smallest possible change.

Rejected because it does not bound what the name promises. With N per parent,
total concurrent subagents is N × (number of parents), up to N^depth. An operator
who sets `20` to bound provider concurrency would get 20 at depth 1, 400 at depth
2. "Max workers" has to be a number you can reason about globally.

### 7.4 Holding the slot until the child's link is resolved

Define occupancy as "admitted and the parent has not yet resolved its
`ChildConversation`". Attractive because it is derivable from durable state alone,
and it makes the cap bound "subagents in flight" literally, including those
waiting at a gate.

Rejected for two reasons:

- A slot would then free only after the **parent's** drive runs
  `_resolve_children`, one loop iteration after the child actually finished — the
  cap would throttle on the parent's scheduling rather than on real work.
- It makes an unanswered approval hold capacity indefinitely. With a cap of 2 and
  two gated subagents, the entire tree stops until a human answers, even though
  nothing is running.

### 7.5 Applying the cap under `autostart_subagents=False`

Two shapes were analyzed.

**Count but never block.** App-driven child drives land in `_runs` like any other,
so they would consume the budget while never being made to wait. It costs nothing
to implement, and in a session that mixes both modes it keeps the framework-owned
side honest.

**Count and block.** `async for event in child` waits before its first event until
a slot frees. Real enforcement, and safe *only* if admission is on demand — at the
first pull — rather than in spawn order. Spawn-order FIFO deadlocks an application
that drives child #3 before child #1: #3 waits for a slot only #1 can free, and #1
is never driven because the application is blocked on #3. Two existing tests drive
children one at a time inside the `SubagentsSpawned` branch
(`tests/agent/subagents/test_parallel_and_control.py:369` and `:402`) and would
hang under the wrong ordering.

Both were rejected in favor of refusing the combination outright (§4.12). The
deciding argument is that the cap is unenforceable under `False` in principle —
the framework cannot make an application call `child()` at all, so a "bound" that
a sequential driver silently ignores is a bound in name only. A loud refusal tells
the developer which knob to reach for; a silent partial bound does not.

### 7.6 A minimum `subagents_max_workers` derived from `subagents_max_depth`

Considered: require something like `2 × subagents_max_depth + 1` to keep a deep
tree from thrashing between parents and children.

There is no such formula, because depth consumes no slots. A parked ancestor holds
nothing (§1.1), and a parent never works while one of its children is unresolved
(§2.3), so a chain of depth 10 needs exactly one slot — for whichever conversation
in it is currently working. The minimum is **1**, at any depth.

The thrash the formula was meant to prevent does not exist either. A
release/re-acquire is a set mutation and an `asyncio.Event`; a re-acquiring parent
redoes no work — no model call, no registry call (`decide()` is not invoked when
nothing is undecided), no re-projection. It wakes, resolves the finished child,
and either continues or parks again. The cost is measured in microseconds and is
bounded by the number of child completions.

The only constraint worth encoding is the validator in §4.1: `Inf` or `>= 1`.

### 7.7 Prioritizing a re-acquiring parent over newly queued children

Under FIFO (§4.4) a parent that parks enqueues behind the children it is waiting
for, so it resumes only once they have all run and resolves their links in a
batch. An alternative policy grants a waking parent priority, so each child's
result is resolved as soon as it lands.

Rejected: it buys earlier link resolution at the cost of cycling a parent in and
out of the pool once per child completion, and it introduces a second admission
rule where one suffices. FIFO's worst case — batched resolution — is a throughput
win, not a defect.

### 7.8 Documenting the wind-down exception instead of releasing the slot

The invariant could have been stated with one carved-out exception — "except
`_wind_down_async`, which holds its slot while waiting for its cancelled children
to settle, and that is safe because those children never request a slot".

Rejected. The one-line release in §4.9 makes the exception unnecessary, frees the
capacity sooner (a cancelled tool body can take up to
`tool_cancellation_grace_period` to settle), and leaves a proof with no
qualifications for whoever next touches cancellation.

### 7.9 Telling the model about the cap

The spawn prompt part could state "at most N subagents run at the same time".

Rejected (§4.15): it is a scheduling detail the model cannot act on, the spawn
tool succeeds either way, nothing in the transcript reflects it, and the number
would drift from the config. Leaving it out also keeps the entire change inside
`core`.

### 7.10 Events tied to slot transitions rather than to lifecycle transitions

The natural first sketch of §11 is one event per pool transition:
`SubagentQueued` when a slot is granted, `SubagentUnqueued` when it is released.

Rejected on both feasibility and meaning:

- **Feasibility.** Slots move in `_pump` / `_release_slot` / `_end_run`, which are
  plain synchronous methods, two of which are reached from inside a *finishing
  child's* `finally` block. None of them runs inside the drive generator, so none
  of them can `yield`. Every emission would need the side channel anyway, and the
  emission points would be spread across the pool internals rather than
  concentrated at two lifecycle boundaries.
- **Meaning.** Slot transitions are not what an application wants to render. A
  nested parent releases and re-acquires its slot around every wait, which would
  flip its row between "waiting" and "running" while its subtree is plainly
  working.

The chosen events fire at conversation lifecycle boundaries, which are both fewer
and more meaningful.

### 7.11 One event carrying a state enum

`SubagentStateChanged(conversation_id, state: SubagentState)` instead of three
classes — a single union member and a single `match` arm.

Rejected for consistency and payload honesty: the framework already models a
three-phase lifecycle as three classes (`CompactionScheduled` /
`CompactionStarted` / `CompactionFinished`), and `SubagentFinished` genuinely
carries a field the others do not (`outcome`). A shared class would either make
that field optional-and-usually-null or lose it.

### 7.12 Emitting the lifecycle events only when a cap is set

Tempting, because the events exist to explain the pool: with `Inf` every subagent
starts immediately, so `SubagentStarted` looks redundant.

Rejected. A consumer would then need two rendering paths — one driven by events
and one inferring state from the absence of them — and would have to know the
session's config to pick. `SubagentFinished` is also useful with no cap at all: it
is the only event that says a subagent's turn closed and with what outcome. The
cost of always emitting is two events per subagent in the default configuration.

### 7.13 Attributing the lifecycle events to the parent conversation

`SubagentsSpawned` carries the *parent* in `conversation_id`, with the children in
a separate list. The lifecycle events could follow that and carry the parent plus
a `subagent_conversation_id`.

Rejected. `SubagentsSpawned` is parent-attributed because it is a batch fact about
the parent's turn — there is no single child it could name. "This subagent
started" is a fact about one child, and the framework's rule is that an event
names the conversation it is about (which is why the tool-lifecycle events carry
the conversation their execution was born in). Attributing to the subagent also
makes the consumer a one-liner, `store[event.conversation_id] = RUNNING`, and
routes correctly in a UI that already keys by `event.conversation_id`.

---

## 8. Configuration guidance

| | |
|---|---|
| **config field** | `RuntimeConfig.subagents_max_workers` |
| **TUI flag** | `--subagents-max-workers N`, alongside `--no-subagents`; writes onto the (possibly resumed) session's runtime config |
| **default** | `Inf` — no limit, today's behavior exactly |
| **minimum** | `1`. Legal at any depth; serializes subagents completely |
| **recommended** | **20–30.** That is the range a real coding-agent workload wants; lower it when your tools are CPU- or memory-heavy, or when your provider's concurrency limits are tighter than that |
| **how to size it** | by the fan-out you want — how many subagents should work at the same time. Subagent work is I/O-bound (model calls and tool waits), so the binding constraint is usually provider concurrency and tool weight, not core count. **Never** derive it from `subagents_max_depth` |
| **scope** | per session, per runner **process**. `_runs` and `_working` are runtime state, so two processes over the same session file get a pool each. That is inherent, not a policy |
| **changing it mid-session** | supported; the pool never interrupts work in progress, it only decides what starts next (§4.13) |
| **incompatible with** | `run(autostart_subagents=False)` — raises (§4.12) |

---

## 9. Tests

`tests/agent/subagents/test_workers.py` (new), plus additions where noted.
Existing tests that assert on event streams are updated to expect the new
lifecycle events.

| # | test | asserts |
|---|---|---|
| T1 | `Inf` schedules nothing | four spawns with the default config produce the identical session as today, and all four start immediately |
| T2 | flat cap | cap 2, four spawns: at the `SubagentsSpawned` yield exactly two conversations have a `TurnStart`; all four eventually resolve; the parent answers last |
| T3 | queued child is `BUSY` and skipped | cap 1, two spawns: the queued child's status is `BUSY`, it holds exactly one node, and no `ChildConversation` resolves against it |
| T4 | admission order is FIFO | cap 1, three spawns: the children's brackets open in model-request order |
| T5 | **nesting does not deadlock** | cap 1, `max_depth=2`, main → `A` → `A1`: the run terminates, every conversation ends `IDLE`, and `A1`'s answer reaches `A`'s projection. The regression test for §7.1 |
| T6 | nesting under a wider cap | cap 2, `max_depth=2`, two children each spawning one grandchild: terminates, all six conversations `IDLE` |
| T7 | a gate releases its slot | cap 1, two spawns where the first gates: the second is admitted while the first is parked |
| T8 | an answered gate re-acquires | after T7, `notify()` + a freed slot re-drives the gated child, which finishes |
| T9 | wind-down releases at entry | cap 1, `max_depth=2`: cancelling a parent subagent that is waiting on a child frees the slot before the subtree finishes unwinding — an unrelated queued subagent starts |
| T10 | a cancelled conversation is never granted a slot | cap 1: a queued child that is cancelled never opens a `TurnStart`; its link resolves with the cancellation result through `_flush_cancelled_children` |
| T11 | cancel while waiting for a slot | a parked parent whose re-acquire is pending still consumes its `CancelRequested` and winds down |
| T12 | reload mid-tree | a session saved with one admitted and one queued child resumes under the cap and finishes |
| T13 | lowering the cap mid-session | a resumed session with a lower cap does not interrupt the running subagents and admits nothing until the count falls below the new limit |
| T14 | `autostart_subagents=False` refusal | `run(autostart_subagents=False)` raises `AgentError` when the cap is set, and does not when it is `Inf`; `start()` is never affected |
| T15 | validator | `subagents_max_workers=0` raises at config construction; `-1` (`Inf`) and `1` are accepted |
| T16 | queued handle is consumable | `await run.child(queued)` returns once the child is admitted and finishes |
| T17 | no double-start | a child granted a slot while a restart path is also running is driven exactly once (`_still_wants_slot` + the `_begin_run` guard) |
| T18 | admission failure surfaces | a `before_entry_written` middleware that raises on the third child's `TurnStart` makes the **parent's** run raise, rather than hang (§4.14) |
| T19 | the model is not told | the system prompt and tool list are byte-identical with and without a cap set (§4.15) |
| T20 | the event sequence, capped | cap 2, four spawns: the stream carries `SubagentsSpawned([4])`, then `Started` for exactly the first two, then a `Started` after each `Finished`, and one `Finished` per child |
| T21 | the event sequence, uncapped | `Inf`, two spawns: every child gets exactly one `Started` and one `Finished` (§11.8) |
| T22 | gate produces `Paused` then a restart produces `Started` | a gated subagent emits `SubagentPaused`, and answering + re-driving emits a second `SubagentStarted` for the same conversation |
| T23 | ordering: paused before started | when a gate frees a slot, `SubagentPaused(S1)` precedes `SubagentStarted(S4)` on the stream (§4.8) |
| T24 | `Finished` carries the outcome | a subagent that is cancelled emits `SubagentFinished(outcome=CANCELLED)`, one that errors emits `ERRORED`, one that answers emits `COMPLETED` — and no `Paused` |
| T25 | attribution | every lifecycle event's `conversation_id` is the subagent's, never the parent's (§11.6) |
| T26 | ownership | under `autostart_subagents=False` no lifecycle event appears on any stream (§11.7) |
| T27 | a never-started child is silent | a queued child that is cancelled before admission emits neither `Started` nor `Paused` nor `Finished` |

---

## 10. Work

| | |
|---|---|
| `RuntimeConfig.subagents_max_workers` + validator | `models.py` |
| `_working`, `_waiters`, `_request_slot` / `_grant` / `_release_slot` / `_acquire_slot` / `_pump` / `_still_wants_slot` / `_drop_waiter` | `runner.py`, `AgentSessionRunner` |
| `AgentRun._start_eager()` extracted from `__init__`, `defer_start=` | `runner.py`, `AgentRun` |
| `_start_children` adopts unstarted, requests slots, returns the started ids | `runner.py:2677` |
| `_restart_unresolved_children` and `_ensure_driven` request slots with `announce=` | `runner.py:1294`, `1350` |
| release/re-acquire in `_await_subtree` | `runner.py:2853` |
| announce + release in `_end_run` | `runner.py:1344` |
| release at the top of `_wind_down_async` | `runner.py:2915` |
| `_task is None` tolerance in `_next_buffered` / `_join` / `_finalize_eager` | `runner.py:741`–`785` |
| the `run()` refusal for `autostart_subagents=False` | `runner.py:1200` |
| admission-failure propagation | `runner.py`, `_grant` + `_drive_loop` top |
| `SubagentStarted` / `SubagentPaused` / `SubagentFinished` + the union | `events.py` |
| `_publish_subagent_event` / `_publish_run_ended`, and the batch yields in `_drive_loop` | `runner.py` |
| `--subagents-max-workers` | `luca/agent/contrib/tui/cli.py` |
| docs: a "how many run at once" section in `docs/agent/13-subagents.md`, the new events in `docs/agent/09-events.md`, the `RuntimeConfig` tables in `docs/agent/04-runner.md` and `AGENTS.agent.md`, the TUI flag | |

**Estimate: one day and a half, tests included.**

---

## 11. Observability — the subagent lifecycle events

### 11.1 What this is for

With a worker pool, "the agent spawned five subagents" no longer means "five
subagents are running". An application — a TUI pane per subagent, a web UI, a log
— has to be able to show which are running, which are waiting, and which are
done, without inspecting the session or knowing anything about slots.

We add **three events**. Together with the existing `SubagentsSpawned` they let a
consumer maintain a table of subagent states by reduction, in the shape of a
Redux-style store: one row per subagent, one assignment per event, no imperative
bookkeeping and no polling.

### 11.2 The state machine

Four states, and every transition is announced:

| state | meaning | entered by |
|---|---|---|
| `waiting` | the subagent exists but is not being driven — either it has never started (no slot yet) or its drive ended with its turn still open | `SubagentsSpawned`, `SubagentPaused` |
| `running` | its drive is live | `SubagentStarted` |
| `finished` | its turn has closed; it will never run again | `SubagentFinished` |

There is deliberately **no separate "queued" event**: a subagent announced by
`SubagentsSpawned` that has not yet had a `SubagentStarted` *is* queued. One
fewer event, and the consumer's initial state is set by the announcement it
already handles.

### 11.3 A worked stream

`subagents_max_workers = 3`, one assistant message spawns five subagents. `S1`
and `S2` each defer on an approval shortly after starting; the application
answers both.

```
SubagentsSpawned([S1, S2, S3, S4, S5])
SubagentStarted(S1)
SubagentStarted(S2)
SubagentStarted(S3)
SubagentPaused(S1)                  # S1 deferred on an approval; its drive ended
SubagentStarted(S4)                 # the freed slot goes to the next waiter
SubagentPaused(S2)
SubagentStarted(S5)
SubagentFinished(S4, COMPLETED)
SubagentStarted(S1)                 # answered earlier; re-driven now that a slot is free
SubagentFinished(S5, COMPLETED)
SubagentStarted(S2)
SubagentFinished(S3, COMPLETED)
SubagentFinished(S1, COMPLETED)
SubagentFinished(S2, COMPLETED)
```

The consumer's table after each group:

```
after SubagentsSpawned      S1 waiting   S2 waiting   S3 waiting   S4 waiting   S5 waiting
after the three Started     S1 running   S2 running   S3 running   S4 waiting   S5 waiting
after Paused(S1)/Start(S4)  S1 waiting   S2 running   S3 running   S4 running   S5 waiting
after Paused(S2)/Start(S5)  S1 waiting   S2 waiting   S3 running   S4 running   S5 running
after S4, S5 finish         S1 running   S2 running   S3 running   S4 finished  S5 finished
after S3 finishes           S1 running   S2 running   S3 finished  S4 finished  S5 finished
after S1, S2 finish         S1 finished  S2 finished  S3 finished  S4 finished  S5 finished
```

### 11.4 The events

Added to `events.py` and to the `AgentEvent` union. `TurnOutcome` is already
imported there (`CompactionFinished` uses it).

```python
class SubagentStarted(AgentEventBase):
    """A subagent's drive has begun: its turn bracket is open and it is now one
    of the conversations doing work.

    `conversation_id` (from `AgentEventBase`) is the SUBAGENT — the conversation
    that started. That differs from `SubagentsSpawned`, which names the PARENT,
    because that one is a batch fact about the parent's turn while this is a
    fact about one child.

    Fires on a subagent's FIRST start and on every RESTART — after an answered
    approval, or on a later `run()` that re-drives a parked child — so a
    consumer that tracks state sees `running` again each time."""

    type: Literal["subagent_started"] = "subagent_started"


class SubagentPaused(AgentEventBase):
    """A subagent's drive has ended with its turn still OPEN: it will run again
    later. Two causes today — it deferred on an approval, or the whole tree was
    suspended when the parent left its `async with` block.

    `conversation_id` is the subagent. There is no durable entry behind this
    event; a paused subagent's conversation looks like any other with an open
    bracket and nothing runnable in it."""

    type: Literal["subagent_paused"] = "subagent_paused"


class SubagentFinished(AgentEventBase):
    """A subagent's turn has CLOSED, whatever the outcome — answered, failed,
    timed out, step-limited or cancelled. A finished subagent never runs again;
    its parent resolves its `ChildConversation` link next.

    `conversation_id` is the subagent. `outcome` comes from the closing
    `TurnFinish`, so this event follows the durable record rather than leading
    it, exactly like every other event in this framework."""

    type: Literal["subagent_finished"] = "subagent_finished"
    outcome: TurnOutcome
```

### 11.5 Where they are emitted

Three call sites, and two delivery channels (§0.8).

**`SubagentStarted`, for the batch that starts immediately** — yielded by the
engine, from `_drive_loop` step 2b, right after `SubagentsSpawned` (the code in
§4.6). It must be the engine rather than the inbox here, for a concrete reason:
the inbox is drained *before* the engine on every pull (`_next_own_or_forwarded`,
`runner.py:617`), so an inbox-published `Started` would overtake the
engine-yielded `SubagentsSpawned` and reach the consumer first — a `Started` for
a subagent it has not been told exists.

`SubagentsSpawned` itself must stay an engine `yield` for a second reason:
under `autostart_subagents=False` that yield is the point at which control
returns to the application so it can drive the children. Publishing it to the
inbox instead would leave the parent's generator running on to its next park with
nobody waiting on the queue, and the run would hang.

**`SubagentStarted`, for every later admission** — published from `_grant`
(§4.3) through the inbox, using the `announce` field on the waiter.
`_restart_unresolved_children` and `_ensure_driven` set it; `_start_children`
and `_acquire_slot` do not (the first is announced by the engine, the second is a
resume rather than a start).

**`SubagentPaused` / `SubagentFinished`** — published from `_end_run` (§4.8),
which is the one place every drive ends. One status read separates the two:

```python
def _publish_run_ended(self, run: AgentRun) -> None:
    """Announce a framework-driven subagent's drive ending, as either a pause
    (its turn is still open, so it will run again) or a finish (its turn
    closed)."""
    if run._task is None:
        return          # never actually drove — the _start_eager failure path
    if run._parent is None or not run._framework_owned:
        return          # not a framework-driven subagent (§11.7)
    status = self.session.get_conversation_status(run.conversation_id).status
    if status is ConversationStatus.IDLE:
        # `_close_turn` (runner.py:3597) records the outcome of the bracket it
        # just closed, and `_begin_run` resets it per drive — so this is the
        # outcome of THIS drive's close.
        outcome = self._closed_outcomes.get(run.conversation_id) or TurnOutcome.COMPLETED
        event = SubagentFinished(conversation_id=run.conversation_id, outcome=outcome)
    else:
        event = SubagentPaused(conversation_id=run.conversation_id)
    self._publish_subagent_event(run, event)
```

```python
def _publish_subagent_event(self, run: AgentRun, event: AgentEvent) -> None:
    """Put a subagent lifecycle event on the stream that OWNS this subagent.

    It goes on the PARENT run's inbox — the same door a child's own events
    already use (`AgentRun._forward`, runner.py:731). `_consume` re-forwards
    everything it pulls, so the event bubbles to the root handle's stream and
    reaches `on_event` exactly like any other event.

    `asyncio.Queue.put_nowait` is synchronous and never blocks, which is what
    lets this be called from `_end_run` and from `_grant` — neither of which
    runs inside the drive generator, and neither of which can `yield`."""
    parent = run._parent
    if parent is None:
        return
    parent._inbox.put_nowait(event)
```

### 11.6 Attribution: the subagent, not the parent

`conversation_id` is the subagent's own id, and there is no second id field. The
framework's rule is that an event names the conversation it is *about* — which is
why the tool-lifecycle events carry the conversation their execution was born in.
`SubagentsSpawned` names the parent only because it is a batch fact with no single
child to name.

Two payoffs: the consumer is one line
(`store[event.conversation_id] = RUNNING`), and a UI that already routes by
`event.conversation_id` — as the TUI does (`app.py:148`) — delivers the event to
the right pane with no special casing. §7.13 records the rejected alternative.

### 11.7 Emitted for framework-driven subagents only

The same ownership rule that already governs event forwarding (`_forward`,
`runner.py:731`): a subagent's events reach the parent's stream only when the
**framework** drives it. Under `autostart_subagents=False` the application starts,
drives and finishes each child itself, so it needs no event to learn what it just
did — and publishing one would put a child's `conversation_id` on the parent's
stream in the one mode where that is explicitly not supposed to happen.

`_publish_run_ended` and `_publish_subagent_event` therefore both check
`run._framework_owned`, and `_start_children` requests slots only when
`run.autostart_subagents` is true.

### 11.8 They fire whether or not a cap is set

With `subagents_max_workers = Inf` every subagent is granted a slot inline, so
every one emits `SubagentStarted` immediately after `SubagentsSpawned` and
`SubagentFinished` when its turn closes. Nothing about the event surface depends
on the configuration.

That is deliberate: a consumer that had to infer state from the *absence* of
events in the uncapped case would need two rendering paths and would have to read
the session's config to choose between them. `SubagentFinished` also carries
information available nowhere else in the stream — the child's `TurnOutcome`.

Cost: two extra events per subagent in the default configuration. §7.12 records
the rejected alternative.

### 11.9 What is deliberately silent

| situation | event | why |
|---|---|---|
| a subagent is spawned but not admitted | none — its state comes from `SubagentsSpawned` | "announced but never `Started`" already means queued; a `SubagentQueued` event would carry no information |
| a nested parent parks to wait on its own children, releasing its slot | none | only reachable at `subagents_max_depth ≥ 2`. Its subtree is plainly working, so flipping its row to `waiting` once per child completion would be misleading noise |
| that parent re-acquires its slot and resumes | none | the same transition in reverse; it never left `running` |
| a queued subagent is cancelled before it ever starts | none | it had no drive, so there is nothing to pause or finish. Its `ChildConversation` link still resolves with the cancellation result |
| a subagent is cancelled while running | `SubagentFinished(outcome=CANCELLED)` | its turn closes, so it is finished. Reporting `Paused` would leave the row looking resumable when it is not |
| the main conversation starts, parks or finishes | none | it is not a subagent. Its lifecycle is the run's own, reported by `RunResult` |

### 11.10 State after a reload, and the one caveat

These are the first events in this framework whose content is not fully
recoverable from the session. `SubagentStarted` and `SubagentFinished` happen to
track durable entries (the child's `TurnStart` and `TurnFinish`), but
`SubagentPaused` has none — a paused subagent and a subagent that is merely
between drives are the same thing on disk.

So a consumer that reloads a session mid-tree **rebuilds its table from the
session, not from replayed events**:

```python
for cid, conversation in session.conversations.items():
    if conversation.depth == 0:
        continue
    status = session.get_conversation_status(cid).status
    store[cid] = FINISHED if status is ConversationStatus.IDLE else WAITING
```

Every subagent that is still unfinished starts as `waiting`, and the next
`run()` emits a `SubagentStarted` for each one the pool admits (§4.10), so the
table converges within one drive.

### 11.11 A reference consumer

```python
from luca.agent.core.events import (
    SubagentsSpawned, SubagentStarted, SubagentPaused, SubagentFinished,
)

states: dict[str, str] = {}

async with runner.run() as run:
    async for event in run:
        match event:
            case SubagentsSpawned(conversation_ids=ids):
                for cid in ids:
                    states[cid] = "waiting"
            case SubagentStarted(conversation_id=cid):
                states[cid] = "running"
            case SubagentPaused(conversation_id=cid):
                states[cid] = "waiting"
            case SubagentFinished(conversation_id=cid, outcome=outcome):
                states[cid] = f"finished ({outcome.value})"
        render(states)
```
