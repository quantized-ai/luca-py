# Implementation plan — 0008 post message to a BLOCKED conversation

Read `prd.md` first. This plan assumes its five findings and four decisions.

Five code changes, in dependency order. Each is independently testable; land
them in this order so the test suite stays green between steps.

---

## Step 1 — the projector carve-out

**File:** `luca/agent/core/projection.py`

The whole model-facing change, and the smallest piece. Purely additive: no
existing projection changes shape.

Add the import:

```python
from .models import (
    NONTERMINAL_STATUSES,
    AnyEntry,
    ApprovalStatus,      # NEW
    AssistantMessage,
    ...
)
```

Add a ClassVar next to `STATUS_ONLY_OUTPUTS`:

```python
    # Derived tool output for a GATED execution — PENDING with an approval the
    # policy explicitly deferred. Deliberately NOT a row in
    # STATUS_ONLY_OUTPUTS: that table is keyed by TERMINAL status, and this is
    # the single nonterminal state that projects at all. `is_error` stays False
    # where every other derived output sets it True — the call has not failed,
    # it has not run, and an error result is exactly what makes a model retry.
    AWAITING_APPROVAL_OUTPUT: ClassVar[str] = (
        "[tool execution is awaiting approval — it has not run, and this is not its result]"
    )
```

In `project_tool_execution`, before the `NONTERMINAL_STATUSES` guard:

```python
        status = entry.status
        if status == ExecutionStatus.PENDING and entry.approval_status == ApprovalStatus.PENDING:
            # THE ONE PROJECTABLE NONTERMINAL STATE. A gated execution is not a
            # runtime bug the way RECEIVED / RUNNING / undecided-PENDING are: it
            # is a durable resting state only the application can move, and the
            # placeholder is what lets a message posted while the gate is open
            # reach the model at all (0008). Replaced by the real result at this
            # same path position once the approval is answered — the one
            # projected tool message that is not final.
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[TextBlock(text=self.AWAITING_APPROVAL_OUTPUT)],
                is_error=False,
            )
        if status in NONTERMINAL_STATUSES:
            raise ProjectionError(...)   # unchanged
```

Update the module docstring's tool-execution status list (currently
"`PENDING` / `RUNNING` are not projectable as tool outputs") to name the
carve-out and say why it is exactly that state.

**Do not** touch `NONTERMINAL_STATUSES` itself. A gated execution *is*
nonterminal; the projector is the only place that treats one specially, and the
frozenset stays the single source of truth for pruning, the wind-down and
context accounting.

---

## Step 2 — the post-only predicate

**File:** `luca/agent/core/models.py`

`open_turn_unseen_material` already locates the open turn's last
`AssistantMessage`. Extract that scan so both predicates share it rather than
duplicating a loop:

```python
def _last_assistant_index(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
    start: int,
) -> int | None:
    for i in range(len(nodes) - 1, start - 1, -1):
        if isinstance(entries.get(nodes[i]), AssistantMessage):
            return i
    return None
```

Then add, immediately after `open_turn_unseen_material` (they belong together
and the docstring must contrast them):

```python
def open_turn_unseen_post(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Does the open turn hold a `UserMessage` recorded AFTER its last
    `AssistantMessage` — a post the model has not been shown?

    The GATE's wake term, and deliberately narrower than
    `open_turn_unseen_material`: a terminal tool execution does not count. A
    gated round's ALLOWED siblings complete in the same round the gate was
    raised, so counting them would fire a model call — with the gate's
    placeholder on the wire — on every approval in the system, before anyone
    posted anything. Only a human's message opens this hatch.

    The same blind spot as its sibling: a message posted while an LLM call is
    in flight lands BEFORE the recorded assistant entry, where this cannot see
    it. The live drive covers that window with its `seen` fingerprint."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return False
    last_assistant = _last_assistant_index(nodes, entries, index)
    if last_assistant is None:
        return False
    return any(isinstance(entries.get(node_id), UserMessage) for node_id in nodes[last_assistant + 1 :])
```

Export it from `luca/agent/core/__init__.py` alongside `open_turn_unseen_material`.

---

## Step 3 — the status derivation

**File:** `luca/agent/core/models.py`, `AgentSession._derive_status`

Two sites, because a gate is reachable with and without children.

Children branch — today's rank 3:

```python
                # 3. A gated execution with nothing above it → BLOCKED, UNLESS
                #    an unseen post can reach the model past it (0008): with the
                #    gate projecting a placeholder the next drive CAN do
                #    something, so the material term below is allowed to speak.
                #    Without a post the old reasoning stands — the drive would
                #    only re-park, and BUSY would hot-loop a poll-for-BLOCKED
                #    consumer.
                if open_turn_has_awaiting_executions(nodes, entries) and not open_turn_unseen_post(nodes, entries):
                    return ConversationStatus.BLOCKED
```

Rank 4 is untouched: a `UserMessage` is already material, so it returns `BUSY`.

No-children branch:

```python
            if open_turn_is_runnable(nodes, entries):
                return ConversationStatus.BUSY
            # Not runnable means only gated executions remain. An unseen post
            # still reaches the model past them (0008).
            if open_turn_unseen_post(nodes, entries):
                return ConversationStatus.BUSY
            return ConversationStatus.BLOCKED
```

Leave `open_turn_is_runnable` alone. It answers "can a DRIVE advance this
path", which is still no — the new term is about what the model can be shown,
which is a different question and belongs where the precedence is readable.

Update the `ConversationStatus` docstring's derivation table and the
`get_conversation_status` precedence list.

---

## Step 4 — the drive loop

**File:** `luca/agent/core/runner.py`, `_drive_loop`

Three edits. This is where the two traps from the PRD's findings 1 and 2 live.

### 4a. Step 3 — fall through on an unseen post

Restructure the gate branch and the progress-`continue` that follows it. Note
the `else`: the fall-through **must not** reach `if undecided or ready …:
continue`, because `undecided` holds the gated execution on every pass
(`ledger.open_turn_undecided_executions` counts `approval_status=PENDING` as
undecided so a re-drive re-asks) and continuing would re-ask `decide()` in a
tight loop forever.

```python
            awaiting = self.ledger.open_turn_awaiting_executions(conversation_id)
            if awaiting:
                cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
                if cancel_entry is not None:
                    for event in await self._wind_down_async(conversation_id, cancel_entry):
                        yield event
                    return
                if not announced_gate:
                    announced_gate = True
                    yield ApprovalRequired(
                        conversation_id=conversation_id,
                        executions=[ex.model_copy(deep=True) for ex in awaiting],
                    )
                # 3b) A POST REACHES THE MODEL PAST THE GATE (0008). The gated
                # execution projects as a placeholder, so the path is
                # well-formed and the user gets an answer while the approval
                # prompt is still up. ONLY a post does this: an allowed sibling
                # that completed in this same round is not new material to the
                # model, and counting it would fire a round on every approval.
                if not self._has_unseen_post(conversation_id, last_seen):
                    if not await self._await_subtree(conversation_id, token, wake):
                        return
                    continue
                # fall through to the model call — and deliberately PAST the
                # progress-continue below: `undecided` holds this gate on every
                # pass, so continuing here would re-ask decide() forever.
            else:
                announced_gate = False
                if undecided or ready or spawned or resolved:
                    continue  # re-run the cancel check before calling the model
```

The cancel re-check the old `continue` bought is preserved on this path: the
gate branch does its own cancel check above, after steps 1–2c, and a cancel
landing between there and the model call trips the token, which
`_race_cancellation` catches.

### 4b. The close site — a gate keeps the turn open

Finding 1. Add the third term:

```python
            elif (
                self._has_unseen_user_message(conversation_id, seen)
                or _unresolved_children(self.session, conversation_id)
                # A LIVE GATE KEEPS THE TURN OPEN (0008). Reachable only since
                # a post can now drive a round past the gate; closing here
                # would strand the approval outside any open turn, where
                # `pending_approvals()` cannot see it and `notify()` has
                # nothing to re-decide, and would freeze the placeholder into
                # the history for a call that never ran. The next pass parks at
                # step 3 instead.
                or self.ledger.has_awaiting_approval(conversation_id)
            ):
```

`has_awaiting_approval` already exists on the ledger. A gate is the only
nonterminal execution reachable at this point — every other kind is settled by
steps 1a/1/2 and their `continue`.

### 4c. Failure closes settle the gate

Finding 4. Extract the terminalization loop out of `_wind_down` so three call
sites share it:

```python
    def _settle_undispatched(self, conversation_id: str) -> list[AgentEvent]:
        """Terminalize every undispatched execution in the open turn as
        CANCELLED — resultless, errorless, approval state untouched — and
        return their `ToolExecuted` events.

        The close-side half of "no close leaves a nonterminal execution
        behind". The cancel wind-down has always done this; the failure closes
        need it too since 0008, because a post can now drive rounds while a
        gate is open, which puts `hard_max_steps` and an LLM failure within
        reach of a live gate. Closing over one strands the approval outside any
        open turn and freezes its placeholder into the projected history."""
```

`_wind_down` becomes `events = self._settle_undispatched(cid)` followed by its
existing `_close_turn`. Then:

- **hard-max close** — before `self._close_turn(conversation_id, TurnOutcome.ERRORED, error=error)`:
  `for event in self._settle_undispatched(conversation_id): yield event`
- **LLM-failure close** — in the `except` block, before
  `self._close_turn(conversation_id, outcome, error=str(exc))`, same yield.
  Yielding then raising from an async generator is fine: the consumer pulls the
  events, and the next pull raises.

The subagent-parent failure close and the cancel paths already route through
`_wind_down` / `_settle_children` and need nothing.

### 4d. The runner-local predicate

Beside `_has_unseen_user_message`:

```python
    def _has_unseen_post(self, conversation_id: str, last_seen: int | None) -> bool:
        """Is there a user post the model has not been shown? Both halves, the
        same pair step 3c uses for the subagent park: the DURABLE predicate
        (`open_turn_unseen_post`) plus this drive's `last_seen` fingerprint,
        which covers the one state the durable half cannot see — a post that
        landed under this drive's own in-flight LLM call, and so sits BEFORE
        the assistant entry that answered the previous round."""
        conversation = self.session.conversations[conversation_id]
        if open_turn_unseen_post(conversation.nodes, self.session.entries):
            return True
        return last_seen is not None and self._has_unseen_user_message(conversation_id, last_seen)
```

### 4e. Docstrings

`post_message`'s ACCEPTED list currently says "Posting into a BLOCKED turn does
not unblock anything: the message waits with the turn." Rewrite it: the post
reaches the model on the next drive with the gate projected as a placeholder,
the gate itself is untouched, and the conversation returns to `BLOCKED` after
the round.

---

## Step 5 — the TUI opts out

**File:** `luca/agent/contrib/tui/app.py`

The shipped TUI must never drive a fall-through round. It is already close:
the frame has one dock slot and `show_composer` / `show_approval` each clear it
before mounting, so an approval prompt physically replaces the composer, and
every exit from `_resolve_approvals` restores it.

What it lacks is a guarantee. The swap is driven by the drive WORKER — the run
is consumed to completion, and only then does `_drive` check `runner.blocked()`
and call `_resolve_approvals()`. The composer is mounted and focused from the
moment the registry defers until that happens, and again inside
`_resolve_approvals` after `_restore_composer()` when a gate re-arms. Today a
keystroke there is inert; after step 4 it drives a model round.

In `on_prompt_input_submitted`, before `self.runner.post_message(parts)`:

```python
        # THE TUI OPTS OUT OF POSTING PAST A GATE (0008). The framework will
        # carry this message to the model with the gated call projected as a
        # placeholder; in this UI that is never what the user meant — the
        # answer they want is the approval prompt two lines below. The composer
        # is normally not even mounted here (the prompt replaces it), but the
        # swap is driven by the drive worker, so there are windows where it is:
        # before the parked run has returned, and after `_restore_composer()`
        # when a partially-answered gate re-arms.
        #
        # `blocked()` and not `pending_approvals()`: a SUBAGENT gate with
        # siblings still working leaves this conversation BUSY, and steering
        # posts into a live orchestration stay supported.
        if self.runner.blocked():
            await self._notice("answer the approval prompt first", error=True)
            return
```

Placement matters: before the post, so this PRD's own `BLOCKED` → `BUSY`
transition cannot bypass it.

Update the module docstring's "The composer stays enabled while the agent
works" line — it now has one exception.

**Do not** drive the composer's lifecycle off the `ApprovalRequired` event.
That event is non-terminal on a parent's stream by design (a subagent gates
while its siblings work), so removing the composer on it would blank the input
during ordinary subagent work.

`post_message` is the TUI's only door: slash commands are already idle-only,
and nothing else in the package calls it.

---

## Step 6 — docs

- **`docs/agent/10-projection.md`** — the fail-loud list ("a nonterminal
  (`RECEIVED`, `PENDING` or `RUNNING`) tool execution") gains the carve-out;
  the customization section gains `AWAITING_APPROVAL_OUTPUT`; add the
  "a projected tool message is no longer always final" property and the
  trailing-assistant hazard (finding 5) with the note that an application
  overrides `project()` to normalize it.
- **`docs/agent/05-permissions.md`** — the claim that the model is never called
  again until every tool call has a terminal execution now has one exception.
- **`docs/agent/04-runner.md`** — status derivation and the `post_message`
  matrix.
- **`docs/agent/contrib/tui/`** — note that the composer refuses a submit while
  the main conversation is `BLOCKED`, and that this is the TUI opting out of a
  framework capability rather than a framework limit.
- **`AGENTS.agent.md`** — the projection fail-loud paragraph, the
  `ConversationStatus` table, both derivation bullets, the `post_message`
  paragraph, the permissions paragraph ("the model is never called again
  until…"), and the `_drive_loop` step list (a new 3b beside 3c).

---

## Tests

Project style: assert the full object — a whole `ToolMessage`, a whole
projected message list, the whole `nodes` path. Declarative: precondition → one
action → postcondition.

### `tests/agent/test_projection.py`

- A gated execution (`PENDING` + `approval_status=PENDING`) projects the full
  expected `ToolMessage` — placeholder text, `is_error=False`, correct
  `tool_call_id`.
- `PENDING` with `approval_status=None` still raises `ProjectionError`.
- `PENDING` with `approval_status=ALLOWED` still raises.
- `RECEIVED` and `RUNNING` still raise.
- A subclass overriding `AWAITING_APPROVAL_OUTPUT` changes the text — the
  override point is real.
- The headline pair from the PRD: the same conversation projects the
  placeholder while gated and the real result once the execution is COMPLETED,
  asserted as two complete message lists.

### `tests/agent/test_models.py`

- Gated, no post → `BLOCKED`.
- Gated + trailing post → `BUSY`.
- Gated + post + an assistant message after it → `BLOCKED` (the post was
  answered; no spin).
- Gated + a completed allowed sibling, no post → `BLOCKED` (the misfire guard —
  this is the test that fails if someone reuses `open_turn_unseen_material`).
- Unresolved children + gate + post → `BUSY`.
- Unresolved children + gate, no post → `BLOCKED` (unchanged).

### `tests/agent/test_runner_post_message.py`

Use `FauxProvider` with an exact response script — a wrong number of rounds
then fails loudly instead of hanging.

- **The headline story.** Gated session, `post_message`, `run()`. Exactly one
  faux request; assert its full message list (placeholder present, post last);
  assert the turn did NOT close, `runner.blocked()`, and
  `pending_approvals()` still returns the gate. Then flip the registry to
  ALLOW, `run()` again, and assert the second request's full message list (real
  result in place, trailing assistant) and a COMPLETED close.
- **No post, no round.** Gated session with a completed allowed sibling,
  `run()` → zero faux requests, still `BLOCKED`.
- **No hot loop.** A one-response script; the post-while-blocked run must
  return after exactly one request. (Without 4a's `else` this hangs or exhausts
  the script.)
- **The close guard.** The fall-through round returns `finish_reason="stop"`;
  assert no `TurnFinish` on the path and the gate still `PENDING`.
- **Several posts, one round.** Two posts before the drive → one request
  carrying both.
- **Post during the fall-through call.** Reuse the existing
  `PostDuringResponse` middleware → a second round, driven by the `last_seen`
  half.
- **`hard_max_steps` over a live gate.** Assert the gated execution ends
  `CANCELLED`, its `ToolExecuted` was emitted, and the turn closed ERRORED.
- **Cancel beats the unseen post at a gate.** Mirrors the existing precedence
  test.

The existing `test_a_post_while_blocked_is_answered_in_the_same_turn_after_the_gate_resolves`
changes meaning: the post is now answered *before* the gate resolves. Rewrite
it rather than deleting it — it is the regression this PRD exists for.

### `tests/agent/subagents/test_post_messages.py`

- A gated CHILD plus a post to that child: the child's drive restarts through
  the existing notify door, answers, and re-parks at its gate; the parent is
  unaffected.

### TUI tests

- Submitting while the main conversation is `BLOCKED` posts nothing and shows
  the notice — assert the session path is unchanged, which is what proves no
  round could have run.
- Submitting while a SUBAGENT is gated and its siblings still work still posts
  — the guard must not over-reach into mid-orchestration steering.

---

## Risks during implementation

**The `else` in 4a is the whole hot-loop fix.** If it is written as a plain
fall-through with the progress-`continue` left at the same nesting level, the
drive spins on `decide()`. The "no hot loop" test is what holds this.

**`_settle_undispatched` in the LLM-failure `except` block** runs before a
`raise`. Verify the events actually reach the consumer in both the lazy
(`run()`) and eager (`start()`) consumption modes — the eager path drains the
generator from a background task.

**Two predicates that must stay in agreement.** `_derive_status` and
`_drive_loop` now each consult a gate term and a material term. If they
disagree, a polling consumer sees `BUSY` from a `run()` that does nothing. The
`test_models.py` cases and the runner stories cover the same scenarios from
both sides on purpose.

**No new dependencies. No data-model change.** No new entry type, no new field
on `ToolExecution`, no migration — the placeholder is derived, and a session
written before this change loads and behaves identically.
