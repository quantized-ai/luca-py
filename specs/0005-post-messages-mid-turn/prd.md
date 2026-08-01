# Mid-turn user messages

## Objective

Let a user post a message to a conversation **while the agent is working on
it**. The message is appended inside the open turn, the model sees it on its
next LLM call, and the turn does not complete until every posted message has
been answered.

Concretely:

```python
def post_message(
    self,
    content: str | list[ContentPart],
    conversation_id: str | None = None,
) -> str: ...
```

- `conversation_id=None` targets the main conversation (resolved at call
  time). The parameter works for ANY conversation whose state accepts input —
  including a live subagent — because this is a framework, not an app: what the
  shipped TUI restricts is TUI policy, not a core rule.
- Posting is legal in far more states than today. The one-line summary of the
  new precondition: **a conversation accepts a message whenever something will
  eventually answer it** — with one deliberate exception (D13): a conversation
  whose open turn has unresolved subagents rejects input while they run, even
  though the turn would eventually answer it.

The core guarantee this feature adds (D5): **a turn never closes COMPLETED
while it contains a user message the model has not seen.** A mid-turn message
is answered within the turn it landed in, not silently absorbed into history.

## Why

The current model is strictly turn-based: `post_message` requires IDLE, so
while the agent works the user's only channel is `cancel()`. "Steering" — the
user adding *"write a regression test first"* or *"keep it short"* while the
agent is mid-task — is impossible; an application can only buffer the text and
post it after the turn ends, at which point the work it was meant to steer is
already done. Interactive agent products treat mid-turn input as table stakes;
the durable data model already supports it (a `UserMessage` is an ordinary
entry, and projection is a flat, order-preserving walk), so the gap is purely
in the runner's precondition and one close-site rule.

## This PRD changes documented behavior

Today's contract is explicit and appears in several places; **this PRD
supersedes it, and every statement of it must be updated** (see "Documentation
updates" below). The three doctrines that change:

1. *"Append a user message to the MAIN conversation. Legal when it is IDLE,
   and in no other state."* (`post_message` docstring) — replaced by the
   acceptance matrix in D2.
2. *"Message queueing is gone — a trailing `UserMessage` derives BUSY, and
   BUSY rejects input; 'let the user type while the agent works' is an
   application-level buffer that posts on the next IDLE."* (AGENTS.agent.md,
   status-derivation notes) — reversed. Posting while BUSY is now the point.
   The application-level input buffer is no longer the recommended pattern.
3. *"`post_message` is main-conversation only — the spawn prompt is the one
   and only user message a child ever receives."* (AGENTS.agent.md, subagents
   limits) — relaxed. A subagent with an open conversational turn accepts
   posts (unless its own subagents are active — D13); its seed prompt is
   merely its *first* user message.

What does NOT change is just as deliberate: status derivation, the projector,
the ledger's write doors, the `ContextManager` contract, the compaction
machinery, and `luca.client` are all untouched (D7, D10).

## Decisions

### D1 — Signature and defaults

`post_message(content, conversation_id=None) -> str`, returning the persisted
entry's id exactly as today. `None` resolves to `session.main_conversation_id`
at call time. The shipped TUI keeps calling `self.runner.post_message(parts)`
with no `conversation_id` — main-only input is TUI policy (D12), and nothing
in core knows or cares about that restriction.

### D2 — The acceptance matrix

A post is accepted or rejected from the target conversation's durable state at
call time. `post_message` stays synchronous (D11), so this picture is exact —
nothing can change between the check and the append.

| Target state | Verdict | Why |
|---|---|---|
| Main, IDLE | **accept** | Today's behavior: trailing message, next `run()` opens a turn. |
| Main, BUSY with no open turn (trailing message already queued) | **accept** | A second trailing message. Queueing is now allowed; both are answered by the next turn. |
| Any conversation, open **conversational** turn (derives BUSY or BLOCKED), no unresolved subagents | **accept** | The mid-turn append. This is the feature. |
| Any conversation, open turn with **unresolved subagents** | **reject** — dedicated exception (D13) | The parent is mid-orchestration: its children cannot see its messages and its next LLM call may be far away. Accepting would be an illusion of steering. |
| Non-main, BUSY with no open turn (spawned child not yet driven) | **accept** | The child is queued behind the worker pool and will be driven; the message projects with its seed prompt. |
| Any, CANCELLING | **reject** — dedicated exception (D3) | The turn is being flushed; an append would be buried in the cancelled bracket, silently unanswered. Outranks the D13 check. |
| Any, open **compaction** bracket (scheduled or in-flight) | **reject** — `AgentError` | Preserves the compaction snapshot invariant (D8). |
| Non-main, IDLE | **reject** — `AgentError` | A finished subagent: its result is already resolved into the parent and nothing will ever drive it again. Accepting would wedge a message forever. |
| Any archived (compacted-away) predecessor, **whatever its derived status** | **reject** — `AgentError` | Nothing ever drives an archived conversation. NOT a status check: a queued message compacted behind leaves the archived path `[…, u, ts_c, cmp, tf_c]`, which derives BUSY (closed compaction brackets are transparent) — a status-based rule would accept into it and wedge the message forever. The test is identity: the target is some conversation's `previous_conversation_id`. |
| Unknown `conversation_id` | **reject** — raise | `KeyError` from the status read is acceptable; a wrapped `AgentError` is equally fine. |

Behavioral notes that follow from the matrix and are requirements, not
suggestions:

- Posting into a turn BLOCKED at an approval gate **does not unblock
  anything**. The message waits with the turn and is included in the next LLM
  call after the block resolves. No wake, no notification — if the turn cannot
  advance, neither can the message. (A parent blocked on its *children* is the
  D13 rejection instead — that state does not accept input at all.)
- A mid-turn post needs **no wake infrastructure at all**: projection
  recomputes from the live path on every LLM call, so a live drive picks the
  message up at its next call, a suspended or crashed-and-reloaded session
  picks it up on the next `run()`, for free.
- The compaction check must test **bracket shape** (`is_compaction_bracket`),
  never status — an in-flight compaction derives BUSY, and a status-based
  check would wrongly accept.

### D3 — Rejection during cancellation is a dedicated exception

Posting into a CANCELLING conversation raises a **new, dedicated exception
type**, subclass of `AgentError` — suggested name
`ConversationCancellingError`; the implementer may pick a better one. It must
be distinct from:

- `asyncio.CancelledError` — task-cancellation semantics; catching it in
  application code is hazardous and it does not mean "your post was refused";
- the existing `luca.agent` `CancelledError` — owned by `CancellationToken`;
- `AlreadyCancellingError` — the `cancel()`-twice diagnostic.

Why dedicated: this is one of the two rejections an interactive application
hits in normal operation (Esc, then immediately typing the correction; the
other is D13's) and must handle gracefully — catch, keep the draft, retry
after the flush. Every other rejection in D2 is a persistent state or a
programming error, and plain `AgentError` remains correct for those.

Two properties make this rejection reliable, and both must hold: `cancel()` is
synchronous and appends the durable `CancelRequested` before returning, so a
`post_message` on the very next line already sees CANCELLING — no window where
the cancel is in flight but invisible. And because the state is derived from a
durable entry, the rejection holds across a process restart: a reloaded
CANCELLING session refuses input until the flush is driven.

### D4 — A mid-turn message is an ordinary append

An accepted post appends the `UserMessage` at the end of the path — which,
when a turn is open, is inside it. Same append door as every other entry:
`context_tokens` is calculated by the `ContextManager`, `before_entry_written`
middleware runs, the ledger commits. The `before_post_message` middleware hook
runs for every accepted post and **keeps its current signature**
(`parts -> parts`); it does not learn the target conversation. No speculative
hook arguments before a real second case exists.

No positional insertion exists anywhere in this feature. The append-only path
plus the runner's existing atomic write windows already guarantee wire-legal
orderings: an assistant message and its `ToolExecution` entries land in one
no-yield window, so a posted message can never fall between a tool request and
its executions — it can only land after a completed round or before the next
assistant message. Projection stays a flat walk; the resulting shapes
(`…, tool_result, tool_result, user, assistant, …`) are orderings the
transports already produce today (multiple tool results already project as
consecutive user-role wire messages on Anthropic).

### D5 — The completed-close guarantee

**A turn must not close COMPLETED while its open span contains a user message
the model has not seen.** "Seen" is defined exactly: the message was on the
path at the instant the projection for the most recent LLM call was taken.

When the model returns a final answer (no tool calls) and an unseen message
exists, the drive records the assistant message as usual and then **runs
another LLM round instead of closing**. The next projection contains both the
premature final answer and the steering message, so the model knows what it
already said and answers what it missed. The recorded-then-superseded
assistant message is real model output and is never dropped or rewritten.

Why this rule is load-bearing and not an edge-case patch: posts can only land
during the drive's awaits, and the longest await — for a tool-less turn, the
*only* await — is the LLM call that turns out to be final. Without this rule,
every mid-turn post to a tool-less turn is deterministically buried: accepted
without error, rendered by the UI, then silently never answered while the
session sits IDLE. Rejecting at post time instead is impossible — an in-flight
final call is indistinguishable from any other in-flight call — so the close
site is the only place the decision can live.

Precedence at the close site, in order:

1. **An unconsumed cancel wins.** The existing rule — a `CancelRequested`
   found when the final answer lands controls the close — is unchanged, and
   it outranks the unseen-message check. The message is buried (D6).
2. **The unseen-message check** — loop again.
3. Close COMPLETED.

Bounds still bound: the extra round re-enters the loop top, so
`hard_max_steps` applies to it (a turn at the hard limit closes ERRORED even
with an unseen message — bounded cost wins, and the burial is the D6 case),
and a soft-max / doom-loop `tool_choice="none"` restriction applies to the
extra round like any other.

The close condition must keep keying off the **presence of tool calls in the
response**, never the provider's `finish_reason` — the existing
misclassifying-provider defense stands.

### D6 — Failure closes may bury; burial is bounded

A turn that closes CANCELLED, ERRORED, or TIMED_OUT writes its `TurnFinish`
regardless of unseen messages. A message caught inside such a close is
**buried**: it derives no status (the turn is closed; the session is IDLE) and
nothing proactively answers it.

Buried is not lost. Projection is a flat walk that visits entries inside
failed brackets, so the buried message appears in the model's history on the
next request — it is answered late, on the user's next engagement, not never.

This is accepted V0 semantics, per close outcome:

- **CANCELLED**: the user said stop; stopping the steering message typed
  moments earlier is the correct reading of the instruction. (And D2/D3 make
  the *post-cancel* ordering — Esc, then type — a loud rejection rather than
  a burial, so this case only covers messages that landed before the cancel.)
- **ERRORED / TIMED_OUT**: answering would mean immediately re-calling a
  provider that just failed — an automatic retry, an explicit non-goal. The
  application surfaces the turn failure; the user's next post carries the
  buried message into context.

A future revision may make burial self-heal (a derivation rule that resurrects
a closed turn's trailing unanswered message as BUSY); it is a pure derivation
change over the same recorded state, requires no migration, and is out of
scope here.

### D7 — Status derivation is untouched

No change to `AgentSession.get_conversation_status` / `_derive_status`. A
mid-turn `UserMessage` does not affect any derivation: an open turn derives
BUSY/BLOCKED/CANCELLING from exactly the facts it derives from today, and a
closed turn containing a buried message derives IDLE (D6 accepts this). The
derivation is the most load-bearing pure function in the framework;
this feature ships without adding a rule to it.

### D8 — The compaction invariant is preserved

`post_message` must reject whenever the target's open bracket is a compaction
bracket — scheduled or mid-flight. Two pieces of the compaction machinery are
built on "nothing can be appended while a compaction bracket is open": the
snapshot's positional assumption that the path tail is `[…, ts_c, cmp]`, and
`check_snapshot`'s path-unchanged validation (G2). A mid-compaction append
would fail the compaction with `CompactionPlanError` at best. The docstrings
that state this invariant (`_snapshot_conversation`,
`ConversationProjector._skip_compaction_bracket`) remain true under this PRD
and need no edits — verify their wording anyway.

### D9 — No event for posts (V0)

No `AgentEvent` announces a posted message. The poster holds the returned
entry id and renders locally; that covers the shipped TUI and any single-view
application. The acknowledged gap: a consumer rendering purely from a run's
forwarded event stream (a remote UI) will not see mid-turn messages. Future
work, not V0.

### D10 — Untouched contracts

No changes to: `ConversationProjector` (flat walk already projects a mid-turn
`UserMessage` in place), the ledger's write doors, `ContextManager` (mid-turn
messages flow through the same `calculate_context` path as any append;
`should_compact` remains a drive-top, closed-bracket check — a turn that grows
mid-flight is not compaction-checked, exactly as a long tool turn is not
today), the compaction step, and all of `luca.client` (no transport or
provider edits; see D4 for why the wire shapes are already legal).

### D11 — Concurrency contract

`post_message` stays **synchronous**, and the feature's correctness leans on
the single-event-loop model: sync code sees a frozen picture, appends
interleave only at awaits, and the no-yield close window makes the D5 check
exact (nothing can land between the check and the `TurnFinish`). "What the
model saw" is fingerprintable by one loop-local integer — the path length at
projection time — because the path is append-only. Calling `post_message`
from another thread is unsupported, unchanged from today.

### D12 — TUI behavior

- The prompt stays enabled while the main conversation is BUSY or BLOCKED;
  submitting calls `self.runner.post_message(parts)` exactly as today (no
  `conversation_id`).
- On success, the TUI renders the user message immediately (it may interleave
  with streaming deltas from the in-flight response; that is correct — the
  message will be answered after the current response completes).
- The call is wrapped in try/except. On the D3 exception (CANCELLING), the
  D13 exception (subagents active) — and on `AgentError` generally — the TUI
  shows a brief notice and **preserves the draft in the input** so the user
  can resubmit after the flush (or once the children resolve). Input is never
  silently discarded; the prompt is not proactively disabled while subagents
  run — the rejection is communicated on submit, uniformly with CANCELLING.
- Slash commands stay idle-only (they mutate runner/session state); a command
  submitted mid-turn gets a notice and the draft is preserved.
- Subagent conversations remain non-postable from the TUI. TUI policy only.

### D13 — No posting while subagents are active

A post targeting a conversation whose open turn contains an unresolved
`ChildConversation` is rejected with a **dedicated exception** —
`SubagentsActiveError`, subclass of `AgentError`, a flat sibling of the D3
exception and dedicated for the same reason: it is the other rejection an
interactive application hits in normal operation and must handle gracefully
(catch, keep the draft, retry once the children resolve).

- **Conversation-generic**, like every rule in D2: it keys on the target's own
  open turn, so it applies to the main conversation and to a live subagent
  that spawned children of its own alike. There is no main-special-casing.
- **The predicate is durable**: `open_turn_unresolved_children` reads the
  `ChildConversation` links off the path, so the rejection holds across a
  reload exactly as CANCELLING does, and `post_message` staying synchronous
  (D11) keeps the check exact.
- **Precedence**: CANCELLING outranks it — a cancelled parent mid-wind-down
  raises the D3 exception, not this one. The compaction rejection cannot
  overlap (no children exist inside a compaction bracket).
- **The boundary is the link, not the spawn call.** A post that lands after a
  spawn tool call exists but before its `ChildConversation` entry is appended
  — or while a gated spawn still awaits approval — is accepted: no child is
  active at that instant. The message sits before the spawn in history and is
  answered after the children resolve, like any other pre-spawn mid-turn
  message. This window is small and closing it would require a fuzzy
  "spawn might be coming" predicate; the link is the exact durable fact.
- Once the last child resolves, posting is legal again **within the same open
  turn**, and D5 then guarantees the message is answered before the turn
  closes COMPLETED.

Why reject rather than accept-and-wait: while subagents run, the parent's next
LLM call may be minutes away, and the children never see parent messages — an
accepted "steering" message could not steer the work actually in flight.
Accepting would be an illusion of responsiveness; the rejection is honest, and
the application can say so ("subagents are running").

## Documentation updates (required, part of this feature)

Every statement of the old contract must be rewritten:

- `AgentSessionRunner.post_message` docstring — the acceptance matrix, the
  mid-turn semantics, the D3 and D13 exceptions, the D5 guarantee, the D6
  burial note.
- `ConversationStatus` docstring — the "post a message?" column of its table
  (BUSY and BLOCKED become "yes, into the open turn"; CANCELLING stays "no").
- `AgentSession.get_conversation_status` docstring — the "a second message
  cannot be queued behind a first" paragraph.
- `runner.py` module docstring — the polling sketch (`if runner.idle():
  post_message(...)`) gets a note that posting is also legal mid-run.
- `AGENTS.agent.md` — the status-derivation section ("message queueing is
  gone" ¶), the subagents limits ("post_message is main-conversation only"),
  and the test-file table row for the post_message matrix.
- `docs/agent/` user docs — wherever posting/IDLE is described (follow
  `docs/llm.txt` when editing), including `13-subagents.md`'s "a subagent
  never receives a user message after the seed" paragraph (now: it accepts
  posts unless D13 applies; a finished child rejects them).
- `schedule_compaction` docstring stays true (it promises `post_message`
  raises while its bracket is open — D8 keeps that promise) — verify wording.

## Non-goals

- Any buffering, queueing, or deferred-injection machinery inside the runner.
  A post either appends immediately or raises; there is no pending state.
- A post-message event (D9).
- Self-healing burial via status derivation (D6 — future).
- Changing the `before_post_message` middleware signature (D4).
- TUI input routing to subagent conversations.
- Cross-thread `post_message`.

## Implementation sketch (non-normative)

The implementer may deviate freely where the Decisions above are satisfied.

**Precondition** (in `post_message`): resolve the target; read
`open_turn_index` and `is_compaction_bracket` for the bracket checks,
`get_conversation_status` for CANCELLING, and
`open_turn_unresolved_children` for D13; apply the D2 matrix; then the
existing normalize → `before_post_message` → `_append` path, unchanged.

**Close-site check** (drive loop, step 4): capture `seen =
len(conversation.nodes)` in the sync region where `prepare_llm_call` runs. In
the final-answer close window — immediately alongside the existing
`open_turn_cancel_requested` re-check — test
`any(isinstance(entries[n], UserMessage) for n in nodes[seen:])`; if true,
yield the recorded events and `continue` instead of `_close_turn(COMPLETED)`.
Roughly ten lines; no new runner state beyond the loop-local `seen`.

**Exceptions**: two new classes in `exceptions.py`, both flat subclasses of
`AgentError`: `ConversationCancellingError` (D3) and `SubagentsActiveError`
(D13). No shared intermediate base — the shipped TUI treats every
`post_message` rejection identically, so a taxonomy would be speculative.

**TUI**: `_set_busy` no longer disables the prompt for the main conversation;
the submit handler gains the try/except and optimistic render.

## Testing

House style throughout: full-object assertions on the resulting
`AgentSession` / event list, driven through `DeterministicRunner` and the faux
provider.

- **Matrix**: every row of D2, accept and reject, including non-main targets
  (live subagent turn → accept; finished subagent → reject; queued seeded
  child → accept), the compaction-bracket rejections (scheduled and
  in-flight), and the archived-predecessor rejection — including the
  BUSY-deriving archived shape (queued message compacted behind).
- **Subagents active (D13)**: a parent with unresolved children raises
  `SubagentsActiveError` (live drive and reloaded session); a live subagent
  with no children of its own accepts; a subagent with its own active
  children rejects; once every child resolves, a mid-turn post is accepted
  and answered before `TurnFinish(COMPLETED)`; CANCELLING outranks D13 on a
  cancelled parent with unresolved children.
- **Mid-turn between tool rounds**: post lands after a tool round completes;
  assert the full session — message inside the turn, answered before
  `TurnFinish(COMPLETED)`.
- **The close race**: post lands during the final LLM call. Deterministically
  simulable without timing tricks — e.g. an `after_llm_response` middleware or
  faux-provider hook that posts during the call — assert the turn does NOT
  close, an extra round answers the message, then closes COMPLETED.
- **Two steering messages** in one turn (including both landing before the
  same close) — one extra round answers both.
- **Post while BLOCKED**: gate resolves → message included in the next call →
  answered in the same turn.
- **Cancellation**: post after `cancel()` raises the D3 exception (live and
  reloaded-CANCELLING sessions); post *before* a cancel is buried and the
  session still projects it into the next request (D6).
- **Precedence**: cancel beats the unseen-message check; `hard_max_steps`
  closes ERRORED over an unseen message.
- **Queued trailing messages**: two posts on an undriven BUSY main
  conversation; one turn answers both.
