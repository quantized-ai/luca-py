# Implementation plan — v2: re-awaken the parent, steer mid-orchestration, stop subagents

Implements `v2.md`. Supersedes three v1/0004 doctrines; everything else in
those specs stands. This revision folds in the findings of a four-lens
adversarial design review (runner, data model/projection, spec fidelity,
test blast-radius); the review-driven changes are marked ⟨R⟩.

## Context: what exists today

A parent that spawns subagents is **parked until every child resolves**. The
drive loop's step 3c waits on the subtree whenever the open turn has an
unresolved `ChildConversation`, and only reaches the model call (step 4) when
none remain. Three mechanisms enforce that:

1. **Projection fails loud** on an unresolved `ChildConversation`
   (`ProjectionError`), like a nonterminal execution.
2. **`post_message` rejects** a conversation whose open turn has unresolved
   children (`SubagentsActiveError`, v1's D13).
3. **The drive parks** in `_await_subtree` until nothing in the subtree can
   advance.

A resolved child projects as one synthetic user message at the LINK's path
position: `<task id={tool_call_id}>\n{content}\n</task>`. Child resolution is
an in-place mutation of the link plus an appended private result-tool
execution (which projects as nothing).

## What v2 changes

1. **The parent re-awakens after each direct child resolves.** The model is
   called with everything new since its last response and can act (text,
   tools, more spawns, stops). The turn still cannot close COMPLETED while
   any child is unresolved.
2. **Users can post mid-orchestration.** `SubagentsActiveError` is deleted; a
   post to a parent with running children is accepted and wakes the parked
   drive.
3. **Child results are delivered append-only**: each resolution projects a
   task-update user message at the RESULT EXECUTION's path position, so every
   wake ends with fresh input after the model's last message and no earlier
   message is ever rewritten. ⟨R — redesigned; see V2-D5⟩
4. **Two new contrib tools**: `stop_subagent` and `list_subagents`, plus the
   core-side stop handshake and `SubagentFinished(CANCELLED)` for flushed
   children.

## Decisions

### V2-D1 — The wake condition is a durable path predicate plus a live fingerprint

The parent parks on its children **iff the open turn holds nothing the model
has not been shown**. Two sources, combined at the park site:

**(a) The durable predicate** — a new pure path derivation in `models.py`:

```python
def open_turn_unseen_material(nodes, entries) -> bool
```

True iff, **after the last `AssistantMessage` of the open turn**, there exists:

- a `UserMessage` (a mid-turn post), or
- a **terminal** `ToolExecution` of a tool that does **not declare spawning**
  — an ordinary tool result, a runtime-minted result-tool execution (the
  durable record that a child resolved), a stop execution, a REJECTED or
  failed call. A spawn-DECLARING call's outcome — spawned, declined, refused
  or failed — is never material, whatever it settled to: keying on the
  declaration rather than the settled payload keeps a mixed all-spawn round
  from waking on its refusals while its committed siblings are still
  starting (an implementation-time refinement: payload-keyed material made
  REFUSED overflow spawns wake the parent instantly, racing the surviving
  children's first calls).

No `AssistantMessage` in the open turn → False.

**(b) The live fingerprint** ⟨R⟩ — a message posted **while a wake round's LLM
call is in flight** lands BEFORE the recorded assistant message, so (a) cannot
see it and the durable state is indistinguishable from an answered post. The
drive therefore carries v1's `seen` fingerprint across iterations: the park
check also runs `_has_unseen_user_message(conversation_id, last_seen)`, where
`last_seen` is the path length captured for this drive's most recent
projection (`None` before the first call — then only (a) applies). A reload in
exactly that window loses the immediacy (the message waits for the next
material) — accepted and documented; the live drive does not.

Why this shape:

- **Durable and resume-exact** for everything except the (b) window: child
  resolution appends the private result execution, so "a child resolved since
  the model last spoke" is readable off the entries.
- **No wake on a pure spawn round** (spec §1's example): after
  `assistant(spawn…) → executions COMPLETED(payload) → links`, everything
  after the last assistant is a settled spawn → no material → park.
- **Normal agentic continuation is preserved**: a woken model that calls a
  tool gets its terminal result as material — the loop continues like any
  ordinary turn until the model answers text-only, then parks.
- **Nonterminal executions are deliberately not material** — they are the
  ordinary loop's business, and a gated execution must not wake a model it
  cannot help.

### V2-D2 — Drive-loop changes (step 3c, close sites)

- **Step 3c** becomes: open turn has unresolved children **and** no unseen
  material (V2-D1 a+b) → `_await_subtree`; otherwise fall through to step 4.
- **The COMPLETED close site**: `elif unseen-message or unresolved-children →
  continue` (never close COMPLETED over either). The next iteration's 3c
  check routes a live unseen message straight to another round via
  `last_seen`; with neither material nor unseen message it parks. ⟨R⟩
- **`hard_max_steps` with unresolved children**: the hard limit is a cost
  stop, so it wins — `_cascade_cancel` the children, settle them
  (`_settle_children`: wait for their drives, flush, resolve every remaining
  link with the cancelled result), **then re-check
  `open_turn_cancel_requested`** ⟨R⟩ — a user cancel that landed during the
  settle's awaits controls the close (close with the cancel's outcome via the
  ordinary wind-down close; otherwise close ERRORED). The alternative — park
  until children resolve — creates a hot-loop BUSY state when children are
  gated.
- **LLM failure (ERRORED / TIMED_OUT) with unresolved children**, split by
  depth ⟨R⟩:
  - **Main conversation (depth 0)**: do **not** close; re-raise with the
    bracket open. Children are independent loops mid-work; a parent provider
    hiccup must not destroy them. The next `run()` resumes (material
    unchanged → the model call is retried by the app's explicit action, never
    automatically).
  - **A subagent parent (depth > 0)**: cascade-cancel + settle its own
    children, close ERRORED, raise — exactly the shape a childless subagent
    failure has today. Nothing outside a subagent ever retries it mid-run, so
    leaving its turn open would hang the tree (a driverless BUSY child
    nothing redrives); closing terminalizes it into "a finished child whose
    result says so", the established doctrine.
- Invariant kept total: **no close ever leaves an unresolved child** — the
  COMPLETED close is guarded, the depth-0 failure close is skipped, the
  depth>0 failure and hard-max closes settle first, and the cancel wind-down
  already resolves everything. A projection-level tripwire backs it (V2-D5).
- Known, accepted degradation ⟨R⟩: after a depth-0 failure consumed via plain
  `await runner.run()` (no context manager), live children keep driving but
  the NEXT run's handle does not adopt them (`_restart_unresolved_children`
  skips live drives), so their events land on the dead handle. Session
  correctness is unaffected (`_ensure_driven` wakes the new drive by
  conversation id); documented, not engineered around.

### V2-D3 — `post_message` accepts mid-orchestration posts and wakes the drive

- Delete the unresolved-children rejection and `SubagentsActiveError`
  entirely. Every other row of v1's D2 acceptance matrix stands.
- After a successful append, `post_message` uses the full notify door ⟨R⟩:
  `self._recheck.add(target)` **and** `self._ensure_driven(target)` — the
  same pair `run.notify()` uses, because `_await_subtree` treats the recheck
  set as the source of truth in its teardown window; the event alone can be
  set on a dying drive and lost. A live parked drive wakes; a framework-owned
  parked child restarts; anything else is a no-op and the next `run()` picks
  the message up.
- v1's close-site unseen-message guarantee is unchanged and now also covers
  wake rounds (V2-D1 b / V2-D2).

### V2-D4 — Status derivation: the children branch, reordered

`_derive_status`'s unresolved-children branch becomes, in order ⟨R — gates
must dominate the material term⟩:

1. any **advanceable** execution (RECEIVED / RUNNING / PENDING with approval
   `None`/ALLOWED) → BUSY — new pure helper
   `open_turn_has_advanceable_executions`, extracted from
   `open_turn_is_runnable` (which becomes "advanceable, or nothing pending
   left");
2. any child BUSY / IDLE / CANCELLING → BUSY (unchanged);
3. any **awaiting** (gated) execution → BLOCKED — the application must act
   first; without this a gated wake-round call plus unrelated material would
   derive BUSY while `run()` can only re-park, hot-looping every
   poll-for-BLOCKED consumer;
4. `open_turn_unseen_material` → BUSY (the next run() calls the model);
5. else BLOCKED.

The no-children branch is untouched. v1's D7 ("derivation untouched") is
superseded.

### V2-D5 — Projection: append-only task updates at the result execution's position

⟨R — full redesign. The draft rendered results at the LINK's path position;
because resolution mutates the link in place, every wake after the first
would end with the model's own stale assistant message and deliver the new
result only as a rewrite of an earlier user message — providers read a
trailing assistant message as a prefill continuation, the stimulus would be
invisible, and the prompt cache would be invalidated mid-history on every
resolution. All three code reviewers found this independently.⟩

The new rendering is **append-only and chronological**:

- **A `ChildConversation` link projects nothing** — neither resolved nor
  unresolved. It is parent bookkeeping; its data reaches the wire through the
  entry whose position records WHEN the result arrived.
- **The task-update message renders at the result execution's path
  position.** `ChildConversation` gains `result_execution_id: str | None`
  (V2-D6). `project()` pre-builds `{result_execution_id → link}` over the
  entry store; a walked node in that map whose link has `execution_result`
  set projects a synthetic user message instead of the private-nothing:

  ```
  Subagent task update:
  <task id={task_id} status={status} completed_at="{iso}">
  {content}
  </task>
  ```

  `task_id` from the spawn payload (via `link.tool_execution_id`), `status` =
  `completed` / `failed` from `execution_result.is_error`, `completed_at`
  from the result execution's own `ended_at` (absolute UTC ISO — a relative
  "10 seconds ago" needs a wall clock, breaking projection determinism),
  `content` from `link.execution_result` (non-text blocks appended after the
  text block, as today). All wording/templates are class vars
  (`CHILD_UPDATE_PREAMBLE`, `CHILD_UPDATE_TEMPLATE`,
  `CHILD_COMPLETED_AT_TEMPLATE`).
- **A link resolved WITHOUT a result execution** (`result_execution_id is
  None`, `execution_result` set — the cancel wind-down and the settle paths)
  renders the same tag at the LINK's position, without `completed_at`. This
  is the only link-position rendering left, and it only occurs inside failed
  brackets, where there are no later wake rounds to contradict.
- **No pending tags.** Spec §1's single-message ideal (completed + pending
  tasks in one update) cannot be reconciled with append-only projection; the
  spec explicitly allows per-entry updates ("if … projection has to project 1
  message per entry that's ok as well"), and the model already knows what it
  spawned (spawn results carry task ids and descriptions; `list_subagents`
  reports live status on demand). This also dissolves the draft's
  pending-vs-`cancelling` vocabulary mismatch between projection and
  `list_subagents`.
- **The fail-loud tripwire moves rather than disappears** ⟨R⟩: `project()`
  raises `ProjectionError` on an unresolved link in a **closed** turn (the
  walk tracks open-turn state; an unresolved link inside the open turn is
  legal and renders nothing). A bug violating V2-D2's close invariant fails
  loudly on the next request instead of silently projecting a task nothing
  will ever finish.
- **`project_pruned` on a PRIVATE original projects nothing** ⟨R⟩ — the
  current behavior emits a `ToolMessage` with a runner-minted
  `tool_call_id` no provider ever issued (protocol-illegal); pruning a result
  execution now simply drops that update from context (and its wake already
  happened). Pruning a result execution mid-open-turn before its wake is an
  application hazard, documented.
- **Context accounting** ⟨R⟩: a link counts `execution_result` content only
  when `result_execution_id is None` (it renders only then); otherwise 0 —
  removing today's silent double-count between the link and the result
  execution, whose own `context_tokens` already cover the content.

### V2-D6 — `ChildConversation.result_execution_id`

New field `result_execution_id: str | None = None`, stamped by
`_derive_child_result` in the same `_persist_entry` write as
`execution_result` (atomic). The wind-down/settle resolutions leave it `None`
— that absence is what selects the link-position rendering. The draft's
`resolved_at` field is dropped ⟨R⟩ — `completed_at` reads the result
execution's `ended_at`, and the no-execution renderings carry no timestamp.

### V2-D7 — The stop handshake (core)

Mirrors the spawn handshake, value-side only:

- `STOP_MARKER = "is_subagent_stop"`; `stop_payload(execution)` returns the
  `structured_content` dict of a COMPLETED execution whose marker is `True`,
  else `None`. Lives in `models.py` beside `spawn_payload` (V2-D9). There is
  no gate half: stopping bypasses no cap.
- New drive step (after the spawn handshake, before the flush/resolve block):
  `_stop_children(conversation_id)` — for each open-turn execution with a
  stop payload: `task_id` missing/empty → `AgentError` (contract violation);
  else find the first matching unresolved direct child **whose
  `ChildConversation` link precedes the stop execution on the path** ⟨R⟩ — a
  stop can only ever target children that existed when it was issued.
  Without the position bound, the stop execution is a standing kill order for
  the rest of the turn: a later spawn reusing the same task id (task ids are
  model-authored) would be killed by a stop nobody issued, on the very next
  loop pass. The position bound is also what makes reload-replay truly
  idempotent. No match, or the child derives IDLE (already finished), or it
  is already CANCELLING → no-op. Otherwise
  `self.cancel(conversation_id=child)` (suppressing
  `AlreadyCancellingError`), cascading to the child's own subtree as any
  cancel does.
- Semantics of stopping are **exactly `cancel()`**: token trip, grace windows
  from `RuntimeConfig`, no new projections for the child, wind-down closes
  its turn CANCELLED, the parent resolves the link through the ordinary
  result tool. **Note the shipped default** ⟨R⟩:
  `tool_cancellation_grace_period` defaults to 0, so a stopped child's
  in-flight tool bodies are hard-cancelled (INTERRUPTED) — the spec's stated
  preference ("allow the ToolExecutions to finish") holds only when the
  application configures a grace window. Honoring it by default would need a
  per-cancel grace override on `CancelRequested`; deliberately not built —
  flagged for review.
- The stopped child's flow back to the model needs no new machinery: the stop
  execution is material (the model promptly sees "signal received"), and the
  child's later resolution appends a result execution → a task update with
  its outcome.

### V2-D8 — `SubagentFinished(CANCELLED)` for flushed children

`_flush_cancelled_children` appends a `SubagentFinished(conversation_id=child,
outcome=cancel.outcome)` after each child it winds down — **only when the
parent's run is framework-owned** (`autostart_subagents=True`) ⟨R⟩: under
`False` no lifecycle events exist, and the flush runs in both modes. Emitted
exactly once (the flush consumes the cancel; a child with a live drive is
skipped there and announces through `_end_run` as today). `Spawned ≺ Finished`
with no `Started` becomes a legal stream shape meaning "cancelled before
admission" — supersedes 0004's "queued cancelled children emit no lifecycle
events", per v2.md's explicit ask.

### V2-D9 — Spawn helpers move to `models.py`

`SPAWN_MARKER`, `SPAWN_REQUIRED_KEYS`, `declares_spawn`, `spawn_payload`,
`spawns_committed` move from `runner.py` to `models.py` (plus the new
`STOP_MARKER` / `stop_payload`). They are pure functions over the data model,
and the projector now needs `spawn_payload`. `core/__init__` re-exports
unchanged — verified: nothing imports them from `luca.agent.core.runner`
directly.

### V2-D10 — The two new tools (contrib)

`luca/agent/contrib/subagents/tools.py`:

- `SubagentStop(BaseModel)`: `is_subagent_stop: bool = True`, `task_id: str`,
  `reason: str | None = None` — declared as `StopSubagent.output_schema`.
- `StopSubagent(Tool)` — `name = "stop_subagent"`. `execute` validates
  against the live session: the `task_id` must name an unresolved, non-IDLE
  direct child of this conversation's open turn. A live match returns the
  full payload plus a "stop signal received" line; no match returns
  `is_error=True` with the known task ids **and the full payload with
  `is_subagent_stop=False`** ⟨R — the declared output schema is the plugin's
  guarantee about what it emits; an absent payload would break that doctrine,
  and the flag already carries "did it happen", exactly like the spawn
  tool's decline⟩. The runner re-checks anyway — a child can finish between
  the tool body and the handler.
- `ListSubagents(Tool)` — `name = "list_subagents"`, read-only, no
  `output_schema`. Renders the open turn's direct children: one
  `<task id=… status=…>` block with description and prompt per child.
  Status: resolved → `completed`; unresolved + child CANCELLING →
  `cancelling`; else `pending`.
- `SubagentToolRegistry.get_tools` additionally withholds both new tools when
  the conversation's open turn has no `ChildConversation`.
- **A second callable prompt part** ⟨R⟩, keyed on the same has-children
  predicate, teaches the update flow and the two tools — it cannot ride on
  `SPAWNING_PROMPT`, whose part is gated by `spawn_gate_open`: a spent spawn
  budget would silence it while stop/list are still offered, violating the
  plugin's own "the prompt and the tool list must never disagree" rule.
  `SubagentsPlugin.get_system_prompt_parts` returns both callables;
  `SPAWNING_PROMPT` itself gains a sentence that results arrive as tasks
  complete.
- `CreateConversationResult` becomes **outcome-aware**: a child whose turn
  closed with a non-COMPLETED outcome gets its last words prefixed with the
  outcome ("The subagent was cancelled; its last message: …", `is_error=True`)
  instead of presenting them as an ordinary answer — without this, a stopped
  child's update reads `status=completed` with its last words as if it
  finished normally. The no-final-message transcript fallback stays.

### V2-D11 — What deliberately does not change

- The worker pool: rules, FIFO, release-on-park, re-acquire on wake. A nested
  parent's wake round is productive work and correctly holds a slot.
- `pretty_print` (renders `execution_result` off the link — still set).
- Depth semantics: a child's completion wakes only its direct parent
  (structural).
- The seed projection, spawn gate, budget, compaction, approvals machinery.
- `autostart_subagents=False`: unchanged mechanics **for the documented
  serial pattern** (drive children inside the `SubagentsSpawned` branch).
  ⟨R⟩ An application driving child handles on separate tasks while iterating
  the parent has no wake source when a lazy child drive ends (pre-existing
  gap, amplified now that wake-per-resolution is the headline); documented as
  a pattern constraint, not engineered around in this change.
- Step economics ⟨R — documented, not changed⟩: wake rounds are real
  assistant steps, so an orchestrating turn now costs ~O(children) steps
  where it cost ~1; `hard_max_steps`/`subagent_hard_max_steps` tuned for the
  old shape will trip earlier, and per V2-D2 a trip now cancels in-flight
  children. The doom-loop flag is per-turn and sticky, so a model repeating
  an identical status-ish call across wake rounds can pin
  `tool_choice="none"` for the rest of a long orchestration. Both get doc
  callouts; neither knob changes meaning.

## File-by-file change list

**`luca/agent/core/models.py`** — spawn helpers in (V2-D9) + `STOP_MARKER` /
`stop_payload`; `ChildConversation.result_execution_id`;
`open_turn_unseen_material`, `open_turn_has_advanceable_executions` (and an
awaiting twin), `open_turn_is_runnable` reworked on them; `_derive_status`
children branch (V2-D4).

**`luca/agent/core/exceptions.py`** — delete `SubagentsActiveError`; module
docstring.

**`luca/agent/core/projection.py`** — V2-D5 wholesale: the result-execution
map + update rendering, link renders nothing (except the no-execution
resolved case), closed-turn unresolved-link tripwire, `project_pruned`
private-original fix, new templates, payload task ids, `completed_at`.

**`luca/agent/core/context_manager.py`** — link counts only when
`result_execution_id is None` (V2-D5).

**`luca/agent/core/runner.py`** — imports from models; `post_message` (drop
rejection, notify door); drive loop: step 3c with material + `last_seen`,
stop step, close-site guard, hard-max settle + cancel re-check, depth-split
failure close; `_settle_children` factored from `_wind_down_async`;
`result_execution_id` stamp; `_stop_children`; `_flush_cancelled_children`
guarded `SubagentFinished`; `_retire_child_failure` (a resolved child's stored
task exception is retrieved so asyncio never reports it unretrieved — a
subagent's failure never propagates, so nobody will ever await that task);
docstring updates.

**`luca/agent/core/__init__.py`** — export updates.

**`luca/agent/contrib/subagents/`** — V2-D10 (tools, registry withholding,
two prompt parts, outcome-aware result tool, package surface).

**`luca/agent/contrib/tui/app.py`** — stale comment only ("the two transient
rejections" → one). Panels/routing/replay already handle interleaved parent
activity (verified).

## Test plan

Scripting discipline for wake rounds ⟨R⟩ — the faux transport pops responses
FIFO across ALL conversations, and a parent now calls the model concurrently
with running children, so:

- **Wire-format and single-update stories run on single-child chains**, where
  the wake round is deterministic.
- **Two-child staggered stories hold the sibling on a scripted PENDING
  approval decision** (a gated child makes no LLM calls), assert the first
  update round exactly, then ALLOW and continue.
- **Anything with ≥2 free-running children** either asserts structure only or
  routes responses per conversation (promote
  `test_integration_full_stack.py`'s `ConversationScript` into
  `tests/agent/subagents/conftest.py`).
- Two children resolving in one loop pass coalesce into ONE wake round
  (`_resolve_children` resolves every idle child per pass) — never assert
  "exactly one wake per child" with free-running siblings.

Updated (complete blast radius per review): `test_child_conversation.py`
(rendering redesign; the SPAWN literal needs a real spawn payload;
`result_execution_id` round-trip), `test_runner_post_message.py` (the two
SUBAGENTS_ACTIVE rows flip to accept+wake), `tests/agent/subagents/*` —
`test_parallel_and_control.py` (the parked-until-all-resolve story becomes
close-blocked + wake-per-resolution; FIFO restructures at 71/93/118/142/278/
529/623), `test_workers.py` (FIFO + the queued-cancelled-child silence
assertion reverses per V2-D8), `test_spawn_handshake.py` (tag format,
positions), `test_depth.py` (tag ids/format), `test_post_messages.py`
(the rejection story inverts; module docstring), `test_spawn_budget.py`
(REFUSED spawn is material → immediate wake), `test_integration_full_stack.py`
(turn 1's todo result is material → main wakes; script gains the wake
response), `tests/agent/contrib/test_subagents.py` (tool lists + prompt
parts), tui `test_app_post_message.py` (accept + script the wake reply),
tui `test_subagents.py` (trees with interleaved parent cells; refused-spawn
story wakes). `test_models.py` / `test_ledger.py` gain the new derivations.

New stories: wake-per-resolution with the exact update projection (gated-
sibling discipline); a post mid-orchestration answered promptly (live, and
the mid-LLM-call race via the posting middleware); reloaded acceptance +
BUSY-from-material; stop end-to-end on a hung child (cancelled, Finished,
link resolved, update wake) + unknown/finished task id + reused task id after
a stop (position bound); `list_subagents` output; stop/list withheld before
the first spawn; hard-max cancelling children (+ cancel landing during the
settle); depth-0 LLM failure leaving the turn open and resuming; depth>0
failure closing ERRORED; gates-above-material status derivation; the
closed-turn unresolved-link projection tripwire; `project_pruned` on a
private original.

## Docs

`13-subagents.md` (steps 4–6, diagram, §5–§7, stop/list, step economics),
`04-runner.md` (matrix, callout), `10-projection.md` (§1, §6, §8 — the
update rendering), `02-data-model.md` (§7, §10), `12-compaction.md`
(unresolved-link bullet), `11-context-and-usage.md` (link counting),
`contrib/subagents/README.md` (four tools), `AGENTS.agent.md` (subagents
section, §11 derivation, post_message matrix, exceptions, engine order, test
table).

## Post-implementation review round (fixes folded in)

A second adversarial review over the finished diff added these:

- **Redriven drives register in `_runs`** — the pre-existing gap ("a
  `_redrive`n drive was never re-registered") became a corruption path once
  stop/post can cancel a child that was restarted out of band: every liveness
  check (the cancel door's token trip, the flush, the settle) keys on
  `_runs`, and an unregistered live drive read as dead — the parent's flush
  would close a turn the drive kept writing to.
- **The next `run()` adopts still-LIVE children** (`_restart_unresolved_
  children` re-parents handles found in `_runs`) — the depth-0
  failure-leaves-turn-open path otherwise orphaned them onto the dead
  handle's queues (events, gates, `child()` lookups all lost).
- **Duplicate-id stop matching is issue-time-positional** — task ids are
  model-authored and non-unique; skipping a RESOLVED match outright would
  turn one consumed stop into a standing kill order for the next same-id
  sibling, while consuming on ANY resolved match would eat a stop plainly
  aimed at the sibling. The target is the first match still unresolved when
  the stop was issued: a match whose result execution PRECEDES the stop on
  the path is skipped (the model was naming its sibling), one that resolved
  after it consumes the signal.
- `list_subagents` reports `failed` for an is_error resolution (the same
  split the task update uses — one vocabulary across the surfaces);
  `stop_subagent` omits the reason clause when none was given; the result
  tool's outcome prefix also covers the no-final-message path and words
  `timed_out` as "timed out".
- Doc corrections: the settle-resolved link's early placement is acknowledged
  (wake rounds recorded while the child lived sit after it), the
  spawn-budget-spans-wake-rounds and soft-max-vs-control-tools interactions
  are called out, and the prune-the-result-execution hazard is documented.

## Flagged decisions

1. **Task updates are per-resolution messages at the result execution's
   position; no pending tags; links render nothing** — the append-only
   redesign forced by the retro-mutation blocker. Spec §1's single
   partial-update message is served by per-entry updates (allowed by
   v2.md:39) + `list_subagents`.
2. Task tags use the **payload `task_id`** (was the spawn `tool_call_id`) —
   required for stop/list coherence.
3. `completed_at` is absolute UTC, not relative (determinism / prompt cache).
4. No wake on a pure spawn round; any non-spawn tool result does wake.
5. Hard-max with running children cancels them; a **main-conversation** LLM
   failure leaves the turn open (re-raised), a **subagent** parent's failure
   settles its children and closes ERRORED.
6. Stopping uses the existing cancel grace config — **with the shipped
   default (grace 0) in-flight tool bodies are hard-interrupted**, which is
   the inverse of the spec's stated preference; honoring it by default needs
   a per-cancel grace override, deliberately not built here.
7. Flushed never-started children emit `SubagentFinished(CANCELLED)`
   (framework mode only) — reverses a 0004 decision per v2.md's ask.
8. `ChildConversation` gains `result_execution_id` (not `resolved_at`).
9. Wake rounds consume real steps; `hard_max_steps` budgets tuned for the
   parked design now trip earlier (docs callout, no knob change).
