# Posting a message to a BLOCKED conversation

## The problem

`post_message()` accepts a message when the conversation is `BLOCKED`, and then
nothing happens with it.

The message is appended to the open turn's path and sits there. The model is
not called, the status does not change, and no event says anything. From the
application's side the call succeeded; from the user's side they typed
something and were ignored.

It is **queuing, not sending**. The message is eventually delivered — bundled
into the same request as the tool result, once the block clears — which is
frequently too late to be the thing the user meant by sending it.

## When a conversation is BLOCKED

There is exactly one cause: **a tool call is waiting for an approval nobody has
answered.**

The model asks to run a tool. The runner asks the registry `decide()`, which
answers ALLOW, DENY, or PENDING. PENDING means "I am not deciding this, go ask
a human." That `ToolExecution` stays at `status=PENDING` with
`approval_status=PENDING`. The runner dispatches every *other* allowed call in
the same response, and only then parks. The conversation now derives `BLOCKED`:
nothing on the path can move until the application answers.

With subagents it is the same cause found by recursion — a parent is `BLOCKED`
when it has no work of its own and every child is `BLOCKED`, and a child is
`BLOCKED` for the same reason. It always bottoms out at an unanswered gate.

Nothing else produces it. An open turn with no unfinished tool call is `BUSY`
(the model can be called). Executions the drive can still move are `BUSY`. A
closed turn is `IDLE`. A parked cancel is `CANCELLING`.

## Why the model is not called

Because the wire payload would be malformed, and the framework refuses to build
it.

Every `tool_call` in an assistant message must be answered by exactly one
`tool` message. A gated `ToolExecution` has no result yet, so
`ConversationProjector` raises rather than emitting a request with a dangling
tool call. That rule is deliberate and load-bearing — it is what makes "the
runtime must never call the model mid-execution" a loud failure instead of a
provider error.

So the block is not a policy choice about messages. It is a consequence of the
path not being projectable.

## Example

Starting state — the user asked for something, the model asked to run a
destructive command, and the registry deferred to a human:

```text
UserMessage("Delete the temp files")
TurnStart()
AssistantMessage(tool_calls=[ToolCall(id=tc1, name="bash", command="rm -rf tmp/")])
ToolExecution(tool_call_id=tc1, status=PENDING, approval_status=PENDING)
```

Status: `BLOCKED`. The application is showing an approval prompt.

The user, instead of answering the prompt, types a question:

```text
UserMessage("Delete the temp files")
TurnStart()
AssistantMessage(tool_calls=[ToolCall(id=tc1, name="bash", command="rm -rf tmp/")])
ToolExecution(tool_call_id=tc1, status=PENDING, approval_status=PENDING)
UserMessage("wait, what are you deleting?")          <- accepted, and inert
```

Status: still `BLOCKED`. No model call. Silence.

The user, getting no answer, approves. Now the path becomes projectable and the
model is finally called:

```text
UserMessage("Delete the temp files")
TurnStart()
AssistantMessage(tool_calls=[ToolCall(id=tc1, name="bash", command="rm -rf tmp/")])
ToolExecution(tool_call_id=tc1, status=COMPLETED, result=...)
UserMessage("wait, what are you deleting?")
AssistantMessage("I removed tmp/ — it contained …")
```

The message that was meant to *stop* the deletion is read by the model after
the deletion happened. It was never lost; it was delivered at the one moment it
could no longer do anything.

## The inconsistency this exposes

The framework already delivers a mid-turn post to the model immediately in a
neighbouring case.

A conversation parked on **unresolved subagents** treats a post as unseen
material, flips to `BUSY`, and calls the model on the next drive — deliberately,
so the user can steer while the children are still working ("forget that, stop
the task"). A live parked drive is even woken for it, through the same notify
door `run.notify()` uses.

So the same act has two different outcomes:

| The conversation is parked on… | A posted message |
|---|---|
| unresolved subagents, no gate of its own | reaches the model on the next drive |
| an unanswered approval gate | sits on the path, inert |

Nothing tells the user which of the two they are in, and the reason for the
difference is not a product decision — it is that one path is projectable and
the other is not.

## How bad it is

Today the window is short. A gate lasts as long as a human takes to press a
key, and the user's own way out is to answer the prompt. The concrete harms are
the silence (an accepted message with no feedback), the late delivery (arriving
after the action it was about), and the inconsistency above.

The harm scales directly with how long a conversation can stay `BLOCKED`. Any
future state that blocks for minutes or hours turns "queued" into "lost", so
the design should be fixed on its own terms rather than because the current
window happens to be tolerable.

---

# The change

**A gated `ToolExecution` projects as a placeholder `tool` message.** The path
becomes well-formed, so the model can be called with the gate still open.

**A posted message — and nothing else — un-parks a gated drive.** The drive
falls through to the model call, the model answers the user while the approval
prompt is still up, and the drive parks again at the same gate.

```text
messages = [
    {role: "user",      content: "Delete the temp files"},
    {role: "assistant", tool_calls: [tc1]},
    {role: "tool",      tool_call_id: "tc1",
                        content: "[tool execution is awaiting approval — it has
                                   not run, and this is not its result]"},
    {role: "user",      content: "wait, what are you deleting?"},
]

# -> "I'm asking to run `rm -rf tmp/`. It's waiting on your approval —
#     it would remove the whole tmp/ directory."
```

Which is the answer the user wanted, at the moment they wanted it.

**There is precedent for fabricating this content.** The projector already
invents tool-message text for executions that produced no result — a
`STATUS_ONLY_OUTPUTS` table yielding `"[tool execution cancelled]"`,
`"[tool execution rejected]"`, `"[tool execution interrupted]"`, plus derived
wording for failures. Nothing is stored on the `ToolExecution`; the placeholder
is derived, exactly like the others.

**The projection stays a pure function of the path.** The carve-out is
unconditional — a gated execution always projects the placeholder, whether or
not a post exists below it. There is no new positional rule and nothing caches
the result, so the same path always projects the same way and a mutated path
projects the new way.

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | What un-parks a gated drive? | **An unseen user post, and nothing else.** A new narrow predicate — a `UserMessage` positioned after the open turn's last `AssistantMessage` — plus the drive-local `seen` fingerprint for a post that landed under an in-flight call. |
| 2 | Where does the placeholder wording live? | **Its own `ConversationProjector` ClassVar**, `AWAITING_APPROVAL_OUTPUT`. Not a row in `STATUS_ONLY_OUTPUTS`: that table is keyed by TERMINAL status and this is the single nonterminal carve-out. |
| 3 | Does the placeholder carry `is_error`? | **No, `is_error=False`.** The call has not failed; it has not run. An error result is exactly what makes a model retry a call. |
| 4 | What happens on a failure close over a live gate? | **The gate is settled, like a cancel does it** — terminalized CANCELLED with a `ToolExecuted` event before the `TurnFinish`. No close leaves a nonterminal execution behind. |
| 5 | The trailing-assistant wire shape (below) | **Accepted and documented**, not fixed here. |

## What the codebase says

Five findings from reading `runner.py`, `models.py`, `projection.py`,
`ledger.py` and the client transports. The first four are work this PRD must
do; the fifth is a hazard it must document.

### 1. There is no close rule — the park IS the close rule

The claim "a turn cannot close COMPLETED while an execution is unfinished" is
not implemented anywhere. The drive's close site checks exactly two things: an
unseen user message, and unresolved children. Nothing about executions.

What actually prevents a close over a gate today is the **park at step 3**,
which returns before the model call, making the close site unreachable while
gated. Removing that park to let a post through therefore closes the turn
COMPLETED with the gate still open — and then `pending_approvals()` (which
reads the OPEN turn only) stops returning it, `notify()` finds nothing to
re-decide, and the model's history shows the placeholder forever for a call
that never ran.

**Required:** the close site gains a third term — a turn does not close
COMPLETED while the open turn holds a gated execution.

### 2. The progress-`continue` hot-loops once the park is bypassed

After the park, the drive runs `if undecided or ready or spawned or resolved:
continue`. `undecided` **includes gated executions** — the ledger deliberately
treats `approval_status=PENDING` as still-undecided so a re-drive re-asks the
registry. That guard is unreachable while gated today, because step 3 parks
first. Falling through reaches it: `continue` → re-ask `decide()` → fall
through → `continue`, a tight loop hammering the permission policy.

**Required:** the fall-through skips the progress-`continue`.

### 3. `open_turn_unseen_material` is the wrong predicate here

It counts any terminal non-spawn execution positioned after the last assistant
message. In a gated round, an **allowed sibling that just completed** sits
exactly there. Reusing it would fire a model round with the placeholder on
*every* approval, before any human typed anything — an extra LLM call and a
prompt racing a model call on every gated tool.

The same predicate is right at the subagent park and wrong here because "new"
means different things: a parent parked on children would otherwise sit still
indefinitely, so a tool result really is something to react to; at a gate the
sibling's result arrived in the *same round* as the gate, and there is nothing
to say until the gate is answered.

**Required:** a narrower, post-only predicate (Decision 1).

### 4. Failure closes bury the gate

`hard_max_steps` and the LLM-failure close both call `_close_turn` with no
settling; only the cancel wind-down terminalizes undispatched executions.
Unreachable with a gate open today. After this change it is reachable — each
post while blocked burns a real step, so a user posting repeatedly can hit
`hard_max_steps` and close the turn ERRORED over a live gate, with the same
orphaning as finding 1.

**Required:** both failure closes settle undispatched executions first
(Decision 4).

### 5. The trailing-assistant wire shape — pre-existing, now routine

Once the approval resolves, the real result replaces the placeholder **in
place**, above the post and above the model's answer to it. So the next request
ends with an assistant message:

```text
[user U1, assistant A1(tool_calls), tool tc1(real result), user U2, assistant A2]
```

Neither this shape nor consecutive `AssistantMessage` entries are new — the
0005 close race already produces both, and there is a shipping test asserting a
projected payload whose last message is an assistant message. What is new is
frequency: there it is a race window, here it is the designed path, every time.

Provider behaviour: Anthropic combines consecutive same-role turns, so the
double-assistant is fine, but a **trailing** assistant message is a *prefill* —
the model continues that message rather than starting a new one, and with
extended thinking enabled Anthropic rejects prefill outright. Luca's Anthropic
transport does no normalization; Bedrock merges same-role runs but does not
strip a trailing assistant.

**Decision:** accept and document. It is pre-existing, provider-specific, and
normalizing it means either discarding the model's own last reply or pushing
provider quirks into a pure projector. An application that needs it fixed today
overrides `project()`.

**Not fixable by reordering.** The real result cannot render at the bottom the
way a subagent result does: a `tool` message correlates by `tool_call_id` and
both provider families require it to answer the assistant turn that issued the
call. In-place replacement is forced.

## Behaviour spec

Given a gated conversation and a post:

```text
UserMessage(M1)                                          # BLOCKED
TurnStart()
AssistantMessage(tool_calls=[tc1])
ToolExecution(tc1, PENDING, approval_status=PENDING)
UserMessage(M2)                                          # -> BUSY
```

1. `post_message(M2)` appends, sets the recheck flag and calls the notify door
   (unchanged). The conversation now derives **`BUSY`**.
2. The next `run()` — or a live parked drive, woken — reaches step 3, finds the
   gate, announces `ApprovalRequired` if it has not already, sees the unseen
   post, and **falls through to the model call**.
3. The request carries the placeholder for tc1 and M2 as its last message.
4. The model's answer A2 is recorded. The close site refuses to close
   COMPLETED (the gate is live) and loops.
5. Step 3 finds no unseen post (A2 sits after M2) and parks. The conversation
   derives **`BLOCKED`** again. `pending_approvals()` still returns tc1.
6. Answering the gate and re-driving resolves tc1 normally. The real result
   replaces the placeholder in place; the request's last message is A2.

### Edge cases

| Case | Behaviour |
|---|---|
| Gate with an allowed sibling that completed, no post | Parks exactly as today. Zero extra LLM calls. |
| Several posts before one drive | One round answers them all — the predicate is "any unseen post", not a count. |
| Post lands during the fall-through round's own LLM call | Covered by the drive-local `seen` fingerprint, same as the subagent park; one more round runs. |
| Model issues a NEW tool call in the fall-through round | Ordinary path. It is born, decided, possibly gated too — two gates is legal, and the projection stays well-formed. |
| Model re-issues the SAME call (the known hazard) | Two gated executions for the same approval. See Known hazards. |
| Gate is DENIED after a fall-through round | Terminal REJECTED, projects the existing rejection wording, drive continues normally. |
| Cancel lands while gated with an unseen post | Cancel wins — the existing check inside the gate branch runs before the fall-through. The post is buried, per existing doctrine. |
| `hard_max_steps` reached while gated | Gate settled CANCELLED with a `ToolExecuted`, then the turn closes ERRORED. |
| LLM call fails during a fall-through round | Same settle, then the existing TIMED_OUT / ERRORED close, then re-raise. |
| Conversation has both unresolved children and a gate | Gate term is evaluated first and now yields to an unseen post; the subagent term is unchanged below it. |
| Subagent is the gated one | Identical. `post_message(conversation_id=child)` restarts its drive through the existing notify door. |
| Compaction | Unreachable. Compaction skips a conversation with an open turn, and `post_message` refuses an open compaction bracket. |
| TUI transcript rendering | Unaffected. It never asks the projector about a nonterminal execution. |
| TUI composer while gated | Refuses the post outright — the shipped TUI opts out of this capability. See below. |

### Acceptance criteria

- A gated `ToolExecution` (`PENDING` + `approval_status=PENDING`) projects a
  `ToolMessage` carrying `AWAITING_APPROVAL_OUTPUT` with `is_error=False`.
- `PENDING` with `approval_status` `None` or `ALLOWED`, and `RECEIVED` /
  `RUNNING`, still raise `ProjectionError`.
- A gated conversation with an unseen post derives `BUSY`; without one,
  `BLOCKED`.
- A post while blocked produces **exactly one** model round, then re-parks.
- A gate with a completed allowed sibling and no post produces **zero** model
  rounds.
- A turn never closes COMPLETED while a gated execution is in its open turn.
- Every close that is not COMPLETED terminalizes gated executions and emits
  their `ToolExecuted` events before the `TurnFinish`.
- `pending_approvals()` returns the gate before and after a fall-through round.
- Answering the gate replaces the placeholder with the real result at the same
  path position.
- The shipped TUI never produces a fall-through round: submitting while the
  main conversation is `BLOCKED` is refused with a notice, in every window
  where the composer happens to be mounted.

## Known hazards

**The model may re-issue the call.** A `tool` message means "your tool
returned". A model reading a placeholder may conclude the call is finished and
issue the same call again, leaving two gated executions asking for the same
approval. The wording ("it has not run, and this is not its result") reduces
this but cannot prevent it.

The framework already has the lever (`tool_choice="none"`, which the runner
applies for soft step limits and doom loops), but choosing when to use it is
**application policy, not framework behavior**. An application that posts into
a blocked conversation owns the consequence, and different applications will
legitimately want different answers.

**A projected tool message is no longer always final.** Every fabricated tool
message today comes from a terminal state and will never change; this one is
replaced by the real result on a later call, so the model's own history is
rewritten underneath it. Consistent with a design that re-derives the whole
payload every call and caches nothing, but new.

**Step limits count these rounds.** Each post becomes a real assistant step, so
`hard_max_steps` and the doom-loop check now see them — the same trade the
subagent wake rounds already accepted. Decision 4 is what keeps that safe.

**The trailing-assistant wire shape** — finding 5.

## The framework / application split

The framework supports posting into a blocked conversation and answering it.
Whether an application *offers* that is its own call, and **the shipped TUI
deliberately will not** — posting instead of answering makes no sense in that
UI. The capability exists for applications whose gates last minutes or hours,
where a queued message is a lost message.

The TUI is already structurally there: the frame has ONE dock slot, and
`show_composer` / `show_approval` both clear it before mounting, so while an
approval prompt is up the composer is not disabled — it is gone. Every exit
from `_resolve_approvals` restores it.

But that property is emergent, not guaranteed, and this change makes the gap
matter. The composer swap is driven by the drive WORKER loop: it consumes the
whole run, and only once the run returns does it check `runner.blocked()` and
call `_resolve_approvals()`. So the composer is live and focused from the
moment the registry defers until the worker gets around to swapping it — and
again inside `_resolve_approvals`, which calls `_restore_composer()` before
looping back into `run()`, where a gate that re-arms (an answer that did not
cover every required pair) leaves the composer up over a still-gated
conversation.

Today a keystroke in either window is inert. After this change it drives a real
model round with the placeholder on the wire.

**The TUI therefore gains an explicit guard in its submit handler:
`if self.runner.blocked(): notice and refuse`.** `blocked()` is the exact
predicate — a plain gate derives `BLOCKED` the instant the registry defers, so
it covers both windows, while a SUBAGENT gate with siblings still working
leaves the main conversation `BUSY`, so mid-orchestration steering posts keep
working. It cannot be bypassed by this PRD's own status change either, since
the guard reads the state before the post.

Deliberately NOT done: driving the composer's lifecycle off the
`ApprovalRequired` event. That event is non-terminal on a parent's stream by
design, so it would blank the composer during ordinary subagent work.

`post_message` is the TUI's only door — slash commands are already idle-only.

## Non-goals

- Changing `post_message()`'s acceptance rules. `BLOCKED` already accepts; this
  is about what happens next.
- Resolving the gate. A post is not an approval and never becomes one.
- Making the projector's fail-loud rule generally softer. Exactly one state
  becomes projectable.
- Framework-level policy about `tool_choice`, composer state, or whether an
  application should allow posting at all.
- Normalizing the trailing-assistant wire shape (finding 5).
- Any event announcing that a post caused a round while the gate is open.
  `ApprovalRequired` already fired; the assistant message is its own signal.
