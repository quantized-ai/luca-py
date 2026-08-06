# 2026-08-06

The initial PRD framed this as a projector problem: `post_message()` on a
`BLOCKED` conversation is inert because a gated `ToolExecution` is not
projectable, so make it projectable with a placeholder `tool` message and the
model can be called. It listed four open questions (placeholder wording,
`is_error`, whether the status change is conditional, whether an event fires)
and asserted that "the close rule is untouched — a turn still cannot close
COMPLETED while an execution is unfinished."

Reading `runner.py`, `models.py`, `ledger.py` and the client transports turned
up five things that changed the scope from "one projector carve-out" to "one
projector carve-out plus four runner corrections":

1. **The close rule does not exist.** The drive's close site checks an unseen
   user message and unresolved children — nothing about executions. What
   actually prevents a close over a gate is the park at step 3, which returns
   before the model call. Bypassing that park to let a post through closes the
   turn COMPLETED over a live gate, which strands the approval outside any open
   turn (`pending_approvals()` reads the open turn only) and freezes the
   placeholder into the history forever. The close site needs a third term.

2. **The progress-`continue` after the park hot-loops.** `undecided` includes
   gated executions by design, so falling through reaches
   `if undecided or ready …: continue` and re-asks `decide()` in a tight loop.

3. **`open_turn_unseen_material` is the wrong wake predicate.** It counts a
   terminal non-spawn execution after the last assistant message, and an
   allowed sibling that completed in the same round as the gate is exactly
   that — so reusing it would fire a model round on every approval in the
   system before anyone posted anything. A narrower post-only predicate was
   introduced instead.

4. **Failure closes bury the gate.** `hard_max_steps` and the LLM-failure close
   call `_close_turn` without settling; only the cancel wind-down terminalizes
   undispatched executions. Since each post while blocked now burns a real
   step, both became reachable with a live gate.

5. **The trailing-assistant wire shape** the user raised is pre-existing, not
   new: the 0005 close race already produces both a projected payload ending in
   an assistant message and two adjacent `AssistantMessage` entries, and there
   is a shipping test asserting it. What changes is frequency — a race window
   becomes the designed path. Accepted and documented rather than fixed, since
   normalizing means either discarding the model's own last reply or pushing
   provider quirks into a pure projector.

A sixth item came from Santiago after the first pass: the shipped TUI must not
use this capability at all. Reading the TUI showed it is already structurally
there — the frame has one dock slot, so an approval prompt physically replaces
the composer — but the swap is driven by the drive worker (run to completion,
then `runner.blocked()`, then `_resolve_approvals()`), leaving windows where the
composer is mounted over a gated conversation: before the parked run returns,
and after `_restore_composer()` when a partially-answered gate re-arms. Inert
today, a real model round after this change. The TUI therefore gains an explicit
`runner.blocked()` guard in its submit handler — chosen over gating on
`pending_approvals()`, which would wrongly block legitimate steering posts while
a subagent is gated and its siblings still work, and over reacting to the
`ApprovalRequired` event, which is non-terminal on a parent's stream by design.

The four open questions were resolved: post-only wake predicate; a dedicated
`AWAITING_APPROVAL_OUTPUT` ClassVar rather than a row in the terminal-only
`STATUS_ONLY_OUTPUTS`; `is_error=False` (the call has not failed, and an error
result is what makes a model retry); no new event. A fifth decision was added:
failure closes settle the gate rather than burying it.

The PRD was rewritten around a decisions table, the five codebase findings, a
behaviour spec with an edge-case table and acceptance criteria, and an explicit
framework/application split (the shipped TUI will keep its composer closed while
an approval prompt is up; the framework capability exists for applications whose
gates last minutes or hours). `implementation-plan.md` was added.
