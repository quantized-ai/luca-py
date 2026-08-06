# Implementation notes — 2026-08-06

Implemented in full, following `implementation-plan.md` step by step. Deviations
and discoveries:

- **The derive-status tests live in `tests/agent/test_ledger.py`, not
  `test_models.py`.** The plan named `test_models.py`, but the existing
  derive-status matrix (the children-branch precedence tests, the gate/material
  ordering) sits in `test_ledger.py`, so the new cases went beside their
  siblings: the no-children gate cases (gated+post → BUSY, post answered →
  BLOCKED, completed sibling + no post → BLOCKED) and the children-branch pair
  (`a_gate_outranks_material` rewritten over a non-post material,
  `a_gate_yields_to_an_unseen_post` added).

- **One existing test changed meaning exactly as predicted.**
  `test_a_post_while_blocked_is_answered_in_the_same_turn_after_the_gate_resolves`
  was rewritten into `test_a_post_while_blocked_is_answered_before_the_gate_resolves`
  — the two-run headline story asserting both full wire payloads (placeholder
  round, then real-result round with the trailing assistant).

- **The fall-through run re-asks `decide()` once per loop pass** (the ledger
  counts a gated execution as undecided by design), so the headline story's
  decide script is `[PENDING, PENDING, ALLOW]` — two re-asks in the
  fall-through run (one per pass), then the answer on the next run. Expected
  behavior, just worth knowing when scripting registry doubles.

- **"Cancel beats the unseen post at the gate"** is exercised through a
  `before_permission_check` middleware that calls `runner.cancel()` — the
  cancel lands after the loop-top check has passed, so the gate branch's own
  cancel check is what catches it before the fall-through. Deterministic
  because the faux registry's `decide` never awaits.

- **The TUI blocked-window test recreates the worker-swap window directly**
  (mounting the composer over the still-gated conversation) since the window
  itself is a race by definition; the sibling test proves the guard does not
  over-reach (subagent gated + sibling working → BUSY → the steering post
  still lands and wakes the parent).

## Post-implementation review findings (multi-agent adversarial review)

A 24-agent review (4 dimensions × adversarial verification) over the finished
diff confirmed three things worth fixing, all fixed:

1. **`announced_gate` was a bool and 0008 broke its invariant.** Pre-0008 the
   awaiting set could not change while the flag was True (no model round runs
   while gated); a fall-through round can mint a NEW gated call, which the
   boolean silently never announced — an application driving prompts off
   `ApprovalRequired` alone would never hear of the second gate. Fixed:
   `announced_gates` is now a per-drive SET of execution ids — a re-park at
   the same gate never repeats the event, a new gate triggers a fresh
   `ApprovalRequired` carrying the full awaiting list. Held by
   `test_a_new_gate_minted_by_the_fall_through_round_is_announced`.

2. **The LLM-failure settle had zero coverage** — a verifier deleted the
   `_settle_undispatched` call in the except block and the whole suite stayed
   green. Two tests added (lazy + eager consumption, per the plan's own risk
   note): the gate settles CANCELLED, its `ToolExecuted` reaches the consumer
   before the `TurnFinish(ERRORED)`, and the failure re-raises.

3. **Three stale comments asserted the pre-0008 invariant** ("the model call
   runs only after every execution is terminal"): the runner module
   docstring, the step-4 header, and the cancel-wins-over-failure safety
   comment — plus the same claim in AGENTS.agent.md's step-4 bullet. All now
   carry the 3b exception.

Everything else landed exactly as planned: the projector carve-out
(`AWAITING_APPROVAL_OUTPUT`), `open_turn_unseen_post` +
`_last_assistant_index`, both `_derive_status` sites, drive-loop step 3b with
the `else` around the progress-`continue`, the close-site third term,
`_settle_undispatched` extracted from `_wind_down` and added to both failure
closes, `_has_unseen_post`, the TUI `blocked()` guard, and the doc updates
(10-projection, 05-permissions, 04-runner, 02-data-model, the TUI README,
AGENTS.agent.md).
