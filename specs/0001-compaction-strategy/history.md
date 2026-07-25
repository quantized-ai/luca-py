# 2026-07-25

The PRD arrived already well developed: compaction opens a new conversation
inside the same session, core owns the contract and the transition, contrib owns
the default policy. That framing was verified against the code and kept — the
invariants it rests on (append-only `entries`, session-wide `tool_executions`,
conversation-keyed `usages`, `PrunedEntry` referents never removed) all hold.

Refinement focused on the core contract and its public API, per the user's
scope: no implementation code, and a check for anything the design would break
as a side effect. Reviewed `models.py`, `ledger.py`, `runner.py`,
`projection.py`, `context_manager.py`, `middleware.py`, `events.py`, `utils.py`,
plus the contrib TUI seams, and ran the suite (910 passed, 14 skipped — only
with `tests/agent/contrib/compaction/` ignored, since seven untracked files
there import a package that does not exist on this branch and break collection).

Eleven holes were found in the core spec. The three that changed the design
rather than just documenting it:

- **`start()` would never compact.** `AgentRun.__init__` opens the turn bracket
  synchronously at `start()` time, so an eager run always presents the drive
  with an open turn and the compaction step would skip forever. `should_compact`
  being sync is what makes the fix possible: `start()` decides at call time and
  opens a compaction bracket instead.
- **`derive_status()` needed a new rule.** A failed compaction closes its
  bracket on the pre-compaction path, so `tf_c(ERRORED)` becomes the leaf and
  hits the retry-ready rule — a polling loop then opens a fresh bracket every
  drive and spins. Worse, any closed compaction bracket buries a queued
  `UserMessage` behind it, so the message stops deriving `PENDING` and the
  question is silently dropped across a suspend or crash. One rule — a closed
  compaction bracket is transparent to derivation — fixes both. The spec had
  claimed no derivation rule would change.
- **A policy handed the live entry could corrupt the conversation silently.**
  Writing `parts` and then failing leaves a summary that projects as a synthetic
  user message onto an unchanged path: the model is told "here is a summary" with
  nothing summarized. The policy now receives a deep copy and returns it on the
  plan.

Also added: usage had no route back from the policy (it owns the LLM call, so it
is the only thing that sees `response.usage`) — `CompactionPlan.usage` now
carries it, and usage is recorded for the *attempt* so a failed compaction still
accounts for the tokens it spent. `llm_config` cannot be stamped by the runner
(the policy picks the model), so `CompactionStarted` carries `None`.
`context_tokens` must be recalculated when `parts` land or the summary counts
zero toward the gauge. The archived conversation must be explicitly set to
`IDLE`, since `_drive` sets `RUNNING` on the way in and archived conversations
are never re-derived. `_build_run_result` had two false assumptions. The
"no open turn" precondition is a real check, not a natural consequence.
`SessionLedger` needed three doors the spec never mentioned. Collateral the
spec missed: `utils.py`'s `pretty_print`, `ContextManager._media_parts`,
`AgentSessionRunner.__eq__`, and the fact that G1 (atomic writes) is
contrib-owned because core owns no persistence.

Sections added: ledger doors, a failure taxonomy, public surface, a core/contrib
test split with a `FakeCompactionPolicy` in `scenarios.py`, a docs plan (the
repo's own rules require one and the draft had none), and a decisions log.

Eight open questions were resolved with the user, notably: `Entry.id` and
`Entry.created_at` both become optional and every existing `id=""` placeholder
site migrates including the public `ToolRegistry.create_execution` contract
(one convention, not two); and failures split by source — a user-scheduled
compaction raises, a policy-initiated one degrades and lets the turn proceed,
because compaction is an optimization and must not cost the user their turn.

# 2026-07-25 — review pass over both documents

A second read of the PRD and the implementation proposal against the code, at
the user's request: find loose ends, change nothing that is already sound. Five
defects were found that do not depend on a policy misbehaving, plus a set of
stale statements. All are now written into the two documents; one question is
left open because it is the user's to settle.

Applied:

- **The policy deadline was unusable.** §5.4 passed
  `client_completion_timeout_in_ms` straight to `asyncio.timeout`. The field is
  int milliseconds, `asyncio.timeout` takes float seconds, and the default is
  the `Inf` sentinel — the integer `-1` — so every compaction on a default
  configuration would have expired instantly, before the policy's call went
  anywhere. Now converted through `_ms_to_seconds` exactly like the LLM step,
  with an explicit "no deadline" branch for `Inf`.
- **The usage write was unguarded**, sitting between the two `try` blocks and
  inside neither. `record_usage` can raise (`Usage` is `extra="forbid"`; it also
  verifies the entry is on the active path), and an escape there left the
  bracket **open** — the resume state — so the next drive would replay the same
  failing policy call forever, with `should_compact` never consulted. Two
  changes: `plan.usage` becomes a typed `UsageCounters` so a wrong counter name
  fails inside the policy (which retires the "unknown usage key" rejection rule
  the PRD listed, and closes the question the PRD had left open); and the write
  moves inside the failure handling, after a G2-only `check_snapshot` so a
  policy that replaced the active conversation still gets G2's error rather than
  the ledger's.
- **`start()` could wedge the runner permanently.** `_begin_run` takes the
  one-run guard, and the proposal then called policy code (`should_compact`)
  that decision 9 lets propagate — with no release, since the background task is
  never created. One `TypeError` in a policy made every later `run()`/`start()`
  raise "another run is already active", unrecoverably. Now released and
  re-raised. The hole predates compaction (middleware inside
  `_ensure_open_turn`), so the fix stands on its own.
- **A cancel with a non-default outcome did not stop the drive.** The check was
  `_closed_outcome == CANCELLED`, but `cancel()` takes its outcome as an
  argument and only forbids `COMPLETED`, so `cancel(ERRORED)` consumed the
  cancellation and then drove the conversational turn anyway. Now a flag set by
  the step, with a regression test that fails only on the value-comparison
  version.
- **Two hazard descriptions understated their consequences**, and the
  compaction-bracket predicate was hardened from "the span contains a
  `CompactionEntry`" to "the node immediately after the `TurnStart` is one".
  The entry outlives its bracket by design and lands where the policy puts it,
  so the weaker test coupled a framework-internal classification to policy
  behavior. A correct policy never triggers it — this is hardening, not a bug
  fix — but the failure it prevents (re-running an already-committed compaction
  and overwriting its `compacted_nodes`) is worse than the "malformed but not
  fatal" the document claimed. The same predicate now also serves `turn_count`,
  replacing the "brackets containing an `AssistantMessage`" rule.
- **Missing pass-throughs**: neither `PluginAgentSessionRunner` — which is how
  the TUI builds its runner — nor the `DeterministicRunner` test double accepts
  `compaction_policy=`. Without them the planned `/compact` command cannot be
  wired and none of the integration tests can be written. Pass-through only;
  plugin *composition* of a policy stays out of scope.
- **Stale statements corrected.** Both documents said the suite was red because
  of seven untracked files under `tests/agent/contrib/compaction/`; that
  directory no longer exists and the suite is green (`910 passed, 14 skipped`,
  no ignores), so work-sequence step 0 was a no-op. The real leftover is an
  empty `luca/agent/contrib/compaction/`. Also: `metadata["source_conversation_id"]`
  was described as surviving, but `metadata` is policy-owned and nothing writes
  it; and nothing re-set `RUNNING` after a committed transition, so the turn
  that followed drove a conversation reporting the derived status.

Left open at the end of that pass, and resolved immediately after (below): the
full-carry plan.

One correction to the review itself, recorded so the reasoning is not trusted
twice: the bracket-predicate issue was first reported as the most critical
finding. It is not — reaching it requires the policy to carry a turn marker
without its pair, which the documents already list as a hazard. The user caught
the overstatement; the finding was downgraded to hardening plus two corrected
hazard descriptions.

# 2026-07-25 — G5 removed: core stops judging whether a compaction was worth it

Prompted by the user, on the principle that the runner should be dumb and
everything delegable should go to the policy: *"We have to implement this
assuming policies will be correctly written. If someone wants to run another
compaction iteration in a current compaction, so be it."*

**G5 — "refuse to summarize a summary" — is deleted**, along with its
implementation `_is_noop`. The argument for removing it is the design's own
rule, stated in §13 of the PRD and §5.5 of the proposal: *the runner validates
structure, never meaning.* G5 was the single violation. Every other rejection
asks whether a plan is well-formed; G5 asked whether the compaction was
**worthwhile** — after `should_compact` had already said yes, after the policy
had paid for an LLM call, and after the plan had passed every structural check —
then discarded it and reported the result as indistinguishable from "nothing to
do". The row directly below it in §13's table already assigned the identical
judgment ("the span is too small to be worth summarizing") to the policy, so the
two owners were inconsistent. It was unreliable besides: a span of
`[ts, cmp, tf]` — the likely real shape of "just a summary" — never triggered it,
because bracket markers are entries too.

Consequences, all recorded in both documents:

- `None` from `compact()` is now the **only** "nothing to do" signal.
- A plan that re-summarizes a summary commits. So does one whose summary is
  larger than the span it replaced.
- **The full-carry question is dissolved rather than decided.** It only existed
  because `all()` over an empty compacted span is vacuously `True`, so `_is_noop`
  would have discarded a plan that decision 10 and its named test require to
  commit. With the guard gone, a full carry commits with `compacted_nodes == []`,
  and `[]` vs `None` keeps the meaning the field's type exists for. Decision 10
  stands unchanged; its test stands unchanged.
- **G4 is now the only bound core places on repeated compaction** — one per
  drive, structural, because the step sits outside the loop. Everything past that
  is `should_compact`'s to decline.
- The `context_tokens` recalculation when `parts` land (§4) stops being merely
  tidy and becomes load-bearing: it is what lets a correct policy's gauge fall
  after a commit, and core no longer second-guesses a gauge that doesn't.
- **The floor moves to the default contrib policy**, where §13 already said it
  belonged, and is now written into the contrib follow-ups as a requirement: a
  minimum span worth summarizing, a refusal to re-summarize a bare previous
  summary, and a gauge that falls. Its own tests are the only thing pinning it.
- New INV-13 for policy authors: a structurally valid plan is always committed.

Accepted cost, stated in the PRD: a policy whose gauge never comes down burns
one LLM call per drive and grows `entries` and `conversation_history` a little
each time. That is waste, not damage — nothing is deleted, every archived
conversation stays intact and recoverable — and the fix is policy-shaped.

Both documents now carry **no open questions.**

# 2026-07-25 — second review pass: one design addition, four corrections

Another read of both documents against the code, at the user's request, with an
explicit instruction not to confuse runner responsibilities with the policy's:
the runner must keep a non-corrupt state, but a buggy policy is allowed to
produce a broken conversation. Everything the documents claim about existing
code was re-verified (`runner.py`, `ledger.py`, `models.py`, `projection.py`,
`context_manager.py`, `contrib/plugins/runner.py`, `contrib/tui/sessions.py`,
and the cited test lines) and holds — with one exception, below.

The design addition:

- **A cancelled compaction was putting a false statement on the wire.**
  `project_turn_finish` (`projection.py:288`) projects
  `"[Request interrupted by user]"` for *any* `TurnFinish(CANCELLED)`, and a
  cancelled compaction never transitions, so its `tf_c` stays on the **active**
  path forever. Every later request then told the model its question had been
  interrupted — a question the model was never shown, about a turn that never
  started. Silent, durable, and reachable on the plainest cancel path the spec
  already tests. Not a policy hazard: core writes that marker.
  The rule lives on `ConversationProjector.project()`. It cannot live on
  `project_turn_finish` — classifying a bracket needs the path, and the
  per-entry methods deliberately do not receive it — so putting it there would
  have meant changing a public override signature for a fact the caller already
  holds. PRD §4 / decision 10, proposal §7 / decision 21.

  **The user simplified the rule, and the simpler one is better.** It was first
  written as two rules: the entry projects iff it has `parts`, plus a special
  case suppressing the bracket's `TurnFinish`. The user's version is one
  positional rule — *a compaction bracket projects as nothing, whatever the
  outcome; a `CompactionEntry` outside any bracket projects its `parts`* —
  which absorbs the `TurnFinish` case rather than special-casing it, since the
  marker is inside the skipped span. It is also less code, because `project()`
  has to track bracket membership either way. Discarding the span loses
  nothing: `post_message` raises while a compaction bracket is open, so only
  markers can ever be inside one.

  It corrects a second thing the two-rule version got wrong, which had been
  written into the PRD as deliberate: an **archived** conversation was left
  projecting its summary. But the archived path still holds every original
  entry, so projecting it emitted the full history *and* a summary of that same
  history. A summary only means anything where the history it replaces is gone
  — the new conversation. The positional rule gets that right for free; the
  content test could not.

The corrections:

- **`_closed_outcome` does not exist.** The proposal treated it as existing
  runner machinery ("`_close_turn` — existing writer; sets `_closed_outcome`");
  `grep` finds it nowhere in `luca/` or `tests/`. It has to be built, and it is
  **not** compaction surface: once `_build_run_result` stops reading
  `nodes[-1].outcome`, every ordinary turn's outcome depends on it. New §5.1a
  specifies the field, its single writer (`_close_turn`, which all four
  existing close paths funnel through), the compaction-specific write in
  `_commit`, and the per-run reset in `_begin_run` — without which a re-driven
  handle reports the previous bracket's outcome.
- **PRD §11's prose headers had drifted two steps** from the renumbered
  pipeline: "the only corruption window" pointed at plan validation rather than
  the transition, and "memory and disk may diverge" at preparation rather than
  the persist. An implementer reading the headers would have protected the
  wrong region.
- **`turn_count` was specified twice, differently.** Proposal §2 said "brackets
  that do not *contain* a `CompactionEntry`"; §4.3 and decisions 3 and 19 say
  adjacency, and §4.3 spends a paragraph arguing "contains" is wrong. §2 now
  uses the one predicate — which also gained a home (`models.py`, upstream of
  its four consumers; `projection.py` does not import `ledger.py`, so putting
  it on the ledger would have forked the definition or added that edge).
- **The drive's post-step `IDLE` return was unconditional**, so it also ran on
  drives with no compaction policy, an open conversational turn, or a `False`
  `should_compact`. `_begin_run` gates on the cached status while that check
  reads the derived one, so any drift between them would have ended an ordinary
  drive silently. Now gated on the step having actually run.

# 2026-07-25 — the policy is handed the path it may rewrite

The phantom-bracket hazard was left with the policy in the pass above. The user
reopened it on different grounds: **not robustness, API quality.** A policy
author should not have to know that the live path ends with the compaction's
own `TurnStart` and entry, nor that one of those two must never be carried
back. They proposed passing the target conversation to `compact()` explicitly,
with the bracket nodes stripped.

Adopted, with two changes, and it turned out to be more than ergonomics.

**`compact()` becomes `(session, nodes, entry)`.** `nodes` is the active path
minus this compaction's `TurnStart` — the path the policy may rewrite. Plans
are validated against that view, so §13's existing rule ("references an id not
on the current path") now catches a carried `ts_c` with **no new rule and no
new judgment**: the runner is checking its own offer, not the plan's wisdom.
The likeliest accidental form of the hazard — a policy slicing the tail of the
path and sweeping up the marker — stops being reachable, and arrives as a
`CompactionPlanError` the author can act on.

The two changes to the user's sketch:

- **Strip `ts_c`, keep `cmp`.** The proposal was to remove both. Keeping the
  entry makes the identity transform legal: `plan.nodes = list(nodes)` is a
  valid full carry. Stripping it would make that same line fail rule 7 ("plan
  omits the compaction entry"), trading one piece of invisible trivia for
  another.
- **A `tuple`, not a `Conversation`.** A `Conversation` with filtered nodes is
  an identity-bearing object claiming to be the live one while not being it —
  and G2 compares plans against the real one. `ConversationSnapshot` now
  carries both paths (`nodes` for G2, `offered` for rule 5 and for the
  `compact()` argument), so the strip happens in exactly one place.

**G6's resume test moves from bracket shape to `entry.parts is None`.** The
`nodes` view removes the likeliest way to counterfeit an open compaction
bracket, but not all of them — a policy can still carry an unrelated
`TurnStart` or invent one. `parts` are written only at the commit point, so an
entry that has them describes a compaction that already finished; the runner
now refuses to resume one. That closes the damage from every remaining variant
without core inspecting a plan's meaning, and it is arguably just G6 ("a
finished one does not") tested against what happened rather than against
shape. The residue is the ordinary phantom open turn — loud, and the policy's.

Recorded as PRD decisions 11 and 12, proposal decisions 23 and 24. Superseded:
the previous pass's note that the `ts_c` guard was considered and declined —
the version adopted here is not that guard, and it is not a guard at all.

A final consistency pass moved G6's content test from the compaction step into
`SessionLedger.open_compaction_entry()`. Same rule, one place instead of two —
and it closes a hole the step-local version left: `schedule_compaction()` also
calls that read, so with the test in the step it would find the committed
entry, report itself idempotent, return its id, and then the drive would refuse
to run it. A call that silently does nothing. With the test in the read, the
open-turn guard fires and it raises like any other open bracket.
