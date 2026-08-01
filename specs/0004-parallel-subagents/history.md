# 2026-08-01

The user's initial intent was: implement the three features specified in v2.md
(subagents_max_depth > 1, subagents_max_per_turn + REFUSED) and max_workers.md
(subagents_max_workers + the three lifecycle events), with three TUI details on
top (--subagents-max-depth default 3, --subagents-max-per-turn default unset =
Inf, a minimal active/waiting note on the subagent panel).

By exploring the codebase at HEAD (8b1134f) the specs held up almost exactly;
the deltas found and the decisions taken are recorded in plan.md §0 and §6
(TUI flags write unconditionally on every launch; library default for
max_depth stays 1 while the TUI opts into 3).

Everything was implemented in one pass — code, tests (test_depth.py,
test_spawn_budget.py, test_workers.py plus enum/validator/TUI/CLI updates)
and docs. Two places where the implementation deliberately deviates from the
max_workers spec's PSEUDOCODE (never from its design):

1. `_end_run` announces and releases OUTSIDE the `_runs`-membership guard: a
   `_redrive`n drive is never re-registered in `_runs`, so the spec's shape
   would leak that drive's slot and swallow its Paused/Finished. A per-drive
   `_ended_published` latch prevents double publication from the other
   `_end_run` call sites.
2. Spawn-batch children that stay QUEUED are announced by `_grant` on
   admission (`announce_when_queued`): the spec's §4.6 sets `announce=None`
   for the whole batch while its worked examples (§5.1, §5.3, T20) require
   later admissions to announce. Inline starts stay engine-yielded so
   `SubagentStarted` cannot overtake `SubagentsSpawned`.

Also fixed while in there: `cancel_drive()` now cascades recursively and
abandons queued (never-started) handles, so a suspended tree's queued children
cannot be started later by the pool; `_live_children` counts a queued child so
a parent's event stream cannot end while one is pending; the third copy of the
`_declares_spawn` predicate (contrib plugin, TUI render, test conftest) was
replaced by the now-public `declares_spawn`.

An adversarial review round (six lenses, per-finding refutation) then
confirmed 15 defects, all fixed the same day:

- **Critical — cancelling a queued child deadlocked the tree.** A queued
  child has no drive and no token, so `cancel()` woke nobody and the parent
  parked forever on a CANCELLING child. `cancel()` now abandons the queued
  handle (waking any joiner) and wakes the parent via `_ensure_driven` — it
  is the only conversation that flushes the child. Regression tests:
  `test_cancelling_a_queued_grandchild_wakes_the_flushing_parent`,
  `test_awaiting_a_queued_handle_survives_its_cancellation`.
- **Major — SubagentStarted could be overtaken by the child's own events**
  (Finished-before-Started, reproduced with 4 fast children and a consumer
  that awaits between pulls). The design changed: in framework mode
  `SubagentsSpawned` and every `SubagentStarted` are published onto the
  parent run's own inbox, synchronously and in order, before any child task
  first runs — the third deviation from max_workers.md §11.5, replacing
  the engine-yield of both events; the fan-in now also arms the inbox while
  the own engine is live and prefers it when a step completes. Regression:
  `test_started_precedes_the_childs_own_events_for_a_slow_consumer`. This
  also made T2's stream-replay concurrency assertion measure real
  admissions.
- Abandoned handles are consumed gracefully everywhere (`await` returns the
  session-derived `RunResult`, iteration ends, `async with` exits) instead
  of raising or hanging; `cancel_drive()` drops pending slot waiters so an
  answered gate's queued redrive cannot run after suspension.
- `subagents_max_depth=Inf` (-1) now means unlimited nesting in both gate
  predicates (it silently disabled spawning before).
- A budget-REFUSED spawn renders a `NoticeCell` in the TUI (live + replay) —
  it was invisible, since a refused spawn has no child panel.
- The TUI limit flags validate through `RuntimeConfig.model_validate`
  (`--subagents-max-workers 0` fails loudly instead of wedging the session
  and poisoning the saved file).
- Wind-down deliberately discards a cancelled child's recorded admission
  error instead of leaking it; the contrib README's gate excerpt was
  re-synced with the shipped `spawn_gate_open`.
