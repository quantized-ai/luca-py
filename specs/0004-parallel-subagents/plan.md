# Implementation plan — parallel subagents V2

> **Status: IMPLEMENTED** (2026-08-01) — all of A, B and C, tests and docs
> included; full suite green. Deviations discovered during implementation are
> recorded in [history.md](history.md).

Implements the three features specified in [v2.md](v2.md) and [max_workers.md](max_workers.md):

- **A** — `subagents_max_depth > 1`: tests, stale-comment cleanup, TUI flag.
- **B** — `subagents_max_per_turn`: two-layer spawn budget + `ExecutionStatus.REFUSED`.
- **C** — `subagents_max_workers`: session-wide worker pool + three lifecycle events.

Both specs were written against `91f6be1`; this plan is against HEAD (`8b1134f`). Every
line reference below was verified against the current working tree. Features land in
order A → B → C; each is independently green (`uv run py.test tests/`, ruff clean).

---

## 0. Verified deltas vs the specs

Facts where the code at HEAD differs from what the specs assume. Everything not listed
here was verified as written (drift ≤ 3 lines).

| # | spec says | code at HEAD |
|---|---|---|
| D1 | `DefaultConversationProjector` | Class is `ConversationProjector` (`projection.py:118`). |
| D2 | "the TUI renders a three-level tree flat" (v2 §1) | Stale since `475edcf`: panels nest by construction — `_open_panel(child_id, source)` mounts into the spawner's panel (`app.py:374–380`), replay recurses (`app.py:518–522`), `child_links()` walks all conversations (`render.py:160–170`). **Depth > 1 needs zero TUI work.** Per the decision to keep the UI simple we leave the existing nesting as is (it is the zero-code option; flattening would be the change). |
| D3 | update `docs/agent/09-events.md` (max_workers §10) | No such file — 09 is plugins. Events are documented in `docs/agent/04-runner.md` §4 (block table, 111–121) and §5 (delta table, 154–158). |
| D4 | "`AGENTS.agent.md:453` says 1 is the only supported value" | The literal phrase lives in `docs/agent/08-runtime-config.md:83` and `docs/agent/13-subagents.md:49`. `AGENTS.agent.md` has the equivalent claims at `:418` ("V0 limits, and the implementation may rely on them") and `:454` (RuntimeConfig row). All four spots change. |
| D5 | `Inf` introduced by max_workers §4.1 | Already exists: `Inf = -1` at `models.py:962`, exported from `luca.agent.core`, used by every int limit field. `subagents_max_depth` is already in the `_inf_or_natural` validator list (`models.py:1034`). |
| D6 | `_ensure_driven` / `_redrive` sketched as new | Both exist (`runner.py:1350–1369`, `394–411`). **`_redrive` does not call `_begin_run` and the redriven drive is not re-registered in `_runs`** — so feature C's release/announce cannot live behind `_end_run`'s `_runs`-membership guard (see §3.4). |
| D7 | — | Pre-existing omission: `AGENTS.agent.md`'s two event lists (`:212` and the `events.py` layout comment `:90–92`) omit `SubagentsSpawned`. Fixed while we're there. |
| D8 | v2 lists the projector + TUI as REFUSED touch points | Also required, or `pretty_print` KeyErrors: `utils.py:72–82` `STATUS_LABELS` is a direct-lookup dict over **all** members (consulted at `utils.py:388`). `tui/render.py:27–37` has its own `STATUS_LABELS` (graceful fallback, but gets a label anyway). `tests/agent/test_models.py:793–805` asserts the full enum member dict. |

Decisions the user supplied on top of the specs:

- `--subagents-max-depth`, TUI default **3**. Verified meaning: the gate is
  `conversation.depth < max_depth`, so `3` allows spawning from depths 0–2 and the
  deepest subagent sits at depth 3 — **main + 3 extra levels**, exactly as intended.
  Library default stays `1` (v2 §1's recommendation).
- `--subagents-max-per-turn`, TUI default unset = `Inf`.
- Worker-pool TUI rendering: a small activity note on the existing `SubagentPanel`,
  nothing more (§3.7).

---

## 1. Feature A — `subagents_max_depth > 1`

Already works (v2 verified it empirically; every mechanism is depth-generic). The lift
is tests, comments, docs, and the flag.

### 1.1 Code

| change | where |
|---|---|
| Rewrite the "only supported value in V0" comment: depth is a real knob; `N` = main + N subagent levels; default 1 | `models.py:1008–1010` |
| `--subagents-max-depth N` flag (`type=int`, default `3`) writing `runtime_config.subagents_max_depth` at launch | `tui/cli.py` (`arg_parser` + `build_session`, beside the `subagents_enabled` write at `:154`) |

Flag-write semantics for all three new TUI flags: same policy as `--no-subagents` —
**written on every launch, fresh or resumed** ("the flag always describes the run you
are starting now", max_workers §4.1/§8, and `cli.py:151–154` precedent). Consequence
worth stating: resuming without flags rewrites these knobs to the TUI defaults
(depth → 3, per-turn → Inf, workers → Inf). See open question Q1.

### 1.2 Tests — `tests/agent/subagents/test_depth.py` (new)

Uses the existing scaffolding (`subagent_session(max_depth=…)`, `SubagentRegistry`,
`DeterministicRunner`, `faux` fixture, `spawn_call`, `faux_assistant_message` /
`faux_text` — note the v2 probe's `assistant()`/`text()` helpers don't exist; the faux
builders do).

| test | asserts |
|---|---|
| grandchild end-to-end | `max_depth=2`, main → child → grandchild; depths `[0, 1, 2]`; results flow leaf → parent (`<task id=…>` in the parent's last request); `runner.idle()` |
| spawn tool withheld at the leaf | `max_depth=2`: the depth-2 conversation's request carries neither the spawn spec nor the spawn prompt part |
| `_verify_gate` at depth 2 | a registry leaking a spawn spec to the leaf raises `AgentError` |
| cancel cascades three levels | cancel main mid-tree; all three conversations close with `CANCELLED`; links resolve with the cancellation result |
| reload mid-tree | serialize with an unresolved grandchild, reload, `run()` finishes the tree (pattern of `test_runner_lifecycle.py:502` / `test_parallel_and_control.py:623`) |
| approval gate on a grandchild | grandchild defers → `pending_approvals()` names it; answer + rerun (or `notify`) unblocks child then main |

### 1.3 Docs

`models.py` comment (above); `AGENTS.agent.md:418` + `:454`;
`docs/agent/08-runtime-config.md:83`; `docs/agent/13-subagents.md:49` + `:170`;
root `AGENTS.md` demo-flag block (`:89–97`) gains the flag;
`docs/agent/contrib/tui/README.md` §1 flag list.

---

## 2. Feature B — `subagents_max_per_turn` + `ExecutionStatus.REFUSED`

Exactly the v2 §2 design: layer 1 withholds the tool + prompt via the gate; layer 2
births the overflow execution terminal. Additions below are the concrete placements.

### 2.1 Data model (`models.py`)

- `RuntimeConfig.subagents_max_per_turn: int = Inf` in the subagents block
  (after `:1011`). Validation: **`Inf` or `>= 1`** — `0` is rejected with the same
  argument max_workers §4.1 makes for the pool ("no spawns ever" is
  `subagents_enabled=False` spelled incorrectly). New shared validator
  `_inf_or_positive` (feature C's `subagents_max_workers` joins it), beside
  `_inf_or_natural` (`:1025–1040`).
- `ExecutionStatus.REFUSED = "refused"` — comment: *a framework runtime limit refused
  the call before dispatch*. Terminality is expressed everywhere as
  `not in (PENDING, RUNNING)`, so REFUSED is terminal at every consulting site by
  construction (verified: no named TERMINAL set exists).

### 2.2 Public helpers (core)

- Rename `_declares_spawn` → `declares_spawn`, `_spawn_payload` → `spawn_payload`
  (module-level in `runner.py:3632/:3645`; keep `SPAWN_MARKER`/`SPAWN_REQUIRED_KEYS`
  private).
- New module-level `spawns_committed(session, conversation_id, *, exclude=None)` in
  `runner.py`, exactly the v2 counter: spawn-declaring only; PENDING/RUNNING count as
  one reserved; settled counts iff `spawn_payload(execution)` is not None; `exclude`
  drops one id. Reads `open_turn_executions(conversation.nodes, session.entries)`
  from `models.py:806` — a pure read, rebuildable after reload.
- Export from `luca.agent.core.__init__`: `declares_spawn`, `spawn_payload`,
  `spawns_committed`, `open_turn_executions`.
- Contrib cleanup: `contrib/subagents/plugin.py:103–105` drops its private copy and
  imports `declares_spawn`; `contrib/tui/render.py:134`'s inline copy in
  `is_runtime_plumbing` switches to the public helper too (third copy dies).

### 2.3 The gate — both predicates, changed together

`spawn_gate_open` (`contrib/subagents/plugin.py:46`) and `_spawn_gate_open`
(`runner.py:1738`) both gain, after the depth clause:

```python
limit = config.subagents_max_per_turn
if limit != Inf and spawns_committed(session, conversation_id, exclude=exclude) >= limit:
    return False
```

(The `!= Inf` guard matters: `Inf` is `-1`, so a naive `committed < limit` would
withhold everything by default.) Both signatures gain `*, exclude: str | None = None`
and stay identical, per v2. `_spawn_one` (`runner.py:2625–2630`) passes
`exclude=execution.id` — the self-counting trap's fix, v2 §2. `_verify_gate`'s error
message (`runner.py:1749–1768`) additionally names the budget so S14 failures are
diagnosable.

### 2.4 Layer 2 — born REFUSED (`_create_executions`, `runner.py:3076–3113`)

Per v2: `refusal = self._spawn_budget_refusal(conversation_id, draft)` computed inside
the append loop (sequential by construction — earlier appends are visible as
reservations), threaded into the existing `build` closure:

```python
if _r is not None:
    update |= {"status": ExecutionStatus.REFUSED, "error": _r}
```

with `ended_at` keyed off the effective status (the closure already stamps
`ended_at` for non-PENDING births). The existing terminal-at-birth branch
(`:3110–3112` → `_finalize_undispatched`) then emits the one `ToolExecuted` — no new
event plumbing. `_spawn_budget_refusal` returns the v2 `ToolExecutionError`
(`error_type="SpawnLimitReached"`, message "Spawn limit reached (N/M subagents this
turn). Complete the remaining work yourself; do not retry.", details `{limit,
committed}`). The second birth site (`_invoke_runtime_tool` → `_birth_draft`,
result-tool dispatch) needs nothing: the result tool declares no spawn.

Note: `ToolExecution.approval_status` stays `None` on a REFUSED birth — the documented
born-terminal precedent (`models.py:506`). `before_tool_execution` middleware still
observes it via `_finalize_undispatched` (`runner.py:1654`).

### 2.5 REFUSED rendering

- `projection.py` `_derived_failure_text` (`:492–514`): new branch before the dict
  fallthrough — `if entry.status == ExecutionStatus.REFUSED and error is not None:
  return error.error_message` — plus a `REFUSED: "[tool execution refused]"` row in
  `STATUS_ONLY_OUTPUTS` (`:133–138`) as the no-error fallback and subclass override
  point. The model reads the real numbers, not a placeholder.
- `utils.py:72–82` `STATUS_LABELS` gains `REFUSED: "refused"` (direct-lookup dict —
  KeyError in `pretty_print` otherwise).
- `tui/render.py:27–37` `STATUS_LABELS` gains a `"refused"` label.

### 2.6 Tests — `tests/agent/subagents/test_spawn_budget.py` (new)

One test per v2 scenario, S1–S15 (S10 is the self-counting regression: the Nth
legitimate spawn converts without raising). Test-design notes:

- S3 (mid-message overflow): one assistant message with two `spawn_call`s at
  `subagents_max_per_turn=1` — exactly one child exists, the second execution is
  `REFUSED` with the `SpawnLimitReached` error, and the next faux request's projected
  tool message carries the refusal text (assert via `faux.requests`).
- S2 (withheld next step): assert the spec absent from the request's tools **and** the
  spawn prompt part absent from its system prompt.
- S5/S6/S7 (non-spawning / rejected / failed spawn calls don't consume budget): use a
  spawn tool double returning `is_subagent_spawn=False`, a scripted `PENDING→DENY`
  decision, and a raising spawn tool respectively.
- S8 reload mid-turn rebuilds the count (serialize between steps, resume, budget still
  enforced); S9 new turn resets; S11 a depth-1 child has its own budget
  (`max_depth=2`); S12 default `Inf` changes nothing (byte-equal session vs today).

Plus: `tests/agent/test_models.py:793–805` (enum members dict) updated; a REFUSED
projection test beside `test_projection.py:724`; a validator test (0 raises, `-1`/`1`
accepted).

### 2.7 Docs

`AGENTS.agent.md`: status list `:154`, subagents section `:404` (the enforcement
story: budget overflow is a born-REFUSED execution, not an `AgentError`), RuntimeConfig
table row after `:454`. `docs/agent/08-runtime-config.md` subagents table (`:80–85`).
`docs/agent/13-subagents.md` §1 config table + §2 sequence + §8 (`:227–229`).
`docs/agent/02-data-model.md` `:194–207` lifecycle table + `:218–221` dispatched
table. `docs/agent/10-projection.md` `:76–81` + `:89–92`. `docs/agent/03-tools.md`
`:203–215` one-output table + `:217`. `docs/agent/04-runner.md` `:115`, dispatch table
`:220–236`. `docs/agent/07-middleware.md` `:124`, `:251–252`, `:259–262` (three
embedded enumerations — llm.txt says embedded source excerpts must match
`middleware.py`, so update the docstrings there too if they enumerate statuses).
`docs/agent/contrib/subagents/README.md` `:37–38`, `:54–55` (the embedded
`spawn_gate_open` source excerpt must be re-synced), `contrib/README.md:26` ("gated by
depth" → depth + budget). TUI flag docs as in §1.3.

---

## 3. Feature C — `subagents_max_workers` + lifecycle events

The max_workers spec is the design; this section maps it onto the real code and
resolves the five places where the spec's pseudocode meets a sharper reality.

### 3.1 Config (`models.py`)

`subagents_max_workers: int = Inf` in the subagents block, validated by the shared
`_inf_or_positive` from §2.1 (`Inf` or `>= 1`; `0` raises).

### 3.2 Events (`events.py`)

`SubagentStarted`, `SubagentPaused`, `SubagentFinished(outcome: TurnOutcome)` exactly
as max_workers §11.4 (docstrings included), added to the `AgentEvent` union
(`:241–259`). `TurnOutcome` is already imported. No `__init__` export churn — events
are deliberately not re-exported from `luca.agent.core`.

### 3.3 `AgentRun` — deferred start

- `__init__(..., defer_start: bool = False)`; the eager bundle (`:309–329`) moves
  verbatim into `_start_eager()` (loop resolve → `_begin_run` →
  `_open_bracket_for_start` with the `_end_run` unwind → `create_task(_consume())`),
  plus `self._wake.set()` at the end and a no-op guard when `_task is not None`.
  `defer_start` defers **the whole bundle** — a queued child is not in `_runs`, has
  no `TurnStart`, and holds only its seed message (this is what makes max_workers
  §5.6's durable-state story true, and what T2 asserts).
- **`_task is None` tolerance** (the spec's §4.11, made concrete — today these crash
  with `AttributeError`/`TypeError`):
  - `_next_buffered` (`:741–757`): treat `_task is None` as "not done" and fall
    through to the `_wake` wait (`_start_eager` sets `_wake`, so admission wakes the
    consumer).
  - `_join` (`:759`) / `_finalize_eager` (`:771`): wait for admission first —
    loop on `_wake` until `_task is not None` (or the handle is abandoned) — then
    await the task. `await run.child(queued)` therefore blocks until admission (T16).
- **`_live_children` (`:659–660`) must count a queued child** — today it requires a
  live task, so a parent whose own engine is exhausted would `StopAsyncIteration`
  with a queued child still pending and the parent's stream would end early. New
  term: `child._eager and child._task is None and not child._abandoned`.
- **`cancel_drive()` (`:426–434`)**: on a queued handle (`_eager and _task is None`)
  set a new `_abandoned = True` flag so the pool's staleness check drops its waiter
  (the lazy `__aexit__` suspension cascade at `:530–531` reaches it); and cascade
  recursively to `self._children.values()` so a suspended subtree's queued
  grandchildren are abandoned too. (The recursion is a small behavioral extension of
  cancel_drive — today it stops at the handle's own task — but it is what "suspension
  cascade" already claims to do, and cancelling an already-done task is a no-op.)
- **`_redrive()` (`:394–411`)**: guard `if self._task is None: return` — a
  never-started handle belongs to the pool; without this, `_ensure_driven` could
  start a drive with no token and no bracket (verified: `_redrive` skips `_begin_run`
  entirely).

### 3.4 Runner — pool state and operations

New runtime-only state on `AgentSessionRunner.__init__` (beside `_wakes`/`_recheck`):
`_working: set[str]`, `_waiters: list[_SlotWaiter]`,
`_admission_errors: dict[str, BaseException]`. Module-level:

```python
@dataclass
class _SlotWaiter:
    conversation_id: str
    grant: Callable[[], None]
    announce: AgentRun | None = None   # publish SubagentStarted on grant
    handle: AgentRun | None = None     # staleness: dropped when abandoned
```

Operations per max_workers §4.3 (`_slot_limit`, `_needs_slot`, `_request_slot`,
`_grant`, `_release_slot`, `_pump`, `_still_wants_slot`, `_drop_waiter`,
`_acquire_slot`), with these hardenings on top of the spec's pseudocode:

- `_still_wants_slot` additionally drops a waiter when: the conversation already has
  a live drive (`self._runs.get(cid) is not None` — the double-start half of T17;
  `_begin_run`'s raise stays as the loud backstop), or `waiter.handle._abandoned`.
  The spec's three checks (missing conversation, derived IDLE, unconsumed
  `CancelRequested` via `ledger.open_turn_cancel_requested`, `ledger.py:452`) stay.
- `_grant` wraps `waiter.grant()` per §4.14: on `BaseException`, discard the working
  entry, record in `_admission_errors[cid]`, and `_ensure_driven(parent)` — the
  parent id read from `waiter.handle._parent`. The drive loop's top (step 0 area)
  pops any admission error recorded against one of its unresolved children and
  re-raises it, preserving today's fail-loud contract for the deferred-start case
  (T18).
- **`_end_run` releases and announces unconditionally** — this is delta D6. A
  `_redrive`n drive is not in `_runs`, yet `_consume`'s `finally` (`:721`) always
  calls `_end_run`; putting release/announce behind the `_runs`-membership guard
  would leak its slot and swallow its Paused/Finished. Shape:

  ```python
  def _end_run(self, run):
      if self._runs.get(run.conversation_id) is run:
          del self._runs[run.conversation_id]
      self._publish_run_ended(run)          # announce BEFORE releasing (§4.8)
      self._release_slot(run.conversation_id)
  ```

  `_publish_run_ended` is guarded per-drive (a `run._ended_published` flag, reset in
  `_start_eager`/`_redrive`) so the extra `_end_run` call sites (`__init__` unwind
  `:327`, lazy paths `:526/:581/:587/:598`) cannot double-publish; the `_task is
  None` / not-framework-owned early returns from §11.5 handle the rest. Releasing is
  idempotent by construction (`set.discard`).
- `_ensure_driven` (`:1350–1369`): the `run._redrive()` arm becomes
  `self._request_slot(cid, run._redrive, announce=run)`, with a
  `if run._task is None: return` guard before it (a queued handle's start belongs to
  the pool; granting `_redrive` to it would no-op and leak the slot). The
  wake-a-live-drive arm is untouched.

### 3.5 Runner — the five slot transitions wired in

| transition | where | change |
|---|---|---|
| acquire at admission | `_start_children` (`:2677–2697`) | construct with `defer_start=True`; `if run.autostart_subagents: self._request_slot(child.conversation_id, child._start_eager)`; return started ids (`child._task is not None`). Drive loop (`:2311–2326`) yields `SubagentsSpawned` then one engine-yielded `SubagentStarted` per started id — engine, not inbox, so `Started` cannot overtake the announcement (§11.5's ordering argument, verified: `_next_own_or_forwarded` drains the inbox first, `:616–618`) |
| release on park / re-acquire on wake | `_await_subtree` (`:2853–2880`) | `_release_slot` before the wait, `_acquire_slot` raced against the token in the `finally`, exactly §4.7. The main conversation passes through both as no-ops (`_needs_slot` False at depth 0) |
| release at wind-down | `_wind_down_async` (`:2915`) | `self._release_slot(conversation_id)` first line (§4.9) |
| release at drive end | `_end_run` | §3.4 above |
| restart admission | `_restart_unresolved_children` (`:1294–1323`) | construct with `defer_start=True`, then `_request_slot(child_id, child._start_eager, announce=child)` — a resume stampede is now paced by the pool, and each admission announces `SubagentStarted` through the parent's inbox (§4.10) |

`_flush_cancelled_children` (`:2953`) needs no change: a queued child is not in
`_runs`, so the parent's drive already winds it down; `_still_wants_slot`'s
cancel check guarantees the pool never grants it first (§5.4).

### 3.6 Runner — publication and the `run()` refusal

- `_publish_subagent_event(run, event)`: `run._parent._inbox.put_nowait(event)` —
  the synchronous side channel, exactly §11.5 (precedent: `_publish_approval`,
  `:2906`). `_publish_run_ended(run)`: §11.5 verbatim — `_task is None` → return;
  not `_framework_owned` or no parent → return; derived status `IDLE` →
  `SubagentFinished(outcome=self._closed_outcomes.get(cid) or COMPLETED)`
  (`_closed_outcomes` verified at `:903`, written by `_close_turn` at `:3597`);
  else `SubagentPaused`.
- `run()` (`:1200`) raises `AgentError` when `autostart_subagents=False` and
  `subagents_max_workers != Inf`, with the spec's message (§4.12). `start()`
  untouched.

### 3.7 TUI

- `cli.py`: `--subagents-max-workers N` (default `Inf`) and the two flags from §§1–2;
  all three written unconditionally in `build_session` beside `subagents_enabled`.
- `app.py`: import the three events; add match arms routing by
  `self._panels.get(event.conversation_id)` (the tool-cell lookup pattern,
  `:339–341`): `SubagentStarted` → subtitle `running…`; `SubagentPaused` → subtitle
  `waiting…`; `SubagentFinished` → no-op (terminal state is owned by the existing
  `_sync_panels` settle path, which reads the resolved link's `is_error`).
- `cells.py`: `SubagentPanel` initial subtitle becomes `waiting…` (honest for a
  queued child; uncapped it flips to `running…` on the immediately-following
  `Started`), plus a tiny `set_activity(text)` used by the arms. That is the whole
  "active/paused note" — no new widget.
- Faux demo: untouched. The FIFO faux transport can script only one subagent
  (`wiring.py:196–203`), so a capped-pool demo is not scriptable; nothing in the
  demo needs the new flags.
- `RuntimeSettings` (tui `config.py:51–64`, the luca.json mirror) deliberately does
  **not** gain the new knobs — it doesn't carry `subagents_enabled` today either;
  the flags are the TUI surface.

### 3.8 Tests — `tests/agent/subagents/test_workers.py` (new)

T1–T27 from max_workers §9, one test each. Design notes for the faux provider's FIFO
constraint (concurrent children race for the next scripted response):

- Capped tests at `cap=1` are fully deterministic (serialized admission = FIFO).
- Tests at `cap≥2` script **identical** responses for all sibling children so the
  race doesn't matter, and assert structure (TurnStart counts at the
  `SubagentsSpawned` yield, event multiset per conversation, final session shape) —
  the style the suite already uses for parallel children.
- T13 (lowering the cap mid-session) and T12 (reload) follow the
  serialize/reload/resume pattern of `test_parallel_and_control.py:623`.
- T18 uses a `before_entry_written` middleware raising on the third child's
  `TurnStart` — asserts the **parent's** run raises rather than hangs.
- T19 asserts byte-identical request payloads (system prompt + tools) with and
  without a cap (`faux.requests`).

Existing-test impact (audited): **no test asserts a full literal event list containing
`SubagentsSpawned`**, so the always-on `Started`/`Finished` events break no literal
asserts. The set-equality tests (`test_parallel_and_control.py:118/:334`,
`test_integration_full_stack.py:354/:424`) stay green: lifecycle events carry child
ids already in the expected sets, and under `autostart_subagents=False` none are
emitted (§11.7). The ~30 no-subagent literal-event suites never spawn, so nothing
fires. TUI panel tests (`tests/agent/contrib/tui/test_subagents.py`) update for the
`waiting…` initial subtitle and gain a Started/Paused arm test.

### 3.9 Docs

`docs/agent/04-runner.md`: three rows in the block-events table (after `:120`, whose
own wording "announced before they start" now points at `SubagentStarted`), the
`autostart_subagents` table `:375–381` ("when a child starts" → when the pool admits
it), §12 prose. `docs/agent/08-runtime-config.md`: `subagents_max_workers` row +
sizing guidance (fan-out, never derived from depth; 20–30 typical; §8 of the spec).
`docs/agent/13-subagents.md`: new "How many run at once" section after §3 (renumber
§§4–8 → 5–9, fix the `Next:` chain and the cross-references listed in the docs audit:
`04-runner.md:120,363`, `08-runtime-config.md:78`, `02-data-model.md:452`,
`contrib/README.md:26`, `contrib/subagents/README.md:6`), plus the lifecycle-event
state machine (§11.2/§11.11's reference consumer). `AGENTS.agent.md`: events lists
(`:212`, `:90–92` — adding the missing `SubagentsSpawned` too, D7), subagents section
(new worker-pool paragraph), `### Change the agent loop` (`:560–563` park/release
description), RuntimeConfig row. TUI README §1 (flag) + §2.1 (`waiting…` subtitle).
Root `AGENTS.md` flag block.

---

## 4. Implementation order

| step | contents | gate |
|---|---|---|
| 1 | A: comment + flag + `test_depth.py` + docs | suite green |
| 2 | B data model: `REFUSED`, `subagents_max_per_turn`, `_inf_or_positive`, enum-consumer updates (projection, utils, tui labels, models test) | suite green |
| 3 | B helpers public + counter: `declares_spawn`/`spawn_payload` renames, `spawns_committed`, exports, contrib de-dup | suite green |
| 4 | B enforcement: both gates + `exclude`, `_spawn_one`, `_spawn_budget_refusal` + append loop, `test_spawn_budget.py`, docs | suite green |
| 5 | C events + `AgentRun` deferred start + `_task is None` tolerance (no pool yet — `defer_start` unused, pure refactor) | suite green |
| 6 | C pool: state, operations, five transitions, restart paths, `run()` refusal, admission errors | suite green |
| 7 | C publication: `_publish_run_ended`/`_publish_subagent_event`, engine yields, `test_workers.py` T1–T27 | suite green |
| 8 | C TUI (flags, arms, panel) + all C docs | suite green |

Estimates: A ≈ half a day; B ≈ half a day; C ≈ 1.5–2 days (tests dominate).

## 5. Risks the implementer must respect

1. **`Inf` is `-1`** — every new comparison must be `limit != Inf and …`, never a bare
   `<`. (Bites silently: the default would withhold all spawning.)
2. **`_end_run` outside the `_runs` guard** (D6) — the redriven-drive slot leak is the
   one bug the spec's pseudocode would ship.
3. **`_live_children` and the queued child** — without the new term, a parent's event
   stream ends while a queued child is pending; shows up as a hang or a truncated
   stream only under a cap, so it needs T2/T16 to cover it.
4. **The self-counting trap** (v2 §2) — `_spawn_one` must pass `exclude`; T-S10 is
   the regression test.
5. **`pytest` runs with `filterwarnings = error`** — any leaked waiter task or
   un-awaited future in the pool fails the suite as a `ResourceWarning`/
   `RuntimeWarning`; every `asyncio.ensure_future` in `_acquire_slot` must be
   `_cancel_quietly`'d exactly like `_await_subtree` does today (`:2874–2879`).
6. **Docs carry embedded source excerpts** (`spawn_gate_open` in
   `contrib/subagents/README.md:54`, middleware docstrings in `07-middleware.md`) —
   llm.txt's "code wins" rule makes these part of the change, not an afterthought.

## 6. Decisions on the two open questions

Resolved without user input (per instruction: simplest option, PRDs win):

- **Q1 — TUI flag semantics on resume: unconditional write.** All three new flags
  write on every launch, fresh or resumed — the `--no-subagents` policy and the
  max_workers spec's stated behavior ("the flag always describes the run you are
  starting now"). Consequence, documented: resuming without flags sets depth to 3
  and both caps to Inf.
- **Q2 — library default for `subagents_max_depth` stays `1`** (v2 §1's
  recommendation: nesting stays opt-in; the TUI opts into 3 via the flag).
