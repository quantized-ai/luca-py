# Implementation plan — mid-turn user messages

Companion to `prd.md` — all decisions D1–D13 are settled (see `history.md`
for how D13 and the D2 refinement landed); this plan maps them onto the code.

## 1. The new rule (D13)

A post targeting a conversation whose **open turn contains an unresolved
`ChildConversation`** — any conversation, main or subagent — is rejected with
`SubagentsActiveError`, exactly as CANCELLING is. Rationale: while subagents
run, the parent is mid-orchestration — its next LLM call may be minutes away
and the children cannot see parent messages, so accepting a "steering"
message would be an illusion of steering; rejecting is honest, and the
durable predicate (`open_turn_unresolved_children`) already exists and
survives reload.

Precedence inside `post_message` (first match wins):

1. unknown conversation → `AgentError` (via `ledger.conversation()`)
2. CANCELLING → the D3 exception
3. open **compaction** bracket → `AgentError` (D8; bracket-shape check, never status)
4. open turn with unresolved children → `SubagentsActiveError` (D13; every
   conversation, not just main)
5. non-main with **no open turn** and not BUSY → `AgentError` (finished subagent)
6. **archived predecessor** (any conversation some other conversation names in
   `previous_conversation_id`) → `AgentError` — see the matrix hole below
7. accept → normalize → `before_post_message` → `_append(target, …)` → id

Boundary note (document, don't fight): a post that lands in the small window
after a spawn tool call exists but before its `ChildConversation` link is
appended (or while a gated spawn awaits approval) is **accepted** — no child
is active yet at that instant. It sits before the spawn in history and is
answered after the children resolve, like any pre-spawn mid-turn message.

### A hole found in the PRD's D2 matrix (fix as part of this work)

D2 row "Non-main, IDLE → reject … archived conversations fall under this rule
too" assumes an archived conversation derives IDLE. Not always: a **queued
trailing `UserMessage` compacted behind** leaves the archived path
`[…, u, ts_c, cmp, tf_c]`, which derives **BUSY** (closed compaction brackets
are transparent). Under a pure status check that archived row would *accept* —
and wedge the message forever, since nothing ever drives an archived
conversation. Hence precondition step 6: a conversation that is someone's
`previous_conversation_id` always rejects. Cheap: one scan over
`session.conversations` values. Record this refinement in `prd.md` (D2 note).

## 2. Code changes

### `luca/agent/core/exceptions.py`

- `ConversationCancellingError(AgentError)` — the D3 exception (PRD-suggested
  name; distinct from `CancelledError`, `AlreadyCancellingError`,
  `asyncio.CancelledError`).
- `SubagentsActiveError(AgentError)` — the D13 exception. Both flat; no
  shared intermediate base.
- Export both from `luca/agent/core/__init__.py`.

### `luca/agent/core/runner.py`

**`post_message(content, conversation_id: str | None = None) -> str`**
(runner.py:1137). `None` resolves to `session.main_conversation_id` at call
time (explicit `is None` check — an empty string is an unknown id, not main).
Precondition per §1; then the existing normalize → `before_post_message`
(signature unchanged, D4) → `_append(target, …)` path. Full docstring rewrite:
acceptance matrix, D3/D13 exceptions, D5 guarantee, D6 burial note.

**Close-site check** (D5), in `_drive_loop`:

- Capture, immediately before `prepare_llm_call` (runner.py:2837, same sync
  region as the projection — a `before_llm_call` middleware that posts still
  counts as unseen):
  `seen = len(self.session.conversations[conversation_id].nodes)`
- In the final-answer close window (runner.py:2978–2987), between the existing
  cancel re-check and `_close_turn(COMPLETED)`:

  ```python
  cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
  if cancel_entry is not None:                       # 1. cancel wins (D5)
      events.extend(await self._wind_down_async(conversation_id, cancel_entry))
  elif self._has_unseen_user_message(conversation_id, seen):   # 2. loop again
      for event in events:
          yield event
      continue
  else:
      self._close_turn(conversation_id, TurnOutcome.COMPLETED)  # 3. close
  for event in events:
      yield event
  return
  ```

  `_has_unseen_user_message` is a small private helper:
  `any(isinstance(entries.get(n), UserMessage) for n in nodes[seen:])`.
  All sync, inside the existing no-yield window — D11's exactness holds.
  Applies identically to streaming and non-streaming, and to subagent drives
  (same loop). Failure closes (`hard_max_steps` ERRORED, LLM
  ERRORED/TIMED_OUT, wind-down) deliberately do NOT check — D6 burial.
  The extra round re-enters the loop top: `hard_max_steps` and the soft-max /
  doom-loop `tool_choice="none"` restriction apply to it unchanged.

**Docstring / comment updates in runner.py**: module docstring polling sketch
(~line 9: posting is also legal mid-run); `idle()` (~1120: no longer "the only
state post_message accepts"); `schedule_compaction` (~1191: rephrase "which
requires a closed bracket" to "which refuses an open compaction bracket");
`_snapshot_conversation` (~2233: still true — verify wording);
`_spawn_one` seed comment (~3060: "post_message is main-only" is gone; the
seed is the child's *first* user message; a live child accepts posts unless it
has active children of its own); `_resolve_children` (~3138: "a closed bracket
means finished" now holds because a finished child *rejects* input, not
because children never receive messages).

### `luca/agent/core/models.py` (docstrings only — D7: no derivation change)

- `ConversationStatus` docstring table: BUSY / BLOCKED "post a message?" →
  "yes, into the open turn (unless subagents are active)"; CANCELLING stays
  "no"; drop "the only postable state" from the IDLE line comment.
- `AgentSession.get_conversation_status` docstring: rewrite the "a second
  message cannot be queued behind a first" paragraph.

### `luca/agent/contrib/tui/app.py` (D12)

- `_set_busy` (app.py:661): stop disabling the prompt; keep the focus/refresh
  behavior.
- `on_prompt_input_submitted` (app.py:193): drop the `if not idle(): return`
  guard. New flow:
  - Slash commands stay **idle-only**: text starting with `/` while non-idle →
    notice, draft preserved, return (commands mutate runner/session state).
  - `try: id = runner.post_message(parts)` — on `AgentError` (the D3/D13
    exceptions included): `self.notify(...)`, **do not clear the input**, keep
    `_pending_images`, return. On success: clear, optimistic
    `UserCell` mount (interleaving with streaming cells is correct).
  - Start the drive only when no drive worker is live — a new
    `self._driving: bool` set in `_start_drive`, cleared in `_drive`'s
    `finally`. Never call `_start_drive` while one runs: the worker group is
    `exclusive=True`, so a second start would *cancel the live run mid-turn*.
    A mid-turn post needs no start at all — the live loop's next LLM call
    picks it up (no wake infrastructure, per the PRD).
- Subagent conversations remain non-postable from the TUI (policy, unchanged).

### Untouched (verify, don't edit)

Status derivation (D7), projector + `projection.py:193` comment (D8 keeps it
true), ledger doors, `ContextManager`, compaction machinery, `luca.client`
(D10), `before_post_message` middleware signature (D4), events (D9),
subagents contrib (`plugin.py` / `tools.py` need no change — the gate and
handshake are untouched).

## 3. Tests

House style: full-object asserts on the session / event list,
`DeterministicRunner` + `FauxProvider`. New file
`tests/agent/test_runner_post_message.py` for the matrix + mid-turn stories
(move the four existing §5.5 matrix tests out of `test_runner_failures.py`;
update the AGENTS.agent.md test table accordingly). Subagent-targeted stories
in `tests/agent/subagents/test_post_messages.py`.

Existing tests that FLIP or change:

- `test_runner.py::test_post_message_refuses_to_queue_behind_an_unanswered_message`
  → becomes the queueing-allowed story (two trailing posts, one turn answers
  both — PRD "Queued trailing messages").
- `test_runner_failures.py::test_post_message_rejects_awaiting_approval`
  → flips: posting while BLOCKED at a gate is accepted; assert the message
  landed inside the open turn and status stays BLOCKED.
- `…::test_post_message_rejects_an_open_resumable_bracket` → flips: the
  mid-turn append is the feature.
- `…::test_post_message_rejects_cancelling` → stays; assert the dedicated
  exception type.
- `test_runner_compaction.py::test_post_message_is_illegal_while_a_compaction_is_scheduled`
  → keep; update the `pytest.raises` match to the new message.

New coverage (PRD Testing section + D13):

- **Matrix**: every D2 row accept/reject — main IDLE; main BUSY-no-open-turn
  (queue); open conversational turn (BUSY and BLOCKED); seeded undriven child
  (accept); finished child (reject); archived conversation (reject — include
  the BUSY-deriving archived shape from §1); unknown id; scheduled + in-flight
  compaction (reject); CANCELLING (dedicated exception, live and reloaded).
- **D13**: parent with unresolved children rejects (live drive and a reloaded
  session literal); accepted again once every child resolved (message then
  answered before `TurnFinish(COMPLETED)`); posting to a live *subagent's*
  open turn accepts; a child that itself has active children rejects;
  CANCELLING outranks D13 on a cancelled parent with unresolved children.
- **Mid-turn between tool rounds**: full-session assert — message inside the
  turn, answered before the close.
- **The close race**: an `after_llm_response` middleware posts during the
  final call → turn does NOT close, extra round answers, then COMPLETED.
- **Two steering messages** before one close → one extra round answers both.
- **Post while BLOCKED** at a gate: resolve → next call includes it → answered
  in the same turn.
- **Burial**: post, then `cancel()` → CANCELLED close buries; the next
  request's projection still carries the message (D6).
- **Precedence**: cancel beats unseen-message; `hard_max_steps` closes ERRORED
  over an unseen message.
- **TUI (Pilot)**: mid-run submit posts + renders a `UserCell`; rejection
  (cancelling / subagents active) → notice + draft preserved; slash command
  mid-turn → notice + draft preserved; prompt no longer disabled while busy.
- `pretty_print`: one shape test with a mid-turn user message (verify only —
  likely already renders correctly).

## 4. Documentation

Per `docs/llm.txt` (validate snippets; density; update `Next:`/index only if
pages are added — none should be):

- `AGENTS.agent.md`: §11 status table + the `post_message requires IDLE` ¶
  (line ~244), subagents limits ¶ (~421 "main-conversation only"), test-file
  table (post_message matrix row + new file).
- `docs/agent/04-runner.md`: the post_message section + the status table's
  "post a message?" column; document the matrix, D5, D6, both exceptions.
- `docs/agent/01-quickstart.md` + `docs/agent/README.md`: polling-sketch note
  (posting is also legal mid-run).
- `docs/agent/13-subagents.md` (~264): rewrite "a subagent never receives a
  user message after the seed"; add the D13 rejection.
- `docs/agent/12-compaction.md` (~66): the warning stays true — verify wording.
- `docs/agent/07-middleware.md`: hook signature unchanged — verify wording.
- `prd.md`: add D13 (+ the §1 archived-row refinement), then `history.md`
  entry.

## 5. Order of work

1. Exceptions + exports.
2. `post_message` precondition + signature (+ models.py docstrings).
3. Core matrix tests green (incl. flips).
4. Close-site D5 check + mid-turn story tests.
5. Subagent stories (D13) + tests.
6. TUI changes + Pilot tests.
7. Docs + AGENTS.agent.md + PRD/history.
8. `uv run ruff check --fix && uv run ruff format`; full `uv run py.test tests/`.

## 6. Risks / things to hold in mind

- The TUI worker-group `exclusive=True` footgun (§2): starting a drive while
  one is live cancels the in-flight run. Guard with `_driving`; test it.
- The `seen` capture must share the sync region with the projection — no await
  between the capture and `prepare_llm_call`.
- Flipped tests may be load-bearing elsewhere: grep for
  `"requires an IDLE conversation"` when changing the message text.
- `filterwarnings = error`: the extra LLM round consumes an extra scripted
  Faux response in several existing tests only if a test posts mid-turn —
  new tests must script the extra response explicitly.
