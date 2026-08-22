# Implement PRD 0011a — rewrite of `luca.client` streaming

You are implementing a fully specified, already-approved refactor. Do not
redesign anything; every open question has a recorded decision.

## Read, in this order

1. `AGENTS.md`, then `AGENTS.client.md` — repo rules (tests, style, tooling).
2. `specs/0011a-client-streaming-refactor/prd.md` — the contract: what is
   built, the three deliberate API changes, the behaviors §6 requires.
3. `specs/0011a-client-streaming-refactor/plan.md` — the how: module layout,
   class designs, the four PoC deviations (§2), timeout mechanics, new test
   design (§6), and the step order you will follow (§7).
4. `specs/0011a-client-streaming-refactor/test_audit.md` — the frozen
   black-box test list, the 22 internal tests to delete, the four forced
   respells.
5. `specs/0011a-client-streaming-refactor/poc_streamer.py` — normative for
   shape, naming, and altitude. Where it and `plan.md` disagree, `plan.md`
   wins.

`history.md` in the same directory is optional context (how decisions were
reached). Read no other files in `specs/` — they are unrelated scratch.

## Rules

- Work on a branch, never on `main`.
- Follow `plan.md` §7 **in order**; each step names its green gate — run it
  before moving on. Step 1 (wire-mixin extraction) must leave the FULL suite
  green before any demolition.
- Frozen black-box tests may only receive the exact edits `test_audit.md`
  lists (four `total_timeout=` renames, two `"RawFinish"` respells, one
  dropped private-attr line, one agent message respell). Nothing else in them
  changes — not assertions, not scenarios, not structure.
- Internal tests are deleted, not ported. New tests are designed from the new
  layers' contracts per `plan.md` §6, written alongside each step.
- `uv run py.test tests/` with warnings-as-errors is the bar; finish with
  `uv run ruff check --fix && uv run ruff format`. No new runtime
  dependencies. Delete freely — this is V1, no compat shims.
- Docs are step 11, not optional: follow `docs/llm.txt`.

Deliverable: all §7 steps done, full suite green, docs updated.
