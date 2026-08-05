# Implementation plan: conversation-aware middleware

Companion to `prd.md`. Nine steps, ordered so the suite stays meaningful
between them. Steps 1–3 are mechanical and touch everything; 4–7 are the
behavior changes; 8–9 are docs and tests.

## Files that change

| File | What |
|---|---|
| `luca/agent/core/middleware.py` | The twelve signatures, the module docstring, the `LucaTool` import removal |
| `luca/agent/core/runner.py` | `_run_middlewares` + every call site, the write helpers, the birth pair, the dispatch restriction, `build_tool_list` |
| `docs/agent/07-middleware.md` | Hook catalogue, the embedded mixin source, hook count 10 → 12 |
| `AGENTS.agent.md` | "Add middleware" recipe, §12's hook-is-the-boundary paragraph, the subagents "conversation-blind" note, the per-step method table, the `core/` layout comment |
| `tests/agent/test_runner_middleware.py` | Every hook test |
| `tests/agent/test_runner_context.py` | `before_entry_written` overrides + `recalculate_context_tokens` |
| `tests/agent/test_runner_post_message.py` | `after_llm_response` overrides |
| `tests/agent/test_runner_cancellation.py` | `before_permission_check` override |
| `tests/agent/test_runner_compaction.py` | `before_entry_written` overrides |
| `tests/agent/subagents/test_wake_rounds.py` | `after_tool_execution` override |
| `tests/agent/subagents/test_workers.py` | `before_entry_written` bomb |
| `tests/agent/test_private_tools.py`, `test_runner.py`, `tests/agent/contrib/tui/test_wiring.py` | `build_tool_list` return type |
| `tests/agent/contrib/test_plugins.py` | `RecordingMiddleware` signatures |

Nothing under `luca/agent/contrib/` changes. No contrib package implements a
hook.

## Step 1 — `_run_middlewares` carries the conversation

`runner.py:1913`. New signature:

```python
def _run_middlewares(self, method_name, conversation_id, value, *ctx_args,
                     unpack_values=False, **ctx_kwargs):
    for mw in self.middleware:
        if not hasattr(mw, method_name):
            continue
        args = (self.session, conversation_id)
        if unpack_values:
            value = getattr(mw, method_name)(*args, *value, *ctx_args, **ctx_kwargs)
        else:
            value = getattr(mw, method_name)(*args, value, *ctx_args, **ctx_kwargs)
    return value
```

`session` comes off `self`, so no call site passes it. `conversation_id` is
positional and second, so every call site must be updated — which is the
point: the compiler-equivalent here is the failing test, and a missed site
raises `TypeError` immediately rather than silently passing the wrong value.

Update the eight existing call sites: `1242` (`before_post_message`), `1943` /
`2012` (`before_entry_written`), `2107` (`after_tool_execution`), `2125`
(`before_tool_execution`, deleted in step 6), `2153` (`build_model_string`),
`2207` (`build_tool_list`), `2296` (`before_llm_call`), `3158`
(`after_llm_response`), `4125` / `4138` (the permission pair).

## Step 2 — thread `conversation_id` through the write helpers

All in `runner.py`. Add `conversation_id: str` as the first parameter:

- `_complete_entry(conversation_id, entry)` — `1934`
- `_complete_uncommitted(conversation_id, build_fn, parent_id, ts)` — `1954`

`_append` (`1945`), `_persist_entry` (`1997`) and `_persist_execution`
(`2016`) already take one; they just forward it.

The compaction transition (`2607`, `2625`, `2639`) passes the **outgoing**
`conversation_id` — the destination conversation's id is minted inside
`ledger.transition_conversation` and does not exist yet. Add a comment saying
so; it is the one non-obvious scope in the change.

Every other caller has a `conversation_id` in scope already.

## Step 3 — `recalculate_context_tokens()` stops running middleware

`runner.py:1970`. Split the context calculation out of `_complete_entry`:

```python
def recalculate_context_tokens(self) -> None:
    for entry_id in list(self.session.entries):
        entry = self.session.entries[entry_id].model_copy()
        entry.context_tokens = self.context_manager.calculate_context(self.session, entry)
        self.ledger.refresh_entry(entry)
```

Signature unchanged (no `conversation_id` parameter — there is no hook to
scope). Rewrite the docstring paragraph that currently says "Through the
middleware door, because middleware has the final say on context": that
sentence stays true for every other write path and becomes false here.

`ledger.refresh_entry` is untouched — it already takes no conversation.

## Step 4 — `build_tool_list` moves to `ToolSpec`

`runner.py:2189`. New body:

```python
def build_tool_list(self, conversation_id: str, specs: list[ToolSpec]) -> list[ToolSpec]:
    visible = [spec for spec in specs if not spec.is_private]
    return self._run_middlewares("build_tool_list", conversation_id, visible)
```

The private filter stays **ahead** of the hook, preserving "a middleware never
sees a private tool".

`_collect_tools` (`2254`) does the adaptation:

```python
specs = await self.resolve_tool_specs(conversation_id)
self._verify_gate(conversation_id, specs)
effective = self.build_tool_list(conversation_id, specs)
return specs, [adapter.tool_spec_to_luca_tool(spec) for spec in effective]
```

`_verify_gate` keeps running on the registry's specs, before the hook — the
gate is a registry-contract check, not a middleware one.

In `middleware.py`, delete the `try: from luca.client.types.tools import Tool
except ImportError` block (`49–52`) and the `LucaTool` name. Import `ToolSpec`
from `.models`. The two `luca.client` message imports (line `39`) stay.

## Step 5 — `before_tool_creation`

`runner.py:_birth_draft` (`4012`). The hook runs on the deep-copied call,
**before** the private-name check, so a middleware that renames a call is
checked against the effective name:

```python
raw = ToolCall(id=tc.id, name=tc.name, arguments=copy.deepcopy(tc.arguments))
raw = self._run_middlewares("before_tool_creation", conversation_id, raw)
if self.tool_registry is None or raw.name in private_names:
    ...
```

Note the existing `tc.name` check on `4047` becomes `raw.name`, and the
`ToolNotFound` message should quote `raw.name`.

`create_execution` on `4061` already receives `raw`, so it picks up the
rewrite for free.

## Step 6 — `after_tool_creation` + restrict `before_tool_execution`

**6a. `after_tool_creation`** — `runner.py:_birth_executions` fold (`3943`).
Build the effective execution, run the hook, then persist:

```python
effective = execution.model_copy(update=changes)
effective = self._run_middlewares("after_tool_creation", conversation_id, effective, exception)
born = self._persist_entry(conversation_id, effective)
```

The existing `_persist_entry(conversation_id, execution, **changes)` on `3966`
becomes the two-line form above. The `if born.status != PENDING` branch below
it now tests the **post-hook** status, which is what makes "the returned
execution determines the next lifecycle step" true.

**6b. Delete `_finalize_undispatched`** (`2115–2126`). Under the new contract
it is identical to `_finalize_outcome`. Replace all five call sites —
`2809`, `3567`, `3584`, `3789`, `3975` — with `_finalize_outcome`.

That is the whole of "restrict `before_tool_execution` to dispatch": the hook
already fires nowhere else. `_dispatch_one`'s call (`4198`) is untouched.

**6c.** Update the `_dispatch_one` docstring (`4192–4193`), which currently
explains why every path from step 3 on must use `_finalize_outcome` rather
than `_finalize_undispatched`. That distinction no longer exists.

## Step 7 — no code for `after_llm_response`

Already correct: `runner.py:3158` is the only call site, streaming and
non-streaming both converge on the assembled `message` before it, an aborted
stream `continue`s (`3084`) and an errored one raises (`3080`). Step 1 updates
the call. Coverage is step 9's job.

## Step 8 — the mixin and the docs

`middleware.py`: rewrite the twelve signatures per the PRD's specification
block. The module docstring needs three edits — the "10 hooks" framing, the
durability paragraph (`before_tool_execution` is now dispatch-only), and a new
paragraph on the `(session, conversation_id)` prefix and per-conversation
state.

`docs/agent/07-middleware.md`: "**10** points" → 12, re-embed the mixin source
verbatim, add `before_tool_creation` / `after_tool_creation` sections, and add
a short subagents example showing one instance receiving two conversation ids.

`AGENTS.agent.md`, four places:
- The `core/` layout comment: "the 10 duck-typed middleware hooks" → 12.
- "Add middleware" recipe: new signatures, the dispatch-only rule, the
  `build_tool_list` type change (the current text explicitly says the hook
  "still receives the post-adapter WIRE list, never `ToolSpec`s" — invert it).
- §12's cancel table paragraph: the `prepare` asymmetry can no longer be
  justified by "would fire the hook twice". State the durable-shape reason
  instead, or delete the justification.
- Subagents section: delete "**Middleware stays conversation-blind** (§12 of
  the PRD, accepted)" and replace with the new contract.
- The per-step method table: `build_tool_list`'s row, and `_finalize_outcome /
  _finalize_undispatched` becomes one row.

## Step 9 — tests

`tests/agent/test_runner_middleware.py` is the primary file. Update every
existing hook test to the new signatures, then add:

1. **Conversation identity** — one middleware instance driven through a turn
   that spawns a subagent; assert the recorded `(hook, conversation_id)` pairs
   include the child's id for the child's operations and the parent's for the
   parent's. This is the acceptance criterion the whole PRD exists for.
2. **Session identity** — `session is runner.session` inside a hook.
3. **Chaining** — three middleware, declared order, for: a single-value hook,
   the `before_llm_call` tuple, and an after hook with exception context
   (assert the same exception object reaches all three).
4. **`build_tool_list` on specs** — receives `ToolSpec`s, never a private one,
   can add/remove/reorder, and the returned list is what the adapter converts.
5. **`before_tool_creation`** — rewriting `name` changes what
   `FakeToolRegistry.create_execution` observes; rewriting to a private name
   records `NOT_FOUND`.
6. **`after_tool_creation`** — (a) fires for a normal PENDING birth; (b) fires
   with the live exception for a raising `create_execution`; (c) forcing a
   terminal status in the hook sends the call straight to the outcome tail
   instead of `decide()`.
7. **`before_tool_execution` is dispatch-only** — a table test over
   terminal-at-birth, REJECTED, REFUSED, cancelled-before-dispatch, and orphan
   recovery asserting the hook did NOT fire, plus a prepare-failure asserting
   it DID.
8. **`after_llm_response` once** — same assertion for `streaming=True` and
   `streaming=False`; zero calls for an aborted stream (`faux_hang()` + cancel)
   and for a provider error.
9. **Entry-write scope** — a recording `before_entry_written` over a turn with
   a tool call, a compaction, and a subagent; assert the full
   `(entry type, conversation_id)` list. The compaction assertion pins the
   outgoing-conversation rule.
10. **`recalculate_context_tokens` runs no middleware** — a bomb middleware
    whose `before_entry_written` raises; the call succeeds and counts update.

Follow the project rules: declarative precondition → one action → whole-object
postcondition, `DeterministicRunner` + `FauxProvider` + `FakeToolRegistry`
from `scenarios.py`, no contrib imports in core tests, no helpers in test
bodies.

Mechanical signature updates in the other eight test files; `build_tool_list`
callers (`test_private_tools.py:98`, `test_runner.py:338`/`365`,
`tests/agent/contrib/tui/test_wiring.py:52`) also change what they assert on —
`ToolSpec`s now, and `test_wiring.py:52` should adapt to wire tools itself if
it needs them.

## Risks

- **Step 1 is a wide blast.** Every `_run_middlewares` call site must be
  updated in the same commit; a missed one is a `TypeError` at runtime, not a
  silent bug, so the suite catches it — but only if that path is covered.
  Grep for `_run_middlewares(` and count against the list in step 1.
- **Step 6a changes when the birth status is read.** Persisting the post-hook
  execution means `_spawn_budget_refusal`'s result can be overridden by
  middleware. That is intended (trust model), but the subagent budget tests
  should be re-read to confirm none of them relies on the old ordering.
- **Step 4 moves the private filter's observable position.** It stays ahead of
  the hook, so behavior is unchanged; `test_private_tools.py` is the guard.
- **Step 3 is a behavior change with no failing test today.** Existing
  coverage asserts middleware DOES run there
  (`test_runner_context.py`'s `recalculate_context_tokens` cases) — those
  tests invert rather than disappear.

## Order

1 → 2 → 3 → 4 → 5 → 6 → 8 → 9, with `uv run py.test tests/agent/` after each.
Step 7 is a no-op. Finish with `uv run ruff check --fix` and
`uv run ruff format`.

## What implementation found that the plan did not

Recorded because both are contract facts, not incidents.

1. **There are TWO birth paths, not one.** `_invoke_runtime_tool`
   (`runner.py:3581`) runs a runner-minted tool call — the private result tool
   that resolves a finished subagent — through `_birth_draft` and then appends
   the draft directly, bypassing `_birth_executions`' fold. Adding
   `before_tool_creation` inside `_birth_draft` therefore fired it there while
   `after_tool_creation` never ran, breaking the pair on every subagent
   resolution. Fixed by running `after_tool_creation` inside that path's
   `_append` build callback. The rule to hold: **the creation pair is paired on
   every path**, and `_birth_draft` is the only shared point, so anything
   calling it owns firing the partner.
2. **`build_tool_list` runs AFTER `before_llm_call`, not before.** The drive
   calls `prepare_llm_call()` (which fires `before_llm_call`) and only then
   `_collect_tools()`, because the tool step is the one raced against the
   cancellation token. The two are independent — neither reads the other's
   result — so this is an implementation detail rather than a contract, but
   the PRD's "Model round" block originally listed them the other way and has
   been corrected to match the code, with a note that their relative order is
   not normative.

Final state: 2158 tests pass (1733 at baseline, +16 new middleware cases,
+the client suite), `ruff check` and `ruff format` clean.
