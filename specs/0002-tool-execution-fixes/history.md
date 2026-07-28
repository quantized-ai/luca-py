# 2026-07-28 — second review round against the codebase

Re-verified P1–P5 against the current source; all five still reproduce, the
contracts in §5–§12 are still implementable, and the scope did not change. Test
scope numbers confirmed exact (70 `ToolSpec(`, 101 `ToolExecution(`, 202
`AgentSession(`, 14 files). Three PRD claims were checked and hold and should
not be re-audited: `get_tools` raising leaves the run resumable (`_pump` /
`_consume` call `_refresh_status()` + `_end_run()` on an engine raise, and an
open turn derives PENDING); the shipped TUI needs no change for the reduced
`ToolExecutionStarted`, because it mounts its cell on `ToolCallReceived` and
re-creates a missing one at `ToolExecuted`; fork is safe, being a plain
`model_copy(deep=True)`.

One scope decision: **the context refresh shrank to a bare public method.**
The `recalculate_context=False` constructor keyword and `main.py
--refresh-context` were cut — the shipped `ContextManager` is a character
estimate no model choice affects, so nothing in this repo would ever set them,
and the project rule is no knobs before a second real case. Also made honest in
§1, which claimed five changes while specifying six.

Corrections, all against the code:

**§16's `ProxyToolRegistry` cold-resume bullet was refuted by §15.5** and is
deleted. Once `decide` and `prepare` both resolve without the route cache, a
call pending approval across processes resolves, is gated, and dispatches;
nothing "ergonomic" remains. §15.5 now says so.

**"blanket-ALLOW removed" would have been implemented as DENY**, which records
`REJECTED` for a tool that never existed and contradicts §6.4's table. Split
explicitly: ALLOW-on-cache-miss goes, ALLOW-on-genuinely-unresolvable stays.

**`prepare()`'s cancellation race was unspecified.** §6.5/§12.5 said "the three
non-body races" while listing four calls. Pinned: grace 0, `detach=False`, same
as the other three; only the prepared callable keeps grace and a detached kill.

**`before_tool_execution` stops being exactly-once per call**, and §8 still
claimed it was. A crash during `prepare()` now leaves the execution `PENDING`
(§6.7), so the next drive fires the hook again — where today the hook fires only
after `RUNNING` persists and a crash recovers as `INTERRUPTED` without
re-firing. Invariant rescoped to "once per dispatch attempt"; §6.7 also notes
that a `raw_tool_call` rewritten by that hook does not survive such a crash.

**The third write door only half-covered.** `transition_conversation` takes
`updates: list[AnyEntry]` as well as `created`, and is public; the spec helper
must run over `updates`, `created` and `closing`.

Three precision fixes: `calculate_context` runs inside the ledger's build
callback on an append, so its entry is NOT yet in `session.entries`; §6.5's
"byte-identical" CANCELLED claim was over-broad (a cancelled birth has
`tool_spec=None` / `approval_status=None`, unlike a prepare-window
cancellation); and "session literals must set both sides" was wrong — the load
validator restores `tool_spec` from `tool_spec_id`, so a literal needs the id
plus its store row, and it is `tool_spec` without an id that raises. §6.4's
outcome table gained the toolless-birth row.

Raised and dropped: a concern that §15.1's restore validator hands every
execution the same shared `ToolSpec` instance, so middleware mutating one in
place would reach into unrelated executions. Out of bounds — luca does not
defend itself against application middleware, and this is the third proposal of
that shape to be rejected. Restored specs stay shared; §6.9 is unchanged.

# 2026-07-28 — review round against the codebase

The PRD arrived at "ready to implement". A full read of the affected code
(`core/` runner, ledger, models, tool_registry, context, context_manager, tools,
adapter, middleware, compaction; all of `contrib/`; the 14 test files carrying
`ToolSpec` / `ToolExecution` literals; the docs tree) confirmed the five changes
and their motivations — P1–P5 each reproduce against real lines — and the
contracts in §5–§12 were implementable as written. The scope did not change.

What changed, all of it correction and precision rather than redirection:

**Storage had a hole.** §15.2 named `SessionLedger.append` / `put_entry` /
`prune` as the write doors that file a spec into `session.tool_specs`. `prune`
never writes a `ToolExecution`, and the real third door —
`transition_conversation`, used by compaction — was missing. A plan-created
`ToolExecution` would keep its spec in memory and lose it on the next save,
because the session serializer strips inline specs and would have no id to write
in their place; the load-time guards in §8 cannot catch that, since the
stripping destroys what they look for. Now stated as three doors, in §6.9, §8,
§12.11, §13 and §15.2.

**Two runtime defects in the dispatch path.** A prepared callable that returns a
non-awaitable would raise `TypeError` from `asyncio.ensure_future`, which sits
outside `_run_tool_body`'s failure handling — crashing the run instead of
failing one execution. And the "returned a non-callable" outcome had no
specified error identity. Both pinned (§5.2, §6.4, §15.3).

**Cancellation of `decide` now wraps its middleware pair**, so a token already
tripped when the batch starts fires no hook at all. It deliberately does not make
`before_permission_check`'s result durable mid-decide — there is nowhere to put
it. Recorded as §12.10, with the general rule (only `before_tool_execution` has
an exactly-once guarantee) in §6.5.

**The context refresh became opt-in.** It was specified as a public method the
application calls on a model switch. It is now `AgentSessionRunner(
recalculate_context=False)` plus `main.py --refresh-context`, default off,
`/model` untouched — the shipped `ContextManager` is a character estimate no
model choice affects, so the automatic version would rewrite every entry to
change nothing.

**`ToolSpec.metadata` stays** and moved from §16 ("dead surface") to §7 as a
documented extension point.

Two proposals were raised and rejected, both for the same reason — they defended
the framework against application code, which contradicts the trust model
already stated in `middleware.py` ("the runtime does not validate or repair what
a hook returns"): freezing `ToolSpec` to prevent middleware from mutating a
shared instance, and having the session serializer re-normalize whatever it is
handed. `ToolSpec` stays mutable; restored specs are shared instances and that
is fine.

Also corrected: §1 claimed the five changes were interdependent, which §15.6
contradicts — they are two independent stacks shipped as one commit for churn
reasons. §13's test scope understated the work (14 files, not 9; 70 `ToolSpec`
literals; `contrib/tui/test_wiring.py` has to await `build_tool_list`), and now
calls for a factory in `scenarios.py` rather than hand-editing every literal.
The docs list gained `09-plugins.md`, `contrib/resource_permissions/README.md`
and `contrib/plugins/README.md`.
