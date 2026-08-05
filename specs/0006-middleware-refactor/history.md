# 2026-08-05

The initial PRD proposed twelve conversation-aware middleware hooks AND a
"language-neutral model boundary": core-owned `ModelMessage` /
`ModelAssistantMessage` DTOs so `luca/agent/core/middleware.py` would import
nothing from `luca.client`.

Codebase inspection found that `luca/agent/core/projection.py` — core, in the
same package — already imports `luca.client` message types deliberately and
states it as policy in its docstring ("The projector targets canonical
`luca.client` DTOs and stops there"), as does `adapter.py`. Core → client is
the sanctioned dependency direction. Cleaning only `middleware.py` would have
bought a cosmetic boundary while requiring three message classes and seven
content-block types plus `MediaSource` to be mirrored into core, a
bidirectional adapter, and a rewrite of the projector,
`SummarizingContextManager` and every projection test.

Decision: **cut it.** Middleware is language-neutral at the level of what a
boundary CARRIES, not what Python class carries it. "The messages about to go
to the model" binds to `luca.client` messages in Python; a TypeScript
implementation would bind the same concept to, say, the Vercel AI SDK's
message type. No new types are introduced by this PRD.

The `build_tool_list` change survived the cut, but for a different reason:
`ToolSpec` is the core's own tool type and carries strictly more than the wire
`Tool` (which drops `tool_kind`, `namespace`, `is_private`, `output_schema`,
`metadata`), so a middleware handed the wire list cannot filter on any of them.
Its public return type becomes `list[ToolSpec]`, with adapter conversion moving
into `_collect_tools`.

Three further corrections came out of the inspection:

- Requirement 7 (streaming and non-streaming converging on one
  `after_llm_response` call site) is **already satisfied** — `runner.py:3158`
  is the only call site. Demoted from code to tests.
- Requirement 8 (conversation scope on every write) is **mostly satisfied** —
  `SessionLedger` already takes a `conversation_id` on every door except
  `refresh_entry`. The gap is in the runner's private write helpers.
- `recalculate_context_tokens()` will now invoke **no** middleware at all
  (today it threads every entry through `before_entry_written`). It rewrites
  every entry across every conversation, so no single id honestly describes
  the operation; it is an operational refresh, not a write with a scope.

Two things absent from the initial PRD were added: the compaction transition's
write scope is the OUTGOING conversation (the destination's id does not exist
when its entries are built), and restricting `before_tool_execution` to
dispatch attempts makes `_finalize_undispatched` identical to
`_finalize_outcome`, so it is deleted and its five call sites collapse.
