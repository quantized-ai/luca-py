"""ContextManager — the runner's context strategy: how big an entry is, what a
tool's output looks like once it is stored, what replaces a pruned entry, and
when and how the conversation is compacted.

One runtime collaborator (passed as `context_manager=`, never serialized —
only its durable results persist) owning five policies over the shared
`Entry` abstraction. All five take the live `AgentSession` first, so one
argument order describes the whole contract and every policy can see the same
state — the active model included, which is what makes a real tokenizer
implementable.

TWO of the five also take a `conversation_id`, and the split is a decision
rather than an oversight. `calculate_context` and `prune_entry` do NOT, because
`context_tokens` is intrinsic to the ENTRY and shared by every conversation
that references it. `process_tool_output` does NOT, because the execution it
already receives carries `conversation_id` — a manager that wants to truncate a
subagent's output harder than the main conversation's reads it from there. The
compaction pair DOES, because compaction is a fact about one path:

- `calculate_context(session, entry)` — the entry's intrinsic context-token
  count. Called by the runner on every new entry before `before_entry_written`
  middleware, and again on a `ToolExecution`'s terminal transition (before
  `after_tool_execution`), when its model-facing outcome is finally known.
  Provider usage is never an input — that is accounting, not content size.
  Because it runs on EVERY new entry, scanning `session.entries` here makes a
  turn quadratic; cross-entry work belongs in `prune_entry` /
  `process_tool_output`, which run rarely. On an append it also runs INSIDE
  the ledger's build callback, so the entry already has its `id` but is not
  yet a member of `session.entries` — an implementation that looks itself up
  there raises `KeyError` on every append.
- `prune_entry(session, entry)` — a durable `PrunedEntry` replacing the
  entry's contribution to a conversation path. Returns a TEMPLATE: identity
  fields (`id`, `parent_id`, `created_at`) are placeholders for the persisting
  door to stamp (ids and clocks are the runner/ledger's, never a strategy's).
  It has NO framework call site — an application calls it and hands the
  template to `SessionLedger.prune()`, so the caller already holds the
  session. Its session argument is uniformity, not need.
- `process_tool_output(session, execution, result)` — an optional
  transformation of a tool's returned `ExecutionResult`, applied before the
  terminal `ToolExecution` is constructed and before any middleware runs. May
  truncate or replace model-facing content (preserving the original in
  `metadata` per its own policy); the base behavior is an identity
  pass-through. `execution` is there to be READ FOR IDENTITY — `tool_spec`,
  `raw_tool_call.name`, `raw_tool_call.arguments` — which is what makes
  "truncate `bash` output but never `read`" expressible. It is IN TRANSITION:
  `status` is still RUNNING and `result` is not yet attached, so it must not
  be inspected for outcome.
- `should_compact(session, conversation_id)` — has that conversation crossed
  the point where compaction is worth doing? Consulted at the top of every
  drive (and at `start()` time, which is why this is SYNC). Unlike
  `calculate_context` this one is MEANT to sum the path: it runs once per
  drive, not once per entry.
- `compact(session, conversation_id, nodes, entry)` — the compaction itself: fill in a deep
  copy of `entry` and return a `CompactionPlan` describing the resulting
  conversation. The base raises `NotImplementedError` and the base
  `should_compact` returns False, so the default manager never compacts and
  only an explicit `schedule_compaction()` can reach `compact` at all.

Compaction: who decides what. The two methods above own everything about the
*decision* — when compaction is worth doing, what the summary says, which model
produces it, which nodes survive, and whether to back off after a failure. Core
triggers, stamps and archives — nothing else. It reads exactly five things off a
returned plan (`entry.parts`, `entry.llm_config`, `entry.metadata`, `nodes`,
`usage`) and discards every other field, which is the whole coupling surface
between the two. The runner opens a turn bracket, appends a `CompactionEntry`,
stamps its `started_at`, and hands `compact` the live session, the path it may
rewrite, and a DEEP COPY of that entry; the returned `CompactionPlan` is
validated for STRUCTURE (never for meaning), its usage recorded, and the
transition performed — one atomic swap in which the pre-compaction conversation
is archived intact and a new one becomes active. Nothing is ever deleted. The
plan's value objects and validator live in `compaction.py`.

Stored counts and the active model. `Entry.context_tokens` is stored on the
entry, so an implementation that counts against `session.session_config.
llm_config` leaves every stored count stale the moment that model changes.
`AgentSessionRunner.recalculate_context_tokens()` re-derives them all; nothing
in the framework calls it, and the application that swaps in a real tokenizer
is the one that should.

This is a CONCRETE class with complete, deliberately simple default behavior
(the same pattern as `ConversationProjector`): estimation is one token per
`CHARS_PER_TOKEN` characters of model-facing text plus a flat `IMAGE_TOKENS`
per image, pruning supports only terminal tool executions (replacing their
output with a fixed marker), and compaction is absent — `should_compact`
declines and `compact` raises, so the shipped default is a pure accountant.
`luca.agent.contrib.simple_context_manager` ships one that also compacts.
Instantiate it directly, subclass and override selected methods, or supply
another object with the same behavior. Luca does not prescribe per-entry-type
methods — dispatch by type here is an internal choice, not a runner contract.

The framework never recalculates, validates, or repairs context after
middleware has run: middleware has the final say, and the application owns
the state it returns.
"""

from __future__ import annotations

import json
from typing import ClassVar

from .compaction import CompactionPlan
from .exceptions import AgentError
from .models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    AssistantMessage,
    ChildConversation,
    CompactionEntry,
    Entry,
    ExecutionResult,
    ImageContent,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    UserMessage,
)

# The replacement content a pruned tool output projects as. Module-level alias
# of the class default.
PRUNED_TOOL_OUTPUT_MARKER = "[tool output has been pruned to reduce context]"


class ContextManager:
    """The default context policy. Every method is an override point."""

    PRUNED_TOOL_OUTPUT_MARKER: ClassVar[str] = PRUNED_TOOL_OUTPUT_MARKER
    CHARS_PER_TOKEN: ClassVar[int] = 4
    IMAGE_TOKENS: ClassVar[int] = 1_000

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        """Estimate the context tokens of `entry`'s model-facing content.

        Ownership per entry type: a user message owns its content; an
        assistant message owns its text, thinking, and tool-call REQUESTS
        (name + arguments — counted here, never again on the execution); a
        tool execution owns only its model-facing outcome (result content,
        else its structured error message), and is 0 while nonterminal; a
        compaction owns its summary `parts` (0 until they land, which is why
        the runner recalculates when they do); a child conversation owns its
        RESULT and nothing of the child's own path (0 until it resolves, and
        recalculated then); a pruned entry owns its replacement content.
        Markers own nothing.

        Non-text content is counted separately by `_media_tokens`, so
        `_estimate_tokens` and `_model_facing_text` stay text-shaped and
        independently overridable."""
        return self._estimate_tokens(self._model_facing_text(entry)) + self._media_tokens(entry)

    def prune_entry(self, session: AgentSession, entry: Entry) -> PrunedEntry:
        """Build the `PrunedEntry` template replacing `entry` in a path.

        Only terminal `ToolExecution`s are prunable by this default; anything
        else fails loudly. The returned template carries no identity —
        `id`/`created_at` stay `None` — the persisting door stamps the real
        `id`/`parent_id`/`created_at` and the runner-side ordering calculates
        `context_tokens` and runs entry middleware, exactly as for any other
        new entry."""
        if not isinstance(entry, ToolExecution):
            raise AgentError(f"Cannot prune entry of type {entry.type!r}: only tool executions are prunable.")
        if entry.status in NONTERMINAL_STATUSES:
            raise AgentError(
                f"Cannot prune ToolExecution {entry.id!r}: a nonterminal "
                f"({entry.status.value}) execution is not prunable."
            )
        return PrunedEntry(
            pruned_entry_type=entry.type,
            pruned_entry_id=entry.id,
            content=[TextContent(text=self.PRUNED_TOOL_OUTPUT_MARKER)],
        )

    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Transform a tool's returned result before it becomes durable.
        Identity by default; override to truncate or replace model-facing
        content (stash the original in `metadata` if your policy keeps it).

        Select on `execution.tool_spec` / `execution.raw_tool_call` to vary the
        policy by tool. `execution` is mid-transition — still RUNNING, no
        result attached — so read it for identity, never for outcome.
        `tool_spec` is `ToolSpec | None`: a registry may dispatch a call it
        never snapshotted, so branch on `raw_tool_call.name` or guard the
        `None` rather than chaining through it."""
        return result

    # ── compaction ───────────────────────────────────────────────────────────

    def should_compact(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> bool:
        """Has `conversation_id` crossed the point where compaction is worth
        doing? Consulted at the top of every drive (and at `start()` time,
        which is why this is SYNC).

        The threshold, the context sum and the window size are all yours —
        core has no context-total API, and `luca.client.catalog` carries
        `context_window` per model. Core remembers nothing about previous
        failures either: a manager that should stop trying returns False."""
        return False

    async def compact(
        self,
        session: AgentSession,
        conversation_id: str,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        """Fill in a DEEP COPY of `entry` and describe the resulting
        conversation.

        `nodes` is THE PATH YOU MAY REWRITE: `conversation_id`'s path with this
        compaction's own `TurnStart` removed, ending with `entry`. It barely
        needs the id — `nodes` already carries the work — but it takes it
        anyway: the signature IS the contract, and adding a parameter later
        breaks every implementor.
        `plan.nodes` may carry any of these ids, in any order, with new
        entries interleaved — and nothing else; an id outside this tuple is a
        plan rejection, so you never have to recognize or route around
        framework markers. `plan.nodes = list(nodes)` is always a legal full
        carry.

        Return `None` for "nothing to do": the entry keeps `parts is None` and
        the bracket closes COMPLETED.

        You own the LLM call end to end — its prompt, its model (record it in
        `entry.llm_config`), its messages. The turn middleware hooks
        deliberately do NOT fire for it. Never mutate `session`: the runner
        refuses to commit if the path moved under you, and the entry you are
        handed is a copy precisely so a failed attempt cannot leave a summary
        projecting onto an unchanged path."""
        raise NotImplementedError()

    # ── derivation helpers ───────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def _model_facing_text(self, entry: Entry) -> str:
        """Concatenate the model-facing text the entry OWNS (see
        `calculate_context`). Unknown entry types contribute nothing rather
        than failing: calculation is an estimate, not a projection."""
        if isinstance(entry, UserMessage):
            return _text_of(entry.parts)
        if isinstance(entry, AssistantMessage):
            chunks: list[str] = []
            for part in entry.parts:
                if isinstance(part, TextContent):
                    chunks.append(part.text)
                elif isinstance(part, ThinkingContent):
                    chunks.append(part.thinking)
                elif isinstance(part, ToolCall):
                    chunks.append(part.name)
                    chunks.append(json.dumps(part.arguments))
            return "".join(chunks)
        if isinstance(entry, ToolExecution):
            if entry.result is not None:
                return _text_of(entry.result.content)
            if entry.error is not None:
                return entry.error.error_message
            return ""
        if isinstance(entry, CompactionEntry):
            return _text_of(entry.parts or [])
        if isinstance(entry, ChildConversation):
            # What the child contributes to its PARENT is its result, and only
            # that: the child's own conversation has its own window. The link
            # counts it only when the link itself RENDERS it (a resolution
            # written without a result execution — a cancel wind-down or a
            # hard-limit settle); otherwise the result execution's own count
            # covers the content, and counting here too would double it.
            if entry.execution_result is not None and entry.result_execution_id is None:
                return _text_of(entry.execution_result.content)
            return ""
        if isinstance(entry, PrunedEntry):
            return _text_of(entry.content)
        return ""

    def _media_tokens(self, entry: Entry) -> int:
        """The entry's non-text context contribution: a flat constant per
        image, deliberately dimension-blind. A URL source has no local bytes
        to measure, reading real dimensions would need an image decoder (a
        new dependency), and the provider formulas disagree by an order of
        magnitude. Override with a per-provider formula if it matters."""
        return self.IMAGE_TOKENS * _image_count(self._media_parts(entry))

    def _media_parts(self, entry: Entry) -> list:
        """The parts an entry owns that may carry non-text content — the same
        ownership `_model_facing_text` applies, from the other side."""
        if isinstance(entry, UserMessage):
            return entry.parts
        if isinstance(entry, ToolExecution):
            return entry.result.content if entry.result is not None else []
        if isinstance(entry, CompactionEntry):
            return entry.parts or []
        if isinstance(entry, ChildConversation):
            # Same ownership rule as `_model_facing_text`: the link carries the
            # content only when it is the one rendering it.
            if entry.execution_result is not None and entry.result_execution_id is None:
                return entry.execution_result.content
            return []
        if isinstance(entry, PrunedEntry):
            return entry.content
        return []


def _text_of(parts) -> str:
    """The text a content-part list contributes. Non-text parts contribute no
    characters — they are counted by `_media_tokens` instead."""
    return "".join(part.text for part in parts if isinstance(part, TextContent))


def _image_count(parts) -> int:
    return sum(isinstance(part, ImageContent) for part in parts)
