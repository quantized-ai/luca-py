"""ContextManager — the runner's context-accounting strategy.

One runtime collaborator (passed as `context_manager=`, never serialized —
only its durable results persist) owning three policies over the shared
`Entry` abstraction:

- `calculate_context(entry)` — the entry's intrinsic context-token count.
  Receives the complete entry and nothing else: the calculation describes
  content owned by the entry itself, so no conversation or session is passed.
  Called by the runner on every new entry before `before_entry_written`
  middleware, and again on a `ToolExecution`'s terminal transition (before
  `after_tool_execution`), when its model-facing outcome is finally known.
  Provider usage is never an input — that is accounting, not content size.
- `prune_entry(entry)` — a durable `PrunedEntry` replacing the entry's
  contribution to a conversation path. Returns a TEMPLATE: identity fields
  (`id`, `parent_id`, `created_at`) are placeholders for the persisting door
  to stamp (ids and clocks are the runner/ledger's, never a strategy's).
- `process_tool_output(execution_result)` — an optional transformation of a
  tool's returned `ExecutionResult`, applied before the terminal
  `ToolExecution` is constructed and before any middleware runs. May truncate
  or replace model-facing content (preserving the original in `metadata` per
  its own policy); the base behavior is an identity pass-through.

This is a CONCRETE class with complete, deliberately simple default behavior
(the same pattern as `ConversationProjector`): estimation is one token per
`CHARS_PER_TOKEN` characters of model-facing text plus a flat `IMAGE_TOKENS`
per image, and pruning supports only terminal tool executions, replacing
their output with a fixed marker.
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

from .exceptions import AgentError
from .models import (
    AssistantMessage,
    CompactionEntry,
    Entry,
    ExecutionResult,
    ExecutionStatus,
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

    def calculate_context(self, entry: Entry) -> int:
        """Estimate the context tokens of `entry`'s model-facing content.

        Ownership per entry type: a user message owns its content; an
        assistant message owns its text, thinking, and tool-call REQUESTS
        (name + arguments — counted here, never again on the execution); a
        tool execution owns only its model-facing outcome (result content,
        else its structured error message), and is 0 while nonterminal; a
        compaction owns its summary; a pruned entry owns its replacement
        content. Markers own nothing.

        Non-text content is counted separately by `_media_tokens`, so
        `_estimate_tokens` and `_model_facing_text` stay text-shaped and
        independently overridable."""
        return (
            self._estimate_tokens(self._model_facing_text(entry))
            + self._media_tokens(entry)
        )

    def prune_entry(self, entry: Entry) -> PrunedEntry:
        """Build the `PrunedEntry` template replacing `entry` in a path.

        Only terminal `ToolExecution`s are prunable by this default; anything
        else fails loudly. The returned template carries placeholder identity
        (`id=""`, `created_at=0`) — the persisting door stamps the real
        `id`/`parent_id`/`created_at` and the runner-side ordering calculates
        `context_tokens` and runs entry middleware, exactly as for any other
        new entry."""
        if not isinstance(entry, ToolExecution):
            raise AgentError(
                f"Cannot prune entry of type {entry.type!r}: only tool "
                "executions are prunable."
            )
        if entry.status in (ExecutionStatus.PENDING, ExecutionStatus.RUNNING):
            raise AgentError(
                f"Cannot prune ToolExecution {entry.id!r}: a nonterminal "
                f"({entry.status.value}) execution is not prunable."
            )
        return PrunedEntry(
            id="",
            created_at=0,
            pruned_entry_type=entry.type,
            pruned_entry_id=entry.id,
            content=[TextContent(text=self.PRUNED_TOOL_OUTPUT_MARKER)],
        )

    def process_tool_output(
        self,
        execution_result: ExecutionResult,
    ) -> ExecutionResult:
        """Transform a tool's returned result before it becomes durable.
        Identity by default; override to truncate or replace model-facing
        content (stash the original in `metadata` if your policy keeps it)."""
        return execution_result

    # ── derivation helpers ───────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def _model_facing_text(self, entry: Entry) -> str:
        """Concatenate the model-facing text the entry OWNS (see
        `calculate_context`). Unknown entry types contribute nothing rather
        than failing: calculation is an estimate, not a projection."""
        if isinstance(entry, UserMessage):
            return "".join(
                part.text for part in entry.parts
                if isinstance(part, TextContent)
            )
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
                return "".join(part.text for part in entry.result.content)
            if entry.error is not None:
                return entry.error.error_message
            return ""
        if isinstance(entry, CompactionEntry):
            return entry.summary
        if isinstance(entry, PrunedEntry):
            return "".join(part.text for part in entry.content)
        return ""

    def _media_tokens(self, entry: Entry) -> int:
        """The entry's non-text context contribution: a flat constant per
        image, deliberately dimension-blind. A URL source has no local bytes
        to measure, reading real dimensions would need an image decoder (a
        new dependency), and the provider formulas disagree by an order of
        magnitude. Override with a per-provider formula if it matters."""
        if isinstance(entry, UserMessage):
            return self.IMAGE_TOKENS * sum(
                isinstance(part, ImageContent) for part in entry.parts
            )
        return 0
