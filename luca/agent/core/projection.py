"""Conversation projection: durable entries → canonical luca.client messages.

`ConversationProjector` is the public strategy that derives the LLM message
history from a `Conversation`. It is a first-class runner collaborator (passed
as `conversation_projector=`, never middleware, never serialized) and a
CONCRETE class with complete default behavior: instantiate it directly,
subclass it and override selected methods, or supply another object with the
same behavior. Policies like dropping history, injecting synthetic messages,
redacting content, or changing tool-execution output all belong in a custom
projector.

Projection is deterministic, read-only derivation:

- Walk `conversation.nodes` in order; resolve each id in `entries`; the path
  is the sole ordering authority (`parent_id` is never traversed).
- Every per-entry `project_*` method takes `(entry, entries)`: the resolved,
  typed entry plus the full read-only entry mapping, so any projection can
  resolve cross-entry references. `project_pruned` uses it to fetch a
  `PrunedEntry`'s original and re-emit the replacement content under the
  original's role and correlation.
- A missing id or an unknown entry type raises `ProjectionError` — durable
  state is never silently omitted or replaced with synthetic content.
- Projected messages are request data, derived on every call and never stored
  in `AgentSession`.
- The projector targets canonical `luca.client` DTOs and stops there:
  provider wire formats (OpenAI dicts, Anthropic tool_result blocks) are
  wholly owned by `luca.client` transports.

Tool executions project by `ExecutionStatus`:

- `PENDING` / `RUNNING` are not projectable as tool outputs — raising here is
  the fail-loud guard against calling the model mid-execution;
- `COMPLETED` projects `result.content` and preserves `result.is_error`
  (an `is_error=True` result is still a completed execution);
- every other terminal status projects derived error content with
  `is_error=True`, worded from the class-level defaults below.

`project_tool_execution` has two consumers that must agree: the `ToolMessage`
in the next LLM request and the presentation fields on the `ToolExecuted`
event. It must therefore stay deterministic for the same durable execution —
no wall clock, no live registry, no transient runner state.

All default derived wording lives ON the class (`STATUS_ONLY_OUTPUTS`,
`CANCELLED_TURN_MARKER`, and the FAILED / NOT_FOUND / INVALID derivations in
`project_tool_execution`) so an application can change any of it in one
subclass without touching the runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import ClassVar

from luca.client.types import (
    MediaBase64,
    MediaFileId,
    MediaURL,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolMessage,
)
from luca.client.types import AssistantMessage as ClientAssistantMessage
from luca.client.types import ImageBlock as ClientImageBlock
from luca.client.types import ToolCall as ClientToolCall
from luca.client.types import UserMessage as ClientUserMessage

from .exceptions import ProjectionError
from .models import (
    AnyEntry,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    Conversation,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    ImageFileId,
    ImageURL,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)

# The synthetic user-role text a CANCELLED TurnFinish projects as — the model's
# only view of a user cancel. Module-level alias of the class default.
CANCELLED_TURN_MARKER = "[Request interrupted by user]"


class ConversationProjector:
    """The default projection policy. Every method is an override point."""

    CANCELLED_TURN_MARKER: ClassVar[str] = CANCELLED_TURN_MARKER

    # Derived tool output for the terminal statuses that are complete
    # lifecycle facts on their own (no ToolExecutionError to elaborate with).
    STATUS_ONLY_OUTPUTS: ClassVar[dict[ExecutionStatus, str]] = {
        ExecutionStatus.REJECTED: "[tool execution rejected]",
        ExecutionStatus.CANCELLED: "[tool execution cancelled]",
        ExecutionStatus.INTERRUPTED: "[tool execution interrupted]",
        ExecutionStatus.TIMED_OUT: "[tool execution timed_out]",
    }

    def project(
        self,
        conversation: Conversation,
        entries: Mapping[str, AnyEntry],
    ) -> list[Message]:
        """Project the ordered conversation path to canonical client messages.

        No adjacent-message merging, role folding, trimming, or token counting
        happens here — a custom projector implements such policy by overriding
        this method. Both inputs are read-only."""
        messages: list[Message] = []
        for node_id in conversation.nodes:
            try:
                entry = entries[node_id]
            except KeyError:
                raise ProjectionError(
                    f"Conversation node {node_id!r} is missing from the entry "
                    "store."
                ) from None
            message = self.project_entry(entry, entries)
            if message is not None:
                messages.append(message)
        return messages

    def project_entry(
        self, entry: AnyEntry, entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """Dispatch one durable entry to its entry-specific projection. Every
        per-entry method receives the full read-only entry mapping so a
        projection may resolve cross-entry references (a `PrunedEntry`'s
        original); most default projections ignore it."""
        if isinstance(entry, UserMessage):
            return self.project_user_message(entry, entries)
        if isinstance(entry, AssistantMessage):
            return self.project_assistant_message(entry, entries)
        if isinstance(entry, ToolExecution):
            return self.project_tool_execution(entry, entries)
        if isinstance(entry, CompactionEntry):
            return self.project_compaction(entry, entries)
        if isinstance(entry, PrunedEntry):
            return self.project_pruned(entry, entries)
        if isinstance(entry, TurnFinish):
            return self.project_turn_finish(entry, entries)
        if isinstance(entry, TurnStart):
            return self.project_turn_start(entry, entries)
        if isinstance(entry, CancelRequested):
            return self.project_cancel_requested(entry, entries)
        raise ProjectionError(
            f"Cannot project entry of type {type(entry).__name__}."
        )

    def project_user_message(
        self, entry: UserMessage, entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage:
        """Content parts in order; no names, timestamps, or synthetic prefixes."""
        return ClientUserMessage(
            content=[self._content_block(part) for part in entry.parts],
        )

    def project_assistant_message(
        self, entry: AssistantMessage, entries: Mapping[str, AnyEntry],
    ) -> ClientAssistantMessage:
        """Content parts in order. Durable provenance (usage, stop reason, the
        producing LLMConfig) is not copied into the projected history —
        projection reconstructs conversation content, not response objects."""
        blocks: list = []
        for part in entry.parts:
            if isinstance(part, ThinkingContent):
                blocks.append(ThinkingBlock(text=part.thinking))
            elif isinstance(part, ToolCall):
                blocks.append(
                    ClientToolCall(
                        id=part.id, name=part.name, arguments=part.arguments,
                    ),
                )
            else:
                blocks.append(self._content_block(part))
        return ClientAssistantMessage(content=blocks)

    def project_tool_execution(
        self, entry: ToolExecution, entries: Mapping[str, AnyEntry],
    ) -> ToolMessage:
        """The single customization point for ALL tool-execution statuses.

        Works exclusively from the durable execution — no registry, no tool
        resolution — and always preserves `entry.tool_call_id` as the
        correlation id. Does not validate or repair application-authored
        state: `ExecutionStatus` is the primary projection fact, and state
        that lacks what the rule needs fails loudly rather than being
        mutated into projectability."""
        status = entry.status
        if status in (ExecutionStatus.PENDING, ExecutionStatus.RUNNING):
            raise ProjectionError(
                f"ToolExecution {entry.id!r} is {status.value}; a nonterminal "
                "execution is not projectable as a tool output."
            )
        if status == ExecutionStatus.COMPLETED:
            if entry.result is None:
                raise ProjectionError(
                    f"ToolExecution {entry.id!r} is COMPLETED but carries no "
                    "ExecutionResult."
                )
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[
                    self._content_block(part) for part in entry.result.content
                ],
                is_error=entry.result.is_error,
            )
        # Every other terminal status: derived error content, never stored as
        # an ExecutionResult.
        return ToolMessage(
            tool_call_id=entry.tool_call_id,
            content=[TextBlock(text=self._derived_failure_text(entry))],
            is_error=True,
        )

    def project_compaction(
        self, entry: CompactionEntry, entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage:
        """The durable summary as a synthetic user message; the summarized ids
        and details are bookkeeping and are not included."""
        return ClientUserMessage(content=[TextBlock(text=entry.summary)])

    def project_pruned(
        self, entry: PrunedEntry, entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """Project the replacement content with the ORIGINAL entry's role and
        protocol correlation: the referenced entry is resolved from the store
        (it remains there unchanged — only the path node was replaced), and
        the pruned content takes its place. A pruned tool execution keeps the
        original's `tool_call_id` so multiple pruned outputs preserve the
        ordering and correlation of the original executions; `is_error` is
        False — the replacement marker is neutral content, not a failure. A
        missing referent, a `pruned_entry_type` disagreeing with the referent,
        or an unprojectable source type fails loudly."""
        original = entries.get(entry.pruned_entry_id)
        if original is None:
            raise ProjectionError(
                f"PrunedEntry {entry.id!r} references entry "
                f"{entry.pruned_entry_id!r}, which is missing from the entry "
                "store."
            )
        if original.type != entry.pruned_entry_type:
            raise ProjectionError(
                f"PrunedEntry {entry.id!r} records pruned_entry_type="
                f"{entry.pruned_entry_type!r} but the referenced entry "
                f"{original.id!r} is {original.type!r}."
            )
        content = [self._content_block(part) for part in entry.content]
        if isinstance(original, ToolExecution):
            return ToolMessage(
                tool_call_id=original.tool_call_id,
                content=content,
                is_error=False,
            )
        if isinstance(original, UserMessage):
            return ClientUserMessage(content=content)
        if isinstance(original, AssistantMessage):
            return ClientAssistantMessage(content=content)
        raise ProjectionError(
            f"PrunedEntry {entry.id!r} references an entry of type "
            f"{original.type!r}, which has no pruned projection."
        )

    def project_turn_finish(
        self, entry: TurnFinish, entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """Only a deliberate user cancel is the model's business; COMPLETED,
        TIMED_OUT, and ERRORED closes contribute nothing (work recorded inside
        a failed bracket still projects — entries are visited independently)."""
        if entry.outcome == TurnOutcome.CANCELLED:
            return ClientUserMessage(
                content=[TextBlock(text=self.CANCELLED_TURN_MARKER)],
            )
        return None

    def project_turn_start(
        self, entry: TurnStart, entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """Bookkeeping; no canonical LLM representation."""
        return None

    def project_cancel_requested(
        self, entry: CancelRequested, entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """A durable runtime signal; the completed turn outcome represents the
        cancellation to the model."""
        return None

    # ── derivation helpers ───────────────────────────────────────────────────

    def _derived_failure_text(self, entry: ToolExecution) -> str:
        """Deterministic, status-appropriate wording for a non-COMPLETED
        terminal execution, from `status` and the structured `error`."""
        error = entry.error
        if entry.status == ExecutionStatus.NOT_FOUND:
            if error is not None:
                return error.error_message
            return f"Unknown tool: {entry.raw_tool_call.name!r}."
        if entry.status == ExecutionStatus.INVALID:
            message = (
                error.error_message
                if error is not None
                else f"Arguments for tool {entry.raw_tool_call.name!r} are "
                "invalid."
            )
            errors = error.details.get("errors") if error is not None else None
            if errors:
                return f"{message}\n{json.dumps(errors)}"
            return message
        if entry.status == ExecutionStatus.FAILED:
            if error is not None:
                return (
                    "Tool execution failed: "
                    f"{error.error_type}: {error.error_message}"
                )
            return "[tool execution failed]"
        return self.STATUS_ONLY_OUTPUTS[entry.status]

    def _content_block(self, part) -> TextBlock | ClientImageBlock:
        """Agent content value → canonical client content block. Shared by
        every entry projection: user messages, tool results and pruned
        replacements all carry the same `ContentPart` union."""
        if isinstance(part, TextContent):
            return TextBlock(text=part.text)
        if isinstance(part, ImageContent):
            return self._image_block(part)
        raise ProjectionError(
            f"Cannot project content of type {type(part).__name__}."
        )

    def _image_block(self, part: ImageContent) -> ClientImageBlock:
        """Agent image part → client `ImageBlock`. Override to rewrite media
        (proxy a URL, upload base64 and swap in a file id). `part.metadata` is
        application-owned and is dropped here by design."""
        source = part.source
        if isinstance(source, ImageBase64):
            return ClientImageBlock(
                source=MediaBase64(
                    data=source.data, media_type=source.media_type,
                ),
            )
        if isinstance(source, ImageURL):
            return ClientImageBlock(
                source=MediaURL(url=source.url, media_type=source.media_type),
            )
        if isinstance(source, ImageFileId):
            return ClientImageBlock(
                source=MediaFileId(
                    file_id=source.file_id, media_type=source.media_type,
                ),
            )
        raise ProjectionError(
            f"Cannot project image source of type {type(source).__name__}."
        )


# Presentation-only stand-in for an image when a tool message is flattened for
# an event. Unlike the model-facing markers above this is a module constant,
# not a class var: `tool_message_text` is a free function, so a projector
# subclass cannot change it.
IMAGE_BLOCK_MARKER = "[image]"


def tool_message_text(message: ToolMessage) -> str:
    """Flatten a projected tool message for event presentation: string content
    is used directly; list content concatenates its blocks in order, an image
    contributing a marker so a caller rendering this never silently loses a
    block it cannot draw."""
    if isinstance(message.content, str):
        return message.content
    chunks: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            chunks.append(block.text)
        elif isinstance(block, ClientImageBlock):
            chunks.append(IMAGE_BLOCK_MARKER)
    return "".join(chunks)
