"""Conversation projection: durable entries → canonical luca.client messages.

`ConversationProjector` is the public strategy that derives the LLM message
history from a conversation's path. It is a first-class runner collaborator (passed
as `conversation_projector=`, never middleware, never serialized) and a
CONCRETE class with complete default behavior: instantiate it directly,
subclass it and override selected methods, or supply another object with the
same behavior. Policies like dropping history, injecting synthetic messages,
redacting content, or changing tool-execution output all belong in a custom
projector.

`project()` takes the PATH — an ordered list of entry ids — rather than a
`Conversation` object. Two reasons: a `Conversation` is live and mutable (the
runner appends to its `nodes` while a projector holds it), and every other
scoped signature in the framework passes an id or plain data; and a caller
that wants to project a SPAN it invented — `SummarizingContextManager`
projecting the nodes it is about to fold — has no conversation to hand over.

Projection is deterministic, read-only derivation:

- Walk `nodes` in order; resolve each id in `entries`; the path is the sole
  ordering authority (`parent_id` is never traversed).
- One path-level rule lives on `project()` itself, because it cannot be
  decided from a single entry: a COMPACTION BRACKET (`ts_c cmp [cr] tf_c`,
  whatever its outcome) projects as nothing, while a `CompactionEntry` outside
  a bracket projects its `parts` as a synthetic user message. That is what
  keeps an archived conversation projecting its originals rather than
  originals-plus-summary, and what stops a cancelled compaction from putting
  the interrupted marker on the wire.
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

A PRIVATE execution (`tool_spec.is_private`) is dispatched to
`project_private_execution` instead, which projects nothing — see there for why
the `ToolMessage` channel is closed to it by protocol rather than by policy.
`project_tool_execution` is still called directly for the `ToolExecuted` event's
presentation fields, so a private execution's event stays self-describing.

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
from collections.abc import Mapping, Sequence
from typing import ClassVar

from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    ImageBlock as ClientImageBlock,
    MediaBase64,
    MediaFileId,
    MediaURL,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall as ClientToolCall,
    ToolMessage,
    UserMessage as ClientUserMessage,
)

from .exceptions import ProjectionError
from .models import (
    NONTERMINAL_STATUSES,
    AnyEntry,
    AssistantMessage,
    CancelRequested,
    ChildConversation,
    CompactionEntry,
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
    is_compaction_bracket,
)

# The synthetic user-role text a CANCELLED TurnFinish projects as — the model's
# only view of a user cancel. Module-level alias of the class default.
CANCELLED_TURN_MARKER = "[Request interrupted by user]"


class ConversationProjector:
    """The default projection policy. Every method is an override point."""

    CANCELLED_TURN_MARKER: ClassVar[str] = CANCELLED_TURN_MARKER

    # How a finished subagent's result reaches the parent's model. One
    # synthetic USER message per child, tagged with the tool call that spawned
    # it so the model can correlate the answer with its own request. Change
    # this — or override `project_child_conversation` outright — to include
    # more of a child's transcript, or to batch several children into one
    # message; both are projector policy, not framework behavior.
    CHILD_TASK_TEMPLATE: ClassVar[str] = "<task id={task_id}>\n{content}\n</task>"

    # Derived tool output for the terminal statuses that are complete
    # lifecycle facts on their own (no ToolExecutionError to elaborate with).
    STATUS_ONLY_OUTPUTS: ClassVar[dict[ExecutionStatus, str]] = {
        ExecutionStatus.REJECTED: "[tool execution rejected]",
        ExecutionStatus.REFUSED: "[tool execution refused]",
        ExecutionStatus.CANCELLED: "[tool execution cancelled]",
        ExecutionStatus.INTERRUPTED: "[tool execution interrupted]",
        ExecutionStatus.TIMED_OUT: "[tool execution timed_out]",
    }

    def project(
        self,
        nodes: Sequence[str],
        entries: Mapping[str, AnyEntry],
    ) -> list[Message]:
        """Project an ordered path of entry ids to canonical client messages.

        One path-level rule lives here, and it is POSITIONAL: a compaction
        bracket — the whole span `ts_c cmp [cr] tf_c`, whatever the outcome —
        projects as nothing, while a `CompactionEntry` reached OUTSIDE a
        bracket projects its `parts`. A summary only means something on the
        path where the history it replaces is gone; inside its bracket the
        entry is the record of the operation, on a path that still holds the
        originals. Deciding that needs the path, which the per-entry methods
        deliberately do not receive.

        No adjacent-message merging, role folding, trimming, or token counting
        happens here — a custom projector implements such policy by overriding
        this method. Both inputs are read-only."""
        messages: list[Message] = []
        index = 0
        while index < len(nodes):
            entry = self._node(nodes[index], entries)
            if isinstance(entry, TurnStart) and is_compaction_bracket(
                nodes,
                entries,
                index,
            ):
                index = self._skip_compaction_bracket(nodes, entries, index)
                continue
            message = self.project_entry(entry, entries)
            if message is not None:
                messages.append(message)
            index += 1
        return messages

    def _skip_compaction_bracket(
        self,
        nodes: Sequence[str],
        entries: Mapping[str, AnyEntry],
        start: int,
    ) -> int:
        """The index just past the compaction bracket opened at `start` — its
        closing `TurnFinish`, or the end of the path when the bracket is still
        open.

        Skipping the whole span is what makes projection CORRECT, not merely
        tidy: `project_turn_finish` emits the interrupted marker for any
        CANCELLED close, and a cancelled compaction never transitions, so
        without this every later request would tell the model its question was
        interrupted — about a question the model was never shown. Nothing is
        lost by discarding the span: only markers can ever be inside one,
        because `post_message` raises while a compaction bracket is open."""
        index = start + 1
        while index < len(nodes):
            entry = self._node(nodes[index], entries)
            index += 1
            if isinstance(entry, TurnFinish):
                break
        return index

    def _node(self, node_id: str, entries: Mapping[str, AnyEntry]) -> AnyEntry:
        try:
            return entries[node_id]
        except KeyError:
            raise ProjectionError(f"Conversation node {node_id!r} is missing from the entry store.") from None

    def project_entry(
        self,
        entry: AnyEntry,
        entries: Mapping[str, AnyEntry],
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
            if entry.tool_spec is not None and entry.tool_spec.is_private:
                return self.project_private_execution(entry, entries)
            return self.project_tool_execution(entry, entries)
        if isinstance(entry, CompactionEntry):
            return self.project_compaction(entry, entries)
        if isinstance(entry, ChildConversation):
            return self.project_child_conversation(entry, entries)
        if isinstance(entry, PrunedEntry):
            return self.project_pruned(entry, entries)
        if isinstance(entry, TurnFinish):
            return self.project_turn_finish(entry, entries)
        if isinstance(entry, TurnStart):
            return self.project_turn_start(entry, entries)
        if isinstance(entry, CancelRequested):
            return self.project_cancel_requested(entry, entries)
        raise ProjectionError(f"Cannot project entry of type {type(entry).__name__}.")

    def project_user_message(
        self,
        entry: UserMessage,
        entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage:
        """Content parts in order; no names, timestamps, or synthetic prefixes."""
        return ClientUserMessage(
            content=[self._content_block(part) for part in entry.parts],
        )

    def project_assistant_message(
        self,
        entry: AssistantMessage,
        entries: Mapping[str, AnyEntry],
    ) -> ClientAssistantMessage:
        """Content parts in order, plus the producing model as provenance.
        Usage and stop reason are not copied — projection reconstructs
        conversation content, not response objects.

        `provider` / `model` ARE copied, and they are load-bearing rather than
        decorative: a `ThinkingContent` carries opaque attestations (an
        Anthropic signature, an OpenAI `rs_…` id plus encrypted payload) that
        only the pair which minted them accepts back, and the transport decides
        replay eligibility by comparing this provenance against the model being
        called. Without it, switching model mid-session (`/model` in the TUI)
        would replay a foreign attestation and every later request would 400."""
        blocks: list = []
        for part in entry.parts:
            if isinstance(part, ThinkingContent):
                blocks.append(
                    ThinkingBlock(
                        text=part.thinking,
                        id=part.id,
                        signature=part.signature,
                        redacted=part.redacted,
                    ),
                )
            elif isinstance(part, ToolCall):
                blocks.append(
                    ClientToolCall(
                        id=part.id,
                        name=part.name,
                        arguments=part.arguments,
                    ),
                )
            else:
                blocks.append(self._content_block(part))
        return ClientAssistantMessage(
            content=blocks,
            provider=entry.llm_config.provider,
            model=entry.llm_config.model,
        )

    def project_tool_execution(
        self,
        entry: ToolExecution,
        entries: Mapping[str, AnyEntry],
    ) -> ToolMessage:
        """The single customization point for ALL tool-execution statuses.

        Works exclusively from the durable execution — no registry, no tool
        resolution — and always preserves `entry.tool_call_id` as the
        correlation id. Does not validate or repair application-authored
        state: `ExecutionStatus` is the primary projection fact, and state
        that lacks what the rule needs fails loudly rather than being
        mutated into projectability."""
        status = entry.status
        if status in NONTERMINAL_STATUSES:
            raise ProjectionError(
                f"ToolExecution {entry.id!r} is {status.value}; a nonterminal "
                "execution is not projectable as a tool output."
            )
        if status == ExecutionStatus.COMPLETED:
            if entry.result is None:
                raise ProjectionError(f"ToolExecution {entry.id!r} is COMPLETED but carries no ExecutionResult.")
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[self._content_block(part) for part in entry.result.content],
                is_error=entry.result.is_error,
            )
        # Every other terminal status: derived error content, never stored as
        # an ExecutionResult.
        return ToolMessage(
            tool_call_id=entry.tool_call_id,
            content=[TextBlock(text=self._derived_failure_text(entry))],
            is_error=True,
        )

    def project_private_execution(
        self,
        entry: ToolExecution,
        entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """A PRIVATE tool's execution. Projects as NOTHING by default.

        That it never projects as a `ToolMessage` is FORCED, not policy: a
        private tool is invoked by the runtime, so no `ToolCall` for it exists
        in any `AssistantMessage` on the path, and a tool result carrying a
        `tool_call_id` the provider never issued is a protocol violation every
        provider rejects. There is no projector setting that makes that legal.

        Whether it projects as anything ELSE is policy, and V0's answer is no —
        for one specific reason: its output already reaches the model through
        the entry that owns it (a subagent result travels on
        `ChildConversation.execution_result`), so projecting it again would
        duplicate the content rather than add it. Override this to render
        private work some other way; a synthetic USER message is the shape this
        framework already uses for framework-authored content, and it is
        available. The `ToolMessage` channel is closed; the entry is not
        structurally invisible."""
        return None

    def project_compaction(
        self,
        entry: CompactionEntry,
        entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage | None:
        """The durable summary as a synthetic user message; `compacted_nodes`,
        `llm_config` and `metadata` are bookkeeping and are not included.

        `None` when the entry carries no content — scheduled, running, a
        no-op, or failed. `parts` land only at the commit point, so this is
        also what stops a failed compaction from ever telling the model "here
        is a summary of the conversation so far"."""
        if not entry.parts:
            return None
        return ClientUserMessage(
            content=[self._content_block(part) for part in entry.parts],
        )

    def project_child_conversation(
        self,
        entry: ChildConversation,
        entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage:
        """A finished subagent's result, as a synthetic user message.

        A synthetic USER message is the established shape for framework-authored
        content in this framework — `project_compaction` and the cancelled-turn
        marker both use it — and it is the only legal shape here: the spawn tool
        already got its own `ToolMessage`, and a second one correlating to the
        same `tool_call_id` would be a protocol violation.

        An UNRESOLVED child raises. That is the same fail-loud rule a PENDING
        `ToolExecution` gets, for the same reason: the runtime must never call
        the model while a child is still working, and inventing a placeholder
        would tell the model a subagent answered when it has not. The runner's
        drive blocks on its children precisely so this is unreachable in
        practice."""
        if entry.execution_result is None:
            raise ProjectionError(
                f"ChildConversation {entry.id!r} has no execution_result; an unresolved subagent is not projectable."
            )
        execution = entries.get(entry.tool_execution_id)
        if not isinstance(execution, ToolExecution):
            raise ProjectionError(
                f"ChildConversation {entry.id!r} references tool execution "
                f"{entry.tool_execution_id!r}, which is missing from the entry "
                "store."
            )
        blocks = [self._content_block(part) for part in entry.execution_result.content]
        text = "".join(block.text for block in blocks if isinstance(block, TextBlock))
        wrapped: list = [
            TextBlock(
                text=self.CHILD_TASK_TEMPLATE.format(
                    task_id=execution.tool_call_id,
                    content=text,
                ),
            ),
        ]
        wrapped += [block for block in blocks if not isinstance(block, TextBlock)]
        return ClientUserMessage(content=wrapped)

    def project_pruned(
        self,
        entry: PrunedEntry,
        entries: Mapping[str, AnyEntry],
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
            # Same provenance rule as project_assistant_message: an assistant
            # message on the wire says which model produced it, pruned or not.
            return ClientAssistantMessage(
                content=content,
                provider=original.llm_config.provider,
                model=original.llm_config.model,
            )
        raise ProjectionError(
            f"PrunedEntry {entry.id!r} references an entry of type {original.type!r}, which has no pruned projection."
        )

    def project_turn_finish(
        self,
        entry: TurnFinish,
        entries: Mapping[str, AnyEntry],
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
        self,
        entry: TurnStart,
        entries: Mapping[str, AnyEntry],
    ) -> Message | None:
        """Bookkeeping; no canonical LLM representation."""
        return None

    def project_cancel_requested(
        self,
        entry: CancelRequested,
        entries: Mapping[str, AnyEntry],
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
                else f"Arguments for tool {entry.raw_tool_call.name!r} are invalid."
            )
            errors = error.details.get("errors") if error is not None else None
            if errors:
                return f"{message}\n{json.dumps(errors)}"
            return message
        if entry.status == ExecutionStatus.FAILED:
            if error is not None:
                return f"Tool execution failed: {error.error_type}: {error.error_message}"
            return "[tool execution failed]"
        if entry.status == ExecutionStatus.REFUSED and error is not None:
            # the limit's own wording, verbatim — the model must read the real
            # reason ("Spawn limit reached (3/3)…"), not a placeholder
            return error.error_message
        return self.STATUS_ONLY_OUTPUTS[entry.status]

    def _content_block(self, part) -> TextBlock | ClientImageBlock:
        """Agent content value → canonical client content block. Shared by
        every entry projection: user messages, tool results and pruned
        replacements all carry the same `ContentPart` union."""
        if isinstance(part, TextContent):
            return TextBlock(text=part.text)
        if isinstance(part, ImageContent):
            return self._image_block(part)
        raise ProjectionError(f"Cannot project content of type {type(part).__name__}.")

    def _image_block(self, part: ImageContent) -> ClientImageBlock:
        """Agent image part → client `ImageBlock`. Override to rewrite media
        (proxy a URL, upload base64 and swap in a file id). `part.metadata` is
        application-owned and is dropped here by design."""
        source = part.source
        if isinstance(source, ImageBase64):
            return ClientImageBlock(
                source=MediaBase64(
                    data=source.data,
                    media_type=source.media_type,
                ),
            )
        if isinstance(source, ImageURL):
            return ClientImageBlock(
                source=MediaURL(url=source.url, media_type=source.media_type),
            )
        if isinstance(source, ImageFileId):
            return ClientImageBlock(
                source=MediaFileId(
                    file_id=source.file_id,
                    media_type=source.media_type,
                ),
            )
        raise ProjectionError(f"Cannot project image source of type {type(source).__name__}.")


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
