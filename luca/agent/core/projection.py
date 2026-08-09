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

- nonterminal executions (`RECEIVED` / `PENDING` / `RUNNING` /
  `AWAITING_RESULT`) are not projectable as tool outputs — raising here is the
  fail-loud guard against calling the model mid-execution — with EXACTLY TWO
  carve-outs, both the same kind of state: a GATED execution (`PENDING` with
  `approval_status=PENDING`) projects the `AWAITING_APPROVAL_OUTPUT`
  placeholder, and a PARKED one (`AWAITING_RESULT`) projects
  `AWAITING_RESULT_OUTPUT`. Both are durable resting points only the
  application can move rather than runtimes in flight, and the placeholder is
  what lets a message posted while one is open reach the model at all (0008);
- `COMPLETED` projects `result.content` and preserves `result.is_error`
  (an `is_error=True` result is still a completed execution);
- every other terminal status projects derived error content with
  `is_error=True`, worded from the class-level defaults below.

A PRIVATE execution (`tool_spec.is_private`) is dispatched to
`project_private_execution` instead, which projects nothing — see there for why
the `ToolMessage` channel is closed to it by protocol rather than by policy —
with ONE path-level carve-out: an execution named by some
`ChildConversation.result_execution_id` is a subagent resolution, and
`project()` renders it as that link's task update at its own position
(`project_child_update`), which is what keeps the projected history
append-only while the link mutates in place. `project_tool_execution` is still
called directly for the `ToolExecuted` event's presentation fields, so a
private execution's event stays self-describing.

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
from datetime import UTC, datetime
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
    ApprovalStatus,
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
    open_turn_index,
    spawn_payload,
)

# The synthetic user-role text a CANCELLED TurnFinish projects as — the model's
# only view of a user cancel. Module-level alias of the class default.
CANCELLED_TURN_MARKER = "[Request interrupted by user]"


class ConversationProjector:
    """The default projection policy. Every method is an override point."""

    CANCELLED_TURN_MARKER: ClassVar[str] = CANCELLED_TURN_MARKER

    # How a subagent's result reaches the parent's model: one synthetic USER
    # message per RESOLUTION, rendered at the path position of the result-tool
    # execution that produced it (`ChildConversation.result_execution_id`) —
    # after the assistant messages that predate it, so a re-awakened model
    # always finds the update BELOW its own last reply and no earlier message
    # is ever rewritten. The tag's `id` is the spawn payload's `task_id` — the
    # same identifier the spawn confirmation, `list_subagents` and
    # `stop_subagent` speak. Override `project_child_update` /
    # `project_child_conversation` (or just these templates) to change the
    # rendering; all of it is projector policy, not framework behavior.
    CHILD_UPDATE_PREAMBLE: ClassVar[str] = "Subagent task update:\n"
    CHILD_UPDATE_TEMPLATE: ClassVar[str] = "<task id={task_id} status={status}{completed_at}>\n{content}\n</task>"
    CHILD_COMPLETED_AT_TEMPLATE: ClassVar[str] = ' completed_at="{iso}"'

    # Derived tool output for a GATED execution — PENDING with an approval the
    # policy explicitly deferred. Deliberately NOT a row in
    # STATUS_ONLY_OUTPUTS: that table is keyed by TERMINAL status, and this is
    # the single nonterminal state that projects at all. `is_error` stays False
    # where every other derived output sets it True — the call has not failed,
    # it has not run, and an error result is exactly what makes a model retry.
    AWAITING_APPROVAL_OUTPUT: ClassVar[str] = (
        "[tool execution is awaiting approval — it has not run, and this is not its result]"
    )

    # Derived tool output for a PARKED execution — dispatched, deferred, and
    # only the application can move it. The SECOND projectable nonterminal
    # state, for the same reason as the gate's: a durable resting point, not a
    # runtime in flight, and the placeholder is what lets a message posted
    # while the call is out reach the model at all. `is_error` stays False for
    # the same documented reason — the call has not failed, and an error result
    # is exactly what makes a model retry.
    #
    # HEAVIER WORDING THAN THE GATE'S, deliberately. A gate can be terse
    # because the model has no available action — it cannot approve itself.
    # Here it does: a bare "still executing" invites a re-call, and a re-call
    # mints a NEW ToolExecution under a new tool_call_id, leaving the tool
    # holding two open calls for one job. This sentence is the framework's one
    # chance to discourage that.
    AWAITING_RESULT_OUTPUT: ClassVar[str] = (
        "[tool execution has not finished — this is not its result; do not call "
        "it again, the result will follow when it completes]"
    )

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

        Two more path-level rules live here, both about subagents, both
        positional:

        - a `ToolExecution` named by some `ChildConversation`'s
          `result_execution_id` projects that link's TASK UPDATE
          (`project_child_update`) — the resolution renders where it HAPPENED,
          after every assistant message that predates it, which is what keeps
          the projected history append-only while links mutate in place;
        - an UNRESOLVED `ChildConversation` outside the open turn raises. No
          close may leave an unresolved child, so this state is a framework
          bug or hand-authored corruption — the same fail-loud stance a
          nonterminal execution gets. Inside the open turn it is legal and
          renders nothing (the model tracks its tasks through the spawn
          confirmations, the updates, and `list_subagents`).

        No adjacent-message merging, role folding, trimming, or token counting
        happens here — a custom projector implements such policy by overriding
        this method. Both inputs are read-only."""
        links_by_result = {
            entry.result_execution_id: entry
            for entry in entries.values()
            if isinstance(entry, ChildConversation) and entry.result_execution_id is not None
        }
        open_turn = open_turn_index(nodes, entries)
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
            if isinstance(entry, ToolExecution) and (link := links_by_result.get(entry.id)) is not None:
                message = self.project_child_update(link, entry, entries)
                if message is not None:
                    messages.append(message)
                index += 1
                continue
            if (
                isinstance(entry, ChildConversation)
                and entry.execution_result is None
                and (open_turn is None or index < open_turn)
            ):
                raise ProjectionError(
                    f"ChildConversation {entry.id!r} is unresolved outside the open "
                    "turn; no close may leave an unresolved subagent behind."
                )
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
        if status == ExecutionStatus.PENDING and entry.approval_status == ApprovalStatus.PENDING:
            # THE FIRST PROJECTABLE NONTERMINAL STATE. A gated execution is not
            # a runtime bug the way RECEIVED / RUNNING / undecided-PENDING are:
            # it is a durable resting state only the application can move, and
            # the placeholder is what lets a message posted while the gate is
            # open reach the model at all (0008). Replaced by the real result at
            # this same path position once the approval is answered.
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[TextBlock(text=self.AWAITING_APPROVAL_OUTPUT)],
                is_error=False,
            )
        if status == ExecutionStatus.AWAITING_RESULT:
            # THE SECOND, and the same kind of state: the tool was dispatched
            # and answered "not yet", so only the application can move it.
            # Replaced by the real result at this same path position once the
            # tool finally returns one. Nothing else in the framework calls the
            # model with a nonterminal execution on the path — compaction skips
            # any conversation with an open turn, pruning refuses, the context
            # manager leaves it at zero tokens — so only a POST ever forces
            # this, plus `build_messages()` being public.
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[TextBlock(text=self.AWAITING_RESULT_OUTPUT)],
                is_error=False,
            )
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

        Whether it projects as anything ELSE is policy, and the default answer
        is no — with one carve-out that does not pass through here: a private
        execution named by some `ChildConversation.result_execution_id` is a
        subagent resolution, and `project()`'s path rule renders it as that
        link's task update (`project_child_update`) before this method is ever
        consulted. Every other private execution projects nothing. Override
        this to render private work some other way; a synthetic USER message
        is the shape this framework already uses for framework-authored
        content, and it is available. The `ToolMessage` channel is closed; the
        entry is not structurally invisible."""
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
    ) -> ClientUserMessage | None:
        """A `ChildConversation` link at its OWN path position. Projects
        nothing in the two ordinary states:

        - UNRESOLVED — the child is still working; the model already saw the
          spawn confirmation, and its result will render when it arrives.
          (`project()` separately refuses this state outside the open turn.)
        - resolved WITH a `result_execution_id` — that execution's position
          owns the rendering (`project_child_update`); rendering here too
          would say it twice, at a position that predates the wake rounds.

        The one state the link itself renders is resolved WITHOUT a result
        execution — the cancel wind-down and the hard-limit settle write the
        result directly, appending nothing — so the outcome still reaches the
        model, at the link's position, without a timestamp. Those resolutions
        only happen inside FAILING brackets, and the placement is knowingly
        early: wake rounds recorded while the child was still alive sit after
        the link on the path, so the next turn's projection shows the
        cancellation notice before them. Accepted — the whole bracket already
        reads as a failure, and the alternative (appending a synthetic entry
        from a close path) would put content authorship inside the
        wind-down."""
        if entry.execution_result is None or entry.result_execution_id is not None:
            return None
        return self._child_task_message(entry, entries, completed_at_ms=None)

    def project_child_update(
        self,
        link: ChildConversation,
        execution: ToolExecution,
        entries: Mapping[str, AnyEntry],
    ) -> ClientUserMessage | None:
        """A subagent's resolution, rendered as a synthetic user message at
        the RESULT EXECUTION's path position — dispatched by `project()`'s
        path rule, never by `project_entry`.

        A synthetic USER message is the established shape for
        framework-authored content (`project_compaction` and the
        cancelled-turn marker both use it), and it is the only legal shape
        here: the result tool is private, so no `ToolCall` for it exists in
        any assistant message and a `ToolMessage` would be a protocol
        violation. Rendering at the resolution's own position is what makes a
        re-awakened model always find the update AFTER its last reply, and
        what keeps every earlier projected message byte-stable while the
        orchestration is still running."""
        if link.execution_result is None:  # unreachable: stamped atomically
            return None
        return self._child_task_message(link, entries, completed_at_ms=execution.finished_at)

    def _child_task_message(
        self,
        link: ChildConversation,
        entries: Mapping[str, AnyEntry],
        *,
        completed_at_ms: int | None,
    ) -> ClientUserMessage:
        """The task-update message for one resolved link: the preamble plus a
        `<task>` tag wrapping the result's text, non-text blocks following.

        The tag's `id` is the spawn payload's `task_id` — the identifier the
        model already met in the spawn confirmation and can hand to
        `stop_subagent`. `status` is the result's own verdict
        (completed / failed); `completed_at` renders only when the resolution
        has a timestamp to report, as absolute UTC — deterministic for the
        same durable state, unlike a relative "10 seconds ago"."""
        spawn = entries.get(link.tool_execution_id)
        if not isinstance(spawn, ToolExecution):
            raise ProjectionError(
                f"ChildConversation {link.id!r} references tool execution "
                f"{link.tool_execution_id!r}, which is missing from the entry "
                "store."
            )
        payload = spawn_payload(spawn)
        if payload is None:
            raise ProjectionError(
                f"ChildConversation {link.id!r}'s spawn execution {spawn.id!r} carries no spawn payload."
            )
        completed_at = (
            self.CHILD_COMPLETED_AT_TEMPLATE.format(iso=_iso_utc(completed_at_ms))
            if completed_at_ms is not None
            else ""
        )
        blocks = [self._content_block(part) for part in link.execution_result.content]
        text = "".join(block.text for block in blocks if isinstance(block, TextBlock))
        wrapped: list = [
            TextBlock(
                text=self.CHILD_UPDATE_PREAMBLE
                + self.CHILD_UPDATE_TEMPLATE.format(
                    task_id=payload["task_id"],
                    status="failed" if link.execution_result.is_error else "completed",
                    completed_at=completed_at,
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
            if original.tool_spec is not None and original.tool_spec.is_private:
                # A private execution's ToolMessage channel is closed by
                # protocol (no ToolCall for it exists), and a runner-minted
                # correlation id on the wire would be rejected by every
                # provider. Pruning one — a subagent-result execution is the
                # natural target — simply removes its contribution.
                return None
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


def _iso_utc(ms: int) -> str:
    """Unix ms → an absolute UTC ISO-8601 second stamp — deterministic for the
    same durable state, which relative wording ("10 seconds ago") cannot be."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
