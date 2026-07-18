"""SessionLedger — the single door onto a session's entry log.

WRITES. Every entry append goes through `append()`, which stamps the injected
id/clock, links `parent_id` to the current path leaf, extends
`Conversation.nodes`, indexes a `ToolExecution` into
`session.tool_executions`, and touches `Conversation.updated_at` — one code
path, so the bookkeeping can't drift between call sites.
`put_execution()` is the only in-place mutation door (`ToolExecution` is the
only mutable entry): it stores the fully formed replacement the runner hands
it — the runner owns building the updated copy, stamping `updated_at`, and
threading it through `before_entry_written` middleware first — and touches
`Conversation.updated_at`.
`record_usage()` is the single write door onto `AgentSession.usages`: it
builds the record itself (conversation id from the active conversation,
entry id from an entry it verifies is on the path), so the store's key/record
agreement and referential invariants cannot drift across call sites.
`prune()` is the single path-replacement door: it stamps a `PrunedEntry`
built by the caller's callback and swaps it for the original node id IN
PLACE — the original entry stays untouched in `session.entries` (and, for a
tool execution, in the `tool_executions` index); only the path stops
visiting it.

READS. The entry-derived queries — open turn, the execution-lifecycle and
approval-state subsets, derived status. These are pure functions of the
session data, equally valid on a freshly deserialized session; they are the
durable truth the runner's `Conversation.status` cache is re-derived from.
Usage and context aggregates deliberately have no reads here: totals are
trivially derived by the application from `AgentSession.usages` and
`Entry.context_tokens` over a conversation's nodes.

Execution vocabulary (over `status` + `approval_status`):
- PENDING — body not started, no terminal outcome. Subsets by approval:
  - UNDECIDED (`approval_status` None or PENDING) — what the runner offers
    to the permission policy;
  - AWAITING (`approval_status` PENDING) — the policy explicitly deferred;
    the application resolves out-of-band. Drives AWAITING_APPROVAL. A
    never-asked execution (None) derives plain PENDING instead, because a
    plain `run()` self-heals it;
  - READY (`approval_status` ALLOWED) — dispatchable.
- RUNNING — body started, no terminal outcome. Any RUNNING execution seen at
  the start of a drive is an orphan (its live task is gone) and is recovered
  to INTERRUPTED.
Approval state is always read from `approval_status` — never reconstructed
from the `approval_decisions` audit log.
"""

from __future__ import annotations

from collections.abc import Callable

from .exceptions import AgentError
from .models import (
    AgentSession,
    AnyEntry,
    ApprovalStatus,
    CancelRequested,
    ConversationStatus,
    ExecutionStatus,
    PrunedEntry,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)


class SessionLedger:
    """Append/read companion over one `AgentSession`. Owns no policy and no
    loop logic — only the entry-log bookkeeping and the derived-state queries."""

    def __init__(
        self,
        session: AgentSession,
        clock: Callable[[], int],
        gen_id: Callable[[], str],
    ) -> None:
        self.session = session
        self.clock = clock
        self.gen_id = gen_id

    # ── writes ───────────────────────────────────────────────────────────────

    def append(self, build: Callable[[str, str | None, int], AnyEntry]) -> AnyEntry:
        """Append one entry to the path. `build(entry_id, parent_id, ts)`
        constructs the entry; the ledger does everything around it."""
        ts = self.clock()
        entry_id = self.gen_id()
        conversation = self.session.active_conversation
        parent_id = conversation.nodes[-1] if conversation.nodes else None
        entry = build(entry_id, parent_id, ts)
        self.session.entries[entry_id] = entry
        conversation.nodes.append(entry_id)
        conversation.updated_at = ts
        if isinstance(entry, ToolExecution):
            self.session.tool_executions.setdefault(
                entry.tool_call_id, [],
            ).append(entry_id)
        return entry

    def put_execution(self, execution: ToolExecution) -> ToolExecution:
        """Store a fully formed `ToolExecution` replacement under its id and
        touch `Conversation.updated_at`. The caller has already stamped
        `updated_at` and run persistence middleware."""
        self.session.entries[execution.id] = execution
        self.session.active_conversation.updated_at = self.clock()
        return execution

    def record_usage(self, entry_id: str, **counters: int) -> Usage:
        """Write the provider-usage record for `entry_id` in the ACTIVE
        conversation. The door builds the `Usage` itself — outer key ==
        `conversation_id`, inner key == `entry_id`, entry verified to exist
        on the conversation's path — so the store's invariants hold at every
        call site. At most one record per (conversation, entry) pair: a
        re-record replaces."""
        conversation = self.session.active_conversation
        if entry_id not in self.session.entries:
            raise AgentError(
                f"Cannot record usage for entry {entry_id!r}: no such entry."
            )
        if entry_id not in conversation.nodes:
            raise AgentError(
                f"Cannot record usage for entry {entry_id!r}: the entry is "
                f"not on conversation {conversation.id!r}'s path."
            )
        usage = Usage(
            conversation_id=conversation.id, entry_id=entry_id, **counters,
        )
        self.session.usages.setdefault(conversation.id, {})[entry_id] = usage
        return usage

    def prune(
        self,
        original_id: str,
        build: Callable[[str, str | None, int], PrunedEntry],
    ) -> PrunedEntry:
        """Replace `original_id`'s node in the active path with a durable
        `PrunedEntry`. `build(entry_id, parent_id, ts)` constructs the entry
        (the caller threads context calculation and entry middleware inside
        it, exactly like `append`); the door stamps identity — the pruned
        entry takes the ORIGINAL's `parent_id`, since it occupies the
        original's position — verifies the replacement (`pruned_entry_id`
        must reference the original, `pruned_entry_type` must agree with it,
        and a pruned tool execution must be terminal), stores it in
        `session.entries`, swaps the node id in place, and touches
        `Conversation.updated_at`. The original entry itself is never
        mutated or deleted."""
        conversation = self.session.active_conversation
        original = self.session.entries.get(original_id)
        if original is None:
            raise AgentError(
                f"Cannot prune entry {original_id!r}: no such entry."
            )
        try:
            node_index = conversation.nodes.index(original_id)
        except ValueError:
            raise AgentError(
                f"Cannot prune entry {original_id!r}: the entry is not on "
                f"conversation {conversation.id!r}'s path."
            ) from None
        ts = self.clock()
        entry = build(self.gen_id(), original.parent_id, ts)
        if entry.pruned_entry_id != original_id:
            raise AgentError(
                f"PrunedEntry references {entry.pruned_entry_id!r} but is "
                f"replacing {original_id!r}."
            )
        if entry.pruned_entry_type != original.type:
            raise AgentError(
                f"PrunedEntry records pruned_entry_type="
                f"{entry.pruned_entry_type!r} but entry {original_id!r} is "
                f"{original.type!r}."
            )
        if isinstance(original, ToolExecution) and original.status in (
            ExecutionStatus.PENDING,
            ExecutionStatus.RUNNING,
        ):
            raise AgentError(
                f"Cannot prune ToolExecution {original_id!r}: a nonterminal "
                f"({original.status.value}) execution is not prunable."
            )
        self.session.entries[entry.id] = entry
        conversation.nodes[node_index] = entry.id
        conversation.updated_at = ts
        return entry

    # ── reads (entry-derived state) ─────────────────────────────────────────

    def open_turn_index(self) -> int | None:
        """Index of the TurnStart opening an unclosed turn, or None. Walking
        back from the leaf, a TurnFinish means the last turn is closed; a
        TurnStart seen first means that turn is still open."""
        nodes = self.session.active_conversation.nodes
        entries = self.session.entries
        for i in range(len(nodes) - 1, -1, -1):
            entry = entries[nodes[i]]
            if isinstance(entry, TurnFinish):
                return None
            if isinstance(entry, TurnStart):
                return i
        return None

    def open_turn_executions(self) -> list[ToolExecution]:
        """Every ToolExecution in the open turn, in path order."""
        idx = self.open_turn_index()
        if idx is None:
            return []
        nodes = self.session.active_conversation.nodes
        entries = self.session.entries
        return [
            entries[node_id]
            for node_id in nodes[idx:]
            if isinstance(entries[node_id], ToolExecution)
        ]

    def open_turn_pending_executions(self) -> list[ToolExecution]:
        """Status PENDING — not dispatched, not terminal. The cancel
        wind-down's input."""
        return [
            execution
            for execution in self.open_turn_executions()
            if execution.status == ExecutionStatus.PENDING
        ]

    def open_turn_running_executions(self) -> list[ToolExecution]:
        """Status RUNNING. At the start of a drive these are orphans — the
        body's live task no longer exists — and are recovered to INTERRUPTED."""
        return [
            execution
            for execution in self.open_turn_executions()
            if execution.status == ExecutionStatus.RUNNING
        ]

    def open_turn_undecided_executions(self) -> list[ToolExecution]:
        """PENDING executions the permission policy should be offered:
        `approval_status` is None (never processed) or PENDING (deferred)."""
        return [
            execution
            for execution in self.open_turn_pending_executions()
            if execution.approval_status
            in (None, ApprovalStatus.PENDING)
        ]

    def open_turn_awaiting_executions(self) -> list[ToolExecution]:
        """PENDING executions whose `approval_status` is PENDING — the policy
        explicitly deferred, so the application must resolve out-of-band
        before the turn can finish."""
        return [
            execution
            for execution in self.open_turn_pending_executions()
            if execution.approval_status == ApprovalStatus.PENDING
        ]

    def open_turn_ready_executions(self) -> list[ToolExecution]:
        """PENDING executions cleared for dispatch: `approval_status` ALLOWED."""
        return [
            execution
            for execution in self.open_turn_pending_executions()
            if execution.approval_status == ApprovalStatus.ALLOWED
        ]

    def has_awaiting_approval(self) -> bool:
        return bool(self.open_turn_awaiting_executions())

    def open_turn_has_doom_loop_flagged(self) -> bool:
        """True if any ToolExecution in the open turn is doom-loop-flagged."""
        return any(
            execution.is_doom_loop_flagged
            for execution in self.open_turn_executions()
        )

    def open_turn_cancel_requested(self) -> CancelRequested | None:
        """The unconsumed `CancelRequested` inside the open turn, or None
        (no open turn, or none requested). At most one can exist — cancel()
        refuses to stack a second; instances in closed turns are consumed."""
        idx = self.open_turn_index()
        if idx is None:
            return None
        nodes = self.session.active_conversation.nodes
        entries = self.session.entries
        for node_id in nodes[idx:]:
            entry = entries[node_id]
            if isinstance(entry, CancelRequested):
                return entry
        return None

    def derive_status(self) -> ConversationStatus:
        """The authoritative status, computed from the entries (used to
        normalize a loaded session and as the runner guard's source of truth).
        Precedence: an unconsumed cancel beats the approval gate beats the
        plain open-turn resume; a closed turn is IDLE unless it failed
        (TIMED_OUT / ERRORED → retry-ready PENDING) or a user message is
        already queued behind it."""
        if self.open_turn_index() is not None:
            if self.open_turn_cancel_requested() is not None:
                return ConversationStatus.CANCELLING
            if self.has_awaiting_approval():
                return ConversationStatus.AWAITING_APPROVAL
            return ConversationStatus.PENDING
        nodes = self.session.active_conversation.nodes
        if not nodes:
            return ConversationStatus.IDLE
        last = self.session.entries[nodes[-1]]
        if isinstance(last, TurnFinish) and last.outcome in (
            TurnOutcome.TIMED_OUT,
            TurnOutcome.ERRORED,
        ):
            return ConversationStatus.PENDING
        if isinstance(last, UserMessage):
            return ConversationStatus.PENDING
        return ConversationStatus.IDLE
