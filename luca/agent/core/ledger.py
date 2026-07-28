"""SessionLedger — the single door onto a session's entry log.

WRITES. Every entry append goes through `append()`, which stamps the injected
id/clock, links `parent_id` to the current path leaf, extends
`Conversation.nodes`, indexes a `ToolExecution` into
`session.tool_executions`, and touches `Conversation.updated_at` — one code
path, so the bookkeeping can't drift between call sites.
`put_entry()` is the only in-place mutation door, serving both mutable entry
types (`ToolExecution` and `CompactionEntry`): it stores the fully formed
replacement the runner hands it — the runner owns building the updated copy,
stamping `updated_at`, and threading it through `before_entry_written`
middleware first — and touches `Conversation.updated_at`. It refuses to
create, so an uncommitted template can never become durable through it.
`transition_conversation()` archives the active conversation and installs a
new one over a given path: the only writer of `conversation_history`, the only
replacer of `active_conversation`, and the atomic region every fallible
operation must happen before.
`record_usage()` is the single write door onto `AgentSession.usages`: it
builds the record itself (conversation id from the active conversation,
entry id from an entry it verifies is on the path), so the store's key/record
agreement and referential invariants cannot drift across call sites.
`prune()` is the single path-replacement door: it stamps a `PrunedEntry`
built by the caller's callback and swaps it for the original node id IN
PLACE — the original entry stays untouched in `session.entries` (and, for a
tool execution, in the `tool_executions` index); only the path stops
visiting it.
`refresh_entry()` is the derived-field door: it replaces an entry in the
store for a recalculation that changes nothing an application asked for (the
runner's `recalculate_context_tokens()`), deliberately separate from
`put_entry`, which is documented as the MUTATION door for the two mutable
entry types.

TOOL SPECS. Every door that puts a `ToolExecution` into the store files its
`tool_spec` in `session.tool_specs` under the spec's content-derived id and
stamps that id on the execution — `append` (an execution's birth),
`put_entry` (every later update), `transition_conversation` (a compaction
plan that carries or creates one), and `refresh_entry`. Writing the same spec
again is a no-op: identical content produces an identical id. The id is
recomputed on every write and never short-circuited when one is already set —
an execution's spec can be replaced between writes (middleware may rewrite
it), and a skipped recompute would leave a stale id pointing at the previous
version. Each door also points the execution's `tool_spec` at the STORED
instance, so `session.tool_specs[e.tool_spec_id] is e.tool_spec` holds in
memory exactly as it does after a reload. `prune()` is not one of these doors:
it only ever writes a `PrunedEntry`. Registries are on none of them — they
return drafts with `tool_spec` populated and never compute an id or touch the
store.

READS. The entry-derived queries — open turn, the execution-lifecycle and
approval-state subsets, the resumable compaction, derived status. These are
pure functions of the session data, equally valid on a freshly deserialized
session; they are the durable truth the runner's `Conversation.status` cache
is re-derived from.
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
    CompactionEntry,
    Conversation,
    ConversationStatus,
    ExecutionStatus,
    PrunedEntry,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
    is_compaction_bracket,
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

    def _store_tool_spec(self, entry: AnyEntry) -> None:
        """File a `ToolExecution`'s spec in `session.tool_specs`, point the
        execution at the STORED instance, and stamp the matching
        `tool_spec_id`. A no-op for every other entry type.

        Always recomputes: hashing a few KB costs on the order of ten
        microseconds, and a short-circuit on an already-set id would leave a
        stale reference behind whenever the spec was replaced.

        Re-pointing `tool_spec` is what makes
        `session.tool_specs[e.tool_spec_id] is e.tool_spec` hold IN MEMORY and
        not only after a reload: a registry mints a fresh `ToolSpec` per call,
        so without it the second execution of a repeated tool would keep its
        own equal-but-distinct copy while the first holds the stored one — the
        same session shaped two different ways depending on whether it had
        been through a save. `AgentSession`'s load validator does exactly this
        for a restored session."""
        if not isinstance(entry, ToolExecution):
            return
        if entry.tool_spec is None:
            entry.tool_spec_id = None
            return
        spec_id = entry.tool_spec.spec_id()
        entry.tool_spec = self.session.tool_specs.setdefault(spec_id, entry.tool_spec)
        entry.tool_spec_id = spec_id

    def append(self, build: Callable[[str, str | None, int], AnyEntry]) -> AnyEntry:
        """Append one entry to the path. `build(entry_id, parent_id, ts)`
        constructs the entry; the ledger does everything around it."""
        ts = self.clock()
        entry_id = self.gen_id()
        conversation = self.session.active_conversation
        parent_id = conversation.nodes[-1] if conversation.nodes else None
        entry = build(entry_id, parent_id, ts)
        self._store_tool_spec(entry)
        self.session.entries[entry_id] = entry
        conversation.nodes.append(entry_id)
        conversation.updated_at = ts
        if isinstance(entry, ToolExecution):
            self.session.tool_executions.setdefault(
                entry.tool_call_id,
                [],
            ).append(entry_id)
        return entry

    def put_entry(self, entry: AnyEntry) -> AnyEntry:
        """Store a fully formed replacement for an entry that is ALREADY in
        the store, and touch `Conversation.updated_at`. The one in-place
        mutation door, serving both mutable entry types (`ToolExecution` and
        `CompactionEntry`); the caller owns building the copy and threading it
        through `before_entry_written` first.

        Refuses an uncommitted entry (`id is None`) and an unknown id: an
        update door must never create — that is `append`'s job, and it is the
        one place a template could otherwise become durable by accident."""
        if entry.id is None:
            raise AgentError(
                "Cannot store an uncommitted entry: `id` is None. put_entry "
                "replaces an entry that already exists; new entries are "
                "created through append()."
            )
        if entry.id not in self.session.entries:
            raise AgentError(f"Cannot update entry {entry.id!r}: no such entry.")
        self._store_tool_spec(entry)
        self.session.entries[entry.id] = entry
        self.session.active_conversation.updated_at = self.clock()
        return entry

    def refresh_entry(self, entry: AnyEntry) -> AnyEntry:
        """Store a replacement for an entry ALREADY in the store, for a
        DERIVED-field refresh — the runner's `recalculate_context_tokens()`.

        Separate from `put_entry` on purpose: that door is documented as the
        mutation path for the two mutable entry types and touches
        `Conversation.updated_at`, and re-deriving a stored estimate is
        neither a mutation of the conversation nor restricted to those two
        types. Refuses to create, exactly like `put_entry`."""
        if entry.id is None or entry.id not in self.session.entries:
            raise AgentError(f"Cannot refresh entry {entry.id!r}: refresh_entry replaces an entry that already exists.")
        self._store_tool_spec(entry)
        self.session.entries[entry.id] = entry
        return entry

    def transition_conversation(
        self,
        *,
        updates: list[AnyEntry],
        created: list[AnyEntry],
        closing: AnyEntry | None,
        nodes: list[str],
        ts: int,
    ) -> Conversation:
        """Archive the active conversation and install a new one over `nodes`.

        The only writer of `conversation_history` and the only replacer of
        `active_conversation`. Deliberately compaction-agnostic — it archives
        one conversation and installs another, and would serve a fork or a
        branch unchanged.

        THE ATOMIC REGION. Every fallible operation happens in the
        precondition block BEFORE the first mutation, after which the
        remaining steps are plain assignments that cannot fail: no
        try/except, no rollback. If it can fail it does not belong after the
        commit point — a transition that failed halfway would leave the same
        conversation both active and archived, a session that loads fine and
        has silently lost its history.

        `updates` replace entries already in the store, `created` are brand-new
        entries whose identity the caller has already stamped, and `closing`
        (a `TurnFinish`) is appended to the OUTGOING path, not the new one.

        This is the third tool-spec write door, and the easiest to miss: no
        ordinary tool call travels through it, only a compaction plan that
        carries or creates a `ToolExecution`. A `ToolExecution` can arrive on
        EITHER list — `updates` is public and takes `list[AnyEntry]` — so the
        helper runs over `updates`, `created` and `closing` alike, not just the
        list the shipped caller happens to use."""
        outgoing = self.session.active_conversation
        entries = self.session.entries

        # ── preconditions: fallible, nothing written ────────────────────────
        for entry in updates:
            if entry.id is None or entry.id not in entries:
                raise AgentError(f"Cannot update entry {entry.id!r} in a transition: no such entry.")
        minted: set[str] = set()
        for entry in [*created, *([closing] if closing is not None else [])]:
            if entry.id is None:
                raise AgentError(
                    "Cannot create an entry with no id in a transition; the "
                    "caller stamps identity before the commit point."
                )
            if entry.id in entries or entry.id in minted:
                raise AgentError(f"Cannot create entry {entry.id!r} in a transition: the id is already taken.")
            minted.add(entry.id)
        conversation_id = self.gen_id()

        # ── THE COMMIT POINT: plain assignments only ────────────────────────
        for entry in updates:
            self._store_tool_spec(entry)
            entries[entry.id] = entry
        for entry in created:
            self._store_tool_spec(entry)
            entries[entry.id] = entry
            if isinstance(entry, ToolExecution):
                self.session.tool_executions.setdefault(
                    entry.tool_call_id,
                    [],
                ).append(entry.id)
        if closing is not None:
            self._store_tool_spec(closing)
            entries[closing.id] = closing
            outgoing.nodes.append(closing.id)
        outgoing.updated_at = ts
        # Explicit: a drive sets RUNNING on its way in and an archived
        # conversation is never re-derived again, so leaving it would freeze a
        # lie in the history.
        outgoing.status = ConversationStatus.IDLE
        self.session.conversation_history.append(outgoing)
        self.session.active_conversation = Conversation(
            id=conversation_id,
            nodes=list(nodes),
            created_at=ts,
            updated_at=ts,
            status=ConversationStatus.IDLE,
        )
        # A pure read over ids validation already proved resolvable — it
        # cannot fail, and installing a conversation whose status nobody
        # derived would be the same frozen lie from the other side.
        self.session.active_conversation.status = self.derive_status()
        return self.session.active_conversation

    def record_usage(self, entry_id: str, **counters: int) -> Usage:
        """Write the provider-usage record for `entry_id` in the ACTIVE
        conversation. The door builds the `Usage` itself — outer key ==
        `conversation_id`, inner key == `entry_id`, entry verified to exist
        on the conversation's path — so the store's invariants hold at every
        call site. At most one record per (conversation, entry) pair: a
        re-record replaces."""
        conversation = self.session.active_conversation
        if entry_id not in self.session.entries:
            raise AgentError(f"Cannot record usage for entry {entry_id!r}: no such entry.")
        if entry_id not in conversation.nodes:
            raise AgentError(
                f"Cannot record usage for entry {entry_id!r}: the entry is "
                f"not on conversation {conversation.id!r}'s path."
            )
        usage = Usage(
            conversation_id=conversation.id,
            entry_id=entry_id,
            **counters,
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
            raise AgentError(f"Cannot prune entry {original_id!r}: no such entry.")
        try:
            node_index = conversation.nodes.index(original_id)
        except ValueError:
            raise AgentError(
                f"Cannot prune entry {original_id!r}: the entry is not on conversation {conversation.id!r}'s path."
            ) from None
        ts = self.clock()
        entry = build(self.gen_id(), original.parent_id, ts)
        if entry.pruned_entry_id != original_id:
            raise AgentError(f"PrunedEntry references {entry.pruned_entry_id!r} but is replacing {original_id!r}.")
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
        return [entries[node_id] for node_id in nodes[idx:] if isinstance(entries[node_id], ToolExecution)]

    def open_turn_pending_executions(self) -> list[ToolExecution]:
        """Status PENDING — not dispatched, not terminal. The cancel
        wind-down's input."""
        return [execution for execution in self.open_turn_executions() if execution.status == ExecutionStatus.PENDING]

    def open_turn_running_executions(self) -> list[ToolExecution]:
        """Status RUNNING. At the start of a drive these are orphans — the
        body's live task no longer exists — and are recovered to INTERRUPTED."""
        return [execution for execution in self.open_turn_executions() if execution.status == ExecutionStatus.RUNNING]

    def open_turn_undecided_executions(self) -> list[ToolExecution]:
        """PENDING executions the permission policy should be offered:
        `approval_status` is None (never processed) or PENDING (deferred)."""
        return [
            execution
            for execution in self.open_turn_pending_executions()
            if execution.approval_status in (None, ApprovalStatus.PENDING)
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
        return any(execution.is_doom_loop_flagged for execution in self.open_turn_executions())

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

    def open_compaction_entry(self) -> CompactionEntry | None:
        """The RESUMABLE `CompactionEntry` inside the open bracket, or None.

        `None` means "there is no compaction to resume", for any of three
        inputs that mean the same thing to every caller: no open bracket at
        all; an open CONVERSATIONAL turn; or an open compaction-shaped bracket
        whose entry already has `parts` — a compaction that already committed
        and whose markers a plan carried, not an interrupted attempt.

        That last test is the one that matters (G6). An open bracket is the
        SIGNAL that a compaction was interrupted; the entry's own state is the
        TEST. `parts` land at the commit point and nowhere else — every
        non-transition ending leaves them None precisely so a failed
        compaction cannot project — so an entry that has them describes
        finished work. Keying on bracket shape instead would let a plan
        counterfeit the shape and make the next drive re-run `compact()` over
        a committed record, overwriting the `compacted_nodes` that say what
        that summary replaced."""
        idx = self.open_turn_index()
        if idx is None:
            return None
        nodes = self.session.active_conversation.nodes
        entries = self.session.entries
        if not is_compaction_bracket(nodes, entries, idx):
            return None
        entry = entries[nodes[idx + 1]]
        return None if entry.parts is not None else entry

    def derive_status(self) -> ConversationStatus:
        """The authoritative status, computed from the entries (used to
        normalize a loaded session and as the runner guard's source of truth).
        Precedence: an unconsumed cancel beats the approval gate beats the
        plain open-turn resume; a CLOSED compaction bracket is transparent and
        is skipped; a closed turn is IDLE unless it failed (TIMED_OUT /
        ERRORED → retry-ready PENDING) or a user message is already queued
        behind it."""
        if self.open_turn_index() is not None:
            if self.open_turn_cancel_requested() is not None:
                return ConversationStatus.CANCELLING
            if self.has_awaiting_approval():
                return ConversationStatus.AWAITING_APPROVAL
            return ConversationStatus.PENDING
        nodes = self.session.active_conversation.nodes
        end = self._before_closed_compaction_brackets()
        if end == 0:
            return ConversationStatus.IDLE
        last = self.session.entries[nodes[end - 1]]
        if isinstance(last, TurnFinish) and last.outcome in (
            TurnOutcome.TIMED_OUT,
            TurnOutcome.ERRORED,
        ):
            return ConversationStatus.PENDING
        if isinstance(last, UserMessage):
            return ConversationStatus.PENDING
        return ConversationStatus.IDLE

    def _before_closed_compaction_brackets(self) -> int:
        """The length of the path with every trailing CLOSED compaction
        bracket dropped — the slice status derives from.

        A compaction bracket is not a conversational turn, and leaving it as
        the leaf gives two wrong answers, both silent: a FAILED compaction
        would look retry-ready (`tf_c(ERRORED)` → PENDING) and a polling loop
        would open a fresh bracket every drive, and a compaction that closed
        behind a queued user message would bury it (the message stops being
        the leaf, so it stops deriving PENDING and is silently never
        answered).

        A trailing `TurnFinish` with no `TurnStart` to anchor it — a policy
        carried one without its pair — is NOT a compaction bracket: stop
        skipping and let the ordinary closed-turn rules apply, so a carried
        `TurnFinish(ERRORED)` still derives retry-ready PENDING rather than
        swallowing the whole path into IDLE."""
        nodes = self.session.active_conversation.nodes
        entries = self.session.entries
        end = len(nodes)
        while end > 0 and isinstance(entries[nodes[end - 1]], TurnFinish):
            start = None
            for i in range(end - 2, -1, -1):
                entry = entries[nodes[i]]
                if isinstance(entry, TurnFinish):
                    break
                if isinstance(entry, TurnStart):
                    start = i
                    break
            if start is None or not is_compaction_bracket(nodes, entries, start):
                break
            end = start
        return end
