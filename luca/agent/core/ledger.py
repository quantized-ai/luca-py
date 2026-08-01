"""SessionLedger — the single door onto a session's entry log.

EVERY DOOR TAKES A `conversation_id`. A session holds several conversations —
the main one, its archived predecessors, and (later) one per subagent — so
there is no "active" one to write through implicitly. The id is passed rather
than the `Conversation` object for the same reason it is everywhere else: the
object is live and mutable, an id cannot go stale, and resolving one is a dict
lookup the ledger already holds the session for.

WRITES. Every entry append goes through `append()`, which stamps the injected
id/clock, links `parent_id` to that conversation's path leaf, extends
`Conversation.nodes`, and touches `Conversation.updated_at` — one code path,
so the bookkeeping can't drift between call sites.
`put_entry()` is the only in-place mutation door, serving all three mutable
entry types (`ToolExecution`, `CompactionEntry` and `ChildConversation`): it stores the fully formed
replacement the runner hands it — the runner owns building the updated copy,
stamping `updated_at`, and threading it through `before_entry_written`
middleware first — and touches `Conversation.updated_at`. It refuses to
create, so an uncommitted template can never become durable through it.
`transition_conversation()` archives one conversation and installs a new one
over a given path: the only writer of `previous_conversation_id`, the only
re-pointer of `main_conversation_id`, and the atomic region every fallible
operation must happen before.
`record_usage()` is the single write door onto `AgentSession.usages`: it
builds the record itself (the conversation it was given, an entry it verifies
is on that conversation's path), so the store's key/record agreement and
referential invariants cannot drift across call sites.
`prune()` is the single path-replacement door: it stamps a `PrunedEntry`
built by the caller's callback and swaps it for the original node id IN
PLACE — the original entry stays untouched in `session.entries`; only the
path stops visiting it.
`refresh_entry()` is the derived-field door: it replaces an entry in the
store for a recalculation that changes nothing an application asked for (the
runner's `recalculate_context_tokens()`), deliberately separate from
`put_entry`, which is documented as the MUTATION door for the two mutable
entry types. It takes NO conversation: a stored count is intrinsic to the
entry and shared by every conversation referencing it.

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
approval-state subsets, the resumable compaction. Each is scoped to one
conversation and delegates to the pure `(nodes, entries)` functions in
`models.py`, which `AgentSession.get_conversation_status` reads too: one
implementation, two doors. Status itself is NOT read here — it is derived, and
`session.get_conversation_status(conversation_id)` is its only door.
Usage and context aggregates deliberately have no reads here: totals are
trivially derived by the application from `AgentSession.usages` and
`Entry.context_tokens` over a conversation's nodes.

Execution vocabulary (over `status` + `approval_status`):
- PENDING — body not started, no terminal outcome. Subsets by approval:
  - UNDECIDED (`approval_status` None or PENDING) — what the runner offers
    to the permission policy;
  - AWAITING (`approval_status` PENDING) — the policy explicitly deferred;
    the application resolves out-of-band. A never-asked execution (None)
    is not awaiting — a plain `run()` self-heals it by asking again;
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
    ExecutionStatus,
    PrunedEntry,
    ToolExecution,
    Usage,
    open_compaction_entry,
    open_turn_cancel_requested,
    open_turn_executions,
    open_turn_index,
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

    # ── resolution ───────────────────────────────────────────────────────────

    def conversation(self, conversation_id: str) -> Conversation:
        """The named conversation. Raises rather than returning None: every
        caller here is about to write to it or walk it, and a silent miss would
        surface as an empty path — a conversation that looks finished."""
        conversation = self.session.conversations.get(conversation_id)
        if conversation is None:
            raise AgentError(f"No conversation {conversation_id!r} in session {self.session.id!r}.")
        return conversation

    def _path(self, conversation_id: str) -> list[str]:
        return self.conversation(conversation_id).nodes

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

    def append(
        self,
        conversation_id: str,
        build: Callable[[str, str | None, int], AnyEntry],
    ) -> AnyEntry:
        """Append one entry to `conversation_id`'s path.
        `build(entry_id, parent_id, ts)` constructs the entry; the ledger does
        everything around it."""
        ts = self.clock()
        entry_id = self.gen_id()
        conversation = self.conversation(conversation_id)
        parent_id = conversation.nodes[-1] if conversation.nodes else None
        entry = build(entry_id, parent_id, ts)
        self._store_tool_spec(entry)
        self.session.entries[entry_id] = entry
        conversation.nodes.append(entry_id)
        conversation.updated_at = ts
        return entry

    def put_entry(self, conversation_id: str, entry: AnyEntry) -> AnyEntry:
        """Store a fully formed replacement for an entry that is ALREADY in
        the store, and touch `Conversation.updated_at`. The one in-place
        mutation door, serving all three mutable entry types (`ToolExecution`,
        `CompactionEntry` and `ChildConversation`); the caller owns building the
        copy and threading it through `before_entry_written` first.

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
        self.conversation(conversation_id).updated_at = self.clock()
        return entry

    def refresh_entry(self, entry: AnyEntry) -> AnyEntry:
        """Store a replacement for an entry ALREADY in the store, for a
        DERIVED-field refresh — the runner's `recalculate_context_tokens()`.

        Separate from `put_entry` on purpose: that door is documented as the
        mutation path for the mutable entry types and touches
        `Conversation.updated_at`, and re-deriving a stored estimate is
        neither a mutation of the conversation nor restricted to those two
        types. Takes no conversation for the same reason — the count is
        intrinsic to the entry and shared by every conversation referencing it.
        Refuses to create, exactly like `put_entry`."""
        if entry.id is None or entry.id not in self.session.entries:
            raise AgentError(f"Cannot refresh entry {entry.id!r}: refresh_entry replaces an entry that already exists.")
        self._store_tool_spec(entry)
        self.session.entries[entry.id] = entry
        return entry

    def transition_conversation(
        self,
        conversation_id: str,
        *,
        updates: list[AnyEntry],
        created: list[AnyEntry],
        closing: AnyEntry | None,
        nodes: list[str],
        ts: int,
    ) -> Conversation:
        """Archive `conversation_id` and install a new conversation over
        `nodes`, re-pointing whoever named the old one.

        The only writer of `previous_conversation_id` and the only re-pointer
        of `main_conversation_id`. Deliberately compaction-agnostic — it
        archives one conversation and installs another, and would serve a fork
        or a branch unchanged.

        THE NAME MOVES WITH THE CONVERSATION. A transition installs a new row,
        so whatever named the old one has to name the new one instead; the
        archived row stays reachable backwards through
        `previous_conversation_id`, and nothing needs a forward link. V0 only
        ever transitions the MAIN conversation — subagent conversations are
        never compaction-checked — so a transition on any other conversation
        raises rather than silently orphaning the successor.

        THE ATOMIC REGION. Every fallible operation happens in the
        precondition block BEFORE the first mutation, after which the
        remaining steps are plain assignments that cannot fail: no
        try/except, no rollback. If it can fail it does not belong after the
        commit point — a transition that failed halfway would leave the same
        conversation both archived and named, a session that loads fine and
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
        outgoing = self.conversation(conversation_id)
        entries = self.session.entries

        # ── preconditions: fallible, nothing written ────────────────────────
        if conversation_id != self.session.main_conversation_id:
            raise AgentError(
                f"Cannot transition conversation {conversation_id!r}: only the "
                f"main conversation is transitioned in V0, and a successor "
                f"nothing names would be unreachable."
            )
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
        new_conversation_id = self.gen_id()
        if new_conversation_id in self.session.conversations:
            raise AgentError(f"Cannot install conversation {new_conversation_id!r}: the id is already taken.")

        # ── THE COMMIT POINT: plain assignments only ────────────────────────
        for entry in updates:
            self._store_tool_spec(entry)
            entries[entry.id] = entry
        for entry in created:
            self._store_tool_spec(entry)
            entries[entry.id] = entry
        if closing is not None:
            self._store_tool_spec(closing)
            entries[closing.id] = closing
            outgoing.nodes.append(closing.id)
        outgoing.updated_at = ts
        incoming = Conversation(
            id=new_conversation_id,
            nodes=list(nodes),
            created_at=ts,
            updated_at=ts,
            previous_conversation_id=outgoing.id,
            depth=outgoing.depth,
        )
        self.session.conversations[new_conversation_id] = incoming
        self.session.main_conversation_id = new_conversation_id
        return incoming

    def record_usage(
        self,
        conversation_id: str,
        entry_id: str,
        **counters: int,
    ) -> Usage:
        """Write the provider-usage record for `entry_id` in `conversation_id`.
        The door builds the `Usage` itself — outer key == `conversation_id`,
        inner key == `entry_id`, entry verified to exist on that conversation's
        path — so the store's invariants hold at every call site. At most one
        record per (conversation, entry) pair: a re-record replaces."""
        conversation = self.conversation(conversation_id)
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
        conversation_id: str,
        original_id: str,
        build: Callable[[str, str | None, int], PrunedEntry],
    ) -> PrunedEntry:
        """Replace `original_id`'s node in `conversation_id`'s path with a
        durable `PrunedEntry`. `build(entry_id, parent_id, ts)` constructs the
        entry (the caller threads context calculation and entry middleware
        inside it, exactly like `append`); the door stamps identity — the
        pruned entry takes the ORIGINAL's `parent_id`, since it occupies the
        original's position — verifies the replacement (`pruned_entry_id`
        must reference the original, `pruned_entry_type` must agree with it,
        and a pruned tool execution must be terminal), stores it in
        `session.entries`, swaps the node id in place, and touches
        `Conversation.updated_at`. The original entry itself is never
        mutated or deleted."""
        conversation = self.conversation(conversation_id)
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

    # ── reads (entry-derived state, scoped to one conversation) ─────────────

    def open_turn_index(self, conversation_id: str) -> int | None:
        """Index of the TurnStart opening an unclosed turn, or None."""
        return open_turn_index(self._path(conversation_id), self.session.entries)

    def open_turn_executions(self, conversation_id: str) -> list[ToolExecution]:
        """Every ToolExecution in the open turn, in path order."""
        return open_turn_executions(self._path(conversation_id), self.session.entries)

    def open_turn_pending_executions(self, conversation_id: str) -> list[ToolExecution]:
        """Status PENDING — not dispatched, not terminal. The cancel
        wind-down's input."""
        return [
            execution
            for execution in self.open_turn_executions(conversation_id)
            if execution.status == ExecutionStatus.PENDING
        ]

    def open_turn_running_executions(self, conversation_id: str) -> list[ToolExecution]:
        """Status RUNNING. At the start of a drive these are orphans — the
        body's live task no longer exists — and are recovered to INTERRUPTED."""
        return [
            execution
            for execution in self.open_turn_executions(conversation_id)
            if execution.status == ExecutionStatus.RUNNING
        ]

    def open_turn_undecided_executions(self, conversation_id: str) -> list[ToolExecution]:
        """PENDING executions the permission policy should be offered:
        `approval_status` is None (never processed) or PENDING (deferred)."""
        return [
            execution
            for execution in self.open_turn_pending_executions(conversation_id)
            if execution.approval_status in (None, ApprovalStatus.PENDING)
        ]

    def open_turn_awaiting_executions(self, conversation_id: str) -> list[ToolExecution]:
        """PENDING executions whose `approval_status` is PENDING — the policy
        explicitly deferred, so the application must resolve out-of-band
        before the turn can finish."""
        return [
            execution
            for execution in self.open_turn_pending_executions(conversation_id)
            if execution.approval_status == ApprovalStatus.PENDING
        ]

    def open_turn_ready_executions(self, conversation_id: str) -> list[ToolExecution]:
        """PENDING executions cleared for dispatch: `approval_status` ALLOWED."""
        return [
            execution
            for execution in self.open_turn_pending_executions(conversation_id)
            if execution.approval_status == ApprovalStatus.ALLOWED
        ]

    def has_awaiting_approval(self, conversation_id: str) -> bool:
        return bool(self.open_turn_awaiting_executions(conversation_id))

    def open_turn_has_doom_loop_flagged(self, conversation_id: str) -> bool:
        """True if any ToolExecution in the open turn is doom-loop-flagged."""
        return any(execution.is_doom_loop_flagged for execution in self.open_turn_executions(conversation_id))

    def open_turn_cancel_requested(self, conversation_id: str) -> CancelRequested | None:
        """The unconsumed `CancelRequested` inside the open turn, or None."""
        return open_turn_cancel_requested(self._path(conversation_id), self.session.entries)

    def open_compaction_entry(self, conversation_id: str) -> CompactionEntry | None:
        """The RESUMABLE `CompactionEntry` inside the open bracket, or None."""
        return open_compaction_entry(self._path(conversation_id), self.session.entries)
