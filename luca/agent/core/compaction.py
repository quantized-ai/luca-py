"""The compaction contract — how an application replaces the older span of a
conversation with a summary of it.

`CompactionPolicy` is compaction's single extension point, and it owns
everything about the *decision*: when compaction is worth doing, what the
summary says, which model produces it, which nodes survive, and whether to
back off after a failure. Core triggers, stamps and archives — nothing else.
It reads exactly five things off a returned plan (`entry.parts`,
`entry.llm_config`, `entry.metadata`, `nodes`, `usage`) and discards every
other field, which is the whole coupling surface between the two.

The runner opens a turn bracket, appends a `CompactionEntry`, stamps its
`started_at`, and hands the policy the live session, the path it may rewrite,
and a DEEP COPY of that entry. The policy fills the copy in and returns a
`CompactionPlan` describing the resulting conversation; the runner validates
its STRUCTURE (never its meaning), records the usage, and performs the
transition — one atomic swap in which the pre-compaction conversation is
archived intact and a new one becomes active. Nothing is ever deleted.

This module is pure: value objects and a validator. It knows nothing about the
session's mutation, the ledger, asyncio, or events.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import CompactionPlanError
from .models import (
    AgentSession,
    AnyEntry,
    CompactionEntry,
    ContentPart,
    TextContent,
)

# ── value objects ─────────────────────────────────────────────────────────────


class UsageCounters(BaseModel):
    """The counters a policy reports for its summarization call.

    Field names and semantics are `Usage`'s, minus the ids the ledger owns —
    one vocabulary, and the same shape the runner produces for an ordinary LLM
    response. `extra="forbid"` is the point: a provider's own counter names
    (`prompt_tokens`, `completion_tokens`) fail HERE, in the policy, at plan
    construction — not later inside a runner-internal ledger write that cannot
    produce a clean plan rejection."""

    model_config = ConfigDict(extra="forbid")

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0


class CompactionPlan(BaseModel):
    """What a policy returns: the filled-in entry plus the new conversation.

    `nodes` is the new path expressed before ids exist — a string CARRIES an
    existing node over at that position, an entry object CREATES one there
    (uncommitted: the runner stamps `id`, `parent_id` and `created_at`,
    threading parents left to right). It cannot be a `Conversation` because a
    conversation's nodes are ids and the entries a policy invents have none
    yet.

    The shape is a flat list rather than "a summary plus a kept suffix"
    because the latter bakes in a policy decision. All of these are legal:

        [entry.id]                                  # fold everything
        [entry.id, "u3"]                            # fold, keep the question
        [UserMessage(parts=…), entry.id, "u2", …]   # a framing message first
    """

    model_config = ConfigDict(extra="forbid")

    entry: CompactionEntry  # the copy you were handed, filled in
    nodes: list[AnyEntry | str]  # the new path
    usage: UsageCounters = Field(default_factory=UsageCounters)


class ConversationSnapshot(BaseModel):
    """The active conversation as it stood when the policy was consulted.

    Two paths, because the two questions are different. `nodes` is the FULL
    live path, markers included — it answers "did the session move under the
    plan?" (G2). `offered` is that path minus this compaction's own
    `TurnStart` — it is what the policy was handed as `nodes`, and what a
    carried id is checked against, so framework bookkeeping can never be
    carried back by accident. Storing both keeps the strip in exactly one
    place, which is where a bug in it would be silent."""

    model_config = ConfigDict(frozen=True)

    id: str
    nodes: tuple[str, ...]
    offered: tuple[str, ...]


# ── the contract ──────────────────────────────────────────────────────────────


class CompactionPolicy:
    """The single extension point for compaction. A concrete base — subclass
    and override the two methods. Core never learns that "which nodes to keep"
    is a decision at all: at the transition the kept set is just a list of
    ids."""

    def should_compact(self, session: AgentSession) -> bool:
        """Has the active conversation crossed the point where compaction is
        worth doing? Consulted at the top of every drive (and at `start()`
        time, which is why this is SYNC).

        The threshold, the context sum and the window size are all yours —
        core has no context-total API, and `luca.client.catalog` carries
        `context_window` per model. Core remembers nothing about previous
        failures either: a policy that should stop trying returns False."""
        return False

    async def compact(
        self,
        session: AgentSession,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        """Fill in a DEEP COPY of `entry` and describe the resulting
        conversation.

        `nodes` is THE PATH YOU MAY REWRITE: the active conversation's path
        with this compaction's own `TurnStart` removed, ending with `entry`.
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
        refuses to commit if the active path moved under you, and the entry
        you are handed is a copy precisely so a failed attempt cannot leave a
        summary projecting onto an unchanged path."""
        raise NotImplementedError()


# ── validation: structure only, never meaning ────────────────────────────────


def has_content(parts: list[ContentPart] | None) -> bool:
    """G3's predicate: a text part with a non-whitespace character, or ANY
    non-text part. An image-only summary is legitimate content; a summary that
    is `None`, empty, or whitespace would replace the history with nothing."""
    if not parts:
        return False
    return any(part.text.strip() if isinstance(part, TextContent) else True for part in parts)


def check_snapshot(
    *,
    session: AgentSession,
    snapshot: ConversationSnapshot,
) -> None:
    """G2: the plan must be validated against the path it was computed from.

    Factored out of `validate_plan` because the runner runs it EARLY — before
    recording usage, which itself verifies the compaction entry is still on
    the active path and would otherwise raise its own unrelated error for a
    policy that replaced the active conversation."""
    active = session.active_conversation
    if snapshot.id != active.id:
        raise CompactionPlanError(f"the active conversation changed under the plan ({snapshot.id!r} → {active.id!r})")
    if snapshot.nodes != tuple(active.nodes):
        raise CompactionPlanError("the active conversation's path changed under the plan")


def validate_plan(
    plan: CompactionPlan,
    *,
    entry_id: str,
    session: AgentSession,
    snapshot: ConversationSnapshot,
) -> None:
    """Refuse a plan that is structurally impossible. STRUCTURE ONLY — the
    runner does not know what a turn is, what a tool call needs, or which
    message must survive, and a framework that second-guesses those cannot be
    extended.

    Deliberately unchecked, and therefore committed as given: node order,
    tool-call/result pairing, orphaned turn markers, a nonterminal execution
    carried over, a summary larger than the span it replaced, whether a
    trailing unanswered user message survives. Those are the policy's
    judgment and the policy's hazards."""
    check_snapshot(session=session, snapshot=snapshot)
    if not plan.nodes:
        raise CompactionPlanError("an empty plan is not a compaction")
    carried: list[str] = []
    for node in plan.nodes:
        if not isinstance(node, str):
            continue  # an entry object is created, not carried
        if node not in session.entries:
            raise CompactionPlanError(f"plan references unknown entry {node!r}")
        if node not in snapshot.offered:
            raise CompactionPlanError(f"plan references entry {node!r}, which is not on conversation {snapshot.id!r}")
        if node in carried:
            raise CompactionPlanError(f"plan references entry {node!r} twice")
        carried.append(node)
    if entry_id not in carried:
        raise CompactionPlanError(f"plan omits the compaction entry {entry_id!r}")
    if not has_content(plan.entry.parts):
        raise CompactionPlanError("plan carries no content")
