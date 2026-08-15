"""The checkpoint index — which conversation position each snapshot belongs to.

Stored in `AgentSession.extras` under one namespaced key, which is exactly what
that field is for: "a tool, a registry or a plugin can keep state that outlives
the process without the application inventing a second file to put it in".
`contrib/memory` already keeps the todo list and the scratchpad the same way.
Riding on the session means a checkpoint list survives `--resume` for free, and
a session copied to another machine carries its index (the shadow repo does not
travel with it, which `Checkpoint.commit` failing to resolve reports honestly).

JSON-CLEAN ALL THE WAY DOWN. `extras` is dumped with the session, so rows go in
as plain dicts and come back validated. A hand-mangled or older index is
dropped rather than raising: losing the ability to undo is a cosmetic loss, and
it must never cost someone their conversation.

THE ANCHOR IS THE PATH LEAF AT CHECKPOINT TIME, not the user message that
follows it. `AgentSessionRunner.rewind_to` truncates so the anchor is the LAST
surviving node, so anchoring at the leaf before the message is what makes
restoring drop the whole turn, message included. `None` is a real anchor and
means "the conversation was empty", which rewinds back to empty.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from luca.agent.core import AgentSession

logger = logging.getLogger(__name__)

# Namespaced, per the rule on `AgentSession.extras`.
CHECKPOINTS_KEY = "luca.checkpoints"


class Checkpoint(BaseModel):
    """One snapshot, bound to the conversation position it was taken at."""

    model_config = ConfigDict(extra="forbid")

    commit: str
    # The node that must remain the path leaf when this checkpoint is restored.
    # None means the conversation had no nodes yet.
    anchor_entry_id: str | None = None
    created_at: int = 0
    # What the user typed, for the picker. Presentation only.
    label: str = ""


class CheckpointIndex(BaseModel):
    """Every checkpoint in one session, oldest first."""

    model_config = ConfigDict(extra="forbid")

    checkpoints: list[Checkpoint] = Field(default_factory=list)


def read_index(session: AgentSession) -> CheckpointIndex:
    """The session's index, or an empty one. Never raises."""
    raw = session.extras.get(CHECKPOINTS_KEY)
    if raw is None:
        return CheckpointIndex()
    try:
        return CheckpointIndex.model_validate(raw)
    except ValidationError:
        logger.warning("the checkpoint index on session %s is unreadable; starting clean", session.id, exc_info=True)
        return CheckpointIndex()


def write_index(session: AgentSession, index: CheckpointIndex) -> None:
    """Store the index back on the session as plain JSON-clean data."""
    session.extras[CHECKPOINTS_KEY] = index.model_dump(mode="json")


def restorable(session: AgentSession, index: CheckpointIndex) -> list[Checkpoint]:
    """The checkpoints whose anchor is still on the MAIN conversation's path,
    newest first.

    A rewind installs a successor over a prefix, so checkpoints taken after the
    point you rewound to have anchors that are no longer reachable. They stay in
    the index (the shadow commits are still there, and nothing in this framework
    deletes history) but they are not offered: restoring one would rewind to a
    position the current conversation never had.

    A checkpoint anchored at `None` is always offered — every conversation can
    be rewound to empty."""
    path = set(session.conversations[session.main_conversation_id].nodes)
    live = [c for c in index.checkpoints if c.anchor_entry_id is None or c.anchor_entry_id in path]
    return list(reversed(live))
