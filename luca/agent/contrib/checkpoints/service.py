"""`CheckpointService` — taking a checkpoint, and putting one back.

The orchestration layer: it owns the two-sided operation that neither half can
do alone, joining `ShadowGitStore` (the workspace) to
`AgentSessionRunner.rewind_to` (the conversation).

A PLAIN SERVICE, NOT A PLUGIN, and that is a decision rather than an omission.
It contributes no tools, no system-prompt parts and no middleware, because
restoring a checkpoint is something a PERSON does. A model that could undo its
own last turn could erase the evidence of a mistake and retry it silently, and
no prompt wording makes that a good idea. The application wires this object and
calls it from its own UI.

IT ALSO STAYS OFF THE MIDDLEWARE CHAIN. Taking a snapshot would fit
`before_entry_written` or `before_post_message` on paper; both hooks are
SYNCHRONOUS, and a git commit over a large workspace on the event loop, once
per entry write, is not a trade worth making. The application awaits `take()`
instead, at the one moment it already knows a turn is starting. Every git call
here goes through `asyncio.to_thread`, matching what `store.py` documents.

ORDER ON RESTORE: files first, then the conversation. If git fails the
conversation is untouched and the user can retry; the reverse order would leave
a conversation running ahead of the workspace, which is the direction that
misleads the model.

That leaves the opposite hazard — the files land and then `rewind_to` REFUSES
(a live run, an open turn) — and it is covered by taking a SAFETY SNAPSHOT
first and rolling back to it if the rewind raises. A safety commit rather than
a pre-flight check on purpose: `rewind_to`'s guards read runner-private state
(the one-run-per-conversation guard is `AgentSessionRunner._runs`), so a contrib
caller cannot ask "would this be allowed?" without reimplementing the guards and
drifting from them. Undoing our own work is something this object can always do.
"""

from __future__ import annotations

import asyncio
import logging
import time

from luca.agent.core import AgentSession, AgentSessionRunner

from .index import Checkpoint, read_index, restorable, write_index
from .store import ShadowGitStore

logger = logging.getLogger(__name__)

# How much of the user's message becomes the picker label.
LABEL_LENGTH = 60


class CheckpointService:
    """Take a checkpoint before a turn; restore one on request."""

    def __init__(self, store: ShadowGitStore, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled
        # Guards the index READ-MODIFY-WRITE, and only that. `read_index`
        # returns a fresh object parsed out of `extras`, so two `take()` calls
        # overlapping across their `to_thread` snapshot would each append to
        # their own copy and the second write would silently drop the first.
        # The git call stays OUTSIDE it — holding a lock across a slow await
        # would serialize snapshots for no reason. This is the shape
        # `luca.agent.core.tool_registry`'s rule 13c prescribes.
        self._index_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and self.store.available()

    def checkpoints(self, session: AgentSession) -> list[Checkpoint]:
        """The checkpoints this session can be restored to, newest first."""
        if not self.available:
            return []
        return restorable(session, read_index(session))

    async def take(self, session: AgentSession, label: str = "") -> Checkpoint | None:
        """Snapshot the workspace and anchor it at the current path leaf.

        Called immediately BEFORE `post_message`, so the anchor is the last
        node of the previous turn and restoring drops the whole turn the
        message is about to open. Returns None when checkpoints are
        unavailable, which is not an error — the feature is simply off."""
        if not self.available:
            return None
        conversation = session.conversations[session.main_conversation_id]
        anchor = conversation.nodes[-1] if conversation.nodes else None
        commit = await asyncio.to_thread(self.store.snapshot, label or "checkpoint")
        if commit is None:
            return None
        checkpoint = Checkpoint(
            commit=commit,
            anchor_entry_id=anchor,
            created_at=_now_ms(),
            label=label[:LABEL_LENGTH],
        )
        async with self._index_lock:
            index = read_index(session)
            index.checkpoints.append(checkpoint)
            write_index(session, index)
        return checkpoint

    async def restore(self, runner: AgentSessionRunner, checkpoint: Checkpoint) -> bool:
        """Put the workspace back, then rewind the conversation to match.

        Raises whatever `rewind_to` raises — a live run or an open turn is a
        caller error, and the caller is the one that can cancel and retry. The
        WORKSPACE is rolled back to where it was before that raise, so a
        refused restore leaves nothing half-done. A git failure is not an
        exception at all: it returns False with the session untouched."""
        if not self.available:
            return False
        if not self.store.has(checkpoint.commit):
            logger.warning("checkpoint %s is not in the shadow repository", checkpoint.commit)
            return False
        safety = await asyncio.to_thread(self.store.snapshot, "before restore")
        if not await asyncio.to_thread(self.store.restore, checkpoint.commit):
            return False
        try:
            runner.rewind_to(checkpoint.anchor_entry_id)
        except BaseException:
            if safety is not None:
                await asyncio.to_thread(self.store.restore, safety)
            raise
        return True


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
