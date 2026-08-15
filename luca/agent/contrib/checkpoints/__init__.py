"""Checkpoints and undo: workspace snapshots bound to conversation positions.

One checkpoint per user turn. Restoring one puts the files back with git and
rewinds the conversation with `AgentSessionRunner.rewind_to`, so the model's
history and the workspace agree again and the undone turn is simply not in the
next projection.

Three pieces, each usable on its own:

- `ShadowGitStore` (`store.py`) snapshots the workspace into a private git
  repository the user never sees. Tool-agnostic, so `bash` is covered like
  everything else.
- `CheckpointIndex` (`index.py`) records which conversation position each
  snapshot belongs to, on `AgentSession.extras`, so the list survives a resume.
- `CheckpointService` (`service.py`) joins the two.

The core knows none of this. It contributes a single policy-free primitive —
rewind the main conversation to a prefix — and everything about what a
checkpoint IS lives here. See `docs/agent/contrib/checkpoints/README.md`.
"""

from .index import CHECKPOINTS_KEY, Checkpoint, CheckpointIndex, read_index, restorable, write_index
from .service import CheckpointService
from .store import DEFAULT_EXCLUDES, ShadowGitStore

__all__ = [
    "CHECKPOINTS_KEY",
    "DEFAULT_EXCLUDES",
    "Checkpoint",
    "CheckpointIndex",
    "CheckpointService",
    "ShadowGitStore",
    "read_index",
    "restorable",
    "write_index",
]
