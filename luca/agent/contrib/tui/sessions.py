"""Session persistence for the TUI: one `<session-id>.json` per session.

The same convention the classic REPL demo used, parameterized by directory so
the app (and its tests) can point the store anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from luca.agent.core.models import AgentSession


def session_path(session_id: str, directory: str | os.PathLike[str] = ".") -> Path:
    return Path(directory) / f"{session_id}.json"


def load_session(session_id: str, directory: str | os.PathLike[str] = ".") -> AgentSession:
    return AgentSession.model_validate_json(
        session_path(session_id, directory).read_text(),
    )


def fork_session(session: AgentSession) -> AgentSession:
    """Clone the session under a fresh id (entries/conversation copied by value)."""
    forked = session.model_copy(deep=True)
    forked.id = uuid4().hex[:8]
    return forked


def save_session(session: AgentSession, directory: str | os.PathLike[str] = ".") -> Path:
    path = session_path(session.id, directory)
    path.write_text(session.model_dump_json(indent=2))
    return path
