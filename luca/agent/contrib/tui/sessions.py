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
    """Write the session ATOMICALLY: a temporary file in the same directory,
    then `os.replace` into place.

    Truncating in place is the one true corruption risk in the system — a
    crash mid-write leaves half a JSON document and an unloadable session,
    losing the whole conversation rather than the last turn. `os.replace` is
    atomic on POSIX and Windows, so a reader either sees the previous session
    or the new one, never a torn one. The core owns no persistence, so this
    guarantee is the application's to provide."""
    path = session_path(session.id, directory)
    temporary = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(session.model_dump_json(indent=2))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
