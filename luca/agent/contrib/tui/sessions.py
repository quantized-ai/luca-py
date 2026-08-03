"""Session persistence for the TUI: one `<session-id>.json` per session, kept
in a per-project directory under a global store.

The store is `~/.luca/projects/<encoded-project-path>/`, where the encoding is
the project's absolute path with its separators turned into `-`. Sessions used
to land in the launch directory, which littered every repo and left no way back
into a conversation except copying its id by hand.

`save_session` / `load_session` still take the directory, so the app and its
tests point the store anywhere; `resolve_session_directory` is what the TUI uses
to work out where that is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from luca.agent.core.models import AgentSession, UserMessage

from .render import user_transcript_text

DEFAULT_STORE = "~/.luca/projects"

TITLE_LENGTH = 60


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
    # The project's directory under the store does not exist until its first save.
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = session_path(session.id, directory)
    temporary = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(session.model_dump_json(indent=2))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


# ── where a project's sessions live ──────────────────────────────────────────


def encode_project_path(path: str | os.PathLike[str]) -> str:
    """A project path as one directory name: `/a/b` becomes `-a-b`.

    The same scheme Claude Code uses for `~/.claude/projects`, so the store is
    readable at a glance. It is lossy — a directory literally named `a-b` and
    the path `a/b` encode alike — and that is accepted: a collision only means
    two projects share a directory, and sessions are keyed by id inside it.

    `\\` and `:` go too, or a Windows drive letter would produce a name the
    filesystem rejects."""
    encoded = Path(path).resolve().as_posix()
    for separator in ("/", "\\", ":"):
        encoded = encoded.replace(separator, "-")
    return encoded


def resolve_session_directory(
    workspace: str | os.PathLike[str] = ".",
    configured: str | None = None,
) -> Path:
    """Where this project's sessions live: `<store>/<encoded workspace>`.

    Keyed on the WORKSPACE rather than the process cwd. The two are the same by
    default, and every other per-project lookup in the app — the shell root,
    skills discovery, the instruction walk — already anchors at the workspace,
    so a session belongs to the project being worked on."""
    store = Path(configured or DEFAULT_STORE).expanduser()
    return store / encode_project_path(workspace)


@dataclass(frozen=True)
class SessionSummary:
    """One row of the resume picker."""

    id: str
    path: Path
    modified: datetime
    title: str
    turns: int
    model: str


def summarize_session(path: Path) -> SessionSummary | None:
    """One stored session as a picker row, or `None` when it cannot be read.
    One unloadable file must never stop the picker opening."""
    try:
        session = AgentSession.model_validate_json(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    nodes = session.conversations[session.main_conversation_id].nodes
    messages = [session.entries[node] for node in nodes if isinstance(session.entries.get(node), UserMessage)]
    lines = user_transcript_text(messages[0].parts).strip().splitlines() if messages else []
    title = lines[0] if lines else "(empty)"
    return SessionSummary(
        id=session.id,
        path=path,
        modified=datetime.fromtimestamp(path.stat().st_mtime),
        title=title[:TITLE_LENGTH],
        turns=len(messages),
        model=session.session_config.llm_config.model,
    )


def list_sessions(directory: str | os.PathLike[str]) -> list[SessionSummary]:
    """Every readable session in `directory`, newest first.

    Each row costs one parse (~3ms for a 500KB session), so no index file: the
    listing is built on demand when the picker opens, and there is nothing to
    keep in sync or rebuild when it drifts."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found = [summarize_session(path) for path in sorted(root.glob("*.json"))]
    return sorted(
        (summary for summary in found if summary is not None),
        key=lambda summary: summary.modified,
        reverse=True,
    )
