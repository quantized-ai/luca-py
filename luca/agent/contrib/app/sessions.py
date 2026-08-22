"""Session persistence: one `<session-id>.json` per session, kept in a
per-project directory under a global store.

The store is `~/.luca/projects/<encoded-project-path>/`, where the encoding is
the project's absolute path with its separators turned into `-`. Sessions used
to land in the launch directory, which littered every repo and left no way back
into a conversation except copying its id by hand.

`save_session` / `load_session` take the directory, so an application and its
tests point the store anywhere; `resolve_session_directory` is what works out
where that is.

Beside each session sits an optional `<session-id>.app.json` — THE FRONT END'S
OWN STORE, for state that belongs to the interface rather than to the
conversation. It is a separate file rather than a key under
`AgentSession.extras` on purpose: `extras` is the SESSION's free-form state,
read by anything that loads the session (another driver, a script, a server),
and a front end's private bookkeeping has no business travelling with it.
Today it holds exactly one thing — the `ask_user` question store, which a
deferred tool call needs back on resume so a parked question can be
re-rendered rather than silently re-asked. See `load_app_state` /
`save_app_state`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from luca.agent.core.models import (
    AgentSession,
    AudioContent,
    ContentPart,
    FileContent,
    ImageContent,
    TextContent,
    UserMessage,
)

from .prompt_files import mention_of

logger = logging.getLogger(__name__)

DEFAULT_STORE = "~/.luca/projects"

SIDECAR_SUFFIX = ".app.json"

TITLE_LENGTH = 60

# The one key the sidecar carries today: `QuestionsTool`'s job store, keyed by
# `tool_call_id`, JSON-shaped all the way down.
QUESTIONS_STORE_KEY = "questions"


def session_path(session_id: str, directory: str | os.PathLike[str] = ".") -> Path:
    return Path(directory) / f"{session_id}.json"


def app_state_path(session_id: str, directory: str | os.PathLike[str] = ".") -> Path:
    """The front end's sidecar for one session: `<session-id>.app.json`."""
    return Path(directory) / f"{session_id}{SIDECAR_SUFFIX}"


def load_app_state(session_id: str, directory: str | os.PathLike[str] = ".") -> dict:
    """The front end's own state for one session, or an empty dict.

    NEVER RAISES, and validates SHAPE rather than only syntax. This is an
    interface convenience, not conversation data: a missing, unreadable or
    hand-mangled sidecar must cost the user their outstanding questions, never
    their session. A parked `ask_user` recovers from the loss on its own anyway
    — `execute()` re-seeds the job from `raw_tool_call.arguments` and defers
    again, so the user is simply asked a second time.

    Shape matters because the store is handed straight to a tool that indexes
    it: a top-level value of the wrong type would turn every `ask_user` call in
    that session into a `FAILED` one, permanently, since the bad shape would be
    written back on the next save. Anything that is not a mapping is dropped
    here instead."""
    path = app_state_path(session_id, directory)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("could not read the app state at %s; starting clean", path, exc_info=True)
        return {}
    if not isinstance(state, dict):
        logger.warning("the app state at %s is not an object; starting clean", path)
        return {}
    store = state.get(QUESTIONS_STORE_KEY)
    if store is not None and not isinstance(store, dict):
        logger.warning("the question store at %s is not an object; dropping it", path)
        state.pop(QUESTIONS_STORE_KEY, None)
    return state


def save_app_state(session_id: str, state: dict, directory: str | os.PathLike[str] = ".") -> Path | None:
    """Write the sidecar atomically, the way `save_session` writes the session.

    Returns the path, or None when the state is empty (nothing to keep) or the
    write failed — the same reasoning as the read: losing this file is a
    cosmetic loss and must never take a turn down with it."""
    path = app_state_path(session_id, directory)
    if not state:
        return None
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
        try:
            temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    except (OSError, TypeError, ValueError):
        logger.warning("could not write the app state at %s", path, exc_info=True)
        return None
    return path


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


def delete_session(session_id: str, directory: str | os.PathLike[str] = ".") -> None:
    """Delete a session AND its sidecar — the two are one artifact to the
    user, and an orphaned `.app.json` would outlive the conversation it
    describes forever."""
    session_path(session_id, directory).unlink(missing_ok=True)
    app_state_path(session_id, directory).unlink(missing_ok=True)


# ── one session, summarized ──────────────────────────────────────────────────


def user_transcript_text(parts: Iterable[ContentPart]) -> str:
    """A user message's parts as transcript text: text verbatim, each image
    as a `[image: name]` placeholder line.

    `@`-mention parts are skipped — an inlined file belongs in its own read
    row, not dumped into the user's own words."""
    lines: list[str] = []
    for part in parts:
        if mention_of(part) is not None:
            continue
        if isinstance(part, TextContent):
            lines.append(part.text)
        elif isinstance(part, ImageContent):
            label = part.metadata.get("name") or part.source.media_type or "image"
            lines.append(f"[image: {label}]")
        elif isinstance(part, AudioContent):
            label = part.metadata.get("name") or part.source.media_type or "audio"
            lines.append(f"[audio: {label}]")
        elif isinstance(part, FileContent):
            label = part.name or part.metadata.get("name") or part.source.media_type or "file"
            lines.append(f"[file: {label}]")
    return "\n".join(lines)


PreviewFn = Callable[[AgentSession], list[str]]
"""How a front end renders the last few transcript rows of a session for its
picker. Optional, and front-end shaped: the TUI's version emits theme-token
markup that means nothing to anyone else, and a JSON-RPC client wants no
preview at all."""


@dataclass(frozen=True)
class SessionSummary:
    """One row of a session picker."""

    id: str
    path: Path
    modified: datetime
    title: str
    turns: int
    model: str
    tokens: int = 0
    cost: float | None = None
    size_bytes: int = 0
    preview: list[str] | None = None


def summarize(
    session: AgentSession,
    *,
    path: Path,
    modified: datetime,
    size_bytes: int,
    preview: PreviewFn | None = None,
) -> SessionSummary:
    """A loaded session as one picker row.

    Split from the file read so a catalogued, in-memory session produces the
    same row a stored one does — a front end has one derivation, not two."""
    from .usage import estimated_cost, usage_totals

    nodes = session.conversations[session.main_conversation_id].nodes
    messages = [session.entries[node] for node in nodes if isinstance(session.entries.get(node), UserMessage)]
    lines = user_transcript_text(messages[0].parts).strip().splitlines() if messages else []
    title = lines[0] if lines else "(empty)"
    return SessionSummary(
        id=session.id,
        path=path,
        modified=modified,
        title=title[:TITLE_LENGTH],
        turns=len(messages),
        model=session.session_config.llm_config.model,
        tokens=usage_totals(session).total,
        cost=estimated_cost(session),
        size_bytes=size_bytes,
        preview=preview(session) if preview is not None else None,
    )


def summarize_session(path: Path, *, preview: PreviewFn | None = None) -> SessionSummary | None:
    """One stored session as one picker row, or `None` when it cannot be read.
    One unloadable file must never stop the picker opening."""
    try:
        session = AgentSession.model_validate_json(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        # Deliberately swallowed so the picker still opens; the log is the only
        # place the reason for a missing row is ever recorded.
        logger.warning("session %s could not be read: %s", path, exc)
        return None
    stat = path.stat()
    return summarize(
        session,
        path=path,
        modified=datetime.fromtimestamp(stat.st_mtime),
        size_bytes=stat.st_size,
        preview=preview,
    )


def list_sessions(
    directory: str | os.PathLike[str],
    *,
    preview: PreviewFn | None = None,
) -> list[SessionSummary]:
    """Every readable session in `directory`, newest first.

    Each row costs one parse (~3ms for a 500KB session), so no index file: the
    listing is built on demand when the picker opens, and there is nothing to
    keep in sync or rebuild when it drifts.

    `*.app.json` sidecars are skipped by NAME rather than by failing to parse:
    they live in the same directory and match the same glob, and letting them
    through would log a warning per session on every open."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found = [
        summarize_session(path, preview=preview)
        for path in sorted(root.glob("*.json"))
        if not path.name.endswith(SIDECAR_SUFFIX)
    ]
    return sorted(
        (summary for summary in found if summary is not None),
        key=lambda summary: summary.modified,
        reverse=True,
    )
