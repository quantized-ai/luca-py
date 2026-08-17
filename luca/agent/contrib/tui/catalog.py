"""The declarative catalog: `screen × world → ScreenState`.

Two axes, combined by `fixtures/catalog.yaml`:

- **Screens** are projections. One pure function per screen of the app, each
  taking the world it reads and returning a whole `ScreenState`. They are the
  SAME functions the live app calls (`transcript_blocks`, `cost_state`,
  `build_settings_state`, `build_sessions_state`, `build_approval_prompts`,
  `question_set_state`), so a catalogued screen cannot drift from the product:
  delete a feature and its catalogued screens change or stop building.
- **Worlds** are data. A committed `AgentSession` under `fixtures/sessions/`
  plus the ambient state no session holds — branch, theme, a typed query, the
  boxes ticked in a picker, how far through a question set the user has got.

Adding a state to the catalog is a line in `catalog.yaml`. Adding a world is a
session file. Adding a screen is a function plus a name in `SCREENS`.

Nothing here reads a live app, a provider, a key, the clock or the filesystem
beyond its own fixtures, so every catalogued screen renders identically on any
machine and snapshots deterministically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
from luca.agent.contrib.questions import QuestionsTool
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    ExecutionStatus,
    ToolExecution,
    ToolSpec,
)
from luca.agent.core.projection import ConversationProjector, tool_message_text

from . import state as vm
from .approvals import build_approval_prompts
from .commands import build_sessions_state, build_settings_state, palette_rows
from .files import rank_files
from .format import HINTS, approval_hints, question_hints_for, short_model
from .mcp_service import build_mcp_detail, build_mcp_state, mcp_hints
from .render import (
    filter_rows,
    is_questions_tool,
    picker_rows,
    plan_from_session,
    question_set_state,
    transcript_blocks,
)
from .sessions import SessionSummary, summarize
from .usage import cost_state, status_counter

CATALOG_PATH = Path(__file__).parent / "fixtures" / "catalog.yaml"
SESSIONS_DIR = Path(__file__).parent / "fixtures" / "sessions"

# A fixed clock. The sessions screen prints `18m ago`, so the catalog needs a
# NOW that never moves or its snapshots would rot overnight.
NOW = datetime(2026, 3, 14, 9, 30, 0)

ScreenName = Literal[
    "chat", "approval", "questions", "palette", "picker", "cost", "settings", "sessions", "mcp", "mcp-detail"
]


class CatalogError(Exception):
    """A catalog entry naming an unknown screen or session, or a malformed
    catalog document."""


class FileEntry(BaseModel):
    """One row of the `@` picker's world: a path and what inlining it costs."""

    path: str
    tokens: int = 0
    model_config = ConfigDict(extra="forbid")


# A small stand-in repo, so `{screen: picker}` renders something without every
# entry having to carry a file list.
DEFAULT_FILES: tuple[FileEntry, ...] = (
    FileEntry(path="luca/events.py", tokens=1400),
    FileEntry(path="luca/store/reader.py", tokens=1900),
    FileEntry(path="luca/store/writer.py", tokens=2200),
    FileEntry(path="luca/store/schema.sql", tokens=400),
    FileEntry(path="luca/cli.py", tokens=3100),
    FileEntry(path="docs/writing-skills.md", tokens=3400),
    FileEntry(path="tests/store/test_writer.py", tokens=1100),
    FileEntry(path="tests/test_events.py", tokens=800),
)


class World(BaseModel):
    """The ambient state a screen reads that no `AgentSession` holds.

    Every field has the ordinary default, so most catalog entries name none of
    it and the ones that do say exactly what makes them different."""

    # frame
    cwd: str = "~/quantized/luca"
    branch: str | None = "main"
    dirty: bool = True
    busy: bool = False  # a turn is running: the composer queues
    # settings
    theme: str = "luca-dark"
    streaming: bool = True
    mode: str = "ask"
    show_counter: bool = True
    # overlays and lists
    query: str = ""
    selected: int = 0
    checked: list[str] = Field(default_factory=list)
    # the question dock — how far the user has got through the set. `answers`
    # maps a question's index to the option indices committed on it; a question
    # the map does not mention is still open.
    active: int = 0
    phase: Literal["asking", "confirming"] = "asking"
    answers: dict[int, list[int]] = Field(default_factory=dict)
    extra: str | None = None
    # Copied, not shared: a `World` must never be able to reach into the
    # module constant and change what every other world sees.
    files: list[FileEntry] = Field(default_factory=lambda: [entry.model_copy() for entry in DEFAULT_FILES])
    model_config = ConfigDict(extra="forbid")


class Entry(BaseModel):
    """One catalogued screen — the catalog's unit."""

    name: str
    screen: ScreenName
    session: str = "empty"
    world: World = Field(default_factory=World)
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Scene:
    """Everything a projection may read. One argument, so adding a screen that
    needs something new never changes the other six signatures."""

    session: AgentSession
    world: World
    library: list[AgentSession]


# ── shared pieces ─────────────────────────────────────────────────────────────


def _resolve_result(session: AgentSession) -> Callable[[ToolExecution], tuple[str | None, bool]]:
    """The projector-backed result text, exactly as the live app supplies it."""
    projector = ConversationProjector()

    def resolve(execution: ToolExecution) -> tuple[str | None, bool]:
        try:
            message = projector.project_tool_execution(execution, session.entries)
        except ProjectionError:
            return None, False
        return tool_message_text(message), message.is_error

    return resolve


def _transcript(scene: Scene) -> list[vm.Block]:
    return transcript_blocks(scene.session, resolve_result=_resolve_result(scene.session))


def _status(scene: Scene, *, label: str | None = None) -> vm.StatusState:
    """The live status bar; a modal replaces the model/branch/counter run with
    a single right-aligned label."""
    if label is not None:
        return vm.StatusState(cwd=scene.world.cwd, label=label)
    tokens, cost = status_counter(scene.session) if scene.world.show_counter else (None, None)
    return vm.StatusState(
        cwd=scene.world.cwd,
        model=short_model(scene.session.session_config.llm_config.model),
        branch=scene.world.branch,
        dirty=scene.world.dirty,
        tokens=tokens,
        cost=cost,
    )


def _composer(scene: Scene) -> vm.ComposerState:
    return vm.ComposerState(placeholder="working…" if scene.world.busy else "ask, or / for commands")


def _base(scene: Scene, *, label: str | None = None) -> dict:
    """Status + transcript + the sticky plan — what every screen sits on. A
    modal covers the frame but this is still what shows underneath it, and the
    plan panel outlives whatever takes the dock."""
    return {
        "status": _status(scene, label=label),
        "transcript": _transcript(scene),
        "plan": plan_from_session(scene.session, running=scene.world.busy),
    }


# ── the screens ───────────────────────────────────────────────────────────────


def chat_screen(scene: Scene) -> vm.ScreenState:
    """1a–1d, 1g — the main conversation column with the composer docked."""
    return vm.ScreenState(
        **_base(scene),
        composer=_composer(scene),
        hints=HINTS["running"] if scene.world.busy else HINTS["idle"],
    )


def approval_screen(scene: Scene) -> vm.ScreenState:
    """1c, 1h — the bordered prompt that REPLACES the composer.

    Built from the session's own gated call through the real prompt model, so
    option 2 says exactly what it widens or is not offered at all."""
    execution = _gated_execution(scene.session)
    if execution is None:
        raise CatalogError("the approval screen needs a session with a gated, unresolved tool call")
    # The FIRST uncovered step. A multi-step gate asks again after this one is
    # answered, which is a sequence, not a state — the catalog holds states.
    prompt = build_approval_prompts(execution, PermissionStrategy())[0]
    approval = prompt.to_state().model_copy(update={"selected": scene.world.selected})
    return vm.ScreenState(
        **_base(scene),
        approval=approval,
        hints=approval_hints(len(approval.options)),
    )


def _gated_execution(session: AgentSession) -> ToolExecution | None:
    for node_id in session.conversations[session.main_conversation_id].nodes:
        entry = session.entries.get(node_id)
        if (
            isinstance(entry, ToolExecution)
            and entry.status in NONTERMINAL_STATUSES
            and entry.extras.get("approval_context")
        ):
            return entry
    return None


def questions_screen(scene: Scene) -> vm.ScreenState:
    """1l — the agent's own `ask_user` set, the fourth dock.

    Rebuilt through the tool's OWN app-facing API and along the same three
    steps the live app takes: the parked call's arguments seed a
    `QuestionsTool` store, `pending()` re-validates them into `Question`
    objects, and `question_set_state` turns those into the dock. The clicks on
    top of it are the world's — nothing in a session records how far through a
    set the user is, because a set is answered all at once."""
    execution = _parked_questions(scene.session)
    if execution is None:
        raise CatalogError("the questions screen needs a session with a parked ask_user call")
    tool = QuestionsTool(
        store={
            execution.tool_call_id: {"questions": execution.raw_tool_call.arguments.get("questions"), "answer": None}
        }
    )
    questions = question_set_state(tool.pending(execution.tool_call_id))
    state = _with_answers(questions, scene.world)
    return vm.ScreenState(
        **_base(scene),
        questions=state,
        hints=question_hints_for(state),
    )


def _parked_questions(session: AgentSession) -> ToolExecution | None:
    """The `ask_user` call holding the dock: parked at `AWAITING_RESULT`, and
    matched by the same declaration test the live app matches on."""
    for node_id in session.conversations[session.main_conversation_id].nodes:
        entry = session.entries.get(node_id)
        if (
            isinstance(entry, ToolExecution)
            and entry.status is ExecutionStatus.AWAITING_RESULT
            and is_questions_tool(entry)
        ):
            return entry
    return None


def _with_answers(state: vm.QuestionSetState, world: World) -> vm.QuestionSetState:
    """The world's answers applied to a freshly built set.

    A single-select question carries its pick on `answer`; a multi-select
    carries it on each option's `checked`. The caret follows the pick, which is
    where the live dock leaves it after answering."""
    questions: list[vm.Question] = []
    for index, question in enumerate(state.questions):
        picked = world.answers.get(index)
        if picked is None:
            questions.append(question)
        elif question.mode == "multi":
            options = [
                option.model_copy(update={"checked": position in picked})
                for position, option in enumerate(question.options)
            ]
            questions.append(question.model_copy(update={"options": options, "answered": True}))
        else:
            first = picked[0] if picked else None
            questions.append(question.model_copy(update={"answer": first, "selected": first or 0, "answered": True}))
    return state.model_copy(
        update={
            "questions": questions,
            "active": world.active,
            "phase": world.phase,
            "extra": world.extra,
        }
    )


def palette_screen(scene: Scene) -> vm.ScreenState:
    """1e — the command palette over a dimmed transcript."""
    everything = palette_rows()
    rows = filter_rows(everything, scene.world.query)
    return vm.ScreenState(
        **_base(scene),
        overlay=vm.OverlayState(
            mode="palette",
            rows=rows,
            query=scene.world.query,
            sigil="/",
            counter=f"{len(rows)} of {len(everything)}",
            selected=scene.world.selected,
        ),
        hints=HINTS["palette"],
    )


def picker_screen(scene: Scene) -> vm.ScreenState:
    """1f — the `@` file picker, fuzzy-ranked, with match spans."""
    tokens_by_path = {entry.path: entry.tokens for entry in scene.world.files}
    ranked = rank_files([entry.path for entry in scene.world.files], scene.world.query)
    matches = [(path, marked, tokens_by_path[path]) for path, marked in ranked]
    return vm.ScreenState(
        **_base(scene),
        overlay=vm.OverlayState(
            mode="picker",
            rows=picker_rows(matches, scene.world.checked),
            query=scene.world.query,
            sigil="@",
            counter=f"{len(matches)} files",
            selected=scene.world.selected,
            column=36,
        ),
        hints=HINTS["picker"],
    )


def cost_screen(scene: Scene) -> vm.ScreenState:
    """1k — token and spend breakdown, meters proportional to COST."""
    return vm.ScreenState(
        **_base(scene, label="cost · this session"),
        modal=vm.ModalState(cost=cost_state(scene.session)),
        hints=HINTS["cost"],
    )


# One of each state, so the catalog shows what every colour means side by side
# and a snapshot catches a regression in any of them.
MCP_SERVERS: tuple[ServerStatus, ...] = (
    ServerStatus(
        label="files",
        state=ServerState.CONNECTED,
        tool_count=14,
        protocol_version="2025-11-25",
    ),
    ServerStatus(
        label="linear",
        state=ServerState.NEEDS_AUTH,
        oauth=True,
    ),
    # Three tools, matching MCP_TOOLS below, so the list and the drill-in agree.
    ServerStatus(
        label="airtable",
        state=ServerState.CONNECTED,
        tool_count=3,
        protocol_version="2026-07-28",
        rejected_tools={"bulk_upsert": "'x-mcp-header' at rows is on a 'array' parameter"},
    ),
    ServerStatus(
        label="staging",
        state=ServerState.INACTIVE,
        error="Could not start MCP server 'staging': No such file or directory",
    ),
    ServerStatus(label="notion", state=ServerState.DISABLED),
    ServerStatus(
        label="github",
        state=ServerState.STALE,
        tool_count=9,
        error="MCP server 'github' is unreachable: timed out",
    ),
)

# The airtable row's tools, for the drill-in. Specs rather than raw MCP tools,
# because that is what the service hands a UI: the catalog derives them once.
MCP_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="mcp__airtable__list_records",
        description="List records in a table, with optional filtering and sorting.",
        input_schema={"type": "object", "properties": {}},
        metadata={"mcp": {"server": "airtable", "tool": "list_records"}},
    ),
    ToolSpec(
        name="mcp__airtable__create_record",
        description="Create one record in a table.",
        input_schema={"type": "object", "properties": {}},
        metadata={"mcp": {"server": "airtable", "tool": "create_record"}},
    ),
    ToolSpec(
        name="mcp__airtable__search_records",
        description="Full-text search across a table's records.",
        input_schema={"type": "object", "properties": {}},
        metadata={"mcp": {"server": "airtable", "tool": "search_records"}},
    ),
)


def mcp_screen(scene: Scene) -> vm.ScreenState:
    """1l — MCP servers, one row per state: connected, stale, waiting on a
    login, failed, and switched off."""
    state = build_mcp_state(list(MCP_SERVERS), selected=scene.world.selected)
    return vm.ScreenState(
        **_base(scene, label="mcp servers"),
        modal=vm.ModalState(mcp=state),
        hints=mcp_hints(state),
    )


def mcp_detail_screen(scene: Scene) -> vm.ScreenState:
    """1m — one server's tools, including the ones excluded and why."""
    status = MCP_SERVERS[2]
    detail = build_mcp_detail(status, MCP_TOOLS)
    state = build_mcp_state(list(MCP_SERVERS), selected=2, detail=detail)
    return vm.ScreenState(
        **_base(scene, label="mcp servers"),
        modal=vm.ModalState(mcp=state),
        hints=mcp_hints(state),
    )


def settings_screen(scene: Scene) -> vm.ScreenState:
    """1j — model, permissions, appearance."""
    return vm.ScreenState(
        **_base(scene, label="settings · luca.json"),
        modal=vm.ModalState(
            settings=build_settings_state(
                scene.session.session_config.llm_config,
                theme=scene.world.theme,
                streaming=scene.world.streaming,
                mode=scene.world.mode,
                show_counter=scene.world.show_counter,
                selected=scene.world.selected,
            )
        ),
        hints=HINTS["settings"],
    )


def sessions_screen(scene: Scene) -> vm.ScreenState:
    """1i — session history and resume, over the whole catalogued library,
    newest first exactly as `list_sessions` orders a real store."""
    summaries = sorted(
        (_summary(session) for session in scene.library),
        key=lambda summary: summary.modified,
        reverse=True,
    )
    state = build_sessions_state(
        summaries,
        directory_name=".luca/sessions",
        now=NOW,
        selected=scene.world.selected,
    )
    if state is None:
        raise CatalogError("the sessions screen needs at least one session in the library")
    return vm.ScreenState(
        **_base(scene, label="sessions"),
        modal=vm.ModalState(sessions=state),
        hints=HINTS["sessions"],
    )


# How long ago each world was last touched. Declared, not read off the file's
# mtime: the catalog must not change because someone checked a fixture out.
AGES_MINUTES: dict[str, int] = {
    "questions": 7,
    "conversation": 18,
    "questions-answered": 25,
    "planning": 42,
    "todos": 55,
    "approval": 96,
    "todos-done": 130,
    "failures": 191,
    "subagents": 320,
    "empty": 1450,
}
DEFAULT_AGE_MINUTES = 60 * 24


def _summary(session: AgentSession) -> SessionSummary:
    """A library session as a sessions-screen row, through the same
    `summarize` the live store uses."""
    return summarize(
        session,
        path=SESSIONS_DIR / f"{session.id}.json",
        modified=NOW - timedelta(minutes=AGES_MINUTES.get(session.id, DEFAULT_AGE_MINUTES)),
        size_bytes=len(session.model_dump_json()),
    )


SCREENS: dict[ScreenName, Callable[[Scene], vm.ScreenState]] = {
    "chat": chat_screen,
    "approval": approval_screen,
    "questions": questions_screen,
    "palette": palette_screen,
    "picker": picker_screen,
    "cost": cost_screen,
    "settings": settings_screen,
    "sessions": sessions_screen,
    "mcp": mcp_screen,
    "mcp-detail": mcp_detail_screen,
}


# ── loading ───────────────────────────────────────────────────────────────────


def load_session_library(directory: Path | None = None) -> dict[str, AgentSession]:
    """Every committed world, keyed by file stem (`conversation`, `subagents`)."""
    root = directory or SESSIONS_DIR
    if not root.is_dir():
        return {}
    library: dict[str, AgentSession] = {}
    for path in sorted(root.glob("*.json")):
        try:
            library[path.stem] = AgentSession.model_validate_json(path.read_text())
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise CatalogError(f"{path} is not a readable session:\n{exc}") from exc
    return library


def load_catalog(path: Path | None = None) -> list[Entry]:
    """`catalog.yaml` — a list of entries, in catalog order."""
    source = path or CATALOG_PATH
    if not source.is_file():
        return []
    import yaml

    try:
        data = yaml.safe_load(source.read_text())
    except Exception as exc:
        raise CatalogError(f"{source}: not parseable ({exc})") from exc
    if not isinstance(data, list):
        raise CatalogError(f"{source}: the top level must be a list of entries")
    try:
        return [Entry.model_validate(row) for row in data]
    except Exception as exc:
        raise CatalogError(f"{source} has an invalid entry:\n{exc}") from exc


def build_screen(entry: Entry, library: dict[str, AgentSession]) -> vm.ScreenState:
    """One catalog entry as a renderable screen. THE pure function: same entry
    plus same library, same screen, on any machine."""
    session = library.get(entry.session)
    if session is None:
        known = ", ".join(sorted(library)) or "none"
        raise CatalogError(f"{entry.name}: no session {entry.session!r} (library: {known})")
    screen = SCREENS.get(entry.screen)
    if screen is None:  # pragma: no cover — the Literal already rejects it
        raise CatalogError(f"{entry.name}: no screen {entry.screen!r}")
    scene = Scene(session=session, world=entry.world, library=list(library.values()))
    return screen(scene).model_copy(update={"name": entry.name})


def build_catalog(
    catalog_path: Path | None = None,
    sessions_dir: Path | None = None,
) -> list[tuple[Entry, vm.ScreenState]]:
    """The whole catalog, built. Used by the gallery and the snapshot suite."""
    library = load_session_library(sessions_dir)
    return [(entry, build_screen(entry, library)) for entry in load_catalog(catalog_path)]


def catalog_names(catalog_path: Path | None = None) -> list[str]:
    return [entry.name for entry in load_catalog(catalog_path)]
