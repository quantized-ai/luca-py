"""The design-system gallery: boot the frame against declarative states.

    uv run python -m luca.agent.contrib.tui --gallery                  # browse all
    uv run python -m luca.agent.contrib.tui --gallery chat/subagents   # one catalog entry
    uv run python -m luca.agent.contrib.tui --gallery 1a_agent_loop    # one fixture
    uv run python -m luca.agent.contrib.tui --gallery path/to/thing.yaml

Two tiers, both browsable here and both snapshot-tested:

- **The catalog** (`fixtures/catalog.yaml`) — `screen × world`, DERIVED through
  the same projections the live app calls. These cannot depict a feature that
  no longer exists: delete it and they change or stop building. See
  `catalog.py`.
- **Fixtures** (`fixtures/*.yaml`) — hand-authored `ScreenState` documents, for
  states no producer exists for yet (the eleven handoff screens `1a`–`1k` plus
  component sheets). These are specifications, not records.

A path is also accepted directly, including a stored `AgentSession`, which is
recognised by shape and rendered as the screen a resume would show.

Keys: `←`/`→` cycle, `g` opens the index, `ctrl+q` quits.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual.binding import Binding

from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import AgentSession, ToolExecution
from luca.agent.core.projection import ConversationProjector, tool_message_text

from . import state as vm
from .format import HINTS, short_model
from .frame import LucaApp
from .render import plan_from_session, transcript_blocks
from .shells import OverlayListView
from .usage import status_counter

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_SUFFIXES = (".yaml", ".yml", ".json")
# Catalog inputs, not fixtures: `catalog.yaml` is the grid and `sessions/`
# holds the worlds it combines. Both live under `fixtures/` so all declarative
# data has one root, and both are excluded from the hand-authored listing.
_NOT_FIXTURES = ("catalog.yaml", "sessions")


class FixtureError(Exception):
    """A fixture that cannot be found, parsed, or validated."""


def list_fixtures(directory: Path | None = None) -> list[Path]:
    """Every hand-authored fixture, screens first, then component sheets."""
    root = directory or FIXTURES_DIR
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.suffix in _SUFFIXES and not set(path.relative_to(root).parts) & set(_NOT_FIXTURES)
        ),
        key=lambda path: str(path.relative_to(root)),
    )


def fixture_name(path: Path, directory: Path | None = None) -> str:
    root = directory or FIXTURES_DIR
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.stem
    return str(relative.with_suffix("")).replace("\\", "/")


def resolve_fixture(ref: str, directory: Path | None = None) -> Path:
    """A path as given, or a bundled fixture by name (`1a_agent_loop`)."""
    direct = Path(ref)
    if direct.is_file():
        return direct
    root = directory or FIXTURES_DIR
    for suffix in _SUFFIXES:
        candidate = root / f"{ref}{suffix}"
        if candidate.is_file():
            return candidate
    known = ", ".join(fixture_name(path) for path in list_fixtures(directory))
    raise FixtureError(f"no fixture {ref!r}; bundled fixtures: {known}")


def is_session_document(data: dict) -> bool:
    """A stored `AgentSession` rather than a hand-authored screen state.
    Keyed on structure, not on the file name, so any session file works."""
    return "entries" in data and "conversations" in data


def session_state(session: AgentSession, *, cwd: str = "~") -> vm.ScreenState:
    """A stored session as the screen the app would resume it into.

    The transcript comes from the same derivation the live replay uses, so
    this renders a real conversation with no agent, no provider and no key —
    which is what makes a session file browsable in the gallery."""
    projector = ConversationProjector()

    def resolve(execution: ToolExecution) -> tuple[str | None, bool]:
        try:
            message = projector.project_tool_execution(execution, session.entries)
        except ProjectionError:
            return None, False
        return tool_message_text(message), message.is_error

    tokens, cost = status_counter(session)
    return vm.ScreenState(
        name=session.id,
        status=vm.StatusState(
            cwd=cwd,
            model=short_model(session.session_config.llm_config.model),
            tokens=tokens,
            cost=cost,
        ),
        transcript=transcript_blocks(session, resolve_result=resolve),
        plan=plan_from_session(session),
        composer=vm.ComposerState(),
        hints=HINTS["idle"],
    )


def load_fixture(path: Path) -> vm.ScreenState:
    text = path.read_text()
    try:
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as exc:
        raise FixtureError(f"{path}: not parseable ({exc})") from exc
    if not isinstance(data, dict):
        raise FixtureError(f"{path}: the top level must be a mapping")
    if is_session_document(data):
        try:
            return session_state(AgentSession.model_validate(data))
        except FixtureError:
            raise
        except Exception as exc:
            raise FixtureError(f"{path} is not a readable session:\n{exc}") from exc
    try:
        return vm.ScreenState.model_validate(data)
    except Exception as exc:
        raise FixtureError(f"{path} is not a valid screen state:\n{exc}") from exc


# ── browsable items: catalog entries and fixtures, one vocabulary ─────────────


@dataclass(frozen=True)
class Item:
    """One browsable state: a name and how to build it. A catalog entry builds
    through its projection; a fixture builds by loading its document."""

    name: str
    build: Callable[[], vm.ScreenState]


def catalog_items() -> list[Item]:
    """The derived tier, in catalog order."""
    from .catalog import Entry, build_screen, load_catalog, load_session_library

    library = load_session_library()

    def make(entry: Entry) -> Item:
        return Item(name=entry.name, build=lambda: build_screen(entry, library))

    return [make(entry) for entry in load_catalog()]


def fixture_items(directory: Path | None = None) -> list[Item]:
    """The hand-authored tier."""
    return [
        Item(name=fixture_name(path, directory), build=lambda path=path: load_fixture(path))
        for path in list_fixtures(directory)
    ]


def all_items() -> list[Item]:
    """Everything the gallery browses: the catalog first, then the fixtures."""
    return [*catalog_items(), *fixture_items()]


def resolve_item(ref: str) -> Item:
    """A catalog entry name, a bundled fixture name, or a path to either a
    `ScreenState` document or a stored `AgentSession`. A name wins over a
    path: `chat/empty` is the catalog entry, never a file that happens to
    sit at that relative path."""
    return _resolve(ref)[0]


def _resolve(ref: str) -> tuple[Item, bool]:
    """The item, plus whether it is one of the bundled ones — a file given by
    path is NOT, even when its stem matches something bundled, or opening
    `./scratch/1a_agent_loop.yaml` would quietly render the bundled `1a`."""
    for item in catalog_items():
        if item.name == ref:
            return item, True
    direct = Path(ref)
    if direct.is_file():
        return Item(name=direct.stem, build=lambda: load_fixture(direct)), False
    path = resolve_fixture(ref)
    return Item(name=fixture_name(path), build=lambda: load_fixture(path)), True


class GalleryApp(LucaApp):
    """The frame booted against declarative states instead of a live agent."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("right", "cycle(1)", show=False, priority=True),
        Binding("left", "cycle(-1)", show=False, priority=True),
        Binding("g", "index", show=False, priority=True),
    ]

    def __init__(self, items: list[Item], *, index: int = 0) -> None:
        super().__init__()
        if not items:
            raise FixtureError("the gallery needs at least one item")
        self.items = items
        self.index = index
        self._browsing = False

    async def on_mount(self) -> None:
        super().on_mount()
        await self.apply_current()

    async def apply_current(self) -> None:
        item = self.items[self.index]
        await self.apply_state(item.build())
        self.sub_title = f"{item.name} · {self.index + 1}/{len(self.items)}"

    async def action_cycle(self, delta: int) -> None:
        self._browsing = False
        self.index = (self.index + delta) % len(self.items)
        await self.apply_current()

    async def action_index(self) -> None:
        rows = [vm.OverlayRow(primary=item.name) for item in self.items]
        self._browsing = True
        await self.show_overlay(
            vm.OverlayState(
                mode="menu",
                rows=rows,
                sigil="g",
                counter=f"{len(rows)} states",
                selected=self.index,
                column=40,
            )
        )
        self.set_hints(["↑↓ move", "enter show", "esc back"])

    async def on_overlay_list_view_committed(self, message: OverlayListView.Committed) -> None:
        if not self._browsing:
            return
        self._browsing = False
        self.index = message.index
        await self.apply_current()

    async def on_overlay_list_view_dismissed(self, message: OverlayListView.Dismissed) -> None:
        if not self._browsing:
            return
        self._browsing = False
        await self.apply_current()


def run_gallery(ref: str | None = None) -> None:
    """CLI entry: one named state, or browse everything.

    A named state still opens inside the whole listing when it belongs to it,
    so `←`/`→` keep working from wherever you started."""
    everything = all_items()
    if ref in (None, "", "all"):
        if not everything:
            raise FixtureError(f"nothing to browse under {FIXTURES_DIR}")
        GalleryApp(everything).run()
        return
    item, bundled = _resolve(ref)
    names = [existing.name for existing in everything]
    if bundled and item.name in names:
        GalleryApp(everything, index=names.index(item.name)).run()
        return
    GalleryApp([item]).run()
