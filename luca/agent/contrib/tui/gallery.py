"""The design-system gallery: boot the frame against declarative fixtures.

    uv run python -m luca.agent.contrib.tui --gallery              # browse all
    uv run python -m luca.agent.contrib.tui --gallery 1a_agent_loop
    uv run python -m luca.agent.contrib.tui --gallery path/to/custom.yaml

A fixture is a YAML (or JSON) document validated as `state.ScreenState` —
the same view-model the live app renders, so anything the product can show
can be authored, reviewed and snapshot-tested here without driving an agent.
The bundled fixtures live in `fixtures/`: the eleven handoff screens
(`1a`–`1k`) plus component sheets (every block and shell variant). This IS
the component catalog; add a fixture for any new component or state.

Keys: `←`/`→` cycle fixtures, `g` opens the fixture index, `ctrl+q` quits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from textual.binding import Binding

from . import state as vm
from .frame import LucaApp
from .shells import OverlayListView

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_SUFFIXES = (".yaml", ".yml", ".json")


class FixtureError(Exception):
    """A fixture that cannot be found, parsed, or validated."""


def list_fixtures(directory: Path | None = None) -> list[Path]:
    """Every bundled fixture, screens first, then component sheets."""
    root = directory or FIXTURES_DIR
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.suffix in _SUFFIXES),
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
    try:
        return vm.ScreenState.model_validate(data)
    except Exception as exc:
        raise FixtureError(f"{path} is not a valid screen state:\n{exc}") from exc


class GalleryApp(LucaApp):
    """The frame booted against fixtures instead of a live agent."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("right", "cycle(1)", show=False, priority=True),
        Binding("left", "cycle(-1)", show=False, priority=True),
        Binding("g", "index", show=False, priority=True),
    ]

    def __init__(self, fixtures: list[Path], *, index: int = 0) -> None:
        super().__init__()
        if not fixtures:
            raise FixtureError("the gallery needs at least one fixture")
        self.fixtures = fixtures
        self.index = index
        self._browsing = False

    async def on_mount(self) -> None:
        super().on_mount()
        await self.apply_current()

    async def apply_current(self) -> None:
        state = load_fixture(self.fixtures[self.index])
        await self.apply_state(state)
        name = fixture_name(self.fixtures[self.index])
        self.sub_title = f"{name} · {self.index + 1}/{len(self.fixtures)}"

    async def action_cycle(self, delta: int) -> None:
        self._browsing = False
        self.index = (self.index + delta) % len(self.fixtures)
        await self.apply_current()

    async def action_index(self) -> None:
        rows = [vm.OverlayRow(primary=fixture_name(path)) for path in self.fixtures]
        self._browsing = True
        await self.show_overlay(
            vm.OverlayState(
                mode="menu",
                rows=rows,
                sigil="g",
                counter=f"{len(rows)} fixtures",
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
    """CLI entry: one named fixture, or browse them all."""
    if ref in (None, "", "all"):
        fixtures = list_fixtures()
        if not fixtures:
            raise FixtureError(f"no fixtures found under {FIXTURES_DIR}")
        GalleryApp(fixtures).run()
        return
    path = resolve_fixture(ref)
    fixtures = list_fixtures()
    index = fixtures.index(path) if path in fixtures else 0
    if path not in fixtures:
        fixtures = [path]
    GalleryApp(fixtures, index=index).run()
