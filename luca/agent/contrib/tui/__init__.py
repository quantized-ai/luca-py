"""luca.agent.contrib.tui — the Textual terminal UI, built as a design system.

A full-screen (alt-screen) app: one scrolling conversation column, the sticky
todo panel beneath it, a single dock slot (composer / approval prompt /
overlay list), and modal screens for sessions, settings and cost. The visual
spec is the design handoff's eleven screens; every state is expressible as
declarative data and renderable without an agent.

Layering (the Textual-free modules are the unit-testable core):

- `theme`     — the palette as a registered Theme; the only hex source.
- `state`     — the view-model / fixture schema (`ScreenState`, the blocks).
- `format` / `render` / `usage` / `approvals` / `sessions` / `files` /
  `gitinfo` / `config` — pure logic, no Textual.
- `blocks` / `chrome` / `shells` / `modals` / `frame` — the widgets and the
  `LucaApp` frame (`app.tcss` holds every geometry and color assignment).
- `catalog`   — the derived catalog: `screen × world → ScreenState`, built
  from committed sessions through the app's own projections.
- `gallery`   — `GalleryApp` (`--gallery`) over the catalog and the
  hand-authored fixtures alike.
- `app`       — `AgentApp`, the live agent wired onto the frame.
- `cli`       — the argparse entry point (`python -m luca.agent.contrib.tui`).

Requires the `tui` dependency group (`textual`, `pyyaml`). Importing this
package root pulls in Textual; the pure modules can be imported directly
without it.
"""

from luca.agent.contrib.app.wiring import build_faux_provider, build_runner

from .app import AgentApp
from .cli import main
from .frame import LucaApp
from .gallery import GalleryApp

__all__ = ["AgentApp", "GalleryApp", "LucaApp", "build_faux_provider", "build_runner", "main"]
