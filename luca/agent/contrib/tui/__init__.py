"""luca.agent.contrib.tui — a Textual terminal UI for the agent loop.

The interactive counterpart of the classic REPL demo: a full-screen chat
transcript, an input box, a modal approval gate, live streaming, and Esc
cancellation — driven by the same `PluginAgentSessionRunner` wiring (shell +
memory plugins, the demo math tools, one shared `PermissionStrategy`).

Layering (the Textual-free modules are the unit-testable core):

- `sessions`  — `<session-id>.json` load / save / fork.
- `wiring`    — `build_runner()`: the full agent composition; `build_faux_provider()`
                for the offline scripted demo.
- `approvals` — the pure approval-prompt model (`ApprovalPrompt`,
                `PromptOption`, `build_approval_prompts`): main-loop policy
                translated to data the modal can display.
- `render`    — pure text formatting for transcript cells.
- `cells` / `screens` / `app` — the Textual widgets, the approval modal, and
                `AgentApp` itself.
- `cli`       — the argparse entry point (`python -m luca.agent.contrib.tui`).

Requires the `tui` dependency group (`textual`). Importing this package root
pulls in Textual; the pure modules can be imported directly without it.
"""

from .app import AgentApp
from .cli import main
from .wiring import build_faux_provider, build_runner

__all__ = ["AgentApp", "build_faux_provider", "build_runner", "main"]
