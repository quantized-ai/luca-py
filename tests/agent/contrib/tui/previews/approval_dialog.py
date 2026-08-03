"""Preview the production approval dialog with an oversized bash request.

Run with:

    uv run textual run tests/agent/contrib/tui/previews/approval_dialog.py

The host opens the real :class:`ApprovalScreen`. Press ``t`` to toggle between
dark and light themes, select an option with its digit, and press ``r`` to open
the dialog again after dismissing it.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.contrib.tui.approvals import PromptOption, build_approval_prompts
from luca.agent.contrib.tui.cells import ToolCallCell, UserCell
from luca.agent.contrib.tui.screens import ApprovalScreen
from luca.agent.core.models import ExecutionStatus, ToolCall, ToolExecution, ToolKind, ToolSpec

LONG_PYTHON_COMMAND = """python -c '
from pathlib import Path
import ast
import json

root = Path("luca")
files = sorted(root.rglob("*.py"))
report = []

for path in files:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = sorted(
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    )
    functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]
    todos = [
        line_number
        for line_number, line in enumerate(source.splitlines(), 1)
        if "TODO" in line or "FIXME" in line
    ]
    report.append(
        {
            "path": str(path),
            "lines": len(source.splitlines()),
            "imports": imports,
            "functions": functions,
            "classes": classes,
            "todos": todos,
        }
    )

summary = {
    "files_scanned": len(report),
    "total_lines": sum(item["lines"] for item in report),
    "files_with_todos": [item["path"] for item in report if item["todos"]],
    "largest_modules": sorted(
        report,
        key=lambda item: item["lines"],
        reverse=True,
    )[:20],
    "all_modules": report,
}
print(json.dumps(summary, indent=2, sort_keys=True))
'"""

PREVIEW_EXECUTION = ToolExecution(
    id="preview-execution",
    created_at=1,
    conversation_id="preview-conversation",
    tool_call_id="preview-tool-call",
    raw_tool_call=ToolCall(
        id="preview-tool-call",
        name="bash",
        arguments={
            "command": LONG_PYTHON_COMMAND,
            "timeout": 120_000,
            "workdir": "/workspace/python-py",
        },
    ),
    tool_spec=ToolSpec(
        name="bash",
        description="Execute a shell command.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": ["integer", "null"]},
                "workdir": {"type": ["string", "null"]},
            },
            "required": ["command"],
        },
        tool_kind=ToolKind.EXECUTE,
    ),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [
                        {
                            "permission": "bash",
                            "resource": LONG_PYTHON_COMMAND,
                        }
                    ],
                    "answer_options": [
                        {
                            "resource_permissions": [
                                {
                                    "permission": "bash",
                                    "resource": "python *",
                                }
                            ],
                            "metadata": {
                                "preview": "Always allow Python commands",
                            },
                        }
                    ],
                    "metadata": {
                        "preview": f"Run command: {LONG_PYTHON_COMMAND}",
                    },
                }
            ]
        }
    },
)

PREVIEW_PROMPT = build_approval_prompts(
    PREVIEW_EXECUTION,
    PermissionStrategy(),
)[0]


class ApprovalDialogStory(Vertical):
    """The reusable catalog story underneath the production approval modal."""

    DEFAULT_CSS = """
    ApprovalDialogStory > .transcript {
        height: 1fr;
        padding: 1 0 0 0;
    }
    ApprovalDialogStory > .preview-status {
        height: auto;
        margin: 0 2;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="transcript"):
            yield UserCell("Inspect the Python package and summarize its structure.")
            yield ToolCallCell(PREVIEW_EXECUTION)
        yield Static(
            "Choose an option, then reopen the dialog from the catalog.",
            classes="preview-status",
        )

    def show_dialog(self) -> None:
        if isinstance(self.app.screen, ApprovalScreen):
            return
        self.app.push_screen(ApprovalScreen(PREVIEW_PROMPT), self._record_choice)

    def _record_choice(self, choice: PromptOption) -> None:
        self.query_one(".preview-status", Static).update(
            f"Selected: {choice.label}. Reopen the dialog to try another option.",
        )


class ApprovalDialogPreview(App[None]):
    """A manually runnable visual scenario for the approval modal."""

    TITLE = "luca · approval dialog preview"
    SUB_TITLE = "textual-dark"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "toggle_preview_theme", "Toggle theme", priority=True),
        Binding("r", "show_dialog", "Reopen dialog"),
        Binding("q", "quit", "Quit", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ApprovalDialogStory(id="approval-dialog-story")
        yield Footer()

    def on_mount(self) -> None:
        self.action_show_dialog()

    def action_show_dialog(self) -> None:
        self.query_one(ApprovalDialogStory).show_dialog()

    def action_toggle_preview_theme(self) -> None:
        self.theme = "atom-one-light" if self.theme == "textual-dark" else "textual-dark"
        self.sub_title = self.theme


app = ApprovalDialogPreview()


if __name__ == "__main__":
    app.run()
