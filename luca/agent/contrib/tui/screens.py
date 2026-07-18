"""The approval modal.

`ApprovalScreen` displays one `ApprovalPrompt` and dismisses with the picked
`PromptOption` — all decision policy lives in `approvals.py`; this screen is
pure presentation. Select by button, by digit (1..n), or Escape / "a" for the
abandon option (always last).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from .approvals import ApprovalPrompt, PromptOption


class ApprovalScreen(ModalScreen[PromptOption]):
    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 70%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    #approval-dialog Label {
        margin-bottom: 1;
    }
    #approval-options {
        height: auto;
        max-height: 20;
    }
    #approval-options Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def __init__(self, prompt: ApprovalPrompt) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        prompt = self.prompt
        title = f"Approval needed: {prompt.tool_name}"
        if prompt.total_steps > 1:
            title += f"  (step {prompt.step}/{prompt.total_steps})"
        with Container(id="approval-dialog"):
            yield Label(title, markup=False, id="approval-title")
            if prompt.resources:
                yield Label(
                    "resources: " + ", ".join(prompt.resources),
                    markup=False, id="approval-resources",
                )
            yield Label(prompt.preview, markup=False, id="approval-preview")
            with VerticalScroll(id="approval-options"):
                for index, option in enumerate(prompt.options):
                    yield Button(
                        Content(f"{index + 1}. {option.label}"),
                        id=f"approval-option-{index}",
                        variant=self._variant(option, index),
                    )

    @staticmethod
    def _variant(option: PromptOption, index: int) -> str:
        if option.is_abandon:
            return "warning"
        if option.is_deny:
            return "error"
        return "primary" if index == 0 else "default"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        index = int(str(event.button.id).rsplit("-", 1)[1])
        self.dismiss(self.prompt.options[index])

    def on_key(self, event: Key) -> None:
        if event.key.isdigit():
            index = int(event.key) - 1
            if 0 <= index < len(self.prompt.options):
                event.stop()
                self.dismiss(self.prompt.options[index])
        elif event.key in ("escape", "a"):
            event.stop()
            self.dismiss(self.prompt.options[-1])  # abandon is always last
