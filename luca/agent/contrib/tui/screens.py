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
from textual.widgets import Button, Label, OptionList

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


class PickerScreen(ModalScreen[str | None]):
    """A single-choice list picker: arrow keys to move, Enter to select, Esc to
    cancel. `self._options` holds the values; the displayed row equals the value
    except the current one, which is shown with a "(current)" suffix. Selection
    maps back by index, so it returns the raw value. Dismisses with the chosen
    string, or None on cancel."""

    DEFAULT_CSS = """
    PickerScreen {
        align: center middle;
    }
    #picker-dialog {
        width: 70%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    #picker-dialog Label {
        margin-bottom: 1;
    }
    #picker-options {
        height: auto;
        max-height: 20;
    }
    """

    def __init__(
        self, title: str, options: list[str], *, current: str | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._current = current

    def compose(self) -> ComposeResult:
        labels = [
            f"{opt} (current)" if opt == self._current else opt
            for opt in self._options
        ]
        with Container(id="picker-dialog"):
            yield Label(self._title, markup=False, id="picker-title")
            yield OptionList(*labels, id="picker-options")

    def on_mount(self) -> None:
        options = self.query_one("#picker-options", OptionList)
        options.highlighted = (
            self._options.index(self._current)
            if self._current in self._options else 0
        )
        options.focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        event.stop()
        self.dismiss(self._options[event.option_index])

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
