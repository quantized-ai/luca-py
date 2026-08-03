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
from textual.widgets import Button, Input, Label, OptionList

from .approvals import ApprovalPrompt, PromptOption

# Keys a filterable picker hands to the option list instead of the filter input.
NAVIGATION_KEYS = {
    "up": "action_cursor_up",
    "down": "action_cursor_down",
    "home": "action_first",
    "end": "action_last",
    "pageup": "action_page_up",
    "pagedown": "action_page_down",
}


class ApprovalScreen(ModalScreen[PromptOption]):
    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 70%;
        max-width: 100;
        height: 80%;
        padding: 1 2 0 2;
        border: thick $primary;
        background: $surface;
    }
    #approval-title {
        height: auto;
        margin-bottom: 1;
    }
    #approval-details {
        height: 1fr;
        min-height: 3;
        margin-bottom: 1;
    }
    #approval-details Label {
        height: auto;
        margin-bottom: 1;
    }
    #approval-options {
        height: auto;
        min-height: 3;
        max-height: 50%;
    }
    #approval-options Button {
        width: 100%;
        margin-bottom: 1;
    }
    #approval-options Button.-last-option {
        margin-bottom: 0;
    }
    """

    def __init__(self, prompt: ApprovalPrompt) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        prompt = self.prompt
        title = f"Approval needed: {prompt.tool_name}"
        if prompt.conversation_id is not None:
            # The task if the caller named it, the id if nobody did — a gate
            # you cannot attribute is worse than one attributed by key.
            who = prompt.conversation_label or prompt.conversation_id
            title += f"  [subagent · {who}]"
        if prompt.total_steps > 1:
            title += f"  (step {prompt.step}/{prompt.total_steps})"
        with Container(id="approval-dialog"):
            yield Label(title, markup=False, id="approval-title")
            with VerticalScroll(id="approval-details"):
                if prompt.resources:
                    yield Label(
                        "resources: " + ", ".join(prompt.resources),
                        markup=False,
                        id="approval-resources",
                    )
                yield Label(prompt.preview, markup=False, id="approval-preview")
            with VerticalScroll(id="approval-options"):
                for index, option in enumerate(prompt.options):
                    yield Button(
                        Content(f"{index + 1}. {option.label}"),
                        id=f"approval-option-{index}",
                        classes="-last-option" if index == len(prompt.options) - 1 else None,
                        variant=self._variant(option, index),
                    )

    def on_mount(self) -> None:
        self.query_one("#approval-options", VerticalScroll).focus()

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
    string, or None on cancel.

    Pass `labels` when the row should read differently from the value it
    returns — a session picker shows a title and a timestamp but returns the
    session id. Same length as `options`, positionally matched.

    Pass `filterable=True` for a list too long to arrow through — openrouter
    alone contributes hundreds of models. Typing narrows the rows, and the
    selected index resolves against what is VISIBLE, never against the full
    list. `self._visible` is that mapping and is the whole reason filtering is
    not just a display concern."""

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
    #picker-filter {
        margin-bottom: 1;
    }
    #picker-options {
        height: auto;
        max-height: 20;
    }
    """

    def __init__(
        self,
        title: str,
        options: list[str],
        *,
        current: str | None = None,
        labels: list[str] | None = None,
        filterable: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._current = current
        self._labels = labels
        self._filterable = filterable
        # Index into `_options` for each row currently on screen.
        self._visible: list[int] = list(range(len(options)))

    def _rows(self) -> list[str]:
        """Every row's text, current-marked, in `_options` order."""
        rows = self._labels if self._labels is not None else self._options
        return [
            f"{row} (current)" if option == self._current else row
            for option, row in zip(self._options, rows, strict=True)
        ]

    def compose(self) -> ComposeResult:
        with Container(id="picker-dialog"):
            yield Label(self._title, markup=False, id="picker-title")
            if self._filterable:
                yield Input(placeholder="filter…", id="picker-filter")
            yield OptionList(*self._rows(), id="picker-options")

    def on_mount(self) -> None:
        options = self.query_one("#picker-options", OptionList)
        options.highlighted = self._options.index(self._current) if self._current in self._options else 0
        # The filter takes focus when there is one, so typing narrows the list
        # instead of jumping the highlight around by first letter.
        if self._filterable:
            self.query_one("#picker-filter", Input).focus()
        else:
            options.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = self._rows()
        self._visible = [index for index, row in enumerate(rows) if needle in row.lower()]
        options = self.query_one("#picker-options", OptionList)
        options.clear_options()
        options.add_options([rows[index] for index in self._visible])
        if self._visible:
            options.highlighted = 0

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        event.stop()
        self.dismiss(self._options[self._visible[event.option_index]])

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if not self._filterable:
            return
        # The filter holds focus so typing narrows the list, which means every
        # navigation key has to be handed on or the list becomes unreachable.
        options = self.query_one("#picker-options", OptionList)
        if event.key == "enter":
            event.stop()
            if options.highlighted is not None and self._visible:
                self.dismiss(self._options[self._visible[options.highlighted]])
        elif action := NAVIGATION_KEYS.get(event.key):
            event.stop()
            getattr(options, action)()
