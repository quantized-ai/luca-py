"""The multiline prompt box.

`PromptInput` is a `TextArea` shaped like a chat input rather than an
editor: Enter submits (posting `PromptInput.Submitted`), Alt+Enter,
Shift+Enter and Ctrl+J insert a newline, and Enter after a trailing
backslash swaps the backslash for a newline. The backslash form is the
universal fallback: Textual requests the kitty keyboard protocol on
startup, but a terminal that doesn't honor it collapses both modified
Enters into a plain Enter before the app ever sees them. A bracketed
paste lands through `TextArea`'s own paste handling — newlines included,
never submitting.

The box grows with its content (`height: auto`, capped) and reuses
`TextArea.suggestion` for slash-command completion: the remainder of the
first match renders inline and Right accepts it, standing in for the
`Input` suggester this widget replaced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from textual import events
from textual.message import Message
from textual.widgets import TextArea

_NEWLINE_KEYS = frozenset({"alt+enter", "shift+enter", "ctrl+j"})


class PromptInput(TextArea):
    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        max-height: 10;
    }
    """

    @dataclass
    class Submitted(Message):
        """Posted on Enter with the full (possibly multiline) text."""

        prompt_input: PromptInput
        value: str

        @property
        def control(self) -> PromptInput:
            return self.prompt_input

    _commands: Sequence[str] = ()

    def __init__(
        self,
        *,
        placeholder: str = "",
        commands: Sequence[str] = (),
        id: str | None = None,  # noqa: A002
    ) -> None:
        super().__init__(placeholder=placeholder, id=id)
        self._commands = list(commands)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in _NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            row, column = self.cursor_location
            if column > 0 and self.document[row][column - 1] == "\\":
                self._replace_via_keyboard("\n", (row, column - 1), (row, column))
                return
            self.post_message(self.Submitted(self, self.text))
            return
        await super()._on_key(event)

    def update_suggestion(self) -> None:
        """Offer the remainder of the first slash command matching the typed
        prefix — only while the whole document is that one prefix, so the
        ghost text always renders where the completion would land."""
        self.suggestion = ""
        if self.document.line_count != 1 or self.cursor_location != self.document.end:
            return
        typed = self.document[0]
        if not typed.startswith("/"):
            return
        for command in self._commands:
            if command.lower().startswith(typed.lower()) and command != typed:
                self.suggestion = command[len(typed) :]
                return
