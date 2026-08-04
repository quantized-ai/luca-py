"""The frame chrome: status bar, hint legend, composer.

All three are custom widgets, not Textual's stock Header/Footer — the design
forbids those. The status bar degrades below the design width by dropping
segments (branch, then cwd, then model) while keeping the mark and the
counter; below 60 columns the app swaps the whole frame for a single centred
line (handled by the app, not here).
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from . import state as vm
from .format import GLYPH_MARK, GLYPH_USER
from .prompt import PromptInput
from .theme import resolve_tokens

_SEGMENT_GAP = "  "  # 2 cells between status segments
_HINT_GAP = "   "  # 3 cells between hint segments


class StatusBar(Static):
    """`◧ luca  ~/quantized/luca  sonnet-4.5  main*        12.4k · $0.08`.

    On modal screens the model/branch/counter run is replaced by a single
    right-aligned label (`sessions`, `settings · .luca/config.toml`)."""

    def __init__(self, model: vm.StatusState | None = None, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(markup=False, id=id)
        self.model = model or vm.StatusState()

    def set_state(self, model: vm.StatusState) -> None:
        self.model = model
        self.refresh()

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        model = self.model
        faint = Style(color=tokens.faint)

        mark = Text()
        mark.append(GLYPH_MARK, style=Style(color=tokens.foreground))
        mark.append(" ")
        mark.append("luca", style=Style(color=tokens.foreground, bold=True))

        if model.label is not None:
            right = Text(model.label, style=faint)
        else:
            counter = " · ".join(part for part in (model.tokens, model.cost) if part)
            right = Text(counter, style=faint)

        cwd = Text(model.cwd, style=faint) if model.cwd else None
        the_model = Text(model.model, style=faint) if model.label is None and model.model else None
        branch = (
            Text(model.branch + ("*" if model.dirty else ""), style=faint)
            if model.label is None and model.branch
            else None
        )

        width = self.size.width
        # Drop order below the design width: branch, then cwd, then model —
        # the mark and the counter always survive.
        for candidate in ([cwd, the_model, branch], [cwd, the_model], [the_model], []):
            kept = [segment for segment in candidate if segment is not None]
            line = self._assemble(mark, kept, right, width)
            if line is not None:
                return line
        return self._assemble(mark, [], right, width) or mark

    @staticmethod
    def _assemble(mark: Text, middles: list[Text], right: Text, width: int) -> Text | None:
        left = Text(no_wrap=True)
        left.append_text(mark.copy())
        for segment in middles:
            left.append(_SEGMENT_GAP)
            left.append_text(segment.copy())
        pad = width - left.cell_len - right.cell_len
        if pad < 2 and middles:
            return None
        line = left
        if right.cell_len:
            line.append(" " * max(pad, 1))
            line.append_text(right.copy())
        return line


class HintLegend(Static):
    """The always-visible key legend; segments 3 cells apart, `$text-faint`.
    The only discoverability mechanism in the app — required chrome."""

    def __init__(self, hints: list[str] | None = None, *, id: str | None = None) -> None:  # noqa: A002
        super().__init__(markup=False, id=id)
        self.hints = hints or []

    def set_hints(self, hints: list[str]) -> None:
        self.hints = list(hints)
        self.refresh()

    def render(self) -> Text:
        return Text(_HINT_GAP.join(self.hints), no_wrap=True, overflow="ellipsis")


class Composer(Horizontal):
    """The bordered input: `›` in accent in cell 1, one space, the input.
    While a turn runs the placeholder becomes `working…`; typing then queues
    (the input stays enabled — queueing is the feature)."""

    def __init__(self, model: vm.ComposerState | None = None, *, commands: list[str] | None = None) -> None:
        super().__init__(id="composer")
        self.model = model or vm.ComposerState()
        self._commands = commands or []

    def compose(self) -> ComposeResult:
        yield Static(GLYPH_USER, markup=False, classes="composer-glyph")
        yield PromptInput(
            placeholder=self.model.placeholder,
            commands=self._commands,
        )

    def on_mount(self) -> None:
        prompt = self.query_one(PromptInput)
        if self.model.text:
            prompt.load_text(self.model.text)
        prompt.disabled = self.model.disabled

    @property
    def input(self) -> PromptInput:
        return self.query_one(PromptInput)

    def set_placeholder(self, placeholder: str) -> None:
        self.model = self.model.model_copy(update={"placeholder": placeholder})
        self.input.placeholder = placeholder
