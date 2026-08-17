"""Pure text/format helpers for the design system. No Textual imports.

Two span vocabularies, deliberately separate:

- `spans()` parses THEME-TOKEN tags (`[accent]…[/]`, `[error]…[/]`) and is
  only ever fed app-authored or fixture-authored strings — never raw model
  output, which may legally contain brackets.
- `code_spans()` colors `` `code spans` `` in `$accent` and is safe for model
  output (backticks are the only markup it reads).

Colors come in as a resolved `Tokens` (from `theme.resolve_tokens`) so this
module never sees a literal.
"""

from __future__ import annotations

import re
from datetime import datetime

from rich.style import Style
from rich.text import Text

from .theme import Tokens

# Glyph vocabulary (the whole set the design allows):
# › ∴ ▸ │ ❯ ☐ ☑ ◉ ✓ ▪ ◧ ← → ↑ ↓ ⇥ − + █ ▎ (+ ▕▏ for the outlined swatch)
GLYPH_USER = "›"
GLYPH_THINKING = "∴"
GLYPH_TOOL = "▸"
GLYPH_CARET = "❯"
GLYPH_BAR = "▎"
GLYPH_MARK = "◧"
GLYPH_CURSOR = "█"
GLYPH_BOX = "☐"
GLYPH_BOX_TICKED = "☑"
GLYPH_ARROW = "→"
MINUS = "−"  # U+2212, never a hyphen

LIST_GLYPHS = {
    "done": GLYPH_BOX_TICKED,
    "active": "◉",
    "pending": GLYPH_BOX,
    "included": "✓",
    "none": " ",
}

TOOL_NAME_COLUMN = 6
TEXT_MEASURE = 84
MIN_COLUMNS = 60

_SPAN_RE = re.compile(r"\[(fg|muted|faint|accent|success|error)\]|\[/\]")


def _token_color(tokens: Tokens, tag: str) -> str:
    return {
        "fg": tokens.foreground,
        "muted": tokens.muted,
        "faint": tokens.faint,
        "accent": tokens.accent,
        "success": tokens.success,
        "error": tokens.error,
    }[tag]


def spans(text: str, tokens: Tokens, default: str | Style | None = None) -> Text:
    """`[accent]…[/]`-tagged text as a Rich Text. Unknown brackets pass
    through literally; tags do not nest (a new tag replaces the current)."""
    result = Text(style=default or "")
    position = 0
    current: str | None = None
    for match in _SPAN_RE.finditer(text):
        if match.start() > position:
            style = Style(color=_token_color(tokens, current)) if current else ""
            result.append(text[position : match.start()], style=style)
        current = match.group(1)  # None for [/]
        position = match.end()
    if position < len(text):
        style = Style(color=_token_color(tokens, current)) if current else ""
        result.append(text[position:], style=style)
    return result


def code_spans(text: str, tokens: Tokens) -> Text:
    """Model-output-safe: `` `span` `` renders in `$accent` without the
    backticks; everything else is verbatim."""
    result = Text()
    accent = Style(color=tokens.accent)
    for index, part in enumerate(text.split("`")):
        if index % 2 == 1 and part:
            result.append(part, style=accent)
        else:
            result.append(part)
    return result


# ── humanized figures ─────────────────────────────────────────────────────────


def fmt_tokens(count: int) -> str:
    """`12.4k` — one decimal above a thousand, plain below."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def fmt_bytes(count: int) -> str:
    """`7.1KB` — the figure shown for content with no line count (an image)."""
    if count >= 1_048_576:
        return f"{count / 1_048_576:.1f}MB"
    if count >= 1024:
        return f"{count / 1024:.1f}KB"
    return f"{count}B"


def fmt_cost(amount: float, *, precision: int = 2) -> str:
    return f"${amount:.{precision}f}"


def fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.0f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{max(seconds, 0):.0f}s"


def fmt_when(moment: datetime, now: datetime) -> str:
    """`18m ago` / `2h ago` / `yesterday` / `3 days ago` — the sessions
    table's WHEN column."""
    delta = now - moment
    minutes = delta.total_seconds() / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes:.0f}m ago"
    if minutes < 60 * 24 and moment.date() == now.date():
        return f"{minutes / 60:.0f}h ago"
    days = (now.date() - moment.date()).days
    if days <= 1:
        return "yesterday"
    return f"{days} days ago"


def home_path(path: str) -> str:
    """`/Users/x/quantized/luca` → `~/quantized/luca`."""
    from pathlib import Path

    try:
        return "~/" + str(Path(path).resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


def short_model(model: str) -> str:
    """`anthropic/claude-sonnet-5` → `claude-sonnet-5` — the status bar shows
    the model, not the route."""
    return model.rsplit("/", 1)[-1]


def inline_paths(prefix: str, paths: list[str]) -> str:
    """The composer text the `@` picker commits: every path carries its own `@`
    (each one is a mention in its own right), comma-joined, surrounded by
    spaces, appended to whatever was typed before the `@` that opened the
    picker. `prefix` is right-trimmed so its trailing space never doubles up."""
    if not paths:
        return prefix
    return f"{prefix.rstrip()} {','.join('@' + path for path in paths)} "


# ── hint legends ──────────────────────────────────────────────────────────────
# One entry per focus context; the legend always reflects the focused context
# and is the app's only discoverability mechanism.

HINTS: dict[str, list[str]] = {
    "idle": ["enter send", "⇥ complete", "^p palette"],
    "running": ["enter queue", "^p palette", "esc interrupt"],
    "approval": ["↑↓ move", "enter confirm", "1–4 pick"],
    "approval-diff": ["↑↓ move", "enter confirm", "1–4 pick", "^d full diff"],
    "retry": ["↑↓ move", "enter confirm", "^r retry now"],
    "palette": ["↑↓ move", "enter run", "esc dismiss"],
    "picker": ["↑↓ move", "space toggle", "enter insert", "esc dismiss"],
    "menu": ["↑↓ move", "enter pick", "esc dismiss"],
    "sessions": ["↑↓ move", "enter resume", "f fork", "d delete", "esc back"],
    "settings": ["↑↓ move", "← → change", "enter edit", "esc save & close"],
    "cost": ["^k compact context", "esc back"],
    # The agent's question set. `N` is filled per question by
    # `question_hints`; these are the shapes.
    "questions-single": ["↑↓ move", "1–N pick", "enter confirm", "⇥ next"],
    "questions-multi": ["↑↓ move", "1–N toggle", "space toggle", "enter confirm"],
    "questions-last": ["↑↓ move", "1–N pick", "enter answer + submit"],
    "questions-custom": ["enter save", "esc back to options", "⇥ next"],
    "questions-confirm": ["enter submit", "esc back to questions"],
}


def approval_hints(option_count: int, *, has_diff: bool = False) -> list[str]:
    hints = ["↑↓ move", "enter confirm", f"1–{option_count} pick"]
    if has_diff:
        hints.append("^d full diff")
    return hints


def question_hints(
    option_count: int,
    *,
    mode: str = "single",
    last: bool = False,
    editing_custom: bool = False,
    confirming: bool = False,
) -> list[str]:
    """The legend for wherever the question set currently is.

    `last` and `enter`'s meaning come from the SAME predicate
    (`question_set_is_last`), so the legend can never disagree with the key. `N`
    is the live row count INCLUDING the custom and chat rows, built the way
    `approval_hints` builds `1–4 pick`."""
    if confirming:
        return list(HINTS["questions-confirm"])
    if editing_custom:
        # While the field has the keyboard, digits and `space` are LITERAL
        # TEXT — advertising them would be actively wrong. `esc` LEAVES the
        # field; it does not clear it, because the row holds a real editor and
        # deleting what you typed is the editor's job.
        return list(HINTS["questions-custom"])
    hints = [hint.replace("1–N", f"1–{option_count}") for hint in HINTS[f"questions-{mode}"]]
    if not last:
        return hints
    # `enter` has exactly two meanings and the legend always states which. On
    # the last open question the same keystroke answers it AND opens the
    # confirmation, so `⇥ next` has nowhere to go.
    submit = HINTS["questions-last"][-1]
    return [submit if hint == "enter confirm" else hint for hint in hints if hint != "⇥ next"]


def question_set_is_last(state) -> bool:
    """Would answering the ACTIVE question leave none open?

    THE ONE PREDICATE behind both `enter`'s two meanings and the legend that
    states which. While any other question is unanswered, `enter` confirms and
    advances; when this is the last one open, the same keystroke answers it and
    opens the confirmation — and the legend then reads `enter answer + submit`,
    not `enter confirm`."""
    return all(question.answered for index, question in enumerate(state.questions) if index != state.active)


def question_hints_for(state) -> list[str]:
    """`question_hints` derived from a whole `state.QuestionSetState`. Duck-typed
    rather than importing the view-model, so `format` stays dependency-free."""
    if state.phase == "confirming":
        return question_hints(0, confirming=True)
    question = state.questions[state.active]
    return question_hints(
        len(question.options),
        mode=question.mode,
        last=question_set_is_last(state),
        editing_custom=state.editing_custom,
    )
