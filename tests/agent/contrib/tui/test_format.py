"""Pure text/format helpers in `format.py`.

Span tests assert the whole `(plain, spans)` shape of the Rich Text — the
segment boundaries and their styles together. `TOKENS` mirrors the shipped
`luca-dark` palette; the values only need to be distinguishable colors.
"""

from datetime import datetime
from pathlib import Path

from rich.style import Style
from rich.text import Span

from luca.agent.contrib.tui.format import (
    HINTS,
    approval_hints,
    code_spans,
    fmt_cost,
    fmt_duration,
    fmt_tokens,
    fmt_when,
    home_path,
    short_model,
    spans,
)
from luca.agent.contrib.tui.theme import Tokens

TOKENS = Tokens(
    foreground="#E8E8E8",
    muted="#999999",
    faint="#7E7E7E",
    accent="#7AA8A6",
    success="#7EA87E",
    error="#C47C7C",
    rule="#2D2D2D",
    hairline="#3A3A3A",
    selection_bg="#212625",
    diff_add_bg="#1F241F",
    diff_del_bg="#2A1E1E",
    surface="#171717",
)


# ── spans() — theme-token tags ────────────────────────────────────────────────


def test_spans_colors_a_tagged_run_and_resets_on_close():
    text = spans("run [accent]tests[/] now", TOKENS)
    assert (text.plain, text.spans) == (
        "run tests now",
        [Span(4, 9, Style(color="#7AA8A6"))],
    )


def test_spans_replace_rather_than_nest():
    text = spans("a [muted]b [error]c", TOKENS)
    assert (text.plain, text.spans) == (
        "a b c",
        [
            Span(2, 4, Style(color="#999999")),
            Span(4, 5, Style(color="#C47C7C")),
        ],
    )


def test_spans_pass_unknown_brackets_through_literally():
    text = spans("[bold]x[/] y", TOKENS)
    assert (text.plain, text.spans) == ("[bold]x y", [])


def test_spans_carry_the_default_style():
    text = spans("hello", TOKENS, default=Style(italic=True))
    assert (text.plain, text.style) == ("hello", Style(italic=True))


# ── code_spans() — model-output-safe backticks ────────────────────────────────


def test_code_spans_color_backticked_runs_in_accent():
    text = code_spans("run `uv sync` twice", TOKENS)
    assert (text.plain, text.spans) == (
        "run uv sync twice",
        [Span(4, 11, Style(color="#7AA8A6"))],
    )


def test_code_spans_never_read_theme_token_tags():
    text = code_spans("[accent]x[/]", TOKENS)
    assert (text.plain, text.spans) == ("[accent]x[/]", [])


# ── humanized figures ─────────────────────────────────────────────────────────


def test_fmt_tokens():
    assert [fmt_tokens(0), fmt_tokens(999), fmt_tokens(1000), fmt_tokens(12437)] == ["0", "999", "1.0k", "12.4k"]


def test_fmt_cost():
    assert [fmt_cost(1.5), fmt_cost(0.214), fmt_cost(0.0042, precision=3)] == ["$1.50", "$0.21", "$0.004"]


def test_fmt_duration():
    assert [fmt_duration(0), fmt_duration(-5), fmt_duration(45), fmt_duration(1800), fmt_duration(7200)] == [
        "0s",
        "0s",
        "45s",
        "30m",
        "2h",
    ]


def test_fmt_when():
    now = datetime(2026, 8, 3, 12, 0, 0)
    assert [
        fmt_when(datetime(2026, 8, 3, 11, 59, 40), now),
        fmt_when(datetime(2026, 8, 3, 11, 18, 0), now),
        fmt_when(datetime(2026, 8, 3, 7, 0, 0), now),
        fmt_when(datetime(2026, 8, 2, 23, 30, 0), now),
        fmt_when(datetime(2026, 7, 31, 12, 0, 0), now),
    ] == ["just now", "42m ago", "5h ago", "yesterday", "3 days ago"]


def test_home_path_contracts_the_home_prefix():
    assert home_path(str(Path.home() / "code" / "luca")) == "~/code/luca"


def test_home_path_leaves_foreign_paths_alone():
    assert home_path("/opt/tools") == "/opt/tools"


def test_short_model_drops_the_route():
    assert [short_model("anthropic/claude-sonnet-5"), short_model("gpt-5.4")] == ["claude-sonnet-5", "gpt-5.4"]


# ── hint legends ──────────────────────────────────────────────────────────────


def test_hints_covers_every_focus_context():
    assert list(HINTS) == [
        "idle",
        "running",
        "approval",
        "approval-diff",
        "retry",
        "palette",
        "picker",
        "menu",
        "sessions",
        "settings",
        "cost",
    ]


def test_approval_hints_scale_with_the_option_count():
    assert approval_hints(3) == ["↑↓ move", "enter confirm", "1–3 pick"]


def test_approval_hints_add_the_diff_toggle():
    assert approval_hints(4, has_diff=True) == ["↑↓ move", "enter confirm", "1–4 pick", "^d full diff"]
