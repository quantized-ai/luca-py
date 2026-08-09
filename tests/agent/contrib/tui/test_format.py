"""Pure text/format helpers in `format.py`.

Span tests assert the whole `(plain, spans)` shape of the Rich Text — the
segment boundaries and their styles together. `TOKENS` mirrors the shipped
`luca-dark` palette; the values only need to be distinguishable colors.
"""

from datetime import datetime
from pathlib import Path

from rich.style import Style
from rich.text import Span

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.format import (
    GLYPH_ARROW,
    GLYPH_BOX,
    GLYPH_BOX_TICKED,
    HINTS,
    approval_hints,
    code_spans,
    fmt_cost,
    fmt_duration,
    fmt_tokens,
    fmt_when,
    home_path,
    inline_paths,
    question_hints,
    question_hints_for,
    question_set_is_last,
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
        "questions-single",
        "questions-multi",
        "questions-last",
        "questions-custom",
        "questions-confirm",
    ]


def test_approval_hints_scale_with_the_option_count():
    assert approval_hints(3) == ["↑↓ move", "enter confirm", "1–3 pick"]


def test_approval_hints_add_the_diff_toggle():
    assert approval_hints(4, has_diff=True) == ["↑↓ move", "enter confirm", "1–4 pick", "^d full diff"]


# ── the `@` picker's composer insertion ───────────────────────────────────────


def test_one_picked_path_lands_surrounded_by_spaces():
    assert inline_paths("", ["luca/agent/core/runner.py"]) == " @luca/agent/core/runner.py "


def test_several_picked_paths_are_comma_joined_each_with_its_own_sigil():
    assert inline_paths("", ["a.py", "b.py", "c.py"]) == " @a.py,@b.py,@c.py "


def test_the_paths_append_to_what_was_typed_before_the_at_sign():
    assert inline_paths("explain", ["a.py"]) == "explain @a.py "


def test_a_trailing_space_on_the_prefix_does_not_double_up():
    # the composer holds "explain " — `@` was stripped when the picker opened
    assert inline_paths("explain ", ["a.py"]) == "explain @a.py "


def test_no_paths_gives_the_prefix_back_untouched():
    assert inline_paths("explain ", []) == "explain "


# ── the agent's question set ──────────────────────────────────────────────────

# Two questions and the four rows each of them ends up with. Only `answered`
# and `mode` ever differ between them, which is all these tests read.
SINGLE_QUESTION = vm.Question(
    tab="backfill",
    title="How should existing sessions be backfilled?",
    options=[
        vm.QuestionOption(label="On first launch, in-process"),
        vm.QuestionOption(label="Lazily, when a session resumes"),
        vm.QuestionOption(label="Custom answer:", kind="custom"),
        vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
    ],
)
MULTI_QUESTION = vm.Question(
    tab="rollout",
    title="Which surfaces read sqlite first?",
    mode="multi",
    options=[
        vm.QuestionOption(label="the sessions screen"),
        vm.QuestionOption(label="the cost screen"),
        vm.QuestionOption(label="Custom answer:", kind="custom"),
        vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
    ],
)
ANSWERED_QUESTION = SINGLE_QUESTION.model_copy(update={"tab": "storage", "answered": True})


def test_the_question_glyphs_are_the_ones_the_design_names():
    # §4 of the design spec is normative about these three — the tick, the
    # empty box and the answer separator — not `x`, `[ ]` or `->`.
    assert [GLYPH_BOX, GLYPH_BOX_TICKED, GLYPH_ARROW] == ["☐", "☑", "→"]


def test_question_hints_scale_with_the_row_count():
    assert question_hints(6) == ["↑↓ move", "1–6 pick", "enter confirm", "⇥ next"]


def test_a_multi_select_question_advertises_space_instead_of_the_next_tab():
    assert question_hints(6, mode="multi") == ["↑↓ move", "1–6 toggle", "space toggle", "enter confirm"]


def test_the_last_open_question_says_enter_answers_and_submits():
    # `⇥ next` is dropped rather than kept and inert: there is nowhere left to
    # go, and the same keystroke now opens the confirmation.
    assert question_hints(6, last=True) == ["↑↓ move", "1–6 pick", "enter answer + submit"]


def test_the_last_open_multi_select_question_keeps_both_toggles():
    assert question_hints(5, mode="multi", last=True) == [
        "↑↓ move",
        "1–5 toggle",
        "space toggle",
        "enter answer + submit",
    ]


def test_editing_a_custom_answer_advertises_only_the_keys_that_are_not_text():
    # While the field has the keyboard, digits and `space` are literal text —
    # advertising them would be actively wrong.
    assert question_hints(6, editing_custom=True) == ["enter save", "esc back to options", "⇥ next"]


def test_the_confirmation_has_exactly_two_keys():
    assert question_hints(0, confirming=True) == ["enter submit", "esc back to questions"]


def test_the_confirmations_legend_wins_over_every_other_branch():
    assert question_hints(6, mode="multi", last=True, editing_custom=True, confirming=True) == [
        "enter submit",
        "esc back to questions",
    ]


def test_the_active_question_is_the_last_one_when_every_other_is_answered():
    state = vm.QuestionSetState(questions=[ANSWERED_QUESTION, SINGLE_QUESTION], active=1)

    assert question_set_is_last(state) is True


def test_it_is_not_the_last_one_while_another_question_is_still_open():
    state = vm.QuestionSetState(questions=[SINGLE_QUESTION, SINGLE_QUESTION], active=1)

    assert question_set_is_last(state) is False


def test_a_one_question_set_is_always_on_its_last_question():
    assert question_set_is_last(vm.QuestionSetState(questions=[SINGLE_QUESTION])) is True


def test_question_hints_for_reads_the_active_question_out_of_the_state():
    state = vm.QuestionSetState(questions=[MULTI_QUESTION, SINGLE_QUESTION], active=0)

    assert question_hints_for(state) == ["↑↓ move", "1–4 toggle", "space toggle", "enter confirm"]


def test_question_hints_for_the_last_open_question_of_a_state():
    state = vm.QuestionSetState(questions=[ANSWERED_QUESTION, SINGLE_QUESTION], active=1)

    assert question_hints_for(state) == ["↑↓ move", "1–4 pick", "enter answer + submit"]


def test_question_hints_for_a_custom_field_that_has_the_keyboard():
    state = vm.QuestionSetState(questions=[SINGLE_QUESTION], editing_custom=True)

    assert question_hints_for(state) == ["enter save", "esc back to options", "⇥ next"]


def test_question_hints_for_the_confirmation_never_reads_a_question():
    state = vm.QuestionSetState(questions=[ANSWERED_QUESTION], phase="confirming")

    assert question_hints_for(state) == ["enter submit", "esc back to questions"]
