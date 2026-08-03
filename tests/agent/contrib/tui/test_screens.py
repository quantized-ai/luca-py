"""`PickerScreen` — labels, and the filter's index remapping.

The filter is not a display concern: the option list reports the index of the
row you picked, so once rows are hidden that index means something different.
`_visible` is the mapping, and these tests exist to keep it honest.
"""

import pytest
from textual.app import App
from textual.widgets import Input, OptionList

from luca.agent.contrib.tui.screens import PickerScreen

MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4-8",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "moonshotai/kimi-k2.7-code",
]


class Host(App):
    """A bare app to push the modal onto."""


async def open_picker(pilot, **kwargs) -> list:
    """Push the modal and return the list its dismissal lands in."""
    picked: list = []
    pilot.app.push_screen(PickerScreen("Select a model", list(MODELS), **kwargs), picked.append)
    await pilot.pause()
    return picked


def rows(app) -> list[str]:
    options = app.screen.query_one("#picker-options", OptionList)
    return [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]


async def type_filter(pilot, text: str) -> None:
    pilot.app.screen.query_one("#picker-filter", Input).value = text
    await pilot.pause()


# ── without a filter ─────────────────────────────────────────────────────────


async def test_an_unfiltered_picker_has_no_filter_input():
    async with Host().run_test() as pilot:
        await open_picker(pilot)

        with pytest.raises(Exception, match="No nodes match"):
            pilot.app.screen.query_one("#picker-filter", Input)


async def test_the_current_option_is_marked():
    async with Host().run_test() as pilot:
        await open_picker(pilot, current="openai/gpt-5.4")

        assert "openai/gpt-5.4 (current)" in rows(pilot.app)


async def test_labels_are_shown_while_the_value_is_returned():
    async with Host().run_test() as pilot:
        picked = await open_picker(pilot, labels=[f"row {i}" for i in range(len(MODELS))])
        shown = rows(pilot.app)  # read before Enter pops the screen
        await pilot.press("down", "enter")

        assert shown[:2] == ["row 0", "row 1"]
        assert picked == [MODELS[1]]


# ── filtering ────────────────────────────────────────────────────────────────


async def test_typing_narrows_the_rows():
    async with Host().run_test() as pilot:
        await open_picker(pilot, filterable=True)
        await type_filter(pilot, "gpt")

        assert rows(pilot.app) == ["openai/gpt-5.4", "openai/gpt-5.4-mini"]


async def test_the_filter_ignores_case():
    async with Host().run_test() as pilot:
        await open_picker(pilot, filterable=True)
        await type_filter(pilot, "CLAUDE")

        assert rows(pilot.app) == ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8"]


async def test_selecting_after_a_filter_returns_the_visible_row_not_the_original_index():
    # THE bug this feature invites: "gpt" leaves two rows, and the second of
    # them is index 3 in the full list. Reading the index against `_options`
    # would hand back `anthropic/claude-opus-4-8`.
    async with Host().run_test() as pilot:
        picked = await open_picker(pilot, filterable=True)
        await type_filter(pilot, "gpt")
        await pilot.press("down", "enter")

        assert picked == ["openai/gpt-5.4-mini"]


async def test_clearing_the_filter_restores_every_row():
    async with Host().run_test() as pilot:
        picked = await open_picker(pilot, filterable=True)
        await type_filter(pilot, "gpt")
        await type_filter(pilot, "")
        shown = rows(pilot.app)  # read before Enter pops the screen
        await pilot.press("enter")

        assert shown == MODELS
        assert picked == [MODELS[0]]


async def test_a_filter_matching_nothing_empties_the_list_and_selects_nothing():
    async with Host().run_test() as pilot:
        await open_picker(pilot, filterable=True)
        await type_filter(pilot, "no-such-model")
        await pilot.press("enter")

        assert rows(pilot.app) == []
        assert isinstance(pilot.app.screen, PickerScreen)  # still open


async def test_escape_still_cancels_while_filtering():
    async with Host().run_test() as pilot:
        picked = await open_picker(pilot, filterable=True)
        await type_filter(pilot, "gpt")
        await pilot.press("escape")

        assert picked == [None]


async def test_end_reaches_the_last_row_even_though_the_filter_holds_focus():
    # the model step puts "back to providers" last, so this key has to work
    async with Host().run_test() as pilot:
        picked = await open_picker(pilot, filterable=True)
        await pilot.press("end", "enter")

        assert picked == [MODELS[-1]]


async def test_the_filter_matches_the_label_not_the_value():
    async with Host().run_test() as pilot:
        picked = await open_picker(
            pilot,
            filterable=True,
            labels=[f"{model}  (fast)" if "mini" in model else model for model in MODELS],
        )
        await type_filter(pilot, "fast")
        await pilot.press("enter")

        assert picked == ["openai/gpt-5.4-mini"]
