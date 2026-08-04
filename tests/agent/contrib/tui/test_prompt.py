"""Pilot tests for `PromptInput` — the multiline prompt.

Enter submits; Alt+Enter, Shift+Enter, Ctrl+J and a trailing backslash
insert a newline instead; a bracketed paste lands verbatim. One parser
test pins the kitty CSI-u encoding of Shift+Enter (`ESC [13;2u`) to the
`shift+enter` key the widget binds — that sequence is what a terminal
(or tmux with extended-keys) actually sends.

Typing `/` in an empty composer opens the palette overlay, so a slash
prefix can no longer be typed key by key: the suggestion tests load the
prefix the way the palette-insert flow does and type only the last
character.
"""

import time

from textual import events
from textual._xterm_parser import XTermParser

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import UserTurn
from luca.agent.contrib.tui.prompt import PromptInput
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def agent_app(tmp_path, provider=None) -> AgentApp:
    return AgentApp(
        fresh_session(),
        provider=provider or FauxProvider(),
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


async def wait_until(pilot, condition, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("condition not met within timeout")


def test_kitty_shift_enter_sequence_parses_to_the_bound_key():
    parser = XTermParser()
    tokens = [*parser.feed("\x1b[13;2u"), *parser.feed("")]
    assert [token.key for token in tokens] == ["shift+enter"]


async def test_alt_enter_inserts_a_newline_instead_of_submitting(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "alt+enter", *"two")

        assert app.query_one(PromptInput).text == "one\ntwo"
        assert list(app.query(UserTurn)) == []


async def test_shift_enter_inserts_a_newline_instead_of_submitting(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "shift+enter", *"two")

        assert app.query_one(PromptInput).text == "one\ntwo"
        assert list(app.query(UserTurn)) == []


async def test_ctrl_j_inserts_a_newline_instead_of_submitting(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "ctrl+j", *"two")

        assert app.query_one(PromptInput).text == "one\ntwo"
        assert list(app.query(UserTurn)) == []


async def test_backslash_enter_swaps_the_backslash_for_a_newline(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one\\", "enter")

        assert app.query_one(PromptInput).text == "one\n"
        assert list(app.query(UserTurn)) == []


async def test_multiline_paste_lands_verbatim_without_submitting(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        # posted to the app, exactly as the terminal driver delivers a
        # bracketed paste; Textual forwards it to the focused prompt
        app.post_message(events.Paste("one\ntwo\nthree"))
        await pilot.pause()

        assert app.query_one(PromptInput).text == "one\ntwo\nthree"
        assert list(app.query(UserTurn)) == []


async def test_enter_submits_the_multiline_text_as_one_message(tmp_path):
    app = agent_app(tmp_path, provider=scripted(faux_assistant_message([faux_text("Got it.")])))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "alt+enter", *"two", "enter")
        await wait_until(pilot, lambda: app.runner.idle() and not app._driving)

        assert [turn.text for turn in app.query(UserTurn)] == ["one\ntwo"]
        assert app.query_one(PromptInput).text == ""


async def test_slash_prefix_suggests_the_first_matching_command(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.load_text("/h")  # the palette-insert flow's shape
        prompt.move_cursor(prompt.document.end)
        prompt.focus()
        await pilot.pause()
        await pilot.press("e")

        assert app.query_one(PromptInput).suggestion == "lp"


async def test_right_accepts_the_suggested_command(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.load_text("/h")
        prompt.move_cursor(prompt.document.end)
        prompt.focus()
        await pilot.pause()
        await pilot.press("e", "right")

        assert app.query_one(PromptInput).text == "/help"


async def test_prompt_grows_with_the_content(tmp_path):
    app = agent_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        assert app.query_one(PromptInput).region.height == 1

        await pilot.press(*"one", "alt+enter", *"two", "alt+enter", *"three")
        await pilot.pause()

        assert app.query_one(PromptInput).region.height == 3
