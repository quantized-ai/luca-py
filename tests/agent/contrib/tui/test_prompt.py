"""Pilot tests for `PromptInput` — the multiline prompt.

Enter submits; Alt+Enter, Shift+Enter, Ctrl+J and a trailing backslash
insert a newline instead; a bracketed paste lands verbatim. One parser
test pins the kitty CSI-u encoding of Shift+Enter (`ESC [13;2u`) to the
`shift+enter` key the widget binds — that sequence is what a terminal
(or tmux with extended-keys) actually sends.
"""

from textual import events
from textual._xterm_parser import XTermParser

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import UserCell
from luca.agent.contrib.tui.prompt import PromptInput
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, idle_again, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def test_kitty_shift_enter_sequence_parses_to_the_bound_key():
    parser = XTermParser()
    tokens = [*parser.feed("\x1b[13;2u"), *parser.feed("")]
    assert [token.key for token in tokens] == ["shift+enter"]


async def test_alt_enter_inserts_a_newline_instead_of_submitting(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"one", "alt+enter", *"two")

        assert app.query_one("#prompt", PromptInput).text == "one\ntwo"
        assert list(app.query(UserCell)) == []


async def test_shift_enter_inserts_a_newline_instead_of_submitting(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"one", "shift+enter", *"two")

        assert app.query_one("#prompt", PromptInput).text == "one\ntwo"
        assert list(app.query(UserCell)) == []


async def test_ctrl_j_inserts_a_newline_instead_of_submitting(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"one", "ctrl+j", *"two")

        assert app.query_one("#prompt", PromptInput).text == "one\ntwo"
        assert list(app.query(UserCell)) == []


async def test_backslash_enter_swaps_the_backslash_for_a_newline(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"one\\", "enter")

        assert app.query_one("#prompt", PromptInput).text == "one\n"
        assert list(app.query(UserCell)) == []


async def test_multiline_paste_lands_verbatim_without_submitting(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        # posted to the app, exactly as the terminal driver delivers a
        # bracketed paste; the app forwards it to the focused prompt
        app.post_message(events.Paste("one\ntwo\nthree"))
        await pilot.pause()

        assert app.query_one("#prompt", PromptInput).text == "one\ntwo\nthree"
        assert list(app.query(UserCell)) == []


async def test_enter_submits_the_multiline_text_as_one_message(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("Got it.")])),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"one", "alt+enter", *"two", "enter")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(UserCell)] == ["one\ntwo"]
        assert app.query_one("#prompt", PromptInput).text == ""


async def test_slash_prefix_suggests_the_first_matching_command(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"/he")

        assert app.query_one("#prompt", PromptInput).suggestion == "lp"


async def test_right_accepts_the_suggested_command(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"/he", "right")

        assert app.query_one("#prompt", PromptInput).text == "/help"


async def test_prompt_grows_with_the_content(tmp_path):
    app = AgentApp(fresh_session(), provider=FauxProvider(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        assert app.query_one("#prompt", PromptInput).region.height == 3

        await pilot.press(*"one", "alt+enter", *"two", "alt+enter", *"three")
        await pilot.pause()

        assert app.query_one("#prompt", PromptInput).region.height == 5
