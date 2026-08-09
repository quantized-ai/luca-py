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

↑/↓ at the edges of the document walk the messages already sent, read
straight off the session — the history tests boot on a session that
already carries some, exactly as a resume does.
"""

import time

from textual import events
from textual._xterm_parser import XTermParser

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import UserTurn
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.core.models import (
    AgentSession,
    TextContent,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def sent(*texts: str) -> AgentSession:
    """A session already carrying these user messages, oldest first — one
    SETTLED turn each, so the app boots idle rather than resuming a drive."""
    session = fresh_session()
    nodes = session.conversations[session.main_conversation_id].nodes
    for index, text in enumerate(texts, start=1):
        at = index * 10
        for entry in (
            TurnStart(id=f"ts{index}", created_at=at),
            UserMessage(id=f"u{index}", created_at=at + 1, parts=[TextContent(text=text)]),
            TurnFinish(id=f"tf{index}", created_at=at + 2, outcome=TurnOutcome.COMPLETED),
        ):
            session.entries[entry.id] = entry
            nodes.append(entry.id)
    return session


def agent_app(tmp_path, provider=None, session=None) -> AgentApp:
    return AgentApp(
        session or fresh_session(),
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


# ── history ───────────────────────────────────────────────────────────────────


async def test_up_recalls_the_last_message_sent(tmp_path):
    app = agent_app(tmp_path, session=sent("first", "second"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("up")

        assert app.query_one(PromptInput).text == "second"


async def test_up_again_walks_further_back_and_stops_at_the_oldest(tmp_path):
    app = agent_app(tmp_path, session=sent("first", "second"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("up", "up", "up")

        assert app.query_one(PromptInput).text == "first"


async def test_down_walks_back_to_the_draft_it_stashed(tmp_path):
    app = agent_app(tmp_path, session=sent("first"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"half typed", "up", "down")

        assert app.query_one(PromptInput).text == "half typed"


async def test_down_past_the_newest_empties_the_composer_again(tmp_path):
    app = agent_app(tmp_path, session=sent("first"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("up", "down")

        assert app.query_one(PromptInput).text == ""


async def test_down_with_nothing_recalled_leaves_the_draft_alone(tmp_path):
    app = agent_app(tmp_path, session=sent("first"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"half typed", "down")

        assert app.query_one(PromptInput).text == "half typed"


async def test_up_inside_a_multiline_draft_moves_the_cursor(tmp_path):
    app = agent_app(tmp_path, session=sent("first"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "alt+enter", *"two", "up")

        prompt = app.query_one(PromptInput)
        assert (prompt.text, prompt.cursor_location) == ("one\ntwo", (0, 3))


async def test_up_from_the_top_of_a_multiline_draft_recalls(tmp_path):
    app = agent_app(tmp_path, session=sent("first"))

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press(*"one", "alt+enter", *"two", "up", "up")

        assert app.query_one(PromptInput).text == "first"


async def test_a_submit_ends_the_walk_and_the_message_joins_the_history(tmp_path):
    # Recall "second", edit it, send it. The walk is back at the newest, which
    # is now the edited message — a stale position would answer "second".
    app = agent_app(
        tmp_path,
        provider=scripted(faux_assistant_message([faux_text("Got it.")])),
        session=sent("first", "second"),
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("up", "!", "enter")
        await wait_until(pilot, lambda: app.runner.idle() and not app._driving)
        await pilot.press("up")

        assert app.query_one(PromptInput).text == "second!"
