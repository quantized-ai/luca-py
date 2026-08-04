"""The `@` context picker, driven through the Pilot.

`@` in the composer (at a word boundary) swaps the dock for the picker,
which lists the workspace's files. `space` checks rows, `enter` commits:
the picked paths go back into the composer as `@path` mentions, comma-joined
and surrounded by spaces, appended to whatever had been typed before the `@`.
`enter` with nothing checked takes the highlighted row.

The workspace is a plain tmp directory, so `git ls-files` fails there and the
listing falls back to the bounded walk — a fixed, sorted set of files.
"""

from luca.agent.contrib.tui import AgentApp, state as vm
from luca.agent.contrib.tui.blocks import ToolBlockView, UserTurn
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.prompt_files import ReadLimits
from luca.agent.contrib.tui.sessions import load_session
from luca.agent.contrib.tui.shells import OverlayListView
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, idle_again, submit, wait_until


def make_app(tmp_path, *responses, session=None, read_limits=None) -> AgentApp:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (workspace / name).write_text(f"# {name}\n")
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return AgentApp(
        session or fresh_session(),
        provider=provider,
        workspace=workspace,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        read_limits=read_limits,
    )


async def open_picker(pilot, typed: str = "") -> None:
    prompt = pilot.app.query_one(PromptInput)
    prompt.load_text(typed)
    prompt.move_cursor(prompt.document.end)  # load_text parks the cursor at 0,0
    prompt.focus()
    await pilot.pause()
    await pilot.press("@")
    await wait_until(pilot, lambda: bool(pilot.app.query(OverlayListView)))


def composer_text(app) -> str:
    return app.query_one(PromptInput).text


async def test_at_opens_the_picker_listing_the_workspace_files(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot)

        [overlay] = app.query(OverlayListView)
        assert [row.primary for row in overlay.model.rows] == ["alpha.py", "beta.py", "gamma.py"]


async def test_enter_with_nothing_checked_inlines_the_highlighted_file(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot, "explain ")

        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        assert composer_text(app) == "explain @alpha.py "


async def test_enter_on_a_moved_highlight_inlines_that_file(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot)
        await pilot.press("down")
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        assert composer_text(app) == " @beta.py "


async def test_several_checked_files_are_comma_joined(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot, "compare ")

        await pilot.press("space")  # alpha.py
        await pilot.press("down")
        await pilot.press("space")  # beta.py
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        assert composer_text(app) == "compare @alpha.py,@beta.py "


async def test_checking_a_row_leaves_the_highlight_where_it_was(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot)
        await pilot.press("down")
        await pilot.press("space")  # check beta.py — the highlight must not jump home
        await pilot.pause()

        [overlay] = app.query(OverlayListView)
        assert overlay.model.selected == 1


async def test_typing_filters_the_list_and_enter_takes_the_match(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot)
        await pilot.press("g", "a", "m")
        await pilot.pause()
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        assert composer_text(app) == " @gamma.py "


async def test_escape_gives_back_the_half_typed_message_untouched(tmp_path):
    app = make_app(tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot, "explain ")

        await pilot.press("escape")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        assert composer_text(app) == "explain "


async def test_the_picked_path_goes_out_expanded_with_the_next_message(tmp_path):
    app = make_app(tmp_path, faux_assistant_message([faux_text("Reading it now.")]))

    async with app.run_test(size=(105, 35)) as pilot:
        await open_picker(pilot, "explain ")
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(PromptInput)))

        app.query_one(PromptInput).focus()
        await pilot.press("enter")
        await wait_until(pilot, lambda: idle_again(app))

        alpha = (tmp_path / "ws" / "alpha.py").resolve()
        user_message = next(iter(app.runner.session.entries.values()))
        # the typed text is untouched; the file rides along as a second part
        assert [part.text for part in user_message.parts] == [
            "explain @alpha.py",
            (
                f'<agent-prompt-file path="{alpha}" status="ok" lines="1" estimated_tokens="2" bytes="11">\n'
                "# alpha.py\n\n"
                "</agent-prompt-file>"
            ),
        ]
        assert user_message.parts[1].metadata["mention"]["path"] == str(alpha)


# ── @-mention expansion on submit ─────────────────────────────────────────────


async def test_a_mention_renders_a_read_row_and_keeps_the_file_out_of_the_user_block(tmp_path):
    app = make_app(tmp_path, faux_assistant_message([faux_text("Ok")]))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, 'Read @alpha.py and reply "Ok"')
        await wait_until(pilot, lambda: idle_again(app))

        # the user's own words only — the 1-line file is NOT dumped in here
        assert [turn.text for turn in app.query(UserTurn)] == ['Read @alpha.py and reply "Ok"']
        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="read",
                arg=str((tmp_path / "ws" / "alpha.py").resolve()),
                status="ok",
                result=vm.ToolResult(summary="1 lines"),
            ),
        ]


async def test_prose_that_looks_like_a_mention_renders_no_read_row(tmp_path):
    app = make_app(tmp_path, faux_assistant_message([faux_text("Ok")]))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "use @property not @staticmethod")
        await wait_until(pilot, lambda: idle_again(app))

        assert list(app.query(ToolBlockView)) == []
        assert [turn.text for turn in app.query(UserTurn)] == ["use @property not @staticmethod"]


async def test_a_file_over_the_cap_renders_the_declined_row(tmp_path):
    app = make_app(tmp_path, faux_assistant_message([faux_text("Ok")]), read_limits=ReadLimits(hard_limit=1))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "summarize @alpha.py")
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="read",
                arg=str((tmp_path / "ws" / "alpha.py").resolve()),
                status="error",
                result=vm.ToolResult(summary="[error]×[/] file too long, defaulting to agent tool calling"),
            ),
        ]


async def test_the_read_row_comes_back_on_reload(tmp_path):
    app = make_app(tmp_path, faux_assistant_message([faux_text("Ok")]))
    alpha = (tmp_path / "ws" / "alpha.py").resolve()

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "Read @alpha.py")
        await wait_until(pilot, lambda: idle_again(app))
        session_id = app.runner.session.id

    replayed = make_app(tmp_path, session=load_session(session_id, tmp_path))
    async with replayed.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        # derived from the stored metadata — the file is never re-read
        assert [turn.text for turn in replayed.query(UserTurn)] == ["Read @alpha.py"]
        assert [view.model for view in replayed.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="read",
                arg=str(alpha),
                status="ok",
                result=vm.ToolResult(summary="1 lines"),
            ),
        ]
