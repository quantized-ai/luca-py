"""The sticky todo panel, driven through the live app.

The panel is docked between the transcript and the composer, so it is asserted
on directly rather than looked for among transcript blocks — and the todo
tools themselves must leave NO transcript row at all, which is the whole
reason the panel exists.

Everything runs against a scripted `FauxProvider`: the model calls
`update_todos`, the memory plugin numbers the list, the app renders it.
"""

from luca.agent.contrib.app.sessions import load_session
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import ListBlockView, ToolBlockView
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def make_app(provider, tmp_path, session=None) -> AgentApp:
    return AgentApp(
        session if session is not None else fresh_session(),
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


def todos(*pairs: tuple[str, str]) -> dict:
    return {"todos": [{"content": content, "status": status} for content, status in pairs]}


def write_todos(*pairs: tuple[str, str], call_id: str = "tc1"):
    return faux_tool_call("update_todos", todos(*pairs), id=call_id)


def panel(app) -> ListBlockView:
    return app.plan_view


def rows(app) -> list[tuple[str, str, bool]]:
    return [(row.glyph, row.text, row.strike) for row in panel(app).model.rows]


async def test_a_todo_write_docks_the_panel_and_leaves_no_transcript_row(tmp_path):
    provider = scripted(
        faux_assistant_message([write_todos(("do the laundry", "pending"), ("take out the dog", "pending"))]),
        faux_assistant_message([faux_text("Created two tasks.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "make me a todo list")
        await wait_until(pilot, lambda: idle_again(app))

        assert panel(app).display is True
        assert panel(app).model.label == "2 tasks (2 open)"
        assert rows(app) == [
            ("pending", "[faint]#1 -[/] do the laundry", False),
            ("pending", "[faint]#2 -[/] take out the dog", False),
        ]
        # The write rendered as the panel and nothing else — no `▸ update_todos`.
        assert [view.model.tool for view in app.transcript.query(ToolBlockView)] == []


async def test_a_read_todo_leaves_no_transcript_row_either(tmp_path):
    # The read that a model so often fires before a write used to be the only
    # visible artifact of the pair, which made it look like the read did the work.
    provider = scripted(
        faux_assistant_message([faux_tool_call("read_todo", {}, id="tc1")]),
        faux_assistant_message([faux_text("Nothing on the list.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what are your todos")
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model.tool for view in app.transcript.query(ToolBlockView)] == []
        assert panel(app).display is False


async def test_completing_an_item_strikes_it_in_place(tmp_path):
    provider = scripted(
        faux_assistant_message([write_todos(("do the laundry", "pending"), ("take out the dog", "pending"))]),
        faux_assistant_message([faux_text("Created two tasks.")]),
        faux_assistant_message(
            [write_todos(("do the laundry", "completed"), ("take out the dog", "pending"), call_id="tc2")]
        ),
        faux_assistant_message([faux_text("Marked complete.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "make me a todo list")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "complete the laundry")
        await wait_until(pilot, lambda: idle_again(app))

        # Two items fit the panel, so nothing moves — the row is simply struck.
        assert panel(app).model.label == "2 tasks (1 done, 1 open)"
        assert rows(app) == [
            ("done", "[faint]#1 -[/] do the laundry", True),
            ("pending", "[faint]#2 -[/] take out the dog", False),
        ]


async def test_a_finished_list_says_done_and_survives_until_the_user_speaks(tmp_path):
    provider = scripted(
        faux_assistant_message([write_todos(("do the laundry", "pending"))]),
        faux_assistant_message([faux_text("Created one task.")]),
        faux_assistant_message([write_todos(("do the laundry", "completed"), call_id="tc2")]),
        faux_assistant_message([faux_text("Done.")]),
        faux_assistant_message([faux_text("Anything else?")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "make me a todo list")
        await wait_until(pilot, lambda: idle_again(app))
        await submit(pilot, "complete it")
        await wait_until(pilot, lambda: idle_again(app))

        # Still docked, and saying so, for the rest of the turn that finished it.
        assert panel(app).model.label == "Done (1 task completed)"

        await submit(pilot, "thanks")

        assert panel(app).display is False
        await wait_until(pilot, lambda: idle_again(app))


async def test_numbering_carries_on_after_a_dismissed_list(tmp_path):
    provider = scripted(
        faux_assistant_message([write_todos(("do the laundry", "completed"), ("walk the dog", "completed"))]),
        faux_assistant_message([faux_text("Both done.")]),
        faux_assistant_message([write_todos(("buy milk", "pending"), call_id="tc2")]),
        faux_assistant_message([faux_text("Added one task.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "two finished chores please")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "now add buy milk")
        await wait_until(pilot, lambda: idle_again(app))

        # #3, not a second #1 — the list was dismissed, the counter was not.
        assert rows(app) == [("pending", "[faint]#3 -[/] buy milk", False)]


async def test_the_list_is_written_into_the_session_the_app_saves(tmp_path):
    # THE PERSISTENCE CONTRACT, end to end. `wiring` hands the memory plugin a
    # dict that lives on the session, so the tools writing the list write into
    # the file the app was already saving — no second store, no sidecar.
    provider = scripted(
        faux_assistant_message([write_todos(("port the reader", "in_progress"))]),
        faux_assistant_message([faux_text("Working through it.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "plan the migration")
        await wait_until(pilot, lambda: idle_again(app))
        session_id = app.runner.session.id

    reloaded = load_session(session_id, tmp_path)

    assert reloaded.extras["todos"] == {
        reloaded.main_conversation_id: {
            "todos": [{"id": 1, "content": "port the reader", "status": "in_progress"}],
            "next_id": 2,
        }
    }


async def test_a_resumed_session_docks_the_plan_it_was_left_with(tmp_path):
    provider = scripted(
        faux_assistant_message([write_todos(("port the reader", "in_progress"), ("port the writer", "pending"))]),
        faux_assistant_message([faux_text("Working through it.")]),
    )
    app = make_app(provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "plan the migration")
        await wait_until(pilot, lambda: idle_again(app))
        stored = app.runner.session.model_copy(deep=True)

    resumed = make_app(scripted(), tmp_path, session=stored)
    async with resumed.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        assert panel(resumed).display is True
        assert rows(resumed) == [
            ("active", "[faint]#1 -[/] port the reader", False),
            ("pending", "[faint]#2 -[/] port the writer", False),
        ]
