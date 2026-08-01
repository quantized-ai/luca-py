"""Subagents in the transcript: one panel per child, and a resume that
reproduces it.

Every assertion here is on the whole transcript TREE rather than on a flat
cell list — nesting is the behavior under test, so a flat `app.query(...)`
would pass just as happily with the subagent's output spliced into the main
column, which is exactly the bug the panel exists to prevent.
"""

from textual.containers import VerticalScroll
from textual.widgets import Label

from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.approvals import build_approval_prompts
from luca.agent.contrib.tui.cells import SubagentPanel
from luca.agent.contrib.tui.screens import ApprovalScreen
from luca.agent.contrib.tui.sessions import load_session
from luca.agent.core.models import (
    ChildConversation,
    RuntimeConfig,
    ToolCall,
    ToolExecution,
    TurnOutcome,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_hang,
    faux_text,
    faux_tool_call,
)

from .helpers import fresh_session, idle_again, submit, wait_until

SUBAGENTS = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1)


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def spawn(description: str, prompt: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


def transcript(app) -> VerticalScroll:
    """The transcript, found on the BASE screen. `app.query` is scoped to the
    top of the screen stack, so a test that reads the transcript while the
    approval modal is up would otherwise find nothing."""
    return app.screen_stack[0].query_one("#transcript", VerticalScroll)


def tree(app, node=None) -> list:
    """The transcript as `(cell type, border title, border subtitle, text,
    children)` tuples — the whole shape in one comparable value."""
    node = node if node is not None else transcript(app)
    return [
        (
            type(child).__name__,
            child.border_title,
            child.border_subtitle,
            getattr(child, "text", None),
            tree(app, child),
        )
        for child in node.children
    ]


def panels(app) -> list[SubagentPanel]:
    return list(transcript(app).query(SubagentPanel))


async def test_a_subagents_work_renders_inside_its_own_panel(tmp_path):
    """The spawn tool and the result tool are plumbing — the panel is what the
    user sees, and the subagent's answer sits inside it."""
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [
                    faux_text("Delegating."),
                    spawn("read alpha.txt", "Read alpha.txt and say what it is.", call_id="c1", task_id="t1"),
                ]
            ),
            faux_assistant_message([faux_text("alpha.txt is a shopping list.")]),
            faux_assistant_message([faux_text("It is a shopping list.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is alpha.txt")
        await wait_until(pilot, lambda: idle_again(app))

        assert tree(app) == [
            ("UserCell", "you", None, "what is alpha.txt", []),
            ("AssistantCell", "assistant", None, "Delegating.", []),
            (
                "SubagentPanel",
                "subagent · read alpha.txt",
                "done",
                None,
                [
                    ("Static", None, None, None, []),  # the task the subagent was given
                    ("AssistantCell", "assistant", None, "alpha.txt is a shopping list.", []),
                ],
            ),
            ("AssistantCell", "assistant", None, "It is a shopping list.", []),
        ]


async def test_parallel_subagents_get_one_panel_each(tmp_path):
    """Two children advance at once and their events interleave on one stream.
    Each answer must land in its own panel — spliced output would show up as
    one panel holding both."""
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ]
            ),
            # Which child pops which response is up to the scheduler; that both
            # panels hold exactly one answer each is not.
            faux_assistant_message([faux_text("one answer")]),
            faux_assistant_message([faux_text("another answer")]),
            faux_assistant_message([faux_text("Both are read.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "read both")
        await wait_until(pilot, lambda: idle_again(app))

        assert [(panel.description, panel.status, len(panel.children)) for panel in panels(app)] == [
            ("read alpha", "done", 2),  # the task line plus one answer
            ("read beta", "done", 2),
        ]
        answers = sorted(cell.text for panel in panels(app) for cell in panel.children if hasattr(cell, "text"))
        assert answers == ["another answer", "one answer"]


async def test_a_subagents_own_tool_calls_render_inside_its_panel(tmp_path):
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [spawn("do the sum", "Add 2 and 3.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3}, id="c_add")]),
            faux_assistant_message([faux_text("2 + 3 = 5.")]),
            faux_assistant_message([faux_text("My helper made it 5.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",  # the gate has its own test below; this one is about placement
    )

    async with app.run_test() as pilot:
        await submit(pilot, "add 2 and 3 with a helper")
        await wait_until(pilot, lambda: idle_again(app))

        assert tree(app) == [
            ("UserCell", "you", None, "add 2 and 3 with a helper", []),
            (
                "SubagentPanel",
                "subagent · do the sum",
                "done",
                None,
                [
                    ("Static", None, None, None, []),
                    ("ToolCallCell", "tool · add", "done", "add(a=2, b=3)\n→ 5.0", []),
                    ("AssistantCell", "assistant", None, "2 + 3 = 5.", []),
                ],
            ),
            ("AssistantCell", "assistant", None, "My helper made it 5.", []),
        ]


async def test_resume_replays_the_panel_exactly_as_it_rendered(tmp_path):
    """A reloaded session must show the delegated work, not a gap where it
    happened — so the replayed tree is compared against the live one."""
    session = fresh_session(SUBAGENTS)
    app = AgentApp(
        session,
        provider=scripted(
            faux_assistant_message(
                [
                    faux_text("Delegating."),
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                ]
            ),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 1}, id="c_add")]),
            faux_assistant_message([faux_text("alpha.txt is a shopping list.")]),
            faux_assistant_message([faux_text("It is a shopping list.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is alpha.txt")
        await wait_until(pilot, lambda: idle_again(app))
        live = tree(app)

    resumed = AgentApp(
        load_session(session.id, tmp_path),
        provider=scripted(),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",
    )
    async with resumed.run_test() as pilot:
        await pilot.pause()

        assert tree(resumed) == live


async def test_a_cancelled_subagents_panel_stops_saying_running(tmp_path):
    """The wind-down resolves a cancelled child WITHOUT running the result
    tool, so a panel driven by that tool's event alone would hang on
    "running…" — on precisely the path where the user needs to see it stop."""
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_hang()]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "delegate then cancel")
        await wait_until(pilot, lambda: bool(panels(app)))
        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))

        assert [(panel.description, panel.status, panel.border_subtitle) for panel in panels(app)] == [
            ("read alpha", "failed", "failed"),
        ]
        session = app.runner.session
        links = [entry for entry in session.entries.values() if isinstance(entry, ChildConversation)]
        assert [entry.execution_result is not None for entry in links] == [True]


async def test_a_queued_subagent_says_waiting_until_the_pool_admits_it(tmp_path):
    """Under `subagents_max_workers` a spawned subagent may be queued: its
    panel opens saying "waiting…" and flips to "running…" only when
    `SubagentStarted` arrives — one field, driven entirely by the events."""
    capped = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1, subagents_max_workers=1)
    app = AgentApp(
        fresh_session(capped),
        provider=scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ],
            ),
            faux_assistant_message([faux_hang()]),  # the first child holds the only slot
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "read both, one at a time")
        await wait_until(pilot, lambda: len(panels(app)) == 2)
        await wait_until(pilot, lambda: panels(app)[0].status == "running")

        # the slot-holder runs; its sibling is queued and says so
        assert [(panel.description, panel.border_subtitle) for panel in panels(app)] == [
            ("read alpha", "running…"),
            ("read beta", "waiting…"),
        ]

        await pilot.press("escape")  # unwind the hung child; both settle
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_refused_spawn_renders_a_notice_not_nothing(tmp_path):
    """A spawn past `subagents_max_per_turn` is born REFUSED and creates no
    child — so the plumbing rule (spawns render as their child's panel) would
    make it invisible. It gets a notice instead, live and on replay."""
    budgeted = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1, subagents_max_per_turn=1)
    app = AgentApp(
        fresh_session(budgeted),
        provider=scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ],
            ),
            faux_assistant_message([faux_text("alpha read")]),
            faux_assistant_message([faux_text("Only alpha; the budget refused beta.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "read both")
        await wait_until(pilot, lambda: idle_again(app))

        notices = [row for row in tree(app) if row[0] == "NoticeCell"]
        assert len(notices) == 1
        assert "Spawn limit reached (1/1 subagents this turn)" in notices[0][3]
        assert [panel.description for panel in panels(app)] == ["read alpha"]

    # and the replay reproduces it
    reloaded = AgentApp(
        load_session(app.runner.session.id, directory=tmp_path),
        provider=FauxProvider(),
        workspace=tmp_path,
        session_dir=tmp_path,
    )
    async with reloaded.run_test() as pilot:
        await wait_until(pilot, lambda: bool(panels(reloaded)))
        notices = [row for row in tree(reloaded) if row[0] == "NoticeCell"]
        assert len(notices) == 1
        assert "Spawn limit reached" in notices[0][3]


async def test_a_subagents_approval_modal_names_it_by_its_task(tmp_path):
    """Two subagents can gate at the same moment, so the modal has to say which
    one is asking in terms the user can act on — the task, not the key."""
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [spawn("do the sum", "Add 2 and 3.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3}, id="c_add")]),
            faux_assistant_message([faux_text("2 + 3 = 5.")]),
            faux_assistant_message([faux_text("My helper made it 5.")]),
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="ask",
    )

    async with app.run_test() as pilot:
        await submit(pilot, "add 2 and 3 with a helper")
        await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))

        prompt = app.screen.prompt
        [child_id] = [panel.conversation_id for panel in panels(app)]
        assert (prompt.tool_name, prompt.conversation_id, prompt.conversation_label) == (
            "add",
            child_id,
            "do the sum",
        )
        assert str(app.screen.query_one("#approval-title", Label).content) == (
            "Approval needed: add  [subagent · do the sum]"
        )

        await pilot.press("1")  # Approve once, so the run finishes cleanly
        await wait_until(pilot, lambda: idle_again(app))


def test_an_unnamed_subagent_prompt_falls_back_to_its_id():
    """`subagent_labels` is optional — attribution and naming are different
    jobs, and a caller that tracks no tasks still gets an attributed gate."""
    execution = ToolExecution(
        id="e1",
        parent_id=None,
        created_at=1,
        conversation_id="c_child",
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    )

    [prompt] = build_approval_prompts(
        execution,
        PermissionStrategy(),
        main_conversation_id="c_main",
    )

    assert (prompt.tool_name, prompt.conversation_id, prompt.conversation_label) == ("add", "c_child", None)


async def test_subagents_off_renders_no_panel(tmp_path):
    """`--no-subagents` withholds the tool, so nothing about the transcript
    changes for a session that never spawns."""
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("No helpers needed.")])),
        workspace=tmp_path,
        session_dir=tmp_path,
        subagents=False,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert tree(app) == [
            ("UserCell", "you", None, "hi", []),
            ("AssistantCell", "assistant", None, "No helpers needed.", []),
        ]
        assert panels(app) == []
        outcomes = [entry.outcome for entry in app.runner.session.entries.values() if hasattr(entry, "outcome")]
        assert outcomes == [TurnOutcome.COMPLETED]
