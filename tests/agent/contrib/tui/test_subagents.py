"""Subagents in the transcript: one `TaskBlockView` per spawned child, its
blocks nested inside `task.body`, and a resume that reproduces it.

Placement assertions read the transcript's direct children and each task
body's direct children rather than a flat `app.query(...)` — nesting is the
behavior under test, and a flat query would pass just as happily with the
subagent's output spliced into the main column.
"""

from luca.agent.contrib.app.approvals import build_approval_prompts
from luca.agent.contrib.app.sessions import load_session
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.contrib.tui import AgentApp, state as vm
from luca.agent.contrib.tui.blocks import (
    AssistantText,
    NoticeLine,
    TaskBlockView,
    ToolBlockView,
)
from luca.agent.contrib.tui.shells import ApprovalPromptView
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


def make_app(session, provider, tmp_path, **kwargs) -> AgentApp:
    return AgentApp(
        session,
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        **kwargs,
    )


def spawn(description: str, prompt: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


def tasks(app) -> list[TaskBlockView]:
    return list(app.query(TaskBlockView))


async def test_a_subagents_work_renders_inside_its_task_block(tmp_path):
    """The spawn tool and the result tool are plumbing — the task block is
    what the user sees, and the subagent's answer sits inside its body."""
    app = make_app(
        fresh_session(SUBAGENTS),
        scripted(
            faux_assistant_message(
                [
                    faux_text("Delegating."),
                    spawn("read alpha.txt", "Read alpha.txt and say what it is.", call_id="c1", task_id="t1"),
                ]
            ),
            faux_assistant_message([faux_text("alpha.txt is a shopping list.")]),
            faux_assistant_message([faux_text("It is a shopping list.")]),
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is alpha.txt")
        await wait_until(pilot, lambda: idle_again(app))

        assert [type(widget).__name__ for widget in app.transcript.children] == [
            "UserTurn",
            "AssistantText",
            "TaskBlockView",
            "AssistantText",
        ]
        [task] = tasks(app)
        assert task.status_model == vm.TaskBlock(description="read alpha.txt", status="done")
        assert [text.model for text in task.body.query(AssistantText)] == [
            vm.TextBlock(text="alpha.txt is a shopping list.", streaming=False),
        ]


async def test_parallel_subagents_get_one_task_block_each(tmp_path):
    """Two children advance at once and their events interleave on one
    stream. Each answer must land in its own task body — spliced output would
    show up as one body holding both."""
    app = make_app(
        fresh_session(SUBAGENTS),
        scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ]
            ),
            # Which child pops which response is up to the scheduler; that
            # both bodies hold exactly one answer each is not.
            faux_assistant_message([faux_text("one answer")]),
            faux_assistant_message([faux_text("another answer")]),
            faux_assistant_message([faux_text("Both are read.")]),
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "read both")
        await wait_until(pilot, lambda: idle_again(app))

        assert [
            (task.status_model.description, task.status_model.status, len(task.body.children)) for task in tasks(app)
        ] == [
            ("read alpha", "done", 1),
            ("read beta", "done", 1),
        ]
        answers = sorted(text.model.text for task in tasks(app) for text in task.body.query(AssistantText))
        assert answers == ["another answer", "one answer"]


async def test_a_subagents_own_tool_calls_render_inside_its_task_body(tmp_path):
    app = make_app(
        fresh_session(SUBAGENTS),
        scripted(
            faux_assistant_message(
                [spawn("do the sum", "Add 2 and 3.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3}, id="c_add")]),
            faux_assistant_message([faux_text("2 + 3 = 5.")]),
            faux_assistant_message([faux_text("My helper made it 5.")]),
        ),
        tmp_path,
        mode="yolo",  # the gate has its own test below; this one is about placement
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "add 2 and 3 with a helper")
        await wait_until(pilot, lambda: idle_again(app))

        assert [type(widget).__name__ for widget in app.transcript.children] == [
            "UserTurn",
            "TaskBlockView",
            "AssistantText",
        ]
        [task] = tasks(app)
        assert [type(widget).__name__ for widget in task.body.children] == ["ToolBlockView", "AssistantText"]
        assert [view.model for view in task.body.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="add",
                arg="a=2, b=3",
                status="ok",
                result=vm.ToolResult(summary="5.0"),
            ),
        ]
        assert [text.model for text in task.body.query(AssistantText)] == [
            vm.TextBlock(text="2 + 3 = 5.", streaming=False),
        ]


async def test_resume_replays_the_task_exactly_as_it_rendered(tmp_path):
    """A reloaded session must show the delegated work, not a gap where it
    happened — so the replayed tree is compared against the live one."""
    session = fresh_session(SUBAGENTS)
    app = make_app(
        session,
        scripted(
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
        tmp_path,
        mode="yolo",
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is alpha.txt")
        await wait_until(pilot, lambda: idle_again(app))
        live_shape = [type(widget).__name__ for widget in app.transcript.children]
        [live_task] = tasks(app)
        live_status = live_task.status_model
        live_body = [type(widget).__name__ for widget in live_task.body.children]
        live_tools = [view.model for view in live_task.body.query(ToolBlockView)]

    resumed = make_app(load_session(session.id, tmp_path), scripted(), tmp_path, mode="yolo")
    async with resumed.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        assert [type(widget).__name__ for widget in resumed.transcript.children] == live_shape
        [task] = tasks(resumed)
        assert task.status_model == live_status
        assert [type(widget).__name__ for widget in task.body.children] == live_body
        assert [view.model for view in task.body.query(ToolBlockView)] == live_tools


async def test_a_cancelled_subagents_task_settles_failed(tmp_path):
    """The wind-down resolves a cancelled child WITHOUT running the result
    tool, so a task driven by that tool's event alone would hang on "running"
    — on precisely the path where the user needs to see it stop."""
    app = make_app(
        fresh_session(SUBAGENTS),
        scripted(
            faux_assistant_message(
                [spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_hang()]),
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "delegate then cancel")
        await wait_until(pilot, lambda: bool(tasks(app)))
        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))

        assert [task.status_model for task in tasks(app)] == [
            vm.TaskBlock(description="read alpha", status="failed"),
        ]
        session = app.runner.session
        links = [entry for entry in session.entries.values() if isinstance(entry, ChildConversation)]
        assert [entry.execution_result is not None for entry in links] == [True]


async def test_a_queued_subagent_says_waiting_until_the_pool_admits_it(tmp_path):
    """Under `subagents_max_workers` a spawned subagent may be queued: its
    task opens "waiting" and flips to "running" only when `SubagentStarted`
    arrives — one field, driven entirely by the events."""
    capped = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1, subagents_max_workers=1)
    app = make_app(
        fresh_session(capped),
        scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ],
            ),
            faux_assistant_message([faux_hang()]),  # the first child holds the only slot
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "read both, one at a time")
        await wait_until(pilot, lambda: len(tasks(app)) == 2)
        await wait_until(pilot, lambda: tasks(app)[0].status_model.status == "running")

        # the slot-holder runs; its sibling is queued and says so
        assert [task.status_model for task in tasks(app)] == [
            vm.TaskBlock(description="read alpha", status="running"),
            vm.TaskBlock(description="read beta", status="waiting"),
        ]

        await pilot.press("escape")  # unwind the hung child; both settle
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_refused_spawn_renders_an_error_notice_not_nothing(tmp_path):
    """A spawn past `subagents_max_per_turn` is born REFUSED and creates no
    child — so the plumbing rule (spawns render as their task block) would
    make it invisible. It gets an error notice instead, live and on replay."""
    budgeted = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1, subagents_max_per_turn=1)
    app = make_app(
        fresh_session(budgeted),
        scripted(
            faux_assistant_message(
                [
                    spawn("read alpha", "Read alpha.txt.", call_id="c1", task_id="t1"),
                    spawn("read beta", "Read beta.txt.", call_id="c2", task_id="t2"),
                ],
            ),
            faux_assistant_message([faux_text("alpha read")]),
            faux_assistant_message([faux_text("Only alpha; the budget refused beta.")]),
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "read both")
        await wait_until(pilot, lambda: idle_again(app))

        [notice] = app.query(NoticeLine)
        assert "Spawn limit reached (1/1 subagents this turn)" in notice.text
        assert notice.error is True
        assert [task.status_model.description for task in tasks(app)] == ["read alpha"]

    # and the replay reproduces it
    reloaded = make_app(load_session(app.runner.session.id, tmp_path), FauxProvider(), tmp_path)
    async with reloaded.run_test(size=(105, 35)) as pilot:
        await wait_until(pilot, lambda: bool(tasks(reloaded)))

        [notice] = reloaded.query(NoticeLine)
        assert "Spawn limit reached" in notice.text
        assert notice.error is True


async def test_a_subagents_approval_prompt_names_its_task(tmp_path):
    """Two subagents can gate at the same moment, so the prompt has to say
    which one is asking in terms the user can act on — the task label, not
    the conversation key."""
    app = make_app(
        fresh_session(SUBAGENTS),
        scripted(
            faux_assistant_message(
                [spawn("do the sum", "Add 2 and 3.", call_id="c1", task_id="t1")],
            ),
            faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3}, id="c_add")]),
            faux_assistant_message([faux_text("2 + 3 = 5.")]),
            faux_assistant_message([faux_text("My helper made it 5.")]),
        ),
        tmp_path,
        mode="ask",
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "add 2 and 3 with a helper")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))

        [prompt] = app.query(ApprovalPromptView)
        assert prompt.model == vm.ApprovalState(
            question="[faint]task · do the sum —[/] Run add?",
            options=[
                vm.ApprovalOption(label="Approve once"),
                vm.ApprovalOption(label="Deny — tell Luca what to do instead"),
                vm.ApprovalOption(label="Cancel turn", key_hint="esc"),
            ],
            selected=0,
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
    assert prompt.question == "[faint]task · c_child —[/] Run add?"


async def test_subagents_off_renders_no_task(tmp_path):
    """`--no-subagents` withholds the tool, so nothing about the transcript
    changes for a session that never spawns."""
    app = make_app(
        fresh_session(),
        scripted(faux_assistant_message([faux_text("No helpers needed.")])),
        tmp_path,
        subagents=False,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [type(widget).__name__ for widget in app.transcript.children] == ["UserTurn", "AssistantText"]
        assert tasks(app) == []
        outcomes = [entry.outcome for entry in app.runner.session.entries.values() if hasattr(entry, "outcome")]
        assert outcomes == [TurnOutcome.COMPLETED]
