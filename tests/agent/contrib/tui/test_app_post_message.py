"""Mid-turn input: the composer stays enabled while the agent works, a submit
posts into the open turn and renders immediately — subagents running included,
where the post wakes the parked parent and the turn answers it — and a slash
command mid-run is refused with a notice, keeping the draft. Input is never
silently discarded.

One deliberate refusal: a submit while the main conversation is BLOCKED at a
gate. The framework would carry it past the gate (0008); the shipped TUI opts
out — the answer the user owes is the approval prompt — so the submit is
refused with a notice and the draft kept. A SUBAGENT's gate with siblings
still working leaves the conversation BUSY, and steering posts keep working."""

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.contrib.tui import AgentApp, state as vm
from luca.agent.contrib.tui.blocks import NoticeLine, UserTurn
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.shells import ApprovalPromptView
from luca.agent.core.models import RuntimeConfig, UserMessage
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


def make_app(session, provider, tmp_path) -> AgentApp:
    return AgentApp(
        session,
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


def spawn(description: str, prompt: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


async def test_typing_mid_run_posts_into_the_open_turn(tmp_path):
    provider = scripted(faux_assistant_message([faux_hang()]))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)
        prompt = app.query_one(PromptInput)
        assert not prompt.disabled  # the composer stays enabled while busy

        await submit(pilot, "and steer this way")
        await pilot.pause()

        # the post landed INSIDE the open turn, rendered immediately, and the
        # input was cleared for the next message
        session = app.runner.session
        nodes = session.conversations[session.main_conversation_id].nodes
        posted = [entry.parts[0].text for n in nodes if isinstance(entry := session.entries[n], UserMessage)]
        assert posted == ["hang please", "and steer this way"]
        assert [turn.text for turn in app.query(UserTurn)] == ["hang please", "and steer this way"]
        assert prompt.text == ""

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_post_while_subagents_run_wakes_the_parent_and_is_answered(tmp_path):
    provider = scripted(
        faux_assistant_message(
            [
                faux_text("Delegating."),
                spawn("read alpha.txt", "Read alpha.txt.", call_id="c1", task_id="t1"),
            ]
        ),
        faux_assistant_message([faux_hang()]),  # the child hangs mid-work
        # the accepted post wakes the parked parent for one round
        faux_assistant_message([faux_text("Noted — still waiting on the task.")]),
    )
    app = make_app(fresh_session(SUBAGENTS), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "go")
        await wait_until(pilot, lambda: len(app._tasks) == 1)

        await submit(pilot, "hurry up")
        await pilot.pause()

        # accepted: the message landed in the parent's open turn, rendered
        # immediately, and the input cleared for the next message
        posted = [e.parts[0].text for e in app.runner.session.entries.values() if isinstance(e, UserMessage)]
        assert "hurry up" in posted
        assert [turn.text for turn in app.query(UserTurn)] == ["go", "hurry up"]
        assert app.query_one(PromptInput).text == ""

        # …and the woken parent's next call SAW it, while the child kept going
        await wait_until(pilot, lambda: len(provider.requests) >= 3)
        assert provider.requests[-1].messages[-1].content[0].text == "hurry up"

        await pilot.press("escape")  # unwind the hung child; everything settles
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_submit_while_blocked_is_refused_with_a_notice(tmp_path):
    # THE 0008 OPT-OUT. The framework would answer a post past the gate with
    # the placeholder on the wire; the shipped TUI refuses instead — the
    # answer the user owes is the approval prompt. The composer is normally
    # not even mounted while blocked (the prompt replaces it), but the swap is
    # worker-driven, so there are real windows where it is; recreate one by
    # mounting the composer over the still-gated conversation.
    provider = scripted(
        faux_assistant_message(
            [faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc1")],
            finish_reason="tool_use",
        ),
    )
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))
        assert app.runner.blocked()
        await app.show_composer(vm.ComposerState())
        await pilot.pause()
        session = app.runner.session
        nodes_before = list(session.conversations[session.main_conversation_id].nodes)

        await submit(pilot, "please hurry")
        await pilot.pause()

        # nothing was posted — the unchanged path is what proves no model
        # round could have run — the refusal is stated, and the draft kept
        assert session.conversations[session.main_conversation_id].nodes == nodes_before
        assert [(notice.text, notice.error) for notice in app.query(NoticeLine)] == [
            ("answer the approval prompt first", True),
        ]
        assert app.query_one(PromptInput).text == "please hurry"

        # settle: the prompt view is gone, so answer the worker's pending ask
        # directly with "Cancel turn"
        app._approval_future.set_result(2)
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_submit_while_a_subagent_is_gated_and_siblings_work_still_posts(tmp_path):
    # The guard must not over-reach: a SUBAGENT's gate with a sibling still
    # working leaves the main conversation BUSY, and mid-orchestration
    # steering posts stay supported — that is exactly what `blocked()` (and
    # not `pending_approvals()`) buys.
    provider = scripted(
        faux_assistant_message(
            [
                faux_text("Delegating both."),
                spawn("do the sum", "Add 2 and 3.", call_id="c1", task_id="t1"),
                spawn("keep researching", "Research things.", call_id="c2", task_id="t2"),
            ]
        ),
        faux_assistant_message([faux_tool_call("add", {"a": 2, "b": 3}, id="c_add")]),  # t1 — gates
        faux_assistant_message([faux_hang()]),  # t2 — keeps the parent BUSY
        # the accepted steering post wakes the parked parent for one round
        faux_assistant_message([faux_text("Noted — the sum still needs your approval.")]),
    )
    app = make_app(fresh_session(SUBAGENTS), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "go")
        await wait_until(pilot, lambda: bool(app.runner.pending_approvals()))
        assert not app.runner.blocked()  # a sibling still works: BUSY

        await submit(pilot, "hurry up")
        await pilot.pause()

        # accepted: the post landed, rendered, and the input cleared
        posted = [e.parts[0].text for e in app.runner.session.entries.values() if isinstance(e, UserMessage)]
        assert "hurry up" in posted
        assert [turn.text for turn in app.query(UserTurn)] == ["go", "hurry up"]
        assert app.query_one(PromptInput).text == ""

        # …and the woken parent's next call saw it
        await wait_until(pilot, lambda: len(provider.requests) >= 4)
        assert provider.requests[-1].messages[-1].content[0].text == "hurry up"

        await pilot.press("escape")  # unwind the hung sibling and the gate
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_command_submitted_mid_run_is_refused_and_the_draft_kept(tmp_path):
    provider = scripted(faux_assistant_message([faux_hang()]))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)

        await submit(pilot, "/model other:model")
        await pilot.pause()

        # commands stay idle-only: nothing dispatched, the refusal is stated,
        # and the draft is preserved
        assert app.runner.session.session_config.llm_config.model == "fake-model"
        assert [(notice.text, notice.error) for notice in app.query(NoticeLine)] == [
            ("commands are available when the agent is idle", True),
        ]
        assert app.query_one(PromptInput).text == "/model other:model"

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))
