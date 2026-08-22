"""End-to-end Pilot tests: `AgentApp` driven by a scripted `FauxProvider`.

Each test scripts the provider, runs the app headless (`run_test`), submits
a prompt, waits for the drive worker to settle, and asserts on transcript
block view-models plus the persisted session.
"""

import base64

from luca.agent.contrib.app.sessions import load_session
from luca.agent.contrib.tui import AgentApp, app as app_module, state as vm
from luca.agent.contrib.tui.blocks import (
    AssistantText,
    NoticeLine,
    ThinkingLine,
    ToolBlockView,
    UserTurn,
)
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.shells import ApprovalPromptView
from luca.agent.core.models import ConversationStatus, LLMConfig
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from tests.agent.scenarios import main_conversation

from .helpers import fresh_session, idle_again, submit, wait_until


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


# a 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def test_text_turn_renders_user_turn_and_assistant_text(tmp_path):
    session = fresh_session()
    app = make_app(session, scripted(faux_assistant_message([faux_text("Hello there!")])), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [turn.text for turn in app.query(UserTurn)] == ["hi"]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="Hello there!", streaming=False),
        ]
        assert app.runner.status is ConversationStatus.IDLE
        assert (tmp_path / f"{session.id}.json").exists()


async def test_streaming_thinking_renders_one_settled_line(tmp_path):
    app = make_app(
        fresh_session(),
        scripted(faux_assistant_message([faux_thinking("pondering the greeting"), faux_text("Hey!")])),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        # the faux stream settles instantly, so the duration floors at 1.0
        assert [line.model for line in app.query(ThinkingLine)] == [vm.ThinkingBlock(duration=1.0)]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="Hey!", streaming=False),
        ]


async def test_non_streaming_renders_the_same_transcript(tmp_path):
    app = make_app(
        fresh_session(),
        scripted(faux_assistant_message([faux_thinking("pondering the greeting"), faux_text("Hey!")])),
        tmp_path,
        streaming=False,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        # no ReasoningStart on the non-streaming tier — the line settles bare
        assert [line.model for line in app.query(ThinkingLine)] == [vm.ThinkingBlock()]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="Hey!", streaming=False),
        ]


async def test_blank_text_renders_no_assistant_block(tmp_path):
    app = make_app(
        fresh_session(),
        scripted(faux_assistant_message([faux_thinking("deciding"), faux_text(" ")])),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert len(app.query(ThinkingLine)) == 1
        assert list(app.query(AssistantText)) == []


async def test_blank_text_renders_no_assistant_block_non_streaming(tmp_path):
    app = make_app(
        fresh_session(),
        scripted(faux_assistant_message([faux_text("   ")])),
        tmp_path,
        streaming=False,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert list(app.query(AssistantText)) == []


async def test_tool_call_lifecycle_renders_a_tool_block_with_its_result(tmp_path):
    app = make_app(
        fresh_session(),
        scripted(
            faux_assistant_message(
                [faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It is 42.")]),
        ),
        tmp_path,
        mode="yolo",
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="multiply",
                arg="a=6, b=7",
                status="ok",
                result=vm.ToolResult(summary="42.0"),
            ),
        ]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="It is 42.", streaming=False),
        ]


async def test_resume_replay_rebuilds_the_transcript(tmp_path):
    session = fresh_session()
    app = make_app(
        session,
        scripted(
            faux_assistant_message(
                [faux_thinking("resumable pondering"), faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("First answer.")]),
        ),
        tmp_path,
        mode="yolo",
    )
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "first question")
        await wait_until(pilot, lambda: idle_again(app))
        live_shape = [type(widget).__name__ for widget in app.transcript.children]
        live_tools = [view.model for view in app.query(ToolBlockView)]
        live_texts = [text.model for text in app.query(AssistantText)]

    resumed = make_app(load_session(session.id, tmp_path), scripted(), tmp_path, mode="yolo")
    async with resumed.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        assert [type(widget).__name__ for widget in resumed.transcript.children] == live_shape
        assert [turn.text for turn in resumed.query(UserTurn)] == ["first question"]
        assert [view.model for view in resumed.query(ToolBlockView)] == live_tools
        assert [text.model for text in resumed.query(AssistantText)] == live_texts
        assert resumed.runner.status is ConversationStatus.IDLE


async def test_an_unresolvable_tool_call_renders_an_error_tool_block(tmp_path):
    # no registry owns `divide`, so the call is terminal at birth: no approval
    # gate, no dispatch — the block is mounted by ToolCallReceived and
    # finished by ToolExecuted without ever going running
    app = make_app(
        fresh_session(),
        scripted(
            faux_assistant_message(
                [faux_tool_call("divide", {"a": 6, "b": 7}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("I have no divide tool.")]),
        ),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 divided by 7?")
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="divide",
                arg="a=6, b=7",
                status="error",
                output=vm.ToolOutput(lines=["Unknown tool: 'divide'."], summary="not found"),
            ),
        ]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="I have no divide tool.", streaming=False),
        ]


async def test_a_drive_failure_shows_the_model_error_block_and_the_retry_prompt(tmp_path):
    session = fresh_session()
    app = make_app(
        session,
        scripted(faux_assistant_message([], error=faux_error("boom"))),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))

        # the error block carries the exception's own class/message noise, so
        # only the load-bearing fields are pinned here
        [error_block] = [view.model for view in app.query(ToolBlockView)]
        assert (error_block.tool, error_block.arg, error_block.status) == ("model", "fake-model", "error")
        assert "boom" in "\n".join(error_block.output.lines)
        # faux has no recommended alternate, so no "Switch to …" option
        [prompt] = app.query(ApprovalPromptView)
        assert prompt.model == vm.ApprovalState(
            question="Keep going, or stop here?",
            options=[
                vm.ApprovalOption(label="Retry now"),
                vm.ApprovalOption(label="Cancel turn", key_hint="esc"),
            ],
            selected=0,
        )

        await pilot.press("2")  # Cancel turn
        await wait_until(pilot, lambda: idle_again(app))

        assert app.runner.status is ConversationStatus.IDLE
        assert app.query_one(PromptInput).text == ""
        assert (tmp_path / f"{session.id}.json").exists()


async def test_pasted_image_is_attached_and_sent_with_the_next_message(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "read_clipboard_image", lambda: PNG)
    session = fresh_session()
    app = make_app(
        session,
        scripted(faux_assistant_message([faux_text("A tiny square.")])),
        tmp_path,
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("ctrl+v")
        await pilot.pause()
        await submit(pilot, "what is this?")
        await wait_until(pilot, lambda: idle_again(app))

        assert [turn.text for turn in app.query(UserTurn)] == [
            "[image: pasted-1.png]\nwhat is this?",
        ]
        first = session.entries[main_conversation(session).nodes[0]]
        assert [part.type for part in first.parts] == ["image", "text"]
        assert app._pending_images == []


async def test_escape_clears_a_pending_attachment(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "read_clipboard_image", lambda: PNG)
    app = make_app(fresh_session(), scripted(), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("ctrl+v")
        await pilot.pause()
        assert len(app._pending_images) == 1

        await pilot.press("escape")
        await pilot.pause()

        assert app._pending_images == []


async def test_pasting_an_image_is_refused_on_a_text_only_model(tmp_path, monkeypatch):
    # the paste path builds the part itself, so the handler chain's capability
    # check never sees it; deepseek rejects the whole request, not just the image
    monkeypatch.setattr(app_module, "read_clipboard_image", lambda: PNG)
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="deepseek-chat", provider="deepseek")
    app = make_app(session, scripted(), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("ctrl+v")
        await pilot.pause()

        assert (app._pending_images, [(line.text, line.error) for line in app.query(NoticeLine)]) == (
            [],
            [("deepseek-chat does not accept images", True)],
        )


async def test_ctrl_q_saves_and_quits(tmp_path):
    session = fresh_session()
    app = make_app(session, scripted(), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()

    assert app.is_running is False


async def test_ctrl_c_clears_the_prompt(tmp_path):
    app = make_app(fresh_session(), scripted(), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.load_text("a half-written question")
        prompt.focus()
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        # the composer empties; ctrl+c is not a quit key
        assert (prompt.text, app.is_running) == ("", True)


async def test_ctrl_d_saves_and_quits_like_ctrl_q(tmp_path):
    session = fresh_session()
    app = make_app(session, scripted(), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.load_text("a half-written question")
        prompt.focus()
        await pilot.pause()

        await pilot.press("ctrl+d")
        await pilot.pause()

    assert app.is_running is False
    assert (tmp_path / f"{session.id}.json").exists()
    assert (tmp_path / f"{session.id}.json").exists()
