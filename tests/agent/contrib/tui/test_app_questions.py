"""The agent's questions through the fourth dock, driven end to end.

`ask_user` is the first DEFERRED tool: its body answers "not yet", the runner
parks the open turn at `AWAITING_RESULT`, and the drive worker blocks on a
human. The question set replaces the composer exactly as the approval prompt
does, and nothing reaches the model until the user passes the confirmation —
so every test here is a real keyboard session against a scripted
`FauxProvider`: park, answer, submit, and the collapsed block left behind.

Two shapes recur. A test that finishes the set (`Chat about this`, or the
confirmation) lets the drive run on and settle; a test that stops with the
dock still open releases the parked worker with `cancel_group`, because a
worker blocked on a human outlives the screen and fails the teardown. The
release changes nothing the dock draws, so it sits between the action and the
assertions.
"""

import json

from luca.agent.contrib.tui import AgentApp, state as vm
from luca.agent.contrib.tui.blocks import ListBlockView, NoticeLine, ToolBlockView
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.sessions import load_session, session_path, tui_state_path
from luca.agent.contrib.tui.shells import (
    OverlayListView,
    QuestionConfirmView,
    QuestionOptionRow,
    QuestionSetView,
    SpanLine,
)
from luca.agent.core.models import (
    ConversationStatus,
    ExecutionAttemptOutcome,
    ExecutionStatus,
    ToolExecution,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types.content import TextBlock
from luca.client.types.messages import ToolMessage

from .helpers import fresh_session, idle_again, submit, wait_until

PROMPT = "migrate the event store to sqlite"

STORAGE_TITLE = "Storage: where should the sqlite file live?"
STORAGE_BODY = "Nine projects, 380MB on disk."
READERS_TITLE = "Readers: which surfaces read sqlite first?"

# What the model asks for: one radio question with a body, one checkbox
# question. The tool's `Question` carries no tab label, so the dock derives one
# from the title's first words — "Storage" and "Readers".
QUESTIONS = {
    "questions": [
        {
            "title": STORAGE_TITLE,
            "body": STORAGE_BODY,
            "options_type": "single_select",
            "options": ["~/.luca/projects/<project>/events.db", "Beside the jsonl files"],
        },
        {
            "title": READERS_TITLE,
            "options_type": "multiple_select",
            "options": ["the sessions screen", "resume / replay", "the cost screen"],
        },
    ]
}

# The same two questions as `QuestionsTool` stores them: `Args`-validated and
# dumped, so the omitted `body` is present as null.
STORED_QUESTIONS = [
    {
        "title": STORAGE_TITLE,
        "body": STORAGE_BODY,
        "options_type": "single_select",
        "options": ["~/.luca/projects/<project>/events.db", "Beside the jsonl files"],
    },
    {
        "title": READERS_TITLE,
        "body": None,
        "options_type": "multiple_select",
        "options": ["the sessions screen", "resume / replay", "the cost screen"],
    },
]

NOTE = "in-process is fine if it's scoped"

ANSWERED_PROSE = f"""User answered all your questions:

{STORAGE_TITLE}
Answer:
- ~/.luca/projects/<project>/events.db

{READERS_TITLE}
Answer:
- the sessions screen
- resume / replay

Additional note from the user: "{NOTE}\""""

DECLINED_PROSE = f"""User declined to answer some questions.
User wants to chat more about: "{READERS_TITLE}"
Respond to them in the conversation — do not ask this question again.

Other questions/answers recorded:

{STORAGE_TITLE}
Answer:
- ~/.luca/projects/<project>/events.db"""


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
        subagents=False,
        skills=False,
        instructions=False,
    )


def transcript_tool_blocks(app) -> list:
    """The tool blocks a RELOAD of this app's session would draw — through the
    same result resolver `_replay_history` uses, so the comparison is about the
    derivation and not about who supplies the projected text."""
    from luca.agent.contrib.tui.render import transcript_blocks

    blocks = transcript_blocks(app.runner.session, resolve_result=app._resolve_result)
    return [block for block in blocks if block.kind == "tool"]


def asks(reaction: str) -> list:
    """The model asks its questions, then reacts to the answers it is given."""
    return [
        faux_assistant_message(
            [faux_tool_call("ask_user", QUESTIONS, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text(reaction)]),
    ]


def storage(
    *,
    selected: int = 0,
    answer: int | None = None,
    answered: bool = False,
    custom: str | None = None,
) -> vm.Question:
    """The radio question as the dock builds it — the model's two options, then
    the custom-answer row, then Chat about this.

    `selected` is the CARET and `answer` is the PICK. They coincide the moment
    a question is answered and diverge as soon as the caret moves, which is
    why only `answer` ever reaches the model."""
    return vm.Question(
        tab="Storage",
        title=STORAGE_TITLE,
        body=[STORAGE_BODY],
        mode="single",
        options=[
            vm.QuestionOption(label="~/.luca/projects/<project>/events.db"),
            vm.QuestionOption(label="Beside the jsonl files"),
            vm.QuestionOption(label="Custom answer:", kind="custom", text=custom),
            vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
        ],
        selected=selected,
        answer=answer,
        answered=answered,
    )


def readers(
    *,
    selected: int = 0,
    answered: bool = False,
    ticks: tuple[int, ...] = (),
    custom: str | None = None,
) -> vm.Question:
    """The checkbox question as the dock builds it; `ticks` are the row indexes
    carrying a `☑`."""
    return vm.Question(
        tab="Readers",
        title=READERS_TITLE,
        body=[],
        mode="multi",
        options=[
            vm.QuestionOption(label="the sessions screen", checked=0 in ticks),
            vm.QuestionOption(label="resume / replay", checked=1 in ticks),
            vm.QuestionOption(label="the cost screen", checked=2 in ticks),
            vm.QuestionOption(label="Custom answer:", kind="custom", text=custom),
            vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
        ],
        selected=selected,
        answered=answered,
    )


# ── the park ──────────────────────────────────────────────────────────────────


async def test_a_deferring_ask_user_parks_the_drive_and_takes_the_dock(tmp_path):
    provider = scripted(*asks("Got it."))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        app.workers.cancel_group(app, "drive")  # release the worker blocked on the human
        await pilot.pause()

        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers()],
        )
        assert app.composer() is None  # the set REPLACES the composer
        assert app.runner.status is ConversationStatus.BLOCKED
        assert len(provider.requests) == 1  # the model was not called again
        # Everything the park is defined by; the ids and timestamps around it
        # belong to the runner, not to this test.
        [execution] = [e for e in app.runner.session.entries.values() if isinstance(e, ToolExecution)]
        assert (execution.tool_call_id, execution.status, [a.outcome for a in execution.attempts]) == (
            "tc1",
            ExecutionStatus.AWAITING_RESULT,
            [ExecutionAttemptOutcome.DEFERRED],
        )


# ── answering the whole set ───────────────────────────────────────────────────


async def test_answering_every_question_and_submitting_completes_the_same_call(tmp_path):
    provider = scripted(*asks("Got it — sqlite it is."))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("1")  # answers Storage and advances to Readers
        await pilot.press("1")  # ticks "the sessions screen"
        await pilot.press("down")
        await pilot.press("space")  # ticks "resume / replay"
        await pilot.press("enter")  # commits the ticks — the set is settled
        await wait_until(pilot, lambda: bool(app.query(QuestionConfirmView)))
        await submit(pilot, NOTE)  # the confirmation's optional field
        await wait_until(pilot, lambda: idle_again(app))

        # ONE tool call, TWO body invocations: it was re-dispatched, not resumed
        [execution] = [e for e in app.runner.session.entries.values() if isinstance(e, ToolExecution)]
        assert (execution.tool_call_id, execution.status, [a.outcome for a in execution.attempts]) == (
            "tc1",
            ExecutionStatus.COMPLETED,
            [ExecutionAttemptOutcome.DEFERRED, ExecutionAttemptOutcome.COMPLETED],
        )
        assert provider.requests[-1].messages[-1] == ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text=ANSWERED_PROSE)],
        )
        assert app.composer() is not None


async def test_the_submitted_set_collapses_into_a_tool_block_and_one_row_per_question(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        # answer both questions, then submit with the confirmation's extra note
        await pilot.press("1")
        await pilot.press("1")
        await pilot.press("down")
        await pilot.press("space")
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(app.query(QuestionConfirmView)))
        await submit(pilot, NOTE)
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.transcript.query(ToolBlockView)] == [
            vm.ToolBlock(tool="Ask the user", arg="2 questions", status="ok", note_right="2 answered"),
        ]
        assert [view.model for view in app.transcript.query(ListBlockView)] == [
            vm.ListBlock(
                rows=[
                    vm.ListRow(glyph="done", text="Storage", annotation="→  ~/.luca/projects/<project>/events.db"),
                    vm.ListRow(glyph="done", text="Readers", annotation="→  the sessions screen, resume / replay"),
                    vm.ListRow(glyph="none", text="", annotation=f"added · {NOTE}"),
                ],
                column=28,
            ),
        ]


# ── the answering rules ───────────────────────────────────────────────────────


async def test_answering_advances_to_the_next_unanswered_tab_wrapping(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        # answer the SECOND question first: the advance wraps back to the first
        await pilot.press("tab", "3", "enter")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers(selected=2, answered=True, ticks=(2,))],
            active=0,
        )


async def test_picking_an_option_clears_the_custom_answer_in_a_radio_question(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("3", *"vendored", "enter")  # a custom answer to Storage

        await pilot.press("shift+tab", "1")  # back to Storage, pick the first option

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(selected=0, answer=0, answered=True, custom=None), readers()],
            active=1,
        )


async def test_typing_a_custom_answer_replaces_the_pick_in_a_radio_question(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("1")  # pick the first option of Storage

        await pilot.press("shift+tab", "3", *"vendored", "enter")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # a radio's pick IS `selected`, so the custom row holding it is what
        # says the option no longer does
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(selected=2, answer=2, answered=True, custom="vendored"), readers()],
            active=1,
        )


async def test_ticks_and_a_custom_answer_compose_in_a_checkbox_question(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("tab", "1", "4", *"replay", "enter")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers(selected=3, answered=True, ticks=(0,), custom="replay")],
            active=0,
        )


async def test_an_empty_custom_answer_is_not_an_answer(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("3", "enter")  # commit the custom row with nothing in it

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # the caret sits on the custom row and nothing else moved: the question
        # is exactly as it was
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(selected=2), readers()],
        )


async def test_esc_while_editing_clears_the_field_and_leaves_the_question_unanswered(tmp_path):
    # `esc` posts `CustomChanged("")` and only THEN `Moved`, so the app sees
    # the clear first and a widget snapshot that still holds the typed text
    # second — the race `on_question_set_view_moved` merges rather than
    # assigns. Both halves are asserted because they can drift apart: the
    # state the payload is built from, and the panel the user is looking at.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("3", *"vendored", "escape")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # the state the payload is built from — submitting must not carry text
        # the user erased…
        assert app._questions_state == vm.QuestionSetState(
            questions=[storage(selected=2), readers()],
        )
        # …and the panel, which must agree with it
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(selected=2), readers()],
        )
        # `esc` never reaches the app's cancel binding: the set still holds the
        # dock and the turn is still parked
        assert app.runner.status is ConversationStatus.BLOCKED


async def test_tabbing_away_and_back_keeps_every_tick(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("tab", "2")  # tick "resume / replay" on Readers

        await pilot.press("tab", "tab")  # away to Storage and back

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers(selected=1, ticks=(1,))],
            active=1,
        )


async def test_tabbing_to_another_question_shows_that_question(tmp_path):
    # The panel must FOLLOW the tab. `QuestionSetView._tab()` moves
    # `model.active` and posts `Moved`; only the app's rebuild makes the
    # title, body and option list follow it. Without that rebuild the dock
    # keeps drawing the PREVIOUS question while every keystroke answers the
    # new one — which is why this asserts the mounted widgets and not the
    # model they were built from.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("tab")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        view = app.query_one(QuestionSetView)
        assert [line.text_value for line in view.query(SpanLine)] == [READERS_TITLE]
        assert [row.option.label for row in view.query(QuestionOptionRow)] == [
            "the sessions screen",
            "resume / replay",
            "the cost screen",
            "Custom answer:",
            "Chat about this",
        ]


async def test_enter_commits_the_ticks_and_never_toggles_the_caret_row(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("tab", "1")  # tick "the sessions screen", caret on it

        await pilot.press("enter")  # commits — the ticked row must stay ticked

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers(answered=True, ticks=(0,))],
            active=0,
        )


# ── the confirmation ──────────────────────────────────────────────────────────


async def test_settling_the_last_question_flips_the_dock_to_the_confirmation(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("1")  # Storage answered; Readers is the last one open

        await pilot.press("1", "enter")  # tick, then answer Readers

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # nothing has reached the agent: submission takes a SECOND press
        assert app.query_one(QuestionConfirmView).model == vm.QuestionSetState(
            questions=[storage(answer=0, answered=True), readers(answered=True, ticks=(0,))],
            active=1,
            phase="confirming",
        )
        assert list(app.query(QuestionSetView)) == []
        assert app.runner.status is ConversationStatus.BLOCKED


async def test_esc_on_the_confirmation_goes_back_with_every_answer_intact(tmp_path):
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("1", "1", "enter")
        await wait_until(pilot, lambda: bool(app.query(QuestionConfirmView)))

        await pilot.press("escape")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # back on the tab that was last active, with both answers kept
        assert app.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(answer=0, answered=True), readers(answered=True, ticks=(0,))],
            active=1,
        )


# ── the way out ───────────────────────────────────────────────────────────────


async def test_chat_about_this_ends_the_set_and_hands_the_composer_back(tmp_path):
    provider = scripted(*asks("Sure — let's talk about the readers."))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("1")  # answer Storage, advance to Readers

        await pilot.press("5")  # Chat about this — no confirmation, the set is over

        await wait_until(pilot, lambda: idle_again(app))
        # a RESULT the model reads, not a cancellation
        [execution] = [e for e in app.runner.session.entries.values() if isinstance(e, ToolExecution)]
        assert (execution.status, execution.result.is_error) == (ExecutionStatus.COMPLETED, False)
        assert provider.requests[-1].messages[-1] == ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text=DECLINED_PROSE)],
        )
        # the answer already given is still recorded, and the composer is back
        assert [view.model for view in app.transcript.query(ListBlockView)] == [
            vm.ListBlock(
                rows=[
                    vm.ListRow(glyph="done", text="Storage", annotation="→  ~/.luca/projects/<project>/events.db"),
                    vm.ListRow(glyph="pending", text="Readers", annotation="→  [accent]chat about this[/]"),
                ],
                column=28,
            ),
        ]
        assert app.composer() is not None


# ── the sidecar ───────────────────────────────────────────────────────────────


async def test_the_sidecar_beside_the_session_carries_the_question_store(tmp_path):
    session = fresh_session()
    app = make_app(session, scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("1", "1", "enter")
        await wait_until(pilot, lambda: bool(app.query(QuestionConfirmView)))
        await submit(pilot, NOTE)
        await wait_until(pilot, lambda: idle_again(app))

        assert session_path(session.id, tmp_path).exists()
        assert json.loads(tui_state_path(session.id, tmp_path).read_text()) == {
            "questions": {
                "tc1": {
                    "questions": STORED_QUESTIONS,
                    "answer": {
                        "answers": [
                            {
                                "question": STORAGE_TITLE,
                                "chat_about_this": False,
                                "answers": ["~/.luca/projects/<project>/events.db"],
                                "custom_answer": None,
                            },
                            {
                                "question": READERS_TITLE,
                                "chat_about_this": False,
                                "answers": ["the sessions screen"],
                                "custom_answer": None,
                            },
                        ],
                        "custom_notes": NOTE,
                    },
                },
            },
        }


async def test_a_resumed_session_re_renders_the_parked_questions(tmp_path):
    session = fresh_session()
    app = make_app(session, scripted(*asks("Got it.")), tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        app.workers.cancel_group(app, "drive")  # leaves the call parked on disk
        await pilot.pause()

    provider = scripted(faux_assistant_message([faux_text("Got it.")]))
    resumed = make_app(load_session(session.id, tmp_path), provider, tmp_path)

    async with resumed.run_test(size=(105, 35)) as pilot:
        await wait_until(pilot, lambda: bool(resumed.query(QuestionSetView)))

        resumed.workers.cancel_group(resumed, "drive")
        await pilot.pause()
        # the same questions, read back out of the sidecar — the model was
        # never asked for them a second time, and a parked call shows no
        # transcript row while it holds the dock
        assert resumed.query_one(QuestionSetView).model == vm.QuestionSetState(
            questions=[storage(), readers()],
        )
        assert provider.requests == []
        assert list(resumed.transcript.query(ToolBlockView)) == []


# ── the composer's post guard ─────────────────────────────────────────────────


async def test_a_submit_while_a_question_set_is_out_names_the_questions(tmp_path):
    # The BLOCKED guard has two causes to tell apart (0007): an approval gate,
    # and a tool that said "not yet". The composer is normally not even mounted
    # while a set is out, but the swap is worker-driven, so there are windows
    # where it is; recreate one by mounting it over the parked call.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await app.show_composer(vm.ComposerState())
        await pilot.pause()
        session = app.runner.session
        nodes_before = list(session.conversations[session.main_conversation_id].nodes)

        await submit(pilot, "please hurry")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        # nothing was posted, the refusal names the questions, the draft stays
        assert session.conversations[session.main_conversation_id].nodes == nodes_before
        assert [(notice.text, notice.error) for notice in app.query(NoticeLine)] == [
            ("answer the questions first", True),
        ]
        assert app.query_one(PromptInput).text == "please hurry"


# ── the caret is not the answer ───────────────────────────────────────────────


async def test_arrowing_over_an_answered_radio_question_does_not_re_answer_it(tmp_path):
    # `selected` is the CARET and `answer` is the PICK. Conflating them makes a
    # pure inspection keystroke a re-answer: pick option 1, go back to re-read
    # the question, arrow down to look at option 2, and the model is told the
    # user chose option 2 — a decision they never made and never saw recorded.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))
        await pilot.press("1")  # answers Storage, advances to Readers
        await pilot.press("shift+tab")

        await pilot.press("down")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        question = app.query_one(QuestionSetView).model.questions[0]
        assert (question.selected, question.answer, question.answered) == (1, 0, True)


# ── the dock is not an overlay's to destroy ───────────────────────────────────


async def test_the_palette_refuses_to_open_over_a_question_set(tmp_path):
    # `show_overlay` empties the dock slot, and the view it would destroy is
    # the ONLY thing that can resolve the future the drive worker is blocked
    # on — after which `esc`, `^c` and the composer are all inert and the turn
    # is unrecoverable. So the palette (and the `@` picker) simply refuse.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        await pilot.press("ctrl+p")

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert bool(app.query(QuestionSetView)) is True
        assert bool(app.query(OverlayListView)) is False


# ── live and replay draw the same thing ───────────────────────────────────────


async def test_a_cancelled_set_renders_the_same_live_and_on_replay(tmp_path):
    # THE COLLAPSED BLOCK IS ONLY FOR A COMPLETED SET. A set the user backed
    # out of settles INTERRUPTED with no result at all, so it renders as the
    # ordinary tool block that says what happened — and the live event handler
    # and `transcript_blocks` go through the SAME derivation, or a session
    # would redraw differently the moment it was reloaded.
    app = make_app(fresh_session(), scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        app.runner.cancel(error="backed out of the questions")
        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        app._start_drive()
        await wait_until(pilot, lambda: idle_again(app))

        live = [view.model for view in app.query(ToolBlockView)]
        assert live == transcript_tool_blocks(app)
        assert [block.output.summary for block in live] == ["interrupted"]


# ── the sidecar ───────────────────────────────────────────────────────────────


async def test_a_session_that_never_asked_a_question_writes_no_sidecar(tmp_path):
    # The store is the app's own reference rather than a key `setdefault`-ed
    # into the TUI state, so the empty-state guard in `save_tui_state` can
    # actually fire and an ordinary session leaves one file, not two.
    app = make_app(fresh_session(), scripted(faux_assistant_message([faux_text("no questions here")])), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: idle_again(app))

        assert sorted(path.name for path in tmp_path.iterdir()) == [f"{app.runner.session.id}.json"]


async def test_a_sidecar_whose_store_is_the_wrong_shape_is_dropped(tmp_path):
    # A hand-mangled store would otherwise be handed straight to a tool that
    # indexes it, turning every `ask_user` call in the session into a FAILED
    # one — permanently, since the bad shape is written back on every save.
    session = fresh_session()
    tui_state_path(session.id, tmp_path).write_text(json.dumps({"questions": ["not", "a", "store"]}))
    app = make_app(session, scripted(*asks("Got it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, PROMPT)
        await wait_until(pilot, lambda: bool(app.query(QuestionSetView)))

        app.workers.cancel_group(app, "drive")
        await pilot.pause()
        assert app.questions.store == {"tc1": {"questions": STORED_QUESTIONS, "answer": None}}
