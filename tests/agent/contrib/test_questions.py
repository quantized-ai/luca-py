"""Self-scoped tests for `luca.agent.contrib.questions`: the spec the model is
shown, `Args`' bounds, `execute()` as a pure predicate, the store, the
app-facing `pending` / `is_answered` / `answer` trio, the overridable renderer,
and the `QuestionsPlugin` surface.

NO RUNNER HERE. Everything the framework does with an `ExecutionDeferred` —
parking at `AWAITING_RESULT`, re-dispatching from scratch on the next drive,
appending one `ExecutionAttempt` per body invocation — is
`tests/agent/test_deferred_tools.py`'s subject. This file only ever calls the
tool the way the registry's prepared callable does, once per "drive".

`QuestionsTool` reads NOTHING from the session — the store is handed in — so one
inert session literal is passed to every entry point and is an invariant
everywhere.
"""

import json

import pytest
from pydantic import ValidationError

from luca.agent.contrib.questions import (
    DESCRIPTION,
    OptionsType,
    Question,
    QuestionsPlugin,
    QuestionsTool,
)
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionDeferred,
    ExecutionResult,
    SessionConfig,
    TextContent,
    ToolSpec,
)
from tests.agent.scenarios import (
    MODEL,
    conversation,
    main_conversation,
)

SESSION = AgentSession(
    id="s_questions",
    conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

CONVERSATION = main_conversation(SESSION).id


# ── the questions, exactly as they sit on the wire ────────────────────────────
# Plain dicts, because that is what `execute()` stores VERBATIM: the registry
# hands it `Args.model_validate(...).model_dump()` and the store keeps it
# untouched. `body` is spelled out because the dump always carries it.

YEAR = {
    "title": "What year is this?",
    "body": None,
    "options_type": "single_select",
    "options": ["2025", "2026"],
}

FRUIT = {
    "title": "What's your favorite fruit?",
    "body": "Pick as many as you like",
    "options_type": "multiple_select",
    "options": ["Banana", "Apple", "Peach"],
}

PLANET = {
    "title": "What's your favorite planet?",
    "body": None,
    "options_type": "single_select",
    "options": ["Mars", "Venus"],
}


# ── the payloads the driver submits ───────────────────────────────────────────

ANSWERED_PAYLOAD = {
    "answers": [
        {
            "question": "What year is this?",
            "chat_about_this": False,
            "answers": ["2026"],
            "custom_answer": None,
        },
        {
            "question": "What's your favorite fruit?",
            "chat_about_this": False,
            "answers": ["Banana", "Apple"],
            "custom_answer": "I also like peaches",
        },
    ],
    "custom_notes": None,
}

DECLINED_PAYLOAD = {
    "answers": [
        {
            "question": "What year is this?",
            "chat_about_this": False,
            "answers": ["2026"],
            "custom_answer": None,
        },
        {
            "question": "What's your favorite fruit?",
            "chat_about_this": False,
            "answers": [],
            "custom_answer": None,
        },
        {
            "question": "What's your favorite planet?",
            "chat_about_this": True,
            "answers": [],
            "custom_answer": None,
        },
    ],
    "custom_notes": None,
}


# ── the rendered strings, whole ───────────────────────────────────────────────
# Written out line by line rather than as a triple-quoted block: the string has
# no trailing newline and one line ends in a quote character.

ALL_ANSWERED_TEXT = (
    "User answered all your questions:\n"
    "\n"
    "What year is this?\n"
    "Answer:\n"
    "- 2026\n"
    "\n"
    "What's your favorite fruit?\n"
    "Answer:\n"
    "- Banana\n"
    "- Apple\n"
    'Custom note: "I also like peaches"'
)

ONE_DECLINED_TEXT = (
    "User declined to answer some questions.\n"
    'User wants to chat more about: "What\'s your favorite planet?"\n'
    "Respond to them in the conversation — do not ask this question again.\n"
    "\n"
    "Other questions/answers recorded:\n"
    "\n"
    "What year is this?\n"
    "Answer:\n"
    "- 2026\n"
    "\n"
    "What's your favorite fruit?\n"
    "Answer: No answer"
)


# ── the spec the model is shown ───────────────────────────────────────────────
# Written out whole rather than derived from `Args`: this is the package's
# public product — the one advertisement a registry hands the provider.

ASK_USER_SPEC = ToolSpec(
    name="ask_user",
    title="Ask the user",
    description=DESCRIPTION,
    namespace="contrib.questions",
    input_schema={
        "$defs": {
            "OptionsType": {
                "description": OptionsType.__doc__,
                "enum": ["single_select", "multiple_select"],
                "title": "OptionsType",
                "type": "string",
            },
            "Question": {
                "additionalProperties": False,
                "description": Question.__doc__,
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "body": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Body",
                    },
                    "options_type": {"$ref": "#/$defs/OptionsType"},
                    "options": {
                        "items": {"type": "string"},
                        "minItems": 2,
                        "title": "Options",
                        "type": "array",
                    },
                },
                "required": ["title", "options_type", "options"],
                "title": "Question",
                "type": "object",
            },
        },
        "additionalProperties": False,
        "properties": {
            "questions": {
                "items": {"$ref": "#/$defs/Question"},
                "maxItems": 4,
                "minItems": 1,
                "title": "Questions",
                "type": "array",
            },
        },
        "required": ["questions"],
        "title": "Args",
        "type": "object",
    },
)


# ── module-level factories ────────────────────────────────────────────────────


def parked(tool_call_id: str, questions: list[dict]) -> dict:
    """One store entry as `execute()` seeds it: the questions verbatim and no
    answer yet."""
    return {tool_call_id: {"questions": questions, "answer": None}}


async def dispatch(tool: QuestionsTool, questions: list[dict], tool_call_id: str = "tc1"):
    """One drive's worth of dispatch — exactly the call the registry's prepared
    callable makes."""
    return await tool.execute(
        {"questions": questions},
        SESSION,
        CONVERSATION,
        tool_name="ask_user",
        tool_call_id=tool_call_id,
        cancellation_token=CancellationToken(),
    )


# ── the spec ──────────────────────────────────────────────────────────────────


def test_the_tool_advertises_ask_user_with_the_schema_derived_from_args():
    assert QuestionsTool().get_tool_spec() == ASK_USER_SPEC


def test_the_tool_declares_no_execution_deadline():
    # Deliberate, and the one ClassVar a reader is most likely to "fix": each
    # poll is an instant dict read, and the PARKED period is unbounded by
    # design, so no per-poll deadline may touch it.
    assert QuestionsTool.timeout_in_ms is None


def test_the_description_tells_the_model_to_ask_everything_in_one_call():
    # Partial by design: the prose is the product's to tune, but these two
    # instructions are the only thing standing between a parked call and a
    # second one under a second tool_call_id.
    assert "Ask everything you need in ONE call" in DESCRIPTION
    assert "do not re-ask" in DESCRIPTION


# ── the argument bounds ───────────────────────────────────────────────────────


def test_a_call_with_no_questions_at_all_is_rejected():
    # min_length=1: an empty call would park the turn on nothing.
    with pytest.raises(ValidationError):
        QuestionsTool.Args.model_validate({"questions": []})


def test_a_call_with_five_questions_is_rejected():
    # max_length=4 is a product call: the answering contract is all-at-once, so
    # a longer set is a wall the user must clear in one sitting.
    with pytest.raises(ValidationError):
        QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT, PLANET, YEAR, FRUIT]})


def test_four_questions_are_accepted():
    args = QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT, PLANET, YEAR]})

    assert args.questions == [
        Question(title="What year is this?", options_type=OptionsType.SINGLE_SELECT, options=["2025", "2026"]),
        Question(
            title="What's your favorite fruit?",
            body="Pick as many as you like",
            options_type=OptionsType.MULTIPLE_SELECT,
            options=["Banana", "Apple", "Peach"],
        ),
        Question(
            title="What's your favorite planet?", options_type=OptionsType.SINGLE_SELECT, options=["Mars", "Venus"]
        ),
        Question(title="What year is this?", options_type=OptionsType.SINGLE_SELECT, options=["2025", "2026"]),
    ]


def test_a_question_offering_a_single_option_is_rejected():
    # min_length=2 on options: a one-option question is not a question, and a
    # question with no good options is one the model should ask in prose.
    with pytest.raises(ValidationError):
        QuestionsTool.Args.model_validate(
            {"questions": [{"title": "Ship it?", "options_type": "single_select", "options": ["yes"]}]}
        )


def test_an_unknown_field_on_a_question_is_rejected():
    with pytest.raises(ValidationError):
        QuestionsTool.Args.model_validate(
            {
                "questions": [
                    {"title": "Ship it?", "options_type": "single_select", "options": ["yes", "no"], "urgent": True}
                ]
            }
        )


# ── execute() as a pure predicate ─────────────────────────────────────────────


async def test_the_first_dispatch_seeds_the_job_and_defers():
    tool = QuestionsTool()

    result = await dispatch(tool, [YEAR, FRUIT])

    assert result == ExecutionDeferred()
    assert tool.store == parked("tc1", [YEAR, FRUIT])


async def test_a_dispatch_after_the_seeding_one_only_reads_the_store():
    # The runner re-dispatches from scratch once per drive and nothing bounds
    # how many drives a parked call sees, so every poll after the first must be
    # a pure read.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT])

    results = [await dispatch(tool, [YEAR, FRUIT]) for _ in range(3)]

    assert results == [ExecutionDeferred(), ExecutionDeferred(), ExecutionDeferred()]
    assert tool.store == parked("tc1", [YEAR, FRUIT])


async def test_a_re_dispatch_never_overwrites_the_seeded_questions():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])

    await dispatch(tool, [PLANET])

    assert tool.store == parked("tc1", [YEAR])


async def test_an_answered_call_returns_one_result_carrying_the_payload_verbatim():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT])
    tool.answer("tc1", ANSWERED_PAYLOAD)

    result = await dispatch(tool, [YEAR, FRUIT])

    assert result == ExecutionResult(
        content=[TextContent(text=ALL_ANSWERED_TEXT)],
        structured_content={"questions": [YEAR, FRUIT], "answer": ANSWERED_PAYLOAD},
        is_error=False,
    )


async def test_a_declined_question_still_completes_the_call_without_an_error():
    # "Chat about this" is a RESULT the model reads, not a failure — and an
    # error result is exactly what makes a model retry.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT, PLANET])
    tool.answer("tc1", DECLINED_PAYLOAD)

    result = await dispatch(tool, [YEAR, FRUIT, PLANET])

    assert result == ExecutionResult(
        content=[TextContent(text=ONE_DECLINED_TEXT)],
        structured_content={"questions": [YEAR, FRUIT, PLANET], "answer": DECLINED_PAYLOAD},
        is_error=False,
    )


async def test_answering_one_of_two_outstanding_calls_leaves_the_other_parked():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR], "tc1")
    await dispatch(tool, [PLANET], "tc2")
    tool.answer("tc1", {"answers": [], "custom_notes": None})

    results = [await dispatch(tool, [YEAR], "tc1"), await dispatch(tool, [PLANET], "tc2")]

    assert results == [
        ExecutionResult(
            content=[TextContent(text="User answered all your questions:\n\nWhat year is this?\nAnswer: No answer")],
            structured_content={"questions": [YEAR], "answer": {"answers": [], "custom_notes": None}},
            is_error=False,
        ),
        ExecutionDeferred(),
    ]


async def test_a_restored_store_brings_a_parked_call_back_parked():
    # The whole resume story: the execution is still AWAITING_RESULT, the next
    # drive re-dispatches, and `execute()` finds the restored job untouched.
    saved = json.loads(json.dumps(parked("tc1", [YEAR, FRUIT])))
    tool = QuestionsTool(store=saved)

    result = await dispatch(tool, [YEAR, FRUIT])

    assert result == ExecutionDeferred()
    assert saved == parked("tc1", [YEAR, FRUIT])


# ── the store ─────────────────────────────────────────────────────────────────


def test_a_tool_given_no_store_makes_its_own():
    assert QuestionsTool().store == {}


async def test_the_store_a_tool_is_given_is_mutated_in_place():
    # The whole persistence contract: hand in a dict and it IS the memory.
    # Whoever owns the dict decides whether it outlives the process.
    saved: dict = {}
    tool = QuestionsTool(store=saved)

    await dispatch(tool, [YEAR])

    assert saved == parked("tc1", [YEAR])


async def test_the_store_round_trips_through_json():
    # Non-negotiable: no Pydantic instance, no datetime, nothing an application
    # cannot write to a file.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT])
    tool.answer("tc1", ANSWERED_PAYLOAD)

    await dispatch(tool, [YEAR, FRUIT])

    assert tool.store == json.loads(json.dumps(tool.store))


async def test_registry_validated_arguments_land_in_the_store_and_still_round_trip():
    # The registry's prepare() hands execute() the
    # Args.model_validate(...).model_dump() dict, whose options_type is an
    # OptionsType member — the store must still serialize and compare equal.
    tool = QuestionsTool()
    args = QuestionsTool.Args.model_validate({"questions": [YEAR]}).model_dump()

    await dispatch(tool, args["questions"])

    assert tool.store == parked("tc1", [YEAR])
    assert tool.store == json.loads(json.dumps(tool.store))


async def test_several_calls_are_outstanding_at_once_under_their_own_keys():
    # `tool_call_id` is the key precisely so one shared instance can serve
    # several calls in one message and several subagents at once.
    tool = QuestionsTool()

    results = [
        await dispatch(tool, [YEAR], "tc1"),
        await dispatch(tool, [FRUIT], "tc2"),
        await dispatch(tool, [PLANET], "tc3"),
    ]

    assert results == [ExecutionDeferred(), ExecutionDeferred(), ExecutionDeferred()]
    assert tool.store == parked("tc1", [YEAR]) | parked("tc2", [FRUIT]) | parked("tc3", [PLANET])


async def test_an_answered_call_stays_in_the_store_when_the_next_one_arrives():
    # The store ACCUMULATES: it is the session's question record, not a working
    # set that needs sweeping. Nothing is ever removed.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR], "tc1")
    tool.answer("tc1", ANSWERED_PAYLOAD)
    await dispatch(tool, [YEAR], "tc1")

    await dispatch(tool, [PLANET], "tc2")

    assert tool.store == {
        "tc1": {"questions": [YEAR], "answer": ANSWERED_PAYLOAD},
        "tc2": {"questions": [PLANET], "answer": None},
    }


# ── pending() and is_answered() ───────────────────────────────────────────────


async def test_pending_re_validates_the_stored_dicts_into_question_objects():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT])

    assert tool.pending("tc1") == [
        Question(title="What year is this?", options_type=OptionsType.SINGLE_SELECT, options=["2025", "2026"]),
        Question(
            title="What's your favorite fruit?",
            body="Pick as many as you like",
            options_type=OptionsType.MULTIPLE_SELECT,
            options=["Banana", "Apple", "Peach"],
        ),
    ]


def test_pending_is_empty_for_an_id_the_store_has_never_seen():
    assert QuestionsTool().pending("nope") == []


def test_pending_skips_a_stored_question_that_no_longer_validates():
    # A hand-edited or half-migrated store is not worth a crash: the readable
    # questions still come back.
    tool = QuestionsTool(store={"tc1": {"questions": [{"title": "no options"}, YEAR], "answer": None}})

    assert tool.pending("tc1") == [
        Question(title="What year is this?", options_type=OptionsType.SINGLE_SELECT, options=["2025", "2026"])
    ]


async def test_a_parked_call_is_not_answered():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])

    assert tool.is_answered("tc1") is False


async def test_a_call_carrying_a_payload_is_answered():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])

    tool.answer("tc1", ANSWERED_PAYLOAD)

    assert tool.is_answered("tc1") is True


def test_an_id_the_store_has_never_seen_is_not_answered():
    assert QuestionsTool().is_answered("nope") is False


# ── answer() never raises ─────────────────────────────────────────────────────
# THE LOAD-BEARING PROPERTY of this package. 0007 has no handler backoff: a
# driver's handler must block until it has made progress, so a raise here
# aborts it before it resolves anything, the drive re-enters, defers again, and
# spins as fast as Python runs. A cosmetic mismatch must never become a hung
# driver — so nothing is validated and nothing is rejected.


def test_answering_an_id_the_store_has_never_seen_creates_the_slot():
    # An answer arriving before the first execute() still works, and a
    # genuinely wrong id is inert instead of fatal.
    tool = QuestionsTool()

    tool.answer("tc-unknown", ANSWERED_PAYLOAD)

    assert tool.store == {"tc-unknown": {"questions": [], "answer": ANSWERED_PAYLOAD}}


async def test_a_payload_missing_every_expected_key_is_stored_verbatim():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])

    tool.answer("tc1", {"selections": ["2026"]})

    assert tool.store == {"tc1": {"questions": [YEAR], "answer": {"selections": ["2026"]}}}


async def test_a_payload_whose_fields_have_the_wrong_types_is_stored_verbatim():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])

    tool.answer("tc1", {"answers": "2026", "custom_notes": 7})

    assert tool.store == {"tc1": {"questions": [YEAR], "answer": {"answers": "2026", "custom_notes": 7}}}


def test_a_payload_that_is_not_a_dict_at_all_is_stored_rather_than_rejected():
    # `answer()` itself has NO failure mode whatsoever — the annotation says
    # `dict`, and even something that is not one is recorded rather than
    # raised on.
    tool = QuestionsTool()

    tool.answer("tc1", "the user said yes")

    assert tool.store == {"tc1": {"questions": [], "answer": "the user said yes"}}


async def test_a_payload_that_is_not_a_dict_at_all_still_renders():
    # The other half of the same contract, and the reason it matters: whatever
    # `answer()` accepted, `execute()` has to render on the next dispatch. A
    # raise here would be a FAILED tool call that loses the questions — the one
    # place in this tool where being strict costs a turn.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])
    tool.answer("tc1", "the user said yes")

    result = await dispatch(tool, [YEAR])

    assert result == ExecutionResult(
        content=[
            TextContent(
                text="User answered all your questions:\n\nWhat year is this?\nAnswer: No answer",
            )
        ],
        structured_content={"questions": [YEAR], "answer": "the user said yes"},
        is_error=False,
    )


async def test_answering_twice_is_last_write_wins():
    tool = QuestionsTool()
    await dispatch(tool, [YEAR])
    tool.answer("tc1", ANSWERED_PAYLOAD)

    tool.answer("tc1", {"answers": [], "custom_notes": "changed my mind"})

    assert tool.store == {"tc1": {"questions": [YEAR], "answer": {"answers": [], "custom_notes": "changed my mind"}}}


async def test_a_malformed_payload_is_rendered_as_best_it_can_be():
    # The other half of "never raises": the rendering runs INSIDE execute(), so
    # a raise there is a FAILED tool call — the one place in this tool where
    # being strict costs a turn.
    tool = QuestionsTool()
    await dispatch(tool, [YEAR, FRUIT])
    tool.answer("tc1", {"answers": "2026", "custom_notes": 7})

    result = await dispatch(tool, [YEAR, FRUIT])

    assert result == ExecutionResult(
        content=[
            TextContent(
                text=(
                    "User answered all your questions:\n"
                    "\n"
                    "What year is this?\n"
                    "Answer: No answer\n"
                    "\n"
                    "What's your favorite fruit?\n"
                    "Answer: No answer"
                )
            )
        ],
        structured_content={"questions": [YEAR, FRUIT], "answer": {"answers": "2026", "custom_notes": 7}},
        is_error=False,
    )


# ── the rendering ─────────────────────────────────────────────────────────────


def test_every_question_answered_renders_the_answered_shape():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT]}).questions

    assert QuestionsTool().generate_content_string_from_answers(questions, ANSWERED_PAYLOAD) == ALL_ANSWERED_TEXT


def test_a_declined_question_renders_the_do_not_ask_again_instruction():
    # Without that third line the model reads "the user wants to chat", has no
    # instruction, and its most likely repair is to re-ask — which mints a
    # SECOND call under a second tool_call_id.
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT, PLANET]}).questions

    assert QuestionsTool().generate_content_string_from_answers(questions, DECLINED_PAYLOAD) == ONE_DECLINED_TEXT


def test_custom_notes_close_the_string():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {"answers": [{"question": "What year is this?", "answers": ["2026"]}], "custom_notes": "ship on Friday"}

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n"
        "\n"
        "What year is this?\n"
        "Answer:\n"
        "- 2026\n"
        "\n"
        'Additional note from the user: "ship on Friday"'
    )


def test_a_question_with_neither_selections_nor_custom_text_renders_no_answer():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {
        "answers": [{"question": "What year is this?", "chat_about_this": False, "answers": [], "custom_answer": "  "}],
        "custom_notes": None,
    }

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n\nWhat year is this?\nAnswer: No answer"
    )


def test_an_entry_whose_title_does_not_match_is_paired_by_position():
    # Fallback two of three: the payload's only destination is prose, so
    # nothing needs the strings to agree.
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {"answers": [{"question": "what year??", "answers": ["2026"]}], "custom_notes": None}

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n\nWhat year is this?\nAnswer:\n- 2026"
    )


def test_an_entry_matching_no_stored_question_is_rendered_standalone():
    # Fallback three of three: the payload carries the text, so nothing is lost.
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {
        "answers": [
            {"question": "What year is this?", "answers": ["2026"]},
            {"question": "A question nobody stored", "answers": ["surprise"]},
        ],
        "custom_notes": None,
    }

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n"
        "\n"
        "What year is this?\n"
        "Answer:\n"
        "- 2026\n"
        "\n"
        "A question nobody stored\n"
        "Answer:\n"
        "- surprise"
    )


def test_a_question_the_payload_omits_entirely_renders_no_answer():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT]}).questions
    payload = {"answers": [{"question": "What year is this?", "answers": ["2026"]}], "custom_notes": None}

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n"
        "\n"
        "What year is this?\n"
        "Answer:\n"
        "- 2026\n"
        "\n"
        "What's your favorite fruit?\n"
        "Answer: No answer"
    )


def test_payload_entries_that_are_not_dicts_are_ignored():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {"answers": [None, "2026", 7], "custom_notes": None}

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User answered all your questions:\n\nWhat year is this?\nAnswer: No answer"
    )


def test_a_declined_entry_with_no_title_of_its_own_borrows_the_stored_one():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR]}).questions
    payload = {"answers": [{"chat_about_this": True}], "custom_notes": None}

    assert QuestionsTool().generate_content_string_from_answers(questions, payload) == (
        "User declined to answer some questions.\n"
        'User wants to chat more about: "What year is this?"\n'
        "Respond to them in the conversation — do not ask this question again.\n"
        "\n"
        "Other questions/answers recorded:"
    )


class ShoutingQuestionsTool(QuestionsTool):
    """A subclass that changes the wording and nothing else — the reason the
    templates are ClassVars rather than literals in the loop."""

    ANSWERED_PREAMBLE = "THE HUMAN HAS SPOKEN:"
    ANSWER_LABEL = "THEY SAID:"
    CUSTOM_NOTE_TEMPLATE = "AND ALSO: {text}"


def test_a_subclass_can_change_the_wording_by_overriding_the_templates():
    questions = QuestionsTool.Args.model_validate({"questions": [YEAR, FRUIT]}).questions

    assert ShoutingQuestionsTool().generate_content_string_from_answers(questions, ANSWERED_PAYLOAD) == (
        "THE HUMAN HAS SPOKEN:\n"
        "\n"
        "What year is this?\n"
        "THEY SAID:\n"
        "- 2026\n"
        "\n"
        "What's your favorite fruit?\n"
        "THEY SAID:\n"
        "- Banana\n"
        "- Apple\n"
        "AND ALSO: I also like peaches"
    )


# ── the plugin ────────────────────────────────────────────────────────────────


def test_the_plugin_holds_one_tool_instance():
    # The application needs a STABLE reference to call answer() on, so the
    # plugin builds the tool once instead of per get_tools() call.
    plugin = QuestionsPlugin()

    assert plugin.get_tools() == [plugin.tool]
    assert plugin.get_tools() == [plugin.tool]


def test_the_plugin_shares_its_store_with_its_tool():
    plugin = QuestionsPlugin()

    assert plugin.tool.store is plugin.store


def test_a_plugin_given_no_store_makes_its_own():
    assert QuestionsPlugin().store == {}


async def test_the_store_a_plugin_is_given_is_the_store_its_tool_writes():
    saved: dict = {}
    plugin = QuestionsPlugin(store=saved)

    await dispatch(plugin.tool, [YEAR])

    assert saved == parked("tc1", [YEAR])


def test_tool_class_lets_a_subclassed_renderer_in():
    plugin = QuestionsPlugin(tool_class=ShoutingQuestionsTool)

    assert type(plugin.tool) is ShoutingQuestionsTool
    assert plugin.tool.store is plugin.store


def test_get_tool_registry_wraps_the_tool_in_an_auto_allowing_registry():
    # Asking a question is not an action to approve — a gate in front of it
    # would mean the user approving BEING ASKED something.
    plugin = QuestionsPlugin()

    registry = plugin.get_tool_registry(SESSION)

    assert type(registry) is SimpleToolRegistry
    assert type(registry.permission_policy) is YoloPermissionPolicy
    assert registry.tools == [plugin.tool]


async def test_the_registry_advertises_the_one_ask_user_spec():
    registry = QuestionsPlugin().get_tool_registry(SESSION)

    specs = await registry.get_tools(SESSION, CONVERSATION)

    assert specs == [ASK_USER_SPEC]
