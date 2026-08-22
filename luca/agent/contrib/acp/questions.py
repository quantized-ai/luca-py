"""Parked `ask_user` calls → `elicitation/create`, and the answers back.

`ask_user` is a DEFERRED tool: it returns `ExecutionDeferred()`, the runner
parks the open turn at `AWAITING_RESULT`, and the same call is re-dispatched on
the next drive, where it finds an answer and returns a result. Resolving it is
the driver's job, out of band.

THE HANDLER MUST MAKE PROGRESS. There is no framework backoff, so a driver that
re-enters `run()` without resolving anything gets another deferral immediately
and spins. Every path here therefore writes an answer, including the path where
the client cannot ask anything at all — that one answers with a note telling the
model to ask in prose instead, which is a worse experience than a form and
infinitely better than a hang.

Elicitation is OPTIONAL in ACP and no client is required to offer it. Zed did
not when this was written.
"""

from __future__ import annotations

import logging

from acp.schema import (
    ElicitationFormRequestMode,
    ElicitationMultiSelectPropertySchema,
    ElicitationSchema,
    ElicitationStringPropertySchema,
    StringMultiSelectItems,
)

from luca.agent.contrib.questions import OptionsType, Question, QuestionsTool
from luca.agent.core.models import ToolExecution

logger = logging.getLogger(__name__)

QUESTIONS_NAMESPACE = "contrib.questions"

NO_ELICITATION_NOTE = (
    "The client cannot display a question form, so these questions were not asked."
    " Ask the user in plain text instead, in your next message."
)


def is_questions_tool(execution: ToolExecution) -> bool:
    """Matched by DECLARATION — namespace and name — not by name alone. An
    application's own tool called `ask_user` is not this one."""
    spec = execution.tool_spec
    return spec is not None and spec.namespace == QUESTIONS_NAMESPACE and spec.name == QuestionsTool.name


def elicitation_schema(questions: list[Question]) -> ElicitationSchema:
    """One question set as an elicitation form: one property per question,
    single-select as an enum and multiple-select as a multi-select.

    Every property is required. The model asked because it cannot proceed
    without the answers, and an optional field invites a form submitted empty,
    which resolves the call while telling the model nothing."""
    properties: dict = {}
    for index, question in enumerate(questions):
        key = f"q{index}"
        if question.options_type is OptionsType.MULTIPLE_SELECT:
            properties[key] = ElicitationMultiSelectPropertySchema(
                type="array",
                title=question.title,
                description=question.body,
                items=StringMultiSelectItems(type="string", enum=list(question.options)),
            )
        else:
            properties[key] = ElicitationStringPropertySchema(
                type="string",
                title=question.title,
                description=question.body,
                enum=list(question.options),
            )
    return ElicitationSchema(
        type="object",
        title="A few questions",
        properties=properties,
        required=list(properties),
    )


def answer_payload(questions: list[Question], content: dict | None) -> dict:
    """A form submission as the payload `QuestionsTool.answer()` stores.

    Pairs by POSITION through the `q<n>` keys the schema minted, and carries
    each question's title so the tool's own title-first pairing lines up.
    Unvalidated by design: everything here is text on its way to becoming a
    string the model reads, and the tool never rejects a payload."""
    content = content or {}
    answers = []
    for index, question in enumerate(questions):
        value = content.get(f"q{index}")
        if isinstance(value, list):
            selected = [str(item) for item in value]
        elif value is None:
            selected = []
        else:
            selected = [str(value)]
        answers.append(
            {
                "question": question.title,
                "chat_about_this": False,
                "answers": selected,
                "custom_answer": None,
            }
        )
    return {"answers": answers, "custom_notes": None}


def declined_payload(questions: list[Question], note: str) -> dict:
    """What to store when the questions could not be put to anyone.

    `chat_about_this` on the first question is the tool's own vocabulary for
    "the user would rather talk about this than pick", and it makes the tool
    render the instruction telling the model to carry on in prose. Reusing it
    means no new wording reaches the model from here."""
    answers = [
        {
            "question": question.title,
            "chat_about_this": index == 0,
            "answers": [],
            "custom_answer": note if index == 0 else None,
        }
        for index, question in enumerate(questions)
    ]
    return {"answers": answers, "custom_notes": note}


class QuestionBridge:
    """Puts a parked question set to the client, or answers it away."""

    def __init__(self, connection, session_id: str, tool: QuestionsTool, *, elicitation: bool) -> None:
        self._connection = connection
        self._session_id = session_id
        self._tool = tool
        self._elicitation = elicitation

    async def resolve(self, execution: ToolExecution) -> None:
        call_id = execution.tool_call_id
        questions = self._tool.pending(call_id)
        if not questions:
            # The store lost the job but the call is still parked. Answering
            # nothing is still an answer, and the turn survives.
            logger.warning("no questions found for the parked call %s; answering empty", call_id)
            self._tool.answer(call_id, {"answers": [], "custom_notes": None})
            return
        if not self._elicitation:
            self._tool.answer(call_id, declined_payload(questions, NO_ELICITATION_NOTE))
            return
        try:
            response = await self._connection.create_elicitation(
                message="The agent has a few questions before continuing.",
                mode=ElicitationFormRequestMode(requested_schema=elicitation_schema(questions)),
                session_id=self._session_id,
            )
        except Exception as exc:
            # A client that advertised elicitation and then refused the call is
            # still a client this turn has to get past.
            logger.error("elicitation failed for %s: %s", call_id, exc, exc_info=True)
            self._tool.answer(call_id, declined_payload(questions, f"The client could not ask: {exc}"))
            return
        if getattr(response, "action", None) != "accept":
            self._tool.answer(
                call_id,
                declined_payload(questions, "The user dismissed the questions without answering."),
            )
            return
        self._tool.answer(call_id, answer_payload(questions, getattr(response, "content", None)))
