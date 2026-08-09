"""`QuestionsTool` — the first deferred tool.

The model asks the user one or more questions and waits for the answers. It is
the smallest real application of 0007's deferred tool calling: `execute()`
returns `ExecutionDeferred()` while the answers are missing, the runner parks
the open turn at `AWAITING_RESULT`, the driver renders the questions and
collects the answers out of band, and the next drive re-dispatches the SAME
call — which now finds an answer and returns its one `ExecutionResult`.

TWO SURFACES, and the split is the whole design:

- the FRAMEWORK-FACING side — `execute()`, called by the runner through the
  registry's prepared callable;
- the APP-FACING side — `pending()` / `is_answered()` / `answer()`, ordinary
  Python methods the framework has never heard of. The driver knows how to
  interact with the tools it installed; the core does not, and must not.

Everything here is contrib. The core learns nothing.

`execute()` IS A PURE PREDICATE. It never waits — no future, no event, no UI.
It reads state and answers *ready / not yet*. That is what makes it safe for
the runner to invoke once per drive, however many drives there are: `setdefault`
makes the first call the seeding call and every later one a read.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionDeferred,
    ExecutionResult,
    TextContent,
)

QUESTIONS_NAMESPACE = "contrib.questions"

DESCRIPTION = """
Ask the user one or more questions and wait for their answers.

Use this when you genuinely cannot proceed without a decision only the user can \
make — not to confirm work you were already asked to do. Ask everything you need \
in ONE call: the user answers every question in a single pass, and a second call \
while the first is outstanding shows them a second prompt.

Each question offers the user your options plus "type your own answer" and \
"let's chat about this". A user who picks "chat about this" is declining the \
question and asking to discuss it — answer them in the conversation, do not \
re-ask.
""".strip()


class OptionsType(str, Enum):
    """How many of a question's options the user may pick."""

    SINGLE_SELECT = "single_select"  # radio
    MULTIPLE_SELECT = "multiple_select"  # checkbox


class Question(BaseModel):
    """One question, exactly as the model authored it.

    `options` is `list[str]` and not a one-field wrapper model: the wrapper
    costs a level of nesting in the JSON Schema the model has to fill in and
    buys nothing today. The one reason to promote it later is a per-option
    `description` — a real second case, and a contained change when it
    arrives."""

    model_config = ConfigDict(extra="forbid")

    title: str  # the question itself
    body: str | None = None  # optional elaboration / why it is being asked
    options_type: OptionsType
    # A one-option question is not a question, and a question with no good
    # options at all is one the model should ask in prose instead — which is
    # why this tool deliberately has no free-text-only mode.
    options: list[str] = Field(min_length=2)


class QuestionsTool(Tool):
    """Ask the user, park, and read the answers back on a later dispatch.

    THE STORE IS A CONSTRUCTOR ARGUMENT, and that is the whole persistence
    story — `MemoryPlugin`'s verbatim:

        QuestionsTool()                 # works, forgets everything on exit
        QuestionsTool(store=mine)       # works, and `mine` IS the memory

    The tool mutates the dict it was handed, in place. Hand in a dict the
    application keeps somewhere durable and a resumed session picks the
    outstanding questions back up; hand in nothing and it starts clean, which
    is right for a script and wrong for anything a user comes back to. The tool
    never reads or writes a file and does not know what a session save is.

    THE STORE IS JSON-SHAPED ALL THE WAY DOWN. Non-negotiable, because the
    whole point is that the application can serialize it:

        store = {
            "<tool_call_id>": {
                "questions": [ {...}, ... ],   # verbatim from args["questions"]
                "answer": {...} | None,        # verbatim from answer()
            },
        }

    No Pydantic instances, no dataclasses, no `datetime`. `questions` is
    exactly what came off the wire (`Args` already validated it), and `answer`
    is exactly the payload the driver submitted. `store == json.loads(
    json.dumps(store))` round-trips. `pending()` re-validates the stored dicts
    into `Question` objects on the way out, so a caller gets typed objects
    without the store holding any.

    KEYED BY `tool_call_id`, and that matters twice. Several calls in one
    assistant message, and several subagents each with their own, can be
    outstanding at once, so the key has to be per-CALL rather than per
    conversation. A `tool_call_id` is globally unique, which buys two things a
    conversation-keyed store does not get: no lock is needed (no two
    conversations can collide on a key), and no compaction re-keying (a
    compaction installs a new conversation id; a `tool_call_id` never moves).

    THE STORE ACCUMULATES, DELIBERATELY. Nothing is ever removed. Every call
    the model made stays keyed by its `tool_call_id`, with its questions and —
    once answered — the payload verbatim: the store is the session's QUESTION
    RECORD, not a working set that needs sweeping. Two consequences follow,
    both wanted: `answer is None` iff the call is still pending (one field is
    the whole status, readable straight off a restored dict without consulting
    the session), and a completed entry is inert (its execution is terminal, so
    the runner never re-dispatches it and the tool never reads it again). It is
    also why this tool needs no lifecycle callback and 0007 needs no cleanup
    hook: there is nothing to clean up."""

    namespace: ClassVar[str] = QUESTIONS_NAMESPACE
    name: ClassVar[str] = "ask_user"
    title: ClassVar[str] = "Ask the user"
    description: ClassVar[str] = DESCRIPTION

    # Deliberately unset. Each poll is an instant dict read, and the PARKED
    # period is unbounded by design — no per-poll deadline touches it.
    timeout_in_ms: ClassVar[int | None] = None

    # Wording lives in ClassVar templates, the same pattern
    # `ConversationProjector` uses, so a subclass that wants one different
    # sentence does not have to reimplement the loop.
    ANSWERED_PREAMBLE: ClassVar[str] = "User answered all your questions:"
    DECLINED_PREAMBLE: ClassVar[str] = "User declined to answer some questions."
    DECLINED_TEMPLATE: ClassVar[str] = 'User wants to chat more about: "{title}"'
    # The point of the whole declined branch. Without it the model reads "the
    # user wants to chat" and has no instruction, and its most likely repair is
    # to re-ask — which mints a SECOND call under a second `tool_call_id`.
    DECLINED_INSTRUCTION: ClassVar[str] = "Respond to them in the conversation — do not ask this question again."
    DECLINED_OTHERS_PREAMBLE: ClassVar[str] = "Other questions/answers recorded:"
    ANSWER_LABEL: ClassVar[str] = "Answer:"
    NO_ANSWER: ClassVar[str] = "Answer: No answer"
    CUSTOM_NOTE_TEMPLATE: ClassVar[str] = 'Custom note: "{text}"'
    CHAT_ABOUT_THIS: ClassVar[str] = "chat about this"
    NOTES_TEMPLATE: ClassVar[str] = 'Additional note from the user: "{text}"'

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        # `min_length=1` rejects an empty call. `max_length=4` is a PRODUCT
        # call, not a technical one: the answering contract is all-at-once, so
        # a ten-question call is a wall the user must clear in one sitting.
        questions: list[Question] = Field(min_length=1, max_length=4)

    def __init__(self, store: dict | None = None) -> None:
        self.store: dict = {} if store is None else store

    # ── the framework-facing side ────────────────────────────────────────────

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        """Seed the job on the first dispatch; answer *ready / not yet* on
        every dispatch, this one included.

        `is_error` is always False on the ready path: a refusal is an ANSWER,
        not a failure — and an error result is exactly what makes a model
        retry. `structured_content` carries the questions and the payload
        verbatim, including whatever the rendering ignores, so the durable
        record stays complete even though the prose is opinionated."""
        job = self.store.setdefault(
            tool_call_id,
            {"questions": args["questions"], "answer": None},
        )

        if job.get("answer") is None:
            return ExecutionDeferred()

        questions = self._questions_of(job)
        return ExecutionResult(
            content=[
                TextContent(
                    text=self.generate_content_string_from_answers(questions, job["answer"]),
                )
            ],
            structured_content=job,
            is_error=False,
        )

    # ── the app-facing side (the framework has never heard of these) ─────────

    def pending(self, tool_call_id: str) -> list[Question]:
        """The questions this call is waiting on, re-validated out of the
        store. An empty list for an id the store has never seen.

        The driver already holds the parked `ToolExecution` (from
        `AgentSessionRunner.pending_deferred_tool_executions()`), so
        `execution.tool_call_id` is the handle. It could equally read
        `execution.raw_tool_call.arguments`; this exists so a driver does not
        have to re-parse the model's raw arguments."""
        job = self.store.get(tool_call_id)
        return self._questions_of(job) if job else []

    def is_answered(self, tool_call_id: str) -> bool:
        """Has this call been answered? `answer is None` is the whole status."""
        job = self.store.get(tool_call_id)
        return bool(job) and job.get("answer") is not None

    def answer(self, tool_call_id: str, payload: dict) -> None:
        """Record every answer for one call, at once.

        THIS METHOD NEVER RAISES, and the reason is structural rather than
        stylistic. 0007's driver contract requires the handler to BLOCK UNTIL
        IT HAS MADE PROGRESS, and there is no framework backoff: a handler that
        returns without resolving anything causes an immediate re-drive,
        another deferral, and a spin. A `raise` here is exactly that — it
        aborts the handler before it resolves anything. A raising validator
        turns a cosmetic mismatch into a hung UI, which is strictly worse than
        the mismatch. Everything downstream of this is text on its way to
        becoming a string the model reads: nothing needs the option strings to
        match what was declared, the list to be complete, or the titles to be
        spelled the same way.

        An unknown `tool_call_id` CREATES the slot rather than failing, so an
        answer arriving before the first `execute()` still works and a
        genuinely wrong id is inert instead of fatal. Calling it twice is
        last-write-wins; once the execution is COMPLETED the runner never
        re-dispatches, so a late second call does nothing."""
        job = self.store.setdefault(tool_call_id, {"questions": [], "answer": None})
        job["answer"] = payload

    # ── rendering (the override point) ───────────────────────────────────────

    def generate_content_string_from_answers(
        self,
        questions: list[Question],
        payload: dict,
    ) -> str:
        """What the MODEL reads. Public, instance-level, overridable — a driver
        subclasses `QuestionsTool` and replaces exactly this method to change
        the wording, without touching state, validation or lifecycle.

        It receives BOTH the questions and the payload, because rendering needs
        the question ORDER and the TITLES of questions the payload omits.

        IT READS EVERYTHING DEFENSIVELY AND HAS NO FAILURE MODE. A missing key,
        an unexpected type, an unmatched entry all render as something rather
        than raising: this runs inside `execute()`, so a raise here is a FAILED
        tool call — the one place in this tool where being strict costs a
        turn."""
        # `answer()` accepts ANYTHING — it never raises — so whatever it stored
        # has to render here, including a payload that is not a mapping at all.
        # A raise on this path is a FAILED tool call that loses the questions.
        payload = payload if isinstance(payload, dict) else {}
        entries = payload.get("answers")
        entries = [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
        # Pair each stored question with its payload entry ONCE, so the
        # "unmatched" pass below cannot disagree with the main loop.
        paired = self._pair(questions, entries)
        matched = {id(entry) for _, entry in paired if entry is not None}
        declined_index = next(
            (index for index, entry in enumerate(entries) if entry.get("chat_about_this")),
            None,
        )
        declined = entries[declined_index] if declined_index is not None else None

        lines: list[str] = []
        if declined is None:
            lines.append(self.ANSWERED_PREAMBLE)
        else:
            title = declined.get("question")
            if not isinstance(title, str) or not title:
                title = questions[declined_index].title if declined_index < len(questions) else ""
            lines.append(self.DECLINED_PREAMBLE)
            lines.append(self.DECLINED_TEMPLATE.format(title=title))
            lines.append(self.DECLINED_INSTRUCTION)
            lines.append("")
            lines.append(self.DECLINED_OTHERS_PREAMBLE)

        for question, entry in paired:
            if declined is not None and entry is declined:
                continue
            lines.append("")
            lines.append(question.title)
            lines.extend(self._answer_lines(entry))

        # A payload entry matching no stored question is still rendered, under
        # its own `question` string — the payload carries the text, so nothing
        # is lost and nothing needs to agree.
        for entry in entries:
            if id(entry) in matched or entry is declined:
                continue
            lines.append("")
            lines.append(str(entry.get("question", "")))
            lines.extend(self._answer_lines(entry))

        notes = payload.get("custom_notes")
        if isinstance(notes, str) and notes.strip():
            lines.append("")
            lines.append(self.NOTES_TEMPLATE.format(text=notes))

        return "\n".join(lines)

    # ── internals (all defensive; none of them raise) ────────────────────────

    def _questions_of(self, job: dict) -> list[Question]:
        stored = job.get("questions")
        stored = stored if isinstance(stored, list) else []
        questions: list[Question] = []
        for item in stored:
            try:
                questions.append(Question.model_validate(item))
            except Exception:  # a hand-edited store is not worth a crash
                continue
        return questions

    def _pair(self, questions: list[Question], entries: list) -> list[tuple[Question, dict | None]]:
        """Pair every stored question with its payload entry: by TITLE first
        across the whole set, then by POSITION for whatever is left over, and
        otherwise not at all.

        EACH ENTRY IS CLAIMED ONCE. Titles are matched in a first pass so a
        positional fallback can never steal an entry that belongs to a later
        question by name — which would report one user answer under two
        questions and tell the model they made a choice they never made. A
        driver sends the full set in order, so in practice the title pass
        claims everything and the second pass does nothing.

        Three fallbacks deep and none of them fail."""
        claimed: set[int] = set()
        matched: dict[int, dict] = {}
        for index, question in enumerate(questions):
            for position, entry in enumerate(entries):
                if position not in claimed and entry.get("question") == question.title:
                    claimed.add(position)
                    matched[index] = entry
                    break
        for index, _question in enumerate(questions):
            if index in matched or index >= len(entries) or index in claimed:
                continue
            claimed.add(index)
            matched[index] = entries[index]
        return [(question, matched.get(index)) for index, question in enumerate(questions)]

    def _answer_lines(self, entry: dict | None) -> list[str]:
        if entry is None:
            return [self.NO_ANSWER]
        if entry.get("chat_about_this"):
            return [f"{self.ANSWER_LABEL} {self.CHAT_ABOUT_THIS}"]
        selected = entry.get("answers")
        selected = [str(item) for item in selected] if isinstance(selected, list) else []
        custom = entry.get("custom_answer")
        custom = custom if isinstance(custom, str) and custom.strip() else None
        if not selected and custom is None:
            return [self.NO_ANSWER]
        lines = []
        if selected:
            lines.append(self.ANSWER_LABEL)
            lines.extend(f"- {item}" for item in selected)
        if custom is not None:
            lines.append(self.CUSTOM_NOTE_TEMPLATE.format(text=custom))
        return lines
