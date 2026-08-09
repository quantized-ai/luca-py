# `luca.agent.contrib.questions` — ask the user, park, read the answers

The model calls `ask_user` with up to four questions and **waits**. This is the
first deferred tool ([`tools/README.md`](../tools/README.md) §10): `execute()`
returns `ExecutionDeferred()` while the answers are missing, the runner parks
the open turn at `AWAITING_RESULT`, the driver renders the questions and
collects answers out of band, and the next drive re-dispatches the *same* call —
which now finds an answer and returns its one `ExecutionResult`.

```python
from luca.agent.contrib.questions import (
    QuestionsPlugin, QuestionsTool,      # the plugin and the tool
    Question, OptionsType,               # what the model authors
)
```

Two surfaces, and the split is the whole design:

| Side | Members | Who calls it |
|---|---|---|
| framework-facing | `execute()` | the runner, through the registry's prepared callable |
| app-facing | `pending()` / `is_answered()` / `answer()` | your driver — plain Python the framework has never heard of |

## 1. Wiring, in full

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.questions import QuestionsPlugin

plugin = QuestionsPlugin(store=session.extras.setdefault("questions", {}))
runner = PluginAgentSessionRunner(session, plugins=[plugin])
tool = plugin.tool                       # the stable reference you answer on

while not runner.idle():
    async with runner.run() as run:
        async for event in run:
            render(event)

    for execution in runner.pending_deferred_tool_executions():
        payload = await ask_the_human(tool.pending(execution.tool_call_id))
        tool.answer(execution.tool_call_id, payload)
```

The loop drives the **same open turn** again; `pending_deferred_tool_executions()`
is a plain read over the session ([`04-runner.md`](../../04-runner.md) §9), so it
also answers for a parked *subagent* while its siblings keep working.

> ⚠️ **The handler must block until it has made progress.** There is no
> framework backoff and deliberately none: a handler that returns without
> resolving anything causes an immediate re-drive, another deferral, and a
> spin.

## 2. What the model authors

```python
class OptionsType(str, Enum):
    SINGLE_SELECT = "single_select"      # radio
    MULTIPLE_SELECT = "multiple_select"  # checkbox

class Question(BaseModel):
    title: str                           # the question itself
    body: str | None = None              # optional elaboration / why it is asked
    options_type: OptionsType
    options: list[str] = Field(min_length=2)

class QuestionsTool(Tool):
    name = "ask_user"
    title = "Ask the user"
    namespace = "contrib.questions"

    class Args(BaseModel):
        questions: list[Question] = Field(min_length=1, max_length=4)
```

Every bound is load-bearing:

| Bound | Why |
|---|---|
| `questions` `min_length=1` | an empty call is not a question |
| `questions` `max_length=4` | a **product** call, not a technical one: the answering contract is all-at-once, so a ten-question call is a wall the user must clear in one sitting. Claude Code caps at four |
| `options` `min_length=2` | a one-option question is not a question. A question with no good options is one the model should ask in prose — the tool deliberately has no free-text-only mode |
| `options` is `list[str]` | a one-field `Option` wrapper costs a level of JSON-Schema nesting the model must fill in and buys nothing. Promote it only when a per-option `description` is really wanted |

`timeout_in_ms` is deliberately unset. Each dispatch is an instant dict read,
and the parked period is unbounded by design — no per-poll deadline touches it.

The `description` is the model's entire instruction set, and two paragraphs of
it are the framework's only chance to stop the model minting a **second** call
while one is parked (a re-call is a new `ToolExecution` under a new
`tool_call_id`, leaving the tool holding two open jobs for one decision):

```
Ask everything you need in ONE call: the user answers every question in a
single pass, and a second call while the first is outstanding shows them a
second prompt.

Each question offers the user your options plus "type your own answer" and
"let's chat about this". A user who picks "chat about this" is declining the
question and asking to discuss it — answer them in the conversation, do not
re-ask.
```

## 3. The store

A constructor argument, and that is the whole persistence story —
`MemoryPlugin`'s verbatim:

```python
QuestionsTool()                 # works, forgets everything on exit
QuestionsTool(store=mine)       # works, and `mine` IS the memory
```

The tool mutates the dict it was handed, in place. It never reads or writes a
file and does not know what a session save is.

**JSON-shaped all the way down**, non-negotiable, because the point is that the
application can serialize it:

```python
store = {
    "<tool_call_id>": {
        "questions": [ {...}, ... ],   # verbatim from args["questions"]
        "answer": {...} | None,        # verbatim from answer(), None while pending
    },
}
```

No Pydantic instances, no dataclasses, no `datetime` —
`store == json.loads(json.dumps(store))` round-trips. `pending()` re-validates
the stored dicts into `Question` objects on the way out, so a caller gets typed
objects without the store holding any.

**Keyed by `tool_call_id`**, and that matters twice. Several calls in one
assistant message, and several subagents each with their own, can be outstanding
at once, so the key has to be per-*call*. A `tool_call_id` is globally unique,
which buys two things a conversation-keyed store does not get: **no lock** (no
two conversations can collide on a key) and **no compaction re-keying** (a
compaction installs a new conversation id; a `tool_call_id` never moves).

**It accumulates, deliberately.** Nothing is ever removed — the store is the
session's *question record*, not a working set that needs sweeping. Two
consequences, both wanted: `answer is None` iff the call is still pending (one
field is the whole status, readable straight off a restored dict without
consulting the session), and a completed entry is inert (its execution is
terminal, so the runner never re-dispatches it and the tool never reads it
again). It is also why nothing here needs a cleanup hook.

Hand in `session.extras` and persistence is free — the dict rides along on every
save ([`02-data-model.md`](../../02-data-model.md) §7):

```python
plugin = QuestionsPlugin(store=session.extras.setdefault("questions", {}))
```

A question parked when the process died comes back parked: the execution is
still `AWAITING_RESULT`, the next drive re-dispatches, `execute()` finds the
restored job, and it defers again — same prompt, nothing lost, nothing re-asked
by the model. **Without a restored store the same path still works**: `execute()`
re-seeds from `raw_tool_call.arguments` and defers, so the user is simply asked a
second time. Persistence upgrades the experience; correctness comes from
`execute()` being a pure predicate, not from the store.

## 4. `execute()` — a pure predicate

```python
job = self.store.setdefault(tool_call_id, {"questions": args["questions"], "answer": None})
if job.get("answer") is None:
    return ExecutionDeferred()
return ExecutionResult(
    content=[TextContent(text=self.generate_content_string_from_answers(questions, job["answer"]))],
    structured_content=job,
    is_error=False,
)
```

- **It never waits.** No future, no event, no UI: it reads state and answers
  *ready / not yet*. `setdefault` makes the first dispatch the seeding one and
  every later one a read, which is what makes it safe to invoke once per drive
  forever.
- **`is_error=False`, always.** A refusal is an *answer*, not a failure — and an
  error result is exactly what makes a model retry.
- **`structured_content` carries the payload verbatim**, including whatever the
  rendering ignores. `content` is the only channel the model sees; the payload
  is for your app ([`tools/README.md`](../tools/README.md) §6).

## 5. The app-facing side

```python
tool.pending(tool_call_id)      -> list[Question]   # [] for an id never seen
tool.is_answered(tool_call_id)  -> bool
tool.answer(tool_call_id, payload) -> None
```

`tool_call_id` is positional because it is *identity*, not data — the same split
the runner uses everywhere. Your driver already holds the parked `ToolExecution`,
so `execution.tool_call_id` is the handle. (It could equally read
`execution.raw_tool_call.arguments`; `pending()` exists so a driver does not have
to re-parse the model's raw arguments.)

The payload, in full:

```python
tool.answer("call_abc", {
    "answers": [
        {"question": "What year is this?",
         "chat_about_this": False, "answers": ["2026"], "custom_answer": None},
        {"question": "What's your favorite fruit?",
         "chat_about_this": False, "answers": ["Banana", "Apple"],
         "custom_answer": "I also like peaches"},
        {"question": "What's your favorite planet?",
         "chat_about_this": True, "answers": [], "custom_answer": None},
    ],
    "custom_notes": None,
})
```

| Key | Meaning |
|---|---|
| `answers` (outer) | one entry per question, in order. Nothing enforces that |
| `answers` (inner) | the selected options. Empty means the question was skipped |
| `custom_answer` | what the user typed under "type your own answer", or `None` |
| `chat_about_this` | the user declined *this* question and wants to discuss it |
| `custom_notes` | free text about the whole set, not about one question |

> ⚠️ **`answer()` never raises, and the reason is structural.** 0007's driver
> contract requires the handler to block until it has made progress, and there
> is no framework backoff — a `raise` here aborts the handler before it resolves
> anything, causing an immediate re-drive, another deferral, and a spin. **A
> raising validator turns a cosmetic mismatch into a hung UI**, which is
> strictly worse than the mismatch. Everything downstream is text on its way to
> becoming a string the model reads, so nothing needs the option strings to
> match what was declared, the list to be complete, or the titles to be spelled
> the same way.

Concretely: an unknown `tool_call_id` **creates** the slot rather than failing
(an answer arriving before the first `execute()` still works, and a wrong id is
inert instead of fatal); every read of the payload is defensive, so a missing or
misspelled key renders as absent; calling it twice is last-write-wins, and once
the execution is COMPLETED the runner never re-dispatches, so a late second call
does nothing.

**"Chat about this" ends the collection, not the other answers.** It makes the
UI stop collecting and submit immediately with whatever state the other
questions had; answers already given are kept and rendered, and at most one
question can carry the flag because submission happens the moment it is set. The
call still completes normally — `COMPLETED`, `is_error=False`. Nothing is
cancelled: this is a result the model *reads*, not an interruption.

## 6. Rendering — the override point

```python
def generate_content_string_from_answers(
    self, questions: list[Question], payload: dict,
) -> str: ...
```

Public, instance-level, overridable: subclass `QuestionsTool`, replace exactly
this method, and the wording the model reads changes without touching state,
validation or lifecycle. It takes **both** the questions and the payload,
because rendering needs the question order and the titles of questions the
payload omits. Wording lives in `ClassVar` templates (`ANSWERED_PREAMBLE`,
`DECLINED_TEMPLATE`, `NO_ANSWER`, …), the same pattern `ConversationProjector`
uses, so a subclass that wants one different sentence does not reimplement the
loop.

All questions answered:

```
User answered all your questions:

What year is this?
Answer:
- 2026

What's your favorite fruit?
Answer:
- Banana
- Apple
Custom note: "I also like peaches"
```

One question declined:

```
User declined to answer some questions.
User wants to chat more about: "What's your favorite planet?"
Respond to them in the conversation — do not ask this question again.

Other questions/answers recorded:

What year is this?
Answer:
- 2026

What's your favorite fruit?
Answer: No answer
```

That third line is the point of the declined branch: without it the model reads
"the user wants to chat", has no instruction, and its most likely repair is to
re-ask — which mints a second call.

Pairing is the one thing that cannot be "just text", and every step of it falls
back rather than failing: **match the payload entry whose `question` equals the
stored title, else by position, else render it standalone.** A stored question
with no entry either way renders `Answer: No answer`; a payload entry matching no
stored question is still rendered under its own `question` string. Three
fallbacks deep and none of them fail — a driver that sends the full set in order
always hits the first rule.

> ⚠️ **A raise here costs a turn.** This runs inside `execute()`, so an
> exception is a `FAILED` tool call — the one place in this tool where being
> strict is expensive. Read the payload defensively, as the shipped
> implementation does.

## 7. The plugin

```python
class QuestionsPlugin:
    def __init__(self, store: dict | None = None,
                 tool_class: type[QuestionsTool] = QuestionsTool) -> None:
        self.store = {} if store is None else store
        self.tool = tool_class(store=self.store)

    def get_tool_registry(self, agent_session) -> SimpleToolRegistry: ...
```

Three things it pins:

- **One tool instance, held on the plugin.** `MemoryPlugin` builds fresh tools
  per `get_tools()` because its tools are stateless over a shared store; here the
  application needs a stable reference to call `answer()` on, so
  `plugin.tool` is it. (The store is shared either way — ergonomics, not
  correctness.)
- **`tool_class=` is how a subclass gets in** — the whole point of an overridable
  renderer is that your UI replaces it, and this is how that instance reaches
  the registry.
- **`YoloPermissionPolicy`.** Asking a question is not an action to approve; a
  gate in front of it would mean the user approving *being asked* something.
  Compose your own registry over `plugin.tool` if you want one.

There is deliberately **no `get_system_prompt_parts`**: the tool's `description`
already carries every instruction the model needs, and a prompt part would
repeat it in a second place that can drift. The hook is duck-typed, so subclass
and add one if "prefer asking over guessing" should be global policy.

## 8. Lifecycle

```text
drive #1
  model  → tool_call(ask_user, questions=[…])           tool_call_id=tc1
  runner → decide() → ALLOWED → prepare() → execute()
  tool   → seeds store["tc1"], returns ExecutionDeferred()
  runner → status=AWAITING_RESULT, attempts=[DEFERRED], parks, returns
  conv   → BLOCKED

driver
  runner.pending_deferred_tool_executions() → [tc1]
  tool.pending("tc1") → the questions
  …renders, blocks on the human…
  tool.answer("tc1", payload)

drive #2                                    (the SAME turn, re-entered)
  runner → re-dispatch tc1 from scratch: prepare() → execute()
  tool   → answers present, returns ExecutionResult
  runner → COMPLETED, attempts=[DEFERRED, COMPLETED], ToolExecuted fires
  model  → reads the rendered answers, keeps working
```

**One call, one `tool_call_id`, one `ToolExecution`, one approval, two
attempts.** Approval is asked once per *call*, never once per dispatch.

## 9. Edge cases

| Situation | Behavior |
|---|---|
| several `ask_user` calls in one assistant message | independent jobs; each parks, `pending_deferred_tool_executions()` returns all |
| a subagent asks while its siblings work | main conversation stays `BUSY`; the parked call is still returned by the subtree read. Resolve it from another task and call `run.notify(execution)` |
| `cancel()` while parked | the core marks the execution `INTERRUPTED`. **The tool is never told** — your UI must drop its own prompt |
| resume with a restored store | still `AWAITING_RESULT`; the next drive re-dispatches, `execute()` finds the restored job and defers, and the same prompt is re-rendered (§3) |
| resume with no store | identical path — `execute()` re-seeds from `raw_tool_call.arguments` and defers, so the user is asked a second time. Nothing wedges either way |
| crash mid-poll | the execution is recovered to `INTERRUPTED` and never re-dispatched; the model reads `[tool execution interrupted]` and may call again under a new id |
| a malformed payload | rendered as best it can be, never rejected (§5). The verbatim payload still reaches `structured_content`, so the mistake is visible without being fatal |
| a dispatch raises | terminal `FAILED`, and the questions go with it. Nothing in this body should be able to raise — a dict read and defensive rendering — so this means a bug, not a condition to handle |
| the model calls `ask_user` again while one is parked | a second job under a second `tool_call_id`, and a second prompt. Discouraged in the description (§2), not prevented |
| `timeout_in_ms` | unset. Each dispatch is an instant dict read; the parked period is unbounded by design |

## 10. Surface

| Name | What |
|---|---|
| `QuestionsPlugin` | `get_tool_registry` + `get_tools`, owns `store` and one `tool` |
| `QuestionsTool` | the `ask_user` tool: `execute` / `pending` / `is_answered` / `answer` / `generate_content_string_from_answers` |
| `Question` | `title`, `body`, `options_type`, `options` |
| `OptionsType` | `SINGLE_SELECT` / `MULTIPLE_SELECT` |
| `DESCRIPTION` | the model-facing instruction text (§2) |
| `QUESTIONS_NAMESPACE` | `"contrib.questions"` |

The TUI's rendering of all of this is in
[`tui/README.md` §2.2](../tui/README.md#22-the-question-set--the-fourth-dock),
and every state of it is browsable without a model or a key:
`uv run python main.py --gallery dock/questions`.

Next: [`plugins/README.md`](../plugins/README.md).
