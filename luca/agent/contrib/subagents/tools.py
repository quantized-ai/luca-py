"""The subagent tools and the payloads that link them to the runner.

THE CONTRACT between contrib and the runner is the tool's STRUCTURED OUTPUT:
declared on the spec via `output_schema`, populated on the result via
`structured_content`. The runner reads that declaration twice, and the two
readings must never disagree about what a spawn is — which is why there is one
declaration rather than two rules to remember:

| | what is read | when | what it means |
|---|---|---|---|
| **Gate** | `ToolSpec.output_schema` declares `is_subagent_spawn` | before the model call, from the spec alone | this tool **can** spawn → it is subject to the depth cap |
| **Handshake** | `structured_content["is_subagent_spawn"] is True` | after the execution completes | this call **did** spawn → create the child |

The asymmetry is deliberate and is what the flag buys: `is_subagent_spawn` is a
`bool`, not a constant. A spawn tool that decides at runtime NOT to spawn — a
rejected task, a failed validation, a request it chose to answer inline —
returns the payload with `is_subagent_spawn=False` and no child is created. It
is still gated, because the gate reads the declaration, not the outcome.

A DEVELOPER CAN SHIP THEIR OWN. Declare `is_subagent_spawn` in your own output
schema and return the payload, and your `delegate_work` tool spawns and gates
correctly — because the runner matches the declaration, never a tool name. A
name match would not have survived that: `delegate_work` would spawn through
the handshake and never be filtered, so a subagent would spawn subagents and
the depth cap would quietly not exist.

THE PAYLOAD IS FREE. `structured_content` never reaches the model and is never
counted toward context, so the handshake — prompt, description, task id, the
result tool's name — costs the parent conversation nothing. The model sees only
the short `content` status line.

THE STOP CONTRACT is the same mechanism, value-side only: a completed
execution whose `structured_content` carries `is_subagent_stop=True` asks the
runner to cancel the direct child whose spawn payload matches `task_id`. There
is no gate half — stopping bypasses no cap — so the spec's `output_schema`
declaration is documentation for the application, not something the runner
reads. `is_subagent_stop=False` (an invalid or already-settled target) asks
for nothing, exactly like a declining spawn.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    AssistantMessage,
    CancellationToken,
    ChildConversation,
    ConversationStatus,
    ExecutionResult,
    TextContent,
    ToolExecution,
    ToolKind,
    TurnFinish,
    TurnOutcome,
    open_turn_index,
    pretty_print,
    spawn_payload,
)

SPAWN_TOOL_NAME = "spawn_subagent"
RESULT_TOOL_NAME = "create_conversation_result"
STOP_TOOL_NAME = "stop_subagent"
LIST_TOOL_NAME = "list_subagents"


def open_turn_children(
    session: AgentSession,
    conversation_id: str,
) -> list[ChildConversation]:
    """Every `ChildConversation` in this conversation's open turn, resolved or
    not, in path order — the roster `list_subagents` renders, `stop_subagent`
    validates against, and the registry keys the two tools' availability on."""
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return []
    index = open_turn_index(conversation.nodes, session.entries)
    if index is None:
        return []
    return [
        entry
        for node_id in conversation.nodes[index:]
        if isinstance(entry := session.entries.get(node_id), ChildConversation)
    ]


def _task_id_of(session: AgentSession, link: ChildConversation) -> str | None:
    spawn = session.entries.get(link.tool_execution_id)
    if not isinstance(spawn, ToolExecution):
        return None
    payload = spawn_payload(spawn)
    return payload.get("task_id") if payload is not None else None


def _task_status(session: AgentSession, link: ChildConversation) -> str:
    """The model-facing status word for one task: `completed` / `failed` once
    resolved (the result's own verdict — the SAME split the projected task
    update uses, so the two surfaces never disagree about a task),
    `cancelling` while a stop is being consumed, `pending` otherwise."""
    if link.execution_result is not None:
        return "failed" if link.execution_result.is_error else "completed"
    if link.conversation_id in session.conversations:
        status = session.get_conversation_status(link.conversation_id).status
        if status is ConversationStatus.CANCELLING:
            return "cancelling"
    return "pending"


class SubagentSpawn(BaseModel):
    """The shape of a spawn tool's result payload — declared on the spec via
    `output_schema`, returned on the result via `structured_content`.

    This is the PLUGIN's guarantee about what it emits. The core cannot import
    it (core never imports contrib) and does not need to: it reads the five
    field names as a convention, and the published schema is what makes them
    discoverable before the call rather than what makes them typed."""

    model_config = ConfigDict(extra="forbid")

    is_subagent_spawn: bool = True
    task_id: str
    prompt: str
    description: str
    process_subagent_result_tool_name: str


class SpawnSubagent(Tool):
    """Start one subagent on an independent task.

    The tool's whole scope is to SPAWN, and it completes as soon as the child
    conversation exists — it does not stay open until the subagent finishes.
    That is what keeps background subagents possible later without changing
    this foundation: a result arriving after the parent's turn has closed
    cannot be represented by a tool call that is still open, because a
    non-terminal execution blocks the parent's projection by design.

    The consequence, accepted: this call's model-facing output is a status
    line, and the subagent's actual answer reaches the model separately, from
    the `ChildConversation` entry."""

    namespace = "contrib.subagents"
    name = SPAWN_TOOL_NAME
    description = (
        "Spawn a subagent to work on an independent task in parallel. "
        "Use one call per task; several subagents run at the same time. "
        "The subagent reports back when it finishes — you do not need to poll."
    )
    tool_kind = ToolKind.OTHER
    output_schema = SubagentSpawn

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        prompt: str = Field(description="The full instructions for the subagent. It sees nothing else.")
        description: str = Field(description="A short description of the task, for the user to read.")
        task_id: str | None = Field(
            default=None,
            description="Make up a unique id for the task, or leave it out and one will be made up for you.",
        )

    def __init__(self, result_tool_name: str = RESULT_TOOL_NAME) -> None:
        # Which tool derives the child's result travels IN THE PAYLOAD rather
        # than being known to the core, so an application can pair its own
        # spawn tool with its own result tool without touching the runner.
        self.result_tool_name = result_tool_name

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        description = args["description"]
        task_id = args.get("task_id") or uuid.uuid4().hex[:8]
        payload = SubagentSpawn(
            is_subagent_spawn=True,
            task_id=task_id,
            prompt=args["prompt"],
            description=description,
            process_subagent_result_tool_name=self.result_tool_name,
        )
        return ExecutionResult(
            content=[TextContent(text=f"Spawned subagent with id {task_id}: {description}")],
            structured_content=payload.model_dump(),
        )


class CreateConversationResult(Tool):
    """Derive the result of a finished subagent conversation.

    PRIVATE: the runtime invokes it, the model never sees it, and a model that
    guesses the name gets NOT_FOUND. It therefore needs no description arguing
    with the model about not calling it.

    It declares no `output_schema` — what it produces is model-facing prose,
    not a machine payload — and that is also what keeps it out of the spawn
    gate: a tool that declares nothing is not a spawn tool."""

    namespace = "contrib.subagents"
    name = RESULT_TOOL_NAME
    description = "Derive the result of a finished subagent conversation."
    is_private = True

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str
        prompt: str
        description: str
        conversation_id: str

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        # NOTE the two ids. The PARAMETER is the parent — where this tool runs.
        # The ARGUMENT is the child being summarized. They are never the same
        # conversation; do not collapse them.
        child_id = args["conversation_id"]
        child = session.conversations[child_id]
        nodes = [session.entries[node_id] for node_id in child.nodes if node_id in session.entries]

        # How the child's last turn CLOSED colors what its last words mean: a
        # stopped or failed subagent's final message is a progress report, not
        # an answer, and presenting it as one would tell the parent a task
        # finished that did not.
        outcome = next(
            (entry.outcome for entry in reversed(nodes) if isinstance(entry, TurnFinish)),
            None,
        )

        # The subagent's own last words are its answer. Its LAST NODE is the
        # closing turn marker, so this looks for the last assistant message on
        # the path rather than the last entry — and takes only text parts,
        # since reasoning and tool calls are the child's working, not its
        # answer. Anything else — cancelled, errored, out of steps — has no
        # final message to take, so hand the parent a readable transcript
        # instead of nothing.
        ended = (
            outcome.value.replace("_", " ") if outcome is not None and outcome is not TurnOutcome.COMPLETED else None
        )
        for entry in reversed(nodes):
            if not isinstance(entry, AssistantMessage):
                continue
            parts = [part for part in entry.parts if isinstance(part, TextContent)]
            if parts:
                if ended is not None:
                    text = "".join(part.text for part in parts)
                    return ExecutionResult(
                        content=[
                            TextContent(
                                text=f"The subagent was {ended} before finishing. Its last message:\n\n{text}",
                            ),
                        ],
                        is_error=True,
                    )
                return ExecutionResult(content=list(parts))
            break
        opening = (
            f"The subagent was {ended} before finishing and left no final message."
            if ended is not None
            else f"The subagent for {args['description']!r} finished without a final message."
        )
        return ExecutionResult(
            content=[
                TextContent(
                    text=f"{opening} Transcript:\n\n{pretty_print(session, child_id)}",
                ),
            ],
            is_error=True,
        )


class SubagentStop(BaseModel):
    """The shape of the stop tool's result payload — declared on the spec via
    `output_schema`, returned on the result via `structured_content`.

    `is_subagent_stop` is the did-it-happen flag, exactly like the spawn
    payload's: a call whose target is unknown or already settled returns the
    payload with `False` and the runner does nothing. The declared schema is
    the plugin's guarantee about what it emits, so the payload is returned in
    BOTH cases, never omitted."""

    model_config = ConfigDict(extra="forbid")

    is_subagent_stop: bool = True
    task_id: str
    reason: str | None = None


class StopSubagent(Tool):
    """Ask the runner to stop one of this conversation's running subagents.

    The tool's whole scope is to SIGNAL: it validates the target against the
    live session and returns the payload; the runner consumes it and issues
    the actual cancellation (best effort — the child winds down through the
    ordinary cancel machinery and its link still resolves, with a result that
    says it was stopped). The runner re-validates on its side too: a child can
    finish between this body and the handler, and only the runner's read is
    authoritative."""

    namespace = "contrib.subagents"
    name = STOP_TOOL_NAME
    description = (
        "Stop one of the subagents you spawned this turn. Best effort: the "
        "subagent is cancelled and reports back one final time as stopped. "
        "Use list_subagents if you are unsure of the task id."
    )
    tool_kind = ToolKind.OTHER
    output_schema = SubagentStop

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str = Field(description="The task id to stop. Has to match perfectly.")
        reason: str | None = Field(
            default=None,
            description="A short description of the reason to stop the task.",
        )

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        task_id = args["task_id"]
        reason = args.get("reason")
        by_status: dict[str, str] = {}
        for link in open_turn_children(session, conversation_id):
            if _task_id_of(session, link) == task_id:
                by_status.setdefault(_task_status(session, link), "seen")
        if "pending" in by_status:
            payload = SubagentStop(is_subagent_stop=True, task_id=task_id, reason=reason)
            acknowledged = f"Stop signal received for task {task_id}."
            if reason:
                acknowledged += f" Reason: {reason}"
            return ExecutionResult(
                content=[TextContent(text=acknowledged)],
                structured_content=payload.model_dump(),
            )
        if "cancelling" in by_status:
            verdict = f"Task {task_id} is already being stopped."
        elif "completed" in by_status or "failed" in by_status:
            verdict = f"Task {task_id} has already finished; there is nothing to stop."
        else:
            known = [
                tid for link in open_turn_children(session, conversation_id) if (tid := _task_id_of(session, link))
            ]
            listing = ", ".join(known) if known else "none"
            verdict = f"No task {task_id} in this turn. Known task ids: {listing}."
        payload = SubagentStop(is_subagent_stop=False, task_id=task_id, reason=reason)
        return ExecutionResult(
            content=[TextContent(text=verdict)],
            structured_content=payload.model_dump(),
            is_error=True,
        )


class ListSubagents(Tool):
    """List this turn's subagents — the roster the model steers with.

    Read-only, and deliberately schema-less: what it produces is model-facing
    prose, not a machine payload, and declaring nothing is also what keeps it
    out of the spawn gate."""

    namespace = "contrib.subagents"
    name = LIST_TOOL_NAME
    description = (
        "List the subagents spawned in the current turn, with each task's id, "
        "status (pending / cancelling / completed / failed), description and prompt."
    )
    tool_kind = ToolKind.READ

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        links = open_turn_children(session, conversation_id)
        if not links:
            return ExecutionResult(
                content=[TextContent(text="No subagents have been spawned in the current turn.")],
            )
        blocks = []
        for link in links:
            spawn = session.entries.get(link.tool_execution_id)
            payload = spawn_payload(spawn) if isinstance(spawn, ToolExecution) else None
            if payload is None:
                continue
            blocks.append(
                f"<task id={payload['task_id']} status={_task_status(session, link)}>\n"
                f"description: {payload['description']}\n"
                f"prompt: {payload['prompt']}\n"
                f"</task>"
            )
        return ExecutionResult(content=[TextContent(text="\n".join(blocks))])
