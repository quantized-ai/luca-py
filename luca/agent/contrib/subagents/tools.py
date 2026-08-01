"""The two subagent tools and the payload that links them.

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
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    AssistantMessage,
    CancellationToken,
    ExecutionResult,
    TextContent,
    ToolKind,
    pretty_print,
)

SPAWN_TOOL_NAME = "spawn_subagent"
RESULT_TOOL_NAME = "create_conversation_result"


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
        payload = SubagentSpawn(
            is_subagent_spawn=True,
            task_id=args.get("task_id") or uuid.uuid4().hex[:8],
            prompt=args["prompt"],
            description=description,
            process_subagent_result_tool_name=self.result_tool_name,
        )
        return ExecutionResult(
            content=[TextContent(text=f"Spawned subagent: {description}")],
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

        # The subagent's own last words are its answer. Its LAST NODE is the
        # closing turn marker, so this looks for the last assistant message on
        # the path rather than the last entry — and takes only text parts,
        # since reasoning and tool calls are the child's working, not its
        # answer. Anything else — cancelled, errored, out of steps — has no
        # final message to take, so hand the parent a readable transcript
        # instead of nothing.
        for entry in reversed(nodes):
            if not isinstance(entry, AssistantMessage):
                continue
            parts = [part for part in entry.parts if isinstance(part, TextContent)]
            if parts:
                return ExecutionResult(content=list(parts))
            break
        return ExecutionResult(
            content=[
                TextContent(
                    text=(
                        f"The subagent for {args['description']!r} finished without a final "
                        f"message. Transcript:\n\n{pretty_print(session, child_id)}"
                    ),
                ),
            ],
            is_error=True,
        )
