"""`SubagentsPlugin` — the subagent tools and the prompts that teach them.

Four tools (spawn / private result / stop / list), two hooks, no middleware.
It is contrib like every other capability bundle, it consumes only core's
public surface, and the core knows nothing about it.

AUTO-APPROVED, AND THAT IS THE RIGHT CALL — not a shortcut. The tools ship in
their own always-allowing registry, the pattern `MemoryPlugin` already uses.
Spawning is cheap to approve because the dangerous work is not in the spawn: it
is in the tools the CHILD calls, and those are gated inside the child by their
own registries. Prompting on the spawn would ask about the wrapper and never
the payload. Stopping only ever cancels work this conversation itself started,
and listing reads. `CreateConversationResult` is private and runtime-invoked,
so a prompt for it would be meaningless — but it still goes through
`decide()`, so it needs a policy that answers ALLOW.

Coexistence with `shell`, `memory` and the rest is free, because there is no
global permission gate anywhere: each registry answers `decide()` for its own
tools, and `PluginAgentSessionRunner` makes every plugin registry a child of one
`ProxyToolRegistry`. Shell keeps its `PermissionStrategy`, memory keeps Yolo,
subagents keeps Yolo. Nothing is shared and nothing negotiates.

TOOL NAMES ARE GLOBALLY UNIQUE — required, and enforced. `ProxyToolRegistry`
routes by `spec.name`; `namespace` does NOT disambiguate. The failure is loud
(the proxy raises on duplicate names), so this is a naming rule rather than a
risk to design around.
"""

from __future__ import annotations

from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    Inf,
    SystemPromptPart,
    ToolSpec,
    declares_spawn,
    spawns_committed,
)

from .tools import (
    LIST_TOOL_NAME,
    RESULT_TOOL_NAME,
    STOP_TOOL_NAME,
    CreateConversationResult,
    ListSubagents,
    SpawnSubagent,
    StopSubagent,
    open_turn_children,
)

SPAWNING_PROMPT = """
### Subagents (spawn_subagent)
You can spawn subagents to work on independent tasks in parallel.
Each subagent starts fresh: it sees only the prompt you give it, so make that prompt self-contained.
Spawn several in one response when the tasks are independent — they run at the same time.
Prefer subagents for wide, self-contained work (searching, reading a lot of files, exploring an approach) and keep the synthesis for yourself.
Subagents report back as each one finishes — you receive a task update per result and can act on it right away.
""".strip()

CONTROL_PROMPT = """
### Managing running subagents
Task updates arrive as each subagent finishes; a task you have not received an update for is still running.
Use list_subagents to check this turn's tasks, and stop_subagent with a task id to cancel one that is no longer needed.
""".strip()


def spawn_gate_open(
    session: AgentSession,
    conversation_id: str,
    *,
    exclude: str | None = None,
) -> bool:
    """Can THIS conversation spawn subagents?

    THE one predicate. The tool list and the system prompt both derive from it,
    and that identity is the whole reason the prompt part is a callable: a
    static part would tell a subagent at the depth cap that it can spawn while
    the tool list withholds the tool, and the model would try. The prompt and
    the tool list must never disagree, and deriving both from one function is
    the cheapest way to guarantee it.

    `exclude` mirrors the core's `_spawn_gate_open`: it drops one execution id
    from the per-turn count. No contrib caller needs it — only the runner
    converts settled spawns — but keeping the signatures identical is cheaper
    than explaining why they differ.
    """
    config = session.session_config.runtime_config
    if not config.subagents_enabled:
        return False
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return False
    if config.subagents_max_depth != Inf and conversation.depth >= config.subagents_max_depth:
        return False
    # the per-turn budget: Inf by default, so this clause never fires for a
    # session that did not ask for a limit
    limit = config.subagents_max_per_turn
    return limit == Inf or spawns_committed(session, conversation_id, exclude=exclude) < limit


def spawning_prompt_part(
    session: AgentSession,
    conversation_id: str,
) -> SystemPromptPart | None:
    """Tell the model it can spawn — but only where it actually can.

    Returns `None` (contribute nothing) when spawning is not possible, which is
    exactly when the gate withholds the tool."""
    if not spawn_gate_open(session, conversation_id):
        return None
    return SystemPromptPart(text=SPAWNING_PROMPT, source="subagents")


def control_prompt_part(
    session: AgentSession,
    conversation_id: str,
) -> SystemPromptPart | None:
    """Teach `stop_subagent` / `list_subagents` — but only while the open turn
    actually has subagents, which is exactly when the registry offers the two
    tools. A SEPARATE part from `spawning_prompt_part` because the two ride
    different predicates: a spent spawn budget silences the spawn teaching
    while this turn's tasks still need managing, and one part gated on either
    predicate would let the prompt and the tool list disagree."""
    if not open_turn_children(session, conversation_id):
        return None
    return SystemPromptPart(text=CONTROL_PROMPT, source="subagents")


class SubagentToolRegistry(SimpleToolRegistry):
    """`SimpleToolRegistry` plus THE GATE.

    The registry decides which tools a conversation is offered — the core makes
    no tool-list policy of its own — so the depth cap and the per-turn spawn
    budget are enforced here, by withholding any tool whose `output_schema`
    declares `is_subagent_spawn`. Reading the DECLARATION rather than the name
    is what makes a custom spawn tool gate correctly too.

    Only `get_tools` is overridden. Everything else is inherited: the runtime
    still resolves and dispatches whatever it is asked for, which is why the
    runner ALSO checks at child-creation time — the cap has to mean something
    at the moment a child would appear, not only where tools are advertised."""

    #: The task-management tools, withheld until the open turn has tasks to
    #: manage. Matched by NAME, not declaration — unlike the spawn gate this
    #: is presentation policy over this registry's OWN tools, not a cap the
    #: runner must be able to verify from a spec alone.
    CONTROL_TOOL_NAMES = frozenset({STOP_TOOL_NAME, LIST_TOOL_NAME})

    async def get_tools(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> list[ToolSpec]:
        specs = await super().get_tools(session, conversation_id)
        if not spawn_gate_open(session, conversation_id):
            specs = [spec for spec in specs if not declares_spawn(spec)]
        if not open_turn_children(session, conversation_id):
            # Nothing to stop or list before the first spawn of the turn; the
            # tools appear on the first wake round, alongside the prompt part
            # that teaches them (`control_prompt_part` — same predicate).
            specs = [spec for spec in specs if spec.name not in self.CONTROL_TOOL_NAMES]
        return specs


class SubagentsPlugin:
    """Bundles the subagent tools with the prompt part that teaches them. A
    plain class implementing only the plugin hooks it needs (no middleware).

    `subagents_enabled` still defaults to False on `RuntimeConfig`, so
    installing this plugin is not on its own enough to turn subagents on — the
    session has to ask for them too. That is deliberate: the capability is
    configuration, not installation."""

    def __init__(self, result_tool_name: str = RESULT_TOOL_NAME) -> None:
        self.result_tool_name = result_tool_name

    def get_tools(self) -> list:
        return [
            SpawnSubagent(result_tool_name=self.result_tool_name),
            CreateConversationResult(),
            StopSubagent(),
            ListSubagents(),
        ]

    def get_tool_registry(self, agent_session: AgentSession) -> SubagentToolRegistry:
        return SubagentToolRegistry(
            tools=self.get_tools(),
            permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        # CALLABLES, and they have to be — see `spawning_prompt_part` and
        # `control_prompt_part`: each is derived from the same predicate that
        # decides its tools' availability, per LLM call.
        return [spawning_prompt_part, control_prompt_part]
