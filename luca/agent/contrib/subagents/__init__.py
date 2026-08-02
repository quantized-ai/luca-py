"""`luca.agent.contrib.subagents` — spawning subagents that work in parallel.

The LLM asks for a subagent by calling a tool, so the whole tool-execution flow
happens naturally and the core needs nothing subagent-specific to run it. What
the core DOES need is a way to recognise a spawn, and that arrives as the tool's
STRUCTURED OUTPUT: the spec declares `is_subagent_spawn` in its `output_schema`,
the result carries the payload in `structured_content`, and the runner reads the
declaration before the call (the depth gate) and the payload after it (create
the child).

    from luca.agent.contrib.plugins import PluginAgentSessionRunner
    from luca.agent.contrib.subagents import SubagentsPlugin
    from luca.agent.core import AgentSessionRunner, LLMConfig, RuntimeConfig

    session = AgentSessionRunner.new_session(
        LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
        runtime_config=RuntimeConfig(subagents_enabled=True),
    )
    runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin()])

Installing the plugin is not enough on its own: `subagents_enabled` defaults to
False, so the session has to ask for the capability too.

Everything else is core's: one conversation per subagent in the same session,
`ChildConversation` linking it into the parent's path, and
`run.children` / `run.child(cid)` for control.
"""

from .plugin import (
    CONTROL_PROMPT,
    SPAWNING_PROMPT,
    SubagentsPlugin,
    SubagentToolRegistry,
    control_prompt_part,
    spawn_gate_open,
    spawning_prompt_part,
)
from .tools import (
    LIST_TOOL_NAME,
    RESULT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    STOP_TOOL_NAME,
    CreateConversationResult,
    ListSubagents,
    SpawnSubagent,
    StopSubagent,
    SubagentSpawn,
    SubagentStop,
    open_turn_children,
)

__all__ = [
    "CONTROL_PROMPT",
    "LIST_TOOL_NAME",
    "RESULT_TOOL_NAME",
    "SPAWNING_PROMPT",
    "SPAWN_TOOL_NAME",
    "STOP_TOOL_NAME",
    "CreateConversationResult",
    "ListSubagents",
    "SpawnSubagent",
    "StopSubagent",
    "SubagentSpawn",
    "SubagentStop",
    "SubagentToolRegistry",
    "SubagentsPlugin",
    "control_prompt_part",
    "open_turn_children",
    "spawn_gate_open",
    "spawning_prompt_part",
]
