"""Built-in sub-agent types.

A `SubAgentType` is a read-only agent profile the `task` tool can spawn: a
system prompt and a step budget. The toolset is fixed (read, glob, grep) and
identical across types in v1 — there is no second capability yet to justify a
per-type toolset. Each type inherits the parent session's model unless the
`SubAgentManager` is built with an override.

`soft_max_steps` nudges the sub-agent to finalize (the model's tool calls are
dropped once it is reached, so the next step must be a text answer);
`hard_max_steps` is the backstop that ends the turn.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubAgentType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    soft_max_steps: int
    hard_max_steps: int


_EXPLORE_PROMPT = (
    "You are a read-only exploration sub-agent. Use the read, glob, and grep "
    "tools to locate and understand the code or files the caller asked about. "
    "You cannot edit files, write files, or run shell commands; you only read. "
    "Finish with a concise summary of what you found, citing concrete "
    "file:line references so the caller can act on it."
)

_GENERAL_PROMPT = (
    "You are a read-only research sub-agent answering one focused question. "
    "Investigate with the read, glob, and grep tools and return a direct, "
    "self-contained answer. You cannot modify anything; you only read."
)

BUILTIN_AGENT_TYPES: dict[str, SubAgentType] = {
    "explore": SubAgentType(
        system_prompt=_EXPLORE_PROMPT,
        soft_max_steps=12,
        hard_max_steps=16,
    ),
    "general": SubAgentType(
        system_prompt=_GENERAL_PROMPT,
        soft_max_steps=20,
        hard_max_steps=25,
    ),
}
