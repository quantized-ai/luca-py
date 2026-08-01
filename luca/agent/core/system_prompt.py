"""System-prompt machinery for the agent loop.

The runner is constructed with `system_prompt_parts` — a list whose items are
static parts or callables producing one, resolved fresh before every LLM call:

- a `SystemPromptPart`;
- a `str` (becomes `SystemPromptPart(text=...)`);
- a dict with `text` and optional `priority` / `source` keys;
- a callable `(session, conversation_id) -> ` any of the above, **or `None`**,
  invoked per call so its part can reflect current session state and the
  conversation it is building a prompt for.

A callable receives the live `AgentSession` and the id of the conversation the
prompt is for, not a session-wide runtime status: a part that wants the step
count asks `session.get_conversation_status(conversation_id)` for the
conversation it is actually building a prompt for. That is what lets a part
say different things to the main agent and to a subagent — and it is what keeps
a prompt from disagreeing with the tool list, since both can be derived from
one predicate.

Returning `None` from a callable means "contribute nothing"; the part is
skipped entirely rather than assembling into a stray blank line.

`coerce_system_prompt_part` normalizes the static forms to a
`SystemPromptPart` (and passes `None` through). An **assembler** flattens the
priority-sorted parts into the final string — a duck-typed concrete base (no
ABC / Protocol): subclass and override the one hook. The default joins part
texts with a newline, so a runner constructed with no parts sends no system
message at all (the runner omits a blank prompt from the request).
"""

from __future__ import annotations

from collections.abc import Callable

from .models import AgentSession, SystemPromptPart

SystemPromptPartLike = SystemPromptPart | str | dict
SystemPromptPartInput = SystemPromptPartLike | Callable[[AgentSession, str], SystemPromptPartLike | None]


def coerce_system_prompt_part(
    value: SystemPromptPartLike | None,
) -> SystemPromptPart | None:
    """Normalize one static part form to a `SystemPromptPart`. Dicts validate
    strictly (`text` plus optional `priority` / `source`); `None` passes
    through as "no part" (only a callable can produce it — a static `None` in
    the constructor list is the same thing and equally harmless); any other
    type raises `TypeError`."""
    if value is None:
        return None
    if isinstance(value, SystemPromptPart):
        return value
    if isinstance(value, str):
        return SystemPromptPart(text=value)
    if isinstance(value, dict):
        return SystemPromptPart.model_validate(value)
    raise TypeError(f"system prompt part must be a SystemPromptPart, str, dict, or None (got {type(value).__name__})")


class SystemPromptAssembler:
    """Flattens the priority-sorted parts into the final system prompt. A
    blank result means "no system message" — the runner drops it."""

    def assemble_system_prompt(self, parts: list[SystemPromptPart]) -> str:
        raise NotImplementedError


class DefaultSystemPromptAssembler(SystemPromptAssembler):
    """Joins the part texts with a newline, in the order given (the runner
    sorts by priority before assembling)."""

    def assemble_system_prompt(self, parts: list[SystemPromptPart]) -> str:
        return "\n".join(part.text for part in parts)
