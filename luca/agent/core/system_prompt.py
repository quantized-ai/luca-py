"""System-prompt machinery for the agent loop.

The runner is constructed with `system_prompt_parts` — a list whose items are
static parts or callables producing one, resolved fresh before every LLM call:

- a `SystemPromptPart`;
- a `str` (becomes `SystemPromptPart(text=...)`);
- a dict with `text` and optional `priority` / `source` keys;
- a callable `(session_config, runtime_status) -> ` any of the above, invoked
  per call so its part can reflect the current session state and runtime
  status (step count, turn count, etc.).

`coerce_system_prompt_part` normalizes the three static forms to a
`SystemPromptPart`. An **assembler** flattens the priority-sorted parts into
the final string — a duck-typed concrete base (no ABC / Protocol): subclass
and override the one hook. The default joins part texts with a newline, so a
runner constructed with no parts sends no system message at all (the runner
omits a blank prompt from the request).
"""

from __future__ import annotations

from collections.abc import Callable

from .models import SessionConfig, SessionRuntimeStatus, SystemPromptPart

SystemPromptPartLike = SystemPromptPart | str | dict
SystemPromptPartInput = (
    SystemPromptPartLike
    | Callable[[SessionConfig, SessionRuntimeStatus], SystemPromptPartLike]
)


def coerce_system_prompt_part(value: SystemPromptPartLike) -> SystemPromptPart:
    """Normalize one static part form to a `SystemPromptPart`. Dicts validate
    strictly (`text` plus optional `priority` / `source`); any other type
    raises `TypeError`."""
    if isinstance(value, SystemPromptPart):
        return value
    if isinstance(value, str):
        return SystemPromptPart(text=value)
    if isinstance(value, dict):
        return SystemPromptPart.model_validate(value)
    raise TypeError(
        "system prompt part must be a SystemPromptPart, str, or dict "
        f"(got {type(value).__name__})"
    )


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
