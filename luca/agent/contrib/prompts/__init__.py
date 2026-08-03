"""System prompts — the agent's persona, tuned to the model, plus the project's
own instruction files.

`SystemPromptPlugin` picks a base prompt for the model family and adds an
environment block; `InstructionsPlugin` reads `LUCA.md` / `AGENTS.md` /
`CLAUDE.md`. No extra dependencies.
"""

from .environment import ENVIRONMENT_TEMPLATE, environment_text
from .instructions import (
    INSTRUCTION_FILES,
    MAX_INSTRUCTION_BYTES,
    InstructionFile,
    apply_budget,
    config_directory,
    find_instructions,
    first_in_directory,
    project_directories,
    read_instruction_file,
)
from .plugin import (
    ENVIRONMENT_PRIORITY,
    INSTRUCTIONS_PRIORITY,
    InstructionsPlugin,
    SystemPromptPlugin,
    format_instructions,
)
from .selection import BASE, FAMILIES, GENERIC, load_prompt, select_family

__all__ = [
    "BASE",
    "ENVIRONMENT_PRIORITY",
    "ENVIRONMENT_TEMPLATE",
    "FAMILIES",
    "GENERIC",
    "INSTRUCTIONS_PRIORITY",
    "INSTRUCTION_FILES",
    "MAX_INSTRUCTION_BYTES",
    "InstructionFile",
    "InstructionsPlugin",
    "SystemPromptPlugin",
    "apply_budget",
    "config_directory",
    "environment_text",
    "find_instructions",
    "first_in_directory",
    "format_instructions",
    "load_prompt",
    "project_directories",
    "read_instruction_file",
    "select_family",
]
