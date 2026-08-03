"""System prompts — the agent's persona, tuned to the model, plus the project's
own instruction files.

`SystemPromptPlugin` picks a base prompt for the model family and adds an
environment block; `InstructionsPlugin` reads `LUCA.md` / `AGENTS.md` /
`CLAUDE.md`. No extra dependencies.
"""

from .environment import ENVIRONMENT_TEMPLATE, format_environment
from .instructions import (
    INSTRUCTION_FILE_NAMES,
    MAX_INSTRUCTION_BYTES,
    InstructionFile,
    InstructionsError,
    apply_budget,
    find_instruction_file,
    find_instructions,
    find_project_directories,
    get_config_directory,
    read_instruction_file,
    read_named_instruction_file,
)
from .plugin import (
    ENVIRONMENT_PRIORITY,
    INSTRUCTIONS_PRIORITY,
    INSTRUCTIONS_PROMPT_HEADER,
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
    "INSTRUCTIONS_PROMPT_HEADER",
    "INSTRUCTION_FILE_NAMES",
    "MAX_INSTRUCTION_BYTES",
    "InstructionFile",
    "InstructionsError",
    "InstructionsPlugin",
    "SystemPromptPlugin",
    "apply_budget",
    "find_instruction_file",
    "find_instructions",
    "find_project_directories",
    "format_environment",
    "format_instructions",
    "get_config_directory",
    "load_prompt",
    "read_instruction_file",
    "read_named_instruction_file",
    "select_family",
]
