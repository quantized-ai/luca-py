"""luca.agent.contrib.shell — filesystem and process tools.

The seven shell tools (read, glob, grep, edit, write, apply_patch, bash)
bound to an active working directory, each exposing its touched resources
through `luca.agent.contrib.resource_permissions.ResourcePermissionToolMixin`,
plus `ShellAccessPlugin`, which bundles them behind one workspace with a
shared `FileReadTracker` and a seeded `PermissionStrategy`.
"""

from .plugin import ShellAccessPlugin
from .tools import (
    ApplyPatchTool,
    BashTool,
    EditTool,
    FileReadTracker,
    GlobTool,
    GrepTool,
    ReadTool,
    ShellTool,
    ShellToolError,
    WriteTool,
)

__all__ = [
    "ApplyPatchTool",
    "BashTool",
    "EditTool",
    "FileReadTracker",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "ShellAccessPlugin",
    "ShellTool",
    "ShellToolError",
    "WriteTool",
]
