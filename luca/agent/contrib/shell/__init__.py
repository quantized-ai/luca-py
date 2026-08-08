"""luca.agent.contrib.shell — filesystem and process tools.

The eight generic shell tools (read, glob, grep, edit, write, apply_patch,
delete_file, bash) bound to an active working directory, each exposing its
touched resources through
`luca.agent.contrib.resource_permissions.ResourcePermissionToolMixin`; the
four PROVIDER-NATIVE tools in [`native/`](native/__init__.py) and the
middleware that projects them; and `ShellAccessPlugin`, which bundles all
twelve behind one workspace with a shared `FileReadTracker` and a seeded
`PermissionStrategy`.

`ShellSession` / `ShellSessionPool` ([`session.py`](session.py)) are the live
bash process behind `anthropic_bash_20250124`, whose wire declaration promises
one. The plugin owns the pool and `aclose()`s it.
"""

from .native import (
    AnthropicBashTool,
    AnthropicTextEditorTool,
    OpenAIApplyPatchTool,
    OpenAIShellTool,
    ShellNativeMiddleware,
    active_natives,
    supported_native_tools,
)
from .plugin import ShellAccessPlugin
from .session import ShellSession, ShellSessionPool
from .tools import (
    ApplyPatchTool,
    BashTool,
    DeleteFileTool,
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
    "DeleteFileTool",
    "EditTool",
    "FileReadTracker",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "ShellAccessPlugin",
    "ShellSession",
    "ShellSessionPool",
    "ShellTool",
    "ShellToolError",
    "WriteTool",
    "OpenAIApplyPatchTool",
    "OpenAIShellTool",
    "AnthropicTextEditorTool",
    "AnthropicBashTool",
    "ShellNativeMiddleware",
    "active_natives",
    "supported_native_tools",
]
