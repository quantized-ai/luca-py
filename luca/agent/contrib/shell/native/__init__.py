"""luca.agent.contrib.shell.native — the provider-native shell tools.

Four ordinary tools under stable internal names — `openai_apply_patch`,
`openai_shell`, `anthropic_text_editor_20250728`, `anthropic_bash_20250124` —
plus `ShellNativeMiddleware`, which is the ONLY thing here that knows a wire
exists: it decides which of them a request advertises, swaps their
declarations for the client's native items, upgrades their projected calls and
results to the native payloads stored at birth, and adopts the calls that come
back.

`supported_native_tools` answers which natives a given (provider, model) pair
can be offered at all; `active_natives` pairs that with the session's on/off
switch, and is what every decision above reads:

    active = active_natives(session)   # supported by the model AND switched on

`ShellAccessPlugin` wires all of it. Nothing outside this package, the plugin
included, needs the tables.
"""

from .anthropic import AnthropicBashTool, AnthropicTextEditorTool
from .middleware import ShellNativeMiddleware, active_natives
from .openai import OpenAIApplyPatchTool, OpenAIShellTool
from .support import supported_native_tools

__all__ = [
    "OpenAIApplyPatchTool",
    "OpenAIShellTool",
    "AnthropicTextEditorTool",
    "AnthropicBashTool",
    "ShellNativeMiddleware",
    "active_natives",
    "supported_native_tools",
]
