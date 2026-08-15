"""The package splits in two around its one optional dependency.

`LucaConfig` names the MCP config models at module scope, so the whole TUI has
to parse and run with the `mcp` group uninstalled. Everything that touches the
wire may refuse, but it has to refuse by naming the group rather than by
raising `ModuleNotFoundError` at whoever imported the TUI.

Run in a subprocess with `mcp_types` blocked, because that is what an
uninstalled group actually looks like and because faking it in-process would
mean mutating `sys.modules` for every other test in the session.
"""

from __future__ import annotations

import subprocess
import sys

WITHOUT_MCP_TYPES = """
import sys
sys.modules["mcp_types"] = None  # any `import mcp_types` now raises ImportError
{body}
"""

IMPORT_THE_PURE_HALF = """
from luca.agent.contrib.mcp import McpServerSettings, McpSettings, StdioServer
from luca.agent.contrib.mcp import errors, headers, servers, sse
built = McpSettings.model_validate({"servers": {"a": {"command": "x"}}}).build(environ={})
print(built["a"].command)
"""

IMPORT_THE_WIRE_HALF = """
try:
    from luca.agent.contrib.mcp import wire
except ImportError as exc:
    print(exc)
else:
    print("imported anyway")
"""


def run(body: str) -> str:
    """One fresh interpreter with the optional group blocked. A module-level
    factory, not logic in a test body: both cases are the same run."""
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_MCP_TYPES.format(body=body)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_the_config_models_import_with_the_group_uninstalled():
    assert run(IMPORT_THE_PURE_HALF) == "x"


def test_the_wire_layer_refuses_by_naming_the_group():
    assert run(IMPORT_THE_WIRE_HALF) == (
        "luca.agent.contrib.mcp needs mcp-types: install the `mcp` dependency group (uv sync)."
    )
