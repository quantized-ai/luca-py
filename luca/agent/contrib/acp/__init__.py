"""luca.agent.contrib.acp — luca as an Agent Client Protocol agent.

ACP is JSON-RPC 2.0 over stdio: a client (Zed, Nori, Pool, acp-ui) spawns the
agent as a subprocess, and the two talk over its stdin and stdout. This package
is the adapter — `LucaAgent` implements the protocol over the same
`AgentApplication` the TUI drives, so the two front ends compose identically
and cannot drift.

Run it:

    uv run python -m luca.agent.contrib.acp

STDOUT IS THE PROTOCOL. Nothing may print to it: logging goes to the session's
own file (`<session dir>/logs/<id>.log`) and `--log-level OFF` turns even that
off. A stray `print` corrupts the JSON-RPC stream and the client disconnects.

Requires the `acp` dependency group.
"""

from . import commands
from .agent import LucaAgent, content_parts
from .permissions import PermissionBridge, permission_options
from .questions import QuestionBridge, elicitation_schema
from .replay import replay
from .server import serve
from .stream import Translator, tool_kind, tool_title

__all__ = [
    "LucaAgent",
    "commands",
    "PermissionBridge",
    "QuestionBridge",
    "Translator",
    "content_parts",
    "elicitation_schema",
    "permission_options",
    "replay",
    "serve",
    "tool_kind",
    "tool_title",
]
