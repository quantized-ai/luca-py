"""The read-only toolset every sub-agent runs with.

A sub-agent gets exactly the three read-only shell tools — read, glob, grep —
gated by a YOLO policy. YOLO is deliberate: a sub-agent is headless, so an
ASK-mode strategy would stall on the first prompt with no human to answer it.
It is safe here because none of these tools mutates anything. The `task` tool
is never in this registry, so a sub-agent cannot spawn another sub-agent.
"""

from __future__ import annotations

import os

from luca.agent.contrib.shell.tools import GlobTool, GrepTool, ReadTool
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.contrib.tools import Tool


def readonly_tools(workspace: str | os.PathLike[str]) -> list[Tool]:
    """The read/glob/grep trio bound to `workspace`."""
    return [
        ReadTool(workdir=workspace),
        GlobTool(workdir=workspace),
        GrepTool(workdir=workspace),
    ]


def build_readonly_registry(workspace: str | os.PathLike[str]) -> SimpleToolRegistry:
    """A `SimpleToolRegistry` of the read-only tools under a YOLO policy."""
    return SimpleToolRegistry(
        tools=readonly_tools(workspace),
        permission_policy=YoloPermissionPolicy(),
    )
