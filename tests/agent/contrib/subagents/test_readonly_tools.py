"""The sub-agent toolset is read-only: read/glob/grep and nothing that mutates,
and no `task` tool, so a sub-agent cannot spawn another sub-agent."""

from __future__ import annotations

from luca.agent.contrib.subagents import build_readonly_registry
from luca.agent.core import AgentSessionRunner

from .support import FAUX_MODEL


async def test_readonly_registry_exposes_only_read_glob_grep(tmp_path):
    registry = build_readonly_registry(tmp_path)
    session = AgentSessionRunner.new_session(FAUX_MODEL)

    names = {spec.name for spec in await registry.get_tools(session)}

    assert names == {"read", "glob", "grep"}
