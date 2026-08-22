"""The whole thing over a real pipe: a subprocess, JSON-RPC, no shortcuts.

`test_agent.py` drives `LucaAgent` in-process, which proves the mapping but
not the wiring. This spawns `python -m luca.agent.contrib.acp` the way Zed
would and talks to it through the SDK's client connection, so it also proves
the entry point starts, the framing works, and NOTHING WRITES TO STDOUT except
the protocol — a single stray `print` anywhere in the import graph breaks the
stream, and that failure is invisible until someone tries it.

`--faux` means no key and no network.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from acp import RequestError, connect_to_agent
from acp.helpers import text_block
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    Implementation,
    RequestPermissionResponse,
)

pytestmark = pytest.mark.acp_stdio


class PipeClient:
    """Approves everything once and records the stream."""

    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self.updates.append(update)

    async def request_permission(self, session_id: str, tool_call, options: list, **kwargs):
        allow = next(option for option in options if option.kind == "allow_once")
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=allow.option_id))

    def __getattr__(self, name: str):
        async def unsupported(*args, **kwargs):
            raise RequestError.method_not_found(name)

        return unsupported

    def on_connect(self, conn) -> None:
        return None


async def test_a_real_client_can_drive_a_real_subprocess(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "luca.agent.contrib.acp",
        "--faux",
        "--no-checkpoints",
        "--no-skills",
        "--no-instructions",
        "--log-level",
        "OFF",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        cwd=str(workspace),
    )
    client = PipeClient()
    connection = connect_to_agent(client, process.stdin, process.stdout)
    try:
        initialized = await asyncio.wait_for(
            connection.initialize(
                protocol_version=1,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="luca-tests", version="0"),
            ),
            timeout=60,
        )
        session = await asyncio.wait_for(
            connection.new_session(cwd=str(workspace), mcp_servers=[]),
            timeout=60,
        )
        answer = await asyncio.wait_for(
            connection.prompt(session_id=session.session_id, prompt=[text_block("what is 6 times 7?")]),
            timeout=120,
        )
    finally:
        await connection.close()
        process.terminate()
        stderr = (await process.communicate())[1]

    assert initialized.protocol_version == 1
    assert initialized.agent_capabilities.load_session is True
    assert session.session_id
    assert answer.stop_reason == "end_turn"
    assert "42" in "".join(
        update.content.text for update in client.updates if update.session_update == "agent_message_chunk"
    )
    # Not an assertion about tidiness: a traceback here means the agent failed
    # in a way the protocol could not carry, and the test above would still
    # have passed if it happened after the last update.
    assert b"Traceback" not in stderr, stderr.decode(errors="replace")
