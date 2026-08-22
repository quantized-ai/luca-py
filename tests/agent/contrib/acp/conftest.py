"""A recording ACP client, and an agent wired to the offline faux provider.

The client is a real `acp.Client` implementation rather than a mock: the SDK
calls it exactly as a live editor would, so a shape it would reject is a shape
these tests reject. It records every `session/update` it receives, which is
what the assertions are written against.

No subprocess, no socket, no key. `--faux` scripts the conversation, so the
same tests run in CI and on a plane.
"""

from __future__ import annotations

import pytest
from acp import RequestError
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)

from luca.agent.contrib.acp import LucaAgent
from luca.agent.contrib.app import build_faux_provider
from luca.client.providers import PROVIDERS

CONFIG_ENV = "LUCA_CONFIG_PATH"


class RecordingClient:
    """Every update the agent sent, in order, plus a scripted answer to any
    permission request.

    `approve` picks by OPTION KIND rather than by index, because the option
    list is built from what the tool suggested and a test that hard-coded
    position 1 would break the moment a tool stops suggesting a widening."""

    def __init__(self, *, approve: str | None = "allow_once", elicitation: dict | None = None) -> None:
        self.updates: list = []
        self.permission_requests: list = []
        self.elicitations: list = []
        self._approve = approve
        self._elicitation = elicitation

    # ── the client side of the protocol ──────────────────────────────────────

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        self.updates.append(update)

    async def request_permission(self, session_id: str, tool_call, options: list, **kwargs):
        self.permission_requests.append((tool_call, options))
        if self._approve is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        for option in options:
            if option.kind == self._approve:
                return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option.option_id))
        raise AssertionError(f"no option of kind {self._approve!r} in {[o.kind for o in options]}")

    async def create_elicitation(self, message: str, mode, **kwargs):
        self.elicitations.append((message, mode))
        if self._elicitation is None:
            raise RequestError.method_not_found("elicitation/create")
        from acp.schema import AcceptElicitationResponse

        return AcceptElicitationResponse(action="accept", content=self._elicitation)

    async def complete_elicitation(self, elicitation_id: str, **kwargs) -> None:
        return None

    # ── everything this agent never calls ────────────────────────────────────

    async def read_text_file(self, *args, **kwargs):
        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(self, *args, **kwargs):
        raise RequestError.method_not_found("fs/write_text_file")

    async def create_terminal(self, *args, **kwargs):
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *args, **kwargs):
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, *args, **kwargs):
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, *args, **kwargs):
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *args, **kwargs):
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict):
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        return None

    def on_connect(self, conn) -> None:
        return None

    # ── reading the recording ────────────────────────────────────────────────

    def of(self, kind: str) -> list:
        return [update for update in self.updates if update.session_update == kind]

    def text(self) -> str:
        return "".join(update.content.text for update in self.of("agent_message_chunk"))

    def thoughts(self) -> str:
        return "".join(update.content.text for update in self.of("agent_thought_chunk"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep discovery off the contributor's home and the catalog off their
    cache, the same guard the TUI tests use."""
    from luca.client.catalog import _store

    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _store._clear_for_tests()
    saved = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)


@pytest.fixture(autouse=True)
def _quiet_logging():
    """`_compose` points the process-wide `luca` logger at a session file.
    Left attached it holds a deleted tmp_path open and blinds `caplog`."""
    import logging

    from luca.agent.contrib.app.boot import remove_log_handlers

    log = logging.getLogger("luca")
    level = log.level
    yield
    remove_log_handlers()
    log.setLevel(level)


@pytest.fixture
def agent(tmp_path):
    """A faux-provider agent whose sessions land under tmp_path."""

    def _build(**overrides) -> LucaAgent:
        options = {
            "provider": build_faux_provider(),
            "faux": True,
            "checkpoints": False,
            "skills": False,
            "instructions": False,
            "log_level": "OFF",
            **overrides,
        }
        return LucaAgent(**options)

    return _build


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root
