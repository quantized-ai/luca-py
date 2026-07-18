"""Shared helpers for the TUI tests.

`fresh_session` builds an empty faux-model session; `submit` types a message
into the prompt and presses Enter; `wait_until` polls an app condition
through the Pilot until it holds (one extra pause after it does, so pending
renders settle). Everything runs against a scripted `FauxProvider` — no
network, no keys.
"""

from __future__ import annotations

import time

from textual.widgets import Input

from luca.agent.core.models import AgentSession, LLMConfig, RuntimeConfig
from luca.agent.core.runner import AgentSessionRunner

FAUX_MODEL = LLMConfig(model="fake-model", provider="faux")


def fresh_session(runtime_config: RuntimeConfig | None = None) -> AgentSession:
    return AgentSessionRunner.new_session(FAUX_MODEL, runtime_config=runtime_config)


async def submit(pilot, text: str) -> None:
    pilot.app.query_one("#prompt", Input).value = text
    pilot.app.query_one("#prompt", Input).focus()
    await pilot.pause()
    await pilot.press("enter")


async def wait_until(pilot, condition, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("condition not met within timeout")


def idle_again(app) -> bool:
    """The drive worker is done: runner idle and the prompt re-enabled."""
    return app.runner.idle() and not app.query_one("#prompt", Input).disabled
