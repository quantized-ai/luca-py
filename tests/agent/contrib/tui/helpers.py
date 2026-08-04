"""Shared helpers for the TUI tests.

`fresh_session` builds an empty faux-model session; `submit` types a message
into the prompt and presses Enter; `wait_until` polls an app condition
through the Pilot until it holds (one extra pause after it does, so pending
renders settle). Everything runs against a scripted `FauxProvider` — no
network, no keys.
"""

from __future__ import annotations

import time

from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.core.models import AgentSession, LLMConfig, RuntimeConfig, TextContent, UserMessage
from luca.agent.core.runner import AgentSessionRunner

FAUX_MODEL = LLMConfig(model="fake-model", provider="faux")


def fresh_session(runtime_config: RuntimeConfig | None = None) -> AgentSession:
    return AgentSessionRunner.new_session(FAUX_MODEL, runtime_config=runtime_config)


def with_user_message(text: str) -> AgentSession:
    """A session carrying one user message — enough for the resume picker to
    title it and for a replay to have something to draw."""
    session = fresh_session()
    entry = UserMessage(id="u1", created_at=1, parts=[TextContent(text=text)])
    session.entries[entry.id] = entry
    session.conversations[session.main_conversation_id].nodes.append(entry.id)
    return session


async def submit(pilot, text: str) -> None:
    prompt = pilot.app.query_one(PromptInput)
    prompt.load_text(text)
    prompt.focus()
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
    """The drive worker is done: runner idle and the drive worker released."""
    return app.runner.idle() and not app._driving
