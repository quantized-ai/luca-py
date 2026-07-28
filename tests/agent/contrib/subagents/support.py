"""Shared builders for the sub-agent tests.

`FAUX_MODEL` is the offline model every child inherits; `scripted` wraps a
`FauxProvider`; `factory_over` builds a `RunnerFactory` handing each spawned
child its own provider (concurrent children cannot share one — the faux queue
interleaves); `until` polls the event loop until a condition holds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

from luca.agent.core import AgentSession, AgentSessionRunner, LLMConfig
from luca.client.testing import FauxProvider

FAUX_MODEL = LLMConfig(model="fake-model", provider="faux")


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def factory_over(
    providers: Iterable[FauxProvider],
) -> Callable[[str, AgentSession], AgentSessionRunner]:
    """A `RunnerFactory` handing each child the next provider — a toolless
    runner, since these children only need to answer."""
    it = iter(providers)

    def factory(agent_type: str, child_session: AgentSession) -> AgentSessionRunner:
        return AgentSessionRunner(child_session, tool_registry=None, provider=next(it))

    return factory


async def until(predicate: Callable[[], bool], *, tries: int = 500) -> None:
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")
