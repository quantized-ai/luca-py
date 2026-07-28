"""Fixtures for the sub-agent tests."""

from __future__ import annotations

import pytest

from luca.agent.contrib.subagents import SubAgentManager


@pytest.fixture
async def make_manager():
    """Build managers and `aclose` them on teardown, so no background child task
    outlives the test (a leaked task fails the suite)."""
    created: list[SubAgentManager] = []

    def _make(*, cls: type[SubAgentManager] = SubAgentManager, **kwargs) -> SubAgentManager:
        manager = cls(**kwargs)
        created.append(manager)
        return manager

    yield _make
    for manager in created:
        await manager.aclose()
