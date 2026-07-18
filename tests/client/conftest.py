"""Top-level test fixtures."""

import time

import pytest


@pytest.fixture
def frozen_time(monkeypatch):
    """Freeze time.time() so SDK-stamped timestamps are deterministic."""
    fixed_seconds = 1748361000.0
    monkeypatch.setattr(time, "time", lambda: fixed_seconds)
    monkeypatch.setattr(time, "time_ns", lambda: int(fixed_seconds * 1_000_000_000))
    return int(fixed_seconds * 1000)


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """Helper caches BaseProvider instances. Clear before/after every test."""
    try:
        from luca.client._client import _provider_cache
    except ImportError:
        _provider_cache = None
    if _provider_cache is not None:
        _provider_cache.clear()
    yield
    if _provider_cache is not None:
        _provider_cache.clear()


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch):
    """Strip provider env vars so forgotten api_key= can't hit a real provider."""
    for var in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
        "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)
