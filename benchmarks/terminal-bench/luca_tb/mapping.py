"""Translations between Harbor's vocabulary and luca's.

No `harbor` import, so it unit-tests on its own. The usage arithmetic
duplicates `luca.agent.contrib.tui.usage`, which cannot be imported here
without pulling in Textual.
"""

from __future__ import annotations

from dataclasses import dataclass

from luca.client import catalog
from luca.client.providers import PROVIDERS

DEFAULT_PROVIDER = "openrouter"


def parse_model(spec: str, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    """Harbor's `-m` value as luca's `(provider, model)`.

    A luca model id can itself contain a slash (`openai/gpt-5.4-mini`), so the
    leading segment counts as a provider only when luca knows that host:

        openrouter/openai/gpt-5.4-mini  -> ("openrouter", "openai/gpt-5.4-mini")
        anthropic/claude-opus-4-5       -> ("anthropic",  "claude-opus-4-5")
        moonshotai/kimi-k2.7-code       -> (default,      "moonshotai/kimi-k2.7-code")

    `openai/...` therefore routes to the OpenAI API, not OpenRouter. Name the
    route explicitly when you want the other one.
    """
    if not spec:
        raise ValueError("model must not be empty")
    head, separator, rest = spec.partition("/")
    if separator and head in PROVIDERS:
        if not rest:
            raise ValueError(f"model {spec!r} names provider {head!r} but no model")
        return head, rest
    return default_provider, spec


def api_key_env_names(provider: str) -> list[str]:
    """Candidate env vars for this provider's key, in precedence order.

    Read off `PROVIDERS`, so a newly registered host needs no edit here.
    `LUCA_API_KEY` comes first as a cross-provider override."""
    entry = PROVIDERS.get(provider)
    if isinstance(entry, dict):
        configured = entry.get("default_api_key_env_var")
    else:
        configured = getattr(entry, "default_api_key_env_var", None)
    return [name for name in ("LUCA_API_KEY", configured) if name]


@dataclass(frozen=True)
class UsageTotals:
    """Summed `AgentSession.usages`. `n_input_tokens` includes cache reads,
    matching harbor's field description."""

    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    cost_usd: float | None = None


def context_from_session(session: dict) -> UsageTotals:
    """A serialized `AgentSession` as the numbers `AgentContext` wants.

    Parsed JSON rather than an `AgentSession`, so a trajectory from a different
    luca version still reports its tokens instead of failing validation."""
    raw_input = raw_output = cache_read = cache_write = 0
    for per_entry in (session.get("usages") or {}).values():
        for usage in (per_entry or {}).values():
            raw_input += usage.get("input") or 0
            raw_output += usage.get("output") or 0
            cache_read += usage.get("cache_read") or 0
            cache_write += usage.get("cache_write") or 0

    config = ((session.get("session_config") or {}).get("llm_config")) or {}
    return UsageTotals(
        n_input_tokens=raw_input + cache_read,
        n_output_tokens=raw_output,
        n_cache_tokens=cache_read,
        cost_usd=estimate_cost(
            provider=config.get("provider"),
            model=config.get("model"),
            input_tokens=raw_input,
            output_tokens=raw_output,
            cache_read=cache_read,
            cache_write=cache_write,
        ),
    )


def estimate_cost(
    *,
    provider: str | None,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
) -> float | None:
    """Dollars from the vendored models.dev catalog, or None when unpriced.

    None means unknown, never zero. A rate left unset within a KNOWN model does
    count as zero, so a partially priced model still reports what it can."""
    if not provider or not model:
        return None
    record = catalog.get(provider, model)
    price = record.cost if record is not None else None
    if price is None:
        return None
    if input_tokens == output_tokens == cache_read == cache_write == 0:
        return None
    return (
        input_tokens * (price.input_per_million_tokens or 0)
        + output_tokens * (price.output_per_million_tokens or 0)
        + cache_read * (price.cached_input_per_million_tokens or 0)
        + cache_write * (price.cache_write_per_million_tokens or 0)
    ) / 1_000_000
