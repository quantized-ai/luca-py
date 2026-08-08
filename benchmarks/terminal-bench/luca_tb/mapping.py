"""The two translations between Harbor's vocabulary and luca's.

Kept free of any `harbor` import so it can be unit-tested on its own, with no
Docker, no container and no benchmark run.

`luca.agent.contrib.tui.usage` already does the usage arithmetic, but importing
it executes the TUI package root and pulls in Textual. Rather than reach around
that (or refactor the TUI to suit a benchmark), the ~20 lines are reimplemented
here against `luca.client.catalog`. It is duplication, and it is the cheaper
of the two mistakes.
"""

from __future__ import annotations

from dataclasses import dataclass

from luca.client import catalog
from luca.client.providers import PROVIDERS

DEFAULT_PROVIDER = "openrouter"


def parse_model(spec: str, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    """Harbor's `-m` value as luca's `(provider, model)`.

    Harbor writes `provider/model`, but a luca model id can itself contain a
    slash: OpenRouter names models `openai/gpt-5.4-mini`. So the leading
    segment only counts as a provider when luca actually knows a host by that
    name, and everything else is passed through whole:

        openrouter/openai/gpt-5.4-mini  → ("openrouter", "openai/gpt-5.4-mini")
        anthropic/claude-opus-4-5       → ("anthropic",  "claude-opus-4-5")
        openai/gpt-5.4-mini             → (default,      "openai/gpt-5.4-mini")

    The third case is the one worth staring at: `openai` IS a known host, so it
    would be read as a provider — which is right if you meant the OpenAI API
    directly and wrong if you meant that model through OpenRouter. There is no
    way to tell from the string alone, so the registry answer wins and you name
    the route explicitly when you want the other one.
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
    """Which environment variables could hold the key for this provider, in
    precedence order.

    Read off `PROVIDERS` rather than hardcoded, so a host added to luca (or
    registered by an application) is picked up without editing this file.
    `LUCA_API_KEY` comes first as a deliberate override: the benchmark often
    routes several providers through one gateway key, and demanding the
    provider's own variable name for that is friction with no upside."""
    entry = PROVIDERS.get(provider)
    if isinstance(entry, dict):
        configured = entry.get("default_api_key_env_var")
    else:
        configured = getattr(entry, "default_api_key_env_var", None)
    return [name for name in ("LUCA_API_KEY", configured) if name]


@dataclass(frozen=True)
class UsageTotals:
    """Summed `AgentSession.usages`, in Harbor's vocabulary.

    `n_input_tokens` includes cache reads, matching harbor's own field
    description ("the number of input tokens used including cache")."""

    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    cost_usd: float | None = None


def context_from_session(session: dict) -> UsageTotals:
    """A serialized `AgentSession` as the numbers `AgentContext` wants.

    Takes the parsed JSON rather than an `AgentSession` so a trajectory from a
    different luca version still reports its tokens instead of failing
    validation over some unrelated field — on the host, after the run, a strict
    parse buys nothing and can only lose data."""
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
    """Dollars from the vendored models.dev catalog, or None when it does not
    price this `(provider, model)`.

    None means unknown, never zero: reporting $0.00 for a model the catalog has
    never heard of would quietly understate a whole run. A rate the catalog
    leaves unset within a known model does count as zero, so a model priced for
    input and output but not for cache still reports what it can."""
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
