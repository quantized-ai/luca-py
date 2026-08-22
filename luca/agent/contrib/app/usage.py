"""Session usage: the token totals and the dollar estimate.

Token figures are real (summed from `AgentSession.usages`); dollar figures
come from the model catalog and are None whenever it does not price the
model. Prices are per `(provider, model)`, so a model reached through two
routes is costed as the route it was actually called on.

They remain estimates. A handful of models price by context tier (above 200k
tokens the rate can double) and `ModelCost` carries one flat rate, so a very
long session under-reports.
"""

from __future__ import annotations

from dataclasses import dataclass

from luca.agent.core.models import AgentSession, LLMConfig
from luca.client import catalog
from luca.client.types import ModelCost


@dataclass(frozen=True)
class UsageTotals:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


def usage_totals(session: AgentSession) -> UsageTotals:
    input_ = output = cache_read = cache_write = 0
    for per_entry in session.usages.values():
        for usage in per_entry.values():
            input_ += usage.input
            output += usage.output
            cache_read += usage.cache_read
            cache_write += usage.cache_write
    return UsageTotals(input=input_, output=output, cache_read=cache_read, cache_write=cache_write)


def _price(config: LLMConfig) -> ModelCost | None:
    """What the catalog charges for this exact route, or None when it does not
    price the model. Keyed on `(provider, model)`, never on the bare name."""
    record = catalog.get(config.provider, config.model)
    return record.cost if record is not None else None


def cost_breakdown(session: AgentSession) -> dict[str, float] | None:
    """Estimated dollars per category, or None for a model the catalog does
    not price. A rate the catalog leaves unset counts as zero, so a model
    priced for input and output but not for cache still reports what it can."""
    price = _price(session.session_config.llm_config)
    if price is None:
        return None
    totals = usage_totals(session)
    return {
        "input": totals.input * (price.input_per_million_tokens or 0) / 1_000_000,
        "output": totals.output * (price.output_per_million_tokens or 0) / 1_000_000,
        "cache read": totals.cache_read * (price.cached_input_per_million_tokens or 0) / 1_000_000,
        "cache write": totals.cache_write * (price.cache_write_per_million_tokens or 0) / 1_000_000,
    }


def estimated_cost(session: AgentSession) -> float | None:
    breakdown = cost_breakdown(session)
    if breakdown is None or usage_totals(session).total == 0:
        return None
    return sum(breakdown.values())
