"""Session usage → the status-bar counter and the cost screen's view-model.

The arithmetic lives in `luca.agent.contrib.app.usage` and is shared with
every other front end; this is the formatting half. Pure module, no Textual.
"""

from __future__ import annotations

from luca.agent.contrib.app.usage import (
    UsageTotals,
    cost_breakdown,
    estimated_cost,
    usage_totals,
)
from luca.agent.contrib.simple_context_manager import (
    calculate_context_used,
    get_context_window_size,
)
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    ToolExecution,
    TurnStart,
    UserMessage,
)

from . import state as vm
from .format import fmt_cost, fmt_duration, fmt_tokens, short_model
from .render import tool_arg

__all__ = ["UsageTotals", "cost_state", "estimated_cost", "status_counter", "usage_totals"]


def status_counter(session: AgentSession) -> tuple[str | None, str | None]:
    """The status-bar counter: `(context tokens, estimated cost)` — either
    side None when there is nothing to show."""
    used = calculate_context_used(session, session.main_conversation_id)
    tokens = fmt_tokens(used) if used else None
    cost = estimated_cost(session)
    return tokens, fmt_cost(cost) if cost is not None else None


# ── the cost screen (1k) ──────────────────────────────────────────────────────

_METER_COLORS: dict[str, vm.MeterColor] = {
    "input": "accent",
    "output": "foreground",
    "cache write": "faint",
    "cache read": "rule",
}


def _session_minutes(session: AgentSession) -> int | None:
    stamps = [entry.created_at for entry in session.entries.values() if entry.created_at]
    if len(stamps) < 2:
        return None
    return round((max(stamps) - min(stamps)) / 60_000)


def cost_state(session: AgentSession) -> vm.CostState:
    totals = usage_totals(session)
    breakdown = cost_breakdown(session)
    model = short_model(session.session_config.llm_config.model)

    nodes = session.conversations[session.main_conversation_id].nodes
    turns = sum(1 for node in nodes if isinstance(session.entries.get(node), TurnStart))
    subline_bits = [f"{turns} {'turn' if turns == 1 else 'turns'}"]
    if (minutes := _session_minutes(session)) is not None:
        subline_bits.append(fmt_duration(minutes * 60))
    subline_bits.append(model)

    token_by_label = {
        "input": totals.input,
        "output": totals.output,
        "cache write": totals.cache_write,
        "cache read": totals.cache_read,
    }
    if breakdown is not None:
        top = max(breakdown.values()) or 1.0
        items = [
            vm.CostItem(
                label=label,
                tokens=fmt_tokens(token_by_label[label]),
                cost=fmt_cost(breakdown[label], precision=3),
                fraction=breakdown[label] / top,
                color=_METER_COLORS[label],
            )
            for label in ("input", "output", "cache write", "cache read")
        ]
        headline = fmt_cost(sum(breakdown.values()))
    else:
        top = max(token_by_label.values()) or 1
        items = [
            vm.CostItem(
                label=label,
                tokens=fmt_tokens(count),
                cost="—",
                fraction=count / top,
                color=_METER_COLORS[label],
            )
            for label, count in token_by_label.items()
        ]
        headline = f"{fmt_tokens(totals.total)} tokens"

    used = calculate_context_used(session, session.main_conversation_id)
    window = get_context_window_size(session)
    free = max(0, window - used)
    context = vm.ContextWindowState(
        used=f"{fmt_tokens(used)} / {fmt_tokens(window)}",
        percent=f"{round(100 * used / window) if window else 0}%",
        context_fraction=(used / window) if window else 0.0,
        reply_fraction=0.0,
        legend=[f"[accent]▪[/] context {fmt_tokens(used)}", f"free {fmt_tokens(free)}"],
    )

    return vm.CostState(
        headline=headline,
        subline=" · ".join(subline_bits),
        items=items,
        context=context,
        consumers=_consumers(session),
    )


def _consumers(session: AgentSession, count: int = 3) -> list[vm.ConsumerRow]:
    """The biggest context holders on the main path, labelled by source."""
    nodes = session.conversations[session.main_conversation_id].nodes
    scored: list[tuple[int, str]] = []
    for node in nodes:
        entry = session.entries.get(node)
        tokens = entry.context_tokens if entry is not None else None
        if not tokens:
            continue
        if isinstance(entry, ToolExecution):
            label = f"output · {entry.raw_tool_call.name} {tool_arg(entry)}".strip()
        elif isinstance(entry, UserMessage):
            label = "message · user"
        elif isinstance(entry, AssistantMessage):
            label = "message · assistant"
        else:
            continue
        scored.append((tokens, label))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [vm.ConsumerRow(label=label[:40], tokens=fmt_tokens(tokens)) for tokens, label in scored[:count]]
