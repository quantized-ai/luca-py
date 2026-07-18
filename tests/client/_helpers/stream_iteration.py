"""Snapshot delta events when materializing a stream into a list.

The streaming protocol uses a SHARED reference on `*_delta` events for `partial`.
Plain `list(stream)` produces a list where every delta's `partial` ends up
pointing at the final mutated state. Always use these helpers for assertions.
"""

from __future__ import annotations

_DELTA_TYPES = frozenset({
    "text_delta", "thinking_delta", "tool_call_delta", "refusal_delta",
})


def _snapshot_if_delta(event):
    if event.type in _DELTA_TYPES:
        return event.model_copy(
            update={"partial": event.partial.model_copy(deep=True)},
        )
    return event


def collect_events_with_snapshots(stream):
    return [_snapshot_if_delta(ev) for ev in stream]


async def acollect_events_with_snapshots(stream):
    out = []
    async for ev in stream:
        out.append(_snapshot_if_delta(ev))
    return out
