"""Choosing the base system prompt for a model.

One shared `base.md` plus a short per-family addendum, rather than a complete
prompt per family: the families differ in a handful of known behaviours, and
four near-identical copies drift apart.

Matching is on the MODEL ID only, never the provider — openrouter serves every
family, so `openrouter` says nothing about which prompt fits.
"""

from __future__ import annotations

from importlib import resources

BASE = "base"
GENERIC = "generic"

# Ordered; the first family whose substring appears in the lowercased model id
# wins, so `claude` is checked before the broader `gpt-`/`codex` patterns.
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude",)),
    ("gpt", ("gpt-", "o1-", "o3-", "codex")),
    ("gemini", ("gemini",)),
)

_cache: dict[str, str] = {}


def select_family(model: str) -> str:
    """The prompt family for a model id, or `GENERIC` when none matches."""
    identifier = model.lower()
    for family, needles in FAMILIES:
        if any(needle in identifier for needle in needles):
            return family
    return GENERIC


def load_prompt(name: str) -> str:
    """One `text/<name>.md`, read once per process."""
    if name not in _cache:
        _cache[name] = resources.files(__package__).joinpath("text", f"{name}.md").read_text(encoding="utf-8").strip()
    return _cache[name]
