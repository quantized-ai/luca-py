"""`@`-mention expansion: a prompt string in, content parts out.

`parse_prompt` finds every `@path` in a submitted message, resolves it against
the workspace, and appends one content part per file that actually exists. The
typed text is always returned first, verbatim and unedited — the mentions stay
visible in it exactly as written.

A mention that does not resolve is left alone and produces no part. That is
what keeps `@property`, `@staticmethod`, `@media` and `@types/node` from being
mistaken for files: prose survives because nothing on disk answers to it.

Resolution never inlines blindly. `process_prompt_file_path` probes the file
once (one stat, one 8KB read) and hands it to the first handler that claims
it; see `handlers.py` for the chain and for how to add a format.
"""

from __future__ import annotations

import re
from pathlib import Path

from luca.agent.core.models import ContentPart, TextContent
from luca.client import catalog
from luca.client.catalog import ModelInfo

from .detection import HEAD_BYTES, looks_binary, read_head, sniff
from .handlers import (
    HANDLERS,
    STATUS_BINARY,
    STATUS_DIRECTORY,
    STATUS_OK,
    STATUS_TOO_LARGE,
    STATUS_TOO_LONG,
    STATUS_UNREADABLE,
    STATUS_UNSUPPORTED,
    TAG,
    FileProbe,
    MediaKind,
    ReadLimits,
    media_kind,
    mention_of,
    probe,
)

__all__ = [
    "mention_of",
    "HANDLERS",
    "HEAD_BYTES",
    "STATUS_BINARY",
    "STATUS_DIRECTORY",
    "STATUS_OK",
    "STATUS_TOO_LARGE",
    "STATUS_TOO_LONG",
    "STATUS_UNREADABLE",
    "STATUS_UNSUPPORTED",
    "TAG",
    "FileProbe",
    "MediaKind",
    "ReadLimits",
    "find_mentions",
    "looks_binary",
    "get_model_info",
    "media_kind",
    "parse_prompt",
    "probe",
    "process_prompt_file_path",
    "read_head",
    "sniff",
]

# `@` only counts at a word boundary, so foo@bar.com is never a mention. A
# comma both opens and closes a token, because the picker emits `@a.py,@b.py`
# — which means a file literally named `a,b.py` is unmentionable.
_MENTION = re.compile(r"(?:(?<=[\s,])|\A)@([^\s,]+)")

# Sentence punctuation that a path is unlikely to end in. Tried only after the
# raw token fails to resolve, so `@weird.name.` still works if it exists.
_TRAILING = ".,;:!?)]}\"'"


def _candidates(token: str) -> list[str]:
    seen = [token]
    trimmed = token
    while trimmed and trimmed[-1] in _TRAILING:
        trimmed = trimmed[:-1]
        if trimmed:
            seen.append(trimmed)
    return seen


def find_mentions(prompt: str, workspace: str | Path = ".") -> list[Path]:
    """Every `@path` in `prompt` that resolves to something on disk, in order
    of appearance, deduplicated. Non-resolving tokens are silently prose."""
    root = Path(workspace)
    found: list[Path] = []
    seen: set[Path] = set()
    for match in _MENTION.finditer(prompt):
        for candidate in _candidates(match.group(1)):
            path = Path(candidate)
            resolved = path if path.is_absolute() else root / path
            if not resolved.exists():
                continue
            key = resolved.resolve()
            if key not in seen:
                seen.add(key)
                found.append(resolved)
            break
    return found


def process_prompt_file_path(
    path: str | Path,
    *,
    workspace: str | Path = ".",
    limits: ReadLimits | None = None,
    context_window: int | None = None,
    model: ModelInfo | None = None,
) -> ContentPart:
    """One path → one content part, through the handler chain."""
    limits = limits or ReadLimits()
    target = Path(path)
    if not target.is_absolute():
        target = Path(workspace) / target
    probed = probe(target)
    for handler in HANDLERS:
        if handler.matches(probed, limits, context_window, model):
            return handler.build(probed, limits, context_window, model)
    raise AssertionError("TextHandler matches everything")  # pragma: no cover


def get_model_info(provider: str | None, model: str | None) -> ModelInfo | None:
    """The active model's catalog record, or None when it is not catalogued.
    None is what keeps a document away from a model we know nothing about."""
    if not provider or not model:
        return None
    return catalog.get(provider, model)


def parse_prompt(
    prompt: str,
    *,
    workspace: str | Path = ".",
    limits: ReadLimits | None = None,
    context_window: int | None = None,
    model: ModelInfo | None = None,
) -> list[ContentPart]:
    """The submitted text, then one part per resolvable `@` mention."""
    parts: list[ContentPart] = []
    if prompt.strip():
        parts.append(TextContent(text=prompt))
    parts.extend(
        process_prompt_file_path(
            path,
            workspace=workspace,
            limits=limits,
            context_window=context_window,
            model=model,
        )
        for path in find_mentions(prompt, workspace)
    )
    return parts
