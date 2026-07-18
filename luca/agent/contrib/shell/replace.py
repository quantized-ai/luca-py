"""Replacement-candidate strategies for the shell `edit` tool.

Ported from OpenCode's replacer chain: each strategy inspects the file
content and the caller-supplied `old_string` and yields candidate substrings
that plausibly correspond to what the caller meant — correcting common LLM
transcription errors (trimmed indentation, normalized whitespace, escaped
newlines, near-miss block middles). The `replace()` driver only ever applies
a literal `str.replace` of a candidate that actually occurs in the content,
so a fuzzy strategy can never invent text that is not in the file.

All strategies operate on LF-normalized text; the edit tool normalizes CRLF
before calling in and restores the file's convention afterwards.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator

BLOCK_ANCHOR_SIMILARITY_THRESHOLD = 0.65
CONTEXT_AWARE_MATCH_RATIO = 0.5
# A fuzzy candidate may legitimately span a little more text than the caller
# quoted (re-indentation, normalized whitespace), but a candidate that is
# disproportionately larger means the strategy anchored onto unrelated code.
MAX_CANDIDATE_GROWTH = 3
CANDIDATE_SLACK = 100

Replacer = Callable[[str, str], Iterator[str]]


class ReplacementError(Exception):
    """Base for replacement failures."""


class OldStringNotFound(ReplacementError):
    """No strategy produced a candidate present in the content."""


class OldStringAmbiguous(ReplacementError):
    """Every present candidate occurs more than once (and replace_all=False)."""


def _find_lines(find: str) -> list[str]:
    lines = find.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _windows(content_lines: list[str], size: int) -> Iterator[list[str]]:
    for start in range(len(content_lines) - size + 1):
        yield content_lines[start : start + size]


def simple_replacer(content: str, find: str) -> Iterator[str]:
    yield find


def line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = _find_lines(find)
    if not find_lines:
        return
    trimmed = [line.strip() for line in find_lines]
    for window in _windows(content.split("\n"), len(find_lines)):
        if [line.strip() for line in window] == trimmed:
            yield "\n".join(window)


def block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = _find_lines(find)
    if len(find_lines) < 3:
        return
    first, last = find_lines[0].strip(), find_lines[-1].strip()
    middle = "\n".join(line.strip() for line in find_lines[1:-1])
    for window in _windows(content.split("\n"), len(find_lines)):
        if window[0].strip() != first or window[-1].strip() != last:
            continue
        window_middle = "\n".join(line.strip() for line in window[1:-1])
        similarity = difflib.SequenceMatcher(None, middle, window_middle).ratio()
        if similarity >= BLOCK_ANCHOR_SIMILARITY_THRESHOLD:
            yield "\n".join(window)


def _normalize_whitespace(line: str) -> str:
    return " ".join(line.split())


def whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = _find_lines(find)
    if not find_lines:
        return
    normalized = [_normalize_whitespace(line) for line in find_lines]
    for window in _windows(content.split("\n"), len(find_lines)):
        if [_normalize_whitespace(line) for line in window] == normalized:
            yield "\n".join(window)


def _dedent(lines: list[str]) -> list[str]:
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    cut = min(indents, default=0)
    return [line[cut:] if line.strip() else "" for line in lines]


def indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = _find_lines(find)
    if not find_lines:
        return
    dedented = _dedent(find_lines)
    for window in _windows(content.split("\n"), len(find_lines)):
        if _dedent(window) == dedented:
            yield "\n".join(window)


_ESCAPES = {
    "\\n": "\n",
    "\\t": "\t",
    "\\r": "\r",
    "\\'": "'",
    '\\"': '"',
    "\\`": "`",
    "\\\\": "\\",
    "\\$": "$",
}


def _unescape(text: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair in _ESCAPES:
            out.append(_ESCAPES[pair])
            index += 2
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    unescaped = _unescape(find)
    if unescaped != find:
        yield unescaped


def trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    trimmed = find.strip()
    if trimmed and trimmed != find:
        yield trimmed


def context_aware_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = _find_lines(find)
    if len(find_lines) < 3:
        return
    for window in _windows(content.split("\n"), len(find_lines)):
        if window[0].strip() != find_lines[0].strip():
            continue
        if window[-1].strip() != find_lines[-1].strip():
            continue
        checked = matched = 0
        for window_line, find_line in zip(window[1:-1], find_lines[1:-1]):
            if not find_line.strip():
                continue
            checked += 1
            if window_line.strip() == find_line.strip():
                matched += 1
        if checked == 0 or matched / checked >= CONTEXT_AWARE_MATCH_RATIO:
            yield "\n".join(window)


def multi_occurrence_replacer(content: str, find: str) -> Iterator[str]:
    yield find


REPLACERS: tuple[Replacer, ...] = (
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
)


def _disproportionate(candidate: str, old_string: str) -> bool:
    if candidate == old_string:
        return False
    return len(candidate) > len(old_string) * MAX_CANDIDATE_GROWTH + CANDIDATE_SLACK


def replace(
    content: str, old_string: str, new_string: str, *, replace_all: bool = False,
) -> tuple[str, int]:
    """Apply the first usable candidate and return `(new_content, count)`.

    A candidate is usable when it literally occurs in `content` and is not
    disproportionately larger than `old_string`. With `replace_all` the first
    usable candidate has every occurrence replaced; otherwise only a candidate
    occurring exactly once is applied, and the search continues past ambiguous
    (multi-occurrence) candidates hoping a later strategy pins a unique one.
    """
    seen: set[str] = set()
    ambiguous = False
    for replacer in REPLACERS:
        for candidate in replacer(content, old_string):
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in content or _disproportionate(candidate, old_string):
                continue
            count = content.count(candidate)
            if replace_all:
                return content.replace(candidate, new_string), count
            if count == 1:
                return content.replace(candidate, new_string, 1), 1
            ambiguous = True
    if ambiguous:
        raise OldStringAmbiguous(old_string)
    raise OldStringNotFound(old_string)
