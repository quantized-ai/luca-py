"""Parser and applier for the `apply_patch` tool's file-oriented patch format.

The format is an envelope (`*** Begin Patch` / `*** End Patch`) around a
sequence of Add / Delete / Update operations; an Update carries hunks whose
expected lines are located in the target with four matching passes: exact,
trailing-whitespace trimmed, both-ends trimmed, then Unicode punctuation
normalized. Parsing and matching are pure — the tool owns all filesystem IO,
so verification can complete before anything is mutated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
ADD_PREFIX = "*** Add File: "
DELETE_PREFIX = "*** Delete File: "
UPDATE_PREFIX = "*** Update File: "
MOVE_PREFIX = "*** Move to: "
EOF_MARKER = "*** End of File"


class PatchError(Exception):
    """A patch failed to parse or a hunk failed to match its target."""


@dataclass
class Hunk:
    context: str | None = None
    lines: list[tuple[str, str]] = field(default_factory=list)  # (prefix, text)
    eof: bool = False


@dataclass
class AddOp:
    path: str
    lines: list[str]


@dataclass
class DeleteOp:
    path: str


@dataclass
class UpdateOp:
    path: str
    move_to: str | None
    hunks: list[Hunk]


PatchOp = AddOp | DeleteOp | UpdateOp

_HEREDOC_RE = re.compile(r"^\s*(?:cat\s*)?<<-?\s*['\"]?(\w+)['\"]?\s*$")


def _strip_heredoc(lines: list[str]) -> list[str]:
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines):
        return lines
    match = _HEREDOC_RE.match(lines[start])
    if match is None:
        return lines
    delimiter = match.group(1)
    end = len(lines) - 1
    while end > start and not lines[end].strip():
        end -= 1
    if lines[end].strip() == delimiter:
        return lines[start + 1 : end]
    return lines[start + 1 :]


def parse_patch(patch_text: str) -> list[PatchOp]:
    lines = _strip_heredoc(patch_text.splitlines())
    begin = next(
        (i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER), None,
    )
    end = next(
        (i for i, line in enumerate(lines) if line.strip() == END_MARKER), None,
    )
    if begin is None or end is None or end < begin:
        raise PatchError(
            "Invalid patch: missing '*** Begin Patch' / '*** End Patch' markers",
        )
    body = lines[begin + 1 : end]
    ops: list[PatchOp] = []
    index = 0
    while index < len(body):
        line = body[index]
        if line.startswith(ADD_PREFIX):
            index, op = _parse_add(body, index)
            ops.append(op)
        elif line.startswith(DELETE_PREFIX):
            ops.append(DeleteOp(path=line[len(DELETE_PREFIX):].strip()))
            index += 1
        elif line.startswith(UPDATE_PREFIX):
            index, op = _parse_update(body, index)
            ops.append(op)
        elif not line.strip():
            index += 1
        else:
            raise PatchError(f"Unknown patch line: {line!r}")
    if not ops:
        raise PatchError("Invalid patch: the envelope contains no operations")
    return ops


def _parse_add(body: list[str], index: int) -> tuple[int, AddOp]:
    path = body[index][len(ADD_PREFIX):].strip()
    index += 1
    added: list[str] = []
    while index < len(body) and not body[index].startswith("*** "):
        if not body[index].startswith("+"):
            raise PatchError(
                f"Invalid Add File line (must start with '+'): {body[index]!r}",
            )
        added.append(body[index][1:])
        index += 1
    return index, AddOp(path=path, lines=added)


def _parse_update(body: list[str], index: int) -> tuple[int, UpdateOp]:
    path = body[index][len(UPDATE_PREFIX):].strip()
    index += 1
    move_to: str | None = None
    if index < len(body) and body[index].startswith(MOVE_PREFIX):
        move_to = body[index][len(MOVE_PREFIX):].strip()
        index += 1
    hunks: list[Hunk] = []
    current: Hunk | None = None
    while index < len(body):
        line = body[index]
        if line.strip() == EOF_MARKER:
            if current is None:
                raise PatchError(f"'{EOF_MARKER}' marker without a hunk in {path}")
            current.eof = True
            index += 1
            continue
        if line.startswith("@@"):
            current = Hunk(context=line[2:].strip() or None)
            hunks.append(current)
            index += 1
            continue
        if line.startswith("*** "):
            break
        if line == "":
            prefix, text = " ", ""
        elif line[0] in (" ", "-", "+"):
            prefix, text = line[0], line[1:]
        else:
            raise PatchError(f"Invalid update line in {path}: {line!r}")
        if current is None:
            current = Hunk()
            hunks.append(current)
        current.lines.append((prefix, text))
        index += 1
    if not any(hunk.lines for hunk in hunks):
        raise PatchError(f"Update for {path} contains no changes")
    return index, UpdateOp(path=path, move_to=move_to, hunks=hunks)


# ── hunk application ─────────────────────────────────────────────────────────

_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-", "―": "-", "−": "-",
})


def _canonical(line: str) -> str:
    return line.translate(_PUNCT_MAP).strip()


_PASSES = (
    lambda line: line,
    lambda line: line.rstrip(),
    lambda line: line.strip(),
    _canonical,
)


def _find_line(haystack: list[str], needle: str, start: int) -> int | None:
    for normalize in _PASSES:
        wanted = normalize(needle)
        for position in range(start, len(haystack)):
            if normalize(haystack[position]) == wanted:
                return position
    return None


def _find_sequence(
    haystack: list[str], needle: list[str], start: int, eof: bool,
) -> int | None:
    if not needle:
        return start
    for normalize in _PASSES:
        wanted = [normalize(line) for line in needle]

        def matches(position: int) -> bool:
            return all(
                normalize(haystack[position + j]) == wanted[j]
                for j in range(len(needle))
            )

        if eof:
            tail = len(haystack) - len(needle)
            if tail >= start and matches(tail):
                return tail
        for position in range(start, len(haystack) - len(needle) + 1):
            if matches(position):
                return position
    return None


def apply_update(original_lines: list[str], op: UpdateOp) -> list[str]:
    """Apply an update's hunks to the file's lines (no trailing-newline
    element) and return the new lines. Kept context lines are copied from the
    original file, so fuzzy matches don't rewrite untouched lines."""
    result: list[str] = []
    index = 0
    for hunk in op.hunks:
        if hunk.context is not None:
            context_position = _find_line(original_lines, hunk.context, index)
            if context_position is None:
                raise PatchError(
                    f"Could not find context line '@@ {hunk.context}' in {op.path}",
                )
            result.extend(original_lines[index : context_position + 1])
            index = context_position + 1
        expected = [text for prefix, text in hunk.lines if prefix in (" ", "-")]
        position = _find_sequence(original_lines, expected, index, hunk.eof)
        if position is None:
            raise PatchError(
                f"Could not find the expected lines in {op.path}: {expected!r}",
            )
        result.extend(original_lines[index:position])
        cursor = position
        for prefix, text in hunk.lines:
            if prefix == " ":
                result.append(original_lines[cursor])
                cursor += 1
            elif prefix == "-":
                cursor += 1
            else:
                result.append(text)
        index = cursor
    result.extend(original_lines[index:])
    return result
