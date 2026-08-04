"""Workspace file listing + fuzzy matching for the `@` context picker.

Listing prefers `git ls-files` (tracked + untracked, exclusions honored) and
falls back to a bounded walk. Token costs are the project's standard estimate
(size / 4).

Matching is SUBSEQUENCE, not substring: the query's characters must appear in
order but need not be adjacent, so `design/prompt` finds
`design_handoff_luca_tui/prompt.md` and `lactui` finds
`luca/agent/contrib/tui/…`. The filter is cheap because it short-circuits —
a file whose first query character is missing costs one `str.find`.

Ranking is where the quality lives, and it is path-aware in ways a general
string-similarity library cannot be: a hit in the basename beats one in a
directory name, a hit at a word boundary (`/`, `_`, `-`, `.`, a camelCase
hump) beats one mid-word, consecutive runs beat scattered characters, and a
shorter path breaks ties. Matched characters are wrapped in `[accent]…[/]` —
several spans now, since a subsequence match is not contiguous.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Safety valve for the non-git walk only. The `git ls-files` listing is NOT
# truncated: capping a sorted list makes every file past the cutoff
# permanently unmatchable, which is a correctness bug, not a tuning knob.
MAX_WALK_FILES = 20_000
MAX_ROWS = 8
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}

# Bonus weights. Tuned by ranking behaviour, not by theory — see test_files.py,
# which pins the orderings these produce.
_CONSECUTIVE = 8.0  # this character continued the previous match
_IN_BASENAME = 6.0  # matched inside the filename rather than a directory
_BOUNDARY = 10.0  # matched at the start of a path/word segment
_LENGTH_PENALTY = 0.1  # per character, so shorter paths win ties
_SEPARATORS = "/_-. "


def list_workspace_files(workspace: str | os.PathLike[str] = ".") -> list[str]:
    root = Path(workspace)
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return sorted(line for line in result.stdout.splitlines() if line)
    except (OSError, subprocess.TimeoutExpired):
        pass
    found: list[str] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for filename in filenames:
            found.append(str((Path(directory) / filename).relative_to(root)))
            if len(found) >= MAX_WALK_FILES:
                return sorted(found)
    return sorted(found)


def estimate_tokens(path: Path) -> int:
    try:
        return max(1, path.stat().st_size // 4)
    except OSError:
        return 0


def score_path(query: str, path: str) -> tuple[float, list[int]] | None:
    """`(score, matched indices)` if `query` is a subsequence of `path`, else
    None. `query` must already be lowercased.

    Greedy left-to-right: the first viable position for each character wins.
    fzf does a full DP for the optimal alignment; greedy is what most editors
    ship for the interactive path and costs a fraction as much.
    """
    low = path.lower()
    # Scanned RIGHT TO LEFT, taking the latest viable position for each
    # character. Left-to-right greedy matches "tui" against the t in "agent"
    # and then has to reach for a distant u and i; scanning backward lands on
    # "/tui/" intact. It also biases matches toward the tail of the path, which
    # is the basename — exactly what a file picker should prefer.
    positions: list[int] = []
    cursor = len(low)
    for character in reversed(query):
        cursor = low.rfind(character, 0, cursor)
        if cursor < 0:
            return None
        positions.append(cursor)
    positions.reverse()

    # `str.lower()` can change length for a few code points; the camelCase
    # bonus indexes the original, so only take it when the two line up.
    aligned = len(low) == len(path)
    basename_at = low.rfind("/") + 1
    score = 0.0
    previous = -2
    for cursor in positions:
        if cursor == previous + 1:
            score += _CONSECUTIVE
        if cursor >= basename_at:
            score += _IN_BASENAME
        if cursor == 0 or low[cursor - 1] in _SEPARATORS:
            score += _BOUNDARY
        elif aligned and path[cursor].isupper() and path[cursor - 1].islower():
            score += _BOUNDARY  # a camelCase hump is a word boundary too
        previous = cursor
    return score - len(path) * _LENGTH_PENALTY, positions


def mark_path(path: str, positions: list[int]) -> str:
    """The path with every matched run wrapped in `[accent]…[/]`."""
    if not positions:
        return path
    pieces: list[str] = []
    last = 0
    start = positions[0]
    end = start + 1
    for index in positions[1:]:
        if index == end:
            end = index + 1
            continue
        pieces.append(f"{path[last:start]}[accent]{path[start:end]}[/]")
        last = end
        start, end = index, index + 1
    pieces.append(f"{path[last:start]}[accent]{path[start:end]}[/]")
    pieces.append(path[end:])
    return "".join(pieces)


def rank_files(files: list[str], query: str, *, limit: int = MAX_ROWS) -> list[tuple[str, str]]:
    """`(path, span-marked path)` best match first — the ranking WITHOUT the
    disk. Every file is scored; only the display is capped. Split from
    `match_files` so a catalogued file list ranks identically to a real one."""
    if not query:
        return [(path, path) for path in files[:limit]]
    needle = query.lower()
    scored: list[tuple[float, str, str]] = []
    for path in files:
        hit = score_path(needle, path)
        if hit is None:
            continue
        score, positions = hit
        scored.append((score, path, mark_path(path, positions)))
    # `-score` first, then the path itself: ties resolve alphabetically so
    # the list never reshuffles between identical queries.
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [(path, marked) for _score, path, marked in scored[:limit]]


def match_files(
    files: list[str],
    query: str,
    workspace: str | os.PathLike[str] = ".",
    *,
    limit: int = MAX_ROWS,
) -> list[tuple[str, str, int]]:
    """`(path, span-marked path, token estimate)` rows for the picker, best
    match first. The ranking plus one `stat` per displayed row."""
    root = Path(workspace)
    return [(path, marked, estimate_tokens(root / path)) for path, marked in rank_files(files, query, limit=limit)]
