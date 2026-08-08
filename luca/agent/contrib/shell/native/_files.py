"""File IO for the native edit tools, matching `edit` / `write` byte for byte.

The provider-native editors reuse `luca.client.native`'s pure text transforms
(`apply_diff`, `str_replace`, `insert`, `view`) and do their own IO here
rather than calling the client's filesystem executors, for two reasons the
client cannot know about: the client confines every path under one `root`,
while in the agent the APPROVAL is the containment (a path outside the
workspace prompts, it is never refused); and the generic tools preserve a
BOM and CRLF line endings, which an edit made through a native tool must
preserve identically or a provider switch would rewrite the file's bytes.
"""

from __future__ import annotations

import difflib
from pathlib import Path

BOM_BYTES = b"\xef\xbb\xbf"


class SourceFile:
    """One file's text plus the two encoding facts a write has to restore."""

    def __init__(self, text: str, *, bom: bool, crlf: bool) -> None:
        self.text = text
        self.bom = bom
        self.crlf = crlf


def read_source(path: Path) -> SourceFile:
    """Read `path` as UTF-8 with its BOM stripped and CRLF normalized to LF,
    remembering both so `write_source` can put them back. Raises
    `UnicodeDecodeError` on a non-UTF-8 file — the caller owns the message."""
    raw = path.read_bytes()
    bom = raw.startswith(BOM_BYTES)
    if bom:
        raw = raw[len(BOM_BYTES) :]
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    return SourceFile(text.replace("\r\n", "\n"), bom=bom, crlf=crlf)


def write_source(path: Path, text: str, source: SourceFile | None = None) -> None:
    """Write `text` to `path`, restoring `source`'s BOM and line endings.
    `source=None` is a new file: LF, no BOM. Creates parent directories.

    The FILE's convention wins, so `text` is normalized to LF first: a model
    that puts `\\r\\n` in a replacement (or a diff whose own endings differ from
    the file's) must not turn the file's `\\r\\n` into `\\r\\r\\n`, and must not
    leave one CRLF line inside an LF file."""
    text = text.replace("\r\n", "\n")
    if source is not None and source.crlf:
        text = text.replace("\n", "\r\n")
    data = text.encode("utf-8")
    if source is not None and source.bom:
        data = BOM_BYTES + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def normalize(text: str) -> str:
    """Model-supplied text with its line endings normalized to LF — the form
    every transform here works in, since `read_source` normalized the file the
    same way. Without it a `\\r\\n` in an `old_str` never matches."""
    return text.replace("\r\n", "\n")


def read_lossy(path: Path) -> str:
    """The file's text for a DISPLAY purpose only (the metadata diff), with
    undecodable bytes replaced rather than raising — deleting a `.pyc` must not
    fail because it is not UTF-8."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def unified_diff(before: str, after: str, label: str) -> str:
    """The `metadata["diff"]` every EDIT tool in this package returns — what
    a UI renders as `+N −M`."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=label,
            tofile=label,
        ),
    )
