r"""Filesystem access shared by the two executors.

One rule about paths: a model-supplied path is resolved against the root the
caller passed, and may not leave it. `root` IS the boundary — a caller that
wants none passes `root="/"`.

IO is byte-faithful about line endings: every read and write opens with
`newline=""`, so `\r\n` and lone `\r` reach the callers as themselves and go
back to disk unchanged. The default (`newline=None`) is universal-newline
mode, which would translate them to `\n` on the way in and back to
`os.linesep` on the way out — silently rewriting a CRLF file on its first
edit.
"""

from __future__ import annotations

from pathlib import Path

from .errors import NativeToolError


def resolve_within(root: str | Path, path: str) -> Path:
    """Resolve a model-supplied `path` against `root`.

    Relative paths join the root; an absolute path replaces it (and then has
    to be inside it anyway). Symlinks are resolved BEFORE the containment
    check, so a link pointing out of the root is refused — resolving is what
    lets the check see where the path really lands.
    """
    if not isinstance(path, str) or not path.strip():
        raise NativeToolError(f"Invalid path: {path!r}")

    base = Path(root).expanduser().resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise NativeToolError(f"Path {path!r} resolves outside the root directory {base}.")
    return target


def read_text(target: Path, *, display: str) -> str:
    """UTF-8 read, newlines untranslated, with a message the model can act on."""
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError as exc:
        raise NativeToolError(f"File not found: {display}") from exc
    except IsADirectoryError as exc:
        raise NativeToolError(f"Not a file: {display}") from exc
    except UnicodeDecodeError as exc:
        raise NativeToolError(f"File is not UTF-8 text: {display}") from exc
    except OSError as exc:
        raise NativeToolError(f"Could not read {display}: {_reason(exc)}.") from exc


def write_text(target: Path, text: str, *, display: str) -> None:
    """UTF-8 write, newlines untranslated, creating missing parent directories."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except OSError as exc:
        raise NativeToolError(f"Could not write {display}: {_reason(exc)}.") from exc


def remove_file(target: Path, *, display: str) -> None:
    """Delete a file, with a message the model can act on."""
    try:
        target.unlink()
    except FileNotFoundError as exc:
        raise NativeToolError(f"File not found: {display}") from exc
    except OSError as exc:
        raise NativeToolError(f"Could not delete {display}: {_reason(exc)}.") from exc


def list_dir(target: Path) -> list[Path]:
    """The entries of a directory, name-sorted."""
    try:
        return sorted(target.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise NativeToolError(f"Could not list {target}: {_reason(exc)}.") from exc


def _reason(exc: OSError) -> str:
    return exc.strerror or str(exc)
