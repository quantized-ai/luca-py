"""`.env` — the convenience path to a provider credential. `auth.json`
(`auth.py`) is the designed one.

Two rules:

  - A name already in `os.environ` is left alone. Exporting is deliberate; a
    checked-out `.env` is a default, and must not shadow it.
  - A line this cannot read is an ERROR naming the file, line and reason.
    `python-dotenv` logs a warning and carries on, which under a TUI that
    writes logs to a file means a stray quote drops a credential in silence.

Applying it mutates `os.environ`, which is what `luca.client` reads when it
builds a provider — an application's business, hence `contrib/tui`.

Grammar:

    # a comment
    KEY=value
    KEY=value # a trailing comment, dropped
    KEY="value"
    KEY='value'
    export KEY=value
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import LucaConfigError, find_project_config

# A `#` that starts a comment: whitespace-preceded, on an UNQUOTED value only.
_INLINE_COMMENT = re.compile(r"\s#")

__all__ = [
    "ENV_ENV_PATH",
    "apply_env_file",
    "load_env_file",
    "parse_env",
    "resolve_env_path",
]

ENV_ENV_PATH = "LUCA_ENV_PATH"
"""Names an env file to use INSTEAD of the discovered one."""

ENV_FILENAME = ".env"


def parse_env(text: str, source: str = ENV_FILENAME) -> dict[str, str]:
    """`.env` text → its variables. Pure; `source` only shapes the message.

    Raises on any line it cannot read: skipping one turns a missing credential
    into a provider's authentication error three layers away."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator:
            raise LucaConfigError(f"{source} line {number}: no '=' in {line!r}")
        if not name:
            raise LucaConfigError(f"{source} line {number}: no name before the '='")

        values[name] = _unquote(value.strip(), name=name, source=source, number=number)
    return values


def _unquote(value: str, *, name: str, source: str, number: int) -> str:
    """Strip one layer of matching quotes, refusing anything ambiguous.

    A quoted value must close and then END: trailing characters mean a stray
    quote, and guessing which reading was meant is how a credential gets
    silently truncated. An unquoted one runs to the end of the line except for
    a whitespace-preceded `#`, matching python-dotenv so the same file means
    the same thing under either."""
    if not value or value[0] not in "\"'":
        comment = _INLINE_COMMENT.search(value)
        return value[: comment.start()].rstrip() if comment else value

    quote = value[0]
    closing = value.find(quote, 1)
    if closing == -1:
        raise LucaConfigError(f"{source} line {number}: {name} has an unterminated {quote} quote")
    if closing != len(value) - 1:
        raise LucaConfigError(f"{source} line {number}: {name} has trailing characters after the closing {quote} quote")
    return value[1:closing]


def resolve_env_path(cli_path: str | None = None, *, cwd: Path | None = None) -> Path | None:
    """Which env file to read, or None. `LUCA_ENV_PATH` names one outright;
    otherwise the nearest `.env` at or above the cwd, bounded by the repo,
    exactly like `luca.json`."""
    for value in (cli_path, os.environ.get(ENV_ENV_PATH)):
        if value:
            return Path(value).expanduser()
    return find_project_config(cwd or Path.cwd(), ENV_FILENAME)


def load_env_file(path: Path | None) -> dict[str, str]:
    """Read and parse one env file. `None` or a missing file is nothing to
    load: running off exported variables or `auth.json` is ordinary."""
    if path is None or not path.is_file():
        return {}
    try:
        text = path.read_text()
    except OSError as exc:
        raise LucaConfigError(f"{path}: cannot be read ({exc})") from exc
    return parse_env(text, source=str(path))


def apply_env_file(path: Path | None = None) -> dict[str, str]:
    """Load the env file and put what is missing into `os.environ`. Returns
    the names it SET, not everything it read."""
    applied = {}
    for name, value in load_env_file(path if path is not None else resolve_env_path()).items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied
