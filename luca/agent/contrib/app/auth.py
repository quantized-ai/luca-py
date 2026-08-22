"""`auth.json` — the provider credentials the TUI hands to the runner.

One user-global file, deliberately separate from `luca.json`: a config file is
the kind of thing you commit to a repo or paste into an issue, and a key is
not. It is read at boot, kept in memory, and passed to
`AgentSessionRunner(api_key=...)` — it never reaches `LLMConfig`, which is
persisted with the session and copied onto every assistant message.

    {
      "openrouter":          {"type": "api", "key": "sk-or-..."},
      "my_custom_provider":  {"type": "api", "key": "sk-..."}
    }

Any provider name is accepted, including one `luca.client` has never heard of
— pairing it with a `providers` entry in `luca.json` that gives a `base_url`
is what makes such a host reachable. A provider with no entry here is not an
error: no key is passed for it, and the client falls back to whatever
environment variable it knows for that provider. `type` is `"api"` today;
`"oauth"` becomes a second member of the union when it lands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .config import LucaConfigError

__all__ = ["AuthEntry", "ENV_AUTH_PATH", "auth_home", "load_auth", "resolve_auth_path"]

ENV_AUTH_PATH = "LUCA_AUTH_PATH"
"""Names an auth file to use INSTEAD of the discovered one."""


class AuthEntry(BaseModel):
    """One provider's credential."""

    type: Literal["api"]
    key: str

    model_config = ConfigDict(extra="forbid")


def auth_home() -> Path:
    """`$XDG_DATA_HOME/luca` or `~/.local/share/luca`.

    The DATA directory, not the config one: this is state luca owns and
    rewrites (an oauth refresh will), not a file you hand-edit and version."""
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "luca"


def resolve_auth_path(cli_path: str | None = None) -> Path:
    """Which auth file to read. `LUCA_AUTH_PATH` over the default location;
    `~` expanded. Separate from `load_auth` so that stays a pure function of
    its argument."""
    for value in (cli_path, os.environ.get(ENV_AUTH_PATH)):
        if value:
            return Path(value).expanduser()
    return auth_home() / "auth.json"


def load_auth(path: Path | None = None) -> dict[str, AuthEntry]:
    """Read and validate the auth file. A missing file is simply no
    credentials — running entirely off environment variables is the default
    experience, not a degraded one. Anything present but wrong is an error:
    silently ignoring a malformed entry would send the request unauthenticated
    and report a provider 401."""
    path = path or resolve_auth_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LucaConfigError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise LucaConfigError(f"{path}: the top level must be a JSON object of provider → credential")
    entries: dict[str, AuthEntry] = {}
    for name, value in data.items():
        try:
            entries[name] = AuthEntry.model_validate(value)
        except ValidationError as exc:
            raise LucaConfigError(f"{path}: provider {name!r} is invalid:\n{exc}") from exc
    return entries


def api_key_for(auth: dict[str, AuthEntry], provider: str) -> str | None:
    """This provider's key, or None to let the client use its environment
    variable. The one read every caller makes, so the "absent is fine" rule
    lives in one place."""
    entry = auth.get(provider)
    return None if entry is None else entry.key
