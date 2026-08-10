"""`auth.json` — the provider credentials the TUI hands to the runner.

One user-global file, deliberately separate from `luca.json`: a config file is
the kind of thing you commit to a repo or paste into an issue, and a key is
not. It is read at boot, kept in memory, and passed to
`AgentSessionRunner(api_key=...)` / `(credentials=...)` — it never reaches
`LLMConfig`, which is persisted with the session and copied onto every
assistant message.

    {
      "openrouter":          {"type": "api", "key": "sk-or-..."},
      "my_custom_provider":  {"type": "api", "key": "sk-..."},
      "bedrock":             {"type": "aws", "profile": "work"}
    }

Any provider name is accepted, including one `luca.client` has never heard of
— pairing it with a `providers` entry in `luca.json` that gives a `base_url`
is what makes such a host reachable. A provider with no entry here is not an
error: no credential is passed for it, and the client falls back to whatever
environment variable or credential chain it knows for that provider.

Two credential kinds, discriminated on `type`:

  - `"api"` is one opaque string, which is every provider that authenticates
    with a bearer token or an api-key header.
  - `"aws"` is the SigV4 tuple, because one string cannot express it. Every
    field is optional — `{"type": "aws", "profile": "work"}` is the ordinary
    entry for someone who has run `aws configure`, and an entry with nothing
    at all still usefully says "use the AWS chain for this provider".

`"oauth"` becomes a third member when it lands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from luca.client import AwsCredentials

from .config import LucaConfigError

__all__ = [
    "ApiAuthEntry",
    "AuthEntry",
    "AwsAuthEntry",
    "ENV_AUTH_PATH",
    "api_key_for",
    "auth_home",
    "credentials_for",
    "load_auth",
    "resolve_auth_path",
]

ENV_AUTH_PATH = "LUCA_AUTH_PATH"
"""Names an auth file to use INSTEAD of the discovered one."""


class ApiAuthEntry(BaseModel):
    """One opaque string — the shape every api-key provider takes."""

    type: Literal["api"] = "api"
    key: str

    model_config = ConfigDict(extra="forbid")


class AwsAuthEntry(BaseModel):
    """The AWS SigV4 inputs. All optional: what is missing here is filled from
    the environment and `~/.aws` by the client, so naming a profile (or
    nothing at all) is a complete entry."""

    type: Literal["aws"]
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    profile: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_credentials(self) -> AwsCredentials:
        return AwsCredentials(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
            region=self.region,
            profile=self.profile,
        )


AuthEntry = Annotated[ApiAuthEntry | AwsAuthEntry, Field(discriminator="type")]

_ENTRY_ADAPTER: TypeAdapter[ApiAuthEntry | AwsAuthEntry] = TypeAdapter(AuthEntry)


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


def load_auth(path: Path | None = None) -> dict[str, ApiAuthEntry | AwsAuthEntry]:
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
    entries: dict[str, ApiAuthEntry | AwsAuthEntry] = {}
    for name, value in data.items():
        try:
            entries[name] = _ENTRY_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise LucaConfigError(f"{path}: provider {name!r} is invalid:\n{exc}") from exc
    return entries


def api_key_for(auth: dict[str, ApiAuthEntry | AwsAuthEntry], provider: str) -> str | None:
    """This provider's key, or None to let the client use its environment
    variable. The one read every caller makes, so the "absent is fine" rule
    lives in one place. An AWS entry has no key — it travels
    `credentials_for`."""
    entry = auth.get(provider)
    return entry.key if isinstance(entry, ApiAuthEntry) else None


def credentials_for(auth: dict[str, ApiAuthEntry | AwsAuthEntry], provider: str) -> AwsCredentials | None:
    """This provider's non-string credential, or None. The sibling of
    `api_key_for`: exactly one of the two answers for any given entry, and
    both answer None for a provider with no entry at all."""
    entry = auth.get(provider)
    return entry.to_credentials() if isinstance(entry, AwsAuthEntry) else None
