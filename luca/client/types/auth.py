"""Non-string credentials.

`api_key` covers every provider that authenticates with one opaque string.
SigV4 needs four values, so the client also carries a `Credentials` object it
never inspects: core stores and forwards, the provider interprets — the same
division `provider_options` uses.

`frozen=True` is load-bearing: a credential is part of the `_provider_cache`
key in `_client.py`.

Every field is optional because this is what a caller MAY specify, not what
the signer needs; `BedrockProvider` completes it from the environment and
`~/.aws` (`transports/bedrock/credentials.py`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Credentials(BaseModel):
    """Marker base. The client core forwards these; it never reads one."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AwsCredentials(Credentials):
    """The AWS credential inputs a caller may supply.

    `profile` names a section in `~/.aws/credentials` / `~/.aws/config` and
    wins over `AWS_PROFILE`. An entry with only `profile` set is the ordinary
    case for someone who has run `aws configure`."""

    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    profile: str | None = None
