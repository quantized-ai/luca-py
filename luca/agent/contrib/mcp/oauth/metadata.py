"""Authorization-server discovery, and the issuer check that guards the code.

Pure parsing plus one validation rule that the 2026-07-28 revision made
mandatory: an authorization server SHOULD return `iss` with the authorization
code, and a client MUST validate a present `iss` against the recorded issuer
BEFORE redeeming it (RFC 9207). Getting that wrong is a mix-up attack, where a
malicious server sends you a code minted by somebody else and you hand it to
the wrong token endpoint.

The check lives here, next to the metadata that supplies the expected value, so
it cannot be forgotten in the flow.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field


class AuthorizationServerMetadata(BaseModel):
    """RFC 8414, trimmed to the fields the flow reads.

    Not `extra="forbid"`: this is a document from a third party, the RFC lets
    them add fields, and refusing an unknown one would break a login over
    something we never look at. The rest of the codebase forbids extras because
    it owns those shapes; this one it does not.
    """

    issuer: str
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    registration_endpoint: str | None = None
    code_challenge_methods_supported: list[str] = Field(default_factory=list)
    scopes_supported: list[str] = Field(default_factory=list)
    # RFC 9207. When true, an authorization response with no `iss` MUST be
    # rejected rather than merely unchecked.
    authorization_response_iss_parameter_supported: bool = False
    model_config = ConfigDict(extra="ignore")


class ProtectedResourceMetadata(BaseModel):
    """RFC 9728. Points at the authorization servers that guard a resource."""

    resource: str | None = None
    authorization_servers: list[str] = Field(default_factory=list)
    scopes_supported: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


def protected_resource_url(server_url: str) -> str:
    """Where to ask a resource who guards it.

    The well-known segment goes after the host and before the resource's own
    path, which is what makes one host able to serve several protected
    resources.
    """
    parts = urlsplit(server_url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, f"/.well-known/oauth-protected-resource{path}", "", ""))


def metadata_urls(issuer: str) -> list[str]:
    """Where to look for an issuer's metadata, best first.

    Three, because deployments disagree. The OAuth form with the issuer path
    inserted is the RFC 8414 rule, the OpenID form is what many providers
    actually serve, and the naive join is what small self-hosted servers do.
    """
    parts = urlsplit(issuer)
    path = parts.path.rstrip("/")
    root = (parts.scheme, parts.netloc)
    return [
        urlunsplit((*root, f"/.well-known/oauth-authorization-server{path}", "", "")),
        urlunsplit((*root, f"/.well-known/openid-configuration{path}", "", "")),
        urljoin(issuer.rstrip("/") + "/", ".well-known/openid-configuration"),
    ]


def fallback_metadata(issuer: str) -> AuthorizationServerMetadata:
    """The endpoints to assume when a server publishes no metadata at all.

    The pre-RFC-8414 convention, still what a lot of small servers implement.
    Better than refusing to log in, and every endpoint is under the issuer, so
    a wrong guess fails locally rather than leaking anything.
    """
    base = issuer.rstrip("/")
    return AuthorizationServerMetadata(
        issuer=issuer,
        authorization_endpoint=f"{base}/authorize",
        token_endpoint=f"{base}/token",
        registration_endpoint=f"{base}/register",
        code_challenge_methods_supported=["S256"],
    )


class IssuerMismatch(Exception):
    """The `iss` on the callback is not the issuer the flow started with."""


def check_issuer(expected: str, received: str | None, *, advertised: bool = False) -> None:
    """RFC 9207 §2.4, as the spec's table spells it out.

    A present `iss` is always compared, whether or not the server advertised
    that it sends one. An ABSENT one is only fatal when the server said it
    would send it: refusing every authorization server that predates the
    parameter would lock people out for no gain, but ignoring its absence from
    a server that promised it is how a mix-up attack gets through.

    Compared verbatim. RFC 9207 forbids case folding, default-port elision,
    trailing-slash and percent-encoding normalization before this comparison,
    because each of those is a way for two different issuers to look equal.
    """
    if received is None:
        if advertised:
            raise IssuerMismatch(
                "The authorization server advertises `iss` on responses but did not send one. Not redeeming the code."
            )
        return
    if received != expected:
        raise IssuerMismatch(
            f"The authorization response came from {received!r}, but the flow started with {expected!r}. "
            "Not redeeming the code."
        )


def canonical_resource(url: str) -> str:
    """The MCP server's canonical URI, for RFC 8707 `resource`.

    The token's audience. Getting it wrong means the authorization server mints
    a token the MCP server then refuses, which surfaces as a 401 immediately
    after a login that looked like it worked.

    Lowercase scheme and host, no fragment, and no trailing slash — the forms
    the spec calls canonical, and the ones a resource server compares against.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


_CHALLENGE_PARAM = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_challenge(header: str | None) -> dict[str, str]:
    """The `Bearer` parameters of a `WWW-Authenticate` header.

    A 401 or 403 from an MCP server carries the two things a client needs to
    recover: `resource_metadata`, saying where to look up who guards it, and
    `scope`, which the spec makes authoritative for the operation that failed.
    """
    if not header:
        return {}
    return {key.lower(): value for key, value in _CHALLENGE_PARAM.findall(header)}
