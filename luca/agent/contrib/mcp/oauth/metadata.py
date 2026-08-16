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


def check_issuer(expected: str, received: str | None) -> None:
    """RFC 9207. A present `iss` MUST match before the code is redeemed.

    An absent one is allowed: the parameter is SHOULD-level for servers, and
    refusing every authorization server that predates it would lock people out
    for no security gain. What is not allowed is a present one that disagrees,
    which is the mix-up attack this exists to stop.
    """
    if received is None:
        return
    if _canonical(received) != _canonical(expected):
        raise IssuerMismatch(
            f"The authorization response came from {received!r}, but the flow started with {expected!r}. "
            "Not redeeming the code."
        )


def _canonical(issuer: str) -> str:
    """Issuers differ only by a trailing slash more often than they should."""
    return issuer.rstrip("/")
