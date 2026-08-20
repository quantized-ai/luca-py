"""Browser OAuth for remote MCP servers.

Split so the parts that can be reasoned about on their own are: `pkce` and
`metadata` are pure, `callback` owns one socket, `store` owns one file, and
`provider` is the only place they meet. The security-relevant rules each live
next to the thing they guard — the issuer check beside the metadata that
supplies the expected value, the constant-time compare beside the state that
needs it.
"""

from __future__ import annotations

from .callback import CallbackTimeout, LoopbackCallback
from .metadata import (
    AuthorizationServerMetadata,
    IssuerMismatch,
    ProtectedResourceMetadata,
    check_issuer,
    fallback_metadata,
    metadata_urls,
    protected_resource_url,
)
from .pkce import Pkce, new_state, state_matches
from .provider import AUTHORIZE_TIMEOUT_S, OAuthProvider
from .store import ClientRegistration, IssuerRecord, OAuthToken, Registration, TokenStore

__all__ = [
    "AUTHORIZE_TIMEOUT_S",
    "AuthorizationServerMetadata",
    "CallbackTimeout",
    "ClientRegistration",
    "IssuerMismatch",
    "IssuerRecord",
    "LoopbackCallback",
    "OAuthProvider",
    "OAuthToken",
    "Pkce",
    "ProtectedResourceMetadata",
    "Registration",
    "TokenStore",
    "check_issuer",
    "fallback_metadata",
    "metadata_urls",
    "new_state",
    "protected_resource_url",
    "state_matches",
]
