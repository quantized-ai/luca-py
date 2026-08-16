"""PKCE and the CSRF state. Pure, no I/O, no clock.

Small enough to read in one sitting, which is the point: this is the part of
the flow where a mistake is silent, so it is separated from anything that
touches a socket and tested on its own.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Final

# RFC 7636 allows 43 to 128 characters. 96 bytes of entropy encodes to 128.
_VERIFIER_BYTES: Final = 64


def _b64(raw: bytes) -> str:
    """base64url without padding, which is what both RFC 7636 and the OAuth
    metadata documents use."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class Pkce:
    verifier: str
    challenge: str
    method: str = "S256"

    @classmethod
    def create(cls) -> Pkce:
        verifier = _b64(secrets.token_bytes(_VERIFIER_BYTES))
        challenge = _b64(hashlib.sha256(verifier.encode("ascii")).digest())
        return cls(verifier=verifier, challenge=challenge)


def new_state() -> str:
    """The CSRF token tying a callback back to the request that started it."""
    return _b64(secrets.token_bytes(32))


def state_matches(expected: str, received: str) -> bool:
    """Constant time, because a timing oracle on this value is a real CSRF
    weakness and `==` on a secret is the classic way to leave one."""
    return hmac.compare_digest(expected, received)
