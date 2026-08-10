"""AWS Signature Version 4.

A leaf module: stdlib plus `httpx.URL`, no clock of its own, no I/O, and no
imports from the rest of the transport. Everything it needs is an argument, so
the AWS-published test vectors can be replayed against it verbatim.

`canonical_request` and `string_to_sign` are public so a mismatch says WHICH
stage diverged; a wrong `Authorization` alone is undebuggable.

Two encoding rules, both of which fail as a 403 that reads like bad
credentials:

  - The canonical path is encoded once MORE than the wire. A Bedrock
    inference-profile id carries a colon (`…-v1:0/converse`) which httpx sends
    literally and the canonical URI encodes to `%3A` — the "encode each
    segment twice" rule for every service but S3.
  - The body hash covers the exact bytes sent, so the caller serializes once
    and passes those bytes both here and to httpx.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from datetime import datetime
from urllib.parse import quote

import httpx

ALGORITHM = "AWS4-HMAC-SHA256"
_DATE_FORMAT = "%Y%m%dT%H%M%SZ"
_DATESTAMP_FORMAT = "%Y%m%d"


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lowercase names, trimmed values, sequential spaces collapsed."""
    return {name.lower(): " ".join(str(value).split()) for name, value in headers.items()}


def _canonical_path(url: httpx.URL) -> str:
    """`raw_path` is the bytes httpx will send. Sign those, not a path rebuilt
    from a string, so the two cannot drift."""
    path = url.raw_path.split(b"?", 1)[0].decode()
    return quote(path or "/", safe="/~")


def _canonical_query(url: httpx.URL) -> str:
    """Pairs sorted, verbatim, as AWS's own SDKs do when the query comes off a
    URL rather than a params dict. Converse sends none; this is here so the
    signer is correct rather than correct-for-one-endpoint."""
    raw = url.raw_path.split(b"?", 1)
    if len(raw) == 1 or not raw[1]:
        return ""
    pairs = []
    for part in raw[1].decode().split("&"):
        name, _, value = part.partition("=")
        pairs.append((name, value))
    return "&".join(f"{name}={value}" for name, value in sorted(pairs))


def host_header(url: httpx.URL) -> str:
    """The `host` value AWS will see. A non-default port is part of it."""
    default_port = {"http": 80, "https": 443}.get(url.scheme)
    if url.port is None or url.port == default_port:
        return url.host
    return f"{url.host}:{url.port}"


def canonical_request(
    method: str,
    url: httpx.URL,
    headers: Mapping[str, str],
    body: bytes,
) -> str:
    """The canonical request. `headers` are the ones to sign, and only those."""
    normalized = _normalize_headers(headers)
    signed_names = sorted(normalized)
    return "\n".join(
        [
            method.upper(),
            _canonical_path(url),
            _canonical_query(url),
            "".join(f"{name}:{normalized[name]}\n" for name in signed_names),
            ";".join(signed_names),
            hashlib.sha256(body).hexdigest(),
        ]
    )


def credential_scope(now: datetime, region: str, service: str) -> str:
    return f"{now.strftime(_DATESTAMP_FORMAT)}/{region}/{service}/aws4_request"


def string_to_sign(canonical: str, now: datetime, region: str, service: str) -> str:
    return "\n".join(
        [
            ALGORITHM,
            now.strftime(_DATE_FORMAT),
            credential_scope(now, region, service),
            hashlib.sha256(canonical.encode()).hexdigest(),
        ]
    )


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def signing_key(secret_access_key: str, now: datetime, region: str, service: str) -> bytes:
    """The four-stage derived key."""
    key = _hmac(f"AWS4{secret_access_key}".encode(), now.strftime(_DATESTAMP_FORMAT))
    key = _hmac(key, region)
    key = _hmac(key, service)
    return _hmac(key, "aws4_request")


def sign(
    method: str,
    url: httpx.URL,
    headers: Mapping[str, str],
    body: bytes,
    *,
    access_key_id: str,
    secret_access_key: str,
    session_token: str | None,
    region: str,
    service: str,
    now: datetime,
) -> dict[str, str]:
    """The headers to ADD to the request. The signable subset is picked here
    rather than by the caller, so it and `SignedHeaders` cannot disagree."""
    amz_date = now.strftime(_DATE_FORMAT)
    to_sign: dict[str, str] = {"host": host_header(url), "x-amz-date": amz_date}
    for name, value in headers.items():
        if name.lower() == "content-type":
            to_sign["content-type"] = value
    if session_token:
        to_sign["x-amz-security-token"] = session_token

    canonical = canonical_request(method, url, to_sign, body)
    signature = hmac.new(
        signing_key(secret_access_key, now, region, service),
        string_to_sign(canonical, now, region, service).encode(),
        hashlib.sha256,
    ).hexdigest()

    signed = {
        "X-Amz-Date": amz_date,
        "Authorization": (
            f"{ALGORITHM} "
            f"Credential={access_key_id}/{credential_scope(now, region, service)}, "
            f"SignedHeaders={';'.join(sorted(to_sign))}, "
            f"Signature={signature}"
        ),
    }
    if session_token:
        signed["X-Amz-Security-Token"] = session_token
    return signed
