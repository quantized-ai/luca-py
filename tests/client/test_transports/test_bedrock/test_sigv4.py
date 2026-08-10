"""SigV4 against AWS's own published vectors.

The cases below are transcribed from AWS's `aws-sig-v4-test-suite` — the
`.req` / `.creq` / `.sts` / `.authz` quadruples it ships. They matter because
they are EXTERNAL: a signer tested only against itself is self-consistent
right up until AWS rejects it, and a 403 from Bedrock says nothing about which
stage diverged. So each case asserts the canonical request, the string to
sign, and the signature separately.

All six use the suite's fixed inputs: access key `AKIDEXAMPLE`, the published
example secret, region `us-east-1`, service `service`, `20150830T123600Z`.

`sign()`'s own header-selection policy is tested further down, separately from
the algorithm. The vectors validate the maths; those tests validate what we
choose to sign.
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import pytest

from luca.client.transports.bedrock import sigv4

ACCESS_KEY_ID = "AKIDEXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
REGION = "us-east-1"
SERVICE = "service"
NOW = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)

EMPTY_BODY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# The suite's session-token case carries this verbatim.
SESSION_TOKEN = (
    "AQoDYXdzEPT//////////wEXAMPLEtc764bNrC9SAPBSM22wDOk4x4HIZ8j4FZTwdQWLWsKWHGBuFqwAeMicRXmxfpSPfIeo"
    "IYRqTflfKD8YUuwthAx7mSEI/qkPpKPi/kMcGdQrmGdeehM4IC1NtBmUpp2wUE8phUZampKsburEDy0KPkyQDYwT7WZ0wq5V"
    "SXDvp75YU9HFvlRd8Tx6q6fE8YQcHNVXAkiY9q6d+xo0rKwT38xVqr7ZD0u0iPPkUL64lIZbqBAz+scqKmlzm8FDrypNC9Yj"
    "c8fPOLn9FX9KSYvKTr4rvx3iSIlTJabIQwj2ICCR/oLxBA=="
)

UNRESERVED_PATH = "/-._~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Vector:
    name: str
    method: str
    url: str
    headers: dict
    body: bytes
    expected_canonical_request: str
    expected_string_to_sign: str
    expected_signature: str


VECTORS = [
    Vector(
        name="get-vanilla",
        method="GET",
        url="https://example.amazonaws.com/",
        headers={"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"},
        body=b"",
        expected_canonical_request=(
            "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\nhost;x-amz-date\n" + EMPTY_BODY_SHA
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
        ),
        expected_signature="5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
    ),
    Vector(
        name="post-vanilla",
        method="POST",
        url="https://example.amazonaws.com/",
        headers={"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"},
        body=b"",
        expected_canonical_request=(
            "POST\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\nhost;x-amz-date\n" + EMPTY_BODY_SHA
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "553f88c9e4d10fc9e109e2aeb65f030801b70c2f6468faca261d401ae622fc87"
        ),
        expected_signature="5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
    ),
    Vector(
        # A POST with a body and a content-type, which is what Bedrock sends.
        #
        # The suite's `.creq` file for this case lists `content-length` among
        # the signed headers, but its own `.sts` and `.authz` were computed
        # WITHOUT it — the published files disagree with each other. The
        # `.authz` is the authoritative half (`SignedHeaders=content-type;
        # host;x-amz-date`), so the canonical request below is the one that
        # actually hashes to the published string to sign.
        name="post-x-www-form-urlencoded",
        method="POST",
        url="https://example.amazonaws.com/",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Host": "example.amazonaws.com",
            "X-Amz-Date": "20150830T123600Z",
        },
        body=b"Param1=value1",
        expected_canonical_request=(
            "POST\n"
            "/\n"
            "\n"
            "content-type:application/x-www-form-urlencoded\n"
            "host:example.amazonaws.com\n"
            "x-amz-date:20150830T123600Z\n"
            "\n"
            "content-type;host;x-amz-date\n"
            "9095672bbd1f56dfc5b65f3e153adc8731a4a654192329106275f4c7b24d0b6e"
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "42a5e5bb34198acb3e84da4f085bb7927f2bc277ca766e6d19c73c2154021281"
        ),
        expected_signature="ff11897932ad3f4e8b18135d722051e5ac45fc38421b1da7b9d196a0fe09473a",
    ),
    Vector(
        name="post-sts-header-before",
        method="POST",
        url="https://example.amazonaws.com/",
        headers={
            "Host": "example.amazonaws.com",
            "X-Amz-Date": "20150830T123600Z",
            "X-Amz-Security-Token": SESSION_TOKEN,
        },
        body=b"",
        expected_canonical_request=(
            "POST\n"
            "/\n"
            "\n"
            "host:example.amazonaws.com\n"
            "x-amz-date:20150830T123600Z\n"
            f"x-amz-security-token:{SESSION_TOKEN}\n"
            "\n"
            "host;x-amz-date;x-amz-security-token\n" + EMPTY_BODY_SHA
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "c237e1b440d4c63c32ca95b5b99481081cb7b13c7e40434868e71567c1a882f6"
        ),
        expected_signature="85d96828115b5dc0cfc3bd16ad9e210dd772bbebba041836c64533a82be05ead",
    ),
    Vector(
        # Query parameters are sorted by key, and the sort is case sensitive.
        name="get-vanilla-query-order-key-case",
        method="GET",
        url="https://example.amazonaws.com/?Param2=value2&Param1=value1",
        headers={"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"},
        body=b"",
        expected_canonical_request=(
            "GET\n"
            "/\n"
            "Param1=value1&Param2=value2\n"
            "host:example.amazonaws.com\n"
            "x-amz-date:20150830T123600Z\n"
            "\n"
            "host;x-amz-date\n" + EMPTY_BODY_SHA
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "816cd5b414d056048ba4f7c5386d6e0533120fb1fcfa93762cf0fc39e2cf19e0"
        ),
        expected_signature="b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500",
    ),
    Vector(
        # The guard against over-encoding: every unreserved character survives
        # the canonical URI untouched.
        name="get-unreserved",
        method="GET",
        url="https://example.amazonaws.com" + UNRESERVED_PATH,
        headers={"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"},
        body=b"",
        expected_canonical_request=(
            "GET\n"
            f"{UNRESERVED_PATH}\n"
            "\n"
            "host:example.amazonaws.com\n"
            "x-amz-date:20150830T123600Z\n"
            "\n"
            "host;x-amz-date\n" + EMPTY_BODY_SHA
        ),
        expected_string_to_sign=(
            "AWS4-HMAC-SHA256\n"
            "20150830T123600Z\n"
            "20150830/us-east-1/service/aws4_request\n"
            "6a968768eefaa713e2a6b16b589a8ea192661f098f37349f4e2c0082757446f9"
        ),
        expected_signature="07ef7494c76fa4850883e2b006601f940f8a34d404d0cfa977f52a65bbf5f24f",
    ),
]


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: v.name)
def test_sigv4_matches_the_published_aws_vector(vector):
    canonical = sigv4.canonical_request(
        vector.method,
        httpx.URL(vector.url),
        vector.headers,
        vector.body,
    )
    to_sign = sigv4.string_to_sign(canonical, NOW, REGION, SERVICE)
    signature = hmac.new(
        sigv4.signing_key(SECRET_ACCESS_KEY, NOW, REGION, SERVICE),
        to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()

    assert (canonical, to_sign, signature) == (
        vector.expected_canonical_request,
        vector.expected_string_to_sign,
        vector.expected_signature,
    )


# ── what `sign()` chooses to sign ────────────────────────────────────────────


def test_sign_returns_the_headers_to_add():
    # The signature hex here is a pin, not an independent check — the maths is
    # what the AWS vectors above cover. What this test is about is WHICH
    # headers `sign` decides to sign, and the exact shape of the header it
    # builds around them.
    signed = sigv4.sign(
        "POST",
        httpx.URL("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse"),
        {"Content-Type": "application/json"},
        b"{}",
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        session_token=None,
        region=REGION,
        service="bedrock",
        now=NOW,
    )
    assert signed == {
        "X-Amz-Date": "20150830T123600Z",
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIDEXAMPLE/20150830/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=content-type;host;x-amz-date, "
            "Signature=ead2b3adab479832c7b4e45d84bfb705280722262a78a62adede500e89c3cc78"
        ),
    }


def test_a_session_token_joins_both_the_signed_set_and_the_headers():
    signed = sigv4.sign(
        "POST",
        httpx.URL("https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse"),
        {"Content-Type": "application/json"},
        b"{}",
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        session_token="session-token-value",
        region=REGION,
        service="bedrock",
        now=NOW,
    )
    # A token that rode along as a header without entering SignedHeaders would
    # be silently ignored by AWS, which is the failure this pins down.
    assert signed["X-Amz-Security-Token"] == "session-token-value"
    assert "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token" in signed["Authorization"]


def test_an_inference_profile_colon_is_encoded_in_the_canonical_uri_only():
    # httpx puts a literal `:` on the wire; AWS canonicalizes it to %3A. Both
    # are correct, and signing the wire form instead is a 403 that reads like
    # a credentials problem.
    url = httpx.URL(
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-20250514-v1:0/converse"
    )
    canonical = sigv4.canonical_request("POST", url, {"Host": url.host}, b"{}")

    assert url.raw_path == b"/model/us.anthropic.claude-sonnet-4-20250514-v1:0/converse"
    assert canonical.split("\n")[1] == "/model/us.anthropic.claude-sonnet-4-20250514-v1%3A0/converse"


def test_a_non_default_port_is_part_of_the_host_header():
    assert sigv4.host_header(httpx.URL("https://proxy.internal:8443/x")) == "proxy.internal:8443"
    assert sigv4.host_header(httpx.URL("https://proxy.internal/x")) == "proxy.internal"
