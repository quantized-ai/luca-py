"""How a Bedrock request is authenticated, over real httpx.

Two schemes share one transport. A Bedrock API key rides the base class's
bearer header and the transport does nothing extra; an AWS credential means
SigV4, which the transport applies itself because the signature covers the
exact request body and the base class never sees one.

The signing algorithm is covered by AWS's published vectors in
`test_sigv4.py`. What is under test HERE is the wiring: that the right scheme
is chosen, that the signed bytes are the sent bytes, and that the path signed
is the path emitted. All three fail as a 403 that reads like bad credentials,
which is why each gets its own test.
"""

import json
from datetime import UTC, datetime

import httpx

from luca.client.transports.bedrock import sigv4
from luca.client.transports.bedrock.credentials import ResolvedAwsCredentials
from luca.client.types import ChatCompletionRequest, UserMessage

from ..._helpers.httpx_mocks import (
    eventstream_frame,
    eventstream_response,
    make_async_client,
    make_sync_client,
)
from ..._helpers.stream_iteration import acollect_events_with_snapshots, collect_events_with_snapshots
from .conftest import TEST_CREDENTIALS

REQUEST = ChatCompletionRequest(
    provider="bedrock",
    model="us.amazon.nova-lite-v1:0",
    messages=[UserMessage(content="Hi")],
)

# The id shape that carries a colon, which is the whole point of one test below.
INFERENCE_PROFILE_REQUEST = ChatCompletionRequest(
    provider="bedrock",
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    messages=[UserMessage(content="Hi")],
)

CONVERSE_REPLY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
}

STREAM_FRAMES = [
    eventstream_frame("messageStart", {"role": "assistant"}),
    eventstream_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hi"}}),
    eventstream_frame("contentBlockStop", {"contentBlockIndex": 0}),
    eventstream_frame("messageStop", {"stopReason": "end_turn"}),
]


def _capture(captured, reply=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["raw_path"] = request.url.raw_path
        captured["content"] = request.content
        captured["authorization"] = request.headers.get("authorization")
        captured["x_amz_date"] = request.headers.get("x-amz-date")
        captured["x_amz_security_token"] = request.headers.get("x-amz-security-token")
        return httpx.Response(200, json=reply if reply is not None else CONVERSE_REPLY)

    return handler


# ── scheme selection ─────────────────────────────────────────────────────────


def test_a_bedrock_api_key_is_sent_as_a_bearer_token_and_nothing_is_signed(bedrock_transport_factory):
    captured: dict = {}
    transport = bedrock_transport_factory(http_client=make_sync_client(_capture(captured)))
    transport.completion(REQUEST)

    # No SigV4 headers at all: a request carrying both would be ambiguous, and
    # this is the path every pre-SigV4 installation is still on.
    assert captured == {
        "method": "POST",
        "raw_path": b"/model/us.amazon.nova-lite-v1:0/converse",
        "content": json.dumps({"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}).encode(),
        "authorization": "Bearer bedrock-token-test",
        "x_amz_date": None,
        "x_amz_security_token": None,
    }


def test_an_aws_credential_is_signed_with_sigv4(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}
    transport = bedrock_sigv4_transport_factory(http_client=make_sync_client(_capture(captured)))
    transport.completion(REQUEST)

    assert captured == {
        "method": "POST",
        "raw_path": b"/model/us.amazon.nova-lite-v1:0/converse",
        "content": json.dumps({"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}).encode(),
        "authorization": (
            "AWS4-HMAC-SHA256 "
            "Credential=AKIDEXAMPLE/20250527/us-east-1/bedrock/aws4_request, "
            "SignedHeaders=content-type;host;x-amz-date, "
            "Signature=3708feab3a401ff7ee795196badf4026d19875c9023c02dcc5ae40014884ea77"
        ),
        "x_amz_date": "20250527T155000Z",
        "x_amz_security_token": None,
    }


def test_a_session_token_is_sent_and_enters_the_signed_header_set(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}
    transport = bedrock_sigv4_transport_factory(
        credentials=ResolvedAwsCredentials(
            access_key_id="AKIDEXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token="temporary-session-token",
            region="us-east-1",
        ),
        http_client=make_sync_client(_capture(captured)),
    )
    transport.completion(REQUEST)

    # A token sent as a header but left out of SignedHeaders is ignored by
    # AWS, which is the failure worth pinning.
    assert captured["x_amz_security_token"] == "temporary-session-token"
    assert "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token" in captured["authorization"]


# ── the path signed is the path sent ─────────────────────────────────────────


def test_an_inference_profile_id_signs_the_path_httpx_emits(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}
    transport = bedrock_sigv4_transport_factory(http_client=make_sync_client(_capture(captured)))
    transport.completion(INFERENCE_PROFILE_REQUEST)

    # httpx puts the colon on the wire literally; the canonical URI AWS
    # verifies against encodes it. Signing the wire form instead is a 403.
    assert captured["raw_path"] == b"/model/us.anthropic.claude-sonnet-4-20250514-v1:0/converse"
    assert captured["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")


# ── the body signed is the body sent ─────────────────────────────────────────


STREAM_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.amazon.nova-lite-v1:0/converse-stream"


def _authorization_over(body: bytes) -> str:
    """The Authorization header SigV4 produces for exactly these bytes.

    Comparing the wire header against this is the direct statement of the
    property: the signature that was sent is the signature OF the body that
    was sent. If the two serializations diverge, the headers differ."""
    return sigv4.sign(
        "POST",
        httpx.URL(STREAM_URL),
        {"Content-Type": "application/json"},
        body,
        access_key_id=TEST_CREDENTIALS.access_key_id,
        secret_access_key=TEST_CREDENTIALS.secret_access_key,
        session_token=TEST_CREDENTIALS.session_token,
        region=TEST_CREDENTIALS.region,
        service="bedrock",
        now=datetime(2025, 5, 27, 15, 50, 0, tzinfo=UTC),
    )["Authorization"]


def test_the_streamed_body_is_the_body_that_was_signed(bedrock_sigv4_transport_factory, frozen_time):
    # The regression this guards: `json=payload` lets httpx serialize a second
    # time, so the signature would cover bytes that never went on the wire.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content"] = request.content
        captured["authorization"] = request.headers.get("authorization")
        return eventstream_response(STREAM_FRAMES)(request)

    transport = bedrock_sigv4_transport_factory(http_client=make_sync_client(handler))
    with transport.completion_stream(REQUEST) as stream:
        collect_events_with_snapshots(stream)

    assert captured["authorization"] == _authorization_over(captured["content"])


async def test_the_async_streamed_body_is_the_body_that_was_signed(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content"] = request.content
        captured["authorization"] = request.headers.get("authorization")
        return eventstream_response(STREAM_FRAMES)(request)

    transport = bedrock_sigv4_transport_factory(async_http_client=make_async_client(handler))
    try:
        async with transport.acompletion_stream(REQUEST) as stream:
            await acollect_events_with_snapshots(stream)
    finally:
        await transport.aclose()

    assert captured["authorization"] == _authorization_over(captured["content"])


def test_a_bearer_stream_carries_no_signature(bedrock_transport_factory):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["content"] = request.content
        return eventstream_response(STREAM_FRAMES)(request)

    transport = bedrock_transport_factory(http_client=make_sync_client(handler))
    with transport.completion_stream(REQUEST) as stream:
        collect_events_with_snapshots(stream)

    assert captured["authorization"] == "Bearer bedrock-token-test"


def test_converse_stamps_the_signing_timestamp(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}
    transport = bedrock_sigv4_transport_factory(http_client=make_sync_client(_capture(captured)))

    transport.completion(REQUEST)

    assert captured["x_amz_date"] == "20250527T155000Z"


def test_converse_stream_stamps_the_same_signing_timestamp(bedrock_sigv4_transport_factory, frozen_time):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["x_amz_date"] = request.headers.get("x-amz-date")
        return eventstream_response(STREAM_FRAMES)(request)

    transport = bedrock_sigv4_transport_factory(http_client=make_sync_client(handler))
    with transport.completion_stream(REQUEST) as stream:
        collect_events_with_snapshots(stream)

    assert captured["x_amz_date"] == "20250527T155000Z"
