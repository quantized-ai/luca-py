"""Bedrock over the real network, with real AWS credentials.

Every other test in this directory is hermetic, which means they can all pass
against a signature AWS would reject. This file is the only one that proves
the signature is ACCEPTED, so it is worth running whenever the signer, the
canonicalization, or the credential chain changes.

Never collected by default: `uv run py.test tests/` stays offline. Opt in with

    uv run py.test -m live tests/client/test_transports/test_bedrock/

It also opts out of the autouse `no_real_env` fixture in
`tests/client/conftest.py`, which exists precisely to strip the variables
these tests need.

Requires: AWS credentials reachable by the normal chain (environment,
`~/.aws/credentials`, or `AWS_PROFILE`), a region, and model access granted in
that account. Override the model with `LUCA_LIVE_BEDROCK_MODEL` — the default
is Nova Lite, which is cheap and marked verified-live in `capabilities.py`.
"""

import os

import pytest

from luca.client.exceptions import ClientError, ConfigurationError
from luca.client.providers.bedrock import BedrockProvider
from luca.client.types import ChatCompletionRequest, Tool, UserMessage

pytestmark = pytest.mark.live

MODEL = os.environ.get("LUCA_LIVE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)


def _request(**over) -> ChatCompletionRequest:
    fields: dict = {
        "provider": "bedrock",
        "model": MODEL,
        "messages": [UserMessage(content="Reply with exactly the word: pong")],
        "max_tokens": 32,
    }
    fields.update(over)
    return ChatCompletionRequest(**fields)


@pytest.fixture
def bedrock():
    """A provider signing with whatever the real chain resolves.

    `AWS_BEARER_TOKEN_BEDROCK` is unset for the duration: it would take
    precedence and silently turn every test here into a bearer-token test,
    which is not what this file is for."""
    os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    try:
        provider = BedrockProvider()
    except ConfigurationError as exc:
        pytest.skip(f"no AWS credentials for a live Bedrock run: {exc}")
    try:
        yield provider
    finally:
        provider.close()


def test_a_signed_request_is_accepted_and_answered(bedrock):
    response = bedrock.completion(_request())

    assert response.message.text
    assert response.usage.output_tokens > 0


async def test_a_signed_streaming_request_decodes(bedrock):
    async with bedrock.acompletion_stream(_request()) as stream:
        events = [event async for event in stream]

    # The binary eventstream framing decoded, which means the signature was
    # accepted before a single frame arrived.
    assert events
    assert stream.message.text


def test_a_signed_tool_call_round_trips(bedrock):
    response = bedrock.completion(
        _request(
            messages=[UserMessage(content="What is the weather in Paris? Use the tool.")],
            tools=[WEATHER_TOOL],
            max_tokens=512,
        )
    )

    assert [call.name for call in response.tool_calls] == ["get_weather"]
    assert response.tool_calls[0].arguments.get("city")


def test_a_corrupted_secret_is_rejected_by_aws(bedrock):
    """The control. Without it, a signature AWS merely TOLERATES is
    indistinguishable from a correct one — every assertion above would still
    pass if the service were ignoring the Authorization header."""
    transport = bedrock.transport
    good = transport._credentials
    transport._credentials = good.model_copy(update={"secret_access_key": good.secret_access_key + "-wrong"})
    try:
        with pytest.raises(ClientError) as exc_info:
            transport.completion(_request())
    finally:
        transport._credentials = good

    assert exc_info.value.provider == "bedrock"
