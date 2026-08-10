import pytest

from luca.client.transports.bedrock.credentials import ResolvedAwsCredentials

TEST_CREDENTIALS = ResolvedAwsCredentials(
    access_key_id="AKIDEXAMPLE",
    secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
    session_token=None,
    region="us-east-1",
)


@pytest.fixture
def bedrock_transport_factory():
    """A transport on the BEARER path. Every test that predates SigV4 uses
    this, which is what keeps them signing-free and unchanged."""
    from luca.client.transports import BedrockTransport

    def make(*, http_client=None, async_http_client=None):
        return BedrockTransport(
            provider="bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key="bedrock-token-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make


@pytest.fixture
def bedrock_sigv4_transport_factory():
    """A transport on the SigV4 path: a resolved credential and no bearer
    token, which is what the provider hands over for an IAM credential."""
    from luca.client.transports import BedrockTransport

    def make(*, credentials=TEST_CREDENTIALS, http_client=None, async_http_client=None, base_url=None):
        return BedrockTransport(
            provider="bedrock",
            base_url=base_url or "https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=None,
            credentials=credentials,
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
