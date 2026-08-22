"""`BedrockProvider`: where the endpoint comes from, and which of the two auth
schemes a given environment selects.

Bedrock is the one provider that resolves something before handing off to the
base class. The region only ever appears in the hostname, so it folds into
`base_url` rather than becoming a field — but SigV4 also signs with it and
cannot read it back out of an arbitrary hostname, so it is resolved even when
the caller supplied their own endpoint.
"""

import pytest

from luca.client.exceptions import ConfigurationError
from luca.client.providers.bedrock import BedrockProvider
from luca.client.transports import BedrockTransport
from luca.client.transports.bedrock.credentials import ResolvedAwsCredentials
from luca.client.types.auth import AwsCredentials

IAM = AwsCredentials(access_key_id="AKIA-TEST", secret_access_key="secret-test")


# ── the endpoint ─────────────────────────────────────────────────────────────


def test_the_region_folds_into_the_runtime_hostname(monkeypatch):
    monkeypatch.setenv("BEDROCK_AWS_REGION", "eu-west-3")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")

    transport = BedrockProvider().transport
    assert isinstance(transport, BedrockTransport)
    assert transport._base_url == "https://bedrock-runtime.eu-west-3.amazonaws.com"


def test_an_explicit_base_url_wins_over_the_region(monkeypatch):
    # A VPC endpoint or a proxy. The region is still resolved, because the
    # signature needs it.
    monkeypatch.setenv("BEDROCK_AWS_REGION", "eu-west-3")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")

    provider = BedrockProvider(base_url="https://vpce-123.bedrock-runtime.eu-west-3.vpce.amazonaws.com")
    assert provider.transport._base_url == "https://vpce-123.bedrock-runtime.eu-west-3.vpce.amazonaws.com"


def test_a_prebuilt_transport_short_circuits_every_resolution():
    # Nothing is configured in this environment at all; passing a transport
    # means the caller has already made every decision.
    built = BedrockTransport(provider="bedrock", base_url="https://example.invalid", api_key="k")
    assert BedrockProvider(transport=built).transport is built


def test_no_region_and_no_base_url_names_what_to_set(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")

    with pytest.raises(ConfigurationError, match="BEDROCK_AWS_REGION"):
        BedrockProvider()


def test_a_region_from_the_aws_profile_is_enough(tmp_path, monkeypatch):
    # Before SigV4 this raised: the region was read from one env var and
    # nowhere else, so `aws configure`-only users could not reach Bedrock.
    config = tmp_path / "config"
    config.write_text("[default]\nregion = sa-east-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")

    assert BedrockProvider().transport._base_url == "https://bedrock-runtime.sa-east-1.amazonaws.com"


# ── which auth scheme ────────────────────────────────────────────────────────


def test_a_bedrock_api_key_selects_the_bearer_path(monkeypatch):
    monkeypatch.setenv("BEDROCK_AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")

    transport = BedrockProvider().transport
    assert (transport._api_key, transport._credentials) == ("bedrock-key", None)


def test_a_bearer_token_wins_even_when_aws_credentials_are_present(monkeypatch):
    # What AWS's own SDKs do, and what keeps an installation that already
    # works from changing behavior when this ships.
    monkeypatch.setenv("BEDROCK_AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-ENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")

    transport = BedrockProvider().transport
    assert (transport._api_key, transport._credentials) == ("bedrock-key", None)


def test_an_aws_credential_selects_sigv4_and_arrives_resolved(monkeypatch):
    monkeypatch.setenv("BEDROCK_AWS_REGION", "us-east-1")

    transport = BedrockProvider(credentials=IAM).transport
    assert (transport._api_key, transport._credentials) == (
        None,
        ResolvedAwsCredentials(
            access_key_id="AKIA-TEST",
            secret_access_key="secret-test",
            session_token=None,
            region="us-east-1",
        ),
    )


def test_environment_credentials_are_found_with_nothing_passed(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-ENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")

    transport = BedrockProvider().transport
    assert transport._credentials == ResolvedAwsCredentials(
        access_key_id="AKIA-ENV",
        secret_access_key="env-secret",
        session_token=None,
        region="ap-northeast-1",
    )


def test_signing_still_needs_a_region_when_the_base_url_is_explicit():
    # The endpoint is known but the signature is not: an explicit base_url
    # excuses the hostname, never the signing input.
    with pytest.raises(ConfigurationError, match="needs a region to sign with"):
        BedrockProvider(base_url="https://vpce-123.vpce.amazonaws.com", credentials=IAM)


def test_no_credentials_and_no_bearer_token_says_what_to_set(monkeypatch):
    monkeypatch.setenv("BEDROCK_AWS_REGION", "us-east-1")

    with pytest.raises(ConfigurationError, match="No AWS credentials found"):
        BedrockProvider()
