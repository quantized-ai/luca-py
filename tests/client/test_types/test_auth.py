"""`AwsCredentials`: what a caller may specify, and the two properties the
client core depends on — frozen (so it can key the provider cache) and
strict (so a typo'd field name is not silently dropped)."""

import pytest
from pydantic import ValidationError

from luca.client import AwsCredentials, Credentials


def test_every_field_is_optional():
    # This is the INPUT form, not what the signer needs: a caller who has a
    # region in ~/.aws/config should not have to restate it here.
    assert AwsCredentials() == AwsCredentials(
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        region=None,
        profile=None,
    )


def test_a_credential_is_hashable_so_it_can_key_the_provider_cache():
    one = AwsCredentials(access_key_id="AKIA1", secret_access_key="s1", region="us-east-1")
    same = AwsCredentials(access_key_id="AKIA1", secret_access_key="s1", region="us-east-1")
    other = AwsCredentials(access_key_id="AKIA2", secret_access_key="s1", region="us-east-1")
    assert {one, same, other} == {one, other}


def test_a_credential_is_frozen():
    credentials = AwsCredentials(access_key_id="AKIA1")
    with pytest.raises(ValidationError):
        credentials.access_key_id = "AKIA2"


def test_an_unknown_field_is_refused():
    with pytest.raises(ValidationError):
        AwsCredentials(secret_key="wrong-name")


def test_aws_credentials_are_credentials():
    # The core forwards `Credentials`; only the Bedrock provider reads inside.
    assert isinstance(AwsCredentials(), Credentials)
