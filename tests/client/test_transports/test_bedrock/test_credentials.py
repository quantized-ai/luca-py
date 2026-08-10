"""Where AWS credentials and the signing region come from.

The autouse `no_real_env` fixture in `tests/client/conftest.py` strips every
AWS variable and points the two file paths at somewhere that does not exist,
so each test below starts from "nothing configured anywhere" and adds exactly
the one thing it is about.
"""

import pytest

from luca.client.exceptions import ConfigurationError
from luca.client.transports.bedrock.credentials import (
    ResolvedAwsCredentials,
    resolve_credentials,
    resolve_region,
)
from luca.client.types.auth import AwsCredentials


def _write_files(tmp_path, monkeypatch, *, credentials=None, config=None):
    """Point the two AWS file paths at content this test owns."""
    if credentials is not None:
        path = tmp_path / "credentials"
        path.write_text(credentials)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(path))
    if config is not None:
        path = tmp_path / "config"
        path.write_text(config)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(path))


# ── credentials ──────────────────────────────────────────────────────────────


def test_an_explicit_credential_is_used_as_given():
    resolved = resolve_credentials(
        AwsCredentials(access_key_id="AKIA-EXPLICIT", secret_access_key="explicit-secret"),
        region="us-east-1",
    )
    assert resolved == ResolvedAwsCredentials(
        access_key_id="AKIA-EXPLICIT",
        secret_access_key="explicit-secret",
        session_token=None,
        region="us-east-1",
    )


def test_the_environment_supplies_a_credential_when_none_is_passed(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-ENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-token")

    assert resolve_credentials(None, region="eu-west-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-ENV",
        secret_access_key="env-secret",
        session_token="env-token",
        region="eu-west-1",
    )


def test_the_environment_beats_the_files(tmp_path, monkeypatch):
    _write_files(
        tmp_path,
        monkeypatch,
        credentials="[default]\naws_access_key_id = AKIA-FILE\naws_secret_access_key = file-secret\n",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-ENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")

    assert resolve_credentials(None, region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-ENV",
        secret_access_key="env-secret",
        session_token=None,
        region="us-east-1",
    )


def test_the_default_profile_is_read_from_the_credentials_file(tmp_path, monkeypatch):
    _write_files(
        tmp_path,
        monkeypatch,
        credentials=(
            "[default]\n"
            "aws_access_key_id = AKIA-DEFAULT\n"
            "aws_secret_access_key = default-secret\n"
            "aws_session_token = default-token\n"
        ),
    )

    assert resolve_credentials(None, region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-DEFAULT",
        secret_access_key="default-secret",
        session_token="default-token",
        region="us-east-1",
    )


def test_aws_profile_selects_a_named_profile(tmp_path, monkeypatch):
    _write_files(
        tmp_path,
        monkeypatch,
        credentials=(
            "[default]\naws_access_key_id = AKIA-DEFAULT\naws_secret_access_key = default-secret\n"
            "\n[work]\naws_access_key_id = AKIA-WORK\naws_secret_access_key = work-secret\n"
        ),
    )
    monkeypatch.setenv("AWS_PROFILE", "work")

    assert resolve_credentials(None, region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-WORK",
        secret_access_key="work-secret",
        session_token=None,
        region="us-east-1",
    )


def test_an_explicit_profile_beats_aws_profile(tmp_path, monkeypatch):
    _write_files(
        tmp_path,
        monkeypatch,
        credentials=(
            "[work]\naws_access_key_id = AKIA-WORK\naws_secret_access_key = work-secret\n"
            "\n[personal]\naws_access_key_id = AKIA-PERSONAL\naws_secret_access_key = personal-secret\n"
        ),
    )
    monkeypatch.setenv("AWS_PROFILE", "work")

    assert resolve_credentials(AwsCredentials(profile="personal"), region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-PERSONAL",
        secret_access_key="personal-secret",
        session_token=None,
        region="us-east-1",
    )


def test_a_named_profile_in_the_config_file_carries_the_profile_prefix(tmp_path, monkeypatch):
    # `~/.aws/config` writes `[profile work]`; `~/.aws/credentials` writes
    # `[work]`. Reading one with the other's convention finds nothing.
    _write_files(
        tmp_path,
        monkeypatch,
        config="[profile work]\naws_access_key_id = AKIA-CONFIG\naws_secret_access_key = config-secret\n",
    )

    assert resolve_credentials(AwsCredentials(profile="work"), region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-CONFIG",
        secret_access_key="config-secret",
        session_token=None,
        region="us-east-1",
    )


def test_the_credentials_file_beats_the_config_file(tmp_path, monkeypatch):
    _write_files(
        tmp_path,
        monkeypatch,
        credentials="[default]\naws_access_key_id = AKIA-CREDS\naws_secret_access_key = creds-secret\n",
        config="[default]\naws_access_key_id = AKIA-CONFIG\naws_secret_access_key = config-secret\n",
    )

    assert resolve_credentials(None, region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-CREDS",
        secret_access_key="creds-secret",
        session_token=None,
        region="us-east-1",
    )


def test_a_secret_containing_a_percent_sign_survives(tmp_path, monkeypatch):
    # The default ConfigParser reads `%` as interpolation and raises. AWS
    # secrets are base64-ish and legitimately contain one.
    _write_files(
        tmp_path,
        monkeypatch,
        credentials="[default]\naws_access_key_id = AKIA-PCT\naws_secret_access_key = abc%2Fdef%ghi\n",
    )

    assert resolve_credentials(None, region="us-east-1") == ResolvedAwsCredentials(
        access_key_id="AKIA-PCT",
        secret_access_key="abc%2Fdef%ghi",
        session_token=None,
        region="us-east-1",
    )


# ── refusals ─────────────────────────────────────────────────────────────────


def test_nothing_configured_anywhere_says_what_to_set():
    with pytest.raises(ConfigurationError, match="AWS_ACCESS_KEY_ID"):
        resolve_credentials(None, region="us-east-1")


def test_half_a_credential_in_the_environment_raises_rather_than_falling_through(tmp_path, monkeypatch):
    # Falling through would pair this access key with the FILE's secret and
    # report a 403 naming neither source.
    _write_files(
        tmp_path,
        monkeypatch,
        credentials="[default]\naws_access_key_id = AKIA-FILE\naws_secret_access_key = file-secret\n",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-ENV")

    with pytest.raises(ConfigurationError, match="secret access key is missing"):
        resolve_credentials(None, region="us-east-1")


def test_a_named_profile_that_does_not_exist_raises(tmp_path, monkeypatch):
    _write_files(tmp_path, monkeypatch, credentials="[default]\naws_access_key_id = A\naws_secret_access_key = B\n")

    with pytest.raises(ConfigurationError, match="'nope' was not found"):
        resolve_credentials(AwsCredentials(profile="nope"), region="us-east-1")


@pytest.mark.parametrize(
    ("key", "value", "mechanism"),
    [
        ("sso_start_url", "https://example.awsapps.com/start", "AWS SSO"),
        ("credential_process", "/usr/local/bin/get-creds", "credential_process"),
        ("role_arn", "arn:aws:iam::1:role/r", "role assumption"),
        ("web_identity_token_file", "/var/run/token", "web identity federation"),
    ],
)
def test_an_unsupported_profile_names_the_mechanism_and_the_way_out(tmp_path, monkeypatch, key, value, mechanism):
    _write_files(tmp_path, monkeypatch, config=f"[profile sso]\n{key} = {value}\nregion = us-east-1\n")

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_credentials(AwsCredentials(profile="sso"), region="us-east-1")

    message = str(exc_info.value)
    assert mechanism in message
    assert "aws configure export-credentials --profile sso" in message


def test_a_malformed_aws_file_is_an_error(tmp_path, monkeypatch):
    _write_files(tmp_path, monkeypatch, credentials="this line has no section header\n")

    with pytest.raises(ConfigurationError, match="not a valid AWS config file"):
        resolve_credentials(None, region="us-east-1")


# ── region ───────────────────────────────────────────────────────────────────


def test_no_region_anywhere_is_none():
    assert resolve_region(None) is None


def test_an_explicit_region_wins():
    assert resolve_region(AwsCredentials(region="ap-south-1")) == "ap-south-1"


@pytest.mark.parametrize(
    "variable",
    ["BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"],
)
def test_each_region_environment_variable_is_read(monkeypatch, variable):
    monkeypatch.setenv(variable, "eu-central-1")
    assert resolve_region(None) == "eu-central-1"


def test_bedrock_aws_region_beats_the_generic_aws_variables(monkeypatch):
    # It is what the provider read before SigV4 existed, so an installation
    # that sets both keeps the region it already had.
    monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    assert resolve_region(None) == "us-west-2"


def test_the_region_falls_back_to_the_profile(tmp_path, monkeypatch):
    _write_files(tmp_path, monkeypatch, config="[profile work]\nregion = ca-central-1\n")
    monkeypatch.setenv("AWS_PROFILE", "work")

    assert resolve_region(None) == "ca-central-1"
