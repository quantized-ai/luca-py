"""Where AWS credentials come from.

`AwsCredentials` is what a caller MAY specify; every field on it is optional.
This module turns one into what the signer needs, filling the gaps from the
environment and from the shared AWS files the `aws` CLI writes.

Supported, in precedence order: the explicit object, the standard environment
variables, then `~/.aws/credentials` and `~/.aws/config` under the selected
profile.

NOT supported: IMDSv2, the ECS/EKS container endpoints, web identity tokens,
the SSO token cache, `credential_process`, `role_arn` chaining. Each needs a
network call at credential time plus expiry handling. A profile using one is a
loud error, not a silent fall-through to unsigned; the message points at
`aws configure export-credentials`.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ...exceptions import ConfigurationError
from ...types.auth import AwsCredentials

ENV_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_SESSION_TOKEN = "AWS_SESSION_TOKEN"
ENV_PROFILE = "AWS_PROFILE"
ENV_SHARED_CREDENTIALS_FILE = "AWS_SHARED_CREDENTIALS_FILE"
ENV_CONFIG_FILE = "AWS_CONFIG_FILE"

# `BEDROCK_AWS_REGION` stays first: it is what the provider read before SigV4
# existed, so an installation that sets it keeps behaving identically.
REGION_ENV_VARS = ("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")

DEFAULT_PROFILE = "default"

# A profile carrying one of these and no access key is asking for a credential
# source this module does not implement. Mapped to the human name of the
# mechanism so the error can say which one.
UNSUPPORTED_PROFILE_KEYS = {
    "sso_start_url": "AWS SSO",
    "sso_session": "AWS SSO",
    "sso_account_id": "AWS SSO",
    "sso_role_name": "AWS SSO",
    "credential_process": "credential_process",
    "role_arn": "role assumption",
    "web_identity_token_file": "web identity federation",
}

_EXPORT_HINT = "run `aws configure export-credentials --profile {profile} --format env` and export those instead"


class ResolvedAwsCredentials(BaseModel):
    """What the signer needs: no optionals except the session token."""

    access_key_id: str
    secret_access_key: str
    region: str
    session_token: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


def _shared_credentials_path() -> Path:
    return Path(os.environ.get(ENV_SHARED_CREDENTIALS_FILE) or "~/.aws/credentials").expanduser()


def _shared_config_path() -> Path:
    return Path(os.environ.get(ENV_CONFIG_FILE) or "~/.aws/config").expanduser()


def _read_ini(path: Path) -> configparser.RawConfigParser | None:
    """A missing file is no credentials; a malformed one is an error.

    RAW, not the interpolating default: a secret containing `%` is legal and
    would otherwise raise `InterpolationSyntaxError`."""
    if not path.is_file():
        return None
    parser = configparser.RawConfigParser()
    try:
        parser.read_string(path.read_text(), source=str(path))
    except (OSError, configparser.Error) as exc:
        raise ConfigurationError(f"{path}: not a valid AWS config file ({exc})", provider="bedrock") from exc
    return parser


def _section(parser: configparser.RawConfigParser | None, profile: str, *, is_config_file: bool) -> dict[str, str]:
    """One profile's settings.

    The two files name their sections differently: `~/.aws/credentials` uses
    the bare profile name, `~/.aws/config` prefixes everything but `default`
    with `profile `."""
    if parser is None:
        return {}
    name = profile
    if is_config_file and profile != DEFAULT_PROFILE:
        name = f"profile {profile}"
    if not parser.has_section(name):
        return {}
    return {key.lower(): value.strip() for key, value in parser.items(name)}


def _profile_name(explicit: AwsCredentials | None) -> tuple[str, bool]:
    """(profile, was_named). `was_named` is what makes a missing profile an
    error rather than a fall-through: nobody asked for `default` by name."""
    if explicit is not None and explicit.profile:
        return explicit.profile, True
    from_env = os.environ.get(ENV_PROFILE)
    if from_env:
        return from_env, True
    return DEFAULT_PROFILE, False


def _profile_settings(profile: str) -> dict[str, str]:
    """The profile's settings, credentials file over config file."""
    return {
        **_section(_read_ini(_shared_config_path()), profile, is_config_file=True),
        **_section(_read_ini(_shared_credentials_path()), profile, is_config_file=False),
    }


def resolve_region(explicit: AwsCredentials | None) -> str | None:
    """The signing region, or None if nothing names one.

    Kept separate from credential resolution because the provider needs a
    region to build the runtime hostname even on the bearer-token path, where
    no credential is resolved at all."""
    if explicit is not None and explicit.region:
        return explicit.region
    for var in REGION_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    profile, _ = _profile_name(explicit)
    return _profile_settings(profile).get("region") or None


def _pair_from(
    source: str,
    access_key_id: str | None,
    secret_access_key: str | None,
    session_token: str | None,
) -> tuple[str, str, str | None] | None:
    """One source's credential pair, or None if it has neither half. Half a
    pair is an error: falling through would combine an access key from one
    source with a secret from another, and the 403 would name neither."""
    if not access_key_id and not secret_access_key:
        return None
    if not access_key_id or not secret_access_key:
        missing = "access key id" if not access_key_id else "secret access key"
        raise ConfigurationError(
            f"{source} supplies only half an AWS credential: the {missing} is missing.",
            provider="bedrock",
        )
    return access_key_id, secret_access_key, session_token


def _check_unsupported(profile: str, settings: dict[str, str]) -> None:
    for key, mechanism in UNSUPPORTED_PROFILE_KEYS.items():
        if key in settings:
            raise ConfigurationError(
                f"AWS profile {profile!r} uses {mechanism} ({key}), which luca does not support. "
                + _EXPORT_HINT.format(profile=profile),
                provider="bedrock",
            )


def resolve_credentials(explicit: AwsCredentials | None, *, region: str) -> ResolvedAwsCredentials:
    """The credential to sign with. Raises rather than returning None: the
    caller has already decided SigV4 is the auth scheme, so having nothing to
    sign with is a configuration failure, not an absence."""
    pair = _pair_from(
        "The passed AwsCredentials",
        explicit.access_key_id if explicit else None,
        explicit.secret_access_key if explicit else None,
        explicit.session_token if explicit else None,
    )
    if pair is None:
        pair = _pair_from(
            "The environment",
            os.environ.get(ENV_ACCESS_KEY_ID),
            os.environ.get(ENV_SECRET_ACCESS_KEY),
            os.environ.get(ENV_SESSION_TOKEN),
        )
    if pair is None:
        profile, was_named = _profile_name(explicit)
        settings = _profile_settings(profile)
        if not settings and was_named:
            raise ConfigurationError(
                f"AWS profile {profile!r} was not found in {_shared_credentials_path()} or {_shared_config_path()}.",
                provider="bedrock",
            )
        _check_unsupported(profile, settings)
        pair = _pair_from(
            f"AWS profile {profile!r}",
            settings.get("aws_access_key_id"),
            settings.get("aws_secret_access_key"),
            settings.get("aws_session_token"),
        )
    if pair is None:
        raise ConfigurationError(
            "No AWS credentials found for Bedrock. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
            "or configure a profile in ~/.aws/credentials, or set AWS_BEARER_TOKEN_BEDROCK to use a "
            "Bedrock API key instead.",
            provider="bedrock",
        )
    access_key_id, secret_access_key, session_token = pair
    return ResolvedAwsCredentials(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        region=region,
    )
