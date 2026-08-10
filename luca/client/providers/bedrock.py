"""AWS Bedrock provider.

Two things Bedrock needs that the base provider has no slot for, and both are
resolved here so the transport receives finished values.

REGION. It only ever appears in the hostname, so it folds into `base_url`
rather than becoming a new field on the base class. An explicit `base_url`
still wins, which keeps VPC endpoints and proxies working — but the region is
resolved either way, because SigV4 signs with it and cannot read it back out
of an arbitrary hostname.

AUTH SCHEME. A Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`, or an explicit
`api_key=`) rides the base class's bearer header and skips SigV4 entirely.
That order matches AWS's own SDKs, and it is what keeps an installation that
already works working. Otherwise the AWS credential chain runs and the
transport signs.
"""

from __future__ import annotations

import os

from ..exceptions import ConfigurationError
from ..transports import BedrockTransport
from ..transports.bedrock.credentials import resolve_credentials, resolve_region
from .base import BaseProvider, ChatCompletionMixin


class BedrockProvider(BaseProvider, ChatCompletionMixin):
    name = "bedrock"
    default_api_key_env_var = "AWS_BEARER_TOKEN_BEDROCK"
    default_transport_class = BedrockTransport
    region_env_var = "BEDROCK_AWS_REGION"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials=None,
        base_url: str | None = None,
        transport=None,
        **kwargs,
    ) -> None:
        if transport is not None:
            super().__init__(transport=transport, **kwargs)
            return

        region = resolve_region(credentials)
        bearer = api_key or os.environ.get(self.default_api_key_env_var)

        if base_url is None:
            if not region:
                raise ConfigurationError(
                    f"Provider 'bedrock' needs a region: set {self.region_env_var} or AWS_REGION in the "
                    "environment, put one in your AWS profile, or pass base_url= explicitly.",
                    provider=self.name,
                )
            base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

        if bearer is None:
            # SigV4 needs the region as a signing input, so an explicit
            # base_url does not excuse a missing one the way it does above.
            if not region:
                raise ConfigurationError(
                    f"Provider 'bedrock' needs a region to sign with: set {self.region_env_var} or "
                    "AWS_REGION, or put one in your AWS profile.",
                    provider=self.name,
                )
            credentials = resolve_credentials(credentials, region=region)
        else:
            credentials = None

        super().__init__(api_key=bearer, credentials=credentials, base_url=base_url, **kwargs)
