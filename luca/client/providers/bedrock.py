"""AWS Bedrock provider.

Region is the one thing Bedrock needs that the base provider has no slot for,
and it only ever appears in the hostname. So the region folds into `base_url`
here rather than becoming a new field: read `BEDROCK_AWS_REGION`, build the
runtime host, and hand the rest to the base class. An explicit `base_url` still
wins, which keeps VPC endpoints and proxies working."""

from __future__ import annotations

import os

from ..exceptions import ConfigurationError
from ..transports import BedrockTransport
from .base import BaseProvider, ChatCompletionMixin


class BedrockProvider(BaseProvider, ChatCompletionMixin):
    name = "bedrock"
    default_api_key_env_var = "AWS_BEARER_TOKEN_BEDROCK"
    default_transport_class = BedrockTransport
    region_env_var = "BEDROCK_AWS_REGION"

    def __init__(self, *, base_url: str | None = None, transport=None, **kwargs) -> None:
        if transport is None and base_url is None:
            region = os.environ.get(self.region_env_var)
            if not region:
                raise ConfigurationError(
                    f"Provider 'bedrock' needs a region: set {self.region_env_var} "
                    "in the environment or pass base_url= explicitly.",
                    provider=self.name,
                )
            base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        super().__init__(base_url=base_url, transport=transport, **kwargs)
