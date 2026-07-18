"""FauxProvider — wraps FauxTransport for tests."""

from __future__ import annotations

from typing import Any

from ..transports.faux import FauxTransport
from .base import BaseProvider, ChatCompletionMixin


class FauxProvider(BaseProvider, ChatCompletionMixin):
    name = "faux"
    default_base_url = ""
    default_api_key_env_var = None
    default_transport_class = FauxTransport

    def __init__(
        self,
        *,
        tokens_per_second: float | int = 0,
        transport: FauxTransport | None = None,
        **kwargs: Any,
    ) -> None:
        if transport is None:
            transport = FauxTransport(
                provider=self.name,
                base_url="",
                api_key=None,
                tokens_per_second=tokens_per_second,
            )
        # Bypass BaseProvider's transport-construction path entirely (the faux
        # has different needs).
        self._transport = transport
        self.tokens_per_second = tokens_per_second

    def set_responses(self, responses: list) -> None:
        self._transport.set_responses(responses)

    @property
    def requests(self) -> list:
        """Every request that reached the underlying transport, in order."""
        return self._transport.requests
