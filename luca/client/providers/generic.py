"""GenericProvider — built from a PROVIDERS config dict entry."""

from __future__ import annotations

from typing import Any

from .base import BaseProvider, ChatCompletionMixin


class GenericProvider(BaseProvider, ChatCompletionMixin):
    """A provider built from a registry config dict. Per-instance attrs avoid
    cross-instance pollution at the class level."""

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        default_api_key_env_var: str | None,
        default_transport_class: type,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.default_base_url = default_base_url
        self.default_api_key_env_var = default_api_key_env_var
        self.default_transport_class = default_transport_class
        super().__init__(**kwargs)
