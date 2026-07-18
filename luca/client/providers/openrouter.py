from ..transports import OpenRouterTransport
from .base import BaseProvider, ChatCompletionMixin


class OpenRouterProvider(BaseProvider, ChatCompletionMixin):
    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_api_key_env_var = "OPENROUTER_API_KEY"
    default_transport_class = OpenRouterTransport
