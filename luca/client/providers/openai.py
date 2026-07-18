from ..transports import OpenAITransport
from .base import BaseProvider, ChatCompletionMixin


class OpenAIProvider(BaseProvider, ChatCompletionMixin):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env_var = "OPENAI_API_KEY"
    default_transport_class = OpenAITransport
