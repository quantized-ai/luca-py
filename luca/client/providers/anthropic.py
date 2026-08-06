from ..transports import AnthropicTransport
from ..transports.anthropic.native_tools import BashTool, TextEditorTool
from .base import BaseProvider, ChatCompletionMixin

__all__ = ["AnthropicProvider", "BashTool", "TextEditorTool"]


class AnthropicProvider(BaseProvider, ChatCompletionMixin):
    name = "anthropic"
    default_base_url = "https://api.anthropic.com"
    default_api_key_env_var = "ANTHROPIC_API_KEY"
    default_transport_class = AnthropicTransport
