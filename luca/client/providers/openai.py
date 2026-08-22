from ..transports import OpenAIResponsesTransport
from ..transports.openai_responses.native_tools import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellCommandResult,
    ShellExitOutcome,
    ShellTimeoutOutcome,
    ShellToolCall,
    ShellToolMessage,
    WebSearchFilters,
    WebSearchImageSettings,
    WebSearchTool,
)
from .base import BaseProvider, ChatCompletionMixin

__all__ = [
    "OpenAIProvider",
    # native tools (Responses wire)
    "ApplyPatchTool",
    "ApplyPatchToolCall",
    "LocalShellTool",
    "ShellCommandResult",
    "ShellExitOutcome",
    "ShellTimeoutOutcome",
    "ShellToolCall",
    "ShellToolMessage",
    "WebSearchFilters",
    "WebSearchImageSettings",
    "WebSearchTool",
]


class OpenAIProvider(BaseProvider, ChatCompletionMixin):
    """OpenAI runs on `/v1/responses`, the protocol its own models are
    developed against and the only one that replays reasoning across a
    multi-step turn. The chat-completions endpoint stays reachable with
    `transport_class=OpenAITransport`; it is also what every OpenAI-compatible
    host in PROVIDERS still uses."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env_var = "OPENAI_API_KEY"
    default_transport_class = OpenAIResponsesTransport
