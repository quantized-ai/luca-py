"""OpenRouter streamer — the chat-completions wire under another host.

Everything OpenRouter-specific about the STREAM (mid-stream error chunks,
`reasoning` thinking deltas) already lives in OpenAIChatCompletionsStreamer,
where it also serves DeepSeek-style hosts; only the label and the default
host change here.
"""

from __future__ import annotations

from ..openai.streamer import OpenAIChatCompletionsStreamer
from ..streamer import AsyncStreamerMixin, SyncStreamerMixin


class OpenRouterChatCompletionsStreamer(OpenAIChatCompletionsStreamer):
    PROVIDER = "openrouter"
    URL = "https://openrouter.ai/api/v1"


class SyncOpenRouterStreamer(SyncStreamerMixin, OpenRouterChatCompletionsStreamer):
    pass


class AsyncOpenRouterStreamer(AsyncStreamerMixin, OpenRouterChatCompletionsStreamer):
    pass
