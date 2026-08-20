"""`SUPPORTS_AUDIO_INPUT` must match what the transport actually does.

The flag exists so a caller can ask before building a request; a flag that
disagrees with the projection is worse than no flag, because it turns a clean
refusal into a lost turn. This asserts the two together, so neither can move
without the other.
"""

from luca.client.exceptions import BadRequestError
from luca.client.transports import (
    AnthropicTransport,
    BedrockTransport,
    OpenAIResponsesTransport,
    OpenAITransport,
    OpenRouterTransport,
)
from luca.client.types import AudioBlock, MediaBase64
from luca.client.types.messages import UserMessage

AUDIO = AudioBlock(source=MediaBase64(data="SUQzBAAAAAAA", media_type="audio/mpeg"))

TRANSPORTS = [
    OpenAITransport(provider="openai", base_url="https://api.openai.com/v1", api_key="k"),
    OpenRouterTransport(provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key="k"),
    OpenAIResponsesTransport(provider="openai", base_url="https://api.openai.com/v1", api_key="k"),
    AnthropicTransport(provider="anthropic", base_url="https://api.anthropic.com", api_key="k"),
    BedrockTransport(provider="bedrock", base_url="https://bedrock-runtime.us-east-1.amazonaws.com", api_key="k"),
]


def _projects_audio(transport) -> bool:
    """Each transport exposes a different door onto the same step."""
    message = UserMessage(content=[AUDIO])
    try:
        if hasattr(transport, "_project_user_block"):
            transport._project_user_block(AUDIO)
        elif hasattr(transport, "_project_user_content"):
            transport._project_user_content(message)
        else:
            transport._project_user_message(message)
    except BadRequestError:
        return False
    return True


def test_the_declared_flag_matches_the_projection_on_every_transport():
    assert {type(t).__name__: (t.SUPPORTS_AUDIO_INPUT, _projects_audio(t)) for t in TRANSPORTS} == {
        "OpenAITransport": (True, True),
        "OpenRouterTransport": (True, True),
        "OpenAIResponsesTransport": (False, False),
        "AnthropicTransport": (False, False),
        "BedrockTransport": (False, False),
    }
