"""No transport may fall back to `str(block)` for a content block.

Every transport used to end `_project_user_block` with
`{"type": "text", "text": str(block)}`. For a `FileBlock` that is the Pydantic
repr — which embeds the ENTIRE base64 payload — so a PDF went to the model as
several hundred KB of text, billed as input tokens and unreadable as a
document. Nothing constructed a `FileBlock` yet, so nothing caught it.

These tests pin the two halves of the fix: a supported block projects to a
real wire shape, and an unsupported one raises instead of degrading.
"""

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports import (
    AnthropicTransport,
    BedrockTransport,
    OpenAIResponsesTransport,
    OpenAITransport,
)
from luca.client.types import AudioBlock, FileBlock, MediaBase64
from luca.client.types.messages import UserMessage

PDF = MediaBase64(data="JVBERi0xLjQKJcOkw7zDtsOfCg==", media_type="application/pdf")
AUDIO = MediaBase64(data="SUQzBAAAAAAA", media_type="audio/mpeg")


def _transports():
    return [
        OpenAITransport(provider="openai", base_url="https://api.openai.com/v1", api_key="k"),
        OpenAIResponsesTransport(provider="openai", base_url="https://api.openai.com/v1", api_key="k"),
        AnthropicTransport(provider="anthropic", base_url="https://api.anthropic.com", api_key="k"),
        BedrockTransport(provider="bedrock", base_url="https://bedrock-runtime.us-east-1.amazonaws.com", api_key="k"),
    ]


def _payloads(transport, blocks):
    """Every transport exposes a different door onto the same step."""
    if hasattr(transport, "_project_user_block"):
        return [transport._project_user_block(block) for block in blocks]
    if hasattr(transport, "_project_user_content"):
        return transport._project_user_content(UserMessage(content=list(blocks)))
    return transport._project_user_message(UserMessage(content=list(blocks)))["content"]


def test_a_file_never_reaches_the_wire_as_a_python_repr():
    block = FileBlock(source=PDF, name="report.pdf")

    for transport in _transports():
        rendered = repr(_payloads(transport, [block]))

        # the tells of the old fallback: the class name and the field syntax
        assert "MediaBase64(" not in rendered, transport.transport_id
        assert "kind='base64'" not in rendered, transport.transport_id
        assert "type='file'" not in rendered, transport.transport_id


def test_audio_is_refused_by_every_transport_that_has_no_shape_for_it():
    # chat completions is the only wire with an audio input part; Anthropic and
    # Bedrock Converse have no audio content block, and the Responses API's
    # own guide sends callers to chat completions for audio
    block = AudioBlock(source=AUDIO)

    for transport in _transports():
        if isinstance(transport, OpenAITransport):
            continue
        with pytest.raises(BadRequestError, match="AudioBlock"):
            _payloads(transport, [block])


def test_audio_reaches_chat_completions_as_a_real_wire_shape():
    transport = OpenAITransport(provider="openai", base_url="https://api.openai.com/v1", api_key="k")

    assert _payloads(transport, [AudioBlock(source=AUDIO)]) == [
        {"type": "input_audio", "input_audio": {"data": "SUQzBAAAAAAA", "format": "mp3"}}
    ]
