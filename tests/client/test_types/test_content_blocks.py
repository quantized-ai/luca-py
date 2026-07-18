import pytest
from pydantic import TypeAdapter, ValidationError

from luca.client.types import (
    ContentBlock,
    ImageBlock,
    MediaBase64,
    MediaURL,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
)


block_adapter = TypeAdapter(ContentBlock)


def test_text_block_coerces():
    b = block_adapter.validate_python({"type": "text", "text": "hello"})
    assert isinstance(b, TextBlock)
    assert b.text == "hello"


def test_thinking_block_coerces():
    b = block_adapter.validate_python({"type": "thinking", "text": "think"})
    assert isinstance(b, ThinkingBlock)


def test_image_block_with_url_source():
    b = block_adapter.validate_python(
        {"type": "image", "source": {"kind": "url", "url": "https://x/y.png"}},
    )
    assert isinstance(b, ImageBlock)
    assert isinstance(b.source, MediaURL)


def test_image_block_with_base64_source_requires_media_type():
    with pytest.raises(ValidationError):
        block_adapter.validate_python(
            {"type": "image", "source": {"kind": "base64", "data": "abc"}},
        )


def test_image_block_with_base64_source():
    b = block_adapter.validate_python(
        {
            "type": "image",
            "source": {"kind": "base64", "data": "abc", "media_type": "image/png"},
        },
    )
    assert isinstance(b.source, MediaBase64)
    assert b.source.media_type == "image/png"


def test_tool_call_defaults():
    tc = ToolCall(id="c", name="get_weather")
    assert tc.arguments == {}
    assert tc.partial_arguments == ""
    assert tc.complete is True


def test_refusal_block_coerces():
    b = block_adapter.validate_python({"type": "refusal", "text": "no"})
    assert isinstance(b, RefusalBlock)
