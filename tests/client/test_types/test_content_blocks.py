import pytest
from pydantic import TypeAdapter, ValidationError

from luca.client.types import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    MediaBase64,
    MediaURL,
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    URLCitationAnnotation,
    Usage,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
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


def test_url_citation_annotation_on_text_block():
    b = block_adapter.validate_python(
        {
            "type": "text",
            "text": "cited",
            "annotations": [
                {"type": "url_citation", "url": "https://x", "title": "X", "start_index": 0, "end_index": 5}
            ],
        }
    )
    assert b == TextBlock(
        text="cited",
        annotations=[URLCitationAnnotation(url="https://x", title="X", start_index=0, end_index=5)],
    )


def test_text_block_annotations_default_to_empty():
    assert TextBlock(text="hi").annotations == []


def test_private_provider_block_coerces():
    b = block_adapter.validate_python(
        {
            "type": "private_provider",
            "format": "openai.responses",
            "data": {"type": "web_search_call", "id": "ws_1"},
        }
    )
    assert b == PrivateProviderBlock(format="openai.responses", data={"type": "web_search_call", "id": "ws_1"})


def test_web_search_block_coerces():
    b = block_adapter.validate_python(
        {
            "type": "web_search",
            "queries": ["apple results"],
            "results": [{"type": "web_page", "url": "https://apple.com"}],
        }
    )
    assert b == WebSearchBlock(queries=["apple results"], results=[WebPagePart(url="https://apple.com")])


def test_web_fetch_block_coerces():
    b = block_adapter.validate_python(
        {
            "type": "web_fetch",
            "web_page": {"type": "web_page", "url": "https://apple.com", "title": "Apple", "content": "hi"},
        }
    )
    assert b == WebFetchBlock(web_page=WebPagePart(url="https://apple.com", title="Apple", content="hi"))


def test_a_message_with_web_blocks_round_trips():
    message = AssistantMessage(
        content=[
            PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s1"}),
            WebSearchBlock(queries=["q"], results=[WebPagePart(url="https://x", title="X")]),
            WebFetchBlock(web_page=WebPagePart(url="https://y")),
            TextBlock(
                text="cited",
                annotations=[URLCitationAnnotation(url="https://x", title="X", start_index=0, end_index=5)],
            ),
        ],
    )
    assert AssistantMessage.model_validate(message.model_dump()) == message


def test_usage_tool_fields_default_empty():
    assert Usage().tool_requests == {}
    assert Usage().provider_tool_usage == {}
