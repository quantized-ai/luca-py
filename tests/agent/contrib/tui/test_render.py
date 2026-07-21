"""Pure formatting helpers."""

from luca.agent.contrib.tui.render import (
    clip_text,
    format_args,
    format_tool_call,
    status_label,
    user_transcript_text,
)
from luca.agent.core.models import (
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    TextContent,
    ToolCall,
)


def test_format_args():
    assert format_args({"a": 1, "path": "/tmp/x"}) == "a=1, path='/tmp/x'"


def test_format_args_empty():
    assert format_args({}) == ""


def test_format_tool_call():
    call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
    assert format_tool_call(call) == "add(a=1, b=2)"


def test_status_label():
    assert status_label(ExecutionStatus.COMPLETED) == "done"
    assert status_label(ExecutionStatus.REJECTED) == "denied"


def test_clip_text_short_is_unchanged():
    assert clip_text("one\ntwo") == "one\ntwo"


def test_clip_text_bounds_lines():
    text = "\n".join(str(i) for i in range(40))
    clipped = clip_text(text, max_lines=3)
    assert clipped.startswith("0\n1\n2\n… (+")
    assert clipped.endswith("more characters)")


def test_clip_text_bounds_chars():
    clipped = clip_text("x" * 5_000)
    assert len(clipped) < 5_000
    assert clipped.endswith("… (+3000 more characters)")


def test_user_transcript_text_renders_images_as_placeholders():
    assert user_transcript_text([
        ImageContent(
            source=ImageBase64(data="aGk=", media_type="image/png"),
            metadata={"name": "receipt.jpg"},
        ),
        TextContent(text="how much did I tip?"),
    ]) == "[image: receipt.jpg]\nhow much did I tip?"


def test_user_transcript_text_falls_back_to_the_media_type():
    assert user_transcript_text([
        ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
    ]) == "[image: image/png]"
