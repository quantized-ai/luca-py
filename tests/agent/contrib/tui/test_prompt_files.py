"""`@`-mention resolution: a path (or a whole prompt) in, content parts out.

Every assertion is on the WHOLE part — the wrapped text and the full metadata
dict together — so the tag format and the metadata contract are both pinned
here. A handler that changes either shows up as a diff.
"""

import base64

import pytest

from luca.agent.contrib.tui.prompt_files import (
    ReadLimits,
    find_mentions,
    looks_binary,
    parse_prompt,
    process_prompt_file_path,
    sniff,
)
from luca.agent.core.models import ImageContent, MediaBase64, TextContent

# a 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TINY = ReadLimits(hard_limit=10, context_percentage=0.05)


def mention(path, **overrides) -> dict:
    """The full metadata shape — every key present, unknowable ones None."""
    base = {
        "path": str(path),
        "status": "ok",
        "success": True,
        "reason": None,
        "guessed_mime": None,
        "lines": None,
        "estimated_tokens": None,
        "bytes": None,
    }
    return {"mention": {**base, **overrides}}


# ── the handler chain, one file at a time ─────────────────────────────────────


def test_a_text_file_is_inlined_inside_the_tag(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("hello\nworld\n")

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="ok" lines="2" estimated_tokens="3" bytes="12">\n'
            "hello\nworld\n\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(path, lines=2, estimated_tokens=3, bytes=12),
    )


def test_a_text_file_over_the_cap_is_declined_and_never_read(tmp_path):
    path = tmp_path / "package-lock.json"
    path.write_text("x" * 400)  # 100 estimated tokens, cap is 10

    assert process_prompt_file_path(path, limits=TINY) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="too_long" estimated_tokens="100" bytes="400">\n'
            "The file is too long to inline (limit 10 estimated tokens). "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="too_long",
            success=False,
            reason="file too long",
            estimated_tokens=100,
            bytes=400,
        ),
    )


def test_an_image_becomes_image_content_not_a_rejection(tmp_path):
    path = tmp_path / "logo.png"
    path.write_bytes(PNG)

    assert process_prompt_file_path(path) == ImageContent(
        source=MediaBase64(data=base64.b64encode(PNG).decode("ascii"), media_type="image/png"),
        metadata=mention(path, guessed_mime="image/png", estimated_tokens=len(PNG) // 4, bytes=len(PNG)),
    )


def test_a_non_image_binary_is_declined_with_its_guessed_type(tmp_path):
    blob = b"%PDF-1.4\n\x00\x01binary junk"
    path = tmp_path / "doc.pdf"
    path.write_bytes(blob)

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="binary" guessed_mime="application/pdf" '
            f'bytes="{len(blob)}">\n'
            "The file is binary and was not inlined. "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="binary",
            success=False,
            reason="can't read binary files",
            guessed_mime="application/pdf",
            estimated_tokens=len(blob) // 4,
            bytes=len(blob),
        ),
    )


def test_a_binary_without_a_known_signature_is_still_declined(tmp_path):
    path = tmp_path / "blob.dat"
    path.write_bytes(b"\x00\x01\x02\x03")

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="binary" bytes="4">\n'
            "The file is binary and was not inlined. "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="binary",
            success=False,
            reason="can't read binary files",
            estimated_tokens=1,
            bytes=4,
        ),
    )


def test_a_directory_is_declined(tmp_path):
    path = tmp_path / "handoff"
    path.mkdir()

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="directory">\n'
            "The path is a directory, not a file. "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(path, status="directory", success=False, reason="is a directory"),
    )


def test_an_empty_file_is_text_not_binary(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("")

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="ok" lines="0" estimated_tokens="1" bytes="0">'
            "\n\n</agent-prompt-file>"
        ),
        metadata=mention(path, lines=0, estimated_tokens=1, bytes=0),
    )


# ── the cap ───────────────────────────────────────────────────────────────────


def test_the_cap_is_the_smaller_of_the_hard_limit_and_the_context_share():
    limits = ReadLimits(hard_limit=25_000, context_percentage=0.05)

    assert [
        limits.max_tokens(None),  # unknown window: the hard limit stands
        limits.max_tokens(128_000),  # small window: 5% wins, 25k would be too big
        limits.max_tokens(1_000_000),  # large window: the hard limit still caps it
    ] == [25_000, 6_400, 25_000]


# ── detection ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),  # marker at offset 8, not a prefix
        (b"%PDF-1.7", "application/pdf"),
        (b"PK\x03\x04\x14\x00", "application/zip"),
        (b"# just markdown", None),
    ],
)
def test_sniff_reads_magic_numbers_not_extensions(head, expected):
    assert sniff(head) == expected


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"", False),  # empty is an empty text file
        (b"plain ascii", False),
        ("héllo wörld".encode(), False),
        (b"\x00\x01", True),  # NUL — git's rule
        (b"\xff\xfe\xfd invalid utf-8", True),
    ],
)
def test_looks_binary(head, expected):
    assert looks_binary(head) is expected


def test_a_multibyte_character_split_by_the_read_boundary_is_not_binary():
    # "€" is three bytes; cutting after two must not report valid UTF-8 as
    # binary, which a naive head.decode("utf-8") would.
    assert looks_binary("a€".encode()[:-1]) is False


# ── parsing a whole prompt ────────────────────────────────────────────────────


def test_parse_prompt_returns_the_text_verbatim_then_one_part_per_resolved_file(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("hi\n")
    prompt = "Read @README.md and @NONEXISTENT.md"

    assert parse_prompt(prompt, workspace=tmp_path) == [
        TextContent(text="Read @README.md and @NONEXISTENT.md"),
        TextContent(
            text=(
                f'<agent-prompt-file path="{readme}" status="ok" lines="1" estimated_tokens="1" bytes="3">\n'
                "hi\n\n"
                "</agent-prompt-file>"
            ),
            metadata=mention(readme, lines=1, estimated_tokens=1, bytes=3),
        ),
    ]


def test_prose_that_looks_like_a_mention_is_left_alone(tmp_path):
    # @property, @staticmethod, @media, @types/node — nothing on disk answers
    # to them, so they stay prose and produce no parts
    prompt = "use @property not @staticmethod, and check @types/node for @media rules"

    assert parse_prompt(prompt, workspace=tmp_path) == [TextContent(text=prompt)]


def test_an_email_address_is_never_a_mention(tmp_path):
    (tmp_path / "bar.com").write_text("x")

    assert find_mentions("mail foo@bar.com", tmp_path) == []


def test_comma_joined_mentions_from_the_picker_each_resolve(tmp_path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")

    assert find_mentions("compare @a.py,@b.py now", tmp_path) == [tmp_path / "a.py", tmp_path / "b.py"]


def test_trailing_sentence_punctuation_is_not_part_of_the_path(tmp_path):
    (tmp_path / "a.py").write_text("a")

    assert find_mentions("look at @a.py.", tmp_path) == [tmp_path / "a.py"]


def test_the_same_file_mentioned_twice_is_inlined_once(tmp_path):
    (tmp_path / "a.py").write_text("a")

    assert find_mentions("@a.py and again @a.py", tmp_path) == [tmp_path / "a.py"]


def test_a_mention_must_start_at_a_word_boundary(tmp_path):
    (tmp_path / "a.py").write_text("a")

    assert find_mentions("no@a.py here", tmp_path) == []
