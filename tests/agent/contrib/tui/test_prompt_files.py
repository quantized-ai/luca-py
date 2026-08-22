"""`@`-mention resolution: a path (or a whole prompt) in, content parts out.

Every assertion is on the WHOLE part — the wrapped text and the full metadata
dict together — so the tag format and the metadata contract are both pinned
here. A handler that changes either shows up as a diff.
"""

import base64
import math

import pytest

from luca.agent.contrib.app.prompt_files import (
    ReadLimits,
    find_mentions,
    get_model_info,
    looks_binary,
    parse_prompt,
    process_prompt_file_path,
    sniff,
)
from luca.agent.core.models import AudioContent, FileContent, ImageContent, MediaBase64, TextContent
from luca.client.catalog import ModelInfo

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


def test_a_non_media_binary_is_declined_with_its_guessed_type(tmp_path):
    # a zip, not a PDF: attachable media routes to UnsupportedMediaHandler now,
    # so a document no longer stands in for "some binary"
    blob = b"PK\x03\x04\x14\x00\x00\x08binary junk"
    path = tmp_path / "bundle.zip"
    path.write_bytes(blob)

    assert process_prompt_file_path(path) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="binary" guessed_mime="application/zip" '
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
            guessed_mime="application/zip",
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


# ── documents ────────────────────────────────────────────────────────────────

PDF = b"%PDF-1.4\n\x00\x01binary junk"
READS_PDF = ModelInfo(supports_pdf_input=True)
NO_PDF = ModelInfo(model="gpt-mini", supports_pdf_input=False)


def test_a_pdf_becomes_file_content_when_the_model_reads_pdfs(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF)

    assert process_prompt_file_path(path, model=READS_PDF) == FileContent(
        source=MediaBase64(data=base64.b64encode(PDF).decode("ascii"), media_type="application/pdf"),
        name="report.pdf",
        metadata=mention(
            path,
            guessed_mime="application/pdf",
            estimated_tokens=len(PDF) // 4,
            bytes=len(PDF),
        ),
    )


def test_a_pdf_names_the_model_when_it_cannot_read_documents(tmp_path):
    # a document keeps the tool advice — text can still be pulled out of it —
    # but the reason now says which model refused rather than "binary"
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF)

    assert process_prompt_file_path(path, model=NO_PDF) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="unsupported" guessed_mime="application/pdf" '
            f'bytes="{len(PDF)}">\n'
            "This is document content and it was NOT attached: gpt-mini does not accept document input. "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="unsupported",
            success=False,
            reason="gpt-mini does not accept document input",
            guessed_mime="application/pdf",
            estimated_tokens=len(PDF) // 4,
            bytes=len(PDF),
        ),
    )


def test_an_oversized_pdf_says_it_is_too_large_and_is_never_read_into_memory(tmp_path):
    path = tmp_path / "huge.pdf"
    path.write_bytes(PDF)

    part = process_prompt_file_path(path, model=READS_PDF, limits=ReadLimits(max_media_bytes=4))

    assert part == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="too_large" guessed_mime="application/pdf" '
            f'bytes="{len(PDF)}">\n'
            "This is document content and it was NOT attached: it is over the 4-byte limit for attached media. "
            "Use your own tools (ranged reads, grep, glob) to satisfy the user's request.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="too_large",
            success=False,
            reason="over the 4-byte limit for attached media",
            guessed_mime="application/pdf",
            estimated_tokens=len(PDF) // 4,
            bytes=len(PDF),
        ),
    )


def test_the_catalog_record_is_what_gates_a_document():
    catalogued = get_model_info("anthropic", "claude-sonnet-5")

    assert (catalogued.supports_pdf_input, get_model_info("nowhere", "made-up"), get_model_info(None, None)) == (
        True,
        None,
        None,
    )


NO_IMAGES = ModelInfo(model="gpt-mini", supports_image_input=False)
READS_IMAGES = ModelInfo(supports_image_input=True)


def test_an_image_is_withheld_from_a_model_the_catalog_says_is_text_only(tmp_path):
    # deepseek-chat rejects the whole request with `unknown variant
    # 'image_url', expected 'text'`, so the turn fails rather than the image
    # simply being ignored
    path = tmp_path / "logo.png"
    path.write_bytes(PNG)

    assert process_prompt_file_path(path, model=NO_IMAGES) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="unsupported" guessed_mime="image/png" '
            f'bytes="{len(PNG)}">\n'
            "This is image content and it was NOT attached: gpt-mini does not accept image input. "
            "Tell the user to switch to a model that does. "
            "Do not try to read, decode or transcribe it with your tools; that cannot work.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="unsupported",
            success=False,
            reason="gpt-mini does not accept image input",
            guessed_mime="image/png",
            estimated_tokens=len(PNG) // 4,
            bytes=len(PNG),
        ),
    )


def test_an_uncatalogued_model_still_gets_the_image(tmp_path):
    # asymmetric with documents on purpose: images have always been sent, and
    # a local or custom-base-url model reports nothing, so declining on
    # "unknown" would stop sending images that work today
    path = tmp_path / "logo.png"
    path.write_bytes(PNG)

    assert process_prompt_file_path(path, model=None) == ImageContent(
        source=MediaBase64(data=base64.b64encode(PNG).decode("ascii"), media_type="image/png"),
        metadata=mention(path, guessed_mime="image/png", estimated_tokens=len(PNG) // 4, bytes=len(PNG)),
    )


def test_a_vision_model_still_gets_the_image(tmp_path):
    path = tmp_path / "logo.png"
    path.write_bytes(PNG)

    assert process_prompt_file_path(path, model=READS_IMAGES) == ImageContent(
        source=MediaBase64(data=base64.b64encode(PNG).decode("ascii"), media_type="image/png"),
        metadata=mention(path, guessed_mime="image/png", estimated_tokens=len(PNG) // 4, bytes=len(PNG)),
    )


# ── audio ────────────────────────────────────────────────────────────────────

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90d"
HEARS_AUDIO = ModelInfo(provider="openrouter", model="hears", supports_audio_input=True)
NO_AUDIO = ModelInfo(provider="openrouter", model="gpt-mini", supports_audio_input=False)


def test_a_recording_becomes_audio_content_when_the_model_listens(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3)

    assert process_prompt_file_path(path, model=HEARS_AUDIO) == AudioContent(
        source=MediaBase64(data=base64.b64encode(MP3).decode("ascii"), media_type="audio/mpeg"),
        metadata=mention(
            path,
            guessed_mime="audio/mpeg",
            estimated_tokens=len(MP3) // 4,
            bytes=len(MP3),
        ),
    )


def test_a_recording_names_the_model_when_it_cannot_hear(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3)

    assert process_prompt_file_path(path, model=NO_AUDIO) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="unsupported" guessed_mime="audio/mpeg" '
            f'bytes="{len(MP3)}">\n'
            "This is audio content and it was NOT attached: gpt-mini does not accept audio input. "
            "Tell the user to switch to a model that does. "
            "Do not try to read, decode or transcribe it with your tools; that cannot work.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="unsupported",
            success=False,
            reason="gpt-mini does not accept audio input",
            guessed_mime="audio/mpeg",
            estimated_tokens=len(MP3) // 4,
            bytes=len(MP3),
        ),
    )


def test_an_uncatalogued_model_does_not_get_the_recording(tmp_path):
    # symmetric with documents, not with images: audio has never been sent, and
    # only the chat-completions wire can carry it at all
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3)

    assert process_prompt_file_path(path, model=None) == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="unsupported" guessed_mime="audio/mpeg" '
            f'bytes="{len(MP3)}">\n'
            "This is audio content and it was NOT attached: the model in this conversation does not accept audio input. "
            "Tell the user to switch to a model that does. "
            "Do not try to read, decode or transcribe it with your tools; that cannot work.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="unsupported",
            success=False,
            reason="audio needs a model known to accept it",
            guessed_mime="audio/mpeg",
            estimated_tokens=len(MP3) // 4,
            bytes=len(MP3),
        ),
    )


def test_an_oversized_recording_says_it_is_too_large_and_is_never_read(tmp_path):
    path = tmp_path / "long.mp3"
    path.write_bytes(MP3)

    part = process_prompt_file_path(path, model=HEARS_AUDIO, limits=ReadLimits(max_media_bytes=4))

    assert part == TextContent(
        text=(
            f'<agent-prompt-file path="{path}" status="too_large" guessed_mime="audio/mpeg" '
            f'bytes="{len(MP3)}">\n'
            "This is audio content and it was NOT attached: it is over the 4-byte limit for attached media. "
            "Tell the user the file is too large to send. "
            "Do not try to read, decode or transcribe it with your tools; that cannot work.\n"
            "</agent-prompt-file>"
        ),
        metadata=mention(
            path,
            status="too_large",
            success=False,
            reason="over the 4-byte limit for attached media",
            guessed_mime="audio/mpeg",
            estimated_tokens=len(MP3) // 4,
            bytes=len(MP3),
        ),
    )


def test_an_audio_model_whose_host_cannot_carry_audio_is_declined(tmp_path):
    # `supports_audio_input` is per MODEL and cannot see the wire. These three
    # are catalogued as audio models on hosts that route to a transport with no
    # audio shape, so building the part would cost the turn at projection.
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3)
    stranded = [
        get_model_info("openai", "gpt-realtime-2.1"),
        get_model_info("bedrock", "mistral.voxtral-mini-3b-2507"),
        get_model_info("bedrock", "mistral.voxtral-small-24b-2507"),
    ]

    assert [(m.supports_audio_input, type(process_prompt_file_path(path, model=m)).__name__) for m in stranded] == [
        (True, "TextContent"),
        (True, "TextContent"),
        (True, "TextContent"),
    ]


def test_the_wire_refusal_blames_the_host_not_the_model(tmp_path):
    # blaming the model would send the user to change the wrong thing: the
    # model does hear, it is the host's API that has no audio part
    path = tmp_path / "clip.mp3"
    path.write_bytes(MP3)

    part = process_prompt_file_path(path, model=get_model_info("openai", "gpt-realtime-2.1"))

    assert part.metadata["mention"]["reason"] == "GPT-Realtime-2.1 takes audio, but the openai wire cannot carry it"


def test_the_audio_formats_the_wire_takes_are_all_sniffable():
    # a format the sniffer misses reaches BinaryHandler and is never sent, so
    # detection and the transport's format table have to agree
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
    m4a = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00"

    assert (
        sniff(wav),
        sniff(m4a),
        sniff(b"\xff\xfb\x90d"),  # mp3 with no ID3 tag
        sniff(b"ID3\x04\x00"),
        sniff(b"OggS\x00\x02"),
        sniff(b"fLaC\x00\x00"),
        sniff(b"\xff\xf1X@"),  # ADTS aac, no CRC
    ) == (
        "audio/wav",
        "audio/mp4",
        "audio/mpeg",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
    )


def test_every_spelling_of_the_mpeg_sync_word_is_read_not_matched():
    # real encoders disagree on the bits after the sync word: afconvert writes
    # ADTS with a CRC (\xff\xf9) where OpenAI's writes none (\xff\xf1), and a
    # literal table missed the first. The LAYER bits are what pick mp3 vs aac.
    assert (
        sniff(b"\xff\xf9\x5c\x60"),  # ADTS with CRC
        sniff(b"\xff\xf8\x5c\x60"),  # ADTS, MPEG-4 no CRC
        sniff(b"\xff\xfa\x90\x64"),  # mp3, MPEG-1 layer III with CRC
        sniff(b"\xff\xf3\xc4\xc4"),  # mp3, MPEG-2 layer III
        sniff(b"\xff\xe3\x18\xc4"),  # mp3, MPEG-2.5 layer III
    ) == ("audio/aac", "audio/aac", "audio/mpeg", "audio/mpeg", "audio/mpeg")


def test_a_binary_that_merely_starts_with_the_sync_bits_is_not_called_audio():
    # the reserved encodings are what separate a real frame header from any
    # binary opening \xff\xe…; without them this would claim arbitrary files
    assert (
        sniff(b"\xff\xeb\x90\x64"),  # reserved MPEG version
        sniff(b"\xff\xfb\xf0\x00"),  # layer III, invalid bitrate index
        sniff(b"\xff\xf1\x3c\x40"),  # ADTS, invalid sampling-frequency index
        sniff(b"\xff\xfd\x90\x64"),  # reserved layer
    ) == (None, None, None, None)


def test_the_document_ceiling_leaves_room_once_base64_encoded():
    # the ceiling is a WIRE budget, not a file size: base64 inflates by 4/3
    # and Anthropic caps the whole request at 32MB, so a file at the ceiling
    # must still leave room for the prompt, history and tool declarations
    ceiling = ReadLimits().max_media_bytes
    encoded = math.ceil(ceiling / 3) * 4
    anthropic_request_cap = 32 * 1024 * 1024

    assert encoded < anthropic_request_cap, f"{encoded} bytes encoded exceeds the 32MB request cap"
    assert encoded <= anthropic_request_cap * 0.7, "no headroom left for the rest of the request"
