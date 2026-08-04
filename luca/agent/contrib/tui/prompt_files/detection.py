"""Is this file text or binary, and if binary, what is it?

Stdlib only — no `libmagic`, no `python-magic`. Detection reads one 8KB head
buffer and answers both questions from it, so nothing reads a whole file
before the harness has decided it is worth reading.

`sniff` matches magic numbers, never extensions: an extension-based classifier
mislabels text files with unusual suffixes, which is a live defect in other
harnesses. `mimetypes` is fine for a LABEL once the type is already known, but
it is not detection. `imghdr`/`sndhdr` are not options at all — removed in
Python 3.13.
"""

from __future__ import annotations

import codecs
from pathlib import Path

HEAD_BYTES = 8192

# Ordered longest-first where prefixes could collide. Every entry is a byte
# signature at offset 0; anything at another offset is special-cased in sniff().
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # also docx/xlsx/pptx/jar/epub
    (b"\x1f\x8b", "application/gzip"),
    (b"BZh", "application/x-bzip2"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "application/vnd.rar"),
    (b"\x7fELF", "application/x-elf"),
    (b"\xca\xfe\xba\xbe", "application/java-vm"),
    (b"SQLite format 3\x00", "application/vnd.sqlite3"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"),
)


def read_head(path: Path, size: int = HEAD_BYTES) -> bytes:
    """The first `size` bytes. Both questions are answered from this one read."""
    with path.open("rb") as handle:
        return handle.read(size)


def sniff(head: bytes) -> str | None:
    """The media type from the file's magic number, or None when nothing
    matches — which means "possibly text", not "unknown binary"."""
    # WebP's marker sits at offset 8, so it is not a prefix match. Any
    # startswith-only table silently misses it.
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    for signature, media_type in _MAGIC:
        if head.startswith(signature):
            return media_type
    return None


def looks_binary(head: bytes) -> bool:
    """Git's rule (a NUL byte in the first 8KB) plus a decodability check that
    catches non-UTF-8 text, which has no NULs but would still garble context.

    UTF-16/UTF-32 are full of NULs and therefore read as binary. That is a
    known, accepted false positive: BOM-sniffing them is possible but they are
    not worth inlining anyway.
    """
    if not head:
        return False  # an empty file is an empty text file
    if b"\x00" in head:
        return True
    try:
        # `final=False` buffers a multi-byte character split by the read
        # boundary instead of raising. A plain head.decode("utf-8") reports
        # perfectly good UTF-8 as binary whenever the cut lands mid-character.
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        return True
    return False
