"""Polymorphic handlers: one probed file → one content part.

Dispatch is a first-match-wins chain over `HANDLERS`. Every handler answers
`matches(probe)` and builds the part itself, so adding a format is adding a
handler and putting it in the list — nothing in `parse_prompt`, the metadata
contract or the TUI changes. A PDF handler slots in ahead of `BinaryHandler`
and starts producing inlined text on the next run.

Every part carries the same `metadata["mention"]` shape, fields it cannot fill
set to None. The TUI reads it best-effort and must not require any single key.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from luca.agent.core.models import (
    AudioContent,
    ContentPart,
    FileContent,
    ImageContent,
    MediaBase64,
    TextContent,
)
from luca.client.catalog import ModelInfo

from .detection import looks_binary, read_head, sniff

TAG = "agent-prompt-file"

# Status values recorded in metadata and echoed in the tag. `ok` is the only
# one that carries file content; the rest tell the agent to use its own tools.
STATUS_OK = "ok"
STATUS_TOO_LONG = "too_long"
STATUS_BINARY = "binary"
STATUS_DIRECTORY = "directory"
STATUS_UNREADABLE = "unreadable"

_FALL_BACK = "Use your own tools (ranged reads, grep, glob) to satisfy the user's request."


@dataclass(frozen=True)
class ReadLimits:
    """The inline ceiling. `min` of the two knobs, so a small-context model is
    never handed a 25k-token file just because the hard limit allows it.

    `max_media_bytes` is a separate, cruder gate for the formats that go to
    the provider as bytes rather than as text: a token estimate says nothing
    about a PDF or a recording. It is measured on the stat, so an oversized
    file is never read.

    The default is set from the WIRE, not from the file. Base64 inflates by
    4/3, and Anthropic caps the whole request at 32MB — so a 24MB PDF is
    already the entire budget once encoded, before the system prompt, the
    history and the tool declarations. 16MB encodes to ~21MB and leaves room
    for the conversation around it. Raising this above ~24MB cannot work: the
    request is rejected no matter what else is in it."""

    hard_limit: int = 25_000
    context_percentage: float = 0.05
    max_media_bytes: int = 16 * 1024 * 1024

    def max_tokens(self, context_window: int | None = None) -> int:
        if not context_window:
            return self.hard_limit
        return min(self.hard_limit, int(context_window * self.context_percentage))


@dataclass(frozen=True)
class FileProbe:
    """Everything known about a path after one stat and one 8KB read."""

    path: Path  # absolute — the source of truth, never the display form
    is_dir: bool
    size_bytes: int | None
    head: bytes
    mime: str | None
    binary: bool
    error: str | None = None

    @property
    def estimated_tokens(self) -> int | None:
        """The project's standard estimate. Deliberately not a real tokenizer:
        that would be a runtime dependency and model-specific besides."""
        return None if self.size_bytes is None else max(1, self.size_bytes // 4)


def probe(path: Path) -> FileProbe:
    resolved = path.resolve()
    if resolved.is_dir():
        return FileProbe(path=resolved, is_dir=True, size_bytes=None, head=b"", mime=None, binary=False)
    try:
        size = resolved.stat().st_size
        head = read_head(resolved)
    except OSError as exc:
        return FileProbe(
            path=resolved,
            is_dir=False,
            size_bytes=None,
            head=b"",
            mime=None,
            binary=False,
            error=exc.strerror or "unreadable",
        )
    return FileProbe(
        path=resolved,
        is_dir=False,
        size_bytes=size,
        head=head,
        mime=sniff(head),
        binary=looks_binary(head),
    )


def _mention(probe: FileProbe, *, status: str, reason: str | None = None, lines: int | None = None) -> dict:
    """The one metadata shape every handler emits. Unknowable fields are None
    rather than absent, so consumers can read a stable set of keys."""
    return {
        "mention": {
            "path": str(probe.path),
            "status": status,
            "success": status == STATUS_OK,
            "reason": reason,
            "guessed_mime": probe.mime,
            "lines": lines,
            "estimated_tokens": probe.estimated_tokens,
            "bytes": probe.size_bytes,
        }
    }


def _attr(value) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _wrap(body: str, **attributes) -> str:
    """`<agent-prompt-file …>body</agent-prompt-file>`. The closing tag is the
    point: without a terminator, two inlined files run together the moment a
    provider flattens a message's text parts into one string."""
    rendered = " ".join(f'{key}="{_attr(value)}"' for key, value in attributes.items() if value is not None)
    return f"<{TAG} {rendered}>\n{body}\n</{TAG}>"


class PromptFileHandler(Protocol):
    """`model` is the ACTIVE model's catalog record, or None when it is not
    catalogued. It is passed whole rather than reduced to booleans: the record
    is already the project's capability table, so a handler for a new format
    reads the field it needs without widening this signature again."""

    def matches(
        self, probe: FileProbe, limits: ReadLimits, context_window: int | None, model: ModelInfo | None
    ) -> bool: ...

    def build(
        self, probe: FileProbe, limits: ReadLimits, context_window: int | None, model: ModelInfo | None
    ) -> ContentPart: ...


class DirectoryHandler:
    def matches(self, probe, limits, context_window, model) -> bool:
        return probe.is_dir

    def build(self, probe, limits, context_window, model) -> ContentPart:
        reason = "is a directory"
        return TextContent(
            text=_wrap(f"The path is a directory, not a file. {_FALL_BACK}", path=probe.path, status=STATUS_DIRECTORY),
            metadata=_mention(probe, status=STATUS_DIRECTORY, reason=reason),
        )


class UnreadableHandler:
    def matches(self, probe, limits, context_window, model) -> bool:
        return probe.error is not None

    def build(self, probe, limits, context_window, model) -> ContentPart:
        reason = f"unreadable · {probe.error}"
        return TextContent(
            text=_wrap(
                f"The file could not be read ({probe.error}). {_FALL_BACK}",
                path=probe.path,
                status=STATUS_UNREADABLE,
            ),
            metadata=_mention(probe, status=STATUS_UNREADABLE, reason=reason),
        )


class ImageHandler:
    """Images inline as real image content, so a vision model actually sees
    them rather than being told one exists.

    Declines when the catalog positively says the model has no image input —
    a text-only model does not ignore an image part, it rejects the whole
    request (`unknown variant 'image_url', expected 'text'`), so the turn
    fails instead of the file simply not being inlined.

    The test is deliberately NOT symmetrical with `DocumentHandler`'s. An
    UNCATALOGUED model still gets the image, because images have always been
    sent and a local or custom-base-url model reports nothing — declining on
    "unknown" would quietly stop sending images that work today. A document is
    the other way round: it has never been sent, so it goes only on a
    positive yes."""

    def matches(self, probe, limits, context_window, model) -> bool:
        if not (probe.mime or "").startswith("image/"):
            return False
        return model is None or model.supports_image_input

    def build(self, probe, limits, context_window, model) -> ContentPart:
        data = probe.path.read_bytes()
        return ImageContent(
            source=MediaBase64(data=base64.b64encode(data).decode("ascii"), media_type=probe.mime),
            metadata=_mention(probe, status=STATUS_OK),
        )


class AudioHandler:
    """Recordings inline as real audio content, so a model that listens gets
    the sound rather than a note saying a file exists.

    Gated like `DocumentHandler` and for the same reason: audio has never been
    sent, so it goes only on a positive `supports_audio_input`, and an
    uncatalogued model does not get it. Only 27 of the catalogued models take
    audio at all, and only the OpenAI chat-completions wire (OpenRouter
    included) can carry it — everything else raises at projection time, which
    would cost the turn.

    Declines fall through to `BinaryHandler`, whose "not inlined, use your own
    tools" message is still the right answer."""

    def matches(self, probe, limits, context_window, model) -> bool:
        if not (probe.mime or "").startswith("audio/"):
            return False
        if model is None or not model.supports_audio_input:
            return False
        return probe.size_bytes is not None and probe.size_bytes <= limits.max_media_bytes

    def build(self, probe, limits, context_window, model) -> ContentPart:
        data = probe.path.read_bytes()
        return AudioContent(
            source=MediaBase64(data=base64.b64encode(data).decode("ascii"), media_type=probe.mime),
            metadata=_mention(probe, status=STATUS_OK),
        )


class DocumentHandler:
    """PDFs inline as real file content, so a model that reads documents gets
    the document rather than a note saying one exists.

    Declines in two cases, and falls through to `BinaryHandler` in both: the
    model does not advertise PDF input, or the file is over
    `limits.max_media_bytes`. Declining is the whole reason this sits above
    `BinaryHandler` rather than replacing it — the old message is still the
    right answer when the document cannot be sent."""

    MEDIA_TYPES = ("application/pdf",)

    def matches(self, probe, limits, context_window, model) -> bool:
        if (probe.mime or "") not in self.MEDIA_TYPES:
            return False
        if model is None or not model.supports_pdf_input:
            return False
        return probe.size_bytes is not None and probe.size_bytes <= limits.max_media_bytes

    def build(self, probe, limits, context_window, model) -> ContentPart:
        data = probe.path.read_bytes()
        return FileContent(
            source=MediaBase64(data=base64.b64encode(data).decode("ascii"), media_type=probe.mime),
            name=probe.path.name,
            metadata=_mention(probe, status=STATUS_OK),
        )


class BinaryHandler:
    """The day-one fallback for every non-image binary. A format that deserves
    better gets its own handler ABOVE this one; nothing else has to change."""

    def matches(self, probe, limits, context_window, model) -> bool:
        return probe.binary or probe.mime is not None

    def build(self, probe, limits, context_window, model) -> ContentPart:
        reason = "can't read binary files"
        return TextContent(
            text=_wrap(
                f"The file is binary and was not inlined. {_FALL_BACK}",
                path=probe.path,
                status=STATUS_BINARY,
                guessed_mime=probe.mime,
                bytes=probe.size_bytes,
            ),
            metadata=_mention(probe, status=STATUS_BINARY, reason=reason),
        )


class TextHandler:
    """The catch-all: inline under the cap, decline over it. Declining is not a
    failure — the agent is told to grep or read ranges instead."""

    def matches(self, probe, limits, context_window, model) -> bool:
        return True

    def build(self, probe, limits, context_window, model) -> ContentPart:
        cap = limits.max_tokens(context_window)
        estimated = probe.estimated_tokens or 0
        if estimated > cap:
            # Rejected on the stat alone — the file is never read into memory.
            return TextContent(
                text=_wrap(
                    f"The file is too long to inline (limit {cap} estimated tokens). {_FALL_BACK}",
                    path=probe.path,
                    status=STATUS_TOO_LONG,
                    estimated_tokens=estimated,
                    bytes=probe.size_bytes,
                ),
                metadata=_mention(probe, status=STATUS_TOO_LONG, reason="file too long"),
            )
        try:
            body = probe.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failed = FileProbe(**{**probe.__dict__, "error": exc.strerror or "unreadable"})
            return UnreadableHandler().build(failed, limits, context_window, model)
        lines = body.count("\n") + (0 if body.endswith("\n") or not body else 1)
        return TextContent(
            text=_wrap(
                body,
                path=probe.path,
                status=STATUS_OK,
                lines=lines,
                estimated_tokens=estimated,
                bytes=probe.size_bytes,
            ),
            metadata=_mention(probe, status=STATUS_OK, lines=lines),
        )


# First match wins. Order is the policy: a new format goes above BinaryHandler.
HANDLERS: tuple[PromptFileHandler, ...] = (
    DirectoryHandler(),
    UnreadableHandler(),
    ImageHandler(),
    AudioHandler(),
    DocumentHandler(),
    BinaryHandler(),
    TextHandler(),
)
