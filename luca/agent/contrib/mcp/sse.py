"""An incremental Server-Sent Events decoder. Pure, no I/O.

A Streamable HTTP server answers a POST with either one JSON object or an SSE
stream carrying progress and log notifications followed by the final response.
So the client needs a decoder, and it needs one that survives arbitrary chunk
boundaries: an event can be split anywhere, including between the CR and the LF
of its own line terminator.

Keeping it byte-oriented and separate from the transport is what makes that
testable. The same stream fed whole, fed one byte at a time, and fed in chunks
that straddle every frame must produce the same events, and that is a test with
no server, no socket and no clock in it.

Deliberately NOT handled: `Last-Event-ID` resumption. The 2026-07-28 revision
removed stream resumability, so `id:` is parsed and carried but never acted on;
a broken stream loses the in-flight request and the caller re-issues it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SseEvent:
    """One dispatched event. `name` defaults to "message" per the SSE spec,
    which is what MCP servers send for an unnamed frame."""

    data: str
    name: str = "message"
    id: str | None = None
    retry: int | None = None


@dataclass
class SseDecoder:
    """Feed it bytes, take events out.

    One instance decodes one response stream and is not reusable, which is the
    same rule the buffer itself implies.
    """

    _buffer: bytes = b""
    _data: list[str] = field(default_factory=list)
    _name: str = ""
    _id: str | None = None
    _retry: int | None = None

    def feed(self, chunk: bytes) -> list[SseEvent]:
        """Every event completed by this chunk, in order. Usually empty."""
        self._buffer += chunk
        events: list[SseEvent] = []
        while (line := self._take_line()) is not None:
            event = self._consume(line)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> list[SseEvent]:
        """Any event left buffered when the stream ends without a final blank
        line. The SSE spec says to discard it; we dispatch it instead, because
        a server that closes right after writing the final response is common
        and throwing that response away would turn a completed tool call into a
        transport failure."""
        pending = self._dispatch()
        self._buffer = b""
        return [pending] if pending is not None else []

    def _take_line(self) -> bytes | None:
        """One line, or None when the buffer does not hold a complete one.

        Handles all three terminators. A trailing lone CR is held back rather
        than treated as a terminator, because the LF that would make it a CRLF
        may be the first byte of the next chunk.
        """
        buffer = self._buffer
        newline = buffer.find(b"\n")
        carriage = buffer.find(b"\r")
        if newline == -1 and carriage == -1:
            return None
        if carriage != -1 and (newline == -1 or carriage < newline):
            if carriage == len(buffer) - 1:
                return None
            line = buffer[:carriage]
            end = carriage + (2 if buffer[carriage + 1 : carriage + 2] == b"\n" else 1)
        else:
            line = buffer[:newline]
            end = newline + 1
        self._buffer = buffer[end:]
        return line

    def _consume(self, raw: bytes) -> SseEvent | None:
        """Apply one line. Returns an event only when the line ends a frame."""
        if not raw:
            return self._dispatch()
        line = raw.decode("utf-8", errors="replace")
        if line.startswith(":"):
            return None  # a comment, which servers also use as a keep-alive
        name, separator, value = line.partition(":")
        if not separator:
            name, value = line, ""
        if value.startswith(" "):
            value = value[1:]
        match name:
            case "data":
                self._data.append(value)
            case "event":
                self._name = value
            case "id":
                # NUL is the one value the SSE spec says to ignore outright.
                if "\x00" not in value:
                    self._id = value
            case "retry":
                if value.isdigit():
                    self._retry = int(value)
        return None

    def _dispatch(self) -> SseEvent | None:
        """Emit the buffered frame and reset. A frame with no data lines is not
        an event: that is what a stream of bare keep-alives decodes to."""
        if not self._data:
            self._name = ""
            return None
        event = SseEvent(
            data="\n".join(self._data),
            name=self._name or "message",
            id=self._id,
            retry=self._retry,
        )
        self._data = []
        self._name = ""
        return event
