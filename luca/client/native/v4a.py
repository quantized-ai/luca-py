"""The V4A diff grammar carried by OpenAI's `apply_patch` operations.

Pure text in, pure text out — no filesystem, no IO. `apply_diff` IS the
meaning of a `create_file` / `update_file` operation's `diff` field.

Adapted from openai-agents-python,
https://github.com/openai/openai-agents-python/blob/main/src/agents/apply_diff.py
— restyled, but the behavior is reproduced verbatim: context lookup, the
three fuzz passes, `*** End of File`, newline detection. The models were
trained against these semantics; do not "improve" them.

The notice below covers that borrowed portion, which is everything in this
module; the rest of `luca` is under the repository's own LICENSE.

    MIT License

    Copyright (c) 2025 OpenAI

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import NativeToolError

ApplyDiffMode = Literal["default", "create"]

END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]


@dataclass
class Chunk:
    """One replacement: `del_lines` at `orig_index` become `ins_lines`."""

    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]


@dataclass
class ParserState:
    lines: list[str]
    index: int = 0
    fuzz: int = 0


@dataclass
class ParsedUpdateDiff:
    chunks: list[Chunk]
    fuzz: int


@dataclass
class ReadSectionResult:
    next_context: list[str]
    section_chunks: list[Chunk]
    end_index: int
    eof: bool


@dataclass
class ContextMatch:
    new_index: int
    fuzz: int


def apply_diff(text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    """Apply a V4A diff to `text`.

    `mode="create"` parses the add-file syntax (every line `+`-prefixed and
    no context); `mode="default"` parses the update syntax with `@@` anchors
    and context hunks. Raises `NativeToolError` when the diff is malformed
    or its context does not match `text`.
    """
    newline = _detect_newline(text, diff, mode)
    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines, newline=newline)

    normalized_text = _normalize_text_newlines(text)
    parsed = _parse_update_diff(diff_lines, normalized_text)
    return _apply_chunks(normalized_text, parsed.chunks, newline=newline)


def _normalize_diff_lines(diff: str) -> list[str]:
    lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _detect_newline_from_text(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _detect_newline(text: str, diff: str, mode: ApplyDiffMode) -> str:
    # A create-file diff has no input to infer the newline style from, so it
    # follows the diff's own style.
    if mode != "create" and "\n" in text:
        return _detect_newline_from_text(text)
    return _detect_newline_from_text(diff)


def _normalize_text_newlines(text: str) -> str:
    # CRLF collapses to LF for parsing and matching; the style detected above
    # is restored when the result is emitted.
    return text.replace("\r\n", "\n")


def _is_done(state: ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines):
        return True
    return any(state.lines[state.index].startswith(prefix) for prefix in prefixes)


def _read_str(state: ParserState, prefix: str) -> str:
    if state.index >= len(state.lines):
        return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix) :]
    return ""


def _parse_create_diff(lines: list[str], newline: str) -> str:
    parser = ParserState(lines=[*lines, END_PATCH])
    output: list[str] = []

    while not _is_done(parser, SECTION_TERMINATORS):
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"):
            raise NativeToolError(f"Invalid Add File Line: {line}")
        output.append(line[1:])

    return newline.join(output)


def _parse_update_diff(lines: list[str], text: str) -> ParsedUpdateDiff:
    parser = ParserState(lines=[*lines, END_PATCH])
    text_lines = text.split("\n")
    chunks: list[Chunk] = []
    cursor = 0

    while not _is_done(parser, END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        has_bare_anchor = anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@"
        if has_bare_anchor:
            parser.index += 1

        if not (anchor or has_bare_anchor or cursor == 0):
            current_line = parser.lines[parser.index] if parser.index < len(parser.lines) else ""
            raise NativeToolError(f"Invalid Line:\n{current_line}")

        if anchor.strip():
            cursor = _advance_cursor_to_anchor(anchor, text_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        found = _find_context(text_lines, section.next_context, cursor, section.eof)
        if found.new_index == -1:
            ctx_text = "\n".join(section.next_context)
            if section.eof:
                raise NativeToolError(f"Invalid EOF Context {cursor}:\n{ctx_text}")
            raise NativeToolError(f"Invalid Context {cursor}:\n{ctx_text}")

        cursor = found.new_index + len(section.next_context)
        parser.fuzz += found.fuzz
        parser.index = section.end_index

        chunks.extend(
            Chunk(
                orig_index=chunk.orig_index + found.new_index,
                del_lines=list(chunk.del_lines),
                ins_lines=list(chunk.ins_lines),
            )
            for chunk in section.section_chunks
        )

    return ParsedUpdateDiff(chunks=chunks, fuzz=parser.fuzz)


def _advance_cursor_to_anchor(
    anchor: str,
    text_lines: list[str],
    cursor: int,
    parser: ParserState,
) -> int:
    found = False

    if not any(line == anchor for line in text_lines[:cursor]):
        for i in range(cursor, len(text_lines)):
            if text_lines[i] == anchor:
                cursor = i + 1
                found = True
                break

    if not found and not any(line.strip() == anchor.strip() for line in text_lines[:cursor]):
        for i in range(cursor, len(text_lines)):
            if text_lines[i].strip() == anchor.strip():
                cursor = i + 1
                parser.fuzz += 1
                break

    return cursor


def _read_section(lines: list[str], start_index: int) -> ReadSectionResult:
    context: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    section_chunks: list[Chunk] = []
    mode: Literal["keep", "add", "delete"] = "keep"
    index = start_index
    orig_index = index

    while index < len(lines):
        raw = lines[index]
        if raw.startswith(("@@", *END_SECTION_MARKERS)):
            break
        if raw == "***":
            break
        if raw.startswith("***"):
            raise NativeToolError(f"Invalid Line: {raw}")

        index += 1
        last_mode = mode
        line = raw if raw else " "
        prefix = line[0]
        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            raise NativeToolError(f"Invalid Line: {line}")

        line_content = line[1:]
        switching_to_context = mode == "keep" and last_mode != mode
        if switching_to_context and (del_lines or ins_lines):
            section_chunks.append(
                Chunk(
                    orig_index=len(context) - len(del_lines),
                    del_lines=list(del_lines),
                    ins_lines=list(ins_lines),
                )
            )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(line_content)
            context.append(line_content)
        elif mode == "add":
            ins_lines.append(line_content)
        else:
            context.append(line_content)

    if del_lines or ins_lines:
        section_chunks.append(
            Chunk(
                orig_index=len(context) - len(del_lines),
                del_lines=list(del_lines),
                ins_lines=list(ins_lines),
            )
        )

    if index < len(lines) and lines[index] == END_FILE:
        return ReadSectionResult(context, section_chunks, index + 1, True)

    if index == orig_index:
        next_line = lines[index] if index < len(lines) else ""
        raise NativeToolError(f"Nothing in this section - index={index} {next_line}")

    return ReadSectionResult(context, section_chunks, index, False)


def _find_context(lines: list[str], context: list[str], start: int, eof: bool) -> ContextMatch:
    if eof:
        end_start = max(0, len(lines) - len(context))
        end_match = _find_context_core(lines, context, end_start)
        if end_match.new_index != -1:
            return end_match
        fallback = _find_context_core(lines, context, start)
        return ContextMatch(new_index=fallback.new_index, fuzz=fallback.fuzz + 10000)
    return _find_context_core(lines, context, start)


def _find_context_core(lines: list[str], context: list[str], start: int) -> ContextMatch:
    if not context:
        return ContextMatch(new_index=start, fuzz=0)

    # Exact, then right-stripped, then fully stripped — each pass costlier and
    # fuzzier than the last.
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value):
            return ContextMatch(new_index=i, fuzz=0)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.rstrip()):
            return ContextMatch(new_index=i, fuzz=1)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.strip()):
            return ContextMatch(new_index=i, fuzz=100)

    return ContextMatch(new_index=-1, fuzz=0)


def _equals_slice(source: list[str], target: list[str], start: int, map_fn: Callable[[str], str]) -> bool:
    if start + len(target) > len(source):
        return False
    return all(map_fn(source[start + offset]) == map_fn(target_value) for offset, target_value in enumerate(target))


def _apply_chunks(text: str, chunks: list[Chunk], newline: str) -> str:
    orig_lines = text.split("\n")
    dest_lines: list[str] = []
    cursor = 0

    for chunk in chunks:
        if chunk.orig_index > len(orig_lines):
            raise NativeToolError(f"apply_diff: chunk.orig_index {chunk.orig_index} > input length {len(orig_lines)}")
        if cursor > chunk.orig_index:
            raise NativeToolError(f"apply_diff: overlapping chunk at {chunk.orig_index} (cursor {cursor})")

        dest_lines.extend(orig_lines[cursor : chunk.orig_index])
        cursor = chunk.orig_index

        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)

        cursor += len(chunk.del_lines)

    dest_lines.extend(orig_lines[cursor:])
    return newline.join(dest_lines)


__all__ = ["ApplyDiffMode", "apply_diff"]
