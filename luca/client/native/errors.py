"""The single error type raised by the native tool handlers.

Deliberately NOT part of the `ClientError` hierarchy: those are reserved for
transport or network failures (see `luca/client/exceptions.py`). A missing
file, an unmatched `old_str` or a diff that will not apply is a domain
outcome of a CALLER-side handler — it can never come out of `completion()`,
and it carries no provider.

Handlers turn it into a tool result:

    try:
        output = execute_text_editor(call.arguments, root=workspace)
    except NativeToolError as exc:
        return ToolMessage(tool_call_id=call.id, content=str(exc), is_error=True)
"""

from __future__ import annotations


class NativeToolError(Exception):
    """A provider-native tool handler could not do what it was asked.

    The message is written for the model: say what failed and what would
    fix it.
    """


__all__ = ["NativeToolError"]
