"""Caller-side handlers for the provider-native file-editing tools.

The transports DECLARE `ApplyPatchTool` and `TextEditorTool`, surface their
calls and project their results; executing them was always the caller's job.
This package is that execution — opt-in, imported by nothing else in
`luca.client`, and depending on nothing but the standard library:

    from luca.client.native import NativeToolError, execute_apply_patch

    try:
        output = execute_apply_patch(call.operation, root=workspace)
    except NativeToolError as exc:
        output, failed = str(exc), True

Two layers. `v4a.apply_diff` and `text_editor.view` / `str_replace` /
`insert` are pure text transforms; `execute_apply_patch` and
`execute_text_editor` add the filesystem on top, resolving model-supplied
paths against a caller-supplied `root` they may not leave.
"""

from .apply_patch import execute_apply_patch
from .errors import NativeToolError
from .text_editor import execute_text_editor
from .v4a import ApplyDiffMode, apply_diff

__all__ = [
    "NativeToolError",
    "apply_diff",
    "ApplyDiffMode",
    "execute_apply_patch",
    "execute_text_editor",
]
