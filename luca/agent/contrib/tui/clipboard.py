"""Reading an image off the OS clipboard. No Textual imports.

A terminal only ever transmits text, so a pasted image never reaches the app
through a key event — the clipboard has to be read directly. Each platform
needs its own helper: `osascript` on macOS, `wl-paste` or `xclip` on Linux,
PowerShell on Windows. All of them are asked for PNG, so the media type is
always `image/png`.

Everything here shells out and blocks; the caller runs it off the UI thread.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

MEDIA_TYPE = "image/png"
_TIMEOUT_SECONDS = 10

_MACOS_SCRIPT = """\
set f to open for access POSIX file "{path}" with write permission
write (the clipboard as «class PNGf») to f
close access f
"""

_WINDOWS_SCRIPT = """\
Add-Type -AssemblyName System.Windows.Forms
$img = [Windows.Forms.Clipboard]::GetImage()
if ($img -eq $null) {{ exit 1 }}
$img.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Png)
"""


class ClipboardUnavailable(Exception):
    """No way to read the clipboard here — an unsupported platform, or the
    helper binary is not installed. Carries a message worth showing."""


def read_clipboard_image() -> bytes | None:
    """The clipboard's image as PNG bytes, or None when it holds no image.

    Raises `ClipboardUnavailable` when the platform has no reader at all —
    which is a different thing from an empty clipboard, and worth telling the
    user only once."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "clipboard.png"
        if not _capture(target):
            return None
        data = target.read_bytes() if target.is_file() else b""
    return data or None


def _capture(target: Path) -> bool:
    """Run the platform's reader. False when the clipboard holds no image."""
    system = platform.system()
    if system == "Darwin":
        return _run(["osascript", "-e", _MACOS_SCRIPT.format(path=target)])
    if system == "Windows":
        return _run(
            ["powershell", "-NoProfile", "-Command",
             _WINDOWS_SCRIPT.format(path=target)],
        )
    if system == "Linux":
        return _capture_linux(target)
    raise ClipboardUnavailable(
        f"Reading the clipboard is not supported on {system}."
    )


def _capture_linux(target: Path) -> bool:
    if shutil.which("wl-paste"):
        return _run(["wl-paste", "--type", MEDIA_TYPE], stdout=target)
    if shutil.which("xclip"):
        return _run(
            ["xclip", "-selection", "clipboard", "-t", MEDIA_TYPE, "-o"],
            stdout=target,
        )
    raise ClipboardUnavailable(
        "Reading the clipboard needs `wl-paste` (Wayland) or `xclip` (X11)."
    )


def _run(command: list[str], *, stdout: Path | None = None) -> bool:
    """True when the reader succeeded. A non-zero exit is how every one of
    these reports "no image here", so it is not an error."""
    try:
        if stdout is None:
            result = subprocess.run(
                command, capture_output=True, timeout=_TIMEOUT_SECONDS,
            )
        else:
            with stdout.open("wb") as handle:
                result = subprocess.run(
                    command, stdout=handle, stderr=subprocess.PIPE,
                    timeout=_TIMEOUT_SECONDS,
                )
    except FileNotFoundError:
        raise ClipboardUnavailable(
            f"`{command[0]}` is not installed."
        ) from None
    except subprocess.TimeoutExpired:
        raise ClipboardUnavailable(
            f"`{command[0]}` did not respond."
        ) from None
    return result.returncode == 0
