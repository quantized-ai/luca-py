"""Skips the whole directory when textual (the `tui` dependency group) is
not installed."""

import pytest

pytest.importorskip("textual")
