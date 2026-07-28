"""Skips the whole directory when the optional `mcp` SDK is not installed."""

import pytest

pytest.importorskip("mcp")
