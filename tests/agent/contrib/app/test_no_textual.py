"""The one invariant `luca.agent.contrib.app` exists to hold.

The package was carved out of `contrib/tui` so a front end that draws nothing —
the ACP stdio server — can compose a configured agent. That only works while
NOTHING in it reaches Textual, and the failure mode is silent: someone adds
`from .render import x` for one constant, and months later a JSON-RPC server
refuses to start on a machine with no `tui` dependency group installed.

It happened once already, twice over. `tui/__init__.py` imports `.app`
eagerly, so importing ANY tui submodule pulled Textual in; and `wiring.py`
reached `textual.theme` through `render -> format -> theme` for two string
constants. Both are why `benchmarks/terminal-bench/luca_tb/runner.py` used to
copy `build_runner` rather than import it.

Checked in a SUBPROCESS with `textual` blocked at the import hook, because the
parent test process has it installed and loaded: an in-process check would pass
no matter what the package imports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[4] / "luca" / "agent" / "contrib" / "app"

PROBE = """
import importlib
import sys


class TextualBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "textual" or name.startswith("textual."):
            raise ImportError(f"blocked by the guard: {name}")
        return None


sys.meta_path.insert(0, TextualBlocker())
for name in sys.argv[1:]:
    importlib.import_module(name)
assert not [m for m in sys.modules if m == "textual" or m.startswith("textual.")]
"""


def _module_names() -> list[str]:
    """Every module in the package, by path rather than by walking the
    imported package — walking would import it here, in a process that already
    has Textual."""
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).with_suffix("")
        parts = [part for part in relative.parts if part != "__init__"]
        names.append(".".join(["luca.agent.contrib.app", *parts]))
    return names


def test_the_package_has_modules_to_check():
    """Guards the guard: a broken glob would make every assertion below pass
    over an empty list."""
    names = _module_names()

    assert "luca.agent.contrib.app" in names
    assert "luca.agent.contrib.app.wiring" in names
    assert len(names) > 5


def test_no_module_in_contrib_app_imports_textual():
    result = subprocess.run(
        [sys.executable, "-c", PROBE, *_module_names()],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
