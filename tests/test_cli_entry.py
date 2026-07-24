"""The `luca` console entry point (`luca.cli:main`)."""

import builtins

import luca.cli
from luca import __version__


def _boom(argv=None):
    raise AssertionError("the TUI must not launch for --version")


def test_version_prints_and_returns_without_launching(capsys, monkeypatch):
    # --version is answered before the TUI is imported; the delegate must not run.
    monkeypatch.setattr("luca.agent.contrib.tui.cli.main", _boom)
    luca.cli.main(["--version"])

    assert capsys.readouterr().out.strip() == f"luca {__version__}"


def test_main_delegates_argv_to_the_tui(monkeypatch):
    seen = {}
    monkeypatch.setattr(luca.cli, "_load_dotenv", lambda: None)
    monkeypatch.setattr(
        "luca.agent.contrib.tui.cli.main",
        lambda argv=None: seen.__setitem__("argv", argv),
    )

    luca.cli.main(["--faux", "--model", "x"])

    assert seen == {"argv": ["--faux", "--model", "x"]}


def test_load_dotenv_is_a_noop_without_python_dotenv(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ModuleNotFoundError("No module named 'dotenv'", name="dotenv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    luca.cli._load_dotenv()  # must not raise
