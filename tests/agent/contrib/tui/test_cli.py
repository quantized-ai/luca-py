"""CLI argument parsing and session building."""

from luca.agent.contrib.tui.cli import arg_parser, build_session
from luca.agent.contrib.tui.sessions import save_session

from .helpers import fresh_session


def test_default_args():
    args = arg_parser().parse_args([])

    assert (args.conversation, args.fork, args.no_streaming, args.faux) == (
        None, False, False, False,
    )
    assert (args.model, args.reasoning_effort) == (None, None)


def test_model_and_reasoning_effort_override_the_fresh_session():
    session = build_session(arg_parser().parse_args([
        "--model", "moonshotai/kimi-k2.7-code", "--reasoning-effort", "high",
    ]))

    llm = session.session_config.llm_config
    assert llm.model == "moonshotai/kimi-k2.7-code"
    assert llm.reasoning_effort == "high"
    assert llm.provider == "openrouter"


def test_model_override_composes_with_faux():
    session = build_session(arg_parser().parse_args([
        "--faux", "--model", "moonshotai/kimi-k2.7-code",
    ]))

    llm = session.session_config.llm_config
    assert llm.model == "moonshotai/kimi-k2.7-code"
    assert llm.provider == "faux"


def test_model_override_applies_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session)

    resumed = build_session(arg_parser().parse_args([
        "--conversation", session.id, "--model", "moonshotai/kimi-k2.7-code",
    ]))

    llm = resumed.session_config.llm_config
    assert llm.model == "moonshotai/kimi-k2.7-code"
    assert llm.provider == session.session_config.llm_config.provider


def test_faux_session_uses_the_faux_model():
    session = build_session(arg_parser().parse_args(["--faux"]))

    assert session.session_config.llm_config.provider == "faux"


def test_resume_and_fork(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session)

    resumed = build_session(
        arg_parser().parse_args(["--conversation", session.id]),
    )
    assert resumed == session

    forked = build_session(
        arg_parser().parse_args(["--conversation", session.id, "--fork"]),
    )
    assert forked.id != session.id
    assert forked.entries == session.entries
