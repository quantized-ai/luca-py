"""CLI argument parsing and session building."""

from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.cli import arg_parser, build_session, main
from luca.agent.contrib.tui.sessions import save_session
from luca.agent.contrib.tui.wiring import default_model
from luca.agent.core.models import LLMConfig

from .helpers import fresh_session


def test_default_args():
    args = arg_parser().parse_args([])

    assert (args.conversation, args.fork, args.no_streaming, args.faux) == (
        None, False, False, False,
    )
    assert (args.model, args.provider, args.reasoning) == (
        None, None, None,
    )


def test_model_and_reasoning_override_the_fresh_session():
    session = build_session(arg_parser().parse_args([
        "--model", "moonshotai/kimi-k2.7-code", "--reasoning", "high",
    ]))

    assert session.session_config.llm_config == LLMConfig(
        model="moonshotai/kimi-k2.7-code",
        provider="openrouter",
        reasoning="high",
    )


def test_provider_override_is_passed_through_as_is():
    session = build_session(arg_parser().parse_args([
        "--provider", "quantized",
    ]))

    llm = session.session_config.llm_config
    assert llm.provider == "quantized"
    assert llm.model == default_model().model


def test_provider_override_applies_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session)

    resumed = build_session(arg_parser().parse_args([
        "--conversation", session.id, "--provider", "quantized",
    ]))

    assert resumed.session_config.llm_config.provider == "quantized"


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


def test_main_prints_the_resume_hint_after_the_app_exits(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["id"] = self.runner.session.id

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    out = capsys.readouterr().out
    assert f"--conversation {seen['id']}" in out
    assert "Goodbye!" in out


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
