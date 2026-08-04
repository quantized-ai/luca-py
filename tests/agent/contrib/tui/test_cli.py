"""CLI argument parsing and session building."""

import json

import pytest

from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.cli import (
    arg_parser,
    build_session,
    main,
    resume_id,
    resume_picker,
)
from luca.agent.contrib.tui.config import ENV_CONFIG_PATH
from luca.agent.contrib.tui.sessions import (
    encode_project_path,
    resolve_session_directory,
    save_session,
)
from luca.agent.contrib.tui.wiring import default_model
from luca.agent.core.models import Inf, LLMConfig, RuntimeConfig
from luca.agent.core.utils import pretty_print

from .helpers import fresh_session


def test_default_args():
    args = arg_parser().parse_args([])

    assert (args.resume, args.fork, args.streaming, args.faux) == (
        None,
        False,
        None,
        False,
    )
    assert args.pretty_print is False
    assert (args.model, args.provider, args.reasoning) == (
        None,
        None,
        None,
    )
    # config-resolvable flags default to None so "unset" is distinguishable
    assert (args.autocompact, args.compact_threshold, args.compact_keep_turns) == (
        None,
        None,
        None,
    )
    assert (args.workspace, args.mode) == (None, None)
    assert args.theme is None
    assert args.gallery is None
    assert args.subagents is True
    assert args.skills is True
    assert args.instructions is True
    assert args.config is None


def test_no_skills_turns_skill_loading_off():
    assert arg_parser().parse_args(["--no-skills"]).skills is False


def test_no_resume_flag_means_a_fresh_session():
    args = arg_parser().parse_args([])

    assert (args.resume, resume_id(args), resume_picker(args)) == (None, None, False)


def test_a_bare_resume_asks_for_the_picker():
    args = arg_parser().parse_args(["--resume"])

    assert (resume_id(args), resume_picker(args)) == (None, True)


def test_resume_with_an_id_names_that_session_and_skips_the_picker():
    args = arg_parser().parse_args(["--resume", "5ac92996"])

    assert (resume_id(args), resume_picker(args)) == ("5ac92996", False)


def test_a_bare_resume_reaches_the_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen: dict[str, bool] = {}
    monkeypatch.setattr(AgentApp, "run", lambda self: seen.update(resume=self._resume))

    main(["--faux", "--resume"])

    assert seen == {"resume": True}


def test_the_session_store_is_the_project_directory_under_the_configured_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"sessions": {"directory": str(tmp_path / "store")}}))
    seen: dict[str, object] = {}
    monkeypatch.setattr(AgentApp, "run", lambda self: seen.update(directory=self._session_dir))

    main(["--faux"])

    assert seen == {"directory": tmp_path / "store" / encode_project_path(tmp_path)}


def test_a_conversation_is_loaded_from_the_store_not_the_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session, resolve_session_directory("."))
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        AgentApp,
        "run",
        lambda self: seen.update(id=self.runner.session.id, picker=self._resume),
    )

    main(["--resume", session.id])

    # named outright, so the app opens on it instead of on the picker
    assert seen == {"id": session.id, "picker": False}
    assert not (tmp_path / f"{session.id}.json").exists()


def test_no_instructions_turns_agents_md_reading_off():
    assert arg_parser().parse_args(["--no-instructions"]).instructions is False


def test_the_config_flag_parses_as_a_path_string():
    assert arg_parser().parse_args(["--config", "./ci.json"]).config == "./ci.json"


def test_the_theme_flag_parses_as_a_textual_theme_name():
    assert arg_parser().parse_args(["--theme", "textual-light"]).theme == "textual-light"


def test_the_gallery_flag_parses_bare_or_with_a_fixture_name():
    # nargs="?" with const "all": bare --gallery browses everything, a value
    # names one fixture (bundled name or path)
    assert arg_parser().parse_args(["--gallery"]).gallery == "all"
    assert arg_parser().parse_args(["--gallery", "1a_agent_loop"]).gallery == "1a_agent_loop"


def test_subagents_are_on_by_default_and_no_subagents_turns_them_off():
    on = build_session(arg_parser().parse_args([]))
    off = build_session(arg_parser().parse_args(["--no-subagents"]))

    assert on.session_config.runtime_config.subagents_enabled is True
    assert off.session_config.runtime_config.subagents_enabled is False


def test_the_subagent_limits_default_to_depth_three_and_no_caps():
    runtime = build_session(arg_parser().parse_args([])).session_config.runtime_config

    assert (
        runtime.subagents_max_depth,
        runtime.subagents_max_per_turn,
        runtime.subagents_max_workers,
    ) == (3, Inf, Inf)


def test_an_invalid_limit_flag_value_fails_loudly():
    # 0 is rejected by the RuntimeConfig validator; a plain attribute write
    # would bypass it, wedge the first spawn, and poison the saved session
    from luca.agent.contrib.tui.config import LucaConfigError

    with pytest.raises(LucaConfigError, match="invalid subagent flag"):
        build_session(arg_parser().parse_args(["--subagents-max-workers", "0"]))
    with pytest.raises(LucaConfigError, match="invalid subagent flag"):
        build_session(arg_parser().parse_args(["--subagents-max-per-turn", "0"]))


def test_the_subagent_limit_flags_land_on_the_session():
    session = build_session(
        arg_parser().parse_args(
            [
                "--subagents-max-depth",
                "2",
                "--subagents-max-per-turn",
                "5",
                "--subagents-max-workers",
                "3",
            ]
        )
    )
    runtime = session.session_config.runtime_config

    assert (
        runtime.subagents_max_depth,
        runtime.subagents_max_per_turn,
        runtime.subagents_max_workers,
    ) == (2, 5, 3)


def test_model_and_reasoning_override_the_fresh_session():
    session = build_session(
        arg_parser().parse_args(
            [
                "--model",
                "moonshotai/kimi-k2.7-code",
                "--reasoning",
                "high",
            ]
        )
    )

    assert session.session_config.llm_config == LLMConfig(
        model="moonshotai/kimi-k2.7-code",
        provider="openrouter",
        reasoning="high",
    )


def test_provider_override_is_passed_through_as_is():
    session = build_session(
        arg_parser().parse_args(
            [
                "--provider",
                "quantized",
            ]
        )
    )

    llm = session.session_config.llm_config
    assert llm.provider == "quantized"
    assert llm.model == default_model().model


def test_provider_override_applies_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session)

    resumed = build_session(
        arg_parser().parse_args(
            [
                "--resume",
                session.id,
                "--provider",
                "quantized",
            ]
        )
    )

    assert resumed.session_config.llm_config.provider == "quantized"


def test_model_override_composes_with_faux():
    session = build_session(
        arg_parser().parse_args(
            [
                "--faux",
                "--model",
                "moonshotai/kimi-k2.7-code",
            ]
        )
    )

    llm = session.session_config.llm_config
    assert llm.model == "moonshotai/kimi-k2.7-code"
    assert llm.provider == "faux"


def test_model_override_applies_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    save_session(session)

    resumed = build_session(
        arg_parser().parse_args(
            [
                "--resume",
                session.id,
                "--model",
                "moonshotai/kimi-k2.7-code",
            ]
        )
    )

    llm = resumed.session_config.llm_config
    assert llm.model == "moonshotai/kimi-k2.7-code"
    assert llm.provider == session.session_config.llm_config.provider


def test_faux_session_uses_the_faux_model():
    session = build_session(arg_parser().parse_args(["--faux"]))

    assert session.session_config.llm_config.provider == "faux"


def test_main_prints_the_resume_hint_after_the_app_exits(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["id"] = self.runner.session.id

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    out = capsys.readouterr().out
    assert "--resume" in out
    assert seen["id"] in out
    assert "Goodbye!" in out


def test_theme_defaults_to_luca_dark_during_app_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["theme"] = self.theme

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    assert seen == {"theme": "luca-dark"}


def test_luca_json_theme_reaches_the_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"theme": {"name": "textual-light"}}))
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["theme"] = self.theme

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    assert seen == {"theme": "textual-light"}


def test_theme_flag_overrides_luca_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"theme": {"name": "textual-light"}}))
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["theme"] = self.theme

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux", "--theme", "textual-dark"])

    assert seen == {"theme": "textual-dark"}


def test_the_config_flag_replaces_the_project_luca_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"model": {"model": "from-project"}}))
    named = tmp_path / "named.json"
    named.write_text(json.dumps({"model": {"model": "from-named"}}))
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["model"] = self.runner.session.session_config.llm_config.model

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux", "--config", str(named)])

    assert seen["model"] == "from-named"


def test_the_config_env_var_is_honored_when_no_flag_is_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"model": {"model": "from-project"}}))
    named = tmp_path / "named.json"
    named.write_text(json.dumps({"model": {"model": "from-env"}}))
    monkeypatch.setenv(ENV_CONFIG_PATH, str(named))
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["model"] = self.runner.session.session_config.llm_config.model

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    assert seen["model"] == "from-env"


def test_an_instructions_entry_that_does_not_exist_exits_with_a_readable_error(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"instructions": ["typo.md"]}))
    monkeypatch.setattr(AgentApp, "run", lambda self: pytest.fail("the app must not start"))

    with pytest.raises(SystemExit) as exit_info:
        main(["--faux"])

    assert exit_info.value.code == 1
    assert "not a readable instruction file" in capsys.readouterr().err


def test_a_missing_config_file_exits_with_a_readable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(AgentApp, "run", lambda self: pytest.fail("the app must not start"))

    with pytest.raises(SystemExit) as exit_info:
        main(["--faux", "--config", str(tmp_path / "nope.json")])

    assert exit_info.value.code == 1
    assert "not a readable config file" in capsys.readouterr().err


def test_pretty_print_writes_the_transcript_and_never_starts_the_app(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    session = fresh_session()
    # into the project's store, not the cwd — which is what --pretty-print
    # now has to look in
    save_session(session, resolve_session_directory("."))
    monkeypatch.setattr(
        AgentApp,
        "run",
        lambda self: pytest.fail("the app must not start"),
    )

    main(["--resume", session.id, "--pretty-print"])

    assert capsys.readouterr().out == pretty_print(session) + "\n"


def test_pretty_print_without_a_conversation_exits_with_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--pretty-print"])

    assert exit_info.value.code == 2
    assert "--pretty-print requires --resume" in capsys.readouterr().err


def test_resume_and_fork(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # subagents on at depth 3 — the defaults every session the CLI builds
    # gets, resumed ones included: the flags describe the run being started
    session = fresh_session(RuntimeConfig(subagents_enabled=True, subagents_max_depth=3))
    save_session(session)

    resumed = build_session(
        arg_parser().parse_args(["--resume", session.id]),
    )
    assert resumed == session

    forked = build_session(
        arg_parser().parse_args(["--resume", session.id, "--fork"]),
    )
    assert forked.id != session.id
    assert forked.entries == session.entries
