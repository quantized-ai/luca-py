"""CLI argument parsing, session building, and the session log."""

import json
import logging
from pathlib import Path

import pytest

from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.auth import ENV_AUTH_PATH
from luca.agent.contrib.tui.cli import (
    arg_parser,
    build_session,
    log_path,
    main,
    resume_id,
    resume_picker,
    setup_logging,
)
from luca.agent.contrib.tui.config import ENV_CONFIG_PATH, LucaConfig, LucaConfigError
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
    assert args.commands is True
    assert args.instructions is True
    assert args.config is None


def test_no_skills_turns_skill_loading_off():
    assert arg_parser().parse_args(["--no-skills"]).skills is False


def test_no_commands_turns_user_defined_commands_off():
    assert arg_parser().parse_args(["--no-commands"]).commands is False


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

    # `--faux` because the stored session names the faux provider, which is
    # only reachable as an injected INSTANCE — without the flag the boot check
    # correctly refuses a provider nothing can build.
    main(["--faux", "--resume", session.id])

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


def test_provider_native_tools_are_on_by_default_and_no_use_native_turns_them_off():
    on = build_session(arg_parser().parse_args([]))
    off = build_session(arg_parser().parse_args(["--no-use-native"]))

    assert on.session_config.use_native_tools is True
    assert off.session_config.use_native_tools is False


def test_the_config_can_turn_provider_native_tools_off():
    session = build_session(arg_parser().parse_args([]), LucaConfig(use_native_tools=False))

    assert session.session_config.use_native_tools is False


def test_the_use_native_flag_beats_the_config():
    session = build_session(
        arg_parser().parse_args(["--use-native"]),
        LucaConfig(use_native_tools=False),
    )

    assert session.session_config.use_native_tools is True


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
        model_options={"reasoning": "high"},
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


def test_the_auth_file_key_reaches_the_runner_and_the_context_manager(tmp_path, monkeypatch):
    # A real (non-faux) launch: `openrouter` is a provider the client knows, so
    # nothing is registered and no call is made — `run` is replaced.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "auth.json").write_text(json.dumps({"openrouter": {"type": "api", "key": "sk-or-live"}}))
    monkeypatch.setenv(ENV_AUTH_PATH, str(tmp_path / "auth.json"))
    seen: dict[str, object] = {}

    def fake_run(self: AgentApp) -> None:
        seen["runner"] = self.runner.api_key
        seen["context_manager"] = self._context_manager.api_key

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main([])

    assert seen == {"runner": "sk-or-live", "context_manager": "sk-or-live"}


def test_a_provider_with_no_auth_entry_passes_no_key_at_all(tmp_path, monkeypatch):
    # Absent, not empty: the client falls back to OPENROUTER_API_KEY.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "auth.json").write_text(json.dumps({"anthropic": {"type": "api", "key": "sk-ant"}}))
    monkeypatch.setenv(ENV_AUTH_PATH, str(tmp_path / "auth.json"))
    seen: dict[str, object] = {}

    monkeypatch.setattr(AgentApp, "run", lambda self: seen.update(key=self.runner.api_key))
    main([])

    assert seen == {"key": None}


def test_an_unreachable_provider_fails_at_boot_with_a_readable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"model": {"provider": "my_host", "model": "some/model"}}))

    with pytest.raises(SystemExit):
        main([])

    assert "cannot be reached" in capsys.readouterr().err


def test_a_custom_host_with_a_base_url_and_transport_boots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(
        json.dumps(
            {
                "model": {"provider": "my_host", "model": "some/model"},
                "providers": {
                    "my_host": {
                        "base_url": "https://custom.api.example/v1",
                        "transport": "luca.client.transports.OpenAITransport",
                    },
                },
            }
        )
    )
    (tmp_path / "auth.json").write_text(json.dumps({"my_host": {"type": "api", "key": "sk-custom"}}))
    monkeypatch.setenv(ENV_AUTH_PATH, str(tmp_path / "auth.json"))
    seen: dict[str, object] = {}

    def fake_run(self: AgentApp) -> None:
        seen["key"] = self.runner.api_key
        seen["options"] = self.runner.session.session_config.llm_config.provider_options

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main([])

    assert seen == {
        "key": "sk-custom",
        "options": {
            "base_url": "https://custom.api.example/v1",
            "transport": "luca.client.transports.OpenAITransport",
        },
    }


def test_luca_json_theme_reaches_the_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"theme": {"name": "textual-light"}}))
    seen: dict[str, str] = {}

    def fake_run(self: AgentApp) -> None:
        seen["theme"] = self.theme

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    assert seen == {"theme": "textual-light"}


def test_luca_json_model_options_reach_the_session_and_survive_a_switch(tmp_path, monkeypatch):
    # The whole chain in one: the issue's config resolves onto the launched
    # session, and the resolver reaches the app so `/model` re-resolves too.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(
        json.dumps(
            {
                "model": {"provider": "faux", "model": "configured-model"},
                "providers": {
                    "faux": {
                        "options": {"temperature": 0.5},
                        "models": {"configured-model": {"options": {"max_tokens": 6000}}},
                    },
                },
            }
        )
    )
    seen: dict[str, LLMConfig] = {}

    def fake_run(self: AgentApp) -> None:
        seen["launched"] = self.runner.session.session_config.llm_config
        seen["switched"] = self.resolve_model_options(
            seen["launched"].model_copy(update={"model": "another-model"}),
        )

    monkeypatch.setattr(AgentApp, "run", fake_run)
    main(["--faux"])

    assert (seen["launched"].model_options, seen["switched"].model_options) == (
        {"max_tokens": 6000, "temperature": 0.5},
        {"temperature": 0.5},
    )


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
    # …and provider-native tools on: the other capability flag every launch
    # stamps onto the session it is about to drive.
    session.session_config.use_native_tools = True
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


# ── --refresh-models ─────────────────────────────────────────────────────────


def test_refresh_models_is_off_unless_asked_for():
    assert arg_parser().parse_args([]).refresh_models is False
    assert arg_parser().parse_args(["--refresh-models"]).refresh_models is True


def test_refresh_models_runs_the_refresh_and_exits(monkeypatch):
    # the refresh owns a parser of its own; this command's flags must not
    # reach it, so it is handed an explicit empty argv
    seen: list = []
    monkeypatch.setattr("luca.agent.contrib.tui.cli.refresh_catalog", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(AgentApp, "run", lambda self: pytest.fail("the app must not start"))

    with pytest.raises(SystemExit) as exit_info:
        main(["--refresh-models"])

    assert exit_info.value.code == 0
    assert seen == [[]]


def test_a_failing_refresh_exits_non_zero(monkeypatch):
    monkeypatch.setattr("luca.agent.contrib.tui.cli.refresh_catalog", lambda argv: 1)

    with pytest.raises(SystemExit) as exit_info:
        main(["--refresh-models"])

    assert exit_info.value.code == 1


# ── logging ──────────────────────────────────────────────────────────────────


def test_logging_flags_default_to_unset():
    args = arg_parser().parse_args([])

    assert (args.log_level, args.log_file) == (None, None)


def test_the_log_lands_beside_the_session_it_belongs_to(tmp_path):
    assert log_path("abc123", tmp_path) == tmp_path / "logs" / "abc123.log"


def test_a_configured_log_file_replaces_the_default_path(tmp_path):
    assert log_path("abc123", tmp_path, "~/elsewhere/luca.log") == Path.home() / "elsewhere" / "luca.log"


def test_setup_logging_writes_luca_records_to_the_file(tmp_path):
    path = tmp_path / "logs" / "s1.log"

    setup_logging(path, "INFO")
    logging.getLogger("luca.agent.core.runner").info("conv=c1 hello")
    logging.getLogger("luca").handlers[-1].flush()

    assert path.read_text().endswith("INFO     luca.agent.core.runner conv=c1 hello\n")


def test_setup_logging_keeps_luca_records_off_the_root_logger(tmp_path):
    # stderr is the TUI's canvas: a root handler must never see luca's records.
    setup_logging(tmp_path / "s1.log", "INFO")

    assert logging.getLogger("luca").propagate is False


def test_setup_logging_off_writes_no_file(tmp_path):
    path = tmp_path / "logs" / "s1.log"

    setup_logging(path, "OFF")

    assert path.parent.exists() is False


def test_an_unknown_log_level_is_a_readable_config_error(tmp_path):
    with pytest.raises(LucaConfigError, match="unknown log level 'LOUD'"):
        setup_logging(tmp_path / "s1.log", "LOUD")


def test_main_logs_the_session_beside_its_own_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"sessions": {"directory": str(tmp_path / "store")}}))
    seen: dict[str, object] = {}
    monkeypatch.setattr(AgentApp, "run", lambda self: seen.update(id=self.runner.session.id))

    main(["--faux"])
    logging.getLogger("luca").warning("conv=c1 something happened")

    store = tmp_path / "store" / encode_project_path(tmp_path)
    assert (store / "logs" / f"{seen['id']}.log").read_text().endswith("conv=c1 something happened\n")


def test_the_log_level_flag_beats_the_env_var_and_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"logging": {"level": "ERROR"}}))
    monkeypatch.setenv("LUCA_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(AgentApp, "run", lambda self: None)

    main(["--faux", "--log-level", "DEBUG"])

    assert logging.getLogger("luca").level == logging.DEBUG


def test_the_env_var_beats_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "luca.json").write_text(json.dumps({"logging": {"level": "ERROR"}}))
    monkeypatch.setenv("LUCA_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(AgentApp, "run", lambda self: None)

    main(["--faux"])

    assert logging.getLogger("luca").level == logging.WARNING
