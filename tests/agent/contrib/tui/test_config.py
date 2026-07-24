"""luca.json: validation, home+project deep-merge, precedence, and the
config → objects mappings (providers, permission rules, compactor)."""

import json

import pytest

from luca.agent.contrib.resource_permissions import (
    PermissionMode,
    ResourcePermission,
    ToolKindRule,
    ToolRule,
)
from luca.agent.contrib.tui.config import (
    CompactionSettings,
    LucaConfig,
    LucaConfigError,
    ModelConfig,
    PermissionRule,
    PermissionSettings,
    RuntimeSettings,
    _deep_merge,
    build_compactor,
    build_permission_rules,
    load_luca_config,
    pick,
    register_config_providers,
    resolve_llm_config,
    resolve_runtime_config,
)
from luca.agent.contrib.compaction import CompactionStrategy, RecentTurnsStrategy
from luca.agent.core.models import ApprovalOption, LLMConfig, RuntimeConfig, ToolKind
from luca.client.providers import PROVIDERS


def _write(directory, payload):
    (directory / "luca.json").write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def _restore_providers():
    """register_config_providers mutates the global PROVIDERS; snapshot it."""
    saved = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)


# ── validation ───────────────────────────────────────────────────────────────


def test_a_partial_config_is_valid_and_defaults_the_rest():
    assert LucaConfig.model_validate({"model": {"model": "x"}}) == LucaConfig(
        model=ModelConfig(model="x"),
    )


def test_an_unknown_key_is_rejected():
    with pytest.raises(Exception):
        LucaConfig.model_validate({"nope": 1})


def test_the_schema_alias_is_accepted_and_ignored():
    assert LucaConfig.model_validate({"$schema": "./luca.schema.json"}).schema_url == (
        "./luca.schema.json"
    )


# ── deep merge (project over home) ───────────────────────────────────────────


def test_a_project_value_overrides_the_same_home_key_leaving_siblings():
    merged = _deep_merge(
        {"model": {"model": "home", "provider": "openrouter"}},
        {"model": {"model": "project"}},
    )
    assert merged == {"model": {"model": "project", "provider": "openrouter"}}


def test_provider_maps_merge_per_key_but_rule_lists_are_replaced(tmp_path):
    home = tmp_path / "home"; project = tmp_path / "project"
    home.mkdir(); project.mkdir()
    _write(home, {
        "providers": {"a": {"base_url": "https://a"}},
        "permissions": {"rules": [{"decision": "allow", "tool_kind": "read"}]},
    })
    _write(project, {
        "providers": {"b": {"base_url": "https://b"}},
        "permissions": {"rules": [{"decision": "deny", "tool_kind": "execute"}]},
    })
    config = load_luca_config(cwd=project, home=home)
    assert set(config.providers) == {"a", "b"}  # provider maps union
    assert len(config.permissions.rules) == 1   # project's list replaces home's
    assert config.permissions.rules[0].tool_kind is ToolKind.EXECUTE


def test_missing_files_give_an_empty_config(tmp_path):
    assert load_luca_config(cwd=tmp_path, home=tmp_path / "nope") == LucaConfig()


# ── precedence (cli > config > base/default) ─────────────────────────────────


def test_pick_prefers_cli_then_config_then_default():
    assert pick("cli", "config", "default") == "cli"
    assert pick(None, "config", "default") == "config"
    assert pick(None, None, "default") == "default"


def test_llm_precedence_is_cli_over_config_over_base():
    base = LLMConfig(model="base", provider="openrouter")
    config = LucaConfig(model=ModelConfig(model="from-config", reasoning="high"))
    assert resolve_llm_config(base, config, {"model": "from-cli"}) == LLMConfig(
        model="from-cli", provider="openrouter", reasoning="high",
    )


def test_runtime_config_fields_apply_over_the_persisted_runtime():
    base = RuntimeConfig(soft_max_steps=5)
    config = LucaConfig(runtime=RuntimeSettings(hard_max_steps=42))
    assert resolve_runtime_config(base, config) == RuntimeConfig(
        soft_max_steps=5, hard_max_steps=42,
    )


def test_pick_treats_falsy_but_set_values_as_set():
    assert pick(None, False, True) is False
    assert pick(None, 0, 5) == 0
    assert pick(None, 0.0, 0.8) == 0.0


def test_runtime_keeps_falsy_values_and_rejects_out_of_range():
    base = RuntimeConfig(soft_max_steps=5)
    kept = resolve_runtime_config(base, LucaConfig(runtime=RuntimeSettings(
        hard_max_steps=0, limit_tool_choice_on_soft_max_steps_reached=False,
    )))
    assert kept == RuntimeConfig(
        soft_max_steps=5, hard_max_steps=0,
        limit_tool_choice_on_soft_max_steps_reached=False,
    )
    with pytest.raises(LucaConfigError):
        resolve_runtime_config(base, LucaConfig(runtime=RuntimeSettings(hard_max_steps=-5)))


def test_compactor_precedence_cli_over_config_over_default():
    config = LucaConfig(compaction=CompactionSettings(threshold=0.6, keep_turns=2))
    from_config = build_compactor(config, enabled=None, threshold=None, keep_turns=None)
    assert from_config.threshold == 0.6
    assert isinstance(from_config.strategy, RecentTurnsStrategy)
    assert from_config.strategy.keep_turns == 2

    cli_wins = build_compactor(config, enabled=False, threshold=0.9, keep_turns=0)
    assert cli_wins.threshold == 0.9
    assert cli_wins.enabled is False
    assert isinstance(cli_wins.strategy, CompactionStrategy)


# ── config → objects ─────────────────────────────────────────────────────────


def test_a_tool_kind_rule_and_a_resource_rule_map_correctly():
    kind = PermissionRule(decision="allow", tool_kind=ToolKind.READ)
    resource = PermissionRule(decision="deny", permission="bash", resource="/etc/*")
    assert kind.to_rule() == ToolKindRule(
        tool_kind=ToolKind.READ, decision=ApprovalOption.ALLOW,
    )
    assert resource.to_rule() == ToolRule(
        resource_permission=ResourcePermission(permission="bash", resource="/etc/*"),
        decision=ApprovalOption.DENY,
    )


def test_a_rule_without_kind_or_permission_is_rejected_at_parse_time():
    # fails during LucaConfig validation, so load_luca_config's guard catches it
    with pytest.raises(Exception):
        LucaConfig.model_validate({"permissions": {"rules": [{"decision": "allow"}]}})


def test_a_bad_rule_in_a_file_surfaces_as_a_config_error(tmp_path):
    (tmp_path / "luca.json").write_text('{"permissions": {"rules": [{"decision": "allow"}]}}')
    with pytest.raises(LucaConfigError):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


def test_shadowing_a_first_class_provider_is_rejected():
    config = LucaConfig.model_validate({
        "providers": {"anthropic": {"base_url": "https://proxy"}},
    })
    with pytest.raises(LucaConfigError, match="built in"):
        register_config_providers(config)


def test_register_config_providers_adds_a_custom_host():
    config = LucaConfig.model_validate({
        "providers": {"lc-test-host": {"base_url": "https://x/v1", "api_key_env": "X_KEY"}},
    })
    register_config_providers(config)
    assert PROVIDERS["lc-test-host"] == {
        "default_base_url": "https://x/v1",
        "default_api_key_env_var": "X_KEY",
        "default_transport_class": __import__(
            "luca.client.transports", fromlist=["OpenAITransport"],
        ).OpenAITransport,
    }


def test_an_unknown_transport_is_rejected():
    config = LucaConfig.model_validate({
        "providers": {"bad": {"base_url": "https://x", "transport": "nope"}},
    })
    with pytest.raises(LucaConfigError, match="unknown transport"):
        register_config_providers(config)


# ── malformed files ──────────────────────────────────────────────────────────


def test_non_json_raises_a_readable_error(tmp_path):
    (tmp_path / "luca.json").write_text("{ not json")
    with pytest.raises(LucaConfigError, match="not valid JSON"):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


def test_a_non_object_top_level_raises(tmp_path):
    (tmp_path / "luca.json").write_text("[1, 2]")
    with pytest.raises(LucaConfigError, match="must be a JSON object"):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


def test_an_invalid_field_raises_luca_config_error(tmp_path):
    (tmp_path / "luca.json").write_text('{"unknown_key": true}')
    with pytest.raises(LucaConfigError, match="invalid"):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


# ── integration: config flows into the running app ───────────────────────────


async def test_luca_json_flows_into_the_running_app(tmp_path):
    from luca.agent.contrib.tui import AgentApp
    from luca.agent.contrib.tui.wiring import faux_model
    from luca.agent.core.models import SessionConfig
    from luca.agent.core.runner import AgentSessionRunner
    from luca.client.testing import FauxProvider

    _write(tmp_path, {
        "model": {"provider": "anthropic", "model": "claude-sonnet-5", "reasoning": "low"},
        "runtime": {"hard_max_steps": 42},
        "compaction": {"threshold": 0.66},
        "permissions": {"mode": "yolo"},
        "workspace": str(tmp_path),
        "models": {"anthropic": ["claude-sonnet-5"]},
    })
    config = load_luca_config(cwd=tmp_path, home=tmp_path / "none")
    session = AgentSessionRunner.new_session(faux_model())
    session.session_config.llm_config = resolve_llm_config(
        session.session_config.llm_config, config,
        {"model": None, "provider": None, "reasoning": None},
    )
    session.session_config.runtime_config = resolve_runtime_config(
        session.session_config.runtime_config, config,
    )
    app = AgentApp(
        session, provider=FauxProvider(),
        mode=config.permissions.mode.value, workspace=config.workspace,
        compactor=build_compactor(config, enabled=None, threshold=None, keep_turns=None),
        permission_rules=build_permission_rules(config) or None,
        recommended_models=config.models or None, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.runner.session.session_config == SessionConfig(
            llm_config=LLMConfig(
                provider="anthropic", model="claude-sonnet-5", reasoning="low",
            ),
            runtime_config=RuntimeConfig(hard_max_steps=42),
        )
        assert app._compactor.threshold == 0.66
        assert app.strategy.mode is PermissionMode.YOLO
        assert app.recommended_models == {"anthropic": ["claude-sonnet-5"]}
