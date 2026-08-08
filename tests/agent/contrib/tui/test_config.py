"""luca.json: validation, home+project deep-merge, precedence, and the
config → objects mappings (providers, permission rules, compaction policy)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import (
    PermissionMode,
    ResourcePermission,
    ToolKindRule,
    ToolRule,
)
from luca.agent.contrib.tui.config import (
    ENV_CONFIG_PATH,
    CompactionSettings,
    LoggingSettings,
    LucaConfig,
    LucaConfigError,
    ModelConfig,
    PermissionRule,
    RuntimeSettings,
    ThemeSettings,
    _deep_merge,
    apply_model_options,
    build_context_manager,
    build_permission_rules,
    load_luca_config,
    pick,
    picker_models,
    register_config_providers,
    resolve_config_path,
    resolve_llm_config,
    resolve_model_options,
    resolve_read_limits,
    resolve_runtime_config,
)
from luca.agent.contrib.tui.prompt_files import ReadLimits
from luca.agent.core.models import ApprovalOption, LLMConfig, ModelOptions, RuntimeConfig, ToolKind
from luca.client.providers import PROVIDERS


def _write(directory, payload):
    (directory / "luca.json").write_text(json.dumps(payload))


# ── validation ───────────────────────────────────────────────────────────────


def test_a_partial_config_is_valid_and_defaults_the_rest():
    assert LucaConfig.model_validate({"model": {"model": "x"}}) == LucaConfig(
        model=ModelConfig(model="x"),
    )


def test_theme_name_is_a_strict_optional_config_section():
    assert LucaConfig.model_validate({"theme": {"name": "nord"}}) == LucaConfig(
        theme=ThemeSettings(name="nord"),
    )


def test_luca_schema_describes_the_theme_section():
    schema = json.loads((Path(__file__).parents[4] / "luca.schema.json").read_text())

    assert (
        schema["properties"]["theme"],
        schema["$defs"]["ThemeSettings"],
    ) == (
        {"$ref": "#/$defs/ThemeSettings"},
        {
            "additionalProperties": False,
            "properties": {
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Name",
                }
            },
            "title": "ThemeSettings",
            "type": "object",
        },
    )


def test_extra_skill_locations_parse_as_a_list_of_strings():
    config = LucaConfig.model_validate({"extra_skill_locations": [".opencode/skills/", "~/.config/opencode/skills/"]})

    assert config.extra_skill_locations == [".opencode/skills/", "~/.config/opencode/skills/"]


def test_extra_skill_locations_default_to_empty():
    assert LucaConfig().extra_skill_locations == []


def test_instructions_parse_as_a_list_of_strings():
    config = LucaConfig.model_validate({"instructions": ["docs/conventions.md", "~/notes/style.md"]})

    assert config.instructions == ["docs/conventions.md", "~/notes/style.md"]


def test_instructions_default_to_empty():
    assert LucaConfig().instructions == []


def test_the_sessions_directory_parses():
    assert LucaConfig.model_validate({"sessions": {"directory": "~/somewhere"}}).sessions.directory == "~/somewhere"


def test_the_sessions_directory_defaults_to_unset():
    assert LucaConfig().sessions.directory is None


def test_an_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        LucaConfig.model_validate({"nope": 1})


def test_the_schema_alias_is_accepted_and_ignored():
    assert LucaConfig.model_validate({"$schema": "./luca.schema.json"}).schema_url == ("./luca.schema.json")


# ── deep merge (project over home) ───────────────────────────────────────────


def test_a_project_value_overrides_the_same_home_key_leaving_siblings():
    merged = _deep_merge(
        {"model": {"model": "home", "provider": "openrouter"}},
        {"model": {"model": "project"}},
    )
    assert merged == {"model": {"model": "project", "provider": "openrouter"}}


def test_provider_maps_merge_per_key_but_rule_lists_are_replaced(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    _write(
        home,
        {
            "providers": {"a": {"base_url": "https://a"}},
            "permissions": {"rules": [{"decision": "allow", "tool_kind": "read"}]},
        },
    )
    _write(
        project,
        {
            "providers": {"b": {"base_url": "https://b"}},
            "permissions": {"rules": [{"decision": "deny", "tool_kind": "execute"}]},
        },
    )
    config = load_luca_config(cwd=project, home=home)
    assert set(config.providers) == {"a", "b"}  # provider maps union
    assert len(config.permissions.rules) == 1  # project's list replaces home's
    assert config.permissions.rules[0].tool_kind is ToolKind.EXECUTE


def test_missing_files_give_an_empty_config(tmp_path):
    (tmp_path / ".git").mkdir()  # bound the walk; else it reaches the real fs
    assert load_luca_config(cwd=tmp_path, home=tmp_path / "nope") == LucaConfig()


# ── finding the project file (walk up, bounded by the repo) ──────────────────


def test_a_project_config_applies_from_a_subdirectory(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, {"model": {"model": "project"}})
    deep = tmp_path / "src" / "deep"
    deep.mkdir(parents=True)

    config = load_luca_config(cwd=deep, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="project"))


def test_the_nearest_project_config_wins(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, {"model": {"model": "outer"}})
    inner = tmp_path / "packages" / "app"
    inner.mkdir(parents=True)
    _write(inner, {"model": {"model": "inner"}})

    config = load_luca_config(cwd=inner, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="inner"))


def test_the_walk_stops_at_the_repository_boundary(tmp_path):
    _write(tmp_path, {"model": {"model": "outside-the-repo"}})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    config = load_luca_config(cwd=repo, home=tmp_path / "none")

    assert config == LucaConfig()


def test_the_repo_root_config_is_read_even_though_it_holds_the_git_marker(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "src"
    source.mkdir(parents=True)
    (repo / ".git").mkdir()
    _write(repo, {"model": {"model": "repo-root"}})

    config = load_luca_config(cwd=source, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="repo-root"))


def test_a_git_FILE_bounds_the_walk_too(tmp_path):
    _write(tmp_path, {"model": {"model": "outside-the-repo"}})
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")

    config = load_luca_config(cwd=worktree, home=tmp_path / "none")

    assert config == LucaConfig()


def test_the_walk_stops_at_home_when_there_is_no_repo(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    work = fake_home / "scratch"
    work.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _write(tmp_path, {"model": {"model": "above-home"}})

    config = load_luca_config(cwd=work, home=tmp_path / "none")

    assert config == LucaConfig()


def test_a_luca_json_in_the_home_directory_is_not_a_project_config(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    work = fake_home / "scratch"
    work.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _write(fake_home, {"model": {"model": "home-root"}})

    config = load_luca_config(cwd=work, home=tmp_path / "none")

    assert config == LucaConfig()


def test_home_is_still_read_when_it_is_where_you_are_standing(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _write(fake_home, {"model": {"model": "home-root"}})

    config = load_luca_config(cwd=fake_home, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="home-root"))


def test_the_home_bound_holds_when_home_is_reached_through_a_symlink(tmp_path, monkeypatch):
    """Both sides are resolved before comparing. Unresolved, `/tmp` never
    equals `/private/tmp` and the bound silently never fires."""
    real_home = tmp_path / "real_home"
    (real_home / "scratch").mkdir(parents=True)
    linked_home = tmp_path / "linked_home"
    linked_home.symlink_to(real_home)
    monkeypatch.setattr(Path, "home", lambda: linked_home)
    _write(real_home, {"model": {"model": "home-root"}})

    config = load_luca_config(cwd=real_home / "scratch", home=tmp_path / "none")

    assert config == LucaConfig()


def test_a_repo_below_home_is_unaffected_by_the_home_bound(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    repo = fake_home / "code" / "myproject"
    source = repo / "src" / "deep"
    source.mkdir(parents=True)
    (repo / ".git").mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _write(repo, {"model": {"model": "project"}})

    config = load_luca_config(cwd=source, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="project"))


def test_an_explicit_path_bypasses_the_walk_entirely(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, {"model": {"model": "project"}})
    deep = tmp_path / "src"
    deep.mkdir()
    named = tmp_path / "named.json"
    named.write_text(json.dumps({"model": {"model": "named"}}))

    config = load_luca_config(cwd=deep, home=tmp_path / "none", path=named)

    assert config == LucaConfig(model=ModelConfig(model="named"))


# ── an explicitly named config file ──────────────────────────────────────────


def test_an_explicit_path_replaces_both_discovered_files(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    _write(home, {"model": {"model": "home"}, "streaming": False})
    _write(tmp_path, {"model": {"model": "project"}, "workspace": "/repo"})
    (tmp_path / "named.json").write_text(json.dumps({"model": {"model": "named"}}))

    config = load_luca_config(cwd=tmp_path, home=home, path=tmp_path / "named.json")

    assert config == LucaConfig(model=ModelConfig(model="named"))


def test_an_explicit_path_that_does_not_exist_raises(tmp_path):
    with pytest.raises(LucaConfigError, match="not a readable config file"):
        load_luca_config(path=tmp_path / "nope.json")


def test_an_explicit_path_that_is_a_directory_raises(tmp_path):
    with pytest.raises(LucaConfigError, match="not a readable config file"):
        load_luca_config(path=tmp_path)


def test_an_explicit_non_json_path_raises_the_same_readable_error(tmp_path):
    named = tmp_path / "named.json"
    named.write_text("{ not json")
    with pytest.raises(LucaConfigError, match="not valid JSON"):
        load_luca_config(path=named)


def test_an_invalid_explicit_config_names_the_file_it_read(tmp_path):
    named = tmp_path / "named.json"
    named.write_text(json.dumps({"unknown_key": True}))
    with pytest.raises(LucaConfigError, match=f"{named} is invalid"):
        load_luca_config(path=named)


# ── resolving which file to read (flag > env > discovery) ────────────────────


def test_the_config_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_CONFIG_PATH, "/from/env.json")
    assert resolve_config_path("/from/flag.json") == Path("/from/flag.json")


def test_the_environment_is_used_when_no_flag_is_given(monkeypatch):
    monkeypatch.setenv(ENV_CONFIG_PATH, "/from/env.json")
    assert resolve_config_path(None) == Path("/from/env.json")


def test_no_flag_and_no_environment_means_discovery(monkeypatch):
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    assert resolve_config_path(None) is None


def test_an_exported_but_empty_environment_variable_means_discovery(monkeypatch):
    monkeypatch.setenv(ENV_CONFIG_PATH, "")
    assert resolve_config_path(None) is None


def test_a_tilde_is_expanded_in_both_channels(monkeypatch):
    monkeypatch.setenv(ENV_CONFIG_PATH, "~/env.json")
    assert (resolve_config_path("~/flag.json"), resolve_config_path(None)) == (
        Path.home() / "flag.json",
        Path.home() / "env.json",
    )


def test_load_luca_config_never_reads_the_environment_itself(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_CONFIG_PATH, str(tmp_path / "ambient.json"))
    _write(tmp_path, {"model": {"model": "project"}})

    config = load_luca_config(cwd=tmp_path, home=tmp_path / "none")

    assert config == LucaConfig(model=ModelConfig(model="project"))


# ── precedence (cli > config > base/default) ─────────────────────────────────


def test_pick_prefers_cli_then_config_then_default():
    assert pick("cli", "config", "default") == "cli"
    assert pick(None, "config", "default") == "config"
    assert pick(None, None, "default") == "default"


def test_llm_precedence_is_cli_over_config_over_base():
    base = LLMConfig(model="base", provider="openrouter")
    config = LucaConfig(model=ModelConfig(model="from-config", reasoning="high"))
    assert resolve_llm_config(base, config, {"model": "from-cli"}) == LLMConfig(
        model="from-cli",
        provider="openrouter",
        reasoning="high",
    )


def test_runtime_config_fields_apply_over_the_persisted_runtime():
    base = RuntimeConfig(soft_max_steps=5)
    config = LucaConfig(runtime=RuntimeSettings(hard_max_steps=42))
    assert resolve_runtime_config(base, config) == RuntimeConfig(
        soft_max_steps=5,
        hard_max_steps=42,
    )


def test_pick_treats_falsy_but_set_values_as_set():
    assert pick(None, False, True) is False
    assert pick(None, 0, 5) == 0
    assert pick(None, 0.0, 0.8) == 0.0


def test_runtime_keeps_falsy_values_and_rejects_out_of_range():
    base = RuntimeConfig(soft_max_steps=5)
    kept = resolve_runtime_config(
        base,
        LucaConfig(
            runtime=RuntimeSettings(
                hard_max_steps=0,
                limit_tool_choice_on_soft_max_steps_reached=False,
            )
        ),
    )
    assert kept == RuntimeConfig(
        soft_max_steps=5,
        hard_max_steps=0,
        limit_tool_choice_on_soft_max_steps_reached=False,
    )
    with pytest.raises(LucaConfigError):
        resolve_runtime_config(base, LucaConfig(runtime=RuntimeSettings(hard_max_steps=-5)))


def test_context_manager_precedence_cli_over_config_over_default():
    config = LucaConfig(compaction=CompactionSettings(threshold=0.6, keep_turns=2))
    from_config = build_context_manager(config, enabled=None, threshold=None, keep_turns=None)
    assert from_config.threshold == 0.6
    assert from_config.keep_turns == 2

    cli_wins = build_context_manager(config, enabled=False, threshold=0.9, keep_turns=0)
    assert cli_wins.threshold == 0.9
    assert cli_wins.enabled is False
    assert cli_wins.keep_turns == 0


# ── config → objects ─────────────────────────────────────────────────────────


def test_a_tool_kind_rule_and_a_resource_rule_map_correctly():
    kind = PermissionRule(decision="allow", tool_kind=ToolKind.READ)
    resource = PermissionRule(decision="deny", permission="bash", resource="/etc/*")
    assert kind.to_rule() == ToolKindRule(
        tool_kind=ToolKind.READ,
        decision=ApprovalOption.ALLOW,
    )
    assert resource.to_rule() == ToolRule(
        resource_permission=ResourcePermission(permission="bash", resource="/etc/*"),
        decision=ApprovalOption.DENY,
    )


def test_a_rule_without_kind_or_permission_is_rejected_at_parse_time():
    # fails during LucaConfig validation, so load_luca_config's guard catches it
    with pytest.raises(ValidationError):
        LucaConfig.model_validate({"permissions": {"rules": [{"decision": "allow"}]}})


def test_a_bad_rule_in_a_file_surfaces_as_a_config_error(tmp_path):
    (tmp_path / "luca.json").write_text('{"permissions": {"rules": [{"decision": "allow"}]}}')
    with pytest.raises(LucaConfigError):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


def test_shadowing_a_first_class_provider_is_rejected():
    config = LucaConfig.model_validate(
        {
            "providers": {"anthropic": {"base_url": "https://proxy"}},
        }
    )
    with pytest.raises(LucaConfigError, match="built in"):
        register_config_providers(config)


def test_register_config_providers_adds_a_custom_host():
    config = LucaConfig.model_validate(
        {
            "providers": {"lc-test-host": {"base_url": "https://x/v1", "api_key_env": "X_KEY"}},
        }
    )
    register_config_providers(config)
    assert PROVIDERS["lc-test-host"] == {
        "default_base_url": "https://x/v1",
        "default_api_key_env_var": "X_KEY",
        "default_transport_class": __import__(
            "luca.client.transports",
            fromlist=["OpenAITransport"],
        ).OpenAITransport,
    }


def test_an_unknown_transport_is_rejected():
    config = LucaConfig.model_validate(
        {
            "providers": {"bad": {"base_url": "https://x", "transport": "nope"}},
        }
    )
    with pytest.raises(LucaConfigError, match="unknown transport"):
        register_config_providers(config)


# ── model options ────────────────────────────────────────────────────────────

# The config from the issue that asked for this, verbatim in our shape: a
# provider-wide default, one model overriding it, and raw keys OpenRouter owns.
OPTIONS_CONFIG = LucaConfig.model_validate(
    {
        "providers": {
            "openrouter": {
                "options": {
                    "max_tokens": 8000,
                    "top_p": 0.9,
                    "provider": {"order": ["baseten", "together"], "allow_fallbacks": True},
                },
                "models": {
                    "moonshotai/kimi-k2:free": {
                        "options": {
                            "max_tokens": 6000,
                            "reasoning": "high",
                            "transforms": ["middle-out"],
                        },
                    },
                },
            },
        },
    }
)


def test_a_model_block_wins_per_key_over_the_provider_wide_one():
    # max_tokens is overridden, top_p is inherited, and the raw keys MERGE
    # rather than replace — the provider-wide routing survives a model that
    # only sets `transforms`.
    assert resolve_model_options(OPTIONS_CONFIG, "openrouter", "moonshotai/kimi-k2:free") == (
        ModelOptions(
            max_tokens=6000,
            top_p=0.9,
            provider_options={
                "openrouter": {
                    "provider": {"order": ["baseten", "together"], "allow_fallbacks": True},
                    "transforms": ["middle-out"],
                },
            },
        ),
        "high",
    )


def test_a_model_with_no_block_of_its_own_gets_the_provider_wide_options():
    assert resolve_model_options(OPTIONS_CONFIG, "openrouter", "some/other-model") == (
        ModelOptions(
            max_tokens=8000,
            top_p=0.9,
            provider_options={
                "openrouter": {"provider": {"order": ["baseten", "together"], "allow_fallbacks": True}},
            },
        ),
        None,
    )


def test_an_unconfigured_provider_resolves_to_nothing():
    assert resolve_model_options(OPTIONS_CONFIG, "anthropic", "claude-sonnet-5") == (None, None)


def test_applying_options_sets_reasoning_from_the_model_block():
    base = LLMConfig(provider="openrouter", model="moonshotai/kimi-k2:free")
    assert apply_model_options(base, OPTIONS_CONFIG).reasoning == "high"


def test_a_cli_reasoning_beats_the_model_block():
    base = LLMConfig(provider="openrouter", model="moonshotai/kimi-k2:free", reasoning="low")
    assert apply_model_options(base, OPTIONS_CONFIG, cli_reasoning="low").reasoning == "low"


def test_switching_to_an_unconfigured_model_clears_the_previous_options():
    # The reason `options` is always assigned: inheriting the last model's
    # max_tokens after a switch would be silent and wrong.
    configured = apply_model_options(
        LLMConfig(provider="openrouter", model="moonshotai/kimi-k2:free"),
        OPTIONS_CONFIG,
    )
    assert apply_model_options(configured.model_copy(update={"provider": "anthropic"}), OPTIONS_CONFIG) == LLMConfig(
        provider="anthropic",
        model="moonshotai/kimi-k2:free",
        reasoning="high",
        options=None,
    )


def test_resolve_llm_config_resolves_options_for_the_pair_it_lands_on():
    base = LLMConfig(model="base", provider="faux")
    config = LucaConfig.model_validate(
        {
            "model": {"provider": "openrouter", "model": "moonshotai/kimi-k2:free"},
            "providers": OPTIONS_CONFIG.model_dump(exclude_none=True)["providers"],
        }
    )
    assert resolve_llm_config(base, config, {"model": None, "provider": None, "reasoning": None}) == LLMConfig(
        provider="openrouter",
        model="moonshotai/kimi-k2:free",
        reasoning="high",
        options=ModelOptions(
            max_tokens=6000,
            top_p=0.9,
            provider_options={
                "openrouter": {
                    "provider": {"order": ["baseten", "together"], "allow_fallbacks": True},
                    "transforms": ["middle-out"],
                },
            },
        ),
    )


def test_a_max_tokens_below_one_is_rejected():
    with pytest.raises(ValidationError):
        LucaConfig.model_validate({"providers": {"openrouter": {"options": {"max_tokens": 0}}}})


def test_options_on_a_built_in_provider_are_allowed_without_a_base_url():
    config = LucaConfig.model_validate({"providers": {"openrouter": {"options": {"max_tokens": 100}}}})
    register_config_providers(config)  # settings only — registers nothing, raises nothing


def test_a_settings_only_entry_naming_an_unreachable_provider_is_rejected():
    config = LucaConfig.model_validate({"providers": {"typoed-host": {"options": {"max_tokens": 100}}}})
    with pytest.raises(LucaConfigError, match="not a known provider"):
        register_config_providers(config)


def test_picker_models_unions_the_settings_table_with_the_models_list():
    config = LucaConfig.model_validate(
        {
            "models": {"openrouter": ["z/listed"], "anthropic": ["claude-sonnet-5"]},
            "providers": {"openrouter": {"models": {"a/configured": {}}}},
        }
    )
    assert picker_models(config) == {
        "openrouter": ["a/configured", "z/listed"],
        "anthropic": ["claude-sonnet-5"],
    }


def test_the_singular_provider_key_is_accepted_as_an_alias(tmp_path):
    # The singular reads naturally for a file configuring one provider, and
    # only the canonical key survives onto the model.
    _write(tmp_path, {"provider": {"openrouter": {"options": {"max_tokens": 6000}}}})

    config = load_luca_config(cwd=tmp_path, home=tmp_path / "none")

    assert config.providers["openrouter"].options.max_tokens == 6000


def test_both_spellings_in_one_file_are_rejected(tmp_path):
    _write(tmp_path, {"provider": {"a": {"base_url": "x"}}, "providers": {"b": {"base_url": "y"}}})

    with pytest.raises(LucaConfigError, match="keep one"):
        load_luca_config(cwd=tmp_path, home=tmp_path / "none")


def test_the_two_spellings_merge_across_files_instead_of_shadowing(tmp_path):
    # Normalizing per file, before the merge, is what makes this one map: with
    # the rename after the merge, whichever key lost would vanish silently.
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    _write(home, {"provider": {"openrouter": {"options": {"max_tokens": 6000}}}})
    _write(project, {"providers": {"anthropic": {"options": {"temperature": 0.5}}}})

    config = load_luca_config(cwd=project, home=home)

    assert sorted(config.providers) == ["anthropic", "openrouter"]


def test_luca_schema_describes_the_provider_options():
    schema = json.loads((Path(__file__).parents[4] / "luca.schema.json").read_text())

    assert (
        schema["$defs"]["ProviderDef"]["properties"]["models"],
        schema["$defs"]["ModelDef"]["properties"],
        # additionalProperties true is the raw escape hatch, and the whole
        # reason this one block is not extra="forbid".
        schema["$defs"]["ModelOptionsBlock"]["additionalProperties"],
        sorted(schema["$defs"]["ModelOptionsBlock"]["properties"]),
    ) == (
        {"additionalProperties": {"$ref": "#/$defs/ModelDef"}, "title": "Models", "type": "object"},
        {"options": {"$ref": "#/$defs/ModelOptionsBlock"}},
        True,
        ["max_tokens", "reasoning", "temperature", "top_p"],
    )


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

    # "luca-dark" rather than a stock Textual theme: the frame's widgets
    # resolve the luca theme variables at render time, and a stock theme
    # (missing them) crashes the first paint. The CLI-layer tests cover the
    # theme value flowing through without rendering.
    _write(
        tmp_path,
        {
            "model": {"provider": "anthropic", "model": "claude-sonnet-5", "reasoning": "low"},
            "theme": {"name": "luca-dark"},
            "runtime": {"hard_max_steps": 42},
            "compaction": {"threshold": 0.66},
            "permissions": {"mode": "yolo"},
            "workspace": str(tmp_path),
            "models": {"anthropic": ["claude-sonnet-5"]},
        },
    )
    config = load_luca_config(cwd=tmp_path, home=tmp_path / "none")
    session = AgentSessionRunner.new_session(faux_model())
    session.session_config.llm_config = resolve_llm_config(
        session.session_config.llm_config,
        config,
        {"model": None, "provider": None, "reasoning": None},
    )
    session.session_config.runtime_config = resolve_runtime_config(
        session.session_config.runtime_config,
        config,
    )
    app = AgentApp(
        session,
        provider=FauxProvider(),
        theme=config.theme.name,
        mode=config.permissions.mode.value,
        workspace=config.workspace,
        context_manager=build_context_manager(config, enabled=None, threshold=None, keep_turns=None),
        permission_rules=build_permission_rules(config) or None,
        recommended_models=config.models or None,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.runner.session.session_config == SessionConfig(
            llm_config=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-5",
                reasoning="low",
            ),
            runtime_config=RuntimeConfig(hard_max_steps=42),
        )
        assert app._context_manager.threshold == 0.66
        assert app.strategy.mode is PermissionMode.YOLO
        assert app.recommended_models == {"anthropic": ["claude-sonnet-5"]}
        assert app.theme == "luca-dark"


# ── client.file_read — the `@`-mention inline cap ────────────────────────────


def test_the_file_read_section_parses():
    config = LucaConfig.model_validate(
        {
            "client": {
                "file_read": {
                    "max_read_file_tokens_hard_limit": 8000,
                    "max_read_file_tokens_context_percentage": 0.1,
                }
            }
        }
    )

    assert resolve_read_limits(config) == ReadLimits(hard_limit=8000, context_percentage=0.1)


def test_the_file_read_section_is_optional_and_falls_back_to_the_defaults():
    assert resolve_read_limits(LucaConfig()) == ReadLimits(hard_limit=25_000, context_percentage=0.05)


def test_a_partial_file_read_section_keeps_the_other_default():
    config = LucaConfig.model_validate({"client": {"file_read": {"max_read_file_tokens_hard_limit": 500}}})

    assert resolve_read_limits(config) == ReadLimits(hard_limit=500, context_percentage=0.05)


def test_an_unknown_key_under_client_is_rejected():
    with pytest.raises(ValidationError):
        LucaConfig.model_validate({"client": {"nope": {}}})


# ── logging ──────────────────────────────────────────────────────────────────


def test_logging_is_a_strict_optional_config_section():
    assert LucaConfig.model_validate({"logging": {"level": "DEBUG", "file": "/tmp/luca.log"}}) == LucaConfig(
        logging=LoggingSettings(level="DEBUG", file="/tmp/luca.log"),
    )


def test_luca_schema_describes_the_logging_section():
    schema = json.loads((Path(__file__).parents[4] / "luca.schema.json").read_text())

    assert (
        schema["properties"]["logging"],
        schema["$defs"]["LoggingSettings"]["properties"],
    ) == (
        {"$ref": "#/$defs/LoggingSettings"},
        {
            "level": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Level",
            },
            "file": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "File",
            },
        },
    )
