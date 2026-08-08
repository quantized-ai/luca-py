"""`luca.json` — layered project + user configuration for the TUI.

By default two files are read and field-level deep-merged, project over home:
`~/.config/luca/luca.json` (personal defaults) then `./luca.json` (repo policy).
Everything is optional, validated with `extra="forbid"`, and applied with the
precedence **CLI flag > luca.json > persisted session > built-in default** — so
the file behaves like sticky CLI flags. It is pure data (no shell execution);
a malformed file raises `LucaConfigError` with a readable message.

The project file is the nearest `luca.json` at or above the cwd, bounded by the
repo. Naming one explicitly (`--config <path>`, or `LUCA_CONFIG_PATH`) REPLACES
that discovery, and a path that does not resolve is an error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from luca.agent.contrib.resource_permissions import (
    PermissionMatchMode,
    PermissionMode,
    ResourcePermission,
    ToolKindRule,
    ToolRule,
)
from luca.agent.contrib.simple_context_manager import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    SummarizingContextManager,
)
from luca.agent.core.models import ApprovalOption, LLMConfig, ModelOptions, RuntimeConfig, ToolKind
from luca.client.providers import PROVIDERS, register_provider
from luca.client.transports import TRANSPORTS
from luca.client.types import Reasoning

from .prompt_files import ReadLimits

_STRICT = ConfigDict(extra="forbid")


class LucaConfigError(Exception):
    """A luca.json that is missing-file-aside unreadable, non-JSON, or invalid."""


class ModelConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    reasoning: Reasoning | None = None
    model_config = _STRICT


class ThemeSettings(BaseModel):
    name: str | None = None
    model_config = _STRICT


class SessionSettings(BaseModel):
    """Where sessions are stored. `directory` is the ROOT; the per-project
    subdirectory is always appended under it."""

    directory: str | None = None
    model_config = _STRICT


class LoggingSettings(BaseModel):
    """Where the session log goes and how loud it is. `file` overrides the
    default `<session dir>/logs/<session-id>.log`; a level of `"OFF"` writes
    nothing at all."""

    level: str | None = None
    file: str | None = None
    model_config = _STRICT


class FileReadSettings(BaseModel):
    """The `@`-mention inline ceiling. The effective cap is the SMALLER of the
    hard limit and the model's context share, so a small-context model is never
    handed a 25k-token file just because the hard limit permits it."""

    max_read_file_tokens_hard_limit: int | None = None
    max_read_file_tokens_context_percentage: float | None = None
    model_config = _STRICT


class ClientSettings(BaseModel):
    file_read: FileReadSettings = Field(default_factory=FileReadSettings)
    model_config = _STRICT


class RuntimeSettings(BaseModel):
    """Every `RuntimeConfig` knob, all optional (unset = leave the session's)."""

    builtin_client_completion_timeout_in_ms: int | None = None
    client_completion_timeout_in_ms: int | None = None
    tool_execution_timeout_in_ms: int | None = None
    llm_completion_cancellation_grace_period: int | None = None
    tool_cancellation_grace_period: int | None = None
    soft_max_steps: int | None = None
    hard_max_steps: int | None = None
    doom_loop_threshold: int | None = None
    limit_tool_choice_on_soft_max_steps_reached: bool | None = None
    limit_tool_choice_on_doom_loop_flagged: bool | None = None
    model_config = _STRICT


class CompactionSettings(BaseModel):
    enabled: bool | None = None
    threshold: float | None = None
    keep_turns: int | None = None
    default_window: int | None = None
    model_config = _STRICT


class ModelOptionsBlock(BaseModel):
    """One `options` block: how a model is invoked, on top of which model it is.

    The ONE model in this file that is not `extra="forbid"`, deliberately.
    Known keys are typed and validated; every unknown key falls through
    verbatim into the provider's raw `provider_options`. That passthrough is
    the point — a provider's own wire fields (OpenRouter's `provider.order`,
    its `transforms`) move faster than this schema ever could, and forbidding
    extras would leave no way to reach them. Nothing is renamed or normalized
    on the way out: write the field the provider documents.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning: Reasoning | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("max_tokens")
    @classmethod
    def _positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be >= 1")
        return value

    @property
    def raw(self) -> dict:
        """The unknown keys, verbatim — the provider's own wire fields."""
        return dict(self.model_extra or {})


class ModelDef(BaseModel):
    options: ModelOptionsBlock = Field(default_factory=ModelOptionsBlock)
    model_config = _STRICT


class ProviderDef(BaseModel):
    """A provider entry: host REGISTRATION (`base_url` and friends), invocation
    SETTINGS (`options` / `models`), or both. `base_url` is what makes it a
    registration, and only a registration may not name a built-in provider —
    setting options on `openrouter` is the ordinary case."""

    base_url: str | None = None
    api_key_env: str | None = None
    transport: str = "openai"
    options: ModelOptionsBlock = Field(default_factory=ModelOptionsBlock)
    models: dict[str, ModelDef] = Field(default_factory=dict)
    model_config = _STRICT


class PermissionRule(BaseModel):
    """A config allow/deny rule → a `ToolKindRule` (when `tool_kind` is set) or
    a `ToolRule` over a `(permission, resource)` glob."""

    decision: Literal["allow", "deny"]
    tool_kind: ToolKind | None = None
    permission: str | None = None
    resource: str | None = None
    tool_name: str | None = None
    model_config = _STRICT

    @model_validator(mode="after")
    def _needs_a_target(self) -> PermissionRule:
        if self.tool_kind is None and self.permission is None:
            raise ValueError("a permission rule needs either 'tool_kind' or 'permission'")
        return self

    def to_rule(self) -> ToolKindRule | ToolRule:
        option = ApprovalOption.ALLOW if self.decision == "allow" else ApprovalOption.DENY
        if self.tool_kind is not None:
            return ToolKindRule(tool_kind=self.tool_kind, decision=option)
        if self.permission is None:
            raise LucaConfigError(
                "a permission rule needs either 'tool_kind' or 'permission'",
            )
        return ToolRule(
            tool_name=self.tool_name,
            resource_permission=ResourcePermission(
                permission=self.permission,
                resource=self.resource,
            ),
            decision=option,
        )


class PermissionSettings(BaseModel):
    mode: PermissionMode | None = None
    match_mode: PermissionMatchMode | None = None
    rules: list[PermissionRule] = Field(default_factory=list)
    model_config = _STRICT


class LucaConfig(BaseModel):
    schema_url: str | None = Field(default=None, alias="$schema")
    model: ModelConfig = Field(default_factory=ModelConfig)
    theme: ThemeSettings = Field(default_factory=ThemeSettings)
    sessions: SessionSettings = Field(default_factory=SessionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    client: ClientSettings = Field(default_factory=ClientSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    providers: dict[str, ProviderDef] = Field(default_factory=dict)
    models: dict[str, list[str]] = Field(default_factory=dict)
    workspace: str | None = None
    additional_directories: list[str] = Field(default_factory=list)
    # Extra roots to scan for `<name>/SKILL.md`; `~` expanded per entry.
    extra_skill_locations: list[str] = Field(default_factory=list)
    # Extra instruction files, on top of the discovered LUCA.md / AGENTS.md /
    # CLAUDE.md; `~` expanded, relative entries resolved against the workspace.
    instructions: list[str] = Field(default_factory=list)
    streaming: bool | None = None
    # Offer the provider's own native tools where the ACTIVE model supports
    # them (`--use-native` / `--no-use-native`; default on). Purely an
    # adaptation input: the same session is valid either way, and the tool set
    # is re-derived before every call.
    use_native_tools: bool | None = Field(
        default=None,
        description=(
            "Offer the provider's own native tools where the active model supports them "
            "(apply_patch + shell on OpenAI, text_editor + bash on Anthropic). Defaults to true."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ── loading ──────────────────────────────────────────────────────────────────

# Names a config file to use INSTEAD of the discovered pair. The `--config`
# flag is the same channel and wins over it.
ENV_CONFIG_PATH = "LUCA_CONFIG_PATH"


def config_home() -> Path:
    """`$XDG_CONFIG_HOME/luca` or `~/.config/luca`."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def resolve_config_path(cli_path: str | None = None) -> Path | None:
    """The explicitly named config file, or `None` to discover. CLI flag over
    `LUCA_CONFIG_PATH`; `~` expanded. Kept out of `load_luca_config` so that
    stays a pure function of its arguments."""
    for value in (cli_path, os.environ.get(ENV_CONFIG_PATH)):
        if value:
            return Path(value).expanduser()
    return None


# Accepted spellings for a canonical top-level key. `provider` reads naturally
# for a file that configures one, and failing a whole config over the singular
# helps nobody. Only the canonical name reaches the model, and only it is in the
# JSON schema.
_KEY_ALIASES = {"provider": "providers"}


def _normalize_keys(data: dict, path: Path) -> dict:
    """Rename accepted aliases to their canonical key. Applied per FILE, before
    the merge: a home file spelling it one way and a project file the other must
    still deep-merge into one map rather than land as two keys, one of which
    silently loses."""
    for alias, canonical in _KEY_ALIASES.items():
        if alias not in data:
            continue
        if canonical in data:
            raise LucaConfigError(f"{path}: has both {alias!r} and {canonical!r}; keep one (they are the same key)")
        data = {(canonical if key == alias else key): value for key, value in data.items()}
    return data


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LucaConfigError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise LucaConfigError(f"{path}: the top level must be a JSON object")
    return _normalize_keys(data, path)


def find_project_config(start: Path) -> Path | None:
    """The nearest `luca.json` at or above `start`, so a project config applies
    from any subdirectory. Two bounds keep a file outside the project from
    silently applying: the directory holding `.git`, and the home directory when
    there is no repo. `exists()`, not `is_dir()`: a worktree's `.git` is a file."""
    # Both sides resolved, or the comparison misses whenever a symlink is in
    # play (/tmp vs /private/tmp on macOS) and the bound silently never fires.
    home = Path.home().resolve()
    start = start.resolve()
    for directory in (start, *start.parents):
        if directory == home and directory != start:
            break  # ~/luca.json is not the project config for everything below it
        candidate = directory / "luca.json"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break
    return None


def _read_named_config(path: Path) -> dict:
    """Read a config file the caller NAMED. A missing one is an error here,
    unlike a discovered file, which is simply empty."""
    if not path.is_file():
        raise LucaConfigError(f"{path}: not a readable config file")
    return _read_json_object(path)


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` wins per key; nested objects merge, scalars and lists replace."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_luca_config(
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    path: Path | None = None,
) -> LucaConfig:
    """Read the home then project `luca.json`, deep-merge, validate. The project
    file is the nearest at or above `cwd`. `path` REPLACES that discovery
    entirely; resolve it with `resolve_config_path()`."""
    if path is not None:
        merged = _read_named_config(path)
        read_from = [path]
    else:
        cwd = cwd or Path.cwd()
        home = home if home is not None else config_home()
        project = find_project_config(cwd)
        home_file = home / "luca.json"
        merged = _deep_merge(
            _read_json_object(home_file),
            _read_json_object(project) if project is not None else {},
        )
        # Named in full: with the walk, the offending file may be several
        # directories up and "luca.json is invalid" would send you to the wrong one.
        read_from = [p for p in (home_file, project) if p is not None and p.is_file()]
    try:
        return LucaConfig.model_validate(merged)
    except ValidationError as exc:
        source = " + ".join(str(p) for p in read_from) or "luca.json"
        raise LucaConfigError(f"{source} is invalid:\n{exc}") from exc


# ── applying (precedence: cli > config > base/default) ───────────────────────


def pick(cli_value, config_value, default):
    """First of cli / config / default that is set (`None` = unset)."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


# Providers with a dedicated transport/behavior: overriding one from config
# would silently swap its wire format (transport defaults to "openai"), so a
# custom host must use a distinct name.
_FIRST_CLASS_PROVIDERS = frozenset({"openai", "anthropic", "openrouter", "bedrock", "faux"})


def register_config_providers(config: LucaConfig) -> None:
    """Register every custom HOST so a call can route to it.

    An entry without `base_url` registers nothing: it only carries settings for
    a provider that already exists. Those are checked after the loop, once every
    registration in this file has landed, so the two can appear in either order."""
    settings_only = []
    for name, defn in config.providers.items():
        if defn.base_url is None:
            settings_only.append(name)
            continue
        if name in _FIRST_CLASS_PROVIDERS:
            raise LucaConfigError(
                f"provider {name!r} is built in; give a custom host a distinct name and point model.provider at it",
            )
        transport = TRANSPORTS.get(defn.transport)
        if transport is None:
            raise LucaConfigError(
                f"provider {name!r}: unknown transport {defn.transport!r} (one of {', '.join(sorted(TRANSPORTS))})",
            )
        register_provider(
            name,
            {
                "default_base_url": defn.base_url,
                "default_api_key_env_var": defn.api_key_env,
                "default_transport_class": transport,
            },
        )
    for name in settings_only:
        # A typo here would otherwise be silent: the options would resolve for
        # a provider nothing ever selects, and the model would run unconfigured.
        # `_FIRST_CLASS_PROVIDERS` as well as the registry, because `faux` is
        # reachable by instance injection and never registered.
        if name not in PROVIDERS and name not in _FIRST_CLASS_PROVIDERS:
            raise LucaConfigError(
                f"provider {name!r} has options but no base_url, and is not a known provider; "
                "add base_url to register the host, or fix the name",
            )


def resolve_model_options(
    config: LucaConfig,
    provider: str,
    model: str,
) -> tuple[ModelOptions | None, str | None]:
    """The `options` for one `(provider, model)` pair, as the two things it
    lands on: the `ModelOptions` the client call takes, and the `reasoning`
    that belongs on `LLMConfig` itself.

    The provider-wide block resolves first and the model's own block wins over
    it, per key — raw passthrough keys deep-merge (nested objects merge,
    scalars and lists replace), so a provider-wide `provider.order` survives a
    model that only sets `transforms`. `(None, None)` when nothing is
    configured, which is what lets an unconfigured session stay untouched."""
    defn = config.providers.get(provider)
    if defn is None:
        return None, None
    blocks = [defn.options]
    model_def = defn.models.get(model)
    if model_def is not None:
        blocks.append(model_def.options)

    typed: dict = {}
    raw: dict = {}
    reasoning: str | None = None
    for block in blocks:
        raw = _deep_merge(raw, block.raw)
        for field in ("max_tokens", "temperature", "top_p"):
            value = getattr(block, field)
            if value is not None:
                typed[field] = value
        if block.reasoning is not None:
            reasoning = block.reasoning

    if not typed and not raw:
        return None, reasoning
    # Keyed by provider name, the shape luca.client takes: a turn a middleware
    # routed to another provider finds no entry under its own name and sends
    # none of this, rather than one provider's wire fields to another.
    return ModelOptions(**typed, provider_options={provider: raw} if raw else None), reasoning


def apply_model_options(
    llm_config: LLMConfig,
    config: LucaConfig,
    *,
    cli_reasoning: str | None = None,
) -> LLMConfig:
    """Resolve `options` for whichever `(provider, model)` this config names.

    `options` is always assigned, including to `None` — switching to a model
    with no block has to CLEAR the previous model's settings, not inherit them.
    `reasoning` is only taken when the CLI did not name one."""
    options, reasoning = resolve_model_options(config, llm_config.provider, llm_config.model)
    updates: dict = {"options": options}
    if reasoning is not None and cli_reasoning is None:
        updates["reasoning"] = reasoning
    return llm_config.model_copy(update=updates)


def resolve_llm_config(base: LLMConfig, config: LucaConfig, cli: dict) -> LLMConfig:
    """`config.model` over `base`, then CLI over both, then the per-provider /
    per-model `options` for whichever pair that lands on."""
    updates: dict = {}
    for field in ("provider", "model", "reasoning"):
        value = getattr(config.model, field)
        if value is not None:
            updates[field] = value
    updates.update({key: value for key, value in cli.items() if value is not None})
    resolved = base.model_copy(update=updates) if updates else base
    return apply_model_options(resolved, config, cli_reasoning=cli.get("reasoning"))


def picker_models(config: LucaConfig) -> dict[str, list[str]]:
    """The `/model` picker's configured list: the top-level `models` map unioned
    with every model carrying a settings block, so configuring a model never
    means also listing it by hand."""
    merged = {provider: list(ids) for provider, ids in config.models.items()}
    for provider, defn in config.providers.items():
        if defn.models:
            merged[provider] = sorted(set(merged.get(provider, [])) | set(defn.models))
    return merged


def resolve_runtime_config(base: RuntimeConfig, config: LucaConfig) -> RuntimeConfig:
    """Set `config.runtime` fields over the session's persisted runtime.

    Re-validated (not `model_copy`) so an out-of-range value from the file hits
    RuntimeConfig's own validator instead of slipping through."""
    updates = config.runtime.model_dump(exclude_none=True)
    if not updates:
        return base
    try:
        return RuntimeConfig.model_validate({**base.model_dump(), **updates})
    except ValidationError as exc:
        raise LucaConfigError(f"luca.json runtime is invalid:\n{exc}") from exc


def resolve_read_limits(config: LucaConfig) -> ReadLimits:
    """`config.client.file_read` over the built-in defaults."""
    settings = config.client.file_read
    defaults = ReadLimits()
    return ReadLimits(
        hard_limit=pick(None, settings.max_read_file_tokens_hard_limit, defaults.hard_limit),
        context_percentage=pick(
            None,
            settings.max_read_file_tokens_context_percentage,
            defaults.context_percentage,
        ),
    )


def build_permission_rules(config: LucaConfig) -> list[ToolKindRule | ToolRule]:
    return [rule.to_rule() for rule in config.permissions.rules]


def build_context_manager(
    config: LucaConfig,
    *,
    provider=None,
    enabled: bool | None,
    threshold: float | None,
    keep_turns: int | None,
) -> SummarizingContextManager:
    enabled = pick(enabled, config.compaction.enabled, True)
    threshold = pick(threshold, config.compaction.threshold, DEFAULT_THRESHOLD)
    keep_turns = pick(keep_turns, config.compaction.keep_turns, 0)
    default_window = pick(None, config.compaction.default_window, DEFAULT_WINDOW)
    return SummarizingContextManager(
        keep_turns=keep_turns,
        threshold=threshold,
        default_window=default_window,
        enabled=enabled,
        provider=provider,
    )
