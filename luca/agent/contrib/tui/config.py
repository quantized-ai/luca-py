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
from luca.agent.core.models import ApprovalOption, LLMConfig, RuntimeConfig, ToolKind
from luca.agent.core.runner import completion_options
from luca.client.exceptions import ClientError
from luca.client.providers import PROVIDERS, resolve_provider
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
    """One `options` block: the `luca.client.acompletion` keyword arguments a
    call is made with, on top of WHICH model it is.

    The ONE model in this file that is not `extra="forbid"`, deliberately.
    The four knobs almost everyone sets are typed and validated here, because
    `max_tokens: 0` and `reasoning: "huge"` are worth catching by name at boot
    rather than as a provider 400 mid-turn. Every other key falls through
    verbatim: the client takes a dozen more (`seed`, `stop`, `top_k`,
    `presence_penalty`, …) and enumerating them would only mean this file
    refusing a legitimate argument every time the client gains one.

    A key here that is NOT an `acompletion` argument is a `TypeError` on the
    first turn. Raw fields the PROVIDER documents belong in the sibling
    `provider_options` block, which is the one with no schema at all."""

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

    def as_options(self) -> dict:
        """The block as the flat kwargs dict `LLMConfig.model_options` holds —
        typed fields that were actually set, plus every extra verbatim."""
        return self.model_dump(exclude_none=True)


class ModelDef(BaseModel):
    """One model's settings, mirroring `LLMConfig`'s two dicts exactly: what
    the model is asked (`options`) and what the provider is told
    (`provider_options`). No key is inspected on the way out of either — which
    block a key is written in IS the routing."""

    options: ModelOptionsBlock = Field(default_factory=ModelOptionsBlock)
    provider_options: dict = Field(default_factory=dict)
    model_config = _STRICT


class ProviderDef(BaseModel):
    """A provider entry: where the host is (`base_url` / `transport`), the
    defaults every model on it is invoked with (`options` /
    `provider_options`), and per-model overrides (`models`). Any subset — an
    entry that only sets `options` on `openrouter` is the ordinary case.

    `base_url` and `transport` are passed PER CALL, never registered globally,
    so pointing a built-in provider at a proxy is as ordinary as configuring a
    host luca has never heard of. A name `luca.client` does not know needs
    both: they are what let it build a generic provider for it.

    `transport` is a dotted path to the transport CLASS
    (`"luca.client.transports.OpenAITransport"`), resolved by the runner at
    call time — an import path survives the round trip through the session
    file that a class object could not.

    No api key here. Credentials live in `auth.json`; see `auth.py`."""

    base_url: str | None = None
    transport: str | None = None
    options: ModelOptionsBlock = Field(default_factory=ModelOptionsBlock)
    provider_options: dict = Field(default_factory=dict)
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
    # Extra roots to scan for `<name>.md` slash commands; `~` expanded per entry.
    extra_command_locations: list[str] = Field(default_factory=list)
    # Extra instruction files, on top of the discovered LUCA.md / AGENTS.md /
    # CLAUDE.md; `~` expanded, relative entries resolved against the workspace.
    instructions: list[str] = Field(default_factory=list)
    streaming: bool | None = None
    # Snapshot the workspace before each turn so `/undo` and `/rewind` can put
    # it back (`--checkpoints` / `--no-checkpoints`; default on). Snapshots go
    # to a private git repository beside the session, never into the workspace,
    # and a machine without git simply has the feature switched off.
    checkpoints: bool | None = Field(
        default=None,
        description=(
            "Snapshot the workspace before each turn so /undo and /rewind can restore it. "
            "Needs a git binary. Defaults to true."
        ),
    )
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


def validate_provider(config: LucaConfig, provider: str) -> None:
    """Fail at BOOT if `luca.client` could not build this provider.

    A name the client knows always resolves. Any other name is only reachable
    through the per-call escape hatch, which needs `base_url` AND `transport`
    together — with either missing, `resolve_provider` raises
    `ProviderNotFoundError` at the first LLM call, which reads as a runtime
    failure of the agent rather than as the config typo it is.

    Nothing is registered here and nothing is refused for naming a built-in:
    settings are carried per call now, so pointing `openai` at a proxy is
    ordinary rather than a global mutation of another provider's wire format."""
    if provider in PROVIDERS:
        return
    defn = config.providers.get(provider)
    if defn is not None and defn.base_url and defn.transport:
        return
    known = ", ".join(sorted(PROVIDERS))
    raise LucaConfigError(
        f"provider {provider!r} is unknown to luca.client and cannot be reached: "
        f'add both to luca.json under providers.{provider} — "base_url" and "transport" '
        f'(e.g. "luca.client.transports.OpenAITransport") — or pick one of: {known}',
    )


def check_provider_buildable(llm_config: LLMConfig, *, api_key=None, credentials=None) -> None:
    """Fail at BOOT if the provider cannot be CONSTRUCTED.

    `validate_provider` only asks whether the name is reachable. A missing
    region, a half-written AWS credential or an unresolvable profile all live
    in the provider's constructor, which otherwise does not run until the first
    LLM call and surfaces there as a failed turn.

    `resolve_provider`, never `get_provider`: the latter hands back a CACHED
    instance and closing it would leave the real call holding a dead client.
    The arguments come from `completion_options` so this builds exactly what
    the runner will. No network happens; construction is local."""
    kwargs = completion_options(llm_config, api_key=api_key, credentials=credentials)
    try:
        provider = resolve_provider(
            llm_config.provider,
            api_key=kwargs.get("api_key"),
            credentials=kwargs.get("credentials"),
            base_url=kwargs.get("base_url"),
            transport_class=kwargs.get("transport_class"),
        )
    except ClientError as exc:
        raise LucaConfigError(str(exc)) from exc
    provider.close()


def resolve_model_options(
    config: LucaConfig,
    provider: str,
    model: str,
) -> tuple[dict, dict]:
    """The `(model_options, provider_options)` for one `(provider, model)`
    pair — the two flat dicts `LLMConfig` carries.

    Both are the model's own block deep-merged over the provider-wide one
    (nested objects merge, scalars and lists replace), so a provider-wide
    `provider.order` survives a model that only sets `transforms`. Which of
    the two a key lands in is decided entirely by which block it was written
    in; no key is inspected. `base_url` and `transport` join
    `provider_options` because that is where the runner looks for them.

    Two empty dicts when nothing is configured, which is what lets an
    unconfigured session stay untouched."""
    defn = config.providers.get(provider)
    if defn is None:
        return {}, {}
    model_def = defn.models.get(model)

    model_options = _deep_merge(
        defn.options.as_options(),
        model_def.options.as_options() if model_def is not None else {},
    )
    provider_options = _deep_merge(
        defn.provider_options,
        model_def.provider_options if model_def is not None else {},
    )
    if defn.base_url is not None:
        provider_options["base_url"] = defn.base_url
    if defn.transport is not None:
        provider_options["transport"] = defn.transport
    return model_options, provider_options


def apply_model_options(
    llm_config: LLMConfig,
    config: LucaConfig,
    *,
    cli_reasoning: str | None = None,
) -> LLMConfig:
    """Resolve both option dicts for whichever `(provider, model)` this config
    names.

    Both are always assigned, including to empty — switching to a model with
    no block has to CLEAR the previous model's settings, not inherit them. The
    top-level `model.reasoning` is a default under the blocks; the CLI's
    `--reasoning` wins over everything."""
    model_options, provider_options = resolve_model_options(
        config,
        llm_config.provider,
        llm_config.model,
    )
    if config.model.reasoning is not None:
        model_options.setdefault("reasoning", config.model.reasoning)
    if cli_reasoning is not None:
        model_options["reasoning"] = cli_reasoning
    return llm_config.model_copy(
        update={"model_options": model_options, "provider_options": provider_options},
    )


def resolve_llm_config(base: LLMConfig, config: LucaConfig, cli: dict) -> LLMConfig:
    """`config.model` over `base`, then CLI over both, then the per-provider /
    per-model options for whichever pair that lands on."""
    updates: dict = {}
    for field in ("provider", "model"):
        value = getattr(config.model, field)
        if value is not None:
            updates[field] = value
    updates.update({key: value for key, value in cli.items() if key != "reasoning" and value is not None})
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
    api_key: str | None = None,
    credentials=None,
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
        api_key=api_key,
        credentials=credentials,
    )
