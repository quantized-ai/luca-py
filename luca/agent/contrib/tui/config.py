"""`luca.json` — layered project + user configuration for the TUI.

Two files are read and field-level deep-merged, project over home:
`~/.config/luca/luca.json` (personal defaults) then `./luca.json` (repo policy).
Everything is optional, validated with `extra="forbid"`, and applied with the
precedence **CLI flag > luca.json > persisted session > built-in default** — so
the file behaves like sticky CLI flags. It is pure data (no shell execution);
a malformed file raises `LucaConfigError` with a readable message.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from luca.agent.contrib.compaction import (
    CompactionStrategy,
    Compactor,
    RecentTurnsStrategy,
)
from luca.agent.contrib.compaction.compactor import DEFAULT_THRESHOLD, DEFAULT_WINDOW
from luca.agent.contrib.resource_permissions import (
    PermissionMatchMode,
    PermissionMode,
    ResourcePermission,
    ToolKindRule,
    ToolRule,
)
from luca.agent.core.models import ApprovalOption, LLMConfig, RuntimeConfig, ToolKind
from luca.client.providers import register_provider
from luca.client.transports import TRANSPORTS
from luca.client.types import Reasoning

_STRICT = ConfigDict(extra="forbid")


class LucaConfigError(Exception):
    """A luca.json that is missing-file-aside unreadable, non-JSON, or invalid."""


class ModelConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    reasoning: Reasoning | None = None
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


class ProviderDef(BaseModel):
    base_url: str
    api_key_env: str | None = None
    transport: str = "openai"
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
    def _needs_a_target(self) -> "PermissionRule":
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
                permission=self.permission, resource=self.resource,
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
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    providers: dict[str, ProviderDef] = Field(default_factory=dict)
    models: dict[str, list[str]] = Field(default_factory=dict)
    workspace: str | None = None
    additional_directories: list[str] = Field(default_factory=list)
    streaming: bool | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ── loading ──────────────────────────────────────────────────────────────────


def config_home() -> Path:
    """`$XDG_CONFIG_HOME/luca` or `~/.config/luca`."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LucaConfigError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise LucaConfigError(f"{path}: the top level must be a JSON object")
    return data


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` wins per key; nested objects merge, scalars and lists replace."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_luca_config(*, cwd: Path | None = None, home: Path | None = None) -> LucaConfig:
    """Read the home then project `luca.json`, deep-merge, validate."""
    cwd = cwd or Path.cwd()
    home = home if home is not None else config_home()
    merged = _deep_merge(
        _read_json_object(home / "luca.json"),
        _read_json_object(cwd / "luca.json"),
    )
    try:
        return LucaConfig.model_validate(merged)
    except ValidationError as exc:
        raise LucaConfigError(f"luca.json is invalid:\n{exc}") from exc


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
    """Register every custom provider so a call can route to it."""
    for name, defn in config.providers.items():
        if name in _FIRST_CLASS_PROVIDERS:
            raise LucaConfigError(
                f"provider {name!r} is built in; give a custom host a distinct "
                "name and point model.provider at it",
            )
        transport = TRANSPORTS.get(defn.transport)
        if transport is None:
            raise LucaConfigError(
                f"provider {name!r}: unknown transport {defn.transport!r} "
                f"(one of {', '.join(sorted(TRANSPORTS))})",
            )
        register_provider(name, {
            "default_base_url": defn.base_url,
            "default_api_key_env_var": defn.api_key_env,
            "default_transport_class": transport,
        })


def resolve_llm_config(base: LLMConfig, config: LucaConfig, cli: dict) -> LLMConfig:
    """`config.model` over `base`, then CLI over both."""
    updates: dict = {}
    for field in ("provider", "model", "reasoning"):
        value = getattr(config.model, field)
        if value is not None:
            updates[field] = value
    updates.update({key: value for key, value in cli.items() if value is not None})
    return base.model_copy(update=updates) if updates else base


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


def build_permission_rules(config: LucaConfig) -> list[ToolKindRule | ToolRule]:
    return [rule.to_rule() for rule in config.permissions.rules]


def build_compactor(
    config: LucaConfig,
    *,
    enabled: bool | None,
    threshold: float | None,
    keep_turns: int | None,
) -> Compactor:
    enabled = pick(enabled, config.compaction.enabled, True)
    threshold = pick(threshold, config.compaction.threshold, DEFAULT_THRESHOLD)
    keep_turns = pick(keep_turns, config.compaction.keep_turns, 0)
    default_window = pick(None, config.compaction.default_window, DEFAULT_WINDOW)
    strategy = RecentTurnsStrategy(keep_turns=keep_turns) if keep_turns > 0 else CompactionStrategy()
    return Compactor(
        strategy, threshold=threshold, default_window=default_window, enabled=enabled,
    )
