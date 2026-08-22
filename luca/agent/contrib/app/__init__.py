"""luca.agent.contrib.app — the headless application layer.

Everything a front end needs to put a configured agent on the road, with no
front end in it: the `luca.json` loader, the credential file, user-defined
slash commands, `@`-mention handling, the plugin composition (`build_runner`),
the session store, the approval-prompt policy, token and cost arithmetic, the
boot sequence, and `AgentApplication`, which ties them together.

NOTHING IN THIS PACKAGE MAY IMPORT TEXTUAL, directly or transitively. The TUI
(`luca.agent.contrib.tui`) is one consumer and the ACP server
(`luca.agent.contrib.acp`) is another; a terminal library has no business in
the import graph of a stdio server. `tests/agent/contrib/app/test_no_textual.py`
enforces it.
"""

from .application import AgentApplication
from .approvals import (
    CANCEL_LABEL,
    DENY_LABEL,
    ApprovalPromptModel,
    PromptOption,
    build_approval_prompts,
)
from .auth import (
    ENV_AUTH_PATH,
    AuthEntry,
    api_key_for,
    auth_home,
    load_auth,
    resolve_auth_path,
)
from .boot import (
    DEFAULT_LOG_LEVEL,
    ENV_LOG_LEVEL,
    BootResult,
    boot,
    build_session,
    credentials,
    log_path,
    remove_log_handlers,
    setup_logging,
)
from .config import (
    ENV_CONFIG_PATH,
    LucaConfig,
    LucaConfigError,
    apply_model_options,
    build_context_manager,
    build_permission_rules,
    load_luca_config,
    pick,
    picker_models,
    resolve_config_path,
    resolve_llm_config,
    resolve_model_options,
    resolve_read_limits,
    resolve_runtime_config,
    validate_provider,
)
from .custom_commands import (
    CustomCommand,
    default_locations,
    discover_commands,
    expand,
    load_command,
    resolve_locations,
)
from .prompt_files import ReadLimits
from .sessions import (
    DEFAULT_STORE,
    QUESTIONS_STORE_KEY,
    SIDECAR_SUFFIX,
    app_state_path,
    delete_session,
    encode_project_path,
    fork_session,
    load_app_state,
    load_session,
    resolve_session_directory,
    save_app_state,
    save_session,
    session_path,
)
from .usage import UsageTotals, cost_breakdown, estimated_cost, usage_totals
from .wiring import (
    SCRATCHPAD_STORE_KEY,
    TODO_STORE_KEY,
    build_faux_provider,
    build_runner,
    default_model,
    faux_model,
)

__all__ = [
    "CANCEL_LABEL",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_STORE",
    "DENY_LABEL",
    "ENV_AUTH_PATH",
    "ENV_CONFIG_PATH",
    "ENV_LOG_LEVEL",
    "QUESTIONS_STORE_KEY",
    "SCRATCHPAD_STORE_KEY",
    "SIDECAR_SUFFIX",
    "TODO_STORE_KEY",
    "AgentApplication",
    "ApprovalPromptModel",
    "AuthEntry",
    "BootResult",
    "CustomCommand",
    "LucaConfig",
    "LucaConfigError",
    "PromptOption",
    "ReadLimits",
    "UsageTotals",
    "api_key_for",
    "app_state_path",
    "apply_model_options",
    "auth_home",
    "boot",
    "build_approval_prompts",
    "build_context_manager",
    "build_faux_provider",
    "build_permission_rules",
    "build_runner",
    "build_session",
    "cost_breakdown",
    "credentials",
    "default_locations",
    "default_model",
    "delete_session",
    "discover_commands",
    "encode_project_path",
    "estimated_cost",
    "expand",
    "faux_model",
    "fork_session",
    "load_app_state",
    "load_auth",
    "load_command",
    "load_luca_config",
    "load_session",
    "log_path",
    "pick",
    "picker_models",
    "remove_log_handlers",
    "resolve_auth_path",
    "resolve_config_path",
    "resolve_llm_config",
    "resolve_locations",
    "resolve_model_options",
    "resolve_read_limits",
    "resolve_runtime_config",
    "resolve_session_directory",
    "save_app_state",
    "save_session",
    "session_path",
    "setup_logging",
    "usage_totals",
    "validate_provider",
]
