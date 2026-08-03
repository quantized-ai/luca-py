This is a WIP. The project is heavily layered so you have to first understand what layer/component your contribution is hitting. The two higher level components are:

* LLM Client: a simple LiteLLM-like framework to unify API Calls to different LLMs. Lives in `luca/client/`.
* LLM Agent: the main purpose of this repo, lives in `luca/agent/`.

For the agent itself, the project is structured in this way:

* core/: the main semantics of the project
* contrib/: all the external packages that extend Luca's core behavior
  * memory/: The scratchpad and todo list plugin.
  * plugins/: The plugin composition layer.
  * resource_permissions/: The rule-based tool permission system.
  * shell/: The filesystem and shell tools.
  * simple_context_manager/: The context manager that handles compaction.
  * simple_tool_registry/: The default tool registry and permission policy.
  * skills/: The system for discovering and loading skills.
  * subagents/: The tools and plugin for running parallel subagents.
  * tui/: The TUI client. See `luca/agent/contrib/tui/CONTRIBUTING.md`
