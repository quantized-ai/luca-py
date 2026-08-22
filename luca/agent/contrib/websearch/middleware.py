"""`WebSearchMiddleware` — the websearch plugin's entire provider knowledge.

Two hooks, two halves of the D-frame:

    adapt_tool_declarations  APPEND the provider's hosted declarations when
                             config ∩ per-model support allows (a hosted tool
                             has no ToolSpec, so the client slot is the only
                             door — the hook's doctrine explicitly permits
                             appending provider-hosted items)
    synthesize_executions    one durable, private, terminal ToolExecution per
                             recorded web operation (the D9 synthetics)

The gate reads the ACTIVE `session.llm_config` — never
`session.use_native_tools`, which governs the native shell/editor swap and
has nothing to do with hosted web tools. Appending the wrong provider's
declaration item is a hard client `BadRequestError` before any HTTP, so the
gate must be exact: provider block enabled AND the support table names the
active model for that tool.
"""

from __future__ import annotations

from urllib.parse import urlparse

from luca.agent.core import (
    AgentSession,
    AssistantMessage,
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    WebFetchContent,
    WebSearchContent,
)

from .config import WebSearchConfig
from .specs import WEB_FETCH_SPEC, WEB_SEARCH_SPEC
from .support import supported_web_tools

# The summary's host list caps at five, with the remainder counted.
_MAX_HOSTS = 5


class WebSearchMiddleware:
    def __init__(self, session: AgentSession, config: WebSearchConfig) -> None:
        self.session = session
        self.config = config

    # ── OUT: the hosted declarations ────────────────────────────────────────

    def adapt_tool_declarations(
        self,
        session: AgentSession,
        conversation_id: str,
        tools: list,
    ) -> list:
        """Append the ACTIVE (provider, model) pair's hosted web declarations.
        The declaration IS the config (D7): the per-provider options validate
        into the client's own tool classes and are handed over as-is."""
        return tools + self._declarations()

    def _declarations(self) -> list:
        if not self.config.enabled:
            return []
        cfg = self.session.llm_config
        supported = supported_web_tools(cfg)
        if cfg.provider == "openai" and self.config.openai.enabled and "web_search" in supported:
            return [self.config.openai.options]
        if cfg.provider == "anthropic" and self.config.anthropic.enabled:
            declarations: list = []
            if "web_search" in supported:
                declarations.append(self.config.anthropic.search)
            if self.config.anthropic.fetch is not None and "web_fetch" in supported:
                declarations.append(self.config.anthropic.fetch)
            return declarations
        return []

    # ── IN: the D9 synthetics ───────────────────────────────────────────────

    def synthesize_executions(
        self,
        session: AgentSession,
        conversation_id: str,
        entry: AssistantMessage,
    ) -> list[ToolExecution]:
        """One template per recorded web operation, in part order. The op id
        is the client-stamped `extras["id"]` — left empty when the provider
        reported none, and the RUNNER backfills it via `generate_id()`
        (middleware does not hold the determinism hooks). COMPLETED unless
        the part carries the client-stamped provider error."""
        templates: list[ToolExecution] = []
        for part in entry.parts:
            if isinstance(part, WebSearchContent):
                templates.append(self._search_template(entry, part))
            elif isinstance(part, WebFetchContent):
                templates.append(self._fetch_template(entry, part))
        return templates

    def _search_template(self, entry: AssistantMessage, part: WebSearchContent) -> ToolExecution:
        op_id = part.extras.get("id") or ""
        error = part.extras.get("error")
        template = ToolExecution(
            tool_call_id=op_id,
            raw_tool_call=ToolCall(id=op_id, name="web_search", arguments={"queries": list(part.queries)}),
            tool_spec=WEB_SEARCH_SPEC,
            status=ExecutionStatus.FAILED if error is not None else ExecutionStatus.COMPLETED,
            extras={"websearch": {"message_entry_id": entry.id}},
        )
        if error is not None:
            template.error = self._provider_error(error)
        else:
            template.result = ExecutionResult(
                content=[TextContent(text=self._summary(part))],
                structured_content=part.model_dump(),
            )
        return template

    def _fetch_template(self, entry: AssistantMessage, part: WebFetchContent) -> ToolExecution:
        op_id = part.extras.get("id") or ""
        error = part.extras.get("error")
        template = ToolExecution(
            tool_call_id=op_id,
            raw_tool_call=ToolCall(id=op_id, name="web_fetch", arguments={"url": part.web_page.url}),
            tool_spec=WEB_FETCH_SPEC,
            status=ExecutionStatus.FAILED if error is not None else ExecutionStatus.COMPLETED,
            extras={"websearch": {"message_entry_id": entry.id}},
        )
        if error is not None:
            template.error = self._provider_error(error)
        else:
            template.result = ExecutionResult(
                content=[TextContent(text=self._summary(part))],
                structured_content=part.model_dump(),
            )
        return template

    @staticmethod
    def _provider_error(error: dict) -> ToolExecutionError:
        """The provider's own report, structured. Anthropic stamps a wire
        error dict with an `error_code`; OpenAI stamps only
        `{"status": "failed"}` — the wording tolerates an error without a
        code."""
        code = error.get("error_code") or error.get("status") or "failed"
        return ToolExecutionError(
            error_type="web_operation_error",
            error_message=code,
            details={"provider_error": error},
        )

    # ── the deterministic one-line summaries (D9 wording table) ─────────────

    def _summary(self, part: WebSearchContent | WebFetchContent) -> str:
        """Overridable, deterministic, one line — the posterity view stored
        as the synthetic's `result.content`."""
        if isinstance(part, WebSearchContent):
            return self._search_summary(part)
        return self._fetch_summary(part)

    def _search_summary(self, part: WebSearchContent) -> str:
        quoted = ", ".join(f'"{query}"' for query in part.queries)
        error = part.extras.get("error")
        if error is not None:
            code = error.get("error_code") or error.get("status") or "failed"
            return f"Web search: {quoted} — failed: {code}"
        if part.results is None:
            return f"Web search: {quoted} (result metadata not returned)"
        if not part.results:
            return f"Web search: {quoted} — no results"
        hosts = [_host(result.url) for result in part.results[:_MAX_HOSTS]]
        more = len(part.results) - _MAX_HOSTS
        listed = ", ".join(hosts) + (f", +{more} more" if more > 0 else "")
        count = len(part.results)
        return f"Web search: {quoted} — {count} result{'s' if count != 1 else ''} ({listed})"

    def _fetch_summary(self, part: WebFetchContent) -> str:
        page = part.web_page
        error = part.extras.get("error")
        if error is not None:
            code = error.get("error_code") or error.get("status") or "failed"
            return f"Web fetch: {page.url} — failed: {code}"
        facts: list[str] = []
        if page.title is not None:
            facts.append(f'"{page.title}"')
        if page.content is not None:
            facts.append(f"{len(page.content):,} chars")
        if facts:
            return f"Web fetch: {page.url} ({', '.join(facts)})"
        return f"Web fetch: {page.url}"


def _host(url: str) -> str:
    """The summary's per-result token: the hostname, `www.` dropped, falling
    back to the raw string for anything unparseable."""
    host = urlparse(url).netloc or url
    return host.removeprefix("www.")
