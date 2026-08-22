"""ContextManager — the runner's context strategy: how big an entry is, what a
tool's output looks like once it is stored, what replaces a pruned entry, and
when and how the conversation is compacted.

One runtime collaborator (passed as `context_manager=`, never serialized —
only its durable results persist) owning five policies over the shared
`Entry` abstraction. All five take the live `AgentSession` first, so one
argument order describes the whole contract and every policy can see the same
state — the active model included, which is what makes a real tokenizer
implementable.

TWO of the five also take a `conversation_id`, and the split is a decision
rather than an oversight. `calculate_context` and `prune_entry` do NOT, because
`context_tokens` is intrinsic to the ENTRY and shared by every conversation
that references it. `process_tool_output` does NOT, because the execution it
already receives carries `conversation_id` — a manager that wants to truncate a
subagent's output harder than the main conversation's reads it from there. The
compaction pair DOES, because compaction is a fact about one path:

- `calculate_context(session, entry)` — the entry's intrinsic context-token
  count. Called by the runner on every new entry before `before_entry_written`
  middleware, and again on a `ToolExecution`'s terminal transition (before
  `after_tool_execution`), when its model-facing outcome is finally known.
  Provider usage is never an input — that is accounting, not content size.
  Because it runs on EVERY new entry, scanning `session.entries` here makes a
  turn quadratic; cross-entry work belongs in `prune_entry` /
  `process_tool_output`, which run rarely. On an append it also runs INSIDE
  the ledger's build callback, so the entry already has its `id` but is not
  yet a member of `session.entries` — an implementation that looks itself up
  there raises `KeyError` on every append.
- `prune_entry(session, entry)` — a durable `PrunedEntry` replacing the
  entry's contribution to a conversation path. Returns a TEMPLATE: identity
  fields (`id`, `parent_id`, `created_at`) are placeholders for the persisting
  door to stamp (ids and clocks are the runner/ledger's, never a strategy's).
  It has NO framework call site — an application calls it and hands the
  template to `SessionLedger.prune()`, so the caller already holds the
  session. Its session argument is uniformity, not need.
- `process_tool_output(session, execution, result)` — an optional
  transformation of a tool's returned `ExecutionResult`, applied before the
  terminal `ToolExecution` is constructed and before any middleware runs. May
  truncate or replace model-facing content (preserving the original in
  `metadata` per its own policy); the base behavior is an identity
  pass-through. `execution` is there to be READ FOR IDENTITY — `tool_spec`,
  `raw_tool_call.name`, `raw_tool_call.arguments` — which is what makes
  "truncate `bash` output but never `read`" expressible. It is IN TRANSITION:
  `status` is still RUNNING and `result` is not yet attached, so it must not
  be inspected for outcome.
- `should_compact(session, conversation_id)` — has that conversation crossed
  the point where compaction is worth doing? Consulted at the top of every
  drive (and at `start()` time, which is why this is SYNC). Unlike
  `calculate_context` this one is MEANT to sum the path: it runs once per
  drive, not once per entry.
- `compact(session, conversation_id, nodes, entry)` — the compaction itself: fill in a deep
  copy of `entry` and return a `CompactionPlan` describing the resulting
  conversation. The base raises `NotImplementedError` and the base
  `should_compact` returns False, so the default manager never compacts and
  only an explicit `schedule_compaction()` can reach `compact` at all.

Compaction: who decides what. The two methods above own everything about the
*decision* — when compaction is worth doing, what the summary says, which model
produces it, which nodes survive, and whether to back off after a failure. Core
triggers, stamps and archives — nothing else. It reads exactly five things off a
returned plan (`entry.parts`, `entry.llm_config`, `entry.metadata`, `nodes`,
`usage`) and discards every other field, which is the whole coupling surface
between the two. The runner opens a turn bracket, appends a `CompactionEntry`,
stamps its `started_at`, and hands `compact` the live session, the path it may
rewrite, and a DEEP COPY of that entry; the returned `CompactionPlan` is
validated for STRUCTURE (never for meaning), its usage recorded, and the
transition performed — one atomic swap in which the pre-compaction conversation
is archived intact and a new one becomes active. Nothing is ever deleted. The
plan's value objects and validator live in `compaction.py`.

Stored counts and the active model. `Entry.context_tokens` is stored on the
entry, so an implementation that counts against `session.session_config.
llm_config` leaves every stored count stale the moment that model changes.
`AgentSessionRunner.recalculate_context_tokens()` re-derives them all; nothing
in the framework calls it, and the application that swaps in a real tokenizer
is the one that should.

This is a CONCRETE class with complete, deliberately simple default behavior
(the same pattern as `ConversationProjector`): estimation is one token per
`CHARS_PER_TOKEN` characters of model-facing text plus a flat `IMAGE_TOKENS`
per image, `AUDIO_TOKENS` per recording and `FILE_TOKENS` per document,
pruning supports terminal tool executions (a fixed marker) and assistant
messages (web machinery summarized, final answer verbatim; refused when
client-executed tool calls are aboard), and compaction is absent —
`should_compact` declines and `compact` raises, so the shipped default is a
pure accountant.
`luca.agent.contrib.simple_context_manager` ships one that also compacts.
Instantiate it directly, subclass and override selected methods, or supply
another object with the same behavior. Luca does not prescribe per-entry-type
methods — dispatch by type here is an internal choice, not a runner contract.

The framework never recalculates, validates, or repairs context after
middleware has run: middleware has the final say, and the application owns
the state it returns.
"""

from __future__ import annotations

import json
from typing import ClassVar
from urllib.parse import urlparse

from luca.client import effective_messages
from luca.client.types import (
    PrivateProviderBlock as ClientPrivateProviderBlock,
    TextBlock as ClientTextBlock,
    ThinkingBlock as ClientThinkingBlock,
    ToolCall as ClientToolCall,
)

from .compaction import CompactionPlan
from .exceptions import AgentError
from .models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    AssistantMessage,
    AudioContent,
    ChildConversation,
    CompactionEntry,
    Entry,
    ExecutionResult,
    FileContent,
    ImageContent,
    LLMConfig,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    UserMessage,
    WebFetchContent,
    WebSearchContent,
)
from .projection import ConversationProjector

# The replacement content a pruned tool output projects as. Module-level alias
# of the class default.
PRUNED_TOOL_OUTPUT_MARKER = "[tool output has been pruned to reduce context]"


class ContextManager:
    """The default context policy. Every method is an override point."""

    PRUNED_TOOL_OUTPUT_MARKER: ClassVar[str] = PRUNED_TOOL_OUTPUT_MARKER
    # The assistant-replacement wording — override points like the marker
    # above; the whole derivation is `_pruned_assistant_text`.
    PRUNED_SEARCH_PREFIX: ClassVar[str] = "Searched the web: "
    PRUNED_FETCH_PREFIX: ClassVar[str] = "Fetched: "
    PRUNED_ANSWER_PREFIX: ClassVar[str] = "Answered: "
    PRUNED_HOSTS_SHOWN: ClassVar[int] = 5
    CHARS_PER_TOKEN: ClassVar[int] = 4
    IMAGE_TOKENS: ClassVar[int] = 1_000
    AUDIO_TOKENS: ClassVar[int] = 10_000
    FILE_TOKENS: ClassVar[int] = 5_000
    # The projector the ASSISTANT branch measures through. A class-level
    # default, overridable by subclass — house OOP, no injected collaborator.
    # Known, stated limitation: a custom projector passed to the RUNNER is not
    # seen here unless the application also overrides the manager.
    PROJECTOR: ClassVar[ConversationProjector] = ConversationProjector()

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        """Estimate the context tokens of `entry`'s model-facing content.

        Ownership per entry type: a user message owns its content; an
        assistant message owns its text, thinking, and tool-call REQUESTS
        (name + arguments — counted here, never again on the execution); a
        tool execution owns only its model-facing outcome (result content,
        else its structured error message), and is 0 while nonterminal; a
        compaction owns its summary `parts` (0 until they land, which is why
        the runner recalculates when they do); a child conversation owns its
        RESULT and nothing of the child's own path (0 until it resolves, and
        recalculated then); a pruned entry owns its replacement content.
        Markers own nothing.

        ASSISTANT entries are measured through the EFFECTIVE wire view —
        project the entry, ask the client what actually serializes for the
        ACTIVE target (`session.llm_config`), and estimate that. This is what
        removes the categorical error hosted web operations introduced: an
        encrypted Anthropic payload counts on Anthropic and stops counting
        after a switch away, without the agent ever learning wire rules. An
        unresolvable target (an unregistered provider — the faux in tests, a
        custom host) falls back to the naive estimate: an estimate is allowed
        to stay an estimate. Every OTHER entry type keeps the naive path ON
        PURPOSE — a projection-based measure of executions would zero private
        ones and count placeholder text.

        Non-text content is counted separately by `_media_tokens`, so
        `_estimate_tokens` and the two text derivations stay text-shaped and
        independently overridable."""
        if isinstance(entry, AssistantMessage):
            text = self._effective_wire_text(session, entry)
        else:
            text = self._model_facing_text(entry)
        return self._estimate_tokens(text) + self._media_tokens(entry)

    def prune_entry(self, session: AgentSession, entry: Entry) -> PrunedEntry:
        """Build the `PrunedEntry` template replacing `entry` in a path.

        Terminal `ToolExecution`s and `AssistantMessage`s are prunable by
        this default; anything else fails loudly. An assistant replacement
        summarizes the web machinery (queries + result hosts — where the
        encrypted bytes live) and keeps the final answer text VERBATIM; a
        message carrying client-executed `ToolCall` parts is REFUSED, because
        a replayed `ToolMessage` whose call vanished from the wire is a 400
        on every provider. The returned template carries no identity —
        `id`/`created_at` stay `None` — the persisting door stamps the real
        `id`/`parent_id`/`created_at` and the runner-side ordering calculates
        `context_tokens` and runs entry middleware, exactly as for any other
        new entry. (Cross-entry reads are fine here — unlike
        `calculate_context`, pruning runs rarely and by explicit
        application choice.)"""
        if isinstance(entry, AssistantMessage):
            return self._prune_assistant_entry(session, entry)
        if not isinstance(entry, ToolExecution):
            raise AgentError(
                f"Cannot prune entry of type {entry.type!r}: only tool executions and assistant messages are prunable."
            )
        if entry.status in NONTERMINAL_STATUSES:
            raise AgentError(
                f"Cannot prune ToolExecution {entry.id!r}: a nonterminal "
                f"({entry.status.value}) execution is not prunable."
            )
        return PrunedEntry(
            pruned_entry_type=entry.type,
            pruned_entry_id=entry.id,
            content=[TextContent(text=self.PRUNED_TOOL_OUTPUT_MARKER)],
        )

    def _prune_assistant_entry(self, session: AgentSession, entry: AssistantMessage) -> PrunedEntry:
        """The assistant replacement template: web machinery condensed, final
        answer verbatim. Annotations drop with the original (their character
        ranges belong to the split privates being shed); thinking is not
        answer text and drops too."""
        if any(isinstance(part, ToolCall) for part in entry.parts):
            raise AgentError(
                f"Cannot prune AssistantMessage {entry.id!r}: it carries "
                "client-executed tool calls, and a replayed ToolMessage whose "
                "call has vanished from the wire is rejected by every provider."
            )
        return PrunedEntry(
            pruned_entry_type=entry.type,  # "assistant" — the discriminator value
            pruned_entry_id=entry.id,
            content=[TextContent(text=self._pruned_assistant_text(session, entry))],
        )

    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Transform a tool's returned result before it becomes durable.
        Identity by default; override to truncate or replace model-facing
        content (stash the original in `metadata` if your policy keeps it).

        Select on `execution.tool_spec` / `execution.raw_tool_call` to vary the
        policy by tool. `execution` is mid-transition — still RUNNING, no
        result attached — so read it for identity, never for outcome.
        `tool_spec` is `ToolSpec | None`: a registry may dispatch a call it
        never snapshotted, so branch on `raw_tool_call.name` or guard the
        `None` rather than chaining through it."""
        return result

    # ── compaction ───────────────────────────────────────────────────────────

    def should_compact(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> bool:
        """Has `conversation_id` crossed the point where compaction is worth
        doing? Consulted at the top of every drive (and at `start()` time,
        which is why this is SYNC).

        The threshold, the context sum and the window size are all yours —
        core has no context-total API, and `luca.client.catalog` carries
        `context_window` per model. Core remembers nothing about previous
        failures either: a manager that should stop trying returns False."""
        return False

    async def compact(
        self,
        session: AgentSession,
        conversation_id: str,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        """Fill in a DEEP COPY of `entry` and describe the resulting
        conversation.

        `nodes` is THE PATH YOU MAY REWRITE: `conversation_id`'s path with this
        compaction's own `TurnStart` removed, ending with `entry`. It barely
        needs the id — `nodes` already carries the work — but it takes it
        anyway: the signature IS the contract, and adding a parameter later
        breaks every implementor.
        `plan.nodes` may carry any of these ids, in any order, with new
        entries interleaved — and nothing else; an id outside this tuple is a
        plan rejection, so you never have to recognize or route around
        framework markers. `plan.nodes = list(nodes)` is always a legal full
        carry.

        Return `None` for "nothing to do": the entry keeps `parts is None` and
        the bracket closes COMPLETED.

        You own the LLM call end to end — its prompt, its model (record it in
        `entry.llm_config`), its messages. The turn middleware hooks
        deliberately do NOT fire for it. Never mutate `session`: the runner
        refuses to commit if the path moved under you, and the entry you are
        handed is a copy precisely so a failed attempt cannot leave a summary
        projecting onto an unchanged path."""
        raise NotImplementedError()

    # ── derivation helpers ───────────────────────────────────────────────────

    def _pruned_assistant_text(self, session: AgentSession, entry: AssistantMessage) -> str:
        """The replacement's one deterministic string: the searches (queries +
        result hosts, where the shed bytes lived), the fetches, then the
        final answer text VERBATIM — TextContent parts only."""
        searches = [self._pruned_search_segment(part) for part in entry.parts if isinstance(part, WebSearchContent)]
        fetches = [self._pruned_fetch_segment(part) for part in entry.parts if isinstance(part, WebFetchContent)]
        answer = "".join(part.text for part in entry.parts if isinstance(part, TextContent))
        pieces: list[str] = []
        if searches:
            pieces.append(self.PRUNED_SEARCH_PREFIX + ", ".join(searches) + ".")
        if fetches:
            pieces.append(self.PRUNED_FETCH_PREFIX + ", ".join(fetches) + ".")
        pieces.append(self.PRUNED_ANSWER_PREFIX + answer)
        return " ".join(pieces)

    def _pruned_search_segment(self, part: WebSearchContent) -> str:
        quoted = ", ".join(f'"{query}"' for query in part.queries)
        error = part.extras.get("error")
        if error is not None:
            code = error.get("error_code") or error.get("status") or "failed"
            return f"{quoted} (failed: {code})"
        if part.results is None:
            return f"{quoted} (result metadata not returned)"
        if not part.results:
            return f"{quoted} (no results)"
        hosts = [_result_host(result.url) for result in part.results[: self.PRUNED_HOSTS_SHOWN]]
        more = len(part.results) - self.PRUNED_HOSTS_SHOWN
        listed = ", ".join(hosts) + (f", +{more} more" if more > 0 else "")
        count = len(part.results)
        return f"{quoted} ({count} result{'s' if count != 1 else ''}: {listed})"

    def _pruned_fetch_segment(self, part: WebFetchContent) -> str:
        error = part.extras.get("error")
        if error is not None:
            code = error.get("error_code") or error.get("status") or "failed"
            return f"{part.web_page.url} (failed: {code})"
        if part.web_page.title is not None:
            return f'{part.web_page.url} ("{part.web_page.title}")'
        return part.web_page.url

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def _effective_wire_text(self, session: AgentSession, entry: AssistantMessage) -> str:
        """The text an ASSISTANT entry actually puts on the wire for the
        ACTIVE target: project the entry (agent policy), run the projection
        through the client's `effective_messages` (wire truth), and textify
        what survives. Measuring the composition of the two is what keeps the
        count correct under any future agent-side projection policy.

        `project_assistant_message` never reads the entries mapping, so the
        inside-the-build-callback call site (the entry has its id but is not
        yet a member of `session.entries`) is safe, and nothing scans the
        store. The target is the ACTIVE pair — `session.llm_config` — with
        `transport` lifted from `provider_options` exactly as
        `completion_options()` lifts it, so a custom-transport session
        measures its real wire. Resolution failure (an unregistered provider,
        an unimportable transport path) falls back to the naive estimate."""
        projected = self.PROJECTOR.project_assistant_message(entry, session.entries)
        cfg = session.llm_config
        try:
            wire_view = effective_messages(
                cfg.model,
                [projected],
                provider=cfg.provider,
                transport_class=self._transport_class(cfg),
            )
        except Exception:
            return self._model_facing_text(entry)
        return self._client_message_text(wire_view)

    def _transport_class(self, llm_config: LLMConfig) -> type | None:
        """The transport class a custom-transport config names, or None.
        Delegates to the runner's public `completion_options` — the single
        `LLMConfig` → client-kwargs translation — so the measured wire can
        never drift from the called one. Imported at call time: the runner
        module imports this one."""
        from .runner import completion_options

        return completion_options(llm_config).get("transport_class")

    def _client_message_text(self, messages: list) -> str:
        """The client-block textifier: what the effective view's blocks
        contribute to the estimate, mirroring `_model_facing_text`'s rules on
        the other side of the wire so a plain assistant entry measures
        identically on both paths — text and thinking contribute their text,
        a tool call its name + JSON arguments (counted once, on the
        assistant), and a surviving `PrivateProviderBlock` its JSON payload
        (it replays verbatim, so its bytes are real context). The portable
        web blocks never reach here on a real transport — no transport sends
        them — and contribute nothing if they ever do."""
        chunks: list[str] = []
        for message in messages:
            for block in message.content:
                if isinstance(block, ClientToolCall):
                    chunks.append(block.name)
                    chunks.append(json.dumps(block.arguments))
                elif isinstance(block, (ClientTextBlock, ClientThinkingBlock)):
                    chunks.append(block.text)
                elif isinstance(block, ClientPrivateProviderBlock):
                    chunks.append(json.dumps(block.data))
        return "".join(chunks)

    def _model_facing_text(self, entry: Entry) -> str:
        """Concatenate the model-facing text the entry OWNS (see
        `calculate_context`). Unknown entry types contribute nothing rather
        than failing: calculation is an estimate, not a projection."""
        if isinstance(entry, UserMessage):
            return _text_of(entry.parts)
        if isinstance(entry, AssistantMessage):
            chunks: list[str] = []
            for part in entry.parts:
                if isinstance(part, TextContent):
                    chunks.append(part.text)
                elif isinstance(part, ThinkingContent):
                    chunks.append(part.thinking)
                elif isinstance(part, ToolCall):
                    chunks.append(part.name)
                    chunks.append(json.dumps(part.arguments))
            return "".join(chunks)
        if isinstance(entry, ToolExecution):
            if entry.result is not None:
                return _text_of(entry.result.content)
            if entry.error is not None:
                return entry.error.error_message
            return ""
        if isinstance(entry, CompactionEntry):
            return _text_of(entry.parts or [])
        if isinstance(entry, ChildConversation):
            # What the child contributes to its PARENT is its result, and only
            # that: the child's own conversation has its own window. The link
            # counts it only when the link itself RENDERS it (a resolution
            # written without a result execution — a cancel wind-down or a
            # hard-limit settle); otherwise the result execution's own count
            # covers the content, and counting here too would double it.
            if entry.execution_result is not None and entry.result_execution_id is None:
                return _text_of(entry.execution_result.content)
            return ""
        if isinstance(entry, PrunedEntry):
            return _text_of(entry.content)
        return ""

    def _media_tokens(self, entry: Entry) -> int:
        """The entry's non-text context contribution: a flat constant per
        image and per file, deliberately dimension-blind. A URL source has no
        local bytes to measure, reading real dimensions would need an image
        decoder (a new dependency), and the provider formulas disagree by an
        order of magnitude. Override with a per-provider formula if it matters.

        A document is the weaker estimate of the two, and knowingly so: cost
        scales with PAGE COUNT, which cannot be read without a PDF parser this
        library will not take a dependency on. `FILE_TOKENS` stands in for a
        short document; a session of long PDFs will under-count until an
        application overrides this.

        Audio has the same weakness against DURATION: Gemini bills 32 tokens
        per second, so `AUDIO_TOKENS` is roughly a five-minute clip."""
        parts = self._media_parts(entry)
        return (
            self.IMAGE_TOKENS * _image_count(parts)
            + self.AUDIO_TOKENS * _audio_count(parts)
            + self.FILE_TOKENS * _file_count(parts)
        )

    def _media_parts(self, entry: Entry) -> list:
        """The parts an entry owns that may carry non-text content — the same
        ownership `_model_facing_text` applies, from the other side."""
        if isinstance(entry, UserMessage):
            return entry.parts
        if isinstance(entry, ToolExecution):
            return entry.result.content if entry.result is not None else []
        if isinstance(entry, CompactionEntry):
            return entry.parts or []
        if isinstance(entry, ChildConversation):
            # Same ownership rule as `_model_facing_text`: the link carries the
            # content only when it is the one rendering it.
            if entry.execution_result is not None and entry.result_execution_id is None:
                return entry.execution_result.content
            return []
        if isinstance(entry, PrunedEntry):
            return entry.content
        return []


def _result_host(url: str) -> str:
    """A pruned summary's per-result token: the hostname, `www.` dropped,
    falling back to the raw string for anything unparseable."""
    host = urlparse(url).netloc or url
    return host.removeprefix("www.")


def _text_of(parts) -> str:
    """The text a content-part list contributes. Non-text parts contribute no
    characters — they are counted by `_media_tokens` instead."""
    return "".join(part.text for part in parts if isinstance(part, TextContent))


def _image_count(parts) -> int:
    return sum(isinstance(part, ImageContent) for part in parts)


def _audio_count(parts) -> int:
    return sum(isinstance(part, AudioContent) for part in parts)


def _file_count(parts) -> int:
    return sum(isinstance(part, FileContent) for part in parts)
