# 2026-08-18

The user's initial intent was: rewrite `luca.client` streaming as the polymorphic
streamer hierarchy specified in prd.md (PoC-normative), with three deliberate API
changes, keeping everything else black-box compatible.

By inspecting the codebase (current stream classes, transports, the agent runner,
the test suite, the docs) and one round of questions, four under-specified areas
were settled and folded into the PRD:

1. **`timeout=` unification now extends beyond streaming.** Total wall-clock on
   all four helpers, plumbed per call (helper → provider → transport → streamer),
   removed from provider construction and the provider cache key. `total_timeout=`
   dies everywhere, including non-streaming `acompletion`. Agent-side edits (two
   runner call sites, collapsing the two RuntimeConfig timeout fields, compaction
   error string, agent doc) are in scope. Sync enforcement is best-effort and
   documented as such. The §3.5 factory sketch was fixed — it forwarded the
   transport's httpx timeout as the stream deadline.
2. **Wire knowledge shared with the non-streaming path** (payload projection,
   error mapping, finish classification) stays in the transport packages and is
   reused by the streamers via inheritance/imports — not moved, not duplicated.
3. **httpx timeouts on stream requests**: per-request override — connect kept,
   read/write/pool disabled — for shared and streamer-owned clients alike;
   caller-injected clients keep being honored.
4. **Test strategy clarified**: internal tests are deleted and are NOT a checklist;
   new tests are designed from the new implementation's layer contracts
   (per-handler behavioral tests, parse-per-framing, mixin contracts), not ported
   from the old suite. §8 gained an explicit "write the new tests" step.

§6 also gained the previously implicit contract edges: async `aclose()` +
external-cancellation survival, never-entered/consumed-once `StreamError`,
`ErrorEvent.usage`, SDK `TimeoutError` staying a non-builtin subclass,
`timeout=None` accepted, OpenRouter quirks staying in the base CC streamer.

Final review round, same day: the user approved the `handle_wire_end()` loop
shape (snippet now recorded in plan.md §3.2 as normative), the Anthropic
unknown-block-ignored behavior (with the intent that server_tool_use/web_search
etc. land later as handler branches — plan.md §3.3), the Bedrock streamer shape
(parse stamps `:event-type` as `"type"` so base dispatch is unchanged), and the
remaining PoC deviations. The PRD's §3.2 sketch was corrected to match
(list-returning parse, handle_wire_end) and §2 now points at plan.md §2 as the
authority where the PoC and plan disagree. Plan is ready for implementation,
starting at step 1 (wire-mixin extraction).

Earlier, after the audit: `test_audit.md` written (frozen/deleted lists, four
forced respells), six codebase fact-checks run, and `plan.md` drafted. The
fact-checks fixed four PoC deviations the plan calls out explicitly: `parse()`
returns a list (Bedrock frames), `handle_wire_end()` emits the terminal at wire
end (CC and Bedrock deliver usage AFTER the finish marker; the PoC's
emit-in-handler approach cannot survive those wires), an `open_wire()` seam so
faux reuses the mixins with empty combo classes, and a `provider=` constructor
datum (OpenAITransport serves DeepSeek/generic hosts, so provenance is instance
data, not class identity).
