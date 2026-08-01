# 2026-08-01

The user's initial intent was: implement the mid-turn user messages PRD as
written, plus one requirement missing from it — posting must be impossible
while subagents are active in the main conversation, raising like the
cancellation case does.

By inspecting the codebase (runner drive loop, status derivation, path
derivations, subagents machinery, TUI input path) and asking three questions,
the following was settled and folded into the PRD:

- **D13 added**: a post targeting a conversation whose open turn has
  unresolved `ChildConversation` links is rejected with a new dedicated
  `SubagentsActiveError`. Generalized to ANY conversation (not main-only) —
  a subagent with its own active children rejects too. CANCELLING outranks
  it. The boundary is the durable link, not the spawn call, so the small
  window between a spawn execution completing and its link appearing accepts
  (documented, not fought).
- **D2 matrix hole fixed**: an archived conversation can derive BUSY (a
  queued message compacted behind leaves `[…, u, ts_c, cmp, tf_c]`), so the
  "non-main IDLE → reject" row would have *accepted* a post into an archived
  conversation nothing ever drives. New matrix row: any conversation that is
  someone's `previous_conversation_id` rejects, whatever its derived status.
- **D3 exception named**: `ConversationCancellingError`, flat next to
  `SubagentsActiveError`; no shared intermediate base (the shipped TUI treats
  every post rejection identically).
- **D12 extended**: the TUI communicates the D13 rejection reactively
  (notice + preserved draft, uniform with CANCELLING) rather than proactively
  disabling the prompt; slash commands stay idle-only.

The implementation plan lives in `implementation-plan.md`.
