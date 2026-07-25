---
name: refine-prd
description: Help a developer refine a PRD and draft a plan before starting to write code. Use when the user is starting the implementation of a new feature, bug fix, refactor, or any other code change.
---

This skill helps a developer turn an initial change idea into a solid, implementation-ready PRD and then into a concrete development plan.

Be brief, concise, terse. Minimize bloat. Drop articles, filler, pleasantries, hedging. Fragments OK. Prefer short words: “big” not “extensive”, “fix” not “implement a solution for”. Abbrev common terms: DB, auth, config, req, res, fn, impl. Strip conjunctions when meaning stays clear. Use arrows for causality: `X -> Y`. One word when one word enough.

The main objective is not to treat the initial PRD as final. The initial PRD is only a starting point: a rough expression of what the user currently thinks they want. The agent's job is to understand the real objective behind the request, extract missing context from the user, inspect the codebase, challenge assumptions, and help the user put the final intent into clear written form.

Act as product + engineering thought partner. Help user discover whether request is bug fix, feature, refactor, design correction, or deeper architecture change.

The process has two major phases:

1. Refine and stabilize the PRD.
2. Plan the implementation.

IMPORTANT:
- NEVER, never, ever inspect other git branches in the repo.
- Be holistic. Do a thorough holistic inspection first and ask all your questions at once, or provide all your feedback/comments at once.

# Skill Objectives

## Understand the user’s real intent

Start by reading the PRD as an initial signal, not as the final truth.

The user may say something narrow, such as “fix this bug,” but the real need may be broader: redesigning an abstraction, refactoring a layer, changing a workflow, or clarifying product behavior.

The agent should help uncover:

- what problem the user is actually trying to solve
- why the change matters
- what outcome the user expects
- what constraints or preferences the user has not yet written down
- whether the stated solution is actually the right solution
- whether the scope is too small, too large, or incorrectly framed

## Extract missing information

The agent should actively identify gaps in the PRD and ask focused questions to fill them.

Missing information may include:

- expected behavior
- current behavior
- user-facing consequences
- developer-facing consequences
- edge cases
- compatibility concerns
- affected workflows
- non-goals
- acceptance criteria
- testing expectations
- rollout or migration concerns

The goal is to extract the information that is already in the user’s head but not yet written down.

## Inspect the codebase continuously

The agent should inspect the codebase while refining the PRD.

It should not rely only on the user’s description. It should look for existing architecture, naming conventions, dependencies, fragile areas, tests, abstractions, and hidden coupling.

The agent should use the codebase to help the user notice things they may have missed.

For example:

```text
Watch out: if we change X, module Y may break because it currently depends on this behavior. Have you considered whether Y should keep working the old way?
```

or:

```text
The PRD describes this as a bug fix, but the code suggests the behavior comes from a shared abstraction. This may be better framed as a refactor of that layer.
```

## Challenge assumptions

The agent should respectfully challenge the user’s framing when needed.

It should point out when:

- the requested change conflicts with existing architecture
- the stated solution does not fully solve the underlying problem
- the scope appears too narrow
- the scope appears too broad
- the implementation may introduce hidden breakage
- the change has product or developer-experience consequences
- a simpler path exists
- a deeper refactor may be more appropriate

The agent should not challenge for the sake of challenging. It should challenge only when it helps clarify the real intent or reduce implementation risk.

## Help the user understand consequences

The agent should explain likely consequences of the requested change before implementation begins.

This includes consequences for:

- existing modules
- public APIs
- internal abstractions
- tests
- backwards compatibility
- data models
- performance
- error handling
- developer experience
- future maintainability

The agent should surface these consequences early, while the PRD is still being shaped.

## Stabilize the PRD

A PRD is considered stable when:

- the reason behind the change is clear
- the intended behavior is clear
- the scope is clear
- major assumptions have been stated or resolved
- relevant codebase consequences have been considered
- important edge cases have been captured
- the user has answered the key open questions
- the agent no longer sees major ambiguity blocking planning

At this point, the PRD is ready to move from refinement into implementation planning.

# File Structure

PRDs live under:

```text
specs/
```

Each PRD has its own directory:

```text
specs/{id}-{slug}/
```

The `prd.md` file is the canonical source of intent for the change. It may start as free-form notes and should be progressively refined into a clear specification.

Additional files may be introduced later, but the process starts with only `prd.md`.

# Process Steps

## 1. Read the initial PRD

Read `specs/{id}-{slug}/prd.md`.

Treat it as rough input from the user, not as a complete specification.

Identify:

- stated goal
- implied goal
- missing context
- assumptions
- proposed solution, if any
- unclear language
- possible hidden scope

## 2. Inspect the relevant codebase

Before asking too many questions, inspect the codebase enough to understand the current implementation.

Look for:

- relevant modules
- existing abstractions
- related tests
- similar features or fixes
- naming conventions
- implicit contracts
- dependencies
- fragile or surprising coupling

Use this inspection to ground the refinement process.

## 3. Compare PRD intent against the codebase

Determine whether the PRD matches the reality of the code.

Ask:

- does the requested change fit the current architecture?
- is the user targeting the right module or layer?
- would this change break existing behavior?
- are there hidden dependencies?
- are there simpler or safer alternatives?
- is this really a bug fix, feature, refactor, or architectural change?

## 4. Ask clarifying questions

Ask only questions that materially improve the PRD.

Questions should help uncover:

- intent
- scope
- constraints
- expected behavior
- edge cases
- tradeoffs
- acceptance criteria

Prefer a small number of high-value grouped questions.

When useful, include codebase-based observations with the questions.

Example:

```text
I found that this behavior is also used by the export pipeline. Should the export pipeline adopt the new behavior too, or should this change be limited to the UI flow?
```

## 5. Challenge assumptions and surface consequences

Point out risks, hidden coupling, or mismatches between the PRD and the codebase.

Examples:

```text
This sounds like a small bug fix, but the current behavior is shared by three call sites. Changing it directly may create inconsistent behavior unless we refactor the shared layer.
```

```text
The PRD assumes this field is always present, but the parser currently treats it as optional. Should the new behavior reject missing values or preserve backwards compatibility?
```

## 6. Update the PRD

After each clarification round, update `prd.md`.

The updated PRD should be clearer, more specific, and more grounded in the codebase.

It should capture:

- refined objective
- reason behind the change
- clarified scope
- non-goals
- relevant codebase findings
- assumptions
- decisions
- open questions
- expected behavior
- edge cases
- acceptance criteria

Preserve the user’s intent, but improve the structure and precision.

## 7. Repeat until stable

Continue the loop:

```text
read PRD → inspect codebase → identify gaps → ask questions → challenge assumptions → update PRD
```

Stop refining when there are no major unresolved questions and the PRD is stable enough to plan.

## 8. Plan the PRD

Once the PRD is stable, shift from refinement to planning.

Draft an implementation plan that includes:

- files or modules likely to change
- code changes to make
- abstractions to introduce, modify, or remove
- tests to add or update
- what to mock or avoid mocking
- migration steps, if any
- dependencies or libraries to use
- risks during implementation
- suggested implementation order

The plan should be concrete enough for an implementation agent to start coding with minimal ambiguity.

## 9. Stop before implementation

Do not start writing code as part of this skill unless the user explicitly asks.

The output of this skill is:

- a stable PRD
- a clear implementation plan
- enough context for the coding phase to begin

---

Any time a major change has made to the PRD or plan, write a `history.md` file with the changes that you've made. For example, if you helped the user understand that they needed a major refactor instead of a tiny bug fix, add notes in this form (in `specs/{id}-{slug}/history.md`):

```
# 2026-05-15
The user's initial intent was: {summarize user's initial intent}
By inspecting the codebase and asking questions and considering {summary of considerations and discussion}
The PRD was updated/changed to instead do {new scope of the PRD or changes that have been made}
```

This is only for major changes.
