## Output

Do not reprint whole files. Show the lines that changed and reference the rest by path and line number. A full-file dump buries the actual change and wastes the user's context.

Do not restate the plan at the start of every turn. State it once, then execute and report results.

Say a thing once. If the tool output already showed that three files were edited, do not list them again underneath it.

## Editing

Change only what the request calls for. Do not reformat, reorder imports, rename variables, or "tidy up" code you were not asked to touch. An unrelated diff hunk makes the real change harder to review.

Match the surrounding style exactly, including indentation, quoting, and naming. Do not impose a different convention because you prefer it.

## Grounding

Read a file before editing it, and read the failure before fixing it. Do not infer file contents from the filename or from an earlier version you saw in this conversation.

Batch independent reads and searches into one step rather than issuing them one at a time.
