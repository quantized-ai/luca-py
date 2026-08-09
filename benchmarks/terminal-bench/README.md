# luca on Terminal-Bench

Measuring how well `luca` actually does on real terminal work, so that a change
to the prompt, the tool set or the compaction strategy can be measured instead
of argued about.

This directory is a **separate uv project**. It is not part of the `luca`
package, and nothing in `luca` knows it exists. It composes an agent through
core + contrib's public surface exactly the way any application would, which is
also the point: if the harness ever needs a private import to work, that is a
finding about the framework boundary and belongs in the spec discussion, not in
a patch here.

Two reasons it is its own project rather than a dependency group:

- `harbor` requires Python >= 3.12; the root `pyproject.toml` declares >= 3.11,
  and uv will not resolve the two together.
- `luca-ai` ships with `httpx` and `pydantic` as its only runtime dependencies.
  A benchmark has no business anywhere near that list.

## In a hurry

Four commands, cheapest first, each one only worth running if the last passed.
The full walkthrough is [below](#running-the-benchmark).

```bash
uv sync && export OPENROUTER_API_KEY=sk-or-...

# 1. does luca work unattended at all?  ~30s, no Docker, a fraction of a cent
uv run python -m luca_tb.runner --model openai/gpt-5.4-mini --provider openrouter \
    --workspace /tmp/luca-scratch --max-steps 20 -- "create hello.py and run it"

# 2. is Docker + Harbor healthy?  no luca involved.  slow once, images cache
uv run harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5      # expect 5/5

# 3. does luca install into a task container?  a container, zero tokens
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini \
    -i terminal-bench/fix-git --install-only

# 4. a real score over ten tasks
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini -l 10 -n 4
```

Step 2 is the one that takes real time: it pulls task images. Everything after
it is fast. If step 2 is not 5/5, the problem is Docker or Harbor and debugging
luca will waste your afternoon.

Note the `terminal-bench/` prefix on the task name in step 3. It is required —
`-i fix-git` matches nothing and aborts the job. See [task names are
namespaced](#task-names-are-namespaced).

## How it fits together

[Terminal-Bench](https://www.tbench.ai) 2.x is no longer run by the old `tb`
CLI. It runs on [Harbor](https://www.harborframework.com), whose model is that
the agent installs itself **into the task's own container** and is invoked
headlessly there. So there are three pieces, split by where they run:

| File | Runs | Does |
|---|---|---|
| `luca_tb/agent.py` | host | `LucaAgent`, a harbor `BaseInstalledAgent`: installs the driver into the task container, invokes it, reads the trajectory back |
| `luca_tb/runner.py` | inside the container | the headless driver: one instruction in, an exit code and an `AgentSession` out |
| `luca_tb/mapping.py` | both | the pure translations (model string, usage rollup); no harbor import, unit-tested on its own |

`luca-ai` is not on PyPI and the repo is private, so `install()` builds a wheel
on the host, uploads it plus `runner.py` into the container, and installs both
into a venv at `/opt/luca/`.

## Setup

```bash
uv sync                            # from this directory; installs harbor, ~1 min
docker info                        # the daemon has to be reachable
df -h                              # task images are large; budget ~30GB
export OPENROUTER_API_KEY=sk-or-...
```

The key has to be **exported**, not merely present in the repo's `.env`. The
driver reads no dotenv and no `luca.json` on purpose: a benchmark run should be
a pure function of its arguments, and silently inheriting whichever config file
happened to be lying around is the opposite of that. Harbor itself will take
`--env-file ../../.env` if you would rather not export anything.

## The driver on its own

Start here. No Docker, about thirty seconds, a fraction of a cent, and it
answers the only question that matters before any of the harness is involved:
does luca work unattended?

```bash
mkdir -p /tmp/luca-scratch && cd /tmp/luca-scratch

uv run --directory ~/path/to/luca-py/benchmarks/terminal-bench \
    python -m luca_tb.runner \
    --model openai/gpt-5.4-mini --provider openrouter \
    --workspace /tmp/luca-scratch \
    --session-out /tmp/luca-scratch/session.json \
    --max-steps 20 \
    -- "create fizzbuzz.py that prints 1 to 20 with fizzbuzz rules, then run it and show the output"
```

Tool calls stream to stdout as they happen (`→ write file_path=…`, `→ bash
command=…`), then the final answer. Check it was real rather than described:

```bash
cat fizzbuzz.py && python3 fizzbuzz.py && echo "exit=$?"
```

Exit codes: `0` completed, `1` the run aborted, `2` blocked on an approval gate
(under the default yolo mode that means something is misconfigured, not that a
human is wanted), `124` wall-clock timeout. The session is written after every
drive, so a timeout or a crash still leaves a readable trajectory to look at.

The `--` before the instruction is not decoration: task text is arbitrary, and
one starting with a dash would otherwise be parsed as a flag.

## Running the benchmark

Run these in order. Each stage rules out one class of failure, so that when
stage 6 gives a bad score you already know it is luca's fault and not Docker's.
The first three cost nothing.

Every command below runs from **this directory**. From the repo root, prefix
them with `uv run --directory benchmarks/terminal-bench` instead of `uv run`.

Flags worth knowing before you spend money (these are harbor 0.20's names, and
they are not the ones in the tbench.ai docs, which describe a newer build):

| Flag | Means |
|---|---|
| `-d` | dataset, `name@version` |
| `-a` | agent: a built-in name, or **our import path** `luca_tb.agent:LucaAgent` |
| `-m` | model, `provider/model` |
| `-l` | max tasks to run |
| `-i` | run one named task (repeatable, glob) — see the note below |
| `-k` | attempts per task — this is what feeds `pass@k` |
| `-n` | concurrent trials (default 4) |
| `--ak` | agent kwarg, `key=value` |
| `--env-file` | load a `.env` for provider keys |
| `--install-only` | install the agent, then stop. No tokens spent |
| `--print-config` | resolve and print the config, then stop. Nothing runs |

### Task names are namespaced

`-i` is an `fnmatch` glob against the **full** task name, and every task in this
dataset is prefixed with its dataset org:

```bash
-i terminal-bench/fix-git     # matches
-i '*/fix-git'                # matches
-i fix-git                    # matches NOTHING, and fails the whole job
```

A bare name silently matches nothing and Harbor aborts with `No tasks matched
the filter(s)`. There is no CLI command that lists a dataset's tasks, so:

```bash
uv run python -c "
import asyncio
from harbor.models.job.config import DatasetConfig
async def main():
    cfgs = await DatasetConfig(name='terminal-bench/terminal-bench-2-1').get_task_configs(
        disable_verification=False)
    for c in sorted(c.name for c in cfgs): print(c)
asyncio.run(main())
"
```

89 tasks, and none of them is a warm-up: the roster runs from `fix-git` and
`regex-log` up through `compile-compcert`, `make-doom-for-mips` and
`install-windows-3.11`. There is no `hello-world` here — that name belongs to
Harbor's own `examples/` directory, not to this dataset.

### Stage 0 — does the command resolve (free, instant)

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini \
    -i terminal-bench/fix-git --print-config
```

Prints the resolved job and exits. No Docker, no tokens. It catches a bad agent
import path and a misspelled dataset name.

It does **not** validate task names: `--print-config` echoes `-i` back verbatim
without resolving the dataset's task list, so a name that matches nothing looks
fine here and fails at job start instead. Check names against the listing above
rather than trusting this stage for them.

### Stage 1 — is the harness sane

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5
```

Expect 5/5 at reward `1.0`. The oracle runs each task's own reference solution,
so anything less is Docker, disk, the dataset download or Harbor itself. This
is the first stage that pulls task images, so it is slow once and fast after.
Do not touch the adapter until it is green.

### Stage 2 — do the tests actually fail

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 -a nop -l 5
```

Expect 0/5. This catches tasks that pass with no agent at all, which would
inflate our score for free. Skipping this stage is how people end up retracting
a number.

### Stage 3 — does luca install (free of tokens)

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini \
    -i terminal-bench/fix-git --install-only
```

Runs `install()` and stops before the agent does any work, so it costs a
container and no model calls. This is the stage that shakes out the whole setup
path: system packages, the uv install, the wheel upload, the venv, and the
`import luca` smoke check. Almost every first-time failure lives here rather
than in the agent.

### Stage 4 — one task, end to end

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent \
    -m openrouter/openai/gpt-5.4-mini \
    -i terminal-bench/fix-git --ak max_steps=40
```

The reward is almost beside the point here. What you are checking:

- `<trial>/result.json` shows `agent_setup` finishing with no `exception_info`.
- `<trial>/agent/session.json` exists and is non-empty, meaning the
  `/logs/agent/` mount worked.
- `agent_result.n_input_tokens` and `n_output_tokens` are non-zero, meaning
  `populate_context_post_run` parsed the trajectory. `cost_usd` may be `null`
  when the catalog does not price that model; that is correct, not a bug.
- `<trial>/agent/luca.txt` shows real tool calls, not approval prompts. Prompts
  would mean yolo did not take.

### Stage 5 — ten-task pilot

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent \
    -m openrouter/openai/gpt-5.4-mini -l 10 -n 4
```

The pilot exists to find systematic failures cheaply. Read every failing
trial's `luca.txt` against the triage table below before scaling up.

### Stage 6 — the real run

```bash
mkdir -p ../../runs
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent \
    -m <model> -k 5 -n 8 2>&1 | tee ../../runs/full-$(date +%F).log
```

All 89 tasks at `-k 5` is 445 trials, which is the leaderboard's minimum and
therefore the number worth getting used to. Harbor prints the total at job
start; size the time and cost budget from that rather than guessing. `-k 5` is
also what puts a confidence interval on the accuracy instead of a single
noisy number.

If you only want a score for ourselves and not a submission, `-k 1` over all
89 tasks is a fifth of the cost and still tells you most of what you need.

Agent knobs go through `--ak`:

```bash
--ak max_steps=300 --ak timeout=1200 --ak reasoning=high --ak subagents=true
```

### Provider keys

Beyond the exported variable from [Setup](#setup), `LUCA_API_KEY` works as a
single override across providers: the adapter forwards it into the container
under whichever variable name luca's provider actually reads, so one gateway
key covers several routes without renaming anything. A missing key fails fast,
before any container is started.

### Through Docker

For a box that already has harbor, uv and a Python 3.12 in it. Run from the
**repo root** — the compose file reads `${PWD}` and everything depends on it
being the repo:

```bash
docker compose -f docker/docker-compose.bench.yml run --rm bench \
    harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5
```

`Dockerfile.harbor` sits at the repo root; the compose file lives in `docker/`
because Compose auto-loads a `.env` next to itself for variable interpolation,
and the root `.env` is the agent's runtime secrets file written for
python-dotenv. Keeping them apart stops one quoting difference from breaking
every benchmark run with an error that points at the wrong file.

The service mounts the host Docker socket, so Harbor's task containers are
**siblings** spawned by the host daemon rather than children, and every bind
mount it asks for resolves against host paths. The repo is therefore mounted at
the same absolute path inside and out. If a run behaves differently here than
on the host, that is the first thing to check.

Two consequences worth knowing. The socket means task containers are not
isolated from the host daemon, which is the accepted trade for avoiding
docker-in-docker — do not point this at task definitions you do not trust. And
the wheel is built inside the container rather than into the mounted `dist/`,
so a Linux build never lands in your checkout.

## What the benchmark actually produces

Each task is a container plus a hidden test script. The agent gets one
instruction and whatever time the task allows; afterwards the verifier runs the
tests it never saw and emits a **reward**, normally `1.0` or `0.0`. Nothing
about how the agent got there is scored: not the tokens, not the explanation,
not the tool choices. Only the state it left the filesystem in.

The headline number is **accuracy**: the mean reward over every trial. With
`-k 5` that is 445 trials, which is what lets the leaderboard put a confidence
interval on it (the current top entry reads `83.8% ± 1.2%`). A run at `-k 1`
gives a single noisy number instead.

Two things that surprise people:

- **Errored trials count as reward 0.** A crash in our adapter is
  indistinguishable from the agent failing, as far as the metric goes.
- **Partial credit is rare.** Most tasks are all-or-nothing, so "nearly worked"
  and "did nothing" score the same.

Alongside the score you get the full trajectory for every trial, which is the
part worth more than the number early on: the agent's stdout, its
`AgentSession`, the verifier's own output, and per-phase timings.

## Reading the results

Harbor writes to `jobs/<job-name>/`. To pull apart the run you just did:

```bash
JOB=$(ls -t jobs | head -1)
TRIAL=$(ls jobs/$JOB | grep -v '\.json$' | head -1)

# the whole job
jq '.stats.evals' jobs/$JOB/result.json
jq '.stats | {n_completed_trials, n_errored_trials, cost_usd}' jobs/$JOB/result.json

# one trial
cat jobs/$JOB/$TRIAL/agent/luca.txt                  # what the agent actually did
jq '.exception_info'   jobs/$JOB/$TRIAL/result.json  # null when nothing broke
jq '.agent_result'     jobs/$JOB/$TRIAL/result.json  # tokens and cost
jq '.verifier_result.rewards.reward' jobs/$JOB/$TRIAL/result.json   # 1.0 or 0.0
cat jobs/$JOB/$TRIAL/verifier/reward.txt             # the same, one line
```

Then browse the whole thing properly, including a side-by-side comparison
matrix across jobs:

```bash
uv run harbor view jobs                              # http://127.0.0.1:8080
```

| Path | What it tells you |
|---|---|
| `result.json` (job) | `stats.evals[<key>].pass_at_k`, `reward_stats`, `n_errored_trials`, aggregate tokens and `cost_usd` |
| `<trial>/result.json` | per-task `verifier_result`, `agent_result` (our `AgentContext`), `exception_info`, timings for `environment_setup` / `agent_setup` / `agent_execution` / `verifier` |
| `<trial>/verifier/test-stdout.txt`, `reward.txt` | what the graders ran and why they failed |
| `<trial>/trial.log` | every command the adapter ran in the container, in order |
| `<trial>/agent/luca.txt` | the driver's stdout |
| `<trial>/agent/session.json` | the full `AgentSession` — replay it offline with `pretty_print` |

`uv run harbor view jobs` opens a local UI over all of it, including a
side-by-side comparison matrix across jobs. Use it for triage; use `result.json`
for anything you report.

## What to expect from the first real run

Probably not a good score, and that is fine. Terminal-Bench 2.x is hard, a
cheap model makes it harder, and this is luca running unattended for the first
time. The number is the least interesting output of the first few runs; the
failure modes in `luca.txt` are the point, because they separate into three
piles that want three different responses.

**Failed** is an ordinary outcome. luca tried and did not get there. Nothing to
file, it is what the benchmark is for.

**Errored** is our bug. A non-null `exception_info`, or a `ProjectionError` or
`ToolExecutionError` in `luca.txt`. These belong in the issue tracker with the
trial's `session.json` attached, since it replays offline.

**A core gap** is the valuable one: something that failed because of how the
data model or the runner works rather than because the model was not good
enough. Context overflow, a tool result that could not be represented, a turn
that could not be resumed. Those go in the `core gaps` line of the results
block and feed back into the spec, which is the whole reason for measuring.

One deliberate decision worth knowing while reading numbers: a timeout is
recorded as a failed task, not an errored trial, so the verifier still scores
it and it does not inflate `n_errored_trials`. Same for an approval gate.

## Failure triage

| Symptom | Likely cause | Where to look |
|---|---|---|
| Every trial errors in `agent_setup` | install broke: uv missing, wheel not uploaded, container Python too old | `<trial>/result.json` `exception_info` |
| `ripgrep (rg) was not found on PATH` in `luca.txt` | `ensure_system_dependencies` lost `ripgrep`; `glob` and `grep` shell out to it | `agent.py` `install()` |
| `luca.txt` stops mid-task with no final message | hit `--max-steps` or `--timeout` | count `TurnStart` entries in `session.json`; raise the bound if it was still making progress |
| Approval prompts in `luca.txt`, exit `2` | yolo did not reach the strategy | `agent.py` `run()` |
| Reward `0` but the agent reported success | declared done without verifying — a prompt problem, not a harness one | `prompt_template.md` against `verifier/test-stdout.txt` |
| Reward `0`, verifier says the file is in the wrong place | `--workspace` does not match the task's `WORKDIR` | `<trial>/config.json` |
| A traceback from luca itself (`ProjectionError`, `ToolExecutionError`, …) | **a core finding**, not a bench bug | file it with `session.json` attached; it replays offline |
| Trial passes but token counts are `null` | `context_from_session` could not read the trajectory | `<trial>/agent/` contents |
| Host and Docker disagree | the sibling-container path problem above | compare mount paths in the two `<trial>/config.json` |

## Submitting to the leaderboard

A scored local run and a leaderboard row are not the same thing. The
[terminal-bench-2-1](https://github.com/harbor-framework/terminal-bench-2-1)
repo runs CI over every submission, and it rejects rather than warns. The rules
below come from its `leaderboard/ci/static_analysis.py`, not from the prose.

**Coverage.** All **89 tasks**, at least **5 trials each** (`-k 5`), so 445
trials minimum. Partial runs are rejected.

**Errored trials count as reward 0.** They are not dropped from the metric.
Every crash in our adapter costs score directly, which is the strongest reason
to get stage 3 clean before spending on a full run.

**No knob-twiddling.** `timeout_multiplier` must be unset or `1.0`, and CI
rejects any of `agent_timeout_multiplier`, `verifier_timeout_multiplier`,
`agent_setup_timeout_multiplier`, `environment_build_timeout_multiplier`,
`override_timeout_sec`, `override_setup_timeout_sec`, `max_timeout_sec`,
`override_cpus`, `override_gpus`, `override_memory_mb`, `override_storage_mb`.
The check runs against the job config *and* every per-trial config, so the two
cannot diverge.

This is also why the adapter passes `--timeout 0` by default. Every task
declares its own `agent.timeout_sec` and Harbor enforces it; a second ceiling
inside the driver could only ever be the smaller of the two, and would hand
back failures the task's own budget allowed. Do not set `--ak timeout=` for a
scored run.

**Effort is keyed on the kwarg name.** CI groups trials by
`(agent, agent version, model, kwargs["reasoning_effort"])`. Our flag is
therefore named `reasoning_effort` rather than `reasoning`:

```bash
--ak reasoning_effort=high      # correct: fills the Effort column
--ak reasoning=high             # wrong: records as "none", merges rows
```

**Public upload.** Trials have to be readable on Harbor Hub, because CI
re-derives every number from there rather than trusting the submitted JSON.

### The run

```bash
uv tool install "harbor[daytona]"      # a cloud sandbox; 445 trials locally is slow
harbor auth login

uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent \
    -m <provider/model> \
    --ak reasoning_effort=<effort> \
    -e daytona -k 5 -n <concurrency> \
    --upload --public
```

Forgot `--upload`? `harbor upload <job-dir> --public` after the fact.

### The PR

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1.git
cd terminal-bench-2-1/leaderboard
uv run lb submit https://hub.harborframework.com/jobs/<uuid>
```

That runs filter → metadata → open-prs in one go. It needs an authenticated
`gh`, and it will prompt for display names and org URLs for luca the first time
since we are not yet in its `display-names.json`. CI then runs static analysis,
maintainers review the trajectories, and a merged PR becomes a leaderboard row.

Budget for it before starting: the current top entry (Claude Code on Fable 5,
83.8%) cost **$552** for its run. A cheap model over 445 trials is far less,
but this is not a thing to start casually at the end of a day.

## Results log

One block per run, appended newest first. Paste the same block in the PR and
attach the job's `result.json`.

Keep errors separate from failures: a task luca *errored* on is our bug and
belongs in the tracker, while `core gaps` is the line that feeds back into the
data model and the spec. Comparisons only mean something between runs that used
the same model and the same `-k`.

```
dataset:      terminal-bench/terminal-bench-2-1
agent:        luca @ <git sha>
model:        <provider/model>, reasoning_effort=<level>
config:       max_steps=<n>, subagents=<on|off>, compaction=<on|off>
tasks:        <n>/89 covered, k=<n>  ->  <total> trials
accuracy:     <xx.x>% ± <x.x>          # mean reward; this is the headline number
errored:      <n> trials               # these count as reward 0, not excluded
tokens:       <in> in / <out> out / <cache> cached
cost:         $<x.xx>
wall clock:   <hh:mm>
failures:     <top 3 recurring causes, with task names>
core gaps:    <anything that looked like a data-model or runner limitation>
```

_No scored run yet. What has been verified so far, on 2026-08-09:_

```
harness:      oracle,  5 tasks -> 5/5 reward 1.0     (Docker + Harbor healthy)
baseline:     nop,     5 tasks -> 0/5 reward 0.0     (no free passes)
                       1 trial errored: VerifierTimeoutError on
                       torch-tensor-parallelism, a slow verifier rather than
                       anything of ours
install:      luca,    terminal-bench/fix-git --install-only -> agent_setup
                       finished in 35s with no exception. apt packages, uv,
                       wheel upload, venv and `import luca` all returned 0.
```

The install path is proven in a real task container. What is still unproven is
everything after it: luca has not yet been asked to solve a task, so there is
no reward, no token count and no cost recorded here.

## Tests

```bash
uv run pytest tests/
```

Covers `mapping.py` only: model-string parsing and the usage rollup, the two
places the adapter could silently get the wrong answer. No Docker, no harbor
import, no network.
