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
uv sync                       # from this directory
docker info                   # the daemon has to be reachable
df -h                         # task images are large; budget ~30GB
export OPENROUTER_API_KEY=...
```

## The driver on its own

Useful for debugging without any of the benchmark machinery:

```bash
uv run python -m luca_tb.runner "create hello.py that prints hi, then run it" \
    --model openai/gpt-5.4-mini --provider openrouter \
    --workspace /tmp/scratch --max-steps 20 \
    --session-out /tmp/session.json
```

Exit codes: `0` completed, `1` the run aborted, `2` blocked on an approval gate
(under the default yolo mode that means something is misconfigured, not that a
human is wanted), `124` wall-clock timeout. The session is written after every
drive, so a timeout or a crash still leaves a readable trajectory.

The driver reads **no** `luca.json`. A benchmark run has to be a pure function
of its arguments, and silently inheriting a config file that happened to be in
the task image is the opposite of that.

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
| `-i` | run one named task (repeatable) |
| `-k` | attempts per task — this is what feeds `pass@k` |
| `-n` | concurrent trials (default 4) |
| `--ak` | agent kwarg, `key=value` |
| `--env-file` | load a `.env` for provider keys |
| `--install-only` | install the agent, then stop. No tokens spent |
| `--print-config` | resolve and print the config, then stop. Nothing runs |

### Stage 0 — does the command resolve (free, instant)

```bash
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini \
    -i hello-world --print-config
```

Prints the resolved job and exits. No Docker, no network, no tokens. If the
agent import path is wrong or the dataset name is misspelled, you find out here
in a second rather than after a dataset download.

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
    -i hello-world --install-only
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
    -i hello-world --ak max_steps=40
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
    -m <model> -k 3 -n 8 2>&1 | tee ../../runs/full-$(date +%F).log
```

Harbor prints the total trial count at job start; size the time and cost budget
from that number rather than guessing. `-k 3` is what makes pass@1 against
pass@3 meaningful and smooths out flaky tasks.

Agent knobs go through `--ak`:

```bash
--ak max_steps=300 --ak timeout=1200 --ak reasoning=high --ak subagents=true
```

### Provider keys

Harbor reads them from the environment, so either export the one your model
needs or point `--env-file` at a file holding it:

```bash
export OPENROUTER_API_KEY=sk-...
# or
uv run harbor run ... --env-file ../../.env
```

`LUCA_API_KEY` works as a single override across providers: the adapter
forwards it into the container under whichever variable name luca's provider
actually reads. A missing key fails fast, before any container is started.

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

## Reading the results

Harbor writes to `jobs/<job-name>/`.

| Path | What it tells you |
|---|---|
| `result.json` (job) | `stats.evals[<key>].pass_at_k`, `reward_stats`, `n_errored_trials`, aggregate tokens and `cost_usd` |
| `<trial>/result.json` | per-task `verifier_result`, `agent_result` (our `AgentContext`), `exception_info`, timings for `environment_setup` / `agent_setup` / `agent_execution` / `verifier` |
| `<trial>/verifier/test-stdout.txt`, `reward.txt` | what the graders ran and why they failed |
| `<trial>/agent/luca.txt` | the driver's stdout |
| `<trial>/agent/session.json` | the full `AgentSession` — replay it offline with `pretty_print` |

`uv run harbor view jobs` opens a local UI over all of it, including a
side-by-side comparison matrix across jobs. Use it for triage; use `result.json`
for anything you report.

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
model:        <provider/model>, reasoning=<level>
config:       max_steps=<n>, timeout=<s>, subagents=<on|off>, compaction=<on|off>
trials:       <completed>/<total>, errored=<n>
pass@1:       <x.xx>
pass@3:       <x.xx>
tokens:       <in> in / <out> out / <cache> cached
cost:         $<x.xx>
wall clock:   <hh:mm>
failures:     <top 3 recurring causes, with task names>
core gaps:    <anything that looked like a data-model or runner limitation>
```

_No runs recorded yet. The first full run is the baseline._

## Tests

```bash
uv run pytest tests/
```

Covers `mapping.py` only: model-string parsing and the usage rollup, the two
places the adapter could silently get the wrong answer. No Docker, no harbor
import, no network.
