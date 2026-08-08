#!/usr/bin/env bash
# Sync the bind-mounted benchmark project, then run whatever was asked for.
#
# The sync happens here rather than at image build time because the project
# depends on `luca-ai` as an editable path install, and that path only exists
# once the repo is mounted. The dependency wheels are already in the image's uv
# cache, so this is fast after the first run.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/workspace/benchmarks/terminal-bench}"
REPO_ROOT="$(cd "${BENCH_DIR}/../.." 2>/dev/null && pwd || true)"

if [[ ! -f "${BENCH_DIR}/pyproject.toml" ]]; then
    echo "bench-entrypoint: no project at ${BENCH_DIR}." >&2
    echo "The repo has to be bind-mounted at the same absolute path it has on the host;" >&2
    echo "run docker compose from the repo root so \${PWD} resolves correctly." >&2
    exit 1
fi

# Build the wheel INSIDE the container, not into the mounted repo's dist/.
# Writing there would leave root-owned artifacts in the contributor's checkout,
# and a Linux wheel where their host tooling expects to find its own.
if [[ -z "${LUCA_WHEEL:-}" ]]; then
    mkdir -p /opt/luca-dist
    # --python is this image's own interpreter, not the repo's `.python-version`.
    # luca builds to a pure-Python `py3-none-any` wheel, so the build
    # interpreter makes no difference to the artifact — but honouring the pin
    # would make uv download a whole managed CPython on every single run.
    uv build --wheel \
        --project "${REPO_ROOT}" \
        --out-dir /opt/luca-dist \
        --python "$(command -v python3)" >/dev/null
    LUCA_WHEEL="$(ls -t /opt/luca-dist/luca_ai-*.whl | head -n 1)"
    export LUCA_WHEEL
fi

uv sync --project "${BENCH_DIR}" --quiet

# Harbor writes its jobs/ tree relative to the working directory. Inside the
# bench project, which is on the mount, so results land on the host.
cd "${BENCH_DIR}"

exec uv run --project "${BENCH_DIR}" --no-sync "$@"
