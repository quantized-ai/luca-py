# 0011 — implementation plan

Execution order for the PRD. Each step leaves the suite green
(`uv run py.test tests/`), then `uv run ruff check --fix && uv run ruff
format`. No new runtime dependencies. The credential model, resolution rules
and error messages live in `prd.md`; this file is the order, the file list and
the test list.

**Standing gate:** the 62 existing tests in
`tests/client/test_transports/test_bedrock/` pass **unchanged** through every
step. Their fixture (`bedrock_transport_factory`) always passes `api_key=`, so
they all ride the bearer path, which SigV4 must not disturb. If one needs
editing, the step is wrong.

---

## Step 1 — AWS env isolation, before any implementation

**File**: `tests/client/conftest.py`

The autouse `no_real_env` fixture strips provider env vars so a forgotten
`api_key=` cannot reach a real provider. AWS is not in its list. Without this,
a contributor with real credentials exported gets different results from CI,
and "missing region raises `ConfigurationError`" passes for the wrong reason.

Add: `AWS_BEARER_TOKEN_BEDROCK`, `BEDROCK_AWS_REGION`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`,
`AWS_DEFAULT_REGION`, `AWS_PROFILE`.

Also point `AWS_SHARED_CREDENTIALS_FILE` and `AWS_CONFIG_FILE` at
non-existent paths, so a developer's real `~/.aws` is never read by a unit
test even though the resolver falls back to `~` when the vars are unset.

First, so every later step is tested against a clean environment.

## Step 2 — `AwsCredentials`

**Files**: `luca/client/types/auth.py` (new),
`luca/client/types/__init__.py`, `luca/client/__init__.py`

- `Credentials(BaseModel)`: empty marker, `extra="forbid"`, `frozen=True`.
- `AwsCredentials(Credentials)`: five optional `str | None` fields per the
  PRD.
- Exported from `luca.client` — the agent layer constructs one.

**Tests** (`tests/client/test_types/test_auth.py`): construction with no
fields; `frozen` rejects assignment; hashable (the `_provider_cache` key
requirement); `extra="forbid"`.

## Step 3 — the signer

**File**: `luca/client/transports/bedrock/sigv4.py` (new)

Pure functions, no I/O, no clock of their own:

```
canonical_request(method, path, query, headers, body) -> str
string_to_sign(canonical, now, region, service) -> str
sign(method, url, headers, body, credentials, now, service="bedrock") -> dict
```

The first two are public so tests can assert the intermediate artifacts and a
failure says which stage broke. `sign` returns only the headers to add.

- Canonical path and query from `httpx.URL.raw_path`, split at `?`.
- Headers sorted by lowercase name, values trimmed.
- `sha256` hex of the body.
- Four-stage signing key: `AWS4` + secret -> date -> region -> service ->
  `aws4_request`.

**Tests** (`tests/client/test_transports/test_bedrock/test_sigv4.py`):
vectors transcribed from AWS's published `aws-sig-v4-test-suite`, asserting
canonical request, string to sign and `Authorization` separately. Plus a
no-session-token case, a session-token case (which adds
`x-amz-security-token` to the signed set) and a colon-in-path case.

## Step 4 — credential and region resolution

**File**: `luca/client/transports/bedrock/credentials.py` (new)

- `_ResolvedAwsCredentials`: frozen, everything required but `session_token`.
- `resolve(explicit) -> _ResolvedAwsCredentials | None`
- The chains, file locations, `RawConfigParser`, profile selection and every
  error message from the PRD.

**Tests** (`.../test_bedrock/test_credentials.py`), `tmp_path` for the files
and `monkeypatch.setenv` for the environment:

- env vars beat files
- `default` profile from the credentials file
- `AWS_PROFILE` selects a named profile; `AwsCredentials.profile` beats it
- `[profile foo]` in config vs bare `[foo]` in credentials
- `AWS_SHARED_CREDENTIALS_FILE` / `AWS_CONFIG_FILE` redirection
- a secret containing `%` survives (the `RawConfigParser` regression)
- the region chain, each rung, `BEDROCK_AWS_REGION` still winning
- an access key with no secret raises rather than falling through
- a named profile that does not exist raises
- SSO-only, `credential_process` and `role_arn` profiles each raise with the
  profile name in the message
- nothing anywhere raises, naming what to set

## Step 5 — provider and transport

**Files**: `luca/client/providers/bedrock.py`,
`luca/client/transports/bedrock/transport.py`,
`luca/client/transports/bedrock/stream.py`

- Provider: resolve region first (its own chain, since `base_url` may be an
  explicit VPC endpoint), then credentials unless the bearer path is
  selected. Pass both down. The existing "no region" `ConfigurationError` must
  consult the whole chain before raising, or a region living in a profile
  produces a spurious failure.
- Transport: `_build_chat_completion_httpx_request` override,
  `_signed_headers(method, url, body)`, and a `_now()` hook implemented as
  `datetime.fromtimestamp(time.time(), tz=UTC)` so the existing `frozen_time`
  fixture in `tests/client/conftest.py` freezes it, rather than inventing a
  second clock-freezing mechanism.
- Stream: both `_open_http` methods onto signed `content=` bytes.
- Rewrite the `transport.py` module docstring, which currently states the
  opposite of what is now true.

**Tests**:

`.../test_bedrock/conftest.py` gains `bedrock_sigv4_transport_factory`. The
existing factory is left alone — that is what keeps the 62 current tests on
the bearer path.

`.../test_bedrock/test_auth.py` (new), using the captured-request idiom from
`test_openai/test_payload_building.py:134` and `frozen_time`, asserting one
whole dict of url + body + headers:

- SigV4 headers well formed for a credentials-configured transport
- session token appears as `X-Amz-Security-Token` and in `SignedHeaders`
- `AWS_BEARER_TOKEN_BEDROCK` gives a `Bearer` header and no SigV4 headers
- a colon-bearing inference-profile id signs the path httpx actually emits

Streaming, over real HTTP, on both the sync and async stream classes: assert
the request body bytes hash to the payload hash inside the signature. This
is the `json=` vs `content=` drift guard, and there is no HTTP-level or async
bedrock stream test today, so both are new.

`tests/client/_helpers/httpx_mocks.py` gains `eventstream_response(frames)`.
The frame builder currently lives as a private `_frame()` in
`test_bedrock/test_stream.py`; promote it so both files share one builder.

`tests/client/test_providers/test_bedrock_provider.py` (new — the test tree
mirrors `luca/client/`, so it does not belong under `test_transports/`).
`BedrockProvider` has no tests at all today, so this covers the existing
behavior as well as the new: region env var folds into the runtime hostname;
explicit `base_url=` wins; `transport=` short-circuits both; no region
anywhere raises; a region from a profile no longer raises; bearer produces no
resolved credentials and credentials produce no bearer.

## Step 6 — `credentials=` through the client core

**Files**: `luca/client/transports/base.py`, `luca/client/providers/base.py`,
`luca/client/providers/__init__.py`, `luca/client/_client.py`

Store and forward through both constructors; add to `resolve_provider`'s
`common` (which already drops `None`, so no other provider changes); extend
the `_get_cached_provider` key; add the kwarg to `completion`, `acompletion`,
`completion_stream`, `acompletion_stream`.

**Tests**: two `AwsCredentials` differing by one field produce two cached
providers, not one (the cache-key requirement); a provider that does not
understand credentials is unaffected.

## Step 7 — agent layer

**Files**: `luca/agent/contrib/tui/auth.py`, `.../app.py`, `.../cli.py`,
`.../wiring.py`, `luca/agent/core/runner.py`,
`luca/agent/contrib/simple_context_manager/manager.py`

- `AuthEntry` becomes a discriminated union on `type`; `api` untouched, `aws`
  added. `credentials_for(auth, provider)` beside `api_key_for`. While here,
  add `api_key_for` to `__all__` — it is the primary read every caller makes
  and is currently omitted.
- `app.py`: a `credentials_for` accessor and a `repoint_credentials` sibling
  to `repoint_api_key`, called from the same site in `commands.py`. It must
  mutate the context manager too, exactly as `repoint_api_key` does.
- `cli.py` and `wiring.py`: thread `credentials` beside `api_key` into the
  runner and `build_context_manager`.
- `runner.py`: runtime-only `credentials` field, included in `__eq__`, and
  threaded through both `completion_options` (the method and the module-level
  function). When `None` the kwarg is left **off entirely**, exactly as
  `api_key` is, so the client falls back to its own resolution.
- `manager.py`: the same field beside `api_key`.
- Nothing on `LLMConfig`.

**Tests**:

- `tests/agent/contrib/tui/test_auth.py`: the union round-trips; an `api`
  entry is identical to today; an `aws` entry validates; a profile-only entry
  validates; a malformed one raises `LucaConfigError` naming the provider.
- `tests/agent/contrib/tui/test_commands.py`: a `/model` switch from Bedrock
  to an API-key provider clears the credentials and sets the key, and back.
  Include a case that passes a `context_manager` — that branch of
  `repoint_api_key` is currently uncovered by any test.
- `tests/agent/contrib/tui/test_cli.py`: an AWS analogue of
  `test_the_auth_file_key_reaches_the_runner_and_the_context_manager`.
- `tests/agent/test_runner_model_options.py`: an AWS analogue of the
  persistence guard, asserting the secret access key does not appear in
  `json.dumps(session.model_dump(mode="json"))`. This is the
  security-relevant test of the whole change.

## Step 8 — live smoke test

**Files**: `.../test_bedrock/test_live.py` (new), `pyproject.toml`

The only check that proves AWS accepts the signature rather than it merely
being self-consistent. Everything else is hermetic.

Marked `live`, skipped without real credentials, never collected by default.
`--strict-markers` is on, so `markers = ["live: hits a real provider; opt
in"]` is registered in the same change. It opts out of the autouse
`no_real_env` fixture, which would otherwise strip what it needs.

Four cases against Nova Micro or Nova Lite (both marked verified-live in
`capabilities.py`):

1. non-streaming `converse`, a real completion returns
2. streaming `converse-stream`, frames decode
3. a tool call round trip
4. a deliberately corrupted secret returns `AuthenticationError`

The fourth matters more than it looks: without it, a signature AWS happens to
tolerate is indistinguishable from a correct one.

Run: `uv run py.test -m live tests/client/test_transports/test_bedrock/`.

## Step 9 — docs

- `docs/client/09-providers-and-transports.md`: the provider table row, and
  the paragraph that currently says SigV4 is not used.
- `docs/agent/contrib/tui/config.md` (`## Credentials`) and
  `docs/agent/contrib/tui/README.md`: the new `auth.json` entry.
- `AGENTS.client.md`: provider table and file-layout tree.
