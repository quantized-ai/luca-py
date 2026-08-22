# AWS SigV4 signing and credential resolution for Bedrock

## Why

`luca.client` already talks to Bedrock. `BedrockProvider` plus the Converse
transport (`transport.py`, `stream.py`, `capabilities.py`) landed in `ede441f`
and carry 62 tests: Converse payload translation, the binary
`vnd.amazon.eventstream` decoder, a per-model capabilities table, error
mapping, catalog wiring.

Authentication is the gap, and the transport says so at
`transports/bedrock/transport.py:12`:

> Auth is a plain bearer token (`AWS_BEARER_TOKEN_BEDROCK`); no SigV4.

A Bedrock API key is a recent AWS feature and most accounts do not issue one.
An IAM user with an access key pair, a developer who ran `aws configure`, and
a CI job with exported credentials are all locked out today. This adds SigV4
request signing and resolves AWS credentials from the environment and the
shared AWS config files.

## Objectives

1. Sign Bedrock Converse requests with AWS Signature Version 4, on both the
   non-streaming and the streaming path.
2. Resolve credentials from explicit arguments, environment variables, and
   `~/.aws/credentials` + `~/.aws/config` including named profiles.
3. Give the client a typed, opaque way to carry a non-string credential, so
   `api_key: str | None` does not have to grow AWS-shaped fields.
4. Let `auth.json` describe an AWS credential, so the TUI holds it the way it
   holds an API key today.
5. Change nothing for a working `AWS_BEARER_TOKEN_BEDROCK` setup.
6. No new runtime dependencies. `hmac` / `hashlib` / `configparser` only.

## Non-objectives

Deliberately not supported, each because it needs network calls at credential
time plus expiry and refresh handling, which is a different kind of change
from a pure signer:

- EC2 instance metadata (IMDSv2)
- ECS and EKS container credential endpoints
- Web identity tokens (`AssumeRoleWithWebIdentity`)
- The SSO token cache
- `assume-role` / `source_profile` chaining
- `credential_process`

Each of these, when detected in a profile, is a loud error rather than a
silent fall-through. `aws configure export-credentials --profile <name>`
covers the gap and the error message says so.

Also unchanged: Bedrock still refuses `response_format` (Converse has no
structured-output field) and still offers no provider-native tools.

## The credential model

Two types, one public.

**`AwsCredentials`** (`luca/client/types/auth.py`, exported from
`luca.client`) is what a caller *may* specify. Every field is optional:

```python
AwsCredentials(
    access_key_id=None,
    secret_access_key=None,
    session_token=None,
    region=None,
    profile=None,
)
```

`frozen=True`, because it becomes part of the `_provider_cache` key in
`_client.py` and has to be hashable. `extra="forbid"` like every model in the
codebase.

It subclasses `Credentials`, an empty marker base. The client core stores and
forwards a `Credentials`; only `BedrockProvider` and `BedrockTransport` know
what is inside one. This is the same division `provider_options` already uses:
core routes, the provider interprets.

**`_ResolvedAwsCredentials`** (`transports/bedrock/credentials.py`, private)
is what the signer needs. `access_key_id`, `secret_access_key` and `region`
are all required; `session_token` stays optional. `BedrockProvider` runs the
resolution chain and hands the resolved form to the transport, so the signer
never sees a partial credential and a caller never has to restate a region
already sitting in `~/.aws/config`.

## Resolution rules

### Auth scheme

1. An explicit `api_key=`, or `AWS_BEARER_TOKEN_BEDROCK` in the environment,
   selects the bearer path. SigV4 is skipped entirely.
2. Otherwise SigV4, using the resolved credentials.
3. Neither available: `ConfigurationError` naming both routes.

Bearer-wins matches AWS's own SDKs, and it is what keeps every existing setup
working with no config change.

### Credentials

First hit wins:

1. Explicit fields on the passed `AwsCredentials`.
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, with `AWS_SESSION_TOKEN`
   when present.
3. The shared credentials file, under the selected profile.
4. The shared config file, under the selected profile.

An access key without its secret is an error, not a partial hit. Falling
through to the next rung would silently mix credentials from two sources.

### Profile selection

`AwsCredentials.profile`, then `AWS_PROFILE`, then `default`.

### File locations

- Credentials: `AWS_SHARED_CREDENTIALS_FILE`, else `~/.aws/credentials`.
  Section names are bare (`[default]`, `[work]`).
- Config: `AWS_CONFIG_FILE`, else `~/.aws/config`. Section names are
  `[profile work]`, except `[default]` which has no prefix.

Parsed with `configparser.RawConfigParser`. Raw, not the interpolating
default: a secret access key containing `%` is legal and would otherwise
raise `InterpolationSyntaxError`.

A missing file is not an error. A missing *named* profile is, because the
user asked for it by name.

### Region

First hit wins:

1. `AwsCredentials.region`
2. `BEDROCK_AWS_REGION` (first among the env vars so today's behavior is
   unchanged)
3. `AWS_REGION`
4. `AWS_DEFAULT_REGION`
5. `region` under the selected profile in the shared config file

An explicit `base_url=` still wins for the endpoint, which keeps VPC
endpoints and proxies working, but the region is still needed as a signing
input and is resolved independently.

## Signing

Service name `bedrock`. Ordinary SigV4 over a single JSON body, on both
`converse` and `converse-stream`: only the *response* is eventstream-framed,
so none of the `aws-chunked` streaming-signature machinery applies.

Headers added to the request:

| Header | When |
|---|---|
| `X-Amz-Date` | always, `YYYYMMDDTHHMMSSZ` |
| `Authorization` | always |
| `X-Amz-Security-Token` | session token present |

Signed header set: `content-type`, `host`, `x-amz-date`, plus
`x-amz-security-token` when present.

### Sign the bytes that go on the wire

Two rules, both load-bearing.

**The body.** SigV4 signs `sha256` of the exact request body, so the body
must be serialized once and both signed and sent. `_headers()` cannot do the
signing because it has never seen the payload. The non-streaming path
overrides `_build_chat_completion_httpx_request`; the streaming path stops
passing `json=payload` (which lets httpx serialize independently) and passes
pre-serialized `content=` bytes.

**The path.** Bedrock inference-profile ids contain a colon:

```
/model/us.anthropic.claude-sonnet-4-20250514-v1:0/converse
```

Any disagreement between the path we canonicalize and the path httpx emits
produces a 403 that reads as a credentials problem. So the signer reads
`httpx.Request.url.raw_path` — the actual bytes — and splits the query string
off it, rather than re-deriving a path from a string we built ourselves.

## `auth.json`

`AuthEntry` becomes a discriminated union on `type`. The existing shape is
untouched:

```jsonc
{
  "openrouter": { "type": "api", "key": "sk-or-..." },
  "bedrock": {
    "type": "aws",
    "access_key_id": "AKIA...",
    "secret_access_key": "...",
    "session_token": null,
    "region": "us-east-1"
  }
}
```

A profile-only entry is also valid:

```jsonc
{ "bedrock": { "type": "aws", "profile": "work" } }
```

The module docstring in `luca/agent/contrib/tui/auth.py` already anticipated
this union ("`"oauth"` becomes a second member of the union when it lands");
`aws` is the first member to actually arrive.

## Where credentials may not go

`LLMConfig` is persisted with the session and copied onto every assistant
entry as provenance. Its docstring says "NO CREDENTIALS" and that stays true.
AWS credentials travel the same runtime-only path `api_key` already does:
`AgentSessionRunner.credentials`, never serialized, left off the client call
entirely when `None` so the client falls back to its own resolution.

The test for this is an assertion that the secret access key does not appear
anywhere in `json.dumps(session.model_dump(mode="json"))`.

## Error messages

Every failure names what to do next.

| Situation | Message |
|---|---|
| No credentials and no bearer token | names `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, and `AWS_BEARER_TOKEN_BEDROCK` |
| No region anywhere | names `BEDROCK_AWS_REGION` / `AWS_REGION` and the profile's `region` |
| Named profile not found | names the profile and the file searched |
| Access key without secret | names the source that supplied the partial pair |
| Profile uses SSO / `credential_process` / `role_arn` | names the profile, the unsupported key, and `aws configure export-credentials --profile <name>` |

All are `ConfigurationError`, which already exists in the client's exception
hierarchy.
