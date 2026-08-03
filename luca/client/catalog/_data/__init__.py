"""Where catalog records are read from and written to.

Two files, in precedence order:

- `models.json`, generated from models.dev and shipped in the package. The
  offline floor — no import ever touches the network, so `catalog.get` works on
  a fresh clone with no connectivity.
- `~/.cache/luca/models.json`, written by a refresh and layered over the
  vendored file, so a model released after the last release is reachable
  without waiting for one.

Records are validated one at a time. A single record this version cannot read —
a file written by a newer luca, a truncated write — is skipped, never fatal:
the compaction gauge reads this catalog on every check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError

from ...types.catalog import ModelInfo

VENDORED_PATH = Path(__file__).with_name("models.json")

SOURCE = "models.dev"


def cache_path() -> Path:
    """`$XDG_CACHE_HOME/luca/models.json` or `~/.cache/luca/models.json`."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "luca" / "models.json"


def load_records(path: Path) -> list[ModelInfo]:
    """Every readable record in `path`, or an empty list when the file is
    missing, unreadable, or not the shape we write."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return []
    records = []
    for raw in payload["records"]:
        try:
            records.append(ModelInfo.model_validate(raw))
        except ValidationError:
            continue  # one unreadable record, not a dead catalog
    return records


def dump_records(path: Path, records: list[ModelInfo]) -> None:
    """Write records atomically — a torn catalog would be read as empty on the
    next start, silently losing every context window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": SOURCE,
        "records": [record.model_dump(exclude_defaults=True) for record in records],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def default_records() -> list[ModelInfo]:
    """The vendored floor."""
    return load_records(VENDORED_PATH)


def cached_records() -> list[ModelInfo]:
    """A refresh's output, if one has been run on this machine."""
    return load_records(cache_path())
