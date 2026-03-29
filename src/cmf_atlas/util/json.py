"""JSON helpers for payload serialization."""

import json
from typing import Any


def dumps(obj: Any) -> str:
    """Compact, deterministic JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def loads(s: str) -> Any:
    """Parse JSON string."""
    return json.loads(s)


def pretty(obj: Any) -> str:
    """Pretty-printed JSON."""
    return json.dumps(obj, sort_keys=True, indent=2, default=str)
