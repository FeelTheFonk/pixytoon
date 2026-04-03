"""Shared input validation utilities."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME = re.compile(r'^[\w\-. ]+$')


_MAX_NAME_LENGTH = 256


def validate_resource_name(name: str, kind: str) -> None:
    """Reject names with path traversal characters or excessive length."""
    if not name or len(name) > _MAX_NAME_LENGTH or not _SAFE_NAME.match(name) or '..' in name:
        raise ValueError(f"Invalid {kind} name: {name!r}")


def validate_path_in_sandbox(resolved: Path, sandbox: Path) -> None:
    """Ensure a resolved path is within the expected sandbox directory.

    Uses Path.is_relative_to() (Python 3.9+) which handles symlinks and
    case-sensitivity correctly, unlike string prefix comparison.
    Rejects symlinks that point outside the sandbox.
    """
    sandbox_resolved = sandbox.resolve()
    path_resolved = resolved.resolve()
    if not path_resolved.is_relative_to(sandbox_resolved):
        raise ValueError(f"Path escapes sandbox: {path_resolved}")
    # Note: symlink targets are already checked by resolve() above —
    # path_resolved follows symlinks, so any symlink pointing outside
    # the sandbox is already caught by the is_relative_to check.
