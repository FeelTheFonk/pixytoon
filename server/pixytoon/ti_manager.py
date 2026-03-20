"""Textual Inversion embedding discovery and resolution."""

from __future__ import annotations

import re
from pathlib import Path

from .config import settings

_SAFE_NAME = re.compile(r'^[\w\-. ]+$')


def _validate_name(name: str, kind: str) -> None:
    """Reject names with path traversal characters."""
    if not name or not _SAFE_NAME.match(name) or '..' in name:
        raise ValueError(f"Invalid {kind} name: {name!r}")


_TI_EXTENSIONS = {".safetensors", ".bin", ".pt"}


def list_embeddings() -> list[str]:
    d = settings.embeddings_dir
    if not d.is_dir():
        return []
    return sorted(
        p.stem
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in _TI_EXTENSIONS
    )


def resolve_embedding_path(name: str) -> Path:
    _validate_name(name, "embedding")
    d = settings.embeddings_dir
    for ext in _TI_EXTENSIONS:
        candidate = d / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Embedding '{name}' not found in {d}. "
        f"Available: {list_embeddings()}"
    )
