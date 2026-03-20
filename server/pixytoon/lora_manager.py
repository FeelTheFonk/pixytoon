"""LoRA discovery and listing."""

from __future__ import annotations

import re
from pathlib import Path

from .config import settings

_SAFE_NAME = re.compile(r'^[\w\-. ]+$')


def _validate_name(name: str, kind: str) -> None:
    """Reject names with path traversal characters."""
    if not name or not _SAFE_NAME.match(name) or '..' in name:
        raise ValueError(f"Invalid {kind} name: {name!r}")


_LORA_EXTENSIONS = {".safetensors", ".bin", ".pt"}


def list_loras() -> list[str]:
    d = settings.loras_dir
    if not d.is_dir():
        return []
    return sorted(
        p.stem
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in _LORA_EXTENSIONS
    )


def resolve_lora_path(name: str) -> Path:
    _validate_name(name, "LoRA")
    d = settings.loras_dir
    for ext in _LORA_EXTENSIONS:
        candidate = d / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"LoRA '{name}' not found in {d}. "
        f"Available: {list_loras()}"
    )
