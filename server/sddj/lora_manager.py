"""LoRA discovery and listing."""

from __future__ import annotations

from pathlib import Path

from .config import settings
from .validation import validate_resource_name as _validate_name


_LORA_EXTENSIONS = {".safetensors", ".bin", ".pt"}


def list_loras() -> list[str]:
    d = settings.loras_dir
    if not d.is_dir():
        return []
    try:
        return sorted(
            p.stem
            for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in _LORA_EXTENSIONS
        )
    except OSError:
        return []


def resolve_lora_path(name: str) -> Path:
    _validate_name(name, "LoRA")
    d = settings.loras_dir
    for ext in _LORA_EXTENSIONS:
        candidate = d / f"{name}{ext}"
        if candidate.is_file():
            # Path traversal guard: resolved path must stay inside loras_dir
            resolved = candidate.resolve()
            if not str(resolved).startswith(str(d.resolve())):
                raise ValueError(f"LoRA path escapes loras_dir: {resolved}")
            return candidate
    raise FileNotFoundError(
        f"LoRA '{name}' not found in {d}. "
        f"Available: {list_loras()}"
    )
