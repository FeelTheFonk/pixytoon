"""Generic resource discovery — shared by LoRA, embedding, and similar managers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .validation import validate_resource_name as _validate_name


_DEFAULT_EXTENSIONS = frozenset({".safetensors", ".bin", ".pt"})


class ResourceManager:
    """Discover and resolve model resources from a directory.

    Eliminates the need for separate manager modules for each resource type
    (LoRA, embeddings, etc.) — all share the same list/resolve pattern.
    """

    __slots__ = ("_kind", "_directory", "_extensions")

    def __init__(
        self,
        kind: str,
        directory: Path,
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    ) -> None:
        self._kind = kind
        self._directory = directory
        self._extensions = frozenset(extensions)

    def list(self) -> list[str]:
        d = self._directory
        if not d.is_dir():
            return []
        try:
            return sorted(
                p.stem
                for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in self._extensions
            )
        except OSError:
            return []

    def resolve(self, name: str) -> Path:
        _validate_name(name, self._kind)
        d = self._directory
        for ext in self._extensions:
            candidate = d / f"{name}{ext}"
            if candidate.is_file():
                resolved = candidate.resolve()
                if not str(resolved).startswith(str(d.resolve())):
                    raise ValueError(f"{self._kind} path escapes directory: {resolved}")
                return candidate
        raise FileNotFoundError(
            f"{self._kind} '{name}' not found in {d}. "
            f"Available: {self.list()}"
        )
