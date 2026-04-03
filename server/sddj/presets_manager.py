"""Preset management — CRUD operations on JSON preset files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import settings
from .validation import validate_resource_name

log = logging.getLogger("sddj.presets_manager")


class PresetsManager:
    """Manages generation presets stored as JSON files."""

    def __init__(self, presets_dir: Path) -> None:
        self._dir = presets_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._list_cache: tuple[str, ...] | None = None
        self._list_cache_mtime: float = 0.0  # mtime of presets dir at last cache fill

    def list_presets(self) -> tuple[str, ...]:
        """Return sorted tuple of available preset names (immutable, no copy needed)."""
        # Invalidate cache if the presets directory has been modified externally
        try:
            dir_mtime = self._dir.stat().st_mtime
        except OSError:
            dir_mtime = 0.0
        if self._list_cache is not None and dir_mtime == self._list_cache_mtime:
            return self._list_cache
        self._list_cache = tuple(sorted(p.stem for p in self._dir.glob("*.json")))
        self._list_cache_mtime = dir_mtime
        return self._list_cache

    def get_preset(self, name: str) -> dict:
        """Load and return a preset by name."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset '{name}' not found")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    _MAX_PRESETS = 100

    def save_preset(self, name: str, data: dict) -> None:
        """Save a preset (create or overwrite)."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        if not path.is_file() and len(self.list_presets()) >= self._MAX_PRESETS:
            raise ValueError(f"Maximum number of presets ({self._MAX_PRESETS}) reached")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._list_cache = None
        log.info("Preset saved: %s", name)

    def delete_preset(self, name: str) -> None:
        """Delete a preset file."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset '{name}' not found")
        path.unlink()
        self._list_cache = None
        log.info("Preset deleted: %s", name)


# Module-level singleton
presets_manager = PresetsManager(settings.presets_dir)
