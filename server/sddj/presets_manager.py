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

    def list_presets(self) -> list[str]:
        """Return sorted list of available preset names."""
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def get_preset(self, name: str) -> dict:
        """Load and return a preset by name."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset '{name}' not found")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_preset(self, name: str, data: dict) -> None:
        """Save a preset (create or overwrite)."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Preset saved: %s", name)

    def delete_preset(self, name: str) -> None:
        """Delete a preset file."""
        validate_resource_name(name, "preset")
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Preset '{name}' not found")
        path.unlink()
        log.info("Preset deleted: %s", name)


# Module-level singleton
presets_manager = PresetsManager(settings.presets_dir)
