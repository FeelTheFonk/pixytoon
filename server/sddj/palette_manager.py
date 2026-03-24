"""Palette loading, listing, save/delete, and enforcement utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import settings
from .validation import validate_resource_name as _validate_name

log = logging.getLogger("sddj.palette")

_MAX_PALETTES = 100


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    elif len(h) == 4:
        raise ValueError(f"Palette hex colors must be RGB (#RRGGBB or #RGB), got alpha: #{h!r}")
    elif len(h) == 8:
        h = h[:6]  # Strip alpha channel (RGBA -> RGB)
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: #{h}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def list_palettes() -> list[str]:
    d = settings.palettes_dir
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_palette(name: str) -> list[tuple[int, int, int]]:
    _validate_name(name, "palette")
    path = settings.palettes_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Palette not found: {name}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "colors" not in data or not isinstance(data["colors"], list):
        raise ValueError(f"Palette '{name}' missing 'colors' array")
    colors = [_hex_to_rgb(c) for c in data["colors"]]
    if not colors:
        raise ValueError(f"Palette '{name}' has no colors")
    return colors


def hex_list_to_rgb(colors: list[str]) -> list[tuple[int, int, int]]:
    return [_hex_to_rgb(c) for c in colors]


def save_palette(name: str, colors: list[str]) -> None:
    """Save a named palette to disk. Colors must be hex strings (#RRGGBB or #RGB)."""
    _validate_name(name, "palette")
    if not colors:
        raise ValueError("Palette must contain at least one color")
    # Validate all colors before writing
    for c in colors:
        _hex_to_rgb(c)
    d = settings.palettes_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    if not path.is_file() and len(list(d.glob("*.json"))) >= _MAX_PALETTES:
        raise ValueError(f"Maximum number of palettes ({_MAX_PALETTES}) reached")
    data = {"name": name, "colors": colors}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Palette saved: %s (%d colors)", name, len(colors))


def delete_palette(name: str) -> None:
    """Delete a named palette from disk."""
    _validate_name(name, "palette")
    path = settings.palettes_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Palette not found: {name}")
    path.unlink()
    log.info("Palette deleted: %s", name)
