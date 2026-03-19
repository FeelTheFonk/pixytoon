"""Palette loading, listing, and enforcement utilities."""

from __future__ import annotations

import json
from pathlib import Path

from .config import settings


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: #{h}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def list_palettes() -> list[str]:
    d = settings.palettes_dir
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_palette(name: str) -> list[tuple[int, int, int]]:
    path = settings.palettes_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Palette not found: {name}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "colors" not in data or not isinstance(data["colors"], list):
        raise ValueError(f"Palette '{name}' missing 'colors' array")
    return [_hex_to_rgb(c) for c in data["colors"]]


def hex_list_to_rgb(colors: list[str]) -> list[tuple[int, int, int]]:
    return [_hex_to_rgb(c) for c in colors]
