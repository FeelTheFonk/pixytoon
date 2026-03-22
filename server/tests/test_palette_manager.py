"""Tests for palette manager — list, load, hex conversion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sddj.palette_manager import _hex_to_rgb, hex_list_to_rgb, list_palettes, load_palette


class TestHexToRgb:
    @pytest.mark.parametrize("hex_str,expected", [
        ("#FF0000", (255, 0, 0)),
        ("#00ff00", (0, 255, 0)),
        ("0000FF", (0, 0, 255)),
        ("#fff", (255, 255, 255)),
        ("#000", (0, 0, 0)),
        ("#FF0000FF", (255, 0, 0)),  # RGBA -> RGB
    ])
    def test_valid_hex(self, hex_str, expected):
        assert _hex_to_rgb(hex_str) == expected

    def test_invalid_hex(self):
        with pytest.raises(ValueError):
            _hex_to_rgb("XYZXYZ")

    def test_hex_list_to_rgb(self):
        colors = hex_list_to_rgb(["#FF0000", "#00FF00"])
        assert colors == [(255, 0, 0), (0, 255, 0)]


class TestListPalettes:
    def test_list_with_palettes(self, tmp_palettes_dir: Path):
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            result = list_palettes()
            assert "pico8" in result

    def test_list_nonexistent_dir(self, tmp_path: Path):
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_path / "ghost"
            result = list_palettes()
            assert result == []


class TestLoadPalette:
    def test_load_valid(self, tmp_palettes_dir: Path):
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            colors = load_palette("pico8")
            assert len(colors) == 16
            assert colors[0] == (0, 0, 0)

    def test_load_nonexistent(self, tmp_palettes_dir: Path):
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            with pytest.raises(FileNotFoundError):
                load_palette("ghost_palette")

    def test_load_missing_colors_key(self, tmp_palettes_dir: Path):
        (tmp_palettes_dir / "bad.json").write_text(json.dumps({"name": "bad"}))
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            with pytest.raises(ValueError, match="missing 'colors'"):
                load_palette("bad")

    def test_load_empty_colors(self, tmp_palettes_dir: Path):
        (tmp_palettes_dir / "empty.json").write_text(json.dumps({"colors": []}))
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            with pytest.raises(ValueError, match="no colors"):
                load_palette("empty")

    def test_path_traversal_rejected(self, tmp_palettes_dir: Path):
        with patch("sddj.palette_manager.settings") as mock_s:
            mock_s.palettes_dir = tmp_palettes_dir
            with pytest.raises(ValueError):
                load_palette("../etc/passwd")
