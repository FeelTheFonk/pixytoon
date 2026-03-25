"""Tests for presets manager — CRUD, validation, edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from sddj.presets_manager import PresetsManager


class TestPresetsManager:
    def test_list_presets_with_defaults(self, tmp_presets_dir: Path):
        mgr = PresetsManager(tmp_presets_dir)
        names = mgr.list_presets()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "pixel_art" in names

    def test_list_empty(self, empty_presets_dir: Path):
        mgr = PresetsManager(empty_presets_dir)
        assert mgr.list_presets() == []

    def test_get_preset(self, tmp_presets_dir: Path):
        mgr = PresetsManager(tmp_presets_dir)
        data = mgr.get_preset("pixel_art")
        assert isinstance(data, dict)
        assert "prompt_prefix" in data

    def test_get_nonexistent(self, tmp_presets_dir: Path):
        mgr = PresetsManager(tmp_presets_dir)
        with pytest.raises(FileNotFoundError):
            mgr.get_preset("nonexistent_preset_xyz")

    def test_save_preset(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        mgr.save_preset("my_preset", sample_preset_data)
        assert "my_preset" in mgr.list_presets()
        loaded = mgr.get_preset("my_preset")
        assert loaded["steps"] == sample_preset_data["steps"]

    def test_overwrite_preset(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        mgr.save_preset("test", sample_preset_data)
        new_data = {**sample_preset_data, "steps": 20}
        mgr.save_preset("test", new_data)
        loaded = mgr.get_preset("test")
        assert loaded["steps"] == 20

    def test_delete_preset(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        mgr.save_preset("deleteme", sample_preset_data)
        assert "deleteme" in mgr.list_presets()
        mgr.delete_preset("deleteme")
        assert "deleteme" not in mgr.list_presets()

    def test_delete_nonexistent(self, empty_presets_dir: Path):
        mgr = PresetsManager(empty_presets_dir)
        with pytest.raises(FileNotFoundError):
            mgr.delete_preset("ghost")

    def test_path_traversal_rejected(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        with pytest.raises(ValueError):
            mgr.get_preset("../etc/passwd")
        with pytest.raises(ValueError):
            mgr.save_preset("../../evil", sample_preset_data)
        with pytest.raises(ValueError):
            mgr.delete_preset("..\\..\\bad")

    def test_empty_name_rejected(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        with pytest.raises(ValueError):
            mgr.save_preset("", sample_preset_data)

    def test_special_chars_rejected(self, empty_presets_dir: Path, sample_preset_data: dict):
        mgr = PresetsManager(empty_presets_dir)
        with pytest.raises(ValueError):
            mgr.save_preset("bad/name", sample_preset_data)
        with pytest.raises(ValueError):
            mgr.save_preset("bad:name", sample_preset_data)

    def test_creates_dir_if_missing(self, tmp_path: Path, sample_preset_data: dict):
        new_dir = tmp_path / "auto_created" / "presets"
        mgr = PresetsManager(new_dir)
        assert new_dir.is_dir()
        mgr.save_preset("test", sample_preset_data)
        assert "test" in mgr.list_presets()

    def test_max_presets_limit(self, empty_presets_dir: Path, sample_preset_data: dict):
        """v0.7.9: reject save when preset count reaches _MAX_PRESETS."""
        mgr = PresetsManager(empty_presets_dir)
        mgr._MAX_PRESETS = 3  # Override for test
        mgr.save_preset("p1", sample_preset_data)
        mgr.save_preset("p2", sample_preset_data)
        mgr.save_preset("p3", sample_preset_data)
        with pytest.raises(ValueError, match="Maximum"):
            mgr.save_preset("p4", sample_preset_data)
        # Overwriting existing should still work
        mgr.save_preset("p1", {**sample_preset_data, "steps": 99})
