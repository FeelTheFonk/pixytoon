"""Tests for ResourceManager — unified resource discovery."""

from __future__ import annotations

import pytest
from pathlib import Path

from sddj.resource_manager import ResourceManager


@pytest.fixture
def resource_dir(tmp_path: Path) -> Path:
    """Create a temp directory with some fake model files."""
    (tmp_path / "model_a.safetensors").write_bytes(b"fake")
    (tmp_path / "model_b.bin").write_bytes(b"fake")
    (tmp_path / "model_c.pt").write_bytes(b"fake")
    (tmp_path / "readme.txt").write_text("not a model")
    (tmp_path / "subdir").mkdir()
    return tmp_path


class TestResourceManagerList:
    def test_lists_matching_extensions(self, resource_dir: Path):
        mgr = ResourceManager("test", resource_dir)
        result = mgr.list()
        assert result == ["model_a", "model_b", "model_c"]

    def test_empty_directory(self, tmp_path: Path):
        mgr = ResourceManager("test", tmp_path)
        assert mgr.list() == []

    def test_nonexistent_directory(self, tmp_path: Path):
        mgr = ResourceManager("test", tmp_path / "nope")
        assert mgr.list() == []

    def test_custom_extensions(self, resource_dir: Path):
        mgr = ResourceManager("test", resource_dir, extensions={".txt"})
        assert mgr.list() == ["readme"]


class TestResourceManagerResolve:
    def test_resolves_existing(self, resource_dir: Path):
        mgr = ResourceManager("test", resource_dir)
        path = mgr.resolve("model_a")
        assert path.name == "model_a.safetensors"

    def test_not_found_raises(self, resource_dir: Path):
        mgr = ResourceManager("test", resource_dir)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            mgr.resolve("nonexistent")

    def test_path_traversal_rejected(self, resource_dir: Path):
        mgr = ResourceManager("test", resource_dir)
        with pytest.raises(ValueError):
            mgr.resolve("../etc/passwd")
