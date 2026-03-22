"""Tests for video_export — quality presets, metadata sanitization, validation."""

from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image

from pixytoon.video_export import (
    QUALITY_PRESETS,
    ExportResult,
    _SAFE_METADATA_RE,
    export_mp4,
    find_ffmpeg,
)


class TestQualityPresets:
    def test_all_presets_exist(self):
        expected = {"web", "high", "archive", "raw"}
        assert set(QUALITY_PRESETS.keys()) == expected

    def test_presets_have_valid_crf(self):
        for name, (crf, preset, scale) in QUALITY_PRESETS.items():
            assert 0 <= crf <= 51, f"{name}: CRF {crf} out of range"

    def test_presets_have_positive_scale(self):
        for name, (crf, preset, scale) in QUALITY_PRESETS.items():
            assert scale >= 1, f"{name}: scale {scale} < 1"


class TestMetadataSanitization:
    def test_safe_string_unchanged(self):
        safe = "pixel art, seed=42, steps=8"
        assert _SAFE_METADATA_RE.sub("", safe) == safe

    def test_dangerous_chars_removed(self):
        dangerous = "test\x00\x01\x02\x03"
        cleaned = _SAFE_METADATA_RE.sub("", dangerous)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned


class TestExportResult:
    def test_dataclass_fields(self):
        r = ExportResult(path="/tmp/video.mp4", size_mb=12.5, duration_s=5.0)
        assert r.path == "/tmp/video.mp4"
        assert r.size_mb == 12.5
        assert r.duration_s == 5.0


class TestExportValidation:
    def test_missing_frame_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Frame directory not found"):
            export_mp4(tmp_path / "nonexistent", None)

    def test_empty_frame_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No frame_.*png files"):
            export_mp4(empty_dir, None)

    def test_invalid_quality_raises(self, tmp_path):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        # Create a dummy frame
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        img.save(frame_dir / "frame_001.png")
        with pytest.raises(ValueError, match="Unknown quality preset"):
            export_mp4(frame_dir, None, quality="superduper")

    def test_ffmpeg_not_found_raises(self, tmp_path):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        img.save(frame_dir / "frame_001.png")
        with pytest.raises((FileNotFoundError, RuntimeError)):
            export_mp4(frame_dir, None, ffmpeg_path="/nonexistent/ffmpeg")
