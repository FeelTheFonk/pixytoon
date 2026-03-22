"""Tests for video_export — quality presets, metadata sanitization, validation."""

from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image

from sddj.video_export import (
    QUALITY_PRESETS,
    ExportResult,
    _FRAME_NUM_RE,
    _SAFE_METADATA_RE,
    _fill_frame_gaps,
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


class TestFrameNumRegex:
    def test_matches_standard_frame(self):
        m = _FRAME_NUM_RE.search("frame_001.png")
        assert m and m.group(1) == "001"

    def test_matches_high_number(self):
        m = _FRAME_NUM_RE.search("frame_120.png")
        assert m and m.group(1) == "120"

    def test_no_match_non_frame(self):
        assert _FRAME_NUM_RE.search("other_file.png") is None


class TestFillFrameGaps:
    def test_no_gaps_unchanged(self, tmp_path):
        """Continuous sequence returns same files."""
        frames = []
        for i in range(1, 4):
            p = tmp_path / f"frame_{i:03d}.png"
            img = Image.new("RGB", (4, 4), (i * 50, 0, 0))
            img.save(p)
            frames.append(p)
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 3

    def test_fills_single_gap(self, tmp_path):
        """Missing frame_002 between 001 and 003 is filled."""
        for i in [1, 3]:
            p = tmp_path / f"frame_{i:03d}.png"
            img = Image.new("RGB", (4, 4), (i * 50, 0, 0))
            img.save(p)
        frames = [tmp_path / "frame_001.png", tmp_path / "frame_003.png"]
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 3
        assert (tmp_path / "frame_002.png").exists()

    def test_fills_multiple_gaps(self, tmp_path):
        """Gaps at 002 and 004 are both filled."""
        for i in [1, 3, 5]:
            p = tmp_path / f"frame_{i:03d}.png"
            img = Image.new("RGB", (4, 4), (0, 0, 0))
            img.save(p)
        frames = [tmp_path / f"frame_{i:03d}.png" for i in [1, 3, 5]]
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 5
        assert (tmp_path / "frame_002.png").exists()
        assert (tmp_path / "frame_004.png").exists()

    def test_empty_list_returns_empty(self, tmp_path):
        result = _fill_frame_gaps(tmp_path, [])
        assert result == []

    def test_single_frame_returns_single(self, tmp_path):
        p = tmp_path / "frame_001.png"
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        img.save(p)
        result = _fill_frame_gaps(tmp_path, [p])
        assert len(result) == 1


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
