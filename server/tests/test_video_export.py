"""Tests for video_export — quality presets, metadata sanitization, validation."""

from __future__ import annotations

import pytest
from PIL import Image

from sddj.video_export import (
    QUALITY_PRESETS,
    ExportResult,
    _FRAME_NUM_RE,
    _SAFE_METADATA_RE,
    _detect_digit_width,
    _fill_frame_gaps,
    export_mp4,
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

    def test_matches_5_digit_frame(self):
        """5-digit frame names produced by %05d must match."""
        m = _FRAME_NUM_RE.search("frame_01000.png")
        assert m and m.group(1) == "01000"

    def test_matches_4_digit_frame(self):
        """4-digit numbers (edge case from old %03d overflow) must still match."""
        m = _FRAME_NUM_RE.search("frame_1000.png")
        assert m and m.group(1) == "1000"


class TestDetectDigitWidth:
    def test_3_digit_returns_5(self, tmp_path):
        """Old %03d frames should return 5 (never shrink below minimum)."""
        p = tmp_path / "frame_001.png"
        p.touch()
        assert _detect_digit_width([p]) == 5

    def test_5_digit_returns_5(self, tmp_path):
        """New %05d frames should return 5."""
        p = tmp_path / "frame_00001.png"
        p.touch()
        assert _detect_digit_width([p]) == 5

    def test_6_digit_returns_6(self, tmp_path):
        """If someone has 6-digit frames, respect that."""
        p = tmp_path / "frame_000001.png"
        p.touch()
        assert _detect_digit_width([p]) == 6

    def test_empty_returns_5(self):
        assert _detect_digit_width([]) == 5


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
        assert (tmp_path / "frame_00002.png").exists()

    def test_fills_multiple_gaps(self, tmp_path):
        """Gaps at 002 and 004 are both filled."""
        for i in [1, 3, 5]:
            p = tmp_path / f"frame_{i:03d}.png"
            img = Image.new("RGB", (4, 4), (0, 0, 0))
            img.save(p)
        frames = [tmp_path / f"frame_{i:03d}.png" for i in [1, 3, 5]]
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 5
        assert (tmp_path / "frame_00002.png").exists()
        assert (tmp_path / "frame_00004.png").exists()

    def test_empty_list_returns_empty(self, tmp_path):
        result = _fill_frame_gaps(tmp_path, [])
        assert result == []

    def test_single_frame_returns_single(self, tmp_path):
        p = tmp_path / "frame_001.png"
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        img.save(p)
        result = _fill_frame_gaps(tmp_path, [p])
        assert len(result) == 1

    def test_fills_gap_with_5_digit_names(self, tmp_path):
        """Gaps in 5-digit frame sequences are filled with 5-digit names."""
        for i in [1, 3]:
            p = tmp_path / f"frame_{i:05d}.png"
            img = Image.new("RGB", (4, 4), (0, 0, 0))
            img.save(p)
        frames = [tmp_path / f"frame_{i:05d}.png" for i in [1, 3]]
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 3
        assert (tmp_path / "frame_00002.png").exists()

    def test_fills_gap_above_999(self, tmp_path):
        """Gap-fill works correctly for frame numbers >999."""
        for i in [999, 1001]:
            p = tmp_path / f"frame_{i:05d}.png"
            img = Image.new("RGB", (4, 4), (0, 0, 0))
            img.save(p)
        frames = [tmp_path / f"frame_{i:05d}.png" for i in [999, 1001]]
        result = _fill_frame_gaps(tmp_path, frames)
        assert len(result) == 3
        assert (tmp_path / "frame_01000.png").exists()


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

