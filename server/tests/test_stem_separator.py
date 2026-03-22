"""Tests for stem_separator — demucs availability check."""

from __future__ import annotations

import pytest

from sddj.stem_separator import STEM_NAMES, StemSeparator, is_available


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


class TestStemSeparator:
    def test_init(self):
        sep = StemSeparator(model_name="htdemucs", device="cpu")
        assert sep._model_name == "htdemucs"
        assert sep._device == "cpu"

    def test_is_available_method(self):
        sep = StemSeparator()
        assert isinstance(sep.is_available(), bool)

    def test_unload_when_not_loaded(self):
        sep = StemSeparator()
        sep.unload()  # should not raise

    def test_separate_file_not_found(self):
        sep = StemSeparator()
        if not sep.is_available():
            pytest.skip("demucs not installed")
        with pytest.raises(FileNotFoundError):
            sep.separate("/nonexistent/file.wav")


class TestStemNames:
    def test_expected_stems(self):
        assert set(STEM_NAMES) == {"drums", "bass", "vocals", "other"}
