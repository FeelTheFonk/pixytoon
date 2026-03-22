"""Tests for input validation utilities."""

from __future__ import annotations

import pytest

from sddj.validation import validate_resource_name


class TestValidateResourceName:
    @pytest.mark.parametrize("name", [
        "pixel_art", "my-preset", "Test Preset", "lora_v1.2",
        "a", "CamelCase", "with spaces",
    ])
    def test_valid_names(self, name):
        validate_resource_name(name, "test")  # Should not raise

    @pytest.mark.parametrize("name", [
        "", "../evil", "..\\bad", "../../etc/passwd",
        "bad/name", "bad:name", "bad\x00name",
    ])
    def test_invalid_names(self, name):
        with pytest.raises(ValueError):
            validate_resource_name(name, "test")

    def test_too_long_name(self):
        with pytest.raises(ValueError):
            validate_resource_name("a" * 257, "test")

    def test_max_length_ok(self):
        validate_resource_name("a" * 256, "test")  # Should not raise

    def test_dotdot_in_name(self):
        with pytest.raises(ValueError):
            validate_resource_name("foo..bar", "test")
