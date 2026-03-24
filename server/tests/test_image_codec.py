"""Tests for image encode/decode/resize utilities."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import pytest
from PIL import Image

from sddj.image_codec import (
    apply_motion_warp,
    composite_with_mask,
    decode_b64_image,
    decode_b64_mask,
    encode_image_b64,
    resize_to_target,
    round8,
)


class TestRound8:
    @pytest.mark.parametrize("inp,expected", [
        (8, 8), (9, 8), (12, 16), (16, 16),
        (64, 64), (100, 104), (128, 128),
        (511, 512), (512, 512), (513, 512),
    ])
    def test_round8(self, inp, expected):
        assert round8(inp) == expected

    def test_round8_zero(self):
        assert round8(0) == 8  # Minimum clamp: prevents 0-dimension VAE tensors

    def test_round8_small_clamps_to_8(self):
        assert round8(1) == 8
        assert round8(3) == 8
        assert round8(7) == 8


class TestDecodeEncodeRoundtrip:
    def _make_b64(self, img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return b64encode(buf.getvalue()).decode("ascii")

    def test_roundtrip_rgb(self):
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        b64 = self._make_b64(img)
        decoded = decode_b64_image(b64)
        assert decoded.size == (64, 64)
        assert decoded.mode in ("RGB", "RGBA")

    def test_roundtrip_rgba(self):
        img = Image.new("RGBA", (32, 32), (0, 255, 0, 128))
        b64 = self._make_b64(img)
        decoded = decode_b64_image(b64)
        assert decoded.mode == "RGBA"

    def test_encode_then_decode(self):
        original = Image.new("RGBA", (48, 48), (100, 200, 50, 255))
        b64 = encode_image_b64(original)
        decoded = decode_b64_image(b64)
        assert decoded.size == original.size

    def test_invalid_b64(self):
        with pytest.raises((ValueError, Exception)):
            decode_b64_image("not-valid-base64!!!")

    def test_palette_mode_converted(self):
        img = Image.new("P", (32, 32))
        b64 = self._make_b64(img)
        decoded = decode_b64_image(b64)
        assert decoded.mode == "RGBA"

    def test_grayscale_converted(self):
        img = Image.new("L", (32, 32), 128)
        b64 = self._make_b64(img)
        decoded = decode_b64_image(b64)
        assert decoded.mode == "RGB"


class TestDecodeMask:
    def test_mask_to_grayscale(self):
        mask = Image.new("RGB", (64, 64), (255, 255, 255))
        buf = BytesIO()
        mask.save(buf, format="PNG")
        b64 = b64encode(buf.getvalue()).decode("ascii")
        decoded = decode_b64_mask(b64)
        assert decoded.mode == "L"
        assert decoded.size == (64, 64)


class TestResizeToTarget:
    def test_same_size_no_op(self):
        img = Image.new("RGB", (512, 512))
        result = resize_to_target(img, 512, 512)
        assert result is img  # Same object

    def test_resize_down(self):
        img = Image.new("RGB", (512, 512))
        result = resize_to_target(img, 256, 256)
        assert result.size == (256, 256)

    def test_resize_up(self):
        img = Image.new("RGB", (64, 64))
        result = resize_to_target(img, 512, 512)
        assert result.size == (512, 512)


class TestCompositeWithMask:
    def test_basic_composite(self):
        original = Image.new("RGB", (64, 64), (255, 0, 0))
        inpainted = Image.new("RGB", (64, 64), (0, 0, 255))
        mask = Image.new("L", (64, 64), 255)
        result = composite_with_mask(original, inpainted, mask)
        assert result.size == (64, 64)
        # Full white mask = all inpainted
        px = result.getpixel((32, 32))
        assert px == (0, 0, 255)

    def test_black_mask_keeps_original(self):
        original = Image.new("RGB", (64, 64), (255, 0, 0))
        inpainted = Image.new("RGB", (64, 64), (0, 0, 255))
        mask = Image.new("L", (64, 64), 0)
        result = composite_with_mask(original, inpainted, mask)
        px = result.getpixel((32, 32))
        assert px == (255, 0, 0)

    def test_size_mismatch_raises(self):
        original = Image.new("RGB", (64, 64))
        inpainted = Image.new("RGB", (32, 32))
        mask = Image.new("L", (64, 64), 255)
        with pytest.raises(ValueError, match="Size mismatch"):
            composite_with_mask(original, inpainted, mask)

    def test_mask_resized_if_different(self):
        original = Image.new("RGB", (64, 64), (255, 0, 0))
        inpainted = Image.new("RGB", (64, 64), (0, 255, 0))
        mask = Image.new("L", (32, 32), 255)
        result = composite_with_mask(original, inpainted, mask)
        assert result.size == (64, 64)


class TestApplyMotionWarp:
    """v0.7.4: Deforum-like smooth 2D affine warp with denoise correlation."""

    def _make_image(self, mode="RGB", size=(64, 64)):
        import numpy as np
        arr = np.random.randint(0, 255, (*size[::-1], 3 if mode == "RGB" else 4), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_identity_no_motion(self):
        img = self._make_image()
        result = apply_motion_warp(img, tx=0, ty=0, zoom=1.0, rotation=0.0)
        assert result is img  # No warp needed → same object returned

    def test_negligible_motion_returns_original(self):
        img = self._make_image()
        result = apply_motion_warp(img, tx=0.001, ty=0.001, zoom=1.0, rotation=0.0)
        assert result is img  # Below threshold

    def test_translation_changes_image(self):
        img = self._make_image()
        import numpy as np
        original_arr = np.array(img)
        result = apply_motion_warp(img, tx=5.0, ty=0.0, denoise_strength=0.8)
        result_arr = np.array(result)
        assert result.size == img.size
        assert not np.array_equal(original_arr, result_arr)

    def test_zoom_changes_image(self):
        img = self._make_image()
        import numpy as np
        original_arr = np.array(img)
        result = apply_motion_warp(img, zoom=1.05, denoise_strength=0.8)
        assert result.size == img.size
        assert not np.array_equal(original_arr, np.array(result))

    def test_rotation_changes_image(self):
        img = self._make_image()
        import numpy as np
        result = apply_motion_warp(img, rotation=2.0, denoise_strength=0.8)
        assert result.size == img.size
        assert not np.array_equal(np.array(img), np.array(result))

    def test_denoise_correlation_scales_motion(self):
        """Higher denoise = more effective motion. Low denoise = dampened."""
        img = self._make_image()
        import numpy as np
        # Same tx, different denoise strengths
        low_denoise = apply_motion_warp(img, tx=5.0, denoise_strength=0.1)
        high_denoise = apply_motion_warp(img, tx=5.0, denoise_strength=0.8)
        # Both produce different images, but different from each other
        assert not np.array_equal(np.array(low_denoise), np.array(high_denoise))

    def test_preserves_rgba_mode(self):
        img = self._make_image(mode="RGBA")
        result = apply_motion_warp(img, tx=3.0, denoise_strength=0.5)
        assert result.mode == "RGBA"
        assert result.size == img.size

    def test_preserves_rgb_mode(self):
        img = self._make_image(mode="RGB")
        result = apply_motion_warp(img, tx=3.0, denoise_strength=0.5)
        assert result.mode == "RGB"
