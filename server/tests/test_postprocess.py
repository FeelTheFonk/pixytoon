"""Tests for post-processing pipeline — pixelation, quantize, palette, dither."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sddj.postprocess import apply
from sddj.protocol import (
    DitherMode,
    PaletteMode,
    PaletteSpec,
    PixelateSpec,
    PostProcessSpec,
    QuantizeMethod,
)


def _make_test_image(w=128, h=128, mode="RGBA"):
    """Create a colorful test image."""
    arr = np.random.randint(0, 255, (h, w, 4 if mode == "RGBA" else 3), dtype=np.uint8)
    if mode == "RGBA":
        arr[:, :, 3] = 255
    return Image.fromarray(arr)


class TestPixelation:
    def test_pixelate_enabled(self):
        np.random.seed(42)
        img = _make_test_image(512, 512)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_pixelate_disabled(self):
        np.random.seed(42)
        img = _make_test_image(128, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (128, 128)

    def test_pixelate_rectangular(self):
        np.random.seed(42)
        img = _make_test_image(256, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        w, h = result.size
        assert max(w, h) == 64


class TestQuantization:
    def test_kmeans_reduces_colors(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_method=QuantizeMethod.KMEANS,
            quantize_colors=8,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        colors = set()
        for px in result.getdata():
            colors.add(px[:3])
        assert len(colors) <= 10  # Allow some slack for alpha compositing

    def test_octree_method(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_method=QuantizeMethod.OCTREE,
            quantize_colors=16,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_median_cut_method(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_method=QuantizeMethod.MEDIAN_CUT,
            quantize_colors=16,
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)


class TestDithering:
    def test_floyd_steinberg(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_colors=8,
            dither=DitherMode.FLOYD_STEINBERG,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_bayer_4x4(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            quantize_colors=8,
            dither=DitherMode.BAYER_4X4,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_no_dither(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)


class TestPalette:
    def test_auto_palette(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)

    def test_custom_palette(self):
        np.random.seed(42)
        img = _make_test_image(64, 64)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            palette=PaletteSpec(
                mode=PaletteMode.CUSTOM,
                colors=["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000"],
            ),
            dither=DitherMode.NONE,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)


class TestFullPipeline:
    def test_complete_postprocess_pipeline(self):
        np.random.seed(42)
        img = _make_test_image(512, 512)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=True, target_size=64),
            quantize_method=QuantizeMethod.KMEANS,
            quantize_colors=16,
            dither=DitherMode.FLOYD_STEINBERG,
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result.size == (64, 64)
        assert result.mode == "RGBA"

    def test_passthrough_no_processing(self):
        np.random.seed(42)
        img = _make_test_image(128, 128)
        spec = PostProcessSpec(
            pixelate=PixelateSpec(enabled=False),
            dither=DitherMode.NONE,
            palette=PaletteSpec(mode=PaletteMode.AUTO),
            remove_bg=False,
        )
        result = apply(img, spec)
        assert result.size == (128, 128)
