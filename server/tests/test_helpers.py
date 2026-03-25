"""Tests for engine helpers — compute_effective_denoise, make_step_callback, frame helpers."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from sddj.engine.helpers import (
    GenerationCancelled,
    _apply_hue_shift,
    apply_frame_motion,
    apply_noise_injection,
    apply_temporal_coherence,
    compute_effective_denoise,
    make_step_callback,
    scale_steps_for_denoise,
)


# ── compute_effective_denoise ──────────────────────────────


class TestComputeEffectiveDenoise:
    def test_above_floor_returns_identity(self):
        strength, scaled, alpha = compute_effective_denoise(8, 0.5)
        assert strength == pytest.approx(0.5)
        assert alpha == pytest.approx(1.0)
        assert scaled >= 8

    def test_below_floor_clamps_and_sets_alpha(self):
        strength, scaled, alpha = compute_effective_denoise(8, 0.05)
        assert strength > 0.05  # clamped up to floor
        assert alpha < 1.0  # sub-floor alpha
        assert alpha == pytest.approx(0.05 / strength)

    def test_at_one_returns_steps(self):
        strength, scaled, alpha = compute_effective_denoise(8, 1.0)
        assert strength == pytest.approx(1.0)
        assert scaled == 8
        assert alpha == pytest.approx(1.0)

    def test_zero_strength_does_not_crash(self):
        strength, scaled, alpha = compute_effective_denoise(8, 0.0)
        assert strength > 0.0
        assert alpha == pytest.approx(0.0)


# ── make_step_callback ────────────────────────────────────


class TestMakeStepCallback:
    def test_calls_on_progress(self):
        cancel = threading.Event()
        progress = MagicMock()
        cb = make_step_callback(cancel, progress, 10, frame_idx=3, total_frames=8)
        result = cb(None, 4, 500.0, {"latents": None})
        assert result == {"latents": None}
        progress.assert_called_once()
        call_args = progress.call_args[0][0]
        assert call_args.step == 5  # step_idx + 1
        assert call_args.total == 10
        assert call_args.frame_index == 3

    def test_raises_on_cancel(self):
        cancel = threading.Event()
        cancel.set()
        cb = make_step_callback(cancel, None, 10)
        with pytest.raises(GenerationCancelled):
            cb(None, 0, 0.0, {})

    def test_no_progress_callback_ok(self):
        cancel = threading.Event()
        cb = make_step_callback(cancel, None, 10)
        result = cb(None, 0, 0.0, {"x": 1})
        assert result == {"x": 1}


# ── scale_steps_for_denoise ───────────────────────────────


class TestScaleStepsForDenoise:
    def test_full_strength_unchanged(self):
        assert scale_steps_for_denoise(8, 1.0) == 8

    def test_half_strength_doubles(self):
        scaled = scale_steps_for_denoise(8, 0.5)
        assert scaled >= 16

    def test_very_low_strength_capped(self):
        scaled = scale_steps_for_denoise(8, 0.01)
        assert scaled <= 8 * 10  # cap should limit


# ── _apply_hue_shift ──────────────────────────────────────


class TestApplyHueShift:
    def test_no_shift_returns_same(self):
        img = Image.new("RGB", (8, 8), (128, 128, 128))
        result = _apply_hue_shift(img, 0.0)
        assert result == img

    def test_empty_image_safe(self):
        img = Image.new("RGB", (0, 0))
        result = _apply_hue_shift(img, 0.5)
        assert result == img

    def test_preserves_alpha(self):
        img = Image.new("RGBA", (8, 8), (128, 128, 128, 200))
        result = _apply_hue_shift(img, 0.5)
        assert result.mode == "RGBA"
        # Alpha channel preserved
        alpha = list(result.split()[-1].getdata())
        assert all(a == 200 for a in alpha)


# ── apply_temporal_coherence ──────────────────────────────


class TestApplyTemporalCoherence:
    def test_returns_image(self):
        img = Image.new("RGB", (16, 16), (100, 100, 100))
        prev = Image.new("RGB", (16, 16), (200, 200, 200))
        result = apply_temporal_coherence(img, prev)
        assert isinstance(result, Image.Image)
        assert result.size == (16, 16)


# ── apply_frame_motion ────────────────────────────────────


class TestApplyFrameMotion:
    def test_no_motion_returns_same_pixels(self):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        params = {"motion_x": 0.0, "motion_y": 0.0, "motion_zoom": 1.0, "motion_rotation": 0.0}
        result = apply_frame_motion(img, params, 0.5)
        assert np.array_equal(np.array(result), np.array(img))

    def test_empty_params_returns_same_pixels(self):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        result = apply_frame_motion(img, {}, 0.5)
        assert np.array_equal(np.array(result), np.array(img))


# ── apply_noise_injection ─────────────────────────────────


class TestApplyNoiseInjection:
    def test_no_noise_amplitude_no_change(self):
        img = Image.new("RGB", (16, 16), (128, 128, 128))
        params = {"noise_amplitude": 0.0}
        result = apply_noise_injection(img, params, seed=42, denoise_strength=0.5)
        assert np.array_equal(np.array(result), np.array(img))

    def test_with_noise_changes_pixels(self):
        img = Image.new("RGB", (16, 16), (128, 128, 128))
        params = {"noise_amplitude": 0.5}
        result = apply_noise_injection(img, params, seed=42, denoise_strength=0.5)
        assert not np.array_equal(np.array(result), np.array(img))

    def test_deterministic_with_seed(self):
        img = Image.new("RGB", (16, 16), (128, 128, 128))
        params = {"noise_amplitude": 0.3}
        r1 = apply_noise_injection(img, params, seed=42, denoise_strength=0.5)
        r2 = apply_noise_injection(img, params, seed=42, denoise_strength=0.5)
        assert np.array_equal(np.array(r1), np.array(r2))
