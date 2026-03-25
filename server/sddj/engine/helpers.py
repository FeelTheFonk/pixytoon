from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ..config import settings
from ..protocol import ProgressResponse


class GenerationCancelled(Exception):
    """Raised when a client cancels an in-progress generation."""


def _apply_hue_shift(image: Image.Image, shift: float) -> Image.Image:
    """Shift hue of an image by `shift` fraction (0-1 maps to 0-360 degrees).

    Preserves alpha channel. Used for audio-driven palette shift modulation.
    """
    if image.size[0] == 0 or image.size[1] == 0:
        return image
    if abs(shift) < 1e-6:
        return image
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h, dtype=np.int16)
    h_arr = (h_arr + int(shift * 255)) % 256
    h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    result = Image.merge("HSV", (h, s, v)).convert("RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result


def scale_steps_for_denoise(steps: int, strength: float) -> int:
    """Scale num_inference_steps so effective denoising steps ≈ requested steps.

    In img2img, diffusers computes: effective = int(steps * strength).
    When strength < 1.0, fewer steps run, degrading quality.
    We compensate by scaling the schedule length: ceil(steps / strength),
    guaranteeing ~`steps` effective denoising passes regardless of strength.

    A cap (``settings.distilled_step_scale_cap``) limits the multiplier to
    avoid wasting compute on distilled models (Hyper-SD) that converge in
    their trained step count.
    """
    if strength >= 1.0:
        return steps
    strength = max(strength, 0.01)  # safety floor
    scaled = math.ceil(steps / strength)
    # Cap scaling for distilled models (Hyper-SD)
    cap = settings.distilled_step_scale_cap
    if cap > 0:
        scaled = min(scaled, steps * cap)
    return max(steps, scaled)


def compute_effective_denoise(
    steps: int, strength: float,
) -> tuple[float, int, float]:
    """Compute effective denoise params with sub-floor blending.

    Returns (effective_strength, scaled_steps, sub_floor_alpha).
    When sub_floor_alpha < 1.0, the result should be alpha-blended toward
    the source image for sub-floor attenuation without quality loss.

    Guarantees ≥2 effective denoising steps while preserving full
    audio/parameter dynamic range.
    """
    cap = max(settings.distilled_step_scale_cap, 1)
    min_denoise = min(1.0, 2.0 / max(steps * cap, 1) + 1e-3)
    sub_floor_alpha = 1.0

    if strength < min_denoise:
        sub_floor_alpha = strength / min_denoise
        effective_strength = min_denoise
    else:
        effective_strength = min(1.0, strength)

    scaled_steps = scale_steps_for_denoise(steps, effective_strength)
    return effective_strength, scaled_steps, sub_floor_alpha


def make_step_callback(cancel_event, on_progress, total_steps,
                       frame_idx=None, total_frames=None):
    """Factory for diffusers callback_on_step_end with cancellation support."""
    def _callback(pipe, step_idx, timestep, callback_kwargs):
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if on_progress:
            on_progress(ProgressResponse(
                step=step_idx + 1, total=total_steps,
                frame_index=frame_idx, total_frames=total_frames,
            ))
        return callback_kwargs
    return _callback


# ── Shared frame-processing helpers ────────────────────────


def apply_temporal_coherence(
    image: Image.Image,
    prev_image: Image.Image,
) -> Image.Image:
    """Apply color coherence + optical flow blending between consecutive frames.

    Uses ``settings.color_coherence_strength`` and ``settings.optical_flow_blend``
    to control intensity.  Both are no-ops when their setting is 0.
    """
    from ..image_codec import match_color_lab, apply_optical_flow_blend
    if settings.color_coherence_strength > 0:
        image = match_color_lab(image, prev_image, settings.color_coherence_strength)
    if settings.optical_flow_blend > 0:
        image = apply_optical_flow_blend(image, prev_image, settings.optical_flow_blend)
    return image


def apply_frame_motion(
    image: Image.Image,
    frame_params: dict[str, float],
    denoise_strength: float,
) -> Image.Image:
    """Apply 2D affine motion warp + perspective tilt from modulation params.

    Applies motion_x/y/zoom/rotation (2D affine) first, then
    tilt_x/tilt_y (perspective) — matching Deforum ordering.
    """
    from ..image_codec import apply_motion_warp, apply_perspective_tilt

    mx = frame_params.get("motion_x", 0.0)
    my = frame_params.get("motion_y", 0.0)
    mz = frame_params.get("motion_zoom", 1.0)
    mr = frame_params.get("motion_rotation", 0.0)
    if abs(mx) > 0.01 or abs(my) > 0.01 or abs(mz - 1.0) > 0.001 or abs(mr) > 0.01:
        image = apply_motion_warp(
            image, tx=mx, ty=my, zoom=mz, rotation=mr,
            denoise_strength=denoise_strength,
        )

    mtx = frame_params.get("motion_tilt_x", 0.0)
    mty = frame_params.get("motion_tilt_y", 0.0)
    if abs(mtx) > 0.01 or abs(mty) > 0.01:
        image = apply_perspective_tilt(
            image, tilt_x=mtx, tilt_y=mty,
            denoise_strength=denoise_strength,
        )
    return image


def apply_noise_injection(
    image: Image.Image,
    frame_params: dict[str, float],
    seed: int,
    denoise_strength: float,
) -> Image.Image:
    """Inject noise into a frame image for temporal variation.

    Auto-coupling: when no ``noise_amplitude`` slot is active, injects subtle
    noise inversely proportional to denoise strength — gated at 0.35 to prevent
    artifact accumulation at low denoise values.
    """
    noise_amp = max(0.0, min(1.0, frame_params.get("noise_amplitude", 0.0)))
    if settings.auto_noise_coupling and "noise_amplitude" not in frame_params:
        if denoise_strength >= 0.35:
            noise_amp = max(0.0, (0.9 - denoise_strength) * 0.1)
        else:
            noise_amp = 0.0
    if noise_amp > 0:
        arr = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.default_rng(seed).standard_normal(
            arr.shape, dtype=np.float32) * noise_amp
        arr = np.clip(arr + noise, 0.0, 1.0)
        image = Image.fromarray((arr * 255).astype(np.uint8))
    return image

