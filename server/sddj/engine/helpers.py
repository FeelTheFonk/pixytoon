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
