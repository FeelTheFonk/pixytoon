from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ..config import settings


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
