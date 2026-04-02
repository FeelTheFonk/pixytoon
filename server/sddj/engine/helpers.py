from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..config import settings
from ..embedding_blend import blend_prompt_embeds
from ..image_codec import match_color_lab, apply_optical_flow_blend, apply_motion_warp, apply_perspective_tilt
from ..protocol import ProgressResponse

log = logging.getLogger("sddj.engine")


def build_prompt_schedule(req) -> "PromptSchedule | None":
    """Build a PromptSchedule from any request type.

    Resolution order:
    1. ``req.prompt_schedule`` (new keyframe format) — if present
    2. ``req.prompt_segments`` (legacy audio segments) — if present
    3. None (no schedule, use static ``req.prompt``)
    """
    from ..prompt_schedule import PromptSchedule, PromptKeyframe

    # New keyframe-based schedule
    schedule_spec = getattr(req, "prompt_schedule", None)
    if schedule_spec is not None:
        # Handle both PromptScheduleSpec (Pydantic) and raw dict
        if hasattr(schedule_spec, "keyframes"):
            kf_dicts = [kf.model_dump() for kf in schedule_spec.keyframes]
        elif isinstance(schedule_spec, dict):
            kf_dicts = schedule_spec.get("keyframes", [])
        else:
            kf_dicts = []
        if kf_dicts:
            # Lua json.lua may encode arrays as objects with numeric keys
            if isinstance(kf_dicts, dict):
                kf_dicts = list(kf_dicts.values())
            default = ""
            if hasattr(schedule_spec, "default_prompt"):
                default = schedule_spec.default_prompt
            elif isinstance(schedule_spec, dict):
                default = schedule_spec.get("default_prompt", "")
            schedule = PromptSchedule.from_keyframe_dicts(
                kf_dicts, default or getattr(req, "prompt", ""),
            )
            is_auto = False
            if hasattr(schedule_spec, "auto_fill"):
                is_auto = schedule_spec.auto_fill
            elif isinstance(schedule_spec, dict):
                is_auto = schedule_spec.get("auto_fill", False)
                
            if schedule and is_auto:
                from ..prompt_schedule import auto_fill_prompts
                from ..prompt_generator import prompt_generator
                schedule = auto_fill_prompts(
                    schedule, prompt_generator, randomness=getattr(req, "randomness", 5),
                    locked_fields=getattr(req, "locked_fields", None)
                )
            return schedule

    return None


_HUE_SHIFT_EPSILON = 1e-6
_DENOISE_FLOOR = 0.01
_MOTION_XY_THRESHOLD = 0.01
_MOTION_ZOOM_THRESHOLD = 0.001
_MOTION_TILT_THRESHOLD = 0.01
_AUTO_NOISE_DENOISE_GATE = 0.35
_AUTO_NOISE_CEILING = 0.9
_AUTO_NOISE_SCALE = 0.1


class GenerationCancelled(Exception):
    """Raised when a client cancels an in-progress generation."""


def _apply_hue_shift(image: Image.Image, shift: float) -> Image.Image:
    """Shift hue of an image by `shift` fraction (0-1 maps to 0-360 degrees).

    Preserves alpha channel. Used for audio-driven palette shift modulation.
    """
    if image.size[0] == 0 or image.size[1] == 0:
        return image
    if abs(shift) < _HUE_SHIFT_EPSILON:
        return image
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[-1] if has_alpha else None
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.array(h, dtype=np.int16)
    h_arr = (h_arr + int(shift * 255)) % 256
    h = Image.fromarray(h_arr.astype(np.uint8))
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
    strength = max(strength, _DENOISE_FLOOR)  # safety floor
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
    mx = frame_params.get("motion_x", 0.0)
    my = frame_params.get("motion_y", 0.0)
    mz = frame_params.get("motion_zoom", 1.0)
    mr = frame_params.get("motion_rotation", 0.0)
    if abs(mx) > _MOTION_XY_THRESHOLD or abs(my) > _MOTION_XY_THRESHOLD or abs(mz - 1.0) > _MOTION_ZOOM_THRESHOLD or abs(mr) > _MOTION_XY_THRESHOLD:
        image = apply_motion_warp(
            image, tx=mx, ty=my, zoom=mz, rotation=mr,
            denoise_strength=denoise_strength,
        )

    mtx = frame_params.get("motion_tilt_x", 0.0)
    mty = frame_params.get("motion_tilt_y", 0.0)
    if abs(mtx) > _MOTION_TILT_THRESHOLD or abs(mty) > _MOTION_TILT_THRESHOLD:
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
    *,
    noise_buf: np.ndarray | None = None,
    work_buf: np.ndarray | None = None,
) -> Image.Image:
    """Inject noise into a frame image for temporal variation.

    O-16: Optional pre-allocated buffers for zero-allocation frame loops:
      - noise_buf: float32 array of shape (H, W, C) for noise generation.
      - work_buf: float32 array of shape (H, W, C) for pixel workspace.
    Pass None (default) for backward compatibility — buffers will be
    allocated on-the-fly.

    Auto-coupling: when no ``noise_amplitude`` slot is active, injects subtle
    noise inversely proportional to denoise strength — gated at 0.35 to prevent
    artifact accumulation at low denoise values.
    """
    noise_amp = max(0.0, min(1.0, frame_params.get("noise_amplitude", 0.0)))
    if settings.auto_noise_coupling and "noise_amplitude" not in frame_params:
        if denoise_strength >= _AUTO_NOISE_DENOISE_GATE:
            noise_amp = max(0.0, (_AUTO_NOISE_CEILING - denoise_strength) * _AUTO_NOISE_SCALE)
        else:
            noise_amp = 0.0
    if noise_amp > 0:
        arr_u8 = np.array(image)
        shape = arr_u8.shape

        # O-16: Reuse pre-allocated buffers when provided
        if work_buf is not None and work_buf.shape == shape:
            np.divide(arr_u8, 255.0, out=work_buf, casting='unsafe')
            arr = work_buf
        else:
            arr = arr_u8.astype(np.float32) / 255.0

        rng = np.random.default_rng(seed)
        if noise_buf is not None and noise_buf.shape == shape:
            rng.standard_normal(out=noise_buf, dtype=np.float32)
            noise_buf *= noise_amp
            arr += noise_buf
        else:
            noise = rng.standard_normal(shape, dtype=np.float32) * noise_amp
            arr += noise

        np.clip(arr, 0.0, 1.0, out=arr)
        image = Image.fromarray((arr * 255).astype(np.uint8))
    return image


# ── Shared prompt resolution + pipeline kwargs ────────────


@dataclass
class FramePromptResult:
    """Resolved prompt info for a single animation frame.

    ``denoise_strength``, ``cfg_scale``, ``steps`` are raw per-keyframe
    overrides (None = use base).  Callers compute effective values
    (scaled steps, sub-floor alpha) from these.
    """
    prompt: str
    negative: str
    blend_embeds: tuple | None = None
    denoise_strength: float | None = None
    cfg_scale: float | None = None
    steps: int | None = None


def resolve_frame_prompt(
    schedule,
    frame_idx: int,
    base_prompt: str,
    base_negative: str,
    ti_suffix: str,
    pipe_for_embed,
    clip_skip: int,
    *,
    audio_fps: float | None = None,
) -> FramePromptResult:
    """Resolve prompt + SLERP blend + per-keyframe overrides for a frame.

    Shared between animation.py and audio_reactive.py chain loops.
    Returns None-valued overrides when the schedule doesn't specify them —
    callers apply their own defaults.

    ``audio_fps`` enables legacy time-based segment fallback (audio_reactive only).
    """
    result = FramePromptResult(prompt=base_prompt, negative=base_negative)

    if not schedule:
        return result

    if schedule.keyframes:
        blend_info = schedule.get_blend_info_for_frame(frame_idx)
        result.prompt = blend_info.effective_prompt or base_prompt
        if blend_info.negative_prompt:
            result.negative = (
                (blend_info.negative_prompt + ", " + ti_suffix)
                if ti_suffix else blend_info.negative_prompt
            )
        if blend_info.is_blending:
            try:
                result.blend_embeds = blend_prompt_embeds(
                    pipe_for_embed,
                    blend_info.prompt_a,
                    blend_info.prompt_b,
                    blend_info.blend_weight,
                    negative_prompt=result.negative,
                    negative_prompt_b=blend_info.negative_prompt_b,
                    clip_skip=clip_skip,
                )
                log.debug(
                    "Frame %d: SLERP blend %.2f (%s → %s)",
                    frame_idx, blend_info.blend_weight,
                    blend_info.prompt_a[:30],
                    blend_info.prompt_b[:30],
                )
            except Exception as e:
                log.warning("SLERP blend failed frame %d: %s", frame_idx, e)
        if blend_info.denoise_strength is not None:
            result.denoise_strength = blend_info.denoise_strength
        if blend_info.cfg_scale is not None:
            result.cfg_scale = blend_info.cfg_scale
        if blend_info.steps is not None:
            result.steps = blend_info.steps
    elif hasattr(schedule, "segments") and schedule.segments and audio_fps is not None:
        result.prompt = schedule.get_prompt(frame_idx / audio_fps)

    return result


def inject_prompt_kwargs(
    kwargs: dict,
    blend_embeds: tuple | None,
    prompt: str,
    negative: str,
) -> None:
    """Set prompt or embedding kwargs on a pipeline call dict in-place.

    Uses pre-computed SLERP embeddings when available, falling back
    to text prompts.
    """
    if blend_embeds is not None:
        kwargs["prompt_embeds"] = blend_embeds[0]
        kwargs["negative_prompt_embeds"] = blend_embeds[1]
    else:
        kwargs["prompt"] = prompt
        kwargs["negative_prompt"] = negative

