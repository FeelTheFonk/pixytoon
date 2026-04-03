from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..config import settings
from ..embedding_blend import blend_prompt_embeds
from ..image_codec import (
    match_color_lab,
    apply_motion_warp,
    apply_perspective_tilt,
    apply_frame_transforms,
    _ensure_rgb3,
    _get_dis_instance,
    _get_flow_grid,
)
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
            # Lua json.lua may encode arrays as objects with numeric string keys
            if isinstance(kf_dicts, dict):
                try:
                    kf_dicts = [kf_dicts[k] for k in sorted(kf_dicts, key=lambda x: int(x))]
                except (ValueError, TypeError):
                    kf_dicts = list(kf_dicts.values())
        else:
            kf_dicts = []
        if kf_dicts:
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

    S5: Uses OpenCV HSV conversion instead of PIL for ~3-5x speedup.
    Preserves alpha channel. Used for audio-driven palette shift modulation.
    """
    if image.size[0] == 0 or image.size[1] == 0:
        return image
    if abs(shift) < _HUE_SHIFT_EPSILON:
        return image
    arr = np.asarray(image)
    has_alpha = arr.shape[2] == 4 if len(arr.shape) == 3 else False
    rgb = arr[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # OpenCV H range is [0, 180], shift is [0, 1]
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + int(shift * 180)) % 180).astype(np.uint8)
    rgb_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if has_alpha:
        result = np.empty_like(arr)
        result[:, :, :3] = rgb_out
        result[:, :, 3] = arr[:, :, 3]
        return Image.fromarray(result)
    return Image.fromarray(rgb_out)


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


def _compute_dis_flow(
    current: Image.Image,
    previous: Image.Image,
) -> np.ndarray:
    """Compute DIS optical flow (previous -> current) for EquiVDM noise warping.

    Returns a float32 (H, W, 2) flow field.  Reuses the same DIS preset
    as ``apply_optical_flow_blend`` in image_codec.py.
    """
    curr_arr = _ensure_rgb3(np.asarray(current, dtype=np.uint8))
    prev_arr = _ensure_rgb3(np.asarray(previous, dtype=np.uint8))
    if prev_arr.shape[:2] != curr_arr.shape[:2]:
        prev_arr = cv2.resize(
            prev_arr, (curr_arr.shape[1], curr_arr.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    curr_gray = cv2.cvtColor(curr_arr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_arr, cv2.COLOR_RGB2GRAY)
    return _get_dis_instance().calc(prev_gray, curr_gray, None)


def apply_temporal_coherence(
    image: Image.Image,
    prev_image: Image.Image,
    *,
    return_flow: bool = False,
    frame_id: int | None = None,
) -> Image.Image | tuple[Image.Image, np.ndarray | None]:
    """Apply color coherence + optical flow blending between consecutive frames.

    Uses ``settings.color_coherence_strength`` and ``settings.optical_flow_blend``
    to control intensity.  Both are no-ops when their setting is 0.

    When *return_flow* is True, also returns the DIS optical flow field
    (prev -> current) for reuse by EquiVDM noise warping.  Returns None
    for the flow component when optical flow is not computed.

    Args:
        frame_id: Passed to match_color_lab as cache key, avoiding per-frame
            MD5 hash computation (~0% hit-rate in animation).
    """
    flow_map: np.ndarray | None = None

    if settings.color_coherence_strength > 0:
        image = match_color_lab(image, prev_image, settings.color_coherence_strength,
                                frame_id=frame_id)
    if settings.optical_flow_blend > 0:
        # Compute flow once and reuse for both blending and EquiVDM
        flow_map = _compute_dis_flow(image, prev_image)
        # Apply optical flow blend using the computed flow
        curr_arr = _ensure_rgb3(np.asarray(image, dtype=np.uint8))
        prev_arr = _ensure_rgb3(np.asarray(prev_image, dtype=np.uint8))
        if prev_arr.shape[:2] != curr_arr.shape[:2]:
            prev_arr = cv2.resize(
                prev_arr, (curr_arr.shape[1], curr_arr.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        h, w = curr_arr.shape[:2]
        grid_y, grid_x = _get_flow_grid(h, w)
        map_x = grid_x + flow_map[..., 0]
        map_y = grid_y + flow_map[..., 1]
        warped = cv2.remap(
            prev_arr, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        strength = settings.optical_flow_blend
        result = cv2.addWeighted(curr_arr, 1.0 - strength, warped, strength, 0)
        image = Image.fromarray(result)
    elif return_flow and settings.equivdm_noise:
        # Compute flow even when optical_flow_blend is disabled,
        # because EquiVDM noise warping needs it.
        flow_map = _compute_dis_flow(image, prev_image)

    if return_flow:
        return image, flow_map
    return image


def apply_frame_motion(
    image: Image.Image,
    frame_params: dict[str, float],
    denoise_strength: float,
) -> Image.Image:
    """Apply 2D affine motion warp + perspective tilt from modulation params.

    Uses apply_frame_transforms to fuse both warps into a single PIL↔numpy
    round-trip when both are active, avoiding an intermediate conversion.
    Matching Deforum ordering: affine first, then perspective.
    """
    mx = frame_params.get("motion_x", 0.0)
    my = frame_params.get("motion_y", 0.0)
    mz = frame_params.get("motion_zoom", 1.0)
    mr = frame_params.get("motion_rotation", 0.0)
    has_warp = (abs(mx) > _MOTION_XY_THRESHOLD or abs(my) > _MOTION_XY_THRESHOLD
                or abs(mz - 1.0) > _MOTION_ZOOM_THRESHOLD or abs(mr) > _MOTION_XY_THRESHOLD)

    mtx = frame_params.get("motion_tilt_x", 0.0)
    mty = frame_params.get("motion_tilt_y", 0.0)
    has_tilt = abs(mtx) > _MOTION_TILT_THRESHOLD or abs(mty) > _MOTION_TILT_THRESHOLD

    warp_params = dict(tx=mx, ty=my, zoom=mz, rotation=mr,
                       denoise_strength=denoise_strength) if has_warp else None
    tilt_params = dict(tilt_x=mtx, tilt_y=mty,
                       denoise_strength=denoise_strength) if has_tilt else None

    return apply_frame_transforms(image, warp_params=warp_params, tilt_params=tilt_params)


def apply_noise_injection(
    image: Image.Image,
    frame_params: dict[str, float],
    seed: int,
    denoise_strength: float,
    *,
    noise_buf: np.ndarray | None = None,
    work_buf: np.ndarray | None = None,
    out_buf: np.ndarray | None = None,
    prev_noise: np.ndarray | None = None,
    flow_map: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray | None]:
    """Inject noise into a frame image for temporal variation.

    O-16: Optional pre-allocated buffers for zero-allocation frame loops:
      - noise_buf: float32 array of shape (H, W, C) for noise generation.
      - work_buf: float32 array of shape (H, W, C) for pixel workspace.
      - out_buf: uint8 array of shape (H, W, C) for final RGB output.
        When provided and shape matches, the final float->uint8 conversion
        reuses this buffer and Image.frombuffer avoids a data copy.
        WARNING: The returned PIL Image does NOT own out_buf — the caller
        must keep out_buf alive for the lifetime of the Image, and must not
        mutate out_buf while the Image is in use.
    Pass None (default) for backward compatibility — buffers will be
    allocated on-the-fly.

    F-O6 EquiVDM temporally coherent noise:
      - prev_noise: float32 (H, W, C) noise array from the previous frame.
      - flow_map: float32 (H, W, 2) DIS optical flow (prev -> current).
      When ``settings.equivdm_noise`` is enabled and both are provided,
      the previous frame's noise is flow-warped and blended with fresh
      noise using ``settings.equivdm_residual`` as the fresh-noise weight.
      This produces temporally coherent noise that reduces structural
      flicker in chain animations at zero VRAM cost.

    Auto-coupling: when no ``noise_amplitude`` slot is active, injects subtle
    noise inversely proportional to denoise strength — gated at 0.35 to prevent
    artifact accumulation at low denoise values.

    Returns:
        (image, noise_used) — the processed image and the noise array used
        for this frame (pass as ``prev_noise`` on the next frame).  Returns
        None for noise_used when no noise was injected.
    """
    noise_amp = max(0.0, min(1.0, frame_params.get("noise_amplitude", 0.0)))
    if settings.auto_noise_coupling and "noise_amplitude" not in frame_params:
        if denoise_strength >= _AUTO_NOISE_DENOISE_GATE:
            noise_amp = max(0.0, (_AUTO_NOISE_CEILING - denoise_strength) * _AUTO_NOISE_SCALE)
        else:
            noise_amp = 0.0
    if noise_amp > 0:
        arr_u8 = np.asarray(image)  # S8: avoid copy when possible
        shape = arr_u8.shape

        # O-16: Reuse pre-allocated buffers when provided
        if work_buf is not None and work_buf.shape == shape:
            np.divide(arr_u8, 255.0, out=work_buf, casting='unsafe')
            arr = work_buf
        else:
            arr = arr_u8.astype(np.float32) / 255.0

        rng = np.random.default_rng(seed)  # Per-frame RNG for reproducibility

        # Generate fresh noise
        if noise_buf is not None and noise_buf.shape == shape:
            rng.standard_normal(out=noise_buf, dtype=np.float32)
            fresh_noise = noise_buf
        else:
            fresh_noise = rng.standard_normal(shape, dtype=np.float32)

        # F-O6: EquiVDM temporally coherent noise — flow-warp previous
        # frame's noise and blend with fresh noise.
        if (settings.equivdm_noise
                and prev_noise is not None
                and flow_map is not None
                and prev_noise.shape == shape):
            h, w = shape[:2]
            grid_y, grid_x = _get_flow_grid(h, w)
            map_x = grid_x + flow_map[..., 0]
            map_y = grid_y + flow_map[..., 1]
            # Warp each channel of prev_noise using the flow field
            warped_noise = cv2.remap(
                prev_noise, map_x, map_y,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            residual = settings.equivdm_residual
            # In-place blend: warped*(1-r) + fresh*r — avoids 3 temp arrays
            np.multiply(warped_noise, 1.0 - residual, out=warped_noise)
            np.multiply(fresh_noise, residual, out=noise_buf if fresh_noise is noise_buf else fresh_noise)
            np.add(warped_noise, fresh_noise, out=warped_noise)
            noise_used = warped_noise
        else:
            noise_used = fresh_noise.copy() if fresh_noise is noise_buf else fresh_noise

        arr += noise_used * noise_amp
        np.clip(arr, 0.0, 1.0, out=arr)

        # F-C1: Zero-alloc final conversion when out_buf is provided.
        if out_buf is not None and out_buf.shape == shape and out_buf.dtype == np.uint8:
            w, h = image.size
            np.multiply(arr, 255, out=work_buf)
            np.copyto(out_buf, work_buf, casting='unsafe')
            image = Image.frombuffer('RGB', (w, h), out_buf.data, 'raw', 'RGB', 0, 1)
        else:
            image = Image.fromarray((arr * 255).astype(np.uint8))

        return image, noise_used
    return image, None


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

