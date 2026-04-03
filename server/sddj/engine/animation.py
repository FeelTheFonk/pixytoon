"""Animation generation — chain + AnimateDiff methods.

Chain mode supports SLERP prompt embedding interpolation for smooth
transitions between keyframes (via PromptBlendInfo).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional

import torch
import torch.compiler
from PIL import Image

from ..config import settings
from ..postprocess import apply as postprocess_apply
from ..protocol import (
    AnimationFrameResponse,
    AnimationMethod,
    AnimationRequest,
    GenerationMode,
    ProgressResponse,
    SeedStrategy,
)
from .. import pipeline_factory
from .compile_utils import eager_pipeline
from ..vram_utils import vram_cleanup
from ..image_codec import (
    composite_with_mask,
    decode_b64_image,
    decode_b64_mask,
    encode_image_raw_bytes,
    resize_to_target,
    round8,
)
from .helpers import (
    GenerationCancelled,
    apply_temporal_coherence,
    build_prompt_schedule,
    compute_effective_denoise,
    inject_prompt_kwargs,
    make_step_callback,
    resolve_frame_prompt,
    scale_steps_for_denoise,
)

log = logging.getLogger("sddj.engine")


def _interpolate_frames(frames: list, factor: int) -> list:
    """Interpolate between animation frames using RIFE or linear blend.

    Falls back to simple alpha blending if RIFE is not available.
    """
    if len(frames) < 2 or factor < 2:
        return frames
    try:
        # Try RIFE first
        from rife_ncnn_vulkan import Rife
        rife = Rife(gpuid=0)
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            for j in range(1, factor):
                t = j / factor
                mid = rife.process(frames[i], frames[i + 1], timestep=t)
                interpolated.append(mid)
        interpolated.append(frames[-1])
        return interpolated
    except ImportError:
        log.warning(
            "RIFE not available — using simple alpha blend (poor quality). "
            "Install rife-ncnn-vulkan-python for proper frame interpolation."
        )
        # Simple alpha blend fallback
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            for j in range(1, factor):
                t = j / factor
                blended = Image.blend(frames[i], frames[i + 1], t)
                interpolated.append(blended)
        interpolated.append(frames[-1])
        return interpolated
    except Exception as e:
        log.warning("Frame interpolation failed: %s — returning original frames", e)
        return frames


class AnimationMixin:
    """Animation generation methods for DiffusionEngine."""

    def generate_animation(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]] = None,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> int:
        """Generate multi-frame animation — dispatches to chain or animatediff method."""
        if not self._loaded:
            self.load()

        self._cancel_event.clear()
        _original_scheduler = None

        try:
            # Handle LoRA same as single generation
            if req.lora is not None:
                if (req.lora.name != self._lora_fuser.current_name
                        or req.lora.weight != self._lora_fuser.current_weight):
                    self.set_style_lora(req.lora.name, req.lora.weight)

            # Per-request scheduler override
            if req.scheduler:
                try:
                    from ..scheduler_factory import create_scheduler
                    _original_scheduler = self._pipe.scheduler
                    new_sched = create_scheduler(req.scheduler, _original_scheduler.config)
                    self._pipe.scheduler = new_sched
                    if self._img2img_pipe is not None:
                        self._img2img_pipe.scheduler = type(new_sched).from_config(new_sched.config)
                except Exception as e:
                    log.warning("Scheduler override '%s' failed: %s", req.scheduler, e)
                    _original_scheduler = None

            try:
                if req.method == AnimationMethod.CHAIN:
                    return self._generate_chain(req, on_frame, on_progress)
                elif req.method == AnimationMethod.ANIMATEDIFF:
                    return self._generate_animatediff(req, on_frame, on_progress)
                else:
                    raise ValueError(f"Unknown animation method: {req.method}")
            finally:
                if _original_scheduler is not None:
                    self._pipe.scheduler = _original_scheduler
                    if self._img2img_pipe is not None:
                        self._img2img_pipe.scheduler = type(_original_scheduler).from_config(
                            _original_scheduler.config)

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM during animation — clearing VRAM cache")
            vram_cleanup()
            raise
        finally:
            self._cancel_event.clear()

    def _generate_chain(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Frame-by-frame chaining: each frame feeds into the next via img2img.

        Uses a temporary UNet swap + dynamo disable to bypass torch.compile + DeepCache:
        1. Suspend DeepCache → unwrap_modules() restores raw UNet's original forwards
        2. Swap pipe.unet and img2img_pipe.unet to raw UNet (_orig_mod)
        3. Disable dynamo globally → prevents eval_frame hook from intercepting raw UNet
        4. Run all frames in pure eager mode (no compile, no DeepCache, FreeU via attrs)
        5. Restore compiled UNet and re-enable DeepCache on exit

        CRITICAL: torch.compiler.disable() is REQUIRED. _orig_mod is the same Python object
        that torch.compile() registered in dynamo's tracking tables. Without disabling dynamo,
        its global eval_frame hook intercepts the raw UNet's forward() call, finds stale
        guards (compiled graph was built with DeepCache active + different step counts),
        and triggers recompilation that hangs indefinitely. AnimateDiff avoids this by using
        a completely different pipeline class (AnimateDiffPipeline), whose __call__ was never
        compiled — dynamo has no guards for it.
        """
        with eager_pipeline(self._pipe, self._img2img_pipe,
                            self._controlnet_pipe, self._deepcache_helper,
                            self._controlnet_img2img_pipe):
            return self._generate_chain_inner(req, on_frame, on_progress)

    @torch.compiler.disable
    def _generate_chain_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Core chain animation logic — dynamo disabled via decorator.

        This decorator prevents torch._dynamo's global eval_frame hook from
        intercepting ANY function calls within this method. Without it, dynamo
        recognizes _orig_mod (the raw UNet) as a compiled object and triggers
        recompilation with stale guards, hanging indefinitely.
        """
        frame_count = 0
        base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        base_seed = base_seed % (2**32)  # clamp to valid range
        chain_source: Optional[Image.Image] = None

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        # Pre-compute TI suffix once (avoids per-frame TI token iteration in blend path)
        _ti_suffix = self._build_ti_suffix(req.negative_ti)

        # Pre-decode base64 images once (avoid re-decoding per frame)
        _source_img = None
        if req.source_image is not None:
            _source_img = decode_b64_image(req.source_image).convert("RGB")
            _source_img = resize_to_target(_source_img, target_w, target_h)
        _mask_img = None
        if req.mask_image is not None:
            _mask_img = decode_b64_mask(req.mask_image)
            _mask_img = resize_to_target(_mask_img, target_w, target_h)
        _control_img = None
        if req.control_image is not None:
            _control_img = decode_b64_image(req.control_image).convert("RGB")
            _control_img = resize_to_target(_control_img, target_w, target_h)

        # Clamp: ensure at least 2 effective denoising steps for img2img
        # (int(steps * strength) == 0 → empty latents → VAE crash)
        chain_denoise, chain_scaled_steps, _ = compute_effective_denoise(req.steps, req.denoise_strength)

        log.info("Chain animation: %d frames, mode=%s, steps=%d (scaled=%d), denoise=%.2f, seed_base=%d",
                 req.frame_count, req.mode.value, req.steps, chain_scaled_steps, chain_denoise, base_seed)

        # Build prompt schedule (resolved per-frame inside loop)
        schedule = build_prompt_schedule(req)

        # Reuse a single CUDA generator (reseed per frame — avoids per-frame allocation)
        _generator = torch.Generator("cuda")

        # S3: Create scheduler once and reuse via set_timesteps() instead of
        # from_config() per frame. Diffusers mutates scheduler state in-place,
        # so we reset via set_timesteps() each frame instead of rebuilding.
        _frame_scheduler = type(self._pipe.scheduler).from_config(self._pipe.scheduler.config)

        with torch.inference_mode():
            for frame_idx in range(req.frame_count):
                if self._cancel_event.is_set():
                    raise GenerationCancelled("Animation cancelled by client")

                t0_frame = time.perf_counter()

                # Resolve seed per strategy
                if req.seed_strategy == SeedStrategy.FIXED:
                    frame_seed = base_seed
                elif req.seed_strategy == SeedStrategy.INCREMENT:
                    frame_seed = base_seed + frame_idx
                else:  # RANDOM
                    frame_seed = random.randint(0, 2**32 - 1)

                generator = _generator
                generator.manual_seed(frame_seed)

                # Progress callback with frame context
                step_callback = make_step_callback(
                    self._cancel_event, on_progress, req.steps,
                    frame_idx=frame_idx, total_frames=req.frame_count,
                )

                # Frame 0: initial generation (direct pipeline call — NO cudagraph markers)
                # Chain mode uses raw (non-compiled) UNet, so _txt2img/_img2img
                # wrappers must be avoided as they call cudagraph_mark_step_begin()
                # Reset img2img scheduler before each frame to prevent
                # stale state accumulation (timesteps, _step_index).
                # This is the root fix for the chain animation infinite loop.
                # TODO(Performance Phase C): diffusers `__call__` already calls `set_timesteps()`.
                # This explicitly clears scheduler state arrays, but may be completely redundant.
                # Must be empirically validated with verify_optimizations.py before removing.
                if frame_idx > 0:
                    self._img2img_pipe.scheduler = _frame_scheduler
                    _frame_scheduler.set_timesteps(frame_steps)
                    log.debug("Chain frame %d: scheduler reset", frame_idx)

                log.debug("Chain frame %d/%d: seed=%d, mode=%s",
                          frame_idx, req.frame_count, frame_seed, req.mode.value)

                # Resolve per-frame prompt from schedule (with SLERP blending)
                fp = resolve_frame_prompt(
                    schedule, frame_idx,
                    base_prompt=req.prompt,
                    base_negative=effective_neg,
                    ti_suffix=_ti_suffix,
                    pipe_for_embed=(
                        self._pipe if frame_idx == 0 else self._img2img_pipe
                    ),
                    clip_skip=req.clip_skip,
                )
                frame_prompt = fp.prompt
                frame_neg = fp.negative
                blend_embeds = fp.blend_embeds
                frame_denoise = chain_denoise
                frame_cfg = req.cfg_scale
                frame_steps = chain_scaled_steps
                frame_steps_full = req.steps
                if fp.denoise_strength is not None:
                    d, s, _ = compute_effective_denoise(req.steps, fp.denoise_strength)
                    frame_denoise = d
                    frame_steps = s
                if fp.cfg_scale is not None:
                    frame_cfg = fp.cfg_scale
                if fp.steps is not None:
                    frame_steps = scale_steps_for_denoise(fp.steps, frame_denoise)
                    frame_steps_full = fp.steps

                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        gen_kwargs = dict(
                            num_inference_steps=frame_steps_full,
                            guidance_scale=frame_cfg,
                            width=target_w,
                            height=target_h,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                        image = self._pipe(**gen_kwargs).images[0]
                    elif req.mode == GenerationMode.IMG2IMG:
                        if _source_img is None:
                            raise ValueError("img2img requires source_image")
                        gen_kwargs = dict(
                            image=_source_img,
                            num_inference_steps=frame_steps,
                            guidance_scale=frame_cfg,
                            strength=frame_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                        image = self._img2img_pipe(**gen_kwargs).images[0]
                    elif req.mode == GenerationMode.INPAINT:
                        if _source_img is None or _mask_img is None:
                            raise ValueError("inpaint requires source_image and mask_image")
                        inp_kwargs = dict(
                            image=_source_img,
                            num_inference_steps=frame_steps,
                            guidance_scale=frame_cfg,
                            strength=frame_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        inject_prompt_kwargs(inp_kwargs, blend_embeds, frame_prompt, frame_neg)
                        inpainted = self._img2img_pipe(**inp_kwargs).images[0]
                        image = composite_with_mask(_source_img, inpainted, _mask_img)
                    elif req.mode.value.startswith("controlnet_"):
                        if _control_img is None:
                            raise ValueError("controlnet requires control_image")
                        self._ensure_controlnet(req.mode)
                        cn_f0_kwargs = dict(
                            image=_control_img,
                            num_inference_steps=frame_steps_full,
                            guidance_scale=frame_cfg,
                            width=target_w,
                            height=target_h,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        cgs = getattr(req, 'control_guidance_start', 0.0)
                        cge = getattr(req, 'control_guidance_end', 1.0)
                        if cgs > 0.0:
                            cn_f0_kwargs["control_guidance_start"] = cgs
                        if cge < 1.0:
                            cn_f0_kwargs["control_guidance_end"] = cge
                        inject_prompt_kwargs(cn_f0_kwargs, blend_embeds, frame_prompt, frame_neg)
                        image = self._controlnet_pipe(**cn_f0_kwargs).images[0]
                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                else:
                    # Frame 1+: img2img from previous frame at denoise_strength
                    if chain_source is None:
                        raise RuntimeError("Chain animation failed: previous frame did not produce output")
                    source = chain_source  # pipeline output already matches target dims

                    log.info("Chain frame %d: img2img (source=%s, steps=%d, strength=%.2f)",
                             frame_idx, source.size, frame_steps, frame_denoise)
                    i2i_kwargs = dict(
                        image=source,
                        num_inference_steps=frame_steps,
                        guidance_scale=frame_cfg,
                        strength=frame_denoise,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )
                    inject_prompt_kwargs(i2i_kwargs, blend_embeds, frame_prompt, frame_neg)
                    image = self._img2img_pipe(**i2i_kwargs).images[0]

                # Temporal coherence: color matching + optical flow (frame 1+)
                if frame_idx > 0 and chain_source is not None:
                    image = apply_temporal_coherence(image, chain_source)

                # Store pre-postprocess image for next frame's img2img source
                # (full-resolution, RGB, no pixelation/quantization artifacts)
                chain_source = image

                # Post-process
                image = postprocess_apply(image, req.post_process)

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Animation cancelled during post-processing")

                # Encode
                raw_bytes = encode_image_raw_bytes(image)
                w, h = image.size
                # frame_time_ms: per-frame generation time (from this frame's t0_frame)
                elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)

                frame_resp = AnimationFrameResponse(
                    frame_index=frame_idx,
                    total_frames=req.frame_count,
                    image="",
                    seed=frame_seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    encoding="raw_rgba",
                )
                frame_resp._raw_bytes = raw_bytes
                frame_count += 1
                if on_frame:
                    on_frame(frame_resp)

        return frame_count

    def _generate_animatediff(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """AnimateDiff motion module generation for temporal consistency."""

        # Lightning frame cap: FreeNoise is incompatible with distilled models
        if settings.is_animatediff_lightning:
            max_lt = settings.animatediff_max_frames_lightning
            if req.frame_count > max_lt:
                raise ValueError(
                    f"AnimateDiff-Lightning is limited to {max_lt} frames "
                    f"(requested {req.frame_count}). FreeNoise long-video is "
                    f"incompatible with distilled few-step models."
                )

        # Smart transition: free ControlNet if not needed for this request
        is_controlnet = req.mode.value.startswith("controlnet_")
        if not is_controlnet and self._controlnet_pipe is not None:
            log.info("Smart transition: unloading ControlNet before AnimateDiff")
            self._controlnet_pipe = None
            self._controlnet_mode = None

        # Mode-aware DeepCache: only toggles on actual mode transition
        # (avoids 100-300ms disable/enable overhead on consecutive AnimateDiff calls)
        if self._dc_state is not None:
            self._dc_state.suppress_for("animatediff")
        try:
            return self._generate_animatediff_inner(req, on_frame, on_progress)
        finally:
            if self._dc_state is not None:
                try:
                    self._dc_state.restore()
                except Exception as e:
                    log.warning("DeepCache restore failed after AnimateDiff (non-critical): %s", e)

    def _generate_animatediff_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Core AnimateDiff generation — FreeNoise native sliding window.

        For sequences longer than context_length, FreeNoise handles temporal
        attention windowing + noise rescheduling internally. This replaces
        the previous manual chunking approach which lacked proper noise
        rescheduling and produced visible seams between chunks.

        Short sequences (≤ context_length) run as a single pass without
        FreeNoise overhead.
        """
        is_controlnet = req.mode.value.startswith("controlnet_")
        is_img2img = req.mode == GenerationMode.IMG2IMG

        if is_controlnet:
            existing_cn = pipeline_factory.get_controlnet_from_pipe(
                self._controlnet_pipe, self._controlnet_mode, req.mode,
            )
            pipe = self._animatediff.ensure_controlnet(
                self._pipe, req.mode, existing_controlnet=existing_cn,
            )
        elif is_img2img:
            pipe = self._animatediff.ensure_vid2vid(self._pipe)
        else:
            pipe = self._animatediff.ensure_base(self._pipe)

        total_frames = req.frame_count

        # FreeNoise: enable sliding-window temporal attention for long sequences
        free_noise_active = self._animatediff.apply_free_noise(pipe, total_frames)

        # FreeInit (not supported on vid2vid or Lightning pipelines)
        freeinit_enabled = False
        if req.enable_freeinit and not is_img2img:
            if settings.is_animatediff_lightning:
                log.warning("FreeInit disabled: incompatible with AnimateDiff-Lightning")
            else:
                try:
                    pipe.enable_free_init(
                        num_iters=req.freeinit_iterations,
                        use_fast_sampling=True,
                    )
                    freeinit_enabled = True
                    log.info("FreeInit enabled (%d iterations)", req.freeinit_iterations)
                except Exception as e:
                    log.warning("FreeInit unavailable: %s", e)

        # Lightning parameter enforcement
        if settings.is_animatediff_lightning:
            effective_steps = settings.animatediff_lightning_steps
            effective_cfg = settings.animatediff_lightning_cfg
            log.info("AnimateDiff-Lightning: enforcing steps=%d, cfg=%.1f "
                     "(request had steps=%d, cfg=%.1f)",
                     effective_steps, effective_cfg, req.steps, req.cfg_scale)
        else:
            effective_steps = req.steps
            effective_cfg = req.cfg_scale

        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        seed = seed % (2**32)
        generator = torch.Generator("cuda").manual_seed(seed)

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        # Prompt schedule: use midpoint prompt for the full sequence
        schedule = build_prompt_schedule(req)
        gen_prompt = req.prompt
        gen_neg = effective_neg
        if schedule and schedule.keyframes:
            mid_frame = total_frames // 2
            kf_prompt = schedule.get_prompt_for_frame(mid_frame)
            if kf_prompt:
                gen_prompt = kf_prompt
            kf_neg = schedule.get_negative_for_frame(mid_frame)
            if kf_neg:
                gen_neg = self._build_effective_negative(kf_neg, req.negative_ti)

        # Pre-decode images
        _source_img = None
        if is_img2img:
            if req.source_image is None:
                raise ValueError("AnimateDiff img2img requires source_image")
            _source_img = decode_b64_image(req.source_image).convert("RGB")
            _source_img = resize_to_target(_source_img, target_w, target_h)

        _control_img = None
        if is_controlnet and req.control_image is not None:
            _control_img = decode_b64_image(req.control_image).convert("RGB")
            _control_img = resize_to_target(_control_img, target_w, target_h)

        t0 = time.perf_counter()

        step_callback = make_step_callback(
            self._cancel_event, on_progress, effective_steps,
            frame_idx=0, total_frames=total_frames,
        )

        log.info("AnimateDiff: %d frames, steps=%d, cfg=%.1f, seed=%d, free_noise=%s",
                 total_frames, effective_steps, effective_cfg, seed, free_noise_active)

        try:
            with torch.inference_mode():
                if is_img2img:
                    strength, scaled_steps, _ = compute_effective_denoise(
                        effective_steps, req.denoise_strength)
                    kwargs = dict(
                        video=[_source_img] * total_frames,
                        prompt=gen_prompt,
                        negative_prompt=gen_neg,
                        num_inference_steps=scaled_steps,
                        guidance_scale=effective_cfg,
                        strength=strength,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )
                else:
                    kwargs = dict(
                        prompt=gen_prompt,
                        negative_prompt=gen_neg,
                        num_frames=total_frames,
                        num_inference_steps=effective_steps,
                        guidance_scale=effective_cfg,
                        width=target_w,
                        height=target_h,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )
                    if is_controlnet and _control_img is not None:
                        kwargs["conditioning_frames"] = [_control_img] * total_frames

                if self._cancel_event.is_set():
                    raise GenerationCancelled("AnimateDiff cancelled before inference")

                # GPU-accurate timing: sync before measuring inference
                torch.cuda.synchronize()
                t_pre_inf = time.perf_counter()

                output = pipe(**kwargs)

                # Sync after inference for accurate measurement
                torch.cuda.synchronize()
                t_inference = time.perf_counter() - t_pre_inf

            pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

        finally:
            # Cleanup: disable FreeNoise + FreeInit regardless of success/failure
            if free_noise_active:
                self._animatediff.remove_free_noise(pipe)
            if freeinit_enabled:
                try:
                    pipe.disable_free_init()
                except Exception:
                    pass

        # Frame interpolation (RIFE or blend fallback)
        if settings.enable_frame_interpolation and getattr(req, 'interpolation_factor', None) and len(pil_frames) >= 2:
            try:
                pil_frames = _interpolate_frames(pil_frames, req.interpolation_factor)
                total_frames = len(pil_frames)
                log.info("Frame interpolation: %dx → %d frames", req.interpolation_factor, total_frames)
            except Exception as e:
                log.warning("Frame interpolation failed: %s — using original frames", e)

        # Post-process and encode all frames
        frame_count = 0
        t_post_start = time.perf_counter()
        for frame_idx, pil_img in enumerate(pil_frames):
            if frame_idx >= total_frames:
                break
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff cancelled during post-processing")
            t0_frame = time.perf_counter()
            image = postprocess_apply(pil_img, req.post_process)
            raw_bytes = encode_image_raw_bytes(image)
            w, h = image.size
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            frame_time_ms = int((time.perf_counter() - t0_frame) * 1000)
            log.debug("AnimateDiff frame %d: post-process %dms, total %dms",
                       frame_idx, frame_time_ms, elapsed_ms)

            frame_resp = AnimationFrameResponse(
                frame_index=frame_idx,
                total_frames=total_frames,
                image="",
                seed=seed,
                time_ms=elapsed_ms,
                width=w,
                height=h,
                encoding="raw_rgba",
            )
            frame_resp._raw_bytes = raw_bytes
            frame_count += 1
            if on_frame:
                on_frame(frame_resp)

        t_post = time.perf_counter() - t_post_start
        t_setup = t_pre_inf - t0
        t_total = time.perf_counter() - t0
        log.info("AnimateDiff timing: setup=%.1fs, inference=%.1fs, post=%.1fs, total=%.1fs",
                 t_setup, t_inference, t_post, t_total)

        return frame_count

