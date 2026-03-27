"""Animation generation — chain + AnimateDiff methods."""

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
from .. import deepcache_manager
from .. import pipeline_factory
from ..animatediff_manager import get_uncompiled_unet
from .compile_utils import eager_pipeline
from ..vram_utils import vram_cleanup
from ..image_codec import (
    composite_with_mask,
    decode_b64_image,
    decode_b64_mask,
    encode_image_raw_b64,
    resize_to_target,
    round8,
)
from .helpers import GenerationCancelled, apply_temporal_coherence, build_prompt_schedule, compute_effective_denoise, make_step_callback, scale_steps_for_denoise

log = logging.getLogger("sddj.engine")


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

        try:
            # Handle LoRA same as single generation
            if req.lora is not None:
                if (req.lora.name != self._lora_fuser.current_name
                        or req.lora.weight != self._lora_fuser.current_weight):
                    self.set_style_lora(req.lora.name, req.lora.weight)

            if req.method == AnimationMethod.CHAIN:
                return self._generate_chain(req, on_frame, on_progress)
            elif req.method == AnimationMethod.ANIMATEDIFF:
                return self._generate_animatediff(req, on_frame, on_progress)
            else:
                raise ValueError(f"Unknown animation method: {req.method}")

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
                            self._controlnet_pipe, self._deepcache_helper):
            return self._generate_chain_inner(req, on_frame, on_progress)

    @torch.compiler.disable
    def _generate_chain_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
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

                generator = torch.Generator("cuda").manual_seed(frame_seed)

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
                if frame_idx > 0:
                    self._img2img_pipe.scheduler = pipeline_factory.fresh_scheduler(self._pipe)
                    log.debug("Chain frame %d: scheduler reset", frame_idx)

                log.info("Chain frame %d/%d: seed=%d, mode=%s",
                         frame_idx, req.frame_count, frame_seed, req.mode.value)

                # Resolve per-frame prompt from schedule
                frame_prompt = req.prompt
                frame_neg = effective_neg
                if schedule and schedule.keyframes:
                    kf_prompt = schedule.get_prompt_for_frame(frame_idx)
                    if kf_prompt:
                        frame_prompt = kf_prompt
                    kf_neg = schedule.get_negative_for_frame(frame_idx)
                    if kf_neg:
                        frame_neg = self._build_effective_negative(kf_neg, req.negative_ti)

                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        image = self._pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            width=target_w,
                            height=target_h,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    elif req.mode == GenerationMode.IMG2IMG:
                        if _source_img is None:
                            raise ValueError("img2img requires source_image")
                        image = self._img2img_pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            image=_source_img,
                            num_inference_steps=chain_scaled_steps,
                            guidance_scale=req.cfg_scale,
                            strength=chain_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    elif req.mode == GenerationMode.INPAINT:
                        if _source_img is None or _mask_img is None:
                            raise ValueError("inpaint requires source_image and mask_image")
                        inpainted = self._img2img_pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            image=_source_img,
                            num_inference_steps=chain_scaled_steps,
                            guidance_scale=req.cfg_scale,
                            strength=chain_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                        image = composite_with_mask(_source_img, inpainted, _mask_img)
                    elif req.mode.value.startswith("controlnet_"):
                        if _control_img is None:
                            raise ValueError("controlnet requires control_image")
                        self._ensure_controlnet(req.mode)
                        image = self._controlnet_pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            image=_control_img,
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            width=target_w,
                            height=target_h,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                else:
                    # Frame 1+: img2img from previous frame at denoise_strength
                    if chain_source is None:
                        raise RuntimeError("Chain animation failed: previous frame did not produce output")
                    source = resize_to_target(chain_source, target_w, target_h)

                    if req.mode.value.startswith("controlnet_") and _control_img is not None:
                        # ControlNet chain: use img2img from previous frame
                        # (ControlNet pipelines don't support img2img directly,
                        # so we fall through to plain img2img for frame coherence)
                        log.info("Chain frame %d: ControlNet mode uses img2img for frame coherence", frame_idx)
                        image = self._img2img_pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            image=source,
                            num_inference_steps=chain_scaled_steps,
                            guidance_scale=req.cfg_scale,
                            strength=chain_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    else:
                        log.info("Chain frame %d: calling img2img pipe (source=%s, steps=%d (scaled=%d), strength=%.2f, unet=%s, scheduler=%s)",
                                 frame_idx, source.size, req.steps, chain_scaled_steps, chain_denoise,
                                 type(self._img2img_pipe.unet).__name__,
                                 type(self._img2img_pipe.scheduler).__name__)
                        image = self._img2img_pipe(
                            prompt=frame_prompt,
                            negative_prompt=frame_neg,
                            image=source,
                            num_inference_steps=chain_scaled_steps,
                            guidance_scale=req.cfg_scale,
                            strength=chain_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                        log.info("Chain frame %d: img2img complete", frame_idx)

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
                b64_image = encode_image_raw_b64(image)
                w, h = image.size
                # frame_time_ms: per-frame generation time (from this frame's t0_frame)
                elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)

                frame_resp = AnimationFrameResponse(
                    frame_index=frame_idx,
                    total_frames=req.frame_count,
                    image=b64_image,
                    seed=frame_seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    encoding="raw_rgba",
                )
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
        # Smart transition: free ControlNet if not needed for this request
        is_controlnet = req.mode.value.startswith("controlnet_")
        if not is_controlnet and self._controlnet_pipe is not None:
            log.info("Smart transition: unloading ControlNet before AnimateDiff")
            self._controlnet_pipe = None
            self._controlnet_mode = None

        with deepcache_manager.suspended(self._deepcache_helper):
            return self._generate_animatediff_inner(req, on_frame, on_progress)

    def _generate_animatediff_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Core AnimateDiff generation — chunked processing for O(n) scaling.

        Processes frames in _AD_CHUNK_SIZE chunks with _AD_OVERLAP overlap.
        Overlap frames are alpha-blended for smooth chunk transitions.
        Prevents O(n²) temporal attention and VRAM explosion on large batches.
        """
        _AD_CHUNK_SIZE = 16
        _AD_OVERLAP = 4

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

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        # Build prompt schedule for per-chunk resolution
        schedule = build_prompt_schedule(req)

        # Pre-decode images for reuse across chunks
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

        # Build chunk ranges: [(start, end), ...]
        total_frames = req.frame_count
        chunks: list[tuple[int, int]] = []
        pos = 0
        while pos < total_frames:
            end = min(pos + _AD_CHUNK_SIZE, total_frames)
            if end - pos < _AD_OVERLAP + 1 and chunks:
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, end)
            else:
                chunks.append((pos, end))
            pos = end - _AD_OVERLAP if end < total_frames else end

        log.info("AnimateDiff: %d frames → %d chunk(s) (size=%d, overlap=%d)",
                 total_frames, len(chunks), _AD_CHUNK_SIZE, _AD_OVERLAP)

        # Results indexed by global frame index
        frame_images: dict[int, Image.Image] = {}

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff cancelled between chunks")

            num_frames = c_end - c_start
            chunk_seed = (seed + c_start) % (2**32)
            generator = torch.Generator("cuda").manual_seed(chunk_seed)

            step_callback = make_step_callback(
                self._cancel_event, on_progress, effective_steps,
                frame_idx=c_start, total_frames=total_frames,
            )

            # FreeInit: only first chunk
            if freeinit_enabled and chunk_idx == 1:
                try:
                    pipe.disable_free_init()
                except Exception:
                    pass

            log.info("Chunk %d/%d [%d-%d): %d frames, seed=%d",
                     chunk_idx + 1, len(chunks), c_start, c_end, num_frames, chunk_seed)

            # Resolve prompt at chunk midpoint
            chunk_prompt = req.prompt
            chunk_neg = effective_neg
            if schedule and schedule.keyframes:
                mid_frame = c_start + num_frames // 2
                kf_prompt = schedule.get_prompt_for_frame(mid_frame)
                if kf_prompt:
                    chunk_prompt = kf_prompt
                kf_neg = schedule.get_negative_for_frame(mid_frame)
                if kf_neg:
                    chunk_neg = self._build_effective_negative(kf_neg, req.negative_ti)

            with torch.inference_mode():
                if is_img2img:
                    strength, scaled_steps, _ = compute_effective_denoise(
                        effective_steps, req.denoise_strength)
                    kwargs = dict(
                        video=[_source_img] * num_frames,
                        prompt=chunk_prompt,
                        negative_prompt=chunk_neg,
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
                        prompt=chunk_prompt,
                        negative_prompt=chunk_neg,
                        num_frames=num_frames,
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
                        kwargs["conditioning_frames"] = [_control_img] * num_frames

                output = pipe(**kwargs)

            pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

            # Blend overlap with previous chunk
            for local_idx, pil_img in enumerate(pil_frames):
                global_idx = c_start + local_idx
                if global_idx >= total_frames:
                    break
                if global_idx in frame_images and chunk_idx > 0:
                    overlap_pos = local_idx
                    alpha = overlap_pos / _AD_OVERLAP
                    frame_images[global_idx] = Image.blend(
                        frame_images[global_idx], pil_img, alpha)
                else:
                    frame_images[global_idx] = pil_img

        # Disable FreeInit if it was enabled
        if freeinit_enabled:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        # Post-process and encode all frames
        frame_count = 0
        for frame_idx in sorted(frame_images.keys()):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff cancelled during post-processing")
            t0_frame = time.perf_counter()
            image = postprocess_apply(frame_images[frame_idx], req.post_process)
            b64_image = encode_image_raw_b64(image)
            w, h = image.size
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            frame_time_ms = int((time.perf_counter() - t0_frame) * 1000)
            log.debug("AnimateDiff frame %d: post-process %dms, total %dms",
                       frame_idx, frame_time_ms, elapsed_ms)

            frame_resp = AnimationFrameResponse(
                frame_index=frame_idx,
                total_frames=total_frames,
                image=b64_image,
                seed=seed,
                time_ms=elapsed_ms,
                width=w,
                height=h,
                encoding="raw_rgba",
            )
            frame_count += 1
            if on_frame:
                on_frame(frame_resp)

        return frame_count
