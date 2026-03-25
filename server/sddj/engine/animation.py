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
from .helpers import GenerationCancelled, apply_temporal_coherence, compute_effective_denoise, make_step_callback, scale_steps_for_denoise

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

                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        image = self._pipe(
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
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
        """Core AnimateDiff generation logic (txt2img, img2img, controlnet)."""
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
        if req.enable_freeinit and not is_img2img:
            if settings.is_animatediff_lightning:
                log.warning("FreeInit disabled: incompatible with AnimateDiff-Lightning (distilled model)")
            else:
                try:
                    pipe.enable_free_init(
                        num_iters=req.freeinit_iterations,
                        use_fast_sampling=True,
                    )
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
        seed = seed % (2**32)  # clamp to valid CUDA generator range
        generator = torch.Generator("cuda").manual_seed(seed)

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        # Progress callback
        step_callback = make_step_callback(
            self._cancel_event, on_progress, effective_steps,
            frame_idx=0, total_frames=req.frame_count,
        )

        t0 = time.perf_counter()

        with torch.inference_mode():
            if is_img2img:
                # AnimateDiff vid2vid: source image → animated frames
                if req.source_image is None:
                    raise ValueError("AnimateDiff img2img requires source_image")
                source = decode_b64_image(req.source_image).convert("RGB")
                source = resize_to_target(source, target_w, target_h)
                # Repeat source image as input "video"
                video_input = [source] * req.frame_count

                strength, scaled_steps, _ = compute_effective_denoise(effective_steps, req.denoise_strength)

                log.info("AnimateDiff img2img: %d frames, steps=%d (scaled=%d), strength=%.2f",
                         req.frame_count, effective_steps, scaled_steps, strength)

                kwargs = dict(
                    video=video_input,
                    prompt=req.prompt,
                    negative_prompt=effective_neg,
                    num_inference_steps=scaled_steps,
                    guidance_scale=effective_cfg,
                    strength=strength,
                    generator=generator,
                    clip_skip=req.clip_skip,
                    callback_on_step_end=step_callback,
                    output_type="pil",
                )
            else:
                # txt2img or controlnet
                kwargs = dict(
                    prompt=req.prompt,
                    negative_prompt=effective_neg,
                    num_frames=req.frame_count,
                    num_inference_steps=effective_steps,
                    guidance_scale=effective_cfg,
                    width=target_w,
                    height=target_h,
                    generator=generator,
                    clip_skip=req.clip_skip,
                    callback_on_step_end=step_callback,
                    output_type="pil",
                )

                if is_controlnet and req.control_image is not None:
                    control = decode_b64_image(req.control_image).convert("RGB")
                    control = resize_to_target(control, target_w, target_h)
                    kwargs["conditioning_frames"] = [control] * req.frame_count

            output = pipe(**kwargs)

        # Extract frames from output
        pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

        # Disable FreeInit for next normal generation
        if req.enable_freeinit and not is_img2img:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        # Post-process and encode each frame
        frame_count = 0
        for frame_idx, pil_img in enumerate(pil_frames):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff cancelled during post-processing")
            t0_frame = time.perf_counter()
            image = postprocess_apply(pil_img, req.post_process)
            b64_image = encode_image_raw_b64(image)
            w, h = image.size
            # time_ms: total elapsed since animation start (intentional —
            # gives the client cumulative progress for the whole batch).
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            # frame_time_ms: time spent encoding/post-processing this individual frame
            frame_time_ms = int((time.perf_counter() - t0_frame) * 1000)
            log.debug("AnimateDiff frame %d: post-process %dms, total %dms",
                       frame_idx, frame_time_ms, elapsed_ms)

            frame_resp = AnimationFrameResponse(
                frame_index=frame_idx,
                total_frames=len(pil_frames),
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
