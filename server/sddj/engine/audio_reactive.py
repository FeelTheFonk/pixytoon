"""Audio-reactive generation — chain + AnimateDiff methods with per-frame modulation."""

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
    AnimationMethod,
    AudioReactiveFrameResponse,
    AudioReactiveRequest,
    GenerationMode,
    ProgressResponse,
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
from .helpers import GenerationCancelled, _apply_hue_shift, apply_frame_motion, apply_noise_injection, apply_temporal_coherence, compute_effective_denoise, make_step_callback, scale_steps_for_denoise

log = logging.getLogger("sddj.engine")


class AudioReactiveMixin:
    """Audio-reactive generation methods for DiffusionEngine."""

    _ANIMATEDIFF_CHUNK_SIZE = 16
    _ANIMATEDIFF_OVERLAP = 4

    def _ensure_audio_modules(self):
        """Lazy-init audio modules (no GPU required)."""
        if self._audio_analyzer is None:
            from ..audio_analyzer import AudioAnalyzer
            from ..audio_cache import AudioCache
            from ..stem_separator import StemSeparator
            from ..modulation_engine import ModulationEngine
            self._audio_analyzer = AudioAnalyzer()
            self._audio_cache = AudioCache(cache_dir=settings.audio_cache_dir)
            self._stem_separator = StemSeparator(
                model_name=settings.stem_model,
                device=settings.stem_device,
            )
            self._modulation_engine = ModulationEngine()

    def stems_available(self) -> bool:
        """Check if stem separation is available."""
        from ..stem_separator import is_available
        return is_available()

    def analyze_audio(self, audio_path: str, fps: float,
                      enable_stems: bool = False):
        """Analyze audio file, returning features. Uses cache when available."""
        self._ensure_audio_modules()
        # Check cache first (invalidate old format without raw_features)
        cached = self._audio_cache.get(audio_path, fps, enable_stems)
        if cached is not None:
            if cached.raw_features:
                return cached
            log.info("Cache outdated (no raw_features), re-analyzing: %s", audio_path)
            self._audio_cache.invalidate(audio_path, fps, enable_stems)

        # Analyze with optional stem separation
        stems = None
        if enable_stems and self._stem_separator.is_available():
            log.info("Separating stems for: %s", audio_path)
            stems = self._stem_separator.separate(audio_path)

        analysis = self._audio_analyzer.analyze(
            audio_path, fps, stems=stems,
            attack_frames=settings.audio_default_attack,
            release_frames=settings.audio_default_release,
        )

        # Enforce max frames
        if analysis.total_frames > settings.audio_max_frames:
            log.warning("Audio too long: %d frames (max %d). Truncating.",
                       analysis.total_frames, settings.audio_max_frames)
            analysis.total_frames = settings.audio_max_frames
            for name in analysis.features:
                analysis.features[name] = analysis.features[name][:settings.audio_max_frames]
            for name in analysis.raw_features:
                analysis.raw_features[name] = analysis.raw_features[name][:settings.audio_max_frames]

        self._audio_cache.put(audio_path, fps, analysis, enable_stems)
        return analysis

    def generate_audio_reactive(
        self,
        req: AudioReactiveRequest,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]] = None,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> int:
        """Generate audio-reactive animation — chain animation with per-frame parameter modulation."""
        if not self._loaded:
            self.load()

        self._cancel_event.clear()
        self._ensure_audio_modules()

        try:
            # Handle LoRA
            if req.lora is not None:
                if (req.lora.name != self._lora_fuser.current_name
                        or req.lora.weight != self._lora_fuser.current_weight):
                    self.set_style_lora(req.lora.name, req.lora.weight)

            # 1. Analyze audio
            analysis = self.analyze_audio(req.audio_path, req.fps, req.enable_stems)

            # 1b. Auto-generate prompt segments from audio structure
            if getattr(req, 'randomness', 0) > 0 and not req.prompt_segments:
                from ..prompt_schedule import auto_generate_segments
                from ..prompt_generator import prompt_generator
                auto_segs = auto_generate_segments(
                    analysis, req.randomness, req.prompt, prompt_generator,
                    locked_fields=getattr(req, 'locked_fields', None),
                )
                if auto_segs:
                    req.prompt_segments = auto_segs
                    log.info("Auto-generated %d prompt segments (randomness=%d)",
                             len(auto_segs), req.randomness)

            # 2. Resolve modulation slots
            from ..modulation_engine import ModulationSlot, ModulationEngine
            if req.modulation_preset:
                slots = ModulationEngine.get_preset(req.modulation_preset)
            else:
                slots = [
                    ModulationSlot(
                        source=s.source, target=s.target,
                        min_val=s.min_val, max_val=s.max_val,
                        attack=s.attack, release=s.release,
                        enabled=s.enabled, invert=s.invert,
                    )
                    for s in req.modulation_slots
                ]

            # 3. Validate expressions if provided
            if req.expressions:
                errors = self._modulation_engine.validate_expressions(
                    req.expressions, analysis.feature_names,
                )
                if errors:
                    raise ValueError(f"Invalid expressions: {errors}")

            # 4. Compute parameter schedule
            schedule = self._modulation_engine.compute_schedule(
                analysis, slots, req.expressions,
            )

            # 5. Honour max_frames limit if set
            if req.max_frames and req.max_frames > 0 and schedule.total_frames > req.max_frames:
                schedule.total_frames = req.max_frames
                schedule.frame_params = schedule.frame_params[:req.max_frames]

            # 6. Run animation with per-frame params — dispatch by method
            if req.method == AnimationMethod.ANIMATEDIFF_AUDIO:
                return self._generate_audio_animatediff(
                    req, analysis, schedule, on_frame, on_progress,
                )
            else:
                return self._generate_audio_chain(
                    req, schedule, on_frame, on_progress,
                    audio_fps=analysis.fps,
                )

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM during audio-reactive generation — clearing VRAM cache")
            vram_cleanup()
            raise
        finally:
            self._cancel_event.clear()

    def _generate_audio_chain(
        self,
        req: AudioReactiveRequest,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
        audio_fps: float = 24.0,
    ) -> int:
        """Audio-reactive chain animation — uses eager_pipeline context manager."""
        with eager_pipeline(self._pipe, self._img2img_pipe,
                            self._controlnet_pipe, self._deepcache_helper):
            return self._generate_audio_chain_inner(req, schedule, on_frame, on_progress, audio_fps)

    @torch.compiler.disable
    def _generate_audio_chain_inner(
        self,
        req: AudioReactiveRequest,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
        audio_fps: float = 24.0,
    ) -> int:
        """Core audio-reactive chain loop — per-frame parameter modulation."""
        frame_count = 0
        base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        base_seed = base_seed % (2**32)
        chain_source: Optional[Image.Image] = None
        total_frames = schedule.total_frames

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        # Pre-decode base64 images
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

        # Build prompt schedule (if segments provided)
        from ..prompt_schedule import PromptSchedule
        prompt_sched = PromptSchedule.from_dicts(
            getattr(req, "prompt_segments", []), req.prompt,
        )

        log.info("Audio-reactive chain: %d frames, mode=%s, steps=%d, seed_base=%d",
                 total_frames, req.mode.value, req.steps, base_seed)

        with torch.inference_mode():
            for frame_idx in range(total_frames):
                if self._cancel_event.is_set():
                    raise GenerationCancelled("Audio-reactive animation cancelled")

                t0_frame = time.perf_counter()

                # Get modulated parameters for this frame
                frame_params = schedule.get_params(frame_idx)

                # Frame cadence: skip frames to save GPU time
                cadence = max(1, int(frame_params.get("frame_cadence", 1.0)))
                if cadence > 1 and frame_idx > 0 and (frame_idx % cadence) != 0:
                    # Reuse the last generated image (chain_source already set)
                    if chain_source is not None:
                        image = chain_source.copy()
                        image = postprocess_apply(image, req.post_process)
                        hue_shift = frame_params.get("palette_shift", 0.0)
                        if hue_shift > 0.01:
                            image = _apply_hue_shift(image, hue_shift)
                        b64_image = encode_image_raw_b64(image)
                        w, h = image.size
                        elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)
                        frame_resp = AudioReactiveFrameResponse(
                            frame_index=frame_idx, total_frames=total_frames,
                            image=b64_image, seed=0, time_ms=elapsed_ms,
                            width=w, height=h, params_used=frame_params,
                            encoding="raw_rgba",
                        )
                        if on_frame:
                            on_frame(frame_resp)
                        frame_count += 1
                        log.debug("Audio frame %d/%d: cadence skip (reuse)", frame_idx, total_frames)
                        continue

                eff_denoise = frame_params.get("denoise_strength", req.denoise_strength)
                _raw_denoise = eff_denoise
                # Sub-floor blending: guarantee ≥2 effective denoising steps
                # while preserving full audio dynamic range.
                eff_denoise, eff_scaled_steps, _sub_floor_alpha = compute_effective_denoise(req.steps, eff_denoise)
                eff_cfg = frame_params.get("cfg_scale", req.cfg_scale)
                seed_offset = int(frame_params.get("seed_offset", frame_idx))
                frame_seed = (base_seed + seed_offset) % (2**32)

                generator = torch.Generator("cuda").manual_seed(frame_seed)

                # Resolve prompt for this frame (may vary with prompt schedule)
                frame_prompt = (
                    prompt_sched.get_prompt(frame_idx / audio_fps)
                    if prompt_sched else req.prompt
                )

                # Progress callback
                step_callback = make_step_callback(
                    self._cancel_event, on_progress, req.steps,
                    frame_idx=frame_idx, total_frames=total_frames,
                )

                # Reset scheduler for frame 1+
                if frame_idx > 0:
                    self._img2img_pipe.scheduler = pipeline_factory.fresh_scheduler(self._pipe)

                log.info("Audio frame %d/%d: seed=%d, denoise=%.3f, cfg=%.2f, steps=%d (scaled=%d)%s",
                         frame_idx, total_frames, frame_seed, eff_denoise, eff_cfg, req.steps, eff_scaled_steps,
                         f", blend={_sub_floor_alpha:.2f}" if _sub_floor_alpha < 1.0 else "")

                # Frame 0: initial generation
                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        image = self._pipe(
                            prompt=frame_prompt,
                            negative_prompt=effective_neg,
                            num_inference_steps=req.steps,
                            guidance_scale=eff_cfg,
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
                            negative_prompt=effective_neg,
                            image=_source_img,
                            num_inference_steps=eff_scaled_steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                        if _sub_floor_alpha < 1.0:
                            image = Image.blend(_source_img, image, _sub_floor_alpha)
                    elif req.mode == GenerationMode.INPAINT:
                        if _source_img is None or _mask_img is None:
                            raise ValueError("inpaint requires source_image and mask_image")
                        inpainted = self._img2img_pipe(
                            prompt=frame_prompt,
                            negative_prompt=effective_neg,
                            image=_source_img,
                            num_inference_steps=eff_scaled_steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                        if _sub_floor_alpha < 1.0:
                            inpainted = Image.blend(_source_img, inpainted, _sub_floor_alpha)
                        image = composite_with_mask(_source_img, inpainted, _mask_img)
                    elif req.mode.value.startswith("controlnet_"):
                        if _control_img is None:
                            raise ValueError("controlnet requires control_image")
                        self._ensure_controlnet(req.mode)
                        cn_scale = max(0.0, min(2.0, frame_params.get("controlnet_scale", 1.0)))
                        image = self._controlnet_pipe(
                            prompt=frame_prompt,
                            negative_prompt=effective_neg,
                            image=_control_img,
                            num_inference_steps=req.steps,
                            guidance_scale=eff_cfg,
                            width=target_w,
                            height=target_h,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            controlnet_conditioning_scale=cn_scale,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                else:
                    # Frame 1+: img2img from previous frame
                    if chain_source is None:
                        raise RuntimeError("Audio chain failed: no previous frame")
                    source = resize_to_target(chain_source, target_w, target_h)

                    # Motion warp: Deforum-like smooth camera (applied BEFORE img2img)
                    source = apply_frame_motion(source, frame_params, _raw_denoise)

                    # Noise amplitude modulation
                    source = apply_noise_injection(source, frame_params, frame_seed, _raw_denoise)

                    if req.mode.value.startswith("controlnet_") and _control_img is not None:
                        log.info("Audio frame %d: ControlNet mode uses img2img for frame coherence", frame_idx)

                    image = self._img2img_pipe(
                        prompt=frame_prompt,
                        negative_prompt=effective_neg,
                        image=source,
                        num_inference_steps=eff_scaled_steps,
                        guidance_scale=eff_cfg,
                        strength=eff_denoise,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    ).images[0]
                    if _sub_floor_alpha < 1.0:
                        image = Image.blend(source, image, _sub_floor_alpha)

                # Temporal coherence: color matching + optical flow (frame 1+)
                if frame_idx > 0 and chain_source is not None:
                    image = apply_temporal_coherence(image, chain_source)

                # Store pre-postprocess for next frame
                chain_source = image

                # Post-process
                image = postprocess_apply(image, req.post_process)

                # Audio-driven palette shift (hue rotation)
                hue_shift = frame_params.get("palette_shift", 0.0)
                if hue_shift > 0.01:
                    image = _apply_hue_shift(image, hue_shift)

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Audio-reactive cancelled during post-processing")

                b64_image = encode_image_raw_b64(image)
                w, h = image.size
                elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)

                frame_resp = AudioReactiveFrameResponse(
                    frame_index=frame_idx,
                    total_frames=total_frames,
                    image=b64_image,
                    seed=frame_seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    params_used=frame_params,
                    encoding="raw_rgba",
                )
                frame_count += 1
                if on_frame:
                    on_frame(frame_resp)

        return frame_count

    # ── AnimateDiff + Audio ──────────────────────────────────

    def _generate_audio_animatediff(
        self,
        req: AudioReactiveRequest,
        analysis,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Audio-reactive AnimateDiff — temporal consistency via chunked batches.

        Divides the audio timeline into overlapping 16-frame AnimateDiff chunks.
        Each chunk uses averaged modulation parameters from the schedule.
        Overlap frames are alpha-blended for smooth transitions between chunks.
        """
        is_controlnet = req.mode.value.startswith("controlnet_")
        if not is_controlnet and self._controlnet_pipe is not None:
            log.info("Smart transition: unloading ControlNet before AnimateDiff audio")
            self._controlnet_pipe = None
            self._controlnet_mode = None

        with deepcache_manager.suspended(self._deepcache_helper):
            return self._generate_audio_animatediff_inner(
                req, analysis, schedule, on_frame, on_progress,
            )

    def _generate_audio_animatediff_inner(
        self,
        req: AudioReactiveRequest,
        analysis,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> int:
        """Core AnimateDiff audio loop — chunked with overlap blending."""
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

        # FreeInit only for first chunk (not supported on vid2vid or Lightning)
        if settings.is_animatediff_lightning:
            if req.enable_freeinit:
                log.warning("FreeInit disabled: incompatible with AnimateDiff-Lightning (distilled model)")
            freeinit_enabled = False
        else:
            freeinit_enabled = req.enable_freeinit and not is_img2img

        # Lightning parameter enforcement
        if settings.is_animatediff_lightning:
            lightning_steps = settings.animatediff_lightning_steps
            lightning_cfg = settings.animatediff_lightning_cfg
            log.info("AnimateDiff-Lightning audio: enforcing steps=%d, cfg=%.1f",
                     lightning_steps, lightning_cfg)
        else:
            lightning_steps = None
            lightning_cfg = None

        base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        base_seed = base_seed % (2**32)

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        target_w, target_h = round8(req.width), round8(req.height)

        total_frames = schedule.total_frames
        chunk_size = self._ANIMATEDIFF_CHUNK_SIZE
        overlap = self._ANIMATEDIFF_OVERLAP

        # Build chunk ranges: [(start, end), ...]
        chunks: list[tuple[int, int]] = []
        pos = 0
        while pos < total_frames:
            end = min(pos + chunk_size, total_frames)
            # Ensure last chunk has at least overlap+1 frames
            if end - pos < overlap + 1 and chunks:
                # Extend previous chunk instead
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, end)
            else:
                chunks.append((pos, end))
            pos = end - overlap if end < total_frames else end

        log.info("AnimateDiff audio: %d total frames, %d chunks (chunk=%d, overlap=%d)",
                 total_frames, len(chunks), chunk_size, overlap)

        # Pre-decode images
        _control_img = None
        if is_controlnet and req.control_image is not None:
            _control_img = decode_b64_image(req.control_image).convert("RGB")
            _control_img = resize_to_target(_control_img, target_w, target_h)

        _source_img = None
        if is_img2img and req.source_image is not None:
            _source_img = decode_b64_image(req.source_image).convert("RGB")
            _source_img = resize_to_target(_source_img, target_w, target_h)
        elif is_img2img:
            raise ValueError("AnimateDiff audio img2img requires source_image")

        # Prompt schedule
        from ..prompt_schedule import PromptSchedule
        prompt_sched = PromptSchedule.from_dicts(
            getattr(req, "prompt_segments", []), req.prompt,
        )

        # Results indexed by frame
        frame_images: dict[int, Image.Image] = {}
        frame_seeds: dict[int, int] = {}
        frame_count_ad = 0
        t0_total = time.perf_counter()

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            if self._cancel_event.is_set():
                raise GenerationCancelled("Audio-reactive AnimateDiff cancelled")

            num_frames = c_end - c_start
            chunk_params = schedule.get_chunk_params(c_start, c_end)
            eff_cfg = lightning_cfg if lightning_cfg is not None else chunk_params.get("cfg_scale", req.cfg_scale)
            eff_denoise = chunk_params.get("denoise_strength", req.denoise_strength)
            eff_steps = lightning_steps if lightning_steps is not None else req.steps

            # Seed: base + chunk midpoint offset for variation
            seed_offset = int(chunk_params.get("seed_offset", c_start))
            chunk_seed = (base_seed + seed_offset) % (2**32)
            generator = torch.Generator("cuda").manual_seed(chunk_seed)

            # Resolve prompt for chunk midpoint
            mid_frame = c_start + num_frames // 2
            chunk_prompt = (
                prompt_sched.get_prompt(mid_frame / analysis.fps)
                if prompt_sched else req.prompt
            )

            # FreeInit: only first chunk
            if freeinit_enabled and chunk_idx == 0:
                try:
                    pipe.enable_free_init(
                        num_iters=req.freeinit_iterations,
                        use_fast_sampling=True,
                    )
                    log.info("FreeInit enabled for first chunk (%d iters)", req.freeinit_iterations)
                except Exception as e:
                    log.warning("FreeInit unavailable: %s", e)
            elif freeinit_enabled and chunk_idx == 1:
                try:
                    pipe.disable_free_init()
                except Exception:
                    pass

            # Progress callback
            step_callback = make_step_callback(
                self._cancel_event, on_progress, eff_steps,
                frame_idx=c_start, total_frames=total_frames,
            )

            _ad_sub_floor_alpha = 1.0

            log.info("Chunk %d/%d [%d-%d): %d frames, cfg=%.2f, denoise=%.3f, seed=%d",
                     chunk_idx + 1, len(chunks), c_start, c_end,
                     num_frames, eff_cfg, eff_denoise, chunk_seed)

            with torch.inference_mode():
                if is_img2img:
                    # Sub-floor blending: same logic as chain loop.
                    chunk_strength, chunk_scaled_steps, _ad_sub_floor_alpha = compute_effective_denoise(eff_steps, eff_denoise)

                    kwargs = dict(
                        video=[_source_img] * num_frames,
                        prompt=chunk_prompt,
                        negative_prompt=effective_neg,
                        num_inference_steps=chunk_scaled_steps,
                        guidance_scale=eff_cfg,
                        strength=chunk_strength,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )
                else:
                    kwargs = dict(
                        prompt=chunk_prompt,
                        negative_prompt=effective_neg,
                        num_frames=num_frames,
                        num_inference_steps=eff_steps,
                        guidance_scale=eff_cfg,
                        width=target_w,
                        height=target_h,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )

                    if is_controlnet and _control_img is not None:
                        kwargs["conditioning_frames"] = [_control_img] * num_frames
                        cn_scale = max(0.0, min(2.0, chunk_params.get("controlnet_scale", 1.0)))
                        kwargs["controlnet_conditioning_scale"] = cn_scale

                output = pipe(**kwargs)

            # Extract frames
            pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

            # Sub-floor blending for vid2vid: attenuate toward source
            if _ad_sub_floor_alpha < 1.0 and is_img2img and _source_img is not None:
                pil_frames = [
                    Image.blend(_source_img, f, _ad_sub_floor_alpha) for f in pil_frames
                ]

            # Blend overlap with previous chunk
            for local_idx, pil_img in enumerate(pil_frames):
                global_idx = c_start + local_idx
                if global_idx >= total_frames:
                    break

                if global_idx in frame_images and chunk_idx > 0:
                    # Overlap region: alpha blend
                    overlap_pos = local_idx  # 0..overlap-1
                    alpha = overlap_pos / overlap  # 0→1: fade from prev to new
                    prev = frame_images[global_idx]
                    blended = Image.blend(prev, pil_img, alpha)
                    frame_images[global_idx] = blended
                else:
                    frame_images[global_idx] = pil_img

                frame_seeds[global_idx] = chunk_seed

        # Post-process and encode all frames
        prev_ad_image: Optional[Image.Image] = None
        for frame_idx in sorted(frame_images.keys()):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff audio cancelled during post-processing")

            # Motion warp for AnimateDiff: cumulative per-frame warp
            frame_params = schedule.get_params(frame_idx)

            # Frame cadence: reuse previous output to match chain method behavior
            cadence = max(1, int(frame_params.get("frame_cadence", 1.0)))
            if (cadence > 1 and frame_idx > 0
                    and (frame_idx % cadence) != 0
                    and prev_ad_image is not None):
                image = prev_ad_image.copy()
                hue_shift = frame_params.get("palette_shift", 0.0)
                if hue_shift > 0.01:
                    image = _apply_hue_shift(image, hue_shift)
                b64_image = encode_image_raw_b64(image)
                w, h = image.size
                elapsed_ms = int((time.perf_counter() - t0_total) * 1000)
                frame_resp = AudioReactiveFrameResponse(
                    frame_index=frame_idx, total_frames=total_frames,
                    image=b64_image, seed=frame_seeds.get(frame_idx, base_seed),
                    time_ms=elapsed_ms, width=w, height=h,
                    params_used=frame_params,
                    encoding="raw_rgba",
                )
                frame_count_ad += 1
                if on_frame:
                    on_frame(frame_resp)
                log.debug("AnimateDiff audio frame %d: cadence skip (reuse)", frame_idx)
                continue

            pil_img = frame_images[frame_idx]

            ad_denoise = frame_params.get("denoise_strength", req.denoise_strength)
            pil_img = apply_frame_motion(pil_img, frame_params, ad_denoise)

            # Noise amplitude injection (parity with chain loop)
            frame_seed_ad = frame_seeds.get(frame_idx, base_seed)
            pil_img = apply_noise_injection(pil_img, frame_params, frame_seed_ad, ad_denoise)

            # Temporal coherence (parity with chain loop)
            if frame_idx > 0 and prev_ad_image is not None:
                pil_img = apply_temporal_coherence(pil_img, prev_ad_image)

            image = postprocess_apply(pil_img, req.post_process)

            # Audio-driven palette shift (hue rotation)
            hue_shift = frame_params.get("palette_shift", 0.0)
            if hue_shift > 0.01:
                image = _apply_hue_shift(image, hue_shift)

            prev_ad_image = image

            b64_image = encode_image_raw_b64(image)
            w, h = image.size
            elapsed_ms = int((time.perf_counter() - t0_total) * 1000)

            frame_resp = AudioReactiveFrameResponse(
                frame_index=frame_idx,
                total_frames=total_frames,
                image=b64_image,
                seed=frame_seeds.get(frame_idx, base_seed),
                time_ms=elapsed_ms,
                width=w,
                height=h,
                params_used=frame_params,
                encoding="raw_rgba",
            )
            frame_count_ad += 1
            if on_frame:
                on_frame(frame_resp)

        # Disable FreeInit if it was enabled
        if freeinit_enabled:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        return frame_count_ad
