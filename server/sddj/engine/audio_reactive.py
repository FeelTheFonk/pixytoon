"""Audio-reactive generation — chain + AnimateDiff methods with per-frame modulation."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.compiler
from PIL import Image

from ..config import settings
from ..postprocess import apply as postprocess_apply, is_processing_active
from ..protocol import (
    AnimationMethod,
    AudioReactiveFrameResponse,
    AudioReactiveRequest,
    GenerationMode,
    ProgressResponse,
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
    _apply_hue_shift,
    apply_frame_motion,
    apply_noise_injection,
    apply_temporal_coherence,
    build_prompt_schedule,
    compute_effective_denoise,
    inject_prompt_kwargs,
    make_step_callback,
    resolve_frame_prompt,
    scale_steps_for_denoise,
)

log = logging.getLogger("sddj.engine")

_ANIMATEDIFF_CHUNK_SIZE = 16
_ANIMATEDIFF_OVERLAP = 4


class AudioReactiveMixin:
    """Audio-reactive generation methods for DiffusionEngine."""

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
        _original_scheduler = None

        try:
            # Handle LoRA
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

            # 1. Analyze audio
            analysis = self.analyze_audio(req.audio_path, req.fps, req.enable_stems)

            # 1b. Auto-generate prompt segments from audio structure
            if getattr(req, 'randomness', 0) > 0 and not getattr(req, 'prompt_schedule', None):
                from ..prompt_schedule import auto_generate_segments
                from ..prompt_generator import prompt_generator
                auto_segs = auto_generate_segments(
                    analysis, req.randomness, req.prompt, prompt_generator,
                    locked_fields=getattr(req, 'locked_fields', None),
                )
                if auto_segs:
                    from ..protocol import PromptScheduleSpec
                    req = req.model_copy(update={"prompt_schedule": PromptScheduleSpec(**auto_segs)})
                    log.info("Auto-mapped %d prompt keyframes from audio (randomness=%d)",
                             len(auto_segs.get("keyframes", [])), req.randomness)

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
            # Restore original scheduler if overridden
            if _original_scheduler is not None:
                self._pipe.scheduler = _original_scheduler
                if self._img2img_pipe is not None:
                    self._img2img_pipe.scheduler = type(_original_scheduler).from_config(
                        _original_scheduler.config)

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
                            self._controlnet_pipe, self._deepcache_helper,
                            self._controlnet_img2img_pipe):
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

        # Pre-compute loop-invariant flags
        _pp_active = is_processing_active(req.post_process)

        # Pre-compute TI suffix once (avoids per-frame TI token iteration in blend path)
        _ti_suffix = self._build_ti_suffix(req.negative_ti)

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

        # Build prompt schedule (if segments or spec provided)
        prompt_sched = build_prompt_schedule(req)

        # Reuse a single CUDA generator (reseed per frame — avoids per-frame allocation)
        _generator = torch.Generator("cuda")

        # S3: Create scheduler once and reuse via set_timesteps() instead of
        # from_config() per frame. Diffusers mutates scheduler state in-place,
        # so we reset via set_timesteps() each frame instead of rebuilding.
        _frame_scheduler = type(self._pipe.scheduler).from_config(self._pipe.scheduler.config)

        # S2: Pre-allocate buffers for noise injection (avoids per-frame heap allocation)
        _noise_buf = np.empty((target_h, target_w, 3), dtype=np.float32)
        _work_buf = np.empty_like(_noise_buf)
        # F-C1: Pre-allocate uint8 output buffer for zero-alloc final conversion
        _out_buf = np.empty((target_h, target_w, 3), dtype=np.uint8)

        # F-O6: EquiVDM temporally coherent noise state
        _prev_noise: np.ndarray | None = None
        _equivdm_flow: np.ndarray | None = None

        # M9: Cache raw_bytes for cadence-skip frames (avoids redundant encode)
        _last_raw_bytes = None

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
                        hue_shift = frame_params.get("palette_shift", 0.0)
                        if _pp_active or hue_shift > 0.01:
                            image = chain_source.copy()
                            if _pp_active:
                                image = postprocess_apply(image, req.post_process)
                            if hue_shift > 0.01:
                                image = _apply_hue_shift(image, hue_shift)
                            raw_bytes = encode_image_raw_bytes(image)
                        elif _last_raw_bytes is not None:
                            # M9: No post-processing or hue shift — reuse cached raw_bytes
                            raw_bytes = _last_raw_bytes
                            image = chain_source
                        else:
                            image = chain_source  # zero-copy: tobytes() is non-mutating
                            raw_bytes = encode_image_raw_bytes(image)
                        w, h = image.size
                        elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)
                        frame_resp = AudioReactiveFrameResponse(
                            frame_index=frame_idx, total_frames=total_frames,
                            image="", seed=0, time_ms=elapsed_ms,
                            width=w, height=h, params_used=frame_params,
                            encoding="raw_rgba",
                        )
                        frame_resp._raw_bytes = raw_bytes
                        if on_frame:
                            on_frame(frame_resp)
                        frame_count += 1
                        log.debug("Audio frame %d/%d: cadence skip (reuse)", frame_idx, total_frames)
                        continue

                # Per-frame LoRA weight modulation (between pipeline calls, not per-step)
                # NOTE: Currently a no-op — LoRAFuser fuses weights and unloads the PEFT
                # adapter, so set_adapters() has nothing to modulate. The except clause
                # handles this gracefully. Will work if/when we switch to unfused PEFT mode.
                if "lora_weight" in frame_params and self._lora_fuser.current_name:
                    try:
                        lw = frame_params["lora_weight"]
                        self._pipe.set_adapters(
                            [self._lora_fuser.current_name],
                            [max(0.0, min(2.0, lw))],
                        )
                    except Exception:
                        pass  # expected: fused LoRA has no active PEFT adapter

                eff_denoise = frame_params.get("denoise_strength", req.denoise_strength)
                _raw_denoise = eff_denoise
                # Sub-floor blending: guarantee ≥2 effective denoising steps
                # while preserving full audio dynamic range.
                eff_denoise, eff_scaled_steps, _sub_floor_alpha = compute_effective_denoise(req.steps, eff_denoise)
                eff_cfg = frame_params.get("cfg_scale", req.cfg_scale)
                seed_offset = int(frame_params.get("seed_offset", frame_idx))
                frame_seed = (base_seed + seed_offset) % (2**32)

                generator = _generator
                generator.manual_seed(frame_seed)

                # Resolve prompt for this frame (keyframe-based with SLERP blending)
                fp = resolve_frame_prompt(
                    prompt_sched, frame_idx,
                    base_prompt=req.prompt,
                    base_negative=effective_neg,
                    ti_suffix=_ti_suffix,
                    pipe_for_embed=(
                        self._pipe if frame_idx == 0 else self._img2img_pipe
                    ),
                    clip_skip=req.clip_skip,
                    audio_fps=audio_fps,
                )
                frame_prompt = fp.prompt
                frame_neg = fp.negative
                blend_embeds = fp.blend_embeds
                if fp.denoise_strength is not None:
                    eff_denoise, eff_scaled_steps, _sub_floor_alpha = compute_effective_denoise(
                        req.steps, fp.denoise_strength)
                    _raw_denoise = fp.denoise_strength
                if fp.cfg_scale is not None:
                    eff_cfg = fp.cfg_scale
                if fp.steps is not None:
                    eff_scaled_steps = scale_steps_for_denoise(fp.steps, eff_denoise)

                # Progress callback
                step_callback = make_step_callback(
                    self._cancel_event, on_progress, req.steps,
                    frame_idx=frame_idx, total_frames=total_frames,
                )

                # S3: Reset scheduler for frame 1+ via set_timesteps() instead of
                # rebuilding from_config() each frame. Assign once, then reset state.
                if frame_idx > 0:
                    self._img2img_pipe.scheduler = _frame_scheduler
                    _frame_scheduler.set_timesteps(eff_scaled_steps)

                log.debug("Audio frame %d/%d: seed=%d, denoise=%.3f, cfg=%.2f, steps=%d (scaled=%d)%s",
                          frame_idx, total_frames, frame_seed, eff_denoise, eff_cfg, req.steps, eff_scaled_steps,
                          f", blend={_sub_floor_alpha:.2f}" if _sub_floor_alpha < 1.0 else "")

                # Frame 0: initial generation
                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        gen_kwargs = dict(
                            num_inference_steps=req.steps,
                            guidance_scale=eff_cfg,
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
                            num_inference_steps=eff_scaled_steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                        image = self._img2img_pipe(**gen_kwargs).images[0]
                        if _sub_floor_alpha < 1.0:
                            image = Image.blend(_source_img, image, _sub_floor_alpha)
                    elif req.mode == GenerationMode.INPAINT:
                        if _source_img is None or _mask_img is None:
                            raise ValueError("inpaint requires source_image and mask_image")
                        gen_kwargs = dict(
                            image=_source_img,
                            num_inference_steps=eff_scaled_steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        )
                        inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                        inpainted = self._img2img_pipe(**gen_kwargs).images[0]
                        if _sub_floor_alpha < 1.0:
                            inpainted = Image.blend(_source_img, inpainted, _sub_floor_alpha)
                        image = composite_with_mask(_source_img, inpainted, _mask_img)
                    elif req.mode.value.startswith("controlnet_"):
                        if _control_img is None:
                            raise ValueError("controlnet requires control_image")
                        self._ensure_controlnet(req.mode)
                        cn_scale = max(0.0, min(2.0, frame_params.get("controlnet_scale", 1.0)))
                        gen_kwargs = dict(
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
                        )
                        inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                        image = self._controlnet_pipe(**gen_kwargs).images[0]
                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                else:
                    # Frame 1+: img2img from previous frame
                    if chain_source is None:
                        raise RuntimeError("Audio chain failed: no previous frame")
                    source = chain_source  # pipeline output already matches target dims

                    # Motion warp: Deforum-like smooth camera (applied BEFORE img2img)
                    source = apply_frame_motion(source, frame_params, _raw_denoise)

                    # Noise amplitude modulation (S2: pre-allocated buffers, F-C1: zero-alloc output)
                    # F-O6: pass prev_noise + flow_map for EquiVDM temporally coherent noise
                    source, _prev_noise = apply_noise_injection(
                        source, frame_params, frame_seed, _raw_denoise,
                        noise_buf=_noise_buf, work_buf=_work_buf,
                        out_buf=_out_buf,
                        prev_noise=_prev_noise, flow_map=_equivdm_flow,
                    )

                    if req.mode.value.startswith("controlnet_") and _control_img is not None:
                        log.info("Audio frame %d: ControlNet mode uses img2img for frame coherence", frame_idx)

                    gen_kwargs = dict(
                        image=source,
                        num_inference_steps=eff_scaled_steps,
                        guidance_scale=eff_cfg,
                        strength=eff_denoise,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    )
                    inject_prompt_kwargs(gen_kwargs, blend_embeds, frame_prompt, frame_neg)
                    image = self._img2img_pipe(**gen_kwargs).images[0]
                    if _sub_floor_alpha < 1.0:
                        image = Image.blend(source, image, _sub_floor_alpha)

                # Temporal coherence: color matching + optical flow (frame 1+)
                # F-O6: request flow map for EquiVDM noise warping on next frame
                _equivdm_flow = None
                if frame_idx > 0 and chain_source is not None:
                    image, _equivdm_flow = apply_temporal_coherence(
                        image, chain_source, return_flow=True,
                    )

                # Store pre-postprocess for next frame
                chain_source = image

                # Post-process
                if _pp_active:
                    image = postprocess_apply(image, req.post_process)

                # Audio-driven palette shift (hue rotation)
                hue_shift = frame_params.get("palette_shift", 0.0)
                if hue_shift > 0.01:
                    image = _apply_hue_shift(image, hue_shift)

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Audio-reactive cancelled during post-processing")

                raw_bytes = encode_image_raw_bytes(image)
                _last_raw_bytes = raw_bytes  # M9: cache for cadence-skip reuse
                w, h = image.size
                elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)

                frame_resp = AudioReactiveFrameResponse(
                    frame_index=frame_idx,
                    total_frames=total_frames,
                    image="",
                    seed=frame_seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    params_used=frame_params,
                    encoding="raw_rgba",
                )
                frame_resp._raw_bytes = raw_bytes
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

        NOTE: Audio-reactive keeps chunk loop (per-chunk parameter modulation is
        the feature). FreeNoise within-chunk coherence is handled by the pipeline.
        """
        # Lightning + audio-reactive: chunked generation handles long sequences.
        # Each chunk is _ANIMATEDIFF_CHUNK_SIZE (16) frames, well within
        # Lightning's per-batch limit.  No total-frame cap needed here —
        # the standard AnimateDiff path rejects because it relies on FreeNoise
        # (incompatible with distilled models), but audio-reactive uses its
        # own chunking strategy instead.
        if settings.is_animatediff_lightning:
            max_lt = settings.animatediff_max_frames_lightning
            if _ANIMATEDIFF_CHUNK_SIZE > max_lt:
                raise ValueError(
                    f"AnimateDiff-Lightning: chunk size {_ANIMATEDIFF_CHUNK_SIZE} "
                    f"exceeds per-batch limit {max_lt}. Reduce chunk size."
                )
            log.info("AnimateDiff-Lightning audio: %d total frames via %d-frame chunks "
                     "(within %d-frame Lightning limit)",
                     schedule.total_frames, _ANIMATEDIFF_CHUNK_SIZE, max_lt)
        is_controlnet = req.mode.value.startswith("controlnet_")
        if not is_controlnet and self._controlnet_pipe is not None:
            log.info("Smart transition: unloading ControlNet before AnimateDiff audio")
            self._controlnet_pipe = None
            self._controlnet_mode = None

        # Mode-aware DeepCache: only toggles on actual mode transition
        # (consistent with animation.py AnimateDiff path)
        if self._dc_state is not None:
            self._dc_state.suppress_for("animatediff")
        try:
            return self._generate_audio_animatediff_inner(
                req, analysis, schedule, on_frame, on_progress,
            )
        finally:
            if self._dc_state is not None:
                try:
                    self._dc_state.restore()
                except Exception as e:
                    log.warning("DeepCache restore failed after audio AnimateDiff (non-critical): %s", e)

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

        # C-02: Reuse a single CUDA generator (reseed per chunk — avoids per-chunk allocation)
        _ad_generator = torch.Generator("cuda")

        total_frames = schedule.total_frames
        chunk_size = _ANIMATEDIFF_CHUNK_SIZE
        overlap = _ANIMATEDIFF_OVERLAP

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

        # Prompt schedule (if segments or spec provided)
        prompt_sched = build_prompt_schedule(req)

        # Overlap buffer — only holds frames pending overlap blending (not all N)
        frame_images: dict[int, Image.Image] = {}
        frame_seeds: dict[int, int] = {}
        frame_count_ad = 0
        t0_total = time.perf_counter()
        prev_ad_image: Optional[Image.Image] = None
        _last_emitted = -1

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
            generator = _ad_generator
            generator.manual_seed(chunk_seed)

            # Resolve prompt for chunk midpoint (keyframe-based with fallback)
            mid_frame = c_start + num_frames // 2
            if prompt_sched and prompt_sched.keyframes:
                blend_info = prompt_sched.get_blend_info_for_frame(mid_frame)
                chunk_prompt = blend_info.effective_prompt or req.prompt
            elif prompt_sched and prompt_sched.segments:
                chunk_prompt = prompt_sched.get_prompt(mid_frame / analysis.fps)
            else:
                chunk_prompt = req.prompt

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
                    # C-19: Overlap region: alpha blend with corrected formula.
                    # overlap_pos / overlap never reaches 1.0 (last overlap frame
                    # gets alpha = (overlap-1)/overlap ≈ 0.75 for overlap=4).
                    # Fix: divide by (overlap - 1) so the last overlap frame = 1.0.
                    overlap_pos = local_idx  # 0..overlap-1
                    alpha = overlap_pos / max(overlap - 1, 1)
                    alpha = min(alpha, 1.0)
                    prev = frame_images[global_idx]
                    blended = Image.blend(prev, pil_img, alpha)
                    frame_images[global_idx] = blended
                else:
                    frame_images[global_idx] = pil_img

                frame_seeds[global_idx] = chunk_seed

            # ── Streaming emit: post-process finalized frames immediately ──
            # After overlap blending, frames before the next chunk's start won't
            # be modified by any future chunk — emit them now and free memory.
            if chunk_idx < len(chunks) - 1:
                _safe_boundary = chunks[chunk_idx + 1][0]
            else:
                _safe_boundary = total_frames

            for frame_idx in range(_last_emitted + 1, _safe_boundary):
                if self._cancel_event.is_set():
                    raise GenerationCancelled("AnimateDiff audio cancelled during post-processing")
                if frame_idx not in frame_images:
                    continue

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
                    raw_bytes = encode_image_raw_bytes(image)
                    w, h = image.size
                    elapsed_ms = int((time.perf_counter() - t0_total) * 1000)
                    frame_resp = AudioReactiveFrameResponse(
                        frame_index=frame_idx, total_frames=total_frames,
                        image="", seed=frame_seeds.get(frame_idx, base_seed),
                        time_ms=elapsed_ms, width=w, height=h,
                        params_used=frame_params,
                        encoding="raw_rgba",
                    )
                    frame_resp._raw_bytes = raw_bytes
                    frame_count_ad += 1
                    if on_frame:
                        on_frame(frame_resp)
                    log.debug("AnimateDiff audio frame %d: cadence skip (reuse)", frame_idx)
                    del frame_images[frame_idx]
                    frame_seeds.pop(frame_idx, None)
                    continue

                pil_img = frame_images[frame_idx]

                ad_denoise = frame_params.get("denoise_strength", req.denoise_strength)
                pil_img = apply_frame_motion(pil_img, frame_params, ad_denoise)

                # Noise amplitude injection (parity with chain loop)
                frame_seed_ad = frame_seeds.get(frame_idx, base_seed)
                pil_img, _ = apply_noise_injection(pil_img, frame_params, frame_seed_ad, ad_denoise)

                # Temporal coherence (parity with chain loop)
                if frame_idx > 0 and prev_ad_image is not None:
                    pil_img = apply_temporal_coherence(pil_img, prev_ad_image)

                image = postprocess_apply(pil_img, req.post_process)

                # Audio-driven palette shift (hue rotation)
                hue_shift = frame_params.get("palette_shift", 0.0)
                if hue_shift > 0.01:
                    image = _apply_hue_shift(image, hue_shift)

                prev_ad_image = image

                raw_bytes = encode_image_raw_bytes(image)
                w, h = image.size
                elapsed_ms = int((time.perf_counter() - t0_total) * 1000)

                frame_resp = AudioReactiveFrameResponse(
                    frame_index=frame_idx,
                    total_frames=total_frames,
                    image="",
                    seed=frame_seeds.get(frame_idx, base_seed),
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    params_used=frame_params,
                    encoding="raw_rgba",
                )
                frame_resp._raw_bytes = raw_bytes
                frame_count_ad += 1
                if on_frame:
                    on_frame(frame_resp)

                del frame_images[frame_idx]
                frame_seeds.pop(frame_idx, None)

            _last_emitted = _safe_boundary - 1

        # Disable FreeInit if it was enabled
        if freeinit_enabled:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        return frame_count_ad
