"""SOTA Diffusion Engine — SD1.5 + Hyper-SD (fused) + DeepCache + FreeU v2 + AnimateDiff.

Orchestrates pipeline lifecycle, LoRA fusing, ControlNet lazy-loading,
AnimateDiff motion module, and progress callbacks for the WebSocket server.

Delegates construction and lifecycle concerns to:
  - pipeline_factory: pipeline construction, attention, scheduler, compile
  - lora_fuser: LoRA fuse/unfuse/dynamo-reset
  - animatediff_manager: AnimateDiff adapter + pipeline lifecycle
  - deepcache_manager: DeepCache enable/disable context manager
  - freeu_applicator: FreeU v2 application
  - image_codec: base64 encode/decode, resize, round8
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, Optional

import torch
import torch.compiler

from diffusers import StableDiffusionPipeline

from ..config import settings
from ..lora_manager import list_loras
from ..ti_manager import list_embeddings, resolve_embedding_path
from ..postprocess import apply as postprocess_apply
from ..protocol import (
    GenerateRequest,
    GenerationMode,
    ProgressResponse,
    ResultResponse,
)
from .. import rembg_wrapper
from ..vram_utils import move_to_cpu, vram_cleanup, vram_log

# Extracted modules
from .. import deepcache_manager
from .. import pipeline_factory
from ..animatediff_manager import AnimateDiffManager
from ..freeu_applicator import apply_freeu
from ..image_codec import (
    composite_with_mask,
    decode_b64_image,
    decode_b64_mask,
    encode_image_b64,
    resize_to_target,
    round8,
)
from ..lora_fuser import LoRAFuser
from .helpers import GenerationCancelled, scale_steps_for_denoise
from .animation import AnimationMixin
from .audio_reactive import AudioReactiveMixin

log = logging.getLogger("sddj.engine")


class DiffusionEngine(AnimationMixin, AudioReactiveMixin):
    """Manages the full SD1.5 pipeline with SOTA optimizations."""

    def __init__(self) -> None:
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._controlnet_mode: Optional[GenerationMode] = None
        self._deepcache_helper = None
        self._lora_fuser = LoRAFuser()
        self._animatediff = AnimateDiffManager()
        self._loaded_ti_tokens: set[str] = set()
        self._loaded = False
        self._cancel_event = threading.Event()
        # Audio reactivity modules (lazy, no GPU)
        self._audio_analyzer = None
        self._audio_cache = None
        self._stem_separator = None
        self._modulation_engine = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def cancel(self) -> None:
        """Signal cancellation — checked at each step callback."""
        self._cancel_event.set()

    def get_status(self) -> dict:
        """Return engine status for /health endpoint."""
        models = []
        if self._pipe is not None:
            models.append("base")
        if self._img2img_pipe is not None:
            models.append("img2img")
        if self._controlnet_pipe is not None:
            models.append(f"controlnet:{self._controlnet_mode.value if self._controlnet_mode else 'unknown'}")
        if self._animatediff.pipe is not None:
            models.append("animatediff")
        return {
            "loaded_models": models,
            "current_lora": self._lora_fuser.current_name,
            "lora_weight": self._lora_fuser.current_weight,
            "deepcache_active": self._deepcache_helper is not None,
        }

    # ─── LIFECYCLE ───────────────────────────────────────────

    def load(self) -> None:
        """Load pipeline with all SOTA optimizations. Retries once on failure."""
        if self._loaded:
            return
        for attempt in range(2):
            try:
                self._load_inner()
                return
            except Exception as e:
                if attempt == 0:
                    log.error("Engine load failed (attempt 1), cleanup + retry: %s", e)
                    vram_cleanup()
                else:
                    raise

    def _load_inner(self) -> None:
        """Internal load — called by load() with retry wrapper."""
        t0 = time.perf_counter()

        # 0. Enable TF32 + high matmul precision (Ampere+, ~15-30% free speedup)
        if torch.cuda.is_available() and settings.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            log.info("TF32 + high matmul precision enabled")

        vram_log("pre-load")

        # 1. Base pipeline
        self._pipe = pipeline_factory.load_base_pipeline()

        # 2. VRAM optimizations
        pipeline_factory.setup_attention(self._pipe)
        pipeline_factory.setup_vae(self._pipe)

        # 3. Hyper-SD LoRA — fuse permanently
        pipeline_factory.setup_hyper_sd(self._pipe)

        # 4. FreeU v2
        apply_freeu(self._pipe)
        if settings.enable_freeu:
            log.info("FreeU v2 enabled (s1=%.1f s2=%.1f b1=%.1f b2=%.1f)",
                     settings.freeu_s1, settings.freeu_s2,
                     settings.freeu_b1, settings.freeu_b2)

        # 5. torch.compile — BEFORE DeepCache
        pipeline_factory.apply_torch_compile(self._pipe)

        # 6. DeepCache — AFTER torch.compile
        self._deepcache_helper = deepcache_manager.create_helper(self._pipe)

        # 7. Create img2img pipeline from same components
        self._img2img_pipe = pipeline_factory.create_img2img_pipeline(self._pipe)

        elapsed = time.perf_counter() - t0
        log.info("Pipeline loaded in %.1fs", elapsed)
        self._loaded = True

        if settings.is_animatediff_lightning:
            log.info(
                "AnimateDiff-Lightning ready: %d-step, CFG=%.1f, FreeU=%s",
                settings.animatediff_lightning_steps,
                settings.animatediff_lightning_cfg,
                "on" if settings.animatediff_lightning_freeu else "OFF",
            )

        # 8. Default style LoRA — fuse BEFORE warmup
        self._load_default_style_lora()

        # 9. Load textual inversion embeddings
        self._load_embeddings()

        # 10. Warmup
        if settings.enable_warmup:
            self._warmup()

    def _load_default_style_lora(self) -> None:
        """Load and fuse the default style LoRA before warmup.

        This ensures torch.compile graphs include the correct fused weights.
        Without this, the first real generation with a LoRA would trigger
        a Dynamo reset and full recompilation (~20s penalty).
        """
        lora_name = settings.default_style_lora
        if not lora_name:
            return

        if lora_name == "auto":
            available = list_loras()
            if not available:
                log.info("No style LoRAs found in %s — skipping", settings.loras_dir)
                return
            lora_name = available[0]

        try:
            self.set_style_lora(lora_name, settings.default_style_lora_weight)
            log.info("Default style LoRA loaded: %s (weight=%.2f)",
                     lora_name, settings.default_style_lora_weight)
        except Exception as e:
            log.warning("Failed to load default style LoRA '%s': %s", lora_name, e)

    def _warmup(self) -> None:
        """Pre-compile the torch.compile graph with a dummy generation.

        CRITICAL: The warmup MUST run in the EXACT same pipeline state as real
        generation.  DeepCache wraps every UNet block's forward() with caching
        closures (enable → wrap_modules).  If warmup runs without DeepCache,
        torch.compile traces the original forwards; when DeepCache re-enables
        afterward, all forward functions change → dynamo guard failure → full
        recompilation on the first real generate() call (~15-25 s penalty).

        Requirements for graph stability:
          - DeepCache ENABLED (same wrapped forwards as real generation)
          - Parameters matching defaults (steps, CFG, resolution, clip_skip)
          - callback_on_step_end provided (parity with real gen path)
        """
        log.info("Warmup: triggering torch.compile + JIT compilation...")
        t0 = time.perf_counter()

        try:
            with torch.inference_mode():
                gen = torch.Generator("cuda").manual_seed(0)
                torch.compiler.cudagraph_mark_step_begin()

                def _noop_callback(pipe, step_idx, timestep, cb_kwargs):
                    return cb_kwargs

                self._pipe(
                    prompt="warmup",
                    negative_prompt="warmup",
                    num_inference_steps=settings.default_steps,
                    guidance_scale=settings.default_cfg,
                    width=settings.default_width,
                    height=settings.default_height,
                    generator=gen,
                    clip_skip=settings.default_clip_skip,
                    callback_on_step_end=_noop_callback,
                    output_type="pil",
                )
            elapsed = time.perf_counter() - t0
            log.info("Warmup complete in %.1fs", elapsed)
        except Exception as e:
            log.warning("Warmup generation failed (non-critical): %s", e)

        # Flush DeepCache's stale cached_output from warmup — prevents
        # dummy features from leaking into the first real generation.
        if self._deepcache_helper is not None:
            self._deepcache_helper.cached_output = {}
            self._deepcache_helper.start_timestep = None
            self._deepcache_helper.cur_timestep = 0

    def _load_embeddings(self) -> None:
        """Load all textual inversion embeddings from embeddings_dir."""
        if not settings.embeddings_dir.is_dir():
            log.info("No embeddings directory found at %s — skipping TI", settings.embeddings_dir)
            return

        available = list_embeddings()
        if not available:
            log.info("No TI embeddings found in %s", settings.embeddings_dir)
            return

        for name in available:
            try:
                path = resolve_embedding_path(name)
                self._pipe.load_textual_inversion(
                    str(path),
                    token=name,
                )
                self._loaded_ti_tokens.add(name)
                log.info("Loaded TI embedding: %s", name)
            except Exception as e:
                log.warning("Failed to load TI embedding '%s': %s", name, e)

        log.info("Loaded %d TI embedding(s): %s",
                 len(self._loaded_ti_tokens), sorted(self._loaded_ti_tokens))

    def unload(self) -> None:
        """Free all GPU memory — .to(cpu) before nullifying for immediate VRAM release."""
        move_to_cpu(self._pipe)
        move_to_cpu(self._img2img_pipe)
        move_to_cpu(self._controlnet_pipe)
        self._pipe = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._controlnet_mode = None
        self._deepcache_helper = None
        self._lora_fuser = LoRAFuser()
        self._loaded_ti_tokens.clear()
        self._loaded = False
        self._animatediff.unload()
        rembg_wrapper.unload()
        vram_cleanup()
        vram_log("post-unload")
        log.info("Pipeline unloaded")

    def cleanup_resources(self) -> dict:
        """Free optional GPU resources (ControlNet, AnimateDiff, rembg).

        Keeps the base pipeline loaded. Returns freed VRAM info.
        """
        freed_before = 0.0
        freed_after = 0.0
        try:
            if torch.cuda.is_available():
                freed_before = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        except Exception:
            pass

        cleaned = []

        if self._controlnet_pipe is not None:
            move_to_cpu(self._controlnet_pipe)
            self._controlnet_pipe = None
            self._controlnet_mode = None
            cleaned.append("ControlNet")

        if self._animatediff.pipe is not None:
            self._animatediff.unload()
            cleaned.append("AnimateDiff")

        rembg_wrapper.unload()

        if self._stem_separator is not None:
            self._stem_separator.unload()
            cleaned.append("StemSeparator")

        vram_cleanup()

        try:
            if torch.cuda.is_available():
                freed_after = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        except Exception:
            pass

        freed_mb = max(0.0, freed_after - freed_before)
        msg = f"Cleaned: {', '.join(cleaned) if cleaned else 'cache only'}"
        log.info("Resource cleanup: freed %.1f MB (%s)", freed_mb, msg)
        return {"freed_mb": round(freed_mb, 1), "message": msg}

    # ─── LORA MANAGEMENT ────────────────────────────────────

    def set_style_lora(self, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch style LoRA (fused into weights, no PEFT runtime)."""
        if not self._loaded or self._pipe is None:
            return
        weight = max(-2.0, min(2.0, weight))
        self._lora_fuser.set_lora(self._pipe, name, weight)

    # ─── CONTROLNET ──────────────────────────────────────────

    def _ensure_controlnet(self, mode: GenerationMode) -> None:
        """Lazy-load ControlNet model for the requested mode."""
        if self._controlnet_mode == mode and self._controlnet_pipe is not None:
            return

        # Smart transition: free AnimateDiff if loaded (reclaim VRAM)
        if self._animatediff.pipe is not None:
            log.info("Smart transition: unloading AnimateDiff before ControlNet load")
            self._animatediff.unload()

        # VRAM budget guard: cleanup if low on memory before loading ControlNet
        from ..vram_utils import check_vram_budget
        if not check_vram_budget(required_mb=800, min_free_mb=settings.vram_min_free_mb):
            log.warning("Low VRAM before ControlNet load — running cleanup")
            vram_cleanup()

        # Unload previous
        move_to_cpu(self._controlnet_pipe)
        self._controlnet_pipe = None
        vram_cleanup()

        self._controlnet_pipe = pipeline_factory.create_controlnet_pipeline(
            self._pipe, mode,
        )
        self._controlnet_mode = mode

    # ─── GENERATION ──────────────────────────────────────────

    def _build_effective_negative(self, neg: str, ti_specs: list | None) -> str:
        """Build final negative prompt with TI embedding tokens."""
        effective = neg or ""
        if not ti_specs:
            return effective
        ti_parts = []
        for ti_spec in ti_specs:
            if ti_spec.name in self._loaded_ti_tokens:
                if abs(ti_spec.weight - 1.0) < 0.01:
                    ti_parts.append(ti_spec.name)
                else:
                    ti_parts.append(f"({ti_spec.name}:{ti_spec.weight:.2f})")
            else:
                log.warning("TI token '%s' not loaded — skipping", ti_spec.name)
        if ti_parts:
            ti_str = ", ".join(ti_parts)
            effective = f"{effective}, {ti_str}" if effective else ti_str
        return effective

    def generate(
        self,
        req: GenerateRequest,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> ResultResponse:
        """Run the full generation pipeline: diffusion → post-process → encode."""
        if not self._loaded:
            self.load()

        self._cancel_event.clear()

        try:
            with torch.inference_mode():
                t0 = time.perf_counter()

                # Resolve seed
                seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
                seed = seed % (2**32)  # clamp to valid range
                generator = torch.Generator("cuda").manual_seed(seed)

                # Set style LoRA — only if client explicitly specifies one.
                if req.lora is not None:
                    lora_name = req.lora.name
                    lora_weight = req.lora.weight
                    if (lora_name != self._lora_fuser.current_name
                            or lora_weight != self._lora_fuser.current_weight):
                        self.set_style_lora(lora_name, lora_weight)

                effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)

                # Progress callback adapter with cancellation support
                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    if self._cancel_event.is_set():
                        raise GenerationCancelled("Generation cancelled by client")
                    if on_progress:
                        on_progress(ProgressResponse(step=step_idx + 1, total=req.steps))
                    return callback_kwargs

                # ── Mode dispatch ─────────────────────────────────────

                if req.mode == GenerationMode.TXT2IMG:
                    image = self._txt2img(req, generator, step_callback, effective_neg)

                elif req.mode == GenerationMode.IMG2IMG:
                    image = self._img2img(req, generator, step_callback, effective_neg)

                elif req.mode == GenerationMode.INPAINT:
                    image = self._inpaint(req, generator, step_callback, effective_neg)

                elif req.mode.value.startswith("controlnet_"):
                    image = self._controlnet_generate(req, generator, step_callback, effective_neg)

                else:
                    raise ValueError(f"Unknown mode: {req.mode}")

                # ── Post-process ──────────────────────────────────────
                image = postprocess_apply(image, req.post_process)

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Cancelled during post-processing")

                # ── Encode result ─────────────────────────────────────
                b64_image = encode_image_b64(image)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                w, h = image.size

                return ResultResponse(
                    image=b64_image,
                    seed=seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                )

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM — clearing VRAM cache")
            vram_cleanup()
            raise
        finally:
            self._cancel_event.clear()

    # ─── PRIVATE GENERATION METHODS ──────────────────────────

    def _txt2img(self, req, generator, callback, effective_neg):
        torch.compiler.cudagraph_mark_step_begin()
        return self._pipe(
            prompt=req.prompt,
            negative_prompt=effective_neg,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=round8(req.width),
            height=round8(req.height),
            generator=generator,
            clip_skip=req.clip_skip,
            callback_on_step_end=callback,
            output_type="pil",
        ).images[0]

    def _img2img(self, req, generator, callback, effective_neg):
        if req.source_image is None:
            raise ValueError("img2img requires source_image")
        source = decode_b64_image(req.source_image)
        source = source.convert("RGB")
        target_w, target_h = round8(req.width), round8(req.height)
        source = resize_to_target(source, target_w, target_h)
        # Clamp: ensure at least 2 effective denoising steps (1 step produces poor quality).
        # Use capped scaled steps for the floor so low denoise isn't killed.
        _cap = max(settings.distilled_step_scale_cap, 1)
        min_denoise = min(1.0, 2.0 / max(req.steps * _cap, 1) + 1e-3)
        strength = max(req.denoise_strength, min_denoise)
        scaled_steps = scale_steps_for_denoise(req.steps, strength)
        torch.compiler.cudagraph_mark_step_begin()
        # DeepCache hooks reference the txt2img scheduler's timesteps, but
        # img2img builds its own truncated schedule → timestep lookup crash.
        # Suspend DeepCache for the img2img call to avoid the mismatch.
        with deepcache_manager.suspended(self._deepcache_helper):
            return self._img2img_pipe(
                prompt=req.prompt,
                negative_prompt=effective_neg,
                image=source,
                num_inference_steps=scaled_steps,
                guidance_scale=req.cfg_scale,
                strength=strength,
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=callback,
                output_type="pil",
            ).images[0]

    def _inpaint(self, req, generator, callback, effective_neg):
        """Inpaint: img2img full image, then composite via mask."""
        if req.source_image is None:
            raise ValueError("inpaint requires source_image")
        if req.mask_image is None:
            raise ValueError("inpaint requires mask_image")

        source = decode_b64_image(req.source_image)
        source = source.convert("RGB")
        mask = decode_b64_mask(req.mask_image)
        target_w, target_h = round8(req.width), round8(req.height)
        source = resize_to_target(source, target_w, target_h)
        mask = resize_to_target(mask, target_w, target_h)

        # Run img2img on the full source (model sees context for coherent inpainting)
        # Clamp: ensure at least 2 effective denoising steps (1 step produces poor quality).
        _cap = max(settings.distilled_step_scale_cap, 1)
        min_denoise = min(1.0, 2.0 / max(req.steps * _cap, 1) + 1e-3)
        strength = max(req.denoise_strength, min_denoise)
        scaled_steps = scale_steps_for_denoise(req.steps, strength)
        torch.compiler.cudagraph_mark_step_begin()
        # Same DeepCache suspension as _img2img — shared UNet, different scheduler.
        with deepcache_manager.suspended(self._deepcache_helper):
            inpainted = self._img2img_pipe(
                prompt=req.prompt,
                negative_prompt=effective_neg,
                image=source,
                num_inference_steps=scaled_steps,
                guidance_scale=req.cfg_scale,
                strength=strength,
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=callback,
                output_type="pil",
            ).images[0]

        # Composite: keep original where mask is black, use inpainted where white
        return composite_with_mask(source, inpainted, mask)

    def _controlnet_generate(self, req, generator, callback, effective_neg):
        if req.control_image is None:
            raise ValueError("ControlNet requires control_image")

        self._ensure_controlnet(req.mode)
        control = decode_b64_image(req.control_image)
        control = control.convert("RGB")
        width = round8(req.width)
        height = round8(req.height)
        control = resize_to_target(control, width, height)

        torch.compiler.cudagraph_mark_step_begin()
        return self._controlnet_pipe(
            prompt=req.prompt,
            negative_prompt=effective_neg,
            image=control,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=width,
            height=height,
            generator=generator,
            clip_skip=req.clip_skip,
            callback_on_step_end=callback,
            output_type="pil",
        ).images[0]
