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

import gc
import hashlib
import logging
import random
import threading
import time
import warnings
from typing import Callable, Optional

import numpy as np
import torch
import torch.compiler
from PIL import Image

# Suppress known harmless library warnings
warnings.filterwarnings("ignore", message=".*safety checker.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
warnings.filterwarnings("ignore", message=".*No LoRA keys associated to CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")
warnings.filterwarnings("ignore", message=".*Torchinductor does not support code generation for complex.*")
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

from diffusers import StableDiffusionPipeline

from .config import settings
from .lora_manager import list_loras
from .ti_manager import list_embeddings, resolve_embedding_path
from .postprocess import apply as postprocess_apply
from .protocol import (
    AnimationFrameResponse,
    AnimationMethod,
    AnimationRequest,
    AudioReactiveFrameResponse,
    AudioReactiveRequest,
    GenerateRequest,
    GenerationMode,
    PostProcessSpec,
    ProgressResponse,
    RealtimeReadyResponse,
    RealtimeResultResponse,
    RealtimeStartRequest,
    RealtimeStoppedResponse,
    RealtimeUpdateRequest,
    ResultResponse,
    SeedStrategy,
)
from . import rembg_wrapper

# Extracted modules
from . import deepcache_manager
from . import pipeline_factory
from .animatediff_manager import AnimateDiffManager, get_uncompiled_unet
from .freeu_applicator import apply_freeu
from .image_codec import (
    composite_with_mask,
    decode_b64_image,
    decode_b64_mask,
    encode_image_b64,
    resize_to_target,
    round8,
)
from .lora_fuser import LoRAFuser

log = logging.getLogger("pixytoon.engine")


# ─────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────

class GenerationCancelled(Exception):
    """Raised when a client cancels an in-progress generation."""


def _apply_hue_shift(image: Image.Image, shift: float) -> Image.Image:
    """Shift hue of an image by `shift` fraction (0-1 maps to 0-360 degrees).

    Preserves alpha channel. Used for audio-driven palette shift modulation.
    """
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


class RealtimeState:
    """Mutable state for a real-time paint session (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active: bool = False
        self.prompt: str = ""
        self.negative_prompt: str = ""
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.negative_prompt_embeds: Optional[torch.Tensor] = None
        self._prompt_hash: Optional[int] = None
        self.denoise_strength: float = 0.5
        self.steps: int = 4
        self.cfg_scale: float = 2.5
        self.seed: int = -1
        self.clip_skip: int = 2
        self.width: int = 512
        self.height: int = 512
        self.post_process: PostProcessSpec = PostProcessSpec()
        self.frame_counter: int = 0
        self._resolved_seed: int = 0  # actual seed used (resolved from -1)
        self._deepcache_was_active: bool = False
        self._scheduler_cls = None   # cached scheduler class for fast reset
        self._scheduler_cfg = None   # cached scheduler config dict

    def reset(self) -> None:
        self.active = False
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self._prompt_hash = None
        self.frame_counter = 0


class DiffusionEngine:
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
        self._realtime = RealtimeState()
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

    # ─── LIFECYCLE ───────────────────────────────────────────

    def load(self) -> None:
        """Load pipeline with all SOTA optimizations."""
        if self._loaded:
            return

        t0 = time.perf_counter()

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

        # 8. Default pixel art LoRA — fuse BEFORE warmup
        self._load_default_pixel_lora()

        # 9. Load textual inversion embeddings
        self._load_embeddings()

        # 10. Warmup
        if settings.enable_warmup:
            self._warmup()

    def _load_default_pixel_lora(self) -> None:
        """Load and fuse the default pixel art LoRA before warmup.

        This ensures torch.compile graphs include the correct fused weights.
        Without this, the first real generation with a LoRA would trigger
        a Dynamo reset and full recompilation (~20s penalty).
        """
        lora_name = settings.default_pixel_lora
        if not lora_name:
            return

        if lora_name == "auto":
            available = list_loras()
            if not available:
                log.info("No pixel art LoRAs found in %s — skipping", settings.loras_dir)
                return
            lora_name = available[0]

        try:
            self.set_pixel_lora(lora_name, settings.default_pixel_lora_weight)
            log.info("Default pixel art LoRA loaded: %s (weight=%.2f)",
                     lora_name, settings.default_pixel_lora_weight)
        except Exception as e:
            log.warning("Failed to load default pixel art LoRA '%s': %s", lora_name, e)

    def _warmup(self) -> None:
        """Run a dummy txt2img to pre-compile the torch.compile graph.

        NOTE: torch.compile is a one-time cost on first run. Subsequent
        generations reuse the compiled graph and are significantly faster.

        CRITICAL: All parameters must match real generation defaults:
        - guidance_scale: Differs → CFG branch changes (single vs double UNet)
        - width/height: Differs → different tensor shapes → graph recompilation
        - steps: Must exercise enough steps for DeepCache cache/skip branches
        """
        log.info("Warmup: triggering torch.compile + JIT compilation...")
        t0 = time.perf_counter()

        # Disable DeepCache during warmup — warmup runs with dummy prompts
        # and DeepCache caching can produce invalid compiled graph states.
        dc_was_active = self._deepcache_helper is not None
        if dc_was_active:
            deepcache_manager.disable(self._deepcache_helper)

        try:
            with torch.inference_mode():
                gen = torch.Generator("cuda").manual_seed(0)
                torch.compiler.cudagraph_mark_step_begin()
                self._pipe(
                    prompt="warmup",
                    negative_prompt="warmup",
                    num_inference_steps=settings.default_steps,
                    guidance_scale=settings.default_cfg,
                    width=settings.default_width,
                    height=settings.default_height,
                    generator=gen,
                    clip_skip=settings.default_clip_skip,
                    output_type="pil",
                )
            elapsed = time.perf_counter() - t0
            log.info("Warmup complete in %.1fs", elapsed)
        except Exception as e:
            log.warning("Warmup generation failed (non-critical): %s", e)
        finally:
            # Re-enable DeepCache after warmup
            if dc_was_active:
                try:
                    deepcache_manager.enable(self._deepcache_helper)
                except Exception as e:
                    log.error("Failed to re-enable DeepCache after warmup: %s", e)
                    self._deepcache_helper = None

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
        """Free all GPU memory."""
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def set_pixel_lora(self, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch pixel art LoRA (fused into weights, no PEFT runtime)."""
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

        # Unload previous
        self._controlnet_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

                # Set pixel art LoRA — only if client explicitly specifies one.
                if req.lora is not None:
                    lora_name = req.lora.name
                    lora_weight = req.lora.weight
                    if (lora_name != self._lora_fuser.current_name
                            or lora_weight != self._lora_fuser.current_weight):
                        self.set_pixel_lora(lora_name, lora_weight)

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
            gc.collect()
            torch.cuda.empty_cache()
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
        torch.compiler.cudagraph_mark_step_begin()
        return self._img2img_pipe(
            prompt=req.prompt,
            negative_prompt=effective_neg,
            image=source,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            strength=req.denoise_strength,
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
        torch.compiler.cudagraph_mark_step_begin()
        inpainted = self._img2img_pipe(
            prompt=req.prompt,
            negative_prompt=effective_neg,
            image=source,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            strength=req.denoise_strength,
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

    # ─── ANIMATION GENERATION ────────────────────────────────

    def generate_animation(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]] = None,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> list[AnimationFrameResponse]:
        """Generate multi-frame animation — dispatches to chain or animatediff method."""
        if not self._loaded:
            self.load()

        self._cancel_event.clear()

        try:
            # Handle LoRA same as single generation
            if req.lora is not None:
                if (req.lora.name != self._lora_fuser.current_name
                        or req.lora.weight != self._lora_fuser.current_weight):
                    self.set_pixel_lora(req.lora.name, req.lora.weight)

            if req.method == AnimationMethod.CHAIN:
                return self._generate_chain(req, on_frame, on_progress)
            elif req.method == AnimationMethod.ANIMATEDIFF:
                return self._generate_animatediff(req, on_frame, on_progress)
            else:
                raise ValueError(f"Unknown animation method: {req.method}")

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM during animation — clearing VRAM cache")
            gc.collect()
            torch.cuda.empty_cache()
            raise
        finally:
            self._cancel_event.clear()

    # ── Audio Reactivity ──────────────────────────────────────

    def _ensure_audio_modules(self):
        """Lazy-init audio modules (no GPU required)."""
        if self._audio_analyzer is None:
            from .audio_analyzer import AudioAnalyzer
            from .audio_cache import AudioCache
            from .stem_separator import StemSeparator
            from .modulation_engine import ModulationEngine
            self._audio_analyzer = AudioAnalyzer()
            self._audio_cache = AudioCache(cache_dir=settings.audio_cache_dir)
            self._stem_separator = StemSeparator(
                model_name=settings.stem_model,
                device=settings.stem_device,
            )
            self._modulation_engine = ModulationEngine()

    def stems_available(self) -> bool:
        """Check if stem separation is available."""
        from .stem_separator import is_available
        return is_available()

    def analyze_audio(self, audio_path: str, fps: float,
                      enable_stems: bool = False):
        """Analyze audio file, returning features. Uses cache when available."""
        self._ensure_audio_modules()
        # Check cache first
        cached = self._audio_cache.get(audio_path, fps, enable_stems)
        if cached is not None:
            return cached

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

        self._audio_cache.put(audio_path, fps, analysis, enable_stems)
        return analysis

    def generate_audio_reactive(
        self,
        req: AudioReactiveRequest,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]] = None,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> list[AudioReactiveFrameResponse]:
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
                    self.set_pixel_lora(req.lora.name, req.lora.weight)

            # 1. Analyze audio
            analysis = self.analyze_audio(req.audio_path, req.fps, req.enable_stems)

            # 2. Resolve modulation slots
            from .modulation_engine import ModulationSlot, ModulationEngine
            if req.modulation_preset:
                slots = ModulationEngine.get_preset(req.modulation_preset)
            else:
                slots = [
                    ModulationSlot(
                        source=s.source, target=s.target,
                        min_val=s.min_val, max_val=s.max_val,
                        attack=s.attack, release=s.release,
                        enabled=s.enabled,
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

            # 5. Run animation with per-frame params — dispatch by method
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
            gc.collect()
            torch.cuda.empty_cache()
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
    ) -> list[AudioReactiveFrameResponse]:
        """Audio-reactive chain animation — wraps raw UNet like _generate_chain."""
        raw_unet = get_uncompiled_unet(self._pipe)
        compiled_unet = self._pipe.unet

        with deepcache_manager.suspended(self._deepcache_helper):
            self._pipe.unet = raw_unet
            self._img2img_pipe.unet = raw_unet
            if self._controlnet_pipe is not None:
                self._controlnet_pipe.unet = raw_unet
            try:
                torch._dynamo.reset()
                return self._generate_audio_chain_inner(req, schedule, on_frame, on_progress, audio_fps)
            finally:
                try:
                    torch._dynamo.reset()
                except Exception:
                    log.warning("torch._dynamo.reset() failed in audio chain cleanup")
                self._pipe.unet = compiled_unet
                self._img2img_pipe.unet = compiled_unet
                if self._controlnet_pipe is not None:
                    self._controlnet_pipe.unet = compiled_unet

    @torch.compiler.disable
    def _generate_audio_chain_inner(
        self,
        req: AudioReactiveRequest,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
        audio_fps: float = 24.0,
    ) -> list[AudioReactiveFrameResponse]:
        """Core audio-reactive chain loop — per-frame parameter modulation."""
        frames: list[AudioReactiveFrameResponse] = []
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
        from .prompt_schedule import PromptSchedule
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
                        b64_image = encode_image_b64(image)
                        w, h = image.size
                        elapsed_ms = int((time.perf_counter() - t0_frame) * 1000)
                        frame_resp = AudioReactiveFrameResponse(
                            frame_index=frame_idx, total_frames=total_frames,
                            image=b64_image, seed=0, time_ms=elapsed_ms,
                            width=w, height=h, params_used=frame_params,
                        )
                        if on_frame:
                            on_frame(frame_resp)
                        frames.append(image)
                        log.debug("Audio frame %d/%d: cadence skip (reuse)", frame_idx, total_frames)
                        continue

                eff_denoise = frame_params.get("denoise_strength", req.denoise_strength)
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
                def step_callback(pipe, step_idx, timestep, callback_kwargs,
                                  _fi=frame_idx, _fc=total_frames):
                    if self._cancel_event.is_set():
                        raise GenerationCancelled("Audio-reactive animation cancelled")
                    if on_progress:
                        on_progress(ProgressResponse(
                            step=step_idx + 1, total=req.steps,
                            frame_index=_fi, total_frames=_fc,
                        ))
                    return callback_kwargs

                # Reset scheduler for frame 1+
                if frame_idx > 0:
                    self._img2img_pipe.scheduler = pipeline_factory.fresh_scheduler(self._pipe)

                log.info("Audio frame %d/%d: seed=%d, denoise=%.3f, cfg=%.2f",
                         frame_idx, total_frames, frame_seed, eff_denoise, eff_cfg)

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
                            num_inference_steps=req.steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
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
                            negative_prompt=effective_neg,
                            image=_source_img,
                            num_inference_steps=req.steps,
                            guidance_scale=eff_cfg,
                            strength=eff_denoise,
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
                        cn_scale = frame_params.get("controlnet_scale", 1.0)
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

                    # Noise amplitude modulation: inject noise into source
                    noise_amp = frame_params.get("noise_amplitude", 0.0)
                    if noise_amp > 0:
                        arr = np.array(source, dtype=np.float32) / 255.0
                        noise = np.random.default_rng(frame_seed).standard_normal(
                            arr.shape, dtype=np.float32) * noise_amp
                        arr = np.clip(arr + noise, 0.0, 1.0)
                        source = Image.fromarray((arr * 255).astype(np.uint8))

                    if req.mode.value.startswith("controlnet_") and _control_img is not None:
                        log.info("Audio frame %d: ControlNet mode uses img2img for frame coherence", frame_idx)

                    image = self._img2img_pipe(
                        prompt=frame_prompt,
                        negative_prompt=effective_neg,
                        image=source,
                        num_inference_steps=req.steps,
                        guidance_scale=eff_cfg,
                        strength=eff_denoise,
                        generator=generator,
                        clip_skip=req.clip_skip,
                        callback_on_step_end=step_callback,
                        output_type="pil",
                    ).images[0]

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

                b64_image = encode_image_b64(image)
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
                )
                frames.append(frame_resp)
                if on_frame:
                    on_frame(frame_resp)

        return frames

    # ── AnimateDiff + Audio ──────────────────────────────────

    _ANIMATEDIFF_CHUNK_SIZE = 16
    _ANIMATEDIFF_OVERLAP = 4

    def _generate_audio_animatediff(
        self,
        req: AudioReactiveRequest,
        analysis,
        schedule,
        on_frame: Optional[Callable[[AudioReactiveFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AudioReactiveFrameResponse]:
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
    ) -> list[AudioReactiveFrameResponse]:
        """Core AnimateDiff audio loop — chunked with overlap blending."""
        is_controlnet = req.mode.value.startswith("controlnet_")

        if is_controlnet:
            existing_cn = pipeline_factory.get_controlnet_from_pipe(
                self._controlnet_pipe, self._controlnet_mode, req.mode,
            )
            pipe = self._animatediff.ensure_controlnet(
                self._pipe, req.mode, existing_controlnet=existing_cn,
            )
        else:
            pipe = self._animatediff.ensure_base(self._pipe)

        # FreeInit only for first chunk
        freeinit_enabled = req.enable_freeinit

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

        # Pre-decode control image if needed
        _control_img = None
        if is_controlnet and req.control_image is not None:
            _control_img = decode_b64_image(req.control_image).convert("RGB")
            _control_img = resize_to_target(_control_img, target_w, target_h)

        # Prompt schedule
        from .prompt_schedule import PromptSchedule
        prompt_sched = PromptSchedule.from_dicts(
            getattr(req, "prompt_segments", []), req.prompt,
        )

        # Results indexed by frame
        frame_images: dict[int, Image.Image] = {}
        frame_seeds: dict[int, int] = {}
        all_frames: list[AudioReactiveFrameResponse] = []
        t0_total = time.perf_counter()

        for chunk_idx, (c_start, c_end) in enumerate(chunks):
            if self._cancel_event.is_set():
                raise GenerationCancelled("Audio-reactive AnimateDiff cancelled")

            num_frames = c_end - c_start
            chunk_params = schedule.get_chunk_params(c_start, c_end)
            eff_cfg = chunk_params.get("cfg_scale", req.cfg_scale)
            eff_denoise = chunk_params.get("denoise_strength", req.denoise_strength)

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
            def step_callback(pipe_ref, step_idx, timestep, callback_kwargs,
                              _ci=chunk_idx, _cs=c_start, _ce=c_end):
                if self._cancel_event.is_set():
                    raise GenerationCancelled("Audio-reactive AnimateDiff cancelled")
                if on_progress:
                    on_progress(ProgressResponse(
                        step=step_idx + 1, total=req.steps,
                        frame_index=_cs, total_frames=total_frames,
                    ))
                return callback_kwargs

            log.info("Chunk %d/%d [%d-%d): %d frames, cfg=%.2f, denoise=%.3f, seed=%d",
                     chunk_idx + 1, len(chunks), c_start, c_end,
                     num_frames, eff_cfg, eff_denoise, chunk_seed)

            with torch.inference_mode():
                kwargs = dict(
                    prompt=chunk_prompt,
                    negative_prompt=effective_neg,
                    num_frames=num_frames,
                    num_inference_steps=req.steps,
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

                output = pipe(**kwargs)

            # Extract frames
            pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

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
        for frame_idx in sorted(frame_images.keys()):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff audio cancelled during post-processing")

            t0_frame = time.perf_counter()
            pil_img = frame_images[frame_idx]
            image = postprocess_apply(pil_img, req.post_process)

            # Audio-driven palette shift (hue rotation)
            frame_params = schedule.get_params(frame_idx)
            hue_shift = frame_params.get("palette_shift", 0.0)
            if hue_shift > 0.01:
                image = _apply_hue_shift(image, hue_shift)

            b64_image = encode_image_b64(image)
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
            )
            all_frames.append(frame_resp)
            if on_frame:
                on_frame(frame_resp)

        # Disable FreeInit if it was enabled
        if freeinit_enabled:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        return all_frames

    def _generate_chain(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
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
        raw_unet = get_uncompiled_unet(self._pipe)
        compiled_unet = self._pipe.unet

        with deepcache_manager.suspended(self._deepcache_helper):
            # After DeepCache.disable(), raw UNet's forwards are restored to original.
            # Swap both pipelines to use the clean raw UNet directly.
            self._pipe.unet = raw_unet
            self._img2img_pipe.unet = raw_unet
            if self._controlnet_pipe is not None:
                self._controlnet_pipe.unet = raw_unet
            try:
                # Purge dynamo code cache to prevent stale guard recompilation
                # on the raw UNet. Without this, dynamo recognizes _orig_mod
                # from prior compilation and may hang on recompilation.
                torch._dynamo.reset()
                return self._generate_chain_inner(req, on_frame, on_progress)
            finally:
                try:
                    torch._dynamo.reset()
                except Exception:
                    log.warning("torch._dynamo.reset() failed in chain cleanup")
                # Restore compiled UNet before DeepCache re-enables
                self._pipe.unet = compiled_unet
                self._img2img_pipe.unet = compiled_unet
                if self._controlnet_pipe is not None:
                    self._controlnet_pipe.unet = compiled_unet

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
        frames: list[AnimationFrameResponse] = []
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

        log.info("Chain animation: %d frames, mode=%s, steps=%d, denoise=%.2f, seed_base=%d",
                 req.frame_count, req.mode.value, req.steps, req.denoise_strength, base_seed)

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
                # Default args capture frame_idx by value (closure would capture by reference)
                def step_callback(pipe, step_idx, timestep, callback_kwargs, _fi=frame_idx, _fc=req.frame_count):
                    if self._cancel_event.is_set():
                        raise GenerationCancelled("Animation cancelled by client")
                    if on_progress:
                        on_progress(ProgressResponse(
                            step=step_idx + 1, total=req.steps,
                            frame_index=_fi, total_frames=_fc,
                        ))
                    return callback_kwargs

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
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            strength=req.denoise_strength,
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
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            strength=req.denoise_strength,
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
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            strength=req.denoise_strength,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                    else:
                        log.info("Chain frame %d: calling img2img pipe (source=%s, steps=%d, strength=%.2f, unet=%s, scheduler=%s)",
                                 frame_idx, source.size, req.steps, req.denoise_strength,
                                 type(self._img2img_pipe.unet).__name__,
                                 type(self._img2img_pipe.scheduler).__name__)
                        image = self._img2img_pipe(
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
                            image=source,
                            num_inference_steps=req.steps,
                            guidance_scale=req.cfg_scale,
                            strength=req.denoise_strength,
                            generator=generator,
                            clip_skip=req.clip_skip,
                            callback_on_step_end=step_callback,
                            output_type="pil",
                        ).images[0]
                        log.info("Chain frame %d: img2img complete", frame_idx)

                # Store pre-postprocess image for next frame's img2img source
                # (full-resolution, RGB, no pixelation/quantization artifacts)
                chain_source = image

                # Post-process
                image = postprocess_apply(image, req.post_process)

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Animation cancelled during post-processing")

                # Encode
                b64_image = encode_image_b64(image)
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
                )
                frames.append(frame_resp)
                if on_frame:
                    on_frame(frame_resp)

        return frames

    def _generate_animatediff(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
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
    ) -> list[AnimationFrameResponse]:
        """Core AnimateDiff generation logic."""
        is_controlnet = req.mode.value.startswith("controlnet_")

        if is_controlnet:
            # Reuse existing ControlNet model if same mode already loaded
            existing_cn = pipeline_factory.get_controlnet_from_pipe(
                self._controlnet_pipe, self._controlnet_mode, req.mode,
            )
            pipe = self._animatediff.ensure_controlnet(
                self._pipe, req.mode, existing_controlnet=existing_cn,
            )
        else:
            pipe = self._animatediff.ensure_base(self._pipe)

        # FreeInit
        if req.enable_freeinit:
            try:
                pipe.enable_free_init(
                    num_iters=req.freeinit_iterations,
                    use_fast_sampling=True,
                )
                log.info("FreeInit enabled (%d iterations)", req.freeinit_iterations)
            except Exception as e:
                log.warning("FreeInit unavailable: %s", e)

        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        seed = seed % (2**32)  # clamp to valid CUDA generator range
        generator = torch.Generator("cuda").manual_seed(seed)

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)

        # Progress callback
        def step_callback(pipe_ref, step_idx, timestep, callback_kwargs):
            if self._cancel_event.is_set():
                raise GenerationCancelled("Animation cancelled by client")
            if on_progress:
                on_progress(ProgressResponse(
                    step=step_idx + 1, total=req.steps,
                    frame_index=0, total_frames=req.frame_count,
                ))
            return callback_kwargs

        t0 = time.perf_counter()

        with torch.inference_mode():
            kwargs = dict(
                prompt=req.prompt,
                negative_prompt=effective_neg,
                num_frames=req.frame_count,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg_scale,
                width=round8(req.width),
                height=round8(req.height),
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=step_callback,
                output_type="pil",
            )

            if is_controlnet and req.control_image is not None:
                control = decode_b64_image(req.control_image).convert("RGB")
                target_w, target_h = round8(req.width), round8(req.height)
                control = resize_to_target(control, target_w, target_h)
                kwargs["conditioning_frames"] = [control] * req.frame_count

            output = pipe(**kwargs)

        # Extract frames from output
        pil_frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames

        # Disable FreeInit for next normal generation
        if req.enable_freeinit:
            try:
                pipe.disable_free_init()
            except Exception:
                pass

        # Post-process and encode each frame
        frames: list[AnimationFrameResponse] = []
        for frame_idx, pil_img in enumerate(pil_frames):
            if self._cancel_event.is_set():
                raise GenerationCancelled("AnimateDiff cancelled during post-processing")
            t0_frame = time.perf_counter()
            image = postprocess_apply(pil_img, req.post_process)
            b64_image = encode_image_b64(image)
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
            )
            frames.append(frame_resp)
            if on_frame:
                on_frame(frame_resp)

        return frames

    # ─── REAL-TIME PAINT MODE ─────────────────────────────────

    def start_realtime(self, req: RealtimeStartRequest) -> RealtimeReadyResponse:
        """Activate real-time paint mode: prepare pipeline for fast img2img loop."""
        if not self._loaded:
            self.load()

        rt = self._realtime
        if rt.active:
            self.stop_realtime()

        # Handle LoRA if specified
        if req.lora is not None:
            if (req.lora.name != self._lora_fuser.current_name
                    or req.lora.weight != self._lora_fuser.current_weight):
                self.set_pixel_lora(req.lora.name, req.lora.weight)

        # Configure state from request
        rt.prompt = req.prompt
        rt.negative_prompt = self._build_effective_negative(req.negative_prompt, req.negative_ti)
        rt.denoise_strength = req.denoise_strength
        rt.steps = req.steps
        rt.cfg_scale = req.cfg_scale
        rt.clip_skip = req.clip_skip
        rt.width = round8(req.width)
        rt.height = round8(req.height)
        rt.post_process = req.post_process
        rt.frame_counter = 0

        # Resolve seed once (fixed for session coherence)
        if req.seed >= 0:
            rt.seed = req.seed
        else:
            rt.seed = random.randint(0, 2**32 - 1)
        rt._resolved_seed = rt.seed % (2**32)

        # Suspend DeepCache (not effective at 2-4 steps)
        if self._deepcache_helper is not None:
            try:
                deepcache_manager.disable(self._deepcache_helper)
                rt._deepcache_was_active = True
                log.info("DeepCache suspended for realtime mode")
            except Exception as e:
                log.warning("Failed to suspend DeepCache: %s", e)
                rt._deepcache_was_active = False
        else:
            rt._deepcache_was_active = False

        # Cache scheduler class+config for fast per-frame reset (avoids pipe access)
        rt._scheduler_cls = type(self._pipe.scheduler)
        rt._scheduler_cfg = self._pipe.scheduler.config

        # Pre-compute prompt embeddings (reused every frame)
        self._cache_prompt_embeds(rt)

        rt.active = True
        log.info("Real-time mode started (steps=%d, cfg=%.1f, denoise=%.2f, seed=%d, %dx%d)",
                 rt.steps, rt.cfg_scale, rt.denoise_strength, rt._resolved_seed,
                 rt.width, rt.height)

        return RealtimeReadyResponse(
            message=f"Real-time mode activated ({rt.steps}-step, seed={rt._resolved_seed})",
        )

    def _cache_prompt_embeds(self, rt: RealtimeState) -> None:
        """Compute and cache prompt embeddings if prompt changed."""
        data = f"{rt.prompt}|||{rt.negative_prompt}|||{rt.clip_skip}".encode()
        prompt_hash = int(hashlib.sha256(data).hexdigest()[:16], 16)
        if prompt_hash == rt._prompt_hash and rt.prompt_embeds is not None:
            return  # Already cached

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self._img2img_pipe.encode_prompt(
                prompt=rt.prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=rt.cfg_scale > 1.0,
                negative_prompt=rt.negative_prompt,
                clip_skip=rt.clip_skip,
            )

        rt.prompt_embeds = prompt_embeds
        rt.negative_prompt_embeds = negative_prompt_embeds
        rt._prompt_hash = prompt_hash
        log.debug("Prompt embeddings cached (hash=%d)", prompt_hash)

    def process_realtime_frame(
        self,
        image_b64: str,
        frame_id: int,
        prompt_override: Optional[str] = None,
        mask_b64: Optional[str] = None,
        roi_x: Optional[int] = None,
        roi_y: Optional[int] = None,
        roi_w: Optional[int] = None,
        roi_h: Optional[int] = None,
    ) -> RealtimeResultResponse:
        """Process a single frame in real-time mode: img2img -> post-process -> encode.

        If ROI parameters are provided, only the dirty region is regenerated
        and composited back for faster, more intuitive live painting.
        """
        rt = self._realtime
        if not rt.active:
            raise RuntimeError("Real-time mode not active")

        t0 = time.perf_counter()

        with rt._lock:
            # Handle prompt override
            if prompt_override is not None and prompt_override != rt.prompt:
                rt.prompt = prompt_override
                self._cache_prompt_embeds(rt)

            # Snapshot volatile state under lock
            _prompt_embeds = rt.prompt_embeds
            _neg_embeds = rt.negative_prompt_embeds
            _steps = rt.steps
            _cfg = rt.cfg_scale
            _denoise = rt.denoise_strength
            _seed = rt._resolved_seed
            _width = rt.width
            _height = rt.height
            _pp = rt.post_process

        has_roi = (roi_x is not None and roi_y is not None
                   and roi_w is not None and roi_h is not None
                   and roi_w > 0 and roi_h > 0)

        try:
            with torch.inference_mode():
                # Decode input canvas
                source = decode_b64_image(image_b64).convert("RGB")
                orig_w, orig_h = source.size
                source = resize_to_target(source, _width, _height)

                # Scale ROI coordinates from original canvas space to target space
                if has_roi and (orig_w != _width or orig_h != _height):
                    sx = _width / orig_w
                    sy = _height / orig_h
                    roi_x = int(roi_x * sx)
                    roi_y = int(roi_y * sy)
                    roi_w = max(1, int(roi_w * sx))
                    roi_h = max(1, int(roi_h * sy))

                if has_roi:
                    result_image = self._process_realtime_roi(
                        source, mask_b64,
                        roi_x, roi_y, roi_w, roi_h,
                        _prompt_embeds, _neg_embeds,
                        _steps, _cfg, _denoise, _seed,
                    )
                else:
                    result_image = self._process_realtime_full(
                        source, _prompt_embeds, _neg_embeds,
                        _steps, _cfg, _denoise, _seed,
                    )

            # Post-process (palette, pixelate, quantize)
            result_image = postprocess_apply(result_image, _pp)

            # Encode (compress_level=0 for realtime speed)
            b64_result = encode_image_b64(result_image, compress_level=0)
            w, h = result_image.size
            latency_ms = int((time.perf_counter() - t0) * 1000)

            rt.frame_counter += 1

            return RealtimeResultResponse(
                image=b64_result,
                latency_ms=latency_ms,
                frame_id=frame_id,
                width=w,
                height=h,
                roi_x=roi_x if has_roi else None,
                roi_y=roi_y if has_roi else None,
            )

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM during realtime frame — clearing cache")
            gc.collect()
            torch.cuda.empty_cache()
            raise

    def _process_realtime_full(
        self,
        source: Image.Image,
        prompt_embeds, neg_embeds,
        steps: int, cfg: float, denoise: float, seed: int,
    ) -> Image.Image:
        """Full-canvas img2img (original behavior)."""
        generator = torch.Generator("cuda").manual_seed(seed)
        rt = self._realtime
        self._img2img_pipe.scheduler = rt._scheduler_cls.from_config(rt._scheduler_cfg)

        kwargs = dict(
            prompt_embeds=prompt_embeds,
            image=source,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=denoise,
            generator=generator,
            output_type="pil",
        )
        if neg_embeds is not None:
            kwargs["negative_prompt_embeds"] = neg_embeds

        return self._img2img_pipe(**kwargs).images[0]

    def _process_realtime_roi(
        self,
        source: Image.Image,
        mask_b64: Optional[str],
        roi_x: int, roi_y: int, roi_w: int, roi_h: int,
        prompt_embeds, neg_embeds,
        steps: int, cfg: float, denoise: float, seed: int,
    ) -> Image.Image:
        """ROI-based realtime: crop dirty region, img2img, composite back."""
        pad = settings.realtime_roi_padding
        min_gen = settings.realtime_roi_min_size
        sw, sh = source.size

        # Compute padded ROI clamped to image bounds
        px1 = max(0, roi_x - pad)
        py1 = max(0, roi_y - pad)
        px2 = min(sw, roi_x + roi_w + pad)
        py2 = min(sh, roi_y + roi_h + pad)
        crop_w = px2 - px1
        crop_h = py2 - py1

        if crop_w < 8 or crop_h < 8:
            # ROI too small, fall back to full canvas
            return self._process_realtime_full(
                source, prompt_embeds, neg_embeds, steps, cfg, denoise, seed,
            )

        # Crop source to padded ROI
        crop = source.crop((px1, py1, px2, py2))

        # Determine generation size (at least min_gen, uniform scale to preserve aspect ratio)
        scale = max(min_gen / crop_w, min_gen / crop_h, 1.0)
        gen_w = round8(int(crop_w * scale))
        gen_h = round8(int(crop_h * scale))
        crop_resized = crop.resize((gen_w, gen_h), Image.LANCZOS)

        # Run img2img on the crop
        generator = torch.Generator("cuda").manual_seed(seed)
        rt = self._realtime
        self._img2img_pipe.scheduler = rt._scheduler_cls.from_config(rt._scheduler_cfg)

        kwargs = dict(
            prompt_embeds=prompt_embeds,
            image=crop_resized,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=denoise,
            generator=generator,
            output_type="pil",
        )
        if neg_embeds is not None:
            kwargs["negative_prompt_embeds"] = neg_embeds

        result_crop = self._img2img_pipe(**kwargs).images[0]

        # Resize result back to original crop dimensions
        result_crop = result_crop.resize((crop_w, crop_h), Image.LANCZOS)

        # Composite with mask if provided
        if mask_b64:
            try:
                mask = decode_b64_mask(mask_b64)
                mask = resize_to_target(mask, sw, sh)
                mask_crop = mask.crop((px1, py1, px2, py2))
                # Blend: result where mask is white, original where black
                result_crop = Image.composite(result_crop, crop, mask_crop)
            except Exception as e:
                log.warning("ROI mask composite failed, using unmasked result: %s", e)

        # Paste result back into full canvas
        full_result = source.copy()
        full_result.paste(result_crop, (px1, py1))

        return full_result

    def update_realtime_params(self, req: RealtimeUpdateRequest) -> None:
        """Hot-update real-time parameters without stopping the session."""
        rt = self._realtime
        if not rt.active:
            return

        with rt._lock:
            prompt_changed = False
            if req.prompt is not None and req.prompt != rt.prompt:
                rt.prompt = req.prompt
                prompt_changed = True
            if req.negative_prompt is not None and req.negative_prompt != rt.negative_prompt:
                rt.negative_prompt = req.negative_prompt
                prompt_changed = True
            if req.denoise_strength is not None:
                rt.denoise_strength = req.denoise_strength
            if req.steps is not None:
                rt.steps = req.steps
            if req.cfg_scale is not None:
                rt.cfg_scale = req.cfg_scale
            if req.clip_skip is not None:
                rt.clip_skip = req.clip_skip
                prompt_changed = True  # clip_skip affects prompt embeddings
            if req.seed is not None:
                if req.seed >= 0:
                    rt.seed = req.seed
                    rt._resolved_seed = req.seed % (2**32)
                else:
                    rt.seed = -1
                    rt._resolved_seed = random.randint(0, 2**32 - 1)

            # Re-cache prompt embeddings if text or clip_skip changed
            if prompt_changed:
                self._cache_prompt_embeds(rt)

        log.debug("Realtime params updated: steps=%d, cfg=%.1f, denoise=%.2f",
                  rt.steps, rt.cfg_scale, rt.denoise_strength)

    def stop_realtime(self) -> RealtimeStoppedResponse:
        """Deactivate real-time paint mode and clean up."""
        rt = self._realtime

        # Re-enable DeepCache if it was active before
        if rt._deepcache_was_active and self._deepcache_helper is not None:
            try:
                deepcache_manager.enable(self._deepcache_helper)
                log.info("DeepCache re-enabled after realtime mode")
            except Exception as e:
                log.warning("Failed to re-enable DeepCache: %s", e)

        frames_processed = rt.frame_counter

        # Explicitly delete GPU tensors before reset to avoid VRAM leak
        if rt.prompt_embeds is not None:
            del rt.prompt_embeds
        if rt.negative_prompt_embeds is not None:
            del rt.negative_prompt_embeds

        rt.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Real-time mode stopped (%d frames processed)", frames_processed)
        return RealtimeStoppedResponse(
            message=f"Real-time mode stopped ({frames_processed} frames processed)",
        )
