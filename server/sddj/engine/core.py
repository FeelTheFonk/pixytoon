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
    EmbeddingSpec,
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
    encode_image_raw_bytes,
    resize_to_target,
    round8,
)
from ..embedding_blend import bump_model_generation, clear_embedding_cache
from PIL import Image as _PILImage
from ..lora_fuser import LoRAFuser, _USE_SET_ADAPTERS, _sanitize_adapter_name
from .helpers import GenerationCancelled, build_prompt_schedule, scale_steps_for_denoise
from .animation import AnimationMixin
from .audio_reactive import AudioReactiveMixin

log = logging.getLogger("sddj.engine")

_MAX_SEED = 2**32 - 1


class DiffusionEngine(AnimationMixin, AudioReactiveMixin):
    """Manages the full SD1.5 pipeline with SOTA optimizations."""

    def __init__(self) -> None:
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._controlnet_img2img_pipe = None
        self._controlnet_mode: Optional[GenerationMode] = None
        self._deepcache_helper = None
        self._dc_state = None  # Mode-aware DeepCache wrapper (avoids redundant toggles)
        self._lora_fuser = LoRAFuser()
        self._lora2_adapter_name: Optional[str] = None
        self._animatediff = AnimateDiffManager()
        self._loaded_ti_tokens: set[str] = set()
        self._loaded = False
        self._cancel_event = threading.Event()
        # IP-Adapter state
        self._ip_adapter_loaded = False
        self._ip_adapter_mode: Optional[str] = None
        # Audio reactivity modules (lazy, no GPU)
        self._tome_applied = False
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
        bump_model_generation()

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

        # 2b. channels_last memory format for better tensor core utilization (Ampere+)
        self._pipe.unet.to(memory_format=torch.channels_last)
        self._pipe.vae.to(memory_format=torch.channels_last)
        log.info("channels_last memory format applied to UNet + VAE")

        # 3. Hyper-SD LoRA — fuse permanently
        pipeline_factory.setup_hyper_sd(self._pipe)

        # 4. FreeU v2
        apply_freeu(self._pipe)
        if settings.enable_freeu:
            log.info("FreeU v2 enabled (s1=%.1f s2=%.1f b1=%.1f b2=%.1f)",
                     settings.freeu_s1, settings.freeu_s2,
                     settings.freeu_b1, settings.freeu_b2)

        # 5. UNet quantization (torchao) — AFTER LoRA fuse, BEFORE torch.compile
        pipeline_factory.apply_unet_quantization(self._pipe)

        # 6. torch.compile — BEFORE DeepCache
        pipeline_factory.apply_torch_compile(self._pipe)

        # 6b. torch.compile VAE decoder
        pipeline_factory.apply_vae_compile(self._pipe)

        # 7. Token Merging (ToMe) — AFTER torch.compile, training-free acceleration
        if settings.enable_tome:
            try:
                import tomesd
                tomesd.apply_patch(self._pipe, ratio=settings.tome_ratio)
                self._tome_applied = True
                log.info("ToMe token merging enabled (ratio=%.2f)", settings.tome_ratio)
            except ImportError:
                log.warning("tomesd not installed — skipping token merging")
            except Exception as e:
                log.warning("ToMe apply_patch failed: %s", e)

        # 8. DeepCache — AFTER torch.compile
        self._deepcache_helper = deepcache_manager.create_helper(self._pipe)
        if self._deepcache_helper is not None:
            self._dc_state = deepcache_manager.DeepCacheState(self._deepcache_helper)

        # 9. Create img2img pipeline from same components
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

        # 10. Default style LoRA — fuse BEFORE warmup
        self._load_default_style_lora()

        # 11. Load textual inversion embeddings
        self._load_embeddings()

        # 12. Warmup — deferred to server lifespan (background task behind
        #     _generate_lock) so WebSocket endpoint is available immediately.

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
        else:
            # Support full file path in .env — extract stem for resolve_lora_path()
            from pathlib import PurePath as _PurePath
            _p = _PurePath(lora_name)
            if _p.suffix.lower() in (".safetensors", ".bin", ".pt"):
                lora_name = _p.stem

        try:
            self.set_style_lora(lora_name, settings.default_style_lora_weight)
            log.info("Default style LoRA loaded: %s (weight=%.2f)",
                     lora_name, settings.default_style_lora_weight)
        except Exception as e:
            log.warning("Failed to load default style LoRA '%s': %s", lora_name, e)

    def warmup(self) -> None:
        """Public warmup — call from server after lifespan yield."""
        if not self._loaded:
            log.warning("Cannot warmup: engine not loaded")
            return
        self._warmup()

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

        # Cancellation-aware callback — allows shutdown to interrupt warmup
        # instead of waiting for all steps to complete.
        cancel_event = self._cancel_event

        def _warmup_callback(pipe, step_idx, timestep, cb_kwargs):
            if cancel_event.is_set():
                raise RuntimeError("Warmup cancelled (server shutting down)")
            return cb_kwargs

        try:
            with torch.inference_mode():
                gen = torch.Generator("cuda").manual_seed(0)
                torch.compiler.cudagraph_mark_step_begin()

                latents = self._pipe(
                    prompt="warmup",
                    negative_prompt="warmup",
                    num_inference_steps=settings.default_steps,
                    guidance_scale=settings.default_cfg,
                    width=settings.default_width,
                    height=settings.default_height,
                    generator=gen,
                    clip_skip=settings.default_clip_skip,
                    callback_on_step_end=_warmup_callback,
                    output_type="latent",
                ).images

                # VAE decode warmup — compiled VAE decode must be exercised
                # here, otherwise the first real generation triggers Inductor
                # compilation AFTER the denoising steps complete, causing a
                # multi-second freeze at "100%" before the image is returned.
                self._decode_latents(latents)

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
        move_to_cpu(self._controlnet_img2img_pipe)

        # Remove ToMe patches before releasing the pipeline
        if getattr(self, '_tome_applied', False):
            try:
                import tomesd
                tomesd.remove_patch(self._pipe)
            except Exception:
                pass
            self._tome_applied = False

        # Restore original SDPA if SageAttention was monkey-patched
        from ..pipeline_factory import restore_attention
        restore_attention()

        self._pipe = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._controlnet_img2img_pipe = None
        self._controlnet_mode = None
        self._deepcache_helper = None
        self._dc_state = None
        self._lora_fuser = LoRAFuser()
        self._loaded_ti_tokens.clear()
        self._loaded = False
        self._ip_adapter_loaded = False
        self._ip_adapter_mode = None
        self._animatediff.unload()
        rembg_wrapper.unload()
        clear_embedding_cache()
        vram_cleanup(force=True)
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
            if self._controlnet_img2img_pipe is not None:
                move_to_cpu(self._controlnet_img2img_pipe)
                self._controlnet_img2img_pipe = None
            self._controlnet_mode = None
            cleaned.append("ControlNet")

        if self._animatediff.pipe is not None:
            self._animatediff.unload()
            cleaned.append("AnimateDiff")

        rembg_wrapper.unload()

        if self._stem_separator is not None:
            self._stem_separator.unload()
            cleaned.append("StemSeparator")

        if self._ip_adapter_loaded:
            try:
                self._pipe.unload_ip_adapter()
                self._ip_adapter_loaded = False
                cleaned.append("IP-Adapter")
            except Exception:
                pass

        vram_cleanup(force=True)

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
            log.debug("set_style_lora(%s) ignored — engine not loaded", name)
            return
        weight = max(-2.0, min(2.0, weight))
        self._lora_fuser.set_lora(self._pipe, name, weight)
        clear_embedding_cache()

    # ─── CONTROLNET ──────────────────────────────────────────

    def _ensure_controlnet(self, mode: GenerationMode) -> None:
        """Lazy-load ControlNet model for the requested mode."""
        if self._controlnet_mode == mode and self._controlnet_pipe is not None:
            return

        # Smart transition: free AnimateDiff if actively loaded (reclaim VRAM)
        if self._animatediff is not None and self._animatediff.pipe is not None:
            log.info("Smart transition: unloading AnimateDiff before ControlNet load")
            self._animatediff.unload()

        # VRAM budget guard: cleanup if low on memory before loading ControlNet
        from ..vram_utils import check_vram_budget
        if not check_vram_budget(required_mb=800, min_free_mb=settings.vram_min_free_mb):
            log.warning("Low VRAM before ControlNet load — running cleanup")
            vram_cleanup()

        # Unload previous
        move_to_cpu(self._controlnet_pipe)
        move_to_cpu(self._controlnet_img2img_pipe)
        self._controlnet_pipe = None
        self._controlnet_img2img_pipe = None
        vram_cleanup()

        self._controlnet_pipe, self._controlnet_img2img_pipe = (
            pipeline_factory.create_controlnet_pipeline(self._pipe, mode)
        )
        self._controlnet_mode = mode

    # ─── IP-ADAPTER ────────────────────────────────────────────

    _IP_ADAPTER_MODELS = {
        "full": "ip-adapter_sd15.bin",
        "style": "ip-adapter-plus_sd15.bin",
        "composition": "ip-adapter-plus-face_sd15.bin",
    }

    def _ensure_ip_adapter(self, mode: str = "full"):
        """Lazy-load IP-Adapter, switching model if mode changed."""
        model_name = self._IP_ADAPTER_MODELS.get(mode, settings.ip_adapter_model)
        if self._ip_adapter_loaded and getattr(self, '_ip_adapter_mode', None) == mode:
            return
        # Unload previous if switching mode
        if self._ip_adapter_loaded:
            try:
                self._pipe.unload_ip_adapter()
            except Exception:
                pass
            self._ip_adapter_loaded = False
        try:
            self._pipe.load_ip_adapter(
                settings.ip_adapter_repo,
                subfolder="models",
                weight_name=model_name,
                local_files_only=True,
            )
            self._ip_adapter_loaded = True
            self._ip_adapter_mode = mode
            log.info("IP-Adapter loaded: %s/%s (mode=%s)", settings.ip_adapter_repo, model_name, mode)
        except Exception as e:
            log.warning("IP-Adapter unavailable (%s): %s", model_name, e)

    def _prepare_ip_adapter_kwargs(self, req, kwargs: dict) -> None:
        """If IP-Adapter is requested, load it and add kwargs for the pipeline call."""
        if not settings.enable_ip_adapter:
            return
        if not req.ip_adapter_image or req.ip_adapter_scale <= 0:
            return
        try:
            mode = req.ip_adapter_mode or "full"
            self._ensure_ip_adapter(mode)
            if not self._ip_adapter_loaded:
                return
            ref_image = decode_b64_image(req.ip_adapter_image).convert("RGB")
            self._pipe.set_ip_adapter_scale(req.ip_adapter_scale)
            kwargs["ip_adapter_image"] = ref_image
        except Exception as e:
            log.warning("IP-Adapter preparation failed: %s", e)

    # ─── MULTI-LORA ─────────────────────────────────────────────

    def _apply_lora2(self, req) -> None:
        """Apply second LoRA on top of LoRA1.

        Fast path (set_adapters): loads LoRA2 as a PEFT adapter and activates
        both via set_adapters([lora1, lora2], [w1, w2]).  No weight fusion.
        Fallback path (fuse/unfuse): fuses LoRA2 into already-fused weights.
        """
        if req.lora2 is None:
            return
        if req.lora is None:
            return  # lora2 requires lora1 to be set
        if req.lora2.name == req.lora.name:
            return  # Same LoRA — skip duplicate load
        try:
            from ..lora_manager import resolve_lora_path
            lora2_path = resolve_lora_path(req.lora2.name)

            if _USE_SET_ADAPTERS:
                # Fast path: load LoRA2 as adapter, activate both via set_adapters.
                # Falls back gracefully if LoRAs target different modules (set_adapters
                # fails on layers that only have one adapter's PEFT injection).
                adapter2_name = _sanitize_adapter_name(req.lora2.name) + "_l2"
                lora1_adapter = _sanitize_adapter_name(req.lora.name)
                self._pipe.load_lora_weights(
                    str(lora2_path.parent),
                    weight_name=lora2_path.name,
                    adapter_name=adapter2_name,
                    local_files_only=True,
                )
                try:
                    self._pipe.set_adapters(
                        [lora1_adapter, adapter2_name],
                        [req.lora.weight, req.lora2.weight],
                    )
                    self._lora2_adapter_name = adapter2_name
                    log.info("Multi-LoRA (fast): %s(%.2f) + %s(%.2f) via set_adapters",
                             req.lora.name, req.lora.weight,
                             req.lora2.name, req.lora2.weight)
                except Exception as e:
                    # LoRAs have incompatible architectures — can't coexist in PEFT.
                    # Clean up LoRA2 and continue with LoRA1 only.
                    log.warning("Multi-adapter set_adapters failed (architecture mismatch), "
                                "proceeding with LoRA1 only: %s", e)
                    try:
                        self._pipe.delete_adapters([adapter2_name])
                    except Exception:
                        pass
                    self._pipe.set_adapters([lora1_adapter], [req.lora.weight])
                    self._lora2_adapter_name = None
                    return
            else:
                # Fallback path: fuse LoRA2 into already-fused weights
                self._pipe.load_lora_weights(
                    str(lora2_path.parent),
                    weight_name=lora2_path.name,
                    adapter_name="lora2_tmp",
                    local_files_only=True,
                )
                self._pipe.fuse_lora(lora_scale=req.lora2.weight)
                self._pipe.unload_lora_weights()
                self._lora2_adapter_name = None
                log.info("Multi-LoRA (fuse): +%s fused at weight %.2f on top of %s",
                         req.lora2.name, req.lora2.weight, req.lora.name)
        except FileNotFoundError:
            log.warning("LoRA2 '%s' not found — proceeding with single LoRA", req.lora2.name)
        except Exception as e:
            log.warning("LoRA2 '%s' load failed — proceeding with single LoRA: %s",
                        req.lora2.name, e)

    def _cleanup_lora2(self) -> None:
        """Restore to single-LoRA state after generation."""
        if not self._lora_fuser.current_name:
            return
        try:
            if _USE_SET_ADAPTERS:
                # Fast path: deactivate LoRA2, restore LoRA1-only via set_adapters
                lora1_adapter = _sanitize_adapter_name(self._lora_fuser.current_name)
                self._pipe.set_adapters(
                    [lora1_adapter],
                    [self._lora_fuser.current_weight],
                )
                # Delete LoRA2 adapter to free memory
                adapter2 = getattr(self, '_lora2_adapter_name', None)
                if adapter2:
                    try:
                        self._pipe.delete_adapters([adapter2])
                    except Exception:
                        pass  # non-critical — adapter will be overwritten on next load
                    self._lora2_adapter_name = None
                log.debug("LoRA2 cleanup (fast): restored single-adapter %s", lora1_adapter)
            else:
                # Fallback path: restore from snapshot + re-fuse LoRA1
                name = self._lora_fuser.current_name
                weight = self._lora_fuser.current_weight
                self._lora_fuser.set_lora(self._pipe, name, weight)
        except Exception as e:
            log.warning("LoRA2 cleanup failed: %s", e)

    # ─── GENERATION ──────────────────────────────────────────

    def _build_effective_negative(self, neg: str, ti_specs: list[EmbeddingSpec] | None) -> str:
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

    def _build_ti_suffix(self, ti_specs: list[EmbeddingSpec] | None) -> str:
        """Build just the TI embedding token suffix string (no base negative).

        Used by animation loops to pre-compute the suffix once and concatenate
        it directly in per-frame blend paths, avoiding repeated _build_effective_negative calls.
        """
        if not ti_specs:
            return ""
        ti_parts = []
        for ti_spec in ti_specs:
            if ti_spec.name in self._loaded_ti_tokens:
                if abs(ti_spec.weight - 1.0) < 0.01:
                    ti_parts.append(ti_spec.name)
                else:
                    ti_parts.append(f"({ti_spec.name}:{ti_spec.weight:.2f})")
        return ", ".join(ti_parts)

    def generate(
        self,
        req: GenerateRequest,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> ResultResponse:
        """Run the full generation pipeline: diffusion → post-process → encode."""
        if not self._loaded:
            self.load()

        self._cancel_event.clear()
        _lora2_active = False  # Defined before try for finally-block access

        try:
            with torch.inference_mode():
                t0 = time.perf_counter()

                # Resolve seed
                # random.randint is not CSPRNG — acceptable for diffusion seeds
                seed = req.seed if req.seed >= 0 else random.randint(0, _MAX_SEED)
                seed = seed % (_MAX_SEED + 1)  # clamp to valid range
                generator = torch.Generator("cuda").manual_seed(seed)

                # Set style LoRA — only if client explicitly specifies one.
                if req.lora is not None:
                    lora_name = req.lora.name
                    lora_weight = req.lora.weight
                    if (lora_name != self._lora_fuser.current_name
                            or lora_weight != self._lora_fuser.current_weight
                            or self._lora_fuser.needs_reapply(self._pipe)):
                        self.set_style_lora(lora_name, lora_weight)

                # Multi-LoRA: load second adapter if specified
                if req.lora2 is not None:
                    try:
                        self._apply_lora2(req)
                        _lora2_active = True
                    except Exception as e:
                        log.warning("Multi-LoRA setup failed: %s", e)

                effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)

                # Progress callback adapter with cancellation + timeout support
                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    if self._cancel_event.is_set():
                        raise GenerationCancelled("Generation cancelled by client")
                    if time.perf_counter() - t0 > settings.generation_timeout:
                        raise GenerationCancelled("Generation timeout exceeded")
                    if on_progress:
                        on_progress(ProgressResponse(step=step_idx + 1, total=req.steps))
                    return callback_kwargs

                # ── Prompt schedule resolution (single-image: use frame 0) ──
                schedule = build_prompt_schedule(req)
                effective_prompt = req.prompt
                if schedule and schedule.keyframes:
                    blend_info = schedule.get_blend_info_for_frame(0)
                    if blend_info.effective_prompt:
                        effective_prompt = blend_info.effective_prompt
                    if blend_info.negative_prompt:
                        effective_neg = self._build_effective_negative(
                            blend_info.negative_prompt, req.negative_ti)
                    # Per-keyframe parameter overrides at frame 0
                    _overrides: dict = {}
                    if blend_info.cfg_scale is not None:
                        _overrides["cfg_scale"] = blend_info.cfg_scale
                    if blend_info.denoise_strength is not None:
                        _overrides["denoise_strength"] = blend_info.denoise_strength
                    if blend_info.steps is not None:
                        _overrides["steps"] = blend_info.steps
                    if _overrides:
                        req = req.model_copy(update=_overrides)

                # ── Per-request scheduler override ────────────────────
                _original_scheduler = None
                if req.scheduler:
                    try:
                        from ..scheduler_factory import create_scheduler
                        _original_scheduler = self._pipe.scheduler
                        new_sched = create_scheduler(req.scheduler, _original_scheduler.config)
                        self._pipe.scheduler = new_sched
                        if self._img2img_pipe is not None:
                            self._img2img_pipe.scheduler = type(new_sched).from_config(new_sched.config)
                        if self._controlnet_pipe is not None:
                            self._controlnet_pipe.scheduler = type(new_sched).from_config(new_sched.config)
                        if self._controlnet_img2img_pipe is not None:
                            self._controlnet_img2img_pipe.scheduler = type(new_sched).from_config(new_sched.config)
                    except Exception as e:
                        log.warning("Scheduler override '%s' failed: %s — using default", req.scheduler, e)
                        _original_scheduler = None

                # ── Mode dispatch (returns latent tensors) ────────────
                _inpaint_compositing = None  # (source, mask) for inpaint mode

                try:
                    if req.mode == GenerationMode.TXT2IMG:
                        latents = self._txt2img(req, generator, step_callback, effective_prompt, effective_neg)

                    elif req.mode == GenerationMode.IMG2IMG:
                        latents = self._img2img(req, generator, step_callback, effective_prompt, effective_neg)

                    elif req.mode == GenerationMode.INPAINT:
                        latents, _inp_source, _inp_mask = self._inpaint(
                            req, generator, step_callback, effective_prompt, effective_neg)
                        _inpaint_compositing = (_inp_source, _inp_mask)

                    elif req.mode.value.startswith("controlnet_"):
                        latents = self._controlnet_generate(req, generator, step_callback, effective_prompt, effective_neg)

                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                finally:
                    # Restore original scheduler if overridden
                    if _original_scheduler is not None:
                        self._pipe.scheduler = _original_scheduler
                        if self._img2img_pipe is not None:
                            self._img2img_pipe.scheduler = type(_original_scheduler).from_config(
                                _original_scheduler.config)
                        if self._controlnet_pipe is not None:
                            self._controlnet_pipe.scheduler = type(_original_scheduler).from_config(
                                _original_scheduler.config)
                        if self._controlnet_img2img_pipe is not None:
                            self._controlnet_img2img_pipe.scheduler = type(_original_scheduler).from_config(
                                _original_scheduler.config)

                # ── VAE decode with progress feedback ─────────────────
                if on_progress:
                    on_progress(ProgressResponse(
                        step=req.steps, total=req.steps, status="decoding"))
                t_vae = time.perf_counter()
                image = self._decode_latents(latents)
                vae_ms = (time.perf_counter() - t_vae) * 1000

                # Inpaint compositing (after VAE decode)
                if _inpaint_compositing is not None:
                    _inp_source, _inp_mask = _inpaint_compositing
                    image = composite_with_mask(_inp_source, image, _inp_mask)

                # ── Post-process ──────────────────────────────────────
                t_pp = time.perf_counter()
                image = postprocess_apply(image, req.post_process)
                pp_ms = (time.perf_counter() - t_pp) * 1000

                if self._cancel_event.is_set():
                    raise GenerationCancelled("Cancelled during post-processing")

                # ── Encode result ─────────────────────────────────────
                t_enc = time.perf_counter()
                raw_bytes = encode_image_raw_bytes(image)
                enc_ms = (time.perf_counter() - t_enc) * 1000
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                log.info("Timing: VAE=%.0fms post=%.0fms enc=%.0fms total=%dms",
                         vae_ms, pp_ms, enc_ms, elapsed_ms)
                w, h = image.size

                result = ResultResponse(
                    image="",
                    seed=seed,
                    time_ms=elapsed_ms,
                    width=w,
                    height=h,
                    encoding="raw_rgba",
                )
                result._raw_bytes = raw_bytes
                return result

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM — clearing VRAM cache")
            vram_cleanup(force=True)
            try:
                from ..pipeline_factory import fresh_scheduler
                self._pipe.scheduler = fresh_scheduler(self._pipe)
            except Exception:
                pass  # best-effort
            raise
        finally:
            # FC-03: LoRA2 cleanup in finally — if OOM occurs after LoRA2
            # fusion, weights remain permanently fused corrupting subsequent
            # generations unless cleaned up unconditionally.
            if _lora2_active:
                try:
                    self._cleanup_lora2()
                except Exception as e:
                    log.warning("LoRA2 cleanup in finally failed: %s", e)
            self._cancel_event.clear()

    # ─── VAE DECODE ───────────────────────────────────────────

    def _decode_latents(self, latents: torch.Tensor) -> _PILImage.Image:
        """Decode latent tensor to PIL image via the pipeline's VAE.

        Separates VAE decode from the pipeline call so a "decoding" progress
        message can be sent between the last denoising step and the actual
        decode, eliminating the perceived freeze at end of generation.
        """
        pipe = self._pipe
        # Scale latents by the VAE's scaling factor
        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
        scaled = latents / scaling
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        # diffusers VaeImageProcessor: denormalize + clamp + convert
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return image

    # ─── PRIVATE GENERATION METHODS ──────────────────────────

    def _txt2img(self, req, generator, callback, prompt, negative):
        torch.compiler.cudagraph_mark_step_begin()
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=round8(req.width),
            height=round8(req.height),
            generator=generator,
            clip_skip=req.clip_skip,
            callback_on_step_end=callback,
            output_type="latent",
        )
        if req.guidance_rescale > 0:
            kwargs["guidance_rescale"] = req.guidance_rescale
        if req.pag_scale > 0 and "PAG" in type(self._pipe).__name__:
            kwargs["pag_scale"] = req.pag_scale
        # IP-Adapter reference image
        self._prepare_ip_adapter_kwargs(req, kwargs)
        return self._pipe(**kwargs).images

    def _img2img(self, req, generator, callback, prompt, negative):
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
        # Suppress DeepCache via mode-aware state to avoid redundant toggles.
        if self._dc_state is not None:
            self._dc_state.suppress_for("img2img")
        try:
            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative,
                image=source,
                num_inference_steps=scaled_steps,
                guidance_scale=req.cfg_scale,
                strength=strength,
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=callback,
                output_type="latent",
            )
            if req.guidance_rescale > 0:
                kwargs["guidance_rescale"] = req.guidance_rescale
            if req.pag_scale > 0 and "PAG" in type(self._img2img_pipe).__name__:
                kwargs["pag_scale"] = req.pag_scale
            # IP-Adapter reference image
            self._prepare_ip_adapter_kwargs(req, kwargs)
            return self._img2img_pipe(**kwargs).images
        finally:
            if self._dc_state is not None:
                self._dc_state.restore()

    def _inpaint(self, req, generator, callback, prompt, negative):
        """Inpaint: img2img full image, then composite via mask.

        Returns (latents, source, mask) — caller decodes latents and composites.
        """
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
        # Same DeepCache suppression as _img2img — shared UNet, different scheduler.
        if self._dc_state is not None:
            self._dc_state.suppress_for("img2img")
        try:
            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative,
                image=source,
                num_inference_steps=scaled_steps,
                guidance_scale=req.cfg_scale,
                strength=strength,
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=callback,
                output_type="latent",
            )
            if req.guidance_rescale > 0:
                kwargs["guidance_rescale"] = req.guidance_rescale
            if req.pag_scale > 0 and "PAG" in type(self._img2img_pipe).__name__:
                kwargs["pag_scale"] = req.pag_scale
            self._prepare_ip_adapter_kwargs(req, kwargs)
            latents = self._img2img_pipe(**kwargs).images
        finally:
            if self._dc_state is not None:
                self._dc_state.restore()

        # Return latents + compositing data — caller decodes and composites
        return latents, source, mask

    def _controlnet_generate(self, req, generator, callback, prompt, negative):
        self._ensure_controlnet(req.mode)
        width = round8(req.width)
        height = round8(req.height)

        # All ControlNet modes: user provides control_image
        if req.control_image is None:
            raise ValueError("ControlNet requires control_image")
        control = decode_b64_image(req.control_image)
        control = control.convert("RGB")
        control = resize_to_target(control, width, height)

        # img2img+ControlNet: source image + control_image + strength
        use_img2img = req.source_image is not None
        source = None
        if use_img2img:
            source = decode_b64_image(req.source_image).convert("RGB")
            source = resize_to_target(source, width, height)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=width,
            height=height,
            generator=generator,
            clip_skip=req.clip_skip,
            callback_on_step_end=callback,
            output_type="latent",
        )
        if req.guidance_rescale > 0:
            call_kwargs["guidance_rescale"] = req.guidance_rescale
        self._prepare_ip_adapter_kwargs(req, call_kwargs)

        if use_img2img:
            # Clamp denoise: ensure at least 2 effective steps (matches _img2img logic)
            _cap = max(settings.distilled_step_scale_cap, 1)
            min_denoise = min(1.0, 2.0 / max(req.steps * _cap, 1) + 1e-3)
            strength = max(req.denoise_strength, min_denoise)
            scaled_steps = scale_steps_for_denoise(req.steps, strength)
            call_kwargs["image"] = source
            call_kwargs["control_image"] = control
            call_kwargs["strength"] = strength
            call_kwargs["num_inference_steps"] = scaled_steps
        else:
            call_kwargs["image"] = control

        # ControlNet conditioning params (all CN modes)
        cgs = req.control_guidance_start
        cge = req.control_guidance_end
        if cgs > 0.0:
            call_kwargs["control_guidance_start"] = cgs
        if cge < 1.0:
            call_kwargs["control_guidance_end"] = cge
        # QR-specific: conditioning scale + step override
        if req.mode == GenerationMode.CONTROLNET_QRCODE:
            cn_scale = getattr(
                req, 'controlnet_conditioning_scale',
                settings.qr_controlnet_conditioning_scale,
            )
            call_kwargs["controlnet_conditioning_scale"] = cn_scale
            # QR defaults from config (override generic guidance if not set)
            if cgs <= 0.0:
                call_kwargs["control_guidance_start"] = settings.qr_control_guidance_start
            if cge >= 1.0:
                call_kwargs["control_guidance_end"] = settings.qr_control_guidance_end
            # Override steps if too low for ControlNet quality
            if req.steps <= 8:
                qr_steps = settings.qr_default_steps
                if use_img2img:
                    qr_steps = scale_steps_for_denoise(qr_steps, strength)
                call_kwargs["num_inference_steps"] = qr_steps
                log.info("QR mode: steps %d → %d for quality",
                         req.steps, qr_steps)

        torch.compiler.cudagraph_mark_step_begin()
        # DeepCache suppressed for ALL ControlNet modes:
        # Cached steps skip UNet blocks where ControlNet injects residuals,
        # causing conditioning to be partially dropped on those steps.
        active_pipe = self._controlnet_img2img_pipe if use_img2img else self._controlnet_pipe
        if req.pag_scale > 0 and "PAG" in type(active_pipe).__name__:
            call_kwargs["pag_scale"] = req.pag_scale
        if self._dc_state is not None:
            self._dc_state.suppress_for("controlnet")
        try:
            latents = active_pipe(**call_kwargs).images
        finally:
            if self._dc_state is not None:
                self._dc_state.restore()

        return latents

