"""SOTA Diffusion Engine — SD1.5 + Hyper-SD (fused) + DeepCache + FreeU v2 + AnimateDiff.

Manages pipeline lifecycle, LoRA fusing, ControlNet lazy-loading,
AnimateDiff motion module, and progress callbacks for the WebSocket server.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from io import BytesIO
from base64 import b64decode, b64encode
from typing import Callable, Optional

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

from diffusers import (
    AnimateDiffPipeline,
    AnimateDiffControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    MotionAdapter,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from diffusers.utils.peft_utils import recurse_remove_peft_layers

from .config import settings
from .lora_manager import list_loras, resolve_lora_path
from .ti_manager import list_embeddings, resolve_embedding_path
from .postprocess import apply as postprocess_apply
from .protocol import (
    AnimationFrameResponse,
    AnimationMethod,
    AnimationRequest,
    GenerateRequest,
    GenerationMode,
    ProgressResponse,
    ResultResponse,
    SeedStrategy,
)
from . import rembg_wrapper

log = logging.getLogger("pixytoon.engine")

# ─────────────────────────────────────────────────────────────
# CONTROLNET MODEL IDS
# ─────────────────────────────────────────────────────────────
_CONTROLNET_IDS: dict[GenerationMode, str] = {
    GenerationMode.CONTROLNET_OPENPOSE: "lllyasviel/control_v11p_sd15_openpose",
    GenerationMode.CONTROLNET_CANNY: "lllyasviel/control_v11p_sd15_canny",
    GenerationMode.CONTROLNET_SCRIBBLE: "lllyasviel/control_v11p_sd15_scribble",
    GenerationMode.CONTROLNET_LINEART: "lllyasviel/control_v11p_sd15_lineart",
}


# ─────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────

class GenerationCancelled(Exception):
    """Raised when a client cancels an in-progress generation."""


def _round8(v: int) -> int:
    """Round to nearest multiple of 8 (SD1.5 VAE requirement)."""
    return ((v + 4) // 8) * 8


class DiffusionEngine:
    """Manages the full SD1.5 pipeline with SOTA optimizations."""

    def __init__(self) -> None:
        self._pipe: Optional[StableDiffusionPipeline] = None
        self._img2img_pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self._controlnet_pipe = None
        self._controlnet_mode: Optional[GenerationMode] = None
        self._deepcache_helper = None
        self._current_pixel_lora: Optional[str] = None
        self._current_pixel_lora_weight: float = 0.0
        self._loaded_ti_tokens: set[str] = set()
        self._loaded = False
        self._cancel_event = threading.Event()
        # AnimateDiff
        self._motion_adapter: Optional[MotionAdapter] = None
        self._animatediff_pipe: Optional[AnimateDiffPipeline] = None
        self._animatediff_controlnet_pipe: Optional[AnimateDiffControlNetPipeline] = None
        self._animatediff_controlnet_mode: Optional[GenerationMode] = None

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

        log.info("Loading SD1.5 pipeline: %s", settings.default_checkpoint)
        t0 = time.perf_counter()

        # 1. Base pipeline
        self._pipe = StableDiffusionPipeline.from_pretrained(
            settings.default_checkpoint,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16",
        )
        self._pipe.to("cuda")

        # 2. VRAM optimizations — SDP > xformers > attention_slicing
        self._setup_attention()
        if settings.enable_vae_tiling:
            self._pipe.vae.enable_tiling()
        self._pipe.vae.enable_slicing()

        # 3. Hyper-SD LoRA — load, fuse permanently, then unload adapter
        self._pipe.scheduler = DDIMScheduler.from_config(
            self._pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        log.info("Loading Hyper-SD LoRA: %s/%s", settings.hyper_sd_repo, settings.hyper_sd_lora_file)
        self._pipe.load_lora_weights(
            settings.hyper_sd_repo,
            weight_name=settings.hyper_sd_lora_file,
            adapter_name="hyper_sd",
        )
        self._pipe.fuse_lora(lora_scale=settings.hyper_sd_fuse_scale)
        self._pipe.unload_lora_weights()
        log.info("Hyper-SD LoRA fused into UNet weights (scale=%.3f)", settings.hyper_sd_fuse_scale)

        # 4. FreeU v2 (free quality boost — no training needed)
        if settings.enable_freeu:
            self._pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )
            log.info("FreeU v2 enabled (s1=%.1f s2=%.1f b1=%.1f b2=%.1f)",
                     settings.freeu_s1, settings.freeu_s2,
                     settings.freeu_b1, settings.freeu_b2)

        # 5. torch.compile — BEFORE DeepCache (DeepCache wraps the forward)
        #    fullgraph=False required: DeepCache introduces dynamic control flow
        #    mode=default: Triton codegen without kernel benchmarking
        #    (reduce-overhead uses CUDAGraphs which is INCOMPATIBLE with DeepCache)
        if settings.enable_torch_compile:
            try:
                self._pipe.unet = torch.compile(
                    self._pipe.unet,
                    mode=settings.compile_mode,
                    fullgraph=False,
                )
                log.info("torch.compile enabled for UNet (%s)", settings.compile_mode)
            except Exception as e:
                log.warning("torch.compile failed: %s", e)

        # 6. DeepCache — AFTER torch.compile (wraps the compiled forward)
        if settings.enable_deepcache:
            try:
                from DeepCache import DeepCacheSDHelper
                self._deepcache_helper = DeepCacheSDHelper(pipe=self._pipe)
                self._deepcache_helper.set_params(
                    cache_interval=settings.deepcache_interval,
                    cache_branch_id=settings.deepcache_branch,
                )
                self._deepcache_helper.enable()
                log.info("DeepCache enabled (interval=%d)", settings.deepcache_interval)
            except Exception as e:
                log.warning("DeepCache unavailable: %s", e)

        # 7. Create img2img pipeline from same components
        self._img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            tokenizer=self._pipe.tokenizer,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        # 7b. Apply FreeU to img2img pipeline (shares components but FreeU needs explicit enable)
        if settings.enable_freeu:
            self._img2img_pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )

        elapsed = time.perf_counter() - t0
        log.info("Pipeline loaded in %.1fs", elapsed)
        self._loaded = True

        # 8. Default pixel art LoRA — fuse BEFORE warmup so compiled graphs
        #    include the correct fused weights (avoids recompilation on first gen)
        self._load_default_pixel_lora()

        # 9. Load textual inversion embeddings
        self._load_embeddings()

        # 10. Warmup — front-load torch.compile + Triton compilation
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
            # Auto-detect: use first available LoRA
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
        """Run a dummy generation to pre-compile the exact torch.compile graph.

        CRITICAL: All parameters must match real generation defaults:
        - guidance_scale: Differs → CFG branch changes (single vs double UNet)
        - width/height: Differs → different tensor shapes → graph recompilation
        - steps: Must exercise enough steps for DeepCache cache/skip branches
        torch.compile graphs are keyed on input shapes — wrong warmup shapes
        mean the first real request recompiles everything from scratch.
        """
        log.info("Warmup: triggering torch.compile + JIT compilation...")
        t0 = time.perf_counter()
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

    def _setup_attention(self) -> None:
        """Configure best available attention: SDP (native) > xformers > slicing.

        PyTorch >= 2.0 uses scaled_dot_product_attention automatically in
        diffusers via AttnProcessor2_0 — no explicit call needed. We only
        need xformers or slicing as fallbacks for older PyTorch.
        """
        # PyTorch >= 2.0: SDP attention is used automatically by diffusers
        torch_major = int(torch.__version__.split(".")[0])
        if torch_major >= 2:
            log.info("PyTorch %s: SDP attention active (native AttnProcessor2_0)", torch.__version__)
            return

        # xformers (for PyTorch < 2.0)
        try:
            import xformers  # noqa: F401
            self._pipe.enable_xformers_memory_efficient_attention()
            log.info("xformers memory-efficient attention enabled")
            return
        except ImportError:
            log.debug("xformers not installed, trying fallback")
        except Exception as e:
            log.warning("xformers init failed (%s), falling back", e)

        # Fallback: attention slicing (slowest)
        if settings.enable_attention_slicing:
            self._pipe.enable_attention_slicing()
            log.info("Attention slicing enabled (fallback)")

    def unload(self) -> None:
        """Free all GPU memory."""
        self._pipe = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._deepcache_helper = None
        self._current_pixel_lora = None
        self._current_pixel_lora_weight = 0.0
        self._loaded_ti_tokens.clear()
        self._loaded = False
        # AnimateDiff
        self._motion_adapter = None
        self._animatediff_pipe = None
        self._animatediff_controlnet_pipe = None
        self._animatediff_controlnet_mode = None
        rembg_wrapper.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Pipeline unloaded")

    # ─── LORA MANAGEMENT ────────────────────────────────────

    def set_pixel_lora(self, name: Optional[str], weight: float = 1.0) -> None:
        """Load or switch pixel art LoRA (fused into weights, no PEFT runtime)."""
        if not self._loaded or self._pipe is None:
            return

        # Track whether we actually unfused — needed for dynamo reset decision
        had_lora = self._current_pixel_lora is not None

        # Unfuse previous pixel art LoRA if any
        if had_lora:
            try:
                self._pipe.unfuse_lora()
                self._pipe.unload_lora_weights()
            except Exception as e:
                log.warning("Failed to unfuse pixel art LoRA '%s': %s",
                            self._current_pixel_lora, e)
            self._current_pixel_lora = None
            self._current_pixel_lora_weight = 0.0

        if name is None:
            # Only reset dynamo if we actually unfused a LoRA (weight change)
            # Skip if no LoRA was loaded — nothing changed, compiled graph is fine
            if not had_lora:
                return
            if settings.enable_torch_compile:
                torch._dynamo.reset()
                log.info("Dynamo cache reset after LoRA removal")
            return

        path = resolve_lora_path(name)
        log.info("Loading pixel art LoRA: %s (weight=%.2f)", name, weight)
        self._pipe.load_lora_weights(
            str(path),
            adapter_name="pixel_art",
        )
        self._pipe.fuse_lora(lora_scale=weight)
        self._pipe.unload_lora_weights()
        self._current_pixel_lora = name
        self._current_pixel_lora_weight = weight

        # Changing fused weights invalidates compiled graph — reset dynamo
        if settings.enable_torch_compile:
            torch._dynamo.reset()
            log.info("Dynamo cache reset after LoRA weight change (will recompile on next generation)")

    # ─── CONTROLNET ──────────────────────────────────────────

    def _ensure_controlnet(self, mode: GenerationMode) -> None:
        """Lazy-load ControlNet model for the requested mode."""
        if self._controlnet_mode == mode and self._controlnet_pipe is not None:
            return

        # Unload previous
        self._controlnet_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_id = _CONTROLNET_IDS.get(mode)
        if model_id is None:
            raise ValueError(f"No ControlNet for mode: {mode}")

        log.info("Loading ControlNet: %s", model_id)
        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda")

        self._controlnet_pipe = StableDiffusionControlNetPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            tokenizer=self._pipe.tokenizer,
            unet=self._pipe.unet,
            controlnet=controlnet,
            scheduler=self._pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        # Apply FreeU + VAE optimizations to ControlNet pipeline
        if settings.enable_freeu:
            self._controlnet_pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )
        if settings.enable_vae_tiling:
            self._controlnet_pipe.vae.enable_tiling()
        self._controlnet_pipe.vae.enable_slicing()

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
                seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
                generator = torch.Generator("cuda").manual_seed(seed)

                # Set pixel art LoRA — only if client explicitly specifies one.
                # When req.lora is None (client omitted field), keep whatever
                # LoRA is currently fused (typically the default_pixel_lora).
                if req.lora is not None:
                    lora_name = req.lora.name
                    lora_weight = req.lora.weight
                    if lora_name != self._current_pixel_lora or lora_weight != self._current_pixel_lora_weight:
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

                elif req.mode.value.startswith("controlnet_"):
                    image = self._controlnet_generate(req, generator, step_callback, effective_neg)

                else:
                    raise ValueError(f"Unknown mode: {req.mode}")

                # ── Post-process ──────────────────────────────────────
                image = postprocess_apply(image, req.post_process)

                # ── Encode result ─────────────────────────────────────
                buf = BytesIO()
                image.save(buf, format="PNG")
                b64_image = b64encode(buf.getvalue()).decode("ascii")

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
            width=_round8(req.width),
            height=_round8(req.height),
            generator=generator,
            clip_skip=req.clip_skip,
            callback_on_step_end=callback,
            output_type="pil",
        ).images[0]

    def _img2img(self, req, generator, callback, effective_neg):
        if req.source_image is None:
            raise ValueError("img2img requires source_image")
        source = _decode_b64_image(req.source_image)
        # Resize source to match requested dimensions for predictable output
        target_w, target_h = _round8(req.width), _round8(req.height)
        if source.size != (target_w, target_h):
            source = source.resize((target_w, target_h), Image.LANCZOS)
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

    def _controlnet_generate(self, req, generator, callback, effective_neg):
        if req.control_image is None:
            raise ValueError("ControlNet requires control_image")

        self._ensure_controlnet(req.mode)
        control = _decode_b64_image(req.control_image)

        width = _round8(req.width)
        height = _round8(req.height)

        # Resize control image to match generation dimensions
        if control.size != (width, height):
            control = control.resize((width, height), Image.LANCZOS)

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

    # ─── ANIMATEDIFF ─────────────────────────────────────────

    def _get_uncompiled_unet(self):
        """Return the raw UNet module, unwrapping torch.compile if present.

        AnimateDiff reshapes latents to 5D [batch, channels, frames, H, W].
        A torch.compile'd UNet has guards specialized for 4D warmup shapes
        and will fail with 'Expected 3D or 4D input to conv2d'.
        """
        unet = self._pipe.unet
        if hasattr(unet, "_orig_mod"):
            return unet._orig_mod
        return unet

    def _strip_peft_from_unet(self, unet) -> None:
        """Strip any remaining PEFT/LoRA wrappers directly from the UNet.

        After fuse_lora() + unload_lora_weights(), PEFT wrappers should be
        removed.  But when the UNet is wrapped by torch.compile, the pipeline
        API may traverse the compiled wrapper instead of the raw module,
        leaving base_layer / lora_A / lora_B artefacts in the state dict.

        This method operates on the raw uncompiled UNet to guarantee a clean
        state dict for UNetMotionModel.from_unet2d().
        """
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
            has_peft = any(isinstance(m, BaseTunerLayer) for m in unet.modules())
        except ImportError:
            has_peft = False

        if has_peft:
            log.info("Stripping residual PEFT wrappers from UNet for AnimateDiff")
            recurse_remove_peft_layers(unet)

        # Clean up PEFT metadata that blocks future adapter injection
        if hasattr(unet, "peft_config"):
            del unet.peft_config
        if hasattr(unet, "_hf_peft_config_loaded"):
            unet._hf_peft_config_loaded = None

    def _ensure_animatediff(self) -> None:
        """Lazy-load AnimateDiff motion adapter and pipeline."""
        if self._motion_adapter is not None:
            return

        log.info("Loading AnimateDiff motion adapter: %s", settings.animatediff_model)
        self._motion_adapter = MotionAdapter.from_pretrained(
            settings.animatediff_model,
            torch_dtype=torch.float16,
        ).to("cuda")

        # AnimateDiff requires the uncompiled UNet — compiled graphs
        # have 4D shape guards that reject the 5D video tensor.
        unet = self._get_uncompiled_unet()

        # Guarantee UNet has no PEFT wrappers before from_unet2d().
        # The fuse/unload pipeline API may miss wrappers when UNet is
        # behind torch.compile; call the low-level stripper directly.
        # NOTE: UNet weights already contain fused Hyper-SD + pixel_art LoRA
        # from set_pixel_lora() — stripping just removes the wrapper, weights
        # stay intact.  No re-fuse needed.
        self._strip_peft_from_unet(unet)

        # Build AnimateDiff pipeline from base components
        self._animatediff_pipe = AnimateDiffPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            tokenizer=self._pipe.tokenizer,
            unet=unet,
            motion_adapter=self._motion_adapter,
            scheduler=self._pipe.scheduler,
            feature_extractor=None,
        )
        self._animatediff_pipe.to("cuda")

        # Apply FreeU to AnimateDiff pipeline
        if settings.enable_freeu:
            self._animatediff_pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )

        log.info("AnimateDiff pipeline ready")

    def _ensure_animatediff_controlnet(self, mode: GenerationMode) -> None:
        """Lazy-load AnimateDiff + ControlNet combined pipeline."""
        if self._animatediff_controlnet_mode == mode and self._animatediff_controlnet_pipe is not None:
            return

        self._ensure_animatediff()

        model_id = _CONTROLNET_IDS.get(mode)
        if model_id is None:
            raise ValueError(f"No ControlNet for mode: {mode}")

        # Reuse existing ControlNet model if same mode already loaded for single-frame
        if self._controlnet_mode == mode and self._controlnet_pipe is not None:
            controlnet = self._controlnet_pipe.controlnet
        else:
            log.info("Loading ControlNet for AnimateDiff: %s", model_id)
            controlnet = ControlNetModel.from_pretrained(
                model_id, torch_dtype=torch.float16,
            ).to("cuda")

        # Use uncompiled UNet — same reason as _ensure_animatediff
        unet = self._get_uncompiled_unet()

        # Strip PEFT wrappers directly (same as _ensure_animatediff).
        # UNet already has fused weights — no re-fuse needed.
        self._strip_peft_from_unet(unet)

        self._animatediff_controlnet_pipe = AnimateDiffControlNetPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            tokenizer=self._pipe.tokenizer,
            unet=unet,
            motion_adapter=self._motion_adapter,
            controlnet=controlnet,
            scheduler=self._pipe.scheduler,
            feature_extractor=None,
        )
        self._animatediff_controlnet_pipe.to("cuda")

        if settings.enable_freeu:
            self._animatediff_controlnet_pipe.enable_freeu(
                s1=settings.freeu_s1, s2=settings.freeu_s2,
                b1=settings.freeu_b1, b2=settings.freeu_b2,
            )

        self._animatediff_controlnet_mode = mode
        log.info("AnimateDiff + ControlNet pipeline ready (mode=%s)", mode.value)

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
                if req.lora.name != self._current_pixel_lora or req.lora.weight != self._current_pixel_lora_weight:
                    self.set_pixel_lora(req.lora.name, req.lora.weight)

            if req.method == AnimationMethod.CHAIN:
                return self._generate_chain(req, on_frame, on_progress)
            elif req.method == AnimationMethod.ANIMATEDIFF:
                return self._generate_animatediff(req, on_frame, on_progress)
            else:
                raise ValueError(f"Unknown animation method: {req.method}")

        except torch.cuda.OutOfMemoryError:
            log.error("CUDA OOM during animation — clearing VRAM cache")
            torch.cuda.empty_cache()
            raise
        finally:
            self._cancel_event.clear()

    def _generate_chain(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
        """Frame-by-frame chaining: each frame feeds into the next via img2img."""
        # Disable DeepCache — its cur_timestep guard causes torch.compile
        # recompilation storm across multiple sequential generations
        deepcache_was_enabled = self._deepcache_helper is not None
        if deepcache_was_enabled:
            try:
                self._deepcache_helper.disable()
                log.info("DeepCache disabled for chain animation")
            except Exception as e:
                log.warning("Failed to disable DeepCache: %s", e)

        # Swap compiled UNet → raw _orig_mod for chain mode.
        # After Dynamo reset, frame 0 (txt2img) and frame 1 (img2img) would
        # each trigger 30-60s full recompilation due to different control flow.
        # Using the raw UNet avoids this entirely — the per-step penalty (~20%)
        # is negligible compared to the recompilation cost.
        compiled_unet = self._pipe.unet
        raw_unet = self._get_uncompiled_unet()
        swapped = raw_unet is not compiled_unet
        if swapped:
            self._pipe.unet = raw_unet
            self._img2img_pipe.unet = raw_unet
            if self._controlnet_pipe is not None:
                self._controlnet_pipe.unet = raw_unet
            log.info("Compiled UNet swapped for raw module (chain mode)")

        try:
            return self._generate_chain_inner(req, on_frame, on_progress)
        finally:
            # Restore compiled UNet for single-frame generation
            if swapped:
                self._pipe.unet = compiled_unet
                self._img2img_pipe.unet = compiled_unet
                if self._controlnet_pipe is not None:
                    self._controlnet_pipe.unet = compiled_unet
                log.info("Compiled UNet restored after chain animation")
            if deepcache_was_enabled and self._deepcache_helper is not None:
                try:
                    self._deepcache_helper.enable()
                    log.info("DeepCache re-enabled after chain animation")
                except Exception as e:
                    log.warning("Failed to re-enable DeepCache: %s", e)

    def _generate_chain_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
        """Core chain animation logic."""
        frames: list[AnimationFrameResponse] = []
        base_seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
        prev_image: Optional[Image.Image] = None
        t0_total = time.perf_counter()

        effective_neg = self._build_effective_negative(req.negative_prompt, req.negative_ti)

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
                    frame_seed = int(torch.randint(0, 2**32, (1,)).item())

                generator = torch.Generator("cuda").manual_seed(frame_seed)

                # Progress callback with frame context
                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    if self._cancel_event.is_set():
                        raise GenerationCancelled("Animation cancelled by client")
                    if on_progress:
                        on_progress(ProgressResponse(
                            step=step_idx + 1, total=req.steps,
                            frame_index=frame_idx, total_frames=req.frame_count,
                        ))
                    return callback_kwargs

                # Frame 0: use source or generate from scratch
                if frame_idx == 0:
                    if req.mode == GenerationMode.TXT2IMG:
                        image = self._txt2img(
                            _to_gen_req(req, frame_seed), generator, step_callback, effective_neg,
                        )
                    elif req.mode == GenerationMode.IMG2IMG:
                        image = self._img2img(
                            _to_gen_req(req, frame_seed), generator, step_callback, effective_neg,
                        )
                    elif req.mode.value.startswith("controlnet_"):
                        image = self._controlnet_generate(
                            _to_gen_req(req, frame_seed), generator, step_callback, effective_neg,
                        )
                    else:
                        raise ValueError(f"Unknown mode: {req.mode}")
                else:
                    # Frame 1+: img2img from previous frame at denoise_strength
                    source = prev_image
                    target_w, target_h = _round8(req.width), _round8(req.height)
                    if source.size != (target_w, target_h):
                        source = source.resize((target_w, target_h), Image.LANCZOS)

                    if req.mode.value.startswith("controlnet_") and req.control_image is not None:
                        # ControlNet chain: use original control_image as anchor,
                        # previous frame as conditioning via img2img-like denoise
                        self._ensure_controlnet(req.mode)
                        control = _decode_b64_image(req.control_image)
                        if control.size != (target_w, target_h):
                            control = control.resize((target_w, target_h), Image.LANCZOS)
                        torch.compiler.cudagraph_mark_step_begin()
                        image = self._controlnet_pipe(
                            prompt=req.prompt,
                            negative_prompt=effective_neg,
                            image=control,
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
                        # Standard img2img chain
                        torch.compiler.cudagraph_mark_step_begin()
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

                # Post-process
                image = postprocess_apply(image, req.post_process)
                prev_image = image

                # Encode
                buf = BytesIO()
                image.save(buf, format="PNG")
                b64_image = b64encode(buf.getvalue()).decode("ascii")
                w, h = image.size
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
        # Disable DeepCache (conflicts with motion module UNet wrapping)
        deepcache_was_enabled = self._deepcache_helper is not None
        if deepcache_was_enabled:
            try:
                self._deepcache_helper.disable()
                log.info("DeepCache disabled for AnimateDiff generation")
            except Exception as e:
                log.warning("Failed to disable DeepCache: %s", e)

        try:
            return self._generate_animatediff_inner(req, on_frame, on_progress)
        finally:
            # Re-enable DeepCache
            if deepcache_was_enabled and self._deepcache_helper is not None:
                try:
                    self._deepcache_helper.enable()
                    log.info("DeepCache re-enabled after AnimateDiff generation")
                except Exception as e:
                    log.warning("Failed to re-enable DeepCache: %s", e)

    def _generate_animatediff_inner(
        self,
        req: AnimationRequest,
        on_frame: Optional[Callable[[AnimationFrameResponse], None]],
        on_progress: Optional[Callable[[ProgressResponse], None]],
    ) -> list[AnimationFrameResponse]:
        """Core AnimateDiff generation logic."""
        is_controlnet = req.mode.value.startswith("controlnet_")

        if is_controlnet:
            self._ensure_animatediff_controlnet(req.mode)
            pipe = self._animatediff_controlnet_pipe
        else:
            self._ensure_animatediff()
            pipe = self._animatediff_pipe

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

        seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
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
                width=_round8(req.width),
                height=_round8(req.height),
                generator=generator,
                clip_skip=req.clip_skip,
                callback_on_step_end=step_callback,
                output_type="pil",
            )

            if is_controlnet and req.control_image is not None:
                control = _decode_b64_image(req.control_image)
                target_w, target_h = _round8(req.width), _round8(req.height)
                if control.size != (target_w, target_h):
                    control = control.resize((target_w, target_h), Image.LANCZOS)
                # Repeat control image for all frames
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
            image = postprocess_apply(pil_img, req.post_process)

            buf = BytesIO()
            image.save(buf, format="PNG")
            b64_image = b64encode(buf.getvalue()).decode("ascii")
            w, h = image.size
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

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


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def _decode_b64_image(data: str) -> Image.Image:
    try:
        raw = b64decode(data)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e


def _to_gen_req(anim_req: AnimationRequest, seed: int) -> GenerateRequest:
    """Convert AnimationRequest to GenerateRequest for single-frame dispatch."""
    return GenerateRequest(
        prompt=anim_req.prompt,
        negative_prompt=anim_req.negative_prompt,
        mode=anim_req.mode,
        width=anim_req.width,
        height=anim_req.height,
        source_image=anim_req.source_image,
        control_image=anim_req.control_image,
        seed=seed,
        steps=anim_req.steps,
        cfg_scale=anim_req.cfg_scale,
        denoise_strength=anim_req.denoise_strength,
        clip_skip=anim_req.clip_skip,
        lora=anim_req.lora,
        negative_ti=anim_req.negative_ti,
        post_process=anim_req.post_process,
    )
