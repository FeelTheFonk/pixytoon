"""SOTA Diffusion Engine — SD1.5 + Hyper-SD (fused) + DeepCache + FreeU v2.

Manages pipeline lifecycle, LoRA fusing, ControlNet lazy-loading,
and progress callbacks for the WebSocket server.
"""

from __future__ import annotations

import logging
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
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from .config import settings
from .lora_manager import list_loras, resolve_lora_path
from .ti_manager import list_embeddings, resolve_embedding_path
from .postprocess import apply as postprocess_apply
from .protocol import (
    GenerateRequest,
    GenerationMode,
    ProgressResponse,
    ResultResponse,
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
        self._cancelled = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def cancel(self) -> None:
        """Signal cancellation — checked at each step callback."""
        self._cancelled = True

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

    def generate(
        self,
        req: GenerateRequest,
        on_progress: Optional[Callable[[ProgressResponse], None]] = None,
    ) -> ResultResponse:
        """Run the full generation pipeline: diffusion → post-process → encode."""
        if not self._loaded:
            self.load()

        self._cancelled = False

        try:
            with torch.inference_mode():
                t0 = time.perf_counter()

                # Resolve seed
                seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
                generator = torch.Generator("cuda").manual_seed(seed)

                # Set pixel art LoRA if requested
                lora_name = req.lora.name if req.lora else None
                lora_weight = req.lora.weight if req.lora else 1.0
                if lora_name != self._current_pixel_lora or lora_weight != self._current_pixel_lora_weight:
                    self.set_pixel_lora(lora_name, lora_weight)

                # Build effective negative prompt with TI tokens
                effective_neg = req.negative_prompt or ""
                if req.negative_ti:
                    ti_parts = []
                    for ti_spec in req.negative_ti:
                        if ti_spec.name in self._loaded_ti_tokens:
                            if abs(ti_spec.weight - 1.0) < 0.01:
                                ti_parts.append(ti_spec.name)
                            else:
                                ti_parts.append(f"({ti_spec.name}:{ti_spec.weight:.2f})")
                        else:
                            log.warning("TI token '%s' not loaded — skipping", ti_spec.name)
                    if ti_parts:
                        ti_str = ", ".join(ti_parts)
                        effective_neg = f"{effective_neg}, {ti_str}" if effective_neg else ti_str

                # Progress callback adapter with cancellation support
                def step_callback(pipe, step_idx, timestep, callback_kwargs):
                    if self._cancelled:
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
        ).images[0]

    def _img2img(self, req, generator, callback, effective_neg):
        if req.source_image is None:
            raise ValueError("img2img requires source_image")
        source = _decode_b64_image(req.source_image)
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
        ).images[0]

    def _controlnet_generate(self, req, generator, callback, effective_neg):
        if req.control_image is None:
            raise ValueError("ControlNet requires control_image")

        self._ensure_controlnet(req.mode)
        control = _decode_b64_image(req.control_image)

        width = _round8(req.width)
        height = _round8(req.height)

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
        ).images[0]


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def _decode_b64_image(data: str) -> Image.Image:
    try:
        raw = b64decode(data)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e
