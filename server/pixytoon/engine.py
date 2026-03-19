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
from PIL import Image

# Suppress known harmless library warnings
warnings.filterwarnings("ignore", message=".*safety checker.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
warnings.filterwarnings("ignore", message=".*No LoRA keys associated to CLIPTextModel.*")
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from .config import settings
from .lora_manager import resolve_lora_path
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


def _round8(v: int) -> int:
    """Round to nearest multiple of 8 (SD1.5 VAE requirement)."""
    return (v // 8) * 8


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
        #    triton / triton-windows is a project dependency
        if settings.enable_torch_compile:
            try:
                self._pipe.unet = torch.compile(
                    self._pipe.unet,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                log.info("torch.compile enabled for UNet (reduce-overhead)")
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

        elapsed = time.perf_counter() - t0
        log.info("Pipeline loaded in %.1fs", elapsed)
        self._loaded = True

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
        self._loaded = False
        rembg_wrapper.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Pipeline unloaded")

    # ─── LORA MANAGEMENT ────────────────────────────────────

    def set_pixel_lora(self, name: Optional[str], weight: float = 0.8) -> None:
        """Load or switch pixel art LoRA (fused into weights, no PEFT runtime)."""
        if not self._loaded or self._pipe is None:
            return

        # Unfuse previous pixel art LoRA if any
        if self._current_pixel_lora is not None:
            try:
                self._pipe.unfuse_lora()
                self._pipe.unload_lora_weights()
            except Exception as e:
                log.warning("Failed to unfuse pixel art LoRA '%s': %s",
                            self._current_pixel_lora, e)
            self._current_pixel_lora = None

        if name is None:
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

        with torch.inference_mode():
            t0 = time.perf_counter()

            # Resolve seed
            seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32, (1,)).item())
            generator = torch.Generator("cuda").manual_seed(seed)

            # Set pixel art LoRA if requested
            lora_name = req.lora.name if req.lora else None
            lora_weight = req.lora.weight if req.lora else 0.8
            if lora_name != self._current_pixel_lora or lora_weight != self._current_pixel_lora_weight:
                self.set_pixel_lora(lora_name, lora_weight)

            # Progress callback adapter with cancellation support
            def step_callback(pipe, step_idx, timestep, callback_kwargs):
                if self._cancelled:
                    raise RuntimeError("Generation cancelled by client")
                if on_progress:
                    on_progress(ProgressResponse(step=step_idx + 1, total=req.steps))
                return callback_kwargs

            # ── Mode dispatch ─────────────────────────────────────

            if req.mode == GenerationMode.TXT2IMG:
                image = self._txt2img(req, generator, step_callback)

            elif req.mode == GenerationMode.IMG2IMG:
                image = self._img2img(req, generator, step_callback)

            elif req.mode.value.startswith("controlnet_"):
                image = self._controlnet_generate(req, generator, step_callback)

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

    # ─── PRIVATE GENERATION METHODS ──────────────────────────

    def _txt2img(self, req, generator, callback):
        return self._pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=_round8(req.width),
            height=_round8(req.height),
            generator=generator,
            callback_on_step_end=callback,
        ).images[0]

    def _img2img(self, req, generator, callback):
        if req.source_image is None:
            raise ValueError("img2img requires source_image")
        source = _decode_b64_image(req.source_image)
        return self._img2img_pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=source,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            strength=req.denoise_strength,
            generator=generator,
            callback_on_step_end=callback,
        ).images[0]

    def _controlnet_generate(self, req, generator, callback):
        if req.control_image is None:
            raise ValueError("ControlNet requires control_image")

        self._ensure_controlnet(req.mode)
        control = _decode_b64_image(req.control_image)

        width = _round8(req.width)
        height = _round8(req.height)

        return self._controlnet_pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=control,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            width=width,
            height=height,
            generator=generator,
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
