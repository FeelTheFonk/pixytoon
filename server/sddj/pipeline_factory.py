"""Pipeline construction — base, img2img, ControlNet, scheduler, attention, compile."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from .config import settings
from .freeu_applicator import apply_freeu
from .protocol import GenerationMode

log = logging.getLogger("sddj.pipeline_factory")

# ─────────────────────────────────────────────────────────────
# CONTROLNET MODEL IDS
# ─────────────────────────────────────────────────────────────
CONTROLNET_IDS: dict[GenerationMode, str] = {
    GenerationMode.CONTROLNET_OPENPOSE: "lllyasviel/control_v11p_sd15_openpose",
    GenerationMode.CONTROLNET_CANNY: "lllyasviel/control_v11p_sd15_canny",
    GenerationMode.CONTROLNET_SCRIBBLE: "lllyasviel/control_v11p_sd15_scribble",
    GenerationMode.CONTROLNET_LINEART: "lllyasviel/control_v11p_sd15_lineart",
}


def load_base_pipeline() -> StableDiffusionPipeline:
    """Load the base SD1.5 pipeline onto CUDA with fp16.

    Accepts three checkpoint formats via settings.default_checkpoint:
      - HuggingFace repo ID:  "Lykon/dreamshaper-8"
      - Local diffusers dir:  "/path/to/model_dir/"
      - Single file (.safetensors/.ckpt): "/path/to/model.safetensors"
    """
    ckpt = settings.default_checkpoint
    ckpt_path = Path(ckpt)
    log.info("Loading base pipeline: %s", ckpt)

    if ckpt_path.is_file() and ckpt_path.suffix in (".safetensors", ".ckpt"):
        pipe = StableDiffusionPipeline.from_single_file(
            str(ckpt_path),
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            ckpt,
            torch_dtype=torch.float16,
            safety_checker=None,
            variant="fp16",
            local_files_only=True,
        )
    pipe.to("cuda")
    return pipe


def setup_attention(pipe: StableDiffusionPipeline) -> None:
    """Configure best available attention: SDP (native) > xformers > slicing.

    PyTorch >= 2.0 uses scaled_dot_product_attention automatically in
    diffusers via AttnProcessor2_0 — no explicit call needed. We only
    need xformers or slicing as fallbacks for older PyTorch.
    """
    torch_major = int(torch.__version__.split(".")[0])
    if torch_major >= 2:
        log.info("PyTorch %s: SDP attention active (native AttnProcessor2_0)", torch.__version__)
        return

    try:
        import xformers  # noqa: F401
        pipe.enable_xformers_memory_efficient_attention()
        log.info("xformers memory-efficient attention enabled")
        return
    except ImportError:
        log.debug("xformers not installed, trying fallback")
    except Exception as e:
        log.warning("xformers init failed (%s), falling back", e)

    if settings.enable_attention_slicing:
        pipe.enable_attention_slicing()
        log.info("Attention slicing enabled (fallback)")


def setup_vae(pipe: StableDiffusionPipeline) -> None:
    """Enable VAE tiling and slicing for VRAM savings."""
    if settings.enable_vae_tiling:
        pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()


def setup_hyper_sd(pipe: StableDiffusionPipeline) -> None:
    """Load Hyper-SD LoRA, fuse permanently, then unload adapter."""
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    log.info("Loading Hyper-SD LoRA: %s/%s", settings.hyper_sd_repo, settings.hyper_sd_lora_file)
    try:
        pipe.load_lora_weights(
            settings.hyper_sd_repo,
            weight_name=settings.hyper_sd_lora_file,
            adapter_name="hyper_sd",
            local_files_only=True,
        )
        pipe.fuse_lora(lora_scale=settings.hyper_sd_fuse_scale)
        pipe.unload_lora_weights()
    except Exception as e:
        log.error("Hyper-SD setup failed: %s", e)
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        raise RuntimeError(f"Failed to setup Hyper-SD: {e}") from e
    log.info("Hyper-SD LoRA fused into UNet weights (scale=%.3f)", settings.hyper_sd_fuse_scale)


def apply_torch_compile(pipe: StableDiffusionPipeline) -> None:
    """torch.compile the UNet if enabled.

    BEFORE DeepCache (DeepCache wraps the forward).
    fullgraph=False required: DeepCache introduces dynamic control flow.
    mode=default: Triton codegen without kernel benchmarking.
    (reduce-overhead uses CUDAGraphs which is INCOMPATIBLE with DeepCache)
    """
    if not settings.enable_torch_compile:
        return
    try:
        pipe.unet = torch.compile(
            pipe.unet,
            mode=settings.compile_mode,
            fullgraph=False,
        )
        log.info("torch.compile enabled for UNet (%s)", settings.compile_mode)
    except Exception as e:
        log.warning("torch.compile failed: %s", e)


def create_img2img_pipeline(
    pipe: StableDiffusionPipeline,
) -> StableDiffusionImg2ImgPipeline:
    """Create img2img pipeline from base components.

    CRITICAL: scheduler must be a separate instance — txt2img and img2img
    both call set_timesteps() which mutates internal state. Sharing one
    scheduler causes chain animation to deadlock on the 3rd frame.
    """
    img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=copy.deepcopy(pipe.scheduler),
        safety_checker=None,
        feature_extractor=None,
    )
    apply_freeu(img2img)
    return img2img


def fresh_scheduler(base_pipe):
    """Create a fresh scheduler from the base pipeline's config.

    Used in chain animation to reset scheduler state between frames.
    The scheduler's internal state (timesteps, _step_index, num_inference_steps)
    mutates during each inference call and can accumulate stale state.
    """
    return type(base_pipe.scheduler).from_config(base_pipe.scheduler.config)


def create_controlnet_pipeline(
    pipe: StableDiffusionPipeline,
    mode: GenerationMode,
) -> StableDiffusionControlNetPipeline:
    """Create ControlNet pipeline for the given mode."""
    model_id = CONTROLNET_IDS.get(mode)
    if model_id is None:
        raise ValueError(f"No ControlNet for mode: {mode}")

    log.info("Loading ControlNet: %s", model_id)
    controlnet = ControlNetModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    cn_pipe = StableDiffusionControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config),
        safety_checker=None,
        feature_extractor=None,
    )

    apply_freeu(cn_pipe)
    setup_vae(cn_pipe)
    return cn_pipe


def get_controlnet_from_pipe(
    cn_pipe: Optional[StableDiffusionControlNetPipeline],
    cn_mode: Optional[GenerationMode],
    mode: GenerationMode,
) -> Optional[ControlNetModel]:
    """Return existing ControlNet model if compatible, else None."""
    if cn_mode == mode and cn_pipe is not None:
        cn = getattr(cn_pipe, 'controlnet', None)
        return cn
    return None
