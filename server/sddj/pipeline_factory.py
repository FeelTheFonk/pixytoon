"""Pipeline construction — base, img2img, ControlNet, scheduler, attention, compile."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from .config import settings
from .freeu_applicator import apply_freeu
from .protocol import GenerationMode

log = logging.getLogger("sddj.pipeline_factory")

# ── Original SDPA reference (for SageAttention restore) ──────
_original_sdpa = None

# ── Torch version (computed once at import) ───────────────────
_TORCH_MAJOR = int(torch.__version__.split(".")[0])

# ── Pipeline caches ───────────────────────────────────────────
# M-37: Thread lock for pipeline cache access
_pipeline_lock = threading.Lock()
_pipeline_cache: dict[str, StableDiffusionPipeline] = {}
# M-37: Keyed by stable string (checkpoint path), not id() which can be reused after GC
_img2img_cache: dict[str, StableDiffusionImg2ImgPipeline] = {}

# ─────────────────────────────────────────────────────────────
# CONTROLNET MODEL IDS
# ─────────────────────────────────────────────────────────────
CONTROLNET_IDS: dict[GenerationMode, str] = {
    GenerationMode.CONTROLNET_OPENPOSE: "lllyasviel/control_v11p_sd15_openpose",
    GenerationMode.CONTROLNET_CANNY: "lllyasviel/control_v11p_sd15_canny",
    GenerationMode.CONTROLNET_SCRIBBLE: "lllyasviel/control_v11p_sd15_scribble",
    GenerationMode.CONTROLNET_LINEART: "lllyasviel/control_v11p_sd15_lineart",
    GenerationMode.CONTROLNET_QRCODE: "monster-labs/control_v1p_sd15_qrcode_monster",
}


def load_base_pipeline() -> StableDiffusionPipeline:
    """Load the base SD1.5 pipeline onto CUDA with fp16.

    Accepts three checkpoint formats via settings.default_checkpoint:
      - HuggingFace repo ID:  "Lykon/dreamshaper-8"
      - Local diffusers dir:  "/path/to/model_dir/"
      - Single file (.safetensors/.ckpt): "/path/to/model.safetensors"
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SDDj diffusion pipeline")

    ckpt = settings.default_checkpoint
    cache_key = str(ckpt)
    # M-13/M-37: Hold lock for the entire load to prevent double GPU allocation.
    # Load is rare (startup only) so contention is negligible.
    with _pipeline_lock:
        if cache_key in _pipeline_cache:
            log.debug("Pipeline cache hit: %s", cache_key)
            return _pipeline_cache[cache_key]

        ckpt_path = Path(ckpt)
        log.info("Loading base pipeline: %s", ckpt)

        if ckpt_path.is_file() and ckpt_path.suffix in (".safetensors", ".ckpt"):
            pipe = StableDiffusionPipeline.from_single_file(
                str(ckpt_path),
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
                config="runwayml/stable-diffusion-v1-5",
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

        # PAG: inform user at load time if enabled
        if settings.enable_pag:
            log.info(
                "PAG config enabled — pag_scale will be passed to pipeline if supported. "
                "To use PAG, load a PAG-compatible pipeline variant "
                "(e.g. StableDiffusionPAGPipeline)."
            )

        _pipeline_cache[cache_key] = pipe
    return pipe


def setup_attention(pipe: StableDiffusionPipeline) -> None:
    """Configure best available attention: SageAttention2 > SDP (native) > xformers > slicing.

    Priority (when backend="auto"):
      1. SageAttention2 — monkey-patches F.scaled_dot_product_attention with sageattn
         for ~1.89× speedup on Ada Lovelace (RTX 40xx) GPUs.
      2. SDP (PyTorch >= 2.0) — native AttnProcessor2_0, no explicit call needed.
      3. xformers — memory-efficient attention for older PyTorch.
      4. Attention slicing — last resort fallback.
    """
    global _original_sdpa
    backend = settings.attention_backend

    # ── SageAttention2 (highest priority when requested or auto) ──
    if backend in ("sage", "auto"):
        try:
            from sageattention import sageattn  # noqa: F401
            import torch.nn.functional as _F
            if _F.scaled_dot_product_attention is not sageattn:
                _original_sdpa = _F.scaled_dot_product_attention
                _F.scaled_dot_product_attention = sageattn
                log.info("SageAttention2 enabled (1.89× attention speedup on Ada Lovelace)")
            return
        except ImportError:
            if backend == "sage":
                log.warning(
                    "SageAttention2 requested but sageattention package not installed — "
                    "falling back to next available backend"
                )
            else:
                log.info("SageAttention2 not available — using SDP (pip install sageattention for ~1.89× speedup)")
        except Exception as e:
            log.warning("SageAttention2 init failed (%s), falling back", e)

    # ── SDP (PyTorch >= 2.0, native) ─────────────────────────────
    if backend in ("sdp", "auto"):
        if _TORCH_MAJOR >= 2:
            log.info("PyTorch %s: SDP attention active (native AttnProcessor2_0)", torch.__version__)
            return
        if backend == "sdp":
            log.warning("SDP requested but PyTorch %s < 2.0 — falling back", torch.__version__)

    # ── xformers ─────────────────────────────────────────────────
    if backend in ("xformers", "auto"):
        try:
            import xformers  # noqa: F401
            pipe.enable_xformers_memory_efficient_attention()
            log.info("xformers memory-efficient attention enabled")
            return
        except ImportError:
            if backend == "xformers":
                log.warning("xformers requested but not installed — falling back")
            else:
                log.debug("xformers not installed, trying fallback")
        except Exception as e:
            log.warning("xformers init failed (%s), falling back", e)

    # ── Attention slicing (last resort) ──────────────────────────
    if settings.enable_attention_slicing:
        pipe.enable_attention_slicing()
        log.info("Attention slicing enabled (fallback)")


def restore_attention():
    """Restore original SDPA if SageAttention was monkey-patched."""
    global _original_sdpa
    if _original_sdpa is not None:
        import torch.nn.functional as _F
        _F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
        log.info("Restored original scaled_dot_product_attention")


def setup_vae(pipe: StableDiffusionPipeline) -> None:
    """Enable VAE tiling and slicing for VRAM savings."""
    if settings.enable_vae_tiling:
        pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()


def setup_hyper_sd(pipe: StableDiffusionPipeline) -> None:
    """Load Hyper-SD LoRA, fuse permanently, then unload adapter."""
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        timestep_spacing="trailing",
    )

    # Enable LoRA hotswap BEFORE the first load_lora_weights() call.
    # This allows subsequent style LoRA switches to avoid torch.compile
    # recompilation (saves ~15-25s per switch).  target_rank must be >= the
    # max rank across ALL LoRAs (HyperSD + any style LoRAs).
    # Caveats: UNet adapters only (no text_encoder), same-layer targeting.
    if settings.enable_lora_hotswap:
        try:
            pipe.enable_lora_hotswap(target_rank=settings.max_lora_rank)
            log.info("LoRA hotswap enabled (target_rank=%d)", settings.max_lora_rank)
        except Exception as e:
            log.warning("LoRA hotswap unavailable (diffusers too old?): %s", e)

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


def apply_unet_quantization(pipe: StableDiffusionPipeline) -> None:
    """Quantize UNet weights via torchao for reduced VRAM and faster matmuls.

    Must be called AFTER LoRA fuse (quantization bakes into the fused weights)
    and BEFORE torch.compile (Inductor fuses int8/fp8 matmuls with epilogues).

    Auto-detection selects the best dtype for the GPU microarchitecture:
      - Ada Lovelace (sm89, RTX 40xx): fp8_dynamic_activation_fp8_weight
      - Ampere (sm80/sm86, RTX 30xx/A100): int8_dynamic_activation_int8_weight
      - Turing or older: skip with warning (no hardware acceleration for quantized matmul)
    """
    if not settings.enable_unet_quantization:
        return

    try:
        from torchao.quantization import quantize_
    except ImportError:
        log.warning("UNet quantization requested but torchao not installed — skipping")
        return

    dtype = settings.unet_quantization_dtype

    if dtype == "auto":
        # Auto-detect based on GPU compute capability
        if not torch.cuda.is_available():
            log.warning("UNet quantization: no CUDA device — skipping")
            return
        device = pipe.unet.device
        cap = torch.cuda.get_device_capability(device)
        sm = cap[0] * 10 + cap[1]  # e.g. (8, 9) → 89

        if sm >= 89:
            # Ada Lovelace (sm89): native FP8 tensor cores
            dtype = "fp8dq"
            log.info("Auto-detected Ada Lovelace (sm%d) → fp8 dynamic quantization", sm)
        elif sm >= 80:
            # Ampere (sm80/sm86): INT8 tensor cores
            dtype = "int8dq"
            log.info("Auto-detected Ampere (sm%d) → int8 dynamic quantization", sm)
        else:
            log.warning(
                "UNet quantization: GPU compute capability sm%d (Turing or older) "
                "lacks efficient quantized matmul — skipping", sm
            )
            return

    # Resolve the quantization function for the chosen dtype
    try:
        if dtype == "int8dq":
            from torchao.quantization import int8_dynamic_activation_int8_weight
            quant_fn = int8_dynamic_activation_int8_weight
        elif dtype == "int8wo":
            from torchao.quantization import int8_weight_only
            quant_fn = int8_weight_only
        elif dtype == "fp8dq":
            from torchao.quantization import float8_dynamic_activation_float8_weight
            quant_fn = float8_dynamic_activation_float8_weight
        elif dtype == "fp8wo":
            from torchao.quantization import float8_weight_only
            quant_fn = float8_weight_only
        else:
            log.warning("Unknown quantization dtype '%s' — skipping", dtype)
            return
    except ImportError:
        log.warning("torchao does not support '%s' quantization — skipping", dtype)
        return

    try:
        quantize_(pipe.unet, quant_fn())
        log.info("UNet quantized with torchao (%s)", dtype)
    except Exception as e:
        log.warning("UNet quantization failed (%s): %s — continuing without quantization", dtype, e)


def _resolve_compile_mode() -> str:
    """Auto-select compile mode based on GPU SM count.

    Both max-autotune and max-autotune-no-cudagraphs trigger exhaustive GEMM
    kernel benchmarking (max_autotune_gemm) which requires a minimum SM count
    to produce any benefit.  GPUs below 40 SMs (RTX 4060=24, RTX 4060 Ti=34)
    waste minutes on GEMM search with zero gain and risk ptxas crashes.

    Thresholds:
      ≥68 SMs + Ampere+  → max-autotune         (full GEMM + CUDA graphs)
      ≥40 SMs + Turing+  → max-autotune-no-cudagraphs  (GEMM search, no graphs)
      <40 SMs             → default              (fast compile, no GEMM search)
    """
    mode = settings.compile_mode
    if settings.auto_compile_mode and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
        cap = torch.cuda.get_device_capability()
        sm_arch = cap[0] * 10 + cap[1]
        if sm_count >= 68 and sm_arch >= 80:
            mode = "max-autotune"
        elif sm_count >= 40 and sm_arch >= 75:
            mode = "max-autotune-no-cudagraphs"
        else:
            mode = "default"
        log.info("Auto-selected compile_mode=%s (sm%d, %d SMs)", mode, sm_arch, sm_count)
    return mode


def apply_torch_compile(pipe: StableDiffusionPipeline) -> None:
    """torch.compile the UNet if enabled.

    BEFORE DeepCache (DeepCache wraps the forward).
    fullgraph=False required: DeepCache introduces dynamic control flow.
    mode=default: Triton codegen without kernel benchmarking.
    (reduce-overhead uses CUDAGraphs which is INCOMPATIBLE with DeepCache)

    dynamic shapes: DeepCache is a *dynamic* inference algorithm that mutates
    the UNet's forward() at runtime.  The DeepCache maintainer has confirmed
    that DeepCache is fundamentally incompatible with torch.compile's graph
    capture (GitHub: horseee/DeepCache).  Our workaround: compile FIRST with
    fullgraph=False, then let DeepCache wrap the forwards.  This works because
    Dynamo traces through DeepCache's modified forwards at warmup time.
    Setting dynamic=True here would destabilize the guards DeepCache relies on,
    so dynamic=True is only safe when DeepCache is DISABLED.
    """
    if not settings.enable_torch_compile:
        return
    mode = _resolve_compile_mode()
    # Inductor config flags for diffusion model optimization (PyTorch blog, July 2025)
    torch._inductor.config.conv_1x1_as_mm = True
    # Coordinate descent kernel tuning — only beneficial for max-autotune modes
    # (exhaustive GEMM benchmarking). Wastes 10-60s on "default" mode with no gain.
    if mode.startswith("max-autotune"):
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.coordinate_descent_check_all_directions = True
    # INT8/mixed-precision fusion flags — only needed for torchao quantized models
    if settings.enable_unet_quantization:
        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.force_fuse_int_mm_with_mul = True
        torch._inductor.config.use_mixed_mm = True
    use_dynamic = settings.compile_dynamic and not settings.enable_deepcache
    try:
        pipe.unet = torch.compile(
            pipe.unet,
            mode=mode,
            fullgraph=False,
            dynamic=use_dynamic,
        )
        log.info("torch.compile enabled for UNet (mode=%s, dynamic=%s)",
                 mode, use_dynamic)
    except Exception as e:
        log.warning("torch.compile failed: %s", e)


def apply_vae_compile(pipe: StableDiffusionPipeline) -> None:
    """torch.compile the VAE decoder for faster image decode."""
    if not settings.enable_torch_compile:
        return
    mode = _resolve_compile_mode()
    try:
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=mode, fullgraph=True)
        log.info("torch.compile enabled for VAE decode (mode=%s)", mode)
    except Exception as e_full:
        log.warning(
            "VAE compile fullgraph=True failed (%s) — this is unexpected for standard SD1.5 VAE. "
            "Falling back to fullgraph=False. Investigate if using a custom VAE.", e_full,
        )
        try:
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode=mode, fullgraph=False)
            log.info("torch.compile enabled for VAE decode (mode=%s, fullgraph=False fallback)", mode)
        except Exception as e:
            log.warning("VAE compile failed: %s — continuing uncompiled", e)


def create_img2img_pipeline(
    pipe: StableDiffusionPipeline,
) -> StableDiffusionImg2ImgPipeline:
    """Create img2img pipeline from base components.

    CRITICAL: scheduler must be a separate instance — txt2img and img2img
    both call set_timesteps() which mutates internal state. Sharing one
    scheduler causes chain animation to deadlock on the 3rd frame.
    """
    # M-37: Stable cache key based on checkpoint path, not id() which can alias after GC
    pipe_key = str(settings.default_checkpoint)
    with _pipeline_lock:
        if pipe_key in _img2img_cache:
            cached = _img2img_cache[pipe_key]
            cached.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
            return cached

    img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config),
        safety_checker=None,
        feature_extractor=None,
    )
    apply_freeu(img2img)
    with _pipeline_lock:
        # Double-check: another thread may have created it concurrently
        if pipe_key in _img2img_cache:
            return _img2img_cache[pipe_key]
        _img2img_cache[pipe_key] = img2img
    return img2img


def fresh_scheduler(base_pipe):
    """Create a fresh scheduler from the base pipeline's config.

    Used in chain animation to reset scheduler state between frames.
    The scheduler's internal state (timesteps, _step_index, num_inference_steps)
    mutates during each inference call and can accumulate stale state.
    """
    return type(base_pipe.scheduler).from_config(base_pipe.scheduler.config)


def create_lightning_scheduler(base_config):
    """Create EulerDiscreteScheduler configured for AnimateDiff-Lightning."""
    from diffusers import EulerDiscreteScheduler
    return EulerDiscreteScheduler.from_config(
        base_config,
        timestep_spacing="trailing",
        beta_schedule="linear",
        clip_sample=False,
    )


def create_controlnet_pipeline(
    pipe: StableDiffusionPipeline,
    mode: GenerationMode,
) -> tuple[StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline]:
    """Create ControlNet pipeline for the given mode."""
    model_id = CONTROLNET_IDS.get(mode)
    if model_id is None:
        raise ValueError(f"No ControlNet for mode: {mode}")

    log.info("Loading ControlNet: %s", model_id)
    load_kwargs: dict = dict(torch_dtype=torch.float16, local_files_only=True)
    # QR Code Monster v2: model weights in v2/ subfolder
    if mode == GenerationMode.CONTROLNET_QRCODE:
        load_kwargs["subfolder"] = "v2"
    controlnet = ControlNetModel.from_pretrained(
        model_id, **load_kwargs,
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

    # Img2Img ControlNet pipeline — shared ControlNet model, no extra VRAM
    # Used for QR illusion art: source image + QR conditioning + denoise
    cn_img2img_pipe = StableDiffusionControlNetImg2ImgPipeline(
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
    apply_freeu(cn_img2img_pipe)
    setup_vae(cn_pipe)
    setup_vae(cn_img2img_pipe)
    return cn_pipe, cn_img2img_pipe


def clear_pipeline_cache():
    """Invalidate all pipeline caches (call when model changes)."""
    with _pipeline_lock:
        _pipeline_cache.clear()
        _img2img_cache.clear()


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
