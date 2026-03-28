"""AnimateDiff pipeline lifecycle — adapter, PEFT strip, pipeline construction."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffPipeline,
    AnimateDiffVideoToVideoPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
    MotionAdapter,
)
from diffusers.utils.peft_utils import recurse_remove_peft_layers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .config import settings
from .freeu_applicator import apply_freeu
from .pipeline_factory import CONTROLNET_IDS
from .protocol import GenerationMode

log = logging.getLogger("sddj.animatediff_manager")


def get_uncompiled_unet(pipe):
    """Return the raw UNet module, unwrapping torch.compile if present.

    AnimateDiff reshapes latents to 5D [batch, channels, frames, H, W].
    A torch.compile'd UNet has guards specialized for 4D warmup shapes
    and will fail with 'Expected 3D or 4D input to conv2d'.
    """
    unet = pipe.unet
    if hasattr(unet, "_orig_mod"):
        return unet._orig_mod
    return unet


def strip_peft_from_unet(unet) -> None:
    """Strip any remaining PEFT/LoRA wrappers directly from the UNet.

    After fuse_lora() + unload_lora_weights(), PEFT wrappers should be
    removed.  But when the UNet is wrapped by torch.compile, the pipeline
    API may traverse the compiled wrapper instead of the raw module,
    leaving base_layer / lora_A / lora_B artefacts in the state dict.

    Operates on the raw uncompiled UNet to guarantee a clean state dict
    for UNetMotionModel.from_unet2d().
    """
    # Fast path: skip full module traversal if no PEFT artifacts exist
    if not hasattr(unet, "peft_config") and not hasattr(unet, "_hf_peft_config_loaded"):
        return

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
        has_peft = any(isinstance(m, BaseTunerLayer) for m in unet.modules())
    except ImportError:
        has_peft = False

    if has_peft:
        log.info("Stripping residual PEFT wrappers from UNet for AnimateDiff")
        recurse_remove_peft_layers(unet)

    if hasattr(unet, "peft_config"):
        del unet.peft_config
    if hasattr(unet, "_hf_peft_config_loaded"):
        unet._hf_peft_config_loaded = None


class AnimateDiffManager:
    """Manages AnimateDiff motion adapter and pipeline lifecycle."""

    def __init__(self) -> None:
        self.motion_adapter: Optional[MotionAdapter] = None
        self.pipe: Optional[AnimateDiffPipeline] = None
        self.vid2vid_pipe: Optional[AnimateDiffVideoToVideoPipeline] = None
        self.controlnet_pipe: Optional[AnimateDiffControlNetPipeline] = None
        self.controlnet_mode: Optional[GenerationMode] = None
        self._peft_stripped: bool = False
        self._lightning_sched_config: Optional[dict] = None

    def _apply_lightning_scheduler(self, pipe) -> None:
        """Apply EulerDiscreteScheduler override for AnimateDiff-Lightning.

        Caches the scheduler config on first call to avoid redundant
        from_config() parsing on subsequent ensure_*() calls.
        """
        if not settings.is_animatediff_lightning:
            return
        if self._lightning_sched_config is None:
            sched = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
                beta_schedule="linear",
                clip_sample=False,
            )
            self._lightning_sched_config = dict(sched.config)
            log.info("Lightning scheduler config cached")
        pipe.scheduler = EulerDiscreteScheduler.from_config(self._lightning_sched_config)

    def _apply_freeu_if_enabled(self, pipe) -> None:
        """Apply FreeU unless explicitly disabled for Lightning."""
        if settings.is_animatediff_lightning and not settings.animatediff_lightning_freeu:
            log.debug("FreeU disabled for AnimateDiff-Lightning")
            return
        apply_freeu(pipe)

    def ensure_base(self, base_pipe) -> AnimateDiffPipeline:
        """Lazy-load AnimateDiff motion adapter and pipeline."""
        if self.motion_adapter is not None and self.pipe is not None:
            return self.pipe

        try:
            if settings.is_animatediff_lightning:
                step = settings.animatediff_lightning_steps
                repo = settings.animatediff_model
                ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
                log.info("Loading AnimateDiff-Lightning %d-step adapter: %s/%s", step, repo, ckpt)
                self.motion_adapter = MotionAdapter().to("cuda", torch.float16)
                ckpt_path = hf_hub_download(repo, ckpt, local_files_only=True)
                self.motion_adapter.load_state_dict(load_file(ckpt_path, device="cuda"))
            else:
                log.info("Loading AnimateDiff motion adapter: %s", settings.animatediff_model)
                self.motion_adapter = MotionAdapter.from_pretrained(
                    settings.animatediff_model,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                ).to("cuda")

            unet = get_uncompiled_unet(base_pipe)
            if not self._peft_stripped:
                strip_peft_from_unet(unet)
                self._peft_stripped = True

            self.pipe = AnimateDiffPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                scheduler=type(base_pipe.scheduler).from_config(base_pipe.scheduler.config),
                feature_extractor=None,
            )
            self.pipe.to("cuda")

            self._apply_lightning_scheduler(self.pipe)
            self._apply_freeu_if_enabled(self.pipe)
        except Exception:
            self.motion_adapter = None
            self.pipe = None
            raise

        log.info("AnimateDiff pipeline ready")
        return self.pipe

    def ensure_vid2vid(self, base_pipe) -> AnimateDiffVideoToVideoPipeline:
        """Lazy-load AnimateDiff vid2vid pipeline for img2img animation.

        Shares all heavy components (UNet, VAE, text encoder, motion adapter)
        with the base AnimateDiff pipeline -- no additional VRAM.
        """
        if self.vid2vid_pipe is not None:
            return self.vid2vid_pipe

        # Ensure base pipeline + motion adapter are loaded first
        self.ensure_base(base_pipe)

        # Reuse the already-converted UNetMotionModel from base AnimateDiff pipeline.
        # AnimateDiffPipeline.__init__ already called UNetMotionModel.from_unet2d()
        # in ensure_base(). Passing UNetMotionModel directly skips the expensive
        # re-conversion (~5-6s). All AnimateDiff pipeline variants share one UNet.
        unet = self.pipe.unet

        try:
            self.vid2vid_pipe = AnimateDiffVideoToVideoPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                scheduler=type(base_pipe.scheduler).from_config(base_pipe.scheduler.config),
                feature_extractor=None,
            )
            self.vid2vid_pipe.to("cuda")

            # Verify UNet sharing is effective (diffusers should not re-wrap)
            if self.vid2vid_pipe.unet is not self.pipe.unet:
                log.error("UNet not shared with vid2vid pipeline — diffusers re-wrapped. "
                          "First vid2vid init will be ~5s slower than expected.")

            self._apply_lightning_scheduler(self.vid2vid_pipe)
            self._apply_freeu_if_enabled(self.vid2vid_pipe)
        except Exception:
            self.vid2vid_pipe = None
            raise

        log.info("AnimateDiff vid2vid pipeline ready")
        return self.vid2vid_pipe

    def ensure_controlnet(
        self,
        base_pipe,
        mode: GenerationMode,
        existing_controlnet: Optional[ControlNetModel] = None,
    ) -> AnimateDiffControlNetPipeline:
        """Lazy-load AnimateDiff + ControlNet combined pipeline."""
        if self.controlnet_mode == mode and self.controlnet_pipe is not None:
            return self.controlnet_pipe

        self.ensure_base(base_pipe)

        model_id = CONTROLNET_IDS.get(mode)
        if model_id is None:
            raise ValueError(f"No ControlNet for mode: {mode}")

        if existing_controlnet is not None:
            controlnet = existing_controlnet
        else:
            log.info("Loading ControlNet for AnimateDiff: %s", model_id)
            load_kwargs: dict = dict(torch_dtype=torch.float16, local_files_only=True)
            # QR Code Monster v2: model weights in v2/ subfolder
            if mode == GenerationMode.CONTROLNET_QRCODE:
                load_kwargs["subfolder"] = "v2"
            controlnet = ControlNetModel.from_pretrained(
                model_id, **load_kwargs,
            ).to("cuda")

        # Reuse UNetMotionModel from base pipeline (same sharing as vid2vid)
        unet = self.pipe.unet

        try:
            self.controlnet_pipe = AnimateDiffControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                controlnet=controlnet,
                scheduler=type(base_pipe.scheduler).from_config(base_pipe.scheduler.config),
                feature_extractor=None,
            )
            self.controlnet_pipe.to("cuda")

            # Verify UNet sharing
            if self.controlnet_pipe.unet is not self.pipe.unet:
                log.error("UNet not shared with controlnet pipeline — diffusers re-wrapped.")

            self._apply_lightning_scheduler(self.controlnet_pipe)
            self._apply_freeu_if_enabled(self.controlnet_pipe)
        except Exception:
            self.controlnet_pipe = None
            self.controlnet_mode = None
            raise

        self.controlnet_mode = mode
        log.info("AnimateDiff + ControlNet pipeline ready (mode=%s)", mode.value)
        return self.controlnet_pipe

    def apply_free_noise(self, pipe, num_frames: int) -> bool:
        """Enable FreeNoise sliding-window temporal attention for long sequences.

        FreeNoise replaces manual chunking with noise rescheduling + weighted
        latent averaging across context windows, producing temporally coherent
        output for arbitrary frame counts.

        Incompatible with AnimateDiff-Lightning (distilled few-step). Returns
        False when skipped, True when activated.
        """
        if settings.is_animatediff_lightning:
            return False

        if num_frames <= settings.animatediff_context_length:
            # Short sequence — no sliding window needed
            return False

        try:
            pipe.enable_free_noise(
                context_length=settings.animatediff_context_length,
                context_stride=settings.animatediff_context_stride,
            )
            num_windows = max(1, (num_frames - settings.animatediff_context_length)
                              // settings.animatediff_context_stride + 1)
            log.info("FreeNoise enabled (context_length=%d, stride=%d, frames=%d, windows=%d)",
                     settings.animatediff_context_length,
                     settings.animatediff_context_stride,
                     num_frames, num_windows)
        except Exception as e:
            log.warning("FreeNoise unavailable: %s", e)
            return False

        # SplitInference: chunked attention/resnet for VRAM optimization on long sequences
        if settings.animatediff_split_inference and num_frames > settings.animatediff_context_length * 2:
            try:
                pipe.enable_free_noise_split_inference(
                    spatial_split_size=settings.animatediff_spatial_split_size,
                    temporal_split_size=settings.animatediff_temporal_split_size,
                )
                log.info("FreeNoise SplitInference enabled (spatial=%d, temporal=%d)",
                         settings.animatediff_spatial_split_size,
                         settings.animatediff_temporal_split_size)
            except Exception as e:
                log.warning("FreeNoise SplitInference unavailable: %s", e)

        return True

    @staticmethod
    def remove_free_noise(pipe) -> None:
        """Disable FreeNoise on the pipeline (cleanup after generation)."""
        if pipe is not None and getattr(pipe, 'free_noise_enabled', False):
            try:
                pipe.disable_free_noise()
                log.info("FreeNoise disabled")
            except Exception as e:
                log.warning("FreeNoise disable failed: %s", e)


    def unload(self) -> None:
        """Release all AnimateDiff resources — .to(cpu) for immediate VRAM release."""
        from .vram_utils import move_to_cpu, vram_cleanup
        move_to_cpu(self.motion_adapter)
        move_to_cpu(self.pipe)
        move_to_cpu(self.vid2vid_pipe)
        move_to_cpu(self.controlnet_pipe)
        self.motion_adapter = None
        self.pipe = None
        self.vid2vid_pipe = None
        self.controlnet_pipe = None
        self.controlnet_mode = None
        self._peft_stripped = False
        self._lightning_sched_config = None
        vram_cleanup()
