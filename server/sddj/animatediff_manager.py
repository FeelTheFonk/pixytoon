"""AnimateDiff pipeline lifecycle — adapter, PEFT strip, pipeline construction."""

from __future__ import annotations

import copy
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

    def _apply_lightning_scheduler(self, pipe) -> None:
        """Apply EulerDiscreteScheduler override for AnimateDiff-Lightning."""
        if not settings.is_animatediff_lightning:
            return
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
            clip_sample=False,
        )
        log.info("Lightning scheduler: EulerDiscreteScheduler (trailing, linear, clip_sample=False)")

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
            strip_peft_from_unet(unet)

            self.pipe = AnimateDiffPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                scheduler=copy.deepcopy(base_pipe.scheduler),
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

        unet = get_uncompiled_unet(base_pipe)

        try:
            self.vid2vid_pipe = AnimateDiffVideoToVideoPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                scheduler=copy.deepcopy(base_pipe.scheduler),
                feature_extractor=None,
            )
            self.vid2vid_pipe.to("cuda")

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

        # UNet already PEFT-stripped by ensure_base() above
        unet = get_uncompiled_unet(base_pipe)

        try:
            self.controlnet_pipe = AnimateDiffControlNetPipeline(
                vae=base_pipe.vae,
                text_encoder=base_pipe.text_encoder,
                tokenizer=base_pipe.tokenizer,
                unet=unet,
                motion_adapter=self.motion_adapter,
                controlnet=controlnet,
                scheduler=copy.deepcopy(base_pipe.scheduler),
                feature_extractor=None,
            )
            self.controlnet_pipe.to("cuda")

            self._apply_lightning_scheduler(self.controlnet_pipe)
            self._apply_freeu_if_enabled(self.controlnet_pipe)
        except Exception:
            self.controlnet_pipe = None
            self.controlnet_mode = None
            raise

        self.controlnet_mode = mode
        log.info("AnimateDiff + ControlNet pipeline ready (mode=%s)", mode.value)
        return self.controlnet_pipe

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
        vram_cleanup()
