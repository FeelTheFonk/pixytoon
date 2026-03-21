"""Server configuration via Pydantic Settings (env vars / .env / defaults)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

log = logging.getLogger("pixytoon.config")


_SERVER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Network ──────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = Field(9876, ge=1, le=65535)

    # ── Paths ────────────────────────────────────────────────
    models_dir: Path = _SERVER_ROOT / "models"
    checkpoints_dir: Path = _SERVER_ROOT / "models" / "checkpoints"
    loras_dir: Path = _SERVER_ROOT / "models" / "loras"
    embeddings_dir: Path = _SERVER_ROOT / "models" / "embeddings"
    palettes_dir: Path = _SERVER_ROOT / "palettes"
    presets_dir: Path = _SERVER_ROOT / "presets"
    prompts_data_dir: Path = _SERVER_ROOT / "data" / "prompts"

    # ── Default checkpoint ───────────────────────────────────
    default_checkpoint: str = "Lykon/dreamshaper-8"

    # ── Hyper-SD (replaces LCM-LoRA — better color fidelity) ─
    hyper_sd_repo: str = "ByteDance/Hyper-SD"
    hyper_sd_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    hyper_sd_fuse_scale: float = Field(0.8, ge=0.0, le=2.0)

    # ── DeepCache ────────────────────────────────────────────
    deepcache_interval: int = Field(3, ge=1, le=10)
    deepcache_branch: int = Field(0, ge=0)

    # ── Defaults ─────────────────────────────────────────────
    default_steps: int = 8
    default_cfg: float = 5.0
    default_width: int = 512
    default_height: int = 512

    # ── Default pixel art LoRA (loaded before warmup) ────────
    # "auto" = first .safetensors in loras_dir, "" = none
    default_pixel_lora: str = "auto"
    default_pixel_lora_weight: float = Field(1.0, ge=0.0, le=2.0)

    # ── CLIP skip (2 = recommended for pixel art / stylized) ──
    default_clip_skip: int = 2

    # ── Performance ──────────────────────────────────────────
    enable_torch_compile: bool = True
    compile_mode: Literal["default", "max-autotune", "reduce-overhead"] = "default"
    enable_deepcache: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_warmup: bool = True

    # ── FreeU v2 (free quality boost, no training needed) ────
    enable_freeu: bool = True
    freeu_s1: float = Field(0.9, ge=0.0, le=2.0)
    freeu_s2: float = Field(0.2, ge=0.0, le=2.0)
    freeu_b1: float = Field(1.5, ge=0.0, le=3.0)
    freeu_b2: float = Field(1.6, ge=0.0, le=3.0)

    # ── Timeouts ─────────────────────────────────────────────
    generation_timeout: float = Field(600.0, gt=0.0)  # 10 minutes max per generation

    # ── rembg ────────────────────────────────────────────────
    rembg_model: str = "u2net"  # Fast CPU (~3-4s). Alt: birefnet-general (best edges, ~10s), bria-rmbg (SOTA quality)
    rembg_on_cpu: bool = True  # Keep GPU free for diffusion

    # ── Animation ────────────────────────────────────────────
    default_anim_frames: int = 8
    default_anim_duration_ms: int = 100
    default_anim_denoise: float = 0.30
    max_animation_frames: int = 120

    # ── AnimateDiff ──────────────────────────────────────────
    animatediff_model: str = "guoyww/animatediff-motion-adapter-v1-5-3"
    enable_freeinit: bool = False
    freeinit_iterations: int = 2

    # ── Real-Time Paint Mode ──────────────────────────────────
    realtime_timeout: float = Field(60.0, gt=0.0)  # auto-stop if no frame for N seconds
    realtime_default_steps: int = Field(4, ge=2, le=8)
    realtime_default_cfg: float = Field(2.5, ge=1.0, le=10.0)
    realtime_default_denoise: float = Field(0.5, ge=0.05, le=0.95)
    realtime_roi_padding: int = Field(32, ge=8, le=128)  # padding around ROI crop
    realtime_roi_min_size: int = Field(64, ge=32, le=256)  # minimum ROI dimension

    model_config = {"env_prefix": "PIXYTOON_"}

    @model_validator(mode='after')
    def _warn_missing_dirs(self):
        for name in ("models_dir", "checkpoints_dir", "loras_dir", "embeddings_dir", "palettes_dir", "presets_dir", "prompts_data_dir"):
            d = getattr(self, name)
            if not d.is_dir():
                log.warning("Directory does not exist: %s=%s", name, d)
        return self


settings = Settings()
