"""Server configuration via Pydantic Settings (env vars / .env / defaults)."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

log = logging.getLogger("sddj.config")


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

    # ── Default style LoRA (loaded before warmup) ─────────────
    # "auto" = first .safetensors in loras_dir, "" = none
    default_style_lora: str = "auto"
    default_style_lora_weight: float = Field(1.0, ge=0.0, le=2.0)

    # ── CLIP skip (2 = recommended for stylized content) ──────
    default_clip_skip: int = 2

    # ── Performance ──────────────────────────────────────────
    enable_torch_compile: bool = True
    compile_mode: Literal["default", "max-autotune", "reduce-overhead"] = "default"
    compile_dynamic: bool = False  # True only when DeepCache disabled (incompatible)
    enable_deepcache: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_warmup: bool = True
    enable_tf32: bool = True  # Ampere+ free ~15-30% speedup
    enable_lora_hotswap: bool = True  # Avoids torch.compile recompilation on LoRA switch
    max_lora_rank: int = Field(128, ge=1)  # Must be >= rank of all LoRA adapters used
    enable_cpu_offload: bool = False  # Mutually exclusive with DeepCache + torch.compile
    vram_min_free_mb: int = Field(512, ge=0)  # VRAM budget guard for lazy-loads
    quantize_unet: Literal["none", "int8", "fp8"] = "none"  # INT8/FP8 UNet quantization

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
    max_animation_frames: int = 120

    # ── AnimateDiff ──────────────────────────────────────────
    animatediff_model: str = "ByteDance/AnimateDiff-Lightning"
    enable_freeinit: bool = False
    freeinit_iterations: int = 2
    # ── AnimateDiff-Lightning (ByteDance distilled, 10× faster) ──
    # Activated when animatediff_model = "ByteDance/AnimateDiff-Lightning"
    animatediff_lightning_steps: int = Field(4, ge=1, le=8)
    animatediff_lightning_cfg: float = Field(2.0, ge=1.0, le=5.0)  # 2.0 preserves negative prompts
    animatediff_motion_lora_strength: float = Field(0.75, ge=0.0, le=1.0)
    animatediff_lightning_freeu: bool = True  # False = force-disable FreeU for Lightning pipelines

    # ── Audio Reactivity ──────────────────────────────────────
    audio_cache_dir: str = ""  # empty = system temp dir
    audio_max_file_size_mb: int = 500
    audio_max_frames: int = 10800  # ~7.5 min at 24fps
    audio_default_attack: int = 2
    audio_default_release: int = 8
    stem_model: str = "htdemucs"
    stem_device: str = "cpu"  # always CPU — keep GPU free for diffusion
    # DSP precision (pinnacle quality)
    audio_sample_rate: int = Field(44100, ge=22050, le=96000)
    audio_hop_length: int = Field(256, ge=64, le=1024)
    audio_n_fft: int = Field(4096, ge=512, le=8192)
    audio_n_mels: int = Field(256, ge=64, le=512)
    audio_perceptual_weighting: bool = True   # K-weighting pre-filter (ITU-R BS.1770)
    audio_smoothing_mode: Literal["ema", "savgol"] = "ema"
    audio_beat_backend: Literal["auto", "librosa", "madmom"] = "auto"
    audio_superflux_lag: int = Field(2, ge=1, le=5)
    audio_superflux_max_size: int = Field(3, ge=1, le=7)

    # ── Temporal Coherence ──────────────────────────────────────
    # Step scaling cap for distilled models (Hyper-SD): max multiplier on steps.
    # E.g. cap=2 means steps=8 can scale to at most 16, preventing wasted compute
    # on models distilled for N-step inference.
    distilled_step_scale_cap: int = Field(2, ge=1, le=10)
    # LAB color coherence between consecutive frames (0 = disabled).
    # Matches each generated frame's LAB statistics to the previous frame,
    # preventing color drift in chains. Recommended: 0.3-0.7.
    color_coherence_strength: float = Field(0.5, ge=0.0, le=1.0)
    # Auto noise-denoise coupling: when no noise_amplitude slot is active,
    # inject subtle noise inversely proportional to denoise strength.
    # Gated below denoise 0.35 — at lower values the model lacks capacity
    # to resolve injected noise, causing progressive artifact accumulation.
    auto_noise_coupling: bool = True
    # Optical flow temporal blending strength (0 = disabled).
    # Blends each frame with a flow-warped previous frame to reduce jitter.
    # Adds ~10-20ms per frame. Recommended: 0.1-0.3 if enabled.
    optical_flow_blend: float = Field(0.0, ge=0.0, le=0.5)

    # ── Video Export ──────────────────────────────────────────
    ffmpeg_path: str = ""  # empty = auto-detect via shutil.which("ffmpeg")

    model_config = {"env_prefix": "SDDJ_"}

    @cached_property
    def is_animatediff_lightning(self) -> bool:
        """True when the configured AnimateDiff model is Lightning."""
        return "animatediff-lightning" in self.animatediff_model.lower()

    @model_validator(mode='after')
    def _warn_missing_dirs(self):
        for name in ("models_dir", "checkpoints_dir", "loras_dir", "embeddings_dir", "palettes_dir", "presets_dir", "prompts_data_dir"):
            d = getattr(self, name)
            if not d.is_dir():
                log.warning("Directory does not exist: %s=%s", name, d)
        return self


settings = Settings()
