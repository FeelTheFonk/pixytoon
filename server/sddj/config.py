from __future__ import annotations

import logging
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
    prompt_schedules_dir: Path = _SERVER_ROOT / "prompt_schedules"
    prompts_data_dir: Path = _SERVER_ROOT / "data" / "prompts"

    # ── Default checkpoint ───────────────────────────────────
    # Relative to _SERVER_ROOT (resolved by engine at load time)
    default_checkpoint: str = "models/checkpoints/liberteRedmond_v10.safetensors"

    # ── Hyper-SD (replaces LCM-LoRA — better color fidelity) ─
    hyper_sd_repo: str = "ByteDance/Hyper-SD"
    hyper_sd_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    hyper_sd_fuse_scale: float = Field(0.8, ge=0.0, le=2.0)

    # ── DeepCache ────────────────────────────────────────────
    deepcache_interval: int = Field(3, ge=1, le=10)
    deepcache_branch: int = Field(0, ge=0)

    # ── Defaults ─────────────────────────────────────────────
    default_steps: int = Field(8, ge=1, le=100)
    default_cfg: float = Field(5.0, ge=0.0, le=30.0)
    default_width: int = Field(512, ge=64, le=2048)
    default_height: int = Field(512, ge=64, le=2048)

    # ── Default style LoRA (loaded before warmup) ─────────────
    # "auto" = first .safetensors in loras_dir, "" = none
    default_style_lora: str = "auto"
    default_style_lora_weight: float = Field(1.0, ge=0.0, le=2.0)

    # ── CLIP skip (2 = recommended for stylized content) ──────
    default_clip_skip: int = Field(2, ge=1, le=12)

    # ── Performance ──────────────────────────────────────────
    enable_torch_compile: bool = True
    compile_mode: Literal["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"] = "default"
    compile_dynamic: bool = False  # True only when DeepCache disabled (incompatible)
    # At 8 steps with interval=3, gain may be marginal — benchmark with/without
    enable_deepcache: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_warmup: bool = True
    enable_tf32: bool = True  # Ampere+ free ~15-30% speedup
    enable_lora_hotswap: bool = True  # Avoids torch.compile recompilation on LoRA switch
    max_lora_rank: int = Field(128, ge=1)  # Must be >= rank of all LoRA adapters used
    enable_cpu_offload: bool = False  # Mutually exclusive with DeepCache + torch.compile
    vram_min_free_mb: int = Field(512, ge=0)  # VRAM budget guard for lazy-loads
    # ── UNet quantization (torchao) ─────────────────────────
    enable_unet_quantization: bool = False
    unet_quantization_dtype: Literal["int8dq", "fp8dq", "int8wo", "fp8wo", "auto"] = "auto"
    # ── Attention backend ────────────────────────────────────
    attention_backend: Literal["auto", "sdp", "sage", "xformers"] = "auto"
    # ── Token merging (ToMe) — training-free UNet acceleration ──
    enable_tome: bool = False
    tome_ratio: float = Field(0.3, ge=0.0, le=0.75)  # Conservative: 0.3-0.4 for quality
    # ── TAESD preview decoder ────────────────────────────────
    enable_taesd_preview: bool = False  # Near-instant intermediate step preview
    # ── Frame compression (remote clients) ───────────────────
    enable_frame_compression: bool = False  # LZ4 compress binary frames before WS send
    # ── Queue management ─────────────────────────────────────
    queue_wait_timeout: float = Field(120.0, gt=0.0)  # Max wait for GPU lock (seconds)

    # ── FreeU v2 (free quality boost, no training needed) ────
    enable_freeu: bool = True
    freeu_s1: float = Field(0.9, ge=0.0, le=2.0)
    freeu_s2: float = Field(0.2, ge=0.0, le=2.0)
    freeu_b1: float = Field(1.5, ge=0.0, le=3.0)
    freeu_b2: float = Field(1.6, ge=0.0, le=3.0)

    # ── ControlNet QR Code Monster ───────────────────────────
    qr_controlnet_conditioning_scale: float = Field(1.5, ge=0.0, le=3.0)
    qr_control_guidance_start: float = Field(0.0, ge=0.0, le=1.0)
    qr_control_guidance_end: float = Field(0.8, ge=0.0, le=1.0)
    qr_default_steps: int = Field(20, ge=4, le=50)

    # ── Timeouts ─────────────────────────────────────────────
    generation_timeout: float = Field(600.0, gt=0.0)  # 10 minutes max per generation

    # ── rembg ────────────────────────────────────────────────
    rembg_model: str = "birefnet-general"  # SOTA edges (IoU 0.87 DIS5K vs u2net 0.39). Alt: u2net (fast CPU ~3-4s)
    rembg_on_cpu: bool = True  # Keep GPU free for diffusion

    # ── Animation ────────────────────────────────────────────
    max_animation_frames: int = Field(256, ge=1, le=10000)

    # ── AnimateDiff ──────────────────────────────────────────
    animatediff_model: str = "ByteDance/AnimateDiff-Lightning"
    enable_freeinit: bool = False
    freeinit_iterations: int = Field(2, ge=1, le=10)
    # ── AnimateDiff-Lightning (ByteDance distilled, 10× faster) ──
    # Activated when animatediff_model = "ByteDance/AnimateDiff-Lightning"
    animatediff_lightning_steps: int = Field(4, ge=1, le=8)
    animatediff_lightning_cfg: float = Field(2.0, ge=1.0, le=5.0)  # 2.0 preserves negative prompts
    animatediff_motion_lora_strength: float = Field(0.75, ge=0.0, le=1.0)
    animatediff_lightning_freeu: bool = True  # False = force-disable FreeU for Lightning pipelines
    # ── FreeNoise — long video temporal coherence (non-Lightning only) ──
    # Replaces manual chunking with diffusers-native sliding window + noise rescheduling.
    animatediff_context_length: int = Field(16, ge=8, le=32)
    animatediff_context_stride: int = Field(4, ge=1, le=16)
    animatediff_split_inference: bool = True  # Auto-enable SplitInference for long sequences
    animatediff_spatial_split_size: int = Field(256, ge=64, le=512)
    animatediff_temporal_split_size: int = Field(16, ge=4, le=32)
    # Lightning hard cap: FreeNoise is incompatible with distilled few-step models.
    animatediff_max_frames_lightning: int = Field(32, ge=8, le=64)

    # ── Audio Reactivity ──────────────────────────────────────
    audio_cache_dir: str = ""  # empty = system temp dir
    audio_max_file_size_mb: int = Field(500, ge=1, le=10000)
    audio_max_frames: int = Field(10800, ge=1)  # ~7.5 min at 24fps
    audio_default_attack: int = Field(2, ge=1, le=100)
    audio_default_release: int = Field(8, ge=1, le=100)
    stem_model: str = "htdemucs"
    stem_backend: Literal["demucs", "roformer"] = "demucs"  # roformer = BS-RoFormer via audio-separator (+3 dB SDR, 6 stems)
    stem_device: str = "cpu"  # always CPU — keep GPU free for diffusion
    # DSP precision (pinnacle quality)
    audio_sample_rate: int = Field(44100, ge=22050, le=96000)
    audio_hop_length: int = Field(256, ge=64, le=1024)
    audio_n_fft: int = Field(4096, ge=512, le=8192)
    audio_n_mels: int = Field(128, ge=64, le=512)
    audio_perceptual_weighting: bool = True   # K-weighting pre-filter (ITU-R BS.1770)
    audio_smoothing_mode: Literal["ema", "savgol"] = "ema"
    audio_beat_backend: Literal["auto", "librosa", "madmom", "beatnet", "allinone"] = "auto"
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
    # EquiVDM-style temporally coherent noise: flow-warp previous frame's noise
    # instead of random noise per frame. Reduces structural flicker at zero VRAM cost.
    equivdm_noise: bool = True
    equivdm_residual: float = Field(0.08, ge=0.0, le=0.5)  # Random residual to prevent stagnation

    # ── Video Export ──────────────────────────────────────────
    ffmpeg_path: str = ""  # empty = auto-detect via shutil.which("ffmpeg")
    audio_cache_ttl_hours: int = Field(24, ge=1, le=168)  # 1h–7d

    model_config = {
        "env_prefix": "SDDJ_",
        "env_file": str(_SERVER_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def is_animatediff_lightning(self) -> bool:
        return "animatediff-lightning" in self.animatediff_model.lower()

    @model_validator(mode='after')
    def _warn_missing_dirs(self):
        for name in ("models_dir", "checkpoints_dir", "loras_dir", "embeddings_dir", "palettes_dir", "presets_dir", "prompt_schedules_dir", "prompts_data_dir"):
            d = getattr(self, name)
            if not d.is_dir():
                log.warning("Directory does not exist: %s=%s", name, d)
        # Validate checkpoint path exists (warning, not exception)
        ckpt = Path(self.default_checkpoint)
        if not ckpt.is_absolute():
            ckpt = _SERVER_ROOT / ckpt
        if not ckpt.exists():
            log.warning("CRITICAL: default_checkpoint not found: %s", ckpt)
        # Cross-validation: mutually exclusive / incompatible settings
        if self.enable_cpu_offload and self.enable_deepcache:
            log.warning("cpu_offload + deepcache are mutually exclusive — disabling DeepCache")
            object.__setattr__(self, 'enable_deepcache', False)
        if self.compile_dynamic and self.enable_deepcache:
            log.warning("compile_dynamic=True is incompatible with DeepCache — forcing compile_dynamic=False")
            object.__setattr__(self, 'compile_dynamic', False)
        if self.enable_torch_compile and not self.enable_lora_hotswap:
            log.warning("torch_compile without lora_hotswap: LoRA switches will trigger full recompilation (~20s)")
        # AnimateDiff performance warnings
        if self.enable_freeinit and self.freeinit_iterations > 2:
            log.warning(
                "FreeInit iterations=%d: each iteration runs a FULL denoising pass. "
                "Recommended: 1-2 for interactive use, 3+ for offline batch only.",
                self.freeinit_iterations,
            )
        if self.animatediff_context_stride < 4:
            log.warning(
                "FreeNoise stride=%d is very aggressive: more overlapping windows = "
                "better coherence but significantly slower. Recommended: 4-8.",
                self.animatediff_context_stride,
            )
        return self


settings = Settings()
