"""Server configuration via Pydantic Settings (env vars / .env / defaults)."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


_SERVER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Network ──────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 9876

    # ── Paths ────────────────────────────────────────────────
    models_dir: Path = _SERVER_ROOT / "models"
    checkpoints_dir: Path = _SERVER_ROOT / "models" / "checkpoints"
    loras_dir: Path = _SERVER_ROOT / "models" / "loras"
    embeddings_dir: Path = _SERVER_ROOT / "models" / "embeddings"
    controlnets_dir: Path = _SERVER_ROOT / "models" / "controlnets"
    palettes_dir: Path = _SERVER_ROOT / "palettes"

    # ── Default checkpoint ───────────────────────────────────
    default_checkpoint: str = "Lykon/dreamshaper-8"

    # ── Hyper-SD (replaces LCM-LoRA — better color fidelity) ─
    hyper_sd_repo: str = "ByteDance/Hyper-SD"
    hyper_sd_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    hyper_sd_fuse_scale: float = 0.8

    # ── DeepCache ────────────────────────────────────────────
    deepcache_interval: int = 3
    deepcache_branch: int = 0

    # ── Defaults ─────────────────────────────────────────────
    default_steps: int = 8
    default_cfg: float = 5.0
    default_width: int = 512
    default_height: int = 512
    default_denoise: float = 0.75
    default_quantize_colors: int = 32

    # ── Default pixel art LoRA (loaded before warmup) ────────
    # "auto" = first .safetensors in loras_dir, "" = none
    default_pixel_lora: str = "auto"
    default_pixel_lora_weight: float = 1.0

    # ── Default negative TI (auto-load from embeddings dir) ───
    default_negative_ti: str = "auto"

    # ── CLIP skip (2 = recommended for pixel art / stylized) ──
    default_clip_skip: int = 2

    # ── Performance ──────────────────────────────────────────
    enable_torch_compile: bool = True
    compile_mode: str = "default"  # "default" | "max-autotune" | "reduce-overhead"
    enable_deepcache: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_warmup: bool = True

    # ── FreeU v2 (free quality boost, no training needed) ────
    enable_freeu: bool = True
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2
    freeu_b1: float = 1.5
    freeu_b2: float = 1.6

    # ── Timeouts ─────────────────────────────────────────────
    generation_timeout: float = 300.0  # 5 minutes max per generation

    # ── rembg ────────────────────────────────────────────────
    rembg_model: str = "birefnet-general"
    rembg_on_cpu: bool = True  # Keep GPU free for diffusion

    model_config = {"env_prefix": "PIXYTOON_"}


settings = Settings()
