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
    controlnets_dir: Path = _SERVER_ROOT / "models" / "controlnets"
    palettes_dir: Path = _SERVER_ROOT / "palettes"

    # ── Default checkpoint ───────────────────────────────────
    default_checkpoint: str = "Lykon/dreamshaper-8"

    # ── Hyper-SD (replaces LCM-LoRA — better color fidelity) ─
    hyper_sd_repo: str = "ByteDance/Hyper-SD"
    hyper_sd_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    hyper_sd_fuse_scale: float = 0.5

    # ── DeepCache ────────────────────────────────────────────
    deepcache_interval: int = 3
    deepcache_branch: int = 0

    # ── Defaults ─────────────────────────────────────────────
    default_steps: int = 8
    default_cfg: float = 5.0
    default_width: int = 512
    default_height: int = 512
    default_denoise: float = 0.75
    default_quantize_colors: int = 16

    # ── Performance ──────────────────────────────────────────
    enable_torch_compile: bool = True
    enable_deepcache: bool = True
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True

    # ── FreeU v2 (free quality boost, no training needed) ────
    enable_freeu: bool = True
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2
    freeu_b1: float = 1.3
    freeu_b2: float = 1.4

    # ── rembg ────────────────────────────────────────────────
    rembg_model: str = "birefnet-general"
    rembg_on_cpu: bool = True  # Keep GPU free for diffusion

    model_config = {"env_prefix": "PIXYTOON_"}


settings = Settings()
